import numpy as np
from scipy.optimize import minimize_scalar
import logging
import h5py
from pathlib import Path
import cupy as cp
from .utils import gpu_to_cpu, cpu_to_gpu

logger = logging.getLogger(__name__)

# CUDA kernel for OASIS algorithm
OASIS_KERNEL = cp.RawKernel(r'''
extern "C" __global__ void oasis_kernel(
    const double* __restrict__ traces,      // Input traces [n_neurons x T]
    double* __restrict__ denoised,          // Output denoised traces [n_neurons x T]
    double* __restrict__ spikes,            // Output spike trains [n_neurons x T]
    double* __restrict__ pool_buffer,       // Global memory buffer for pool data
    const int T,                            // Number of time points
    const double g,                         // Decay constant
    const double lambda,                    // L1 penalty
    const double smin,                      // Minimum spike size (optional)
    const double spike_threshold            // Threshold for binary spike detection
) {
    // Get neuron index (one neuron per block)
    const int neuron_idx = blockIdx.x;
    const int trace_offset = neuron_idx * T;
    const int pool_offset = neuron_idx * T * 4;  // 4 arrays per neuron
    
    // Use global memory for pool data
    double* v_pool = &pool_buffer[pool_offset];                  // Pool values
    double* w_pool = &pool_buffer[pool_offset + T];             // Pool weights
    int* start_idx = (int*)&pool_buffer[pool_offset + 2*T];     // Pool start indices
    int* pool_length = (int*)&pool_buffer[pool_offset + 3*T];   // Pool lengths
    
    // Shared memory for temporary calculations
    __shared__ double temp_vals[128];  // Small buffer for intermediate calculations
    __shared__ int n_pools;            // Number of active pools
    
    // Only thread 0 in each block does the work (sequential algorithm)
    if (threadIdx.x == 0) {
        // Initialize pools
        n_pools = 0;
        
        // Calculate mu from lambda (penalty terms)
        double mu_t;
        
        // Initialize first pool with first data point
        mu_t = lambda * (1.0 - g);
        v_pool[0] = traces[trace_offset] - mu_t;
        w_pool[0] = 1.0;
        start_idx[0] = 0;
        pool_length[0] = 1;
        n_pools = 1;
        
        // Process each time point
        for (int t = 1; t < T; t++) {
            // Add new pool
            mu_t = (t == T-1) ? lambda : lambda * (1.0 - g);
            v_pool[n_pools] = traces[trace_offset + t] - mu_t;
            w_pool[n_pools] = 1.0;
            start_idx[n_pools] = t;
            pool_length[n_pools] = 1;
            n_pools++;
            
            // Merge pools if necessary
            while (n_pools > 1) {
                int i = n_pools - 2;  // Second to last pool
                
                // Calculate g^l where l is pool length
                double g_power = 1.0;
                for (int p = 0; p < pool_length[i]; p++) {
                    g_power *= g;
                }
                
                // Check merging condition
                bool should_merge;
                if (smin <= 0.0) {
                    // L1 penalty version
                    should_merge = v_pool[i+1] < g_power * v_pool[i];
                } else {
                    // Hard threshold version
                    should_merge = v_pool[i+1] < g_power * v_pool[i] + smin;
                }
                
                if (!should_merge) break;
                
                // Merge pools i and i+1
                double g_power_2 = g_power * g_power;
                double w_new = w_pool[i] + g_power_2 * w_pool[i+1];
                double v_new = (w_pool[i]*v_pool[i] + g_power*w_pool[i+1]*v_pool[i+1]) / w_new;
                
                // Update pool i
                v_pool[i] = v_new;
                w_pool[i] = w_new;
                pool_length[i] += pool_length[i+1];
                
                // Remove pool i+1
                n_pools--;
            }
            
            // Safety check: if too many pools, merge the oldest ones
            if (n_pools >= T-10) {
                // Merge the first two pools
                double g_power = 1.0;
                for (int p = 0; p < pool_length[0]; p++) {
                    g_power *= g;
                }
                double w_new = w_pool[0] + g_power * g_power * w_pool[1];
                double v_new = (w_pool[0]*v_pool[0] + g_power*w_pool[1]*v_pool[1]) / w_new;
                
                // Update first pool
                v_pool[0] = v_new;
                w_pool[0] = w_new;
                pool_length[0] += pool_length[1];
                
                // Shift remaining pools
                for (int j = 1; j < n_pools-1; j++) {
                    v_pool[j] = v_pool[j+1];
                    w_pool[j] = w_pool[j+1];
                    start_idx[j] = start_idx[j+1];
                    pool_length[j] = pool_length[j+1];
                }
                n_pools--;
            }
        }
        
        // Construct solution
        for (int p = 0; p < n_pools; p++) {
            double v = fmax(0.0, v_pool[p]);  // Enforce non-negativity
            for (int tau = 0; tau < pool_length[p]; tau++) {
                // Calculate g^tau
                double g_tau = 1.0;
                for (int k = 0; k < tau; k++) {
                    g_tau *= g;
                }
                denoised[trace_offset + start_idx[p] + tau] = v * g_tau;
            }
        }
        
        // Calculate spikes
        spikes[trace_offset] = fmax(0.0, denoised[trace_offset]);
        for (int t = 1; t < T; t++) {
            double s = fmax(0.0, denoised[trace_offset + t] - g * denoised[trace_offset + t - 1]);
            spikes[trace_offset + t] = (s > spike_threshold) ? 1.0 : 0.0;
        }
    }
}
''', 'oasis_kernel')

def determine_pilot_lambdas(time_series, n_pilots, g, sigma):
    """Determine lambda values from pilot neurons.
    
    Parameters
    ----------
    time_series : ndarray
        Full time series data for all neurons
    n_pilots : int
        Number of pilot neurons to use (typically 32)
    g : float
        Decay constant
    sigma : float
        Noise standard deviation
    
    Returns
    -------
    float
        Median lambda value from pilot neurons
    """
    n_neurons = time_series.shape[0]
    
    # Randomly select pilot neurons
    pilot_indices = np.random.choice(n_neurons, size=n_pilots, replace=False)
    lambdas = np.zeros(n_pilots)
    
    # Process each pilot neuron
    for i, idx in enumerate(pilot_indices):
        trace = time_series[idx]
        T = len(trace)
        
        # Normalize trace
        trace_mean = np.mean(trace)
        trace_std = np.std(trace)
        if trace_std > 0:
            trace = (trace - trace_mean) / trace_std
        
        # Initialize OASIS for this pilot
        oasis = OASIS(g=g, smin=None, lambda_=None)
        
        # Function to minimize: distance between RSS and sigma^2 * T
        def objective(lambda_):
            oasis.lambda_ = float(lambda_)
            c, _ = oasis.fit(trace)
            RSS = np.sum((trace - c)**2)
            return (RSS - sigma**2 * T)**2
        
        # Find optimal lambda
        max_lambda = 10.0 * np.abs(trace).max()
        res = minimize_scalar(objective, bounds=(0, max_lambda), method='bounded')
        lambdas[i] = float(res.x)
    
    # Return median lambda
    return float(np.median(lambdas))

class OASIS:
    """OASIS (Online Active Set method to Infer Spikes) algorithm for spike inference.
    
    This implementation is based on:
    Friedrich J, Zhou P, Paninski L (2017) Fast online deconvolution of calcium imaging data.
    PLOS Computational Biology 13(3): e1005423. https://doi.org/10.1371/journal.pcbi.1005423
    
    Parameters
    ----------
    g : float
        Calcium decay time constant (gamma)
    smin : float, optional
        Minimum spike amplitude. If None, L1 penalty is used instead
    lambda_ : float, optional 
        L1 sparsity penalty. If None, it is optimized using noise estimate
    """
    def __init__(self, g, smin=None, lambda_=None):
        self.g = g
        self.smin = smin
        self.lambda_ = lambda_
        
    def fit(self, y, sigma=None):
        """Infer spikes from fluorescence trace."""
        y = np.asarray(y, dtype=np.float64)  # Use float64 for better precision
        T = len(y)
        
        # Initialize pools
        pools = []  # Each pool is (value, weight, time, length)
        
        # Initialize solution
        c = np.zeros(T, dtype=np.float64)
        s = np.zeros(T, dtype=np.float64)
        
        # If lambda_ not provided, initialize it to 0
        lambda_ = float(self.lambda_) if self.lambda_ is not None else 0.0
        
        # Calculate mu from lambda_
        mu = lambda_ * (1 - self.g + np.zeros(T, dtype=np.float64))
        mu[-1] = lambda_  # Last time point has different mu
        
        # Initialize first pool with first data point
        pools.append((float(y[0] - mu[0]), 1.0, 0, 1))
        
        # Process each time point
        for t in range(1, T):
            # Add new pool
            pools.append((float(y[t] - mu[t]), 1.0, t, 1))
            
            # Merge pools if necessary
            while len(pools) > 1:
                i = len(pools) - 2  # Second to last pool
                if self.smin is None:
                    # L1 penalty version
                    if pools[i+1][0] >= self.g**pools[i][3] * pools[i][0]:
                        break
                else:
                    # Hard threshold version
                    if pools[i+1][0] >= self.g**pools[i][3] * pools[i][0] + self.smin:
                        break
                
                # Merge pools i and i+1
                vi = pools[i][0]
                wi = pools[i][1]
                li = pools[i][3]
                
                vip1 = pools[i+1][0]
                wip1 = pools[i+1][1]
                
                # New pool parameters with careful handling of numerical precision
                g_power = self.g**li
                g_power_2 = self.g**(2*li)
                w_new = wi + g_power_2 * wip1
                v_new = (wi*vi + g_power*wip1*vip1) / w_new
                t_new = pools[i][2]
                l_new = li + pools[i+1][3]
                
                # Replace pools i and i+1 with merged pool
                pools[i] = (float(v_new), float(w_new), t_new, l_new)
                pools.pop(i+1)
        
        # Construct solution
        for v, w, t, l in pools:
            v = max(0, v)  # Enforce non-negativity
            for tau in range(l):
                c[t+tau] = v * (self.g**tau)
        
        # Calculate continuous spike train with non-negativity constraint
        s_continuous = np.zeros(T, dtype=np.float64)
        s_continuous[1:] = np.maximum(0, c[1:] - self.g * c[:-1])
        s_continuous[0] = max(0, c[0])
        
        # Convert to binary spikes using threshold
        threshold = 0.1 * s_continuous.max() if s_continuous.max() > 0 else 0
        s = (s_continuous > threshold).astype(np.float64)
        
        # If lambda_ was not provided, optimize it
        if self.lambda_ is None and sigma is not None:
            # Function to minimize: distance between RSS and sigma^2 * T
            def objective(lambda_):
                self.lambda_ = float(lambda_)
                c_new, _ = self.fit(y)
                RSS = np.sum((y - c_new)**2)
                return (RSS - sigma**2 * T)**2
            
            # Find optimal lambda_
            # Use a reasonable upper bound based on data scale
            max_lambda = 10.0 * np.abs(y).max()
            res = minimize_scalar(objective, bounds=(0, max_lambda), method='bounded')
            self.lambda_ = float(res.x)
            
            # Rerun with optimal lambda_
            c, s = self.fit(y)
        
        return c, s

def detect_spikes(neuron_data, config):
    """Detect spikes from calcium traces using GPU-accelerated OASIS algorithm.
    
    Parameters
    ----------
    neuron_data : dict
        Dictionary containing neuron positions and time series
    config : dict
        Configuration dictionary
        
    Returns
    -------
    dict
        Dictionary containing original data plus spike information
    """
    logger.info("Starting GPU-accelerated spike detection")
    
    # Get parameters from config
    g = config.get('spike_detection', {}).get('decay_constant', 0.95)
    smin = config.get('spike_detection', {}).get('minimum_spike', None)
    lambda_ = config.get('spike_detection', {}).get('lambda', None)
    sigma = config.get('spike_detection', {}).get('noise_std', None)
    
    # Get time series data
    time_series = gpu_to_cpu(neuron_data['time_series'])  # Start on CPU for normalization
    n_neurons, n_timepoints = time_series.shape
    
    logger.info(f"Processing {n_neurons} neurons with {n_timepoints} time points each")
    
    # Normalize all traces
    traces_mean = np.mean(time_series, axis=1, keepdims=True)
    traces_std = np.std(time_series, axis=1, keepdims=True)
    traces_std[traces_std == 0] = 1.0  # Avoid division by zero
    time_series_norm = (time_series - traces_mean) / traces_std
    
    # If lambda is not provided, determine it from pilot neurons
    if lambda_ is None and sigma is not None:
        n_pilots = 8
        logger.info(f"Determining optimal lambda from {n_pilots} pilot neurons")
        lambda_ = determine_pilot_lambdas(time_series_norm, n_pilots=n_pilots, g=g, sigma=sigma)
        logger.info(f"Using lambda = {lambda_:.6f} (determined from pilot neurons)")
    elif lambda_ is None:
        lambda_ = 0.0
        logger.warning("No lambda or sigma provided, using lambda = 0.0")
    
    # Move normalized data to GPU
    traces_gpu = cp.asarray(time_series_norm, dtype=cp.float64)
    
    # Allocate output arrays on GPU
    denoised_gpu = cp.zeros_like(traces_gpu)
    spikes_gpu = cp.zeros_like(traces_gpu)
    
    # Allocate pool buffer in global memory
    # 4 arrays per neuron: v_pool, w_pool, start_idx, pool_length
    pool_buffer = cp.zeros(n_neurons * n_timepoints * 4, dtype=cp.float64)
    
    # Calculate spike threshold
    spike_threshold = 0.1 if sigma is None else 2.0 * sigma
    
    # Launch kernel
    logger.info("Launching OASIS kernel")
    threads_per_block = 32  # One thread per neuron within each block
    blocks = (n_neurons, 1, 1)  # One block per neuron
    
    try:
        OASIS_KERNEL(
            blocks,
            (threads_per_block, 1, 1),
            (
                traces_gpu,           # Input traces
                denoised_gpu,         # Output denoised traces
                spikes_gpu,           # Output spike trains
                pool_buffer,          # Global memory buffer for pool data
                n_timepoints,         # Number of time points
                g,                    # Decay constant
                lambda_,              # L1 penalty
                float(smin if smin is not None else 0.0),  # Minimum spike size
                spike_threshold       # Threshold for binary spike detection
            )
        )
        
        # Synchronize to ensure kernel completion
        cp.cuda.Stream.null.synchronize()
        
        logger.info("Kernel execution complete")
        
        # Denormalize denoised traces
        denoised_gpu = denoised_gpu * cp.asarray(traces_std) + cp.asarray(traces_mean)
        
        # Return results
        return {
            'positions': neuron_data['positions'],
            'time_series': neuron_data['time_series'],
            'denoised_time_series': denoised_gpu,
            'spikes': spikes_gpu,
            'metadata': neuron_data['metadata']
        }
        
    except Exception as e:
        logger.error(f"Error during kernel execution: {str(e)}")
        raise

def write_spike_results(results, config):
    """Write spike detection results to HDF5 file."""
    output_path = Path(config['output']['base_dir']) / 'spike_neuron_data.h5'
    
    logger.info(f"Writing spike detection results to: {output_path}")
    
    try:
        with h5py.File(output_path, 'w') as f:
            # Create groups
            neurons = f.create_group('neurons')
            
            # Store data
            neurons.create_dataset('positions', data=gpu_to_cpu(results['positions']))
            neurons.create_dataset('time_series', data=gpu_to_cpu(results['time_series']))
            neurons.create_dataset('denoised_time_series', data=gpu_to_cpu(results['denoised_time_series']))
            neurons.create_dataset('spikes', data=gpu_to_cpu(results['spikes']))
            
            # Store metadata
            neurons.attrs['total_neurons'] = len(results['positions'])
            neurons.attrs['time_points'] = results['time_series'].shape[1]
            
            logger.info(f"Saved spike data for {neurons.attrs['total_neurons']} neurons")
            
    except Exception as e:
        logger.error(f"Error writing spike results: {str(e)}")
        raise 