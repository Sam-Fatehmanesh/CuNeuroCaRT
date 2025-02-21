import numpy as np
from scipy.optimize import minimize_scalar
import logging
import h5py
from pathlib import Path
import cupy as cp
from .utils import gpu_to_cpu, cpu_to_gpu

logger = logging.getLogger(__name__)

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
    """Detect spikes from calcium traces using OASIS algorithm.
    
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
    logger.info("Starting spike detection")
    
    # Get parameters from config
    g = config.get('spike_detection', {}).get('decay_constant', 0.95)
    smin = config.get('spike_detection', {}).get('minimum_spike', None)
    lambda_ = config.get('spike_detection', {}).get('lambda', None)
    sigma = config.get('spike_detection', {}).get('noise_std', None)
    
    # Initialize OASIS
    oasis = OASIS(g=g, smin=smin, lambda_=lambda_)
    
    # Get time series data
    time_series = gpu_to_cpu(neuron_data['time_series'])
    n_neurons, n_timepoints = time_series.shape
    
    # Initialize output arrays
    denoised = np.zeros_like(time_series)
    spikes = np.zeros_like(time_series)
    
    # Process each neuron
    logger.info(f"Processing {n_neurons} neurons")
    for i in range(n_neurons):
        if i % 100 == 0:
            logger.info(f"Processed {i}/{n_neurons} neurons")
            
        # Get calcium trace for this neuron
        trace = time_series[i]
        
        # Normalize trace
        trace_mean = np.mean(trace)
        trace_std = np.std(trace)
        if trace_std > 0:
            trace = (trace - trace_mean) / trace_std
        
        # Detect spikes
        c, s = oasis.fit(trace, sigma=sigma)
        
        # Store results (denormalized)
        denoised[i] = c * trace_std + trace_mean
        spikes[i] = s
    
    logger.info("Spike detection complete")
    
    # Convert results to GPU arrays for consistency
    denoised = cpu_to_gpu(denoised)
    spikes = cpu_to_gpu(spikes)
    
    # Return results
    return {
        'positions': neuron_data['positions'],
        'time_series': neuron_data['time_series'],
        'denoised_time_series': denoised,
        'spikes': spikes,
        'metadata': neuron_data['metadata']
    }

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