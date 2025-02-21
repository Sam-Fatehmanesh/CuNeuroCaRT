import cupy as cp
import numpy as np
import logging
from .utils import gpu_to_cpu, cpu_to_gpu

logger = logging.getLogger(__name__)

def create_circular_mask(radius):
    """Create a circular mask for averaging around neuron centers."""
    y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
    mask = x*x + y*y <= radius*radius
    return cp.asarray(mask)

def extract_time_series(volume, neuron_data, config):
    """Extract time series for each detected neuron from memory-mapped volume.
    
    Parameters
    ----------
    volume : numpy.memmap
        Memory-mapped 4D array (time, z, y, x)
    neuron_data : dict
        Dictionary containing neuron positions and time series
    config : dict
        Configuration dictionary
    """
    logger.info("Starting time series extraction")
    
    # Get parameters
    neighborhood_size = config['extraction']['neighborhood_size']
    
    # Create circular mask for averaging
    mask = create_circular_mask(neighborhood_size)
    mask_height, mask_width = mask.shape
    mask_offset = neighborhood_size
    
    time_points, z_slices, height, width = volume.shape
    positions = neuron_data['positions']
    n_neurons = len(positions)
    
    # Initialize time series array on CPU
    time_series = np.zeros((n_neurons, time_points), dtype=np.float32)
    
    try:
        # Calculate optimal chunk size based on available GPU memory
        free_memory = cp.cuda.runtime.memGetInfo()[0]
        bytes_per_pixel = np.dtype(np.float32).itemsize
        
        # Memory needed per time point:
        # 1. Z-plane data (height * width)
        # 2. Region extracts ((2*mask_offset+1)^2)
        # 3. Mask and intermediate computations
        memory_per_timepoint = bytes_per_pixel * (
            height * width +  # Z-plane data
            (2*mask_offset+1)**2 +  # Region extract
            height * width * 2  # Processing overhead
        )
        
        # Use 80% of available memory
        time_chunk_size = int(0.8 * free_memory / memory_per_timepoint)
        time_chunk_size = max(1, min(time_chunk_size, time_points))
        logger.info(f"Processing in chunks of {time_chunk_size} time points")
        
        # Group neurons by z-plane to minimize data loading
        z_plane_neurons = {}
        for i, (z, y, x) in enumerate(positions):
            z = int(z)
            if z not in z_plane_neurons:
                z_plane_neurons[z] = []
            z_plane_neurons[z].append((i, int(y), int(x)))
        
        # Process each z-plane
        for z_plane, neurons in z_plane_neurons.items():
            logger.info(f"Processing {len(neurons)} neurons in z-plane {z_plane}")
            
            # Process time points in chunks
            for t_start in range(0, time_points, time_chunk_size):
                t_end = min(t_start + time_chunk_size, time_points)
                
                # Load only the required z-plane data for this time chunk
                z_plane_data = volume[t_start:t_end, z_plane]
                
                # Move to GPU and ensure float32
                z_plane_gpu = cp.asarray(z_plane_data, dtype=cp.float32)
                
                # Process each neuron in this z-plane
                for neuron_idx, y, x in neurons:
                    # Define region bounds
                    y_start = max(0, y - mask_offset)
                    y_end = min(height, y + mask_offset + 1)
                    x_start = max(0, x - mask_offset)
                    x_end = min(width, x + mask_offset + 1)
                    
                    # Adjust mask if near borders
                    mask_y_start = max(0, mask_offset - y)
                    mask_y_end = mask_height - max(0, (y + mask_offset + 1) - height)
                    mask_x_start = max(0, mask_offset - x)
                    mask_x_end = mask_width - max(0, (x + mask_offset + 1) - width)
                    
                    # Get region mask
                    region_mask = mask[mask_y_start:mask_y_end, mask_x_start:mask_x_end]
                    
                    # Extract and average the region for all time points in chunk
                    regions = z_plane_gpu[:, y_start:y_end, x_start:x_end]
                    time_series[neuron_idx, t_start:t_end] = gpu_to_cpu(
                        cp.sum(regions * region_mask, axis=(1,2)) / cp.sum(region_mask)
                    )
                
                # Clear GPU memory
                del z_plane_gpu
                cp.get_default_memory_pool().free_all_blocks()
                
                if (t_end - t_start) < time_chunk_size:
                    logger.info(f"Processed final chunk: frames {t_start} to {t_end}")
                elif t_start == 0:
                    logger.info(f"Processing chunks of {time_chunk_size} frames")
        
        logger.info("Time series extraction complete")
        
        # Convert time series to GPU for consistency with other returns
        time_series_gpu = cpu_to_gpu(time_series)
        
        # Return results
        return {
            'positions': neuron_data['positions'],
            'time_series': time_series_gpu,
            'metadata': neuron_data['metadata']
        }
        
    except Exception as e:
        logger.error(f"Error during time series extraction: {str(e)}")
        raise 