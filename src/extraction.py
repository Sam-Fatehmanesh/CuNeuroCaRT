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
    """Extract time series for each detected neuron."""
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
    
    # Initialize time series array on CPU to save GPU memory
    time_series = np.zeros((n_neurons, time_points), dtype=volume.dtype)
    
    try:
        # Calculate chunk sizes based on available GPU memory
        total_memory = cp.cuda.runtime.memGetInfo()[1]
        bytes_per_pixel = np.dtype(volume.dtype).itemsize
        
        # Calculate memory needed per time point
        # 1. Volume chunk
        # 2. Region extracts
        # 3. Mask operations
        memory_per_timepoint = bytes_per_pixel * (height * width * z_slices + 
                                                n_neurons * (2*mask_offset+1)**2)
        
        # Use 20% of available memory for processing
        time_chunk_size = int(0.9 * total_memory / memory_per_timepoint)
        time_chunk_size = max(1, min(time_chunk_size, time_points))
        
        # Calculate neuron chunk size
        memory_per_neuron = bytes_per_pixel * time_chunk_size * (2*mask_offset+1)**2
        neuron_chunk_size = int(0.9 * total_memory / memory_per_neuron)
        neuron_chunk_size = max(1, min(neuron_chunk_size, n_neurons))
        
        logger.info(f"Processing in chunks of {time_chunk_size} time points and {neuron_chunk_size} neurons")
        
        # Process in chunks
        for n_start in range(0, n_neurons, neuron_chunk_size):
            n_end = min(n_start + neuron_chunk_size, n_neurons)
            logger.info(f"Processing neurons {n_start+1} to {n_end}/{n_neurons}")
            
            # Get positions for this chunk of neurons
            chunk_positions = positions[n_start:n_end]
            
            for t_start in range(0, time_points, time_chunk_size):
                t_end = min(t_start + time_chunk_size, time_points)
                
                # Move volume chunk to GPU
                volume_chunk = cpu_to_gpu(volume[t_start:t_end])
                
                # Process each neuron in the chunk
                for i, (z, y, x) in enumerate(chunk_positions):
                    # Convert coordinates to integers
                    z, y, x = int(z), int(y), int(x)
                    
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
                    
                    # Extract time series using masked average
                    region_mask = mask[mask_y_start:mask_y_end, mask_x_start:mask_x_end]
                    
                    # Process all time points in this chunk at once
                    regions = volume_chunk[:, z, y_start:y_end, x_start:x_end]
                    time_series[n_start + i, t_start:t_end] = gpu_to_cpu(
                        cp.sum(regions * region_mask, axis=(1,2)) / cp.sum(region_mask)
                    )
                
                # Clear GPU memory after processing time chunk
                del volume_chunk
                cp.get_default_memory_pool().free_all_blocks()
            
            # Log progress
            logger.info(f"Completed {n_end}/{n_neurons} neurons")
        
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