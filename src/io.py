import tifffile
import h5py
import numpy as np
from pathlib import Path
import logging
from .utils import gpu_to_cpu, cpu_to_gpu, ensure_gpu_memory

logger = logging.getLogger(__name__)

def read_tiff(filepath, config):
    """Read multi-page TIFF file and organize it as a memory-mapped 4D array (time, z, y, x).
    Handles both 3D (pages, y, x) and 4D (time, z, y, x) TIFF files."""
    logger.info(f"Memory mapping TIFF file: {filepath}")
    try:
        # Create a memory map (read-only) of the TIFF
        mem = tifffile.memmap(filepath, mode='r')
        
        # Check if file is already 4D
        if len(mem.shape) == 4:
            logger.info(f"Found 4D TIFF with shape: {mem.shape}")
            volume_data = mem
        else:
            total_pages = mem.shape[0]
            
            # Get the shape of a single page
            page_shape = mem.shape[1:]
            logger.info(f"Found 3D TIFF with shape: {mem.shape} | Single page shape: {page_shape}")
            
            # Get dimensions from config
            z_slices = config['input']['z_slices_per_volume']
            time_points = config['input'].get('time_points_per_volume', total_pages // z_slices)
            
            expected_pages = time_points * z_slices
            if total_pages != expected_pages:
                logger.warning(f"Total pages ({total_pages}) does not match expected pages "
                             f"(time_points {time_points} * z_slices {z_slices} = {expected_pages})")
            
            # Reshape the memmap to 4D: (time, z, y, x)
            volume_data = mem.reshape((time_points, z_slices) + page_shape)
            
        logger.info(f"Final volume shape: {volume_data.shape}, dtype: {volume_data.dtype}")
        return volume_data  # Returns a numpy memmap array
                
    except Exception as e:
        logger.error(f"Error memory-mapping TIFF file: {str(e)}")
        logger.error(f"File path: {filepath}")
        raise

def read_chunk(volume_data, t_start, t_end, z_start, z_end):
    """Return a chunk from the memory-mapped volume data.
    
    Parameters
    ----------
    volume_data : numpy.memmap
        Memory-mapped 4D array (time, z, y, x)
    t_start, t_end : int
        Start and end time points
    z_start, z_end : int
        Start and end z-planes
    """
    # Simple slicing of the memmap - only loads requested chunk into memory
    chunk = volume_data[t_start:t_end, z_start:z_end]
    logger.debug(f"Read chunk with shape: {chunk.shape}")
    return chunk

def write_results(neuron_data, config):
    """Write neuron detection and time series results to HDF5 file."""
    output_path = Path(config['output']['base_dir']) / config['output']['results_file']
    
    logger.info(f"Writing results to: {output_path}")
    
    try:
        with h5py.File(output_path, 'w') as f:
            # Create groups for different types of data
            neurons = f.create_group('neurons')
            
            # Store neuron positions
            positions = neurons.create_dataset('positions', 
                                            data=gpu_to_cpu(neuron_data['positions']))
            
            # Store time series data
            time_series = neurons.create_dataset('time_series', 
                                               data=gpu_to_cpu(neuron_data['time_series']))
            
            # Store metadata
            neurons.attrs['total_neurons'] = len(neuron_data['positions'])
            neurons.attrs['time_points'] = neuron_data['time_series'].shape[1]
            
            logger.info(f"Saved data for {neurons.attrs['total_neurons']} neurons")
            
    except Exception as e:
        logger.error(f"Error writing results: {str(e)}")
        raise

def save_registered_volume(registered_data, config):
    """Save registered volume as a single TIFF file."""
    reg_dir = Path(config['output']['base_dir']) / config['output']['registered_dir']
    output_path = reg_dir / "registered.tif"
    
    logger.info(f"Saving registered volume to: {output_path}")
    
    try:
        # Convert to CPU if necessary
        reg_data = gpu_to_cpu(registered_data)
        
        # Save as TIFF
        tifffile.imwrite(output_path, reg_data)
        logger.info(f"Saved registered volume with shape {reg_data.shape}")
        
    except Exception as e:
        logger.error(f"Error saving registered volume: {str(e)}")
        raise 