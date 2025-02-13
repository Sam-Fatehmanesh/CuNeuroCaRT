import tifffile
import h5py
import numpy as np
from pathlib import Path
import logging
from .utils import gpu_to_cpu, cpu_to_gpu, ensure_gpu_memory

logger = logging.getLogger(__name__)

def read_tiff(filepath, config):
    """Read multi-page TIFF file and organize it as a 4D array (time, z, y, x)."""
    logger.info(f"Reading TIFF file: {filepath}")
    
    try:
        # Read the TIFF file
        with tifffile.TiffFile(filepath) as tif:
            # Read all pages into a single array
            data = tif.asarray()
            logger.debug(f"Raw data shape: {data.shape}")
            
            # Handle different dimension cases
            if len(data.shape) == 3:  # (pages, height, width)
                # Check if this is raw input data or already processed data
                if str(filepath).endswith('registered.tif'):
                    # This is already processed data, just return as is
                    logger.info(f"Loading pre-processed volume with shape: {data.shape}")
                    return data
                else:
                    # This is raw input data, reshape according to config
                    z_slices = config['input']['z_slices_per_volume']
                    total_pages = data.shape[0]
                    time_points = total_pages // z_slices
                    height, width = data.shape[1:]
                    volume_data = data.reshape(time_points, z_slices, height, width)
            elif len(data.shape) == 4:  # Already in (time, z, y, x) format
                logger.info("Data already in correct 4D format")
                volume_data = data
            else:
                raise ValueError(f"Unexpected TIFF dimensions: {data.shape}")
            
            logger.info(f"Loaded volume with shape: {volume_data.shape}")
            
            # Check GPU memory and transfer if possible
            if ensure_gpu_memory(volume_data.shape, volume_data.dtype):
                volume_data = cpu_to_gpu(volume_data)
            
            return volume_data
            
    except Exception as e:
        logger.error(f"Error reading TIFF file: {str(e)}")
        logger.error(f"File path: {filepath}")
        if 'data' in locals():
            logger.error(f"Data shape: {data.shape}")
        raise

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