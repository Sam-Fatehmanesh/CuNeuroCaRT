import argparse
import logging
import os
import time
from pathlib import Path
import h5py
import numpy as np
import cupy as cp
import cv2
from tqdm import tqdm

# Import pipeline modules
from src.utils import load_config, init_logger, setup_output_dirs, ensure_gpu_memory
from src.io import read_tiff, write_results, save_registered_volume
from src.registration import register_volume
from src.detection import detect_neurons
from src.extraction import extract_time_series
from src.diagnostics import generate_comparison_video

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_grab_config(tiff_file):
    """
    Create a configuration for GRAB sensor data processing.
    
    Parameters:
    -----------
    tiff_file : str
        Path to the TIFF file
    
    Returns:
    --------
    dict
        Configuration dictionary
    """
    # Create output directory based on input filename
    base_name = os.path.splitext(os.path.basename(tiff_file))[0]
    output_dir = os.path.join("output", base_name)
    
    # Create configuration
    config = {
        "input": {
            "tiff_file": tiff_file,
            "z_slices_per_volume": 30  # From the filename, we know there are 30 slices
        },
        "output": {
            "base_dir": output_dir,
            "registered_dir": "registered",
            "detection_dir": "detection",
            "results_file": "neuron_data.h5"
        },
        "registration": {
            "fft_block_size": 512,
            "phase_correlation_threshold": 0.3,
            "max_shift": 20
        },
        "detection": {
            "local_contrast_radius": 2,
            "brightness_threshold": 60,
            "min_neuron_area": 2,
            "max_neuron_area": 100
        },
        "extraction": {
            "neighborhood_size": 3
        },
        "diagnostics": {
            "generate_video": True,
            "z_plane": 15,  # Middle slice
            "video_fps": 20,
            "video_codec": "XVID",
            "video_filename": "diagnostic_comparison_video.avi",
            "max_frames": 600
        },
        "gpu": {
            "device_id": 0,
            "memory_limit": 0.95
        }
    }
    
    return config

def process_grab_data(tiff_file, custom_config=None):
    """
    Process GRAB sensor data using the CuNeuroCaRT pipeline.
    
    Parameters:
    -----------
    tiff_file : str
        Path to the TIFF file
    custom_config : dict, optional
        Custom configuration to override defaults
    """
    start_time = time.time()
    
    # Create configuration
    config = create_grab_config(tiff_file)
    
    # Override with custom config if provided
    if custom_config:
        for section, values in custom_config.items():
            if section in config:
                config[section].update(values)
            else:
                config[section] = values
    
    # Create output directories
    base_dir, reg_dir = setup_output_dirs(config)
    
    # Check for existing results
    results_path = Path(config['output']['base_dir']) / config['output']['results_file']
    registered_path = reg_dir / "registered.tif"
    
    if results_path.exists() and registered_path.exists():
        logger.info("Found existing results file and registered volume")
        with h5py.File(results_path, 'r') as f:
            results = {
                'positions': f['neurons/positions'][()],
                'time_series': f['neurons/time_series'][()],
                'metadata': [{'z': p[0], 'y': p[1], 'x': p[2]} for p in f['neurons/positions'][()]]
            }
        # Load registered volume as memmap
        registered_volume = read_tiff(str(registered_path), config)
        logger.info(f"Loaded existing results with {len(results['positions'])} neurons")
    else:
        # Read input TIFF file
        logger.info("Reading input data")
        volume = read_tiff(tiff_file, config)
        
        # Register volume
        logger.info("Performing volume registration")
        try:
            registered_volume = register_volume(volume, config)
            logger.info("Registration completed successfully")
        except Exception as e:
            logger.error(f"Registration failed: {str(e)}")
            raise
        
        # Save registered volume
        logger.info("Saving registered volume")
        save_registered_volume(registered_volume, config)
        
        # Free memory
        del volume
        
        # Detect neurons
        logger.info("Detecting neurons")
        try:
            neuron_data = detect_neurons(registered_volume, config)
            logger.info(f"Detection completed. Found {len(neuron_data['positions'])} neurons")
        except Exception as e:
            logger.error(f"Neuron detection failed: {str(e)}")
            raise
        
        # Extract time series
        logger.info("Extracting neuron time series")
        try:
            results = extract_time_series(registered_volume, neuron_data, config)
            logger.info("Time series extraction completed")
        except Exception as e:
            logger.error(f"Time series extraction failed: {str(e)}")
            raise
        
        # Save results
        logger.info("Saving results")
        write_results(results, config)
    
    # Generate comparison video if requested
    if config.get('diagnostics', {}).get('generate_video', True):
        logger.info("Generating diagnostic video")
        try:
            generate_comparison_video(registered_volume, results, config)
            logger.info("Video generation completed")
        except Exception as e:
            logger.error(f"Video generation failed: {str(e)}")
            raise
    
    # Calculate and log processing time
    end_time = time.time()
    processing_time = end_time - start_time
    logger.info(f"Total processing time: {processing_time:.2f} seconds")
    
    return results, registered_volume

def create_summary_report(results, config, output_file):
    """
    Create a summary report of the processed data.
    
    Parameters:
    -----------
    results : dict
        Results from the pipeline
    config : dict
        Configuration dictionary
    output_file : str
        Path to the output report file
    """
    logger.info(f"Creating summary report: {output_file}")
    
    # Extract information
    num_neurons = len(results['positions'])
    time_points = results['time_series'].shape[1]
    
    # Create report
    with open(output_file, 'w') as f:
        f.write("# GRAB Sensor Data Processing Report\n\n")
        f.write(f"## Input File\n")
        f.write(f"- File: {config['input']['tiff_file']}\n")
        f.write(f"- Z-slices per volume: {config['input']['z_slices_per_volume']}\n\n")
        
        f.write(f"## Processing Results\n")
        f.write(f"- Detected neurons: {num_neurons}\n")
        f.write(f"- Time points: {time_points}\n\n")
        
        f.write(f"## Neuron Distribution\n")
        # Group neurons by z-slice
        z_slices = {}
        for pos in results['positions']:
            z = int(pos[0])
            if z not in z_slices:
                z_slices[z] = 0
            z_slices[z] += 1
        
        # Write distribution
        f.write("| Z-slice | Neuron Count |\n")
        f.write("|---------|-------------|\n")
        for z in sorted(z_slices.keys()):
            f.write(f"| {z} | {z_slices[z]} |\n")
    
    logger.info(f"Summary report created: {output_file}")

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Process GRAB sensor data')
    parser.add_argument('--tiff_file', type=str, default="20230331_HuC_GRAB_5HT4_fish1_5uWside_50msExp_30slices_4x4bin_fullRun_30minE3_30minDMSO_1hr_MT_1.tif",
                        help='Path to the TIFF file')
    args = parser.parse_args()
    
    # Process the data
    results, registered_volume = process_grab_data(args.tiff_file)
    
    # Create summary report
    config = create_grab_config(args.tiff_file)
    report_file = os.path.join(config['output']['base_dir'], "processing_report.md")
    create_summary_report(results, config, report_file)
    
    logger.info("Processing completed successfully!") 