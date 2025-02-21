import argparse
import logging
from pathlib import Path
import h5py
import sys
import traceback
import signal
import os
import subprocess
import time
import cupy as cp

from .utils import load_config, init_logger, setup_output_dirs
from .io import read_tiff, write_results, save_registered_volume
from .registration import register_volume
from .detection import detect_neurons
from .extraction import extract_time_series
from .spike_detection import detect_spikes, write_spike_results
from .diagnostics import generate_comparison_video

class DiagnosticHandler:
    def __init__(self):
        self.start_time = time.time()
        self.last_checkpoint = self.start_time
        self.stage = "initialization"
        self.gpu_memory_start = None
        self.original_sigint = signal.getsignal(signal.SIGINT)
        
    def log_gpu_state(self):
        """Log GPU memory usage and state."""
        try:
            # Get GPU info from nvidia-smi
            nvidia_smi = subprocess.check_output(['nvidia-smi'], text=True)
            logging.info("GPU State from nvidia-smi:")
            for line in nvidia_smi.split('\n'):
                if any(x in line.lower() for x in ['mib', 'gpu', 'memory', 'volatile']):
                    logging.info(f"nvidia-smi: {line.strip()}")
            
            # Get CuPy memory info
            mem_info = cp.cuda.runtime.memGetInfo()
            free_mem = mem_info[0]
            total_mem = mem_info[1]
            used_mem = total_mem - free_mem
            logging.info(f"CuPy Memory - Used: {used_mem/1e9:.2f}GB, Free: {free_mem/1e9:.2f}GB, Total: {total_mem/1e9:.2f}GB")
            
            # Get GPU driver info
            driver_info = subprocess.check_output(['nvidia-smi', '--query-gpu=driver_version,gpu_name,temperature.gpu,power.draw,power.limit', '--format=csv,noheader'], text=True)
            logging.info(f"GPU Driver Info: {driver_info.strip()}")
            
            return used_mem, free_mem, total_mem
        except Exception as e:
            logging.error(f"Error getting GPU info: {str(e)}")
            return None, None, None
            
    def log_system_state(self):
        """Log system state including journal and GPU logs."""
        try:
            # Try to get recent journal logs
            try:
                journal = subprocess.check_output(['journalctl', '--no-pager', '-n', '10'], text=True)
                logging.info("Recent system journal entries:")
                for line in journal.split('\n'):
                    if any(x in line.lower() for x in ['error', 'warn', 'fail', 'cuda', 'nvidia', 'gpu']):
                        logging.info(f"journal: {line}")
            except Exception as e:
                logging.error(f"Could not read journal: {str(e)}")
            
            # Get process memory info
            with open('/proc/self/status') as f:
                for line in f:
                    if 'Vm' in line:
                        logging.info(f"Process memory: {line.strip()}")
            
            # Check for GPU errors
            try:
                gpu_errors = subprocess.check_output(['nvidia-smi', '-q', '-d', 'ERROR'], text=True)
                if 'No errors found' not in gpu_errors:
                    logging.error("GPU Errors detected:")
                    logging.error(gpu_errors)
            except Exception as e:
                logging.error(f"Could not check GPU errors: {str(e)}")
                
        except Exception as e:
            logging.error(f"Error getting system state: {str(e)}")
    
    def check_point(self, new_stage):
        """Log checkpoint timing and state."""
        current_time = time.time()
        duration = current_time - self.last_checkpoint
        total_duration = current_time - self.start_time
        logging.info(f"Stage '{self.stage}' completed in {duration:.2f}s (Total: {total_duration:.2f}s)")
        self.stage = new_stage
        self.last_checkpoint = current_time
        self.log_gpu_state()
    
    def handle_interrupt(self, signum, frame):
        """Custom interrupt handler."""
        logging.error(f"\nInterrupt detected during stage: {self.stage}")
        self.log_system_state()
        self.log_gpu_state()
        logging.error("Stack trace at interrupt:")
        traceback.print_stack(frame)
        # Restore original handler and re-raise
        signal.signal(signal.SIGINT, self.original_sigint)
        raise KeyboardInterrupt

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Process 3D brain imaging data')
    parser.add_argument('-c', '--config', type=str, required=True,
                      help='Path to configuration YAML file')
    return parser.parse_args()

def main():
    """Main pipeline function."""
    # Parse arguments and load configuration
    args = parse_args()
    config = load_config(args.config)
    
    # Initialize logger
    logger = init_logger()
    logger.info("Starting brain imaging processing pipeline")
    
    try:
        # Create output directories
        base_dir, reg_dir = setup_output_dirs(config)
        
        # Check for existing results
        results_path = Path(config['output']['base_dir']) / config['output']['results_file']
        if results_path.exists():
            logger.info("Found existing results file, skipping to video generation")
            with h5py.File(results_path, 'r') as f:
                results = {
                    'positions': f['neurons/positions'][()],
                    'time_series': f['neurons/time_series'][()],
                    'metadata': [{'z': p[0], 'y': p[1], 'x': p[2]} for p in f['neurons/positions'][()]]
                }
            registered_volume = None  # Will be loaded during video generation
        else:
            # Read input TIFF file
            logger.info("Reading input data")
            volume = read_tiff(config['input']['tiff_file'], config)
            
            # Register volume
            logger.info("Performing volume registration")
            try:
                registered_volume = register_volume(volume, config)
                logger.info("Registration completed successfully")
            except KeyboardInterrupt:
                logger.warning("Registration interrupted. You can resume later from the last saved state.")
                return
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
            except KeyboardInterrupt:
                logger.warning("Neuron detection interrupted.")
                return
            except Exception as e:
                logger.error(f"Neuron detection failed: {str(e)}")
                raise
            
            # Extract time series
            logger.info("Extracting neuron time series")
            try:
                results = extract_time_series(registered_volume, neuron_data, config)
                logger.info("Time series extraction completed")
            except KeyboardInterrupt:
                logger.warning("Time series extraction interrupted.")
                return
            except Exception as e:
                logger.error(f"Time series extraction failed: {str(e)}")
                raise
            
            # Save results
            logger.info("Saving results")
            write_results(results, config)
            
            # Free memory before video generation
            del registered_volume
            registered_volume = None
        
        # Generate comparison video if requested
        if config.get('diagnostics', {}).get('generate_video', True):
            logger.info("Generating diagnostic video")
            try:
                # Load volume if needed
                if registered_volume is None:
                    logger.info("Loading registered volume for video generation")
                    reg_dir = Path(config['output']['base_dir']) / config['output']['registered_dir']
                    registered_volume = read_tiff(str(reg_dir / "registered.tif"), config)
                
                generate_comparison_video(registered_volume, results, config)
                logger.info("Video generation completed")
            except KeyboardInterrupt:
                logger.warning("Video generation interrupted.")
                return
            except Exception as e:
                logger.error(f"Video generation failed: {str(e)}")
                raise
        
        logger.info("Pipeline completed successfully")
        
    except KeyboardInterrupt:
        logger.warning("Pipeline interrupted by user")
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise

def diagnostic_main():
    """Diagnostic wrapper around main pipeline."""
    handler = DiagnosticHandler()
    current_sigint = signal.getsignal(signal.SIGINT)
    logging.info(f"Initial SIGINT handler: {current_sigint}")
    signal.signal(signal.SIGINT, handler.handle_interrupt)
    
    try:
        # Parse arguments and load configuration
        args = parse_args()
        config = load_config(args.config)
        
        # Initialize logger with more detailed format
        logger = init_logger()
        logger.info("Starting brain imaging processing pipeline with diagnostics")
        
        # Log initial GPU state
        handler.log_gpu_state()
        handler.log_system_state()
        
        try:
            # Create output directories
            handler.stage = "setup"
            base_dir, reg_dir = setup_output_dirs(config)
            handler.check_point("input")
            
            # Check for existing results
            results_path = Path(config['output']['base_dir']) / config['output']['results_file']
            spike_results_path = Path(config['output']['base_dir']) / 'spike_neuron_data.h5'
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
                handler.check_point("results_loaded")
                
                # Perform spike detection if not already done
                if not spike_results_path.exists():
                    handler.stage = "spike_detection"
                    try:
                        spike_results = detect_spikes(results, config)
                        write_spike_results(spike_results, config)
                        handler.check_point("spike_detection_complete")
                    except Exception as e:
                        logger.error(f"Spike detection failed: {str(e)}")
                        handler.log_system_state()
                        raise
            else:
                # Read input TIFF file
                handler.stage = "reading_input"
                volume = read_tiff(config['input']['tiff_file'], config)
                handler.check_point("registration")
                
                # Register volume
                logger.info("Performing volume registration")
                try:
                    registered_volume = register_volume(volume, config)
                    logger.info("Registration completed successfully")
                    handler.check_point("registration_complete")
                except Exception as e:
                    logger.error(f"Registration failed: {str(e)}")
                    handler.log_system_state()
                    raise
                
                # Save registered volume
                handler.stage = "saving_registration"
                save_registered_volume(registered_volume, config)
                
                # Free memory
                del volume
                handler.check_point("detection")
                
                # Detect neurons
                logger.info("Detecting neurons")
                try:
                    neuron_data = detect_neurons(registered_volume, config)
                    logger.info(f"Detection completed. Found {len(neuron_data['positions'])} neurons")
                    handler.check_point("detection_complete")
                except Exception as e:
                    logger.error(f"Neuron detection failed: {str(e)}")
                    handler.log_system_state()
                    raise
                
                # Extract time series
                handler.stage = "extraction"
                try:
                    results = extract_time_series(registered_volume, neuron_data, config)
                    logger.info("Time series extraction completed")
                    handler.check_point("extraction_complete")
                except Exception as e:
                    logger.error(f"Time series extraction failed: {str(e)}")
                    handler.log_system_state()
                    raise
                
                # Save results
                handler.stage = "saving_results"
                write_results(results, config)
                
                # Perform spike detection
                handler.stage = "spike_detection"
                try:
                    spike_results = detect_spikes(results, config)
                    write_spike_results(spike_results, config)
                    handler.check_point("spike_detection_complete")
                except Exception as e:
                    logger.error(f"Spike detection failed: {str(e)}")
                    handler.log_system_state()
                    raise
            
            # Generate comparison video if requested
            if config.get('diagnostics', {}).get('generate_video', True):
                handler.stage = "video_generation"
                try:
                    # Pass the memory-mapped array directly
                    generate_comparison_video(registered_volume, results, config)
                    logger.info("Video generation completed")
                    handler.check_point("complete")
                except Exception as e:
                    logger.error(f"Video generation failed: {str(e)}")
                    handler.log_system_state()
                    raise
            
            logger.info("Pipeline completed successfully")
            
        except KeyboardInterrupt:
            logger.warning(f"Pipeline interrupted during stage: {handler.stage}")
            handler.log_system_state()
            return
        except Exception as e:
            logger.error(f"Pipeline failed during stage {handler.stage}: {str(e)}")
            handler.log_system_state()
            raise
            
    finally:
        # Restore original signal handler
        signal.signal(signal.SIGINT, current_sigint)

if __name__ == '__main__':
    diagnostic_main() 