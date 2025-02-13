import cv2
import numpy as np
import cupy as cp
import logging
from pathlib import Path
from .utils import gpu_to_cpu, ensure_gpu_memory
import subprocess

logger = logging.getLogger(__name__)

def normalize_frame(frame):
    """Normalize frame to 0-255 range for video."""
    frame_min = frame.min()
    frame_max = frame.max()
    if frame_max > frame_min:
        frame = 255 * (frame - frame_min) / (frame_max - frame_min)
    return frame.astype(np.uint8)

def create_neuron_frame(positions, time_series, t, shape, z_plane):
    """Create a frame showing neuron activities at time t."""
    height, width = shape
    frame = np.zeros((height, width), dtype=np.float32)
    
    # For each neuron in this z-plane
    for i, (z, y, x) in enumerate(positions):
        if int(z) == z_plane:
            # Get intensity from time series and normalize it
            raw_intensity = time_series[i, t]
            # Get the full time series for this neuron for normalization
            neuron_series = time_series[i, :]
            min_intensity = np.min(neuron_series)
            max_intensity = np.max(neuron_series)
            
            # Normalize intensity to [0, 1] range
            if max_intensity > min_intensity:
                intensity = (raw_intensity - min_intensity) / (max_intensity - min_intensity)
            else:
                intensity = 0
            
            # Scale to make it more visible
            intensity = intensity * 255
            
            # Draw a Gaussian spot at neuron location
            y, x = int(y), int(x)
            radius = 2
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    yy = y + dy
                    xx = x + dx
                    if 0 <= yy < height and 0 <= xx < width:
                        # Create a Gaussian falloff from center
                        dist = np.sqrt(dy**2 + dx**2)
                        if dist <= radius:
                            gaussian_factor = np.exp(-(dist**2) / (2 * (radius/2)**2))
                            frame[yy, xx] = max(frame[yy, xx], intensity * gaussian_factor)
    
    return frame

def generate_comparison_video(volume, neuron_data, config):
    """Generate side-by-side comparison video of original vs. reconstructed data."""
    logger.info("Generating comparison video")
    video = None
    
    try:
        # Get parameters
        z_plane = config['diagnostics']['z_plane']
        fps = config['diagnostics']['video_fps']
        codec = config['diagnostics']['video_codec']
        output_path = Path(config['output']['base_dir']) / config['diagnostics']['video_filename']
        
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Get volume dimensions
        time_points, z_slices, height, width = volume.shape
        
        # Calculate chunk size based on available GPU memory
        total_memory = cp.cuda.runtime.memGetInfo()[1]
        bytes_per_pixel = np.dtype(volume.dtype).itemsize
        
        # Calculate memory needed per frame:
        # 1. Original frame (height × width)
        # 2. Reconstructed frame (height × width)
        # 3. Comparison frame (height × (2×width + 10) × 3 channels)
        # 4. Temporary arrays for processing
        memory_per_frame = bytes_per_pixel * (
            height * width +  # Original frame
            height * width +  # Reconstructed frame
            height * (2 * width + 10) * 3 +  # RGB comparison frame
            height * width * 2  # Processing overhead
        )
        
        # Use 20% of available memory
        chunk_size = int(0.9 * total_memory / memory_per_frame)
        chunk_size = max(1, min(chunk_size, time_points))  # Ensure valid chunk size
        
        logger.info(f"Processing video in chunks of {chunk_size} frames")
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*codec)
        video_width = width * 2 + 10  # Space for two frames side by side with gap
        video = cv2.VideoWriter(str(output_path), fourcc, fps, (video_width, height))
        
        if not video.isOpened():
            raise RuntimeError(f"Failed to open video writer for {output_path}")
        
        # Convert data to CPU if needed (one chunk at a time)
        positions = gpu_to_cpu(neuron_data['positions'])
        time_series = gpu_to_cpu(neuron_data['time_series'])
        
        # Process chunks of frames
        for chunk_start in range(0, time_points, chunk_size):
            chunk_end = min(chunk_start + chunk_size, time_points)
            logger.info(f"Processing frames {chunk_start+1} to {chunk_end}/{time_points}")
            
            # Get chunk of volume data
            if isinstance(volume, cp.ndarray):
                chunk_volume = gpu_to_cpu(volume[chunk_start:chunk_end, z_plane])
            else:
                chunk_volume = volume[chunk_start:chunk_end, z_plane]
            
            # Process each frame in the chunk
            for t in range(chunk_start, chunk_end):
                frame_idx = t - chunk_start
                
                # Get original frame
                original = chunk_volume[frame_idx]
                original_norm = normalize_frame(original)
                
                # Create reconstructed frame from neuron data
                reconstructed = create_neuron_frame(positions, time_series, t, (height, width), z_plane)
                reconstructed_norm = normalize_frame(reconstructed)
                
                # Create side-by-side comparison
                comparison = np.zeros((height, video_width), dtype=np.uint8)
                comparison[:, :width] = original_norm
                comparison[:, -width:] = reconstructed_norm
                
                # Convert to RGB for video
                comparison_rgb = cv2.cvtColor(comparison, cv2.COLOR_GRAY2BGR)
                
                # Write frame
                video.write(comparison_rgb)
                
                # Clear some memory
                del comparison, comparison_rgb
            
            # Clear chunk memory
            del chunk_volume
            
            # Force garbage collection
            import gc
            gc.collect()
            
            # Clear GPU memory
            cp.get_default_memory_pool().free_all_blocks()
        
        # Properly close video writer
        if video is not None:
            video.release()
        
        logger.info(f"Video saved to: {output_path}")
        
    except KeyboardInterrupt:
        logger.warning("Video generation interrupted by user")
        if video is not None:
            video.release()
        raise
        
    except Exception as e:
        logger.error(f"Error generating comparison video: {str(e)}")
        if video is not None:
            video.release()
        raise
        
    finally:
        # Ensure video writer is always closed
        if video is not None:
            video.release()

def log_system_state(self):
    # Check for GPU errors
    try:
        gpu_errors = subprocess.check_output(['nvidia-smi', '-q', '-d', 'ERROR'], text=True, stderr=subprocess.DEVNULL)
        if 'No errors found' not in gpu_errors:
            logging.error("GPU Errors detected:")
            logging.error(gpu_errors)
    except subprocess.CalledProcessError:
        logging.warning("GPU error check not supported on this system.")
    except Exception as e:
        logging.warning(f"GPU error check failed with exception: {e}") 