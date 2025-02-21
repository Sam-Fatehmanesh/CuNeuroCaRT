import cv2
import numpy as np
import cupy as cp
import logging
from pathlib import Path
from .utils import gpu_to_cpu, ensure_gpu_memory
import subprocess
import h5py
import tifffile

logger = logging.getLogger(__name__)

def normalize_frames_batch(frames):
    """Normalize a batch of frames to 0-255 range on GPU."""
    # Compute min/max across spatial dimensions but keep batch dimension
    frame_min = cp.min(frames, axis=(1, 2), keepdims=True)
    frame_max = cp.max(frames, axis=(1, 2), keepdims=True)
    
    # Avoid division by zero
    denominator = frame_max - frame_min
    denominator = cp.maximum(denominator, 1e-10)
    
    # Normalize to [0, 255]
    frames_norm = 255 * (frames - frame_min) / denominator
    return frames_norm.astype(cp.uint8)

def create_gaussian_kernel(radius, sigma=1.0, sharp=False):
    """Create a Gaussian kernel on GPU."""
    size = 2 * radius + 1
    y, x = cp.ogrid[-radius:radius+1, -radius:radius+1]
    if sharp:
        # Sharper falloff for spikes
        kernel = cp.exp(-(x*x + y*y) / (2 * (sigma/2)**2))
    else:
        kernel = cp.exp(-(x*x + y*y) / (2 * sigma**2))
    return kernel.astype(cp.float32)

def create_neuron_frames_batch(positions, time_series, t_start, t_end, shape, z_plane):
    """Create frames showing neuron activities for a batch of time points using vectorized operations."""
    height, width = shape
    batch_size = t_end - t_start
    
    # Filter neurons in this z-plane
    z_mask = positions[:, 0] == z_plane
    if not cp.any(z_mask):
        return cp.zeros((batch_size, height, width), dtype=cp.float32)
    
    plane_positions = positions[z_mask]
    plane_series = time_series[z_mask]
    n_neurons = len(plane_positions)
    
    # Get and normalize intensities for all neurons across the batch
    raw_intensities = plane_series[:, t_start:t_end]  # shape: (n_neurons, batch_size)
    series_min = cp.min(plane_series, axis=1, keepdims=True)
    series_max = cp.max(plane_series, axis=1, keepdims=True)
    denominator = cp.maximum(series_max - series_min, 1e-10)
    intensities = 255 * (raw_intensities - series_min) / denominator  # shape: (n_neurons, batch_size)
    
    # Precompute Gaussian kernel
    radius = 2
    kernel = create_gaussian_kernel(radius, sigma=1.0, sharp=False)  # shape: (kernel_size, kernel_size)
    kernel_size = 2 * radius + 1
    
    # Get neuron positions
    y_pos = plane_positions[:, 1].astype(cp.int32)  # shape: (n_neurons,)
    x_pos = plane_positions[:, 2].astype(cp.int32)  # shape: (n_neurons,)
    
    # Create position indices for the kernel around each neuron
    y_indices = cp.arange(-radius, radius + 1)  # shape: (kernel_size,)
    x_indices = cp.arange(-radius, radius + 1)  # shape: (kernel_size,)
    
    # Create meshgrid for kernel positions
    Y_kernel, X_kernel = cp.meshgrid(y_indices, x_indices, indexing='ij')
    
    # Expand dimensions for broadcasting
    # Shape transformations for vectorized operations:
    # y_pos: (n_neurons, 1, 1, 1) - for broadcasting across kernel positions and batch
    # intensities: (n_neurons, batch_size, 1, 1) - for broadcasting across kernel positions
    # Y_kernel: (1, 1, kernel_size, kernel_size) - for broadcasting across neurons and batch
    y_pos = y_pos.reshape(-1, 1, 1, 1)  # (n_neurons, 1, 1, 1)
    x_pos = x_pos.reshape(-1, 1, 1, 1)  # (n_neurons, 1, 1, 1)
    intensities = intensities.reshape(n_neurons, batch_size, 1, 1)  # (n_neurons, batch_size, 1, 1)
    kernel = kernel.reshape(1, 1, kernel_size, kernel_size)  # (1, 1, kernel_size, kernel_size)
    
    # Calculate all Y, X positions for all neurons at once
    Y = y_pos + Y_kernel.reshape(1, 1, kernel_size, kernel_size)  # (n_neurons, 1, kernel_size, kernel_size)
    X = x_pos + X_kernel.reshape(1, 1, kernel_size, kernel_size)  # (n_neurons, 1, kernel_size, kernel_size)
    
    # Create valid mask
    valid = (Y >= 0) & (Y < height) & (X >= 0) & (X < width)  # (n_neurons, 1, kernel_size, kernel_size)
    
    # Initialize output frames
    frames = cp.zeros((batch_size, height, width), dtype=cp.float32)
    
    # Calculate contributions for all neurons and all frames at once
    kernel_values = kernel * intensities  # (n_neurons, batch_size, kernel_size, kernel_size)
    
    # Apply mask
    kernel_values = cp.where(valid, kernel_values, 0)
    
    # Reshape coordinates and values for scatter_add
    valid_mask = valid.reshape(n_neurons, -1)  # (n_neurons, kernel_size * kernel_size)
    Y_flat = Y.reshape(n_neurons, -1)  # (n_neurons, kernel_size * kernel_size)
    X_flat = X.reshape(n_neurons, -1)  # (n_neurons, kernel_size * kernel_size)
    kernel_values = kernel_values.reshape(n_neurons, batch_size, -1)  # (n_neurons, batch_size, kernel_size * kernel_size)
    
    # For each frame in batch (still needed but much more efficient now)
    for b in range(batch_size):
        # Get all valid positions and values for this frame
        frame_values = kernel_values[:, b]  # (n_neurons, kernel_size * kernel_size)
        
        # Get valid indices
        valid_indices = valid_mask
        Y_valid = Y_flat[valid_indices]
        X_valid = X_flat[valid_indices]
        values_valid = frame_values[valid_indices]
        
        # Add all contributions at once
        frames[b].scatter_add((Y_valid, X_valid), values_valid)
    
    return frames

def create_spike_frames_batch(positions, spikes, t_start, t_end, shape, z_plane):
    """Create frames showing spikes for a batch of time points using vectorized operations."""
    height, width = shape
    batch_size = t_end - t_start
    
    # Filter neurons in this z-plane
    z_mask = positions[:, 0] == z_plane
    if not cp.any(z_mask):
        return cp.zeros((batch_size, height, width), dtype=cp.float32)
    
    plane_positions = positions[z_mask]
    plane_spikes = spikes[z_mask, t_start:t_end]  # shape: (n_neurons, batch_size)
    n_neurons = len(plane_positions)
    
    # Precompute spike kernel (sharper than neuron kernel)
    radius = 3
    kernel = create_gaussian_kernel(radius, sigma=1.0, sharp=True)  # shape: (kernel_size, kernel_size)
    kernel_size = 2 * radius + 1
    
    # Get neuron positions
    y_pos = plane_positions[:, 1].astype(cp.int32)  # shape: (n_neurons,)
    x_pos = plane_positions[:, 2].astype(cp.int32)  # shape: (n_neurons,)
    
    # Create position indices for the kernel around each neuron
    y_indices = cp.arange(-radius, radius + 1)  # shape: (kernel_size,)
    x_indices = cp.arange(-radius, radius + 1)  # shape: (kernel_size,)
    
    # Create meshgrid for kernel positions
    Y_kernel, X_kernel = cp.meshgrid(y_indices, x_indices, indexing='ij')
    
    # Expand dimensions for broadcasting
    y_pos = y_pos.reshape(-1, 1, 1, 1)  # (n_neurons, 1, 1, 1)
    x_pos = x_pos.reshape(-1, 1, 1, 1)  # (n_neurons, 1, 1, 1)
    plane_spikes = plane_spikes.reshape(n_neurons, batch_size, 1, 1)  # (n_neurons, batch_size, 1, 1)
    kernel = kernel.reshape(1, 1, kernel_size, kernel_size)  # (1, 1, kernel_size, kernel_size)
    
    # Calculate all Y, X positions for all neurons at once
    Y = y_pos + Y_kernel.reshape(1, 1, kernel_size, kernel_size)  # (n_neurons, 1, kernel_size, kernel_size)
    X = x_pos + X_kernel.reshape(1, 1, kernel_size, kernel_size)  # (n_neurons, 1, kernel_size, kernel_size)
    
    # Create valid mask
    valid = (Y >= 0) & (Y < height) & (X >= 0) & (X < width)  # (n_neurons, 1, kernel_size, kernel_size)
    
    # Initialize output frames
    frames = cp.zeros((batch_size, height, width), dtype=cp.float32)
    
    # Calculate contributions for all neurons and all frames at once
    # Multiply by 255 for full intensity where spikes occur
    kernel_values = kernel * (plane_spikes > 0).astype(cp.float32) * 255  # (n_neurons, batch_size, kernel_size, kernel_size)
    
    # Apply mask
    kernel_values = cp.where(valid, kernel_values, 0)
    
    # Reshape coordinates and values for scatter_add
    valid_mask = valid.reshape(n_neurons, -1)  # (n_neurons, kernel_size * kernel_size)
    Y_flat = Y.reshape(n_neurons, -1)  # (n_neurons, kernel_size * kernel_size)
    X_flat = X.reshape(n_neurons, -1)  # (n_neurons, kernel_size * kernel_size)
    kernel_values = kernel_values.reshape(n_neurons, batch_size, -1)  # (n_neurons, batch_size, kernel_size * kernel_size)
    
    # For each frame in batch (still needed but much more efficient now)
    for b in range(batch_size):
        # Get all valid positions and values for this frame
        frame_values = kernel_values[:, b]  # (n_neurons, kernel_size * kernel_size)
        
        # Get valid indices
        valid_indices = valid_mask & (frame_values > 0).reshape(valid_mask.shape)
        Y_valid = Y_flat[valid_indices]
        X_valid = X_flat[valid_indices]
        values_valid = frame_values[valid_indices]
        
        # Add all contributions at once
        frames[b].scatter_add((Y_valid, X_valid), values_valid)
    
    return frames

def process_video_batch(original_batch, neuron_data, spikes, t_start, t_end, z_plane, video_writer):
    """Process a batch of frames for video generation."""
    batch_size, height, width = original_batch.shape
    
    # Move batch to GPU and normalize
    original_gpu = cp.asarray(original_batch)
    original_norm = normalize_frames_batch(original_gpu)
    
    # Create neuron visualization frames
    neuron_frames = create_neuron_frames_batch(
        cp.asarray(neuron_data['positions']),
        cp.asarray(neuron_data['time_series']),
        t_start, t_end,
        (height, width),
        z_plane
    )
    neuron_frames_norm = normalize_frames_batch(neuron_frames)
    
    # Create spike frames if available
    if spikes is not None:
        spike_frames = create_spike_frames_batch(
            cp.asarray(neuron_data['positions']),
            cp.asarray(spikes),
            t_start, t_end,
            (height, width),
            z_plane
        )
        spike_frames_norm = normalize_frames_batch(spike_frames)
    else:
        spike_frames_norm = cp.zeros((batch_size, height, width), dtype=cp.uint8)
    
    # Process each frame in the batch
    for b in range(batch_size):
        # Create comparison frame (on CPU since we need to use OpenCV)
        video_height = height * 2 + 10
        video_width = width * 2 + 10
        comparison = np.zeros((video_height, video_width), dtype=np.uint8)
        
        # Transfer normalized frames to CPU and compose the layout
        comparison[:height, :width] = gpu_to_cpu(original_norm[b])
        comparison[:height, -width:] = gpu_to_cpu(neuron_frames_norm[b])
        comparison[-height:, -width:] = gpu_to_cpu(spike_frames_norm[b])
        
        # Convert to RGB for adding colored text
        comparison_rgb = cv2.cvtColor(comparison, cv2.COLOR_GRAY2BGR)
        
        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1
        
        cv2.putText(comparison_rgb, 'Original', (10, 20), font, font_scale, (0,255,0), font_thickness)
        cv2.putText(comparison_rgb, 'Detected Neurons', (width+20, 20), font, font_scale, (255,0,0), font_thickness)
        cv2.putText(comparison_rgb, 'Spike Activity', (width+20, height+30), font, font_scale, (0,0,255), font_thickness)
        
        # Add timestamp
        time_str = f"Time: {t_start+b}/{neuron_data['time_series'].shape[1]}"
        cv2.putText(comparison_rgb, time_str, (10, video_height-10), font, font_scale, (255,255,255), font_thickness)
        
        # Write frame
        video_writer.write(comparison_rgb)
    
    # Clear GPU memory
    del original_gpu, neuron_frames, spike_frames_norm
    cp.get_default_memory_pool().free_all_blocks()

def read_tiff_z_plane(volume_data, z_plane):
    """Extract the specified z-plane from a memory-mapped volume.
    
    Parameters
    ----------
    volume_data : numpy.memmap
        Memory-mapped 4D array (time, z, y, x)
    z_plane : int
        Z-plane to extract
    """
    logger.info(f"Extracting z-plane {z_plane} from volume data")
    
    # Extract z-plane data
    z_plane_data = volume_data[:, z_plane, :, :]
    
    # Ensure data is float32 and normalized to a reasonable range
    z_plane_data = z_plane_data.astype(np.float32)
    
    # Normalize each frame individually to preserve temporal dynamics
    min_vals = z_plane_data.min(axis=(1,2), keepdims=True)
    max_vals = z_plane_data.max(axis=(1,2), keepdims=True)
    
    # Avoid division by zero
    denom = np.maximum(max_vals - min_vals, 1e-10)
    z_plane_data = (z_plane_data - min_vals) * (255.0 / denom)
    
    logger.info(f"Loaded z-plane data with shape: {z_plane_data.shape}")
    return z_plane_data

def generate_comparison_video(volume_data, neuron_data, config):
    """Generate comparison video with original data and neuron detection+spikes.
    
    Parameters
    ----------
    volume_data : numpy.memmap
        Memory-mapped 4D array (time, z, y, x)
    neuron_data : dict
        Dictionary containing neuron positions and time series
    config : dict
        Configuration dictionary containing mandatory 'generate_video' parameter
    """
    # Check if video generation is enabled
    if not config['diagnostics'].get('generate_video', False):
        logger.info("Video generation is disabled in config")
        return
        
    logger.info("Generating comparison video")
    video = None
    
    try:
        # Get parameters
        z_plane = config['diagnostics']['z_plane']
        fps = config['diagnostics']['video_fps']
        codec = config['diagnostics']['video_codec']
        output_path = Path(config['output']['base_dir']) / "diagnostic_comparison_video.avi"
        
        # Get optional max_frames parameter
        max_frames = config['diagnostics'].get('max_frames', None)
        
        # Load spike data if available
        spike_path = Path(config['output']['base_dir']) / 'spike_neuron_data.h5'
        if spike_path.exists():
            logger.info("Loading spike data for visualization")
            with h5py.File(spike_path, 'r') as f:
                spikes = f['neurons/spikes'][()]
        else:
            logger.warning("No spike data found, video will only show original and reconstructed data")
            spikes = None
        
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Read only the required z-plane
        z_plane_data = read_tiff_z_plane(volume_data, z_plane)
        time_points, height, width = z_plane_data.shape
        
        # Limit number of frames if max_frames is set
        if max_frames is not None:
            time_points = min(time_points, max_frames)
            z_plane_data = z_plane_data[:time_points]
            logger.info(f"Limiting video to first {time_points} frames")
        
        # Calculate batch size based on available GPU memory
        total_memory = cp.cuda.runtime.memGetInfo()[0]
        bytes_per_pixel = np.dtype(z_plane_data.dtype).itemsize
        
        # Memory needed per frame:
        # 1. Original frame
        # 2. Reconstructed frame
        # 3. Spike frame
        # 4. RGB output frame
        # Plus overhead for processing
        memory_per_frame = bytes_per_pixel * (
            height * width +  # Original
            height * width +  # Reconstructed
            height * width +  # Spikes
            height * width * 3 +  # RGB output
            height * width * 4  # Processing overhead
        )
        
        batch_size = int(0.5 * total_memory / memory_per_frame)
        batch_size = max(1, min(batch_size, time_points)) 
        logger.info(f"Processing video in batches of {batch_size} frames")
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*codec)
        video_width = width * 2 + 10
        video_height = height * 2 + 10
        video = cv2.VideoWriter(str(output_path), fourcc, fps, (video_width, video_height))
        
        if not video.isOpened():
            raise RuntimeError(f"Failed to open video writer for {output_path}")
        
        # Process batches
        for t_start in range(0, time_points, batch_size):
            t_end = min(t_start + batch_size, time_points)
            logger.info(f"Processing frames {t_start+1} to {t_end}/{time_points}")
            
            # Get batch of frames
            batch_data = z_plane_data[t_start:t_end]
            
            # Process batch
            process_video_batch(
                batch_data,
                neuron_data,
                spikes,
                t_start,
                t_end,
                z_plane,
                video
            )
            
            # Force garbage collection
            import gc
            gc.collect()
        
        logger.info(f"Video saved to: {output_path}")
        
    except Exception as e:
        logger.error(f"Error generating comparison video: {str(e)}")
        raise
        
    finally:
        # Ensure video writer is always closed
        if video is not None:
            video.release()

def log_system_state():
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