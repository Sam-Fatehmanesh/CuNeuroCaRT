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
    """Process a batch of frames for the comparison video."""
    # Get parameters
    positions = neuron_data['positions']
    time_series = neuron_data['time_series']
    
    # Create neuron visualization frames
    neuron_frames = create_neuron_frames_batch(positions, time_series, t_start, t_end, original_batch.shape[1:], z_plane)
    
    # Create spike visualization frames if spikes are available
    if spikes is not None:
        spike_frames = create_spike_frames_batch(positions, spikes['spikes'], t_start, t_end, original_batch.shape[1:], z_plane)
    else:
        # Create blank frames if no spikes
        spike_frames = np.zeros_like(original_batch)
    
    # Process each frame in the batch
    for i in range(len(original_batch)):
        # Create 2x2 grid
        frame = np.zeros((original_batch.shape[1]*2, original_batch.shape[2]*2), dtype=np.uint8)
        
        # Top left: Original
        frame[:original_batch.shape[1], :original_batch.shape[2]] = normalize_frames_batch(original_batch[i])
        
        # Top right: Neurons
        frame[:original_batch.shape[1], original_batch.shape[2]:] = neuron_frames[i]
        
        # Bottom right: Spikes (if available)
        frame[original_batch.shape[1]:, original_batch.shape[2]:] = spike_frames[i]
        
        # Add frame counter
        cv2.putText(frame, f"Frame {t_start+i}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 1)
        
        # Write frame
        video_writer.write(frame)

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

def generate_comparison_video(volume_data, neuron_data, config, spike_results=None):
    """Generate comparison video showing original data, detected neurons, and spikes.
    
    Parameters
    ----------
    volume_data : numpy.ndarray
        4D array (time, z, y, x)
    neuron_data : dict
        Dictionary containing neuron positions and time series
    config : dict
        Configuration dictionary
    spike_results : dict, optional
        Dictionary containing spike detection results. If None, spike visualization will be disabled.
    """
    logger.info("Generating comparison video")
    
    # Get parameters
    z_plane = config['diagnostics']['z_plane']
    fps = config['diagnostics'].get('video_fps', 10)
    codec = config['diagnostics'].get('video_codec', 'XVID')
    video_filename = config['diagnostics'].get('video_filename', 'diagnostic_comparison_video.avi')
    max_frames = config['diagnostics'].get('max_frames', None)
    
    # Get output path
    output_dir = Path(config['output']['base_dir'])
    video_path = output_dir / video_filename
    
    # Get dimensions
    time_points = volume_data.shape[0]
    height = volume_data.shape[2]
    width = volume_data.shape[3]
    
    # Limit frames if requested
    if max_frames is not None:
        time_points = min(time_points, max_frames)
        logger.info(f"Limiting video to first {time_points} frames")
    
    try:
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*codec)
        video_writer = cv2.VideoWriter(str(video_path), fourcc, fps, (width*2, height*2), False)
        
        # Process in batches
        batch_size = 100  # Adjust based on memory constraints
        for t_start in range(0, time_points, batch_size):
            t_end = min(t_start + batch_size, time_points)
            logger.debug(f"Processing frames {t_start} to {t_end}")
            
            # Get batch of original frames
            z_plane_data = read_tiff_z_plane(volume_data, z_plane)
            original_batch = z_plane_data[t_start:t_end]
            
            # Process batch
            process_video_batch(original_batch, neuron_data, spike_results, t_start, t_end, z_plane, video_writer)
            
            # Log progress
            if t_end - t_start < batch_size:
                logger.info(f"Processed final batch: frames {t_start} to {t_end}")
            elif t_start == 0:
                logger.info(f"Processing in batches of {batch_size} frames")
        
        # Release video writer
        video_writer.release()
        logger.info(f"Video saved to: {video_path}")
        
    except Exception as e:
        logger.error(f"Error generating video: {str(e)}")
        if 'video_writer' in locals():
            video_writer.release()
        raise

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