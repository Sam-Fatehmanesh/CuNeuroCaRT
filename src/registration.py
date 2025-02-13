import cupy as cp
import numpy as np
import logging
from .utils import gpu_to_cpu, cpu_to_gpu
from pathlib import Path
import faulthandler
import signal

logger = logging.getLogger(__name__)

def compute_phase_correlation(ref_frame, moving_frame):
    """Compute phase correlation between two frames using FFT."""
    # Enable faulthandler
    faulthandler.enable()
    
    # # Log pending signals
    # try:
    #     pending = signal.sigpending()
    #     logging.info(f"Pending signals before FFT: {pending}")
    # except AttributeError:
    #     logging.info("sigpending() not available on this platform")
    
    # Ensure inputs are 2D arrays and convert to float32
    ref_frame = cp.asarray(ref_frame, dtype=cp.float32)
    moving_frame = cp.asarray(moving_frame, dtype=cp.float32)
    
    if ref_frame.ndim != 2 or moving_frame.ndim != 2:
        raise ValueError(f"Input frames must be 2D arrays. Got shapes {ref_frame.shape} and {moving_frame.shape}")
    
    # Normalize inputs to [0, 1] range
    ref_frame = (ref_frame - cp.min(ref_frame)) / (cp.max(ref_frame) - cp.min(ref_frame) + 1e-10)
    moving_frame = (moving_frame - cp.min(moving_frame)) / (cp.max(moving_frame) - cp.min(moving_frame) + 1e-10)
    
    # Apply Hanning window to reduce edge effects
    window = cp.hanning(ref_frame.shape[0])[:, None] * cp.hanning(ref_frame.shape[1])[None, :]
    ref_frame = ref_frame * window
    moving_frame = moving_frame * window
    
    try:
        # Compute FFTs with explicit axes specification
        fft1 = cp.fft.fft2(ref_frame, axes=(-2, -1))
        fft2 = cp.fft.fft2(moving_frame, axes=(-2, -1))
        
        # Compute normalized cross-power spectrum
        cross_power = fft1 * cp.conj(fft2)
        magnitude = cp.abs(cross_power) + 1e-10
        cross_power_norm = cross_power / magnitude
        
        # Inverse FFT and get real component
        correlation = cp.real(cp.fft.ifft2(cross_power_norm, axes=(-2, -1)))
        
        # Find peak
        peak_idx = int(cp.argmax(correlation))  # Convert to Python int immediately
        
        # Convert to 2D indices
        peak_y, peak_x = np.unravel_index(peak_idx, correlation.shape)
        
        # Convert to shifts
        shift_y = peak_y if peak_y < correlation.shape[0]//2 else peak_y - correlation.shape[0]
        shift_x = peak_x if peak_x < correlation.shape[1]//2 else peak_x - correlation.shape[1]
        
        # Log state before GPU->CPU transfer
        # logging.info("Starting GPU->CPU transfer of peak index")
        try:
            # Use non-blocking transfer with timeout
            peak_idx_cpu = cp.asnumpy(peak_idx, stream=cp.cuda.Stream(non_blocking=True))
            peak_y, peak_x = np.unravel_index(peak_idx_cpu, correlation.shape)
        except cp.cuda.runtime.CUDARuntimeError as e:
            logging.error(f"CUDA error during transfer: {e}")
            raise
        except Exception as e:
            logging.error(f"Error during transfer: {type(e).__name__}: {e}")
            raise
            
        return shift_y, shift_x
        
    except Exception as e:
        logging.error(f"FFT computation failed: {str(e)}")
        logging.error(f"Input shapes - ref: {ref_frame.shape}, moving: {moving_frame.shape}")
        logging.error(f"Input dtypes - ref: {ref_frame.dtype}, moving: {moving_frame.dtype}")
        raise

def apply_shift(frame, shift_y, shift_x):
    """Apply shift to frame using cupy operations."""
    height, width = frame.shape
    shifted = cp.zeros_like(frame)
    
    # Calculate valid regions after shift
    src_y_start = max(0, shift_y)
    src_y_end = min(height, height + shift_y)
    src_x_start = max(0, shift_x)
    src_x_end = min(width, width + shift_x)
    
    dst_y_start = max(0, -shift_y)
    dst_y_end = min(height, height - shift_y)
    dst_x_start = max(0, -shift_x)
    dst_x_end = min(width, width - shift_x)
    
    # Copy valid region
    src_slice = frame[src_y_start:src_y_end, src_x_start:src_x_end]
    shifted[dst_y_start:dst_y_end, dst_x_start:dst_x_end] = src_slice
    
    return shifted

def process_chunk(chunk, mean_frame, max_shift):
    """Process a chunk of time points for a given z-plane."""
    chunk_size, height, width = chunk.shape
    # logger.debug(f"Processing chunk with shape: {chunk.shape}")
    # logger.debug(f"Mean frame shape: {mean_frame.shape}")
    
    registered_chunk = cp.zeros_like(chunk)
    
    for t in range(chunk_size):
        frame = chunk[t]
        # logger.debug(f"Frame shape: {frame.shape}")
        
        # Compute shift using phase correlation
        shift_y, shift_x = compute_phase_correlation(mean_frame, frame)
        # logger.debug(f"Computed shifts: y={shift_y}, x={shift_x}")
        
        # Clip shifts to maximum allowed value
        shift_y = int(cp.clip(shift_y, -max_shift, max_shift))
        shift_x = int(cp.clip(shift_x, -max_shift, max_shift))
        
        # Apply shift
        registered_chunk[t] = apply_shift(frame, shift_y, shift_x)
    
    # logger.debug(f"Registered chunk shape: {registered_chunk.shape}")
    return registered_chunk

def register_volume(volume, config):
    """Register 4D volume using phase correlation with chunked processing."""
    logger.info("Starting volume registration")
    logger.debug(f"Input volume shape: {volume.shape}")
    
    # Get parameters
    max_shift = config['registration']['max_shift']
    output_dir = Path(config['output']['base_dir']) / config['output']['registered_dir']
    output_dir.mkdir(parents=True, exist_ok=True)
    
    time_points, z_slices, height, width = volume.shape
    
    # Determine if input is on GPU
    is_gpu_input = isinstance(volume, cp.ndarray)
    logger.debug(f"Input volume is on GPU: {is_gpu_input}")
    
    # Create output array on CPU to avoid memory issues
    registered = np.zeros_like(gpu_to_cpu(volume) if is_gpu_input else volume)
    logger.debug(f"Created output array with shape: {registered.shape}")
    
    try:
        # Calculate chunk size based on available GPU memory
        total_memory = cp.cuda.runtime.memGetInfo()[1]
        bytes_per_pixel = np.dtype(volume.dtype).itemsize
        pixels_per_frame = height * width
        
        memory_per_frame = bytes_per_pixel * pixels_per_frame * 4  # 4x for processing overhead
        chunk_size = int(0.9 * total_memory / memory_per_frame)
        chunk_size = max(1, min(chunk_size, 50))
        
        logger.info(f"Processing in chunks of {chunk_size} time points")
        
        # Process z-planes
        for z in range(z_slices):
            logger.info(f"Processing z-plane {z}/{z_slices}")
            
            # Compute mean frame for this z-plane
            z_plane_data = volume[:, z, :, :]
            if not is_gpu_input:
                z_plane_data = cpu_to_gpu(z_plane_data)
            
            mean_frame = cp.mean(z_plane_data, axis=0)
            logger.debug(f"Mean frame shape: {mean_frame.shape}")
            
            # Process time points in chunks
            for t_start in range(0, time_points, chunk_size):
                t_end = min(t_start + chunk_size, time_points)
                logger.debug(f"Processing time points {t_start} to {t_end}")
                
                # Extract and process chunk
                chunk_data = z_plane_data[t_start:t_end]
                if not is_gpu_input:
                    chunk_data = cpu_to_gpu(chunk_data)
                
                registered_chunk = process_chunk(chunk_data, mean_frame, max_shift)
                logger.debug(f"Registered chunk shape: {registered_chunk.shape}")
                
                # Store results (always convert to CPU for storage)
                logger.debug(f"Storing at indices [{t_start}:{t_end}, {z}, :, :]")
                registered_chunk_cpu = gpu_to_cpu(registered_chunk)
                registered[t_start:t_end, z, :, :] = registered_chunk_cpu
                
                # Clear GPU memory
                del chunk_data, registered_chunk
                cp.get_default_memory_pool().free_all_blocks()
            
            # Clear GPU memory for this z-plane
            if not is_gpu_input:
                del z_plane_data
            del mean_frame
            cp.get_default_memory_pool().free_all_blocks()
        
        logger.info("Registration complete")
        return registered
            
    except KeyboardInterrupt:
        logger.warning("Registration interrupted by user")
        logger.info("Preserving temporary files for recovery")
        raise
    except Exception as e:
        logger.error(f"Error during registration: {str(e)}")
        raise 