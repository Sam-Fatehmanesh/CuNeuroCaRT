import cupy as cp
import numpy as np
from skimage import measure
import logging
import triton
import triton.language as tl
from .utils import gpu_to_cpu, cpu_to_gpu

logger = logging.getLogger(__name__)

@triton.jit
def local_contrast_kernel(
    input_ptr,
    output_ptr,
    width: tl.constexpr,
    height: tl.constexpr,
    radius: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Triton kernel for computing local contrast."""
    # Get program ID and compute position
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE
    
    # Create block-wide position indices
    row = (offset + tl.arange(0, BLOCK_SIZE)) // width
    col = (offset + tl.arange(0, BLOCK_SIZE)) % width
    
    # Create mask for valid positions
    is_valid = (row < height) & (col < width)
    
    # Load center values
    center_vals = tl.load(input_ptr + offset + tl.arange(0, BLOCK_SIZE), mask=is_valid, other=0.0)
    min_vals = center_vals
    
    # Compute local contrast
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            # Compute neighbor coordinates
            n_row = row + dy
            n_col = col + dx
            
            # Check bounds
            is_valid_n = is_valid & (n_row >= 0) & (n_row < height) & (n_col >= 0) & (n_col < width)
            
            # Load neighbor values
            if tl.sum(is_valid_n) > 0:
                n_offset = n_row * width + n_col
                n_vals = tl.load(input_ptr + n_offset, mask=is_valid_n, other=float('inf'))
                min_vals = tl.minimum(min_vals, n_vals)
    
    # Store results
    contrast = center_vals - min_vals
    tl.store(output_ptr + offset + tl.arange(0, BLOCK_SIZE), contrast, mask=is_valid)

def compute_local_contrast(image, radius):
    """Compute local contrast using CuPy operations."""
    logger.debug("Starting local contrast computation")
    height, width = image.shape
    logger.debug(f"Input shape: {image.shape}, dtype: {image.dtype}")
    
    # Ensure input is float32
    if image.dtype != cp.float32:
        logger.debug(f"Converting input from {image.dtype} to float32")
        image = image.astype(cp.float32)
    
    # Initialize output
    output = cp.zeros_like(image)
    
    # Create a sliding window view for minimum computation
    pad_width = ((radius, radius), (radius, radius))
    padded = cp.pad(image, pad_width, mode='edge')
    
    # Compute local minima using a sliding window approach
    min_vals = cp.ones_like(image) * float('inf')
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            y_start = radius + dy
            y_end = y_start + height
            x_start = radius + dx
            x_end = x_start + width
            window = padded[y_start:y_end, x_start:x_end]
            min_vals = cp.minimum(min_vals, window)
    
    # Compute contrast
    contrast = image - min_vals
    
    logger.debug("Local contrast computation completed")
    return contrast

def detect_neurons(volume, config):
    """Detect neurons in the registered volume."""
    logger.info("Starting neuron detection")
    
    # Get parameters
    contrast_radius = config['detection']['local_contrast_radius']
    brightness_threshold = config['detection']['brightness_threshold']
    min_area = config['detection']['min_neuron_area']
    max_area = config['detection']['max_neuron_area']
    
    time_points, z_slices, height, width = volume.shape
    
    # Store detected neuron information
    neurons = []
    
    try:
        # Calculate chunk size based on available GPU memory
        total_memory = cp.cuda.runtime.memGetInfo()[1]
        bytes_per_pixel = np.dtype(volume.dtype).itemsize
        pixels_per_frame = height * width * z_slices
        
        # Account for GPU memory needed for processing:
        # 1. Input chunk
        # 2. Mean computation
        # 3. Local contrast computation (includes padded array)
        # 4. Binary mask
        memory_per_frame = bytes_per_pixel * pixels_per_frame * 4  # 4x for processing overhead
        chunk_size = int(0.9 * total_memory / memory_per_frame)  # Use 90% of available memory
        chunk_size = max(1, min(chunk_size, time_points))  # Ensure valid chunk size
        
        logger.info(f"Processing in chunks of {chunk_size} time points")
        
        # Initialize mean volume
        mean_volume = np.zeros((z_slices, height, width), dtype=np.float32)
        
        logger.info("Computing mean volume in chunks")
        for t_start in range(0, time_points, chunk_size):
            t_end = min(t_start + chunk_size, time_points)
            chunk = volume[t_start:t_end]
            
            # Process chunk on GPU
            with cp.cuda.Device(0):
                # Move chunk to GPU and compute mean
                chunk_gpu = cpu_to_gpu(chunk)
                chunk_mean = cp.mean(chunk_gpu, axis=0)
                mean_volume += gpu_to_cpu(chunk_mean) * (t_end - t_start) / time_points
                
                # Clear GPU memory immediately
                del chunk_gpu, chunk_mean
                cp.get_default_memory_pool().free_all_blocks()
        
        logger.info("Processing each z-plane")
        # Calculate z-plane batch size based on remaining GPU memory
        total_memory = cp.cuda.runtime.memGetInfo()[1]
        memory_per_zplane = bytes_per_pixel * height * width * 4  # 4x for processing overhead
        z_batch_size = int(0.9 * total_memory / memory_per_zplane)  # Use 20% of available memory
        z_batch_size = max(1, min(z_batch_size, z_slices))
        
        logger.info(f"Processing z-planes in batches of {z_batch_size}")
        
        for z_start in range(0, z_slices, z_batch_size):
            z_end = min(z_start + z_batch_size, z_slices)
            logger.info(f"Processing z-planes {z_start} to {z_end-1}")
            
            for z in range(z_start, z_end):
                # Move mean image for this z-plane to GPU
                mean_image = cpu_to_gpu(mean_volume[z])
                
                # Compute local contrast
                contrast = compute_local_contrast(mean_image, contrast_radius)
                
                # Threshold the contrast image
                binary = (contrast > brightness_threshold) & (mean_image > brightness_threshold)
                
                # Convert to CPU for connected component analysis
                binary_cpu = gpu_to_cpu(binary)
                
                # Clear GPU memory
                del mean_image, contrast, binary
                cp.get_default_memory_pool().free_all_blocks()
                
                # Find connected components
                labels = measure.label(binary_cpu)
                regions = measure.regionprops(labels)
                
                # Filter regions by area and store neuron information
                for region in regions:
                    if min_area <= region.area <= max_area:
                        y, x = region.centroid
                        neurons.append({
                            'z': z,
                            'y': y,
                            'x': x,
                            'area': region.area
                        })
                
                # Clear CPU memory
                del binary_cpu, labels, regions
            
            # Log progress
            logger.info(f"Found {len(neurons)} neurons so far")
            
            # Force garbage collection
            import gc
            gc.collect()
        
        n_neurons = len(neurons)
        logger.info(f"Detected {n_neurons} neurons")
        
        if n_neurons == 0:
            logger.warning("No neurons detected! Check detection parameters in config.")
            return {'positions': np.array([]), 'metadata': []}
        
        # Convert to numpy arrays for easier handling
        positions = np.array([[n['z'], n['y'], n['x']] for n in neurons])
        
        return {
            'positions': positions,
            'metadata': neurons
        }
        
    except Exception as e:
        logger.error(f"Error during neuron detection: {str(e)}")
        raise 