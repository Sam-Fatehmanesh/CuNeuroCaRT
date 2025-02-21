import cupy as cp
import numpy as np
from skimage import measure
import logging
import triton
import triton.language as tl
from .utils import gpu_to_cpu, cpu_to_gpu
from pathlib import Path
import cv2

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

def detect_neurons(volume_data, config):
    """Detect neurons in a 4D volume using local contrast and thresholding."""
    logger.info("Starting neuron detection")
    
    # Get parameters from config with defaults
    detection_config = config.get('detection', {})
    output_config = config.get('output', {})
    
    # Detection parameters with defaults
    local_contrast_radius = detection_config.get('local_contrast_radius', 2)
    brightness_threshold = detection_config.get('brightness_threshold', 60)
    min_size = detection_config.get('min_neuron_area', 2)
    max_size = detection_config.get('max_neuron_area', 100)
    
    # Output directory setup
    base_dir = Path(output_config.get('base_dir', 'output'))
    detection_dir = output_config.get('detection_dir', 'detection')
    output_dir = base_dir / detection_dir
    
    # Log parameters
    logger.info(f"Detection parameters:")
    logger.info(f"  Local contrast radius: {local_contrast_radius}")
    logger.info(f"  Brightness threshold: {brightness_threshold}")
    logger.info(f"  Min neuron area: {min_size}")
    logger.info(f"  Max neuron area: {max_size}")
    logger.info(f"  Output directory: {output_dir}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get shape and dtype from volume_data
    time_points, z_slices, height, width = volume_data.shape
    dtype = volume_data.dtype
    
    # Initialize results array
    neuron_locations = []
    
    try:
        # Calculate optimal chunk size based on available GPU memory
        free_memory = cp.cuda.runtime.memGetInfo()[0]
        bytes_per_pixel = np.dtype(dtype).itemsize
        
        # Memory needed per frame:
        # 1. Input frame
        # 2. Local contrast computation
        # 3. Binary mask
        # 4. Connected components
        # Plus overhead for intermediate computations
        memory_per_frame = bytes_per_pixel * (
            height * width +  # Input frame
            height * width * 2 +  # Local contrast (float32)
            height * width +  # Binary mask
            height * width +  # Connected components
            height * width * 2  # Overhead
        )
        
        # Use 80% of available memory for safety
        chunk_size = int(0.8 * free_memory / memory_per_frame)
        chunk_size = max(1, min(chunk_size, time_points))  # Ensure valid chunk size
        logger.info(f"Processing in chunks of {chunk_size} frames")
        
        # Process z-planes
        for z in range(z_slices):
            logger.info(f"Processing z-plane {z}/{z_slices}")
            
            # Initialize mean frame computation
            mean_frame = None
            frames_processed = 0
            
            # Process time points in chunks to compute mean frame
            for t_start in range(0, time_points, chunk_size):
                t_end = min(t_start + chunk_size, time_points)
                
                # Get chunk data directly from memmap
                chunk_data = volume_data[t_start:t_end, z]
                
                # Move to GPU and convert to float32 for computations
                chunk_gpu = cp.asarray(chunk_data, dtype=cp.float32)
                
                # Update mean frame
                if mean_frame is None:
                    mean_frame = cp.sum(chunk_gpu, axis=0, dtype=cp.float32)
                else:
                    mean_frame += cp.sum(chunk_gpu, axis=0, dtype=cp.float32)
                frames_processed += len(chunk_gpu)
                
                # Clear GPU memory
                del chunk_gpu
                cp.get_default_memory_pool().free_all_blocks()
            
            # Finalize mean frame
            mean_frame = mean_frame / float(frames_processed)  # Explicit float division
            
            # Normalize mean frame to [0, 255] range for consistent thresholding
            mean_min = cp.min(mean_frame)
            mean_max = cp.max(mean_frame)
            if mean_max > mean_min:
                mean_frame = ((mean_frame - mean_min) * (255.0 / (mean_max - mean_min))).astype(cp.float32)
            
            # Compute local contrast
            local_contrast = compute_local_contrast(mean_frame, local_contrast_radius)
            
            # Normalize local contrast to [0, 255] range
            contrast_min = cp.min(local_contrast)
            contrast_max = cp.max(local_contrast)
            if contrast_max > contrast_min:
                local_contrast = ((local_contrast - contrast_min) * (255.0 / (contrast_max - contrast_min))).astype(cp.float32)
            
            # Apply threshold and convert to uint8
            binary_mask = (local_contrast > brightness_threshold).astype(cp.uint8)
            
            # Find connected components
            binary_mask_cpu = cp.asnumpy(binary_mask)
            num_features, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask_cpu)
            
            # Filter components by size and store locations
            for i in range(1, num_features):  # Skip background (label 0)
                area = stats[i, cv2.CC_STAT_AREA]
                if min_size <= area <= max_size:  # Apply both min and max size filters
                    neuron_locations.append({
                        'z': z,
                        'y': centroids[i][1],
                        'x': centroids[i][0],
                        'size': area
                    })
            
            # Clear GPU memory
            del mean_frame, local_contrast, binary_mask
            cp.get_default_memory_pool().free_all_blocks()
            
            # Log progress
            logger.info(f"Found {len(neuron_locations)} neurons in z-plane {z+1}/{z_slices}")
        
        # Save neuron locations to file
        locations_file = output_dir / "neuron_locations.npy"
        np.save(locations_file, np.array([[n['z'], n['y'], n['x']] for n in neuron_locations]))
        logger.info(f"Saved neuron locations to {locations_file}")
        
        logger.info(f"Detection complete. Found {len(neuron_locations)} total neurons")
        return {
            'positions': np.array([[n['z'], n['y'], n['x']] for n in neuron_locations]),
            'metadata': neuron_locations
        }
            
    except KeyboardInterrupt:
        logger.warning("Detection interrupted by user")
        logger.info("Preserving temporary files for recovery")
        raise
    except Exception as e:
        logger.error(f"Error during detection: {str(e)}")
        raise 