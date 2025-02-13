# Technical Details

## Pipeline Overview

This document provides a detailed technical explanation of how the brain imaging analysis pipeline works.

## 1. Registration Process

### Phase Correlation
The registration process uses phase correlation to detect shifts between frames:

1. **Preprocessing**:
   - Input frames are normalized to [0,1] range
   - Hanning window is applied to reduce edge effects
   ```python
   ref_frame = (ref_frame - cp.min(ref_frame)) / (cp.max(ref_frame) - cp.min(ref_frame) + 1e-10)
   window = cp.hanning(ref_frame.shape[0])[:, None] * cp.hanning(ref_frame.shape[1])[None, :]
   ```

2. **FFT Computation**:
   - 2D FFT is computed for both reference and moving frames
   - Cross-power spectrum is normalized
   ```python
   fft1 = cp.fft.fft2(ref_frame)
   fft2 = cp.fft.fft2(moving_frame)
   cross_power = fft1 * cp.conj(fft2)
   cross_power_norm = cross_power / (cp.abs(cross_power) + 1e-10)
   ```

3. **Shift Detection**:
   - Inverse FFT gives correlation surface
   - Peak location indicates shift
   - Shifts are clipped to maximum allowed value

### Memory Management
- Processing is done in chunks to manage GPU memory
- Chunk size is dynamically calculated based on available memory
- Each z-plane is processed independently

## 2. Neuron Detection

### Local Contrast Enhancement
1. **Computation**:
   - For each pixel, compute minimum in local neighborhood
   - Subtract minimum from center pixel
   - This enhances local peaks (potential neurons)

2. **Parameters**:
   - `local_contrast_radius`: Controls neighborhood size
   - Larger radius can detect dimmer neurons but increases noise sensitivity

### Region Detection
1. **Thresholding**:
   - Apply brightness threshold to contrast-enhanced image
   - Binary mask creation using both contrast and intensity

2. **Connected Components**:
   - Label connected regions in binary mask
   - Filter regions by area constraints
   - Extract centroids as neuron positions

### Memory Optimization
- Mean volume computation in chunks
- Z-plane batch processing
- Automatic cleanup of GPU memory

## 3. Time Series Extraction

### Signal Extraction
1. **Region Definition**:
   - Circular mask around each neuron centroid
   - Radius defined by `neighborhood_size`
   ```python
   mask = x*x + y*y <= radius*radius
   ```

2. **Time Series Computation**:
   - Average intensity within masked region
   - Process multiple time points simultaneously
   - Handle border cases properly

### Chunked Processing
1. **Time Chunks**:
   - Process subset of time points
   - Reduce memory usage
   - Dynamic chunk size based on GPU memory

2. **Neuron Chunks**:
   - Process subset of neurons
   - Balance memory usage and parallelism

## 4. Memory Management

### GPU Memory
1. **Dynamic Chunk Sizing**:
   ```python
   total_memory = cp.cuda.runtime.memGetInfo()[1]
   memory_per_frame = bytes_per_pixel * pixels_per_frame * overhead_factor
   chunk_size = int(fraction * total_memory / memory_per_frame)
   ```

2. **Memory Cleanup**:
   - Explicit deallocation after chunk processing
   - GPU memory pool clearing
   ```python
   del temporary_data
   cp.get_default_memory_pool().free_all_blocks()
   ```

### CPU Memory
1. **Efficient Storage**:
   - Store large arrays on CPU
   - Transfer chunks to GPU as needed
   - Use memory-mapped files when appropriate

## 5. Performance Considerations

### GPU Optimization
1. **Memory Transfer**:
   - Minimize CPU-GPU transfers
   - Use asynchronous transfers when possible
   - Batch operations where possible

2. **Computation**:
   - Use CuPy for GPU-accelerated operations
   - Vectorize operations when possible
   - Balance chunk size and GPU utilization

### Pipeline Optimization
1. **Chunk Size Selection**:
   - Too small: overhead dominates
   - Too large: memory issues
   - Dynamic adjustment based on available resources

2. **Parallel Processing**:
   - Independent z-plane processing
   - Batch processing of neurons
   - Memory-aware parallelization

## 6. Error Handling

### Recovery Mechanisms
1. **Interruption Handling**:
   - Save intermediate results
   - Clean resource cleanup
   - Proper error logging

2. **Memory Management**:
   - Graceful handling of out-of-memory conditions
   - Automatic chunk size adjustment
   - Resource cleanup in error cases 