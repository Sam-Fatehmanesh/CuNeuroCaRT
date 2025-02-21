# Technical Details

## Pipeline Overview

This document provides a detailed technical explanation of how the brain imaging analysis pipeline works.

## 1. Registration Process

### Data Loading and Organization
1. **TIFF Loading**:
   - Input is a multi-page TIFF file containing sequential z-planes
   - Pages are automatically organized into volumes based on z_slices_per_volume
   - Number of time points is calculated as: total_pages / z_slices_per_volume
   - Manual override of time points possible via config if needed
   - Data is reshaped into 4D array: (time, z, y, x)

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

## 4. Spike Detection

### OASIS Algorithm
1. **Algorithm Overview**:
   - Online Active Set method for Sparse Inference (OASIS) [1]
   - Deconvolves calcium traces into spike trains
   - Handles both L1 penalty and hard threshold approaches
   - Implementation based on Friedrich et al. (2017) [1]

2. **GPU Implementation**:
   - Parallel processing of neurons using CUDA
   - One block per neuron, one thread per block
   - Memory-efficient pool management:
     ```
     Global Memory:
     ├── Pool Buffer (per neuron)
     │   ├── v_pool[T]      # Pool values
     │   ├── w_pool[T]      # Pool weights
     │   ├── start_idx[T]   # Pool start indices
     │   └── pool_length[T] # Pool lengths
     ```
   - Dynamic pool merging to prevent memory overflow
   - Automatic lambda optimization using pilot neurons

3. **Parameters**:
   ```yaml
   spike_detection:
     decay_constant: 0.95  # Calcium decay time constant (g)
     minimum_spike: 0.1    # Minimum spike amplitude (smin)
     lambda: null          # L1 sparsity penalty (auto-optimized)
     noise_std: 0.1       # For lambda optimization
   ```

4. **Processing Steps**:
   - Trace normalization
   - Lambda determination from pilot neurons
   - Parallel OASIS processing on GPU
   - Binary spike train generation
   - Denormalization of denoised traces

### Memory Management
1. **Global Memory Usage**:
   - Pool data stored in global memory buffer
   - 4 arrays per neuron (values, weights, indices, lengths)
   - Buffer size: n_neurons × n_timepoints × 4

2. **Shared Memory Usage**:
   - Small buffer (128 doubles) for temporary calculations
   - Single counter for active pools
   - Minimized to stay within hardware limits

3. **Pool Management**:
   - Dynamic merging of pools when count approaches limit
   - Oldest pools merged first to maintain memory efficiency
   - Automatic handling of long time series

### Performance Considerations
1. **GPU Optimization**:
   - Coalesced memory access patterns
   - Minimal thread divergence
   - Efficient pool merging strategy
   - Balanced memory usage vs speed

2. **Scalability**:
   - Handles arbitrary length time series
   - Automatic memory management
   - Parallel processing of multiple neurons

3. **Precision**:
   - Double precision for numerical stability
   - Careful handling of exponential terms
   - Robust spike threshold determination

### Output Format
```
spike_neuron_data.h5
├── neurons/
    ├── positions              # (N, 3) array
    ├── time_series           # (N, T) raw traces
    ├── denoised_time_series  # (N, T) denoised
    ├── spikes               # (N, T) binary spikes
    └── attributes
        ├── total_neurons
        └── time_points
```

## 5. Diagnostics

### Video Generation
1. **Layout**:
   - 2x2 grid arrangement:
     ```
     ┌────────────┬────────────┐
     │  Original  │  Detected  │
     │            │  Neurons   │
     ├────────────┼────────────┤
     │   (blank)  │   Spike    │
     │            │  Activity  │
     └────────────┴────────────┘
     ```
   - Original data in natural size in top-left
   - Bottom-left quadrant left empty
   - Detected neurons and spikes on right

2. **Features**:
   - Frame counter
   - Color-coded labels
   - Gaussian spot visualization
   - Real-time spike display

3. **Memory Optimization**:
   - Chunk-based processing
   - Dynamic memory allocation
   - Efficient frame composition

### Performance Monitoring
1. **GPU Optimization**:
   - Minimize CPU-GPU transfers
   - Use asynchronous transfers when possible
   - Batch operations where possible

2. **Computation**:
   - Use CuPy for GPU-accelerated operations
   - Vectorize operations when possible
   - Balance chunk size and GPU utilization

3. **Pipeline Optimization**:
   - Chunk Size Selection
   - Parallel Processing
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

## References

[1] Friedrich J, Zhou P, Paninski L (2017) Fast online deconvolution of calcium imaging data. PLOS Computational Biology 13(3): e1005423. https://doi.org/10.1371/journal.pcbi.1005423 