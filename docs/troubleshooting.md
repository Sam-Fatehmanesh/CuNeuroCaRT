# Troubleshooting Guide

This document covers common issues and their solutions.

## Common Issues

### 1. Memory-Related Problems

#### Out of Memory Errors
```
RuntimeError: out of memory
```

**Solutions:**
1. Reduce chunk sizes:
   ```yaml
   gpu:
     memory_limit: 0.8  # Reduce from 0.95
   ```

2. Clear GPU memory:
   ```bash
   # Before running pipeline
   nvidia-smi -r  # Requires sudo
   ```

3. Check memory usage:
   ```bash
   nvidia-smi
   ```

#### Slow Processing
**Symptoms:**
- Very small chunk sizes
- Frequent memory transfers

**Solutions:**
1. Optimize memory limit:
   ```yaml
   gpu:
     memory_limit: 0.9  # Balance between 0.8 and 0.95
   ```

2. Reduce FFT block size:
   ```yaml
   registration:
     fft_block_size: 256  # Down from 512
   ```

### 2. Registration Issues

#### Poor Motion Correction
**Symptoms:**
- Blurry output
- Visible motion artifacts

**Solutions:**
1. Adjust correlation threshold:
   ```yaml
   registration:
     phase_correlation_threshold: 0.2  # More permissive
   ```

2. Increase maximum shift:
   ```yaml
   registration:
     max_shift: 30  # Up from 20
   ```

#### Registration Failures
**Symptoms:**
- Error messages about shift computation
- NaN values in output

**Solutions:**
1. Check input data normalization
2. Adjust FFT parameters:
   ```yaml
   registration:
     fft_block_size: 1024  # More precise
     phase_correlation_threshold: 0.4  # Stricter
   ```

### 3. Neuron Detection Issues

#### Too Few Neurons Detected
**Symptoms:**
- Missing obvious neurons
- Very few detections

**Solutions:**
1. Lower detection thresholds:
   ```yaml
   detection:
     brightness_threshold: 40  # Lower threshold
     min_neuron_area: 2  # Smaller minimum size
   ```

2. Increase contrast radius:
   ```yaml
   detection:
     local_contrast_radius: 3  # Larger radius
   ```

#### Too Many False Positives
**Symptoms:**
- Detecting noise as neurons
- Unrealistic number of neurons

**Solutions:**
1. Increase thresholds:
   ```yaml
   detection:
     brightness_threshold: 80  # Higher threshold
     min_neuron_area: 4  # Larger minimum size
   ```

2. Adjust area constraints:
   ```yaml
   detection:
     max_neuron_area: 50  # More restrictive
   ```

### 4. Time Series Extraction Issues

#### Noisy Time Series
**Symptoms:**
- High-frequency noise
- Unrealistic signal fluctuations

**Solutions:**
1. Increase averaging area:
   ```yaml
   extraction:
     neighborhood_size: 4  # Larger averaging area
   ```

2. Check registration quality
3. Verify neuron detection parameters

#### Missing Signal
**Symptoms:**
- Flat time series
- Missing known activity

**Solutions:**
1. Decrease averaging area:
   ```yaml
   extraction:
     neighborhood_size: 2  # More localized
   ```

2. Check neuron positions
3. Verify brightness thresholds

### 5. CUDA and GPU Issues

#### CUDA Version Mismatch
**Symptoms:**
```
ImportError: CUDA driver version is insufficient for CUDA runtime version
```

**Solutions:**
1. Check CUDA version:
   ```bash
   nvidia-smi
   ```

2. Install matching CuPy:
   ```bash
   pip uninstall cupy-cuda11x
   pip install cupy-cuda12x  # Match your CUDA version
   ```

#### GPU Not Detected
**Symptoms:**
```
RuntimeError: CUDA initialization failed
```

**Solutions:**
1. Verify GPU visibility:
   ```bash
   nvidia-smi
   ```

2. Check CUDA installation:
   ```bash
   nvcc --version
   ```

3. Set CUDA_VISIBLE_DEVICES:
   ```bash
   export CUDA_VISIBLE_DEVICES=0
   ```

### 6. Output Issues

#### Corrupted Video
**Symptoms:**
- Video won't play
- Incomplete video file

**Solutions:**
1. Try different codec:
   ```yaml
   diagnostics:
     video_codec: "MJPG"  # Alternative codec
   ```

2. Check disk space
3. Verify OpenCV installation

#### Missing or Incomplete Results
**Symptoms:**
- Missing output files
- Incomplete HDF5 file

**Solutions:**
1. Check disk space
2. Verify output permissions:
   ```bash
   chmod -R 755 output_directory
   ```

3. Monitor progress through logs

## Debugging Tips

### 1. Enable Detailed Logging
```python
logging.basicConfig(level=logging.DEBUG)
```

### 2. Monitor GPU Usage
```bash
watch -n 1 nvidia-smi
```

### 3. Check Intermediate Results
- Inspect registered.tif
- View comparison video
- Check neuron positions in HDF5

### 4. Performance Optimization
1. Monitor timing:
   - Registration time per plane
   - Detection time per volume
   - Extraction time per neuron

2. Optimize chunk sizes:
   - Balance memory usage
   - Monitor processing speed
   - Adjust based on GPU capacity

### 5. Data Validation
1. Check input data:
   - TIFF file format
   - Dimensions match config
   - Value ranges appropriate

2. Verify intermediate results:
   - Registration quality
   - Neuron detection accuracy
   - Time series signal quality 