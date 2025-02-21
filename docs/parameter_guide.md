# Parameter Guide

This document provides detailed information about each configuration parameter and how to tune them for optimal results.

## Input/Output Parameters

### Input Settings
```yaml
input:
  tiff_file: "/path/to/data.tif"
  z_slices_per_volume: 10
  # time_points_per_volume is optional
```

- **tiff_file**: Path to input TIFF stack
  - Must be a multi-page TIFF file
  - Typically raw calcium imaging data
  - File size = total_pages × height × width × bytes_per_pixel

- **z_slices_per_volume**: Number of z-planes in each volume
  - Must match your microscopy setup
  - Affects memory usage and processing time
  - Used to reshape raw pages into 4D volumes

- **time_points_per_volume**: (Optional) Number of time points
  - Automatically calculated as: total_tiff_pages / z_slices_per_volume
  - Only specify if automatic calculation needs to be overridden
  - Example use case: when TIFF contains extra pages (calibration frames, metadata)
  - If provided, a warning will be logged if it differs from the calculated value

### Output Settings
```yaml
output:
  base_dir: "/path/to/output"
  registered_dir: "registered"
  results_file: "neuron_data.h5"
```

- **base_dir**: Base output directory
  - Will be created if it doesn't exist
  - Should have enough disk space for results

- **registered_dir**: Subdirectory for registered data
  - Contains motion-corrected TIFF file
  - Size similar to input file

- **results_file**: HDF5 file for results
  - Contains neuron positions and time series
  - Much smaller than TIFF files

## Registration Parameters

```yaml
registration:
  fft_block_size: 512
  phase_correlation_threshold: 0.3
  max_shift: 20
```

- **fft_block_size**: Size for FFT computation
  - Larger values: more precise but slower
  - Should be power of 2 for efficiency
  - Typical values: 256, 512, 1024

- **phase_correlation_threshold**: Correlation threshold
  - Range: 0.0 to 1.0
  - Higher values: more strict matching
  - Lower values: more permissive matching
  - Recommended range: 0.2 - 0.4

- **max_shift**: Maximum allowed pixel shift
  - Limits maximum motion correction
  - Prevents large erroneous shifts
  - Should match expected motion range
  - Typical values: 10-30 pixels

## Neuron Detection Parameters

```yaml
detection:
  local_contrast_radius: 2
  brightness_threshold: 60
  min_neuron_area: 2
  max_neuron_area: 100
```

- **local_contrast_radius**: Radius for contrast enhancement
  - Affects neuron detection sensitivity
  - Larger values: better for dim neurons
  - Smaller values: better spatial resolution
  - Typical values: 2-4 pixels

- **brightness_threshold**: Minimum brightness
  - Range: 0-255 (after normalization)
  - Higher: fewer false positives
  - Lower: catches dimmer neurons
  - Start with 60, adjust based on results

- **min_neuron_area**: Minimum region size
  - In pixels
  - Filters out noise and artifacts
  - Should match smallest expected neuron
  - Typical values: 2-4 pixels

- **max_neuron_area**: Maximum region size
  - In pixels
  - Filters out large artifacts
  - Should match largest expected neuron
  - Typical values: 50-200 pixels

## Time Series Extraction Parameters

```yaml
extraction:
  neighborhood_size: 3
```

- **neighborhood_size**: Radius for signal extraction
  - Defines area around neuron center
  - Larger: more spatial averaging
  - Smaller: more precise localization
  - Typical values: 2-4 pixels

## Diagnostics Parameters

```yaml
diagnostics:
  z_plane: 7
  video_fps: 1
  video_codec: "XVID"
  video_filename: "comparison_video.avi"
```

- **z_plane**: Z-plane for video generation
  - Usually middle plane (z_slices/2)
  - Should show good neuron visibility
  - Range: 0 to z_slices-1

- **video_fps**: Frames per second
  - Affects playback speed
  - Higher: faster playback
  - Lower: easier to inspect details

- **video_codec**: Video compression codec
  - "XVID": good compression, widely compatible
  - Other options: "MJPG", "X264"
  - Affects file size and quality

- **video_filename**: Output video name
  - Will be saved in base_dir
  - Common formats: .avi, .mp4

## GPU Settings

```yaml
gpu:
  device_id: 0
  memory_limit: 0.95
```

- **device_id**: GPU device to use
  - 0 for single GPU systems
  - Range: 0 to number_of_gpus - 1

- **memory_limit**: Maximum GPU memory fraction
  - Range: 0.0 to 1.0
  - Higher: more memory for processing
  - Lower: safer, leaves memory for system
  - Recommended: 0.8 - 0.95

## Spike Detection Parameters

```yaml
spike_detection:
  decay_constant: 0.95
  minimum_spike: 0.1
  lambda: null
  noise_std: 0.1
```

- **decay_constant**: Calcium decay time constant (g)
  - Range: 0.8 to 0.98
  - Controls calcium signal decay rate
  - Higher values: slower decay
  - Lower values: faster decay
  - Typical values: 0.95-0.97

- **minimum_spike**: Minimum spike amplitude
  - Range: 0.05 to 0.5
  - Threshold for spike detection
  - Set to null to use L1 penalty instead
  - Higher values: fewer false positives
  - Lower values: more sensitive detection

- **lambda**: L1 sparsity penalty
  - Range: null or 0.0 to 10.0
  - Controls spike train sparsity
  - null: auto-optimize using noise_std
  - Higher values: sparser solutions
  - Lower values: more spikes detected

- **noise_std**: Noise standard deviation
  - Range: 0.01 to 0.5
  - Used for lambda optimization
  - Estimate from quiet periods
  - Higher values: more regularization
  - Lower values: less regularization

## Parameter Tuning Guide

### For Weak Signals
```yaml
detection:
  local_contrast_radius: 3
  brightness_threshold: 40
  min_neuron_area: 2
extraction:
  neighborhood_size: 4
```

### For Strong Signals
```yaml
detection:
  local_contrast_radius: 2
  brightness_threshold: 80
  min_neuron_area: 4
extraction:
  neighborhood_size: 2
```

### For Memory-Limited Systems
```yaml
gpu:
  memory_limit: 0.8
registration:
  fft_block_size: 256
```

### For High-Motion Data
```yaml
registration:
  max_shift: 30
  phase_correlation_threshold: 0.2
```

### For Noisy Data
```yaml
spike_detection:
  decay_constant: 0.97
  minimum_spike: 0.2
  lambda: null
  noise_std: 0.2
```

### For Clean Data
```yaml
spike_detection:
  decay_constant: 0.95
  minimum_spike: 0.08
  lambda: null
  noise_std: 0.05
```

### For Fast Events
```yaml
spike_detection:
  decay_constant: 0.9
  minimum_spike: 0.15
  lambda: null
  noise_std: 0.1
```

### For Slow Events
```yaml
spike_detection:
  decay_constant: 0.98
  minimum_spike: 0.1
  lambda: null
  noise_std: 0.1
``` 