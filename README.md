# CuNeuroCaRT (CUDA Neuron Calcium Registration & Tracking)

A high-performance, GPU-accelerated pipeline for processing 3D brain calcium imaging data. CuNeuroCaRT leverages CUDA technology to provide fast and efficient analysis of large-scale neural recordings, implementing registration, neuron detection, and time series extraction. The implementation is inspired by the Kawashima et al. (Cell, 2016) approach.

## Key Features

- ⚡ GPU-accelerated image registration using phase correlation
- 🧠 Automatic neuron detection with local contrast enhancement
- 📊 Efficient time series extraction for detected neurons
- 🎥 Real-time diagnostic video generation
- 💾 Memory-efficient processing with dynamic chunk sizing
- 🚀 CUDA optimization via CuPy and Triton

## Performance

- Processes large 4D datasets (time × z-planes × height × width)
- Dynamic GPU memory management for optimal performance
- Parallel processing of multiple z-planes and time points
- Typical speedup: 5-10x faster than CPU implementations

## Requirements

- CUDA-capable NVIDIA GPU
- CUDA 11.x or later
- Anaconda or Miniconda
- Python 3.8+

## Quick Start

1. Clone this repository:
   ```bash
   git clone <repository-url>
   cd CuNeuroCaRT
   ```

2. Set up the environment:
   ```bash
   chmod +x setup_env.sh
   ./setup_env.sh
   ```

3. Activate the environment:
   ```bash
   conda activate brain_reg
   ```

4. Run the pipeline:
   ```bash
   python -m src.main -c config.yaml
   ```

## Pipeline Overview

1. **Registration** 🔄
   - Motion correction using phase correlation
   - GPU-accelerated FFT computations
   - Sub-pixel precision alignment

2. **Neuron Detection** 🔍
   - Local contrast enhancement
   - Connected component analysis
   - Size and brightness filtering

3. **Time Series Extraction** 📈
   - Signal extraction from detected neurons
   - Circular averaging around centroids
   - Efficient batch processing

4. **Diagnostics** 📊
   - Side-by-side comparison videos
   - Activity visualization
   - Performance metrics

## Output Files

- `registered.tif`: Motion-corrected 4D volume
- `neuron_data.h5`: HDF5 file containing:
  ```
  neuron_data.h5
  ├── neurons/
  │   ├── positions         # (z, y, x) coordinates
  │   ├── time_series      # Fluorescence traces
  │   └── attributes
  │       ├── total_neurons
  │       └── time_points
  ```
- `comparison_video.avi`: Diagnostic visualization

## Configuration

Example configuration (config.yaml):
```yaml
input:
  tiff_file: "/path/to/data.tif"
  z_slices_per_volume: 10
  time_points_per_volume: 768

output:
  base_dir: "/path/to/output"
  registered_dir: "registered"
  results_file: "neuron_data.h5"

registration:
  fft_block_size: 512
  phase_correlation_threshold: 0.3
  max_shift: 20

detection:
  local_contrast_radius: 2
  brightness_threshold: 60
  min_neuron_area: 2
  max_neuron_area: 100

extraction:
  neighborhood_size: 3

gpu:
  device_id: 0
  memory_limit: 0.95
```

## Documentation

Detailed documentation is available in the [docs](./docs) directory:
- [Technical Details](./docs/technical_details.md)
- [Parameter Guide](./docs/parameter_guide.md)
- [Troubleshooting](./docs/troubleshooting.md)

## Performance Tuning

For optimal performance:
1. Adjust `memory_limit` based on your GPU
2. Use appropriate chunk sizes for your data
3. Monitor GPU memory usage with `nvidia-smi`

## Contributing

Contributions are welcome! Please read our [Contributing Guidelines](CONTRIBUTING.md) for details on how to submit pull requests, report issues, and contribute to the project.

## Citation

If you use CuNeuroCaRT in your research, please cite:
```bibtex
@software{CuNeuroCaRT2024,
  title = {CuNeuroCaRT: GPU-Accelerated Neural Activity Analysis},
  year = {2024},
  url = {https://github.com/yourusername/CuNeuroCaRT}
}
```

## License

MIT License - see [LICENSE](LICENSE) for details
