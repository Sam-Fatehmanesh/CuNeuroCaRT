import yaml
import logging
import os
import cupy as cp
import numpy as np
from pathlib import Path

def load_config(config_path):
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def init_logger(level=logging.INFO):
    """Initialize logger with specified level."""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        force=True  # Ensure we override any existing logging configuration
    )
    return logging.getLogger(__name__)

def setup_output_dirs(config):
    """Create output directories if they don't exist."""
    base_dir = Path(config['output']['base_dir'])
    reg_dir = base_dir / config['output']['registered_dir']
    
    for dir_path in [base_dir, reg_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    return base_dir, reg_dir

def gpu_to_cpu(data):
    """Convert cupy array to numpy array if necessary."""
    if isinstance(data, cp.ndarray):
        return cp.asnumpy(data)
    return data

def cpu_to_gpu(data):
    """Convert numpy array to cupy array if necessary."""
    if isinstance(data, np.ndarray):
        return cp.asarray(data)
    return data

def ensure_gpu_memory(shape, dtype=np.float32, fraction=0.8):
    """Check if there's enough GPU memory for array of given shape."""
    try:
        mem_needed = np.prod(shape) * np.dtype(dtype).itemsize
        mem_total = cp.cuda.runtime.memGetInfo()[1]
        if mem_needed > mem_total * fraction:
            raise MemoryError(f"Operation would require {mem_needed/1e9:.2f}GB, but only "
                            f"{mem_total*fraction/1e9:.2f}GB is available")
        return True
    except Exception as e:
        logging.error(f"GPU memory check failed: {str(e)}")
        return False 