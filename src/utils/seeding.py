"""Utility functions for robot learning system.

This module provides common utilities including seeding, device management,
logging configuration, and other helper functions.
"""

from __future__ import annotations

import logging
import os
import random
from typing import Optional, Union

import numpy as np
import torch

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility.

    Sets seeds for Python random, NumPy, PyTorch, and CUDA (if available).

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Set CUDA seeds if available
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Make CUDA operations deterministic
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    logger.info(f"Set random seed to {seed}")


def get_device() -> torch.device:
    """Get the best available device for PyTorch operations.

    Returns:
        PyTorch device (CUDA, MPS, or CPU)
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using Apple Silicon MPS device")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU device")
    
    return device


def setup_logging(
    level: Union[str, int] = logging.INFO,
    log_file: Optional[str] = None,
    format_string: Optional[str] = None,
) -> None:
    """Setup logging configuration.

    Args:
        level: Logging level
        log_file: Optional log file path
        format_string: Custom format string
    """
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Configure root logger
    logging.basicConfig(
        level=level,
        format=format_string,
        handlers=[
            logging.StreamHandler(),
            *([logging.FileHandler(log_file)] if log_file else []),
        ],
    )
    
    # Set specific logger levels
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    
    logger.info(f"Logging configured with level {level}")


def ensure_dir(directory: str) -> None:
    """Ensure directory exists, create if it doesn't.

    Args:
        directory: Directory path to ensure exists
    """
    os.makedirs(directory, exist_ok=True)
    logger.debug(f"Ensured directory exists: {directory}")


def format_time(seconds: float) -> str:
    """Format time duration in a human-readable format.

    Args:
        seconds: Time duration in seconds

    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def format_number(number: float, precision: int = 2) -> str:
    """Format number with appropriate precision and units.

    Args:
        number: Number to format
        precision: Decimal precision

    Returns:
        Formatted number string
    """
    if abs(number) < 1e-6:
        return f"{number:.{precision}e}"
    elif abs(number) < 1e-3:
        return f"{number * 1e6:.{precision}f}μ"
    elif abs(number) < 1:
        return f"{number * 1e3:.{precision}f}m"
    elif abs(number) < 1e3:
        return f"{number:.{precision}f}"
    elif abs(number) < 1e6:
        return f"{number / 1e3:.{precision}f}K"
    elif abs(number) < 1e9:
        return f"{number / 1e6:.{precision}f}M"
    else:
        return f"{number / 1e9:.{precision}f}B"


class Timer:
    """Simple timer context manager for measuring execution time."""

    def __init__(self, name: str = "Operation") -> None:
        """Initialize timer.

        Args:
            name: Name of the operation being timed
        """
        self.name = name
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None

    def __enter__(self) -> Timer:
        """Start timing."""
        import time
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Stop timing and log duration."""
        import time
        self.end_time = time.time()
        duration = self.end_time - self.start_time
        logger.info(f"{self.name} completed in {format_time(duration)}")

    @property
    def elapsed(self) -> Optional[float]:
        """Get elapsed time in seconds."""
        if self.start_time is None:
            return None
        end_time = self.end_time or time.time()
        return end_time - self.start_time


def moving_average(data: list[float], window: int) -> list[float]:
    """Calculate moving average of data.

    Args:
        data: Input data
        window: Window size for moving average

    Returns:
        Moving average values
    """
    if len(data) < window:
        return data
    
    result = []
    for i in range(len(data)):
        start_idx = max(0, i - window + 1)
        window_data = data[start_idx:i + 1]
        result.append(sum(window_data) / len(window_data))
    
    return result


def calculate_confidence_interval(
    data: list[float], 
    confidence: float = 0.95
) -> tuple[float, float, float]:
    """Calculate confidence interval for data.

    Args:
        data: Input data
        confidence: Confidence level (0-1)

    Returns:
        (mean, lower_bound, upper_bound)
    """
    import scipy.stats as stats
    
    mean = np.mean(data)
    std = np.std(data, ddof=1)
    n = len(data)
    
    # Calculate confidence interval
    alpha = 1 - confidence
    t_value = stats.t.ppf(1 - alpha / 2, n - 1)
    margin_error = t_value * (std / np.sqrt(n))
    
    lower_bound = mean - margin_error
    upper_bound = mean + margin_error
    
    return mean, lower_bound, upper_bound


def save_config(config: dict, filepath: str) -> None:
    """Save configuration dictionary to file.

    Args:
        config: Configuration dictionary
        filepath: Path to save config
    """
    import json
    
    ensure_dir(os.path.dirname(filepath))
    
    with open(filepath, 'w') as f:
        json.dump(config, f, indent=2, default=str)
    
    logger.info(f"Saved configuration to {filepath}")


def load_config(filepath: str) -> dict:
    """Load configuration from file.

    Args:
        filepath: Path to config file

    Returns:
        Configuration dictionary
    """
    import json
    
    with open(filepath, 'r') as f:
        config = json.load(f)
    
    logger.info(f"Loaded configuration from {filepath}")
    return config
