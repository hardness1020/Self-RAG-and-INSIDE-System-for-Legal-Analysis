"""
Device Utilities

Helper functions for automatic device detection and management
across CPU, CUDA (NVIDIA GPU), and MPS (Mac GPU - Apple Silicon).
"""

import torch
from typing import Optional


def get_optimal_device(prefer_gpu: bool = True, verbose: bool = True) -> str:
    """
    Automatically detect and return the best available device.

    Priority order (when prefer_gpu=True):
    1. MPS (Mac GPU - Apple Silicon M1/M2/M3)
    2. CUDA (NVIDIA GPU)
    3. CPU (fallback)

    Args:
        prefer_gpu: Whether to prefer GPU over CPU
        verbose: Whether to print device information

    Returns:
        Device string: 'mps', 'cuda', or 'cpu'

    Example:
        >>> device = get_optimal_device()
        >>> model = model.to(device)
    """
    device = "cpu"
    device_info = []

    if prefer_gpu:
        # Check for Mac GPU (MPS)
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            device = "mps"
            device_info.append("✓ Mac GPU (MPS) available and will be used")
            device_info.append(f"  PyTorch MPS backend: {torch.backends.mps.is_built()}")
        # Check for NVIDIA GPU (CUDA)
        elif torch.cuda.is_available():
            device = "cuda"
            device_info.append(f"✓ CUDA GPU available: {torch.cuda.get_device_name(0)}")
            device_info.append(f"  CUDA devices: {torch.cuda.device_count()}")
        else:
            device = "cpu"
            device_info.append("⚠ No GPU detected, using CPU")
            if torch.backends.mps.is_built():
                device_info.append("  Note: MPS is built but not available (older macOS?)")
    else:
        device = "cpu"
        device_info.append("Using CPU (prefer_gpu=False)")

    if verbose:
        print(f"\n{'='*60}")
        print(f"Device Selection: {device.upper()}")
        print(f"{'='*60}")
        for info in device_info:
            print(info)
        print(f"{'='*60}\n")

    return device


def get_training_args_for_device(device: str) -> dict:
    """
    Get HuggingFace TrainingArguments kwargs for the specified device.

    This function returns the appropriate device-specific parameters needed
    for HuggingFace's Trainer class to correctly use the specified device.

    Uses the modern transformers API (4.35+) with automatic MPS/CUDA detection.

    Args:
        device: Device string from get_optimal_device() ('cuda', 'mps', or 'cpu')

    Returns:
        Dictionary of kwargs to pass to TrainingArguments

    Example:
        >>> device = get_optimal_device()
        >>> device_args = get_training_args_for_device(device)
        >>> training_args = TrainingArguments(..., **device_args)

    Device-specific behavior:
        - 'cuda': Returns {'use_cpu': False} (enables CUDA auto-detection)
        - 'mps': Returns {'use_cpu': False} (enables MPS auto-detection)
        - 'cpu': Returns {'use_cpu': True} (forces CPU usage)
    """
    device_lower = device.lower()

    if device_lower == "cuda":
        # Allow CUDA auto-detection
        return {"use_cpu": False}
    elif device_lower == "mps":
        # Allow MPS auto-detection (modern transformers auto-detect MPS)
        return {"use_cpu": False}
    elif device_lower == "cpu":
        # Force CPU usage
        return {"use_cpu": True}
    else:
        # Unknown device, default to CPU behavior
        return {"use_cpu": True}


def get_device_info() -> dict:
    """
    Get detailed information about available devices.

    Returns:
        Dictionary with device availability and specifications

    Example:
        >>> info = get_device_info()
        >>> print(f"MPS available: {info['mps_available']}")
    """
    info = {
        'cpu_available': True,
        'cuda_available': torch.cuda.is_available(),
        'mps_available': torch.backends.mps.is_available() and torch.backends.mps.is_built(),
        'mps_built': torch.backends.mps.is_built(),
    }

    # CUDA details
    if info['cuda_available']:
        info['cuda_device_count'] = torch.cuda.device_count()
        info['cuda_device_name'] = torch.cuda.get_device_name(0)
        info['cuda_version'] = torch.version.cuda

    # PyTorch version
    info['pytorch_version'] = torch.__version__

    return info


def verify_device_compatibility(device: str) -> bool:
    """
    Verify that the specified device is available.

    Args:
        device: Device string ('cpu', 'cuda', or 'mps')

    Returns:
        True if device is available, False otherwise

    Example:
        >>> if verify_device_compatibility('mps'):
        >>>     model = model.to('mps')
    """
    device_lower = device.lower()

    if device_lower == 'cpu':
        return True
    elif device_lower == 'cuda':
        return torch.cuda.is_available()
    elif device_lower == 'mps':
        return torch.backends.mps.is_available() and torch.backends.mps.is_built()
    else:
        return False


def get_device_or_fallback(device: Optional[str] = None, verbose: bool = True) -> str:
    """
    Get specified device or fallback to optimal device if not available.

    Args:
        device: Requested device ('cpu', 'cuda', 'mps', or None for auto)
        verbose: Whether to print warnings

    Returns:
        Available device string

    Example:
        >>> device = get_device_or_fallback('mps')  # Will fallback to CPU if MPS unavailable
    """
    if device is None:
        return get_optimal_device(prefer_gpu=True, verbose=verbose)

    if verify_device_compatibility(device):
        if verbose:
            print(f"✓ Using requested device: {device}")
        return device
    else:
        optimal = get_optimal_device(prefer_gpu=True, verbose=False)
        if verbose:
            print(f"⚠ Requested device '{device}' not available")
            print(f"  Falling back to: {optimal}")
        return optimal


def print_device_summary():
    """
    Print a formatted summary of all available devices.

    Useful for troubleshooting and configuration.
    """
    info = get_device_info()

    print(f"\n{'='*60}")
    print("Device Availability Summary")
    print(f"{'='*60}")
    print(f"PyTorch Version: {info['pytorch_version']}")
    print(f"\n{'Device':<15} {'Available':<12} {'Details'}")
    print(f"{'-'*60}")

    # CPU
    print(f"{'CPU':<15} {'✓ Yes':<12} Always available")

    # CUDA
    if info['cuda_available']:
        print(f"{'CUDA (NVIDIA)':<15} {'✓ Yes':<12} {info['cuda_device_name']}")
        print(f"{'':<15} {'':<12} {info['cuda_device_count']} device(s), CUDA {info['cuda_version']}")
    else:
        print(f"{'CUDA (NVIDIA)':<15} {'✗ No':<12} Not detected")

    # MPS
    if info['mps_available']:
        print(f"{'MPS (Mac GPU)':<15} {'✓ Yes':<12} Apple Silicon GPU")
    else:
        if info['mps_built']:
            print(f"{'MPS (Mac GPU)':<15} {'⚠ Built':<12} Built but not available (check macOS version)")
        else:
            print(f"{'MPS (Mac GPU)':<15} {'✗ No':<12} Not available")

    print(f"{'='*60}")
    print(f"\nRecommended device: {get_optimal_device(verbose=False).upper()}")
    print(f"{'='*60}\n")


# Example usage and testing
if __name__ == "__main__":
    print("Device Utilities Test\n")

    # Print full summary
    print_device_summary()

    # Test optimal device detection
    print("\nAutomatic device detection:")
    device = get_optimal_device(prefer_gpu=True, verbose=True)

    # Test device verification
    print("\nTesting device compatibility:")
    for test_device in ['cpu', 'cuda', 'mps']:
        available = verify_device_compatibility(test_device)
        status = "✓ Available" if available else "✗ Not available"
        print(f"  {test_device}: {status}")

    # Test fallback
    print("\nTesting fallback mechanism:")
    result = get_device_or_fallback('mps', verbose=True)
    print(f"Final device: {result}")
