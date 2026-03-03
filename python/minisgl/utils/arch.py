from __future__ import annotations

import functools
from typing import Tuple


@functools.cache
def is_hip() -> bool:
    """Check if running on AMD ROCm (HIP) platform."""
    import torch

    return hasattr(torch.version, "hip") and torch.version.hip is not None


@functools.cache
def is_cuda() -> bool:
    """Check if running on NVIDIA CUDA platform (not HIP)."""
    import torch

    return torch.cuda.is_available() and not is_hip()


def is_cuda_alike() -> bool:
    """Check if running on any CUDA-compatible platform (NVIDIA or AMD ROCm)."""
    return is_cuda() or is_hip()


@functools.cache
def is_gfx942() -> bool:
    """Check if running on AMD MI300X (gfx942) GPU."""
    if not is_hip():
        return False
    import torch

    props = torch.cuda.get_device_properties(0)
    return "gfx942" in getattr(props, "gcnArchName", "")


@functools.cache
def _get_torch_cuda_version() -> Tuple[int, int] | None:
    import torch
    import torch.version

    if not torch.cuda.is_available() or not torch.version.cuda:
        return None
    if is_hip():
        return None  # SM capabilities are NVIDIA-only
    return torch.cuda.get_device_capability()


def is_arch_supported(major: int, minor: int = 0) -> bool:
    arch = _get_torch_cuda_version()
    if arch is None:
        return False
    return arch >= (major, minor)


def is_sm90_supported() -> bool:
    return is_arch_supported(9, 0)


def is_sm100_supported() -> bool:
    return is_arch_supported(10, 0)
