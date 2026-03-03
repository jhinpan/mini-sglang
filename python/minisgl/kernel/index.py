from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Tuple

from minisgl.utils.arch import is_hip

if TYPE_CHECKING:
    import torch
    from tvm_ffi import Module


def _indexing_pytorch(
    weights: torch.Tensor,
    indices: torch.Tensor,
    *,
    output: torch.Tensor | None = None,
    vocab_range: Tuple[int, int] | None = None,
) -> torch.Tensor:
    if vocab_range is not None:
        start, length = vocab_range
        valid_mask = (indices >= start) & (indices < start + length)
        adjusted_indices = (indices - start).clamp(0, length - 1)
        if output is None:
            output = weights.new_zeros(indices.shape[0], weights.shape[1])
        result = weights[adjusted_indices]
        output[valid_mask] = result[valid_mask]
        output[~valid_mask] = 0
        return output
    if output is None:
        output = weights[indices]
    else:
        output.copy_(weights[indices])
    return output


if not is_hip():
    from .utils import KernelConfig, load_jit, make_cpp_args

    DEFAULT_INDEX_KERNEL_CONFIG = KernelConfig(num_threads=128, max_occupancy=1, use_pdl=False)

    @functools.cache
    def _jit_index_module(
        element_size: int,
        *,
        num_splits: int = 1,
        config: KernelConfig = DEFAULT_INDEX_KERNEL_CONFIG,
    ) -> Module:
        args = make_cpp_args(element_size, num_splits, *config)
        return load_jit(
            "index",
            *args,
            cuda_files=["index.cu"],
            cuda_wrappers=[("launch", f"IndexKernel<{args}>::run")],
        )


def indexing(
    weights: torch.Tensor,
    indices: torch.Tensor,
    *,
    output: torch.Tensor | None = None,
    vocab_range: Tuple[int, int] | None = None,  # (start, length)
) -> torch.Tensor:
    if is_hip():
        return _indexing_pytorch(weights, indices, output=output, vocab_range=vocab_range)

    if output is None:
        output = weights.new_empty(indices.shape[0], weights.shape[1])

    element_size = weights.shape[1] * weights.element_size()
    if element_size % 2048 == 0:
        num_splits = 4
    elif element_size % 1024 == 0:
        num_splits = 2
    else:
        num_splits = 1
    module = _jit_index_module(element_size, num_splits=num_splits)
    module.launch(weights, indices, output, vocab_range)
    return output
