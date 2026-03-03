from __future__ import annotations

import functools
from typing import TYPE_CHECKING

from minisgl.utils.arch import is_hip

if TYPE_CHECKING:
    import torch
    from tvm_ffi import Module


def _store_cache_pytorch(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    indices: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> None:
    num_tokens = k_cache.shape[0]
    k_cache_flat = k_cache.view(num_tokens, -1)
    v_cache_flat = v_cache.view(num_tokens, -1)
    k_cache_flat[indices] = k.view(indices.shape[0], -1)
    v_cache_flat[indices] = v.view(indices.shape[0], -1)


if not is_hip():
    from .utils import KernelConfig, load_jit, make_cpp_args

    DEFAULT_INDEX_KERNEL_CONFIG = KernelConfig(num_threads=128, max_occupancy=1, use_pdl=False)

    @functools.cache
    def _jit_store_module(
        element_size: int,
        *,
        config: KernelConfig = DEFAULT_INDEX_KERNEL_CONFIG,
    ) -> Module:
        args = make_cpp_args(element_size, *config)
        return load_jit(
            "store",
            *args,
            cuda_files=["store.cu"],
            cuda_wrappers=[("launch", f"StoreKernel<{args}>::run")],
        )


def store_cache(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    indices: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> None:
    if is_hip():
        _store_cache_pytorch(k_cache, v_cache, indices, k, v)
        return

    num_tokens = k_cache.shape[0]
    k_cache = k_cache.view(num_tokens, -1)
    v_cache = v_cache.view(num_tokens, -1)
    element_size = k_cache.shape[1] * k_cache.element_size()
    module = _jit_store_module(element_size)
    module.launch(k_cache, v_cache, indices, k, v)
