from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from minisgl.utils.arch import is_hip

if TYPE_CHECKING:
    import torch


def _activation_and_mul_pytorch(
    activation_fn: Callable, x: torch.Tensor, out: torch.Tensor | None = None
) -> torch.Tensor:
    d = x.shape[-1] // 2
    gate = x[..., :d]
    up = x[..., d:]
    result = activation_fn(gate) * up
    if out is not None:
        out.copy_(result)
        return out
    return result


def silu_and_mul(x: torch.Tensor, out: torch.Tensor | None = None):
    if is_hip():
        import torch.nn.functional as F

        return _activation_and_mul_pytorch(F.silu, x, out=out)
    from flashinfer import silu_and_mul

    return silu_and_mul(x, out=out)


def gelu_and_mul(x: torch.Tensor, out: torch.Tensor | None = None):
    if is_hip():
        import torch.nn.functional as F

        return _activation_and_mul_pytorch(F.gelu, x, out=out)
    from flashinfer import gelu_and_mul

    return gelu_and_mul(x, out=out)


__all__ = ["silu_and_mul", "gelu_and_mul"]
