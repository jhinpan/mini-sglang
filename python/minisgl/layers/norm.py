from typing import Tuple

import torch

from minisgl.utils.arch import is_hip

from .base import BaseOP


def _rmsnorm_pytorch(
    x: torch.Tensor, weight: torch.Tensor, eps: float, out: torch.Tensor | None = None
) -> torch.Tensor:
    orig_dtype = x.dtype
    x_float = x.to(torch.float32)
    variance = x_float.pow(2).mean(dim=-1, keepdim=True)
    x_normed = x_float * torch.rsqrt(variance + eps)
    result = (x_normed * weight).to(orig_dtype)
    if out is not None:
        out.copy_(result)
        return out
    return result


def _fused_add_rmsnorm_pytorch(
    x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float
) -> None:
    x += residual
    residual.copy_(x)
    x_float = x.to(torch.float32)
    variance = x_float.pow(2).mean(dim=-1, keepdim=True)
    x_normed = x_float * torch.rsqrt(variance + eps)
    x.copy_((x_normed * weight).to(x.dtype))


class RMSNorm(BaseOP):
    def __init__(self, size: int, eps: float) -> None:
        self.eps = eps
        self.weight = torch.empty(size)
        if is_hip():
            self.rmsnorm = _rmsnorm_pytorch
        else:
            from flashinfer import rmsnorm

            self.rmsnorm = rmsnorm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.rmsnorm(x, self.weight, self.eps)

    def forward_inplace(self, x: torch.Tensor) -> None:
        self.rmsnorm(x, self.weight, self.eps, out=x)


class RMSNormFused(BaseOP):
    def __init__(self, size: int, eps: float) -> None:
        self.eps = eps
        self.weight = torch.empty(size)
        if is_hip():
            self.rmsnorm = _rmsnorm_pytorch
            self.fused_add_rmsnorm = _fused_add_rmsnorm_pytorch
        else:
            from flashinfer import fused_add_rmsnorm, rmsnorm

            self.rmsnorm = rmsnorm
            self.fused_add_rmsnorm = fused_add_rmsnorm

    def forward(
        self, x: torch.Tensor, residual: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            return self.rmsnorm(x, self.weight, self.eps), x
        self.fused_add_rmsnorm(x, residual, self.weight, self.eps)
        return x, residual
