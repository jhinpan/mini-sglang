from __future__ import annotations

import functools
import math
from typing import Any, Callable, Dict, Tuple

import torch

from .base import StateLessOP


class RotaryEmbedding(StateLessOP):
    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
        post_process: None | Callable[[torch.Tensor], torch.Tensor] = None,
    ) -> None:
        from minisgl.utils.arch import is_hip

        super().__init__()
        self.head_size = head_size
        self._is_hip = is_hip()
        assert rotary_dim == head_size
        inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
        if post_process is not None:
            inv_freq = post_process(inv_freq)
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        # buffer, so don't load/save
        self._cos_sin_cache = torch.cat((cos, sin), dim=-1)
        assert self.head_size in [64, 128, 256, 512]

        if not self._is_hip:
            from flashinfer import apply_rope_with_cos_sin_cache_inplace

            self.apply_rope_with_cos_sin_cache_inplace = apply_rope_with_cos_sin_cache_inplace

    def _apply_rope_pytorch(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> None:
        # AMD uses int64 positions for RoPE compatibility (following SGLang)
        positions = positions.to(torch.int64)
        cos_sin = self._cos_sin_cache[positions]
        half_dim = self.head_size // 2
        cos = cos_sin[:, :half_dim]
        sin = cos_sin[:, half_dim:]

        def _rotate(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> None:
            # x may be 2D (num_tokens, num_heads * head_size) or 3D (num_tokens, num_heads, head_size)
            if x.dim() == 2:
                x = x.view(x.shape[0], -1, self.head_size)
            num_tokens, num_heads, _ = x.shape
            cos_expand = cos[:, None, :].expand(num_tokens, num_heads, half_dim)
            sin_expand = sin[:, None, :].expand(num_tokens, num_heads, half_dim)
            x1 = x[..., :half_dim]
            x2 = x[..., half_dim:]
            r1 = x1 * cos_expand - x2 * sin_expand
            r2 = x2 * cos_expand + x1 * sin_expand
            x[..., :half_dim] = r1
            x[..., half_dim:] = r2

        _rotate(query, cos, sin)
        _rotate(key, cos, sin)

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._is_hip:
            self._apply_rope_pytorch(positions, query, key)
        else:
            self.apply_rope_with_cos_sin_cache_inplace(
                positions=positions,
                query=query,
                key=key,
                head_size=self.head_size,
                cos_sin_cache=self._cos_sin_cache,
            )
        return query, key


def _get_rope(
    head_dim: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    rope_scaling: Dict[str, Any] | None = None,
) -> RotaryEmbedding:
    if rope_scaling is None:
        return RotaryEmbedding(head_dim, rotary_dim, max_position, base)
    # need to test some cases:
    match rope_scaling["rope_type"]:
        case "llama3":
            scaling_factor: float = rope_scaling["factor"]
            low_freq_factor: float = rope_scaling["low_freq_factor"]
            high_freq_factor: float = rope_scaling["high_freq_factor"]
            original_max_position: int = rope_scaling["original_max_position_embeddings"]

            def post_process(inv_freq: torch.Tensor) -> torch.Tensor:
                # no smooth if low_freq_factor == high_freq_factor
                wave_len = 2 * math.pi / inv_freq
                if low_freq_factor == high_freq_factor:
                    return torch.where(
                        wave_len < original_max_position / high_freq_factor,
                        inv_freq,
                        inv_freq / scaling_factor,
                    )

                delta = high_freq_factor - low_freq_factor
                smooth = (original_max_position / wave_len - low_freq_factor) / delta
                smooth = torch.clamp(smooth, 0, 1)
                factor = (1 - smooth) / scaling_factor + smooth
                return factor * inv_freq

            return RotaryEmbedding(head_dim, rotary_dim, max_position, base, post_process)

    raise ValueError(f"Unsupported {rope_scaling = }")


_ROPE_DEVICE: torch.device | None = None


def set_rope_device(device: torch.device):
    global _ROPE_DEVICE
    _ROPE_DEVICE = device


@functools.cache
def get_rope(
    head_dim: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    rope_scaling: Tuple[Tuple[str, Any], ...] | None = None,
) -> RotaryEmbedding:
    rope_map = dict(rope_scaling) if rope_scaling is not None else None
    t = torch.tensor([])
    if t.device == torch.device("meta"):
        # we cannot use meta device for rope
        if _ROPE_DEVICE is None:
            raise RuntimeError(
                "We cannot use meta device for rope. Please call set_rope_device() first."
            )
        with torch.device(_ROPE_DEVICE):
            return _get_rope(head_dim, rotary_dim, max_position, base, rope_map)
    return _get_rope(head_dim, rotary_dim, max_position, base, rope_map)


__all__ = ["get_rope", "RotaryEmbedding", "set_rope_device"]
