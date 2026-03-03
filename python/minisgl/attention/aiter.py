from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List

import torch
import torch.nn.functional as F
from minisgl.core import Batch, get_global_ctx

from .base import BaseAttnBackend, BaseAttnMetadata
from .utils import BaseCaptureData

if TYPE_CHECKING:
    from minisgl.models import ModelConfig

# Maximum KV sequence length for CUDA graph capture.
# Sequences exceeding this use non-graph (eager) mode.
_MAX_GRAPH_KV_LEN = 4096


@dataclass
class AITERCaptureData(BaseCaptureData):
    pass


@dataclass
class AITERMetadata(BaseAttnMetadata):
    cu_seqlens_k: torch.Tensor
    cu_seqlens_q: torch.Tensor
    cache_seqlens: torch.Tensor
    max_seqlen_k: int
    max_seqlen_q: int

    page_table: torch.Tensor

    def get_last_indices(self, bs: int) -> torch.Tensor:
        return self.cu_seqlens_q[1 : 1 + bs] - 1


class AITERBackend(BaseAttnBackend):
    def __init__(self, config: ModelConfig):
        ctx = get_global_ctx()
        self.config = config
        self.kvcache = ctx.kv_cache
        self.page_size = ctx.page_size
        self.capture: AITERCaptureData | None = None
        self.max_graph_bs = 0
        self.capture_bs: List[int] = []
        self.scale = config.head_dim**-0.5
        self.num_qo_heads = config.num_qo_heads
        self.num_kv_heads = config.num_kv_heads

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, layer_id: int, batch: Batch
    ) -> torch.Tensor:
        metadata = batch.attn_metadata
        assert isinstance(metadata, AITERMetadata)
        self.kvcache.store_kv(k, v, batch.out_loc, layer_id)

        k_cache = self.kvcache.k_cache(layer_id)
        v_cache = self.kvcache.v_cache(layer_id)

        if metadata.max_seqlen_q == 1:
            return _sdpa_decode(
                q=q,
                k_cache=k_cache,
                v_cache=v_cache,
                page_table=metadata.page_table,
                cache_seqlens=metadata.cache_seqlens,
                page_size=self.page_size,
                softmax_scale=self.scale,
            )
        else:
            return _sdpa_prefill(
                q=q,
                k_cache=k_cache,
                v_cache=v_cache,
                page_table=metadata.page_table,
                cache_seqlens=metadata.cache_seqlens,
                cu_seqlens_q=metadata.cu_seqlens_q,
                page_size=self.page_size,
                softmax_scale=self.scale,
                num_qo_heads=self.num_qo_heads,
                num_kv_heads=self.num_kv_heads,
            )

    def prepare_metadata(self, batch: Batch) -> None:
        reqs = batch.padded_reqs

        padded_size = len(reqs)
        seqlens_q = [req.extend_len for req in reqs]
        seqlens_k = [req.device_len for req in reqs]
        cached_lens = [req.cached_len for req in reqs]
        max_seqlen_k = max(seqlens_k)
        max_seqlen_q = max(seqlens_q)
        CPU_KWARGS = {"device": "cpu", "dtype": torch.int32, "pin_memory": True}

        device = self.kvcache.device
        cache_seqlens = torch.tensor(seqlens_k, **CPU_KWARGS)
        cache_seqlens = cache_seqlens.to(device, non_blocking=True)
        cu_seqlens_k = torch.tensor([0] + seqlens_k, **CPU_KWARGS).cumsum_(dim=0)
        cu_seqlens_k = cu_seqlens_k.to(device, non_blocking=True)

        if max_seqlen_q == 1:
            cu_seqlens_q = torch.arange(0, padded_size + 1, device=device, dtype=torch.int32)
        elif all(l == 0 for l in cached_lens):  # prefill with no cache hit
            cu_seqlens_q = cu_seqlens_k
        else:  # normal extend prefill, with partial cache hit
            cu_seqlens_q = torch.tensor([0] + seqlens_q, **CPU_KWARGS).cumsum_(dim=0)
            cu_seqlens_q = cu_seqlens_q.to(self.kvcache.device, non_blocking=True)

        page_table = get_global_ctx().page_table
        new_page_table = torch.stack(  # NOTE: global page table treat page_size = 1, we need slice
            [page_table[req.table_idx, : max_seqlen_k : self.page_size] for req in reqs]
        )
        if self.page_size > 1:
            new_page_table.div_(self.page_size, rounding_mode="floor")
        batch.attn_metadata = AITERMetadata(
            cu_seqlens_k=cu_seqlens_k,
            cu_seqlens_q=cu_seqlens_q,
            cache_seqlens=cache_seqlens,
            max_seqlen_k=max_seqlen_k,
            max_seqlen_q=max_seqlen_q,
            page_table=new_page_table,
        )

    def init_capture_graph(self, max_seq_len: int, bs_list: List[int]) -> None:
        assert self.capture is None, "Capture already initialized."
        max_bs = max(bs_list)
        # Cap the graph capture KV length to avoid OOM with SDPA
        capped_kv_len = min(max_seq_len, _MAX_GRAPH_KV_LEN)
        capture = AITERCaptureData.create(
            max_bs, capped_kv_len // self.page_size, self.kvcache.device
        )
        self.max_graph_bs = max_bs
        self.capture = capture
        self.capture_bs = sorted(bs_list)

    def prepare_for_capture(self, batch: Batch) -> None:
        assert (bs := batch.size) in self.capture_bs and self.capture
        capture = self.capture
        metadata = AITERMetadata(
            cu_seqlens_k=capture.cu_seqlens_k[: bs + 1],
            cu_seqlens_q=capture.cu_seqlens_q[: bs + 1],
            cache_seqlens=capture.seq_lens[:bs],
            max_seqlen_k=capture.page_table.size(1) * self.page_size,
            max_seqlen_q=1,  # decode only
            page_table=capture.page_table[:bs, :],
        )
        batch.attn_metadata = metadata

    def prepare_for_replay(self, batch: Batch) -> None:
        metadata, bs = batch.attn_metadata, batch.padded_size
        assert isinstance(metadata, AITERMetadata)
        assert self.capture is not None and bs in self.capture_bs
        # cu_seqlens_q is always [0, 1, 2, ..., bs] for decode (i.e. no-op)
        table_len = min(metadata.page_table.size(1), self.capture.page_table.size(1))
        self.capture.cu_seqlens_k[: bs + 1].copy_(metadata.cu_seqlens_k)
        self.capture.seq_lens[:bs].copy_(metadata.cache_seqlens)
        self.capture.page_table[:bs, :table_len].copy_(metadata.page_table[:, :table_len])


def _sdpa_decode(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    page_size: int,
    softmax_scale: float,
) -> torch.Tensor:
    """SDPA-based decode attention. CUDA-graph safe.

    q: (batch, num_qo_heads, head_dim)
    k_cache/v_cache: (num_pages, page_size, kv_heads, head_dim)
    page_table: (batch, max_num_pages)
    cache_seqlens: (batch,)
    """
    # Gather K/V from paged cache
    # k_cache[page_table]: (batch, max_num_pages, page_size, kv_heads, head_dim)
    batch_size, max_num_pages = page_table.shape
    kv_heads = k_cache.shape[2]
    head_dim = k_cache.shape[3]
    max_tokens = max_num_pages * page_size

    k_pages = k_cache[page_table]
    v_pages = v_cache[page_table]
    k_gathered = k_pages.reshape(batch_size, max_tokens, kv_heads, head_dim)
    v_gathered = v_pages.reshape(batch_size, max_tokens, kv_heads, head_dim)

    # Attention mask: only attend to valid cached tokens
    token_positions = torch.arange(max_tokens, device=q.device, dtype=cache_seqlens.dtype)
    attn_mask = token_positions.unsqueeze(0) < cache_seqlens.unsqueeze(1)
    attn_mask = attn_mask.unsqueeze(1).unsqueeze(1)  # (batch, 1, 1, max_tokens)

    # Reshape for SDPA: (batch, heads, seq_len, head_dim)
    q_sdpa = q.unsqueeze(2)  # (batch, num_qo_heads, 1, head_dim)
    k_sdpa = k_gathered.permute(0, 2, 1, 3)  # (batch, kv_heads, max_tokens, head_dim)
    v_sdpa = v_gathered.permute(0, 2, 1, 3)

    out = F.scaled_dot_product_attention(
        q_sdpa, k_sdpa, v_sdpa,
        attn_mask=attn_mask,
        scale=softmax_scale,
        is_causal=False,
        enable_gqa=True,
    )
    return out.squeeze(2)  # (batch, num_qo_heads, head_dim)


def _sdpa_prefill(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    page_size: int,
    softmax_scale: float,
    num_qo_heads: int,
    num_kv_heads: int,
) -> torch.Tensor:
    """SDPA-based prefill attention. Processes each sequence individually."""
    head_dim = q.shape[2]
    batch_size = page_table.shape[0]

    # Flatten paged cache for token-level indexing
    num_pages, ps, kv_heads, hd = k_cache.shape
    k_flat = k_cache.reshape(num_pages * ps, kv_heads, hd)
    v_flat = v_cache.reshape(num_pages * ps, kv_heads, hd)

    outputs = []
    for i in range(batch_size):
        q_start = cu_seqlens_q[i].item()
        q_end = cu_seqlens_q[i + 1].item()
        q_len = q_end - q_start
        if q_len == 0:
            continue

        q_seq = q[q_start:q_end]  # (q_len, num_qo_heads, head_dim)
        k_len = cache_seqlens[i].item()

        # Get token indices from page table
        num_pages_needed = (k_len + page_size - 1) // page_size
        page_indices = page_table[i, :num_pages_needed]
        offsets = torch.arange(page_size, device=q.device)
        token_indices = (page_indices.unsqueeze(-1) * page_size + offsets).reshape(-1)[:k_len]

        k_seq = k_flat[token_indices]  # (k_len, kv_heads, head_dim)
        v_seq = v_flat[token_indices]  # (k_len, kv_heads, head_dim)

        # Reshape for SDPA: (1, heads, seq_len, head_dim)
        q_sdpa = q_seq.permute(1, 0, 2).unsqueeze(0)  # (1, num_qo_heads, q_len, head_dim)
        k_sdpa = k_seq.permute(1, 0, 2).unsqueeze(0)  # (1, kv_heads, k_len, head_dim)
        v_sdpa = v_seq.permute(1, 0, 2).unsqueeze(0)

        # For prefill: use causal mask when q_len == k_len (full prefill)
        is_causal = q_len == k_len
        if not is_causal and q_len > 1:
            # Extend prefill with partial cache hit
            attn_mask = torch.ones(q_len, k_len, device=q.device, dtype=torch.bool)
            q_offset = k_len - q_len
            for qi in range(q_len):
                attn_mask[qi, q_offset + qi + 1 :] = False
            attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
        else:
            attn_mask = None

        out = F.scaled_dot_product_attention(
            q_sdpa, k_sdpa, v_sdpa,
            attn_mask=attn_mask,
            scale=softmax_scale,
            is_causal=is_causal,
            enable_gqa=True,
        )
        outputs.append(out.squeeze(0).permute(1, 0, 2))

    return torch.cat(outputs, dim=0)
