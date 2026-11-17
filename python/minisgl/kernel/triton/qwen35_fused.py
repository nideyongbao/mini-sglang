"""Qwen3.5-specific fused operations used around the shared SGLang kernels."""

from __future__ import annotations

from functools import lru_cache

import torch
import triton
import triton.language as tl


@triton.jit
def _mrope_cache_kernel(
    cache,
    positions,
    output,
    length: tl.constexpr,
    dim: tl.constexpr,
    h_section: tl.constexpr,
    w_section: tl.constexpr,
    block: tl.constexpr,
):
    token = tl.program_id(0)
    col = tl.arange(0, block)
    frequency = col % (dim // 2)
    axis = tl.where(
        (frequency % 3 == 1) & (frequency < h_section * 3),
        1,
        tl.where((frequency % 3 == 2) & (frequency < w_section * 3), 2, 0),
    )
    pos = tl.load(positions + axis * length + token)
    values = tl.load(cache + pos * dim + col, col < dim, 0)
    tl.store(output + token * dim + col, values, col < dim)


def build_mrope_cache(cache: torch.Tensor, positions: torch.Tensor, sections):
    """Select THW frequencies once per batch; reuse fused QK norm/RoPE at every layer."""
    length, dim = positions.shape[1], cache.shape[1]
    output = torch.empty((length, dim), device=cache.device, dtype=cache.dtype)
    _mrope_cache_kernel[(length,)](
        cache,
        positions.contiguous(),
        output,
        length,
        dim,
        sections[1],
        sections[2],
        triton.next_power_of_2(dim),
        num_warps=1,
    )
    return output


@triton.jit
def _gated_rmsnorm_kernel(
    core,
    gate,
    weight,
    eps,
    stride_core_token: tl.constexpr,
    stride_core_head: tl.constexpr,
    stride_gate_token: tl.constexpr,
    stride_gate_head: tl.constexpr,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    block_size: tl.constexpr,
    sigmoid_gate: tl.constexpr,
):
    row = tl.program_id(0)
    token = row // num_heads
    head = row % num_heads
    offsets = tl.arange(0, block_size)
    mask = offsets < head_dim
    core_ptr = core + token * stride_core_token + head * stride_core_head + offsets
    gate_ptr = gate + token * stride_gate_token + head * stride_gate_head + offsets

    values = tl.load(core_ptr, mask=mask, other=0.0).to(tl.float32)
    gates = tl.load(gate_ptr, mask=mask, other=0.0).to(tl.float32)
    weights = tl.load(weight + offsets, mask=mask, other=0.0).to(tl.float32)
    values *= tl.rsqrt(tl.sum(values * values) / head_dim + eps) * weights
    values *= tl.sigmoid(gates) if sigmoid_gate else gates * tl.sigmoid(gates)
    tl.store(core_ptr, values, mask=mask)


def gated_rmsnorm_(
    core: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    sigmoid_gate: bool = False,
) -> torch.Tensor:
    """Apply RMSNorm and a SiLU (default) or sigmoid gate in one launch, in place."""
    if core.ndim != 3 or gate.ndim != 3 or core.shape != gate.shape:
        raise ValueError("core and gate must have identical [tokens, heads, dim] shapes")
    tokens, num_heads, head_dim = core.shape
    block_size = triton.next_power_of_2(head_dim)
    _gated_rmsnorm_kernel[(tokens * num_heads,)](
        core,
        gate,
        weight,
        eps,
        core.stride(0),
        core.stride(1),
        gate.stride(0),
        gate.stride(1),
        num_heads,
        head_dim,
        block_size,
        sigmoid_gate,
        num_warps=1,
        num_stages=2,
    )
    return core


@lru_cache(maxsize=1)
def _load_attention_kernels():
    from sglang.kernels.ops.attention.fused_qk_rmsnorm_rope_gate import (
        fused_qk_gemma_rmsnorm_rope_gate,
    )
    from sglang.kernels.ops.elementwise.elementwise import fused_sigmoid_mul

    return fused_qk_gemma_rmsnorm_rope_gate, fused_sigmoid_mul


def prepare_qk_gate(
    q_gate: torch.Tensor,
    key: torch.Tensor,
    *,
    query_weight: torch.Tensor,
    key_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    eps: float,
    num_query_heads: int,
    num_key_heads: int,
    head_dim: int,
    rotary_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fuse Q/K Gemma RMSNorm, partial RoPE, and gate deinterleaving."""
    fused_prepare, _ = _load_attention_kernels()
    query, key, gate = fused_prepare(
        q_gate,
        key,
        query_weight,
        key_weight,
        cos_sin_cache,
        positions,
        eps,
        num_query_heads,
        num_key_heads,
        head_dim,
        rotary_dim,
        has_gate=True,
    )
    assert gate is not None
    return query, key, gate


def apply_attention_gate_(output: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """Apply the attention output gate in one in-place launch."""
    _, fused_sigmoid_mul = _load_attention_kernels()
    return fused_sigmoid_mul(output, gate, inplace=True)


__all__ = ["apply_attention_gate_", "gated_rmsnorm_", "prepare_qk_gate"]
