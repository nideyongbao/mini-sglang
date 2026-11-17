"""Sparse paged GQA over per-query physical token indices."""

import torch
import triton
import triton.language as tl


@triton.jit
def _sparse(
    Q,
    K,
    V,
    I,
    O,
    QS: tl.constexpr,
    KS: tl.constexpr,
    H: tl.constexpr,
    KV: tl.constexpr,
    D: tl.constexpr,
    TOP: tl.constexpr,
    BH: tl.constexpr,
    BN: tl.constexpr,
):
    row, group = tl.program_id(0), tl.program_id(1)
    h = tl.arange(0, BH)
    d = tl.arange(0, D)
    nh = H // KV
    q = tl.load(Q + row * QS + (group * nh + h[:, None]) * D + d[None, :], h[:, None] < nh, 0)
    q = (q * (D**-0.5 * 1.4426950408889634)).to(q.dtype)
    acc = tl.zeros((BH, D), tl.float32)
    maximum = tl.full((BH,), -1.0e30, tl.float32)
    denom = tl.zeros((BH,), tl.float32)
    for start in range(tl.cdiv(TOP, BN)):
        col = start * BN + tl.arange(0, BN)
        loc = tl.load(I + row * TOP + col, col < TOP, -1)
        valid = loc >= 0
        k = tl.load(K + loc[None, :] * KS + group * D + d[:, None], valid[None, :], 0)
        v = tl.load(V + loc[:, None] * KS + group * D + d[None, :], valid[:, None], 0)
        score = tl.where(valid[None, :], tl.dot(q, k), -1.0e30)
        nextmax = tl.maximum(maximum, tl.max(score, 1))
        alpha = tl.exp2(maximum - nextmax)
        prob = tl.where(valid[None, :], tl.exp2(score - nextmax[:, None]), 0.0)
        acc = tl.dot(prob.to(v.dtype), v, acc * alpha[:, None])
        denom = denom * alpha + tl.sum(prob, 1)
        maximum = nextmax
    output = acc / tl.maximum(denom[:, None], 1.0e-20)
    tl.store(O + row * H * D + (group * nh + h[:, None]) * D + d[None, :], output, h[:, None] < nh)


def sparse_attention(q, k, v, indices):
    q = q.contiguous()
    out = torch.empty_like(q)
    _sparse[(q.shape[0], k.shape[1])](
        q,
        k,
        v,
        indices,
        out,
        q.stride(0),
        k.stride(0),
        q.shape[1],
        k.shape[1],
        q.shape[2],
        indices.shape[1],
        max(16, triton.next_power_of_2(q.shape[1] // k.shape[1])),
        32,
        num_warps=4,
    )
    return out


@triton.jit
def _gather(
    K,
    V,
    I,
    CU,
    PK,
    PV,
    TOP: tl.constexpr,
    KV: tl.constexpr,
    D: tl.constexpr,
    B: tl.constexpr,
    BD: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.program_id(1) * B + tl.arange(0, B)
    d = tl.arange(0, BD)
    loc = tl.load(I + row * TOP + col, col < TOP, -1)
    start = tl.load(CU + row)
    end = tl.load(CU + row + 1)
    valid = (col < TOP) & (loc >= 0) & (start + col < end)
    kval = tl.load(
        K + loc[:, None] * KV * D + d[None, :], valid[:, None] & (d[None, :] < KV * D), 0
    )
    vval = tl.load(
        V + loc[:, None] * KV * D + d[None, :], valid[:, None] & (d[None, :] < KV * D), 0
    )
    tl.store(
        PK + (start + col[:, None]) * KV * D + d[None, :],
        kval,
        valid[:, None] & (d[None, :] < KV * D),
    )
    tl.store(
        PV + (start + col[:, None]) * KV * D + d[None, :],
        vval,
        valid[:, None] & (d[None, :] < KV * D),
    )


def sparse_decode(q, k, v, indices):
    """Pack rows (valid indices before -1 padding) and use the reference FA3 kernel."""
    from sgl_kernel.flash_attn import flash_attn_varlen_func

    batch, top = indices.shape
    lengths = (indices >= 0).sum(-1, dtype=torch.int32)
    cu = torch.cat((lengths.new_zeros(1), lengths.cumsum(0, dtype=torch.int32)))
    packed_k = k.new_empty(batch * top, k.shape[1], k.shape[2])
    packed_v = torch.empty_like(packed_k)
    _gather[(batch, triton.cdiv(top, 32))](
        k,
        v,
        indices,
        cu,
        packed_k,
        packed_v,
        top,
        k.shape[1],
        k.shape[2],
        32,
        triton.next_power_of_2(k.shape[1] * k.shape[2]),
    )
    return flash_attn_varlen_func(
        q=q.contiguous(),
        k=packed_k,
        v=packed_v,
        cu_seqlens_q=torch.arange(batch + 1, device=q.device, dtype=torch.int32),
        cu_seqlens_k=cu,
        max_seqlen_q=1,
        max_seqlen_k=top,
        softmax_scale=q.shape[-1] ** -0.5,
        causal=True,
        ver=3,
    )
