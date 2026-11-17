"""Grouped Gemma norm used by Qwen3.8 HC and PLE."""

import torch
import triton
import triton.language as tl


@triton.jit
def _group_norm(X, W, Y, G: tl.constexpr, NG: tl.constexpr, EPS: tl.constexpr, B: tl.constexpr):
    row = tl.program_id(0)
    d = tl.arange(0, B)
    x = tl.load(X + row * G + d, d < G, 0).to(tl.float32)
    w = tl.load(W + (row % NG) * G + d, d < G, 0).to(tl.float32)
    y = x * tl.rsqrt(tl.sum(x * x, 0) / G + EPS) * (1.0 + w)
    tl.store(Y + row * G + d, y, d < G)


def grouped_norm(x, w, group, eps):
    y = torch.empty_like(x)
    _group_norm[(x.numel() // group,)](
        x, w, y, group, w.numel() // group, eps, triton.next_power_of_2(group), num_warps=1
    )
    return y


@triton.jit
def _hc_mix(
    X,
    NW,
    WD,
    WU,
    Y,
    NORM,
    HC: tl.constexpr,
    HS: tl.constexpr,
    R: tl.constexpr,
    EPS: tl.constexpr,
    BR: tl.constexpr,
):
    row = tl.program_id(0)
    c = tl.arange(0, HC)
    d = tl.arange(0, HS)
    pos = c[:, None] * HS + d[None, :]
    x = tl.load(X + row * HC * HS + pos).to(tl.float32)
    w = tl.load(NW + pos).to(tl.float32)
    norm = x * tl.rsqrt(tl.sum(x * x, 1)[:, None] / HS + EPS) * (1.0 + w)
    norm = norm.to(X.dtype.element_ty)
    tl.store(NORM + row * HC * HS + pos, norm)
    k = tl.arange(0, HC * HS)
    r = tl.arange(0, BR)
    nv = tl.reshape(norm, (HC * HS,)).to(tl.float32)
    down = tl.load(WD + r[:, None] * HC * HS + k[None, :], r[:, None] < R, 0).to(tl.float32)
    h = tl.sum(down * nv[None, :], 1).to(X.dtype.element_ty).to(tl.float32) / HC
    h = (h * tl.sigmoid(h)).to(X.dtype.element_ty).to(tl.float32)
    up = tl.load(WU + k[:, None] * R + r[None, :], r[None, :] < R, 0).to(tl.float32)
    u = tl.sum(up * h[None, :], 1).to(X.dtype.element_ty).to(tl.float32)
    mixed = tl.reshape(tl.sigmoid(u) * nv, (HC, HS))
    tl.store(Y + row * HS + d, tl.sum(mixed, 0) / HC)


@triton.jit
def _hc_combine(Y, X, N, W, O, HC: tl.constexpr, HS: tl.constexpr):
    row = tl.program_id(0)
    c = tl.arange(0, HC)
    k = tl.arange(0, HC * HS)
    n = tl.load(N + row * HC * HS + k).to(tl.float32)
    w = tl.load(W + c[:, None] * HC * HS + k[None, :]).to(tl.float32)
    gate = tl.sum(w * n[None, :], 1).to(X.dtype.element_ty).to(tl.float32)
    gate = 2.0 * tl.sigmoid(gate / HC)
    d = tl.arange(0, HS)
    y = tl.load(Y + row * HS + d).to(tl.float32)
    x = tl.load(X + row * HC * HS + c[:, None] * HS + d[None, :]).to(tl.float32)
    tl.store(O + row * HC * HS + c[:, None] * HS + d[None, :], x + gate[:, None] * y[None, :])


def tiny_hc_mix(x, nw, wd, wu, hc, hs, eps):
    out = x.new_empty(x.shape[0], hs)
    normed = torch.empty_like(x)
    _hc_mix[(x.shape[0],)](
        x,
        nw,
        wd,
        wu,
        out,
        normed,
        hc,
        hs,
        wd.shape[0],
        eps,
        triton.next_power_of_2(wd.shape[0]),
        num_warps=4,
        enable_fp_fusion=False,
    )
    return out, normed


def tiny_hc_combine(y, x, normed, w, hc, hs):
    out = torch.empty_like(x)
    _hc_combine[(x.shape[0],)](y, x, normed, w, out, hc, hs, num_warps=4, enable_fp_fusion=False)
    return out
