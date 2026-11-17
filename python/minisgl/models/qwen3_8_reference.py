"""Pinned Qwen38 reference operators. No global Mini-SGLang monkeypatches.

Imports are lazy so engine construction does not initialize CUDA early.
"""

import torch


def reference_hc(self):
    from sglang.srt.layers.hyperconnection import GatedResidual, HyperConnectionConfig

    if not hasattr(self, "_alignment_reference"):
        c = HyperConnectionConfig(
            hc_count=self.hc,
            hidden_size=self.hs,
            hc_lowrank=self.input_mix_weight_down.weight.shape[0],
            rms_norm_eps=self.hc_norm.eps,
            hc_per_branch_norm=True,
        )
        ref = GatedResidual(c, use_combine=hasattr(self, "block_inject_weight"))
        ref.hc_norm.weight = torch.nn.Parameter(self.hc_norm.weight, requires_grad=False)
        ref.input_mix_weight_down.weight = torch.nn.Parameter(
            self.input_mix_weight_down.weight, requires_grad=False
        )
        ref.input_mix_weight_up.weight = torch.nn.Parameter(
            self.input_mix_weight_up.weight, requires_grad=False
        )
        if hasattr(self, "block_inject_weight"):
            ref.block_inject_weight.weight = torch.nn.Parameter(
                self.block_inject_weight.weight, requires_grad=False
            )
        self._alignment_reference = ref
    return self._alignment_reference


def reference_prefill(q, k, v, indices):
    from minisgl.core import get_global_ctx
    from sglang.srt.layers.attention.qsa.sparse_attn import (
        sparse_gqa_fwd_interface_triton,
        sparse_gqa_fwd_interface_triton_ck,
    )

    ctx = get_global_ctx()
    lens, ext, keys, values, logical = [], [], [], [], []
    offset = 0
    for req in ctx.batch.reqs:
        n, end = req.extend_len, req.cached_len + req.extend_len
        slots = ctx.page_table[req.table_idx, :end].long()
        inverse = torch.empty(k.shape[0], device=q.device, dtype=torch.int32)
        inverse[slots] = torch.arange(end, device=q.device, dtype=torch.int32)
        part = indices[offset : offset + n]
        logical.append(torch.where(part >= 0, inverse[part.clamp_min(0).long()], -1))
        keys.append(k[slots])
        values.append(v[slots])
        lens.append(end)
        ext.append(n)
        offset += n
    lengths = torch.tensor(lens, device=q.device, dtype=torch.int32)
    extend = torch.tensor(ext, device=q.device, dtype=torch.int32)
    cuq = torch.cat((extend.new_zeros(1), extend.cumsum(0, dtype=torch.int32)))
    cuk = torch.cat((lengths.new_zeros(1), lengths.cumsum(0, dtype=torch.int32)))
    logical = torch.cat(logical).int().contiguous()
    if lens == ext:
        return sparse_gqa_fwd_interface_triton(
            q.contiguous(),
            torch.cat(keys),
            torch.cat(values),
            max(lens),
            logical,
            cuq,
            q.shape[-1] ** -0.5,
        )
    return sparse_gqa_fwd_interface_triton_ck(
        q.contiguous(),
        torch.cat(keys),
        torch.cat(values),
        logical,
        cuq,
        cuk,
        lengths,
        q.shape[-1] ** -0.5,
    )


def reference_decode(
    mixed_qkv,
    a,
    b,
    A_log,
    dt_bias,
    state,
    state_indices,
    num_q_heads,
    num_v_heads,
    head_k_dim,
    head_v_dim,
    scale,
    round_beta=False,
):
    from sglang.kernels.ops.attention.fla.fused_recurrent import (
        fused_recurrent_gated_delta_rule_packed_decode,
    )

    out = mixed_qkv.new_empty(mixed_qkv.shape[0], 1, num_v_heads, head_v_dim)
    fused_recurrent_gated_delta_rule_packed_decode(
        mixed_qkv,
        a,
        b,
        A_log,
        dt_bias,
        scale,
        state,
        out,
        state_indices,
        use_qk_l2norm_in_kernel=True,
    )
    return out[:, 0]


def reference_decode_scores(self, q, keys, positions):
    from sglang.srt.layers.attention.qsa.mqa import qsa_mqa_decode

    batch, blocks, dim = keys.shape
    assert blocks % 64 == 0
    table = torch.arange(batch * (blocks // 64), device=q.device, dtype=torch.int32).reshape(
        batch, blocks // 64
    )
    lengths = ((positions + 1) // self.ratio).int()
    width = blocks
    from minisgl.core import get_global_ctx

    ctx = get_global_ctx()
    if ctx.attn_backend.gdn_backend._capture_active_bs is None:
        width = ((max(r.cached_len + 1 for r in ctx.batch.reqs) + 63) // 64) * (64 // self.ratio)
    return qsa_mqa_decode(
        q, keys.reshape(batch * (blocks // 64), 64, 1, dim), table, lengths, width
    )


def reference_prefill_scores(self, q, keys, positions):
    from sglang.srt.layers.attention.qsa.mqa import qsa_mqa_prefill

    ends = ((positions + 1) // self.ratio).int()
    return qsa_mqa_prefill(q, keys[:, None], torch.zeros_like(ends), ends)


def reference_moe_reduce(x, out):
    from sgl_kernel import moe_sum_reduce
    from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import moe_sum_reduce_torch_compile

    if x.shape[0] <= 32:
        moe_sum_reduce_torch_compile(x, out, 1.0)
    else:
        moe_sum_reduce(x, out, 1.0)


def stable_topk(logits, lengths, topk, row_starts=None):
    starts = torch.zeros_like(lengths) if row_starts is None else row_starts
    columns = torch.arange(logits.shape[1], device=logits.device)[None]
    valid = (columns >= starts[:, None]) & (columns < (starts + lengths)[:, None])
    scores = logits.masked_fill(~valid, -float("inf"))
    order = scores.argsort(dim=-1, descending=True, stable=True)[:, :topk]
    ids = order - starts[:, None]
    ranks = torch.arange(order.shape[1], device=logits.device)[None]
    ids = torch.where(ranks < lengths[:, None], ids, -1).int()
    if ids.shape[1] < topk:
        ids = torch.nn.functional.pad(ids, (0, topk - ids.shape[1]), value=-1)
    ranks = torch.arange(topk, device=logits.device)[None].expand_as(ids)
    all_ids = torch.where(ranks < lengths[:, None], ranks, -1).int()
    ids = torch.where((lengths <= topk)[:, None], all_ids, ids)
    return ids.contiguous()


def install_reference_policy():
    """Explicit opt-in for the reference Qwen model in benchmark subprocesses."""
    import sglang.kernels.ops.elementwise.fast_topk as module

    module.fast_topk = stable_topk


def reference_index_q(self, qk, positions):
    from sglang.kernels.ops.attention.qsa_indexer import qsa_index_q_norm_rope_store

    cache = self._rotary._cos_sin_cache
    slots = torch.arange(qk.shape[0], device=qk.device, dtype=torch.int64)
    return qsa_index_q_norm_rope_store(
        qk.contiguous(),
        positions.long(),
        cache,
        torch.zeros(cache.shape[-1] // 2, device=qk.device, dtype=torch.int32),
        self.q_layernorm.weight,
        slots,
        qk.new_empty(qk.shape[0], self.dim),
        slots.new_empty(qk.shape[0], 3),
        self.nh,
        cache.shape[-1],
        self.q_layernorm.eps,
        True,
    )


def reference_index_k(self, groups, positions):
    from sglang.kernels.ops.attention.qsa_indexer import qsa_index_k_compress_store

    cache = self._rotary._cos_sin_cache
    count = groups.shape[0]
    members = torch.arange(count * self.ratio, device=groups.device, dtype=torch.int32)
    rope = (positions[:, None] + torch.arange(self.ratio, device=groups.device)).reshape(-1)
    out = groups.new_empty(count, self.dim)
    qsa_index_k_compress_store(
        groups.flatten(0, 1).contiguous(),
        members.reshape(count, self.ratio),
        rope[:, None].expand(-1, 3).contiguous().long(),
        cache,
        torch.zeros(cache.shape[-1] // 2, device=groups.device, dtype=torch.int32),
        self.k_layernorm.weight,
        torch.arange(count, device=groups.device, dtype=torch.int32),
        out,
        self.ratio,
        cache.shape[-1],
        self.k_layernorm.eps,
        True,
    )
    return out
