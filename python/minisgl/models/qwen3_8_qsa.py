"""QSA compressed indexer with request-local state and paged sparse attention."""

import torch
from minisgl.compilation import get_forward_context
from minisgl.core import get_global_ctx
from minisgl.kernel.triton.qsa import sparse_attention, sparse_decode
from minisgl.kernel.triton.qwen35_fused import apply_attention_gate_, prepare_qk_gate
from minisgl.layers import BaseOP, GemmaRMSNorm, LinearReplicated

from .utils import RopeAttn


class QSAIndexer(BaseOP):
    def __init__(self, config, rotary):
        c = config.next_config
        self.nh, self.dim = c["indexer_n_heads"], c["indexer_head_dim"]
        self.ratio, self.budget = c["indexer_compress_ratio"], c["indexer_budget"]
        if c["indexer_kv_heads"] != 1 or self.budget % self.ratio:
            raise ValueError("QSA requires one index KV head and a divisible token budget")
        self.index_qk_proj = LinearReplicated(config.hidden_size, (self.nh + 1) * self.dim, False)
        self.q_layernorm = GemmaRMSNorm(self.dim, config.rms_norm_eps)
        self.k_layernorm = GemmaRMSNorm(self.dim, config.rms_norm_eps)
        self._rotary = rotary
        self._pending = None
        self._compressed = None

    def rope(self, x, pos):
        cache = self._rotary._cos_sin_cache[pos.long()]
        half = cache.shape[-1] // 2
        cos, sin = cache[:, :half, None].transpose(1, 2), cache[:, half:, None].transpose(1, 2)
        # Same FP32 rotation / BF16 cast boundary as the reference indexer.
        a, b = x[..., :half].float(), x[..., half : 2 * half].float()
        y = torch.cat((a * cos - b * sin, b * cos + a * sin), -1).to(x.dtype)
        return torch.cat((y, x[..., 2 * half :]), -1)

    def forward(self, x):
        from sglang.srt.layers.attention.qsa.kernel import expand_qsa_block_indices, qsa_fast_topk

        ctx = get_global_ctx()
        batch = ctx.batch
        slots_count, maxlen = ctx.page_table.shape
        nb = (maxlen + self.ratio - 1) // self.ratio
        if self._pending is None:
            self._pending = x.new_zeros(slots_count, self.ratio, self.dim)
            self._compressed = x.new_zeros(slots_count, nb, self.dim)
        qk = self.index_qk_proj.forward(x)
        q = self.q_layernorm.forward(qk[:, : self.nh * self.dim].reshape(-1, self.dim)).reshape(
            -1, self.nh, self.dim
        )
        q = self.rope(q, batch.positions)
        k = qk[:, self.nh * self.dim :]
        if batch.is_decode:
            fb = get_forward_context().forward_batch
            slots = ctx.attn_backend.gdn_backend._get_decode_state_indices(
                batch.padded_reqs, x.device, fb
            ).long()
            pos = batch.positions.long()
            self._pending[slots, pos % self.ratio] = k
            pooled = self._pending[slots].float().mean(1).to(x.dtype)
            compressed = self.rope(
                self.k_layernorm.forward(pooled)[:, None], (pos // self.ratio) * self.ratio
            )[:, 0]
            block = (pos // self.ratio).clamp(max=nb - 1)
            old = self._compressed[slots, block]
            self._compressed[slots, block] = torch.where(
                ((pos + 1) % self.ratio == 0)[:, None], compressed, old
            )
            keys = self._compressed[slots]
            scores = (
                torch.relu(torch.einsum("bhd,bnd->bhn", q.float(), keys.float())).sum(1)
                / self.dim**0.5
            )
            ends = ((pos + 1) // self.ratio).int()
            scores.masked_fill_(
                torch.arange(nb, device=x.device)[None] >= ends[:, None], -float("inf")
            )
            selected = qsa_fast_topk(
                scores.contiguous(), torch.zeros_like(ends), ends, self.budget // self.ratio
            )
            indices = expand_qsa_block_indices(selected, pos, pos + 1, self.ratio, self.budget)
            physical = ctx.page_table[slots[:, None], indices.long().clamp_min(0)]
        else:
            physical_parts = []
            offset = 0
            for req in batch.reqs:
                n, start, slot = req.extend_len, req.cached_len, req.table_idx
                end = start + n
                if start == 0:
                    self._pending[slot].zero_()
                    self._compressed[slot].zero_()
                rem = start % self.ratio
                keys = torch.cat((self._pending[slot, :rem], k[offset : offset + n]), 0)
                groups = keys.shape[0] // self.ratio
                first = start // self.ratio
                if groups:
                    pooled = (
                        keys[: groups * self.ratio]
                        .reshape(groups, self.ratio, self.dim)
                        .float()
                        .mean(1)
                        .to(x.dtype)
                    )
                    positions = torch.arange(first, first + groups, device=x.device) * self.ratio
                    self._compressed[slot, first : first + groups] = self.rope(
                        self.k_layernorm.forward(pooled)[:, None], positions
                    )[:, 0]
                tail = end % self.ratio
                if tail:
                    self._pending[slot, :tail] = keys[-tail:]
                pos = batch.positions[offset : offset + n].long()
                cached = self._compressed[slot, : max(1, end // self.ratio)]
                scores = (
                    torch.relu(
                        torch.einsum("mhd,nd->mhn", q[offset : offset + n].float(), cached.float())
                    ).sum(1)
                    / self.dim**0.5
                )
                ends = ((pos + 1) // self.ratio).int()
                scores.masked_fill_(
                    torch.arange(cached.shape[0], device=x.device)[None] >= ends[:, None],
                    -float("inf"),
                )
                selected = qsa_fast_topk(
                    scores.contiguous(), torch.zeros_like(ends), ends, self.budget // self.ratio
                )
                indices = expand_qsa_block_indices(selected, pos, pos + 1, self.ratio, self.budget)
                physical_parts.append(
                    torch.where(indices >= 0, ctx.page_table[slot, indices.long().clamp_min(0)], -1)
                )
                offset += n
            return torch.cat(physical_parts).contiguous()
        return torch.where(indices >= 0, physical, -1).contiguous()


class QSAAttention(RopeAttn):
    def __init__(self, config, layer_id):
        super().__init__(
            config, layer_id, has_qk_norm=True, has_attn_output_gate=True, use_gemma_norm=True
        )
        self.indexer = QSAIndexer(config, self.attn.rotary)
        self.budget = config.next_config["indexer_budget"]

    def forward(self, x):
        ctx = get_global_ctx()
        batch = ctx.batch
        # Exact dense equivalence when the entire configured context fits the budget.
        if ctx.page_table.shape[1] <= self.budget:
            return super().forward(x)
        indices = self.indexer.forward(x)
        qdim, kdim = self.attn.qo_attn_dim, self.attn.kv_attn_dim
        qg, k, v = self.qkv_proj.forward(x).split([2 * qdim, kdim, kdim], -1)
        q, k, gate = prepare_qk_gate(
            qg,
            k,
            query_weight=self.q_norm.weight,
            key_weight=self.k_norm.weight,
            cos_sin_cache=self.attn.rotary._cos_sin_cache,
            positions=batch.positions,
            eps=self.q_norm.eps,
            num_query_heads=self.attn.num_qo_heads,
            num_key_heads=self.attn.num_kv_heads,
            head_dim=self.attn.head_dim,
            rotary_dim=self.attn.rotary._cos_sin_cache.shape[-1],
        )
        ctx.kv_cache.store_kv(k, v.contiguous(), batch.out_loc, self.attn.layer_id)
        kcache = ctx.kv_cache.k_cache(self.attn.layer_id).view(
            -1, self.attn.num_kv_heads, self.attn.head_dim
        )
        vcache = ctx.kv_cache.v_cache(self.attn.layer_id).view_as(kcache)
        fn = sparse_decode if batch.is_decode else sparse_attention
        out = fn(
            q.reshape(-1, self.attn.num_qo_heads, self.attn.head_dim), kcache, vcache, indices
        ).flatten(1)
        apply_attention_gate_(out, gate)
        return self.o_proj.forward(out)
