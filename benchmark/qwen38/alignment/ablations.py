"""Experimental operator substitutions; explicit opt-in, no production defaults."""

import torch


def install(variant):
    import minisgl.attention.gdn as gdn_backend
    import minisgl.models.qwen3_8_qsa as qsa_module
    from minisgl.kernel.triton.gdn_decode import packed_decode as local_decode
    from minisgl.kernel.triton.qsa import sparse_decode as local_sparse_decode
    from minisgl.models.qwen3_8 import HyperConnection, NextMoE
    from minisgl.models.qwen3_8_qsa import QSAIndexer

    def legacy_shared(self, x):
        shared = self.shared_expert.forward(x) * torch.sigmoid(self.shared_expert_gate.forward(x))
        return self.experts.forward(x, self.gate.forward(x)) + shared

    NextMoE.forward = legacy_shared

    def legacy_rope(self, x, pos):
        cache = self._rotary._cos_sin_cache[pos.long()]
        half = cache.shape[-1] // 2
        cos, sin = cache[:, None, :half], cache[:, None, half:]
        a, b = x[..., :half].float(), x[..., half : 2 * half].float()
        y = torch.cat((a * cos - b * sin, b * cos + a * sin), -1).to(x.dtype)
        return torch.cat((y, x[..., 2 * half :]), -1)

    QSAIndexer.rope = legacy_rope

    def legacy_select(self, scores, ends):
        from sglang.srt.layers.attention.qsa.kernel import qsa_fast_topk

        return qsa_fast_topk(scores, torch.zeros_like(ends), ends, self.budget // self.ratio)

    QSAIndexer.select_decode_blocks = legacy_select

    def legacy_decode(*args, **kwargs):
        kwargs["round_beta"] = False
        return local_decode(*args, **kwargs)

    gdn_backend.gdn_packed_decode = legacy_decode
    qsa_module.sparse_decode = lambda *args: local_sparse_decode(*args, zero_scratch=False)
    flags = set() if variant == "baseline" else set(variant.split("+"))
    assert flags <= {
        "shared",
        "gdn",
        "hc",
        "rounded",
        "qsa",
        "reduce",
        "index",
        "rope",
        "zero",
        "direct",
        "width",
        "router",
    }, flags
    assert "width" not in flags or "index" in flags, "width requires index"
    if "shared" in flags:
        from sglang.kernels.ops.elementwise.elementwise import fused_gate_sigmoid_mul_add

        def forward(self, x):
            shared = self.shared_expert.forward(x)
            routed = self.experts.forward(x, self.gate.forward(x))
            fused_gate_sigmoid_mul_add(x, self.shared_expert_gate.weight.squeeze(), shared, routed)
            return routed

        NextMoE.forward = forward
    if "gdn" in flags:
        import minisgl.attention.gdn as backend

        def packed_decode(
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

        backend.gdn_packed_decode = packed_decode
    if "hc" in flags:

        def reference(self):
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

        HyperConnection.mix = lambda self, x: reference(self).mix(x)
        HyperConnection.combine = lambda self, y, res: reference(self).combine(y, res)

    if "rounded" in flags:
        import minisgl.attention.gdn as backend

        def rounded_decode(*args, **kwargs):
            kwargs["round_beta"] = True
            return local_decode(*args, **kwargs)

        backend.gdn_packed_decode = rounded_decode
    if "qsa" in flags:
        import minisgl.models.qwen3_8_qsa as module

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

        module.sparse_attention = reference_prefill

    if "reduce" in flags:
        import minisgl.kernel as kernels

        def reduce(x, out):
            from sgl_kernel import moe_sum_reduce
            from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import (
                moe_sum_reduce_torch_compile,
            )

            if x.shape[0] <= 32:
                moe_sum_reduce_torch_compile(x, out, 1.0)
            else:
                moe_sum_reduce(x, out, 1.0)

        kernels.moe_sum_reduce_triton = reduce

    if "index" in flags:
        from minisgl.models.qwen3_8_qsa import QSAIndexer

        def decode_scores(self, q, keys, positions):
            from sglang.srt.layers.attention.qsa.mqa import qsa_mqa_decode

            batch, blocks, dim = keys.shape
            assert blocks % 64 == 0
            table = torch.arange(
                batch * (blocks // 64), device=q.device, dtype=torch.int32
            ).reshape(batch, blocks // 64)
            lengths = ((positions + 1) // self.ratio).int()
            width = blocks
            if "width" in flags:
                from minisgl.core import get_global_ctx

                ctx = get_global_ctx()
                if ctx.attn_backend.gdn_backend._capture_active_bs is None:
                    width = ((max(r.cached_len + 1 for r in ctx.batch.reqs) + 63) // 64) * 16
            return qsa_mqa_decode(
                q, keys.reshape(batch * (blocks // 64), 64, 1, dim), table, lengths, width
            )

        def prefill_scores(self, q, keys, positions):
            from sglang.srt.layers.attention.qsa.mqa import qsa_mqa_prefill

            ends = ((positions + 1) // self.ratio).int()
            return qsa_mqa_prefill(q, keys[:, None], torch.zeros_like(ends), ends)

        QSAIndexer.decode_scores = decode_scores
        QSAIndexer.prefill_scores = prefill_scores

    if "rope" in flags:
        from minisgl.models.qwen3_8_qsa import QSAIndexer

        def rope(self, x, positions):
            cache = self._rotary._cos_sin_cache[positions.long()].to(x.dtype)
            half = cache.shape[-1] // 2
            cos, sin = cache[:, None, :half], cache[:, None, half:]
            a, b = x[..., :half], x[..., half : 2 * half]
            y = torch.cat((a * cos - b * sin, b * cos + a * sin), -1)
            return torch.cat((y, x[..., 2 * half :]), -1)

        QSAIndexer.rope = rope
    if "zero" in flags:
        import minisgl.models.qwen3_8_qsa as module
        from minisgl.kernel.triton.qsa import sparse_decode

        module.sparse_decode = lambda *args: sparse_decode(*args, zero_scratch=True)

    if "direct" in flags:
        import sglang.srt.layers.attention.qsa.kernel as kernel

        original_topk = kernel.qsa_fast_topk

        def topk(logits, starts, ends, topk):
            from minisgl.core import get_global_ctx

            if get_global_ctx().batch.is_decode:
                from sglang.kernels.ops.elementwise.fast_topk import fast_topk

                return fast_topk(logits, ends.int(), topk=topk, row_starts=None)
            return original_topk(logits, starts, ends, topk)

        kernel.qsa_fast_topk = topk

    if "router" in flags:
        import minisgl.moe.fused as routing

        def reference_topk(*args, **kwargs):
            from sglang.srt.layers.moe.topk import fused_topk

            return fused_topk(*args, **kwargs)

        routing.fused_topk = reference_topk
