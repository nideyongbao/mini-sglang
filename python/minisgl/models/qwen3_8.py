"""Qwen3.8 Flash Next text inference, with native block-FP8 routed experts.

The model/scheduler/state ownership is Mini-SGLang's. SGLang supplies the same
low-level FP8 MoE kernels already used by the reference container.
"""

from __future__ import annotations

import re
from dataclasses import replace
from pathlib import Path

import torch
import torch.nn.functional as F
from minisgl.compilation import get_forward_context
from minisgl.core import get_global_ctx
from minisgl.layers import BaseOP, LinearReplicated, OPList, ParallelLMHead, VocabParallelEmbedding
from safetensors import safe_open

from .base import BaseLLMModel
from .qwen3_5 import Qwen3_5GatedDeltaNet, _DepthwiseConv1D
from .qwen3_8_qsa import QSAAttention
from .utils import GatedMLP
from .weight import _get_merge_info, _normalize_weight_name


class GroupNorm(BaseOP):
    def __init__(self, size, group, eps):
        self.weight = torch.empty(size)
        self.group, self.eps = group, eps

    def forward(self, x):
        if x.is_cuda and x.is_contiguous():
            from minisgl.kernel.triton.qwen38 import grouped_norm

            return grouped_norm(x, self.weight, self.group, self.eps)
        y = x.float().unflatten(-1, (-1, self.group))
        y = y * torch.rsqrt(y.square().mean(-1, keepdim=True) + self.eps)
        return (y.flatten(-2) * (1 + self.weight.float())).to(x.dtype)


@torch.compile
def _mix(x, down, up, hc, hs):
    gates = F.silu(F.linear(x, down) / hc)
    gates = torch.sigmoid(F.linear(gates, up))
    return (gates.unflatten(-1, (hc, hs)) * x.unflatten(-1, (hc, hs))).mean(-2)


@torch.compile
def _combine(y, x, normed, weight, hc, hs):
    gates = 2 * torch.sigmoid(F.linear(normed, weight) / hc)
    return (x.unflatten(-1, (hc, hs)) + y.unsqueeze(-2) * gates.unsqueeze(-1)).flatten(-2)


class HyperConnection(BaseOP):
    def __init__(self, config, combine=True):
        c = config.next_config
        self.hc, self.hs = c["hc_count"], config.hidden_size
        self._tiny = self.hc == 4 and self.hs == 32 and c["hc_lowrank"] == 4
        size = self.hc * self.hs
        self.hc_norm = GroupNorm(size, self.hs, config.rms_norm_eps)
        self.input_mix_weight_down = LinearReplicated(size, c["hc_lowrank"], False)
        self.input_mix_weight_up = LinearReplicated(c["hc_lowrank"], size, False)
        if combine:
            self.block_inject_weight = LinearReplicated(size, self.hc, False)

    def mix(self, x):
        if self._tiny and x.is_cuda:
            from minisgl.kernel.triton.qwen38 import tiny_hc_mix

            out, normed = tiny_hc_mix(
                x,
                self.hc_norm.weight,
                self.input_mix_weight_down.weight,
                self.input_mix_weight_up.weight,
                self.hc,
                self.hs,
                self.hc_norm.eps,
            )
            return out, (x, normed)
        normed = self.hc_norm.forward(x)
        return _mix(
            normed,
            self.input_mix_weight_down.weight,
            self.input_mix_weight_up.weight,
            self.hc,
            self.hs,
        ), (x, normed)

    def combine(self, y, residual):
        if self._tiny and y.is_cuda:
            from minisgl.kernel.triton.qwen38 import tiny_hc_combine

            return tiny_hc_combine(y, *residual, self.block_inject_weight.weight, self.hc, self.hs)
        return _combine(y, *residual, self.block_inject_weight.weight, self.hc, self.hs)


class FP8Experts(BaseOP):
    def __init__(self, config):
        self.block_shape = list(config.quantization_config["weight_block_size"])
        e, h, m = config.num_experts, config.hidden_size, config.moe_intermediate_size
        bn, bk = self.block_shape
        self.gate_up_proj = torch.empty(e, 2 * m, h, dtype=torch.float8_e4m3fn)
        self.down_proj = torch.empty(e, h, m, dtype=torch.float8_e4m3fn)
        self.gate_up_scale = torch.empty(
            e, 2 * ((m + bn - 1) // bn), (h + bk - 1) // bk, dtype=torch.float32
        )
        self.down_scale = torch.empty(
            e, (h + bn - 1) // bn, (m + bk - 1) // bk, dtype=torch.float32
        )
        self.top_k = config.num_experts_per_tok

    def forward(self, x, logits):
        import triton.language as tl
        from minisgl.kernel import moe_sum_reduce_triton
        from minisgl.layers import silu_and_mul
        from minisgl.moe.fused import fused_topk, moe_align_block_size
        from sglang.kernels.ops.moe.fused_moe_triton_kernels import invoke_fused_moe_kernel

        weights, ids = fused_topk(x, logits, self.top_k, True)
        config = dict(
            BLOCK_SIZE_M=64,
            BLOCK_SIZE_N=self.block_shape[0],
            BLOCK_SIZE_K=self.block_shape[1],
            GROUP_SIZE_M=32,
            num_warps=4,
            num_stages=3,
        )
        sorted_ids, experts, padded = moe_align_block_size(ids, 64, self.gate_up_proj.shape[0])
        m, h = x.shape
        n = self.gate_up_proj.shape[1]
        up = x.new_empty(m * self.top_k, n)
        act = x.new_empty(m * self.top_k, n // 2)
        down = x.new_empty(m * self.top_k, h)

        def gemm(a, w, scale, out, routed, topk):
            invoke_fused_moe_kernel(
                a,
                w,
                None,
                out,
                None,
                scale,
                None,
                weights,
                ids,
                sorted_ids,
                experts,
                padded,
                routed,
                topk,
                config,
                compute_type=tl.bfloat16,
                use_fp8_w8a8=True,
                use_int8_w8a8=False,
                use_int8_w8a16=False,
                use_int4_w4a16=False,
                per_channel_quant=False,
                block_shape=self.block_shape,
                filter_expert=False,
            )

        gemm(x, self.gate_up_proj, self.gate_up_scale, up, False, self.top_k)
        silu_and_mul(up, act)
        gemm(act, self.down_proj, self.down_scale, down, True, 1)
        out = torch.empty_like(x)
        moe_sum_reduce_triton(down.view(m, self.top_k, h), out)
        return out


class NextMoE(BaseOP):
    def __init__(self, config):
        self.gate = LinearReplicated(config.hidden_size, config.num_experts, False)
        self.experts = FP8Experts(config)
        self.shared_expert = GatedMLP(
            replace(config, intermediate_size=config.next_config["shared_expert_intermediate_size"])
        )
        self.shared_expert_gate = LinearReplicated(config.hidden_size, 1, False)

    def forward(self, x):
        shared = self.shared_expert.forward(x) * torch.sigmoid(self.shared_expert_gate.forward(x))
        return self.experts.forward(x, self.gate.forward(x)) + shared


class NGramEmbedding(BaseOP):
    def __init__(self, config):
        c = config.next_config
        self.order, self.heads = c["ngram_size"], c["heads_per_ngram"]
        self.eos = c["eos_token_id"]
        n = (self.order - 1) * self.heads
        self.layer_multipliers = torch.empty(self.order, dtype=torch.int64)
        self.ngram_heads_vocab_sizes = torch.empty(n, dtype=torch.int64)
        self.ngram_heads_offsets = torch.empty(n, dtype=torch.int64)
        # Table size comes from serialized shards; validate at load time.
        self._table = None
        self._scale = None

    def load_state_dict(self, state_dict, *, prefix="", _internal=False):
        self._table = state_dict.pop(prefix + ".ngram_embedding.weight")
        self._scale = state_dict.pop(prefix + ".ngram_embedding.weight_scale")
        super().load_state_dict(state_dict, prefix=prefix, _internal=True)
        if self._table.shape[0] < int(
            (self.ngram_heads_offsets + self.ngram_heads_vocab_sizes).max()
        ):
            raise ValueError("PLE table is smaller than its hash layout")

    def forward(self, contexts):
        # Contexts are oldest -> current. EOS ends the preceding segment;
        # the current EOS itself may still look back into that segment.
        shifted = [contexts[:, -1]]
        valid = torch.ones_like(shifted[0], dtype=torch.bool)
        for shift in range(1, self.order):
            previous = contexts[:, -1 - shift]
            valid = valid & (previous != self.eos)
            shifted.append(torch.where(valid, previous, self.eos))
        blocks = []
        mix = shifted[0] * self.layer_multipliers[0]
        for order in range(2, self.order + 1):
            mix = mix ^ (shifted[order - 1] * self.layer_multipliers[order - 1])
            sl = slice((order - 2) * self.heads, (order - 1) * self.heads)
            blocks.append(
                mix[:, None].remainder(self.ngram_heads_vocab_sizes[sl])
                + self.ngram_heads_offsets[sl]
            )
        ids = torch.cat(blocks, -1)
        # GPU indexing of float8 tables is not supported by all PyTorch builds.
        return (F.embedding(ids, self._table) * self._scale).flatten(-2)


class PLE(BaseOP):
    def __init__(self, config):
        c = config.next_config
        self.hs, self.hc = config.hidden_size, c["hc_count"]
        self.dilation, self.kernel = c["ngram_size"], c["ple_conv_kernel_size"]
        self.hist = (self.kernel - 1) * self.dilation
        self.ple_embedding = NGramEmbedding(config)
        width = self.hs * self.hc
        self.key_proj = LinearReplicated(c["ple_embed_dim"], width, False)
        self.value_proj = LinearReplicated(c["ple_embed_dim"], self.hs, False)
        self.norm_key = GroupNorm(width, self.hs, config.rms_norm_eps)
        self.norm_query = GroupNorm(width, self.hs, config.rms_norm_eps)
        self.norm_conv = GroupNorm(width, self.hs, config.rms_norm_eps)
        self.conv1d = _DepthwiseConv1D(width, self.kernel)
        self._state = None

    def forward(self, x, contexts, slots):
        ctx = get_global_ctx()
        if self._state is None:
            self._state = x.new_zeros(ctx.page_table.shape[0], self.hs * self.hc, self.hist)
        emb = self.ple_embedding.forward(contexts)
        k = self.norm_key.forward(self.key_proj.forward(emb)).unflatten(-1, (self.hc, self.hs))
        q = self.norm_query.forward(x).unflatten(-1, (self.hc, self.hs))
        gate = (k * q).sum(-1, keepdim=True) / self.hs**0.5
        gate = torch.sigmoid(gate.abs().clamp_min(1e-6).sqrt() * gate.sign())
        value = (gate * self.value_proj.forward(emb).unsqueeze(-2)).flatten(-2)
        normed = self.norm_conv.forward(value)
        if ctx.batch.is_decode:
            inputs = torch.cat((self._state[slots], normed.unsqueeze(-1)), -1)
            self._state[slots] = inputs[:, :, 1:]
            out = F.conv1d(
                inputs, self.conv1d.weight, groups=self.hs * self.hc, dilation=self.dilation
            ).squeeze(-1)
        else:
            outputs, offset = [], 0
            for req in ctx.batch.reqs:
                n, slot = req.extend_len, req.table_idx
                if req.cached_len == 0:
                    self._state[slot].zero_()
                inputs = torch.cat((self._state[slot], normed[offset : offset + n].T), -1)
                self._state[slot] = inputs[:, -self.hist :]
                outputs.append(
                    F.conv1d(
                        inputs[None],
                        self.conv1d.weight,
                        groups=self.hs * self.hc,
                        dilation=self.dilation,
                    )[0].T
                )
                offset += n
            out = torch.cat(outputs)
        return value + F.silu(out)


class NextLayer(BaseOP):
    def __init__(self, config, layer_id):
        if config.layer_types[layer_id] == "linear_attention":
            self.linear_attn = Qwen3_5GatedDeltaNet(config, layer_id)
        else:
            self.self_attn = QSAAttention(config, layer_id)
        self.attn_hyper_connection = HyperConnection(config)
        self.mlp_hyper_connection = HyperConnection(config)
        self.mlp = NextMoE(config)
        if layer_id + 1 in config.next_config["ple_layer_ids"]:
            self.ple = PLE(config)

    def forward(self, x, contexts, slots):
        if hasattr(self, "ple"):
            x = x + self.ple.forward(x, contexts, slots)
        h, residual = self.attn_hyper_connection.mix(x)
        attn = self.linear_attn if hasattr(self, "linear_attn") else self.self_attn
        x = self.attn_hyper_connection.combine(attn.forward(h), residual)
        h, residual = self.mlp_hyper_connection.mix(x)
        return self.mlp_hyper_connection.combine(self.mlp.forward(h), residual)


class NextModel(BaseOP):
    def __init__(self, config):
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = OPList([NextLayer(config, i) for i in range(config.num_layers)])
        self.hyper_connection_mixer = HyperConnection(config, combine=False)
        self.hc = config.next_config["hc_count"]
        self.order = config.next_config["ngram_size"]
        self.eos = config.next_config["eos_token_id"]
        self._history = None

    def forward(self):
        ctx = get_global_ctx()
        batch, ids = ctx.batch, ctx.batch.input_ids
        if self._history is None:
            self._history = torch.full(
                (ctx.page_table.shape[0], self.order - 1),
                self.eos,
                dtype=torch.int64,
                device=ids.device,
            )
        slots = None
        if batch.is_decode:
            fb = get_forward_context().forward_batch
            slots = ctx.attn_backend.gdn_backend._get_decode_state_indices(
                batch.padded_reqs, ids.device, fb
            ).long()
            contexts = torch.cat((self._history[slots], ids[:, None].long()), -1)
            self._history[slots] = contexts[:, 1:]
        else:
            contexts, offset = [], 0
            for req in batch.reqs:
                n, slot = req.extend_len, req.table_idx
                if req.cached_len == 0:
                    self._history[slot].fill_(self.eos)
                seq = torch.cat((self._history[slot], ids[offset : offset + n].long()))
                contexts.append(seq.unfold(0, self.order, 1))
                self._history[slot] = seq[-(self.order - 1) :]
                offset += n
            contexts = torch.cat(contexts)
        x = self.embed_tokens.forward(ids).repeat(1, self.hc)
        for layer in self.layers.op_list:
            x = layer.forward(x, contexts, slots)
        return self.hyper_connection_mixer.mix(x)[0]


class Qwen38ForCausalLM(BaseLLMModel):
    def __init__(self, config):
        if not config.quantization_config or config.quantization_config["quant_method"] != "fp8":
            raise ValueError("Qwen38 local requires a block-FP8 checkpoint")
        self.model = NextModel(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        self._config = config

    def forward(self):
        if any(r.image_inputs is not None for r in get_global_ctx().batch.reqs):
            raise ValueError("Qwen38 local supports text inputs only")
        return self.lm_head.forward(self.model.forward())

    def load_checkpoint(self, path, device):
        raw = {}
        for file in sorted(Path(path).glob("*.safetensors")):
            with safe_open(file, framework="pt", device=str(device)) as f:
                for key in f.keys():
                    if key.startswith(("model.visual.", "mtp.")):
                        continue
                    raw[_normalize_weight_name(key)] = f.get_tensor(key)
        # PLE shards have numeric ordering, not checkpoint lexicographic ordering.
        for key in list(raw):
            if ".ngram_embedding.shard_0.weight" in key:
                prefix = key.split(".shard_")[0]
                parts = [
                    raw.pop(f"{prefix}.shard_{i}.weight")
                    for i in range(self._config.next_config["split_ngram_parts"])
                ]
                raw[prefix + ".weight"] = torch.cat(parts).to(torch.bfloat16)
        out, merges, experts = {}, {}, {}
        for key, value in raw.items():
            if ".experts." in key:
                m = re.match(
                    r"(.+\.experts)\.(\d+)\.(gate_proj|up_proj|down_proj)\.(weight|weight_scale_inv)$",
                    key,
                )
                if m is None:
                    raise ValueError(f"Unknown expert tensor {key}")
                prefix, idx, proj, kind = m.groups()
                experts.setdefault(prefix, {})[(int(idx), proj, kind)] = value
            elif (info := _get_merge_info(key)) is not None:
                target, slot, names = info
                merges.setdefault(target, {})[slot] = value
                if all(n in merges[target] for n in names):
                    out[target] = torch.cat([merges[target][n] for n in names])
                    del merges[target]
            else:
                out[key] = value
        if merges:
            raise ValueError(f"Incomplete projection groups: {list(merges)}")
        for prefix, tensors in experts.items():
            for kind, suffix in [("weight", "proj"), ("weight_scale_inv", "scale")]:
                out[prefix + ".gate_up_" + suffix] = torch.stack(
                    [
                        torch.cat([tensors[(i, p, kind)] for p in ("gate_proj", "up_proj")])
                        for i in range(self._config.num_experts)
                    ]
                )
                out[prefix + ".down_" + suffix] = torch.stack(
                    [tensors[(i, "down_proj", kind)] for i in range(self._config.num_experts)]
                )
                if kind != "weight":
                    out[prefix + ".gate_up_" + suffix] = out[prefix + ".gate_up_" + suffix].float()
                    out[prefix + ".down_" + suffix] = out[prefix + ".down_" + suffix].float()
        return out
