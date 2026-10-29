from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import torch
from minisgl.compilation import get_forward_context
from minisgl.core import get_global_ctx
from minisgl.distributed import get_tp_info
from minisgl.kernel.triton.qwen35_fused import gated_rmsnorm_
from minisgl.layers import (
    BaseOP,
    GemmaRMSNormFused,
    LinearColParallelMerged,
    LinearRowParallel,
    OPList,
    ParallelLMHead,
    RMSNorm,
    RadixLinearAttention,
    VocabParallelEmbedding,
)
from minisgl.utils import div_even, nvtx_annotate

from .base import BaseLLMModel
from .utils import GatedMLP as Qwen3_5MLP
from .utils import RopeAttn as Qwen3_5Attn

if TYPE_CHECKING:
    from .config import ModelConfig


def _get_layer_type(config: "ModelConfig", layer_id: int) -> str:
    if not config.layer_types:
        return "attention"
    layer_type = config.layer_types[layer_id]
    if layer_type == "full_attention":
        return "attention"
    return layer_type


class _DepthwiseConv1D(BaseOP):
    def __init__(self, channels: int, kernel_size: int):
        self.weight = torch.empty(channels, 1, kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class Qwen3_5GatedDeltaNet(BaseOP):
    def __init__(self, config: "ModelConfig", layer_id: int):
        if config.linear_num_key_heads <= 0 or config.linear_num_value_heads <= 0:
            raise ValueError("Invalid qwen3.5 linear attention config.")

        tp_size = get_tp_info().size
        self.layer_id = layer_id
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.num_q_heads = div_even(config.linear_num_key_heads, tp_size, allow_replicate=True)
        self.num_v_heads = div_even(config.linear_num_value_heads, tp_size, allow_replicate=True)
        if self.num_v_heads % self.num_q_heads != 0:
            raise ValueError("num_v_heads must be divisible by num_q_heads for GDN.")

        key_dim = config.linear_num_key_heads * self.head_k_dim
        value_dim = config.linear_num_value_heads * self.head_v_dim
        self.hidden_act = config.hidden_act

        self.in_proj_qkvz = self.create_qkvz_proj(
            hidden_size=config.hidden_size,
            key_dim=key_dim,
            value_dim=value_dim,
        )
        self.in_proj_ba = self.create_ba_proj(
            hidden_size=config.hidden_size,
            num_v_heads=config.linear_num_value_heads,
        )

        self.local_qk_dim = self.num_q_heads * self.head_k_dim
        self.local_v_dim = self.num_v_heads * self.head_v_dim
        self.local_conv_dim = 2 * self.local_qk_dim + self.local_v_dim

        self.conv1d = _DepthwiseConv1D(self.local_conv_dim, config.linear_conv_kernel_dim)
        self.A_log = torch.empty(self.num_v_heads)
        self.dt_bias = torch.empty(self.num_v_heads)
        self.attn = RadixLinearAttention(
            layer_id=layer_id,
            num_q_heads=self.num_q_heads,
            num_k_heads=self.num_q_heads,
            num_v_heads=self.num_v_heads,
            head_q_dim=self.head_k_dim,
            head_k_dim=self.head_k_dim,
            head_v_dim=self.head_v_dim,
            # Keep these as non-Tensor at init time to avoid state_dict loading
            # expecting duplicated keys under linear_attn.attn.*.
            conv_weights=None,
            bias=None,
            activation=self.hidden_act,
            A_log=None,
            dt_bias=None,
        )

        self.norm = RMSNorm(self.head_v_dim, eps=config.rms_norm_eps)
        self.out_proj = LinearRowParallel(
            value_dim,
            config.hidden_size,
            has_bias=False,
        )

    def create_qkvz_proj(
        self,
        hidden_size: int,
        key_dim: int,
        value_dim: int,
    ) -> LinearColParallelMerged:
        return LinearColParallelMerged(
            hidden_size,
            [key_dim, key_dim, value_dim, value_dim],
            has_bias=False,
        )

    def create_ba_proj(
        self,
        hidden_size: int,
        num_v_heads: int,
    ) -> LinearColParallelMerged:
        return LinearColParallelMerged(
            hidden_size,
            [num_v_heads, num_v_heads],
            has_bias=False,
        )

    def _unpack_input_projections(
        self,
        mixed_qkvz: torch.Tensor,
        mixed_ba: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mixed_qkv, z = mixed_qkvz.split([self.local_conv_dim, self.local_v_dim], dim=-1)
        b, a = mixed_ba.split([self.num_v_heads, self.num_v_heads], dim=-1)
        z = z.view(z.shape[0], self.num_v_heads, self.head_v_dim)
        return mixed_qkv, z, b, a

    def _forward_input_proj(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        projected_states_qkvz = self.in_proj_qkvz.forward(x)
        projected_states_ba = self.in_proj_ba.forward(x)
        return projected_states_qkvz, projected_states_ba

    @nvtx_annotate("GatedDeltaNet")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        forward_ctx = get_forward_context()
        forward_batch = forward_ctx.forward_batch if forward_ctx is not None else None

        projected_states_qkvz, projected_states_ba = self._forward_input_proj(x)
        mixed_qkv, z, b, a = self._unpack_input_projections(
            projected_states_qkvz, projected_states_ba
        )

        # Keep RadixLinearAttention references aligned after state_dict loading.
        self.attn.conv_weights = self.conv1d.weight
        self.attn.A_log = self.A_log
        self.attn.dt_bias = self.dt_bias
        core = self.attn.forward(
            forward_batch=forward_batch,
            mixed_qkv=mixed_qkv,
            a=a,
            b=b,
        ).to(dtype=z.dtype)

        core = core.reshape(-1, self.num_v_heads, self.head_v_dim)
        gated_rmsnorm_(core, z, self.norm.weight, self.norm.eps)
        core = core.reshape(x.shape[0], -1)
        return self.out_proj.forward(core)


class _Qwen3_5BaseDecoderLayer(BaseOP):
    def __init__(self, config: "ModelConfig", layer_id: int):
        self.mlp = Qwen3_5MLP(config)
        self.input_layernorm = GemmaRMSNormFused(
            size=config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.post_attention_layernorm = GemmaRMSNormFused(
            size=config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self._layer_id = layer_id

    def _run_attention(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @nvtx_annotate("Layer_{}", layer_id_field="_layer_id")
    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x, residual = self.input_layernorm.forward(x, residual)
        x = self._run_attention(x)
        x, residual = self.post_attention_layernorm.forward(x, residual)
        x = self.mlp.forward(x)
        return x, residual


class Qwen3_5LinearDecoderLayer(_Qwen3_5BaseDecoderLayer):
    """Qwen3.5 Decoder Layer with Linear Attention (GatedDeltaNet)."""

    def __init__(self, config: "ModelConfig", layer_id: int):
        self.linear_attn = Qwen3_5GatedDeltaNet(config, layer_id)
        super().__init__(config, layer_id)

    def _run_attention(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_attn.forward(x)


class Qwen3_5AttentionDecoderLayer(_Qwen3_5BaseDecoderLayer):
    """Qwen3.5 Decoder Layer with Full Attention."""

    def __init__(self, config: "ModelConfig", layer_id: int):
        self.self_attn = Qwen3_5Attn(
            config,
            layer_id,
            has_qk_norm=True,
            has_attn_output_gate=config.attn_output_gate,
            use_gemma_norm=True,
        )
        super().__init__(config, layer_id)

    def _run_attention(self, x: torch.Tensor) -> torch.Tensor:
        return self.self_attn.forward(x)


_DECODER_LAYER_REGISTRY = {
    "linear_attention": Qwen3_5LinearDecoderLayer,
    "attention": Qwen3_5AttentionDecoderLayer,
}


class Qwen3_5Model(BaseOP):
    def __init__(self, config: "ModelConfig"):
        self.embed_tokens = VocabParallelEmbedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
        )
        layers: list[BaseOP] = []
        for layer_id in range(config.num_layers):
            layer_type = _get_layer_type(config, layer_id)
            layer_cls = _DECODER_LAYER_REGISTRY.get(layer_type)
            if layer_cls is None:
                raise ValueError(f"Unsupported qwen3.5 layer type: {layer_type}")
            layers.append(layer_cls(config, layer_id))
        self.layers = OPList(layers)
        self.norm = GemmaRMSNormFused(
            size=config.hidden_size,
            eps=config.rms_norm_eps,
        )

    def forward(
        self, input_ids: torch.Tensor, inputs_embeds: torch.Tensor | None = None
    ) -> torch.Tensor:
        x = self.embed_tokens.forward(input_ids) if inputs_embeds is None else inputs_embeds
        residual: torch.Tensor | None = None
        for layer in self.layers.op_list:
            x, residual = layer.forward(x, residual)
        return self.norm.forward(x, residual)[0]


class Qwen3_5ForCausalLM(BaseLLMModel):
    def __init__(self, config: "ModelConfig"):
        self.model = Qwen3_5Model(config)
        if config.vision_config is not None:
            from .vision import Qwen35Vision

            self.visual = Qwen35Vision(config.vision_config)
        self.lm_head = ParallelLMHead(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            tie_word_embeddings=config.tie_word_embeddings,
            tied_embedding=self.model.embed_tokens if config.tie_word_embeddings else None,
        )
        super().__init__()

    def forward(self) -> torch.Tensor:
        batch = get_global_ctx().batch
        embeddings = None
        if batch.is_prefill and any(req.image_inputs is not None for req in batch.reqs):
            embeddings = self.model.embed_tokens.forward(batch.input_ids)
            offset = 0
            with torch.profiler.record_function("minisgl.vlm.vision_encoder"):
                for req in batch.reqs:
                    mm = req.image_inputs
                    if mm is not None:
                        if not hasattr(self, "visual"):
                            raise ValueError("This checkpoint has no vision encoder")
                        if mm.embeddings is None:
                            mm.embeddings = self.visual.forward(mm.pixel_values, mm.grid_thw)
                            if mm.embeddings.shape[0] != len(mm.image_indices):
                                raise ValueError(
                                    "Vision output does not match image placeholder count"
                                )
                        mask = (mm.image_indices >= req.cached_len) & (
                            mm.image_indices < req.device_len
                        )
                        indices = mm.image_indices[mask] - req.cached_len + offset
                        selected = mask.nonzero().flatten().to(embeddings.device)
                        embeddings[indices.to(embeddings.device)] = mm.embeddings[selected].to(
                            embeddings.dtype
                        )
                    offset += req.extend_len
        with torch.profiler.record_function(
            "minisgl.vlm.language_model_prefill" if batch.is_prefill else "minisgl.decode"
        ):
            output = self.model.forward(batch.input_ids, embeddings)
        logits = self.lm_head.forward(output)
        return logits


__all__ = ["Qwen3_5GatedDeltaNet", "Qwen3_5ForCausalLM"]
