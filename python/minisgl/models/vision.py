"""Qwen3.5 vision weights and HF vision architecture, using PyTorch SDPA."""

from __future__ import annotations

import torch
from minisgl.layers.base import BaseOP


class Qwen35Vision(BaseOP):
    def __init__(self, config: dict):
        from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5VisionConfig
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel

        cfg = Qwen3_5VisionConfig(**config)
        cfg._attn_implementation = "sdpa"
        # Non-persistent rotary buffers must be materialized, not left on meta.
        with torch.device("cpu"):
            self._module = Qwen3_5VisionModel(cfg).eval()

    def state_dict(self, *, prefix="", result=None):
        result = {} if result is None else result
        result.update({f"{prefix}.{k}": v for k, v in self._module.state_dict().items()})
        return result

    def load_state_dict(self, state_dict, *, prefix="", _internal=False):
        keys = list(self._module.state_dict())
        weights = {k: state_dict.pop(f"{prefix}.{k}") for k in keys}
        device = next(iter(weights.values())).device
        self._module.load_state_dict(weights, strict=True, assign=True)
        self._module.to(device=device)

    def forward(self, pixels, grid):
        weight = self._module.patch_embed.proj.weight
        return self._module(
            pixels.to(device=weight.device, dtype=weight.dtype),
            grid.to(device=weight.device),
        ).pooler_output
