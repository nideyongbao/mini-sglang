"""Qwen3.5 vision weights and HF vision architecture, using PyTorch SDPA."""

from __future__ import annotations

import torch
from minisgl.layers.base import BaseOP
from minisgl.image_cache import ImageEmbeddingCache
from minisgl.multimodal import ImageInputs, image_hashes


class Qwen35Vision(BaseOP):
    def __init__(self, config: dict):
        from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5VisionConfig
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel

        cfg = Qwen3_5VisionConfig(**config)
        cfg._attn_implementation = "sdpa"
        # Non-persistent rotary buffers must be materialized, not left on meta.
        with torch.device("cpu"):
            self._module = Qwen3_5VisionModel(cfg).eval()
        self._cache = ImageEmbeddingCache(0)

    def configure_cache(self, capacity_bytes: int):
        self._cache = ImageEmbeddingCache(capacity_bytes)

    def encode_images(self, inputs: ImageInputs) -> torch.Tensor:
        if self._cache.capacity_bytes == 0:
            with torch.profiler.record_function("minisgl.vlm.vision_encode_miss"):
                return self.forward(inputs.pixel_values, inputs.grid_thw)
        keys = inputs.image_hashes
        if keys is None:
            keys = image_hashes(inputs.pixel_values, inputs.grid_thw)
        if len(keys) != len(inputs.grid_thw):
            raise ValueError("Image cache keys do not match image grid")
        embeddings = []
        offset = 0
        for i, grid in enumerate(inputs.grid_thw.tolist()):
            size = grid[0] * grid[1] * grid[2]
            embedding = self._cache.get(keys[i])
            if embedding is None:
                with torch.profiler.record_function("minisgl.vlm.vision_encode_miss"):
                    embedding = self.forward(
                        inputs.pixel_values[offset : offset + size], inputs.grid_thw[i : i + 1]
                    )
                self._cache.put(keys[i], embedding)
            else:
                with torch.profiler.record_function("minisgl.vlm.image_cache_hit"):
                    pass
            embeddings.append(embedding)
            offset += size
        return embeddings[0] if len(embeddings) == 1 else torch.cat(embeddings, dim=0)

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
        self._cache.clear()  # Reloaded weights must never reuse old embeddings.

    def forward(self, pixels, grid):
        weight = self._module.patch_embed.proj.weight
        return self._module(
            pixels.to(device=weight.device, dtype=weight.dtype),
            grid.to(device=weight.device),
        ).pooler_output
