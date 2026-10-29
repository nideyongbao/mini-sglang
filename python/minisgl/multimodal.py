"""Image input preparation for Qwen3.5. All preprocessing runs on the CPU.

Image embeddings are reused only within a request (for chunked prefill), never
across requests. KV token offsets and MRoPE coordinates are deliberately separate.
"""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from typing import Any

import torch
from PIL import Image


@dataclass
class ImageInputs:
    pixel_values: torch.Tensor
    grid_thw: torch.Tensor
    positions: torch.Tensor
    image_indices: torch.Tensor
    rope_delta: int
    embeddings: torch.Tensor | None = None


def image_positions(
    input_ids: torch.Tensor, grid_thw: torch.Tensor, image_token_id: int, merge_size: int
) -> tuple[torch.Tensor, int]:
    """Build interleaved temporal/height/width coordinates for still images."""
    ids = input_ids.tolist()
    positions = torch.empty((3, len(ids)), dtype=torch.int64, device="cpu")
    cursor = coordinate = 0
    for t, h, w in grid_thw.tolist():
        if t != 1 or h % merge_size or w % merge_size:
            raise ValueError("Only still-image grids divisible by spatial_merge_size are supported")
        h, w = h // merge_size, w // merge_size
        try:
            start = ids.index(image_token_id, cursor)
        except ValueError as exc:
            raise ValueError("Image grid has no matching image tokens") from exc
        n = h * w
        if ids[start : start + n] != [image_token_id] * n:
            raise ValueError("Image placeholder count does not match its grid")
        positions[:, cursor:start] = torch.arange(start - cursor) + coordinate
        coordinate += start - cursor
        positions[0, start : start + n] = coordinate
        positions[1, start : start + n] = torch.arange(h).repeat_interleave(w) + coordinate
        positions[2, start : start + n] = torch.arange(w).repeat(h) + coordinate
        coordinate += max(h, w)
        cursor = start + n
    if image_token_id in ids[cursor:]:
        raise ValueError("Image tokens have no matching image grid")
    positions[:, cursor:] = torch.arange(len(ids) - cursor) + coordinate
    return positions, int(positions.max()) + 1 - len(ids)


class ImageProcessor:
    def __init__(self, model_path: str):
        from transformers import AutoConfig, AutoProcessor

        self.processor = AutoProcessor.from_pretrained(model_path)
        self.config = AutoConfig.from_pretrained(model_path)
        if self.config.model_type != "qwen3_5":
            raise ValueError("Image inference currently supports dense Qwen3.5 checkpoints")

    def prepare(self, messages: list[dict[str, Any]], **template_kwargs):
        images = []
        for message in messages:
            content = message["content"]
            if isinstance(content, str):
                continue
            for part in content:
                if part["type"] == "image_url":
                    url = part["image_url"]["url"]
                    if not url.startswith("data:image/") or ";base64," not in url:
                        raise ValueError("Images must be base64 data:image/... URLs")
                    with Image.open(
                        io.BytesIO(base64.b64decode(url.split(",", 1)[1], validate=True))
                    ) as im:
                        images.append(im.convert("RGB"))
                elif part["type"] != "text":
                    raise ValueError("Only text and image_url content parts are supported")
        if not images:
            raise ValueError("An image request must contain at least one image")
        with torch.profiler.record_function("minisgl.vlm.image_preprocess"):
            prompt = self.processor.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, **template_kwargs
            )
            encoded = self.processor(text=[prompt], images=images, return_tensors="pt")
            ids = encoded["input_ids"][0].to(dtype=torch.int32, device="cpu")
            grid = encoded["image_grid_thw"].cpu()
            positions, delta = image_positions(
                ids, grid, self.config.image_token_id, self.config.vision_config.spatial_merge_size
            )
            mm = ImageInputs(
                encoded["pixel_values"].cpu(),
                grid,
                positions,
                (ids == self.config.image_token_id).nonzero().flatten(),
                delta,
            )
        return ids, mm


def has_images(messages) -> bool:
    return isinstance(messages, list) and any(
        isinstance(m.get("content"), list)
        and any(p.get("type") == "image_url" for p in m["content"])
        for m in messages
    )
