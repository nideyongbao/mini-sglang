from __future__ import annotations

from typing import List

import torch
from minisgl.message import TokenizeMsg
from transformers import PreTrainedTokenizerBase


class TokenizeManager:
    def __init__(
        self, tokenizer: PreTrainedTokenizerBase, image_cache_enabled: bool = True
    ) -> None:
        self.tokenizer = tokenizer
        self.image_cache_enabled = image_cache_enabled
        self.image_processor = None

    def tokenize(self, msgs: List[TokenizeMsg]) -> List[torch.Tensor]:
        results: List[torch.Tensor] = []
        # TODO: batch tokenization
        for msg in msgs:
            if isinstance(msg.text, list):
                prompt = self.tokenizer.apply_chat_template(
                    msg.text,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                assert isinstance(prompt, str)
            else:
                prompt = msg.text
            input_ids: torch.Tensor = (  # type: ignore
                self.tokenizer.encode(prompt, return_tensors="pt")
            )
            results.append(input_ids.view(-1).to(torch.int32))
        return results

    def prepare(self, msg: TokenizeMsg):
        from minisgl.multimodal import ImageProcessor, has_images

        kwargs = dict(msg.sampling_params.chat_template_kwargs or {})
        if msg.sampling_params.reasoning_effort:
            kwargs["reasoning_effort"] = msg.sampling_params.reasoning_effort
        if has_images(msg.text):
            if self.image_processor is None:
                self.image_processor = ImageProcessor(
                    self.tokenizer.name_or_path, cache_enabled=self.image_cache_enabled
                )
            return self.image_processor.prepare(msg.text, **kwargs)
        prompt = (
            self.tokenizer.apply_chat_template(
                msg.text, tokenize=False, add_generation_prompt=True, **kwargs
            )
            if isinstance(msg.text, list)
            else msg.text
        )
        return self.tokenizer.encode(prompt, return_tensors="pt").view(-1).to(torch.int32), None
