from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
from minisgl.core import SamplingParams
from minisgl.distributed import DistributedInfo
from minisgl.message import (
    BaseBackendMsg,
    DetokenizeMsg,
    UserMsg,
)
from minisgl.scheduler import Scheduler, SchedulerConfig


class RequestAllFinished(Exception):
    pass


@dataclass
class RequestStatus:
    uid: int
    input_ids: List[int]
    output_ids: List[int]


class LLM(Scheduler):
    def __init__(self, model_path: str, dtype: torch.dtype = torch.bfloat16, **kwargs):
        config = SchedulerConfig(
            model_path=model_path,
            tp_info=DistributedInfo(0, 1),
            dtype=dtype,
            offline_mode=True,
            **kwargs,
        )
        super().__init__(config)
        self._image_cache_enabled = config.image_cache_size_mb > 0
        self._model_path = model_path
        self._image_processor = None
        self.pending_requests: List[Tuple[List[int] | str | dict, SamplingParams]] = []
        self.status_map: Dict[int, RequestStatus] = {}
        self.counter = 0

    def _tokenize_one(self, prompt: List[int] | str) -> torch.Tensor:
        if isinstance(prompt, str):
            return self.tokenizer.encode(prompt, return_tensors="pt").view(-1).to(torch.int32)
        else:
            return torch.tensor(prompt, dtype=torch.int32, device="cpu")

    def offline_receive_msg(self, blocking: bool = False) -> List[BaseBackendMsg]:
        if blocking and len(self.pending_requests) == 0:
            raise RequestAllFinished()
        results: List[BaseBackendMsg] = []
        added, sum_input_len = 0, 0
        for tokens_or_prompt, sampling_params in self.pending_requests:
            if sum_input_len >= self.prefill_budget:
                break
            image_inputs = None
            if isinstance(tokens_or_prompt, dict):
                from minisgl.multimodal import ImageProcessor

                if self._image_processor is None:
                    self._image_processor = ImageProcessor(
                        self._model_path, cache_enabled=self._image_cache_enabled
                    )
                template_kwargs = dict(sampling_params.chat_template_kwargs or {})
                if sampling_params.reasoning_effort:
                    template_kwargs["reasoning_effort"] = sampling_params.reasoning_effort
                input_ids, image_inputs = self._image_processor.prepare(
                    tokens_or_prompt["messages"], **template_kwargs
                )
            else:
                input_ids = self._tokenize_one(tokens_or_prompt)
            sum_input_len += len(input_ids)
            uid, added = self.counter + added, added + 1
            results.append(
                UserMsg(
                    uid=uid,
                    input_ids=input_ids,
                    sampling_params=sampling_params,
                    image_inputs=image_inputs,
                )
            )
            self.status_map[uid] = RequestStatus(
                uid=uid,
                input_ids=(
                    input_ids.tolist()
                    if isinstance(tokens_or_prompt, (str, dict))
                    else tokens_or_prompt
                ),
                output_ids=[],
            )
        self.counter += added
        self.pending_requests = self.pending_requests[added:]
        return results

    def offline_send_result(self, reply: List[DetokenizeMsg]) -> None:
        for msg in reply:
            status = self.status_map[msg.uid]
            if not (msg.finished and msg.next_token == self.eos_token_id):
                status.output_ids.append(msg.next_token)

    def generate(
        self,
        prompts: List[str] | List[List[int]] | List[dict],
        sampling_params: List[SamplingParams] | SamplingParams,
    ) -> List[Dict[str, str | List[int]]]:
        self.pending_requests = []
        self.status_map = {}
        self.counter = 0
        if isinstance(sampling_params, SamplingParams):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            self.pending_requests.append((prompt, sp))
        try:
            self.run_forever()
        except RequestAllFinished:
            pass
        results: List[Dict[str, str | List[int]]] = []
        for i in range(len(prompts)):
            status = self.status_map[i]
            output_text = self.tokenizer.decode(status.output_ids)
            results.append({"text": output_text, "token_ids": status.output_ids})
        return results

    def clear_image_cache(self) -> None:
        """Clear cross-request image embeddings and counters between generate calls."""
        visual = getattr(self.engine.model, "visual", None)
        if visual is not None:
            visual._cache.clear()

    def image_cache_stats(self) -> dict[str, int]:
        visual = getattr(self.engine.model, "visual", None)
        return visual._cache.stats() if visual is not None else {}
