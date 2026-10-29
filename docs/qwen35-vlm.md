# Qwen3.5 image inference

This branch supports still images with dense Qwen3.5 conditional-generation checkpoints. It loads the checkpoint's vision encoder and merger, expands image placeholders with the Hugging Face processor, inserts visual embeddings into the language model, and applies interleaved temporal/height/width MRoPE. Generated tokens continue with the request's MRoPE delta; KV-cache addresses remain absolute. Decode CUDA graphs carry a separate RoPE-position buffer.

The vision architecture comes from Transformers (`Qwen3_5VisionModel`, SDPA attention); the language model, scheduler, KV cache, sampler and serving stack remain Mini-SGLang. This is not a Transformers language-model generation wrapper. The existing text path still uses the fused QK norm/RoPE kernel; a small Triton kernel prepares the multi-axis cosine/sine table once per image prefill batch.

## Environment

Validated on a single NVIDIA H20 with the local image `lmsysorg/sglang:dev-cu12-qwen38-27b-dflash2`, PyTorch 2.13.0+cu129 and Transformers 5.12.1. The branch uses the existing SGLang/FlashInfer kernels in that image. Set `PYTHONPATH` to this repository's `python` directory when using the image. Images require Pillow and torchvision. Older Transformers 4.x releases do not supply the Qwen3.5 vision architecture.

## Online serving

```bash
python -m minisgl \
  --model /shard_data/brooksli/models/Qwen/Qwen3.5-2B \
  --dtype bfloat16 --port 8000 --max-running-requests 8 \
  --max-seq-len-override 32768 --max-extend-length 4096 \
  --memory-ratio 0.8 --cache-type naive --num-tokenizer 1
```

POST to `/v1/chat/completions`, using OpenAI-style content parts:

```python
import base64
import requests

with open("image.png", "rb") as f:
    image_url = "data:image/png;base64," + base64.b64encode(f.read()).decode()
messages = [{"role": "user", "content": [
    {"type": "image_url", "image_url": {"url": image_url}},
    {"type": "text", "text": "Describe the objects in the image."},
]}]
response = requests.post("http://127.0.0.1:8000/v1/chat/completions", json={
    "model": "/shard_data/brooksli/models/Qwen/Qwen3.5-2B",
    "messages": messages, "max_tokens": 128, "temperature": 0,
    "chat_template_kwargs": {"enable_thinking": False},
})
print(response.json())
```

For SSE use `stream: true`. `stream_options: {"include_usage": true}` includes cumulative actual prompt/output counts and `prompt_tokens_details.image_tokens`, including steps whose decoded text is empty. The final finish reason follows collected host tokens, rather than the device cursor advanced by overlap scheduling.

## Offline inference

```python
from minisgl.llm import LLM
from minisgl.core import SamplingParams

llm = LLM("/shard_data/brooksli/models/Qwen/Qwen3.5-2B",
          max_running_req=1, max_extend_tokens=4096,
          max_seq_len_override=32768, cuda_graph_bs=[1], cache_type="naive")
outputs = llm.generate([{"messages": messages}],
                      SamplingParams(max_tokens=128, temperature=0,
                                     chat_template_kwargs={"enable_thinking": False}))
print(outputs)
llm.shutdown()
```

## Cache and scope

- Image requests bypass token-only prefix matching and insertion even when text radix caching is enabled. Identical placeholder IDs do not identify identical image contents.
- Vision embeddings are retained within one request for chunked prefill, then released with the request. There is no cross-request image embedding cache.
- Still-image inputs may coexist with text requests. Multiple image grids and non-square images use the same position builder. The measured workload is one 1024×1024 image with 1024 text tokens on Qwen3.5-2B, TP=1.
- The HTTP interface accepts base64 image data URLs. Remote URL downloading, video/audio inputs and Qwen3.5 MoE multimodal models are outside this implementation's validated scope. TP>1 image inference has not been benchmarked.

## Validation and profiles

`tests/core/test_multimodal.py` compares image coordinates to Transformers, checks malformed grids and message serialization, and covers overlap completion accounting. `tests/kernel/test_mrope.py` verifies the fused multi-axis cache against a tensor reference.

Actual model validation includes graph/eager decode, 512-token chunked prefill crossing image boundaries, mixed text/image batches, repeat requests, and red/blue image answers. See the benchmark artifacts under `../benchmarks/qwen35-vlm-20261029` relative to the repository root.

Performance uses the unchanged `0826/inference_bench/runners/online.py` client and metrics. Profile collection runs after timed measurements. Open `mini-sglang-vlm-prefill.trace.json` and `sglang-vlm-prefill.trace.json` in Perfetto; compare vision and language-prefill ranges separately from overlap decode. Profiler timings contain instrumentation overhead.
