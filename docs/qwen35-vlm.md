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
  --memory-ratio 0.8 --cache-type naive --num-tokenizer 1 --image-cache-size-mb 100
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
- A model-local GPU image embedding cache is enabled by default, with a 100 MiB LRU budget per rank (matching the tested SGLang default). Use `--image-cache-size-mb 0` online or `LLM(..., image_cache_size_mb=0)` offline to disable cross-request reuse. Automatic KV allocation reserves this cache budget.
- Cache keys hash each image's normalized processor pixels, grid, shape and dtype with SHA-256. Model identity and embedding dtype are scoped to the owning model instance. URLs, prompt text and placeholder token IDs are not cache identities. The same image with different text, or in another multi-image order, can hit the cache.
- Only the vision embedding is cached. Text/image decoding, image preprocessing, language prefill and autoregressive decode still run. This cache is independent of text KV-prefix caching.
- Cache entries own their tensor storage and are byte-accounted; least-recently-used images are evicted when necessary. An image larger than the budget is computed without insertion and does not evict useful smaller entries. Embeddings referenced by an active chunked request remain valid after eviction; their temporary memory is separate from the retained-cache budget. Weight reload and shutdown clear cached embeddings.
- For offline inspection use `llm.image_cache_stats()` and `llm.clear_image_cache()` between `generate` calls. Stats include hits, misses, entries, retained bytes, capacity and evictions. Clearing resets the counters. This version does not expose an HTTP cache-management endpoint.
- Still-image inputs may coexist with text requests. Multiple image grids and non-square images use the same position builder. The measured workload is one 1024×1024 image with 1024 text tokens on Qwen3.5-2B, TP=1.
- The HTTP interface accepts base64 image data URLs. Remote URL downloading, video/audio inputs and Qwen3.5 MoE multimodal models are outside this implementation's validated scope. TP>1 image inference has not been benchmarked.

## Validation and profiles

`tests/core/test_multimodal.py` compares image coordinates to Transformers, checks malformed grids and message serialization, and covers overlap completion accounting. `tests/kernel/test_mrope.py` verifies the fused multi-axis cache against a tensor reference.

Actual model validation includes graph/eager decode, 512-token chunked prefill crossing image boundaries, mixed text/image batches, repeat requests, and red/blue image answers. See the benchmark artifacts under `../benchmarks/qwen35-vlm-20261029` relative to the repository root.

Performance uses the unchanged `0826/inference_bench/runners/online.py` client and metrics. Profile collection runs after timed measurements. Open `mini-sglang-vlm-prefill.trace.json` and `sglang-vlm-prefill.trace.json` in Perfetto; compare vision and language-prefill ranges separately from overlap decode. Profiler timings contain instrumentation overhead.

## Image cache validation and benchmarks

`tests/core/test_image_cache.py` covers hash inputs, per-image reordering, model isolation, capacity accounting, LRU, oversized images and disabling. Real Qwen3.5-2B tests also verify that warm images cause zero vision forward calls and produce the same tokens as cold/disabled requests. The cache-enabled comparison is stored in `../benchmarks/qwen35-vlm-cache-20261029`; the earlier `qwen35-vlm-20261029` report deliberately disabled image caching.

For a fair SGLang comparison, keep the same text KV-cache policy and image-preprocessing-cache policy, and pair Mini's `--image-cache-size-mb 100` with `SGLANG_VLM_CACHE_SIZE_MB=100`. A cold image miss still runs vision; repeated-image hit latency must be reported separately from misses. Profile annotations distinguish `minisgl.vlm.vision_encode_miss` and `minisgl.vlm.image_cache_hit`.
