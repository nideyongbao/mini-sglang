# Matched single-request offline / HTTP comparison

Use the pinned local image and the same model mount as the other Qwen38 benchmarks.
This is a controlled benchmark, not a production launch path: the benchmark servers
force token 1000 after every sample, including spawned Mini-SGLang schedulers.
SGLang requires the existing FA3 and opt-in reference/continuation patches.
Production launch scripts and model operators are unchanged.

Prepare chat inputs once:

```bash
python benchmark/qwen38/single/client.py --prepare --cases /tmp/cases.json
```

This uses the checkpoint's chat template with thinking disabled and produces
140/2060/3084 token prompts. Offline generation consumes these exact token IDs.
HTTP uses the corresponding chat messages. The first request at each length saves
the server tokenizer's actual token IDs; the client requires exact equality before
measuring. Later requests only check the already-seen input length; there is no
per-step tensor capture or audit-file I/O during timing.

Offline example, CUDA graph enabled, batch=1, chunk=512:

```bash
export PYTHONPATH=/workspace/python:/workspace/benchmark/qwen38/modes
python benchmark/qwen38/performance.py --backend mini --mode strict \
  --fixed-continuation --chunk 512 --batches 1 --warmups 3 --repeats 15 \
  --prompts-json /tmp/cases.json --output /tmp/offline.json
```

Use `--backend sglang` or `--mode fast` for the other paths. For HTTP, launch
`serve_mini.py` or `serve_sglang.py` with the normal server arguments, identical
chunk512/context4096/graph1–8/token-pool32768/max-running8 configurations, and:

```bash
export Q38_SINGLE_AUDIT=/tmp/audit
export MINISGL_QWEN38_REFERENCE_MODE=strict
export QWEN38_FIXED_CONTINUATION=1
python benchmark/qwen38/single/client.py --cases /tmp/cases.json \
  --url http://127.0.0.1:31039 --backend strict --audit /tmp/audit \
  --output /tmp/http.json
```

The client makes one synchronous request at a time over a persistent loopback HTTP
connection, with 3 warmups and 15 measured requests per input/output shape. It
checks prompt length, output length and forced output text on every response.
Measure output1 and output128 separately; output1 is end-to-end one-token latency,
not streaming TTFT. Offline timings bracket HTTP timings to expose temporal drift.
Both offline and HTTP perform the normal sampler followed by one GPU fill per
sampled batch. Therefore these timings compare equal model histories but are not
natural-generation serving benchmarks.

The recorded exclusive shell script in the result directory is specific to this
host and includes the user's authorized stop/restore procedure for both services.
Do not run that script blindly on another deployment.
