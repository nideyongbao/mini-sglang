# Strict / fast numerical modes

Run in the pinned local image with the repository mounted at `/workspace` and the tiny checkpoint at `/model`. Reference SGLang needs the H20 FA3 compatibility patch and the opt-in deterministic QSA tie policy:

```bash
export PYTHONPATH=/workspace/python:/workspace/benchmark/qwen38/modes
python benchmark/qwen38/patch_reference.py
python benchmark/qwen38/alignment/patch_modes.py
```

The patches only affect this test container. `MINISGL_QWEN38_REFERENCE_MODE=strict` enables the deterministic reference policy. Ordinary reference execution remains available with that variable unset or `original`.

Precision with CUDA graph replay (batch 1–8 captured, no padding):

```bash
python benchmark/qwen38/modes/precision.py --backend sglang --graphs --directory /tmp/precision
python benchmark/qwen38/modes/precision.py --backend mini --mode strict --graphs --directory /tmp/precision
python benchmark/qwen38/modes/precision.py --backend mini --mode fast --graphs --directory /tmp/precision
python benchmark/qwen38/modes/compare.py --directory /tmp/precision --mode strict --output /tmp/strict.json
python benchmark/qwen38/modes/compare.py --directory /tmp/precision --mode fast --output /tmp/fast.json
```

The default full-prefill budget is 32768 so all requested sequences enter the same forward batch. SGLang's independent `max_prefill_tokens` is also 32768. The comparator requires every requested output frame and exact input IDs, decode positions and logits shapes. Logits are saved before sampling and compared by FP32 bit representation; BF16 model logits are losslessly converted to FP32. Chunk-internal outputs and SGLang's discarded overlap decode are excluded. Raw tensors are large and should stay outside Git.

For chunked-prefill verification use `--chunk 512 --batches 1` on both sides. For identical performance histories use `--constant-continuation --steps 128 --lengths 128 2048 3072 --batches 1 4 8` and compare with `--steps 128`. `--retain-local-gdn` is a benchmark-only ablation that must be installed **before** graph capture. It saves under `strict-local-gdn`.

Exclusive timing, serially (stop other GPU workloads beforehand and restore them afterward):

```bash
python benchmark/qwen38/performance.py --backend sglang --fixed-continuation --lengths 128 2048 3072 --batches 1 4 8 --output /tmp/sglang.json
python benchmark/qwen38/performance.py --backend mini --mode strict --fixed-continuation --lengths 128 2048 3072 --batches 1 4 8 --output /tmp/strict.json
python benchmark/qwen38/performance.py --backend mini --mode fast --fixed-continuation --lengths 128 2048 3072 --batches 1 4 8 --output /tmp/fast.json
```

Timing disables tensor capture. Both samplers perform one device `fill_(1000)` per sampled batch to keep all subsequent model inputs identical, including n-gram state. Finite-distribution validation and warmup are outside timing. Results contain all five repetitions and prompt SHA-256. `--eager` measures strict without graph, `--retain-local-gdn` measures the local recurrence ablation, and `--reference-policy original` measures SGLang without stable QSA top-k.

HTTP uses `reference.sh` / `mini.sh` with chunk512 and graph1–8. Set `MINISGL_QWEN38_REFERENCE_MODE=strict` for the reference, and `QWEN38_MODE=strict|fast` for Mini-SGLang. `online.py` measures natural generation; it is a serving comparison, not a claim of identical internal batches. `service_check.py --url ... --output ...` verifies streaming/nonstreaming agreement, recovery after disconnect, mixed output lengths and repeated slot reuse.
