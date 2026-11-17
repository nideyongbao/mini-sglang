# Qwen3.8 intermediate-tensor alignment

Run in the pinned `lmsysorg/sglang:dev-cu12-qwen38-next-local` development container, with the repository at `/workspace` and tiny checkpoint at `/model`. The reference FA3 compatibility patch in `../patch_reference.py` is required on this H20.

The complete core matrix is available as `bash benchmark/qwen38/alignment/run.sh`. Individual steps:

```bash
export PYTHONPATH=/workspace/python:/workspace/benchmark/qwen38/alignment
python benchmark/qwen38/alignment/patch_probe.py
python benchmark/qwen38/alignment/capture.py --backend sglang
python benchmark/qwen38/alignment/capture.py --backend mini
python benchmark/qwen38/alignment/compare.py --output /tmp/qwen38-alignment/baseline-comparison.json
python benchmark/qwen38/alignment/replay_operators.py
```

Capture uses eager execution, TP=1, context=4096, chunk=512, fixed token IDs and eight generated tokens. The true full-vocabulary logits are saved **before** forcing the same continuation on both backends. Consequently output JSON contains forced tokens and is not an accuracy score. Compare `.pt` logits and intermediates instead. The comparison asserts identical token inputs and positions. SGLang may execute an extra unused decode forward; only common semantic positions are compared. Intermediate prefill chunk logits are diagnostic outputs, not returned generations.

`--random --lengths 2048 3073` reuses the prior long-accuracy prompts when their lengths match. `--batch 2 --random --lengths 16 128` covers packed prefill and multi-request decode. Use a separate `--directory` for these cases. The random-prompt source is the tracked benchmark result `../results/2026-11-17/accuracy-long.json`.

`--variant production` runs the current defaults. `baseline` and all other diagnostic variants explicitly restore the historical arithmetic first, so the original comparisons remain reproducible after fixes land.

Experimental substitutions are explicit `--variant` flags joined by `+`:

| Flag | Substitution |
|---|---|
| shared | Reference fused shared gate projection, sigmoid, multiply and routed add |
| gdn | Reference packed recurrent decode |
| rounded | Local recurrent decode with beta rounded to the input dtype |
| hc | Actual reference GatedResidual mix/combine and loaded mini weights |
| qsa | Reference sparse prefill and chunk-prefill kernels with logical/paged index conversion |
| reduce | Reference routed-expert sum dispatch |
| index | Reference TileLang index scoring |
| rope | Reference per-operation BF16 rounding in indexer RoPE |
| zero | Zero unused FA3 gather scratch |
| direct | Reference decode fast-topk entry without row-start tensor |
| width | Reference eager score width rounded to compressed page=16; requires index |
| router | Reference routed-MoE fused_topk dispatch, including normalization |

These substitutions are diagnostic and pinned to this image/configuration, not a general production backend. `probe.py` includes tiny-profile normalization for indexer heads and compression ratio. Raw captures include model inputs and should be handled as inference data. No tensor captures or forced-token hooks are active unless `QWEN38_TRACE_DIR/armed.json` exists.

Run performance without `QWEN38_TRACE_DIR`, serially on an idle GPU:

```bash
PYTHONPATH=/workspace/python python benchmark/qwen38/performance.py \
  --backend mini --ablation shared+rounded+rope+zero+direct \
  --lengths 128 2048 --batches 1 4 --output /tmp/ablation.json
```

Performance excludes loading, compilation, graph capture and tensor inspection. An untimed finite-output check precedes each measurement, followed by another warmup and five measured repetitions. Reference substitutions may cost additional gathers, kernels and host synchronization. HTTP and operator-level latency are not inferred from this offline measurement.

For a separate deterministic-index diagnostic, use `--stable-topk` on **both** backends with a fresh `--directory`. This replaces fast-topk tie handling by stable score/index ordering. It changes the reference policy, is deliberately slow, and is not part of production or reported ordinary-top-k performance.
