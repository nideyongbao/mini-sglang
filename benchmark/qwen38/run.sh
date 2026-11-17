#!/usr/bin/env bash
set -euo pipefail
repo=$(cd "$(dirname "$0")/../.." && pwd)
model=${MODEL_PATH:-/shard_data/brooksli/models/tiny-random/qwen3.8-flash-next-fp8}
image=${IMAGE:-lmsysorg/sglang:dev-cu12-qwen38-next-local}
mode=${1:-test}
results=${RESULTS_PATH:-$repo/benchmark/qwen38/results/local}
mkdir -p "$results"
docker_cmd=(docker)
if ! docker info >/dev/null 2>&1; then docker_cmd=(sudo -n docker); fi
args=(run --rm --gpus all --shm-size 2g --pull never
  -v "$repo:/workspace" -v "$model:/model:ro" -v "$results:/results"
  -e PYTHONPATH=/workspace/python:/workspace/benchmark/qwen38/modes
  -e QWEN38_MODE="${QWEN38_MODE:-strict}" -w /workspace)
case "$mode" in
  test) command='python -m pytest tests/models/test_qwen38.py tests/kernel/test_qwen35_fused.py tests/core/test_attention_backend.py -q -o addopts=' ;;
  accuracy) command='python benchmark/qwen38/patch_reference.py
    python benchmark/qwen38/alignment/patch_modes.py
    export MINISGL_QWEN38_REFERENCE_MODE=strict
    bash benchmark/qwen38/reference.sh > /results/reference.log 2>&1 &
    reference_pid=$!
    trap "kill $reference_pid 2>/dev/null || true" EXIT
    python benchmark/qwen38/wait_ready.py
    python benchmark/qwen38/accuracy.py --graphs --output /results/accuracy-short.json
    python benchmark/qwen38/accuracy.py --graphs --long --output /results/accuracy-long.json
    python benchmark/qwen38/accuracy.py --graphs --padded-graphs --chunk 127 --output /results/accuracy-chunk127-padded.json
    python benchmark/qwen38/accuracy.py --graphs --padded-graphs --long --output /results/accuracy-long-padded.json' ;;
  performance) command='python benchmark/qwen38/patch_reference.py
    python benchmark/qwen38/alignment/patch_modes.py
    export MINISGL_QWEN38_REFERENCE_MODE=strict
    python benchmark/qwen38/performance.py --backend sglang --fixed-continuation --output /results/performance-sglang.json
    python benchmark/qwen38/performance.py --backend mini --mode strict --fixed-continuation --output /results/performance-strict.json
    python benchmark/qwen38/performance.py --backend mini --mode fast --fixed-continuation --output /results/performance-fast.json' ;;
  online-sglang) command='python benchmark/qwen38/patch_reference.py
    python benchmark/qwen38/alignment/patch_modes.py
    export MINISGL_QWEN38_REFERENCE_MODE=strict
    bash benchmark/qwen38/reference.sh > /results/reference-online.log 2>&1 &
    reference_pid=$!
    trap "kill $reference_pid 2>/dev/null || true" EXIT
    python benchmark/qwen38/wait_ready.py
    python benchmark/qwen38/online.py --backend sglang --url http://127.0.0.1:31038 --output /results/online-sglang.json' ;;
  online-mini) command='bash benchmark/qwen38/mini.sh > /results/mini-online.log 2>&1 &
    mini_pid=$!
    trap "kill $mini_pid 2>/dev/null || true" EXIT
    python benchmark/qwen38/wait_ready.py http://127.0.0.1:31039/v1
    python benchmark/qwen38/online.py --backend mini --url http://127.0.0.1:31039 --output /results/online-mini.json' ;;
  serve) args+=(-p "${PORT:-31039}:31039")
    command='bash benchmark/qwen38/mini.sh' ;;
  *) echo "Usage: $0 {test|accuracy|performance|online-sglang|online-mini|serve}" >&2; exit 2 ;;
esac
exec "${docker_cmd[@]}" "${args[@]}" --entrypoint bash "$image" -lc "set -euo pipefail
$command"
