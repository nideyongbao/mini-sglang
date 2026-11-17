#!/usr/bin/env bash
set -euo pipefail
exec python -m minisgl --model "${MODEL_PATH:-/model}" --dtype bfloat16 --qwen38-mode "${QWEN38_MODE:-strict}" \
  --host 0.0.0.0 --port 31039 --max-running-requests 8 \
  --max-seq-len-override 4096 --num-pages 32768 --max-prefill-length 512 \
  --cuda-graph-bs 1 2 3 4 5 6 7 8 --attention-backend fa --cache naive --disable-pynccl
