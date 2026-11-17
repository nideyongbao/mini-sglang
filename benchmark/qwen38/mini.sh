#!/usr/bin/env bash
set -euo pipefail
exec python -m minisgl --model "${MODEL_PATH:-/model}" --dtype bfloat16 \
  --host 0.0.0.0 --port 31039 --max-running-requests 8 \
  --max-seq-len-override 4096 --num-pages 32768 --max-prefill-length 512 \
  --graph 8 --attention-backend fa --cache naive --disable-pynccl
