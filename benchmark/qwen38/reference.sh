#!/usr/bin/env bash
set -euo pipefail
exec python -m sglang.launch_server \
  --model-path "${MODEL_PATH:-/model}" --host 127.0.0.1 --port 31038 \
  --language-only --fp8-gemm-backend triton --moe-runner-backend triton \
  --context-length 4096 --max-total-tokens 32768 --max-running-requests 8 \
  --max-mamba-cache-size 16 --chunked-prefill-size 512 --mem-fraction-static 0.9 \
  --cuda-graph-bs-decode 1 2 3 4 5 6 7 8 --disable-cuda-graph-padding --disable-radix-cache
