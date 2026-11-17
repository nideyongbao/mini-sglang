#!/usr/bin/env bash
set -euo pipefail
python3 - <<'PY'
import time,urllib.request
op=urllib.request.build_opener(urllib.request.ProxyHandler({}))
for _ in range(300):
 try:
  if op.open('http://127.0.0.1:8000/health',timeout=3).status==200:break
 except Exception:pass
 time.sleep(2)
else:raise SystemExit('Original service not ready')
print('Original service healthy before additional graph boundary checks')
PY
docker exec -e PYTHONPATH=/workspace/python:/workspace/benchmark/qwen38/modes -w /workspace minisgl-qwen38-dev bash -c '
set -e
for backend in sglang mini; do
 python benchmark/qwen38/modes/precision.py --backend $backend --graphs --lengths 1 3 4 16 2049 --batches 3 5 6 7 --directory /tmp/q38-graph-boundaries > /tmp/q38-graph-boundaries-$backend.log 2>&1
done
python benchmark/qwen38/modes/compare.py --directory /tmp/q38-graph-boundaries --output /tmp/q38-modes-results/precision-graph-boundaries.json
python -c "import json; s=json.load(open(\"/tmp/q38-modes-results/precision-graph-boundaries.json\"))[\"summary\"]; assert s[\"exact\"] == s[\"frames\"], s"
'
docker run -d --name minisgl-qwen38-strict --gpus all --shm-size 2g --pull never  -p 127.0.0.1:31039:31039  -v /shard_data/brooksli/workspace/0904/mini-sglang-qwen3.5-textonly:/workspace  -v /shard_data/brooksli/models/tiny-random/qwen3.8-flash-next-fp8:/model:ro  -e PYTHONPATH=/workspace/python -w /workspace  --entrypoint bash lmsysorg/sglang:dev-cu12-qwen38-next-local  -lc 'exec bash benchmark/qwen38/mini.sh'
docker exec -w /workspace minisgl-qwen38-strict python benchmark/qwen38/wait_ready.py http://127.0.0.1:31039/v1/models
docker exec -w /workspace minisgl-qwen38-strict python benchmark/qwen38/modes/service_check.py --url http://127.0.0.1:31039 --output /workspace/benchmark/qwen38/results/2026-11-17-modes/deployment-service.json
docker stop -t 10 minisgl-qwen38-dev >/dev/null
