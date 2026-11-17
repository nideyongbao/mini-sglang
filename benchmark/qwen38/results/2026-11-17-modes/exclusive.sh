#!/usr/bin/env bash
set -euo pipefail
restore() {
 code=$?
 trap - EXIT INT TERM
 docker stop -t 10 minisgl-qwen38-dev >/dev/null || true
 MONITORING=0 bash /shard_data/brooksli/workspace/0826/deploy.sh
 python3 - <<'PY'
import time,urllib.request,json
op=urllib.request.build_opener(urllib.request.ProxyHandler({}))
for _ in range(300):
 try:
  with op.open('http://127.0.0.1:8000/health', timeout=5) as r:
   if r.status==200:
    print('ORIGINAL_SERVICE_RESTORED_HEALTHY',time.strftime('%Y-%m-%dT%H:%M:%S%z'),flush=True)
    print(op.open('http://127.0.0.1:8000/v1/models').read().decode());break
 except Exception: pass
 time.sleep(2)
else: raise SystemExit('Original service health check timed out')
PY
 exit "$code"
}
docker inspect sglang-qwen38-dflash2 > /tmp/qwen38-modes-original-inspect.json
trap restore EXIT INT TERM
docker stop -t 30 sglang-qwen38-dflash2
docker exec minisgl-qwen38-dev mkdir -p /tmp/q38-modes-results
nvidia-smi --query-gpu=name,uuid,memory.used,utilization.gpu,temperature.gpu,power.draw --format=csv
for spec in 'sglang strict' 'mini strict' 'mini fast' 'mini strict --eager' 'mini strict --retain-local-gdn' 'sglang strict --reference-policy=original'; do
 read -ra options <<< "$spec"
 label="${options[0]}-${options[1]}${options[2]:-}"
 echo "MEASURE $label $(date -Is)"
 docker exec -e PYTHONPATH=/workspace/python:/workspace/benchmark/qwen38/modes -w /workspace minisgl-qwen38-dev python benchmark/qwen38/performance.py --backend "${options[0]}" --mode "${options[1]}" "${options[@]:2}" --fixed-continuation --lengths 128 2048 3072 --batches 1 4 8 --output "/tmp/q38-modes-results/offline-$label.json" > "/tmp/q38-offline-$label.log" 2>&1
done
for mode in sglang strict fast; do
 if [[ "$mode" == sglang ]]; then backend=sglang; numerical=strict; else backend=mini; numerical="$mode"; fi
 echo "LONG_PRECISION $mode $(date -Is)"
 docker exec -e PYTHONPATH=/workspace/python:/workspace/benchmark/qwen38/modes -w /workspace minisgl-qwen38-dev python benchmark/qwen38/modes/precision.py --backend "$backend" --mode "$numerical" --graphs --constant-continuation --steps 128 --lengths 128 2048 3072 --batches 1 4 8 --directory /tmp/q38-performance-precision > "/tmp/q38-long-precision-$mode.log" 2>&1
 if [[ "$mode" != sglang ]]; then
  docker exec -w /workspace minisgl-qwen38-dev python benchmark/qwen38/modes/compare.py --directory /tmp/q38-performance-precision --mode "$mode" --steps 128 --output "/tmp/q38-modes-results/precision-$mode-128steps.json"
 fi
done
for mode in sglang strict fast; do
 echo "HTTP $mode $(date -Is)"
 docker stop -t 10 minisgl-qwen38-dev >/dev/null
 docker start minisgl-qwen38-dev >/dev/null
 if [[ "$mode" == sglang ]]; then script=reference; port=31038; else script=mini; port=31039; fi
 docker exec -d -e PYTHONPATH=/workspace/python:/workspace/benchmark/qwen38/modes -e MINISGL_QWEN38_REFERENCE_MODE=strict -e QWEN38_MODE="$mode" -w /workspace minisgl-qwen38-dev bash -c "exec bash benchmark/qwen38/$script.sh > /tmp/q38-http-$mode.log 2>&1"
 docker exec -e PYTHONPATH=/workspace/python -w /workspace minisgl-qwen38-dev python benchmark/qwen38/wait_ready.py "http://127.0.0.1:$port/v1/models"
 docker exec -w /workspace minisgl-qwen38-dev python benchmark/qwen38/online.py --url "http://127.0.0.1:$port" --backend "$mode" --output "/tmp/q38-modes-results/http-$mode.json" > "/tmp/q38-http-perf-$mode.log" 2>&1
 docker exec -w /workspace minisgl-qwen38-dev python benchmark/qwen38/modes/service_check.py --url "http://127.0.0.1:$port" --output "/tmp/q38-modes-results/service-$mode.json" > "/tmp/q38-service-$mode.log" 2>&1
done
nvidia-smi --query-gpu=name,memory.used,utilization.gpu,temperature.gpu,power.draw --format=csv
