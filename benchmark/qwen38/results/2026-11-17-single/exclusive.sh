#!/usr/bin/env bash
set -euo pipefail
wait_url() {
 python3 - "$1" <<'PY'
import sys,time,urllib.request
op=urllib.request.build_opener(urllib.request.ProxyHandler({}))
for _ in range(300):
 try:
  if op.open(sys.argv[1],timeout=3).status==200:break
 except Exception:pass
 time.sleep(2)
else:raise SystemExit('Service restore timeout: '+sys.argv[1])
print('READY',sys.argv[1],time.strftime('%Y-%m-%dT%H:%M:%S%z'),flush=True)
PY
}
restore() {
 code=$?
 trap - EXIT INT TERM
 set +e
 docker stop -t 10 minisgl-qwen38-dev >/dev/null
 if [[ "$(docker inspect -f '{{.State.Running}}' sglang-qwen38-dflash2 2>/dev/null)" != true ]]; then
  MONITORING=0 bash /shard_data/brooksli/workspace/0826/deploy.sh
 fi
 wait_url http://127.0.0.1:8000/health
 original_health=$?
 docker start minisgl-qwen38-strict >/dev/null
 wait_url http://127.0.0.1:31039/v1/models
 strict_health=$?
 echo "RESTORE original=$original_health strict=$strict_health"
 if ((original_health || strict_health)); then exit 1; fi
 exit "$code"
}
offline() {
 local mode=$1 phase=$2 backend numerical
 if [[ "$mode" == sglang ]]; then backend=sglang; numerical=strict; else backend=mini; numerical=$mode; fi
 echo "OFFLINE $mode $phase $(date -Is)"
 docker exec -e PYTHONPATH=/workspace/python:/workspace/benchmark/qwen38/modes -w /workspace minisgl-qwen38-dev python benchmark/qwen38/performance.py --backend "$backend" --mode "$numerical" --fixed-continuation --chunk 512 --batches 1 --repeats 15 --warmups 3 --prompts-json /tmp/q38-single/cases.json --output "/tmp/q38-single/offline-$mode-$phase.json" > "/tmp/q38-single-$mode-$phase.log" 2>&1
}
docker inspect sglang-qwen38-dflash2 minisgl-qwen38-strict > /tmp/q38-single-before-inspect.json
trap restore EXIT INT TERM
docker stop -t 10 minisgl-qwen38-dev >/dev/null
docker stop -t 30 minisgl-qwen38-strict sglang-qwen38-dflash2
docker start minisgl-qwen38-dev >/dev/null
nvidia-smi --query-gpu=name,uuid,memory.used,utilization.gpu,temperature.gpu,power.draw --format=csv
for mode in sglang strict fast; do
 offline "$mode" before
 echo "HTTP $mode $(date -Is)"
 if [[ "$mode" == sglang ]]; then
  port=31038
  command='python benchmark/qwen38/single/serve_sglang.py --model-path /model --host 127.0.0.1 --port 31038 --language-only --fp8-gemm-backend triton --moe-runner-backend triton --context-length 4096 --max-total-tokens 32768 --max-running-requests 8 --max-mamba-cache-size 16 --chunked-prefill-size 512 --max-prefill-tokens 32768 --mem-fraction-static 0.9 --cuda-graph-bs-decode 1 2 3 4 5 6 7 8 --disable-cuda-graph-padding --disable-radix-cache'
 else
  port=31039
  command="python benchmark/qwen38/single/serve_mini.py --model /model --dtype bfloat16 --qwen38-mode $mode --host 127.0.0.1 --port 31039 --max-running-requests 8 --max-seq-len-override 4096 --num-pages 32768 --max-prefill-length 512 --cuda-graph-bs 1 2 3 4 5 6 7 8 --attention-backend fa --cache naive --disable-pynccl"
 fi
 docker exec -d -e PYTHONPATH=/workspace/python:/workspace/benchmark/qwen38/modes -e QWEN38_FIXED_CONTINUATION=1 -e MINISGL_QWEN38_REFERENCE_MODE=strict -e Q38_SINGLE_AUDIT="/tmp/q38-single/audit-$mode" -w /workspace minisgl-qwen38-dev bash -c "exec $command > /tmp/q38-single/server-$mode.log 2>&1"
 docker exec -w /workspace minisgl-qwen38-dev python benchmark/qwen38/wait_ready.py "http://127.0.0.1:$port/v1/models"
 docker exec -w /workspace minisgl-qwen38-dev python benchmark/qwen38/single/client.py --cases /tmp/q38-single/cases.json --url "http://127.0.0.1:$port" --backend "$mode" --audit "/tmp/q38-single/audit-$mode" --output "/tmp/q38-single/http-$mode.json" > "/tmp/q38-single-http-$mode.log" 2>&1
 docker stop -t 10 minisgl-qwen38-dev >/dev/null
 docker start minisgl-qwen38-dev >/dev/null
 offline "$mode" after
done
nvidia-smi --query-gpu=name,memory.used,utilization.gpu,temperature.gpu,power.draw --format=csv
