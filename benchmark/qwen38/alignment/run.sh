#!/usr/bin/env bash
# Inside the pinned development container; captures intentionally disable CUDA graphs.
set -euo pipefail
cd /workspace
export PYTHONPATH=/workspace/python:/workspace/benchmark/qwen38/alignment
python benchmark/qwen38/patch_reference.py
python benchmark/qwen38/alignment/patch_probe.py
full=shared+gdn+hc+qsa+reduce+index+rope+zero+direct+width
python benchmark/qwen38/alignment/capture.py --backend sglang
for variant in baseline shared gdn hc shared+gdn+hc shared+gdn+hc+qsa "$full" production; do
  python benchmark/qwen38/alignment/capture.py --backend mini --variant "$variant"
  candidate=mini-$variant
  [[ "$variant" != baseline ]] || candidate=mini
  python benchmark/qwen38/alignment/compare.py --candidate "$candidate" --output "/tmp/qwen38-alignment/final-$candidate.json"
done
for backend in sglang mini; do
  for spec in '1 2048 3073' '2 16 128'; do
    read -r -a shape <<< "$spec"
    python benchmark/qwen38/alignment/capture.py --backend "$backend" --variant production --batch "${shape[0]}" --lengths "${shape[@]:1}" --random --directory /tmp/qwen38-alignment-extra
  done
  python benchmark/qwen38/alignment/capture.py --backend "$backend" --variant "$full+router" --stable-topk --directory /tmp/qwen38-alignment-stable
  for spec in '1 2048 3073' '2 16 128'; do
    read -r -a shape <<< "$spec"
    python benchmark/qwen38/alignment/capture.py --backend "$backend" --variant "$full+router" --batch "${shape[0]}" --lengths "${shape[@]:1}" --random --stable-topk --directory /tmp/qwen38-alignment-stable-extra
  done
done
python benchmark/qwen38/alignment/compare.py --directory /tmp/qwen38-alignment-extra --candidate mini-production --output /tmp/qwen38-alignment-extra/final-mini-production.json
for directory in /tmp/qwen38-alignment-stable /tmp/qwen38-alignment-stable-extra; do
  python benchmark/qwen38/alignment/compare.py --directory "$directory" --candidate "mini-$full+router" --output "$directory/comparison-router.json"
done
python benchmark/qwen38/alignment/replay_operators.py
python benchmark/qwen38/alignment/replay_topk.py
python benchmark/qwen38/alignment/replay_scratch.py
