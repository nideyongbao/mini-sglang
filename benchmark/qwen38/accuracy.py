"""Fixed-token, teacher-forced next-token/logprob and free-running comparison."""

import argparse
import json
import math
import random
from pathlib import Path

import requests
import torch
from minisgl.core import SamplingParams
from minisgl.llm import LLM

p = argparse.ArgumentParser()
p.add_argument("--model", default="/model")
p.add_argument("--reference", default="http://127.0.0.1:31038")
p.add_argument("--output", required=True)
p.add_argument("--graphs", action="store_true")
p.add_argument("--padded-graphs", action="store_true", help="Test native CLI graph tiers 1,2,4,8")
p.add_argument("--chunk", type=int, default=512)
p.add_argument("--long", action="store_true")
p.add_argument("--max-logprob-error", type=float, default=0.005)
a = p.parse_args()
rng = random.Random(20261117)
prompts = [[100, 200, 300, 400], [1, 2, 3, 4, 5, 6]]
prompts += [
    [rng.randrange(100, 240000) for _ in range(n)]
    for n in [1, 7, 16, 63, 128, 511, 512, 513, 1024, 2000]
]
prompts += [[248044, 17, 28, 248044, 39, 50, 61], [100] * 65]
if a.long:
    prompts = [
        [rng.randrange(100, 240000) for _ in range(n)] for n in [15, 2047, 2048, 2049, 3073, 4000]
    ]
params = {"temperature": 0, "max_new_tokens": 16, "ignore_eos": True}
ref = requests.post(
    a.reference + "/generate",
    json={
        "input_ids": prompts,
        "sampling_params": params,
        "return_logprob": True,
        "top_logprobs_num": 20,
    },
    timeout=300,
)
ref.raise_for_status()
ref = ref.json()
Path(a.output + ".reference.json").write_text(json.dumps(ref, indent=2))
for i, r in enumerate(ref):
    for step, top in enumerate(r["meta_info"]["output_top_logprobs"]):
        if any(v is None or not math.isfinite(v) for v, _, _ in top):
            raise RuntimeError(
                f"Reference returned nonfinite logprobs: prompt {i}, step {step}, length {len(prompts[i])}"
            )
llm = LLM(
    a.model,
    max_running_req=8,
    max_seq_len_override=4096 if a.long else 2048,
    num_page_override=32768,
    max_extend_tokens=a.chunk,
    cuda_graph_bs=([1, 2, 4, 8] if a.padded_graphs else list(range(1, 9))) if a.graphs else [],
    attention_backend="fa",
    cache_type="naive",
    use_pynccl=False,
)
try:
    free = llm.generate(prompts, SamplingParams(max_tokens=16, ignore_eos=True))
    # Teacher forcing: identical histories even when greedy trajectories differ.
    forced, specs = [], []
    for i, (prompt, out) in enumerate(zip(prompts, ref)):
        for step in (0, 1, 3, 7, 15):
            forced.append(prompt + out["output_ids"][:step])
            specs.append((i, step))
    collected = []
    sample = llm.engine.sampler.sample

    def capture(logits, args):
        # Chunked-prefill intermediates are not sampled into returned requests.
        lp = logits.float().log_softmax(-1)
        collected.append(lp.detach().cpu())
        return sample(logits, args)

    # Sequential calls identify the final prefill unambiguously even with chunks.
    llm.engine.sampler.sample = capture
    rows = []
    for prompt, (i, step) in zip(forced, specs):
        collected.clear()
        llm.generate([prompt], SamplingParams(max_tokens=1, ignore_eos=True))
        lp = collected[-1][0]
        reference_top = ref[i]["meta_info"]["output_top_logprobs"][step]
        errors = [abs(lp[token].item() - logprob) for logprob, token, _ in reference_top]
        top_id = int(lp.argmax())
        ref_id = ref[i]["output_ids"][step]
        rows.append(
            dict(
                prompt=i,
                step=step,
                length=len(prompt),
                mini=top_id,
                reference=ref_id,
                top1_match=top_id == ref_id,
                reference_top1_tie_match=any(
                    token == top_id and abs(logprob - reference_top[0][0]) <= 1e-6
                    for logprob, token, _ in reference_top
                ),
                top20_max_logprob_error=max(errors),
                top20_mean_logprob_error=sum(errors) / len(errors),
                reference_token_logprob=lp[ref_id].item(),
                mini_top1_logprob=lp[top_id].item(),
            )
        )
    result = dict(
        graphs=a.graphs,
        padded_graphs=a.padded_graphs,
        chunk=a.chunk,
        prompts=prompts,
        reference=ref,
        mini=free,
        teacher_forced=rows,
        summary=dict(
            free_token_matches=sum(
                sum(x == y for x, y in zip(m["token_ids"], r["output_ids"]))
                for m, r in zip(free, ref)
            ),
            free_token_total=sum(len(r["output_ids"]) for r in ref),
            sequence_matches=sum(m["token_ids"] == r["output_ids"] for m, r in zip(free, ref)),
            sequences=len(ref),
            teacher_top1_matches=sum(r["top1_match"] for r in rows),
            teacher_total=len(rows),
            teacher_reference_tie_matches=sum(r["reference_top1_tie_match"] for r in rows),
            max_top20_logprob_error=max(r["top20_max_logprob_error"] for r in rows),
            mean_top20_logprob_error=sum(r["top20_mean_logprob_error"] for r in rows) / len(rows),
        ),
    )
    Path(a.output).write_text(json.dumps(result, indent=2))
    print(json.dumps(result["summary"], indent=2))
    assert result["summary"]["max_top20_logprob_error"] <= a.max_logprob_error, (
        "Numerical tolerance exceeded",
        result["summary"],
    )
finally:
    llm.shutdown()
