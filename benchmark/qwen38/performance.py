"""Like-for-like offline generate latency; execute backends serially on an idle GPU."""

import argparse
import hashlib
import json
import os
import random
import statistics
import time
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backend", choices=["mini", "sglang"], required=True)
    p.add_argument("--ablation", default="production")
    p.add_argument("--mode", choices=["strict", "fast"], default="strict")
    p.add_argument("--reference-policy", choices=["strict", "original"], default="strict")
    p.add_argument("--fixed-continuation", action="store_true")
    p.add_argument("--retain-local-gdn", action="store_true")
    p.add_argument("--eager", action="store_true")
    p.add_argument("--prompts-json", help="Prepared prompt cases; requires --batches 1")
    p.add_argument("--warmups", type=int, default=2)
    p.add_argument("--model", default="/model")
    p.add_argument("--output", required=True)
    p.add_argument("--chunk", type=int, default=32768)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--context", type=int, default=4096)
    p.add_argument("--lengths", type=int, nargs="+", default=[128, 512, 1536, 3072])
    p.add_argument("--batches", type=int, nargs="+", default=[1, 4, 8])
    a = p.parse_args()
    prepared = None
    if a.prompts_json:
        assert a.batches == [1], "Prepared prompts require single-request execution"
        prepared = {
            len(c["input_ids"]): c["input_ids"]
            for c in json.loads(Path(a.prompts_json).read_text())
        }
        a.lengths = list(prepared)
    assert a.warmups >= 1
    os.environ["MINISGL_QWEN38_REFERENCE_MODE"] = a.reference_policy
    if a.fixed_continuation:
        os.environ["QWEN38_FIXED_CONTINUATION"] = "1"
    if a.retain_local_gdn:
        from minisgl.kernel.triton.gdn_decode import packed_decode
        from minisgl.models import qwen3_8_reference

        qwen3_8_reference.reference_decode = packed_decode
    if a.backend == "mini":
        if a.ablation != "production":
            from alignment.ablations import install

            install(a.ablation)
        from minisgl.core import SamplingParams
        from minisgl.llm import LLM

        engine = LLM(
            a.model,
            qwen38_mode=a.mode,
            max_running_req=8,
            max_seq_len_override=a.context,
            num_page_override=32768,
            max_extend_tokens=a.chunk,
            cuda_graph_bs=[] if a.eager else list(range(1, 9)),
            attention_backend="fa",
            cache_type="naive",
            use_pynccl=False,
        )

        if a.fixed_continuation:
            from modes.mode_probe import install_fixed_continuation_mini

            install_fixed_continuation_mini(engine)

        def generate(ids, n):
            return engine.generate(ids, SamplingParams(max_tokens=n, ignore_eos=True))

    else:
        import sglang

        engine = sglang.Engine(
            model_path=a.model,
            language_only=True,
            tp_size=1,
            fp8_gemm_runner_backend="triton",
            moe_runner_backend="triton",
            context_length=a.context,
            max_total_tokens=32768,
            max_running_requests=8,
            max_mamba_cache_size=16,
            chunked_prefill_size=a.chunk,
            max_prefill_tokens=32768,
            mem_fraction_static=0.9,
            cuda_graph_bs_decode=list(range(1, 9)),
            disable_cuda_graph_padding=True,
            disable_cuda_graph=a.eager,
            disable_radix_cache=True,
            log_level="warning",
        )

        def generate(ids, n):
            return engine.generate(
                input_ids=ids,
                sampling_params=dict(temperature=0, max_new_tokens=n, ignore_eos=True),
            )

    import torch

    def validate(ids, n):
        # Outside timed regions, check the actual sampled distributions.
        if a.backend == "mini":
            original_sample = engine.engine.sampler.sample

            def checked_sample(logits, args):
                assert torch.isfinite(logits).all(), "Nonfinite mini logits"
                return original_sample(logits, args)

            engine.engine.sampler.sample = checked_sample
            try:
                generate(ids, n)
            finally:
                engine.engine.sampler.sample = original_sample
        else:
            import math

            out = engine.generate(
                input_ids=ids,
                sampling_params=dict(temperature=0, max_new_tokens=n, ignore_eos=True),
                return_logprob=True,
                top_logprobs_num=1,
            )
            assert all(
                v is not None and math.isfinite(v)
                for r in out
                for top in r["meta_info"]["output_top_logprobs"]
                for v, *_ in top
            ), "Nonfinite reference logprobs"

    def timed(ids, n):
        torch.cuda.synchronize()
        start = time.perf_counter()
        out = generate(ids, n)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        key = "token_ids" if a.backend == "mini" else "output_ids"
        assert len(out) == len(ids) and all(len(r[key]) == n for r in out), (n, out)
        if a.fixed_continuation:
            assert all(set(r[key]) == {1000} for r in out), "Continuation was not forced"
        return elapsed

    rows = []
    try:
        for length in a.lengths:
            for batch in a.batches:
                rng = random.Random(20261117 + length * 31 + batch)
                ids = [[rng.randrange(100, 240000) for _ in range(length)] for _ in range(batch)]
                if prepared is not None:
                    ids = [prepared[length]]
                for output in [1, 128]:
                    validate(ids, output)
                    for _ in range(a.warmups - 1):
                        timed(ids, output)
                    times = [timed(ids, output) for _ in range(a.repeats)]
                    med = statistics.median(times)
                    row = dict(
                        input_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
                        batch=batch,
                        input_tokens=length,
                        output_tokens=output,
                        median_ms=med * 1000,
                        min_ms=min(times) * 1000,
                        max_ms=max(times) * 1000,
                        output_tokens_per_s=batch * output / med,
                        seconds=times,
                    )
                    rows.append(row)
                    print(json.dumps(row), flush=True)
                    Path(a.output).write_text(
                        json.dumps(
                            dict(
                                backend=a.backend,
                                ablation=a.ablation,
                                mode=a.mode,
                                fixed_continuation=a.fixed_continuation,
                                reference_policy=a.reference_policy,
                                graphs=not a.eager,
                                retain_local_gdn=a.retain_local_gdn,
                                finite_preflight=True,
                                context=a.context,
                                chunk=a.chunk,
                                repeats=a.repeats,
                                warmups=a.warmups,
                                rows=rows,
                            ),
                            indent=2,
                        )
                    )
    finally:
        engine.shutdown()


if __name__ == "__main__":
    main()
