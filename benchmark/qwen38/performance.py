"""Like-for-like offline generate latency; execute backends serially on an idle GPU."""

import argparse
import json
import random
import statistics
import time
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backend", choices=["mini", "sglang"], required=True)
    p.add_argument("--model", default="/model")
    p.add_argument("--output", required=True)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--context", type=int, default=4096)
    p.add_argument("--lengths", type=int, nargs="+", default=[128, 512, 1536, 3072])
    p.add_argument("--batches", type=int, nargs="+", default=[1, 4, 8])
    a = p.parse_args()
    if a.backend == "mini":
        from minisgl.core import SamplingParams
        from minisgl.llm import LLM

        engine = LLM(
            a.model,
            max_running_req=8,
            max_seq_len_override=a.context,
            num_page_override=32768,
            max_extend_tokens=512,
            cuda_graph_bs=list(range(1, 9)),
            attention_backend="fa",
            cache_type="naive",
            use_pynccl=False,
        )

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
            chunked_prefill_size=512,
            mem_fraction_static=0.9,
            cuda_graph_bs_decode=list(range(1, 9)),
            disable_cuda_graph_padding=True,
            disable_radix_cache=True,
            log_level="warning",
        )

        def generate(ids, n):
            return engine.generate(
                input_ids=ids,
                sampling_params=dict(temperature=0, max_new_tokens=n, ignore_eos=True),
            )

    import torch

    def timed(ids, n):
        torch.cuda.synchronize()
        start = time.perf_counter()
        out = generate(ids, n)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        key = "token_ids" if a.backend == "mini" else "output_ids"
        assert len(out) == len(ids) and all(len(r[key]) == n for r in out), (n, out)
        return elapsed

    rows = []
    try:
        for length in a.lengths:
            for batch in a.batches:
                rng = random.Random(20261117 + length * 31 + batch)
                ids = [[rng.randrange(100, 240000) for _ in range(length)] for _ in range(batch)]
                for output in [1, 128]:
                    timed(ids, output)
                    timed(ids, output)
                    times = [timed(ids, output) for _ in range(a.repeats)]
                    med = statistics.median(times)
                    row = dict(
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
                                backend=a.backend, context=a.context, repeats=a.repeats, rows=rows
                            ),
                            indent=2,
                        )
                    )
    finally:
        engine.shutdown()


if __name__ == "__main__":
    main()
