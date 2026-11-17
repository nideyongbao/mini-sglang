import argparse
import json
import os
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backend", choices=["mini", "sglang"], required=True)
    p.add_argument("--directory", default="/tmp/qwen38-alignment")
    p.add_argument("--lengths", nargs="+", type=int, default=[16, 128, 2048])
    p.add_argument("--variant", default="production")
    p.add_argument("--mode", choices=["strict", "fast"], default="strict")
    p.add_argument("--chunk", type=int, default=512)
    p.add_argument("--mode-inputs", action="store_true")
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--random", action="store_true")
    p.add_argument("--stable-topk", action="store_true")
    a = p.parse_args()
    root = Path(a.directory)
    root.mkdir(parents=True, exist_ok=True)
    arm = root / "armed.json"
    arm.unlink(missing_ok=True)
    os.environ["QWEN38_TRACE_DIR"] = a.directory
    if a.stable_topk:
        os.environ["QWEN38_STABLE_TOPK"] = "1"
    os.environ["MINISGL_QWEN38_REFERENCE_MODE"] = a.mode
    if a.backend == "mini":
        from probe import install_mini

        if a.variant != "production":
            from ablations import install

            install(a.variant)
        os.environ["QWEN38_TRACE_LABEL"] = (
            "mini" if a.variant == "baseline" else "mini-" + a.variant
        )
        install_mini()
        from minisgl.core import SamplingParams
        from minisgl.llm import LLM

        engine = LLM(
            "/model",
            qwen38_mode=a.mode,
            max_running_req=a.batch,
            max_seq_len_override=4096,
            num_page_override=32768,
            max_extend_tokens=a.chunk,
            cuda_graph_bs=[],
            attention_backend="fa",
            cache_type="naive",
            use_pynccl=False,
        )

        def generate(ids):
            return engine.generate(ids, SamplingParams(max_tokens=a.steps, ignore_eos=True))

    else:
        import sglang

        engine = sglang.Engine(
            model_path="/model",
            language_only=True,
            tp_size=1,
            fp8_gemm_runner_backend="triton",
            moe_runner_backend="triton",
            context_length=4096,
            max_total_tokens=32768,
            max_running_requests=a.batch,
            max_mamba_cache_size=max(16, a.batch * 2),
            chunked_prefill_size=a.chunk,
            max_prefill_tokens=32768,
            mem_fraction_static=0.9,
            disable_cuda_graph=True,
            disable_radix_cache=True,
            log_level="warning",
        )

        def generate(ids):
            return engine.generate(
                input_ids=ids,
                sampling_params=dict(temperature=0, max_new_tokens=a.steps, ignore_eos=True),
            )

    try:
        for n in a.lengths:
            case = (
                f"length{n}"
                + ("-random" if a.random else "")
                + (f"-batch{a.batch}" if a.batch != 1 else "")
            )
            if a.random:
                import random

                rng = random.Random(20261117 + n)
                known = json.loads(
                    (
                        Path(__file__).parents[1] / "results/2026-11-17/accuracy-long.json"
                    ).read_text()
                )["prompts"]
                prompt = next(
                    (p for p in known if len(p) == n),
                    [rng.randrange(100, 240000) for _ in range(n)],
                )
                ids = [prompt] + [
                    [rng.randrange(100, 240000) for _ in range(n)] for _ in range(a.batch - 1)
                ]
            else:
                ids = [[100 + (i + 37 * b) % 1000 for i in range(n)] for b in range(a.batch)]
            if a.mode_inputs:
                import random

                rng = random.Random(20261117 + n * 31 + a.batch)
                ids = [[rng.randrange(100, 240000) for _ in range(n)] for _ in range(a.batch)]
            arm.write_text(json.dumps(dict(case=case, prompt_length=n)))
            result = generate(ids)
            (root / f"{a.backend}-{a.variant}-{case}-output.json").write_text(
                json.dumps(result, indent=2)
            )
            arm.unlink()
    finally:
        arm.unlink(missing_ok=True)
        engine.shutdown()


if __name__ == "__main__":
    main()
