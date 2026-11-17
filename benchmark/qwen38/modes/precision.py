import argparse
import json
import os
import random
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backend", choices=["mini", "sglang"], required=True)
    p.add_argument("--mode", choices=["strict", "fast"], default="strict")
    p.add_argument("--graphs", action="store_true")
    p.add_argument("--constant-continuation", action="store_true")
    p.add_argument("--retain-local-gdn", action="store_true")
    p.add_argument("--lengths", type=int, nargs="+", default=[16, 128, 2048, 3073])
    p.add_argument("--batches", type=int, nargs="+", default=[1, 2, 4, 8])
    p.add_argument("--chunk", type=int, default=32768)
    p.add_argument("--steps", type=int, default=16)
    p.add_argument("--directory", default="/tmp/q38-mode-precision")
    a = p.parse_args()
    root = Path(a.directory)
    root.mkdir(exist_ok=True, parents=True)
    arm = root / "armed.json"
    arm.unlink(missing_ok=True)
    os.environ["MINISGL_QWEN38_REFERENCE_MODE"] = "strict"
    os.environ["QWEN38_MODE_PROBE_DIR"] = str(root)
    if a.backend == "mini":
        from minisgl.core import SamplingParams
        from minisgl.llm import LLM
        from mode_probe import install_mini

        if a.retain_local_gdn:
            from minisgl.kernel.triton.gdn_decode import packed_decode
            from minisgl.models import qwen3_8_reference

            qwen3_8_reference.reference_decode = packed_decode
        engine = LLM(
            "/model",
            qwen38_mode=a.mode,
            max_running_req=8,
            max_seq_len_override=4096,
            num_page_override=32768,
            max_extend_tokens=a.chunk,
            cuda_graph_bs=list(range(1, 9)) if a.graphs else [],
            attention_backend="fa",
            cache_type="naive",
            use_pynccl=False,
        )
        install_mini(engine, a.mode + ("-local-gdn" if a.retain_local_gdn else ""))

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
            max_running_requests=8,
            max_mamba_cache_size=16,
            chunked_prefill_size=a.chunk,
            max_prefill_tokens=32768,
            mem_fraction_static=0.9,
            cuda_graph_bs_decode=list(range(1, 9)),
            disable_cuda_graph_padding=True,
            disable_cuda_graph=not a.graphs,
            disable_radix_cache=True,
            log_level="warning",
        )

        def generate(ids):
            return engine.generate(
                input_ids=ids,
                sampling_params=dict(temperature=0, max_new_tokens=a.steps, ignore_eos=True),
            )

    try:
        for length in a.lengths:
            for batch in a.batches:
                rng = random.Random(20261117 + length * 31 + batch)
                ids = [[rng.randrange(100, 240000) for _ in range(length)] for _ in range(batch)]
                case = f"n{length}-b{batch}"
                arm.write_text(
                    json.dumps(
                        dict(
                            case=case, length=length, constant_continuation=a.constant_continuation
                        )
                    )
                )
                result = generate(ids)
                arm.unlink()
                (root / f"{a.backend}-{a.mode}-{case}.json").write_text(
                    json.dumps(dict(inputs=ids, output=result))
                )
    finally:
        arm.unlink(missing_ok=True)
        engine.shutdown()


if __name__ == "__main__":
    main()
