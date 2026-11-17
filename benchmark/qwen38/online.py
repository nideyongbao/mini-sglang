"""Native OpenAI chat HTTP E2E latency; loopback client, fixed text and concurrency."""

import argparse
import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import requests


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", required=True)
    p.add_argument("--backend", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--repeats", type=int, default=5)
    a = p.parse_args()
    rows = []
    for words in (128, 2048):
        prompt = " ".join(["hello"] * words)
        for batch in (1, 4, 8):
            payload = dict(
                model="/model",
                messages=[dict(role="user", content=prompt)],
                temperature=0,
                max_tokens=128,
                ignore_eos=True,
                chat_template_kwargs=dict(enable_thinking=False),
            )

            def request(_):
                start = time.perf_counter()
                r = requests.post(a.url + "/v1/chat/completions", json=payload, timeout=300)
                r.raise_for_status()
                result = r.json()
                elapsed = time.perf_counter() - start
                assert result["usage"]["completion_tokens"] == 128, result
                return dict(seconds=elapsed, usage=result["usage"])

            samples = []
            with ThreadPoolExecutor(max_workers=batch) as pool:
                for repeat in range(2 + a.repeats):
                    start = time.perf_counter()
                    results = list(pool.map(request, range(batch)))
                    elapsed = time.perf_counter() - start
                    if repeat >= 2:
                        samples.append(dict(seconds=elapsed, requests=results))
            prompt_tokens = {r["usage"]["prompt_tokens"] for s in samples for r in s["requests"]}
            assert len(prompt_tokens) == 1, prompt_tokens
            median = statistics.median(s["seconds"] for s in samples)
            row = dict(
                words=words,
                batch=batch,
                input_tokens=prompt_tokens.pop(),
                output_tokens=128,
                median_batch_ms=median * 1000,
                output_tokens_per_s=batch * 128 / median,
                median_request_ms=statistics.median(
                    r["seconds"] for s in samples for r in s["requests"]
                )
                * 1000,
                samples=samples,
            )
            rows.append(row)
            Path(a.output).write_text(
                json.dumps(
                    dict(
                        backend=a.backend,
                        repeats=a.repeats,
                        warmups=2,
                        updated_utc=datetime.now(timezone.utc).isoformat(),
                        rows=rows,
                    ),
                    indent=2,
                )
            )
            print(json.dumps({k: v for k, v in row.items() if k != "samples"}), flush=True)


if __name__ == "__main__":
    main()
