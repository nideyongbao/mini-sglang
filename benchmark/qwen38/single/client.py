"""Sequential single-request HTTP measurements with matched model work."""

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

import requests
from transformers import AutoTokenizer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prepare", action="store_true")
    p.add_argument("--cases", required=True)
    p.add_argument("--url")
    p.add_argument("--backend")
    p.add_argument("--audit")
    p.add_argument("--output")
    p.add_argument("--repeats", type=int, default=15)
    p.add_argument("--warmups", type=int, default=3)
    a = p.parse_args()
    tokenizer = AutoTokenizer.from_pretrained("/model")
    if a.prepare:
        cases = []
        for words in [128, 2048, 3072]:
            messages = [dict(role="user", content=" ".join(["hello"] * words))]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
            ids = tokenizer.encode(text)
            cases.append(
                dict(
                    words=words,
                    messages=messages,
                    input_ids=ids,
                    input_sha256=hashlib.sha256(json.dumps([ids]).encode()).hexdigest(),
                )
            )
        Path(a.cases).write_text(json.dumps(cases, indent=2))
        print([(c["words"], len(c["input_ids"])) for c in cases])
        return
    cases = json.loads(Path(a.cases).read_text())
    client = requests.Session()
    client.trust_env = False
    rows = []
    for case in cases:
        for n in [1, 128]:
            payload = dict(
                model="/model",
                messages=case["messages"],
                temperature=0,
                max_tokens=n,
                ignore_eos=True,
                stream=False,
                chat_template_kwargs=dict(enable_thinking=False),
            )
            expected = tokenizer.decode([1000] * n)

            def call():
                start = time.perf_counter()
                response = client.post(a.url + "/v1/chat/completions", json=payload, timeout=180)
                response.raise_for_status()
                result = response.json()
                elapsed = time.perf_counter() - start
                assert result["usage"]["completion_tokens"] == n, result
                assert result["usage"]["prompt_tokens"] == len(case["input_ids"]), result
                assert result["choices"][0]["message"]["content"] == expected, result
                return elapsed

            for _ in range(a.warmups):
                call()
            audit = json.loads((Path(a.audit) / f"{len(case['input_ids'])}.json").read_text())
            assert audit["input_ids"] == case["input_ids"], "HTTP token IDs differ from offline"
            times = [call() for _ in range(a.repeats)]
            med = statistics.median(times)
            row = dict(
                input_tokens=len(case["input_ids"]),
                batch=1,
                output_tokens=n,
                input_sha256=case["input_sha256"],
                median_ms=med * 1000,
                output_tokens_per_s=n / med,
                seconds=times,
                min_ms=min(times) * 1000,
                max_ms=max(times) * 1000,
                input_ids_verified=True,
                fixed_output_verified=True,
            )
            rows.append(row)
            Path(a.output).write_text(
                json.dumps(
                    dict(
                        backend=a.backend,
                        interface="HTTP",
                        concurrency=1,
                        chunk=512,
                        graphs=True,
                        reference_policy="strict",
                        fixed_continuation=True,
                        warmups=a.warmups,
                        repeats=a.repeats,
                        keep_alive=True,
                        rows=rows,
                    ),
                    indent=2,
                )
            )
            print(json.dumps({k: v for k, v in row.items() if k != "seconds"}), flush=True)
    client.close()


if __name__ == "__main__":
    main()
