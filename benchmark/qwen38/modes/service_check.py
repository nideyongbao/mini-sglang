"""HTTP availability, streaming equivalence and slot reuse; outside timed runs."""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    base = dict(
        model="/model",
        messages=[dict(role="user", content="hello " * 128)],
        max_tokens=16,
        temperature=0,
        ignore_eos=True,
        chat_template_kwargs=dict(enable_thinking=False),
    )

    def request(payload):
        r = requests.post(a.url + "/v1/chat/completions", json=payload, timeout=120)
        r.raise_for_status()
        result = r.json()
        assert result["usage"]["completion_tokens"] == payload["max_tokens"], result
        assert result["choices"][0]["finish_reason"] == "length", result
        return result

    initial = request(base)
    chunks = []
    with requests.post(
        a.url + "/v1/chat/completions", json=dict(base, stream=True), stream=True, timeout=120
    ) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if line.startswith(b"data: "):
                value = line[6:]
                if value == b"[DONE]":
                    break
                chunks.append(json.loads(value))
    streamed = "".join(
        c.get("choices", [{}])[0].get("delta", {}).get("content", "") or ""
        for c in chunks
        if c.get("choices")
    )
    assert streamed == (initial["choices"][0]["message"].get("content") or "")
    assert any(c["choices"][0].get("finish_reason") == "length" for c in chunks if c.get("choices"))
    # Disconnect a long stream, then exercise more requests than available slots.
    with requests.post(
        a.url + "/v1/chat/completions",
        json=dict(base, stream=True, max_tokens=512),
        stream=True,
        timeout=120,
    ) as r:
        r.raise_for_status()
        next(r.iter_lines())
    checks = []
    for _ in range(2):
        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(request, [dict(base, max_tokens=8 + i) for i in range(8)]))
        checks.extend(r["usage"] for r in results)
    repeated = request(base)
    assert repeated["choices"][0]["message"] == initial["choices"][0]["message"]
    Path(a.output).write_text(
        json.dumps(
            dict(
                initial=initial,
                stream_matches=True,
                slot_reuse_requests=checks,
                after_disconnect=repeated,
            ),
            indent=2,
        )
    )
    print("PASS streaming, cancellation recovery, mixed lengths, slot reuse, repeatability")


if __name__ == "__main__":
    main()
