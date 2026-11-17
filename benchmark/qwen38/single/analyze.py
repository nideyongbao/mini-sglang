"""Compare HTTP to pooled bracketing offline samples; keep timing drift visible."""

import argparse
import json
import statistics
from pathlib import Path


def indexed(path):
    data = json.loads(path.read_text())
    assert data["graphs"] and data["chunk"] == 512 and data["fixed_continuation"]
    assert all(row["batch"] == 1 for row in data["rows"])
    return {(row["input_tokens"], row["output_tokens"]): row for row in data["rows"]}


def describe(seconds):
    quantiles = statistics.quantiles(seconds, n=10)
    return dict(
        median_ms=1000 * statistics.median(seconds),
        p10_ms=1000 * quantiles[0],
        p90_ms=1000 * quantiles[-1],
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--directory", required=True)
    a = p.parse_args()
    root = Path(a.directory)
    rows = []
    for mode in ["sglang", "strict", "fast"]:
        pre = indexed(root / f"offline-{mode}-before.json")
        post = indexed(root / f"offline-{mode}-after.json")
        http = indexed(root / f"http-{mode}.json")
        assert pre.keys() == post.keys() == http.keys()
        for key in sorted(pre):
            x, y, z = pre[key], post[key], http[key]
            assert x["input_sha256"] == y["input_sha256"] == z["input_sha256"]
            off = describe(x["seconds"] + y["seconds"])
            wire = describe(z["seconds"])
            row = dict(
                mode=mode,
                input_tokens=key[0],
                output_tokens=key[1],
                input_sha256=x["input_sha256"],
                offline=off,
                http=wire,
                offline_tokens_per_s=key[1] * 1000 / off["median_ms"],
                http_tokens_per_s=key[1] * 1000 / wire["median_ms"],
                added_ms=wire["median_ms"] - off["median_ms"],
                added_latency_percent=100 * (wire["median_ms"] / off["median_ms"] - 1),
                offline_before_ms=x["median_ms"],
                offline_after_ms=y["median_ms"],
                offline_drift_percent=100 * (y["median_ms"] / x["median_ms"] - 1),
            )
            rows.append(row)
            if key[1] == 128:
                print(
                    mode,
                    key[0],
                    f"{off['median_ms']:.2f} -> {wire['median_ms']:.2f} ms",
                    f"{row['offline_tokens_per_s']:.1f} -> {row['http_tokens_per_s']:.1f} tok/s",
                    f"delta {row['added_latency_percent']:.2f}%",
                )
    (root / "summary.json").write_text(json.dumps(dict(rows=rows), indent=2))


if __name__ == "__main__":
    main()
