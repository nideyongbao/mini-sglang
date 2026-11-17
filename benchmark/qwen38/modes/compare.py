import argparse
import json
from pathlib import Path

import torch

p = argparse.ArgumentParser()
p.add_argument("--directory", required=True)
p.add_argument("--mode", default="strict")
p.add_argument("--output", required=True)
p.add_argument("--steps", type=int, default=16)
a = p.parse_args()
torch.set_num_threads(1)
root = Path(a.directory)
rows = []
for case in sorted(root.glob("n*-b*")):
    reference = {f.name: f for f in (case / "sglang").glob("*.pt")}
    candidate = {f.name: f for f in (case / a.mode).glob("*.pt")}
    length = int(case.name.split("-b")[0][1:])
    expected = {f"prefill-{length - 1:05d}.pt"} | {
        f"decode-{length + step:05d}.pt" for step in range(a.steps - 1)
    }
    # Compare all requested output steps; omit chunk-internal sampling and the
    # reference overlap scheduler's discarded decode after request completion.
    reference = {k: v for k, v in reference.items() if k in expected}
    candidate = {k: v for k, v in candidate.items() if k in expected}
    assert reference.keys() == expected, (case, "incomplete reference")
    assert candidate.keys() == expected, (case, "incomplete candidate")
    assert reference and candidate, case
    assert reference.keys() == candidate.keys(), (case, "missing frames")
    for name in sorted(reference):
        x = torch.load(reference[name], weights_only=True)
        y = torch.load(candidate[name], weights_only=True)
        assert torch.equal(x["ids"], y["ids"]), (case, name, "inputs")
        if x["decode"]:
            assert torch.equal(x["positions"], y["positions"]), (case, name, "positions")
        xx, yy = x["logits"], y["logits"]
        assert xx.shape == yy.shape
        delta = xx - yy
        assert torch.isfinite(delta).all()
        rows.append(
            dict(
                case=case.name,
                frame=name,
                equal=torch.equal(xx.view(torch.int32), yy.view(torch.int32)),
                bitwise_rows=int((xx.view(torch.int32) == yy.view(torch.int32)).all(-1).sum()),
                rows=xx.shape[0],
                top1=int((xx.argmax(-1) == yy.argmax(-1)).sum()),
                reference_tie_hits=int(
                    (
                        xx.max(-1).values == xx.gather(-1, yy.argmax(-1, keepdim=True)).squeeze(-1)
                    ).sum()
                ),
                top1_margin_max=(
                    xx.max(-1).values - xx.gather(-1, yy.argmax(-1, keepdim=True)).squeeze(-1)
                )
                .max()
                .item(),
                max_abs=delta.abs().max().item(),
                rms=delta.square().mean().sqrt().item(),
            )
        )
s = dict(
    frames=len(rows),
    exact=sum(r["equal"] for r in rows),
    logit_rows=sum(r["rows"] for r in rows),
    top1=sum(r["top1"] for r in rows),
    bitwise_rows=sum(r["bitwise_rows"] for r in rows),
    reference_tie_hits=sum(r["reference_tie_hits"] for r in rows),
    top1_margin_max=max(r["top1_margin_max"] for r in rows),
    max_abs=max(r["max_abs"] for r in rows),
    rms=(sum(r["rms"] ** 2 * r["rows"] for r in rows) / sum(r["rows"] for r in rows)) ** 0.5,
)
Path(a.output).write_text(json.dumps(dict(mode=a.mode, summary=s, rows=rows), indent=2))
print(s)
print("differences", [(r["case"], r["frame"], r["max_abs"]) for r in rows if not r["equal"]][:20])
