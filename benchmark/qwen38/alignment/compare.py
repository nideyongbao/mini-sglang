import argparse
import json
from pathlib import Path

import torch


def metrics(a, b):
    a, b = a.reshape(-1).float(), b.reshape(-1).float()
    assert a.shape == b.shape
    delta = a - b
    return dict(
        equal=bool(torch.equal(a, b)),
        max_abs=delta.abs().max().item(),
        mean_abs=delta.abs().mean().item(),
        rms=delta.square().mean().sqrt().item(),
        unequal=int((a != b).sum()),
        elements=a.numel(),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--directory", default="/tmp/qwen38-alignment")
    p.add_argument("--candidate", default="mini")
    p.add_argument("--output", required=True)
    p.add_argument("--allow-nonfinite", action="store_true")
    a = p.parse_args()
    torch.set_num_threads(1)
    rows = []
    invalid = []
    for case in sorted(Path(a.directory).glob("length*")):
        ref = {f.name: f for f in (case / "sglang").glob("*.pt")}
        cand = {f.name: f for f in (case / a.candidate).glob("*.pt")}
        assert ref and cand, (case, ref, cand)
        for name in sorted(ref.keys() & cand.keys(), key=lambda s: int(s.split("-")[1])):
            x = torch.load(ref[name], weights_only=False, map_location="cpu")
            y = torch.load(cand[name], weights_only=False, map_location="cpu")
            assert torch.equal(x["input_ids"], y["input_ids"]), name
            xp = x["positions"]
            yp = y["positions"]
            if xp.ndim == 2 and yp.ndim == 1:
                assert torch.equal(xp, yp.expand_as(xp))
            else:
                assert torch.equal(xp, yp)
            stages = []
            for key, value in y["tensors"].items():
                if key not in x["tensors"]:
                    continue
                other = x["tensors"][key]
                if value.numel() != other.numel():
                    continue
                stages.append(dict(name=key, **metrics(other, value)))
            xl = x["tensors"]["logits"].float()
            yl = y["tensors"]["logits"].float()
            if not (torch.isfinite(xl).all() and torch.isfinite(yl).all()):
                invalid.append(
                    dict(
                        case=case.name,
                        frame=name,
                        reference_nonfinite=int((~torch.isfinite(xl)).sum()),
                        candidate_nonfinite=int((~torch.isfinite(yl)).sum()),
                    )
                )
                print("INVALID", invalid[-1])
                continue
            lp = xl.log_softmax(-1)
            mp = yl.log_softmax(-1)
            top = lp.topk(20, -1).indices
            row = dict(
                case=case.name,
                frame=name,
                input_shape=list(x["input_ids"].shape),
                logit_rows=xl.shape[0],
                top1_matching_rows=int((xl.argmax(-1) == yl.argmax(-1)).sum()),
                reference_tie_matching_rows=int(
                    (
                        xl.gather(-1, yl.argmax(-1, keepdim=True)).squeeze(-1) == xl.max(-1).values
                    ).sum()
                ),
                reference_margin_to_candidate=(
                    xl.max(-1).values - xl.gather(-1, yl.argmax(-1, keepdim=True)).squeeze(-1)
                ).tolist(),
                first_difference=next((s for s in stages if not s["equal"]), None),
                stages=stages,
                logits=metrics(xl, yl),
                top1_match=bool(torch.equal(xl.argmax(-1), yl.argmax(-1))),
                top20_logprob_max=(lp.gather(-1, top) - mp.gather(-1, top)).abs().max().item(),
                reference_top1=xl.argmax(-1).tolist(),
                candidate_top1=yl.argmax(-1).tolist(),
            )
            rows.append(row)
            print(
                case.name,
                name,
                "first",
                row["first_difference"] and row["first_difference"]["name"],
                "logit max",
                row["logits"]["max_abs"],
                "top1",
                row["top1_match"],
            )
    summary = dict(
        frames=len(rows),
        top1_matches=sum(r["top1_match"] for r in rows),
        exact_logit_frames=sum(r["logits"]["equal"] for r in rows),
        max_logit_abs=max(r["logits"]["max_abs"] for r in rows),
        max_top20_logprob_error=max(r["top20_logprob_max"] for r in rows),
    )
    elements = sum(r["logits"]["elements"] for r in rows)
    summary.update(
        invalid_frames=len(invalid),
        logit_rows=sum(r["logit_rows"] for r in rows),
        top1_matching_rows=sum(r["top1_matching_rows"] for r in rows),
        reference_tie_matching_rows=sum(r["reference_tie_matching_rows"] for r in rows),
        logit_rms=(sum(r["logits"]["rms"] ** 2 * r["logits"]["elements"] for r in rows) / elements)
        ** 0.5,
        logit_mean_abs=sum(r["logits"]["mean_abs"] * r["logits"]["elements"] for r in rows)
        / elements,
    )
    Path(a.output).write_text(
        json.dumps(
            dict(candidate=a.candidate, summary=summary, invalid_frames=invalid, rows=rows),
            indent=2,
        )
    )
    print(summary)
    assert not invalid or a.allow_nonfinite, invalid


if __name__ == "__main__":
    main()
