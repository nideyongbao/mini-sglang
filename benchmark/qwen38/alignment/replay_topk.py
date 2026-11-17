import json
from pathlib import Path

import torch
from sglang.kernels.ops.elementwise.fast_topk import fast_topk

p = Path("/tmp/qwen38-alignment/length2048")
a = torch.load(p / "sglang/decode-2051-2051.pt", weights_only=False)
b = torch.load(
    p / "mini-shared+gdn+hc+qsa+reduce+index+rope+zero+direct+width/decode-2051-2051.pt",
    weights_only=False,
)
x = a["tensors"]["layer.3.qsa.index.scores"].cuda()
y = b["tensors"]["layer.3.qsa.index.scores"].cuda()
print(
    "widths",
    a.get("index_score_width"),
    b.get("index_score_width"),
    "scoreequal",
    torch.equal(x, y),
)
ends = torch.full((1,), x.shape[1], device="cuda", dtype=torch.int32)
out = {
    "valid_scores_equal": torch.equal(x, y),
    "reference_width": a.get("index_score_width"),
    "candidate_width": b.get("index_score_width"),
    "min_score": x.min().item(),
    "min_ties": int((x == x.min()).sum()),
    "runs": [],
}
for width in [528, 1024]:
    z = torch.full((1, width), -float("inf"), device="cuda")
    z[:, : x.shape[1]] = x
    for i in range(8):
        ids = fast_topk(z, ends, topk=512, row_starts=None).cpu().flatten().tolist()
        out["runs"].append(
            dict(width=width, run=i, omitted=sorted(set(range(x.shape[1])) - set(ids)))
        )
Path("/tmp/qwen38-alignment/topk-replay.json").write_text(json.dumps(out, indent=2))
print(out)

from compare import metrics

# Routed MoE: inspect the remaining first divergent expert module.
from minisgl.moe.fused import fused_topk as mini_topk
from sgl_kernel import topk_softmax
from sglang.srt.layers.moe.topk import fused_topk as reference_topk

frame = torch.load(
    "/tmp/qwen38-alignment-stable-extra/length128-random-batch2/sglang/prefill-0000-0127.pt",
    weights_only=False,
)["tensors"]
hidden = frame["layer.2.mlp_hc.mix"].cuda()
logits = frame["layer.2.router"].cuda()
rw, ri = reference_topk(hidden, logits, 10, True)
mw, mi = mini_topk(hidden, logits, 10, True)
raw_w = torch.empty_like(mw)
raw_i = torch.empty_like(mi)
topk_softmax(raw_w, raw_i, logits.float(), True)
result = dict(
    ids_equal=torch.equal(ri, mi),
    weights=metrics(rw, mw),
    single_normalization_ids_equal=torch.equal(ri, raw_i),
    single_normalization_weights=metrics(rw, raw_w),
)
Path("/tmp/qwen38-alignment/moe-topk-replay.json").write_text(json.dumps(result, indent=2))
print(result)
