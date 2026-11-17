import json
from pathlib import Path

import torch
from sgl_kernel.flash_attn import flash_attn_varlen_func

torch.manual_seed(83)
q = torch.randn(2, 12, 32, device="cuda", dtype=torch.bfloat16)
k = torch.randn(34, 1, 32, device="cuda", dtype=q.dtype)
v = torch.randn_like(k)
cu = torch.tensor([0, 17, 34], device="cuda", dtype=torch.int32)
cuq = torch.arange(3, device="cuda", dtype=torch.int32)
result = {}
for fill in [0.0, float("nan")]:
    pk = torch.full((4102, 1, 32), fill, device="cuda", dtype=q.dtype)
    pv = torch.full_like(pk, fill)
    pk[:34] = k
    pv[:34] = v
    out = flash_attn_varlen_func(
        q=q,
        k=pk,
        v=pv,
        cu_seqlens_q=cuq,
        cu_seqlens_k=cu,
        max_seqlen_q=1,
        max_seqlen_k=2051,
        softmax_scale=32**-0.5,
        causal=True,
        ver=3,
    )
    result[str(fill)] = dict(nonfinite=int((~torch.isfinite(out)).sum()), elements=out.numel())
Path("/tmp/qwen38-alignment/scratch-replay.json").write_text(json.dumps(result, indent=2))
print(result)
