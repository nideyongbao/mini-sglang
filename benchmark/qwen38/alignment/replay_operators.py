"""Isolate operators using captured identical inputs; no serving timing here."""

import json
from pathlib import Path

import torch
import triton
import triton.language as tl


@triton.jit
def hc_linear(N, W, O):
    row = tl.program_id(0)
    c = tl.arange(0, 4)
    k = tl.arange(0, 128)
    n = tl.load(N + row * 128 + k).to(tl.float32)
    w = tl.load(W + c[:, None] * 128 + k[None, :]).to(tl.float32)
    tl.store(O + row * 4 + c, tl.sum(w * n[None, :], 1))


from compare import metrics
from minisgl.kernel.triton.gdn_decode import packed_decode
from minisgl.kernel.triton.qwen38 import _hc_combine
from safetensors import safe_open
from sglang.kernels.ops.attention.fla.fused_recurrent import (
    fused_recurrent_gated_delta_rule_packed_decode,
)
from sglang.kernels.ops.elementwise.elementwise import fused_gate_sigmoid_mul_add

root = Path("/tmp/qwen38-alignment")
weights = {}
with safe_open(next(Path("/model").glob("*.safetensors")), framework="pt", device="cuda") as f:
    for suffix in [
        "layers.0.attn_hyper_connection.block_inject_weight.weight",
        "layers.0.mlp.shared_expert_gate.weight",
        "layers.0.linear_attn.A_log",
        "layers.0.linear_attn.dt_bias",
    ]:
        key = next(k for k in f.keys() if k.endswith(suffix))
        weights[suffix] = f.get_tensor(key)
results = {}
for length in [16, 128, 2048]:
    n = min(length, 512)
    name = f"prefill-0000-{n-1:04d}.pt"
    r = torch.load(root / f"length{length}" / "sglang" / name, weights_only=False)["tensors"]
    r = {k: v.cuda() for k, v in r.items()}
    x = r["layer.0.attn_hc.mix.input"]
    y = r["layer.0.attention"]
    norm = r["layer.0.attn_hc.mix.norm"]
    w = weights["layers.0.attn_hyper_connection.block_inject_weight.weight"]
    gate_ref = torch.nn.functional.linear(norm, w)
    gate_mini = torch.empty_like(gate_ref)
    hc_linear[(n,)](norm, w, gate_mini, num_warps=4, enable_fp_fusion=False)
    results[f"hc_{length}_gate_projection"] = metrics(gate_ref, gate_mini)
    for fma in [False, True]:
        out = torch.empty_like(x)
        _hc_combine[(n,)](y, x, norm, w, out, 4, 32, num_warps=4, enable_fp_fusion=fma)
        results[f"hc_{length}_fma{fma}"] = metrics(r["layer.0.attn_hc.combine"], out)
    inp = r["layer.0.mlp_hc.mix"]
    shared = r["layer.0.shared"]
    routed = r["layer.0.experts"]
    gate = weights["layers.0.mlp.shared_expert_gate.weight"]
    separated = routed + shared * torch.sigmoid(torch.nn.functional.linear(inp, gate))
    fused = routed.clone()
    fused_gate_sigmoid_mul_add(inp, gate.squeeze(), shared, fused)
    results[f"shared_{length}_separate"] = metrics(r["layer.0.mlp"], separated)
    results[f"shared_{length}_reference"] = metrics(r["layer.0.mlp"], fused)
# GDN recurrence: identical supplied input/state, independent of earlier layers.
torch.manual_seed(73)
mixed = torch.randn(1, 640, device="cuda", dtype=torch.bfloat16) * 0.02
a = torch.randn(1, 12, device="cuda", dtype=torch.bfloat16) * 0.02
b = torch.randn_like(a) * 0.02
state = torch.randn(1, 12, 32, 32, device="cuda", dtype=torch.float32) * 0.001
indices = torch.zeros(1, device="cuda", dtype=torch.int32)
A = weights["layers.0.linear_attn.A_log"]
dt = weights["layers.0.linear_attn.dt_bias"]
refstate = state.clone()
refout = mixed.new_empty(1, 1, 12, 32)
fused_recurrent_gated_delta_rule_packed_decode(
    mixed, a, b, A, dt, 32**-0.5, refstate, refout, indices, True
)
for rounded in [False, True]:
    s = state.clone()
    out = packed_decode(mixed, a, b, A, dt, s, indices, 4, 12, 32, 32, 32**-0.5, round_beta=rounded)
    results[f"gdn_round_beta{rounded}"] = dict(
        output=metrics(refout, out), state=metrics(refstate, s)
    )
(root / "isolated-operators.json").write_text(json.dumps(results, indent=2))
print(json.dumps(results, indent=2))
