import pytest
import torch
from minisgl.kernel import indexing, store_cache
from minisgl.kernel.triton.qsa import sparse_attention, sparse_decode
from minisgl.kernel.triton.qwen35_fused import gated_rmsnorm_

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_narrow_embedding_and_kv(dtype):
    weights = torch.randn(97, 32, device="cuda", dtype=dtype)
    indices = torch.tensor([0, 96, 14, 5], device="cuda", dtype=torch.int32)
    assert torch.equal(indexing(weights, indices), weights[indices.long()])
    masked = indexing(weights, indices, vocab_range=(10, 20))
    expected = weights[(indices - 10).clamp(0, 96).long()]
    expected[(indices < 10) | (indices >= 30)] = 0
    assert torch.equal(masked, expected)
    k, v = torch.zeros_like(weights), torch.zeros_like(weights)
    x, y = weights[:4].clone(), weights[4:8].clone()
    store_cache(k, v, indices, x, y)
    assert torch.equal(k[indices.long()], x)
    assert torch.equal(v[indices.long()], y)


@pytest.mark.parametrize("sigmoid", [False, True])
def test_gdn_gate(sigmoid):
    torch.manual_seed(19)
    x = torch.randn(11, 12, 32, device="cuda", dtype=torch.bfloat16)
    z = torch.randn_like(x)
    w = torch.randn(32, device="cuda", dtype=x.dtype)
    ref = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6) * w.float()
    ref *= torch.sigmoid(z.float()) if sigmoid else torch.nn.functional.silu(z.float())
    gated_rmsnorm_(x, z, w, 1e-6, sigmoid)
    torch.testing.assert_close(x, ref.to(x.dtype), atol=0.03125, rtol=0.01)


def test_sparse_attention_mask_and_tail():
    torch.manual_seed(23)
    q = torch.randn(3, 12, 32, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(89, 1, 32, device="cuda", dtype=q.dtype)
    v = torch.randn_like(k)
    ids = torch.full((3, 67), -1, device="cuda", dtype=torch.int32)
    ids[0, :3] = torch.tensor([5, 7, 1], device="cuda")
    ids[1, :] = torch.randperm(89, device="cuda")[:67].int()
    ids[2, 64:] = torch.tensor([88, 3, 8], device="cuda")
    result = sparse_attention(q, k, v, ids)
    refs = []
    for row in range(3):
        selected = ids[row][ids[row] >= 0].long()
        scores = q[row].float() @ k[selected, 0].float().T / (32**0.5)
        refs.append(scores.softmax(-1) @ v[selected, 0].float())
    torch.testing.assert_close(result, torch.stack(refs).to(q.dtype), atol=0.02, rtol=0.03)


def test_sparse_decode_compaction():
    torch.manual_seed(17)
    q = torch.randn(3, 12, 32, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(89, 1, 32, device="cuda", dtype=q.dtype)
    v = torch.randn_like(k)
    ids = torch.full((3, 67), -1, device="cuda", dtype=torch.int32)
    for i, n in enumerate([1, 15, 67]):
        ids[i, :n] = torch.randperm(89, device="cuda")[:n].int()
    out = sparse_decode(q, k, v, ids)
    refs = []
    for row in range(3):
        selected = ids[row][ids[row] >= 0].long()
        scores = q[row].float() @ k[selected, 0].float().T / (32**0.5)
        refs.append(scores.softmax(-1) @ v[selected, 0].float())
    torch.testing.assert_close(out, torch.stack(refs).to(q.dtype), atol=0.01, rtol=0.02)


@pytest.mark.parametrize("tokens", [1, 7, 129])
def test_grouped_gemma_norm(tokens):
    from minisgl.kernel.triton.qwen38 import grouped_norm

    torch.manual_seed(37)
    x = torch.randn(tokens, 128, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(128, device="cuda", dtype=x.dtype)
    grouped = x.float().reshape(tokens, 4, 32)
    ref = (grouped * torch.rsqrt(grouped.square().mean(-1, keepdim=True) + 1e-6)).flatten(1)
    ref = (ref * (w.float() + 1)).to(x.dtype)
    torch.testing.assert_close(grouped_norm(x, w, 32, 1e-6), ref, atol=0.015625, rtol=0.008)


@pytest.mark.parametrize("tokens", [1, 8, 127])
def test_tiny_hyperconnection(tokens):
    from minisgl.kernel.triton.qwen38 import tiny_hc_combine, tiny_hc_mix
    from minisgl.models.qwen3_8 import _combine, _mix

    torch.manual_seed(42)
    x = torch.randn(tokens, 128, device="cuda", dtype=torch.bfloat16)
    norm_w = torch.randn(128, device="cuda", dtype=x.dtype) * 0.02
    down = torch.randn(4, 128, device="cuda", dtype=x.dtype) * 0.02
    up = torch.randn(128, 4, device="cuda", dtype=x.dtype) * 0.02
    inject = torch.randn(4, 128, device="cuda", dtype=x.dtype) * 0.02
    grouped = x.float().reshape(tokens, 4, 32)
    norm = (grouped * torch.rsqrt(grouped.square().mean(-1, keepdim=True) + 1e-6)).flatten(1)
    norm = (norm * (1 + norm_w.float())).to(x.dtype)
    expected = _mix(norm, down, up, 4, 32)
    actual, actual_norm = tiny_hc_mix(x, norm_w, down, up, 4, 32, 1e-6)
    torch.testing.assert_close(actual_norm, norm, atol=0.008, rtol=0.008)
    torch.testing.assert_close(actual, expected, atol=0.004, rtol=0.01)
    y = torch.randn_like(actual)
    torch.testing.assert_close(
        tiny_hc_combine(y, x, actual_norm, inject, 4, 32),
        _combine(y, x, norm, inject, 4, 32),
        atol=0.016,
        rtol=0.008,
    )
