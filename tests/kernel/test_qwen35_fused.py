import torch

from minisgl.kernel.triton.qwen35_fused import gated_rmsnorm_


def test_gated_rmsnorm_supports_strided_gate() -> None:
    torch.manual_seed(0)
    tokens, heads, head_dim = 17, 8, 128
    core = torch.randn(tokens, heads, head_dim, device="cuda", dtype=torch.bfloat16)
    packed_gate = torch.randn(
        tokens,
        heads * head_dim + 64,
        device="cuda",
        dtype=torch.bfloat16,
    )
    gate = packed_gate[:, : heads * head_dim].view(tokens, heads, head_dim)
    weight = torch.randn(head_dim, device="cuda", dtype=torch.bfloat16)
    eps = 1e-6

    expected = core.float()
    expected *= torch.rsqrt(expected.square().mean(dim=-1, keepdim=True) + eps)
    expected *= weight.float()
    expected *= torch.nn.functional.silu(gate.float())

    actual = gated_rmsnorm_(core.clone(), gate, weight, eps)

    torch.testing.assert_close(actual.float(), expected, atol=3e-2, rtol=3e-2)
