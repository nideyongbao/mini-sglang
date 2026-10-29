import torch

from minisgl.kernel.triton.qwen35_fused import build_mrope_cache


def test_fused_mrope_cache_selects_hf_interleaved_axes():
    torch.manual_seed(0)
    cache = torch.randn(128, 64, device="cuda")
    positions = torch.randint(0, 128, (3, 73), device="cuda")
    reference = cache[positions[0]].clone()
    for axis, section in [(1, 11), (2, 10)]:
        for offset in [0, 32]:
            cols = torch.arange(axis, section * 3, 3, device="cuda") + offset
            reference[:, cols] = cache[positions[axis, :, None], cols]
    torch.testing.assert_close(build_mrope_cache(cache, positions, [11, 11, 10]), reference)
