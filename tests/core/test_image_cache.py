from types import SimpleNamespace

import pytest
import torch

from minisgl.image_cache import ImageEmbeddingCache
from minisgl.multimodal import image_hashes
from minisgl.models.vision import Qwen35Vision


def test_lru_bytes_eviction_and_storage_ownership():
    cache = ImageEmbeddingCache(16)
    source = torch.arange(100, dtype=torch.float32)
    cache.put("a", source[:2])
    source[:2] = -1
    assert cache.get("a").tolist() == [0, 1]
    assert cache.get("a").untyped_storage().nbytes() == 8
    cache.put("b", torch.ones(2))
    retained = cache.get("a")  # a is most recently used
    cache.put("c", torch.ones(2))
    assert cache.get("b") is None
    assert cache.stats()["size_bytes"] == 16
    assert cache.stats()["evictions"] == 1
    cache.put("oversized", torch.zeros(10))
    assert cache.get("a") is not None and cache.get("c") is not None
    cache.clear()
    assert retained.tolist() == [0, 1]  # in-flight request references survive eviction
    assert cache.stats()["size_bytes"] == 0 and cache.stats()["entries"] == 0


def test_hashes_cover_pixels_grid_dtype_and_image_order():
    pixels = torch.arange(16, dtype=torch.float32).reshape(8, 2)
    grid = torch.tensor([[1, 2, 2], [1, 2, 2]])
    a, b = image_hashes(pixels, grid)
    assert a != b
    assert image_hashes(torch.cat([pixels[4:], pixels[:4]]), grid) == [b, a]
    assert image_hashes(pixels.clone(), grid) == [a, b]
    changed = pixels.clone()
    changed[0, 0] += 1
    assert image_hashes(changed, grid)[0] != a
    assert image_hashes(pixels, torch.tensor([[1, 1, 4], [1, 2, 2]]))[0] != a
    assert image_hashes(pixels.to(torch.bfloat16), grid)[0] != a
    with pytest.raises(ValueError):
        image_hashes(pixels[:-1], grid)


def test_per_image_reuse_order_isolation_clear_and_disable():
    visual = Qwen35Vision.__new__(Qwen35Vision)
    visual.configure_cache(1024)
    calls = []

    def forward(pixels, grid):
        calls.append(grid.tolist())
        return pixels.reshape(len(grid), 4, 2).sum(1)

    visual.forward = forward
    grid = torch.tensor([[1, 2, 2], [1, 2, 2]])
    pixels = torch.arange(16, dtype=torch.float32).reshape(8, 2)
    inputs = SimpleNamespace(pixel_values=pixels, grid_thw=grid, image_hashes=None)
    cold = visual.encode_images(inputs)
    torch.testing.assert_close(visual.encode_images(inputs), cold)
    assert len(calls) == 2
    reverse = SimpleNamespace(
        pixel_values=torch.cat([pixels[4:], pixels[:4]]), grid_thw=grid, image_hashes=None
    )
    torch.testing.assert_close(visual.encode_images(reverse), cold.flip(0))
    assert len(calls) == 2
    other_model = Qwen35Vision.__new__(Qwen35Vision)
    other_model.configure_cache(1024)
    other_model.forward = forward
    other_model.encode_images(inputs)
    assert len(calls) == 4
    visual._cache.clear()
    visual.encode_images(inputs)
    assert len(calls) == 6
    visual.configure_cache(0)
    visual.encode_images(inputs)
    visual.encode_images(inputs)
    assert len(calls) == 8  # disabled uses one batched vision call per request
    assert visual._cache.stats()["entries"] == 0


def test_negative_capacity_rejected():
    with pytest.raises(ValueError):
        ImageEmbeddingCache(-1)
