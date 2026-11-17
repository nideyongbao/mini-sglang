import pytest
import torch
from minisgl.models.qwen3_8_reference import stable_topk


def test_stable_topk_ties_and_partial_rows():
    scores = torch.tensor([[4.0, 4.0, 4.0, 4.0, 8.0], [7.0, 6.0, 5.0, 4.0, 3.0]])
    actual = stable_topk(scores, torch.tensor([4, 2]), 3)
    assert actual.tolist() == [[0, 1, 2], [0, 1, -1]]


def test_stable_topk_relative_offsets_and_all_selected_order():
    scores = torch.tensor([[99.0, 4.0, 8.0, 9.0, 5.0]])
    assert stable_topk(scores, torch.tensor([3]), 2, torch.tensor([1])).tolist() == [[2, 1]]
    assert stable_topk(scores, torch.tensor([3]), 5, torch.tensor([1])).tolist() == [
        [0, 1, 2, -1, -1]
    ]


def test_strict_default_and_invalid_mode():
    from minisgl.distributed import DistributedInfo
    from minisgl.engine.config import EngineConfig

    kwargs = dict(model_path="/unused", tp_info=DistributedInfo(0, 1), dtype=torch.bfloat16)
    assert EngineConfig(**kwargs).qwen38_mode == "strict"
    assert EngineConfig(**kwargs, qwen38_mode="fast").qwen38_mode == "fast"
    with pytest.raises(ValueError, match="qwen38_mode"):
        EngineConfig(**kwargs, qwen38_mode="typo")


def test_strict_graph_requires_exact_batch():
    from types import SimpleNamespace

    from minisgl.engine.graph import GraphRunner

    runner = GraphRunner.__new__(GraphRunner)
    runner.max_graph_bs = 8
    runner.graph_map = {1: None, 2: None, 4: None, 8: None}
    runner._exact_batches = True
    assert runner.can_use_cuda_graph(SimpleNamespace(is_decode=True, size=4))
    assert not runner.can_use_cuda_graph(SimpleNamespace(is_decode=True, size=3))
    assert not runner.can_use_cuda_graph(SimpleNamespace(is_decode=False, size=4))
    runner._exact_batches = False
    assert runner.can_use_cuda_graph(SimpleNamespace(is_decode=True, size=3))
