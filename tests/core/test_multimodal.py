from types import SimpleNamespace, MethodType

import pytest
import torch

from minisgl.message.backend import BaseBackendMsg, UserMsg
from minisgl.core import Batch, SamplingParams
from minisgl.multimodal import ImageInputs, image_positions
from minisgl.scheduler.scheduler import _make_rope_positions


def test_image_coordinates_match_hf_and_continue_during_decode():
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Model

    image_id = 248056
    # Two non-square images with text before, between and after them.
    ids = torch.tensor([1, 2] + [image_id] * 6 + [3, 4, 5] + [image_id] * 2 + [6])
    grid = torch.tensor([[1, 4, 6], [1, 2, 4]])
    positions, delta = image_positions(ids, grid, image_id, 2)
    fake = SimpleNamespace(config=Qwen3_5Config())
    fake.get_vision_position_ids = MethodType(Qwen3_5Model.get_vision_position_ids, fake)
    reference, reference_delta = Qwen3_5Model.get_rope_index(
        fake, ids[None], (ids == image_id).int()[None], image_grid_thw=grid
    )
    torch.testing.assert_close(positions, reference[:, 0])
    assert delta == reference_delta.item()
    image_req = SimpleNamespace(cached_len=len(ids), image_inputs=SimpleNamespace(rope_delta=delta))
    text_req = SimpleNamespace(cached_len=17, image_inputs=None)
    batch = Batch([image_req, text_req], "decode")
    batch.padded_reqs = batch.reqs
    actual = _make_rope_positions(batch, torch.device("cpu"))
    assert actual.tolist() == [positions.max().item() + 1, 17]


def test_image_payload_survives_message_boundary():
    mm = ImageInputs(
        torch.randn(4, 12),
        torch.tensor([[1, 2, 2]]),
        torch.arange(6).expand(3, -1),
        torch.tensor([2]),
        -3,
    )
    original = UserMsg(7, torch.arange(6, dtype=torch.int32), SamplingParams(), mm)
    decoded = BaseBackendMsg.decoder(original.encoder())
    torch.testing.assert_close(decoded.image_inputs.pixel_values, mm.pixel_values)
    torch.testing.assert_close(decoded.image_inputs.positions, mm.positions)
    assert decoded.image_inputs.rope_delta == -3


@pytest.mark.parametrize(
    "ids,grid",
    [([1, 9, 2], [[1, 4, 4]]), ([1, 2], [[1, 2, 2]]), ([9, 9], [[1, 2, 2]]), ([9], [[2, 2, 2]])],
)
def test_malformed_placeholder_grids_are_rejected(ids, grid):
    with pytest.raises(ValueError):
        image_positions(torch.tensor(ids), torch.tensor(grid), 9, 2)


def test_overlap_finishes_only_after_last_host_token_is_collected():
    from contextlib import nullcontext
    from minisgl.core import Req
    from minisgl.scheduler import Scheduler

    req = Req(
        torch.tensor([1, 2, 3]), 0, 0, 2, 7, SamplingParams(max_tokens=2, ignore_eos=True), None
    )
    # Two device steps are submitted, but neither result has reached the host yet.
    req.complete_one()
    req.complete_one()
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.cache_manager = SimpleNamespace(lazy_free_region=nullcontext)
    scheduler.decode_manager = SimpleNamespace(remove_req=lambda _: None)
    scheduler.finished_reqs = set()
    freed = []
    scheduler._free_req_resources = lambda r: freed.append(r)
    replies = []
    scheduler.send_result = replies.extend
    batch = Batch([req], "decode")
    data = (
        SimpleNamespace(batch=batch),
        (None, torch.tensor([42]), SimpleNamespace(synchronize=lambda: None)),
    )
    scheduler._process_last_data(data)
    scheduler._process_last_data(data)
    assert [r.finished for r in replies] == [False, True]
    assert [r.completion_tokens for r in replies] == [1, 2]
    assert replies[-1].finish_reason == "length"
    assert len(freed) == 1
