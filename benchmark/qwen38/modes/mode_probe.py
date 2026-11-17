"""Observe and teacher-force sampled logits after graph replay, never during timing."""

import json
import os
from pathlib import Path

import torch


def record(logits, ids, positions, decode, label):
    root = Path(os.environ["QWEN38_MODE_PROBE_DIR"])
    arm = root / "armed.json"
    if not arm.exists():
        return
    spec = json.loads(arm.read_text())
    positions = positions.detach().cpu()
    end = int(positions.max())
    folder = root / spec["case"] / label
    folder.mkdir(parents=True, exist_ok=True)
    assert torch.isfinite(logits).all(), (label, spec["case"], end)
    torch.save(
        dict(
            logits=logits.detach().float().cpu(),
            ids=ids.detach().cpu(),
            positions=positions,
            decode=decode,
        ),
        folder / f'{"decode" if decode else "prefill"}-{end:05d}.pt',
    )
    token = 1000 if spec.get("constant_continuation") else 1000 + max(0, end - (spec["length"] - 1))
    logits.fill_(-10000)
    logits[:, token] = 10000


def install_reference():
    from sglang.srt.model_executor.model_runner import ModelRunner

    original = ModelRunner.sample
    if getattr(original, "_q38_probe", False):
        return

    def sample(self, output, batch):
        decode = batch.forward_mode.is_decode()
        positions = batch.positions if decode else batch.seq_lens - 1
        record(output.next_token_logits, batch.input_ids, positions, decode, "sglang")
        return original(self, output, batch)

    sample._q38_probe = True
    ModelRunner.sample = sample


def install_mini(engine, label):
    original = engine.engine.sampler.sample
    forward = engine.engine.forward_batch
    active = [None]

    def forward_batch(batch, args):
        active[0] = batch
        try:
            return forward(batch, args)
        finally:
            active[0] = None

    engine.engine.forward_batch = forward_batch

    def sample(logits, args):
        batch = active[0]
        record(logits, batch.input_ids, batch.positions, batch.is_decode, label)
        return original(logits, args)

    engine.engine.sampler.sample = sample


def install_fixed_continuation_reference():
    """Constant continuation for timings: one identical device fill per sampled batch."""
    from sglang.srt.layers.sampler import Sampler

    original = Sampler.forward

    def sample(self, *args, **kwargs):
        ids = original(self, *args, **kwargs)
        return ids.fill_(1000)

    Sampler.forward = sample


def install_fixed_continuation_mini(engine):
    original = engine.engine.sampler.sample

    def sample(*args, **kwargs):
        return original(*args, **kwargs).fill_(1000)

    engine.engine.sampler.sample = sample
