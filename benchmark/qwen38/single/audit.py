"""First-request tokenization audit, outside timed repetitions."""

import hashlib
import json
import os
from pathlib import Path

SEEN = set()


def record(ids):
    n = len(ids)
    if n in SEEN:
        return
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    root = Path(os.environ["Q38_SINGLE_AUDIT"])
    root.mkdir(parents=True, exist_ok=True)
    (root / f"{n}.json").write_text(
        json.dumps(
            dict(input_ids=ids, input_sha256=hashlib.sha256(json.dumps([ids]).encode()).hexdigest())
        )
    )
    SEEN.add(n)


def install_mini():
    from minisgl.tokenizer.tokenize import TokenizeManager

    original = TokenizeManager.prepare

    def prepare(self, msg):
        result = original(self, msg)
        record(result[0])
        return result

    TokenizeManager.prepare = prepare
    from minisgl.engine.sample import Sampler

    original_sample = Sampler.sample

    def sample(self, *args, **kwargs):
        return original_sample(self, *args, **kwargs).fill_(1000)

    Sampler.sample = sample


def install_sglang():
    from sglang.srt.managers.tokenizer_manager import TokenizerManager

    original = TokenizerManager._tokenize_one_request

    async def tokenize(self, obj):
        result = await original(self, obj)
        record(result.input_ids)
        return result

    TokenizerManager._tokenize_one_request = tokenize
