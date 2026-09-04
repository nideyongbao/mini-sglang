from types import SimpleNamespace

from minisgl.attention import (
    SUPPORTED_ATTENTION_BACKENDS,
    HybridBackend,
    create_attention_backend,
)
from minisgl.attention.gdn import HybridLinearBackend


class _Backend:
    pass


def test_stage_hybrid_is_wrapped_by_one_linear_backend(monkeypatch):
    monkeypatch.setitem(
        SUPPORTED_ATTENTION_BACKENDS._registry,
        "test_prefill",
        lambda config: _Backend(),
    )
    monkeypatch.setitem(
        SUPPORTED_ATTENTION_BACKENDS._registry,
        "test_decode",
        lambda config: _Backend(),
    )
    config = SimpleNamespace(has_linear_layers=True)

    backend = create_attention_backend("test_prefill,test_decode", config)

    assert isinstance(backend, HybridLinearBackend)
    assert isinstance(backend.full_backend, HybridBackend)
    assert isinstance(backend.full_backend.prefill_backend, _Backend)
    assert isinstance(backend.full_backend.decode_backend, _Backend)


def test_non_linear_model_keeps_stage_hybrid_outermost(monkeypatch):
    monkeypatch.setitem(
        SUPPORTED_ATTENTION_BACKENDS._registry,
        "test_prefill",
        lambda config: _Backend(),
    )
    monkeypatch.setitem(
        SUPPORTED_ATTENTION_BACKENDS._registry,
        "test_decode",
        lambda config: _Backend(),
    )
    config = SimpleNamespace(has_linear_layers=False)

    backend = create_attention_backend("test_prefill,test_decode", config)

    assert isinstance(backend, HybridBackend)
    assert isinstance(backend.prefill_backend, _Backend)
    assert isinstance(backend.decode_backend, _Backend)
