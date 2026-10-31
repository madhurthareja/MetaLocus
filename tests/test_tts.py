"""Wiring tests only -- SeamlessTTSBackend loads a ~9GB real model on first
use, so actual synthesis is verified by a one-off script, not here (same
split as SAM 3.1: fast unit tests vs. a real-model smoke test)."""

import pytest

from locus.actuation.tts import SeamlessTTSBackend, build_tts_backend


def test_build_tts_backend_returns_none_when_unconfigured():
    assert build_tts_backend({"tts": {"backend": None}}) is None
    assert build_tts_backend({}) is None


def test_build_tts_backend_returns_seamless_backend():
    backend = build_tts_backend({"tts": {"backend": "seamless_m4t_v2"}})
    assert isinstance(backend, SeamlessTTSBackend)


def test_build_tts_backend_rejects_unknown_backend():
    with pytest.raises(ValueError):
        build_tts_backend({"tts": {"backend": "not_a_real_backend"}})


def test_backend_construction_does_not_touch_the_model():
    """Lazy loading: constructing the backend must not import torch/transformers
    or touch the GPU -- only synthesize() should."""
    backend = SeamlessTTSBackend()
    assert backend._model is None
    assert backend._processor is None
