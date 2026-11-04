"""Wiring tests only -- a real propose() call costs real Meta API tokens,
so it's verified by a one-off script/manual check, not run on every
`pytest tests/` (same split as SAM 3.1 and TTS: fast unit tests vs. a real
external call)."""

import pytest

from locus.perception.spark_proposer import SparkProposer


def test_client_not_created_until_first_propose_call():
    """Lazy loading: constructing the proposer must not touch the network
    or require a key -- only propose() should."""
    proposer = SparkProposer()
    assert proposer._client is None


def test_propose_raises_clearly_when_key_missing(monkeypatch):
    monkeypatch.delenv("META_KEY", raising=False)
    # load_dotenv() would otherwise re-populate META_KEY from this machine's
    # real .env, defeating the point of this test.
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: None)
    proposer = SparkProposer(api_key=None)
    with pytest.raises(RuntimeError, match="META_KEY"):
        proposer._client_lazy()


def test_classify_motion_raises_clearly_when_key_missing(monkeypatch):
    """Same lazy-client contract as propose() -- classify_motion() is the
    other real Spark call this module makes."""
    monkeypatch.delenv("META_KEY", raising=False)
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: None)
    proposer = SparkProposer(api_key=None)
    with pytest.raises(RuntimeError, match="META_KEY"):
        proposer.classify_motion(["suitcase"])


def test_classify_motion_parses_a_mixed_static_and_movable_response(monkeypatch):
    """Wiring only, no network: confirms classify_motion() calls through
    with the right response_format and reshapes the parsed result into
    the {concept: (kind, speed)} shape motion_classes.py expects --
    static drops its speed field regardless of what the model returned."""
    from locus.perception.spark_proposer import _MotionClassification, _MotionClassifications

    parsed = _MotionClassifications(
        classifications=[
            _MotionClassification(concept="suitcase", motion_class="movable", max_speed_mps=2.0),
            _MotionClassification(concept="fire escape", motion_class="static", max_speed_mps=0.0),
        ]
    )

    class _FakeMessage:
        def __init__(self, parsed):
            self.parsed = parsed

    class _FakeCompletions:
        def parse(self, **kwargs):
            assert kwargs["response_format"] is _MotionClassifications
            assert "suitcase" in kwargs["messages"][-1]["content"]
            return type("R", (), {"choices": [type("C", (), {"message": _FakeMessage(parsed)})]})()

    proposer = SparkProposer(api_key="x")
    proposer._client = type("Client", (), {"beta": type("Beta", (), {"chat": type("Chat", (), {"completions": _FakeCompletions()})})})()

    result = proposer.classify_motion(["suitcase", "fire escape"])
    assert result == {"suitcase": ("movable", 2.0), "fire escape": ("static", None)}
