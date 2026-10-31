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
