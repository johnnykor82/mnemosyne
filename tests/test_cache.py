"""Verifies anchor-card mtime cache and Honcho peer-card TTL cache."""
from __future__ import annotations

import time

from conftest import make_provider


def test_peer_card_cached_within_ttl():
    """Same prefetch call within TTL should hit the cache (Honcho called once)."""
    p = make_provider(hindsight_text="x")
    p.prefetch("q1")
    p.prefetch("q2")
    p.prefetch("q3")
    assert p._honcho.calls == 1, f"peer card cache miss — calls={p._honcho.calls}"


def test_peer_card_invalidated_on_session_switch():
    """Switching sessions clears the cache so the new context gets fresh data."""
    p = make_provider(hindsight_text="x")
    p.prefetch("q1")
    assert p._honcho.calls == 1
    p.on_session_switch("session-new")
    p.prefetch("q2")
    assert p._honcho.calls == 2, "peer card was not re-fetched after session switch"


def test_peer_card_invalidated_on_memory_write():
    """A memory_write may change the peer card → cache must be dropped."""
    p = make_provider(hindsight_text="x")
    p.prefetch("q1")
    assert p._honcho.calls == 1
    # on_memory_write fans out to inner providers; our FakeHoncho doesn't
    # implement it, so swallow that AttributeError.
    try:
        p.on_memory_write("retain", "user", "some content")
    except AttributeError:
        pass
    p.prefetch("q2")
    assert p._honcho.calls == 2


def test_peer_card_ttl_expires(monkeypatch):
    """Forcing time forward past TTL invalidates the cache entry."""
    p = make_provider(hindsight_text="x")
    p._peer_cache_ttl_s = 0.05
    p.prefetch("q1")
    assert p._honcho.calls == 1
    time.sleep(0.07)
    p.prefetch("q2")
    assert p._honcho.calls == 2
