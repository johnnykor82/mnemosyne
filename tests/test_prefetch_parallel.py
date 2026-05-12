"""Validates that prefetch() now fans out anchor/peer/hindsight in parallel
instead of running them serially."""
from __future__ import annotations

import time
import pytest

from conftest import make_provider


def test_prefetch_parallel_faster_than_sum_of_branches():
    """Branches take 250ms each → serial would be ~750ms, parallel must be
    closer to one branch (~300ms). Asserting <500ms gives generous headroom."""
    p = make_provider(honcho_sleep=0.25, hindsight_sleep=0.25)
    t0 = time.perf_counter()
    out = p.prefetch("any query")
    elapsed = time.perf_counter() - t0
    assert elapsed < 0.5, f"prefetch too slow ({elapsed:.3f}s) — branches not parallel"
    assert "User profile" in out
    assert "Facts (relevant)" in out


def test_prefetch_sections_correct_order():
    """Parallel execution must not break the rendered section order:
    anchor → peer card → facts."""
    p = make_provider()
    out = p.prefetch("hello")
    # Anchor card may be absent (no anchor_card.md). The presence/absence
    # is fine, but if both peer and facts are present, peer comes first.
    peer_idx = out.find("# User profile")
    facts_idx = out.find("# Facts (relevant)")
    assert peer_idx >= 0 and facts_idx >= 0
    assert peer_idx < facts_idx, "peer card must precede facts"


def test_prefetch_branch_failure_does_not_break_others():
    """A failing branch (raising/empty) leaves the other sections intact."""
    p = make_provider(honcho_card=())  # empty card → peer section omitted
    out = p.prefetch("query")
    # facts section still present
    assert "Facts (relevant)" in out
