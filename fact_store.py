"""Local SQLite store for fact metadata — repetition count, dates, sources.

Drives plan items 1 (date tags), 2 (mention counter) and 10 (forced strong
signal for user-explicit facts). Item 3 (pre-write dedup against semantic
neighbours) uses Hindsight `recall` separately — fact_store only handles
**exact-key** duplicates, not fuzzy semantics.

The store is a small companion to Hindsight, not a replacement: tags here
mirror what we tell Hindsight to attach to each retained memory, so we can
filter/rerank by them at recall time.
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
import threading
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import config

logger = logging.getLogger(__name__)


_PUNCT_RE = re.compile(r"[^\w\s]+", flags=re.UNICODE)
_WS_RE = re.compile(r"\s+", flags=re.UNICODE)


def _canonical_key(text: str) -> str:
    """Normalize text for exact-key matching: lowercase, strip punctuation,
    collapse whitespace. Word order is preserved (semantically meaningful).
    """
    if not text:
        return ""
    s = text.strip().lower()
    s = _PUNCT_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip()
    return s


def today_iso() -> str:
    return date.today().isoformat()


def date_tag(d: Optional[date] = None) -> str:
    return f"ts:{(d or date.today()).isoformat()}"


class FactStore:
    """SQLite-backed counter/date store. Thread-safe via a single lock."""

    SCHEMA = """
        CREATE TABLE IF NOT EXISTS facts (
            canonical_key TEXT PRIMARY KEY,
            mention_count INTEGER NOT NULL DEFAULT 1,
            first_seen TEXT NOT NULL,
            last_seen TEXT NOT NULL,
            sources TEXT NOT NULL DEFAULT '[]',
            forgotten_at TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_last_seen ON facts(last_seen);
        CREATE INDEX IF NOT EXISTS idx_mention_count ON facts(mention_count);

        -- Semantic forget signatures.
        --
        -- One row per logical forget operation, NOT one per matched
        -- candidate: storing 30 near-identical sigs for one operation
        -- would balloon the table and add nothing. Instead we keep the
        -- union of token sets covered by that operation, so a single
        -- match on any future paraphrase fires the filter.
        --
        -- `tokens`     — JSON array of lowercase content tokens (set semantics)
        -- `examples`   — JSON array of representative sample texts (debug only)
        -- `query`      — the original user-facing query that triggered the op
        -- `last_match_ts` — bumped on every successful read-side match, used
        --                   for vacuuming sigs that no longer match anything.
        CREATE TABLE IF NOT EXISTS forgotten_signatures (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tokens TEXT NOT NULL,
            examples TEXT NOT NULL DEFAULT '[]',
            query TEXT,
            created_at TEXT NOT NULL,
            last_match_ts TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_sig_last_match ON forgotten_signatures(last_match_ts);
    """

    def __init__(self, db_path: Optional[Path] = None) -> None:
        self._db_path = db_path or (config.plugin_dir() / "fact_store.db")
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path), timeout=30, isolation_level=None)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _init_schema(self) -> None:
        with self._lock, self._connect() as conn:
            conn.executescript(self.SCHEMA)

    # ------------------------------------------------------------------
    # Write path
    # ------------------------------------------------------------------

    def bump(self, text: str, *, source: Optional[str] = None) -> int:
        """Record a new mention of `text`. Returns the new mention_count."""
        key = _canonical_key(text)
        if not key:
            return 0
        now = today_iso()
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT mention_count, sources FROM facts WHERE canonical_key = ?",
                (key,),
            ).fetchone()
            if row is None:
                sources = [source] if source else []
                conn.execute(
                    "INSERT INTO facts(canonical_key, mention_count, first_seen, last_seen, sources) "
                    "VALUES(?, 1, ?, ?, ?)",
                    (key, now, now, json.dumps(sources, ensure_ascii=False)),
                )
                return 1
            count, sources_json = row[0], row[1]
            try:
                sources = json.loads(sources_json or "[]")
            except Exception:
                sources = []
            if source and source not in sources:
                sources.append(source)
            conn.execute(
                "UPDATE facts SET mention_count = mention_count + 1, "
                "last_seen = ?, sources = ?, forgotten_at = NULL "
                "WHERE canonical_key = ?",
                (now, json.dumps(sources, ensure_ascii=False), key),
            )
            return int(count) + 1

    def force_strong(self, text: str, *, source: str, level: Optional[int] = None) -> int:
        """Plan item 10: user-explicit facts get max mention_count immediately."""
        key = _canonical_key(text)
        if not key:
            return 0
        target = level if level is not None else config.get(
            "fact_store", "user_explicit_mention_count", default=10
        )
        now = today_iso()
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT mention_count, sources FROM facts WHERE canonical_key = ?",
                (key,),
            ).fetchone()
            if row is None:
                sources = [source] if source else []
                conn.execute(
                    "INSERT INTO facts(canonical_key, mention_count, first_seen, last_seen, sources) "
                    "VALUES(?, ?, ?, ?, ?)",
                    (key, int(target), now, now, json.dumps(sources, ensure_ascii=False)),
                )
                return int(target)
            current = int(row[0])
            try:
                sources = json.loads(row[1] or "[]")
            except Exception:
                sources = []
            if source and source not in sources:
                sources.append(source)
            new_count = max(current, int(target))
            conn.execute(
                "UPDATE facts SET mention_count = ?, last_seen = ?, sources = ?, "
                "forgotten_at = NULL WHERE canonical_key = ?",
                (new_count, now, json.dumps(sources, ensure_ascii=False), key),
            )
            return new_count

    # ------------------------------------------------------------------
    # Read path
    # ------------------------------------------------------------------

    def get(self, text: str) -> Optional[Dict[str, Any]]:
        """Look up by canonical key. Returns row or None."""
        key = _canonical_key(text)
        if not key:
            return None
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT canonical_key, mention_count, first_seen, last_seen, sources, forgotten_at "
                "FROM facts WHERE canonical_key = ?",
                (key,),
            ).fetchone()
        if not row:
            return None
        try:
            sources = json.loads(row[4] or "[]")
        except Exception:
            sources = []
        return {
            "canonical_key": row[0],
            "mention_count": int(row[1]),
            "first_seen": row[2],
            "last_seen": row[3],
            "sources": sources,
            "forgotten_at": row[5],
        }

    def get_strength(self, text: str) -> int:
        row = self.get(text)
        if not row or row.get("forgotten_at"):
            return 0
        return int(row["mention_count"])

    def is_strong(self, text: str) -> bool:
        threshold = int(config.get("fact_store", "strong_signal_threshold", default=3))
        return self.get_strength(text) >= threshold

    def is_known(self, text: str) -> bool:
        """True if the canonical key already exists (regardless of strength)."""
        return self.get(text) is not None

    # ------------------------------------------------------------------
    # Forget
    # ------------------------------------------------------------------

    def mark_forgotten(self, text: str) -> bool:
        """Mark a fact as forgotten. Inserts a tombstone row if no existing
        row matches the canonical key — necessary because recall surfaces
        LLM-extracted text whose canonical key won't match the raw user-turn
        text we keep in the bump() path."""
        key = _canonical_key(text)
        if not key:
            return False
        now = today_iso()
        with self._lock, self._connect() as conn:
            cur = conn.execute(
                "UPDATE facts SET forgotten_at = ? WHERE canonical_key = ?",
                (now, key),
            )
            if cur.rowcount > 0:
                return True
            conn.execute(
                "INSERT INTO facts(canonical_key, mention_count, first_seen, "
                "last_seen, sources, forgotten_at) VALUES(?, 0, ?, ?, '[]', ?)",
                (key, now, now, now),
            )
            return True

    def is_forgotten(self, text: str) -> bool:
        row = self.get(text)
        return bool(row and row.get("forgotten_at"))

    # ------------------------------------------------------------------
    # Semantic forget signatures
    #
    # Why a separate mechanism: ``mark_forgotten`` is exact-key only.
    # Hindsight stores dozens of paraphrases of the same idea, each with
    # a different canonical_key, so per-key marks don't catch the family.
    # Signatures store the *set of content tokens* covered by a forget
    # op; the read-side filter computes Jaccard against every signature
    # and drops candidates that match. One signature can mask hundreds
    # of paraphrases.
    # ------------------------------------------------------------------

    def add_signature(
        self,
        tokens: List[str],
        *,
        examples: Optional[List[str]] = None,
        query: Optional[str] = None,
        merge_jaccard: float = 0.7,
    ) -> int:
        """Insert a forget signature, or merge into a near-identical one.

        Returns the row id (existing or new). De-dup by Jaccard against
        prior sigs so repeat ``forget("Barsik")`` calls don't pile up.
        """
        token_set = {t for t in tokens if t}
        if not token_set:
            return 0
        examples = examples or []
        now = today_iso()
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT id, tokens, examples FROM forgotten_signatures"
            ).fetchall()
            for rid, tjson, ejson in rows:
                try:
                    existing = set(json.loads(tjson or "[]"))
                except Exception:
                    existing = set()
                inter = token_set & existing
                union = token_set | existing
                jac = len(inter) / len(union) if union else 0.0
                if jac >= merge_jaccard:
                    new_tokens = sorted(union)
                    try:
                        old_examples = list(json.loads(ejson or "[]"))
                    except Exception:
                        old_examples = []
                    new_examples = old_examples + [
                        e for e in examples if e and e not in old_examples
                    ]
                    new_examples = new_examples[:10]  # cap debug list
                    conn.execute(
                        "UPDATE forgotten_signatures SET tokens=?, examples=?, "
                        "last_match_ts=? WHERE id=?",
                        (json.dumps(new_tokens, ensure_ascii=False),
                         json.dumps(new_examples, ensure_ascii=False),
                         now, rid),
                    )
                    return int(rid)

            cur = conn.execute(
                "INSERT INTO forgotten_signatures(tokens, examples, query, created_at) "
                "VALUES(?, ?, ?, ?)",
                (
                    json.dumps(sorted(token_set), ensure_ascii=False),
                    json.dumps(examples[:10], ensure_ascii=False),
                    query,
                    now,
                ),
            )
            return int(cur.lastrowid or 0)

    def list_signatures(self) -> List[Dict[str, Any]]:
        """Read all signatures. Cheap — table is tiny by design."""
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT id, tokens, examples, query, created_at, last_match_ts "
                "FROM forgotten_signatures"
            ).fetchall()
        out: List[Dict[str, Any]] = []
        for rid, tjson, ejson, q, cts, lts in rows:
            try:
                tokens = json.loads(tjson or "[]")
            except Exception:
                tokens = []
            try:
                examples = json.loads(ejson or "[]")
            except Exception:
                examples = []
            out.append({
                "id": int(rid),
                "tokens": tokens,
                "examples": examples,
                "query": q,
                "created_at": cts,
                "last_match_ts": lts,
            })
        return out

    def touch_signature(self, sig_id: int) -> None:
        if not sig_id:
            return
        now = today_iso()
        with self._lock, self._connect() as conn:
            conn.execute(
                "UPDATE forgotten_signatures SET last_match_ts=? WHERE id=?",
                (now, sig_id),
            )

    def vacuum_signatures(self, *, max_count: int = 1000,
                          stale_days: Optional[int] = None) -> int:
        """Drop oldest sigs over ``max_count``, plus any whose
        last_match_ts (or created_at if never matched) is older than
        ``stale_days``. Returns rows removed."""
        from datetime import datetime, timedelta
        removed = 0
        with self._lock, self._connect() as conn:
            if stale_days is not None and stale_days > 0:
                cutoff = (datetime.utcnow() - timedelta(days=stale_days)).date().isoformat()
                cur = conn.execute(
                    "DELETE FROM forgotten_signatures "
                    "WHERE COALESCE(last_match_ts, created_at) < ?",
                    (cutoff,),
                )
                removed += cur.rowcount or 0
            count = conn.execute(
                "SELECT COUNT(*) FROM forgotten_signatures"
            ).fetchone()[0]
            if count > max_count:
                cur = conn.execute(
                    "DELETE FROM forgotten_signatures WHERE id IN ("
                    "SELECT id FROM forgotten_signatures "
                    "ORDER BY COALESCE(last_match_ts, created_at) ASC LIMIT ?"
                    ")",
                    (count - max_count,),
                )
                removed += cur.rowcount or 0
        return removed

    # ------------------------------------------------------------------
    # Tagging helpers — what we attach to Hindsight on retain.
    # ------------------------------------------------------------------

    def tags_for_retain(self, *, source: Optional[str] = None,
                        when: Optional[date] = None) -> List[str]:
        tags = [date_tag(when)]
        if source:
            tags.append(f"source:{source}")
        return tags
