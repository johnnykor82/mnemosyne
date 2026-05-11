"""Plan item 11 — explicit forgetting.

Two paths:

1. Point delete via built-in memory: when the agent calls
   `memory(action='remove', target=user, old_text=...)`, the
   `on_memory_write(action='remove')` hook on MnemosyneMemoryProvider
   forwards to `forget_text(...)` here, which marks the canonical key
   in fact_store and writes a tombstone to Hindsight.

2. Fuzzy forget via tool/CLI: `memory_forget(query)` does a recall,
   runs the candidates through `forget_text(...)` for each match.

We can't physically delete from Hindsight (no public API). What we do:
- Mark the canonical key in fact_store as forgotten — at recall time,
  Mnemosyne filters out any candidate whose normalized text matches a
  forgotten key.
- Write a tombstone to Hindsight: a new memory whose body is
  `[FORGOTTEN <date>] <original>` with tags `forgotten:<date>` and
  `supersedes:<key>`. This lets recall callers that bypass our filter
  still see the marker.
- Append to `forgotten.jsonl` for an audit trail.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import config
from .fact_store import FactStore, _canonical_key, date_tag, today_iso

logger = logging.getLogger(__name__)


def _audit_log_path() -> Path:
    return config.plugin_dir() / "forgotten.jsonl"


def _append_audit(entry: Dict[str, Any]) -> None:
    path = _audit_log_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("a") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as exc:
        logger.debug("mnemosyne.forget: audit log write failed: %s", exc)


def is_forgotten(fact_store: FactStore, text: str) -> bool:
    """Used by the recall filter — does this candidate match a forgotten key?"""
    return fact_store.is_forgotten(text)


def forget_text(
    fact_store: FactStore,
    hindsight_provider: Optional[Any],
    text: str,
    *,
    reason: str = "user_request",
) -> Dict[str, Any]:
    """Mark a single fact as forgotten. Returns audit entry."""
    key = _canonical_key(text)
    when = today_iso()

    fact_store.mark_forgotten(text)

    tombstone_written = False
    if hindsight_provider is not None:
        try:
            hindsight_provider.handle_tool_call(
                "hindsight_retain",
                {
                    "content": f"[FORGOTTEN {when}] {text}",
                    "tags": [f"forgotten:{when}", f"supersedes:{key}", f"reason:{reason}"],
                },
            )
            tombstone_written = True
        except Exception as exc:
            logger.debug("mnemosyne.forget: tombstone write failed: %s", exc)

    entry = {
        "ts": when,
        "canonical_key": key,
        "text": text,
        "reason": reason,
        "tombstone_written": tombstone_written,
    }
    _append_audit(entry)
    return entry


_WORD_RE = re.compile(r"\w+", flags=re.UNICODE)
_FORGET_STOP = {
    "the", "a", "an", "is", "are", "was", "were", "of", "in", "on", "at",
    "to", "from", "by", "for", "with", "and", "or", "but", "this", "that",
    "и", "в", "на", "не", "что", "это", "как", "по", "из", "к", "у", "о",
    "от", "за", "со", "до", "для", "при", "без", "то", "же",
    "user", "пользователь", "user's", "пользователя", "пользователю",
}


def _content_tokens(text: str) -> set:
    return {
        t.lower() for t in _WORD_RE.findall(text or "")
        if len(t) > 1 and t.lower() not in _FORGET_STOP
    }


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    inter = a & b
    union = a | b
    return len(inter) / len(union) if union else 0.0


def _containment(query_tokens: set, cand_tokens: set) -> float:
    """How much of the query is present in the candidate.

    Jaccard punishes long candidates against short queries (a 1-token
    query matched in a 20-token candidate scores 0.05) — that breaks the
    forget UX when the agent passes a single concrete keyword like
    "Barsik". Containment ``|q ∩ c| / |q|`` ignores candidate length and
    asks the right question: did the query land inside this candidate?
    """
    if not query_tokens:
        return 0.0
    inter = query_tokens & cand_tokens
    return len(inter) / len(query_tokens)


def forget_by_query(
    fact_store: FactStore,
    hindsight_provider: Optional[Any],
    query: str,
    *,
    confirm: Optional[Any] = None,
    confirmed: bool = False,
    indices: Optional[List[int]] = None,
    max_items: int = 30,
    min_overlap: float = 0.0,
) -> Dict[str, Any]:
    """Recall candidates matching `query`, mark the chosen ones as forgotten.

    Symmetry with memory_recall is the goal: the agent should see in the
    forget preview the same items it sees from recall. Otherwise it
    can't actually delete what it wants to delete.

      * Default behaviour is **dry-run** — return candidates with
        ``confirmed=False`` in the response, do not mark anything.
      * Caller must re-invoke with ``confirmed=True`` to actually forget.
      * Optional ``indices`` (1-based) lets the caller pick a subset.
      * ``max_items`` defaults to 30, matching what recall returns.
      * ``min_overlap`` defaults to 0 (off). Raise it to filter out
        candidates whose query tokens aren't present (containment, not
        Jaccard) — useful if the agent passes a vague query, but the
        explicit two-step confirm already handles misuse.

    The two-step confirm + agent's own visual review IS the safety. The
    overlap filter and the small max_items cap were double-locking after
    the May 6 incident, which had a different root cause (reranker
    drift, not insufficient filtering).

    ``confirm`` is the legacy callback API used by the interactive CLI.
    Tool callers should use ``confirmed=True/False`` instead.
    """
    if hindsight_provider is None:
        return {"error": "hindsight unavailable", "forgotten": [], "candidates": []}

    try:
        raw = hindsight_provider.handle_tool_call(
            "hindsight_recall",
            {"query": query, "max_tokens": 4096},
        )
    except Exception as exc:
        return {"error": f"recall failed: {exc}", "forgotten": [], "candidates": []}

    raw_candidates = _extract_candidates(raw)
    query_tokens = _content_tokens(query)

    # Score every candidate by query-containment (how much of the query
    # is present), drop any below threshold. Containment, not Jaccard,
    # so a short query like "Barsik" matches long candidates that
    # mention it. Default threshold is 0 — show everything recall would
    # show, let the agent + user decide.
    scored: List[Dict[str, Any]] = []
    seen: set = set()
    for c in raw_candidates:
        text = c if isinstance(c, str) else (c.get("text") or c.get("content") or "")
        text = (text or "").strip()
        if not text:
            continue
        ckey = text.lower()
        if ckey in seen:
            continue
        seen.add(ckey)
        overlap = _containment(query_tokens, _content_tokens(text))
        if overlap < min_overlap:
            continue
        scored.append({"text": text, "overlap": round(overlap, 3)})

    scored.sort(key=lambda x: x["overlap"], reverse=True)
    candidates = scored[:max_items]

    # Legacy callback path (interactive CLI confirm)
    if confirm is not None:
        try:
            chosen = list(confirm([c["text"] for c in candidates]) or [])
        except Exception as exc:
            logger.debug("mnemosyne.forget: confirm callback failed: %s", exc)
            chosen = []
        forgotten = [
            forget_text(fact_store, hindsight_provider, t, reason="forget_by_query")
            for t in chosen if t
        ]
        return {"forgotten": forgotten, "query": query, "ts": today_iso(),
                "candidates": [c["text"] for c in candidates]}

    # Tool-caller path: dry-run unless confirmed=True.
    if not confirmed:
        return {
            "preview": True,
            "forgotten": [],
            "candidates": [
                {"index": i + 1, "text": c["text"], "overlap": c["overlap"]}
                for i, c in enumerate(candidates)
            ],
            "instructions": (
                "DRY-RUN. Show these candidates to the user, get explicit "
                "confirmation, then re-invoke memory_forget with "
                "confirmed=true (and optionally indices=[1,3,...] to pick "
                "a subset). Without confirmed=true nothing is forgotten."
            ),
            "query": query,
            "min_overlap": min_overlap,
        }

    # Confirmed path — apply selected indices, or all if none given.
    if indices:
        chosen = [c for c in candidates
                  if any(i for i in indices if 1 <= int(i) <= len(candidates)
                         and candidates[int(i) - 1] is c)]
    else:
        chosen = candidates

    # Phase 1 — synchronous, fast: mark canonical keys forgotten and
    # register ONE semantic signature for the whole op. The signature
    # is what actually drops paraphrases at recall time, so the user
    # sees the effect immediately even though tombstones are still
    # being written in the background.
    op_tokens: set = set()
    chosen_texts: List[str] = []
    for c in chosen:
        text = c["text"]
        chosen_texts.append(text)
        try:
            fact_store.mark_forgotten(text)
        except Exception as exc:
            logger.debug("mnemosyne.forget: mark_forgotten failed for %r: %s",
                         text[:80], exc)
        op_tokens |= _content_tokens(text)
    # Also fold in query tokens so the sig fires on future paraphrases
    # that share the user's intent vocabulary.
    op_tokens |= _content_tokens(query)

    sig_id = 0
    if op_tokens:
        try:
            merge = float(config.get("forget", "signature_merge_jaccard",
                                     default=0.7))
            sig_id = fact_store.add_signature(
                list(op_tokens),
                examples=chosen_texts[:5],
                query=query,
                merge_jaccard=merge,
            )
        except Exception as exc:
            logger.debug("mnemosyne.forget: add_signature failed: %s", exc)

    audit_now = today_iso()
    audit_entries: List[Dict[str, Any]] = []
    for text in chosen_texts:
        entry = {
            "ts": audit_now,
            "canonical_key": _canonical_key(text),
            "text": text,
            "reason": "forget_by_query",
            "tombstone_written": False,  # filled in by background worker
            "sig_id": sig_id,
        }
        _append_audit(entry)
        audit_entries.append(entry)

    # Phase 2 — fire-and-forget tombstones. Each retain takes ~5-15s
    # because Hindsight runs them through embedding + reranker. We
    # don't block the agent on this; the signature filter already hides
    # the records from recall.
    write_ts = bool(config.get("forget", "write_tombstones", default=True))
    async_ts = bool(config.get("forget", "write_tombstones_async", default=True))
    if write_ts and hindsight_provider is not None and chosen_texts:
        if async_ts:
            from . import _spawn_tombstone_writer  # late import — set in __init__.py
            _spawn_tombstone_writer(hindsight_provider, chosen_texts, audit_now)
        else:
            for text in chosen_texts:
                _write_tombstone(hindsight_provider, text, audit_now)

    return {
        "forgotten": audit_entries,
        "candidates": [c["text"] for c in candidates],
        "query": query,
        "ts": audit_now,
        "signature_id": sig_id,
        "tombstones": (
            "queued" if (write_ts and async_ts) else
            "written" if write_ts else
            "skipped"
        ),
        "note": (
            "Marked as forgotten in the read-side filter — these "
            "memories will not appear in future recall results "
            "regardless of paraphrasing. Tombstone records in the "
            "Hindsight bank are written in the background as a "
            "secondary safety net (no need to wait for them)."
        ),
    }


def _write_tombstone(hindsight_provider: Any, text: str, when: str) -> bool:
    """Single tombstone retain. Blocking. Logs and returns False on error."""
    if hindsight_provider is None:
        return False
    try:
        hindsight_provider.handle_tool_call(
            "hindsight_retain",
            {
                "content": f"[FORGOTTEN {when}] {text}",
                "tags": [f"forgotten:{when}", "reason:forget_by_query"],
            },
        )
        return True
    except Exception as exc:
        logger.debug("mnemosyne.forget: tombstone write failed: %s", exc)
        return False


def _extract_candidates(raw_recall_result: Any) -> List[Any]:
    """Best-effort extraction of candidate strings from Hindsight's recall.

    Hindsight (via the hermes plugin) returns a JSON string of the shape
    ``{"result": "1. fact A\n2. fact B\n…"}`` — a numbered plain-text list,
    not a structured array. We also keep the older list/dict shapes for
    forward-compat with future Hindsight versions.
    """
    if isinstance(raw_recall_result, list):
        return raw_recall_result
    if isinstance(raw_recall_result, str):
        try:
            data = json.loads(raw_recall_result)
        except Exception:
            return _split_numbered_lines(raw_recall_result)
        return _extract_from_dict(data)
    if isinstance(raw_recall_result, dict):
        return _extract_from_dict(raw_recall_result)
    return []


def _extract_from_dict(data: Any) -> List[Any]:
    if isinstance(data, list):
        return data
    if not isinstance(data, dict):
        return []
    # Numbered-list format used by hermes hindsight plugin
    result = data.get("result")
    if isinstance(result, str) and result.strip():
        items = _split_numbered_lines(result)
        if items:
            return items
    if isinstance(result, list) and result:
        return result
    # Other recall shapes (future / direct API)
    for key in ("memories", "results", "items", "matches", "data"):
        v = data.get(key)
        if isinstance(v, list) and v:
            return v
    return []


_NUMBER_PREFIX = re.compile(r"^\s*\d+\.\s*", flags=re.UNICODE)


def _split_numbered_lines(text: str) -> List[str]:
    """Parse "1. foo\n2. bar | Involving: …\n3. baz" into clean fact strings.

    Drops the empty-result placeholder. Strips per-line metadata after the
    " | Involving: …" separator that Hindsight appends."""
    if not text:
        return []
    if "No relevant memories found" in text:
        return []
    out: List[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        line = _NUMBER_PREFIX.sub("", line)
        head, sep, _ = line.partition(" | ")
        out.append(head if sep else line)
    # Dedup while preserving order — Hindsight often returns the same fact
    # multiple times in different surface forms.
    seen: set = set()
    deduped: List[str] = []
    for item in out:
        if item not in seen:
            deduped.append(item)
            seen.add(item)
    return deduped


# Tool schema served via the curated tools surface
MEMORY_FORGET_SCHEMA = {
    "name": "memory_forget",
    "description": (
        "TWO-STEP forget tool. First call WITHOUT confirmed=true returns a "
        "preview list of candidate facts that would be forgotten — show this "
        "list to the user verbatim and ask them to confirm. Then call again "
        "with confirmed=true (and optionally indices=[1,3,...] to pick "
        "specific candidates). Forgotten facts will not appear in future "
        "memory_recall results. Use ONLY when the user explicitly asks to "
        "forget something specific."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "What to forget. Phrase it CONCRETELY — the more specific "
                    "the words, the more accurate the match. Vague queries "
                    "('cat', 'project') will match too broadly and the "
                    "preview will reject most candidates."
                ),
            },
            "confirmed": {
                "type": "boolean",
                "description": (
                    "Set to true ONLY after the user has explicitly approved "
                    "the candidates from the previous preview call. Default "
                    "false (preview only)."
                ),
            },
            "indices": {
                "type": "array",
                "items": {"type": "integer"},
                "description": (
                    "Optional 1-based indices into the previous preview to "
                    "narrow the forget to a subset. Omit to forget every "
                    "candidate that survived the overlap filter."
                ),
            },
            "max_items": {
                "type": "integer",
                "description": "Max candidates to consider (default 30, matching recall).",
            },
        },
        "required": ["query"],
    },
}
