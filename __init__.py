"""Mnemosyne — composite memory provider for Hermes.

Wraps HonchoMemoryProvider (user model) and HindsightMemoryProvider
(facts) via composition. Adds:

  * Plan item 1 — date tags on every retained fact
  * Plan item 2 — repetition counter (FactStore SQLite)
  * Plan item 3 — pre-write dedup against semantic neighbours
  * Plan item 5 — anchor_card.md always-pinned facts
  * Plan item 6 — sectioned 3-block prefetch + conflict-aware labeling
  * Plan item 7 — recovery from session transcripts on startup
  * Plan item 8 — bulk import (via CLI)
  * Plan item 10 — bridge from built-in memory (USER.md/MEMORY.md)
                  to Hindsight with mention_count=10 forced strong signal
  * Plan item 11 — explicit forgetting via memory_forget tool & filter

See README.md for the full design and roadmap.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Parent-package + submodule bootstrap.
#
# The hermes plugin loader registers user-installed plugins under the synthetic
# package `_hermes_user_memory.<name>`, but it does NOT register the parent
# `_hermes_user_memory` itself. It also pre-registers each submodule in
# sys.modules and then exec_module()s them — so any submodule using
# `from . import …` blows up with ModuleNotFoundError on the parent. The
# loader logs that exception at DEBUG and continues with empty submodule stubs
# in sys.modules, which then poison every relative import we make from this
# file. The fix below: synthesise the parent packages and force-reload our
# submodules in dependency order so each `from . import …` here picks up
# fully-initialised module objects.
# ---------------------------------------------------------------------------
import importlib.util as _importlib_util
import os as _os
import sys as _sys
import types as _types

_PLUGIN_DIR = _os.path.dirname(_os.path.abspath(__file__))
_PARENT_PKG = "_hermes_user_memory"
_FULL_PKG = f"{_PARENT_PKG}.mnemosyne"

for _pkg_name, _pkg_paths in (
    (_PARENT_PKG, [_os.path.dirname(_PLUGIN_DIR)]),
    (_FULL_PKG, [_PLUGIN_DIR]),
):
    if _pkg_name not in _sys.modules:
        _ns = _types.ModuleType(_pkg_name)
        _ns.__path__ = _pkg_paths
        _sys.modules[_pkg_name] = _ns


def _mnemosyne_force_reload(submodule_name: str):
    full = f"{_FULL_PKG}.{submodule_name}"
    fpath = _os.path.join(_PLUGIN_DIR, f"{submodule_name}.py")
    if not _os.path.exists(fpath):
        return None
    spec = _importlib_util.spec_from_file_location(full, fpath)
    if spec is None or spec.loader is None:
        return None
    mod = _importlib_util.module_from_spec(spec)
    _sys.modules[full] = mod
    spec.loader.exec_module(mod)
    return mod


for _sub in ("config", "conflict", "fact_store", "forget", "recovery", "importer", "dedup"):
    try:
        _mnemosyne_force_reload(_sub)
    except Exception as _exc:  # pragma: no cover
        import logging as _logging
        _logging.getLogger(__name__).warning(
            "mnemosyne: failed to load submodule %s: %s", _sub, _exc
        )

import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from typing import Any, Dict, List, Optional

from agent.memory_provider import MemoryProvider

from . import config
from .conflict import is_contradiction, label_pair
from .fact_store import FactStore, _canonical_key, today_iso
from .forget import (
    MEMORY_FORGET_SCHEMA,
    forget_by_query,
    forget_text,
    is_forgotten as _is_forgotten,
    _write_tombstone,
)
from .recovery import initialize_cursor_if_missing, replay_missed

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Background tombstone writer.
#
# `forget.py` calls into us here via late-import so the heavy executor lives
# on the provider but the forget logic stays self-contained. One pool keeps
# tombstone backlog from blocking other writes (it has its own worker, not
# stolen from the prefetch/recall pool).
# ---------------------------------------------------------------------------

_tombstone_executor: Optional[ThreadPoolExecutor] = None
_tombstone_lock = threading.Lock()


def _get_tombstone_executor() -> ThreadPoolExecutor:
    global _tombstone_executor
    with _tombstone_lock:
        if _tombstone_executor is None:
            _tombstone_executor = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="mnemosyne-tombstone"
            )
        return _tombstone_executor


def _spawn_tombstone_writer(hindsight_provider, texts, when: str) -> None:
    """Submit one job that writes all tombstones sequentially. Sequential
    (not parallel) on purpose: each retain hits the same Hindsight
    embedding pipeline, parallelising would just queue inside Hindsight."""
    if not hindsight_provider or not texts:
        return
    executor = _get_tombstone_executor()

    def _runner():
        ok = 0
        for text in texts:
            if _write_tombstone(hindsight_provider, text, when):
                ok += 1
        logger.info(
            "mnemosyne: tombstones written %d/%d (when=%s)",
            ok, len(texts), when,
        )

    try:
        executor.submit(_runner)
    except Exception as exc:
        logger.debug("mnemosyne: tombstone executor submit failed: %s", exc)


# ---------------------------------------------------------------------------
# Curated tool schemas (plan item 6 — tightened role descriptions)
# ---------------------------------------------------------------------------

_PROFILE_SCHEMA = {
    "name": "memory_profile",
    "description": (
        "ONLY for the user's profile card: name, role, communication style, "
        "stable preferences. Read or update. Do NOT use for general facts or "
        "past-conversation history — for those use memory_recall."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "card": {
                "type": "array",
                "items": {"type": "string"},
                "description": "New card as a list of fact strings. Omit to read.",
            },
        },
        "required": [],
    },
}

_REASONING_SCHEMA = {
    "name": "memory_reasoning",
    "description": (
        "ONLY questions about the user as a person: their style, habits, "
        "behavioral patterns, what approach works best with them. NOT for "
        "general knowledge and NOT for facts from past conversations — for "
        "those use memory_recall or memory_reflect."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Natural-language question about the user as a person.",
            },
            "reasoning_level": {
                "type": "string",
                "enum": ["minimal", "low", "medium", "high", "max"],
                "description": "Depth control. Omit for default (low).",
            },
        },
        "required": ["query"],
    },
}

_CONCLUDE_SCHEMA = {
    "name": "memory_conclude",
    "description": (
        "Record a stable user-related conclusion (preference, habit, style). "
        "NOT for technical facts or events — those go through memory_recall."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "conclusion": {"type": "string", "description": "The conclusion to persist."},
        },
        "required": ["conclusion"],
    },
}

_RECALL_SCHEMA = {
    "name": "memory_recall",
    "description": (
        "FIRST CHOICE for 'do you remember when we did X?', 'we discussed this', "
        "'how did we fix that before'. Multi-strategy search (semantic + entity "
        "graph) over all past conversations. Returns relevant facts and "
        "fragments. THIS IS THE MAIN LONG-TERM MEMORY TOOL."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to look for."},
            "max_tokens": {
                "type": "integer",
                "description": "Token budget (default 800, max 4096).",
            },
        },
        "required": ["query"],
    },
}

_REFLECT_SCHEMA = {
    "name": "memory_reflect",
    "description": (
        "LLM synthesis across past-conversation facts. Use when you need a "
        "summary spanning multiple sources ('what did we conclude about X?', "
        "'what facts do we have on topic Y?'). NOT for questions about the "
        "user as a person."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Natural-language question."},
        },
        "required": ["query"],
    },
}


# Maps curated tool names → (inner_provider_attr, inner_tool_name).
_TOOL_DISPATCH = {
    "memory_profile":   ("honcho",    "honcho_profile"),
    "memory_reasoning": ("honcho",    "honcho_reasoning"),
    "memory_conclude":  ("honcho",    "honcho_conclude"),
    "memory_recall":    ("hindsight", "hindsight_recall"),
    "memory_reflect":   ("hindsight", "hindsight_reflect"),
}


def _truncate_to_chars(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    last_nl = cut.rfind("\n")
    if last_nl > max_chars * 0.5:
        cut = cut[:last_nl]
    return cut + "\n…[truncated]"


# Hindsight enforces "Query too long: N tokens exceeds maximum of 500".
# 1500 chars ≈ 350-450 tokens (RU runs higher chars/token than EN); leaves
# headroom for any query expansion Hindsight does internally.
_RECALL_QUERY_MAX_CHARS = 1500


def _truncate_recall_query(query: str) -> str:
    """Trim query so it never trips Hindsight's 500-token recall limit.
    Prefers to cut at a word boundary near the end."""
    if not query:
        return query
    if len(query) <= _RECALL_QUERY_MAX_CHARS:
        return query
    cut = query[:_RECALL_QUERY_MAX_CHARS]
    last_space = cut.rfind(" ")
    if last_space > _RECALL_QUERY_MAX_CHARS * 0.7:
        cut = cut[:last_space]
    return cut


class MnemosyneMemoryProvider(MemoryProvider):
    """Composite provider — Honcho for user model, Hindsight for facts."""

    def __init__(self) -> None:
        self._honcho: Optional[MemoryProvider] = None
        self._hindsight: Optional[MemoryProvider] = None
        # 4 workers: 2 for write fan-out (sync_turn), 2 spare for parallel
        # tool calls so agent-driven recall isn't queued behind background
        # retain jobs.
        self._executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="mnemosyne")
        self._fact_store: Optional[FactStore] = None
        self._init_lock = threading.Lock()
        self._initialized = False
        # Last prefetch payload — stripped from assistant_content before
        # we hand the turn to Hindsight's LLM extractor. Without this the
        # extractor re-extracts whatever we just put into the prompt,
        # producing endless paraphrases of the same fact (the "Barsik
        # loop"). Updated atomically; no lock needed for last-write-wins
        # semantics.
        self._last_prefetch: str = ""
        self._load_inner_providers()

    def _inject_hindsight_routing_env(self) -> None:
        """Set Hindsight embedding/reranker routing in os.environ so the
        embedded daemon (which inherits parent environ) picks it up.

        Routes:
          * Embeddings → local omlx Jina v5 multilingual via OpenAI-compatible
            API on :8000 (fast, runs on this Mac).
          * Reranker → cloud rerank via litellm on :4000 (alias model name
            `rerank`; the actual upstream is configured in litellm).

        Values are read from mnemosyne config.json under "hindsight_env"
        with hard-coded defaults so a fresh install Just Works.
        """
        import os as _o

        defaults = {
            "HINDSIGHT_API_EMBEDDINGS_PROVIDER": "openai",
            "HINDSIGHT_API_EMBEDDINGS_OPENAI_API_KEY": "sk-local-litellm",
            "HINDSIGHT_API_EMBEDDINGS_OPENAI_BASE_URL": "http://localhost:8000/v1",
            "HINDSIGHT_API_EMBEDDINGS_OPENAI_MODEL":
                "jina-embeddings-v5-text-small-retrieval-mlx",
            "HINDSIGHT_API_RERANKER_PROVIDER": "cohere",
            "HINDSIGHT_API_RERANKER_COHERE_API_KEY": "sk-local-litellm",
            "HINDSIGHT_API_RERANKER_COHERE_BASE_URL": "http://localhost:4000/v1/rerank",
            "HINDSIGHT_API_RERANKER_COHERE_MODEL": "rerank",
        }
        overrides = config.get("hindsight_env", default={}) or {}
        for k, v in defaults.items():
            # Don't clobber if the user has set something themselves at the
            # shell level — they win over our defaults.
            if k in _o.environ and _o.environ[k]:
                continue
            value = overrides.get(k, v)
            if value:
                _o.environ[k] = str(value)

    def _load_inner_providers(self) -> None:
        try:
            from plugins.memory.honcho import HonchoMemoryProvider
            self._honcho = HonchoMemoryProvider()
        except Exception as exc:
            logger.warning("mnemosyne: failed to load Honcho inner provider: %s", exc)

        try:
            from plugins.memory.hindsight import HindsightMemoryProvider
            self._hindsight = HindsightMemoryProvider()
        except Exception as exc:
            logger.warning("mnemosyne: failed to load Hindsight inner provider: %s", exc)

    @property
    def name(self) -> str:
        return "mnemosyne"

    def is_available(self) -> bool:
        h_ok = bool(self._honcho and self._honcho.is_available())
        i_ok = bool(self._hindsight and self._hindsight.is_available())
        return h_ok and i_ok

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(self, session_id: str, **kwargs) -> None:
        with self._init_lock:
            # The Hindsight embedded daemon inherits the parent process's
            # os.environ at start time (daemon_embed_manager.py:331). Inject
            # our Jina/Qwen3 routing here so the daemon picks them up via
            # env without needing to patch the hermes-shipped hindsight
            # plugin or rely on the auto-managed profile.env (which gets
            # re-materialised on every start with only LLM keys).
            self._inject_hindsight_routing_env()

            if self._honcho:
                try:
                    self._honcho.initialize(session_id, **kwargs)
                except Exception as exc:
                    logger.warning("mnemosyne: Honcho initialize failed: %s", exc)
            if self._hindsight:
                try:
                    self._hindsight.initialize(session_id, **kwargs)
                except Exception as exc:
                    logger.warning("mnemosyne: Hindsight initialize failed: %s", exc)

            try:
                self._fact_store = FactStore()
                # One-shot vacuum: bound the signatures table so it
                # never silently grows past the configured ceiling.
                try:
                    max_sigs = int(config.get("forget", "max_signatures",
                                              default=1000))
                    stale_days = config.get("forget", "signature_stale_days",
                                            default=365)
                    stale = int(stale_days) if stale_days else None
                    removed = self._fact_store.vacuum_signatures(
                        max_count=max_sigs, stale_days=stale,
                    )
                    if removed:
                        logger.info("mnemosyne: vacuumed %d forget signature(s)",
                                    removed)
                except Exception as exc:
                    logger.debug("mnemosyne: signature vacuum skipped: %s", exc)
            except Exception as exc:
                logger.warning("mnemosyne: FactStore init failed: %s", exc)
                self._fact_store = None

            # Plan item 7 — recovery from session transcripts.
            try:
                if initialize_cursor_if_missing():
                    logger.info("mnemosyne: recovery cursor stamped at current state")
                else:
                    summary = replay_missed(self._hindsight, max_pairs=50)
                    if summary.get("replayed", 0):
                        logger.info("mnemosyne: recovery replayed %d turn pair(s)",
                                    summary["replayed"])
            except Exception as exc:
                logger.debug("mnemosyne: recovery skipped: %s", exc)

            self._initialized = True

    def shutdown(self) -> None:
        if self._honcho:
            try:
                self._honcho.shutdown()
            except Exception as exc:
                logger.debug("mnemosyne: Honcho shutdown failed: %s", exc)
        if self._hindsight:
            try:
                self._hindsight.shutdown()
            except Exception as exc:
                logger.debug("mnemosyne: Hindsight shutdown failed: %s", exc)
        self._executor.shutdown(wait=False)

    def system_prompt_block(self) -> str:
        """Tell the agent about Mnemosyne's curated tool surface.

        Crucially we DO NOT delegate to ``self._honcho.system_prompt_block()``
        or ``self._hindsight.system_prompt_block()`` — those describe their
        native tool names (``honcho_*`` / ``hindsight_*``) which we
        deliberately hide behind our 6 curated tools. Letting them through
        would tell the LLM that ``honcho_search`` etc. exist when in fact
        only the curated set is callable, leading to phantom tool calls and
        confused tool selection."""
        return (
            "# Memory (Mnemosyne)\n"
            "You have a long-term memory system backed by two layers: a user "
            "model (style, preferences, behavioral patterns) and a fact store "
            "(prior conversations, entities, decisions). Both are accessed "
            "through these six tools — DO NOT call any tool name that starts "
            "with `honcho_` or `hindsight_`; those are not exposed.\n"
            "\n"
            "When to use which:\n"
            "- `memory_recall(query)`        — FIRST CHOICE for 'do you "
            "remember…', 'we discussed…', 'how did we fix…'.\n"
            "- `memory_reflect(query)`       — synthesised summary across "
            "multiple past facts ('what did we conclude about X?').\n"
            "- `memory_profile(card?)`       — read or update the user's "
            "profile card (stable preferences, role, communication style).\n"
            "- `memory_reasoning(query)`     — questions about the user "
            "**as a person** (style, habits). Slow — use sparingly.\n"
            "- `memory_conclude(conclusion)` — record a stable fact about "
            "the user.\n"
            "- `memory_forget(query)`        — TWO STEPS. First call with "
            "just the query returns a preview list of candidates. Show that "
            "list to the user verbatim, get explicit confirmation, then "
            "re-invoke with `confirmed=true` (or `indices=[1,3]` to pick a "
            "subset). NEVER call with `confirmed=true` on the first try.\n"
        )

    # ------------------------------------------------------------------
    # Prefetch (plan item 6/7 fusion: anchor → peer card → Hindsight recall)
    # ------------------------------------------------------------------

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        max_total = int(config.get("prefetch", "max_total_tokens", default=4500))
        anchor_budget = int(config.get("prefetch", "anchor_token_budget", default=200))
        peer_card_budget = int(config.get("prefetch", "honcho_card_token_budget", default=200))
        hindsight_budget = int(config.get("prefetch", "hindsight_token_budget", default=4096))

        sections: List[str] = []

        # 1) Anchor card (plan item 5)
        anchor_text = self._read_anchor_card()
        if anchor_text:
            sections.append("# Pinned (anchor card)\n" +
                            _truncate_to_chars(anchor_text, anchor_budget * 4))

        # 2) Honcho peer card — static, no LLM
        peer_card_text = self._fetch_honcho_peer_card()
        if peer_card_text:
            sections.append("# User profile\n" +
                            _truncate_to_chars(peer_card_text, peer_card_budget * 4))

        # 3) Hindsight recall — relevance-filtered facts.
        hindsight_text = self._fetch_hindsight_recall(query, hindsight_budget)
        if hindsight_text:
            # Apply forget filter (plan item 11): drop lines whose canonical
            # key is in fact_store.forgotten.
            hindsight_text = self._filter_forgotten(hindsight_text)
            if hindsight_text:
                sections.append("# Facts (relevant)\n" +
                                _truncate_to_chars(hindsight_text, hindsight_budget * 4))

        # 4) Conflict-aware labeling between peer card and facts (item 6).
        if len(sections) >= 3:
            sections = self._apply_conflict_resolver(sections)

        result = "\n\n".join(sections)
        # Final hard cap
        capped = _truncate_to_chars(result, max_total * 4)
        self._last_prefetch = capped
        return capped

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        if self._honcho:
            try:
                self._honcho.queue_prefetch(query, session_id=session_id)
            except Exception:
                pass
        if self._hindsight:
            try:
                self._hindsight.queue_prefetch(query, session_id=session_id)
            except Exception:
                pass

    def _read_anchor_card(self) -> str:
        fn = config.get("anchor_card", "filename", default="anchor_card.md")
        path = config.plugin_dir() / fn
        if not path.exists():
            return ""
        try:
            text = path.read_text(encoding="utf-8")
        except Exception:
            return ""
        # Drop comment lines (#-prefixed, or lines starting with '# ')
        keep = []
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("#"):
                continue
            keep.append(stripped)
        return "\n".join(f"- {line}" for line in keep)

    def _fetch_honcho_peer_card(self) -> str:
        """Static peer card via Honcho — no LLM, no representation summary."""
        if self._honcho is None:
            return ""
        try:
            raw = self._honcho.handle_tool_call("honcho_profile", {})
            data = json.loads(raw) if isinstance(raw, str) else raw
            card = data.get("card") if isinstance(data, dict) else None
            if isinstance(card, list) and card:
                return "\n".join(f"- {item}" for item in card)
            if isinstance(data, dict) and data.get("hint"):
                return f"_{data['hint']}_"
        except Exception as exc:
            logger.debug("mnemosyne: honcho profile fetch failed: %s", exc)
        return ""

    def _fetch_hindsight_recall(self, query: str, max_tokens: int) -> str:
        if self._hindsight is None or not query:
            return ""
        try:
            raw = self._hindsight.handle_tool_call(
                "hindsight_recall",
                {"query": _truncate_recall_query(query),
                 "max_tokens": min(max_tokens, 4096)},
            )
            if isinstance(raw, str):
                # Try JSON, fall back to plain text
                try:
                    data = json.loads(raw)
                    return self._format_hindsight_results(data)
                except Exception:
                    return raw
            return self._format_hindsight_results(raw)
        except Exception as exc:
            logger.debug("mnemosyne: hindsight recall failed: %s", exc)
            return ""

    def _format_hindsight_results(self, data: Any) -> str:
        if isinstance(data, str):
            return self._dedupe_recall_text(data)
        if isinstance(data, list):
            joined = "\n".join(self._extract_text(item) for item in data
                               if self._extract_text(item))
            return self._dedupe_recall_text(joined)
        if isinstance(data, dict):
            # 'result' is the key the hermes hindsight plugin uses for
            # numbered-list recall output. Check it first.
            for key in ("result", "memories", "results", "items", "matches", "data", "text"):
                v = data.get(key)
                if v:
                    return self._format_hindsight_results(v)
        return str(data)

    @staticmethod
    def _dedupe_recall_text(text: str) -> str:
        """Hindsight returns top-N candidates ranked by similarity to the
        QUERY, not pairwise-distinct. With 40+ paraphrases of one event in
        the bank, all of them survive the existing exact-canonical-key
        filter and bloat context.

        Hybrid clustering (see ``dedup.cluster_lines``):
          - Jaccard token overlap groups paraphrases.
          - Embedding cosine confirms the merge — pairs that pass
            Jaccard but fail cosine are kept apart, protecting against
            "Barsik got sick" vs "Barsik died" type collapses.
        Configurable via ``prefetch.dedup_*`` keys."""
        if not text:
            return text
        from .dedup import cluster_lines

        cleaned: List[str] = []
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            content = stripped
            for prefix in range(1, 100):
                pfx = f"{prefix}. "
                if content.startswith(pfx):
                    content = content[len(pfx):]
                    break
            head, sep, _ = content.partition(" | Involving:")
            content = head if sep else content
            if content:
                cleaned.append(content)

        kept = cluster_lines(cleaned)
        return "\n".join(f"- {item}" for item in kept)

    @staticmethod
    def _extract_text(item: Any) -> str:
        if isinstance(item, str):
            return item
        if isinstance(item, dict):
            return (item.get("text") or item.get("content") or
                    item.get("body") or "")
        return ""

    def _filter_forgotten(self, text: str) -> str:
        """Drop lines whose canonical key was point-forgotten OR whose
        token set hits any stored semantic forget signature.

        Matching uses **candidate-containment**: ``|cand ∩ sig| / |cand|``.
        Asks "is most of this candidate's vocabulary covered by the
        forget signature?" — the right question for asymmetric sizes (a
        signature accumulates many tokens across an op; a candidate is
        one line). Symmetric Jaccard fails here: a 6-token candidate vs
        an 11-token signature with 4 shared words scores 0.31 — below
        any safe threshold — even though every content word in the
        candidate IS in the signature. Containment scores 0.66 and
        catches the paraphrase as intended.
        """
        if not self._fact_store:
            return text

        # Refresh the in-memory signature cache on demand. The table is
        # tiny (designed to stay <1k rows), so we just re-read it on
        # each filter pass for correctness; cost is microseconds.
        try:
            sigs = self._fact_store.list_signatures()
        except Exception:
            sigs = []
        try:
            cont_min = float(config.get("forget", "signature_jaccard_min",
                                        default=0.5))
        except Exception:
            cont_min = 0.5

        # Pre-compute signature token sets once per call.
        sig_tokens: List[tuple] = []
        for sig in sigs:
            tokens = set(sig.get("tokens") or [])
            if tokens:
                sig_tokens.append((sig.get("id"), tokens))

        from .dedup import _content_tokens
        kept: List[str] = []
        sig_hits: set = set()
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                kept.append(line)
                continue
            # Already-tombstoned lines from Hindsight
            if "[FORGOTTEN" in line:
                continue
            # Exact-key forget marks
            if _is_forgotten(self._fact_store, line):
                continue
            # Semantic signature filter (candidate-containment)
            if sig_tokens:
                content = stripped
                # strip "1. " number prefix and trailing "| Involving:"
                head, sep, _ = content.partition(" | Involving:")
                content = head if sep else content
                line_tokens = _content_tokens(content)
                if line_tokens:
                    matched = False
                    for sid, tokens in sig_tokens:
                        inter = line_tokens & tokens
                        cont = len(inter) / len(line_tokens)
                        if cont >= cont_min:
                            matched = True
                            if sid:
                                sig_hits.add(sid)
                            break
                    if matched:
                        continue
            kept.append(line)

        # Bump last_match_ts on signatures that did real work — keeps
        # vacuum from dropping useful sigs.
        for sid in sig_hits:
            try:
                self._fact_store.touch_signature(int(sid))
            except Exception:
                pass

        return "\n".join(kept)

    def _apply_conflict_resolver(self, sections: List[str]) -> List[str]:
        """If we detect a contradiction between the user profile and a fact,
        annotate both inline. Best-effort; rule-based detector."""
        if len(sections) < 3:
            return sections
        anchor, profile, facts = sections[0], sections[1], sections[2]
        profile_lines = [l for l in profile.splitlines() if l.startswith("- ")]
        fact_lines = [l for l in facts.splitlines() if l.startswith("- ") or l.startswith("# Facts") is False and l.strip()]
        annotated_facts: List[str] = []
        today = today_iso()
        for fact_line in facts.splitlines():
            if not fact_line.strip() or fact_line.startswith("#"):
                annotated_facts.append(fact_line)
                continue
            conflict = False
            for profile_line in profile_lines:
                if is_contradiction(fact_line, profile_line):
                    a, b = label_pair(
                        fact_line, {"label": "Hindsight", "when": today},
                        profile_line, {"label": "Honcho profile"},
                    )
                    annotated_facts.append(a)
                    annotated_facts.append(b)
                    conflict = True
                    break
            if not conflict:
                annotated_facts.append(fact_line)
        return [anchor, profile, "\n".join(annotated_facts)]

    # ------------------------------------------------------------------
    # Write path (plan items 1, 2, 3, 10)
    # ------------------------------------------------------------------

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        # Bump fact_store on user turn (cheap, exact-key dedup only).
        if self._fact_store and user_content.strip():
            try:
                self._fact_store.bump(user_content, source="conversation")
            except Exception as exc:
                logger.debug("mnemosyne: fact_store bump failed: %s", exc)

        # Break the prefetch→extract→retain feedback loop: strip from the
        # assistant turn anything we already injected as recalled context,
        # so Hindsight's extractor can't re-mint paraphrases of facts it
        # just gave us. user_content is left untouched — that's the only
        # actual new information in the turn.
        cleaned_assistant = self._strip_prefetched(assistant_content)

        futures = []
        if self._honcho:
            futures.append(self._executor.submit(
                self._honcho.sync_turn, user_content, cleaned_assistant,
                session_id=session_id,
            ))
        if self._hindsight:
            futures.append(self._executor.submit(
                self._hindsight.sync_turn, user_content, cleaned_assistant,
                session_id=session_id,
            ))
        for f in futures:
            try:
                f.result(timeout=5)
            except Exception as exc:
                logger.debug("mnemosyne: sync_turn fan-out failure: %s", exc)

    _SHINGLE_SIZE = 8        # words per shingle
    _SHINGLE_HIT_RATIO = 0.5 # fraction of a paragraph's shingles that must
                             # be in the prefetch to qualify for stripping

    def _strip_prefetched(self, assistant_content: str) -> str:
        """Drop paragraphs from `assistant_content` that mostly repeat
        text we just put into the prompt via prefetch.

        Why: Hindsight.sync_turn LLM-extracts facts from the JSON of the
        turn. If the assistant cited or paraphrased the prefetched
        memories, the extractor mints fresh records of them — every
        session. Recall picks them up next time, and the bank grows
        without bound (the Barsik loop).

        Approach: word-shingles, paragraph granularity. If a paragraph's
        shingle hit rate against the prefetch is high, drop it. Conservative
        thresholds: short paragraphs (<8 words) untouched, partial
        paraphrases survive."""
        if not assistant_content or not self._last_prefetch:
            return assistant_content
        if not bool(config.get("prefetch", "strip_from_extraction", default=True)):
            return assistant_content

        prefetch_shingles = self._build_shingles(self._last_prefetch)
        if not prefetch_shingles:
            return assistant_content

        kept_paragraphs: List[str] = []
        for para in assistant_content.split("\n\n"):
            if not para.strip():
                kept_paragraphs.append(para)
                continue
            para_shingles = self._build_shingles(para)
            if len(para_shingles) < 2:
                kept_paragraphs.append(para)
                continue
            hits = sum(1 for sh in para_shingles if sh in prefetch_shingles)
            ratio = hits / len(para_shingles)
            if ratio >= self._SHINGLE_HIT_RATIO:
                logger.debug("mnemosyne: stripped paraphrased paragraph "
                             "(%.0f%% shingle overlap with prefetch)",
                             ratio * 100)
                continue
            kept_paragraphs.append(para)
        return "\n\n".join(kept_paragraphs)

    @classmethod
    def _build_shingles(cls, text: str) -> set:
        if not text:
            return set()
        # Lowercase + strip non-word characters; same logic as
        # fact_store._canonical_key but token-level so word order matters.
        import re as _re
        words = _re.findall(r"\w+", text.lower(), flags=_re.UNICODE)
        if len(words) < cls._SHINGLE_SIZE:
            return set()
        return {
            tuple(words[i:i + cls._SHINGLE_SIZE])
            for i in range(len(words) - cls._SHINGLE_SIZE + 1)
        }

    def on_memory_write(
        self,
        action: str,
        target: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Plan item 10 — built-in memory bridge.

        Mirror every memory(...) call from the user-facing tool into Hindsight
        with `source:user_explicit` and a max-strength fact_store mark."""
        # Pass-through to inner providers so they can do their own bookkeeping.
        if self._honcho:
            try:
                self._honcho.on_memory_write(action, target, content, metadata)
            except Exception:
                pass
        if self._hindsight:
            try:
                self._hindsight.on_memory_write(action, target, content, metadata)
            except Exception:
                pass

        if not content or not action:
            return

        if action == "remove":
            if self._fact_store:
                try:
                    self._fact_store.mark_forgotten(content)
                except Exception:
                    pass
            if self._hindsight:
                try:
                    self._hindsight.handle_tool_call(
                        "hindsight_retain",
                        {
                            "content": f"[FORGOTTEN {today_iso()}] {content}",
                            "tags": [f"forgotten:{today_iso()}", "source:built_in_remove"],
                        },
                    )
                except Exception:
                    pass
            return

        if action in ("add", "replace"):
            if self._fact_store:
                try:
                    self._fact_store.force_strong(content, source="user_explicit")
                except Exception:
                    pass
            if self._hindsight:
                try:
                    tags = [f"ts:{today_iso()}", "source:user_explicit",
                            f"target:{target or 'memory'}"]
                    if action == "replace":
                        tags.append("supersedes:built_in")
                    self._hindsight.handle_tool_call(
                        "hindsight_retain",
                        {"content": content, "tags": tags},
                    )
                except Exception as exc:
                    logger.debug("mnemosyne: hindsight retain mirror failed: %s", exc)

    # ------------------------------------------------------------------
    # Tools (plan item 6 — curated 6 with tightened descriptions)
    # ------------------------------------------------------------------

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        exposed = config.get("tools", "expose", default=[
            "memory_profile", "memory_reasoning", "memory_conclude",
            "memory_recall", "memory_reflect", "memory_forget",
        ])
        catalogue = {
            "memory_profile":   _PROFILE_SCHEMA,
            "memory_reasoning": _REASONING_SCHEMA,
            "memory_conclude":  _CONCLUDE_SCHEMA,
            "memory_recall":    _RECALL_SCHEMA,
            "memory_reflect":   _REFLECT_SCHEMA,
            "memory_forget":    MEMORY_FORGET_SCHEMA,
        }
        return [catalogue[n] for n in exposed if n in catalogue]

    # Per-tool timeout (seconds), env-overridable — see config.py _ENV_MAP.
    # Defaults are generous (3-5 min for reasoning paths) so genuine deep
    # synthesis isn't truncated; tighten via MNEMOSYNE_TIMEOUT_* env vars
    # if a specific call misbehaves.
    @staticmethod
    def _timeout_for(tool_name: str) -> Optional[float]:
        key_map = {
            "memory_recall":    "recall",
            "memory_reasoning": "reasoning",
            "memory_reflect":   "reflect",
            "memory_profile":   "profile",
            "memory_conclude":  "conclude",
            "memory_forget":    "forget",
        }
        key = key_map.get(tool_name)
        if key is None:
            return None
        try:
            return float(config.get("timeouts", key,
                                    default=config.get("timeouts", "default",
                                                       default=120)))
        except Exception:
            return None

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        if tool_name == "memory_forget":
            return self._handle_forget(args)

        mapping = _TOOL_DISPATCH.get(tool_name)
        if not mapping:
            raise NotImplementedError(f"mnemosyne does not handle tool {tool_name}")
        provider_attr, inner_name = mapping
        provider = getattr(self, f"_{provider_attr}", None)
        if provider is None:
            return json.dumps({"error": f"{provider_attr} not available"})

        # Truncate the query argument for tools that hit Hindsight's 500-token
        # recall limit. Defensive — without this, an LLM-formed long query
        # comes back as 400 Bad Request from Hindsight.
        if tool_name in ("memory_recall", "memory_reflect"):
            q = args.get("query")
            if isinstance(q, str) and len(q) > _RECALL_QUERY_MAX_CHARS:
                args = dict(args)
                args["query"] = _truncate_recall_query(q)
                logger.debug("mnemosyne: truncated %s query from %d to %d chars",
                             tool_name, len(q), len(args["query"]))

        timeout = self._timeout_for(tool_name)
        if timeout and timeout > 0:
            future = self._executor.submit(
                provider.handle_tool_call, inner_name, args, **kwargs
            )
            try:
                raw = future.result(timeout=timeout)
            except FuturesTimeout:
                logger.warning(
                    "mnemosyne: %s (→ %s) timed out after %.0fs",
                    tool_name, inner_name, timeout,
                )
                return json.dumps({
                    "error": f"{tool_name} timed out after {timeout:.0f}s "
                             f"(inner: {inner_name}). The underlying memory "
                             f"backend is slow or stuck — try a more focused "
                             f"query or a different memory tool.",
                }, ensure_ascii=False)
            except Exception as exc:
                logger.warning("mnemosyne: %s (→ %s) raised: %s",
                               tool_name, inner_name, exc)
                return json.dumps({"error": f"{tool_name} failed: {exc}"},
                                  ensure_ascii=False)
        else:
            raw = provider.handle_tool_call(inner_name, args, **kwargs)

        # Post-process recall: dedup duplicate surface forms and drop forgotten lines.
        if tool_name == "memory_recall":
            try:
                cleaned = self._format_hindsight_results(json.loads(raw)
                                                        if isinstance(raw, str) else raw)
                cleaned = self._filter_forgotten(cleaned)
                if cleaned.strip():
                    return json.dumps({"result": cleaned}, ensure_ascii=False)
                return json.dumps({"result": "No relevant memories found."},
                                  ensure_ascii=False)
            except Exception as exc:
                logger.debug("mnemosyne: recall post-process failed: %s", exc)
                return raw
        return raw

    def _handle_forget(self, args: Dict[str, Any]) -> str:
        query = (args.get("query") or "").strip()
        if not query:
            return json.dumps({"error": "query required"}, ensure_ascii=False)
        if not self._fact_store:
            return json.dumps({"error": "fact_store unavailable"}, ensure_ascii=False)

        confirmed = bool(args.get("confirmed", False))
        indices = args.get("indices")
        if indices is not None:
            try:
                indices = [int(i) for i in indices]
            except Exception:
                indices = None
        max_items = int(args.get("max_items") or 30)

        result = forget_by_query(
            self._fact_store, self._hindsight, query,
            confirmed=confirmed,
            indices=indices,
            max_items=max_items,
        )
        return json.dumps(result, ensure_ascii=False)

    # ------------------------------------------------------------------
    # Other ABC hooks (fan-out)
    # ------------------------------------------------------------------

    def on_turn_start(self, turn_number: int, message: str, **kwargs) -> None:
        if self._honcho:
            try:
                self._honcho.on_turn_start(turn_number, message, **kwargs)
            except Exception:
                pass
        if self._hindsight:
            try:
                self._hindsight.on_turn_start(turn_number, message, **kwargs)
            except Exception:
                pass

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        if self._honcho:
            try:
                self._honcho.on_session_end(messages)
            except Exception:
                pass
        if self._hindsight:
            try:
                self._hindsight.on_session_end(messages)
            except Exception:
                pass

    def on_session_switch(
        self,
        new_session_id: str,
        *,
        parent_session_id: str = "",
        reset: bool = False,
        **kwargs,
    ) -> None:
        if self._honcho:
            try:
                self._honcho.on_session_switch(
                    new_session_id, parent_session_id=parent_session_id,
                    reset=reset, **kwargs,
                )
            except Exception:
                pass
        if self._hindsight:
            try:
                self._hindsight.on_session_switch(
                    new_session_id, parent_session_id=parent_session_id,
                    reset=reset, **kwargs,
                )
            except Exception:
                pass

    def on_pre_compress(self, messages: List[Dict[str, Any]]) -> str:
        parts: List[str] = []
        if self._honcho:
            try:
                s = self._honcho.on_pre_compress(messages) or ""
                if s:
                    parts.append(s)
            except Exception:
                pass
        if self._hindsight:
            try:
                s = self._hindsight.on_pre_compress(messages) or ""
                if s:
                    parts.append(s)
            except Exception:
                pass
        return "\n\n".join(parts)

    def on_delegation(self, task: str, result: str, *, child_session_id: str = "", **kwargs) -> None:
        if self._honcho:
            try:
                self._honcho.on_delegation(task, result, child_session_id=child_session_id, **kwargs)
            except Exception:
                pass
        if self._hindsight:
            try:
                self._hindsight.on_delegation(task, result, child_session_id=child_session_id, **kwargs)
            except Exception:
                pass

    def get_config_schema(self) -> List[Dict[str, Any]]:
        return []

    def save_config(self, values: Dict[str, Any], hermes_home: str) -> None:
        pass


def register(ctx) -> None:
    """Hermes plugin entry point — register Mnemosyne as a memory provider."""
    ctx.register_memory_provider(MnemosyneMemoryProvider())
