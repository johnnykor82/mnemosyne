"""Mnemosyne config — env vars > config.json > built-in defaults.

Every knob can be overridden via an env var (preferred — listed in the
``ENV_*`` table) or via ``$HERMES_HOME/plugins/mnemosyne/config.json``.

Env precedence is highest so you can adjust runtime behaviour from
``~/.hermes/.env`` without touching JSON.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


_DEFAULTS: Dict[str, Any] = {
    "delegation": {
        "auto_inject_user_model": "honcho",
        "auto_inject_facts": "hindsight",
        "writes": ["honcho", "hindsight"],
    },
    "fact_store": {
        "strong_signal_threshold": 3,
        "duplicate_similarity": 0.85,
        "user_explicit_mention_count": 10,
    },
    "forget": {
        # Read-side semantic filter. A recall result is dropped if its
        # token set has Jaccard >= signature_jaccard_min with ANY stored
        # forget signature. Catches the whole family of paraphrases from
        # one forget op, not just the literal canonical_keys.
        "signature_jaccard_min": 0.5,
        # Sigs whose Jaccard with a new sig is >= this get merged
        # instead of inserted, capping table growth.
        "signature_merge_jaccard": 0.7,
        # Hard ceiling on rows; oldest (by last_match_ts) get dropped.
        "max_signatures": 1000,
        # Auto-vacuum sigs that haven't matched anything in this many
        # days (None disables time-based pruning).
        "signature_stale_days": 365,
        # Tombstones in Hindsight are best-effort safety; the signature
        # filter is the real mechanism. Write them async so forget
        # returns instantly.
        "write_tombstones": True,
        "write_tombstones_async": True,
    },
    "anchor_card": {
        "max_tokens": 200,
        "filename": "anchor_card.md",
    },
    "recovery": {
        "enabled": True,
        "cursor_filename": "recovery_cursor.json",
    },
    "import": {
        "default_days": 90,
        "min_turns": 5,
    },
    "tools": {
        "expose": [
            "memory_profile",
            "memory_reasoning",
            "memory_conclude",
            "memory_recall",
            "memory_reflect",
            "memory_forget",
        ],
    },
    "prefetch": {
        "max_total_tokens": 4500,
        "anchor_token_budget": 200,
        "honcho_card_token_budget": 200,
        "hindsight_token_budget": 4096,
        # Hybrid recall-time dedup. Jaccard groups candidates by token
        # overlap (cheap, catches paraphrases). Embedding cosine confirms
        # the merge — if cosine < cosine_min we keep them apart, even when
        # Jaccard says "duplicate". This protects against fact-loss in
        # cases like "Barsik got sick" vs "Barsik died" (high token
        # overlap but different facts).
        "dedup_enabled": True,
        "dedup_jaccard_min": 0.3,
        "dedup_cosine_min": 0.85,
        "dedup_use_embeddings": True,
        "dedup_embedding_timeout": 4.0,
        # Strip paragraphs from assistant_content before handing the turn
        # to Hindsight's LLM extractor when they mostly repeat what we
        # just put into the prompt via prefetch. Breaks the prefetch →
        # extract → retain feedback loop. Disable only for debugging.
        "strip_from_extraction": True,
    },
    # Per-tool dispatch timeouts in seconds. Generous defaults — the
    # underlying providers have their own retry/backoff logic and we don't
    # want to truncate genuine reasoning. Tighten via env if a specific
    # call starts misbehaving.
    "timeouts": {
        "recall":    180,
        "reasoning": 240,
        "reflect":   300,
        "profile":   60,
        "conclude":  60,
        "forget":    180,
        "default":   120,
    },
}


# ---------------------------------------------------------------------------
# Env var → config-path mapping. Env values are strings; we coerce via
# the type of the corresponding default when reading.
# ---------------------------------------------------------------------------
_ENV_MAP: Dict[str, List[str]] = {
    # Timeouts (seconds)
    "MNEMOSYNE_TIMEOUT_RECALL":     ["timeouts", "recall"],
    "MNEMOSYNE_TIMEOUT_REASONING":  ["timeouts", "reasoning"],
    "MNEMOSYNE_TIMEOUT_REFLECT":    ["timeouts", "reflect"],
    "MNEMOSYNE_TIMEOUT_PROFILE":    ["timeouts", "profile"],
    "MNEMOSYNE_TIMEOUT_CONCLUDE":   ["timeouts", "conclude"],
    "MNEMOSYNE_TIMEOUT_FORGET":     ["timeouts", "forget"],
    "MNEMOSYNE_TIMEOUT_DEFAULT":    ["timeouts", "default"],
    # Prefetch budgets (tokens)
    "MNEMOSYNE_PREFETCH_MAX_TOKENS":         ["prefetch", "max_total_tokens"],
    "MNEMOSYNE_PREFETCH_ANCHOR_TOKENS":      ["prefetch", "anchor_token_budget"],
    "MNEMOSYNE_PREFETCH_HONCHO_CARD_TOKENS": ["prefetch", "honcho_card_token_budget"],
    "MNEMOSYNE_PREFETCH_HINDSIGHT_TOKENS":   ["prefetch", "hindsight_token_budget"],
    "MNEMOSYNE_DEDUP_ENABLED":               ["prefetch", "dedup_enabled"],
    "MNEMOSYNE_DEDUP_JACCARD_MIN":           ["prefetch", "dedup_jaccard_min"],
    "MNEMOSYNE_DEDUP_COSINE_MIN":            ["prefetch", "dedup_cosine_min"],
    "MNEMOSYNE_DEDUP_USE_EMBEDDINGS":        ["prefetch", "dedup_use_embeddings"],
    "MNEMOSYNE_DEDUP_EMBED_TIMEOUT":         ["prefetch", "dedup_embedding_timeout"],
    "MNEMOSYNE_STRIP_FROM_EXTRACTION":       ["prefetch", "strip_from_extraction"],
    # Fact store
    "MNEMOSYNE_FACT_STRONG_THRESHOLD":   ["fact_store", "strong_signal_threshold"],
    "MNEMOSYNE_FACT_DUP_SIMILARITY":     ["fact_store", "duplicate_similarity"],
    "MNEMOSYNE_FACT_USER_EXPLICIT":      ["fact_store", "user_explicit_mention_count"],
    # Forget signatures
    "MNEMOSYNE_FORGET_SIG_JACCARD_MIN":  ["forget", "signature_jaccard_min"],
    "MNEMOSYNE_FORGET_SIG_MERGE":        ["forget", "signature_merge_jaccard"],
    "MNEMOSYNE_FORGET_MAX_SIGS":         ["forget", "max_signatures"],
    "MNEMOSYNE_FORGET_SIG_STALE_DAYS":   ["forget", "signature_stale_days"],
    "MNEMOSYNE_FORGET_WRITE_TOMBSTONES": ["forget", "write_tombstones"],
    "MNEMOSYNE_FORGET_TOMBSTONES_ASYNC": ["forget", "write_tombstones_async"],
    # Anchor card
    "MNEMOSYNE_ANCHOR_MAX_TOKENS":  ["anchor_card", "max_tokens"],
    "MNEMOSYNE_ANCHOR_FILENAME":   ["anchor_card", "filename"],
    # Recovery
    "MNEMOSYNE_RECOVERY_ENABLED":   ["recovery", "enabled"],
    # Import
    "MNEMOSYNE_IMPORT_DAYS":       ["import", "default_days"],
    "MNEMOSYNE_IMPORT_MIN_TURNS":  ["import", "min_turns"],
}


def _coerce(value: str, like: Any) -> Any:
    """Best-effort string→type coercion based on the default value's type."""
    if isinstance(like, bool):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    if isinstance(like, int):
        try:
            return int(value)
        except ValueError:
            return like
    if isinstance(like, float):
        try:
            return float(value)
        except ValueError:
            return like
    return value


def _hermes_home() -> Path:
    home = os.environ.get("HERMES_HOME")
    if home:
        return Path(home)
    return Path.home() / ".hermes"


def plugin_dir() -> Path:
    return _hermes_home() / "plugins" / "mnemosyne"


def config_path() -> Path:
    return plugin_dir() / "config.json"


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _set_path(d: Dict[str, Any], path: List[str], value: Any) -> None:
    node = d
    for key in path[:-1]:
        if key not in node or not isinstance(node[key], dict):
            node[key] = {}
        node = node[key]
    node[path[-1]] = value


def _read_path(d: Dict[str, Any], path: List[str]) -> Any:
    node: Any = d
    for key in path:
        if not isinstance(node, dict):
            return None
        node = node.get(key)
    return node


def _apply_env_overrides(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Layer env-var overrides on top of cfg, in place."""
    for env_var, path in _ENV_MAP.items():
        raw = os.environ.get(env_var)
        if raw is None or raw == "":
            continue
        like = _read_path(_DEFAULTS, path)
        _set_path(cfg, path, _coerce(raw, like))
    return cfg


def load() -> Dict[str, Any]:
    """Load config: defaults → config.json overrides → env-var overrides."""
    cfg = json.loads(json.dumps(_DEFAULTS))  # deep copy of defaults
    path = config_path()
    if path.exists():
        try:
            with path.open() as f:
                user_cfg = json.load(f) or {}
            cfg = _deep_merge(cfg, user_cfg)
        except Exception as exc:
            logger.warning("mnemosyne: failed to read %s: %s — using defaults",
                           path, exc)
    return _apply_env_overrides(cfg)


def get(*keys: str, default: Any = None) -> Any:
    """Convenience: cfg.get('fact_store', 'strong_signal_threshold')."""
    node: Any = load()
    for key in keys:
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node
