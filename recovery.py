"""Plan item 7 — replay missed turns from hermes session transcripts.

Hermes writes every turn synchronously to ~/.hermes/sessions/<id>.jsonl
regardless of memory plugin state. If Hindsight's async writer was killed
mid-flight (process crash, ctrl-C, OS kill), the last N turns may not
have reached Hindsight. On startup, recovery diff's the on-disk
transcripts against a cursor and replays anything missing.

Cursor file: ~/.hermes/plugins/mnemosyne/recovery_cursor.json
{
  "last_filename": "20260505_120134_a3b2.jsonl",
  "last_offset": 12,
  "updated_at": "2026-05-05T18:32:01"
}

`last_offset` is the line number (excluding session_meta) we've already
flushed — anything with index >= last_offset still needs replay.

First-time initialization sets the cursor to "now" without replaying
anything: the user goes from "no Hindsight" to "Hindsight live" without
re-living the entire archive. The bulk importer (item 8) handles the
historical backfill explicitly when invoked.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from . import config
from .fact_store import date_tag

logger = logging.getLogger(__name__)

# 20260505_091839_e8a4f1d2.jsonl
_FILENAME_RE = re.compile(r"^(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})_[0-9a-f]+\.jsonl$")


def _sessions_dir() -> Path:
    return Path.home() / ".hermes" / "sessions"


def _cursor_path() -> Path:
    fn = config.get("recovery", "cursor_filename", default="recovery_cursor.json")
    return config.plugin_dir() / fn


def _filename_to_iso_date(filename: str) -> Optional[str]:
    m = _FILENAME_RE.match(filename)
    if not m:
        return None
    yyyy, mm, dd, *_ = m.groups()
    return f"{yyyy}-{mm}-{dd}"


def _load_cursor() -> Dict[str, Any]:
    path = _cursor_path()
    if not path.exists():
        return {}
    try:
        with path.open() as f:
            return json.load(f) or {}
    except Exception as exc:
        logger.warning("mnemosyne.recovery: cursor read failed: %s", exc)
        return {}


def _save_cursor(cursor: Dict[str, Any]) -> None:
    path = _cursor_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    cursor = dict(cursor)
    cursor["updated_at"] = datetime.utcnow().isoformat()
    try:
        tmp = path.with_suffix(".tmp")
        with tmp.open("w") as f:
            json.dump(cursor, f, ensure_ascii=False, indent=2)
        tmp.replace(path)
    except Exception as exc:
        logger.warning("mnemosyne.recovery: cursor save failed: %s", exc)


def _list_session_files() -> List[Path]:
    d = _sessions_dir()
    if not d.is_dir():
        return []
    return sorted(d.glob("*.jsonl"))


def _iter_turns(jsonl_path: Path) -> Iterator[Tuple[int, Dict[str, Any]]]:
    """Yield (line_index, record) for every non-session_meta line."""
    try:
        with jsonl_path.open() as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                if rec.get("role") == "session_meta":
                    continue
                yield idx, rec
    except Exception as exc:
        logger.debug("mnemosyne.recovery: read %s failed: %s", jsonl_path, exc)


def _pair_user_assistant(records: List[Dict[str, Any]]) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    """Walk records and emit user→assistant pairs.
    Tool calls and other non-user/non-assistant rows are ignored. The next
    assistant after a user becomes that user's pair; orphaned users without
    a following assistant are dropped (incomplete turn)."""
    pairs: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    pending_user: Optional[Dict[str, Any]] = None
    for r in records:
        role = r.get("role")
        if role == "user":
            pending_user = r
        elif role == "assistant" and pending_user is not None:
            pairs.append((pending_user, r))
            pending_user = None
    return pairs


def _retain_pair(
    hindsight_provider: Any,
    user_rec: Dict[str, Any],
    asst_rec: Dict[str, Any],
    *,
    iso_date: str,
    extra_tags: Optional[List[str]] = None,
) -> bool:
    user_text = (user_rec.get("content") or "").strip()
    asst_text = (asst_rec.get("content") or "").strip()
    if not user_text and not asst_text:
        return False

    body = f"User: {user_text}\nAssistant: {asst_text}"
    tags = [f"ts:{iso_date}", "source:transcript"]
    if extra_tags:
        tags.extend(extra_tags)
    try:
        hindsight_provider.handle_tool_call(
            "hindsight_retain",
            {"content": body, "tags": tags},
        )
        return True
    except Exception as exc:
        logger.debug("mnemosyne.recovery: retain failed: %s", exc)
        return False


def initialize_cursor_if_missing() -> bool:
    """First-time setup: stamp cursor at the latest session so we don't
    replay history. Returns True if a fresh cursor was created."""
    if _cursor_path().exists():
        return False
    files = _list_session_files()
    cursor: Dict[str, Any] = {"last_offset": 0}
    if files:
        cursor["last_filename"] = files[-1].name
    _save_cursor(cursor)
    return True


def replay_missed(
    hindsight_provider: Any,
    *,
    max_pairs: int = 50,
) -> Dict[str, Any]:
    """Look for pairs newer than the cursor and resend them to Hindsight.

    Stops after max_pairs to keep startup fast. Subsequent startups pick
    up from the new cursor."""
    if hindsight_provider is None:
        return {"replayed": 0, "skipped": "hindsight unavailable"}

    if not config.get("recovery", "enabled", default=True):
        return {"replayed": 0, "skipped": "disabled in config"}

    cursor = _load_cursor()
    last_filename = cursor.get("last_filename")
    last_offset = int(cursor.get("last_offset") or 0)

    files = _list_session_files()
    if not files:
        return {"replayed": 0, "skipped": "no sessions on disk"}

    # Find the starting point in the files list.
    start_idx = 0
    if last_filename:
        for i, f in enumerate(files):
            if f.name == last_filename:
                start_idx = i
                break
        else:
            # Cursor file no longer exists — start from the latest only.
            start_idx = len(files) - 1

    replayed = 0
    new_cursor = dict(cursor)
    for f in files[start_idx:]:
        iso = _filename_to_iso_date(f.name) or date_tag().split(":", 1)[1]

        records: List[Dict[str, Any]] = []
        max_idx_seen = 0
        for idx, rec in _iter_turns(f):
            records.append(rec)
            max_idx_seen = max(max_idx_seen, idx)

        # If this is the cursor file, skip pairs we've already flushed.
        is_cursor_file = (f.name == last_filename)
        pairs = _pair_user_assistant(records)
        if is_cursor_file and last_offset > 0:
            pairs = pairs[last_offset:]

        for u, a in pairs:
            if replayed >= max_pairs:
                new_cursor["last_filename"] = f.name
                # Best-effort offset: number of pairs we got through.
                new_cursor["last_offset"] = (
                    last_offset + replayed if is_cursor_file else replayed
                )
                _save_cursor(new_cursor)
                return {"replayed": replayed, "stopped_at_limit": True}
            ok = _retain_pair(hindsight_provider, u, a, iso_date=iso,
                              extra_tags=["recovery:true"])
            if ok:
                replayed += 1

        new_cursor["last_filename"] = f.name
        new_cursor["last_offset"] = (
            last_offset + len(pairs) if is_cursor_file else len(pairs)
        )

    _save_cursor(new_cursor)
    return {"replayed": replayed, "stopped_at_limit": False}
