"""CLI for mnemosyne — `hermes mnemosyne ...`.

Subcommands:
  status            Show provider availability and config summary
  anchor edit       Open anchor_card.md in $EDITOR
  anchor list       Print anchor_card.md contents
  anchor add TEXT   Append a line to anchor_card.md
  anchor remove TEXT  Remove first line containing TEXT
  import [--days 90] [--min-turns 5]
  forget QUERY [--yes]
  honcho-quiet      Switch all Honcho hosts to recallMode=tools (kill noisy auto-inject)
"""

from __future__ import annotations

# Parent-package bootstrap.
#
# Hermes' CLI discovery loads ``cli.py`` *standalone* (without importing the
# plugin's ``__init__.py``) — see ``plugins/memory/__init__.py:359-371``.
# So when the user runs ``hermes mnemosyne ...`` BEFORE any code path that
# loads the full provider, we have two problems:
#
#   1. Relative imports below (``from . import config`` etc.) need
#      ``_hermes_user_memory.mnemosyne`` registered as a real package.
#   2. If we register it as an empty ``ModuleType`` namespace, a later
#      ``load_memory_provider("mnemosyne")`` call sees a cached, empty
#      module in ``sys.modules`` and returns None — so ``hermes memory
#      status`` reports "NOT installed" even though the plugin is fine.
#
# Fix: load the real ``__init__.py`` here ourselves if it isn't loaded yet.
# That populates ``_hermes_user_memory.mnemosyne`` with the actual
# ``MnemosyneMemoryProvider`` class and ``register()`` function, so the
# subsequent memory provider load reuses it and works.
import importlib.util as _importlib_util
import os as _os
import sys as _sys
import types as _types

_PLUGIN_DIR = _os.path.dirname(_os.path.abspath(__file__))
_PARENT_PKG = "_hermes_user_memory"
_FULL_PKG = f"{_PARENT_PKG}.mnemosyne"

if _PARENT_PKG not in _sys.modules:
    _ns = _types.ModuleType(_PARENT_PKG)
    _ns.__path__ = [_os.path.dirname(_PLUGIN_DIR)]
    _sys.modules[_PARENT_PKG] = _ns

_existing = _sys.modules.get(_FULL_PKG)
# Treat an entry without ``register`` (i.e. an empty namespace ModuleType
# we may have left from a prior load) as "not loaded" and reload it.
if _existing is None or not hasattr(_existing, "register"):
    _spec = _importlib_util.spec_from_file_location(
        _FULL_PKG,
        _os.path.join(_PLUGIN_DIR, "__init__.py"),
        submodule_search_locations=[_PLUGIN_DIR],
    )
    if _spec is not None and _spec.loader is not None:
        _mod = _importlib_util.module_from_spec(_spec)
        _sys.modules[_FULL_PKG] = _mod
        try:
            _spec.loader.exec_module(_mod)
        except Exception:
            # Don't crash CLI dispatch if the plugin's main module fails to
            # initialise (e.g. inner provider unavailable). The CLI commands
            # that don't touch the plugin runtime (e.g. ``status``) still work.
            pass

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

from . import config


def register_cli(subparser) -> None:
    """Build the `hermes mnemosyne` argparse subcommand tree.

    The hermes top-level CLI dispatches via ``args.func(args)``, so we MUST
    call ``subparser.set_defaults(func=mnemosyne_command)`` — otherwise hermes
    silently falls through to the root help when a sub-action is invoked.
    """
    sub = subparser.add_subparsers(dest="action", required=True)

    sub.add_parser("status", help="Show mnemosyne status")

    p_anchor = sub.add_parser("anchor", help="Manage anchor card")
    p_anchor.add_argument("anchor_action",
                          choices=["edit", "list", "add", "remove"])
    p_anchor.add_argument("--text", help="Text for add/remove")

    p_import = sub.add_parser("import", help="Bulk import past sessions to Hindsight")
    p_import.add_argument("--days", type=int, default=None)
    p_import.add_argument("--min-turns", type=int, default=None)

    p_forget = sub.add_parser("forget", help="Mark memories matching a query as forgotten")
    p_forget.add_argument("query")
    p_forget.add_argument("--yes", action="store_true",
                          help="Skip interactive confirmation")
    p_forget.add_argument("--max-items", type=int, default=20)

    sub.add_parser("honcho-quiet",
                   help="Switch all Honcho hosts to recallMode=tools (disable noisy auto-inject)")

    subparser.set_defaults(func=mnemosyne_command)


def mnemosyne_command(args) -> int:
    action = getattr(args, "action", None)
    if action == "status":
        return _cmd_status()
    if action == "anchor":
        return _cmd_anchor(args)
    if action == "import":
        return _cmd_import(args)
    if action == "forget":
        return _cmd_forget(args)
    if action == "honcho-quiet":
        return _cmd_honcho_quiet()
    print(f"Unknown action: {action}", file=sys.stderr)
    return 2


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------

def _cmd_status() -> int:
    print("Mnemosyne plugin status")
    print(f"  Plugin dir: {config.plugin_dir()}")
    print(f"  Config:     {config.config_path()} {'(exists)' if config.config_path().exists() else '(default)'}")

    # Inner provider availability
    try:
        from plugins.memory.honcho import HonchoMemoryProvider
        h = HonchoMemoryProvider()
        print(f"  Honcho:     available={h.is_available()}")
    except Exception as e:
        print(f"  Honcho:     load failed: {e}")

    try:
        from plugins.memory.hindsight import HindsightMemoryProvider
        i = HindsightMemoryProvider()
        print(f"  Hindsight:  available={i.is_available()}")
    except Exception as e:
        print(f"  Hindsight:  load failed: {e}")

    # Local artefacts
    fact_db = config.plugin_dir() / "fact_store.db"
    anchor = _anchor_path()
    rec_cur = config.plugin_dir() / "recovery_cursor.json"
    imp_cur = config.plugin_dir() / "import_cursor.json"
    print(f"  fact_store.db:       {'present' if fact_db.exists() else 'missing'}")
    print(f"  anchor_card.md:      {'present' if anchor.exists() else 'missing'}")
    print(f"  recovery_cursor:     {'present' if rec_cur.exists() else 'missing'}")
    print(f"  import_cursor:       {'present' if imp_cur.exists() else 'missing'}")
    return 0


# ---------------------------------------------------------------------------
# anchor card
# ---------------------------------------------------------------------------

def _anchor_path() -> Path:
    fn = config.get("anchor_card", "filename", default="anchor_card.md")
    return config.plugin_dir() / fn


def _ensure_anchor() -> Path:
    p = _anchor_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    if not p.exists():
        p.write_text(
            "# Anchor card — pinned facts about the user\n"
            "#\n"
            "# Each non-comment line is one fact. Keep it short — the\n"
            "# whole file is injected into every prefetch (budget ~200 tokens).\n"
            "\n",
            encoding="utf-8",
        )
    return p


def _cmd_anchor(args) -> int:
    sub = args.anchor_action
    p = _ensure_anchor()
    if sub == "edit":
        editor = os.environ.get("EDITOR") or os.environ.get("VISUAL") or "nano"
        return subprocess.call([editor, str(p)])
    if sub == "list":
        sys.stdout.write(p.read_text(encoding="utf-8"))
        return 0
    if sub == "add":
        text = (args.text or "").strip()
        if not text:
            print("--text required for add", file=sys.stderr)
            return 2
        content = p.read_text(encoding="utf-8")
        if not content.endswith("\n"):
            content += "\n"
        content += text + "\n"
        p.write_text(content, encoding="utf-8")
        print(f"Added: {text}")
        return 0
    if sub == "remove":
        text = (args.text or "").strip()
        if not text:
            print("--text required for remove", file=sys.stderr)
            return 2
        lines = p.read_text(encoding="utf-8").splitlines()
        kept = []
        removed = False
        for line in lines:
            if not removed and text in line and not line.lstrip().startswith("#"):
                removed = True
                continue
            kept.append(line)
        if not removed:
            print(f"No line containing: {text}", file=sys.stderr)
            return 1
        p.write_text("\n".join(kept) + "\n", encoding="utf-8")
        print(f"Removed first line containing: {text}")
        return 0
    print(f"Unknown anchor action: {sub}", file=sys.stderr)
    return 2


# ---------------------------------------------------------------------------
# import
# ---------------------------------------------------------------------------

def _cmd_import(args) -> int:
    from .importer import run_import
    provider = _make_hindsight()
    if provider is None:
        print("Hindsight is not available — install via `hermes memory setup`.",
              file=sys.stderr)
        return 1

    def progress(stage, info):
        if stage == "start":
            print(f"  → import {info['files']} files, {info['pairs']} turn pairs "
                  f"(last {info['days']} days, min {info['min_turns']} turns)")
        elif stage == "checkpoint":
            print(f"    {info['file']}: {info['pairs_done']}/{info['pairs_total']} "
                  f"(total imported: {info['imported']})")
        elif stage == "file_done":
            print(f"    ✓ {info['file']}  files {info['files_done']}/{info['files_total']}")
        elif stage == "interrupted":
            print(f"  Interrupted — imported {info['imported']} pairs so far.")

    print("Mnemosyne bulk import → Hindsight")
    result = run_import(
        provider,
        days=args.days,
        min_turns=args.min_turns,
        on_progress=progress,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))

    try:
        provider.shutdown()
    except Exception:
        pass
    return 0


# ---------------------------------------------------------------------------
# forget
# ---------------------------------------------------------------------------

def _cmd_forget(args) -> int:
    from .fact_store import FactStore
    from .forget import forget_by_query

    provider = _make_hindsight()
    if provider is None:
        print("Hindsight is not available.", file=sys.stderr)
        return 1
    fs = FactStore()

    confirm = None
    if not args.yes:
        confirm = _interactive_confirm
    result = forget_by_query(
        fs, provider, args.query,
        confirm=confirm,
        max_items=args.max_items,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))

    try:
        provider.shutdown()
    except Exception:
        pass
    return 0


def _interactive_confirm(candidates):
    print(f"\nFound {len(candidates)} candidate(s) to forget:")
    for i, c in enumerate(candidates, 1):
        text = c if isinstance(c, str) else (c.get("text") or c.get("content") or str(c))
        preview = text.strip().replace("\n", " ")
        if len(preview) > 200:
            preview = preview[:200] + "…"
        print(f"  {i}. {preview}")
    print()
    choice = input("Forget all? [y/N/<comma-separated indices>] ").strip().lower()
    if choice in ("y", "yes"):
        return candidates
    if choice in ("", "n", "no"):
        return []
    chosen = []
    for tok in choice.split(","):
        tok = tok.strip()
        if tok.isdigit():
            idx = int(tok) - 1
            if 0 <= idx < len(candidates):
                chosen.append(candidates[idx])
    return chosen


# ---------------------------------------------------------------------------
# honcho-quiet
# ---------------------------------------------------------------------------

def _cmd_honcho_quiet() -> int:
    """Set recallMode: tools on every Honcho host. Kills noisy auto-inject."""
    honcho_path = Path.home() / ".hermes" / "honcho.json"
    if not honcho_path.exists():
        print(f"Honcho config not found at {honcho_path}", file=sys.stderr)
        return 1
    try:
        original_text = honcho_path.read_text(encoding="utf-8")
        cfg = json.loads(original_text)
    except Exception as exc:
        print(f"Failed to read {honcho_path}: {exc}", file=sys.stderr)
        return 1

    changed = 0
    for host_key, host_block in (cfg.get("hosts") or {}).items():
        if not isinstance(host_block, dict):
            continue
        if host_block.get("recallMode") != "tools":
            host_block["recallMode"] = "tools"
            changed += 1

    if changed == 0:
        print("All Honcho hosts already use recallMode: tools.")
        return 0

    backup = honcho_path.with_suffix(".json.before-honcho-quiet")
    backup.write_text(original_text, encoding="utf-8")
    honcho_path.write_text(
        json.dumps(cfg, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Updated {changed} host(s) in {honcho_path} → recallMode: tools")
    print(f"Backup of original: {backup}")
    return 0


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _make_hindsight():
    try:
        from plugins.memory.hindsight import HindsightMemoryProvider
    except Exception as exc:
        print(f"Hindsight plugin import failed: {exc}", file=sys.stderr)
        return None
    p = HindsightMemoryProvider()
    if not p.is_available():
        print("Hindsight reports is_available()=False. "
              "Run `hermes memory setup` and pick hindsight.", file=sys.stderr)
        return None
    try:
        from hermes_constants import get_hermes_home
        hermes_home = str(get_hermes_home())
    except Exception:
        hermes_home = str(Path.home() / ".hermes")
    p.initialize(session_id="cli-mnemosyne", hermes_home=hermes_home, platform="cli")
    return p
