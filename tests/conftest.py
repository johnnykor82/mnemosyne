"""Test bootstrap for mnemosyne.

Mnemosyne is a Hermes memory provider. Its __init__.py imports
`agent.memory_provider.MemoryProvider` (a Hermes-internal base class) and
expects to be loaded via the Hermes plugin discovery path. For pytest we
stub `agent.memory_provider` *before* importing the plugin and then load
the package directly via importlib so the relative imports inside the
plugin still resolve."""
from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path
from typing import Any, Dict


# 1. Stub agent.memory_provider so `from agent.memory_provider import
#    MemoryProvider` succeeds without pulling the whole Hermes runtime.
class MemoryProvider:  # noqa: D401 — minimal ABC-like stub
    """Stand-in for the real Hermes MemoryProvider base class."""


_agent_pkg = types.ModuleType("agent")
_mp_module = types.ModuleType("agent.memory_provider")
_mp_module.MemoryProvider = MemoryProvider
_agent_pkg.memory_provider = _mp_module
sys.modules.setdefault("agent", _agent_pkg)
sys.modules.setdefault("agent.memory_provider", _mp_module)


# 2. Load mnemosyne/__init__.py under its production package name
#    `_hermes_user_memory.mnemosyne`. The parent package must exist in
#    sys.modules BEFORE we exec the submodule, otherwise its `from . import`
#    statements raise "attempted relative import with no known parent package".
_PLUGIN_DIR = Path(__file__).resolve().parent.parent
_PARENT_PKG = "_hermes_user_memory"
_PKG_NAME = f"{_PARENT_PKG}.mnemosyne"
if _PARENT_PKG not in sys.modules:
    _parent = types.ModuleType(_PARENT_PKG)
    _parent.__path__ = [str(_PLUGIN_DIR.parent)]
    sys.modules[_PARENT_PKG] = _parent
_SPEC = importlib.util.spec_from_file_location(
    _PKG_NAME,
    _PLUGIN_DIR / "__init__.py",
    submodule_search_locations=[str(_PLUGIN_DIR)],
)
mnemosyne = importlib.util.module_from_spec(_SPEC)
sys.modules[_PKG_NAME] = mnemosyne
_SPEC.loader.exec_module(mnemosyne)


# 3. Fake inner providers — they record calls, sleep to emulate the real
#    Hindsight recall pipeline (HTTP → embed → vector → rerank ≈ 6-8s in prod).
class _FakeProvider:
    def __init__(self, name: str, fixed_response: Any, sleep_s: float = 0.0):
        self.name = name
        self._sleep_s = sleep_s
        self._fixed = fixed_response
        self.calls = 0

    def handle_tool_call(self, name: str, args: Dict[str, Any]):
        import time as _t
        self.calls += 1
        if self._sleep_s:
            _t.sleep(self._sleep_s)
        return self._fixed


def make_provider(*, honcho_sleep: float = 0.0, hindsight_sleep: float = 0.0,
                  honcho_card=("alpha", "beta"),
                  hindsight_text: str = "Fact 1"):
    """Construct a MnemosyneMemoryProvider with the inner Honcho/Hindsight
    swapped for `_FakeProvider`s with configurable latency. Bypasses the
    real `_load_inner_providers` (which can't run without Hermes installed)."""
    provider = mnemosyne.MnemosyneMemoryProvider.__new__(mnemosyne.MnemosyneMemoryProvider)
    # Replicate the bits of __init__ we need.
    from concurrent.futures import ThreadPoolExecutor
    import threading as _threading
    provider._honcho = _FakeProvider(
        "honcho", json.dumps({"card": list(honcho_card)}),
        sleep_s=honcho_sleep,
    )
    provider._hindsight = _FakeProvider(
        "hindsight", json.dumps({"result": hindsight_text}),
        sleep_s=hindsight_sleep,
    )
    provider._executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="mnemo-test")
    provider._fact_store = None
    provider._init_lock = _threading.Lock()
    provider._initialized = True
    provider._last_prefetch = ""
    provider._anchor_cache = None
    provider._peer_cache = None
    provider._peer_cache_ttl_s = 60.0
    provider._cache_lock = _threading.Lock()
    return provider
