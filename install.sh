#!/usr/bin/env bash
# Mnemosyne installer.
# Detects the Hermes venv and installs plugin dependencies into it.
# Works on macOS and Linux.

set -euo pipefail

PLUGIN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HERMES_VENV="${HERMES_VENV:-$HOME/.hermes/hermes-agent/venv}"

# Friendly OS label (purely cosmetic — both branches do the same thing).
case "$(uname -s)" in
  Darwin)  OS_LABEL="macOS" ;;
  Linux)   OS_LABEL="Linux" ;;
  *)       OS_LABEL="$(uname -s)" ;;
esac

echo "Mnemosyne installer ($OS_LABEL)"
echo "  Plugin dir: $PLUGIN_DIR"
echo "  Hermes venv: $HERMES_VENV"
echo

if [[ ! -x "$HERMES_VENV/bin/pip" ]]; then
  echo "ERROR: Hermes venv not found at $HERMES_VENV"
  echo "Set HERMES_VENV=/path/to/venv and re-run, or install Hermes Agent first."
  exit 1
fi

echo "Installing plugin dependencies into $HERMES_VENV ..."
"$HERMES_VENV/bin/pip" install -q \
  "honcho-ai" \
  "hindsight-client>=0.4.22"

echo "Verifying installation ..."
"$HERMES_VENV/bin/python" - <<'PY'
import importlib.util, sys
mods = ["honcho", "hindsight"]
missing = [m for m in mods if not importlib.util.find_spec(m)]
if missing:
    print("MISSING:", missing); sys.exit(1)
print("OK — all deps present.")
PY

echo
echo "Mnemosyne installed at: $PLUGIN_DIR"
echo
echo "To activate as the memory provider:"
echo "  hermes config set memory.provider mnemosyne"
echo "  hermes gateway restart"
echo
echo "To roll back to the previous provider (e.g. honcho):"
echo "  hermes config set memory.provider honcho"
echo "  hermes gateway restart"
echo
echo "Verify activation:"
echo "  tail -f ~/.hermes/logs/agent.log | grep -i mnemosyne"
