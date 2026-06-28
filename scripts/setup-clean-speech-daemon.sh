#!/usr/bin/env bash
# Sets up the bundled clean-speech-daemon (the `clean-speech` submodule) so the agent can
# auto-start it: creates a virtualenv and installs the daemon into it. Run once after cloning.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DAEMON_DIR="$REPO_ROOT/clean-speech"

if [ ! -e "$DAEMON_DIR/pyproject.toml" ]; then
  echo "Submodule not initialized. Run: git submodule update --init clean-speech" >&2
  exit 1
fi

echo "Creating virtualenv in $DAEMON_DIR/.venv ..."
python3 -m venv "$DAEMON_DIR/.venv"
"$DAEMON_DIR/.venv/bin/pip" install --upgrade pip

# Install with the [neural] extras (onnxruntime, soxr, torch) because the daemon's default
# echo canceller is hybrid_localvqe. This pulls torch and is a large download. If you only want
# the lightweight nlms canceller, install without the extras: pip install -e "$DAEMON_DIR"
echo "Installing clean-speech-daemon with neural extras (this downloads torch; may take a while)..."
"$DAEMON_DIR/.venv/bin/pip" install -e "$DAEMON_DIR[neural]"

# Create a default daemon config if the user does not already have one.
CONFIG="$HOME/.config/clean-speech-daemon/config.toml"
if [ ! -e "$CONFIG" ]; then
  echo "Writing default daemon config to $CONFIG ..."
  "$DAEMON_DIR/.venv/bin/clean-speech-daemon" write-config || true
fi

echo
echo "Done. The agent auto-starts the daemon by default (see sttsettings.json Capture settings)."
echo "Default echo canceller is hybrid_localvqe (neural); it falls back to nlms if anything is missing."
