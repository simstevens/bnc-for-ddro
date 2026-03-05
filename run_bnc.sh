#!/usr/bin/env bash
set -euo pipefail

# Resolve repository root so the script works from any directory.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Allow overriding the Python executable via $PYTHON, default to python3.
PYTHON_BIN="${PYTHON:-python3}"

exec "$PYTHON_BIN" "$ROOT_DIR/src/run_bnc.py" "$@"