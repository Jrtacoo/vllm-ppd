#!/usr/bin/env bash
# Start only the local dcn10 consumer, in the foreground. No cleanup or SSH.
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ -n "${PYTHON_BIN:-}" ]]; then
    DCN10_PYTHON="$PYTHON_BIN"
elif command -v python3 >/dev/null 2>&1; then
    DCN10_PYTHON="$(command -v python3)"
elif command -v python >/dev/null 2>&1; then
    DCN10_PYTHON="$(command -v python)"
else
    echo "Python not found. Activate your environment or set PYTHON_BIN to its executable." >&2
    exit 1
fi

exec "$DCN10_PYTHON" "$PROJECT_DIR/ppd/dcn10.py" "$@"
