#!/usr/bin/env bash
# One foreground local process. No cleanup, remote launch, or global pkill.
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
DCN9_PYTHON="${PYTHON_BIN:-}"
if [[ -z "$DCN9_PYTHON" ]]; then
    DCN9_PYTHON="$(command -v python3 || command -v python || true)"
fi
if [[ -z "$DCN9_PYTHON" ]]; then
    echo 'Python not found. Activate your environment or set PYTHON_BIN.' >&2
    exit 1
fi
exec "$DCN9_PYTHON" "$PROJECT_DIR/ppd/dcn9.py" "$@"
