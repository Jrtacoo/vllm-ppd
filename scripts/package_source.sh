#!/usr/bin/env bash
# Package the complete source tree, including vendored vLLM and local changes.
# Runtime data, model weights, environments, credentials, and git history are not
# part of this source handoff. This script never contacts another host.
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_DIR="${1:-$PROJECT_DIR/artifacts}"
mkdir -p -- "$OUTPUT_DIR"
OUTPUT_DIR="$(cd -- "$OUTPUT_DIR" && pwd)"
case "$OUTPUT_DIR/" in
    "$PROJECT_DIR/artifacts/"*) ;;
    "$PROJECT_DIR/"*) echo "Inside the project, archive output must be under artifacts/." >&2; exit 1 ;;
esac
PROJECT_NAME="$(basename -- "$PROJECT_DIR")"
ARCHIVE_NAME="vllm-ppd-two-node-$(date -u +%Y%m%dT%H%M%SZ).tar.gz"
ARCHIVE_PATH="$OUTPUT_DIR/$ARCHIVE_NAME"

tar --sort=name \
    --exclude="$PROJECT_NAME/.git" \
    --exclude="$PROJECT_NAME/artifacts" \
    --exclude="$PROJECT_NAME/logs" \
    --exclude="$PROJECT_NAME/data" \
    --exclude="$PROJECT_NAME/results" \
    --exclude="$PROJECT_NAME/build" \
    --exclude="$PROJECT_NAME/dist" \
    --exclude="$PROJECT_NAME/env" \
    --exclude="$PROJECT_NAME/venv" \
    --exclude='__pycache__' --exclude='*.pyc' --exclude='.pytest_cache' \
    --exclude='.venv' --exclude='.env' \
    -czf "$ARCHIVE_PATH" -C "$(dirname -- "$PROJECT_DIR")" "$PROJECT_NAME"

tar -tzf "$ARCHIVE_PATH" >/dev/null
(
    cd -- "$OUTPUT_DIR"
    sha256sum "$ARCHIVE_NAME" > "$ARCHIVE_NAME.sha256"
)
printf '%s\n' "$ARCHIVE_PATH" "$ARCHIVE_PATH.sha256"
