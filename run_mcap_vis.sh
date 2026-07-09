#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_PATH="${1:-/ssd/data}"

shift || true

exec streamlit run "$SCRIPT_DIR/mcap_vis.py" --server.address 0.0.0.0 --server.port 8501 -- --data-path "$DATA_PATH" "$@"
