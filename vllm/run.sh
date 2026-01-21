#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -z "$1" ]; then
    echo "Usage: $0 <config-name>"
    echo "  Config is looked up in runs/ folder, e.g.:"
    echo "    $0 gpt-oss-120b-session-1"
    echo "    $0 gpt-oss-120b-session-1.yaml"
    exit 1
fi

CONFIG_FILE="$1"

uv pip install -r "${SCRIPT_DIR}/requirements.txt"
uv run python "${SCRIPT_DIR}/run.py" --config "$CONFIG_FILE"
