#!/usr/bin/env sh
# SoloCropper
# Copyright (c) 2026 Solo
# Original work by Solo | https://sololo.xyz
set -eu

cd "$(dirname "$0")"

echo "[1/2] Activating virtual environment..."
if [ ! -f "./venv/bin/activate" ]; then
    echo "[Error] Could not find ./venv/bin/activate in the current directory."
    echo "On Linux/macOS, create a local virtual environment for this system first."
    echo "Example:"
    echo "  python3 -m venv venv"
    echo "  . ./venv/bin/activate"
    echo "  pip install -r requirements.txt"
    exit 1
fi

# shellcheck disable=SC1091
. "./venv/bin/activate"

echo "[2/2] Running SoloCropper.py..."
python SoloCropper.py
