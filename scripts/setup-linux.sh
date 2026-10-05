#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
sudo apt-get update
sudo apt-get install -y python3-venv python3-dev build-essential git libegl1 libgl1 libgles2 libglfw3 libosmesa6
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-resolved.txt
python -m pip check
echo 'Run bash scripts/check.sh, then a smoke run before full training.'
