#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ -z "${VIRTUAL_ENV:-}" ]]; then source .venv/bin/activate; fi
export MUJOCO_GL=egl
export PYTHONPATH="$PWD/scripts:$PWD/variants/elevated-pivc256-film/dreamerv3"
python scripts/check_release.py
python scripts/test_pivc256.py
python scripts/test_pi256.py
python scripts/test_elevated_task.py
python scripts/test_vc256.py
python scripts/test_verified_checkpoint.py
