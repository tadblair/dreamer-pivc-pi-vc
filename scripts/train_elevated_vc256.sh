#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ -z "${VIRTUAL_ENV:-}" ]]; then source .venv/bin/activate; fi
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 LP_NUM_THREADS=1 MALLOC_ARENA_MAX=2
export MUJOCO_GL=egl XLA_PYTHON_CLIENT_MEM_FRACTION=.80
export JAX_COMPILATION_CACHE_DIR="$PWD/.cache/jax-elevated-vc256-film"
export PYTHONPATH="$PWD/scripts:$PWD/variants/elevated-vc256-film/dreamerv3"
mode="${1:-smoke}"
if [[ $# -gt 0 ]]; then shift; fi
case "$mode" in
 smoke) exec python variants/elevated-vc256-film/dreamerv3/dreamerv3/main.py --configs elevated_track size50m --run.envs 2 --batch_size 4 --run.steps 512 --run.train_ratio 8 --run.save_every 60 --run.report_every 60 --logdir "$PWD/runs/elevated-vc256-film-smoke" "$@" ;;
 train) exec python variants/elevated-vc256-film/dreamerv3/dreamerv3/main.py --configs elevated_track size50m --run.steps inf --batch_size 4 --batch_length 256 --logdir "$PWD/runs/elevated-track-vc256-seed0" "$@" ;;
 *) echo 'Usage: train_elevated_film.sh {smoke|train}' >&2; exit 2 ;;
esac
