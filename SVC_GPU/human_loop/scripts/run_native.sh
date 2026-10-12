#!/usr/bin/env bash
set -euo pipefail
umask 077
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
app_dir=$(cd -- "$script_dir/.." && pwd)
project_root=$(cd -- "$app_dir/../.." && pwd)
python=${HISTOLOOP_PYTHON:-"${HISTOLOOP_VENV:-$app_dir/.venv-portable}/bin/python"}
if [[ -d /blue && -d /apps && -z ${SLURM_JOB_ID:-} ]]; then
  echo "HiPerGator: request a compute allocation before running HistoLoop." >&2; exit 2
fi
[[ -x $python ]] || { echo "Python not executable: $python. Run setup_native.sh or set HISTOLOOP_PYTHON." >&2; exit 2; }
export HISTOLOOP_DATASETS_FILE=${HISTOLOOP_DATASETS_FILE:-"${HISTOLOOP_DATA_ROOT:?Set HISTOLOOP_DATA_ROOT or HISTOLOOP_DATASETS_FILE}/datasets.json"}
[[ -r $HISTOLOOP_DATASETS_FILE ]] || { echo "Dataset configuration not readable: $HISTOLOOP_DATASETS_FILE" >&2; exit 2; }
export HISTOLOOP_STATE_DIR=${HISTOLOOP_STATE_DIR:-"$app_dir/state"}
export HISTOLOOP_CACHE_DIR=${HISTOLOOP_CACHE_DIR:-"${SLURM_TMPDIR:-${TMPDIR:-/tmp}}/histoloop-${SLURM_JOB_ID:-local}-$(id -u)"}
mkdir -p -- "$HISTOLOOP_STATE_DIR" "$HISTOLOOP_CACHE_DIR"
[[ -w $HISTOLOOP_STATE_DIR && -w $HISTOLOOP_CACHE_DIR ]] || { echo "State and cache directories must be writable." >&2; exit 2; }
export HISTOLOOP_DEVICE=${HISTOLOOP_DEVICE:-cpu}
export HISTOLOOP_ENABLE_COMPUTE=${HISTOLOOP_ENABLE_COMPUTE:-0}
export PYTHONPATH="$project_root${PYTHONPATH:+:$PYTHONPATH}"
exec "$python" -m SVC_GPU.human_loop.server "$@"
