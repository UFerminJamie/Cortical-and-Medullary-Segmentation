#!/usr/bin/env bash
set -euo pipefail
export HISTOLOOP_DATA_ROOT=${HISTOLOOP_DATA_ROOT:-/data}
export HISTOLOOP_DATASETS_FILE=${HISTOLOOP_DATASETS_FILE:-"$HISTOLOOP_DATA_ROOT/datasets.json"}
export HISTOLOOP_STATE_DIR=${HISTOLOOP_STATE_DIR:-/state}
export HISTOLOOP_CACHE_DIR=${HISTOLOOP_CACHE_DIR:-/cache}
export HISTOLOOP_DEVICE=cpu HISTOLOOP_ENABLE_COMPUTE=0
export HISTOLOOP_PYTHON=/usr/local/bin/python
exec /bin/bash /opt/histoloop/SVC_GPU/human_loop/scripts/run_native.sh "$@"
