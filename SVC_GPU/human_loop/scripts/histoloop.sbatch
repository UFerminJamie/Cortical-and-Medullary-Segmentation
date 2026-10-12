#!/usr/bin/env bash
# Submit with: sbatch --account=YOUR_ACCOUNT --qos=YOUR_QOS scripts/histoloop.sbatch
#SBATCH --job-name=histoloop
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=histoloop-%j.log
set -euo pipefail
script_dir=${HISTOLOOP_SCRIPTS_DIR:?Set HISTOLOOP_SCRIPTS_DIR to the absolute human_loop/scripts directory}
if [[ -n ${HISTOLOOP_IMAGE:-} ]]; then
  module load apptainer
  exec bash "$script_dir/run_apptainer.sh"
else
  exec bash "$script_dir/run_native.sh"
fi
