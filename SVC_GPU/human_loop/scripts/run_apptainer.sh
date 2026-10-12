#!/usr/bin/env bash
set -euo pipefail
umask 077
[[ -n ${SLURM_JOB_ID:-} ]] || { echo "Run inside a Slurm compute allocation." >&2; exit 2; }
runtime=${HISTOLOOP_CONTAINER_RUNTIME:-apptainer}
command -v "$runtime" >/dev/null || { echo "Load the apptainer module, or set HISTOLOOP_CONTAINER_RUNTIME=singularity." >&2; exit 2; }
image=${HISTOLOOP_IMAGE:?Set HISTOLOOP_IMAGE to the .sif image}
data=${HISTOLOOP_DATA_ROOT:?Set HISTOLOOP_DATA_ROOT to the dataset directory}
state=${HISTOLOOP_STATE_DIR:?Set HISTOLOOP_STATE_DIR to persistent annotation storage}
cache=${HISTOLOOP_CACHE_DIR:-"${SLURM_TMPDIR:?Set HISTOLOOP_CACHE_DIR or SLURM_TMPDIR}/histoloop-$SLURM_JOB_ID"}
[[ -r $image && -d $data && -r $data/datasets.json ]] || { echo "Image, dataset directory, or datasets.json is missing." >&2; exit 2; }
mkdir -p -- "$state" "$cache"
[[ -w $state && -w $cache ]] || { echo "State and cache directories must be writable." >&2; exit 2; }
for path in "$data" "$state" "$cache"; do
  [[ $path != *:* && $path != *,* ]] || { echo "Bind paths cannot contain ':' or ','." >&2; exit 2; }
done
# --cleanenv keeps host Python/Conda settings out; Apptainer uses the invoking UID.
annotation_binds=()
annotation_env=()
if [[ -n ${HISTOLOOP_ANNOTATION_ROOT:-} ]]; then
  [[ -d $HISTOLOOP_ANNOTATION_ROOT && -w $HISTOLOOP_ANNOTATION_ROOT ]] || { echo "Pre-create a writable annotation root." >&2; exit 2; }
  [[ $HISTOLOOP_ANNOTATION_ROOT != *:* && $HISTOLOOP_ANNOTATION_ROOT != *,* ]] || exit 2
  annotation_binds+=(--bind "$HISTOLOOP_ANNOTATION_ROOT:/annotations:rw")
  annotation_env+=(--env HISTOLOOP_ANNOTATION_ROOT=/annotations)
fi
if [[ -n ${HISTOLOOP_CONTOUR_ROOT:-} ]]; then
  [[ -d $HISTOLOOP_CONTOUR_ROOT && $HISTOLOOP_CONTOUR_ROOT != *:* && $HISTOLOOP_CONTOUR_ROOT != *,* ]] || exit 2
  annotation_binds+=(--bind "$HISTOLOOP_CONTOUR_ROOT:/contours:ro")
  annotation_env+=(--env HISTOLOOP_CONTOUR_ROOT=/contours)
fi
exec "$runtime" exec --cleanenv --no-home \
  --bind "$data:/data:ro,$state:/state:rw,$cache:/cache:rw" \
  "${annotation_binds[@]}" "${annotation_env[@]}" \
  --env "SLURM_JOB_ID=$SLURM_JOB_ID,HISTOLOOP_HOST=127.0.0.1,HISTOLOOP_PORT=${HISTOLOOP_PORT:-8765},HISTOLOOP_ORIGINAL_MANIFEST=/data/case_split.xlsx" \
  "$image" /bin/bash /opt/histoloop/SVC_GPU/human_loop/scripts/container_entrypoint.sh "$@"
