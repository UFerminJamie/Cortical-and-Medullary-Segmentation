# Annotation workspace update

The 2026-10-12 feature validation, real production migration and local-only real
screenshots are recorded in [workspace-validation.md](workspace-validation.md).
The earlier demo-only checks below describe the original release.

# HistoLoop validation

Validation used isolated state and synthetic inputs on a HiPerGator CPU compute
allocation. Production annotations and source research data were not modified.

| Check | Result |
|---|---|
| Synthetic application and model-pipeline tests | **64 passed**, including browser tests |
| Fresh native environment | Hashed Python 3.10 CPU dependencies installed successfully |
| Native startup and restart | Authenticated bootstrap, four demo slides, JPEG tiles, binary overlays, annotation writes and restart persistence passed |
| Actual browser workflows | Branding, dataset/split selection, sample navigation, WSI loading, overlays, rectangle autosave, history, reload persistence and responsive layouts passed; no page errors or failed requests |
| Screenshot generation | Six actual 1600 × 1000 captures using synthetic data, including the score histogram |
| Apptainer | Final clean-checkout SIF built successfully; authenticated startup, read-only data bind, JPEGs, binary overlays, annotation save, graceful shutdown and restart persistence passed |
| Docker Compose and launch scripts | Compose configuration, Bash syntax and JavaScript syntax validated |
| Docker build/run | **Passed on GitHub Actions**: image built, generated demo data loaded, authenticated API/JPEG/overlay checks passed, annotations persisted after container restart |
| GPU execution | **Not rerun** in this CPU allocation; existing worker tests simulate compute and containers disable GPU jobs |
| Singularity executable | Launcher supports it; runtime validation used Apptainer 1.5.3 |

The broader export test initially failed because it expected filename extensions
in annotation layer names. Its stale expectation was corrected to the existing
export format; exporter behavior is unchanged. The final suite has no failures.
Warnings concern dependency deprecations and a single-class metric fixture.

The clean publishing checkout also passed all 64 tests and native runtime checks
without installation-specific settings. An initial concurrent startup check
timed out during image compression in the one-CPU/4-GiB allocation; its sequential
rerun passed. SIF compression is limited to one CPU and 256 MiB in the documented
build command.

The GitHub workflow `.github/workflows/histoloop.yaml` builds the Dockerfile,
generates synthetic data inside the image, checks authenticated APIs and tiles,
saves corrections to a host mount, and verifies persistence after restart.
The [successful Docker run](https://github.com/UFerminJamie/Cortical-and-Medullary-Segmentation/actions/runs/38275261000)
validated commit `22d02d9`. Local Docker is unavailable on this node; its Podman
alternative could not extract the base image because subordinate UID/GID mappings
are not configured. Docker success above refers to the executed GitHub runner.

Existing navigation/cache regressions cover request cancellation, ETags, byte
bounds, worker fallback, selected-grid correction scope, unchanged prediction
rendering during drawing, and cache reuse. A three-sample backend navigation
benchmark was also rerun on existing inputs with isolated state. This is a smoke
measurement, not a statistically supported before/after performance claim; raw
research-identifying reports remain outside Git.

To reproduce synthetic checks from the parent of `SVC_GPU`, after portable setup:

```bash
SVC_GPU/human_loop/.venv-portable/bin/python -m pip install -r SVC_GPU/human_loop/requirements-test.txt
SVC_GPU/human_loop/.venv-portable/bin/python -m playwright install chromium
SVC_GPU/human_loop/.venv-portable/bin/python -m pytest SVC_GPU/human_loop/tests SVC_GPU/tests -q
```

Generate demo data and screenshots from `human_loop`:

```bash
.venv-portable/bin/python scripts/generate_demo.py --output "$PWD/demo"
HISTOLOOP_PYTHON="$PWD/.venv-portable/bin/python" \
  .venv-portable/bin/python scripts/capture_screenshots.py \
  --data "$PWD/demo" --state "$PWD/demo-screenshot-state"
```

On HiPerGator, use a compute allocation for server and browser checks. Chromium
also needs its OS runtime libraries. Use a fresh screenshot state directory;
the capture utility refuses existing state to protect prior annotations.
