# Annotation workspace validation — 2026-10-12

This feature was validated on the native project host, actual project WSIs and
prediction CSVs, and an isolated Apptainer CPU runtime. Real images and annotation
documents were not added to Git. Existing synthetic demo screenshots remain
explicitly labeled in the README.

## Checks performed

The complete application suite passed: **75 tests**, no skips, 32.61 seconds.
Chromium exercised file layers, colors/visibility, paginated expansion, vertex
coordinates and insertion/removal, drawing, delete/undo/redo, retry after a failed
save, reload persistence, regeneration, all four prediction methods, threshold
changes, background-work cancellation during navigation, keyboard disclosure,
sidebar resizing, remembered preferences, Focus mode, and a 390-pixel viewport.
Python coverage includes migration conflicts/idempotence, schema metadata,
concurrent/external revisions, interrupted saves, partial generation recovery,
holes/multipolygons, missing classes, empty intersections and annotation/correction
separation. Additional targeted tests cover root confinement during discovery.

The shared-math Node check passed order permutations for all methods, majority
ties, threshold changes, fixed manual labels, uncovered cells, tissue gaps and
distinct touching tissue component IDs. Overview and zoom use the same raster;
the browser suite verifies that renderer and stale worker/field-transfer handling.

```bash
python -m pytest SVC_GPU/human_loop/tests -q
node SVC_GPU/human_loop/tests/test_prediction_field.js
```

Browser tests require Playwright and its Chromium installation. The new browser
test also accepts `HISTOLOOP_BROWSER=/absolute/path/to/chromium`. Without browser
dependencies, those tests skip; the reported run included them. The only warning
was the pre-existing Starlette/AnyIO deprecated portal alias.

The pinned CPU Apptainer sandbox and packaged SIF built successfully. Smoke checks
passed authenticated binary views, WSI tiles, tile corrections, working annotation
creation, imported tissue, derived regeneration, and persistence after stopping
and restarting the server with the same state/annotation binds. The current
feature source was bound read-only over the packaged runtime for these checks.
Docker is unavailable on this host; a local Docker build was not claimed. The
existing GitHub container workflow builds and exercises the same expanded smoke
script when CI runs.

## Real production migration

The requested eight source roots and contour root were available. Source GT was
Histomics `annotation.elements`, while matched tissue contours were GeoJSON
FeatureCollections; existing tile export JSON lists are a different format.

The reviewed dry run identified **1,871 sample identities**, including 1,865
manifest entries and six source-only directories. Execution copied **509 GT
files** into the requested production annotation root. A subsequent dry run and
execution copied **zero** and preserved all 509 existing destination files.
No GT files were malformed, no roots were missing, and no annotation source-file
conflicts occurred. Six unmatched source directories, one ambiguous cohort mapping
and one duplicated basename were reported and kept separate. Missing GT files
were not created. Original Labelled_files remain unchanged. Private reports and
source-path/checksum provenance are retained locally outside Git.

## Measured interaction performance

Hardware: AMD EPYC 7702 host, x86-64; allocation one CPU and 4,096 MiB.
Installed Chromium ran headless with CPU/software rendering. The representative
dataset contains 1,069 slides. Physical tile width was 507 level-zero pixels and
overlap 74.951%. Production inputs were read-only; captures used separate state
and working annotation copies. Before/after SHA-256 checks found no change to
source predictions or original GT. No browser errors occurred.

The original filter path measured p50 **40.4 ms**, p95 **58.2 ms** over 12 updates
on a 11,986-tile slide (111,607 × 41,957 pixels). That baseline changed score
visibility, whereas this feature reclassifies an aggregate field; they are not
equivalent workloads. After baseline measurement, the interaction targets were
p95 below 200 ms for cached simple aggregation and 500 ms for bounded CRF, with
an inexpensive mean preview during dragging. The two measured slides met those
targets. These are interaction timings, not a 60 FPS claim.

Feature timings below are p50 / p95 milliseconds from a programmatic threshold
update to accepted worker output and the next animation frame, 20 updates per
method. They include browser scheduling; they exclude first slide loading,
initial field creation and the slider's CRF debounce interval. The median is p50;
p95 uses sorted index `floor(0.95*(n-1))`. Actual slider dragging was separately
exercised in the browser workflow test.

| Method | Real slide 1: 14,773 tiles | Real slide 2: 91,174 tiles |
|---|---:|---:|
| Mean score | 49.2 / 83.2 | 66.6 / 100.8 |
| Majority vote | 100.7 / 226.5 | 51.1 / 107.3 |
| Gaussian score | 42.8 / 64.9 | 45.1 / 71.4 |
| Spatial CRF | 249.3 / 333.4 | 358.0 / 483.5 |

Slide 1 was 69,754 × 56,943 pixels, rendered at 1,517 × 1,238 cells with 46
pixels/cell. Slide 2 was 145,488 × 70,725 pixels, rendered at 1,532 × 745 cells
with 95 pixels/cell. Chromium `performance.memory.usedJSHeapSize` was 43.7 / 21.8
MiB before the method loop and 37.7 / 51.1 MiB afterward. These page-reported metrics
do not establish total worker, canvas/GPU or browser process memory. Server
peak RSS was 766.4 MiB during the real-slide session. Worker temporary allocation
and complete process memory were not separately sampled. Shared-host contention
and software rendering can change tail timings substantially.

## Real screenshots: local only

Four genuine captures were inspected: normal multiple-layer workspace with derived
layers, expanded region list and vertex editor, aggregated heatmap and legend,
and Focus viewer mode. The threshold is visible in each. Identifying slide titles,
queue entries, reviewer information and contour names were replaced in the capture
DOM; private paths and tokens are absent from the visible interface.

The user requested that real screenshots remain local. They are in the ignored
`docs/private-screenshots/` folder of the native checkout with a local index. They
are **not published in this repository or PR**. Public image publication remains
blocked pending clearance; synthetic images are not presented as real captures.

## Concrete limits

- Prediction fields are bounded to 1,536 cells on the longest axis. The same
  resolution is shown at overview and zoom, so fine boundaries can look blocky.
  There is no higher-resolution viewport chunk renderer. Vector annotation
  derivation remains exact at level zero and does not use this raster.
- Tissue masks are sampled at field resolution. Sub-cell gaps/holes may disappear;
  distinct represented vector components remain separate in CRF. Tiny coverage
  regions can be omitted at coarse resolutions. Increase the bounded resolution
  only after measuring memory/latency for the target slides.
- The CRF is an adjustable local axial Potts adaptation using uncalibrated score
  evidence and a bounded thumbnail. It is neither a fully connected DenseCRF nor
  an anatomical boundary detector; no calibrated probabilities are available.
- Supported polygon imports are closed Histomics polylines and GeoJSON
  Polygon/MultiPolygon FeatureCollections. Other element types and tile-export
  lists require explicit conversion; unsupported geometry is never discarded.
- File revision conflicts require explicit reconciliation. Advisory locks protect
  cooperating writers; an external writer ignoring locks can race after the
  final revision check. Staged generation recovery is coordinated by its manifest;
  external readers must inspect that manifest before treating both outputs as current.
- Display PNG/parameter exports create new artifacts. Registering a threshold
  preview as a new inference run is not exposed. Original predictions and training
  snapshots remain independent.
