# Working annotations and prediction review

HistoLoop keeps three independent kinds of state: polygon annotation working files,
tile-label correction events, and immutable model predictions. Editing a polygon
does not create training corrections. Imports do not enter training snapshots.

## Migration and identity

Run from the repository root with the CPU runtime installed. Set the dataset
registry to the workbooks used by the viewer. Migration discovers every manifest
slide and every immediate sample directory under the configured source roots.

```bash
export HISTOLOOP_DATASETS_FILE=/path/to/datasets.json
export HISTOLOOP_STATE_DIR=/path/to/review-state
export HISTOLOOP_ANNOTATION_ROOT=/orange/pinaki.sarder/j.fermin/cort_med_wsi/trident_processed/annotations
export HISTOLOOP_CONTOUR_ROOT=/orange/pinaki.sarder/j.fermin/cort_med_wsi/trident_processed/contours_geojson
python -m SVC_GPU.human_loop.migrate_annotations --native-sources --report /tmp/annotation-plan.json
# Review the JSON report before execution; it contains private identities and paths.
python -m SVC_GPU.human_loop.migrate_annotations --report /tmp/annotation-plan.json --execute
python -m SVC_GPU.human_loop.server --device cpu --disable-compute
```

`--native-sources` selects the three `/blue/.../Cort_Med_Project/KPMP_{barisoni,pathologist3,hodgin}/Labelled_files`
roots and five `/orange/.../cort_med_wsi/KPMP_batch{2,3,4,5,6}/Labelled_files`
roots specified for this project. Other installations set
`HISTOLOOP_ANNOTATION_SOURCES` to a JSON array of directories instead. Config CLI
flags include `--annotation-root`, `--contour-root`, and `--annotation-sources`.
Original roots are only read. Tissue QC files are imported explicitly; migration
copies only the two requested ground-truth classes.

```text
annotations/
  .samples.json                         # manifest/slide identity registry
  .history/                             # backups, locks, provenance, staged generations
  <sample-name>/
    cortical_gt.json                     # editable; missing stays missing
    medullary_gt.json                    # editable; empty stays present
    cortical.json                       # derived, read-only
    medullary.json                       # derived, read-only
    Tissue_qc.json                       # optional explicitly imported tissue mask
    <imported-layer>.json                # optional; not a training input
    .generation.json                    # publication/status manifest
    .tissue.json                        # explicit tissue selection
```

Sample directory names may contain the semicolons and parentheses present in the
real manifests. Names are treated as filesystem data, never shell commands.
Manifest names are matched with the source cohort in the slide path. A basename
alone cannot establish cohort identity. Colliding names get a stable identity
suffix; unresolvable sources become separate source-only identities and appear in
the report. They are not silently attached to a different cohort's slide.
Unregistered nonempty destination folders require identity resolution before
adoption. Existing working files are never overwritten by a migration rescan.

`cortex_gt.json` is accepted as an import alias for canonical `cortical_gt.json`.
Both names in one source folder constitute a conflict, even when byte-identical.
Multiple candidate source files are reported and skipped. Dry-run reports list
missing roots, malformed files, unmatched directories, ambiguous matches, duplicate
names, proposed copies and preserved destinations. Execution rechecks source
hashes and publishes byte-identical copies exclusively, with fsync. Provenance
records preserve source paths and SHA-256 hashes outside layer discovery.

Production verification on 2026-10-12 identified **1,871 identities**, copied
**509 ground-truth files**, and preserved all 509 on rescan with **zero copies**.
All eight roots were available. There were six unmatched source directories and
one ambiguous cohort match; these remain separate identities. No malformed GT
files, missing roots, or conflicting GT sources were found. Private reports are
local artifacts, not repository contents.

## Schemas, editing and recovery

Actual ground truth on the host uses Histomics JSON wrappers with
`annotation.elements`, closed `polyline` elements, and `[x,y,z]` points. Bare
`elements` documents are also supported. Actual TRIDENT contours use GeoJSON
`FeatureCollection` files named `<sample>.geojson`, containing Polygon features.
Polygon and MultiPolygon features, including holes, are supported. Coordinates
are interpreted as level-zero slide pixels from these producers, rather than
longitude/latitude inferred from a filename or a producer's CRS label.

Unknown schemas, open polylines, nonpolygon features and invalid geometries are
rejected with an explanation. They are never silently dropped. Existing metadata,
class properties, wrappers, and untouched regions remain intact. Display colors,
opacity and visibility live only in reviewer preferences. Existing IDs are kept;
regions without IDs receive persistent `_histoloop_id` values on the first edit.
IDs before that edit are deterministically derived from the unchanged source.

Choose **Draw into → File: ...** for polygon work, or **Tile corrections** for the
existing training-label tools. Create missing GT files explicitly. File drawing
adds a draft; **Save file** persists it to that exact file. Expand a file to see up
to 30 regions per page. Region actions provide selection, zoom, visibility, vertex
editing and deletion. The vertex editor exposes every ring/component, optional
name, numeric coordinates, insertion/removal and pointer dragging. Apply geometry,
then save the file. Draft undo/redo covers supported edits and deletions. After a
save, **Saved versions / restore** can recover earlier file contents. Restoring
creates another recoverable version with a concurrency check.

Histomics polylines cannot represent holes or multipolygons within one element.
An edit requiring those features is refused; export and explicitly convert to a
GeoJSON layer before that edit. Existing hole/multipolygon GeoJSON is edited
without flattening other rings or components. No automatic geometry repair runs.

Imports require an explicit destination slide and layer. They can create a new
named layer or fill a missing file, and cannot overwrite an existing layer through
the import dialog. Source schema metadata is retained, so treat downloads and
backups as sensitive even if the UI displays a neutral name.

Each file uses a SHA-256 content revision. Saves require the loaded revision,
take a process/thread and shared-filesystem advisory lock, create a recoverable
backup, and use a temporary file with flush/fsync and atomic replacement. A second
revision check catches external changes during preparation. Cooperating external
editors should use the same advisory lock; an uncooperative writer racing after
the final check cannot be made transactional by a filesystem API.

Navigation and reload warn/block while file drafts or vertex edits exist. Failed
saves retain drafts, show **Retry save**, and offer a schema-preserving **Download
draft**. A revision conflict requires explicit reconciliation with the changed
file; retry does not bypass revision protection. Backups, locks and generation
staging never appear as annotation layers. Back up both the working root and the
existing event/state directory. Stale temporary files are never layer inputs.

## Exact derived geometry

For unions `C` (cortical GT), `M` (medullary GT) and `T` (selected tissue), GEOS via
Shapely computes:

```text
medullary.json = M ∩ T
cortical.json  = (C \ M) ∩ T
```

Consequently, interiors do not overlap and the union equals `(C ∪ M) ∩ T`.
Medulla priority applies only to these annotation outputs, never to model tile
overlap rendering. Missing cortex produces only medulla; missing medulla produces
only cortex. An existing empty source produces an existing empty derived file.
Removed source classes invalidate their output and remove it recoverably when a
valid generation can publish. Missing/invalid tissue leaves sources intact and
marks derivation unavailable; prior outputs are stale, not current results.

Exact-name contour candidates and imported `Tissue_qc.json` are offered for
selection. Multiple masks or ambiguous sample basenames require an explicit choice.
Source saves schedule background regeneration; **Regenerate** also runs it
manually. Dirty/pending derived rows are marked stale. The manifest records both
source hashes, tissue identity/hash, algorithm, generation ID and time. Both
outputs use one captured input snapshot. Staged outputs and a pending publication
manifest recover partial publication on the next layer request. Consumers reading
the files outside HistoLoop must also check `.generation.json` is current.

## Prediction display and threshold

The default is an **aggregated class map with mean score**, replacing the
order-dependent rectangle overpainting used in both previous overlay paths.
Overview and zoom display the same cached raster with nearest-neighbor sampling.
Raw label inspection uses outlines; raw score inspection uses transparent fills
with inspectable contributions. Optional tile boundaries and hover IDs/scores
preserve access to the original observations. Nothing paints medulla last as an
overlap fix.

The four methods are:

| Method | Rule at a covered raster cell |
|---|---|
| Majority vote | Threshold every covering tile. More votes wins; ties compare mean score with the same threshold. |
| Mean score | Average covering decision scores, then threshold. |
| Gaussian score | `sum(w*score)/sum(w)`, with Gaussian weights centered on physical tile footprints; sigma is 0.05–0.75 tile widths, default 0.25. |
| Spatial CRF | Bounded local binary Potts mean-field refinement of the mean field, using score-based unary evidence and spatial/RGB similarity. Strength 0–5, radius 1–3 cells, iterations 1–12. |

The CRF uses uncalibrated unary evidence `(mean_score-threshold)/scale`, with
`scale=max(0.1, score_span/6)`, mapped through a sigmoid for its internal update.
That mapping is **not calibrated confidence**. The graph uses axial neighbors;
it is a practical local adaptation, not the fully connected algorithm in the
reference paper. Thumbnail RGB similarity is optional while its bounded image
guide loads. Color edges are not anatomical truth. CRF never fills uncovered cells
or intentionally links separate vector tissue components/background gaps. No
morphological cleanup removes small components.

The Cividis-like blue/yellow score ramp is fixed to the selected run's full score
range. It does not rescale during panning or change classification. SVM margins
are not probabilities. Correlated tile vote fractions/dispersion are not
independent-sample statistical confidence. Ground truth remains independent.

The live slider/numeric input classifies **score ≥ threshold as medulla**, as
confirmed in `gpu_model.predict_gpu` and the host model's encoded class mapping
(`0=cortical`, `1=medullary`). **Reset run threshold** restores the selected saved
run's original threshold. Majority thresholds tiles before voting; mean/Gaussian
threshold the aggregate; CRF updates its evidence. CRF dragging immediately uses
the mean preview, then debounces refinement by 220 ms; pending/stale responses are
marked/discarded. Existing human labels stay fixed. Score visibility filters remain
separate. Tile-correction eligibility can use the preview threshold while its
audit record retains original prediction labels and both thresholds.

Aggregate counts are sampled area in level-zero pixels squared (`≈ M px²`);
raw counts and histograms describe tiles. Uncovered/outside-mask cells are
transparent. Score heatmaps retain aggregate SVM margins even when CRF refines the
class map; they do not display a fabricated calibrated probability field.

Threshold changes are preview-only. No inference runs or prediction fetches occur
per slider movement. Baseline CSVs, GT and tile correction events remain unchanged.
Export overlay PNG and parameters JSON use new timestamped filenames with
prediction identity, fingerprint, tissue revision, method/parameters, threshold,
resolution and pending-refinement flag. These display exports are not registered
as training/prediction runs. There is no "replace baseline" or save-preview-run
action.

Primary references: [MONAI overlap blending](https://monai.readthedocs.io/en/stable/inferers.html),
[scikit-learn calibration](https://scikit-learn.org/stable/modules/calibration.html),
[Krähenbühl & Koltun CRF inference](https://www.philkr.net/2011/kraehenbuehl2011efficient/).
No MONAI/dense-CRF framework is imported for these scalar tile fields.

## Workspace and deployment

Sample filters, annotations, prediction filters, overlap/refinement, histogram,
history and compute are independent accessible disclosures in `panels.js`.
Annotations expand initially; advanced controls start collapsed. Panels do not
change layer eye states. Sidebars reclaim their width when collapsed; top restore
buttons remain reachable. Bounded pointer/keyboard resizing and Focus viewer keep
the original OSD/canvas lifecycle. Reviewer preferences restore expansion,
visibility, widths, file display settings and method controls, with viewport
clamping. Narrow screens use independent drawers. Collapsed lists/charts defer
rendering and update on reopen. File tools, navigation, save indicators and the
live threshold remain reachable in Focus mode. Disabled compute has no large empty
panel.

| Data | Native location | Container bind |
|---|---|---|
| WSIs, PT, XLSX, prediction CSVs | Dataset registry/manifests | `/data:ro` or binds preserving native absolute paths |
| Correction events/preferences/jobs/token | `HISTOLOOP_STATE_DIR` | `/state:rw` |
| Working polygons/history/provenance | `HISTOLOOP_ANNOTATION_ROOT` | `/annotations:rw` |
| Tissue contours | `HISTOLOOP_CONTOUR_ROOT` | `/contours:ro` |
| SQLite/NumPy/JPEG caches | Node-local cache | `/cache:rw` |
| Original Labelled_files, for migration | Source-root JSON array | Bind read-only at the paths named in the source array |

Fresh native instances default to `<state>/annotations` if no working root is
set. Do not point the working root at originals. Pre-create writable state and
annotation directories. Missing contour masks are supported; an empty contour
directory can be mounted. Compose now requires annotation/contour directories in
`.env`. The Apptainer launcher binds both when their environment settings are
present; without them it retains `/state/annotations` and no contour root.

Add to the existing Docker command:

```bash
--mount "type=bind,src=$HISTOLOOP_ANNOTATION_ROOT,dst=/annotations" \
--mount "type=bind,src=$HISTOLOOP_CONTOUR_ROOT,dst=/contours,readonly" \
-e HISTOLOOP_ANNOTATION_ROOT=/annotations -e HISTOLOOP_CONTOUR_ROOT=/contours
```

For native-manifest absolute paths on this host, also bind the referenced `/blue`
and `/orange` dataset locations read-only at identical container paths, with a
more specific writable bind for the annotation destination. Do not rewrite
slide/feature identities just to run a container against existing review state.

## Validation, performance and limits

See [workspace validation](workspace-validation.md) for executed tests and timings.
The prediction field is bounded to at most 1,536 cells on its longest axis. The
actual step is `max(requested_step, ceil(max_slide_dimension/1536))`; there is no
full-resolution WSI raster allocation. Coverage/score fields survive threshold
updates. Typed arrays, one worker, a coalesced next request, generation IDs and
requestAnimationFrame retain cancellation and bounded processing. Gaussian/mean
fields reuse scores; majority reruns voting; refinement remains asynchronous.

The finite display raster can omit details narrower than its sampling step,
including small holes/gaps/components. There is no morphological deletion, but
sampling is not vector-accurate. Saved GT/derived geometries remain exact vectors.
There is no viewport-adaptive finer raster or calibrated probability input path.
CRF neighbors cannot cross separate vector component IDs or uncovered cells;
subcell background gaps remain a display-resolution limitation. The main/worker
math is shared; a workerless fallback uses the same aggregate math, with mean
preview rather than image-guided CRF.

Real screenshots were captured locally from real WSIs and saved predictions,
with slide identifiers, reviewer data and private paths hidden. They show normal
multi-layer review, expanded region editing, heatmap/threshold and Focus viewer.
**Publication is blocked: the project owner requested local-only screenshots.**
No real tissue captures or patient annotations are committed. Local installation
artifacts live in the ignored `docs/private-screenshots/` directory; the older
public screenshots in README remain explicitly synthetic.
