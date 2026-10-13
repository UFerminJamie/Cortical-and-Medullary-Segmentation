# Train Model and viewer controls

## Objective

Add a focused training workflow to HistoLoop: select disjoint training and validation
samples, freeze the saved predictions and human tile corrections, submit a GPU SVC
regularization search through HiPerGator Slurm, compare validation performance, and
choose a candidate for subsequent prediction. Add configurable heatmap colors and
explicit polygon-annotation visibility controls.

## Goals and implementation plan

1. Keep the existing original-workbook retraining path available as an optional source.
2. Add a dedicated **Train Model** workspace with a model name, C grid, decision
   threshold, and a scrollable WSI thumbnail grid. Filter by cohort/split, search sample
   names, sort by cohort/split/medullary percentage, and choose the number of columns.
3. Use separate Training and Validation checkboxes. Selecting one disables the other
   for that sample; the backend also rejects shared sample IDs, feature files, or WSIs.
4. Freeze all eligible native feature tiles from each selected sample, independent of
   viewer filters, overlap display settings, and tile budgets. Saved prediction labels
   provide the default; human tile corrections override them. Exclusions are omitted.
5. Fit scaler/PCA/RBF SVC on training samples only for each C. Submit the search as
   one separate Slurm GPU job using the configured account, QoS, partition, GPU,
   CPU, RAM, and time settings. Evaluate on validation samples only.
6. Publish each successful candidate separately with sample records, tile provenance,
   metrics, and a confusion matrix. Let the reviewer choose the candidate in the GUI.
7. Add heatmap presets and custom low/middle/high colors. Start each sample with
   polygon annotations hidden and unloaded; leave prediction/correction tiles visible.
8. Verify the request and snapshot rules, existing workflows, browser behavior, and
   a synthetic end-to-end GPU search before publication.

## Use

Click **Train Model**, name the model, enter positive C values separated by commas
or spaces (up to 20), and select at least one sample for each set. Thumbnails show
the WSI under all saved prediction tiles and resolved human corrections: blue cortex,
red medulla, grey excluded, pale outlines for corrected tiles. Samples lacking saved
predictions cannot be selected; run a prediction for them first.

Medullary percentage uses the current saved predictions plus human corrections,
excluding omitted tiles. Refreshing samples updates prediction versions and correction
revisions for retained selections. If a correction changes after selection, preparation
fails with a refresh message instead of silently training with different labels.

**Submit GPU training** freezes labels in the background, then submits `sbatch`.
Check **GPU Jobs** for preparation, Slurm ID, candidate progress, and errors.
The selected-sample path does not require the original split workbook or original
training labels. It fits a new pipeline rather than updating the selected model's weights.
The selected model's PCA component count is retained when compatible with the data.

**Models & validation results** shows accuracy, balanced accuracy, and AUROC for
each candidate, plus accuracy, balanced accuracy, AUROC, sensitivity, and specificity
for cortex and medulla. Expand a candidate to see its confusion matrix (reference
rows, predicted columns) and training/validation sample membership. **Use model**
selects it for the next prediction run; then run it on the desired slides to display its
predictions. Completed candidates remain available after server restart.

Validation reference labels include predictions and human corrections; these metrics
measure agreement with that reference. AUROC and class metrics with missing
denominators show an em dash when undefined. Sample disjointness is enforced;
choose independent cases if patient-level independence is needed.

Large training sets are never silently subsampled. GPU memory preflight may ask for
fewer selected samples or a larger configured GPU. Candidate failures are recorded
individually; successful candidates remain usable if other C values fail.

## Records

Each search keeps the following under its persistent `state/jobs/train-*/` directory:

- `job.json`: name, exact sample selections, C values, execution status, resources,
  candidate progress, and errors.
- `request.json` and `run.sh`: worker configuration and Slurm launch command.
- `snapshot/snapshot.json`: sample IDs/names/cohorts, membership, prediction IDs,
  correction revisions, class counts, feature stamps, and label checksums.
- `snapshot/<sample-id>.npz` and event JSON: immutable eligible tile IDs,
  coordinates, labels, human-override flags, source event IDs, and correction history.
- `artifacts/sample_membership.json`, `training_tile_provenance.csv`, and
  `search_results.json`: shared records for the search.
- `artifacts/<candidate-id>/`: saved GPU pipeline, metadata, validation metrics,
  validation tile provenance, and `validation_confusion_matrix.png`.
- `state/models/<candidate-id>.json`: persistent GUI model registration.

## Viewer behavior

Heatmap colors apply to both raw and aggregated score views, including the legend,
PNG exports, and exported parameters. Presets include blue/yellow, blue/white/red,
and purple/green/yellow. Editing any endpoint switches to custom colors. Colors
are display settings and never change predictions, corrections, or decision scores.

Polygon files are listed when a slide opens; their geometry is read when a user
shows a layer, opens its region list, chooses it as the annotation target, or clicks
**Show all annotations**. **Hide & close all** hides layers and collapses region lists.
Changing samples resets polygon visibility and expansion, and enables the combined
prediction/correction tile overlay with both tissue classes shown. File drafts retain
the existing save/undo protections.

## Verification

From the repository root:

```bash
PYTHONPATH=. HISTOLOOP_BROWSER=/path/to/chromium \
  SVC_GPU/human_loop/.venv/bin/python -m pytest SVC_GPU/human_loop/tests -q
node SVC_GPU/human_loop/tests/test_prediction_field.js
```

For an isolated real GPU search using the configured HiPerGator defaults:

```bash
PYTHONPATH=. SVC_GPU/human_loop/.venv/bin/python \
  -m SVC_GPU.human_loop.tests.gpu_smoke_selected
```

The GPU harness uses synthetic slides/features and a separate model registry under
ignored `test_artifacts/`; it does not change the review dataset or active models.
