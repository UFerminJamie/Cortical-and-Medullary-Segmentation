"""GPU worker for reproducible review snapshots and single-slide prediction."""
from __future__ import annotations

import argparse
import hashlib
import os
import socket
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.spatial import cKDTree

from SVC_GPU.data import load_sample_features
from SVC_GPU.evaluation import classification_metrics, save_confusion_matrix
from SVC_GPU.gpu_model import base_metadata, load_bundle, predict_gpu, require_gpu, save_bundle, train_gpu_pipeline
from SVC_GPU.manifest import load_split_samples
from SVC_GPU.outputs import (prediction_dataframe, save_combined_geojson, save_mask_geojson,
                             save_prediction_csv, save_thresholded_mask)

from .catalog import Catalog
from .config import Config
from .geometry import region_indices
from .snapshots import verify_stamp
from .storage import Store, atomic_json, now, read_json
from .wsi import WsiPool


def sample_corrections(slides: list[dict], budget: int, seed: int = 20261004) -> dict[str, np.ndarray]:
    """Fairly distribute an optional cap across slide/class groups, then sample reproducibly."""
    groups = []
    for row in slides:
        with np.load(row["labels_path"], allow_pickle=False) as data:
            for class_id in (0, 1):
                indices = np.flatnonzero(data["labels"] == class_id)
                if len(indices):
                    groups.append((row["id"], indices))
    capacities = np.array([len(ids) for _, ids in groups], dtype=np.int64)
    target = min(budget, int(capacities.sum())) if budget else int(capacities.sum())
    quotas = np.zeros(len(groups), dtype=np.int64)
    remaining = target
    while remaining:
        available = np.flatnonzero(quotas < capacities)
        amount = max(1, remaining // len(available))
        for index in available:
            take = min(amount, int(capacities[index] - quotas[index]), remaining)
            quotas[index] += take
            remaining -= take
            if not remaining:
                break
    rng = np.random.default_rng(seed)
    selected: dict[str, list[np.ndarray]] = {}
    for (slide_id, candidates), count in zip(groups, quotas):
        chosen = rng.choice(candidates, size=int(count), replace=False)
        selected.setdefault(slide_id, []).append(chosen)
    return {key: np.sort(np.concatenate(value)) for key, value in selected.items()}


def case_keys(row: dict) -> set[str]:
    name = Path(str(row.get("sample_name", row.get("name", "")))).stem
    keys = {"sample:" + name, "patient:" + name.split("_")[0]}
    case_id = str(row.get("case_id") or "").strip()
    if case_id and case_id.lower() != "nan":
        keys.add("case:" + case_id)
    return keys


def partition_holdout(train_rows: list[dict], test_rows: list[dict], corrected: list[dict]) -> tuple[list[dict], list[dict]]:
    fitted = set().union(*(case_keys(row) for row in [*train_rows, *corrected]))
    kept, excluded = [], []
    for row in test_rows:
        (excluded if case_keys(row) & fitted else kept).append(row)
    return kept, excluded


def train(config: Config, store: Store, catalog: Catalog, job: dict) -> dict:
    if job.get("training_mode") == "selected_samples":
        return train_selected(config, store, job)
    snapshot = read_json(Path(job["snapshot_path"]))
    if job.get("model_stamp"):
        verify_stamp(job["model_stamp"])
    parent = job.get("parent_metadata", read_json(Path(job["model_path"]).parent / "metadata.json", {}))
    expected_dim = int(parent.get("feature_dimension", 1024))
    baseline = read_json(Path(config.baseline_model).parent / "metadata.json", {})
    original_count = int(baseline.get("tile_counts", {}).get("fit", 0))
    added_count = min(job["budget"], snapshot["corrected_tiles"]) if job["budget"] else snapshot["corrected_tiles"]
    # The dense scaler/PCA fit needs several copies on GPU; fail before reading large matrices.
    # This is a conservative preflight, not a promise that an arbitrary RBF fit will finish.
    if original_count:
        cp, _, _, _ = require_gpu()
        free_bytes, _ = cp.cuda.runtime.memGetInfo()
        estimated_bytes = (original_count + added_count) * expected_dim * 4 * 4 + 4096 * 1024**2
        if estimated_bytes > free_bytes * .85:
            raise ValueError(f"Estimated GPU fit memory is {estimated_bytes / 1024**3:.1f} GiB; "
                f"only {free_bytes / 1024**3:.1f} GiB is free. Reduce the additional human tile budget.")
    original = {row["pt_path"]: row for row in snapshot["original_sources"]}
    train_rows = load_split_samples(snapshot["original_manifest"], config.original_features, "train").to_dict("records")
    test_rows = load_split_samples(snapshot["original_manifest"], config.original_features, "test").to_dict("records")
    retained, excluded = partition_holdout(train_rows, test_rows, snapshot["slides"])
    selected = sample_corrections(snapshot["slides"], job["budget"])
    corrections = {}
    for row in snapshot["slides"]:
        verify_stamp(row["feature_stamp"])
        with np.load(row["labels_path"], allow_pickle=False) as data:
            corrections[row["name"]] = {"row": row, "coords": data["coordinates"].copy(),
                "labels": data["labels"].copy(), "tile_ids": data["tile_ids"].copy(),
                "event_ids": data["source_event_ids"].copy() if "source_event_ids" in data else np.full(len(data["labels"]), "", dtype="U32"),
                "chosen": selected.get(row["id"], np.array([], dtype=np.int64))}
        corrections[row["name"]]["events"] = {event["id"]: event for event in read_json(Path(job["snapshot_path"]).parent / f"{row['id']}_events.json", [])}
    features, labels, provenance, exclusion_provenance = [], [], [], []
    overwritten = deduplicated = original_excluded = 0
    for row in train_rows:
        source = original[row["pt_path"]]
        verify_stamp(source["feature_stamp"])
        sample = load_sample_features(row["pt_path"], csv_path=source["csv_copy"], load_labels=True)
        y = sample.labels.numpy().copy()
        manual = np.zeros(len(y), dtype=bool)
        source_events = np.full(len(y), "", dtype="U32")
        keep = np.ones(len(y), dtype=bool)
        correction = corrections.get(Path(row["pt_path"]).stem)
        if correction:
            centers = correction["coords"] + correction["row"]["tile_size"] / 2
            tree = cKDTree(centers)
            distances, nearest = tree.query(sample.coordinates + sample.tile_size / 2, p=np.inf)
            manual = distances <= correction["row"]["tile_size"] / 2
            owners = correction["event_ids"][nearest]
            # Region boundaries still apply when transferring labels to the old tiling grid.
            for owner in np.unique(owners[manual]):
                event = correction["events"].get(owner)
                if event and event.get("grid_mode", "native") != "native":
                    group = manual & (owners == owner)
                    # A subset correction cannot spread to neighboring original
                    # tiles during reconciliation for model retraining.
                    exact = np.all(sample.coordinates == correction["coords"][nearest], axis=1)
                    manual[group & ~exact] = False
                if event and event.get("polygon"):
                    group = manual & (owners == owner)
                    inside = region_indices(sample.coordinates, sample.tile_size, event["polygon"], group)
                    manual[group] = False
                    manual[inside] = True
            y[manual] = correction["labels"][nearest[manual]]
            source_events[manual] = correction["event_ids"][nearest[manual]]
            overwritten += int(manual.sum())
            if sample.tile_size == correction["row"]["tile_size"]:
                duplicate_keys = {tuple(coordinate) for coordinate in correction["coords"][correction["chosen"]]}
                keep = np.array([tuple(coordinate) not in duplicate_keys for coordinate in sample.coordinates])
                deduplicated += int((~keep).sum())
        removed = y == -2
        original_excluded += int(removed.sum())
        if removed.any():
            exclusion_provenance.append(pd.DataFrame({"sample_name": sample.sample_name,
                "x": sample.coordinates[removed, 0], "y": sample.coordinates[removed, 1],
                "label_encoded": -2, "source": "original_train_excluded", "source_event_id": source_events[removed],
                "feature_path": str(sample.pt_path)}))
        keep &= (y == 0) | (y == 1)
        features.append(sample.features[torch.from_numpy(keep)])
        labels.append(torch.from_numpy(y[keep].astype(np.int64)))
        provenance.append(pd.DataFrame({"sample_name": sample.sample_name, "x": sample.coordinates[keep, 0],
            "y": sample.coordinates[keep, 1], "label_encoded": y[keep], "source": "original_train",
            "human_override": manual[keep], "feature_path": str(sample.pt_path)}))
        provenance[-1]["source_event_id"] = source_events[keep]
        print(f"Original train: {sample.sample_name} | {int(keep.sum()):,} tiles", flush=True)
    for correction in corrections.values():
        excluded_ids = correction["labels"] == -2
        if excluded_ids.any():
            exclusion_provenance.append(pd.DataFrame({"sample_name": correction["row"]["name"],
                "x": correction["coords"][excluded_ids, 0], "y": correction["coords"][excluded_ids, 1],
                "label_encoded": -2, "source": "human_review_excluded",
                "source_event_id": correction["event_ids"][excluded_ids], "feature_path": correction["row"]["pt_path"]}))
        chosen = correction["chosen"]
        if not len(chosen):
            continue
        row = correction["row"]
        sample = load_sample_features(row["pt_path"])
        if not np.array_equal(sample.coordinates[correction["tile_ids"]], correction["coords"]):
            raise ValueError(f"Correction coordinates do not match features for {sample.sample_name}")
        ids = correction["tile_ids"][chosen]
        features.append(sample.features[torch.from_numpy(ids)])
        labels.append(torch.from_numpy(correction["labels"][chosen].astype(np.int64)))
        provenance.append(pd.DataFrame({"sample_name": row["name"], "x": sample.coordinates[ids, 0],
            "y": sample.coordinates[ids, 1], "label_encoded": correction["labels"][chosen],
            "source": "human_review", "human_override": True, "feature_path": row["pt_path"]}))
        provenance[-1]["source_event_id"] = correction["event_ids"][chosen]
        print(f"Human review: {row['name']} | {len(ids):,} tiles", flush=True)
    x, y = torch.cat(features), torch.cat(labels)
    del features, labels
    if not len(y) or set(y.numpy().tolist()) != {0, 1}:
        raise ValueError("Retraining needs remaining cortex and medulla tiles after applying exclusions")
    if x.shape[1] != expected_dim:
        raise ValueError(f"Expected {expected_dim} feature dimensions; found {x.shape[1]}")
    n_components = min(int(parent.get("n_components", 1024)), min(x.shape))
    output = Path(job["directory"]) / "artifacts"
    output.mkdir(parents=True, exist_ok=True)
    pd.concat(provenance, ignore_index=True).to_csv(output / "training_tile_provenance.csv", index=False)
    if exclusion_provenance:
        pd.concat(exclusion_provenance, ignore_index=True).to_csv(output / "excluded_tile_provenance.csv", index=False)
    print(f"Fitting SVC: {len(y):,} tiles, {x.shape[1]} features, {n_components} PCA components", flush=True)
    bundle = train_gpu_pipeline(x, y, n_components=n_components, c_value=float(parent.get("C", 2.0)), cache_size_mb=4096)
    metadata = {**base_metadata(), "model_type": "cuml StandardScaler -> PCA -> RBF SVC",
        "feature_dimension": x.shape[1], "n_components": n_components, "C": float(parent.get("C", 2.0)),
        "decision_threshold": job["threshold"], "parent_model_id": job["model_id"],
        "parent_model_path": job["model_path"], "snapshot_path": job["snapshot_path"],
        "fitted_splits": ["original_train", "human_review"], "original_training_samples": len(train_rows),
        "original_train_labels_overridden": overwritten, "exact_duplicate_original_tiles_removed": deduplicated,
        "original_training_tiles_excluded": original_excluded, "human_tiles_excluded": snapshot.get("excluded_tiles", 0),
        "corrected_tiles_available": snapshot["corrected_tiles"], "corrected_tiles_sampled": sum(len(ids) for ids in selected.values()),
        "additional_tile_budget": job["budget"], "sampling_seed": 20261004, "training_tiles": len(y),
        "class_counts_fit": {"cortical": int((y == 0).sum()), "medullary": int((y == 1).sum())},
        "heldout_samples": len(retained), "excluded_heldout_samples": [row["sample_name"] for row in excluded],
        "evaluation_policy": "Exclude all original test slides sharing a recorded case ID or filename patient prefix with fitted data",
        "gpu": bundle["gpu"]}
    model_path = save_bundle(bundle, output / "svc_pipeline.pkl", metadata)
    atomic_json(output / "metadata.json", metadata)
    prediction, scores = predict_gpu(bundle, x, batch_size=32768, threshold=job["threshold"])
    metrics = classification_metrics(y.numpy(), prediction, scores)
    atomic_json(output / "train_metrics.json", metrics)
    del x, y
    test_features, test_labels = [], []
    for row in retained:
        source = original[row["pt_path"]]
        verify_stamp(source["feature_stamp"])
        sample = load_sample_features(row["pt_path"], csv_path=source["csv_copy"], load_labels=True)
        test_features.append(sample.features)
        test_labels.append(sample.labels)
    test_metrics = None
    if test_features:
        prediction, scores = predict_gpu(bundle, torch.cat(test_features), batch_size=32768, threshold=job["threshold"])
        test_metrics = classification_metrics(torch.cat(test_labels).numpy(), prediction, scores)
        test_metrics.update(threshold=job["threshold"], samples=len(retained), excluded_samples=len(excluded))
        atomic_json(output / "test_metrics.json", test_metrics)
        save_confusion_matrix(test_metrics, output / "test_confusion_matrix.png")
    else:
        atomic_json(output / "test_metrics.json", {"samples": 0, "excluded_samples": len(excluded),
            "reason": "No untouched original test cases remain; provide a new independent evaluation set"})
    version = {"id": job["id"], "name": f"Reviewed SVC · {job['created_at'][:16].replace('T', ' ')}",
        "path": str(model_path), "created_at": now(), "job_id": job["id"],
        "sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(), "metadata": metadata,
        "test_metrics": test_metrics}
    with store.lock():
        atomic_json(store.root / "models" / f"{job['id']}.json", version)
    return {"model_id": job["id"], "model_path": str(model_path), "training_tiles": metadata["training_tiles"],
            "heldout_samples": len(retained), "excluded_heldout_samples": len(excluded)}


def load_selected_sample(row):
    verify_stamp(row["feature_stamp"])
    path = Path(row["labels_path"])
    if hashlib.sha256(path.read_bytes()).hexdigest() != row["labels_sha256"]:
        raise ValueError("Frozen training labels changed")
    sample = load_sample_features(row["pt_path"])
    verify_stamp(row["feature_stamp"])
    with np.load(path, allow_pickle=False) as saved:
        ids, labels = saved["tile_ids"].copy(), saved["labels"].copy()
        if not np.array_equal(sample.coordinates[ids], saved["coordinates"]):
            raise ValueError(f"Frozen coordinates differ from features: {row['name']}")
        provenance = pd.DataFrame({"slide_id": row["id"], "sample_name": row["name"],
            "split": row["split"], "tile_id": ids, "x": saved["coordinates"][:, 0],
            "y": saved["coordinates"][:, 1], "label_encoded": labels,
            "human_override": saved["human_override"], "source_event_id": saved["source_event_ids"],
            "prediction_id": row["prediction_id"], "feature_path": row["pt_path"]})
    return sample.features[torch.from_numpy(ids)], torch.from_numpy(labels.astype(np.int64)), provenance


def train_selected(config, store, job):
    """Fit every C on training samples and evaluate only on frozen validation labels."""
    import gc

    snapshot = read_json(Path(job["snapshot_path"]))
    rows = snapshot["slides"]
    training = [row for row in rows if row["split"] == "training"]
    validation = [row for row in rows if row["split"] == "validation"]
    if not training or not validation or {r["id"] for r in training} & {r["id"] for r in validation}:
        raise ValueError("Frozen training and validation sets must be nonempty and disjoint")
    output = Path(job["directory"]) / "artifacts"
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "sample_membership.json", snapshot)
    # Check estimated fit memory before materializing dense matrices.
    probe = torch.load(training[0]["pt_path"], map_location="cpu", weights_only=False, mmap=True)
    dim = int(probe["features"].shape[1])
    del probe
    cp, *_ = require_gpu()
    free, _ = cp.cuda.runtime.memGetInfo()
    count = sum(row["tiles"] for row in training)
    estimate = count * dim * 4 * 4 + 4096 * 1024**2
    if estimate > free * .85:
        raise ValueError(f"All selected training tiles need approximately {estimate / 1024**3:.1f} GiB GPU memory; "
            f"{free / 1024**3:.1f} GiB is free. Select fewer samples or configure a larger GPU.")
    features, labels, provenance = [], [], []
    for row in training:
        x_row, y_row, frame = load_selected_sample(row)
        if x_row.shape[1] != dim:
            raise ValueError("Selected samples use different embedding dimensions")
        features.append(x_row); labels.append(y_row); provenance.append(frame)
    x, y = torch.cat(features), torch.cat(labels)
    del features, labels, x_row, y_row
    if set(torch.unique(y).tolist()) != {0, 1}:
        raise ValueError("Selected training tiles must contain cortex and medulla")
    pd.concat(provenance, ignore_index=True).to_csv(output / "training_tile_provenance.csv", index=False)
    del provenance
    components = min(int(job.get("parent_metadata", {}).get("n_components", dim)), min(x.shape))
    candidates = []
    for index, c_value in enumerate(job["c_values"]):
        candidate_id = f"{job['id']}-c{index + 1}"
        directory = output / candidate_id
        directory.mkdir(parents=True, exist_ok=True)
        bundle = None
        try:
            print(f"Candidate {index + 1}/{len(job['c_values'])}: C={c_value}, {len(y):,} training tiles", flush=True)
            bundle = train_gpu_pipeline(x, y, n_components=components, c_value=c_value, cache_size_mb=4096)
            truth, predicted, scores, val_provenance = [], [], [], []
            for row in validation:
                vx, vy, frame = load_selected_sample(row)
                if vx.shape[1] != dim:
                    raise ValueError("Validation and training embedding dimensions differ")
                pred, score = predict_gpu(bundle, vx, batch_size=32768, threshold=job["threshold"])
                truth.append(vy.numpy()); predicted.append(pred); scores.append(score)
                val_provenance.append(frame)
                del vx, vy
            metrics = classification_metrics(np.concatenate(truth), np.concatenate(predicted), np.concatenate(scores))
            metrics["auroc"] = metrics["per_class"]["medullary"]["auc"]
            for values in metrics["per_class"].values():
                values["auroc"] = values["auc"]
                values["balanced_accuracy"] = ((values["sensitivity"] + values["specificity"]) / 2
                    if values["sensitivity"] is not None and values["specificity"] is not None else None)
            metrics.update(samples=len(validation), tiles=sum(row["tiles"] for row in validation), threshold=job["threshold"])
            metadata = {**base_metadata(), "model_type": "cuml StandardScaler -> PCA -> RBF SVC",
                "training_mode": "selected_samples", "feature_dimension": dim, "n_components": components,
                "C": c_value, "kernel": "rbf", "gamma": "scale", "class_weight": "balanced",
                "decision_threshold": job["threshold"], "snapshot_path": job["snapshot_path"],
                "sample_membership_path": str(output / "sample_membership.json"),
                "training_samples": training, "validation_samples": validation,
                "training_tiles": len(y), "validation_tiles": metrics["tiles"],
                "class_counts_fit": {"cortical": int((y == 0).sum()), "medullary": int((y == 1).sum())},
                "label_policy": snapshot["label_policy"], "evaluation_policy": snapshot["evaluation_policy"], "gpu": bundle["gpu"]}
            model_path = save_bundle(bundle, directory / "svc_pipeline.pkl", metadata)
            atomic_json(directory / "metadata.json", metadata)
            atomic_json(directory / "validation_metrics.json", metrics)
            save_confusion_matrix(metrics, directory / "validation_confusion_matrix.png")
            pd.concat(val_provenance, ignore_index=True).to_csv(directory / "validation_tile_provenance.csv", index=False)
            record = {"id": candidate_id, "name": f"{job['model_name']} · C={c_value:g}", "path": str(model_path),
                "created_at": now(), "job_id": job["id"], "sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
                "metadata": metadata, "validation_metrics": metrics}
            with store.lock():
                atomic_json(store.root / "models" / f"{candidate_id}.json", record)
            candidates.append({"model_id": candidate_id, "C": c_value, "status": "completed", "validation_metrics": metrics})
        except Exception as error:
            traceback.print_exc()
            candidates.append({"model_id": candidate_id, "C": c_value, "status": "failed", "error": str(error)})
        finally:
            del bundle
            gc.collect()
            cp.get_default_memory_pool().free_all_blocks()
        atomic_json(output / "search_results.json", candidates)
        with store.lock():
            path = Path(job["directory"]) / "job.json"
            current = read_json(path, job)
            current.update(candidates=candidates, completed=sum(c["status"] == "completed" for c in candidates),
                failed=sum(c["status"] == "failed" for c in candidates))
            atomic_json(path, current)
    succeeded = [c for c in candidates if c["status"] == "completed"]
    if not succeeded:
        raise ValueError("Every SVC candidate failed; inspect the GPU job logs")
    # Deliberately leave model choice to the reviewer.
    return {"model_ids": [c["model_id"] for c in succeeded], "candidates": candidates,
        "training_tiles": len(y), "completed": len(succeeded), "failed": len(candidates) - len(succeeded)}


def predict(config: Config, store: Store, catalog: Catalog, job: dict, bundle=None) -> dict:
    if job.get("model_stamp"):
        verify_stamp(job["model_stamp"])
    slide = catalog.slide(job["slide_id"])
    if job.get("feature_stamp"):
        verify_stamp(job["feature_stamp"])
        if str(Path(slide["pt_path"]).resolve()) != job["feature_stamp"]["path"]:
            raise ValueError("Slide source path changed after submission")
    data = catalog.geometry(slide["id"])
    if job.get("fingerprint") and data["fingerprint"] != job["fingerprint"]:
        raise ValueError("Slide features changed after prediction was requested")
    sample = load_sample_features(slide["pt_path"])
    if job.get("feature_stamp"):
        verify_stamp(job["feature_stamp"])
    if not np.array_equal(sample.coordinates, data["coordinates"]):
        raise ValueError("Tile coordinates changed")
    bundle = bundle if bundle is not None else load_bundle(job["model_path"])
    if sample.features.shape[1] != int(bundle["metadata"].get("feature_dimension", 1024)):
        raise ValueError("Selected model and slide feature dimensions differ")
    print(f"Predicting {slide['name']}: {len(sample.features):,} tiles; threshold {job['threshold']}", flush=True)
    prediction, scores = predict_gpu(bundle, sample.features, batch_size=32768, threshold=job["threshold"])
    directory = store.root / "predictions" / slide["id"]
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{job['id']}.npz"
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, predictions=prediction.astype(np.int8), scores=scores,
                            fingerprint=np.array(data["fingerprint"]))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    frame = prediction_dataframe(sample_name=slide["name"], case_id=slide["case_id"], pt_path=sample.pt_path,
        coordinates=sample.coordinates, predictions=prediction, decision_scores=scores, threshold=job["threshold"])
    output = Path(job.get("output_directory") or (Path(job["directory"]) / "outputs"))
    save_prediction_csv(frame, output / "predictions", slide["name"])
    save_combined_geojson(frame, output / "geojson", slide["name"], tile_size=sample.tile_size, threshold=job["threshold"])
    save_mask_geojson(frame, output / "geojson", slide["name"], tile_size=sample.tile_size, threshold=job["threshold"])
    pool = WsiPool()
    try:
        info = pool.info(slide["wsi_path"])
        save_thresholded_mask(frame, output / "masks", slide["name"], tile_size=sample.tile_size,
            downsample=64, threshold=job["threshold"], wsi_dimensions=(info["width"], info["height"]))
    finally:
        pool.close()
    with store.lock():
        version = {"job_id": job["id"], "model_id": job["model_id"], "path": str(path),
            "threshold": job["threshold"], "created_at": now(), "tiles": len(prediction),
            "cortical": int((prediction == 0).sum()), "medullary": int((prediction == 1).sum())}
        versions = read_json(directory / "versions.json", {})
        versions[job["id"]] = version
        atomic_json(directory / "versions.json", versions)
        models = read_json(store.root / "prediction_models.json", {})
        models.setdefault(slide["id"], {})[job["model_id"]] = version
        atomic_json(store.root / "prediction_models.json", models)
        atomic_json(directory / "active.json", version)
        index = read_json(store.root / "prediction_index.json", {})
        index[slide["id"]] = {"cortical": int((prediction == 0).sum()), "medullary": int((prediction == 1).sum()),
            "prediction_id": job["id"], "model_id": job["model_id"], "prediction_threshold": job["threshold"]}
        atomic_json(store.root / "prediction_index.json", index)
    return {"predictions": str(path), "tiles": len(prediction), "cortex": int((prediction == 0).sum()),
            "medulla": int((prediction == 1).sum()), "outputs": str(output)}


def predict_group(config, store, catalog, job):
    """One model and GPU worker per allocation, with independently published slides."""
    verify_stamp(job["model_stamp"])
    bundle = load_bundle(job["model_path"])
    results = []
    entries = job.get("slides") or [{"id": job["slide_id"], "fingerprint": job["fingerprint"]}]
    for entry in entries:
        child = {**job, "slide_id": entry["id"], **{key: entry[key] for key in ("fingerprint", "feature_stamp") if key in entry}}
        if len(entries) > 1:
            child["output_directory"] = str(Path(job["directory"]) / "outputs" / entry["id"])
        try:
            result = predict(config, store, catalog, child, bundle=bundle)
            results.append({"slide_id": entry["id"], "status": "completed", **result})
        except Exception as error:
            traceback.print_exc()
            results.append({"slide_id": entry["id"], "status": "failed", "error": str(error)})
        with store.lock():
            path = Path(job["directory"]) / "job.json"
            current = read_json(path, job)
            current.update(completed=sum(row["status"] == "completed" for row in results),
                           failed=sum(row["status"] == "failed" for row in results), total=len(entries), results=results)
            atomic_json(path, current)
        print(f"Progress: {len(results)}/{len(entries)}; {current['failed']} failed", flush=True)
    if not current["completed"]:
        raise ValueError("All selected slides failed: " + "; ".join(row["error"] for row in results))
    return ({**results[0], "completed": current["completed"], "failed": current["failed"]}
            if len(entries) == 1 else {"slides": results, "completed": current["completed"], "failed": current["failed"]})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    args = parser.parse_args()
    request = read_json(args.request)
    config, job = Config(**request["config"]), request["job"]
    store = Store(config.state_dir)
    path = Path(job["directory"]) / "job.json"
    try:
        with store.lock():
            current = read_json(path, job)
            current.update(status="running", started_at=now())
            atomic_json(path, current)
        from SVC_GPU.gpu_model import require_gpu
        cp, *_ = require_gpu()
        resources = {"hostname": socket.gethostname(), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                     "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                     "cpu_affinity": sorted(os.sched_getaffinity(0)),
                     "cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
                     "gpu_device": int(cp.cuda.runtime.getDevice()),
                     "gpu_name": cp.cuda.runtime.getDeviceProperties(0)["name"].decode()}
        with store.lock():
            current = read_json(path, job)
            current["resources"] = resources
            atomic_json(path, current)
        print("Allocated resources:", resources, flush=True)
        catalog = Catalog(config, store)
        result = train(config, store, catalog, job) if job["kind"] == "train" else predict_group(config, store, catalog, job)
        with store.lock():
            current = read_json(path, job)
            current.update(status="completed_with_errors" if result.get("failed") else "completed", finished_at=now(), result=result)
            atomic_json(path, current)
        print("Completed", result, flush=True)
        return 0
    except Exception as error:
        traceback.print_exc()
        with store.lock():
            current = read_json(path, job)
            current.update(status="failed", finished_at=now(), error=str(error))
            atomic_json(path, current)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
