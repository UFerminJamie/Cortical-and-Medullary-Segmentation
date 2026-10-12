from __future__ import annotations

import hashlib
import re
import threading
import uuid
from zipfile import BadZipFile
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .config import Config
from .cache import BoundedCache, JsonCache, ReadCache, signature
from .grid import grid_info
from .storage import Store, atomic_json, read_json


def cohort(path: str) -> str:
    match = re.search(r"/KPMP_([^/]+)", path)
    return match.group(1) if match else "Other"


class Catalog:
    def __init__(self, config: Config, store: Store, *, refresh: bool = False):
        self.config, self.store = config, store
        self.read_cache = ReadCache(config.state_dir, config.cache_dir)
        self.json_cache = JsonCache()
        self._refresh = refresh
        # Split metadata is optional for viewing; training still requires it.
        original = (pd.read_excel(config.original_manifest).fillna("")
                    if Path(config.original_manifest).is_file()
                    else pd.DataFrame(columns=["sample_name", "split", "case_id"]))
        self.original_rows = original.to_dict("records")
        self._original_rows = {Path(str(row.sample_name)).stem: row._asdict() for row in original.itertuples(index=False)}
        self.slides: dict[str, dict] = {}
        self.datasets: list[dict] = []
        self._lock = threading.RLock()
        self._cache: OrderedDict[str, tuple[tuple, dict]] = OrderedDict()
        self._tile_locks = tuple(threading.RLock() for _ in range(64))
        self._saved_cache = BoundedCache(6, config.prediction_cache_mib * 1024**2)
        self.generation = uuid.uuid4().hex
        for source in config.dataset_sources():
            dataset = {"id": source["id"], "name": source["name"], "available": False,
                       "slide_count": 0, "missing_predictions": 0}
            self.datasets.append(dataset)
            if not Path(source["manifest"]).is_file():
                dataset["reason"] = "Workbook not found"
                continue
            self.slides.update(self._load_dataset(source, dataset, self._original_rows))
            dataset["available"] = True
        self._migrate_prediction_versions()

    def _load_dataset(self, source: dict, dataset: dict, original_rows: dict):
        source_stamp = signature(source["manifest"], Path(source["predictions"]) / "inference_summary.json", self.config.original_manifest)
        source_stamp += str((source["manifest"], source["predictions"], source["name"]))
        cached = None if self._refresh else self.read_cache.get("dataset:" + source["id"], source_stamp)
        if cached:
            dataset.update(cached["dataset"])
            slides = cached["slides"]
            if any(key in self.slides for key in slides):
                raise ValueError("A sample appears in multiple datasets with the same feature path")
            return slides
        frame = pd.read_excel(source["manifest"]).fillna("")
        if "feature_path" not in frame or "sample_name" not in frame:
            raise ValueError(f"Dataset {source['name']} requires sample_name and feature_path columns")
        summary = read_json(Path(source["predictions"]) / "inference_summary.json", {"samples": []})
        threshold = float(summary.get("threshold", 0.0))
        summaries = {row["sample"]: row for row in summary["samples"]}
        names, slides = set(), {}
        for row in frame.to_dict("records"):
            if not str(row["sample_name"]).strip() or not str(row["feature_path"]).strip():
                raise ValueError(f"Dataset {source['name']} has a sample without a name or feature path")
            name = Path(str(row["sample_name"])).stem
            pt_path = str(Path(row["feature_path"]).resolve())
            slide_id = hashlib.sha256((name + "\0" + pt_path).encode()).hexdigest()[:20]
            if name in names:
                raise ValueError(f"Duplicate sample in inference manifest: {name}")
            names.add(name)
            if slide_id in self.slides:
                raise ValueError(f"Sample {name} appears in multiple datasets with the same feature path")
            wsi_path = str(row.get("wsi_path") or row.get("folder_path") or "")
            previous = original_rows.get(name, {})
            stats = summaries.get(name, {})
            csv_path = Path(source["predictions"]) / "predictions" / f"{name}.csv"
            baseline_available = csv_path.is_file()
            dataset["slide_count"] += 1
            dataset["missing_predictions"] += int(not baseline_available)
            slides[slide_id] = {"id": slide_id, "name": name, "pt_path": pt_path,
                "dataset_id": source["id"], "dataset_name": source["name"], "baseline_threshold": threshold,
                "baseline_available": baseline_available,
                "wsi_path": wsi_path, "cohort": cohort(wsi_path), "original_split": self.split(previous.get("split", "new")),
                "case_id": str(previous.get("case_id") or row.get("case_id") or ""),
                "tiles": int(stats.get("tiles", 0)), "cortical": int(stats.get("cortical", 0)),
                "medullary": int(stats.get("medullary", 0)),
                "baseline_csv": str(csv_path)}
        self.read_cache.put("dataset:" + source["id"], source_stamp, {"dataset": dataset, "slides": slides})
        return slides

    def add_dataset(self, name: str, manifest: str, predictions: str) -> dict:
        name = name.strip()
        if not name or len(name) > 120:
            raise ValueError("Enter a cohort name of 1–120 characters")
        paths = {}
        for key, value in (("manifest", manifest), ("predictions", predictions)):
            path = Path(value.strip()).expanduser()
            if not value.strip() or not path.is_absolute():
                raise ValueError("Enter absolute server paths for the workbook and prediction folder")
            paths[key] = str(path.resolve())
        if not Path(paths["manifest"]).is_file() or Path(paths["manifest"]).suffix.lower() != ".xlsx":
            raise ValueError("Choose an existing .xlsx sample workbook")
        if not Path(paths["predictions"]).is_dir():
            raise ValueError("Prediction folder does not exist")
        with self.store.lock(), self._lock:
            sources = self.config.dataset_sources()
            if any(item["name"].casefold() == name.casefold() for item in sources):
                raise ValueError("A cohort with that name is already configured")
            if any(Path(item["manifest"]).resolve() == Path(paths["manifest"]) for item in sources):
                raise ValueError("That workbook is already configured")
            source = {"id": "cohort-" + uuid.uuid4().hex[:16], "name": name, **paths}
            dataset = {"id": source["id"], "name": name, "available": True,
                       "slide_count": 0, "missing_predictions": 0}
            try:
                slides = self._load_dataset(source, dataset, self._original_rows)
            except (OSError, ValueError, KeyError, TypeError, BadZipFile) as error:
                raise ValueError(f"Cannot load cohort: {error}") from error
            if not slides:
                raise ValueError("The workbook contains no samples")
            for slide in slides.values():
                if not Path(slide["pt_path"]).is_file():
                    raise ValueError(f"Feature file not found for {slide['name']}: {slide['pt_path']}")
                if not slide["wsi_path"] or not Path(slide["wsi_path"]).is_file():
                    raise ValueError(f"Slide image not found for {slide['name']}: {slide['wsi_path']}")
            path = self.store.root / "added_datasets.json"
            added = read_json(path, [])
            # Persist only after validation; invalid registrations never change the live catalog.
            atomic_json(path, [*added, source])
            self.slides.update(slides)
            self.datasets.append(dataset)
            self.generation = uuid.uuid4().hex
            return dataset

    def refresh_datasets(self) -> list[dict]:
        with self.store.lock(), self._lock:
            refreshed = Catalog(self.config, self.store, refresh=True)
            self.slides, self.datasets = refreshed.slides, refreshed.datasets
            self._original_rows = refreshed._original_rows
            self.original_rows = refreshed.original_rows
            self._cache.clear()
            self._saved_cache.clear()
            self.generation = uuid.uuid4().hex
            refreshed.read_cache.close()
            return self.datasets

    @staticmethod
    def split(value: str) -> str:
        value = str(value).strip().lower()
        return "val" if value in ("val", "valid", "validation") else value if value in ("train", "test") else "new"

    def _migrate_prediction_versions(self):
        """Index existing saved outputs without loading feature matrices or altering active views."""
        if (self.store.root / "prediction_models.json").exists():
            return
        with self.store.lock():
            if (self.store.root / "prediction_models.json").exists():
                return
            index = {}
            for path in sorted((self.store.root / "jobs").glob("predict-*/job.json")):
                job = read_json(path, {})
                slide_id = job.get("slide_id")
                saved = self.store.root / "predictions" / str(slide_id) / f"{job.get('id')}.npz"
                if slide_id not in self.slides or job.get("status") != "completed" or not saved.is_file():
                    continue
                result = job.get("result", {})
                entry = {"job_id": job["id"], "model_id": job["model_id"], "path": str(saved),
                    "threshold": job["threshold"], "created_at": job.get("finished_at", job["created_at"]),
                    "cortical": result.get("cortex", 0), "medullary": result.get("medulla", 0),
                    "tiles": result.get("tiles", 0)}
                versions_path = saved.parent / "versions.json"
                versions = read_json(versions_path, {})
                versions[job["id"]] = entry
                atomic_json(versions_path, versions)
                current = index.setdefault(slide_id, {}).get(job["model_id"])
                if not current or entry["created_at"] > current["created_at"]:
                    index[slide_id][job["model_id"]] = entry
            # Older installations may have an active output without a completed job record.
            for path in (self.store.root / "predictions").glob("*/active.json"):
                entry = read_json(path)
                slide_id = path.parent.name
                if slide_id not in self.slides:
                    continue
                stats = read_json(self.store.root / "prediction_index.json", {}).get(slide_id, {})
                entry = {**entry, "cortical": stats.get("cortical", 0), "medullary": stats.get("medullary", 0),
                         "tiles": self.slides[slide_id]["tiles"]}
                versions = read_json(path.parent / "versions.json", {})
                versions.setdefault(entry["job_id"], entry)
                atomic_json(path.parent / "versions.json", versions)
                index.setdefault(slide_id, {}).setdefault(entry["model_id"], entry)
            atomic_json(self.store.root / "prediction_models.json", index)

    def slide(self, slide_id: str) -> dict:
        if slide_id not in self.slides:
            raise KeyError("Unknown slide")
        return self.slides[slide_id]

    def listing(self, pass_id: str, model_id: str = "active", dataset_id: str | None = None) -> list[dict]:
        if dataset_id is not None:
            dataset = next((item for item in self.datasets if item["id"] == dataset_id), None)
            if dataset is None:
                raise KeyError("Unknown dataset")
            if not dataset["available"]:
                raise ValueError(f"Dataset {dataset['name']}: {dataset['reason']}")
        summaries = self.store.summary()
        predictions = self.json_cache.read(self.store.root / "prediction_index.json", {})
        by_model = self.json_cache.read(self.store.root / "prediction_models.json", {})
        if model_id != "active":
            self.store.model(model_id, self.config.baseline_model)
        rows = []
        with self._lock:
            slide_rows = [row for row in self.slides.values() if dataset_id is None or row["dataset_id"] == dataset_id]
        for row in sorted(slide_rows, key=lambda value: (value["cohort"], value["name"])):
            if dataset_id is not None and row["dataset_id"] != dataset_id:
                continue
            stats = predictions.get(row["id"], {}) if model_id == "active" else by_model.get(row["id"], {}).get(model_id, {}) if model_id != "baseline" else {}
            available = bool(stats) or (row["baseline_available"] and model_id in ("active", "baseline"))
            item = {**row, **summaries.get(row["id"], {}), **stats,
                    "reviewed": pass_id in summaries.get(row["id"], {}).get("reviewed_passes", {}),
                    "model_id": stats.get("model_id", "baseline") if available else model_id,
                    "prediction_available": available,
                    "prediction_id": stats.get("job_id", stats.get("prediction_id", "baseline")) if available else None}
            if not available:
                item.update(cortical=0, medullary=0)
            total = item["cortical"] + item["medullary"]
            item["medulla_fraction"] = item["medullary"] / total if total else None
            item["medulla_cortex_ratio"] = item["medullary"] / item["cortical"] if item["cortical"] else None
            rows.append(item)
        return rows

    def prediction_versions(self, slide_id: str) -> list[dict]:
        slide = self.slide(slide_id)
        base = {"job_id": "baseline", "model_id": "baseline", "threshold": slide["baseline_threshold"],
                "created_at": None, "cortical": slide["cortical"], "medullary": slide["medullary"]}
        versions = self.json_cache.read(self.store.root / "predictions" / slide_id / "versions.json", {})
        return ([base] if slide["baseline_available"] else []) + sorted(versions.values(), key=lambda item: item["created_at"], reverse=True)

    def geometry(self, slide_id: str) -> dict:
        """Feature geometry permits a first prediction before a baseline CSV exists."""
        slide = self.slide(slide_id)
        path = Path(slide["pt_path"])
        stamp = path.stat()
        stamp = (stamp.st_mtime_ns, stamp.st_size)
        cache_stamp = str((path, stamp))
        cached = self.read_cache.arrays("geometry:" + slide_id, cache_stamp)
        if cached:
            return cached
        payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        coordinates = np.asarray(payload.get("coordinates", payload.get("coords")), dtype=np.int64)
        if coordinates.ndim != 2 or coordinates.shape[1] != 2 or not len(coordinates):
            raise ValueError("Features require nonempty x/y tile coordinates")
        if len(np.unique(coordinates, axis=0)) != len(coordinates) or np.any(coordinates < 0):
            raise ValueError("Feature coordinates must be unique and nonnegative")
        size = payload.get("tile_size") or payload.get("metadata", {}).get("read_plan", {}).get("patch_size_level0") or 512
        if isinstance(size, (tuple, list)):
            if len(size) != 2 or size[0] != size[1]:
                raise ValueError("The reviewer requires square feature patches")
            size = size[0]
        if int(size) <= 0:
            raise ValueError("Tile size must be positive")
        digest = hashlib.sha256(coordinates.astype("<i8").tobytes())
        digest.update(str((path, stamp)).encode())
        data = {"coordinates": coordinates, "scores": np.zeros(len(coordinates), np.float32),
                "predictions": np.zeros(len(coordinates), np.int8), "tile_size": int(size),
                "fingerprint": digest.hexdigest(), "grid": grid_info(coordinates, int(size)),
                "model_id": "baseline", "prediction_id": "unavailable", "prediction_threshold": slide["baseline_threshold"]}
        self.read_cache.put_arrays("geometry:" + slide_id, cache_stamp, data)
        return data

    def tiles(self, slide_id: str) -> dict:
        lock = self._tile_locks[int(hashlib.sha256(slide_id.encode()).hexdigest()[:8], 16) % len(self._tile_locks)]
        with lock:
            return self._load_tiles(slide_id)

    def _trim_tiles(self):
        limit = self.config.prediction_cache_mib * 1024**2
        while self._cache and (len(self._cache) > 6 or
                sum(sum(value.nbytes for value in data.values() if isinstance(value, np.ndarray))
                    for _, data in self._cache.values()) > limit):
            self._cache.popitem(last=False)

    def _load_tiles(self, slide_id: str) -> dict:
        slide = self.slide(slide_id)
        csv_path, pt_path = Path(slide["baseline_csv"]), Path(slide["pt_path"])
        if not csv_path.is_file():
            return self.geometry(slide_id)
        try:
            csv_stat, pt_stat = csv_path.stat(), pt_path.stat()
        except FileNotFoundError as error:
            raise FileNotFoundError(f"Prediction CSV or features are missing for {slide['name']}") from error
        stamp = (csv_stat.st_mtime_ns, csv_stat.st_size, pt_stat.st_mtime_ns, pt_stat.st_size)
        with self._lock:
            existing = self._cache.get(slide_id)
            if existing and existing[0] == stamp:
                self._cache.move_to_end(slide_id)
                return existing[1]
        artifact_stamp = str((str(csv_path), str(pt_path), stamp, slide["baseline_threshold"]))
        cached = self.read_cache.arrays("tiles:" + slide_id, artifact_stamp)
        if cached:
            with self._lock:
                self._cache[slide_id] = (stamp, cached)
                self._trim_tiles()
            return cached
        frame = pd.read_csv(csv_path, usecols=["x", "y", "decision_score", "predicted_label_encoded"])
        coordinates = frame[["x", "y"]].to_numpy(dtype=np.int64)
        # mmap exposes metadata and coordinates without reading the complete feature matrix.
        payload = torch.load(pt_path, map_location="cpu", weights_only=False, mmap=True)
        pt_coords = payload.get("coordinates", payload.get("coords")) if isinstance(payload, dict) else None
        if pt_coords is None or not np.array_equal(coordinates, np.asarray(pt_coords)):
            raise ValueError(f"Prediction rows and feature coordinates differ for {slide['name']}")
        if len(np.unique(coordinates, axis=0)) != len(coordinates):
            raise ValueError(f"Duplicate tile coordinates in {slide['name']}")
        meta = payload.get("metadata", {})
        tile_size = payload.get("tile_size") or meta.get("read_plan", {}).get("patch_size_level0") or 512
        if isinstance(tile_size, (tuple, list)):
            if len(tile_size) != 2 or tile_size[0] != tile_size[1]:
                raise ValueError("The reviewer currently requires square feature patches")
            tile_size = tile_size[0]
        digest = hashlib.sha256(coordinates.astype("<i8").tobytes())
        digest.update(str((pt_path, stamp[2:])).encode())
        result = {"coordinates": coordinates, "scores": frame.decision_score.to_numpy(dtype=np.float32),
                  "predictions": frame.predicted_label_encoded.to_numpy(dtype=np.int8),
                  "tile_size": int(tile_size), "fingerprint": digest.hexdigest(),
                  "model_id": "baseline", "prediction_id": "baseline", "prediction_threshold": slide["baseline_threshold"]}
        result["grid"] = grid_info(coordinates, int(tile_size))
        if not np.isfinite(result["scores"]).all() or not np.isin(result["predictions"], [0, 1]).all():
            raise ValueError("Prediction data contains invalid scores or labels")
        del payload
        self.read_cache.put_arrays("tiles:" + slide_id, artifact_stamp, result)
        with self._lock:
            self._cache[slide_id] = (stamp, result)
            self._cache.move_to_end(slide_id)
            self._trim_tiles()
        return result

    def current_tiles(self, slide_id: str, model_id: str = "active", prediction_id: str | None = None) -> dict:
        base = self.tiles(slide_id)
        directory = self.store.root / "predictions" / slide_id
        if prediction_id == "baseline" or (not prediction_id and model_id == "baseline"):
            if base["prediction_id"] == "unavailable":
                raise FileNotFoundError("Initial predictions are missing. Run SVC on the selected slides first.")
            if model_id not in ("active", "baseline"):
                raise ValueError("Initial prediction does not belong to the selected model")
            return base
        if prediction_id:
            active = self.json_cache.read(directory / "versions.json", {}).get(prediction_id)
            if not active or (model_id != "active" and active["model_id"] != model_id):
                raise ValueError("Saved prediction does not belong to the selected model")
        elif model_id == "active":
            active = self.json_cache.read(directory / "active.json")
        else:
            self.store.model(model_id, self.config.baseline_model)
            active = self.json_cache.read(self.store.root / "prediction_models.json", {}).get(slide_id, {}).get(model_id)
            if not active:
                raise FileNotFoundError("No saved prediction for this slide and model. Run this model on the slide first.")
        if active:
            key = (slide_id, base["fingerprint"], signature(active["path"]),
                   active["job_id"], active["model_id"], active["threshold"])
            cached = self._saved_cache.get(key)
            if cached is not None:
                return {**base, **cached}
            with np.load(active["path"], allow_pickle=False) as saved:
                if saved["fingerprint"].item() != base["fingerprint"]:
                    raise ValueError("The updated prediction no longer matches the source features")
                if len(saved["scores"]) != len(base["coordinates"]):
                    raise ValueError("Updated prediction length differs from source tiles")
                if (len(saved["predictions"]) != len(base["coordinates"]) or
                    not np.isfinite(saved["scores"]).all() or not np.isin(saved["predictions"], [0, 1]).all()):
                    raise ValueError("Saved prediction contains invalid scores or labels")
                base = {**base, "scores": saved["scores"].copy(), "predictions": saved["predictions"].copy(),
                        "model_id": active["model_id"], "prediction_id": active["job_id"],
                        "prediction_threshold": active["threshold"]}
            payload = {name: base[name] for name in ("scores", "predictions", "model_id", "prediction_id", "prediction_threshold")}
            payload["source_signature"] = key[2]
            base["source_signature"] = key[2]
            self._saved_cache.put(key, payload, base["scores"].nbytes + base["predictions"].nbytes)
        if base["prediction_id"] == "unavailable":
            raise FileNotFoundError("Predictions are missing. Select the slide in the queue and run SVC first.")
        return base
