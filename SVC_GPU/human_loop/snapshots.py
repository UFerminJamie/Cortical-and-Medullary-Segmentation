from __future__ import annotations

import hashlib
import io
import shutil
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from SVC_GPU.manifest import load_split_samples

from .catalog import Catalog
from .storage import Store, atomic_json, now


def file_stamp(path: str | Path) -> dict:
    path = Path(path)
    stat = path.stat()
    return {"path": str(path.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def verify_stamp(stamp: dict) -> None:
    actual = file_stamp(stamp["path"])
    if actual != stamp:
        raise ValueError(f"Source file changed after the training snapshot: {stamp['path']}")


def resolved_owners(events: list[dict]) -> dict:
    undone = {event["target_id"] for event in events if event["action"] == "undo"}
    owners = {}
    for event in events:
        if event["id"] in undone:
            continue
        if event["action"] == "label":
            owners.update({tile_id: event for tile_id in event["tile_ids"]})
        elif event["action"] == "clear":
            for tile_id in event["tile_ids"]:
                owners.pop(tile_id, None)
    return owners


def corrected_records(catalog: Catalog, store: Store):
    for slide_id, summary in store.summary().items():
        if not summary.get("manual_tiles") and not summary.get("excluded_tiles"):
            continue
        slide, data = catalog.slide(slide_id), catalog.tiles(slide_id)
        events = store.events(slide_id)
        for event in events:
            if event.get("fingerprint") and event["fingerprint"] != data["fingerprint"]:
                raise ValueError(f"Saved annotations no longer match the features for {slide['name']}")
        labels, revision, _ = store.resolved(slide_id, len(data["coordinates"]), events)
        ids = np.flatnonzero(labels != -1)
        if len(ids):
            yield slide, data, events, ids, labels[ids], revision


def create_snapshot(catalog: Catalog, store: Store, directory: Path) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    corrected = []
    with store.lock():
        for slide, data, events, ids, labels, revision in corrected_records(catalog, store):
            name = slide["id"]
            path = directory / f"{name}.npz"
            owners = resolved_owners(events)
            np.savez_compressed(path, tile_ids=ids, coordinates=data["coordinates"][ids], labels=labels,
                                source_event_ids=np.array([owners[int(tile_id)]["id"] for tile_id in ids], dtype="U32"),
                                fingerprint=np.array(data["fingerprint"]))
            atomic_json(directory / f"{name}_events.json", events)
            corrected.append({**slide, "labels_path": str(path), "tile_size": data["tile_size"],
                "revision": revision, "corrected_tiles": int((labels >= 0).sum()), "excluded_tiles": int((labels == -2).sum()),
                "feature_stamp": file_stamp(slide["pt_path"]),
                "cortex": int((labels == 0).sum()), "medulla": int((labels == 1).sum())})
        if not corrected:
            raise ValueError("Annotate at least one tile before retraining")
        original_manifest = directory / "original_case_split.xlsx"
        shutil.copyfile(catalog.config.original_manifest, original_manifest)
        original = []
        for split in ("train", "test"):
            rows = load_split_samples(original_manifest, catalog.config.original_features, split)
            for row in rows.to_dict("records"):
                csv_copy = directory / "original_labels" / split / f"{Path(row['pt_path']).stem}.csv"
                csv_copy.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(row["csv_path"], csv_copy)
                original.append({"split": split, "pt_path": row["pt_path"], "csv_copy": str(csv_copy),
                    "feature_stamp": file_stamp(row["pt_path"]),
                    "label_sha256": hashlib.sha256(csv_copy.read_bytes()).hexdigest()})
        result = {"schema_version": 2, "created_at": now(), "passes": store.passes(),
            "original_manifest": str(original_manifest), "original_sources": original,
            "slides": corrected, "corrected_tiles": sum(row["corrected_tiles"] for row in corrected),
            "excluded_tiles": sum(row["excluded_tiles"] for row in corrected),
            "label_encoding": {"0": "cortex", "1": "medulla", "-2": "excluded", "-1": "unannotated"}}
        atomic_json(directory / "snapshot.json", result)
        return result


def export_zip(catalog: Catalog, store: Store) -> bytes:
    buffer = io.BytesIO()
    with store.lock(), zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        import json

        archive.writestr("passes.json", json.dumps(store.passes(), indent=2))
        exported = []
        for slide_id in store.summary():
            events = store.events(slide_id)
            if not events:
                continue
            slide, data = catalog.slide(slide_id), catalog.tiles(slide_id)
            for event in events:
                if event.get("fingerprint") and event["fingerprint"] != data["fingerprint"]:
                    raise ValueError(f"Feature files changed for {slide['name']}")
            labels, revision, _ = store.resolved(slide_id, len(data["coordinates"]), events)
            ids = np.flatnonzero(labels != -1)
            owners = resolved_owners(events)
            rows = pd.DataFrame({"tile_id": ids, "x": data["coordinates"][ids, 0],
                "y": data["coordinates"][ids, 1], "label_encoded": labels[ids],
                "label": np.select([labels[ids] == 0, labels[ids] == 1], ["cortex", "medulla"], default="excluded"),
                "tile_size_level0": data["tile_size"],
                "source_event_id": [owners[int(index)]["id"] for index in ids],
                "pass_id": [owners[int(index)]["pass_id"] for index in ids],
                "reviewer": [owners[int(index)]["reviewer"] for index in ids],
                "corrected_at": [owners[int(index)]["created_at"] for index in ids]})
            archive.writestr(f"slides/{slide_id}/corrections.csv", rows.to_csv(index=False))
            archive.writestr(f"slides/{slide_id}/history.json", json.dumps(events, indent=2))
            exported.append({**slide, "revision": revision, "manual_tiles": int((labels >= 0).sum()),
                             "excluded_tiles": int((labels == -2).sum()), "fingerprint": data["fingerprint"]})
        archive.writestr("manifest.json", json.dumps({"schema_version": 2, "created_at": now(),
            "coordinates": "level-zero top-left pixels; regions select tile centers", "slides": exported}, indent=2))
    return buffer.getvalue()
