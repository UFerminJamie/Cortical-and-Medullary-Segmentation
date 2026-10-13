"""Explicit sample selection, frozen tile labels, and training thumbnails."""
from __future__ import annotations

import hashlib
import io
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .snapshots import file_stamp, resolved_owners, verify_stamp
from .cache import signature
from .storage import Conflict, atomic_json, now


def validate_selection(catalog, training, validation, name, c_values):
    if not isinstance(name, str) or not 1 <= len(name.strip()) <= 120:
        raise ValueError("Model name must contain 1–120 characters")
    if not isinstance(c_values, list) or not 1 <= len(c_values) <= 20:
        raise ValueError("Provide 1–20 positive C values")
    if any(isinstance(c, bool) or not isinstance(c, (int, float)) or not math.isfinite(c) or c <= 0 for c in c_values):
        raise ValueError("Every C value must be a finite positive number")
    groups = []
    for title, entries in (("training", training), ("validation", validation)):
        if not isinstance(entries, list) or not entries:
            raise ValueError(f"Select at least one {title} sample")
        ids, rows = set(), []
        for entry in entries:
            if not isinstance(entry, dict) or not isinstance(entry.get("id"), str):
                raise ValueError("Sample selections require an ID")
            if entry["id"] in ids:
                raise ValueError(f"Duplicate {title} sample")
            ids.add(entry["id"])
            slide = catalog.slide(entry["id"])
            version = entry.get("prediction_id")
            if not isinstance(version, str) or version not in {v["job_id"] for v in catalog.prediction_versions(slide["id"])}:
                raise ValueError(f"Choose a saved prediction for {slide['name']}; refresh the sample picker")
            revision = entry.get("revision", 0)
            if isinstance(revision, bool) or not isinstance(revision, int) or revision < 0:
                raise ValueError("Invalid correction revision")
            rows.append({"id": slide["id"], "prediction_id": version, "revision": revision})
        groups.append(rows)
    # Reject aliases of the same WSI/features as well as repeated IDs.
    def identities(rows):
        keys = set()
        for row in rows:
            slide = catalog.slide(row["id"])
            keys.update(("id:" + row["id"], "features:" + str(Path(slide["pt_path"]).resolve())))
            if slide.get("wsi_path"):
                keys.add("wsi:" + str(Path(slide["wsi_path"]).resolve()))
        return keys
    if identities(groups[0]) & identities(groups[1]):
        raise ValueError("Training and validation samples must not overlap")
    return groups[0], groups[1], name.strip(), list(dict.fromkeys(float(c) for c in c_values))


def selected_labels(catalog, store, entry):
    data = catalog.current_tiles(entry["id"], prediction_id=entry["prediction_id"])
    events = store.events(entry["id"], refresh=True)
    if any(e.get("fingerprint") and e["fingerprint"] != data["fingerprint"] for e in events):
        raise ValueError("Corrections no longer match sample features")
    human, revision, _ = store.resolved(entry["id"], len(data["coordinates"]), events)
    if revision != entry.get("revision", 0):
        raise Conflict("Corrections changed since sample selection. Refresh the sample picker before training.")
    labels = np.where(human != -1, human, data["predictions"]).astype(np.int8)
    return data, human, labels, events


def selection_fraction(catalog, store, entry):
    """Keep cohort sorting cheap; only recompute samples whose labels/sources changed."""
    slide = catalog.slide(entry["id"])
    sources = [slide["pt_path"], slide["baseline_csv"]]
    if entry["prediction_id"] != "baseline":
        sources.append(store.root / "predictions" / entry["id"] / f"{entry['prediction_id']}.npz")
    stamp = signature(*sources) + str((entry["prediction_id"], entry.get("revision", 0)))
    key = "training-summary-v1:" + entry["id"]
    cached = catalog.read_cache.get(key, stamp)
    if cached is not None:
        return cached["medulla_fraction"]
    _, _, labels, _ = selected_labels(catalog, store, entry)
    total = int(np.isin(labels, [0, 1]).sum())
    fraction = float((labels == 1).sum() / total) if total else None
    if signature(*sources) + str((entry["prediction_id"], entry.get("revision", 0))) != stamp:
        raise Conflict("Sample sources changed; refresh the sample picker")
    catalog.read_cache.put(key, stamp, {"medulla_fraction": fraction})
    return fraction


def selection_snapshot(catalog, store, directory, job):
    directory.mkdir(parents=True, exist_ok=True)
    records = []
    with store.lock():
        for split in ("training", "validation"):
            for entry in job[split + "_samples"]:
                slide = catalog.slide(entry["id"])
                stamp = file_stamp(slide["pt_path"])
                data, human, labels, events = selected_labels(catalog, store, entry)
                verify_stamp(stamp)
                ids = np.flatnonzero(np.isin(labels, [0, 1]))
                if not len(ids):
                    raise ValueError(f"{slide['name']} has no eligible tiles after exclusions")
                owners = resolved_owners(events)
                path = directory / f"{slide['id']}.npz"
                np.savez_compressed(path, tile_ids=ids, coordinates=data["coordinates"][ids], labels=labels[ids],
                    human_override=(human[ids] >= 0), source_event_ids=np.array([owners.get(int(i), {}).get("id", "") for i in ids], dtype="U32"))
                atomic_json(directory / f"{slide['id']}_events.json", events)
                records.append({**slide, **entry, "split": split, "labels_path": str(path),
                    "labels_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "feature_stamp": stamp,
                    "fingerprint": data["fingerprint"], "prediction_model_id": data["model_id"],
                    "prediction_threshold": data["prediction_threshold"], "tile_size": data["tile_size"],
                    "tiles": len(ids), "human_tiles": int((human[ids] >= 0).sum()),
                    "excluded_tiles": int((labels == -2).sum()), "class_counts": [int((labels[ids] == k).sum()) for k in (0, 1)]})
        counts = [sum(row["class_counts"][k] for row in records if row["split"] == "training") for k in (0, 1)]
        if not all(counts):
            raise ValueError("Training samples must contain both cortex and medulla tiles")
        snapshot = {"schema_version": 3, "mode": "selected_samples", "created_at": now(),
            "model_name": job["model_name"], "c_values": job["c_values"], "slides": records,
            "label_policy": "Saved predictions overridden by human corrections; excluded tiles omitted; all native feature tiles used",
            "evaluation_policy": "Selected validation samples only; reference includes predictions and human corrections"}
        atomic_json(directory / "snapshot.json", snapshot)
    return snapshot


def training_thumbnail(pool, slide, data, human, labels):
    image = Image.open(io.BytesIO(pool.thumbnail(slide["wsi_path"]))).convert("RGBA")
    image.thumbnail((480, 300))
    info = pool.info(slide["wsi_path"])
    sx, sy = image.width / info["width"], image.height / info["height"]
    overlay = Image.new("RGBA", image.size)
    draw = ImageDraw.Draw(overlay)
    for manual in (False, True):
        for i in np.flatnonzero((human != -1) == manual):
            x, y = data["coordinates"][i]
            box = (int(x*sx), int(y*sy), max(int(x*sx), int((x+data["tile_size"])*sx)-1), max(int(y*sy), int((y+data["tile_size"])*sy)-1))
            color = (82, 156, 255) if labels[i] == 0 else (241, 133, 130) if labels[i] == 1 else (170, 178, 185)
            draw.rectangle(box, fill=(*color, 135 if manual else 80), outline=(240, 250, 255, 220) if manual else None)
    output = io.BytesIO()
    Image.alpha_composite(image, overlay).convert("RGB").save(output, format="JPEG", quality=85)
    return output.getvalue()
