"""Shared correction records for drawn regions and full-slide annotations."""
from __future__ import annotations

import numpy as np


def correction_record(slide, data, labels, ids, payload, polygon):
    action = payload.get("action", "label")
    shape = payload.get("shape", "polygon")
    if shape not in ("rectangle", "polygon", "freehand", "whole_slide"):
        raise ValueError("Unknown annotation tool")
    return {"action": action, "label": payload["label"] if action == "label" else -1,
        "polygon": polygon, "shape": shape,
        "sample_name": slide["name"], "wsi_path": slide["wsi_path"], "feature_path": slide["pt_path"],
        "tile_ids": ids.tolist(), "tile_coordinates": data["coordinates"][ids].tolist(),
        "prior_labels": labels[ids].tolist(), "predicted_labels": data["predictions"][ids].tolist(),
        "decision_scores": data["scores"][ids].astype(float).tolist(),
        "reviewer": payload["reviewer"], "pass_id": payload["pass_id"], "note": str(payload.get("note", ""))[:2000],
        "fingerprint": data["fingerprint"], "model_id": data["model_id"], "prediction_id": data["prediction_id"],
        "prediction_threshold": data["prediction_threshold"], "filters": payload.get("filters", {}),
        "display_threshold": payload.get('display_threshold', data['prediction_threshold']),
        "visible_only": payload.get("visible_only", True),
        "grid_mode": str(payload.get("grid_mode", "native")), "source_overlap": str(payload.get("source_overlap", "auto")),
        "grid_origin": data.get("grid", {}).get("origin"),
        "selection_rule": ("all_feature_tiles" if str(payload.get("grid_mode", "native")) == "native" else "selected_grid_tiles") if shape == "whole_slide" else "tile_center_inside_region",
        "tile_size_level0": data["tile_size"], **({"bulk_id": payload["bulk_id"]} if payload.get("bulk_id") else {})}


def full_slide_polygon(data):
    minimum = data["coordinates"].min(axis=0)
    maximum = data["coordinates"].max(axis=0) + data["tile_size"]
    x0, y0 = minimum.astype(float)
    x1, y1 = maximum.astype(float)
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
