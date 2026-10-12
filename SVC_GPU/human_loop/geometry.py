from __future__ import annotations

import numpy as np


def filter_mask(scores: np.ndarray, labels: np.ndarray, manual: np.ndarray, filters: dict,
                predictions: np.ndarray | None = None) -> np.ndarray:
    source = filters.get("source", "combined")
    if source == "prediction":
        if predictions is None:
            raise ValueError("Prediction layer requires model labels")
        labels = predictions
        manual = np.full(len(manual), -1, dtype=np.int8)
    elif source == "human":
        labels = manual
    elif source != "combined":
        raise ValueError("Overlay source must be combined, prediction, or human")
    keep = np.zeros(len(scores), dtype=bool)
    for class_id, key in ((0, "cortex"), (1, "medulla")):
        config = filters.get(key, {})
        if not config.get("enabled", True):
            continue
        threshold = float(config.get("threshold", 0.0))
        if not np.isfinite(threshold):
            raise ValueError("Filter thresholds must be finite")
        selected = labels == class_id
        comparison = config.get("comparison", "lte" if class_id == 0 else "gte")
        if comparison == "lte":
            selected &= scores <= threshold
        elif comparison == "gte":
            selected &= scores >= threshold
        elif comparison != "all":
            raise ValueError("Filter comparison must be lte, gte or all")
        if filters.get("keep_manual", True):
            selected |= (labels == class_id) & (manual >= 0)
        keep |= selected
    if source != "prediction" and filters.get("excluded_enabled", True):
        keep |= manual == -2
    return keep


def region_indices(coordinates: np.ndarray, tile_size: int, polygon: list, eligible: np.ndarray | None = None) -> np.ndarray:
    points = np.asarray(polygon, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or not 3 <= len(points) <= 512 or not np.isfinite(points).all():
        raise ValueError("Region needs 3–512 finite level-zero [x, y] points")
    area = abs(np.sum(points[:, 0] * np.roll(points[:, 1], -1) - points[:, 1] * np.roll(points[:, 0], -1))) / 2
    if area < 1:
        raise ValueError("The selected region is too small")
    centers = coordinates.astype(np.float64) + tile_size / 2
    minimum, maximum = points.min(axis=0), points.max(axis=0)
    selected = ((centers >= minimum).all(axis=1) & (centers <= maximum).all(axis=1))
    if eligible is not None:
        selected &= eligible
    ids = np.flatnonzero(selected)
    x, y = centers[ids].T
    inside, boundary = np.zeros(len(ids), dtype=bool), np.zeros(len(ids), dtype=bool)
    for a, b in zip(points, np.roll(points, -1, axis=0)):
        crosses = (a[1] > y) != (b[1] > y)
        denominator = b[1] - a[1]
        intersection = (b[0] - a[0]) * (y - a[1]) / (denominator if denominator else 1) + a[0]
        inside ^= crosses & (x < intersection)
        cross = (x - a[0]) * (b[1] - a[1]) - (y - a[1]) * (b[0] - a[0])
        boundary |= (np.abs(cross) <= 1e-6) & (x >= min(a[0], b[0])) & (x <= max(a[0], b[0])) & (y >= min(a[1], b[1])) & (y <= max(a[1], b[1]))
    return ids[inside | boundary]
