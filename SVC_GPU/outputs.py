"""Per-tile CSV, Athena annotation uploads, masks, and WSI visualizations."""

from __future__ import annotations

import json
import os
import secrets
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from .config import LABEL_NAMES

CLASS_COLORS = {
    0: (0, 102, 255),
    1: (255, 0, 0),
}


def prediction_dataframe(
    *,
    sample_name: str,
    pt_path: Path,
    coordinates: np.ndarray,
    predictions: np.ndarray,
    decision_scores: np.ndarray,
    threshold: float,
    case_id: object | None = None,
) -> pd.DataFrame:
    predictions = np.asarray(predictions, dtype=np.int32).reshape(-1)
    decision_scores = np.asarray(decision_scores, dtype=np.float32).reshape(-1)
    coordinates = np.asarray(coordinates, dtype=np.int64)
    if not (len(coordinates) == len(predictions) == len(decision_scores)):
        raise ValueError("Coordinates, predictions, and decision scores must have equal length")
    return pd.DataFrame(
        {
            "sample_name": sample_name,
            "case_id": "" if case_id is None or pd.isna(case_id) else str(case_id),
            "pt_path": str(pt_path),
            "tile_name": [f"{x}_{y}" for x, y in coordinates],
            "x": coordinates[:, 0],
            "y": coordinates[:, 1],
            "predicted_label_encoded": predictions,
            "predicted_label": [LABEL_NAMES[int(value)] for value in predictions],
            "decision_score": decision_scores,
            "absolute_decision_score": np.abs(decision_scores),
            "threshold": float(threshold),
        }
    )


def save_prediction_csv(df: pd.DataFrame, output_dir: str | Path, stem: str) -> Path:
    output_path = Path(output_dir) / f"{stem}.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path


def _scaled_confidences(df: pd.DataFrame, threshold: float) -> np.ndarray:
    scores = df["decision_score"].to_numpy(dtype=np.float64)
    labels = df["predicted_label_encoded"].to_numpy(dtype=np.int32)
    distances = np.abs(scores - threshold)
    scaled = np.zeros(len(df), dtype=np.float64)
    for class_id in (0, 1):
        selected = labels == class_id
        if not selected.any():
            continue
        maximum = float(distances[selected].max())
        scaled[selected] = 1.0 if np.isclose(maximum, 0.0) else distances[selected] / maximum
    return np.clip(scaled, 0.0, 1.0)


def save_combined_geojson(
    df: pd.DataFrame,
    output_dir: str | Path,
    stem: str,
    *,
    tile_size: int,
    threshold: float,
) -> Path:
    """Save two Athena annotations in one uploadable file.

    Athena assigns each annotation its own database ID on import. A JSON list
    is required for this; a GeoJSON FeatureCollection imports as one annotation.
    """
    output_path = Path(output_dir) / f"{stem}.geojson"
    scaled = _scaled_confidences(df, threshold)
    elements: dict[int, list[dict[str, Any]]] = {0: [], 1: []}

    for row, confidence in zip(df.itertuples(index=False), scaled):
        x = int(row.x)
        y = int(row.y)
        class_id = int(row.predicted_label_encoded)
        class_name = LABEL_NAMES[class_id]
        red, green, blue = CLASS_COLORS[class_id]
        alpha = float(max(0.05, confidence))
        properties = {
            "class_id": class_id,
            "class_name": class_name,
            "classification": {
                "name": class_name,
                "color": [red, green, blue],
            },
            "decision_score": float(row.decision_score),
            "absolute_decision_score": float(row.absolute_decision_score),
            "scaled_confidence": float(confidence),
            "threshold": float(threshold),
            "tile_x": x,
            "tile_y": y,
            "tile_size": int(tile_size),
        }
        elements[class_id].append(
            {
                "id": secrets.token_hex(12),
                "type": "rectangle",
                "center": [x + tile_size / 2, y + tile_size / 2, 0],
                "width": tile_size,
                "height": tile_size,
                "rotation": 0,
                "group": class_name,
                "fillColor": f"rgba({red}, {green}, {blue}, {alpha})",
                "lineColor": "rgba(0, 0, 0, 0)",
                "lineWidth": 0,
                "user": properties,
            }
        )

    payload = _annotation_documents(elements, stem, "heatmap", tile_size, threshold)
    return _write_annotation_file(output_path, payload)


def save_mask_geojson(
    df: pd.DataFrame,
    output_dir: str | Path,
    stem: str,
    *,
    tile_size: int,
    threshold: float,
) -> Path:
    """Save cortex and medullary thresholded masks as separate Athena layers."""
    elements: dict[int, list[dict[str, Any]]] = {0: [], 1: []}
    for row in df.itertuples(index=False):
        x, y = int(row.x), int(row.y)
        class_id = int(row.predicted_label_encoded)
        red, green, blue = CLASS_COLORS[class_id]
        elements[class_id].append({
            "id": secrets.token_hex(12),
            "type": "rectangle",
            "center": [x + tile_size / 2, y + tile_size / 2, 0],
            "width": tile_size,
            "height": tile_size,
            "rotation": 0,
            "group": LABEL_NAMES[class_id],
            "fillColor": f"rgba({red}, {green}, {blue}, 1)",
            "lineColor": "rgba(0, 0, 0, 0)",
            "lineWidth": 0,
            "user": {"class_id": class_id, "class_name": LABEL_NAMES[class_id],
                     "tile_x": x, "tile_y": y, "tile_size": int(tile_size),
                     "threshold": float(threshold)},
        })
    payload = _annotation_documents(elements, stem, "mask", tile_size, threshold)
    return _write_annotation_file(Path(output_dir) / f"{stem}_mask.geojson", payload)


def _annotation_documents(
    elements: dict[int, list[dict[str, Any]]],
    stem: str,
    kind: str,
    tile_size: int,
    threshold: float,
) -> list[dict[str, Any]]:
    documents = []
    for class_id, layer_name in ((0, "cortex"), (1, "medullary")):
        documents.append({"annotation": {
            "name": f"{layer_name}_{kind}" if kind == "heatmap" else f"{layer_name}_mask",
            "description": f"{layer_name} {kind} for {stem}; signed SVC threshold {threshold:g}",
            "attributes": {"params": {
                "sampleName": stem,
                "class": LABEL_NAMES[class_id],
                "effectiveThreshold": float(threshold),
                "tileSize": int(tile_size),
                "numberOfTiles": len(elements[class_id]),
            }},
            "elements": elements[class_id],
        }})
    return documents


def _write_annotation_file(output_path: Path, payload: list[dict[str, Any]]) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, separators=(",", ":"), allow_nan=False)
    os.replace(temporary_path, output_path)
    return output_path


def resolve_wsi_path(
    *,
    sample_name: str,
    metadata_path: Path | None,
    manifest_value: object | None,
    wsi_root: str | Path | None,
) -> Path | None:
    candidates: list[Path] = []
    if metadata_path is not None:
        candidates.append(metadata_path)
    if manifest_value is not None and not pd.isna(manifest_value):
        candidates.append(Path(str(manifest_value)).expanduser())

    extensions = (".svs", ".tif", ".tiff", ".ndpi")
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
        if candidate.is_dir():
            for extension in extensions:
                nested = candidate / f"{sample_name}{extension}"
                if nested.is_file():
                    return nested.resolve()

    if wsi_root is not None:
        root = Path(wsi_root).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"WSI root does not exist: {root}")
        matches: list[Path] = []
        for extension in extensions:
            matches.extend(root.rglob(f"{sample_name}{extension}"))
        unique_matches = sorted({path.resolve() for path in matches})
        if len(unique_matches) > 1:
            raise ValueError(f"Multiple WSIs matched {sample_name}: {unique_matches}")
        if unique_matches:
            return unique_matches[0]
    return None


def _slide_dimensions(wsi_path: Path | None) -> tuple[int, int] | None:
    if wsi_path is None:
        return None
    try:
        from tiffslide import TiffSlide
    except ImportError as error:
        raise RuntimeError("tiffslide is required when a WSI path is available") from error
    with TiffSlide(str(wsi_path)) as slide:
        return tuple(map(int, slide.dimensions))


def _output_dimensions(
    df: pd.DataFrame,
    *,
    tile_size: int,
    downsample: float,
    wsi_dimensions: tuple[int, int] | None,
) -> tuple[int, int]:
    if downsample <= 0:
        raise ValueError("downsample must be positive")
    if wsi_dimensions is None:
        width = int(df["x"].max()) + tile_size
        height = int(df["y"].max()) + tile_size
    else:
        width, height = wsi_dimensions
    return int(np.ceil(width / downsample)), int(np.ceil(height / downsample))


def save_thresholded_mask(
    df: pd.DataFrame,
    output_dir: str | Path,
    stem: str,
    *,
    tile_size: int,
    downsample: float,
    threshold: float,
    wsi_dimensions: tuple[int, int] | None,
) -> Path:
    """Save a palette PNG: 0=no tile, 1=cortical, 2=medullary."""
    output_path = Path(output_dir) / f"{stem}.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    width, height = _output_dimensions(
        df,
        tile_size=tile_size,
        downsample=downsample,
        wsi_dimensions=wsi_dimensions,
    )
    mask = _mask_array(
        df, width=width, height=height, tile_size=tile_size,
        downsample=downsample, threshold=threshold,
    )
    image = Image.fromarray(mask, mode="P")
    palette = [0] * 768
    palette[3:6] = list(CLASS_COLORS[0])
    palette[6:9] = list(CLASS_COLORS[1])
    image.putpalette(palette)
    image.info["transparency"] = 0
    # image.save(output_path, optimize=True)        # dont save mask
    return output_path


def _mask_array(
    df: pd.DataFrame,
    *,
    width: int,
    height: int,
    tile_size: int,
    downsample: float,
    threshold: float,
) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    for row in df.itertuples(index=False):
        x0 = max(0, int(np.floor(row.x / downsample)))
        y0 = max(0, int(np.floor(row.y / downsample)))
        x1 = min(width, int(np.ceil((row.x + tile_size) / downsample)))
        y1 = min(height, int(np.ceil((row.y + tile_size) / downsample)))
        class_value = 2 if float(row.decision_score) >= threshold else 1
        mask[y0:y1, x0:x1] = class_value
    return mask


def _score_map(
    df: pd.DataFrame,
    *,
    width: int,
    height: int,
    tile_size: int,
    downsample: float,
) -> np.ndarray:
    score_sum = np.zeros((height, width), dtype=np.float32)
    score_count = np.zeros((height, width), dtype=np.uint16)
    for row in df.itertuples(index=False):
        x0 = max(0, int(np.floor(row.x / downsample)))
        y0 = max(0, int(np.floor(row.y / downsample)))
        x1 = min(width, int(np.ceil((row.x + tile_size) / downsample)))
        y1 = min(height, int(np.ceil((row.y + tile_size) / downsample)))
        score_sum[y0:y1, x0:x1] += float(row.decision_score)
        score_count[y0:y1, x0:x1] += 1
    result = np.full((height, width), np.nan, dtype=np.float32)
    valid = score_count > 0
    result[valid] = score_sum[valid] / score_count[valid]
    return result


def save_wsi_visualizations(
    df: pd.DataFrame,
    output_dir: str | Path,
    stem: str,
    *,
    wsi_path: Path,
    tile_size: int,
    downsample: float,
    threshold: float,
    alpha: float = 0.55,
) -> Path:
    try:
        from tiffslide import TiffSlide
    except ImportError as error:
        raise RuntimeError("tiffslide is required for --visualize") from error
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap, TwoSlopeNorm
    from matplotlib.patches import Patch

    visualization_path = Path(output_dir) / f"{stem}_visualization.png"
    visualization_path.parent.mkdir(parents=True, exist_ok=True)
    with TiffSlide(str(wsi_path)) as slide:
        wsi_width, wsi_height = map(int, slide.dimensions)
        width = int(np.ceil(wsi_width / downsample))
        height = int(np.ceil(wsi_height / downsample))
        thumbnail = np.asarray(
            slide.get_thumbnail((width, height)).convert("RGB").resize(
                (width, height), Image.Resampling.BILINEAR
            )
        )

    score_map = _score_map(
        df,
        width=thumbnail.shape[1],
        height=thumbnail.shape[0],
        tile_size=tile_size,
        downsample=downsample,
    )
    finite = np.abs(score_map[np.isfinite(score_map)] - threshold)
    limit = float(np.percentile(finite, 99)) if finite.size else 1.0
    if not np.isfinite(limit) or np.isclose(limit, 0.0):
        limit = 1.0

    mask = _mask_array(
        df,
        width=width,
        height=height,
        tile_size=tile_size,
        downsample=downsample,
        threshold=threshold,
    )
    figure, axes = plt.subplots(1, 3, figsize=(24, 8))
    axes[0].imshow(thumbnail)
    axes[0].set_title(f"{stem} WSI")
    axes[0].axis("off")
    axes[1].imshow(thumbnail)
    heatmap = axes[1].imshow(
        score_map,
        cmap="coolwarm",
        alpha=alpha,
        norm=TwoSlopeNorm(vmin=threshold - limit, vcenter=threshold, vmax=threshold + limit),
    )
    axes[1].set_title("SVC heatmap (blue=cortical, red=medullary)")
    axes[1].axis("off")
    colorbar = figure.colorbar(heatmap, ax=axes[1], fraction=0.046, pad=0.04)
    colorbar.set_label("Signed SVC decision score")
    axes[2].imshow(thumbnail)
    axes[2].imshow(
        np.ma.masked_equal(mask, 0),
        cmap=ListedColormap(["#0066ff", "#ff0000"]),
        norm=BoundaryNorm([0.5, 1.5, 2.5], 2),
        alpha=alpha,
        interpolation="nearest",
    )
    axes[2].set_title("Thresholded mask (blue=cortical, red=medullary)")
    axes[2].legend(
        handles=[
            Patch(color="#0066ff", label="Cortical"),
            Patch(color="#ff0000", label="Medullary"),
        ],
        loc="lower right",
    )
    axes[2].axis("off")
    figure.tight_layout()
    figure.savefig(visualization_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    for suffix in ("heatmap", "mask"):
        (visualization_path.parent / f"{stem}_{suffix}.png").unlink(missing_ok=True)
    return visualization_path


def get_wsi_dimensions(wsi_path: Path | None) -> tuple[int, int] | None:
    return _slide_dimensions(wsi_path)
