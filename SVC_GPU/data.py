"""Load UNI feature tensors, labels, and level-zero tile coordinates."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


@dataclass(frozen=True)
class SampleFeatures:
    sample_name: str
    pt_path: Path
    features: torch.Tensor
    coordinates: np.ndarray
    tile_size: int
    wsi_path: Path | None
    labels: torch.Tensor | None = None


@dataclass(frozen=True)
class LabeledDataset:
    features: torch.Tensor
    labels: torch.Tensor
    sample_names: np.ndarray
    coordinates: np.ndarray
    pt_paths: np.ndarray


def _as_feature_tensor(value: Any, pt_path: Path) -> torch.Tensor:
    features = value.get("features") if isinstance(value, dict) else value
    if not isinstance(features, torch.Tensor):
        raise TypeError(f"Expected a tensor or dict['features'] in {pt_path}")
    if features.ndim != 2:
        raise ValueError(f"Expected 2D features in {pt_path}; found {tuple(features.shape)}")
    if features.shape[0] == 0:
        raise ValueError(f"Feature tensor is empty: {pt_path}")
    return features.detach().to(device="cpu", dtype=torch.float32).contiguous()


def _coordinates_from_payload(payload: Any, row_count: int) -> np.ndarray | None:
    if not isinstance(payload, dict):
        return None
    coordinates = payload.get("coordinates")
    if coordinates is None:
        coordinates = payload.get("coords")
    if coordinates is None:
        return None
    if isinstance(coordinates, torch.Tensor):
        coordinates = coordinates.detach().cpu().numpy()
    coordinates = np.asarray(coordinates)
    if coordinates.shape != (row_count, 2):
        raise ValueError(
            f"Expected coordinates with shape ({row_count}, 2); found {coordinates.shape}"
        )
    if not np.isfinite(coordinates).all():
        raise ValueError("Tile coordinates contain non-finite values")
    return coordinates.astype(np.int64, copy=False)


def _read_coordinate_csv(csv_path: Path, row_count: int) -> tuple[pd.DataFrame, np.ndarray]:
    if not csv_path.is_file():
        raise FileNotFoundError(f"Coordinate CSV does not exist: {csv_path}")
    df = pd.read_csv(csv_path)
    missing = sorted({"x", "y"}.difference(df.columns))
    if missing:
        raise ValueError(f"Missing coordinate columns in {csv_path}: {missing}")
    if len(df) != row_count:
        raise ValueError(
            f"{csv_path.stem}: CSV rows={len(df)}, PT rows={row_count}"
        )
    coordinates = df[["x", "y"]].to_numpy(dtype=np.int64)
    return df, coordinates


def load_sample_features(
    pt_path: str | Path,
    *,
    csv_path: str | Path | None = None,
    load_labels: bool = False,
) -> SampleFeatures:
    """Load one sample. Coordinates come from PT metadata, with CSV fallback."""
    pt_path = Path(pt_path).expanduser().resolve()
    if not pt_path.is_file():
        raise FileNotFoundError(f"PT feature file does not exist: {pt_path}")

    payload = torch.load(pt_path, map_location="cpu", weights_only=False)
    features = _as_feature_tensor(payload, pt_path)
    coordinates = _coordinates_from_payload(payload, len(features))
    label_df: pd.DataFrame | None = None

    resolved_csv = Path(csv_path).expanduser().resolve() if csv_path else pt_path.with_suffix(".csv")
    if coordinates is None or load_labels:
        label_df, csv_coordinates = _read_coordinate_csv(resolved_csv, len(features))
        if coordinates is None:
            coordinates = csv_coordinates
        elif not np.array_equal(coordinates, csv_coordinates):
            raise ValueError(f"PT and CSV coordinates do not match for {pt_path.stem}")

    assert coordinates is not None
    labels: torch.Tensor | None = None
    if load_labels:
        assert label_df is not None
        if "label" not in label_df.columns:
            raise ValueError(f"Missing label column in {resolved_csv}")
        raw_labels = pd.to_numeric(label_df["label"], errors="raise").to_numpy(dtype=np.int64)
        unexpected = sorted(set(np.unique(raw_labels).tolist()).difference({-1, 0, 1, 2}))
        if unexpected:
            raise ValueError(f"Unexpected labels in {resolved_csv}: {unexpected}")
        keep = (raw_labels == 1) | (raw_labels == 2)
        if not keep.any():
            raise ValueError(f"No labeled cortex/medulla tiles remain in {resolved_csv}")
        features = features[torch.from_numpy(keep)]
        coordinates = coordinates[keep]
        # Only tissue labels are eligible for cortical/medullary training.
        encoded = raw_labels[keep] - 1
        labels = torch.from_numpy(encoded.astype(np.int64, copy=False))

    metadata = payload if isinstance(payload, dict) else {}
    nested_metadata = metadata.get("metadata", {})
    if not isinstance(nested_metadata, dict):
        nested_metadata = {}
    read_plan = nested_metadata.get("read_plan", {})
    if not isinstance(read_plan, dict):
        read_plan = {}
    tile_value = metadata.get("tile_size") or read_plan.get("patch_size_level0") or 512
    if isinstance(tile_value, (tuple, list)):
        if len(tile_value) != 2 or tile_value[0] != tile_value[1]:
            raise ValueError(f"Expected square level-zero tiles in {pt_path}; found {tile_value}")
        tile_value = tile_value[0]
    tile_size = int(tile_value)
    if tile_size <= 0:
        raise ValueError(f"Invalid tile_size={tile_size} in {pt_path}")
    wsi_value = metadata.get("svs_path") or metadata.get("wsi_path") or nested_metadata.get("slide_path")
    wsi_path = Path(str(wsi_value)).expanduser() if wsi_value else None
    sample_name = str(metadata.get("slide_name") or nested_metadata.get("slide_name") or pt_path.stem)

    return SampleFeatures(
        sample_name=sample_name,
        pt_path=pt_path,
        features=features,
        coordinates=coordinates,
        tile_size=tile_size,
        wsi_path=wsi_path,
        labels=labels,
    )


def load_labeled_dataset(samples_df: pd.DataFrame) -> LabeledDataset:
    """Load and concatenate all labeled samples in a manifest DataFrame."""
    if samples_df.empty:
        raise ValueError("Cannot load an empty sample DataFrame")

    all_features: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    all_names: list[np.ndarray] = []
    all_coordinates: list[np.ndarray] = []
    all_paths: list[np.ndarray] = []

    for row in samples_df.itertuples(index=False):
        sample = load_sample_features(
            row.pt_path,
            csv_path=row.csv_path,
            load_labels=True,
        )
        assert sample.labels is not None
        count = len(sample.labels)
        cortical = int((sample.labels == 0).sum())
        medullary = int((sample.labels == 1).sum())
        print(
            f"{sample.sample_name} | tiles={count:,} | "
            f"cortical={cortical:,} | medullary={medullary:,}",
            flush=True,
        )
        all_features.append(sample.features)
        all_labels.append(sample.labels)
        all_names.append(np.full(count, sample.sample_name, dtype=object))
        all_coordinates.append(sample.coordinates)
        all_paths.append(np.full(count, str(sample.pt_path), dtype=object))

    return LabeledDataset(
        features=torch.cat(all_features, dim=0),
        labels=torch.cat(all_labels, dim=0),
        sample_names=np.concatenate(all_names),
        coordinates=np.concatenate(all_coordinates, axis=0),
        pt_paths=np.concatenate(all_paths),
    )
