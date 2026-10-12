"""cuML scaler/PCA/SVC fitting, persistence, and batched inference."""

from __future__ import annotations

import json
import os
import pickle
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch


def require_gpu() -> tuple[Any, Any, Any, Any]:
    """Import RAPIDS lazily and fail with an actionable CUDA error."""
    try:
        import cupy as cp
        from cuml.decomposition import PCA
        from cuml.preprocessing import StandardScaler
        from cuml.svm import SVC

        device_count = cp.cuda.runtime.getDeviceCount()
        if device_count < 1:
            raise RuntimeError("No CUDA devices are visible")
        cp.cuda.Device(0).use()
        return cp, PCA, StandardScaler, SVC
    except Exception as error:
        raise RuntimeError(
            "A working CUDA GPU and the RAPIDS environment are required. "
            "Run this command on a GPU compute node with the rapids_svc conda environment."
        ) from error


def gpu_details(cp: Any) -> dict[str, Any]:
    device_id = int(cp.cuda.runtime.getDevice())
    properties = cp.cuda.runtime.getDeviceProperties(device_id)
    name = properties.get("name", "unknown")
    if isinstance(name, bytes):
        name = name.decode(errors="replace")
    return {
        "device_id": device_id,
        "name": str(name),
        "total_memory_bytes": int(properties.get("totalGlobalMem", 0)),
    }


def train_gpu_pipeline(
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    n_components: int,
    c_value: float,
    cache_size_mb: float,
) -> dict[str, Any]:
    cp, PCA, StandardScaler, SVC = require_gpu()

    if features.ndim != 2:
        raise ValueError(f"features must be 2D; found {tuple(features.shape)}")
    maximum_components = min(features.shape)
    if not 1 <= n_components <= maximum_components:
        raise ValueError(
            f"n_components={n_components} exceeds min(samples, features)={maximum_components}"
        )
    unique_labels = torch.unique(labels).tolist()
    if unique_labels != [0, 1]:
        raise ValueError(f"Training requires both encoded classes [0, 1]; found {unique_labels}")

    x_gpu = cp.asarray(features.numpy(), dtype=cp.float32)
    y_gpu = cp.asarray(labels.numpy(), dtype=cp.int32)

    scaler = StandardScaler(output_type="cupy")
    x_scaled = scaler.fit_transform(x_gpu)
    del x_gpu

    pca = PCA(n_components=n_components, whiten=False, output_type="cupy")
    x_reduced = pca.fit_transform(x_scaled)
    del x_scaled

    svc = SVC(
        C=c_value,
        kernel="rbf",
        gamma="scale",
        class_weight="balanced",
        cache_size=cache_size_mb,
        output_type="cupy",
    )
    svc.fit(x_reduced, y_gpu)
    del x_reduced, y_gpu

    cp.get_default_memory_pool().free_all_blocks()
    return {
        "format_version": 1,
        "scaler": scaler,
        "pca": pca,
        "svc": svc,
        "gpu": gpu_details(cp),
    }


def predict_gpu(
    bundle: dict[str, Any],
    features: torch.Tensor,
    *,
    batch_size: int,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return thresholded class predictions and signed SVC decision scores."""
    cp, _, _, _ = require_gpu()
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    predictions: list[np.ndarray] = []
    scores: list[np.ndarray] = []
    for start in range(0, len(features), batch_size):
        stop = min(start + batch_size, len(features))
        x_gpu = cp.asarray(features[start:stop].numpy(), dtype=cp.float32)
        x_scaled = bundle["scaler"].transform(x_gpu)
        x_reduced = bundle["pca"].transform(x_scaled)
        score_gpu = bundle["svc"].decision_function(x_reduced)
        score = cp.asnumpy(score_gpu).reshape(-1).astype(np.float32, copy=False)
        prediction = (score >= threshold).astype(np.int32)
        scores.append(score)
        predictions.append(prediction)
        del x_gpu, x_scaled, x_reduced, score_gpu

    return np.concatenate(predictions), np.concatenate(scores)


def save_bundle(
    bundle: dict[str, Any],
    output_path: str | Path,
    metadata: dict[str, Any],
) -> Path:
    """Atomically save one self-contained scaler/PCA/SVC artifact."""
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    complete_bundle = dict(bundle)
    complete_bundle["metadata"] = metadata
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary_path.open("wb") as handle:
        pickle.dump(complete_bundle, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary_path, output_path)
    return output_path


def load_bundle(model_path: str | Path) -> dict[str, Any]:
    model_path = Path(model_path).expanduser().resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"Model artifact does not exist: {model_path}")
    # Initialize CUDA before unpickling RAPIDS estimators.
    require_gpu()
    with model_path.open("rb") as handle:
        bundle = pickle.load(handle)
    required = {"scaler", "pca", "svc", "metadata"}
    missing = sorted(required.difference(bundle))
    if missing:
        raise ValueError(f"Model artifact is missing keys: {missing}")
    return bundle


def base_metadata() -> dict[str, Any]:
    """Capture reproducibility information without requiring CUDA imports."""
    import pandas as pd
    import sklearn

    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "torch": torch.__version__,
        "sklearn": sklearn.__version__,
    }


def write_json(payload: dict[str, Any], output_path: str | Path) -> Path:
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary_path, output_path)
    return output_path
