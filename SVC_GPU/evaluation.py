"""Metrics and tabular prediction helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .config import LABEL_NAMES


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    decision_scores: np.ndarray,
) -> dict:
    """Report both classes as one-vs-rest using signed medullary SVC scores."""
    y_true = np.asarray(y_true, dtype=np.int32).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.int32).reshape(-1)
    decision_scores = np.asarray(decision_scores, dtype=np.float64).reshape(-1)
    if not (len(y_true) == len(y_pred) == len(decision_scores)) or not len(y_true):
        raise ValueError("Labels, predictions, and scores must have equal nonzero length")
    if not np.isin(y_true, [0, 1]).all() or not np.isin(y_pred, [0, 1]).all():
        raise ValueError("Labels and predictions must be encoded as 0 or 1")
    if not np.isfinite(decision_scores).all():
        raise ValueError("Decision scores must be finite")

    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    per_class = {}
    for class_id, class_name in LABEL_NAMES.items():
        true_positive = int(matrix[class_id, class_id])
        false_negative = int(matrix[class_id, 1 - class_id])
        false_positive = int(matrix[1 - class_id, class_id])
        true_negative = int(matrix[1 - class_id, 1 - class_id])
        class_scores = decision_scores if class_id == 1 else -decision_scores
        per_class[class_name] = {
            "accuracy": float(accuracy_score(y_true == class_id, y_pred == class_id)),
            "f1_score": float(f1_score(y_true, y_pred, pos_label=class_id, zero_division=0)),
            "sensitivity": (
                true_positive / (true_positive + false_negative)
                if true_positive + false_negative else None
            ),
            "specificity": (
                true_negative / (true_negative + false_positive)
                if true_negative + false_positive else None
            ),
            "auc": (
                float(roc_auc_score(y_true == class_id, class_scores))
                if np.unique(y_true).size == 2 else None
            ),
            "support": true_positive + false_negative,
        }
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_medullary": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_medullary": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_medullary": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": matrix.tolist(),
        "per_class": per_class,
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=[LABEL_NAMES[0], LABEL_NAMES[1]],
            output_dict=True,
            zero_division=0,
        ),
    }


def labeled_predictions_dataframe(
    *,
    sample_names: np.ndarray,
    coordinates: np.ndarray,
    pt_paths: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    decision_scores: np.ndarray,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_name": sample_names,
            "pt_path": pt_paths,
            "x": coordinates[:, 0],
            "y": coordinates[:, 1],
            "true_label_encoded": y_true,
            "true_label": [LABEL_NAMES[int(value)] for value in y_true],
            "predicted_label_encoded": y_pred,
            "predicted_label": [LABEL_NAMES[int(value)] for value in y_pred],
            "decision_score": decision_scores,
            "absolute_decision_score": np.abs(decision_scores),
            "correct_prediction": y_true == y_pred,
        }
    )


def save_confusion_matrix(metrics: dict, output_path: str | Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(6, 6))
    display = ConfusionMatrixDisplay(
        confusion_matrix=np.asarray(metrics["confusion_matrix"]),
        display_labels=["Cortical", "Medullary"],
    )
    display.plot(ax=axis, values_format="d", colorbar=False)
    axis.set_title("Test confusion matrix")
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return output_path
