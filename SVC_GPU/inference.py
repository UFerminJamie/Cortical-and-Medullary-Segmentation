"""Run a saved GPU SVC on unseen PT features and create spatial outputs."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    DEFAULT_FEATURE_ROOT,
    DEFAULT_INFERENCE_BATCH_SIZE,
    DEFAULT_THRESHOLD,
)
from .data import load_sample_features
from .gpu_model import load_bundle, predict_gpu, write_json
from .manifest import load_inference_samples
from .outputs import (
    get_wsi_dimensions,
    prediction_dataframe,
    resolve_wsi_path,
    save_combined_geojson,
    save_mask_geojson,
    save_prediction_csv,
    save_thresholded_mask,
    save_wsi_visualizations,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="GPU inference with per-tile CSV, GeoJSON, mask, and optional WSI overlay.",
    )
    parser.add_argument("--xlsx", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, default=DEFAULT_FEATURE_ROOT)
    parser.add_argument("--model", type=Path, default=Path("SVC_GPU/artifacts/svc_pipeline.pkl"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--split",
        choices=["train", "val", "test", "test1", "test2", "test3"],
        default=None,
        help="Optional subset when the unseen workbook has a split column.",
    )
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_INFERENCE_BATCH_SIZE)
    parser.add_argument("--tile-size", type=int, default=None, help="Override PT tile_size metadata.")
    parser.add_argument("--mask-downsample", type=float, default=64.0)
    parser.add_argument("--visualization-downsample", type=float, default=64.0)
    parser.add_argument("--wsi-root", type=Path, default=None)
    parser.add_argument("--visualize", action="store_true")
    return parser


def _optional_row_value(row: object, name: str) -> object | None:
    return getattr(row, name, None)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    started = time.monotonic()
    output_dir = args.output_dir.expanduser().resolve()
    prediction_dir = output_dir / "predictions"
    geojson_dir = output_dir / "geojson"
    mask_dir = output_dir / "masks"
    visualization_dir = output_dir / "visualizations"
    for directory in (prediction_dir, geojson_dir, mask_dir):
        directory.mkdir(parents=True, exist_ok=True)
    if args.visualize:
        visualization_dir.mkdir(parents=True, exist_ok=True)

    samples_df = load_inference_samples(args.xlsx, args.feature_root, split=args.split)
    bundle = load_bundle(args.model)
    expected_dimension = int(bundle["metadata"].get("feature_dimension", 0))
    summaries: list[dict] = []

    for index, row in enumerate(samples_df.itertuples(index=False), start=1):
        sample = load_sample_features(row.pt_path, load_labels=False)
        if expected_dimension and sample.features.shape[1] != expected_dimension:
            raise ValueError(
                f"{sample.pt_path}: expected {expected_dimension} features, "
                f"found {sample.features.shape[1]}"
            )
        tile_size = args.tile_size or sample.tile_size
        print(
            f"[{index}/{len(samples_df)}] Predicting {sample.pt_path.stem} "
            f"({len(sample.features):,} tiles)",
            flush=True,
        )
        predictions, scores = predict_gpu(
            bundle,
            sample.features,
            batch_size=args.batch_size,
            threshold=args.threshold,
        )
        result_df = prediction_dataframe(
            sample_name=sample.sample_name,
            case_id=_optional_row_value(row, "case_id"),
            pt_path=sample.pt_path,
            coordinates=sample.coordinates,
            predictions=predictions,
            decision_scores=scores,
            threshold=args.threshold,
        )
        stem = sample.pt_path.stem
        csv_path = save_prediction_csv(result_df, prediction_dir, stem)
        geojson_path = save_combined_geojson(
            result_df,
            geojson_dir,
            stem,
            tile_size=tile_size,
            threshold=args.threshold,
        )
        mask_geojson_path = save_mask_geojson(
            result_df,
            geojson_dir,
            stem,
            tile_size=tile_size,
            threshold=args.threshold,
        )

        manifest_wsi = _optional_row_value(row, "wsi_path")
        if manifest_wsi is None or pd.isna(manifest_wsi) or str(manifest_wsi).strip() == "":
            manifest_wsi = _optional_row_value(row, "folder_path")
        wsi_path = resolve_wsi_path(
            sample_name=sample.sample_name,
            metadata_path=sample.wsi_path,
            manifest_value=manifest_wsi,
            wsi_root=args.wsi_root,
        )
        try:
            wsi_dimensions = get_wsi_dimensions(wsi_path)
        except RuntimeError:
            if args.visualize:
                raise
            print(
                f"Warning: tiffslide unavailable; {stem} mask extent will use tile coordinates",
                flush=True,
            )
            wsi_dimensions = None

        mask_path = save_thresholded_mask(
            result_df,
            mask_dir,
            stem,
            tile_size=tile_size,
            downsample=args.mask_downsample,
            threshold=args.threshold,
            wsi_dimensions=wsi_dimensions,
        )
        visualization_path = None
        if args.visualize:
            if wsi_path is None:
                raise FileNotFoundError(
                    f"Cannot visualize {stem}: no WSI was found in PT metadata, workbook, or --wsi-root"
                )
            visualization_path = save_wsi_visualizations(
                result_df,
                visualization_dir,
                stem,
                wsi_path=wsi_path,
                tile_size=tile_size,
                downsample=args.visualization_downsample,
                threshold=args.threshold,
            )

        cortical = int(np.sum(predictions == 0))
        medullary = int(np.sum(predictions == 1))
        summary = {
            "sample": stem,
            "tiles": int(len(predictions)),
            "cortical": cortical,
            "medullary": medullary,
            "csv": str(csv_path),
            "geojson": str(geojson_path),
            "mask_geojson": str(mask_geojson_path),
            "mask": str(mask_path),
            "visualization": str(visualization_path) if visualization_path else None,
            "wsi": str(wsi_path) if wsi_path else None,
        }
        summaries.append(summary)
        print(
            f"Saved {stem} | cortical={cortical:,} | medullary={medullary:,}",
            flush=True,
        )

    run_summary = {
        "model": str(args.model.expanduser().resolve()),
        "xlsx": str(args.xlsx.expanduser().resolve()),
        "feature_root": str(args.feature_root.expanduser().resolve()),
        "threshold": float(args.threshold),
        "elapsed_seconds": float(time.monotonic() - started),
        "samples": summaries,
    }
    write_json(run_summary, output_dir / "inference_summary.json")
    print(f"Inference complete: {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
