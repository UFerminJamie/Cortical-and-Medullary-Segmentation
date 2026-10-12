"""Create an all-test inference workbook from Trident embeddings and source slides."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd



from .config import LOCAL_SETTINGS

DEFAULT_FEATURE_ROOT = Path(LOCAL_SETTINGS.get("prepare_features", "data/features"))
DEFAULT_SLIDE_ROOTS = [Path(path) for path in LOCAL_SETTINGS.get("prepare_slide_roots", ["data/slides"])]

DEFAULT_OUTPUT = Path("SVC_GPU/UF_CKD_all_test.xlsx")
SLIDE_EXTENSIONS = {".tif", ".tiff", ".svs", ".ndpi", ".mrxs", ".scn", ".vms", ".vmu", ".bif"}

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-root", type=Path, default=DEFAULT_FEATURE_ROOT)
    parser.add_argument("--slide-root", type=Path, action="append", dest="slide_roots")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    feature_root = args.feature_root.expanduser().resolve()
    slide_roots = args.slide_roots or DEFAULT_SLIDE_ROOTS
    if not feature_root.is_dir():
        parser.error(f"Feature directory does not exist: {feature_root}")
    missing_roots = [str(root) for root in slide_roots if not root.is_dir()]
    if missing_roots:
        parser.error(f"Slide directories do not exist: {missing_roots}")

    features = sorted(feature_root.glob("*.pt"))
    if not features:
        parser.error(f"No .pt files in {feature_root}")

    slides: dict[str, list[Path]] = {}
    for root in slide_roots:
        for slide in root.rglob("*"):
            if slide.suffix.lower() not in SLIDE_EXTENSIONS or not slide.is_file():
                continue
            slides.setdefault(slide.stem.casefold(), []).append(slide.resolve())

    rows = []
    missing = []
    ambiguous = []
    for pt in features:
        matches = sorted(set(slides.get(pt.stem.casefold(), [])))
        if not matches:
            missing.append(pt.name)
            continue
        if len(matches) > 1:
            import torch

            payload = torch.load(pt, map_location="cpu", weights_only=False)
            slide_path = payload.get("metadata", {}).get("slide_path") if isinstance(payload, dict) else None
            matches = [path for path in matches if str(path) == str(slide_path)]
            del payload
            if len(matches) != 1:
                ambiguous.append((pt.name, [str(path) for path in matches]))
                continue
        slide = matches[0]
        rows.append({
            "sample_name": slide.name,
            "case_id": "",
            "folder_path": str(slide),
            "source_directory": str(slide.parent),
            "has_medullary": None,
            "case_has_medullary": None,
            "split": "test",
            "feature_path": str(pt.resolve()),
            "wsi_path": str(slide),
        })

    if missing or ambiguous:
        raise ValueError(
            f"Cannot make a complete manifest: {len(missing)} unmatched embeddings "
            f"and {len(ambiguous)} ambiguous slide matches. "
            f"Unmatched examples: {missing[:20]}; ambiguous examples: {ambiguous[:5]}"
        )

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_excel(output, index=False, engine="openpyxl")
    print(f"Saved {len(rows)} test samples to {output}")


if __name__ == "__main__":
    main()
