"""Read XLSX sample manifests and resolve their feature files."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

ALLOWED_SPLITS = frozenset({"train", "val", "test"})
COHORT_PATTERN = re.compile(r"(?:^|/)KPMP_([^/]+)(?:/|$)", flags=re.IGNORECASE)


def _require_columns(df: pd.DataFrame, required: set[str], xlsx_path: Path) -> None:
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(
            f"Missing columns in {xlsx_path}: {missing}. "
            f"Available columns: {df.columns.tolist()}"
        )


def _extract_cohort(source_directory: object) -> str:
    normalized = str(source_directory).replace("\\", "/")
    match = COHORT_PATTERN.search(normalized)
    if match is None:
        raise ValueError(f"Could not determine cohort from: {source_directory}")
    return match.group(1)


def _resolve_manifest(
    xlsx_path: str | Path,
    root_dir: str | Path,
    *,
    split: str | None,
    require_csv: bool,
) -> pd.DataFrame:
    xlsx_path = Path(xlsx_path).expanduser().resolve()
    root_dir = Path(root_dir).expanduser().resolve()

    if not xlsx_path.is_file():
        raise FileNotFoundError(f"XLSX manifest does not exist: {xlsx_path}")
    if not root_dir.is_dir():
        raise FileNotFoundError(f"Feature root does not exist: {root_dir}")

    df = pd.read_excel(xlsx_path, engine="openpyxl")
    _require_columns(df, {"sample_name"}, xlsx_path)

    if split is not None:
        split = split.strip().lower()
        if split not in ALLOWED_SPLITS:
            raise ValueError(
                f"split must be one of {sorted(ALLOWED_SPLITS)}; received {split!r}"
            )
        _require_columns(df, {"split"}, xlsx_path)
        normalized_splits = df["split"].astype("string").str.strip().str.lower()
        df = df.loc[normalized_splits == split].copy()
        if df.empty:
            raise ValueError(f"No rows with split={split!r} were found in {xlsx_path}")
        df["split"] = split
    else:
        df = df.copy()

    df["sample_name"] = df["sample_name"].map(lambda value: Path(str(value)).stem)

    if "feature_path" in df.columns:
        pt_paths = df["feature_path"].map(lambda value: Path(str(value)).expanduser())
        pt_paths = pt_paths.map(
            lambda path: path if path.suffix.lower() == ".pt" else path.with_suffix(".pt")
        )
        cohorts = df.get("cohort", pd.Series([None] * len(df), index=df.index))
    else:
        if "cohort" in df.columns:
            cohorts = df["cohort"].astype(str).str.strip()
        else:
            _require_columns(df, {"source_directory"}, xlsx_path)
            cohorts = df["source_directory"].map(_extract_cohort)
        pt_paths = pd.Series(
            [root_dir / cohort / f"{sample_name}.pt" for cohort, sample_name in zip(cohorts, df["sample_name"])],
            index=df.index,
            dtype=object,
        )

    df["cohort"] = cohorts
    df["pt_path"] = pt_paths.map(lambda path: str(Path(path).resolve()))
    df["csv_path"] = df["pt_path"].map(lambda path: str(Path(path).with_suffix(".csv")))
    df["sample_path"] = df["pt_path"].map(lambda path: str(Path(path).with_suffix("")))

    duplicated = df["pt_path"].duplicated(keep=False)
    if duplicated.any():
        duplicate_paths = sorted(df.loc[duplicated, "pt_path"].unique().tolist())
        raise ValueError(f"Duplicate feature paths in {xlsx_path}: {duplicate_paths}")

    missing_pt = [path for path in df["pt_path"] if not Path(path).is_file()]
    if missing_pt:
        preview = "\n".join(missing_pt[:10])
        raise FileNotFoundError(
            f"Missing {len(missing_pt)} PT feature file(s). First missing paths:\n{preview}"
        )

    if require_csv:
        missing_csv = [path for path in df["csv_path"] if not Path(path).is_file()]
        if missing_csv:
            preview = "\n".join(missing_csv[:10])
            raise FileNotFoundError(
                f"Missing {len(missing_csv)} label CSV file(s). First missing paths:\n{preview}"
            )

    return df.reset_index(drop=True)


def load_split_samples(
    xlsx_path: str | Path,
    root_dir: str | Path,
    split: str,
) -> pd.DataFrame:
    """Return one labeled train/val/test split from a sample workbook."""
    return _resolve_manifest(
        xlsx_path=xlsx_path,
        root_dir=root_dir,
        split=split,
        require_csv=True,
    )


def load_inference_samples(
    xlsx_path: str | Path,
    root_dir: str | Path,
    split: str | None = None,
) -> pd.DataFrame:
    """Return unseen samples; the workbook may omit both labels and split."""
    return _resolve_manifest(
        xlsx_path=xlsx_path,
        root_dir=root_dir,
        split=split,
        require_csv=False,
    )
