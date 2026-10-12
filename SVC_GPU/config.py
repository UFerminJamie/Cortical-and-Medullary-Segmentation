"""Shared constants for training and inference."""

from pathlib import Path
import json
import os

PACKAGE = Path(__file__).resolve().parent
LOCAL_SETTINGS_FILE = PACKAGE / "human_loop/local_settings.json"
LOCAL_SETTINGS = json.loads(LOCAL_SETTINGS_FILE.read_text()) if LOCAL_SETTINGS_FILE.is_file() else {}
DEFAULT_SPLIT_XLSX = Path(os.environ.get("HISTOLOOP_ORIGINAL_MANIFEST", LOCAL_SETTINGS.get("original_manifest", str(PACKAGE / "data/case_split.xlsx"))))
DEFAULT_FEATURE_ROOT = Path(os.environ.get("HISTOLOOP_ORIGINAL_FEATURES", LOCAL_SETTINGS.get("original_features", str(PACKAGE / "data/features"))))

BEST_N_COMPONENTS = 1024
BEST_C = 2.0
DEFAULT_THRESHOLD = 0.0
DEFAULT_CACHE_SIZE_MB = 4096.0
DEFAULT_INFERENCE_BATCH_SIZE = 32768

LABEL_NAMES = {0: "cortical", 1: "medullary"}
RAW_LABEL_MAPPING = {-1: "excluded", 0: "background (excluded)", 1: "cortical", 2: "medullary"}
