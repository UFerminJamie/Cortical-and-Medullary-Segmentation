from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json
import os
import sys

from SVC_GPU.config import DEFAULT_SPLIT_XLSX, DEFAULT_FEATURE_ROOT

PROJECT = Path(__file__).resolve().parents[2]
PACKAGE = Path(__file__).resolve().parent


def allocated_gpu() -> bool:
    """Use only GPUs advertised by this compute allocation."""
    gpu_ids = os.environ.get("SLURM_JOB_GPUS", "")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if "CUDA_VISIBLE_DEVICES" in os.environ and visible in ("", "-1", "NoDevFiles", "none", "N/A"):
        return False
    return bool(gpu_ids and gpu_ids not in ("N/A", "none", "-1") or visible)


@dataclass
class Config:
    manifest: str = str(PROJECT / "SVC_GPU/trident_all_test.xlsx")
    predictions: str = str(PROJECT / "SVC_GPU/KPMP_inference_outputs")
    datasets_file: str = ""
    cache_dir: str = ""
    annotation_root: str = ""
    contour_root: str = ""
    annotation_sources: str = "[]"
    request_threads: int = 8
    wsi_handles: int = 4
    jpeg_cache_mib: int = 96
    prediction_cache_mib: int = 64
    response_cache_mib: int = 48
    original_manifest: str = str(DEFAULT_SPLIT_XLSX)
    original_features: str = str(DEFAULT_FEATURE_ROOT)
    baseline_model: str = str(PROJECT / "SVC_GPU/artifacts/svc_pipeline.pkl")
    state_dir: str = str(PACKAGE / "state")
    python: str = str(PACKAGE / ".venv/bin/python")
    executor: str = "slurm"
    device: str = "gpu"
    account: str = "sarder-hubmap"
    qos: str = "sarder-hubmap"
    gpu_partition: str = "hpg-turin"
    gpu_resource: str = "gpu:l4:1"
    job_cpus: int = 8
    job_memory: str = "64G"
    job_time: str = "12:00:00"
    enable_compute: bool = True

    @classmethod
    def from_env(cls):
        """Portable launch settings; direct Config callers retain legacy defaults."""
        config = cls()
        names = {"datasets_file": "DATASETS_FILE", "manifest": "MANIFEST",
                 "annotation_root": "ANNOTATION_ROOT", "contour_root": "CONTOUR_ROOT", "annotation_sources": "ANNOTATION_SOURCES",
                 "predictions": "PREDICTIONS", "original_manifest": "ORIGINAL_MANIFEST",
                 "original_features": "ORIGINAL_FEATURES", "baseline_model": "BASELINE_MODEL",
                 "state_dir": "STATE_DIR", "cache_dir": "CACHE_DIR", "python": "WORKER_PYTHON",
                 "executor": "EXECUTOR", "device": "DEVICE", "account": "ACCOUNT", "qos": "QOS",
                 "gpu_partition": "GPU_PARTITION", "gpu_resource": "GPU_RESOURCE",
                 "job_cpus": "JOB_CPUS", "job_memory": "JOB_MEMORY", "job_time": "JOB_TIME",
                 "request_threads": "REQUEST_THREADS", "wsi_handles": "WSI_HANDLES",
                 "jpeg_cache_mib": "JPEG_CACHE_MIB", "prediction_cache_mib": "PREDICTION_CACHE_MIB",
                 "response_cache_mib": "RESPONSE_CACHE_MIB"}
        for field, name in names.items():
            value = os.environ.get("HISTOLOOP_" + name)
            if value is not None:
                setattr(config, field, int(value) if isinstance(getattr(config, field), int) else value)
        value = os.environ.get("HISTOLOOP_ENABLE_COMPUTE", "1").lower()
        if value not in ("0", "1", "false", "true"):
            raise ValueError("HISTOLOOP_ENABLE_COMPUTE must be 0 or 1")
        config.enable_compute = value in ("1", "true")
        if "HISTOLOOP_WORKER_PYTHON" not in os.environ and not Path(config.python).is_file():
            config.python = sys.executable
        return config

    def dict(self) -> dict:
        return asdict(self)

    def dataset_sources(self) -> list[dict]:
        """Keep explicit single-workbook configurations compatible with existing callers."""
        path = Path(self.datasets_file) if self.datasets_file else PACKAGE / "datasets.json"
        if not self.datasets_file and (self.manifest != Config.manifest or self.predictions != Config.predictions):
            sources = [{"id": "default", "name": "Review dataset", "manifest": self.manifest,
                        "predictions": self.predictions}]
        elif not path.is_file() and not self.datasets_file:
            sources = [{"id": "default", "name": "KPMP", "manifest": self.manifest,
                        "predictions": self.predictions}]
        else:
            sources = json.loads(path.read_text())
        if not isinstance(sources, list) or not sources:
            raise ValueError("Dataset configuration must be a nonempty list")
        result, seen = [], set()
        for source in sources:
            if not isinstance(source, dict) or any(not isinstance(source.get(key), str) or not source[key].strip()
                                                   for key in ("id", "name", "manifest", "predictions")):
                raise ValueError("Each dataset requires id, name, manifest, and predictions strings")
            if source["id"] in seen:
                raise ValueError(f"Duplicate dataset ID: {source['id']}")
            seen.add(source["id"])
            item = dict(source)
            for key in ("manifest", "predictions"):
                value = Path(item[key]).expanduser()
                item[key] = str((path.parent / value).resolve() if not value.is_absolute() else value.resolve())
            result.append(item)
        added_path = Path(self.state_dir) / "added_datasets.json"
        if added_path.is_file():
            added = json.loads(added_path.read_text())
            if not isinstance(added, list):
                raise ValueError("Saved cohorts must be a list")
            for source in added:
                if not isinstance(source, dict) or any(not isinstance(source.get(key), str) or not source[key].strip()
                                                       for key in ("id", "name", "manifest", "predictions")):
                    raise ValueError("Saved cohorts require id, name, manifest, and predictions strings")
                if source["id"] in seen:
                    raise ValueError(f"Duplicate dataset ID: {source['id']}")
                seen.add(source["id"])
                item = dict(source)
                for key in ("manifest", "predictions"):
                    value = Path(item[key]).expanduser()
                    item[key] = str((added_path.parent / value).resolve() if not value.is_absolute() else value.resolve())
                result.append(item)
        return result
