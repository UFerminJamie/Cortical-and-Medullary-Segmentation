from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tifffile
import torch
from fastapi.testclient import TestClient

from SVC_GPU.human_loop.config import Config
from SVC_GPU.human_loop.server import create_app
from SVC_GPU.human_loop.storage import atomic_json


@pytest.fixture
def setup(tmp_path):
    features, original_root, predictions = tmp_path / "features", tmp_path / "original", tmp_path / "predictions"
    for directory in (features, original_root, predictions / "predictions"):
        directory.mkdir(parents=True)
    coords = np.array([[0, 0], [256, 0], [0, 256], [256, 256]], dtype=np.int64)
    names = ["A_PAS_1", "B_PAS_1", "C_PAS_1"]
    rows, original_rows = [], []
    for index, name in enumerate(names):
        slide = tmp_path / "KPMP_batch2" / f"{name}.svs"
        slide.parent.mkdir(exist_ok=True)
        tifffile.imwrite(slide, np.full((512, 512, 3), 240, dtype=np.uint8), photometric="rgb")
        pt = features / f"{name}.pt"
        torch.save({"features": torch.arange(64, dtype=torch.float32).reshape(4, 16) / 30,
                    "coords": torch.from_numpy(coords), "metadata": {"slide_name": name,
                        "slide_path": str(slide), "read_plan": {"patch_size_level0": (256, 256)}}}, pt)
        frame = pd.DataFrame({"x": coords[:, 0], "y": coords[:, 1], "decision_score": [-2.0, -.5, .5, 2.0],
                              "predicted_label_encoded": [0, 0, 1, 1]})
        frame.to_csv(predictions / "predictions" / f"{name}.csv", index=False)
        rows.append({"sample_name": name + ".svs", "feature_path": str(pt), "wsi_path": str(slide), "split": "test"})
        if index < 2:
            old_pt = original_root / f"{name}.pt"
            torch.save({"features": torch.randn(4, 16), "coordinates": torch.from_numpy(coords), "tile_size": 256}, old_pt)
            pd.DataFrame({"x": coords[:, 0], "y": coords[:, 1], "label": [1, 2, 1, 2]}).to_csv(old_pt.with_suffix(".csv"), index=False)
            original_rows.append({"sample_name": name + ".svs", "feature_path": str(old_pt), "case_id": name[0],
                                  "split": "train" if index == 0 else "test"})
    manifest, original_manifest = tmp_path / "manifest.xlsx", tmp_path / "case_split.xlsx"
    pd.DataFrame(rows).to_excel(manifest, index=False)
    pd.DataFrame(original_rows).to_excel(original_manifest, index=False)
    atomic_json(predictions / "inference_summary.json", {"threshold": 0, "samples": [
        {"sample": name, "tiles": 4, "cortical": 2, "medullary": 2} for name in names]})
    model = tmp_path / "baseline" / "svc_pipeline.pkl"
    model.parent.mkdir()
    model.write_bytes(b"test model")
    atomic_json(model.parent / "metadata.json", {"feature_dimension": 16, "n_components": 8, "C": 2,
                                                "tile_counts": {"fit": 4}})
    config = Config(manifest=str(manifest), predictions=str(predictions), original_manifest=str(original_manifest),
        original_features=str(original_root), baseline_model=str(model), state_dir=str(tmp_path / "state"))
    app = create_app(config, token="test-access")
    with TestClient(app) as client:
        client.get("/?token=test-access")
        slides = client.get("/api/slides").json()
        yield config, app, client, {row["name"][0]: row for row in slides}


def correction(client, slide_id, *, label=1, polygon=None, revision=None, **kwargs):
    detail = client.get(f"/api/slides/{slide_id}").json()
    return client.post(f"/api/slides/{slide_id}/corrections", json={
        "label": label, "polygon": polygon or [[0, 0], [256, 0], [256, 256], [0, 256]],
        "revision": detail["revision"] if revision is None else revision, "fingerprint": detail["fingerprint"],
        "reviewer": "Test reviewer", "pass_id": "pass-1", "visible_only": False, "note": "Verified test region", "grid_format": 2,
        **kwargs})
