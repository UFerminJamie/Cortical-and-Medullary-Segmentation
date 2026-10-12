"""Model comparison and whole-slide annotation invariants."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from io import BytesIO
import zipfile
import json

import numpy as np
import pytest

from SVC_GPU.human_loop.catalog import Catalog
from SVC_GPU.human_loop.config import Config
from SVC_GPU.human_loop.server import create_app
from SVC_GPU.human_loop.storage import Store, atomic_json, read_json
from SVC_GPU.human_loop.worker import predict
from SVC_GPU.human_loop.worker import train, sample_corrections
from SVC_GPU.human_loop.snapshots import create_snapshot
from SVC_GPU.human_loop.geometry import filter_mask
from conftest import correction


def saved_version(app, slide_id, job_id, model_id, prediction, created_at):
    base = app.state.catalog.tiles(slide_id)
    root = app.state.store.root
    directory = root / "predictions" / slide_id
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{job_id}.npz"
    np.savez_compressed(path, scores=np.array([-3, -1, 1, 3], dtype=np.float32),
                        predictions=np.array(prediction, dtype=np.int8), fingerprint=np.array(base["fingerprint"]))
    version = {"job_id": job_id, "model_id": model_id, "path": str(path), "threshold": -1,
        "created_at": created_at, "tiles": 4, "cortical": prediction.count(0), "medullary": prediction.count(1)}
    versions = read_json(directory / "versions.json", {})
    versions[job_id] = version
    atomic_json(directory / "versions.json", versions)
    models = read_json(root / "prediction_models.json", {})
    models.setdefault(slide_id, {})[model_id] = version
    atomic_json(root / "prediction_models.json", models)
    atomic_json(directory / "active.json", version)
    index = read_json(root / "prediction_index.json", {})
    index[slide_id] = {**version, "prediction_id": job_id}
    atomic_json(root / "prediction_index.json", index)
    return version


def test_original_splits_and_prediction_statistics(setup):
    config, app, client, slides = setup
    assert [slides[key]["original_split"] for key in "ABC"] == ["train", "test", "new"]
    assert Catalog.split(" Validation ") == "val"
    assert slides["A"]["medulla_fraction"] == .5
    assert slides["A"]["medulla_cortex_ratio"] == 1


def test_visible_selection_matches_the_displayed_annotation_layer():
    scores = np.array([-2, -.5, .5, 2])
    predictions = np.array([0, 0, 1, 1])
    manual = np.array([1, -1, 0, -1])
    effective = np.where(manual >= 0, manual, predictions)
    filters = {"cortex": {"enabled": True, "comparison": "all"},
               "medulla": {"enabled": False}, "keep_manual": True}
    assert filter_mask(scores, effective, manual, {**filters, "source": "combined"}, predictions).tolist() == [False, True, True, False]
    assert filter_mask(scores, effective, manual, {**filters, "source": "prediction"}, predictions).tolist() == [True, True, False, False]
    assert filter_mask(scores, effective, manual, {**filters, "source": "human"}, predictions).tolist() == [False, False, True, False]
    with pytest.raises(ValueError, match="Overlay source"):
        filter_mask(scores, effective, manual, {"source": "invalid"}, predictions)


def test_model_switch_preserves_humans_and_pins_each_prediction_run(setup):
    config, app, client, slides = setup
    slide_id = slides["A"]["id"]
    correction(client, slide_id, label=1)
    atomic_json(app.state.store.root / "models/model-two.json", {
        "id": "model-two", "name": "Reviewed SVC 2", "path": config.baseline_model})
    saved_version(app, slide_id, "predict-first", "model-two", [0, 0, 0, 0], "2026-10-05T01:00:00Z")
    saved_version(app, slide_id, "predict-second", "model-two", [1, 1, 1, 1], "2026-10-05T02:00:00Z")
    for query, expected in [("?model_id=baseline", [0, 0, 1, 1]),
                            ("?prediction_id=predict-first", [0, 0, 0, 0]),
                            ("?model_id=model-two", [1, 1, 1, 1])]:
        response = client.get(f"/api/slides/{slide_id}/tiles.bin{query}")
        assert response.status_code == 200
        data = np.frombuffer(response.content, "<f4").reshape(-1, 5)
        assert data[:, 3].tolist() == expected
        assert data[:, 4].tolist() == [1, -1, -1, -1]
    listing = client.get("/api/slides?model_id=model-two").json()
    row = next(row for row in listing if row["id"] == slide_id)
    assert row["medulla_fraction"] == 1 and row["medulla_cortex_ratio"] is None
    assert not next(row for row in listing if row["id"] == slides["B"]["id"])["prediction_available"]
    assert client.get(f"/api/slides/{slides['B']['id']}?model_id=model-two").status_code == 404
    assert client.get(f"/api/slides/{slide_id}?model_id=model-two&prediction_id=baseline").status_code == 400
    detail = client.get(f"/api/slides/{slide_id}?prediction_id=predict-first").json()
    response = correction(client, slide_id, label=0, view_prediction_id="predict-first", prediction_id="predict-first", shape="freehand")
    assert response.status_code == 200
    event = app.state.store.events(slide_id)[-1]
    assert event["prediction_id"] == "predict-first" and event["model_id"] == "model-two" and event["shape"] == "freehand"
    assert detail["predicted_medulla"] == 0 and detail["medulla"] == 1


def test_bulk_labels_all_tiles_ignoring_filters_and_undo(setup, monkeypatch):
    config, app, client, slides = setup
    monkeypatch.setattr("SVC_GPU.human_loop.bulk.threading.Thread.start", lambda self: None)
    ids = [slides["A"]["id"], slides["B"]["id"]]
    payload = {"slides": [{"id": key, "revision": 0} for key in ids], "label": 1,
               "reviewer": "Reviewer", "pass_id": "pass-1", "note": "Verified whole slides",
               "filters": {"medulla": {"enabled": False}}}
    response = client.post("/api/bulk-labels", json=payload)
    assert response.status_code == 202
    batch_id = response.json()["id"]
    assert client.post("/api/bulk-labels", json=payload).status_code == 409
    app.state.bulk.run(read_json(app.state.store.root / "bulk" / batch_id / "request.json"))
    status = client.get("/api/bulk-labels").json()[0]
    assert status["status"] == "completed" and status["completed"] == 2 and status["tiles"] == 8
    for slide_id in ids:
        labels, revision, undone = app.state.store.resolved(slide_id, 4)
        assert labels.tolist() == [1, 1, 1, 1]
        event = app.state.store.events(slide_id)[0]
        assert event["shape"] == "whole_slide" and event["selection_rule"] == "all_feature_tiles"
        assert event["bulk_id"] == batch_id and len(event["tile_coordinates"]) == 4
        result = client.post(f"/api/slides/{slide_id}/undo", json={"revision": revision,
            "target_id": event["id"], "reviewer": "Reviewer", "pass_id": "pass-1"})
        assert result.status_code == 200
        assert app.state.store.resolved(slide_id, 4)[0].tolist() == [-1, -1, -1, -1]
    assert client.get("/api/slides").json()[0]["medulla_fraction"] == .5  # human labels never change predicted rank


def test_bulk_skips_concurrent_edits_instead_of_overwriting(setup, monkeypatch):
    config, app, client, slides = setup
    monkeypatch.setattr("SVC_GPU.human_loop.bulk.threading.Thread.start", lambda self: None)
    slide_id = slides["A"]["id"]
    batch = client.post("/api/bulk-labels", json={"slides": [{"id": slide_id, "revision": 0}],
        "label": 0, "reviewer": "Reviewer", "pass_id": "pass-1"}).json()
    correction(client, slide_id, label=1)
    app.state.bulk.run(read_json(app.state.store.root / "bulk" / batch["id"] / "request.json"))
    status = client.get("/api/bulk-labels").json()[0]
    assert status["status"] == "completed_with_skips" and status["failed"] == 1
    assert app.state.store.resolved(slide_id, 4)[0].tolist() == [1, -1, -1, -1]


def test_cpu_mode_config_and_review_completion(setup):
    config, app, client, slides = setup
    assert Config().device == "gpu"
    with pytest.raises(ValueError, match="CPU review mode"):
        create_app(replace(config, device="cpu", executor="local"))
    app.state.config.device = "cpu"
    assert client.get("/api/bootstrap").json()["device"] == "cpu"
    slide_id = slides["A"]["id"]
    result = client.post(f"/api/slides/{slide_id}/review", json={"revision": 0,
        "reviewer": "Reviewer", "pass_id": "pass-1", "view_prediction_id": "baseline"})
    assert result.status_code == 200
    assert "pass-1" in client.get(f"/api/slides/{slide_id}").json()["reviewed_passes"]
    event = app.state.store.events(slide_id)[0]
    assert event["model_id"] == "baseline" and event["prediction_id"] == "baseline"


def test_prediction_worker_publishes_version_and_model_index(setup, monkeypatch):
    config, app, client, slides = setup
    slide_id = slides["A"]["id"]
    monkeypatch.setattr("SVC_GPU.human_loop.worker.load_bundle", lambda path: {"metadata": {"feature_dimension": 16}})
    monkeypatch.setattr("SVC_GPU.human_loop.worker.predict_gpu", lambda *args, **kwargs:
        (np.array([1, 0, 1, 1], dtype=np.int8), np.array([2, -2, -1, .5], dtype=np.float32)))
    job = {"id": "predict-publish-test", "slide_id": slide_id, "model_id": "baseline",
           "model_path": config.baseline_model, "threshold": -1,
           "fingerprint": app.state.catalog.tiles(slide_id)["fingerprint"],
           "directory": str(app.state.store.root / "jobs/predict-publish-test")}
    result = predict(config, app.state.store, app.state.catalog, job)
    version = app.state.catalog.prediction_versions(slide_id)[1]
    assert result["tiles"] == 4 and version["job_id"] == job["id"]
    assert version["cortical"] == 1 and version["medullary"] == 3
    assert app.state.catalog.current_tiles(slide_id, prediction_id=job["id"])["predictions"].tolist() == [1, 0, 1, 1]
    assert read_json(app.state.store.root / "prediction_models.json")[slide_id]["baseline"]["job_id"] == job["id"]


def test_existing_outputs_migrate_without_losing_active_prediction(setup):
    config, app, client, slides = setup
    slide_id = slides["A"]["id"]
    version = saved_version(app, slide_id, "predict-legacy", "legacy-model", [0, 0, 0, 0], "2026-10-05T00:00:00Z")
    atomic_json(app.state.store.root / "models/legacy-model.json", {
        "id": "legacy-model", "name": "Legacy reviewed model", "path": config.baseline_model})
    atomic_json(app.state.store.root / "jobs/predict-legacy/job.json", {
        "id": "predict-legacy", "kind": "predict", "status": "completed", "slide_id": slide_id,
        "model_id": "legacy-model", "threshold": -1, "created_at": version["created_at"],
        "result": {"cortex": 4, "medulla": 0, "tiles": 4}})
    (app.state.store.root / "prediction_models.json").unlink()
    (app.state.store.root / "predictions" / slide_id / "versions.json").unlink()
    recovered = Catalog(config, Store(config.state_dir))
    assert recovered.current_tiles(slide_id)["prediction_id"] == "predict-legacy"
    assert recovered.current_tiles(slide_id, "legacy-model")["predictions"].tolist() == [0, 0, 0, 0]
    assert recovered.prediction_versions(slide_id)[1]["job_id"] == "predict-legacy"


def test_exclusion_persists_exports_and_undo_restores_prediction(setup, tmp_path):
    config, app, client, slides = setup
    slide_id = slides["A"]["id"]
    response = correction(client, slide_id, label=-2)
    assert response.status_code == 200 and response.json()["label"] == -2
    detail = client.get(f"/api/slides/{slide_id}").json()
    assert detail["excluded_tiles"] == 1 and detail["manual_tiles"] == 0
    assert detail["cortex"] == 1 and detail["medulla"] == 2
    snapshot = create_snapshot(app.state.catalog, app.state.store, tmp_path / "snapshot")
    assert snapshot["excluded_tiles"] == 1 and snapshot["corrected_tiles"] == 0
    assert not sample_corrections(snapshot["slides"], 0)
    with zipfile.ZipFile(BytesIO(client.get("/api/export").content)) as archive:
        assert b"excluded" in archive.read(f"slides/{slide_id}/corrections.csv")
    Path(config.state_dir, "review_index.json").unlink()
    assert Store(config.state_dir).summary()[slide_id]["excluded_tiles"] == 1
    event_id = response.json()["event_id"]
    assert client.post(f"/api/slides/{slide_id}/undo", json={"target_id": event_id,
        "revision": 1, "reviewer": "Test", "pass_id": "pass-1"}).status_code == 200
    assert client.get(f"/api/slides/{slide_id}").json()["excluded_tiles"] == 0


def test_training_omits_excluded_original_and_review_tiles(setup, tmp_path, monkeypatch):
    config, app, client, slides = setup
    correction(client, slides["A"]["id"], label=-2)
    correction(client, slides["C"]["id"], label=-2)
    directory = tmp_path / "exclude-fit"
    snapshot = create_snapshot(app.state.catalog, app.state.store, directory / "snapshot")
    captured = {}
    def fit(x, y, **kwargs):
        captured["labels"] = y.numpy()
        return {"scaler": "fake", "pca": "fake", "svc": "fake", "gpu": {"name": "test"}}
    def prediction(bundle, features, **kwargs):
        scores = np.linspace(-1, 1, len(features)).astype(np.float32)
        return (scores >= kwargs["threshold"]).astype(np.int32), scores
    monkeypatch.setattr("SVC_GPU.human_loop.worker.train_gpu_pipeline", fit)
    monkeypatch.setattr("SVC_GPU.human_loop.worker.predict_gpu", prediction)
    monkeypatch.setattr("SVC_GPU.human_loop.worker.require_gpu", lambda: (
        SimpleNamespace(cuda=SimpleNamespace(runtime=SimpleNamespace(memGetInfo=lambda:(24*1024**3,24*1024**3)))),None,None,None))
    job = {"id": "train-exclusion-test", "snapshot_path": str(directory / "snapshot/snapshot.json"),
        "budget": 0, "model_path": config.baseline_model, "model_id": "baseline", "threshold": 0,
        "directory": str(directory), "created_at": "2026-10-05T12:00:00"}
    result = train(config, app.state.store, app.state.catalog, job)
    assert captured["labels"].tolist() == [1, 0, 1]
    metadata = read_json(directory / "artifacts/metadata.json")
    assert metadata["original_training_tiles_excluded"] == 1 and metadata["human_tiles_excluded"] == 2
    assert metadata["corrected_tiles_sampled"] == 0
    assert (directory / "artifacts/excluded_tile_provenance.csv").is_file()


def test_session_gpu_jobs_do_not_submit_slurm_and_slurm_remains_optional(setup, monkeypatch):
    config, app, client, slides = setup
    monkeypatch.setenv("SLURM_JOB_GPUS", "0")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    spawned, submitted = [], []
    def popen(command, **kwargs):
        spawned.append(command)
        return SimpleNamespace(pid=12345678, poll=lambda:None)
    def sbatch(command, **kwargs):
        submitted.append(command)
        assert command[0] == "sbatch"
        return SimpleNamespace(stdout="98765\n", returncode=0)
    monkeypatch.setattr("SVC_GPU.human_loop.jobs.subprocess.Popen", popen)
    monkeypatch.setattr("SVC_GPU.human_loop.jobs.subprocess.run", sbatch)
    config.executor = "local"
    payload = {"kind": "predict", "slide_id": slides["A"]["id"], "model_id": "baseline",
               "reviewer": "Test", "pass_id": "pass-1"}
    first = client.post("/api/jobs", json=payload).json()
    app.state.jobs._threads[first["id"]].join(timeout=5)
    assert spawned and not submitted and first["executor"] == "local"
    request = read_json(Path(first["directory"]) / "request.json")
    assert request["config"]["executor"] == "local"
    second = client.post("/api/jobs", json={**payload, "slide_id": slides["B"]["id"], "executor": "slurm"}).json()
    app.state.jobs._threads[second["id"]].join(timeout=5)
    assert len(submitted) == 1 and second["executor"] == "slurm"
    assert "--account=sarder-hubmap" in submitted[0]


def test_local_worker_exit_is_reaped_and_failure_is_not_overwritten(setup):
    config, app, client, slides = setup
    manager = app.state.jobs
    directory = app.state.store.root / "jobs/local-exit"
    job = {"id": "local-exit", "directory": str(directory), "executor": "local",
           "pid": 12345678, "status": "running", "created_at": "2026-10-05T12:00:00"}
    atomic_json(directory / "job.json", job)
    manager._processes[job["id"]] = SimpleNamespace(poll=lambda:137)
    result = manager.listing()[0]
    assert result["status"] == "failed" and "137" in result["error"]
    assert job["id"] not in manager._processes
    manager._publish_launch({**job, "status": "submitted"})
    assert read_json(directory / "job.json")["status"] == "failed"
