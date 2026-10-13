import json
import zipfile
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from SVC_GPU.human_loop.geometry import filter_mask, region_indices
from SVC_GPU.human_loop.jobs import Jobs
from SVC_GPU.human_loop.snapshots import create_snapshot
from SVC_GPU.human_loop.storage import Store, read_json
from SVC_GPU.human_loop.worker import partition_holdout, sample_corrections, train

from conftest import correction


def test_centers_boundaries_and_concave_regions():
    coordinates = np.array([[0, 0], [20, 0], [0, 20], [20, 20]])
    assert region_indices(coordinates, 20, [[10, 10], [30, 10], [30, 30], [10, 30]]).tolist() == [0, 1, 2, 3]
    assert region_indices(coordinates, 20, [[0, 0], [40, 0], [40, 20], [20, 20], [20, 40], [0, 40]]).tolist() == [0, 1, 2]
    with pytest.raises(ValueError):
        region_indices(coordinates, 20, [[0, 0], [1, 1]])


def test_independent_signed_filters_and_manual_visibility():
    scores = np.array([-2, -1, -.5, 0, .5, 2])
    labels = np.array([0, 0, 0, 1, 1, 1])
    manual = np.array([-1, -1, 0, -1, -1, -1])
    filters = {"cortex": {"comparison": "lte", "threshold": -1}, "medulla": {"comparison": "gte", "threshold": .5}, "keep_manual": False}
    assert filter_mask(scores, labels, manual, filters).tolist() == [True, True, False, False, True, True]
    filters["cortex"].update(comparison="gte", threshold=-1)
    assert filter_mask(scores, labels, manual, filters).tolist() == [False, True, True, False, True, True]
    filters["cortex"].update(comparison="lte", threshold=-1)
    filters["keep_manual"] = True
    assert filter_mask(scores, labels, manual, filters)[2]
    filters["cortex"]["enabled"] = False
    assert not filter_mask(scores, labels, manual, filters)[:3].any()


def test_auth_and_origin(setup):
    config, app, client, slides = setup
    assert client.post("/api/passes", json={"name": "Bad"}, headers={"Origin": "https://other.example"}).status_code == 403
    client.cookies.clear()
    assert client.get("/api/slides").status_code == 401
    assert client.get("/health").status_code == 200


def test_selection_filters_passes_undo_and_export(setup):
    config, app, client, slides = setup
    slide_id = slides["A"]["id"]
    response = correction(client, slide_id)
    assert response.status_code == 200
    first = response.json()
    assert first["tile_ids"] == [0]
    assert np.frombuffer(client.get(f"/api/slides/{slide_id}/tiles.bin").content, "<f4").reshape(-1, 5)[:, 4].tolist() == [1, -1, -1, -1]
    assert correction(client, slide_id, label=0, revision=0).status_code == 409
    passed = client.post("/api/passes", json={"name": "Pass 2"}).json()
    second = correction(client, slide_id, label=0, pass_id=passed["id"]).json()
    assert client.post(f"/api/slides/{slide_id}/undo", json={"revision": 2, "target_id": second["event_id"],
        "pass_id": passed["id"], "reviewer": "Test reviewer"}).status_code == 200
    detail = client.get(f"/api/slides/{slide_id}").json()
    assert detail["manual_tiles"] == 1 and detail["revision"] == 3
    labels, revision, undone = Store(config.state_dir).resolved(slide_id, 4)
    assert labels.tolist() == [1, -1, -1, -1]
    assert second["event_id"] in undone
    history = client.get(f"/api/slides/{slide_id}/history").json()["events"]
    assert len(history) == 3 and history[-1]["pass_id"] == "pass-1"
    assert history[-1]["polygon"] and history[-1]["note"] == "Verified test region"
    with zipfile.ZipFile(BytesIO(client.get("/api/export").content)) as archive:
        events = json.loads(archive.read(f"slides/{slide_id}/history.json"))
        assert events[0]["tile_coordinates"] == [[0, 0]]
        assert events[0]["decision_scores"] == [-2]
        assert b"medulla" in archive.read(f"slides/{slide_id}/corrections.csv")
        assert b"source_event_id,pass_id,reviewer,corrected_at" in archive.read(f"slides/{slide_id}/corrections.csv")


def test_filter_limited_region_and_clear(setup):
    config, app, client, slides = setup
    slide_id = slides["A"]["id"]
    polygon = [[0, 0], [512, 0], [512, 512], [0, 512]]
    filters = {"cortex": {"comparison": "lte", "threshold": -1}, "medulla": {"enabled": False}}
    response = correction(client, slide_id, polygon=polygon, visible_only=True, filters=filters)
    assert response.json()["tile_ids"] == [0]
    response = correction(client, slide_id, action="clear", label=-1)
    assert response.status_code == 200
    assert client.get(f"/api/slides/{slide_id}").json()["manual_tiles"] == 0


def test_live_preview_filters_do_not_change_prediction_sources(setup):
    config, app, client, slides = setup
    sid = slides['A']['id']
    before = Path(slides['A']['baseline_csv']).read_bytes()
    response = correction(client, sid, polygon=[[0,0],[512,0],[512,512],[0,512]], visible_only=True,
        display_threshold=3, filters={'cortex':{'comparison':'all'},'medulla':{'enabled':False}})
    assert response.status_code == 200 and response.json()['tile_ids'] == [0,1,2,3]
    assert Path(slides['A']['baseline_csv']).read_bytes() == before
    assert app.state.store.events(sid)[0]['predicted_labels'] == [0,0,1,1]
    assert app.state.store.events(sid)[0]['display_threshold'] == 3


def test_crash_recovery_and_review_status(setup):
    config, app, client, slides = setup
    slide_id = slides["A"]["id"]
    correction(client, slide_id)
    assert client.post(f"/api/slides/{slide_id}/review", json={"revision": 1, "pass_id": "pass-1", "reviewer": "Test"}).status_code == 200
    assert next(row for row in client.get("/api/slides").json() if row["id"] == slide_id)["reviewed"]
    Path(config.state_dir, "review_index.json").unlink()
    recovered = Store(config.state_dir).summary()[slide_id]
    assert recovered["manual_tiles"] == 1 and recovered["revision"] == 2
    assert "pass-1" in recovered["reviewed_passes"]
    correction(client, slide_id)
    assert not Store(config.state_dir).summary()[slide_id]["reviewed_passes"]


def test_wsi_native_deep_zoom_tiles(setup):
    config, app, client, slides = setup
    slide_id = slides["A"]["id"]
    detail = client.get(f"/api/slides/{slide_id}").json()
    assert detail["tile_size_level0"] == 256
    assert b'TileSize="256"' in client.get(f"/api/wsi/{slide_id}.dzi").content
    response = client.get(f"/api/wsi/{slide_id}_files/{detail['max_level']}/0_0.jpg")
    assert response.status_code == 200 and response.content.startswith(b"\xff\xd8")
    assert client.get(f"/api/wsi/{slide_id}_files/{detail['max_level']}/99_99.jpg").status_code == 400


def test_snapshot_is_frozen_and_jobs_submit(setup, monkeypatch):
    config, app, client, slides = setup
    slide_id = slides["A"]["id"]
    correction(client, slide_id)
    calls = []
    def fake_run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(stdout="12345\n", returncode=0)
    monkeypatch.setattr("SVC_GPU.human_loop.jobs.subprocess.run", fake_run)
    response = client.post("/api/jobs", json={"kind": "train", "threshold": -1, "budget": 100,
        "reviewer": "Test", "pass_id": "pass-1"})
    assert response.status_code == 202
    initial = response.json()
    app.state.jobs._threads[initial["id"]].join(timeout=2)
    job = read_json(Path(initial["directory"]) / "job.json")
    assert job["slurm_id"] == "12345" and job["threshold"] == -1
    assert calls[0][0] == "sbatch" and "--gres=gpu:l4:1" in calls[0]
    snapshot = read_json(Path(job["snapshot_path"]))
    assert snapshot["corrected_tiles"] == 1
    correction(client, slide_id, label=0)
    with np.load(snapshot["slides"][0]["labels_path"]) as frozen:
        assert frozen["labels"].tolist() == [1]
    assert Path(snapshot["original_sources"][0]["csv_copy"]).is_file()
    assert client.post("/api/jobs", json={"kind": "train", "reviewer": "Test"}).status_code == 400


def test_case_level_holdout_isolation():
    train = [{"sample_name": "A_PAS_1.svs", "case_id": "case-a"}]
    test = [{"sample_name": "B_PAS_1.svs", "case_id": "case-b"},
            {"sample_name": "A_PAS_2.svs", "case_id": "erroneous-case-id"},
            {"sample_name": "D_PAS_1.svs", "case_id": "case-d"},
            {"sample_name": "E_PAS_1.svs", "case_id": "case-b"}]
    corrected = [{"name": "B_PAS_2", "case_id": "case-b"}]
    kept, excluded = partition_holdout(train, test, corrected)
    assert [row["case_id"] for row in kept] == ["case-d"]
    assert len(excluded) == 3


def test_slow_submission_recovers_unique_slurm_job_without_duplicate(setup, monkeypatch):
    import subprocess

    config, app, client, slides = setup
    calls = []
    def slow(command, **kwargs):
        calls.append(command)
        if command[0] == "sbatch":
            raise subprocess.TimeoutExpired(command, 60)
        return SimpleNamespace(stdout="54321\n", returncode=0)
    monkeypatch.setattr("SVC_GPU.human_loop.jobs.subprocess.run", slow)
    response = client.post("/api/jobs", json={"kind": "predict", "slide_id": slides["A"]["id"],
        "threshold": -1, "reviewer": "Test", "pass_id": "pass-1"})
    assert response.status_code == 202
    initial = response.json()
    app.state.jobs._threads[initial["id"]].join(timeout=2)
    job = read_json(Path(initial["directory"]) / "job.json")
    assert job["status"] == "pending" and job["slurm_id"] == "54321"
    assert any(argument.startswith("--name=rv_") for argument in calls[1])
    assert sum(command[0] == "sbatch" for command in calls) == 1


def test_sampler_is_fair_and_reproducible(tmp_path):
    slides = []
    for index in range(3):
        path = tmp_path / f"{index}.npz"
        np.savez(path, labels=np.array([0] * 10 + [1] * 10))
        slides.append({"id": str(index), "labels_path": str(path)})
    first, second = sample_corrections(slides, 12), sample_corrections(slides, 12)
    assert sum(map(len, first.values())) == 12
    assert all(len(value) == 4 for value in first.values())
    assert all(np.array_equal(first[key], second[key]) for key in first)
    assert sum(map(len, sample_corrections(slides, 0).values())) == 60


def test_training_combines_original_and_human_with_override_and_provenance(setup, tmp_path, monkeypatch):
    config, app, client, slides = setup
    correction(client, slides["A"]["id"], label=1)
    correction(client, slides["C"]["id"], label=0)
    store, catalog = app.state.store, app.state.catalog
    directory = tmp_path / "training"
    snapshot = create_snapshot(catalog, store, directory / "snapshot")
    captured = {}
    def fit(features, labels, **kwargs):
        captured["labels"] = labels.numpy()
        captured["features"] = features.numpy()
        return {"scaler": "fake", "pca": "fake", "svc": "fake", "gpu": {"name": "test"}}
    def predict(bundle, features, **kwargs):
        scores = np.linspace(-1, 1, len(features)).astype(np.float32)
        return (scores >= kwargs["threshold"]).astype(np.int32), scores
    monkeypatch.setattr("SVC_GPU.human_loop.worker.train_gpu_pipeline", fit)
    monkeypatch.setattr("SVC_GPU.human_loop.worker.predict_gpu", predict)
    monkeypatch.setattr("SVC_GPU.human_loop.worker.require_gpu", lambda: (SimpleNamespace(cuda=SimpleNamespace(runtime=SimpleNamespace(memGetInfo=lambda:(24*1024**3,24*1024**3)))), None, None, None))
    job = {"id": "train-test", "snapshot_path": str(directory / "snapshot/snapshot.json"),
        "budget": 0, "model_path": config.baseline_model, "model_id": "baseline", "threshold": 0,
        "directory": str(directory), "created_at": "2026-10-04T12:00:00"}
    result = train(config, store, catalog, job)
    assert len(captured["labels"]) == 5  # Three original vectors + one replacement + one new tile.
    metadata = read_json(directory / "artifacts/metadata.json")
    assert metadata["original_train_labels_overridden"] == 1
    assert metadata["exact_duplicate_original_tiles_removed"] == 1
    assert metadata["corrected_tiles_sampled"] == 2
    assert result["heldout_samples"] == 1
    assert store.model("train-test", config.baseline_model)["path"].endswith("svc_pipeline.pkl")
    assert (directory / "artifacts/training_tile_provenance.csv").is_file()


def test_source_identity_rejects_changed_embeddings(setup):
    config, app, client, slides = setup
    slide_id = slides["A"]["id"]
    correction(client, slide_id)
    feature_path = Path(slides["A"]["pt_path"])
    import os

    stat = feature_path.stat()
    os.utime(feature_path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 10_000_000))
    response = client.get(f"/api/slides/{slide_id}")
    assert response.status_code == 400 and "changed" in response.json()["detail"]
