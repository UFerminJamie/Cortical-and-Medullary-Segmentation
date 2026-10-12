from dataclasses import replace
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from SVC_GPU.human_loop.config import Config
from SVC_GPU.human_loop.server import create_app
from conftest import correction


def test_environment_overrides(monkeypatch):
    monkeypatch.setenv("HISTOLOOP_STATE_DIR", "/state")
    monkeypatch.setenv("HISTOLOOP_REQUEST_THREADS", "3")
    monkeypatch.setenv("HISTOLOOP_ENABLE_COMPUTE", "0")
    monkeypatch.setenv("HISTOLOOP_WORKER_PYTHON", "/custom/python")
    config = Config.from_env()
    assert config.state_dir == "/state" and config.request_threads == 3
    assert not config.enable_compute and config.python == "/custom/python"
    monkeypatch.setenv("HISTOLOOP_ENABLE_COMPUTE", "invalid")
    with pytest.raises(ValueError, match="must be 0 or 1"):
        Config.from_env()


def test_bootstrap_with_unregistered_container_uid(setup, monkeypatch):
    _, _, client, _ = setup
    def missing_user():
        raise KeyError("UID is not in /etc/passwd")
    monkeypatch.delenv("HISTOLOOP_REVIEWER", raising=False)
    monkeypatch.setattr("SVC_GPU.human_loop.server.getpass.getuser", missing_user)
    assert client.get("/api/bootstrap").json()["reviewer"] == "Reviewer"
    monkeypatch.setenv("HISTOLOOP_REVIEWER", "Container reviewer")
    assert client.get("/api/bootstrap").json()["reviewer"] == "Container reviewer"


def test_cpu_review_without_original_training_data(setup, tmp_path):
    config, _, _, _ = setup
    config = replace(config, original_manifest=str(tmp_path/"missing.xlsx"),
                     state_dir=str(tmp_path/"portable-state"), device="cpu", enable_compute=False)
    app = create_app(config, token="portable-test")
    with TestClient(app) as client:
        assert client.get("/api/bootstrap").status_code == 401
        assert client.get("/health").status_code == 200
        client.get("/?token=portable-test")
        assert client.get("/api/bootstrap").json()["compute_enabled"] is False
        assert "HistoLoop" in client.get("/").text
        rows = client.get("/api/slides").json()
        assert all(row["original_split"] == "new" for row in rows)
        slide_id = rows[0]["id"]
        assert client.get(f"/api/wsi/{slide_id}.dzi").status_code == 200
        assert client.get(f"/api/wsi/{slide_id}_files/9/0_0.jpg").status_code == 200
        assert correction(client, slide_id).status_code == 200
        assert client.get(f"/api/slides/{slide_id}").json()["manual_tiles"] == 1
        response = client.post("/api/jobs", json={"kind":"predict", "slide_id":slide_id,
                               "reviewer":"Tester", "pass_id":"pass-1"})
        assert response.status_code == 400 and "disabled" in response.text
    restarted = create_app(config, token="portable-test")
    with TestClient(restarted) as client:
        client.get("/?token=portable-test")
        assert client.get(f"/api/slides/{slide_id}").json()["manual_tiles"] == 1
    assert list(Path(config.state_dir).glob("events/*/*.json"))
