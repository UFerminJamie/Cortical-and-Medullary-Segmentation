"""Dataset selection preserves source predictions, slide identity, and corrections."""
from dataclasses import replace
from pathlib import Path
import json
import shutil
import socket
import threading
import time

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from SVC_GPU.human_loop.server import create_app
from SVC_GPU.human_loop.storage import atomic_json
from conftest import correction


def multi_config(config):
    root = Path(config.manifest).parent
    frame = pd.read_excel(config.manifest)
    uf = frame.iloc[:1].copy()
    features = root / "uf_features"
    features.mkdir()
    feature = features / Path(uf.iloc[0].feature_path).name
    shutil.copyfile(uf.iloc[0].feature_path, feature)
    uf.loc[uf.index[0], "feature_path"] = str(feature)
    uf_manifest = root / "uf.xlsx"
    uf.to_excel(uf_manifest, index=False)
    predictions = root / "uf_outputs"
    (predictions / "predictions").mkdir(parents=True)
    name = Path(uf.iloc[0].sample_name).stem
    csv = pd.read_csv(Path(config.predictions) / "predictions" / f"{name}.csv")
    csv["decision_score"] = [-5, -4, -3, -2]
    csv["predicted_label_encoded"] = 0
    csv.to_csv(predictions / "predictions" / f"{name}.csv", index=False)
    atomic_json(predictions / "inference_summary.json", {"threshold": -1, "samples": [
        {"sample": name, "tiles": 4, "cortical": 4, "medullary": 0}]})
    dn = root / "dn.xlsx"
    frame.iloc[2:].to_excel(dn, index=False)
    sources = [
        {"id": "kpmp", "name": "KPMP", "manifest": config.manifest, "predictions": config.predictions},
        {"id": "uf", "name": "UF CKD", "manifest": "uf.xlsx", "predictions": "uf_outputs"},
        {"id": "dn", "name": "DN", "manifest": "dn.xlsx", "predictions": "dn_outputs"},
        {"id": "fsgs", "name": "FSGS", "manifest": "missing.xlsx", "predictions": "missing_outputs"},
    ]
    # DN uses a distinct feature path too, even though these synthetic names overlap.
    dn_frame = pd.read_excel(dn)
    dn_feature = root / "dn.pt"
    shutil.copyfile(dn_frame.iloc[0].feature_path, dn_feature)
    dn_frame.loc[dn_frame.index[0], "feature_path"] = str(dn_feature)
    dn_frame.to_excel(dn, index=False)
    registry = root / "datasets.json"
    registry.write_text(json.dumps(sources))
    return replace(config, datasets_file=str(registry))


def new_cohort(config):
    root = Path(config.manifest).parent
    frame = pd.read_excel(config.manifest).iloc[:1].copy()
    feature = root / "new_cohort.pt"
    shutil.copyfile(frame.iloc[0].feature_path, feature)
    frame.loc[frame.index[0], "feature_path"] = str(feature)
    frame.loc[frame.index[0], "sample_name"] = "New_Cohort.svs"
    manifest = root / "new_cohort.xlsx"
    frame.to_excel(manifest, index=False)
    predictions = root / "new_cohort_outputs"
    (predictions / "predictions").mkdir(parents=True)
    shutil.copyfile(Path(config.predictions) / "predictions/A_PAS_1.csv",
                    predictions / "predictions/New_Cohort.csv")
    atomic_json(predictions / "inference_summary.json", {"threshold": -.75, "samples": [
        {"sample": "New_Cohort", "tiles": 4, "cortical": 2, "medullary": 2}]})
    return {"reviewer": "Tester", "name": "New cohort", "manifest": str(manifest), "predictions": str(predictions)}


def test_add_cohort_persists_for_restart_and_workers(setup):
    config, app, client, slides = setup
    assert correction(client, slides["A"]["id"]).status_code == 200
    response = client.post("/api/datasets", json=new_cohort(config))
    assert response.status_code == 200, response.text
    dataset = response.json()["dataset"]
    assert dataset["slide_count"] == 1 and dataset["missing_predictions"] == 0
    rows = client.get("/api/slides", params={"dataset_id": dataset["id"]}).json()
    assert len(rows) == 1
    assert client.get(f"/api/slides/{rows[0]['id']}").json()["prediction_threshold"] == -.75
    saved = Path(config.state_dir) / "added_datasets.json"
    assert json.loads(saved.read_text())[0]["id"] == dataset["id"]
    assert any(source["id"] == dataset["id"] for source in config.dataset_sources())
    restarted = create_app(config, token="test-access")
    with TestClient(restarted) as other:
        other.get("/?token=test-access")
        assert other.get("/api/slides", params={"dataset_id": dataset["id"]}).json()[0]["id"] == rows[0]["id"]
        assert other.get(f"/api/slides/{slides['A']['id']}").json()["manual_tiles"] > 0
    assert app.state.jobs.catalog.slide(rows[0]["id"])["dataset_id"] == dataset["id"]


def test_add_cohort_rejects_invalid_without_partial_changes(setup):
    config, app, client, slides = setup
    payload = new_cohort(config)
    count = len(app.state.catalog.slides)
    for change in ({"manifest": "relative.xlsx"}, {"manifest": "/tmp/nonexistent_cohort.xlsx"},
                   {"predictions": "/tmp/nonexistent_cohort_outputs"}, {"name": ""}, {"name": 42}):
        assert client.post("/api/datasets", json={**payload, **change}).status_code == 400
        assert len(app.state.catalog.slides) == count
    frame = pd.read_excel(payload["manifest"])
    original = frame.copy()
    frame.loc[frame.index[0], "feature_path"] = "/tmp/nonexistent_cohort.pt"
    frame.to_excel(payload["manifest"], index=False)
    assert client.post("/api/datasets", json=payload).status_code == 400
    assert len(app.state.catalog.slides) == count
    assert not (Path(config.state_dir) / "added_datasets.json").exists()
    original.to_excel(payload["manifest"], index=False)
    assert client.post("/api/datasets", json=payload).status_code == 200
    before = (Path(config.state_dir) / "added_datasets.json").read_bytes()
    assert client.post("/api/datasets", json=payload).status_code == 400
    assert client.post("/api/datasets", json={**payload, "name": "Renamed cohort"}).status_code == 400
    assert (Path(config.state_dir) / "added_datasets.json").read_bytes() == before


def test_refresh_cohorts_discovers_completed_outputs(setup):
    config, app, client, slides = setup
    multi = multi_config(config)
    refreshed_app = create_app(multi, token="test-access")
    with TestClient(refreshed_app) as other:
        other.get("/?token=test-access")
        dn = other.get("/api/slides?dataset_id=dn").json()[0]
        assert not dn["prediction_available"]
        outputs = Path(config.manifest).parent / "dn_outputs"
        (outputs / "predictions").mkdir(parents=True)
        shutil.copyfile(Path(config.predictions) / "predictions/C_PAS_1.csv", outputs / "predictions/C_PAS_1.csv")
        response = other.post("/api/datasets/refresh")
        assert response.status_code == 200, response.text
        dn_after = other.get("/api/slides?dataset_id=dn").json()[0]
        assert dn_after["id"] == dn["id"] and dn_after["prediction_available"]
        assert other.get(f"/api/slides/{dn['id']}").status_code == 200


def test_dataset_sources_and_selection_preserve_reviews(setup):
    config, old_app, old_client, old_slides = setup
    original_id = old_slides["A"]["id"]
    assert correction(old_client, original_id).status_code == 200
    app = create_app(multi_config(config), token="test-access")
    with TestClient(app) as client:
        client.get("/?token=test-access")
        datasets = {row["id"]: row for row in client.get("/api/bootstrap").json()["datasets"]}
        assert datasets["kpmp"]["slide_count"] == 3
        assert datasets["uf"]["slide_count"] == 1
        assert datasets["dn"]["missing_predictions"] == 1
        assert not datasets["fsgs"]["available"]
        kpmp = client.get("/api/slides?dataset_id=kpmp").json()
        uf = client.get("/api/slides?dataset_id=uf").json()
        assert len(kpmp) == 3 and len(uf) == 1
        assert uf[0]["name"] == old_slides["A"]["name"]
        assert uf[0]["id"] != original_id
        assert all(row["dataset_id"] == "kpmp" for row in kpmp)
        assert client.get(f"/api/slides/{original_id}").json()["manual_tiles"] > 0
        detail = client.get(f"/api/slides/{uf[0]['id']}").json()
        assert detail["manual_tiles"] == 0
        assert detail["prediction_threshold"] == -1
        assert detail["predicted_cortex"] == 4
        assert detail["prediction_versions"][0]["threshold"] == -1
        # Requests do not change a process-wide active dataset.
        assert len(client.get("/api/slides?dataset_id=kpmp").json()) == 3
        assert client.get("/api/slides?dataset_id=unknown").status_code == 404
        assert client.get("/api/slides?dataset_id=fsgs").status_code == 400
        dn = client.get("/api/slides?dataset_id=dn").json()
        assert len(dn) == 1 and not dn[0]["prediction_available"]
        assert client.get(f"/api/slides/{dn[0]['id']}").status_code == 404
        assert client.post("/api/preferences", json={"reviewer": "Tester", "preferences": {"dataset_id": "uf"}}).status_code == 200


def test_single_dataset_override_and_duplicate_ids(setup):
    config, app, client, slides = setup
    assert config.dataset_sources()[0]["manifest"] == config.manifest
    multi = multi_config(config)
    registry = Path(multi.datasets_file)
    sources = json.loads(registry.read_text())
    sources[1]["id"] = "kpmp"
    registry.write_text(json.dumps(sources))
    try:
        multi.dataset_sources()
    except ValueError as error:
        assert "Duplicate dataset ID" in str(error)
    else:
        raise AssertionError("Duplicate dataset IDs accepted")


def test_browser_dataset_switch_and_restore(setup):
    sync_playwright = pytest.importorskip("playwright.sync_api").sync_playwright
    import uvicorn

    config, old_app, old_client, old_slides = setup
    payload = new_cohort(config)
    app = create_app(multi_config(config), token="test-access")
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, log_level="error"))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 10
        while not server.started and time.monotonic() < deadline:
            time.sleep(.02)
        assert server.started
        with sync_playwright() as manager:
            if not Path(manager.chromium.executable_path).is_file():
                pytest.skip("Set PLAYWRIGHT_BROWSERS_PATH to an installed Chromium browser")
            browser = manager.chromium.launch(headless=True, args=["--no-sandbox"])
            page = browser.new_page(viewport={"width": 1600, "height": 1000})
            errors = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.goto(f"http://127.0.0.1:{port}/?token=test-access")
            page.wait_for_function("S.detail && !S.loading && !S.queueBusy")
            assert page.locator("#datasetSelect").input_value() == "kpmp"
            assert page.locator("#totalSlides").inner_text() == "3"
            assert page.locator('#datasetSelect option[value="fsgs"]').evaluate("option => option.disabled")
            # An unsaved region cannot be abandoned by switching workbooks.
            page.evaluate("S.draft=[[0,0],[10,0],[10,10]]; $('datasetSelect').value='uf'; switchDataset()")
            assert page.locator("#datasetSelect").input_value() == "kpmp"
            page.evaluate("S.draft=[]; updateButtons()")
            page.locator("#datasetSelect").select_option("uf")
            page.wait_for_function("S.detail?.dataset_id==='uf' && !S.loading")
            assert page.locator("#totalSlides").inner_text() == "1"
            assert page.evaluate("S.detail.prediction_threshold") == -1
            page.wait_for_function("!S.queueBusy && !S.saving")
            # Wait for the debounced preference save by inspecting the backing file.
            deadline = time.monotonic() + 5
            preference = Path(config.state_dir) / "preferences"
            while time.monotonic() < deadline:
                files = list(preference.glob("*.json"))
                if files and json.loads(files[0].read_text()).get("dataset_id") == "uf":
                    break
                page.wait_for_timeout(50)
            else:
                raise AssertionError("Selected dataset was not saved")
            page.reload()
            page.wait_for_function("S.detail?.dataset_id==='uf' && !S.loading")
            assert page.locator("#datasetSelect").input_value() == "uf"
            page.locator("#datasetSelect").select_option("dn")
            page.wait_for_function("S.dataset==='dn' && !S.queueBusy")
            assert page.evaluate("S.detail") is None
            assert page.locator("#totalSlides").inner_text() == "1"
            assert "need predictions" in page.locator("#datasetStatus").inner_text()
            assert page.locator("#predictButton").is_disabled()
            page.locator("#datasetSelect").select_option("kpmp")
            page.wait_for_function("S.detail?.dataset_id==='kpmp' && !S.loading")
            assert page.locator("#totalSlides").inner_text() == "3"
            page.locator("#addDatasetButton").click()
            page.locator("#datasetName").fill(payload["name"])
            page.locator("#datasetManifest").fill(payload["manifest"])
            page.locator("#datasetPredictions").fill(payload["predictions"])
            page.locator("#submitDataset").click()
            page.wait_for_function("S.detail?.dataset_name==='New cohort' && !S.loading && !S.datasetBusy")
            assert page.locator("#totalSlides").inner_text() == "1"
            assert page.evaluate("S.detail.prediction_threshold") == -.75
            assert page.locator("#datasetDialog").evaluate("dialog => !dialog.open")
            page.locator("#refreshDatasetsButton").click()
            page.wait_for_function("!S.datasetBusy && !S.queueBusy")
            assert page.evaluate("S.detail.dataset_name") == "New cohort"
            assert not errors
            browser.close()
    finally:
        server.should_exit = True
        thread.join(timeout=10)
        sock.close()
