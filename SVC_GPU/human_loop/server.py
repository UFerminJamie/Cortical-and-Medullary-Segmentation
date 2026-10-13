from __future__ import annotations

import argparse
import getpass
import hashlib
import hmac
import math
import json
import logging
import resource
import secrets
import socket
import time
import os
from dataclasses import replace
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import anyio.to_thread
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles

from .catalog import Catalog
from .cache import BoundedCache, signature
from .annotations import correction_record
from .annotation_files import AnnotationFiles, Document, SOURCE_NAMES
from .bulk import BulkLabels
from .config import PACKAGE, Config, allocated_gpu
from .geometry import filter_mask, region_indices
from .jobs import Jobs
from .snapshots import export_zip
from .storage import Conflict, Store, atomic_json, read_json
from .wsi import WsiPool
from .transport import ApiCompression, encode_view
from .grid import describe_grid, grid_mask
from .performance import RequestMetrics
from .training import selected_labels, training_thumbnail

logger = logging.getLogger("renal_review")


def create_app(config: Config | None = None, token: str | None = None) -> FastAPI:
    config = config or Config()
    if config.device not in ("gpu", "cpu") or (config.device == "cpu" and config.executor == "local"):
        raise ValueError("CPU review mode submits GPU work through Slurm; use the Slurm executor")
    for name in ("request_threads", "wsi_handles", "jpeg_cache_mib", "prediction_cache_mib", "response_cache_mib"):
        if getattr(config, name) < 1:
            raise ValueError(f"{name} must be positive")
    store, pool = Store(config.state_dir), WsiPool(config.wsi_handles, config.jpeg_cache_mib * 1024**2)
    responses = BoundedCache(16, config.response_cache_mib * 1024**2)
    timings = RequestMetrics()
    catalog = Catalog(config, store)
    annotation_files = AnnotationFiles(config, catalog)
    jobs = Jobs(config, store, catalog)
    bulk = BulkLabels(store, catalog)
    if token is None:
        token_path = store.root / "access_token"
        if not token_path.exists():
            token_path.write_text(secrets.token_urlsafe(32))
            token_path.chmod(0o600)
        token = token_path.read_text().strip()
    atomic_json(store.root / "config.json", config.dict())

    @asynccontextmanager
    async def lifespan(app):
        limiter = anyio.to_thread.current_default_thread_limiter()
        previous = limiter.total_tokens
        limiter.total_tokens = config.request_threads
        try:
            yield
        finally:
            pool.close()
            catalog.read_cache.close()
            responses.clear()
            limiter.total_tokens = previous

    app = FastAPI(title="HistoLoop", lifespan=lifespan, docs_url=None, redoc_url=None)
    app.state.config, app.state.store, app.state.catalog = config, store, catalog
    app.state.pool, app.state.jobs, app.state.token = pool, jobs, token
    app.state.bulk = bulk
    app.state.annotation_files = annotation_files
    app.state.responses = responses
    app.add_middleware(ApiCompression, minimum_size=2048, compresslevel=3)

    @app.middleware("http")
    async def access(request: Request, call_next):
        started = time.perf_counter()
        if request.url.path == "/health":
            return await call_next(request)
        supplied = request.cookies.get("review_session", "")
        bootstrap = request.query_params.get("token", "") if request.url.path == "/" else ""
        if bootstrap and hmac.compare_digest(bootstrap, token):
            response = RedirectResponse("/", status_code=303)
            response.set_cookie("review_session", token, httponly=True, samesite="strict", max_age=7 * 86400)
            response.headers["Referrer-Policy"] = "no-referrer"
            return response
        if not hmac.compare_digest(supplied, token):
            return JSONResponse({"detail": "Open the access link printed in the server log to sign in."}, status_code=401)
        origin = request.headers.get("origin")
        if request.method not in ("GET", "HEAD") and origin and urlparse(origin).netloc != request.headers.get("host"):
            return JSONResponse({"detail": "Cross-origin changes are not accepted"}, status_code=403)
        length = request.headers.get("content-length")
        if length and int(length) > 4_000_000:
            return JSONResponse({"detail": "Request is too large"}, status_code=413)
        response = await call_next(request)
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["X-Content-Type-Options"] = "nosniff"
        if request.url.path.startswith("/api/"):
            response.headers.setdefault("Cache-Control", "no-store")
            response.headers["Server-Timing"] = f"app;dur={(time.perf_counter()-started)*1000:.2f}"
            duration = (time.perf_counter() - started) * 1000
            timings.observe(request.url.path, duration, response.status_code)
            if duration > 1000 or response.status_code >= 500:
                route = request.scope.get("route")
                logger.warning(json.dumps({"event": "slow_or_failed_request", "route": getattr(route, "path", "/api/other"),
                                           "status": response.status_code, "duration_ms": round(duration, 2)}))
        return response

    @app.exception_handler(Conflict)
    async def conflict(request, error):
        return JSONResponse({"detail": str(error)}, status_code=409)

    @app.exception_handler(ValueError)
    async def invalid(request, error):
        return JSONResponse({"detail": str(error)}, status_code=400)

    @app.exception_handler(KeyError)
    async def missing(request, error):
        return JSONResponse({"detail": str(error)}, status_code=404)

    @app.exception_handler(FileNotFoundError)
    async def missing_file(request, error):
        return JSONResponse({"detail": str(error)}, status_code=404)

    def actor(payload: dict) -> str:
        value = str(payload.get("reviewer", "")).strip()
        if not value or len(value) > 120:
            raise ValueError("Enter a reviewer name before annotating")
        return value

    def state_data(slide_id: str, model_id: str = "active", prediction_id: str | None = None):
        data = catalog.current_tiles(slide_id, model_id, prediction_id)
        events = store.events(slide_id, refresh=True)
        if any(event.get("fingerprint") and event["fingerprint"] != data["fingerprint"] for event in events):
            raise ValueError("Source feature files have changed. Saved correction indices cannot be reused.")
        labels, revision, undone = store.resolved(slide_id, len(data["coordinates"]), events)
        return data, labels, revision, events, undone

    def history(slide_id: str, limit: int | None = None, before: int | None = None):
        if limit is not None and not 1 <= limit <= 500:
            raise ValueError("History limit must be between 1 and 500")
        if before is not None and before < 1:
            raise ValueError("History cursor must be positive")
        data, labels, revision, events, undone = state_data(slide_id)
        omitted = {"tile_ids", "tile_coordinates", "prior_labels", "predicted_labels", "decision_scores"}
        selected = [event for event in reversed(events) if before is None or event["sequence"] < before]
        more = limit is not None and len(selected) > limit
        if limit is not None:
            selected = selected[:limit]
        return {"revision": revision, "events": [{**{key: value for key, value in event.items() if key not in omitted},
                "tile_count": len(event.get("tile_ids", [])), "undone": event["id"] in undone} for event in selected],
                "next_before": selected[-1]["sequence"] if more else None}

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/api/performance")
    def performance():
        return {"wsi": pool.metrics(), "responses": responses.metrics(),
                "saved_predictions": catalog._saved_cache.metrics(), "request_threads": config.request_threads,
                "requests": timings.snapshot(), "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024}

    @app.get('/api/slides/{slide_id}/annotations')
    def annotation_layers(slide_id: str):
        annotation_files.recover(slide_id)
        return annotation_files.list(slide_id)

    @app.get('/api/slides/{slide_id}/annotations/{layer}')
    def annotation_document(slide_id: str, layer: str):
        return annotation_files.read(slide_id, layer)

    @app.put('/api/slides/{slide_id}/annotations/{layer}')
    def annotation_save(slide_id: str, layer: str, payload: dict, background: BackgroundTasks):
        payload['reviewer'] = actor(payload)
        result = annotation_files.save(slide_id, layer, payload)
        if layer in SOURCE_NAMES:
            background.add_task(annotation_files.regenerate, slide_id)
        return result

    @app.post('/api/slides/{slide_id}/annotation-generation')
    def annotation_generation(slide_id: str, payload: dict):
        actor(payload)
        return annotation_files.regenerate(slide_id, payload.get('tissue_id'))

    @app.post('/api/annotation-preview')
    def annotation_preview(payload: dict):
        actor(payload)
        return Document(payload['document']).edit(payload.get('operations', []))

    @app.get('/api/slides/{slide_id}/annotation-versions/{layer}')
    def annotation_versions(slide_id: str, layer: str):
        return annotation_files.versions(slide_id, layer)

    @app.post('/api/slides/{slide_id}/annotation-versions/{layer}')
    def annotation_restore(slide_id: str, layer: str, payload: dict, background: BackgroundTasks):
        payload['reviewer'] = actor(payload)
        result = annotation_files.restore(slide_id, layer, payload)
        if layer in SOURCE_NAMES:
            background.add_task(annotation_files.regenerate, slide_id)
        return result

    @app.get('/api/slides/{slide_id}/prediction-tissue')
    def prediction_tissue(slide_id: str):
        try:
            identity, path, document = annotation_files.tissue(slide_id)
            from .annotation_files import checksum
            return {'status': 'available', 'identity': identity, 'revision': checksum(path), 'geometry': __import__('shapely').geometry.mapping(document.union())}
        except ValueError as error:
            return {'status': 'unavailable', 'error': str(error)}

    @app.get('/api/wsi/{slide_id}/thumbnail')
    def prediction_image_guide(slide_id: str):
        slide = catalog.slide(slide_id)
        key = 'guide:' + signature(slide['wsi_path'])
        image = responses.get(key)
        if image is None:
            image = pool.thumbnail(slide['wsi_path'])
            responses.put(key, image, len(image))
        return Response(image, media_type='image/jpeg')

    @app.get("/")
    def index():
        return FileResponse(PACKAGE / "static/index.html")

    @app.get("/api/bootstrap")
    def bootstrap():
        # Docker bind mounts commonly use a host UID without an /etc/passwd row.
        try:
            reviewer = os.environ.get("HISTOLOOP_REVIEWER") or getpass.getuser()
        except (KeyError, OSError):
            reviewer = "Reviewer"
        preference_path = store.root / "preferences" / f"{hashlib.sha256(reviewer.encode()).hexdigest()}.json"
        return {"passes": store.passes(), "models": store.models(config.baseline_model),
            "reviewer": reviewer, "preferences": read_json(preference_path, {}),
            "state_dir": str(store.root), "executor": config.executor, "device": config.device,
            "local_gpu_available": config.device == "gpu" and allocated_gpu(),
            "compute_enabled": config.enable_compute,
            "slide_count": len(catalog.slides), "datasets": catalog.datasets}

    @app.post("/api/preferences")
    def preferences(payload: dict):
        reviewer = actor(payload)
        preferences = payload.get("preferences", {})
        if not isinstance(preferences, dict):
            raise ValueError("Preferences must be an object")
        path = store.root / "preferences" / f"{hashlib.sha256(reviewer.encode()).hexdigest()}.json"
        with store.lock():
            atomic_json(path, preferences)
        return {"saved": True}

    @app.post("/api/datasets")
    def add_dataset(payload: dict):
        actor(payload)
        for key in ("name", "manifest", "predictions"):
            if not isinstance(payload.get(key), str):
                raise ValueError("Enter a cohort name, workbook path, and prediction folder")
        dataset = catalog.add_dataset(payload["name"], payload["manifest"], payload["predictions"])
        return {"dataset": dataset, "datasets": catalog.datasets}

    @app.post("/api/datasets/refresh")
    def refresh_datasets():
        return {"datasets": catalog.refresh_datasets()}

    @app.post("/api/passes")
    def add_pass(payload: dict):
        return store.create_pass(str(payload.get("name", "")))

    @app.get("/api/slides")
    def slides(request: Request, pass_id: str = "pass-1", model_id: str = "active", dataset_id: str | None = None, compact: bool = False):
        stamp = signature(store.root / "review_index.json", store.root / "prediction_index.json",
                          store.root / "prediction_models.json")
        etag = 'W/"' + hashlib.sha256(str((catalog.generation, stamp, pass_id, model_id, dataset_id, compact)).encode()).hexdigest() + '"'
        headers = {"ETag": etag, "Cache-Control": "private, no-cache"}
        if request.headers.get("if-none-match") == etag:
            return Response(status_code=304, headers=headers)
        cached = responses.get(etag)
        if cached is not None:
            return Response(cached, media_type="application/json", headers=headers)
        rows = catalog.listing(pass_id, model_id, dataset_id)
        if compact:
            omitted = {"pt_path", "wsi_path", "baseline_csv", "baseline_threshold", "baseline_available", "path", "dataset_name"}
            rows = [{key: value for key, value in row.items() if key not in omitted} for row in rows]
        response = JSONResponse(rows, headers=headers)
        responses.put(etag, response.body, len(response.body))
        return response

    def slide_detail(slide_id: str, data, labels, revision):
        slide = catalog.slide(slide_id)
        info = pool.info(slide["wsi_path"])
        effective = np.where(labels != -1, labels, data["predictions"])
        return {**slide, **info, "tile_size_level0": data["tile_size"], "tiles": len(labels),
            "fingerprint": data["fingerprint"], "revision": revision, "manual_tiles": int((labels >= 0).sum()),
            "excluded_tiles": int((labels == -2).sum()),
            "cortex": int((effective == 0).sum()), "medulla": int((effective == 1).sum()),
            "model_id": data["model_id"], "prediction_id": data["prediction_id"],
            "prediction_threshold": data["prediction_threshold"],
            "prediction_versions": catalog.prediction_versions(slide_id),
            "predicted_cortex": int((data["predictions"] == 0).sum()),
            "predicted_medulla": int((data["predictions"] == 1).sum()),
            "reviewed_passes": store.summary().get(slide_id, {}).get("reviewed_passes", {}),
            "dzi_url": f"/api/wsi/{slide_id}.dzi", "score_range": [float(data["scores"].min()), float(data["scores"].max())],
            "grid": describe_grid(data)}

    @app.get("/api/slides/{slide_id}")
    def detail(slide_id: str, model_id: str = "active", prediction_id: str | None = None):
        data, labels, revision, events, undone = state_data(slide_id, model_id, prediction_id)
        return slide_detail(slide_id, data, labels, revision)

    @app.get("/api/slides/{slide_id}/view")
    def view(slide_id: str, request: Request, model_id: str = "active", prediction_id: str | None = None):
        data, labels, revision, events, undone = state_data(slide_id, model_id, prediction_id)
        source = catalog.slide(slide_id)
        stamp = signature(source["baseline_csv"], Path(source["wsi_path"]),
                          store.root / "predictions" / slide_id / "versions.json")
        etag = 'W/"' + hashlib.sha256(str((catalog.generation, slide_id, data["fingerprint"], revision,
                    data["prediction_id"], data["prediction_threshold"], stamp,
                    data.get("source_signature"),
                    signature(store.root / "predictions" / slide_id / "active.json"))).encode()).hexdigest() + '"'
        headers = {"X-View-Format": "renal-v1", "ETag": etag, "Cache-Control": "private, no-cache"}
        if request.headers.get("if-none-match") == etag:
            return Response(status_code=304, headers=headers)
        content = responses.get(etag)
        if content is None:
            content = encode_view(slide_detail(slide_id, data, labels, revision), data, labels)
            responses.put(etag, content, len(content))
        return Response(content, media_type="application/octet-stream", headers=headers)

    @app.get("/api/slides/{slide_id}/tiles.bin")
    def binary(slide_id: str, model_id: str = "active", prediction_id: str | None = None):
        data, labels, revision, events, undone = state_data(slide_id, model_id, prediction_id)
        array = np.column_stack((data["coordinates"], data["scores"], data["predictions"], labels)).astype("<f4")
        return Response(array.tobytes(), media_type="application/octet-stream",
                        headers={"X-Review-Revision": str(revision), "X-Prediction-ID": data["prediction_id"]})

    @app.get("/api/slides/{slide_id}/history")
    def slide_history(slide_id: str, limit: int | None = None, before: int | None = None):
        return history(slide_id, limit, before)

    @app.post("/api/slides/{slide_id}/corrections")
    def correct(slide_id: str, payload: dict):
        if str(payload.get("grid_mode", "native")) != "native" and payload.get("grid_format") != 2:
            raise Conflict("Reload the page before applying corrections on an overlap subset.")
        slide = catalog.slide(slide_id)
        data, labels, revision, events, undone = state_data(slide_id, payload.get("view_model_id", "active"), payload.get("view_prediction_id"))
        if payload.get("fingerprint") != data["fingerprint"]:
            raise Conflict("Tile dataset changed; reload this slide")
        if payload.get("prediction_id", data["prediction_id"]) != data["prediction_id"]:
            raise Conflict("New predictions arrived while you were drawing. Reload this slide.")
        action = payload.get("action", "label")
        if action not in ("label", "clear"):
            raise ValueError("Correction action must be label or clear")
        label = payload.get("label")
        if action == "label" and label not in (0, 1, -2):
            raise ValueError("Label must be cortex (0), medulla (1), or excluded (-2)")
        polygon = payload.get("polygon", [])
        filters = payload.get("filters", {})
        display_threshold = float(payload.get('display_threshold', data['prediction_threshold']))
        if not math.isfinite(display_threshold):
            raise ValueError('Display threshold must be finite')
        preview_predictions = (data['scores'] >= display_threshold).astype(np.int8)
        effective = np.where(labels != -1, labels, preview_predictions)
        eligible = filter_mask(data["scores"], effective, labels, filters, preview_predictions) if payload.get("visible_only", True) else None
        selected_grid = grid_mask(data["coordinates"], data["tile_size"], payload.get("grid_mode", "native"),
                                  payload.get("source_overlap", "auto"), data.get("grid"))
        eligible = selected_grid if eligible is None else eligible & selected_grid
        ids = region_indices(data["coordinates"], data["tile_size"], polygon, eligible)
        if not len(ids):
            raise ValueError("No eligible tile centers are inside this region. Enlarge it or change the filters.")
        record = correction_record(slide, data, labels, ids,
            {**payload, "action": action, "label": label, "reviewer": actor(payload)}, polygon)
        event = store.append(slide_id, len(labels), int(payload["revision"]), record)
        return {"event_id": event["id"], "revision": event["sequence"], "tile_ids": ids.tolist(),
                "label": label if action == "label" else -1, "saved_at": event["created_at"]}

    @app.post("/api/slides/{slide_id}/undo")
    def undo(slide_id: str, payload: dict):
        data, labels, revision, events, undone = state_data(slide_id)
        event = store.append(slide_id, len(labels), int(payload["revision"]), {"action": "undo",
            "target_id": payload["target_id"], "pass_id": payload["pass_id"], "reviewer": actor(payload),
            "fingerprint": data["fingerprint"]})
        return {"revision": event["sequence"]}

    @app.post("/api/slides/{slide_id}/review")
    def reviewed(slide_id: str, payload: dict):
        data, labels, revision, events, undone = state_data(slide_id, prediction_id=payload.get("view_prediction_id"))
        event = store.append(slide_id, len(labels), int(payload["revision"]), {"action": "review",
            "pass_id": payload["pass_id"], "reviewer": actor(payload), "fingerprint": data["fingerprint"],
            "model_id": data["model_id"], "prediction_id": data["prediction_id"],
            "prediction_threshold": data["prediction_threshold"]})
        return {"revision": event["sequence"]}

    @app.get("/api/bulk-labels")
    def bulk_list():
        return bulk.listing()

    @app.post("/api/bulk-labels", status_code=202)
    def bulk_label(payload: dict):
        return bulk.submit({**payload, "reviewer": actor(payload)})

    @app.post("/api/bulk-labels/plan")
    def bulk_plan(payload: dict):
        entries = payload.get("slides", [])
        if not isinstance(entries, list):
            raise ValueError("Select slides for the bulk operation")
        result, total, seen = [], 0, set()
        for entry in entries:
            slide_id = entry["id"]
            if slide_id in seen:
                continue
            seen.add(slide_id)
            data, labels, revision, events, undone = state_data(slide_id, prediction_id=entry.get("prediction_id"))
            count = int(grid_mask(data["coordinates"], data["tile_size"], payload.get("grid_mode", "native"),
                                  payload.get("source_overlap", "auto"), data.get("grid")).sum())
            result.append({"id": slide_id, "revision": revision, "prediction_id": data["prediction_id"],
                           "grid_format": 2,
                           "grid_mode": payload.get("grid_mode", "native"),
                           "source_overlap": payload.get("source_overlap", "auto"), "tiles": count})
            total += count
        return {"slides": result, "tiles": total}

    @app.get("/api/wsi/{slide_id}.dzi")
    def dzi(slide_id: str):
        info = pool.info(catalog.slide(slide_id)["wsi_path"])
        xml = f'<Image xmlns="http://schemas.microsoft.com/deepzoom/2008" TileSize="256" Overlap="0" Format="jpg"><Size Width="{info["width"]}" Height="{info["height"]}"/></Image>'
        return Response(xml, media_type="application/xml")

    @app.get("/api/wsi/{slide_id}_files/{level}/{filename}")
    def image_tile(slide_id: str, level: int, filename: str):
        import re

        match = re.fullmatch(r"(\d+)_(\d+)\.jpg", filename)
        if not match:
            raise HTTPException(404)
        content = pool.tile(catalog.slide(slide_id)["wsi_path"], level, int(match.group(1)), int(match.group(2)))
        return Response(content, media_type="image/jpeg", headers={"Cache-Control": "private, max-age=86400"})

    @app.get("/api/training-plan")
    def training_plan(budget: int = 100000):
        from .worker import partition_holdout

        import pandas as pd

        original_rows = catalog.original_rows
        train_rows = [row for row in original_rows if catalog.split(row.get("split", "")) == "train"]
        test_rows = [row for row in original_rows if catalog.split(row.get("split", "")) == "test"]
        summary = store.summary()
        corrected = [catalog.slide(key) for key, row in summary.items() if row.get("manual_tiles") or row.get("excluded_tiles")]
        kept, excluded = partition_holdout(train_rows, test_rows, corrected)
        available = sum(row.get("manual_tiles", 0) for row in summary.values())
        metadata = read_json(Path(config.baseline_model).parent / "metadata.json", {})
        return {"original_train_slides": len(train_rows), "original_train_tiles": metadata.get("tile_counts", {}).get("fit"),
            "corrected_slides": len(corrected), "corrected_tiles": available,
            "excluded_tiles": sum(row.get("excluded_tiles", 0) for row in summary.values()),
            "additional_tiles_selected": min(available, budget) if budget else available,
            "untouched_test_slides": len(kept), "excluded_test_slides": len(excluded),
            "reviewed_slides": sum(bool(row.get("reviewed_passes")) for row in summary.values()), "total_slides": len(catalog.slides)}

    @app.post("/api/jobs", status_code=202)
    def submit(payload: dict):
        return jobs.submit(str(payload.get("kind")), slide_id=payload.get("slide_id"),
            slide_ids=payload.get("slide_ids"),
            model_id=str(payload.get("model_id", "baseline")), threshold=float(payload.get("threshold", -1.0)),
            budget=int(payload.get("budget", 100000)), reviewer=actor(payload), pass_id=str(payload.get("pass_id", "pass-1")),
            executor=payload.get("executor"), training_samples=payload.get("training_samples"),
            validation_samples=payload.get("validation_samples"), model_name=payload.get("model_name", ""),
            c_values=payload.get("c_values"))

    @app.get("/api/training/samples")
    def training_samples():
        rows = catalog.listing("pass-1")
        for row in rows:
            row["revision"] = row.get("revision", 0)
            # Only corrected slides need tile reads to adjust the cheap prediction summary.
            if row["prediction_available"] and (row.get("manual_tiles") or row.get("excluded_tiles")):
                try:
                    _, _, labels, _ = selected_labels(catalog, store, row)
                    count = int(np.isin(labels, [0, 1]).sum())
                    row["medulla_fraction"] = float((labels == 1).sum() / count) if count else None
                except (ValueError, FileNotFoundError) as error:
                    row.update(prediction_available=False, selection_error=str(error))
        return rows

    @app.get("/api/training/samples/{slide_id}/thumbnail")
    def training_image(slide_id: str, prediction_id: str, revision: int = 0):
        slide = catalog.slide(slide_id)
        data, human, labels, _ = selected_labels(catalog, store, {"id": slide_id, "prediction_id": prediction_id, "revision": revision})
        key = "training:" + signature(slide["wsi_path"], slide["pt_path"], slide["baseline_csv"]) + str((slide_id, prediction_id, revision))
        cached = responses.get(key)
        if cached is None:
            cached = training_thumbnail(pool, slide, data, human, labels)
            responses.put(key, cached, len(cached))
        return Response(cached, media_type="image/jpeg", headers={"Cache-Control": "private, no-cache"})

    @app.get("/api/jobs")
    def job_list():
        return jobs.listing()

    @app.get("/api/jobs/{job_id}/logs")
    def job_logs(job_id: str):
        return jobs.logs(job_id)

    @app.get("/api/export")
    def export():
        content = export_zip(catalog, store)
        return Response(content, media_type="application/zip", headers={"Content-Disposition": 'attachment; filename="kpmp_review_history.zip"'})

    app.mount("/static", StaticFiles(directory=PACKAGE / "static"), name="static")
    return app


def main():
    defaults = Config.from_env()
    parser = argparse.ArgumentParser(description="HistoLoop whole-slide annotation server")
    parser.add_argument("--host", default=os.environ.get("HISTOLOOP_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("HISTOLOOP_PORT", "8765")))
    parser.add_argument("--executor", choices=["auto", "slurm", "local"], default=os.environ.get("HISTOLOOP_EXECUTOR", "auto"))
    parser.add_argument("--device", choices=["gpu", "cpu"], default=defaults.device,
                        help="CPU mode reviews slides without a GPU and submits compute jobs through Slurm")
    for flag in ("state-dir", "manifest", "predictions", "datasets-file", "cache-dir", "annotation-root", "contour-root", "annotation-sources",
                 "original-manifest", "original-features", "baseline-model", "account", "qos",
                 "gpu-partition", "gpu-resource", "job-memory", "job-time"):
        parser.add_argument("--" + flag, default=getattr(defaults, flag.replace("-", "_")))
    parser.add_argument("--worker-python", default=defaults.python)
    parser.add_argument("--disable-compute", action="store_true", default=not defaults.enable_compute)
    for flag in ("request-threads", "wsi-handles", "jpeg-cache-mib", "prediction-cache-mib",
                 "response-cache-mib", "job-cpus"):
        parser.add_argument("--" + flag, type=int, default=getattr(defaults, flag.replace("-", "_")))
    args = parser.parse_args()
    if args.executor == "auto":
        args.executor = "local" if args.device == "gpu" and allocated_gpu() else "slurm"
    if args.device == "cpu" and args.executor == "local":
        parser.error("CPU review mode uses --executor slurm for GPU prediction and training")
    config = replace(defaults, state_dir=str(Path(args.state_dir).resolve()), executor=args.executor, device=args.device,
                    manifest=str(Path(args.manifest).resolve()), predictions=str(Path(args.predictions).resolve()),
                    datasets_file=str(Path(args.datasets_file).resolve()) if args.datasets_file else "",
                    cache_dir=str(Path(args.cache_dir).resolve()) if args.cache_dir else "",
                    annotation_root=args.annotation_root, contour_root=args.contour_root, annotation_sources=args.annotation_sources,
                    request_threads=args.request_threads, wsi_handles=args.wsi_handles,
                    jpeg_cache_mib=args.jpeg_cache_mib, prediction_cache_mib=args.prediction_cache_mib,
                    response_cache_mib=args.response_cache_mib,
                    job_cpus=args.job_cpus, job_memory=args.job_memory, job_time=args.job_time,
                    original_manifest=args.original_manifest, original_features=args.original_features,
                    baseline_model=args.baseline_model, python=args.worker_python,
                    account=args.account, qos=args.qos, gpu_partition=args.gpu_partition,
                    gpu_resource=args.gpu_resource, enable_compute=not args.disable_compute)
    app = create_app(config)
    print(f"Node: {socket.gethostname()} | Port: {args.port} | {len(app.state.catalog.slides)} review slides", flush=True)
    print(f"Saved corrections: {app.state.store.root}", flush=True)
    print(f"Access: http://127.0.0.1:{args.port}/?token={app.state.token}", flush=True)
    print("HistoLoop: open the access link locally, or forward this port through SSH / VS Code.", flush=True)
    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, workers=1, proxy_headers=True)


if __name__ == "__main__":
    main()
