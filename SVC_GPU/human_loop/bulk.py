"""Background CPU annotation batches with frozen slide IDs and revision guards."""
from __future__ import annotations

import threading
import uuid

import numpy as np

from .annotations import correction_record, full_slide_polygon
from .storage import Conflict, atomic_json, now, read_json
from .grid import grid_mask


class BulkLabels:
    def __init__(self, store, catalog):
        self.store, self.catalog = store, catalog
        self.threads = {}
        # The committed per-slide events are authoritative; never silently resume a batch.
        for path in (store.root / "bulk").glob("*/status.json"):
            status = read_json(path)
            if status["status"] in ("queued", "running"):
                status.update(status="interrupted", finished_at=now(), error="Server restarted. Completed slide labels remain saved; review results before retrying.")
                atomic_json(path, status)

    def listing(self):
        return [read_json(path) for path in sorted((self.store.root / "bulk").glob("*/status.json"), reverse=True)]

    def submit(self, payload):
        entries = payload.get("slides")
        if not isinstance(entries, list) or not 1 <= len(entries) <= len(self.catalog.slides):
            raise ValueError("Choose at least one slide from this workspace")
        if payload.get("label") not in (0, 1):
            raise ValueError("Whole-slide label must be cortex or medulla")
        if payload.get("pass_id") not in {p["id"] for p in self.store.passes()}:
            raise ValueError("Unknown review pass")
        seen = set()
        for entry in entries:
            mode = str(entry.get("grid_mode", payload.get("grid_mode", "native")))
            if mode != "native" and entry.get("grid_format", payload.get("grid_format")) != 2:
                raise ValueError("Reload the page before applying labels on an overlap subset")
            self.catalog.slide(entry["id"])
            if entry["id"] in seen or not isinstance(entry.get("revision"), int) or entry["revision"] < 0:
                raise ValueError("Duplicate slide or invalid annotation revision")
            seen.add(entry["id"])
        model_id = payload.get("model_id", "active")
        if model_id != "active":
            self.store.model(model_id, self.catalog.config.baseline_model)
        with self.store.lock():
            if any(row["status"] in ("queued", "running") for row in self.listing()):
                raise Conflict("A bulk annotation is still running. Wait before starting another.")
            bulk_id = "bulk-" + uuid.uuid4().hex[:16]
            directory = self.store.root / "bulk" / bulk_id
            request = {**payload, "id": bulk_id, "created_at": now()}
            status = {"id": bulk_id, "status": "queued", "created_at": request["created_at"],
                "label": payload["label"], "reviewer": payload["reviewer"], "pass_id": payload["pass_id"],
                "total": len(entries), "completed": 0, "failed": 0, "tiles": 0, "results": []}
            atomic_json(directory / "request.json", request)
            atomic_json(directory / "status.json", status)
        thread = threading.Thread(target=self.run, args=(request,), daemon=True, name=bulk_id)
        self.threads[bulk_id] = thread
        thread.start()
        return status

    def run(self, request):
        path = self.store.root / "bulk" / request["id"] / "status.json"
        status = read_json(path)
        status["status"] = "running"
        atomic_json(path, status)
        for entry in request["slides"]:
            slide_id = entry["id"]
            try:
                slide = self.catalog.slide(slide_id)
                data = self.catalog.current_tiles(slide_id, request.get("model_id", "active"), entry.get("prediction_id"))
                events = self.store.events(slide_id)
                if any(e.get("fingerprint") and e["fingerprint"] != data["fingerprint"] for e in events):
                    raise ValueError("Source features changed; labels cannot be reused")
                labels, revision, undone = self.store.resolved(slide_id, len(data["coordinates"]), events)
                if revision != entry["revision"]:
                    raise Conflict("Slide changed since the batch was confirmed; skipped")
                mode = entry.get("grid_mode", request.get("grid_mode", "native"))
                source_overlap = entry.get("source_overlap", request.get("source_overlap", "auto"))
                ids = np.flatnonzero(grid_mask(data["coordinates"], data["tile_size"], mode, source_overlap, data.get("grid")))
                if not len(ids):
                    raise ValueError("Slide has no feature tiles")
                payload = {**request, "action": "label", "shape": "whole_slide", "bulk_id": request["id"], "visible_only": False,
                           "grid_mode": mode, "source_overlap": source_overlap}
                event = self.store.append(slide_id, len(labels), revision,
                    correction_record(slide, data, labels, ids, payload, full_slide_polygon(data)))
                result = {"slide_id": slide_id, "name": slide["name"], "status": "saved",
                          "event_id": event["id"], "revision": event["sequence"], "tiles": len(ids)}
                status["completed"] += 1
                status["tiles"] += len(ids)
            except Exception as error:
                result = {"slide_id": slide_id, "status": "skipped", "error": str(error)}
                status["failed"] += 1
            status["results"].append(result)
            atomic_json(path, status)
        status.update(status="completed" if not status["failed"] else "completed_with_skips", finished_at=now())
        atomic_json(path, status)
