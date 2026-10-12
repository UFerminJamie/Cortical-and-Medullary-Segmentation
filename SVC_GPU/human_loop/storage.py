"""Atomic JSON records and append-only events on the shared /blue filesystem."""
from __future__ import annotations

import fcntl
import json
import os
import threading
import time
import uuid
from collections import OrderedDict
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from .cache import JsonCache


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def read_json(path: Path, default=None):
    if not path.is_file():
        return default
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def atomic_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(data, handle, separators=(",", ":"), allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)


class Conflict(ValueError):
    pass


class Store:
    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._mutex = threading.RLock()
        self._event_cache = OrderedDict()
        self._label_cache = OrderedDict()
        self._json_cache = JsonCache()
        with self.lock():
            if not (self.root / "passes.json").exists():
                atomic_json(self.root / "passes.json", [{"id": "pass-1", "name": "Pass 1", "created_at": now()}])
            self._recover_index()

    def _recover_index(self):
        """The event files are authoritative if a crash interrupted the summary write."""
        summaries = read_json(self.root / "review_index.json", {})
        changed = False
        for directory in (self.root / "events").glob("*"):
            if not directory.is_dir():
                continue
            files = sorted(directory.glob("*.json"))
            revision = int(files[-1].name.split("_", 1)[0]) if files else 0
            if summaries.get(directory.name, {}).get("revision") == revision:
                continue
            events = self.events(directory.name)
            count = max((max(event.get("tile_ids") or [-1]) + 1 for event in events), default=0)
            labels, _, _ = self.resolved(directory.name, count, events)
            reviewed = {}
            for event in events:
                if event["action"] == "review":
                    reviewed[event["pass_id"]] = event["created_at"]
                else:
                    reviewed.pop(event["pass_id"], None)
            summaries[directory.name] = {"revision": revision, "manual_tiles": int((labels >= 0).sum()),
                "excluded_tiles": int((labels == -2).sum()),
                "reviewed_passes": reviewed, "last_changed": events[-1]["created_at"] if events else None}
            changed = True
        if changed:
            atomic_json(self.root / "review_index.json", summaries)

    @contextmanager
    def lock(self):
        # One server process; flock also coordinates snapshotting and GPU workers.
        with self._mutex:
            with (self.root / ".lock").open("a") as handle:
                fcntl.flock(handle, fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    fcntl.flock(handle, fcntl.LOCK_UN)

    def passes(self) -> list[dict]:
        return read_json(self.root / "passes.json", [])

    def create_pass(self, name: str) -> dict:
        name = name.strip()
        if not name or len(name) > 80:
            raise ValueError("Review pass needs a name of 1–80 characters")
        with self.lock():
            passes = self.passes()
            item = {"id": "pass-" + uuid.uuid4().hex[:12], "name": name, "created_at": now()}
            passes.append(item)
            atomic_json(self.root / "passes.json", passes)
            return item

    def events(self, slide_id: str, *, refresh: bool = False) -> list[dict]:
        directory = self.root / "events" / slide_id
        if not directory.exists():
            return []
        with self._mutex:
            while True:
                stamp = directory.stat().st_mtime_ns
                cached = self._event_cache.get(slide_id)
                if cached and cached[0] == stamp and not refresh:
                    self._event_cache.move_to_end(slide_id)
                    return cached[1]
                # Events are immutable. Reuse parsed records but always discover the
                # current file list for revision validation on a shared filesystem.
                known = {event["id"]: event for event in cached[1]} if cached else {}
                events = [known.get(path.stem.split("_", 1)[-1]) or read_json(path)
                          for path in sorted(directory.glob("*.json"))]
                if directory.stat().st_mtime_ns != stamp:
                    continue
                self._event_cache[slide_id] = (stamp, events)
                self._event_cache.move_to_end(slide_id)
                # Python objects use more space than the on-disk JSON. Bound an
                # intentionally conservative estimate as well as entry count.
                def footprint(records):
                    return sum(2048 + len(event.get("polygon") or []) * 160 +
                        sum(len(event.get(key) or []) * size for key, size in
                            (("tile_ids", 40), ("tile_coordinates", 160), ("prior_labels", 40),
                             ("predicted_labels", 40), ("decision_scores", 40))) for event in records)
                while self._event_cache and (len(self._event_cache) > 8 or
                        sum(footprint(item[1]) for item in self._event_cache.values()) > 64 * 1024**2):
                    self._event_cache.popitem(last=False)
                return events

    def resolved(self, slide_id: str, count: int, events: list[dict] | None = None) -> tuple[np.ndarray, int, set[str]]:
        events = self.events(slide_id) if events is None else events
        key = (slide_id, count, events[-1]["id"] if events else "empty")
        with self._mutex:
            if key in self._label_cache:
                self._label_cache.move_to_end(key)
                return self._label_cache[key]
        labels = np.full(count, -1, dtype=np.int8)
        undone = {event["target_id"] for event in events if event["action"] == "undo"}
        for event in events:
            if event["id"] in undone:
                continue
            if event["action"] in ("label", "clear"):
                indices = np.asarray(event["tile_ids"], dtype=np.int64)
                if len(indices) and (indices.min() < 0 or indices.max() >= count):
                    raise ValueError("Saved correction indices do not match this slide's tile dataset")
                labels[indices] = int(event["label"]) if event["action"] == "label" else -1
        labels.flags.writeable = False
        result = (labels, max((event["sequence"] for event in events), default=0), undone)
        with self._mutex:
            self._label_cache[key] = result
            self._label_cache.move_to_end(key)
            while self._label_cache and (len(self._label_cache) > 16 or
                    sum(value[0].nbytes for value in self._label_cache.values()) > 32 * 1024**2):
                self._label_cache.popitem(last=False)
        return result

    def append(self, slide_id: str, count: int, expected_revision: int, payload: dict) -> dict:
        with self.lock():
            events = self.events(slide_id, refresh=True)
            labels, revision, undone = self.resolved(slide_id, count, events)
            if expected_revision != revision:
                raise Conflict("This slide was changed in another window. Reload it before saving.")
            if payload["pass_id"] not in {item["id"] for item in self.passes()}:
                raise ValueError("Unknown review pass")
            if payload["action"] == "undo":
                targets = [event for event in events if event["id"] == payload["target_id"]]
                if not targets or targets[0]["action"] not in ("label", "clear") or payload["target_id"] in undone:
                    raise ValueError("This correction cannot be undone")
            event = {**payload, "id": uuid.uuid4().hex, "slide_id": slide_id,
                     "sequence": revision + 1, "created_at": now(), "time_ns": time.time_ns()}
            atomic_json(self.root / "events" / slide_id / f"{revision + 1:08d}_{event['id']}.json", event)
            # A successful append must never leave a pre-save history cached.
            self._event_cache.pop(slide_id, None)
            labels, revision, _ = self.resolved(slide_id, count, [*events, event])
            summaries = read_json(self.root / "review_index.json", {})
            previous = summaries.get(slide_id, {})
            reviewed = previous.get("reviewed_passes", {})
            if event["action"] == "review":
                reviewed[event["pass_id"]] = event["created_at"]
            else:
                reviewed.pop(event["pass_id"], None)
            summaries[slide_id] = {"revision": revision, "manual_tiles": int((labels >= 0).sum()),
                                  "excluded_tiles": int((labels == -2).sum()),
                                  "reviewed_passes": reviewed, "last_changed": event["created_at"]}
            atomic_json(self.root / "review_index.json", summaries)
            return event

    def summary(self) -> dict:
        return self._json_cache.read(self.root / "review_index.json", {})

    def models(self, baseline: str) -> list[dict]:
        result = [{"id": "baseline", "name": "Original SVC", "path": baseline, "created_at": None}]
        result.extend(read_json(path) for path in sorted((self.root / "models").glob("*.json")))
        return result

    def model(self, model_id: str, baseline: str) -> dict:
        for model in self.models(baseline):
            if model["id"] == model_id:
                return model
        raise ValueError("Unknown model version")
