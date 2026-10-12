"""Rebuildable node-local read caches; shared audit records remain authoritative."""
from __future__ import annotations

from collections import OrderedDict
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import threading
import uuid

import numpy as np


class BoundedCache:
    """Thread-safe LRU bounded by both entry count and caller-supplied byte size."""
    def __init__(self, capacity=16, max_bytes=48 * 1024 * 1024):
        if capacity < 1 or max_bytes < 1:
            raise ValueError("Cache limits must be positive")
        self.capacity, self.max_bytes = capacity, max_bytes
        self._lock = threading.RLock()
        self._items = OrderedDict()
        self._bytes = self.hits = self.misses = self.evictions = 0

    def get(self, key):
        with self._lock:
            item = self._items.get(key)
            if item is None:
                self.misses += 1
                return None
            self.hits += 1
            self._items.move_to_end(key)
            return item[0]

    def put(self, key, value, size):
        with self._lock:
            old = self._items.pop(key, None)
            if old is not None:
                self._bytes -= old[1]
            if size > self.max_bytes:
                return
            self._items[key] = (value, size)
            self._bytes += size
            while len(self._items) > self.capacity or self._bytes > self.max_bytes:
                _, (_, removed) = self._items.popitem(last=False)
                self._bytes -= removed
                self.evictions += 1

    def clear(self):
        with self._lock:
            self._items.clear()
            self._bytes = 0

    def metrics(self):
        with self._lock:
            return {"entries": len(self._items), "bytes": self._bytes,
                    "max_bytes": self.max_bytes, "hits": self.hits,
                    "misses": self.misses, "evictions": self.evictions}


def signature(*paths) -> str:
    parts = []
    for value in paths:
        path = Path(value)
        try:
            stat = path.stat()
            parts.append((str(path), stat.st_mtime_ns, stat.st_size))
        except FileNotFoundError:
            parts.append((str(path), None, None))
    return hashlib.sha256(json.dumps(parts).encode()).hexdigest()


class JsonCache:
    def __init__(self, capacity=128):
        self.capacity = capacity
        self._lock = threading.RLock()
        self._items = OrderedDict()

    def read(self, path: Path, default=None):
        try:
            stamp = path.stat()
        except FileNotFoundError:
            return default
        key = str(path)
        stamp = (stamp.st_mtime_ns, stamp.st_size)
        with self._lock:
            item = self._items.get(key)
            if item and item[0] == stamp:
                self._items.move_to_end(key)
                return item[1]
            with path.open(encoding="utf-8") as handle:
                value = json.load(handle)
            self._items[key] = (stamp, value)
            while len(self._items) > self.capacity:
                self._items.popitem(last=False)
            return value


class ReadCache:
    """SQLite indexes validated cohort metadata and bounded NumPy cache artifacts."""
    def __init__(self, state_dir: str, root: str = "", max_bytes=512 * 1024 * 1024):
        identity = hashlib.sha256(str(Path(state_dir).absolute()).encode()).hexdigest()[:20]
        self.root = Path(root) if root else Path("/tmp") / f"renal-review-cache-{os.getuid()}" / identity
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.max_bytes = max_bytes
        self._lock = threading.RLock()
        self._db = sqlite3.connect(self.root / "index.sqlite3", timeout=10, check_same_thread=False)
        self._db.execute("PRAGMA journal_mode=DELETE")
        self._db.execute("CREATE TABLE IF NOT EXISTS cache_entries (key TEXT PRIMARY KEY, signature TEXT NOT NULL, payload TEXT NOT NULL)")
        self._db.commit()

    def get(self, key: str, stamp: str):
        with self._lock:
            row = self._db.execute("SELECT payload FROM cache_entries WHERE key=? AND signature=?", (key, stamp)).fetchone()
        return json.loads(row[0]) if row else None

    def close(self):
        with self._lock:
            self._db.close()

    def put(self, key: str, stamp: str, value):
        with self._lock, self._db:
            self._db.execute("INSERT OR REPLACE INTO cache_entries VALUES (?, ?, ?)", (key, stamp, json.dumps(value)))

    def arrays(self, key: str, stamp: str):
        metadata = self.get(key, stamp)
        if not metadata:
            return None
        path = self.root / metadata["artifact"]
        try:
            with np.load(path, allow_pickle=False) as saved:
                arrays = {name: saved[name] for name in ("coordinates", "scores", "predictions")}
            path.touch()
        except (OSError, ValueError, KeyError):
            return None
        return {**arrays, **{name: value for name, value in metadata.items() if name != "artifact"}}

    def put_arrays(self, key: str, stamp: str, data: dict):
        name = hashlib.sha256((key + stamp).encode()).hexdigest() + ".npz"
        destination = self.root / name
        temporary = self.root / f".{uuid.uuid4().hex}.tmp"
        try:
            with temporary.open("wb") as handle:
                np.savez(handle, **{name: data[name] for name in ("coordinates", "scores", "predictions")})
            temporary.replace(destination)
            metadata = {name: value for name, value in data.items() if name not in ("coordinates", "scores", "predictions")}
            self.put(key, stamp, {**metadata, "artifact": name})
            self.prune()
        finally:
            temporary.unlink(missing_ok=True)

    def prune(self):
        artifacts = []
        for path in self.root.glob("*.npz"):
            try:
                artifacts.append((path.stat().st_mtime_ns, path))
            except FileNotFoundError:
                pass
        artifacts = [path for _, path in sorted(artifacts, reverse=True)]
        total = 0
        for path in artifacts:
            try:
                total += path.stat().st_size
                if total > self.max_bytes:
                    path.unlink(missing_ok=True)
            except FileNotFoundError:
                pass
