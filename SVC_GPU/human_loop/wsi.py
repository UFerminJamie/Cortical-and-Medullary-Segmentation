"""On-demand Deep Zoom tiles, bounded JPEG cache, and pooled WSI readers."""
from __future__ import annotations

import io
import math
import threading
from concurrent.futures import Future
from dataclasses import dataclass
from collections import OrderedDict
from pathlib import Path

from PIL import Image
from tiffslide import TiffSlide
from tifffile import TiffFileError

TILE_SIZE = 256


@dataclass
class Reader:
    slide: TiffSlide
    lock: threading.RLock
    leases: int = 0


class WsiPool:
    def __init__(self, capacity: int = 4, cache_bytes: int = 96 * 1024 * 1024):
        if capacity < 1 or cache_bytes < 1:
            raise ValueError("WSI cache limits must be positive")
        self.capacity, self.cache_bytes = capacity, cache_bytes
        self._lock = threading.RLock()
        self._readers: OrderedDict[str, Reader] = OrderedDict()
        self._opens = tuple(threading.RLock() for _ in range(64))
        self._flights = {}
        self._info = OrderedDict()
        self._jpeg: OrderedDict[tuple, bytes] = OrderedDict()
        self._jpeg_bytes = 0
        self._hits = self._misses = self._coalesced = self._opened = 0

    def _acquire(self, path: str):
        opener = self._opens[hash(path) % len(self._opens)]
        with opener:
            with self._lock:
                entry = self._readers.get(path)
                if entry is not None:
                    entry.leases += 1
                    self._readers.move_to_end(path)
            if entry is None:
                if not Path(path).is_file():
                    raise FileNotFoundError(f"Source WSI does not exist: {path}")
                try:
                    entry = Reader(TiffSlide(path), threading.RLock())
                except TiffFileError as error:
                    raise ValueError(f"Cannot open WSI {Path(path).name}: corrupted or unsupported TIFF") from error
                with self._lock:
                    self._opened += 1
                    self._readers[path] = entry
                    entry.leases += 1
                    self._readers.move_to_end(path)
        # Waiting for a busy slide never holds the global pool/cache lock.
        entry.lock.acquire()
        return entry

    def _release(self, entry):
        entry.lock.release()
        closing = []
        with self._lock:
            entry.leases -= 1
            for path, candidate in list(self._readers.items()):
                if len(self._readers) <= self.capacity:
                    break
                if candidate.leases == 0:
                    del self._readers[path]
                    closing.append(candidate.slide)
        for slide in closing:
            slide.close()

    def info(self, path: str) -> dict:
        with self._lock:
            if path in self._info:
                self._info.move_to_end(path)
                return dict(self._info[path])
        entry = self._acquire(path)
        slide = entry.slide
        try:
            width, height = map(int, slide.dimensions)
            info = {"width": width, "height": height, "max_level": math.ceil(math.log2(max(width, height))),
                    "tile_size": TILE_SIZE, "mpp": slide.properties.get("openslide.mpp-x")}
            with self._lock:
                self._info[path] = info
                while len(self._info) > 256:
                    self._info.popitem(last=False)
            return dict(info)
        finally:
            self._release(entry)

    def thumbnail(self, path: str) -> bytes:
        """Bounded RGB image guide for spatial refinement; independent of viewport."""
        entry = self._acquire(path)
        try:
            image = entry.slide.get_thumbnail((1024, 1024)).convert('RGB')
            output = io.BytesIO()
            image.save(output, format='JPEG', quality=85)
            return output.getvalue()
        finally:
            self._release(entry)

    def tile(self, path: str, level: int, x: int, y: int) -> bytes:
        key = (path, level, x, y)
        with self._lock:
            cached = self._jpeg.get(key)
            if cached is not None:
                self._hits += 1
                self._jpeg.move_to_end(key)
                return cached
            flight = self._flights.get(key)
            leader = flight is None
            if leader:
                self._misses += 1
                flight = self._flights[key] = Future()
            else:
                self._coalesced += 1
        if not leader:
            return flight.result()
        try:
            result = self._decode(path, level, x, y)
            with self._lock:
                self._jpeg[key] = result
                self._jpeg_bytes += len(result)
                while self._jpeg_bytes > self.cache_bytes and self._jpeg:
                    _, item = self._jpeg.popitem(last=False)
                    self._jpeg_bytes -= len(item)
            flight.set_result(result)
            return result
        except BaseException as error:
            if isinstance(error, TiffFileError):
                error = ValueError(f"Cannot decode WSI {Path(path).name}: corrupted TIFF tile")
            flight.set_exception(error)
            raise error
        finally:
            with self._lock:
                self._flights.pop(key, None)

    def _decode(self, path: str, level: int, x: int, y: int) -> bytes:
        entry = self._acquire(path)
        slide = entry.slide
        try:
            width, height = slide.dimensions
            max_level = math.ceil(math.log2(max(width, height)))
            if not 0 <= level <= max_level or x < 0 or y < 0:
                raise ValueError("Invalid Deep Zoom tile")
            downsample = 2 ** (max_level - level)
            origin = (x * TILE_SIZE * downsample, y * TILE_SIZE * downsample)
            if origin[0] >= width or origin[1] >= height:
                raise ValueError("Tile is outside the slide")
            out_width = min(TILE_SIZE, math.ceil((width - origin[0]) / downsample))
            out_height = min(TILE_SIZE, math.ceil((height - origin[1]) / downsample))
            native_level = slide.get_best_level_for_downsample(downsample)
            native_downsample = slide.level_downsamples[native_level]
            read_size = (math.ceil(out_width * downsample / native_downsample), math.ceil(out_height * downsample / native_downsample))
            if read_size[0] * read_size[1] > 16_000_000:
                raise ValueError("WSI lacks pyramid levels needed for safe overview viewing")
            region = slide.read_region(origin, native_level, read_size).convert("RGBA")
            image = Image.new("RGB", region.size, "white")
            image.paste(region, mask=region.getchannel("A"))
            image = image.resize((out_width, out_height), Image.Resampling.LANCZOS)
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=85)
            result = buffer.getvalue()
        finally:
            self._release(entry)
        return result

    def metrics(self):
        with self._lock:
            return {"jpeg_hits": self._hits, "jpeg_misses": self._misses,
                    "coalesced": self._coalesced, "jpeg_bytes": self._jpeg_bytes,
                    "jpeg_entries": len(self._jpeg), "readers": len(self._readers),
                    "opens": self._opened, "inflight": len(self._flights),
                    "metadata_entries": len(self._info)}

    def close(self):
        with self._lock:
            readers = list(self._readers.values())
            self._readers.clear()
            self._info.clear()
            self._jpeg.clear()
            self._jpeg_bytes = 0
        for entry in readers:
            with entry.lock:
                entry.slide.close()
