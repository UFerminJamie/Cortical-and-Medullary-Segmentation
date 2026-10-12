"""Compressed API transport with unchanged JPEG/ZIP cache semantics."""
import json
import struct

import numpy as np
from starlette.middleware.gzip import GZipMiddleware


class ApiCompression(GZipMiddleware):
    async def __call__(self, scope, receive, send):
        path = scope.get("path", "")
        if not path.startswith("/api/") or path.startswith("/api/wsi/") or path == "/api/export":
            return await self.app(scope, receive, send)
        return await super().__call__(scope, receive, send)


def encode_view(detail: dict, data: dict, labels: np.ndarray) -> bytes:
    metadata = json.dumps(detail, separators=(",", ":"), allow_nan=False).encode()
    # Pad metadata so Float32Array can view the payload without a copy.
    metadata += b" " * (-len(metadata) % 4)
    array = np.empty((len(labels), 5), dtype="<f4")
    array[:, :2] = data["coordinates"]
    array[:, 2] = data["scores"]
    array[:, 3] = data["predictions"]
    array[:, 4] = labels
    return struct.pack("<I", len(metadata)) + metadata + array.tobytes()
