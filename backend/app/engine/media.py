"""Store-backed media handle with the three D13 accessors (ARCHITECTURE §2.2 `RunContext.media`).

- `original_bytes()` returns a read-only mmap of the write-once original in the object store. Use it for bitstream,
  metadata and codec forensics.
- `native_pixels(tile=None)` decodes at native resolution, after EXIF orientation, with the decompression-bomb guard.
- `analysis_copy()` returns a ≤ 4096 px LANCZOS copy and `scale_to_original` (native width / copy width).
- **No partial decode (L-1).** A truncated file fails to decode ("image file is truncated"). Pillow's process-wide
  `ImageFile.LOAD_TRUNCATED_IMAGES`, which WeasyPrint switches on when it is imported, would pad it with grey instead, so
  `refuse_truncated_images` switches it off before every decode.
"""
from __future__ import annotations

import io
import mmap
import threading
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageFile, ImageOps

from app.core.logging import get_logger
from app.storage.objects import ObjectStore

MAX_SIDE = 4096
MAX_PIXELS = 80_000_000
log = get_logger(__name__)


def refuse_truncated_images() -> None:
    """Switch Pillow's `LOAD_TRUNCATED_IMAGES` back off: Veritas never analyses a partially decoded image."""
    if ImageFile.LOAD_TRUNCATED_IMAGES:
        ImageFile.LOAD_TRUNCATED_IMAGES = False
        log.warning("media.truncated_images_refused")


class StoredMedia:
    def __init__(self, sha256: str, kind: str, store: ObjectStore, *, path: Path | None = None, max_side: int = MAX_SIDE,
                 max_pixels: int = MAX_PIXELS):
        self.sha256 = sha256
        self.kind = kind
        self.store = store
        self.path = path  # the upload path when it survives (fixture sidecars read it); detectors use the accessors
        self.max_side = max_side
        self.max_pixels = max_pixels
        self._lock = threading.Lock()
        self._file: Any = None
        self._mmap: mmap.mmap | None = None
        self._native: np.ndarray | None = None
        self._copy: tuple[np.ndarray, float] | None = None
        self._info: dict[str, Any] | None = None

    def original_bytes(self) -> memoryview:
        with self._lock:
            if self._mmap is None:
                self._file = self.store.open(self.sha256)
                try:
                    self._mmap = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
                except ValueError:  # an empty file cannot be mapped
                    return memoryview(b"")
            return memoryview(self._mmap)

    def _decode(self) -> np.ndarray:
        with self._lock:
            if self._native is not None:
                return self._native
        data = bytes(self.original_bytes())
        refuse_truncated_images()
        with Image.open(io.BytesIO(data)) as src:
            if src.width * src.height > self.max_pixels:  # checked before decoding (decompression-bomb guard)
                raise ValueError(f"The image exceeds the {self.max_pixels // 1_000_000} MP pixel limit")
            src.load()
            orientation = int(src.getexif().get(0x0112, 1) or 1)
            info = {"format": src.format, "mode": src.mode, "width": src.width, "height": src.height,
                    "n_frames": int(getattr(src, "n_frames", 1)),
                    "jpeg_quant_tables": len(getattr(src, "quantization", None) or {}),
                    "exif_orientation": orientation, "exif_transposed": orientation != 1}
            oriented = ImageOps.exif_transpose(src)
            pixels = np.ascontiguousarray(np.asarray(oriented.convert("RGB")))
        info["oriented_width"], info["oriented_height"] = int(pixels.shape[1]), int(pixels.shape[0])
        with self._lock:
            self._native, self._info = pixels, info
        return pixels

    def info(self) -> dict[str, Any]:
        self._decode()
        assert self._info is not None
        return dict(self._info)

    def native_pixels(self, tile: tuple[int, int, int, int] | None = None) -> np.ndarray:
        pixels = self._decode()
        if tile is None:
            return pixels
        x, y, w, h = tile
        if w <= 0 or h <= 0 or x < 0 or y < 0:
            raise ValueError("a tile needs a non-negative origin and a positive size")
        return pixels[y:y + h, x:x + w]

    def analysis_copy(self) -> tuple[np.ndarray, float]:
        with self._lock:
            if self._copy is not None:
                return self._copy
        pixels = self._decode()
        height, width = pixels.shape[:2]
        if max(height, width) <= self.max_side:
            copy = (pixels, 1.0)
        else:
            im = Image.fromarray(pixels)
            im.thumbnail((self.max_side, self.max_side), Image.Resampling.LANCZOS)
            arr = np.ascontiguousarray(np.asarray(im))
            copy = (arr, width / arr.shape[1])
        with self._lock:
            self._copy = copy
        return copy

    def close(self) -> None:
        with self._lock:
            if self._mmap is not None:
                self._mmap.close()
                self._mmap = None
            if self._file is not None:
                self._file.close()
                self._file = None
