"""`FrameStore`: a bounded LRU of decoded uint8 frames (ARCHITECTURE §3.1 `decode`).

Frames are never persisted. The sampling plan is, and any frame is re-derived by decoding its table index again, so an
eviction costs time, not correctness. The bound counts pixel bytes (2 GiB by default). A request larger than the bound
still returns every frame it asked for; only the most recently used frames stay cached. Cached arrays are read-only, so
no consumer can change the pixels another consumer sees. Consumers should ask for their frames in one call: each call
with uncached frames is one decoder pass.
"""
from __future__ import annotations

import hashlib
import threading
from collections import OrderedDict
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path

import numpy as np

from app.detectors.video.decode import DecodeError, FrameTable, decode_frames

DEFAULT_MAX_BYTES = 2 * 1024**3


class FrameStore:
    def __init__(self, path: Path, table: FrameTable, *, max_bytes: int = DEFAULT_MAX_BYTES,
                 check: Callable[[], None] | None = None, decoder: Callable[..., Iterable] = decode_frames):
        self.path, self.table, self.max_bytes = path, table, int(max_bytes)
        self._check, self._decoder = check, decoder
        self._cache: OrderedDict[tuple[int, int | None], np.ndarray] = OrderedDict()
        self._native_anchors: dict[int, dict] = {}
        self._verified_native_pts: set[int] = set()
        self._lock = threading.Lock()
        self.bytes = 0
        self.hits = self.misses = self.evictions = self.decode_passes = 0

    def __contains__(self, key: tuple[int, int | None]) -> bool:
        return key in self._cache

    def _put(self, key: tuple[int, int | None], rgb: np.ndarray) -> None:
        if key in self._cache or rgb.nbytes > self.max_bytes:
            return
        while self._cache and self.bytes + rgb.nbytes > self.max_bytes:
            _, old = self._cache.popitem(last=False)
            self.bytes -= old.nbytes
            self.evictions += 1
        self._cache[key] = rgb
        self.bytes += rgb.nbytes

    def iter_many(self, indices: Iterable[int], *, max_side: int | None = None,
                  timeout_s: float | None = None) -> Iterator[tuple[int, np.ndarray]]:
        """(table index, frame) pairs: cached frames first, then the uncached ones from one decoder pass as they are
        decoded. A consumer that keeps only crops holds one decoded frame at a time. Frames the decoder cannot deliver are
        absent. An optional timeout applies to the uncached decoder pass."""
        wanted = sorted({int(i) for i in indices})
        cached: list[tuple[int, np.ndarray]] = []
        missing: list[int] = []
        with self._lock:
            for i in wanted:
                rgb = self._cache.get((i, max_side))
                if rgb is None:
                    missing.append(i)
                else:
                    self._cache.move_to_end((i, max_side))
                    cached.append((i, rgb))
            self.hits += len(cached)
            self.misses += len(missing)
        yield from cached
        if not missing:
            return
        self.decode_passes += 1
        options = {} if timeout_s is None else {"timeout_s": timeout_s}
        frames = self._decoder(self.path, self.table, missing, max_side=max_side, check=self._check, **options)
        try:
            for frame in frames:
                frame.rgb.setflags(write=False)
                with self._lock:
                    if max_side is None:
                        anchor = {"frame_index": frame.index, "pts": frame.pts, "time_base": self.table.time_base,
                                  "width": int(frame.rgb.shape[1]), "height": int(frame.rgb.shape[0]),
                                  "decoded_rgb_sha256": hashlib.sha256(frame.rgb.tobytes()).hexdigest()}
                        previous = self._native_anchors.get(frame.index)
                        if previous is not None and previous != anchor:
                            raise DecodeError("Video could not be decoded consistently: a repeated frame changed")
                        self._native_anchors[frame.index] = anchor
                        if frame.pts_verified:
                            self._verified_native_pts.add(frame.index)
                    self._put((frame.index, max_side), frame.rgb)
                yield frame.index, frame.rgb
        finally:
            close = getattr(frames, "close", None)
            if close is not None:
                close()  # a consumer that stops early ends the decoder child now, not at garbage collection

    def get_many(self, indices: Iterable[int], *, max_side: int | None = None) -> dict[int, np.ndarray]:
        """Frames by table index (native resolution unless `max_side`). Frames the decoder cannot deliver are absent."""
        return dict(self.iter_many(indices, max_side=max_side))

    def get(self, index: int, *, max_side: int | None = None) -> np.ndarray:
        frames = self.get_many([index], max_side=max_side)
        if index not in frames:
            raise DecodeError(f"Video could not be decoded: frame {index} is not available")
        return frames[index]

    def native_anchors(self) -> list[dict]:
        """Pixels actually decoded at native resolution, independent of cache eviction.

        Decode coverage is not detector coverage; callers must join these anchors to
        successful scoring windows before presenting a frame as scored evidence.
        """
        with self._lock:
            return [dict(self._native_anchors[i]) for i in sorted(self._native_anchors)]

    def native_pts_verified(self, index: int) -> bool:
        """Whether the decoder supplied a timestamp for this native frame."""
        with self._lock:
            return index in self._verified_native_pts
