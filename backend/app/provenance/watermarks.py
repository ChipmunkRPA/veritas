"""Open watermark decoders: the public `dwtDct` messages of SDXL, the FLUX reference code and SD1 (I-5; IF §4.5, D-15).

- **Codec.** invisible-watermark 0.2.0 `dwtDct` (MIT, vendored in `app/third_party/imwatermark`): BGR → YUV, a one-level
  Haar DWT of the U channel (scales [0, 36, 36]: only U carries the mark), 4 × 4 blocks of the LL band in row order, and
  block k carries bit k mod n. A block votes 1 when |its largest AC value| mod 36 > 18; a bit is 1 when its mean vote × 255
  > 127. `block_votes` computes every vote at once and equals the vendored loop (tested).
- **Messages** (source constants rechecked on 2026-11-08):
  - SDXL: diffusers `pipelines/stable_diffusion_xl/watermark.py` and Stability AI generative-models, 0xB3EC907BB19E, 48 bits;
  - FLUX reference code: black-forest-labs/flux `src/flux/util.py`, `0b0010101011111110…`. The embedder uses
    `bin(x)[2:]`, which drops the two leading zeros, so the embedded message has **46 bits**;
  - SD1: CompVis `scripts/txt2img.py`, the UTF-8 bytes of "StableDiffusionV1", 136 bits.
  Every embedder converts RGB to BGR before encoding; the decoder reads the stored pixels, without EXIF orientation.
- **Threshold.** A message is present when its equal bits reach the smallest count whose binomial tail under chance
  (p = ½) is at most P(X ≥ 44 | n = 48) ≈ 7.6e-10 (IF §4.5): 44 of 48, 43 of 46, 103 of 136.
- **Meaning.** The messages are public and unkeyed, so anyone can embed them in a genuine image. A hit is an
  unverified marking observation, outside AI-origin confidence and the early-stop gate. The historical D-15
  confidence floor was retired. An absent mark establishes neither originality nor absence of AI use; resizing and
  cropping can remove the mark.
"""
from __future__ import annotations

import functools
import io
from dataclasses import dataclass

import cv2
import numpy as np
import pywt
from PIL import Image
from scipy.stats import binom

SCALE = 36
BLOCK = 4
MIN_PIXELS = 256 * 256  # the library refuses smaller images
REFERENCE_BITS, REFERENCE_EQUAL = 48, 44
SCHEME = "dwtDct"


@dataclass(frozen=True)
class Message:
    key: str
    name: str
    bits: tuple[int, ...]
    source: str


def int_bits(value: int) -> tuple[int, ...]:
    """The embedders' `[int(bit) for bit in bin(value)[2:]]` (leading zeros dropped)."""
    return tuple(int(bit) for bit in bin(value)[2:])


def byte_bits(content: bytes) -> tuple[int, ...]:
    return tuple(int(bit) for bit in np.unpackbits(np.frombuffer(content, dtype=np.uint8)))


MESSAGES = (
    Message("sdxl", "Stable Diffusion XL (diffusers, Stability AI)", int_bits(0b101100111110110010010000011110111011000110011110),
            "diffusers pipelines/stable_diffusion_xl/watermark.py; Stability-AI/generative-models"),
    Message("flux", "FLUX reference code (Black Forest Labs)", int_bits(0b001010101111111010000111100111001111010100101110),
            "black-forest-labs/flux src/flux/util.py"),
    Message("sd1", "Stable Diffusion 1 reference script (CompVis)", byte_bits(b"StableDiffusionV1"),
            "CompVis/stable-diffusion scripts/txt2img.py"),
)
BY_KEY = {m.key: m for m in MESSAGES}


@functools.cache
def threshold(n_bits: int) -> int:
    reference = float(binom.sf(REFERENCE_EQUAL - 1, REFERENCE_BITS, 0.5))
    return next(k for k in range(n_bits + 1) if binom.sf(k - 1, n_bits, 0.5) <= reference)


@dataclass(frozen=True)
class Reading:
    message: Message
    equal_bits: int
    threshold: int

    @property
    def n_bits(self) -> int:
        return len(self.message.bits)

    @property
    def present(self) -> bool:
        return self.equal_bits >= self.threshold

    @property
    def chance(self) -> float:
        """P(at least this many equal bits) for an unmarked image (binomial, p = ½)."""
        return float(binom.sf(self.equal_bits - 1, self.n_bits, 0.5))


def block_votes(bgr: np.ndarray) -> np.ndarray:
    """One vote per 4 × 4 block of the U channel's LL band, in the library's row order."""
    rows, cols = bgr.shape[0] // 4 * 4, bgr.shape[1] // 4 * 4
    yuv = cv2.cvtColor(np.ascontiguousarray(bgr), cv2.COLOR_BGR2YUV)
    ll, _ = pywt.dwt2(yuv[:rows, :cols, 1], "haar")
    br, bc = ll.shape[0] // BLOCK, ll.shape[1] // BLOCK
    blocks = ll[:br * BLOCK, :bc * BLOCK].reshape(br, BLOCK, bc, BLOCK).transpose(0, 2, 1, 3).reshape(br * bc, BLOCK * BLOCK)
    position = np.argmax(np.abs(blocks[:, 1:]), axis=1) + 1
    values = np.abs(blocks[np.arange(len(blocks)), position])
    return (values % SCALE) > 0.5 * SCALE


def decode_bits(votes: np.ndarray, n_bits: int) -> np.ndarray:
    index = np.arange(len(votes)) % n_bits
    sums = np.bincount(index, weights=votes.astype(np.float64), minlength=n_bits)
    counts = np.bincount(index, minlength=n_bits)
    mean = np.divide(sums, counts, out=np.zeros(n_bits), where=counts > 0)
    return mean * 255 > 127


def scan(rgb: np.ndarray) -> list[Reading] | None:
    """Every message against 8-bit RGB pixels; None below 256 × 256 pixels or for other shapes."""
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8 or rgb.shape[0] * rgb.shape[1] < MIN_PIXELS:
        return None
    votes = block_votes(rgb[:, :, ::-1])
    readings = []
    for message in MESSAGES:
        bits = decode_bits(votes, len(message.bits))
        equal = int((bits == np.asarray(message.bits, dtype=bool)).sum())
        readings.append(Reading(message, equal, threshold(len(message.bits))))
    return readings


def read_rgb(data: bytes | memoryview) -> np.ndarray | None:
    """The stored 8-bit RGB pixels (no EXIF orientation), or None when the file cannot be decoded."""
    try:
        with Image.open(io.BytesIO(bytes(data))) as im:
            return np.asarray(im.convert("RGB"))
    except (OSError, ValueError, SyntaxError, Image.DecompressionBombError):
        return None


def metrics(readings: list[Reading]) -> dict[str, float]:
    out: dict[str, float] = {}
    for r in readings:
        out |= {f"wm_{r.message.key}_equal_bits": float(r.equal_bits), f"wm_{r.message.key}_bits": float(r.n_bits),
                f"wm_{r.message.key}_threshold": float(r.threshold), f"wm_{r.message.key}_present": float(r.present)}
    return out


def assessment_entries(metrics_: dict[str, float]) -> list[dict[str, object]]:
    """H3 `watermarks[]` (ARCHITECTURE §4.4) from a decoder run's metrics: one entry per message read."""
    out = []
    for m in MESSAGES:
        if f"wm_{m.key}_equal_bits" not in metrics_:
            continue
        out.append({"scheme": SCHEME, "message": m.key, "name": m.name, "equal_bits": int(metrics_[f"wm_{m.key}_equal_bits"]),
                    "bits": int(metrics_[f"wm_{m.key}_bits"]), "threshold": int(metrics_[f"wm_{m.key}_threshold"]),
                    "present": bool(metrics_[f"wm_{m.key}_present"])})
    return out
