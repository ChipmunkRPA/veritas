"""Aligned double-JPEG analysis and the global 8 × 8 grid origin (I-4; IF §4.1, F5). Clean-room numpy.

- **Coefficients.** Quantised luma DCT coefficients and the luma table come from the original bytes through `jpeglib`
  (MPL-2.0; its bundled libjpeg 6b reads coefficients without decoding). Only Huffman-coded 8-bit files are read (checked
  with `jpeg_bitstream` first), so libjpeg never meets a frame type it rejects.
- **Reference statistics (calibration).** The file's luma, decoded from the original bytes, cropped by 4 px (which breaks
  the alignment with any earlier grid) and recompressed with the file's own luma table: its coefficient histogram p0
  estimates what a file compressed once with that table would show.
- **Model per AC frequency** (the first `N_FREQUENCIES` in zigzag order whose current step q2 ≤ `MAX_Q2`):
  - single compression: P_S = p0;
  - aligned double compression with an earlier step q1: each p0 bin's mass is spread uniformly over its q2-wide
    interval, requantised with q1, perturbed by the rounding noise between the two compressions (Gaussian,
    σ = `NOISE_SIGMA`) and quantised again with q2;
  - q1 runs over 1 … min(`MAX_Q1_FACTOR`·q2, `MAX_Q1`), skipping q2 (equal steps leave no trace). A frequency supports
    double compression when the log-likelihood gain over the single model is ≥ `MIN_GAIN` nats per coefficient; the file
    is double-compressed when ≥ `MIN_DQ_FREQUENCIES` frequencies support it.
- **Map.** Per 8 × 8 block, the sum over the supporting frequencies of ln P_D'(x) / P_S(x) with P_D' = (1 − η)P_D + ηP_S,
  summed over the 3 × 3 neighbourhood. `posterior()` = 1 / (1 + e^LLR), the probability that the block lacks the trace,
  with a prior of ½ and no calibration. Neighbourhoods with fewer than `MIN_INFORMATIVE` non-zero coefficients are
  unreliable.
- **Earlier quality.** The libjpeg quality whose standard luma table equals most of the estimated q1 steps.
- **Grid origin.** From the decoded luma in the stored orientation, the offset of the strongest 8-periodic boundary
  discontinuity per axis; `detected` when both contrasts reach `GRID_MIN_CONTRAST`.
- **Thresholds** (`MIN_GAIN`, `GRID_MIN_CONTRAST`) were set on fixtures built from `demo/real-photo.jpg` and 27 vb-mini
  reals (the first three per stratum), which are also in the FPR record (`docs/v2/qa/i4_jpeg_history_fpr.md`).
- **Limits.** No trace when q1 = q2 or q1 divides q2 (a finer earlier compression), after resizing or cropping between the
  compressions (non-aligned; not built), in smooth or saturated areas, or in a file compressed once. The absence of a
  trace is not evidence of authenticity.
"""
from __future__ import annotations

import io
import math
import os
import tempfile
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.special import ndtr

from app.detectors.image import jpeg_bitstream
from app.engine.fusion.features import fp_guarded
from app.provenance.metadata_rules import STD_LUMA, ZIGZAG, ijg_table

N_FREQUENCIES = 15
MAX_Q2 = 24
MAX_Q1 = 48
MAX_Q1_FACTOR = 4
MAX_ABS = 48  # histogram range; larger coefficients are neutral in the map
CALIBRATION_CROP = 4
PSEUDO_COUNT = 0.5
NOISE_SIGMA = 1.0
ETA = 0.05
MIN_GAIN = 0.05
MIN_NONZERO = 400
MIN_DQ_FREQUENCIES = 3
MIN_INFORMATIVE = 6
READABLE_FRAMES = frozenset({"baseline", "extended sequential", "progressive"})
MIN_REGION_BLOCKS = 16
MIN_REGION_SHARE = 0.005
MAX_REGION_SHARE = 0.5
REGION_POSTERIOR = 0.9
GRID_MIN_CONTRAST = 0.06  # margin of the strongest offset over the runner-up, per mean absolute pixel difference


@dataclass(frozen=True)
class Coefficients:
    y: np.ndarray  # (rows, cols, 8, 8) quantised luma coefficients, stored orientation
    table: np.ndarray  # (8, 8) luma quantisation table
    width: int
    height: int


@dataclass(frozen=True)
class FrequencyFit:
    zigzag: int
    q2: int
    n: int
    q1: int | None  # the earlier step when the frequency supports double compression
    gain: float  # nats per coefficient, double over single


@dataclass(frozen=True)
class Region:
    x: int  # blocks, stored orientation
    y: int
    w: int
    h: int
    blocks: int
    share: float  # of the reliable blocks


@dataclass(frozen=True)
class GridOrigin:
    dx: int
    dy: int
    contrast_x: float
    contrast_y: float

    @property
    def detected(self) -> bool:
        return min(self.contrast_x, self.contrast_y) >= GRID_MIN_CONTRAST


@dataclass
class DQResult:
    fits: list[FrequencyFit]
    llr: np.ndarray  # (rows, cols) neighbourhood log-likelihood ratio; > 0 favours the double-compression trace
    reliable: np.ndarray  # (rows, cols) bool
    width: int
    height: int
    grid: GridOrigin

    @property
    def supporting(self) -> list[FrequencyFit]:
        return [f for f in self.fits if f.q1 is not None]

    @property
    def double_compression(self) -> bool:
        return len(self.supporting) >= MIN_DQ_FREQUENCIES

    @property
    def earlier_quality(self) -> int | None:
        """The libjpeg quality whose standard luma steps equal most estimated q1 values (≥ 60 % of them), else None."""
        fits = self.supporting
        if len(fits) < MIN_DQ_FREQUENCIES:
            return None
        positions = np.array([ZIGZAG[f.zigzag] for f in fits])
        steps = np.array([f.q1 for f in fits])
        scored = []
        for quality in range(1, 101):
            table = ijg_table(STD_LUMA, quality)[positions]
            scored.append((int((table == steps).sum()), -int(np.abs(table - steps).sum()), quality))
        matches, _, quality = max(scored)
        return quality if matches >= max(MIN_DQ_FREQUENCIES, math.ceil(0.6 * len(fits))) else None

    def posterior(self) -> np.ndarray:
        """Per-block probability that the double-compression trace is absent (uncalibrated, prior ½)."""
        return (1.0 / (1.0 + np.exp(np.clip(self.llr, -50.0, 50.0)))).astype(np.float32)

    def regions(self) -> list[Region]:
        """Connected reliable blocks lacking the trace (posterior ≥ `REGION_POSTERIOR`), largest first."""
        if not self.double_compression:
            return []
        total = int(self.reliable.sum())
        mask = (self.posterior() >= REGION_POSTERIOR) & self.reliable
        n, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
        found = [Region(int(x), int(y), int(w), int(h), int(area), round(area / max(total, 1), 4))
                 for x, y, w, h, area in stats[1:] if area >= max(MIN_REGION_BLOCKS, MIN_REGION_SHARE * total)]
        return sorted(found, key=lambda r: (-r.blocks, r.y, r.x))

    @property
    def localised(self) -> Region | None:
        """The largest region when it covers at most `MAX_REGION_SHARE` of the reliable blocks."""
        found = self.regions()
        return found[0] if found and found[0].share <= MAX_REGION_SHARE else None


def read_coefficients(data: bytes | memoryview) -> Coefficients | None:
    """Luma coefficients of a JPEG, or None for other formats, unsupported frames and unreadable files.

    Raises ImportError when `jpeglib` is not installed."""
    payload = bytes(data)
    structure = jpeg_bitstream.read(payload)
    if structure is None or structure.frame not in READABLE_FRAMES or structure.precision != 8:
        return None
    import jpeglib  # noqa: PLC0415 - loads the bundled libjpeg only for JPEG inputs

    fd, path = tempfile.mkstemp(suffix=".jpg")  # jpeglib reads from a path
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
        jpeg = jpeglib.read_dct(path)
        y, tables, numbers = np.asarray(jpeg.Y), np.asarray(jpeg.qt), np.asarray(jpeg.quant_tbl_no)
        return Coefficients(y=y.astype(np.int32), table=tables[int(numbers[0])].astype(np.int64), width=int(jpeg.width),
                            height=int(jpeg.height))
    except (OSError, ValueError, IndexError, RuntimeError):
        return None
    finally:
        os.unlink(path)


def decode_luma(data: bytes | memoryview) -> np.ndarray | None:
    """The luma of any image the decoder reads, in the stored orientation (no EXIF transpose)."""
    try:
        with Image.open(io.BytesIO(bytes(data))) as im:
            return np.asarray(im.convert("L"))
    except (OSError, ValueError, SyntaxError, Image.DecompressionBombError):
        return None


def calibration(gray: np.ndarray, table: np.ndarray) -> Coefficients | None:
    """`gray` cropped by `CALIBRATION_CROP` px and recompressed with `table` (the single-compression reference)."""
    if min(gray.shape) < CALIBRATION_CROP + 16:
        return None
    buf = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(gray[CALIBRATION_CROP:, CALIBRATION_CROP:])).save(buf, "JPEG",
                                                                                         qtables=[table.ravel().tolist()])
    return read_coefficients(buf.getvalue())


def _histogram(values: np.ndarray) -> np.ndarray:
    flat = values.ravel().astype(np.int64)
    return np.bincount(flat[np.abs(flat) <= MAX_ABS] + MAX_ABS, minlength=2 * MAX_ABS + 1).astype(np.float64)


def _uniform_cdf(v: np.ndarray, p0: np.ndarray, q2: int) -> np.ndarray:
    """CDF of the density that spreads each reference bin's mass uniformly over its q2-wide interval."""
    u = np.clip(v / q2 + 0.5 + MAX_ABS, 0.0, float(len(p0)))
    index = np.minimum(np.floor(u).astype(np.int64), len(p0) - 1)
    cumulative = np.concatenate([[0.0], np.cumsum(p0)])
    return cumulative[index] + p0[index] * (u - index)


@fp_guarded
def fit_frequency(values: np.ndarray, reference: np.ndarray, q2: int, zigzag: int = 0) -> tuple[FrequencyFit, np.ndarray | None]:
    """Single vs aligned double compression for one frequency; the per-bin ln P_D'/P_S table when double is supported."""
    h = _histogram(values)
    n = int(h.sum())
    if n - int(h[MAX_ABS]) < MIN_NONZERO:
        return FrequencyFit(zigzag, q2, n, None, 0.0), None
    p0 = _histogram(reference) + PSEUDO_COUNT
    p0 /= p0.sum()
    ll_single = float((h * np.log(p0)).sum())
    x = np.arange(-MAX_ABS, MAX_ABS + 1, dtype=np.float64)
    best_ll, best_q1, best_double = -np.inf, None, None
    for q1 in range(1, min(MAX_Q1_FACTOR * q2, MAX_Q1) + 1):
        if q1 == q2:
            continue
        kmax = math.ceil(((MAX_ABS + 0.5) * q2 + 4 * NOISE_SIGMA) / q1) + 1
        k = np.arange(-kmax, kmax + 1, dtype=np.float64)
        pk = _uniform_cdf((k + 0.5) * q1, p0, q2) - _uniform_cdf((k - 0.5) * q1, p0, q2)
        centre = k[:, None] * q1
        spread = ndtr(((x + 0.5) * q2 - centre) / NOISE_SIGMA) - ndtr(((x - 0.5) * q2 - centre) / NOISE_SIGMA)
        double = pk @ spread
        double /= double.sum()
        if not np.all(np.isfinite(double)):
            raise FloatingPointError(f"non-finite double-compression model (q2 {q2}, q1 {q1})")
        ll = float((h * np.log(np.maximum(double, 1e-12))).sum())
        if ll > best_ll:
            best_ll, best_q1, best_double = ll, q1, double
    gain = (best_ll - ll_single) / n
    if best_double is None or gain < MIN_GAIN:
        return FrequencyFit(zigzag, q2, n, None, round(gain, 5)), None
    table = np.log(np.maximum((1 - ETA) * best_double + ETA * p0, 1e-12) / p0)
    return FrequencyFit(zigzag, q2, n, best_q1, round(gain, 5)), table


def analyse(data: bytes | memoryview) -> DQResult | None:
    """Double-compression fits, block map and grid origin of a JPEG; None for other formats and unreadable files."""
    coefficients = read_coefficients(data)
    gray = decode_luma(data) if coefficients is not None else None
    reference = calibration(gray, coefficients.table) if coefficients is not None and gray is not None else None
    if coefficients is None or gray is None or reference is None:
        return None
    y, rows, cols = coefficients.y, coefficients.y.shape[0], coefficients.y.shape[1]
    llr = np.zeros((rows, cols), dtype=np.float64)
    informative = np.zeros((rows, cols), dtype=np.float64)
    fits = []
    for zigzag in range(1, N_FREQUENCIES + 1):
        i, j = divmod(int(ZIGZAG[zigzag]), 8)
        q2 = int(coefficients.table[i, j])
        if q2 > MAX_Q2:
            continue
        values = y[:, :, i, j]
        fit, table = fit_frequency(values, reference.y[:, :, i, j], q2, zigzag)
        fits.append(fit)
        if table is None:
            continue
        inside = np.abs(values) <= MAX_ABS
        llr += np.where(inside, table[np.clip(values, -MAX_ABS, MAX_ABS) + MAX_ABS], 0.0)
        informative += values != 0
    area = 9.0  # 3 × 3 neighbourhood sums
    llr = ndimage.uniform_filter(llr, size=3, mode="nearest") * area
    reliable = ndimage.uniform_filter(informative, size=3, mode="nearest") * area >= MIN_INFORMATIVE
    return DQResult(fits=fits, llr=llr.astype(np.float32), reliable=reliable, width=coefficients.width,
                    height=coefficients.height, grid=grid_origin(gray))


def _fold(profile: np.ndarray) -> tuple[int, float]:
    """Offset r of the 8-periodic boundary (between samples t and t + 1 with (t + 1) mod 8 = r) and its contrast."""
    excess = profile[1:-1] - 0.5 * (profile[:-2] + profile[2:])
    t = np.arange(1, len(profile) - 1)
    folded = np.array([excess[(t + 1) % 8 == r].mean() for r in range(8)])
    r = int(np.argmax(folded))
    runner_up = float(np.sort(folded)[-2])
    return r, round(float((folded[r] - runner_up) / (profile.mean() + 1e-6)), 4)


def grid_origin(gray: np.ndarray) -> GridOrigin:
    """The dominant 8 × 8 grid offset of a grayscale image in its stored orientation."""
    g = gray.astype(np.float32)
    columns = np.abs(np.diff(g[::2], axis=1)).mean(axis=0)
    rows = np.abs(np.diff(g[:, ::2], axis=0)).mean(axis=1)
    if len(columns) < 16 or len(rows) < 16:
        return GridOrigin(0, 0, 0.0, 0.0)
    dx, cx = _fold(columns)
    dy, cy = _fold(rows)
    return GridOrigin(dx, dy, cx, cy)


def grid_from_bytes(data: bytes | memoryview) -> GridOrigin | None:
    gray = decode_luma(data)
    return None if gray is None else grid_origin(gray)
