"""Classical image forensics used as *explanatory* evidence alongside learned detectors.

Each function returns (metrics, artifact_rgb | None). Interpretations are conservative and
stated as reference ranges in the Signal, not as proof.

1. Frequency spectrum: azimuthally averaged log power spectrum. Generators (esp. GAN/diffusion
   upsamplers) leave periodic peaks and an abnormally flat/steep high-frequency tail compared to
   camera sensors (Durall et al. 2020; Frank et al. 2020).
2. Error Level Analysis (ELA): re-compress at known JPEG quality; uniform error means one
   generation; localised error means a region was pasted/edited (splice or face swap).
3. Noise residual: sensor noise is spatially consistent; synthesised or swapped regions show
   noise variance discontinuities, notably face-vs-background.
"""
from __future__ import annotations

import io
import math

import cv2
import numpy as np
from matplotlib.figure import Figure

  # OO API: no pyplot global state, safe in worker threads
from PIL import Image
from scipy import ndimage

# Folio palette for figures (matches frontend/design/direction-b-stripe-notion.md)
INK, MUTED, RULE, PAPER = "#1B1A17", "#625D54", "#E4E0D7", "#FCFBF9"
SCALE = ["#0B6B54", "#2A7460", "#6E685D", "#AE4617", "#9B2915"]  # authentic → AI


def _style(ax, fig=None) -> None:
    if fig is not None:
        fig.patch.set_facecolor(PAPER)
    ax.set_facecolor(PAPER)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(RULE)
    ax.tick_params(colors=MUTED, labelsize=8, length=3)
    ax.yaxis.label.set_color(MUTED)
    ax.xaxis.label.set_color(MUTED)
    ax.grid(axis="y", color=RULE, linewidth=0.6)
    ax.set_axisbelow(True)



def _gray(rgb: np.ndarray, size: int = 512) -> np.ndarray:
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    return cv2.resize(g, (size, size), interpolation=cv2.INTER_AREA)


def radial_profile(mag: np.ndarray) -> np.ndarray:
    h, w = mag.shape
    cy, cx = h // 2, w // 2
    y, x = np.indices(mag.shape)
    r = np.hypot(x - cx, y - cy).astype(np.int32)
    tbin = np.bincount(r.ravel(), mag.ravel())
    nr = np.bincount(r.ravel())
    prof = tbin / np.maximum(nr, 1)
    return prof[: min(cy, cx)]


def spectrum_analysis(rgb: np.ndarray) -> tuple[dict, np.ndarray]:
    g = _gray(rgb)
    g = g - g.mean()
    win = np.outer(np.hanning(g.shape[0]), np.hanning(g.shape[1]))
    f = np.fft.fftshift(np.fft.fft2(g * win))
    metrics, art = spectrum_from_magnitude(np.log1p(np.abs(f)))
    assert art is not None
    return metrics, art


def spectrum_from_magnitude(mag: np.ndarray, *, figure: bool = True) -> tuple[dict, np.ndarray | None]:
    """Metrics and the figure from a centred log-magnitude spectrum (shared with `native_forensics.spectrum`)."""
    prof = radial_profile(mag)
    n = len(prof)
    freqs = np.arange(1, n) / n
    p = prof[1:]
    # slope of log-power vs log-frequency in mid/high band (natural images ~ 1/f^a, a≈1-1.2)
    band = (freqs > 0.1) & (freqs < 0.9)
    slope, _ = np.polyfit(np.log(freqs[band]), p[band], 1)
    # high-frequency energy fraction (top 25% radii) relative to mid band
    hf = p[freqs >= 0.75].mean()
    mf = p[(freqs >= 0.25) & (freqs < 0.5)].mean()
    hf_ratio = float(hf / mf) if mf > 0 else 0.0
    # periodic-peak detection: residual after smoothing, peaks above 3 sigma in the high band
    smooth = ndimage.uniform_filter1d(p, size=9)
    resid = p - smooth
    hb = freqs > 0.4
    sigma = resid[hb].std() + 1e-6
    peaks = int(((resid[hb] / sigma) > 3.0).sum())
    peak_strength = float(resid[hb].max() / sigma)
    # "grid" peaks: energy at exact f/2 (upsampling) and f/4 spatial frequencies (2D spectrum)
    h, w = mag.shape
    grid = float(
        np.mean([mag[h // 2, w // 2 + w // 4], mag[h // 2 + h // 4, w // 2], mag[h // 2, w // 2 + w // 2 - 1], mag[h // 2 + h // 2 - 1, w // 2]])
        - np.median(mag)
    )
    metrics = {
        "spectral_slope": round(float(slope), 3),
        "hf_ratio": round(hf_ratio, 3),
        "periodic_peaks": peaks,
        "peak_strength_sigma": round(peak_strength, 2),
        "grid_energy": round(grid, 3),
    }
    if not figure:
        return metrics, None
    # artifact figure: 2D spectrum + radial profile
    fig = Figure(figsize=(8, 3.4), dpi=110)
    ax = fig.subplots(1, 2)
    ax[0].imshow(mag, cmap="magma")
    ax[0].axis("off")
    _style(ax[1], fig)
    ax[1].plot(freqs, p, lw=1.2, color=INK, label="observed")
    ax[1].plot(freqs, smooth, lw=1, color=SCALE[3], ls="--", label="smoothed")
    ax[1].set_xlabel("normalised spatial frequency")
    ax[1].set_ylabel("log magnitude")
    ax[1].legend(fontsize=8, frameon=False, labelcolor=MUTED)
    fig.tight_layout()
    art = _fig_to_rgb(fig)
    return metrics, art


def spectrum_score(m: dict) -> tuple[float, float, float]:
    """(p, p_peaks, p_slope): v0.3 heuristic mapping, capped to [0.40, 0.75]."""
    p_peaks = logistic(m["peak_strength_sigma"], 4.5, 0.8)
    p_slope = logistic(abs(m["spectral_slope"] + 1.1), 0.9, 0.35)  # far from natural ~ -1.1
    return 0.40 + 0.35 * (0.5 * p_peaks + 0.5 * p_slope), p_peaks, p_slope


def ela_score(m: dict) -> tuple[float, float]:
    """(p, raw): face/background error ratio when a face was measured, else block inconsistency; capped to [0.40, 0.75]."""
    ratio = m.get("ela_face_bg_ratio")
    if ratio is not None:  # face region compressed differently from the background → splice / swap indicator
        raw = logistic(abs(np.log(max(ratio, 1e-3))), 0.45, 0.15)
    else:
        raw = logistic(m["ela_block_cv"], 0.9, 0.25)
    return 0.40 + 0.35 * raw, raw


def noise_score(m: dict) -> tuple[float, float, float | None]:
    """(p, p_flat, p_ratio): over-smooth residual and face/background inconsistency; capped to [0.40, 0.75]."""
    p_flat = logistic(m["noise_flat_fraction"], 0.62, 0.08)  # over-smooth renders
    p, p_ratio = p_flat, None
    ratio = m.get("noise_face_bg_ratio")
    if ratio is not None:
        p_ratio = logistic(abs(np.log(max(ratio, 1e-3))), 0.5, 0.15)
        p = 0.5 * p_flat + 0.5 * p_ratio
    return 0.40 + 0.35 * p, p_flat, p_ratio


def ela(rgb: np.ndarray, quality: int = 90, scale: int = 12, face_box: tuple[int, int, int, int] | None = None) -> tuple[dict, np.ndarray]:
    metrics, _err, diff = ela_map(rgb, quality=quality, face_box=face_box)
    return metrics, np.clip(diff * scale, 0, 255).astype(np.uint8)


def ela_map(rgb: np.ndarray, quality: int = 90, face_box: tuple[int, int, int, int] | None = None) -> tuple[dict, np.ndarray, np.ndarray]:
    """(metrics, per-pixel mean absolute error map, RGB difference), both at ≤ 1024 px."""
    im = Image.fromarray(rgb)
    if max(im.size) > 1024:
        im.thumbnail((1024, 1024))
        ratio = im.size[0] / rgb.shape[1]
    else:
        ratio = 1.0
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=quality)
    buf.seek(0)
    rec = Image.open(buf).convert("RGB")
    diff = np.abs(np.asarray(im, dtype=np.int16) - np.asarray(rec, dtype=np.int16)).astype(np.float32)
    err = diff.mean(axis=2)
    # block-wise means to measure spatial inconsistency
    bs = 16
    hh, ww = (err.shape[0] // bs) * bs, (err.shape[1] // bs) * bs
    blocks = err[:hh, :ww].reshape(hh // bs, bs, ww // bs, bs).mean(axis=(1, 3))
    mean_err = float(err.mean())
    block_cv = float(blocks.std() / (blocks.mean() + 1e-6))
    metrics = {"ela_mean": round(mean_err, 3), "ela_block_cv": round(block_cv, 3)}
    if face_box is not None:
        x, y, w, h = [int(v * ratio) for v in face_box]
        face = err[y : y + h, x : x + w]
        ring = _ring_mask(err.shape, x, y, w, h)
        if face.size > 0 and ring.sum() > 0:
            metrics["ela_face_bg_ratio"] = round(float(face.mean() / (err[ring].mean() + 1e-6)), 3)
    return metrics, err, diff


def noise_residual(rgb: np.ndarray, face_box: tuple[int, int, int, int] | None = None) -> tuple[dict, np.ndarray]:
    metrics, lstd = noise_map(rgb, face_box=face_box)
    vis = cv2.normalize(lstd, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return metrics, cv2.applyColorMap(vis, cv2.COLORMAP_INFERNO)[:, :, ::-1]


def noise_map(rgb: np.ndarray, face_box: tuple[int, int, int, int] | None = None) -> tuple[dict, np.ndarray]:
    """(metrics, local standard deviation of the median-filter residual at ≤ 1024 px)."""
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    if max(g.shape) > 1024:
        s = 1024 / max(g.shape)
        g = cv2.resize(g, (int(g.shape[1] * s), int(g.shape[0] * s)), interpolation=cv2.INTER_AREA)
    else:
        s = 1.0
    den = cv2.medianBlur(g.astype(np.uint8), 3).astype(np.float32)
    res = g - den
    # local variance map
    k = 16
    var = ndimage.uniform_filter(res**2, size=k) - ndimage.uniform_filter(res, size=k) ** 2
    var = np.maximum(var, 0)
    global_std = float(res.std())
    # consistency: CV of local std across the image (natural photos: moderate; flat synthetic: low; splice: high)
    lstd = np.sqrt(var)
    cv = float(lstd.std() / (lstd.mean() + 1e-6))
    # smoothness: proportion of near-zero residual pixels (over-smooth renders)
    flat = float((np.abs(res) < 0.5).mean())
    metrics = {"noise_std": round(global_std, 3), "noise_local_cv": round(cv, 3), "noise_flat_fraction": round(flat, 3)}
    if face_box is not None:
        x, y, w, h = [int(v * s) for v in face_box]
        f = lstd[y : y + h, x : x + w]
        ring = _ring_mask(lstd.shape, x, y, w, h)
        if f.size > 0 and ring.sum() > 0:
            metrics["noise_face_bg_ratio"] = round(float(f.mean() / (lstd[ring].mean() + 1e-6)), 3)
    return metrics, lstd


def jpeg_info(pil_img: Image.Image) -> dict:
    q = getattr(pil_img, "quantization", None)
    if not q:
        return {"jpeg": False}
    # estimate quality from luminance table (IJG scaling of the standard table)
    std_lum = np.array(
        [16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57, 69, 56, 14, 17, 22, 29, 51, 87, 80, 62,
         18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92, 49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99],
        dtype=np.float64,
    )
    tbl = np.array(list(q[0]), dtype=np.float64)[:64]
    scale = float(np.mean(tbl / std_lum)) * 100
    quality = 50 * 100 / scale if scale >= 100 else 100 - scale / 2
    return {"jpeg": True, "estimated_quality": int(round(max(1, min(100, quality)))), "tables": len(q)}


def _ring_mask(shape: tuple[int, ...], x: int, y: int, w: int, h: int, expand: float = 1.7) -> np.ndarray:
    """Boolean mask of the region around the face box (hair/neck/shoulders) excluding the face itself."""
    cx, cy = x + w / 2, y + h / 2
    X0, Y0 = int(max(0, cx - w * expand / 2)), int(max(0, cy - h * expand / 2))
    X1, Y1 = int(min(shape[1], cx + w * expand / 2)), int(min(shape[0], cy + h * expand / 2))
    m = np.zeros(shape[:2], dtype=bool)
    m[Y0:Y1, X0:X1] = True
    m[y : y + h, x : x + w] = False
    return m


def logistic(x: float, center: float, scale: float) -> float:
    """Map a measurement to a probability with a logistic centred at `center`."""
    return 1 / (1 + math.exp(-(x - center) / scale))


def _fig_to_rgb(fig) -> np.ndarray:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    return np.asarray(Image.open(buf).convert("RGB"))
