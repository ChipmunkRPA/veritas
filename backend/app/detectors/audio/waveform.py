"""Waveform envelope and spectrogram of the analysed signal (BUILD_SPEC L-7; design review P1-9).

What the recording looks like over time, beside the voice lane. Both are descriptive pictures of the 16 kHz mono signal the
detectors read; neither is a detector output, and no score is derived from them.

- **Envelope:** per bin the lowest and highest sample and the RMS level, in shares of full scale. Bins are 10 ms or longer,
  at most 2,000 across the recording.
- **Spectrogram:** Hann-windowed 32 ms frames every 10 ms (512-point FFT), power in dB below the recording's loudest cell,
  floored at -80 dB; 0-8 kHz on 128 rows (low frequencies at the bottom), at most 1,600 columns (the loudest frame of each
  column). Coloured with the eleven inferno stops the heat maps use.
"""
from __future__ import annotations

import io
from typing import Any

import numpy as np

SR = 16000
ENVELOPE_MAX_BINS = 2000
ENVELOPE_MIN_BIN_S = 0.01
N_FFT = 512
HOP = 160
SPECTROGRAM_ROWS = 128
SPECTROGRAM_MAX_COLUMNS = 1600
SPECTROGRAM_FLOOR_DB = -80.0
_CHUNK_FRAMES = 4096

ENVELOPE_TITLE = "Waveform of the analysed signal"
SPECTROGRAM_TITLE = "Spectrogram of the analysed signal"


def envelope(pcm: np.ndarray, sr: int = SR, *, amplitude_limit: float = 1.0, decimals: int = 4) -> dict[str, Any]:
    """{"bin_s", "t", "low", "high", "rms"}: bin centres in seconds and the three levels per bin, rounded to the requested precision."""
    n = int(len(pcm))
    if n == 0:
        return {"bin_s": ENVELOPE_MIN_BIN_S, "t": [], "low": [], "high": [], "rms": []}
    size = max(int(round(ENVELOPE_MIN_BIN_S * sr)), -(-n // ENVELOPE_MAX_BINS))
    bins = -(-n // size)
    padded = np.full(bins * size, np.nan, dtype=np.float64)
    padded[:n] = np.clip(np.asarray(pcm, dtype=np.float64), -amplitude_limit, amplitude_limit)
    frames = padded.reshape(bins, size)
    counts = np.minimum(size, n - np.arange(bins) * size)
    low, high = np.nanmin(frames, axis=1), np.nanmax(frames, axis=1)
    rms = np.sqrt(np.nansum(frames ** 2, axis=1) / counts)
    centres = (np.arange(bins) * size + counts / 2.0) / sr
    rounded = lambda a: [round(float(v), decimals) for v in a]  # noqa: E731
    return {"bin_s": round(size / sr, 6), "t": [round(float(v), 4) for v in centres], "low": rounded(low), "high": rounded(high), "rms": rounded(rms)}


def envelope_series(pcm: np.ndarray, sr: int = SR) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """The series of the envelope exhibit and its facts."""
    env = envelope(pcm, sr)
    series = [{"name": name, "points": [[t, v] for t, v in zip(env["t"], env[key], strict=True)]}
              for name, key in (("level_low", "low"), ("level_high", "high"), ("level_rms", "rms"))]
    return series, {"bin_s": env["bin_s"], "bins": len(env["t"])}


def envelope_caption(facts: dict[str, Any]) -> str:
    return (f"The lowest and highest sample and the RMS level of each {facts['bin_s'] * 1000:g} ms of the 16 kHz mono signal, as shares "
            "of full scale. It shows loudness over time; it is not a detector output.")


def spectrogram(pcm: np.ndarray, sr: int = SR) -> tuple[np.ndarray, dict[str, Any]] | None:
    """(8-bit levels, rows × columns with row 0 the highest frequency; facts), or None for a signal shorter than one frame."""
    x = np.asarray(pcm, dtype=np.float32)
    frames = 1 + (len(x) - N_FFT) // HOP if len(x) >= N_FFT else 0
    if frames <= 0:
        return None
    columns = min(frames, SPECTROGRAM_MAX_COLUMNS)
    column_of = np.minimum(columns - 1, (np.arange(frames) * columns) // frames)
    window = np.hanning(N_FFT).astype(np.float32)
    pooled = np.zeros((SPECTROGRAM_ROWS, columns), dtype=np.float32)
    bins_per_row = (N_FFT // 2) // SPECTROGRAM_ROWS
    for start in range(0, frames, _CHUNK_FRAMES):
        stop = min(frames, start + _CHUNK_FRAMES)
        index = (np.arange(start, stop) * HOP)[:, None] + np.arange(N_FFT)[None, :]
        power = np.abs(np.fft.rfft(x[index] * window, axis=1)).astype(np.float32) ** 2
        rows = power[:, :SPECTROGRAM_ROWS * bins_per_row].reshape(stop - start, SPECTROGRAM_ROWS, bins_per_row).max(axis=2)
        np.maximum.at(pooled, (slice(None), column_of[start:stop]), rows.T)
    peak = float(pooled.max())
    if peak <= 0.0:
        levels = np.zeros_like(pooled, dtype=np.uint8)
    else:
        db = 10.0 * np.log10(np.maximum(pooled, peak * 10 ** (SPECTROGRAM_FLOOR_DB / 10.0)) / peak)
        levels = np.rint((db - SPECTROGRAM_FLOOR_DB) / -SPECTROGRAM_FLOOR_DB * 255.0).astype(np.uint8)
    facts = {"columns": columns, "rows": SPECTROGRAM_ROWS, "frame_s": N_FFT / sr, "hop_s": HOP / sr, "frames": int(frames),
             "f_max_hz": sr / 2, "floor_db": SPECTROGRAM_FLOOR_DB, "seconds": round((N_FFT + (frames - 1) * HOP) / sr, 6)}
    return levels[::-1], facts


def spectrogram_png(pcm: np.ndarray, sr: int = SR) -> tuple[bytes, dict[str, Any]] | None:
    """The spectrogram as an RGB PNG (inferno stops), with its facts."""
    from PIL import Image

    from app.reports.scalar_heatmap import _lut

    made = spectrogram(pcm, sr)
    if made is None:
        return None
    levels, facts = made
    out = io.BytesIO()
    Image.fromarray(np.asarray(_lut(), dtype=np.uint8)[levels]).save(out, "PNG", optimize=False, compress_level=6)
    return out.getvalue(), facts


def spectrogram_caption(facts: dict[str, Any]) -> str:
    return (f"Power of the 16 kHz mono signal from 0 Hz (bottom) to {facts['f_max_hz'] / 1000:g} kHz (top) over the first "
            f"{facts['seconds']:.1f} s, in dB below the loudest cell down to {facts['floor_db']:g} dB (dark); {facts['frame_s'] * 1000:g} ms "
            f"frames every {facts['hop_s'] * 1000:g} ms. It shows what the recording holds; it is not a detector output.")
