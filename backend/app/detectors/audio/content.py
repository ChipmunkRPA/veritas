"""Speech, music, noise and silence per second of a recording (A-1; DESIGN "≤ 4 s windows with speech / music / silence
gating"; ARCHITECTURE §4.4 abstention: music- or silence-dominated audio is `not_assessable`).

Model-free and heuristic (`sms-heuristic-v1`). 20 ms frames of 16 kHz mono PCM give an energy, a level (dBFS) and a
spectral flatness (100–4000 Hz, as the V-5 VAD). Each 1 s block is labelled from its own frames and from a 2 s context
centred on it:
- **silence:** at least 80 % of the block's frames below -50 dBFS;
- **noise:** the median flatness of the block's non-silent frames is at least 0.5 (broadband sound without harmonic
  structure);
- **speech:** the level envelope is modulated like syllables: at least 30 % of the context's frames below half its mean
  energy, and a level spread (standard deviation of the frame levels, floored at -60 dBFS) of at least 4.5 dB;
- **music:** every other block: tonal sound with a steady envelope. Singing, and speech over loud music, are not separated.

A recording is **not assessable** when it holds less than 1 s of speech (it is music-, noise- or silence-dominated) or more
music than speech. Otherwise the speech spans gate the voice windows: a window is scored when at least half of it lies
inside speech. The thresholds are provisional (see `docs/v2/qa/a1_audio_content.md` for what was measured).
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

METHOD = "sms-heuristic-v1"
SR = 16000
FRAME_S = 0.02
BLOCK_S = 1.0
CONTEXT_S = 2.0
SILENCE_DBFS = -50.0
LEVEL_FLOOR_DBFS = -60.0
SILENT_BLOCK = 0.8
NOISE_FLATNESS = 0.5
SPEECH_LOW_ENERGY = 0.3
SPEECH_SPREAD_DB = 4.5
MIN_SPEECH_S = 1.0
WINDOW_SPEECH_FRACTION = 0.5
MIN_TAIL_S = 0.25
LABELS = ("speech", "music", "noise", "silence")
PARAMS = {"frame_s": FRAME_S, "block_s": BLOCK_S, "context_s": CONTEXT_S, "silence_dbfs": SILENCE_DBFS,
          "silent_block_fraction": SILENT_BLOCK, "noise_flatness": NOISE_FLATNESS, "speech_low_energy_fraction": SPEECH_LOW_ENERGY,
          "speech_level_spread_db": SPEECH_SPREAD_DB, "min_speech_s": MIN_SPEECH_S, "window_speech_fraction": WINDOW_SPEECH_FRACTION}


@dataclass(frozen=True)
class Block:
    start_s: float
    end_s: float
    label: str
    silent_fraction: float
    low_energy_fraction: float
    level_spread_db: float
    flatness: float | None  # median over the block's non-silent frames


@dataclass(frozen=True)
class Content:
    duration_s: float
    blocks: tuple[Block, ...]

    def seconds(self) -> dict[str, float]:
        out = dict.fromkeys(LABELS, 0.0)
        for b in self.blocks:
            out[b.label] += b.end_s - b.start_s
        return {k: round(v, 6) for k, v in out.items()}

    def shares(self) -> dict[str, float]:
        seconds = self.seconds()
        total = sum(seconds.values())
        return {k: round(v / total, 6) if total > 0 else 0.0 for k, v in seconds.items()}

    def runs(self) -> list[tuple[float, float, str]]:
        """Consecutive blocks with the same label, merged: (start_s, end_s, label)."""
        out: list[tuple[float, float, str]] = []
        for b in self.blocks:
            if out and out[-1][2] == b.label:
                out[-1] = (out[-1][0], b.end_s, b.label)
            else:
                out.append((b.start_s, b.end_s, b.label))
        return out

    def spans(self, label: str = "speech") -> list[tuple[float, float]]:
        return [(a, b) for a, b, lab in self.runs() if lab == label]

    def assessment(self) -> tuple[bool, str | None]:
        """(assessable, the reason when not)."""
        if self.duration_s < MIN_SPEECH_S:
            return False, f"The recording is {self.duration_s:.2f} s long, shorter than the 1 s the voice detectors need."
        seconds, shares = self.seconds(), self.shares()
        if seconds["speech"] < MIN_SPEECH_S:
            dominant = max(("music", "noise", "silence"), key=lambda k: (seconds[k], k == "silence"))
            return False, (f"The recording is {dominant}-dominated ({describe(shares)}): less than 1 s of speech was found, and "
                           "the voice detectors are trained on speech.")
        if seconds["music"] > seconds["speech"]:
            return False, (f"The recording is music-dominated ({describe(shares)}): the voice detectors are trained on speech, and "
                           "music is a known source of false positives.")
        return True, None

    def as_json(self) -> dict[str, Any]:
        assessable, reason = self.assessment()
        return {"method": METHOD, "params": PARAMS, "duration_s": round(self.duration_s, 6), "seconds": self.seconds(),
                "shares": self.shares(), "assessable": assessable, "reason": reason,
                "speech_spans": [[a, b] for a, b in self.spans("speech")], "runs": [[a, b, lab] for a, b, lab in self.runs()],
                "blocks": [asdict(b) for b in self.blocks]}


def describe(shares: dict[str, float]) -> str:
    """'62 % music, 20 % speech, 18 % silence': the labels present, largest first."""
    present = sorted(((v, k) for k, v in shares.items() if round(100 * v) > 0), key=lambda x: (-x[0], LABELS.index(x[1])))
    return ", ".join(f"{round(100 * v)} % {k}" for v, k in present) or "no content"


def frame_features(pcm: np.ndarray, sr: int = SR) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(energy, level dBFS, flatness 100–4000 Hz) per 20 ms frame."""
    hop = int(round(FRAME_S * sr))
    n = len(pcm) // hop
    if n == 0:
        return np.zeros(0), np.zeros(0), np.zeros(0)
    frames = np.asarray(pcm[:n * hop], np.float64).reshape(n, hop)
    energy = np.mean(frames ** 2, axis=1)
    level_db = 10.0 * np.log10(np.maximum(energy, 1e-12))
    power = np.abs(np.fft.rfft(frames * np.hanning(hop), axis=1)) ** 2
    freqs = np.fft.rfftfreq(hop, 1.0 / sr)
    band = power[:, (freqs >= 100.0) & (freqs <= 4000.0)] + 1e-12
    flatness = np.exp(np.mean(np.log(band), axis=1)) / np.mean(band, axis=1)
    return energy, level_db, flatness


def _block(start: float, end: float, features: tuple[np.ndarray, np.ndarray, np.ndarray], a: int, b: int, ca: int, cb: int) -> Block:
    energy, level_db, flatness = features
    if b <= a:
        return Block(round(start, 6), round(end, 6), "silence", 1.0, 0.0, 0.0, None)
    silent = level_db[a:b] < SILENCE_DBFS
    silent_fraction = float(np.mean(silent))
    context = energy[ca:cb]
    mean = float(np.mean(context))
    low = float(np.mean(context < 0.5 * mean)) if mean > 0 else 1.0
    spread = float(np.std(np.maximum(level_db[ca:cb], LEVEL_FLOOR_DBFS)))
    sounding = flatness[a:b][~silent]
    flat = float(np.median(sounding)) if len(sounding) else None
    if silent_fraction >= SILENT_BLOCK:
        label = "silence"
    elif flat is not None and flat >= NOISE_FLATNESS:
        label = "noise"
    elif low >= SPEECH_LOW_ENERGY and spread >= SPEECH_SPREAD_DB:
        label = "speech"
    else:
        label = "music"
    return Block(round(start, 6), round(end, 6), label, round(silent_fraction, 6), round(low, 6), round(spread, 6),
                 None if flat is None else round(flat, 6))


def classify(pcm: np.ndarray | None, sr: int = SR) -> Content:
    """1 s blocks labelled speech, music, noise or silence. A tail shorter than 0.25 s joins the previous block."""
    if pcm is None or len(pcm) == 0:
        return Content(0.0, ())
    duration = len(pcm) / sr
    features = frame_features(pcm, sr)
    n_frames = len(features[0])
    per_block, half_context = int(round(BLOCK_S / FRAME_S)), int(round((CONTEXT_S - BLOCK_S) / 2.0 / FRAME_S))
    n_blocks = max(1, math.ceil(duration / BLOCK_S - 1e-9))
    if n_blocks > 1 and duration - (n_blocks - 1) * BLOCK_S < MIN_TAIL_S:
        n_blocks -= 1
    blocks = []
    for k in range(n_blocks):
        start, end = k * BLOCK_S, duration if k == n_blocks - 1 else (k + 1) * BLOCK_S
        a, b = k * per_block, min(n_frames, int(round(end / FRAME_S)))
        blocks.append(_block(start, end, features, a, b, max(0, a - half_context), min(n_frames, b + half_context)))
    return Content(duration, tuple(blocks))


def window_in_speech(spans: Sequence[Sequence[float]], start_s: float, end_s: float,
                     fraction: float = WINDOW_SPEECH_FRACTION) -> bool:
    """True when at least `fraction` of [start_s, end_s] lies inside the speech spans."""
    if end_s <= start_s:
        return False
    overlap = sum(max(0.0, min(end_s, float(b)) - max(start_s, float(a))) for a, b in spans)
    return overlap + 1e-9 >= fraction * (end_s - start_s)
