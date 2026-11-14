"""Find bounded soundtrack shifts relative to corresponding pictures."""

from __future__ import annotations

import io
from bisect import bisect_left, bisect_right
from contextlib import closing
from fractions import Fraction
from hashlib import sha256

import numpy as np
from PIL import Image
from scipy.signal import correlate

from app.core.canonical import canonical_hash
from app.detectors.audio.decode import SAMPLE_RATE, wav_bytes
from app.detectors.audio.waveform import envelope
from app.detectors.video.decode import DecodeError, decode_frames
from app.evidence import video_reference_soundtrack_reencode as reencode
from app.evidence.audio_reference import _wav_asset

WINDOW = 2560  # 160 ms
STEP = 640  # 40 ms
SEARCH = 5760  # 360 ms on either side
MIN_SHIFT = 1920  # 120 ms
MIN_RMS = .003
MIN_CORRELATION = .86
MIN_MARGIN = .06
ALTERNATIVE = 96  # treat peaks within 6 ms as one local correlation peak
STABLE = 16
MIN_WINDOWS = 4
MAX_SKIPPED_WINDOWS = 5
MIN_WINDOW_FRACTION = Fraction(2, 3)
FLANK_WINDOWS = 2
FLANK_SEARCH = 5760
MIN_SAMPLES = SAMPLE_RATE * 3 // 2
MAX_SAMPLES = SAMPLE_RATE * 12
SCOPE = ("For clips with identical decoded pictures and picture timing, this check looks for the same sound at a different "
         "time inside a local interval, with same-time sound on either side. Ordinary edits can cause this result. "
         "This comparison carries zero weight in the AI estimate.")


def _matches(earlier, submitted):
    base = np.asarray(earlier, dtype=np.float64)
    sums = np.r_[0, np.cumsum(base)]
    squares = np.r_[0, np.cumsum(base * base)]
    energy = np.maximum(0.0, squares[WINDOW:] - squares[:-WINDOW] - (sums[WINDOW:] - sums[:-WINDOW]) ** 2 / WINDOW)
    for start in range(0, len(submitted) - WINDOW + 1, STEP):
        query = np.asarray(submitted[start:start + WINDOW], dtype=np.float64)
        query = query - query.mean()
        power = np.dot(query, query)
        if power < WINDOW * MIN_RMS ** 2:
            yield start, None, None
            continue
        low, high = max(0, start - SEARCH), min(len(base) - WINDOW, start + SEARCH)
        options = energy[low:high + 1]
        numerator = correlate(base[low:high + WINDOW], query, mode="valid", method="fft")
        correlation = np.clip(numerator / np.sqrt(np.maximum(options * power, 1e-20)), -1, 1)
        correlation[options < WINDOW * MIN_RMS ** 2] = 0
        index = int(correlation.argmax())
        best = float(correlation[index])
        distinct = max(float(correlation[:max(0, index - ALTERNATIVE + 1)].max(initial=0)),
                       float(correlation[index + ALTERNATIVE:].max(initial=0)))
        if best < MIN_CORRELATION or best - distinct < MIN_MARGIN:
            yield start, None, None
        else:
            yield start, start - (low + index), best


def _shift_runs(matches):
    run = []
    for row in matches:
        start, shift, _ = row
        if shift is None:
            continue
        if run and (abs(shift - run[0][1]) > STABLE or start - run[-1][0] > (MAX_SKIPPED_WINDOWS + 1) * STEP):
            yield run
            run = []
        if abs(shift) >= MIN_SHIFT:
            run.append(row)
    if run:
        yield run


def _brackets(matches):
    zeros = [start for start, shift, _ in matches if shift is not None and abs(shift) <= STABLE]
    for candidate in _shift_runs(matches):
        if len(candidate) < MIN_WINDOWS:
            continue
        start, end = candidate[0][0], candidate[-1][0] + WINDOW
        if Fraction(len(candidate), (end - WINDOW - start) // STEP + 1) < MIN_WINDOW_FRACTION:
            continue
        shifts = [row[1] for row in candidate]
        before = [tick for tick in zeros if start - FLANK_SEARCH <= tick <= start - WINDOW // 2]
        after = [tick for tick in zeros if end - WINDOW // 2 <= tick <= end + FLANK_SEARCH]
        if len(before) < FLANK_WINDOWS or len(after) < FLANK_WINDOWS:
            continue
        start, end = max(start, before[-1] + WINDOW), min(end, after[0])
        if end - start < WINDOW + (MIN_WINDOWS - 1) * STEP:
            continue
        offset = sorted(shifts)[len(shifts) // 2]
        yield {"submitted_samples": [start, end], "earlier_samples": [start - offset, end - offset],
               "shift_samples": offset, "supporting_windows": len(candidate),
               "minimum_correlation": round(min(row[2] for row in candidate), 6),
               "window_start_samples": [row[0] for row in candidate],
               "same_time_support": {"before": before[-FLANK_WINDOWS:], "after": after[:FLANK_WINDOWS]}}


def measure(earlier, submitted):
    """Return the longest locally bracketed shift, plus assessed waveform support."""
    matches = list(_matches(earlier, submitted))
    candidates = list(_brackets(matches))
    best = min(candidates, key=lambda span: (-span["supporting_windows"], span["submitted_samples"][0]), default=None)
    return best, sum(shift is not None for _, shift, _ in matches)


def _clip(assets, side, track, interval, origin):
    a, b = interval
    pcm = track[1][a:b]
    data = wav_bytes(pcm)
    pts = track[2]["source_pts"]
    return {"samples": interval, "start_s": float(origin + Fraction(a, SAMPLE_RATE)),
            "end_s": float(origin + Fraction(b, SAMPLE_RATE)), "source_pts": [pts + a, pts + b],
            "pcm_sha256": sha256((pcm * 32768).astype("<i2").tobytes()).hexdigest(), "waveform": envelope(pcm),
            "audio": _wav_asset(assets, f"soundtrack-shift-{side}.wav", data)}


def _picture_at(assets, path, table, index, *, prefix="soundtrack-shift-picture"):
    with closing(decode_frames(path, table, [index], timeout_s=8)) as frames:
        frame = next(frames, None)
        if frame is None or frame.index != index or frame.pts != table.pts[index] or not frame.pts_verified:
            raise DecodeError("Could not retrieve the native picture at the matched sound excerpt.")
        filename = f"{prefix}-{index:03d}.png"
        with Image.fromarray(frame.rgb) as picture, io.BytesIO() as output:
            picture.save(output, "PNG")
            data = output.getvalue()
        assets[filename] = data
        return {"frame_index": index, "pts": frame.pts, "time_base": table.time_base,
                "time_s": float((frame.pts - table.pts[0]) * table.tb), "rgb_sha256": sha256(frame.rgb).hexdigest(),
                "asset": {"file": filename, "sha256": sha256(data).hexdigest(), "media_type": "image/png"}}


def _picture(assets, path, table, origin, span):
    clock = [(tick - table.pts[0]) * table.tb for tick in table.pts]
    middle = origin + Fraction(sum(span), 2 * SAMPLE_RATE)
    return _picture_at(assets, path, table, max(0, bisect_right(clock, middle) - 1))


def _assess(table, picture, audio, tracks, assets, result, picture_witness, summary_end):
    if audio["status"] != "measured" or not MIN_SAMPLES <= audio["earlier"]["samples"] <= MAX_SAMPLES:
        return result | {"summary": "This timing check needs two matching continuous soundtrack clocks and at least 1.5 seconds of audio."}
    origin = Fraction(audio["earlier"]["relative_start"])
    best, support = measure(tracks[0][1], tracks[1][1])
    result["matched_windows"] = support
    if support < 2 * FLANK_WINDOWS + MIN_WINDOWS:
        return result | {"summary": "Too few distinctive sound excerpts could be matched to assess a local timing change."}
    if best is None:
        return result | {"status": "no_local_shift", "summary": "No qualifying local soundtrack shift was found within the matched excerpts."}
    start, end = best["submitted_samples"]
    earlier_start, earlier_end = best["earlier_samples"]
    last = (table.pts[-1] - table.pts[0]) * table.tb
    if not (origin + Fraction(min(start, earlier_start), SAMPLE_RATE) >= 0
            and origin + Fraction(max(end, earlier_end), SAMPLE_RATE) <= last):
        return result | {"summary": "The matched sound excerpts extend beyond the verified picture timeline."}
    direction = "later" if best["shift_samples"] > 0 else "earlier"
    milliseconds = best["shift_samples"] * 1000 / SAMPLE_RATE
    return result | {"status": "local_shift",
                     "summary": f"Matching soundtrack audio moved {abs(milliseconds):.1f} ms {direction} {summary_end}",
                     "offset_ms": milliseconds, "sample_rate": SAMPLE_RATE, "picture_pair_sha256": picture["frame_pairs_sha256"],
                     "range": best | {"earlier": _clip(assets, "earlier", tracks[0], best["earlier_samples"], origin),
                                      "submitted": _clip(assets, "submitted", tracks[1], best["submitted_samples"], origin),
                                      **picture_witness(origin, best["submitted_samples"]),
                                      "support_sha256": canonical_hash(best)}}


def assess(path, table, picture, audio, tracks, assets):
    """Bind the selected audio timing and picture to the original supplied clips."""
    result = {"status": "unassessed", "scope": SCOPE, "affects_overall_confidence": False}
    if picture is None or picture["changed_frames"]:
        return result | {"summary": "This timing check requires identical decoded pictures and matching picture clocks."}
    return _assess(table, picture, audio, tracks, assets, result,
                   lambda origin, span: {"picture": _picture(assets, path, table, origin, span)},
                   "relative to the same pictures.")


def assess_reencode(paths, tables, picture, audio, tracks, assets):
    """Measure a shift only when whole-picture and local-grid agreement both hold."""
    result = {"status": "unassessed", "basis": reencode.BASIS, "scope": reencode.scope(picture),
              "affects_overall_confidence": False, **reencode.report_profile(picture)}
    if not picture["soundtrack_picture_comparison"]["within_bounds"]:
        return result | {"summary": "Local soundtrack timing could not be assessed because the paired pictures differ too much."}
    profile = reencode.profile(tables, allow_hd=True)
    if (profile is None or audio["earlier"]["relative_start"] != "0" or not profile["samples"]
            <= audio["earlier"]["samples"] <= profile["samples"] + 1024):
        return result | {"summary": "Local soundtrack timing requires matching continuous sound clocks starting at the first picture."}

    clock = [(tick - tables[1].pts[0]) * tables[1].tb for tick in tables[1].pts]

    def witnesses(_origin, span):
        start, end = span
        positions = {"before": max(0, bisect_right(clock, Fraction(start, SAMPLE_RATE)) - 2),
                     "during": bisect_right(clock, Fraction(start + end, 2 * SAMPLE_RATE)) - 1,
                     "after": bisect_left(clock, Fraction(end, SAMPLE_RATE))}
        return {"paired_pictures": {
            role: {side: _picture_at(assets, path, table, index, prefix=f"soundtrack-shift-{side}-{role}")
                   for side, path, table in zip(("earlier", "submitted"), paths, tables, strict=True)}
            for role, index in positions.items()
        }}

    return _assess(tables[1], picture, audio, tracks, assets, result, witnesses,
                   "between the clips over corresponding pictures.")
