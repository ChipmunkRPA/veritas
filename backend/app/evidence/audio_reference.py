"""Exact time-aligned comparisons of decoded mono audio."""

from __future__ import annotations

import hashlib
import math
import os
from fractions import Fraction
from pathlib import Path

import numpy as np

from app.core.canonical import canonical_hash
from app.detectors.audio.decode import SAMPLE_RATE as AUDIO_RATE
from app.detectors.audio.decode import AudioStreamUnavailable, decode_audio, wav_bytes
from app.detectors.audio.waveform import envelope
from app.evidence import audio_level_adjustment
from app.storage.files import sniff_audio

SCHEMA = "chipmunk.audio-reference-comparison/1"
SCHEMA_WITH_LEVEL_ADJUSTMENT = "chipmunk.audio-reference-comparison/2"
SCHEMA_WITH_TRIM_ALIGNMENT = "chipmunk.audio-reference-comparison/3"
SCHEMA_WITH_TRIM_AND_LEVEL = "chipmunk.audio-reference-comparison/4"
MAX_BYTES = 48 * 1024 * 1024
MAX_SECONDS = 30
MAX_PRESENTED_RANGES = 3
AUDIO_BIN_SAMPLES = AUDIO_RATE // 50
AUDIO_PREVIEW_SAMPLES = 2 * AUDIO_RATE
NATIVE_LOSSLESS_CODECS = frozenset({"pcm_s16le", "pcm_s16be", "flac", "alac"})
TRIM_LIMIT = AUDIO_RATE // 4
ANCHOR_SAMPLES = 1024
INTERPRETATION = (
    "This comparison measures differences in the first audio track after conversion to 16 kHz mono. "
    "Encoding, volume changes, or ordinary edits can change samples. "
    "The difference waveform shows where the decoded copies disagree. "
    "The comparison does not determine who spoke or whether AI was used."
)
LEVEL_INTERPRETATION = (
    "The first and last half second of the decoded 16 kHz mono track support the same constant change in volume. Each counted sample differs by more "
    "than half a quantization step from every value predicted by that range of gains. The residual waveform uses the "
    "midpoint of the fitted gain range and shows submitted minus volume-adjusted earlier samples. Ordinary processing "
    "and edits can cause these differences. This comparison contributes zero weight to the AI estimate."
)
TRIM_INTERPRETATION = (
    "Two distinct sections of the original audio support the same sample offset. "
    "The waveforms show differences inside the shared span. Samples outside that span were left unassessed. "
    "A trim or an ordinary edit can cause these differences. This comparison contributes zero weight to the AI estimate."
)
TRIM_LEVEL_INTERPRETATION = (
    "Two distinct sections support the same sample offset and constant volume change. The adjusted waveform shows "
    "analyzed minus volume-adjusted earlier samples within the shared span. Samples outside it were left unassessed. "
    "Ordinary processing and edits can produce differences. This comparison contributes zero weight to the AI estimate."
)


def _identity(path: Path, filename: str) -> dict:
    with path.open("rb") as stream:
        count = os.fstat(stream.fileno()).st_size
        if not 0 < count <= MAX_BYTES:
            raise ValueError("Each recording must be nonempty and at most 48 MiB.")
        if sniff_audio(stream.read(8192), filename) is None:
            raise ValueError("Choose two readable audio files.")
        stream.seek(0)
        return {"sha256": hashlib.file_digest(stream, "sha256").hexdigest(), "bytes": count}


def marked_ranges(marked):
    edges = np.flatnonzero(np.diff(np.r_[False, marked, False]))
    return [(int(start), int(end)) for start, end in zip(edges[::2], edges[1::2], strict=True)]


def _wav_asset(assets, filename, content):
    assets[filename] = content
    return {"file": filename, "sha256": hashlib.sha256(content).hexdigest(), "media_type": "audio/wav"}


def _read_track(path, source, *, max_seconds, origin, video, stream_position=0):
    try:
        options = {"stream_position": stream_position} if stream_position else {}
        decoded = decode_audio(path, source_sha256=source["sha256"], max_seconds=max_seconds + 1, **options)
    except AudioStreamUnavailable:
        return ("no_stream", None, None), False, False
    except (ValueError, RuntimeError):
        return ("unverified_audio_clock", None, None), False, False
    mapping = decoded.mapping
    if (not video or stream_position) and mapping["samples"] > max_seconds * AUDIO_RATE:
        return ("comparison_limit", None, None), False, False
    if len(mapping["timeline"]) != 1 or mapping["samples"] >= mapping["limit_samples"]:
        return ("unverified_audio_clock", None, None), False, False
    pts = mapping["timeline"][0]["source_pts"]
    relative = Fraction(pts, AUDIO_RATE) - origin
    native_lossless = (mapping["source_channels"] == 1 and mapping["source_sample_rate"] == AUDIO_RATE
                       and mapping["source_codec"] in NATIVE_LOSSLESS_CODECS)
    exact_s16 = native_lossless and (
        mapping["source_codec"] in {"pcm_s16le", "pcm_s16be"}
        or (mapping["source_codec"] == "flac" and mapping.get("source_bit_depth") == 16)
    )
    return ("measured", decoded.pcm, {"pcm_sha256": mapping["pcm_sha256"], "source_pts": pts,
                                     "relative_start": str(relative), "samples": len(decoded.pcm)}), native_lossless, exact_s16


def read_track(path, source, *, max_seconds, origin=Fraction(0), video=False, stream_position=0):
    track, _, _ = _read_track(path, source, max_seconds=max_seconds, origin=origin, video=video, stream_position=stream_position)
    return track


def _source_ranges(first, second, start, end):
    return {side: {"start_pts": track[2]["source_pts"] + start, "end_exclusive_pts": track[2]["source_pts"] + end}
            for side, track in (("earlier", first), ("submitted", second))}


def _measure_changes(first, second, assets, difference, delta, *, origin, prefix="", id_prefix="A"):
    changed_per_bin = np.add.reduceat(difference, np.arange(0, len(difference), AUDIO_BIN_SAMPLES), dtype=np.int64)
    ranges = marked_ranges(changed_per_bin > 0)
    top = set(sorted(ranges, key=lambda pair: (-(pair[1] - pair[0]), pair[0]))[:MAX_PRESENTED_RANGES])
    result = []
    for number, (first_bin, after_bin) in enumerate(ranges, 1):
        start, end = first_bin * AUDIO_BIN_SAMPLES, min(after_bin * AUDIO_BIN_SAMPLES, len(difference))
        item = {"id": f"{id_prefix}{number:02d}", "start_sample": start, "end_sample": end,
                "start_s": float(origin + Fraction(start, AUDIO_RATE)), "end_s": float(origin + Fraction(end, AUDIO_RATE)),
                "changed_samples": int(np.sum(changed_per_bin[first_bin:after_bin])),
                "source_ranges": _source_ranges(first, second, start, end)}
        if delta is not None:
            span = delta[start:end].astype(np.float64)
            item["difference"] = {"peak": float(np.max(np.abs(span))), "rms": float(np.sqrt(np.mean(span**2)))}
        if (first_bin, after_bin) in top:
            preview_end = min(end, start + AUDIO_PREVIEW_SAMPLES)
            excerpts = {"earlier": first[1][start:preview_end], "submitted": second[1][start:preview_end]}
            item["preview"] = {"start_sample": start, "end_sample": preview_end,
                               "end_s": float(origin + Fraction(preview_end, AUDIO_RATE)),
                               "waveforms": {label: envelope(pcm) for label, pcm in excerpts.items()},
                               "assets": {label: _wav_asset(assets, f"{prefix}{label}-audio-{number:02d}.wav", wav_bytes(pcm))
                                          for label, pcm in excerpts.items()}}
            if delta is not None:
                item["preview"]["difference_waveform"] = envelope(delta[start:preview_end], amplitude_limit=2.0, decimals=8)
        result.append(item)
    return {"sample_rate": AUDIO_RATE, "bin_samples": AUDIO_BIN_SAMPLES, "changed_samples": int(np.count_nonzero(difference)),
            "bin_counts_sha256": canonical_hash(changed_per_bin.tolist()), "ranges": result}


def compare_tracks(first, second, assets, *, standalone=False, preview_prefix="", difference_waveform=False):
    if first[0] == second[0] == "no_stream":
        return {"status": "no_stream"}
    if first[0] != "measured" or second[0] != "measured":
        return {"status": "audio_unassessed", "earlier": first[0], "submitted": second[0]}
    left, right = first[2], second[2]
    if (left["relative_start"], left["samples"]) != (right["relative_start"], right["samples"]):
        return {"status": "audio_timelines_differ", "earlier": left, "submitted": right, "sample_rate": AUDIO_RATE}

    delta = second[1] - first[1] if standalone or difference_waveform else None
    origin = Fraction(0) if standalone else Fraction(right["relative_start"])
    return {"status": "measured", "earlier": left, "submitted": right,
            **_measure_changes(first, second, assets, first[1] != second[1], delta, origin=origin, prefix=preview_prefix)}


def _gain_bounds(bounds):
    return {"lower": str(bounds[0]), "upper": str(bounds[1])}


def _level_adjusted(first, second, assets, *, support_spans=None, file_offsets=None):
    count = len(first[1])
    width = audio_level_adjustment.SUPPORT_SAMPLES
    support_spans = support_spans or ((0, width), (count - width, count))
    fit = audio_level_adjustment.compare(first[1], second[1], supports=tuple(slice(a, b) for a, b in support_spans))
    if fit is None:
        return None
    origin = Fraction(file_offsets[1], AUDIO_RATE) if file_offsets else Fraction(0)
    changes = _measure_changes(first, second, assets, fit.changed, fit.residual, origin=origin,
                               prefix="level-adjusted-", id_prefix="V")
    support = []
    for (start, end), bounds in zip(support_spans, fit.support_bounds, strict=True):
        support.append({"start_sample": start, "end_sample": end, "gain_bounds": _gain_bounds(bounds),
                        "source_ranges": _source_ranges(first, second, start, end),
                        "pcm_sha256": {side: hashlib.sha256((track[1][start:end] * 32768).astype("<i2").tobytes()).hexdigest()
                                       for side, track in (("earlier", first), ("submitted", second))}})
    if file_offsets:
        _attach_file_ranges([*changes["ranges"], *support], (first, second), file_offsets)
    midpoint = sum(fit.bounds) / 2
    return {"status": "measured", **changes, "gain_bounds": _gain_bounds(fit.bounds),
            "gain_db": round(20 * math.log10(midpoint), 6), "support": support,
            "interpretation": TRIM_LEVEL_INTERPRETATION if file_offsets else LEVEL_INTERPRETATION}


def _clock_range(track, start, end):
    pts = track[2]["source_pts"]
    return {"start_sample": start, "end_sample": end,
            "start_pts": pts + start, "end_exclusive_pts": pts + end}


def _attach_file_ranges(rows, tracks, offsets):
    for span in rows:
        span["file_ranges"] = {}
        for side, track, offset in zip(("earlier", "submitted"), tracks, offsets, strict=True):
            times = _clock_range(track, span["start_sample"], span["end_sample"])
            times["start_sample"] += offset
            times["end_sample"] += offset
            span["file_ranges"][side] = times


def _scaled_matches(anchor, submitted, low, high):
    # The correlation is only a cheap proposal. Each surviving position must
    # explain the original integer samples under one compatible positive gain.
    window = submitted[low:high + len(anchor)].astype(np.float64)
    reference = anchor.astype(np.float64)
    products = np.correlate(window, reference, mode="valid")
    squares = np.r_[0., np.cumsum(window ** 2)]
    norms = np.sqrt(np.dot(reference, reference) * (squares[len(anchor):] - squares[:-len(anchor)]))
    positions = np.flatnonzero((norms > 0) & (products >= 0.999 * norms)) + low
    return [int(pos) for pos in positions
            if audio_level_adjustment.compatible_gain(anchor, submitted[pos:pos + len(anchor)]) is not None]


def _audio_trim(first, second, assets, *, adjust_volume=False):
    earlier, submitted = (np.rint(track[1] * 32768).astype(np.int64) for track in (first, second))
    count = min(len(earlier), len(submitted))
    if count < 2 * AUDIO_RATE or abs(len(earlier) - len(submitted)) > TRIM_LIMIT:
        return None
    supports, offsets = [], []
    for start in (count // 5, 4 * count // 5):
        anchor = earlier[start:start + ANCHOR_SAMPLES]
        if np.count_nonzero(np.abs(anchor.astype(np.int32)) >= 512) < 128 or len(np.unique(anchor)) < 64:
            return None
        low, high = start - TRIM_LIMIT, min(start + TRIM_LIMIT, len(submitted) - ANCHOR_SAMPLES)
        if adjust_volume:
            matches = _scaled_matches(anchor, submitted, low, high)
        else:
            candidates = np.flatnonzero(submitted[low:high + 1] == anchor[0]) + low
            matches = [int(pos) for pos in candidates if np.array_equal(submitted[pos:pos + ANCHOR_SAMPLES], anchor)]
        if len(matches) != 1:
            return None
        pos = matches[0]
        offsets.append(start - pos)
        supports.append({"earlier": _clock_range(first, start, start + ANCHOR_SAMPLES),
                         "submitted": _clock_range(second, pos, pos + ANCHOR_SAMPLES),
                         "pcm_sha256": hashlib.sha256(anchor.astype("<i2").tobytes()).hexdigest()})
    if offsets[0] != offsets[1] or supports[0]["pcm_sha256"] == supports[1]["pcm_sha256"]:
        return None
    first_start, second_start = max(0, offsets[0]), max(0, -offsets[0])
    common = min(len(earlier) - first_start, len(submitted) - second_start)
    if max(len(earlier), len(submitted)) - common > TRIM_LIMIT:
        return None
    clipped = []
    overlap = {}
    for side, track, start in (("earlier", first, first_start), ("submitted", second, second_start)):
        overlap[side] = _clock_range(track, start, start + common)
        clipped.append(("measured", track[1][start:start + common],
                        {"source_pts": track[2]["source_pts"] + start}))
    offsets = (first_start, second_start)
    adjusted = None
    if adjust_volume:
        spans = tuple((w["earlier"]["start_sample"] - first_start, w["earlier"]["end_sample"] - first_start) for w in supports)
        adjusted = _level_adjusted(*clipped, assets, support_spans=spans, file_offsets=offsets)
        if adjusted is None:
            return None
        supports = adjusted["support"]
    delta = clipped[1][1] - clipped[0][1]
    changes = _measure_changes(*clipped, assets, delta != 0, delta, origin=Fraction(second_start, AUDIO_RATE))
    _attach_file_ranges(changes["ranges"], clipped, offsets)
    return {"status": "measured", "earlier": first[2], "submitted": second[2], **changes,
            "trim_alignment": {"earlier_minus_submitted_samples": first_start - second_start, "compared_samples": common,
                               "overlap": overlap, "support": supports}}, adjusted


def compare(earlier: Path, submitted: Path, *, filenames: tuple[str, str] | None = None,
            include_level_adjustment: bool = True,
            include_trim_alignment: bool = True,
            include_trim_level_adjustment: bool = True) -> tuple[dict, dict[str, bytes]]:
    """Compare continuous recordings and recover a small, exactly supported edge trim."""
    paths = earlier, submitted
    filenames = filenames or (earlier.name, submitted.name)
    sources = tuple(_identity(path, name) for path, name in zip(paths, filenames, strict=True))
    first, second = [_read_track(path, source, max_seconds=MAX_SECONDS, origin=Fraction(0), video=False)
                     for path, source in zip(paths, sources, strict=True)]
    tracks = first[0], second[0]
    assets: dict[str, bytes] = {}
    audio = compare_tracks(*tracks, assets, standalone=True)
    status = "comparison_limit" if any(track[0] == "comparison_limit" for track in tracks) else audio["status"]
    record = {"schema": SCHEMA, "status": status, "earlier": sources[0], "submitted": sources[1],
              "audio": audio, "interpretation": INTERPRETATION, "affects_overall_confidence": False}
    if (include_trim_alignment and status == "audio_timelines_differ" and first[2] and second[2]
            and audio["earlier"]["samples"] != audio["submitted"]["samples"]):
        aligned = _audio_trim(*tracks, assets)
        if aligned is not None:
            record.update(schema=SCHEMA_WITH_TRIM_ALIGNMENT, status="measured", audio=aligned[0],
                          interpretation=TRIM_INTERPRETATION)
        elif include_trim_level_adjustment and include_level_adjustment:
            aligned = _audio_trim(*tracks, assets, adjust_volume=True)
            if aligned is not None:
                record.update(schema=SCHEMA_WITH_TRIM_AND_LEVEL, status="measured", audio=aligned[0],
                              interpretation=TRIM_LEVEL_INTERPRETATION, level_adjusted=aligned[1])
    if include_level_adjustment and status == "measured" and audio["changed_samples"] and first[1] and second[1]:
        adjusted = _level_adjusted(*tracks, assets)
        if adjusted is not None:
            record.update(schema=SCHEMA_WITH_LEVEL_ADJUSTMENT, level_adjusted=adjusted)
    if sources != tuple(_identity(path, name) for path, name in zip(paths, filenames, strict=True)):
        raise ValueError("One of the recordings changed while the comparison was running.")
    return record, assets
