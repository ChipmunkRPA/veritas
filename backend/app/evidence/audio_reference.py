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
MAX_BYTES = 48 * 1024 * 1024
MAX_SECONDS = 30
MAX_PRESENTED_RANGES = 3
AUDIO_BIN_SAMPLES = AUDIO_RATE // 50
AUDIO_PREVIEW_SAMPLES = 2 * AUDIO_RATE
NATIVE_LOSSLESS_CODECS = frozenset({"pcm_s16le", "pcm_s16be", "flac", "alac"})
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


def _read_track(path, source, *, max_seconds, origin, video):
    try:
        decoded = decode_audio(path, source_sha256=source["sha256"], max_seconds=max_seconds + 1)
    except AudioStreamUnavailable:
        return ("no_stream", None, None), False
    except (ValueError, RuntimeError):
        return ("unverified_audio_clock", None, None), False
    mapping = decoded.mapping
    if not video and mapping["samples"] > max_seconds * AUDIO_RATE:
        return ("comparison_limit", None, None), False
    if len(mapping["timeline"]) != 1 or mapping["samples"] >= mapping["limit_samples"]:
        return ("unverified_audio_clock", None, None), False
    pts = mapping["timeline"][0]["source_pts"]
    relative = Fraction(pts, AUDIO_RATE) - origin
    native_lossless = (mapping["source_channels"] == 1 and mapping["source_sample_rate"] == AUDIO_RATE
                       and mapping["source_codec"] in NATIVE_LOSSLESS_CODECS)
    return ("measured", decoded.pcm, {"pcm_sha256": mapping["pcm_sha256"], "source_pts": pts,
                                     "relative_start": str(relative), "samples": len(decoded.pcm)}), native_lossless


def read_track(path, source, *, max_seconds, origin=Fraction(0), video=False):
    track, _ = _read_track(path, source, max_seconds=max_seconds, origin=origin, video=video)
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


def compare_tracks(first, second, assets, *, standalone=False):
    if first[0] == second[0] == "no_stream":
        return {"status": "no_stream"}
    if first[0] != "measured" or second[0] != "measured":
        return {"status": "audio_unassessed", "earlier": first[0], "submitted": second[0]}
    left, right = first[2], second[2]
    if (left["relative_start"], left["samples"]) != (right["relative_start"], right["samples"]):
        return {"status": "audio_timelines_differ", "earlier": left, "submitted": right, "sample_rate": AUDIO_RATE}

    delta = second[1] - first[1] if standalone else None
    origin = Fraction(0) if standalone else Fraction(right["relative_start"])
    return {"status": "measured", "earlier": left, "submitted": right,
            **_measure_changes(first, second, assets, first[1] != second[1], delta, origin=origin)}


def _gain_bounds(bounds):
    return {"lower": str(bounds[0]), "upper": str(bounds[1])}


def _level_adjusted(first, second, assets):
    fit = audio_level_adjustment.compare(first[1], second[1])
    if fit is None:
        return None
    changes = _measure_changes(first, second, assets, fit.changed, fit.residual, origin=Fraction(0),
                               prefix="level-adjusted-", id_prefix="V")
    count = len(first[1])
    width = audio_level_adjustment.SUPPORT_SAMPLES
    support = []
    for (start, end), bounds in zip(((0, width), (count - width, count)), fit.support_bounds, strict=True):
        support.append({"start_sample": start, "end_sample": end, "gain_bounds": _gain_bounds(bounds),
                        "source_ranges": _source_ranges(first, second, start, end),
                        "pcm_sha256": {side: hashlib.sha256((track[1][start:end] * 32768).astype("<i2").tobytes()).hexdigest()
                                       for side, track in (("earlier", first), ("submitted", second))}})
    midpoint = sum(fit.bounds) / 2
    return {"status": "measured", **changes, "gain_bounds": _gain_bounds(fit.bounds),
            "gain_db": round(20 * math.log10(midpoint), 6), "support": support,
            "interpretation": LEVEL_INTERPRETATION}


def compare(earlier: Path, submitted: Path, *, filenames: tuple[str, str] | None = None,
            include_level_adjustment: bool = True) -> tuple[dict, dict[str, bytes]]:
    """Compare complete recordings with the same continuous start and sample count."""
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
    if include_level_adjustment and status == "measured" and audio["changed_samples"] and first[1] and second[1]:
        adjusted = _level_adjusted(*tracks, assets)
        if adjusted is not None:
            record.update(schema=SCHEMA_WITH_LEVEL_ADJUSTMENT, level_adjusted=adjusted)
    if sources != tuple(_identity(path, name) for path, name in zip(paths, filenames, strict=True)):
        raise ValueError("One of the recordings changed while the comparison was running.")
    return record, assets
