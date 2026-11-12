"""Exact time-aligned comparisons of decoded mono audio."""

from __future__ import annotations

import hashlib
import os
from fractions import Fraction
from pathlib import Path

import numpy as np

from app.core.canonical import canonical_hash
from app.detectors.audio.decode import SAMPLE_RATE as AUDIO_RATE
from app.detectors.audio.decode import AudioStreamUnavailable, decode_audio, wav_bytes
from app.detectors.audio.waveform import envelope
from app.storage.files import sniff_audio

SCHEMA = "chipmunk.audio-reference-comparison/1"
MAX_BYTES = 48 * 1024 * 1024
MAX_SECONDS = 30
MAX_PRESENTED_RANGES = 3
AUDIO_BIN_SAMPLES = AUDIO_RATE // 50
AUDIO_PREVIEW_SAMPLES = 2 * AUDIO_RATE
INTERPRETATION = (
    "This comparison measures differences in the first audio track after conversion to 16 kHz mono. "
    "Encoding, volume changes, or ordinary edits can change samples. "
    "The difference waveform shows where the decoded copies disagree. "
    "The comparison does not determine who spoke or whether AI was used."
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


def read_track(path, source, *, max_seconds, origin=Fraction(0), video=False):
    try:
        decoded = decode_audio(path, source_sha256=source["sha256"], max_seconds=max_seconds + 1)
    except AudioStreamUnavailable:
        return "no_stream", None, None
    except (ValueError, RuntimeError):
        return "unverified_audio_clock", None, None
    mapping = decoded.mapping
    if not video and mapping["samples"] > max_seconds * AUDIO_RATE:
        return "comparison_limit", None, None
    if len(mapping["timeline"]) != 1 or mapping["samples"] >= mapping["limit_samples"]:
        return "unverified_audio_clock", None, None
    pts = mapping["timeline"][0]["source_pts"]
    relative = Fraction(pts, AUDIO_RATE) - origin
    return "measured", decoded.pcm, {"pcm_sha256": mapping["pcm_sha256"], "source_pts": pts,
                                    "relative_start": str(relative), "samples": len(decoded.pcm)}


def compare_tracks(first, second, assets, *, standalone=False):
    if first[0] == second[0] == "no_stream":
        return {"status": "no_stream"}
    if first[0] != "measured" or second[0] != "measured":
        return {"status": "audio_unassessed", "earlier": first[0], "submitted": second[0]}
    left, right = first[2], second[2]
    if (left["relative_start"], left["samples"]) != (right["relative_start"], right["samples"]):
        return {"status": "audio_timelines_differ", "earlier": left, "submitted": right, "sample_rate": AUDIO_RATE}

    difference = first[1] != second[1]
    delta = second[1] - first[1] if standalone else None
    changed_per_bin = np.add.reduceat(difference, np.arange(0, len(difference), AUDIO_BIN_SAMPLES), dtype=np.int64)
    ranges = marked_ranges(changed_per_bin > 0)
    top = set(sorted(ranges, key=lambda pair: (-(pair[1] - pair[0]), pair[0]))[:MAX_PRESENTED_RANGES])
    result = []
    origin = Fraction(0) if standalone else Fraction(right["relative_start"])
    for number, (first_bin, after_bin) in enumerate(ranges, 1):
        start, end = first_bin * AUDIO_BIN_SAMPLES, min(after_bin * AUDIO_BIN_SAMPLES, len(difference))
        item = {"id": f"A{number:02d}", "start_sample": start, "end_sample": end,
                "start_s": float(origin + Fraction(start, AUDIO_RATE)), "end_s": float(origin + Fraction(end, AUDIO_RATE)),
                "changed_samples": int(np.sum(changed_per_bin[first_bin:after_bin])),
                "source_ranges": {
                    side: {"start_pts": details["source_pts"] + start, "end_exclusive_pts": details["source_pts"] + end}
                    for side, details in (("earlier", left), ("submitted", right))
                }}
        if delta is not None:
            span = delta[start:end].astype(np.float64)
            item["difference"] = {"peak": float(np.max(np.abs(span))), "rms": float(np.sqrt(np.mean(span**2)))}
        if (first_bin, after_bin) in top:
            preview_end = min(end, start + AUDIO_PREVIEW_SAMPLES)
            excerpts = {"earlier": first[1][start:preview_end], "submitted": second[1][start:preview_end]}
            item["preview"] = {"start_sample": start, "end_sample": preview_end,
                               "end_s": float(origin + Fraction(preview_end, AUDIO_RATE)),
                               "waveforms": {label: envelope(pcm) for label, pcm in excerpts.items()},
                               "assets": {label: _wav_asset(assets, f"{label}-audio-{number:02d}.wav", wav_bytes(pcm))
                                          for label, pcm in excerpts.items()}}
            if delta is not None:
                item["preview"]["difference_waveform"] = envelope(delta[start:preview_end], amplitude_limit=2.0, decimals=8)
        result.append(item)
    return {"status": "measured", "earlier": left, "submitted": right, "sample_rate": AUDIO_RATE,
            "bin_samples": AUDIO_BIN_SAMPLES, "changed_samples": int(np.count_nonzero(difference)),
            "bin_counts_sha256": canonical_hash(changed_per_bin.tolist()), "ranges": result}


def compare(earlier: Path, submitted: Path, *, filenames: tuple[str, str] | None = None) -> tuple[dict, dict[str, bytes]]:
    """Compare complete recordings with the same continuous start and sample count."""
    paths = earlier, submitted
    filenames = filenames or (earlier.name, submitted.name)
    sources = tuple(_identity(path, name) for path, name in zip(paths, filenames, strict=True))
    tracks = [read_track(path, source, max_seconds=MAX_SECONDS) for path, source in zip(paths, sources, strict=True)]
    assets: dict[str, bytes] = {}
    audio = compare_tracks(*tracks, assets, standalone=True)
    status = "comparison_limit" if any(track[0] == "comparison_limit" for track in tracks) else audio["status"]
    record = {"schema": SCHEMA, "status": status, "earlier": sources[0], "submitted": sources[1],
              "audio": audio, "interpretation": INTERPRETATION, "affects_overall_confidence": False}
    if sources != tuple(_identity(path, name) for path, name in zip(paths, filenames, strict=True)):
        raise ValueError("One of the recordings changed while the comparison was running.")
    return record, assets
