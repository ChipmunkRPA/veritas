"""Measure picture and soundtrack differences between two clock-matched videos."""

from __future__ import annotations

import hashlib
import io
import os
from contextlib import closing
from fractions import Fraction
from itertools import pairwise, zip_longest
from pathlib import Path

import numpy as np
from PIL import Image

from app.core.canonical import canonical_hash
from app.detectors.video.decode import DecodeError, FrameTable, decode_frames, probe_frames
from app.evidence import video_reference_local
from app.evidence.audio_reference import compare_tracks, marked_ranges, read_track
from app.storage.files import VIDEO_MIMES, sniff

SCHEMA = "chipmunk.video-reference-comparison/1"
SCHEMA_WITH_LOCAL = "chipmunk.video-reference-comparison/2"
SCHEMA_WITH_TIMING = "chipmunk.video-reference-comparison/3"
SCHEMA_WITH_SHORT_TIMING = "chipmunk.video-reference-comparison/4"
SCHEMA_WITH_SOUNDTRACK = "chipmunk.video-reference-comparison/5"
MAX_BYTES = 48 * 1024 * 1024
MAX_FRAMES = 360
MAX_PIXELS = 1280 * 720
MAX_SECONDS = 12
MAX_PRESENTED_RANGES = 3
PREVIEW_SIDE = 420

INTERPRETATION = (
    "The comparison shows changes in decoded pictures and the first soundtrack. "
    "Encoding, color conversion, and ordinary edits can change decoded values. "
    "Establishing edit intent or AI authorship requires other evidence."
)


def _identity(path: Path) -> dict:
    with path.open("rb") as stream:
        count = os.fstat(stream.fileno()).st_size
        if not 0 < count <= MAX_BYTES:
            raise ValueError("Each clip must be nonempty and at most 48 MiB.")
        mime, _ = sniff(stream.read(8192), path.name)
        if mime not in VIDEO_MIMES:
            raise ValueError("Choose two readable video files.")
        stream.seek(0)
        return {"sha256": hashlib.file_digest(stream, "sha256").hexdigest(), "bytes": count}


def _reference(table: FrameTable) -> dict:
    return {"width": table.width, "height": table.height, "time_base": table.time_base,
            "first_pts": table.pts[0], "last_pts": table.pts[-1], "frames": len(table),
            "frame_pts_sha256": canonical_hash(list(table.pts))}


def _elapsed(table: FrameTable) -> list[Fraction]:
    return [(pts - table.pts[0]) * table.tb for pts in table.pts]


def _comparison_issue(tables: tuple[FrameTable, FrameTable]) -> str | None:
    if any(table.truncated for table in tables):
        return "comparison_limit"
    if any(len(table) < 2 or table.interpolated or any(a >= b for a, b in pairwise(table.pts)) for table in tables):
        return "unverified_picture_clock"
    earlier, later = tables
    if (earlier.width, earlier.height) != (later.width, later.height):
        return "picture_geometry_differs"
    clocks = [_elapsed(table) for table in tables]
    if any(clock[-1] > MAX_SECONDS for clock in clocks):
        return "comparison_limit"
    if clocks[0] != clocks[1]:
        return "picture_timelines_differ"
    return None


def _asset(assets, filename, content, media_type):
    assets[filename] = content
    return {"file": filename, "sha256": hashlib.sha256(content).hexdigest(), "media_type": media_type}


def _image(assets, name, rgb):
    with Image.fromarray(rgb) as picture, io.BytesIO() as output:
        picture.thumbnail((PREVIEW_SIDE, PREVIEW_SIDE), Image.Resampling.LANCZOS)
        picture.save(output, "PNG")
        return _asset(assets, name, output.getvalue(), "image/png")


def _bounds(mask) -> dict:
    xs, = np.nonzero(mask.any(axis=0))
    ys, = np.nonzero(mask.any(axis=1))
    return {"x": int(xs[0]), "y": int(ys[0]), "w": int(xs[-1] - xs[0] + 1), "h": int(ys[-1] - ys[0] + 1)}


def _frame_pairs(paths, tables):
    with (closing(decode_frames(paths[0], tables[0], None, timeout_s=40, max_pixels=MAX_PIXELS)) as earlier,
          closing(decode_frames(paths[1], tables[1], None, timeout_s=40, max_pixels=MAX_PIXELS)) as later):
        for expected, (first, second) in enumerate(zip_longest(earlier, later)):
            if (first is None or second is None or first.index != expected or second.index != expected or
                    not first.pts_verified or not second.pts_verified):
                raise DecodeError("Video could not be decoded with verified picture timestamps.")
            yield first, second


def _clock_range(tables, start, end):
    return {label: {"start_pts": table.pts[start], "last_pts": table.pts[end - 1],
                    "end_exclusive_pts": table.pts[end] if end < len(table) else None}
            for label, table in zip(("earlier", "submitted"), tables, strict=True)}


def _picture(paths, tables, assets, *, include_local):
    clock = _elapsed(tables[1])
    local_check = video_reference_local.eligible(paths, tables, clock) if include_local else None
    rects = video_reference_local.rectangles(tables[0].height, tables[0].width) if local_check else None
    measurements = []
    signals, background_errors = [], []
    for first, second in _frame_pairs(paths, tables):
        difference = first.rgb != second.rgb
        mask = difference[..., 0] | difference[..., 1] | difference[..., 2]
        count = int(np.count_nonzero(mask))
        measurements.append({"frame": first.index, "earlier_pts": first.pts, "submitted_pts": second.pts,
                             "earlier_rgb_sha256": hashlib.sha256(first.rgb).hexdigest(),
                             "submitted_rgb_sha256": hashlib.sha256(second.rgb).hexdigest(),
                             "changed_pixels": count, "difference_bounds": _bounds(mask) if count else None})
        if rects is not None:
            signal, background = video_reference_local.measure(first.rgb, second.rgb, rects)
            signals.append(signal)
            background_errors.append(background)
    if len(measurements) != len(tables[0]):
        raise DecodeError("Video could not be decoded with every recorded picture.")

    sizes = [row["changed_pixels"] for row in measurements]
    ranges = marked_ranges(np.asarray(sizes) > 0)
    top = set(sorted(ranges, key=lambda pair: (-(pair[1] - pair[0]), pair[0]))[:MAX_PRESENTED_RANGES])
    output, needed = [], set()
    for number, (start, end) in enumerate(ranges, 1):
        strongest = max(range(start, end), key=sizes.__getitem__)
        examples = {}
        if (start, end) in top:
            examples = {"during": strongest}
            if start:
                examples["before"] = start - 1
            if end < len(measurements):
                examples["after"] = end
            needed.update(examples.values())
        output.append({"id": f"V{number:02d}", "start_frame": start, "end_frame_exclusive": end,
                       "start_s": float(clock[start]), "last_s": float(clock[end - 1]),
                       "end_s": float(clock[end]) if end < len(clock) else None,
                       "source_ranges": _clock_range(tables, start, end), "strongest_frame": measurements[strongest],
                       "preview_indices": examples})

    local = choice = None
    if local_check is not None and rects is not None:
        dt, scope = local_check
        status, choice = video_reference_local.assess(signals, background_errors, dt, rects)
        local = {"status": status, "summary": video_reference_local.STATUS[status], "scope": scope}
        if choice:
            needed.update(choice["samples"].values())

    previews, crops = {}, {}
    crop_indices = set(choice["samples"].values()) if choice else set()
    for side, path, table in zip(("earlier", "submitted"), paths, tables, strict=True):
        with closing(decode_frames(path, table, sorted(needed), timeout_s=40, max_pixels=MAX_PIXELS)) as frames:
            for frame in frames:
                digest = hashlib.sha256(frame.rgb).hexdigest()
                if not frame.pts_verified or digest != measurements[frame.index][f"{side}_rgb_sha256"]:
                    raise DecodeError("Video could not be decoded with consistent picture evidence.")
                previews.setdefault(frame.index, {})[side] = _image(assets, f"{side}-frame-{frame.index:03d}.png", frame.rgb)
                if choice and frame.index in crop_indices:
                    x, y, w, h = (choice["tile"][key] for key in ("x", "y", "w", "h"))
                    crops.setdefault(frame.index, {})[side] = _image(
                        assets, f"{side}-local-crop-{frame.index:03d}.png", frame.rgb[y:y+h, x:x+w]
                    )
    if set(previews) != needed or any(len(pictures) != 2 for pictures in previews.values()):
        raise DecodeError("Video could not be decoded with all requested evidence pictures.")

    for item in output:
        item["previews"] = {position: {**measurements[index], "assets": previews[index], "time_s": float(clock[index])}
                            for position, index in item.pop("preview_indices").items()}
    changed = sum(size > 0 for size in sizes)
    picture = {"status": "measured", "matching_frames": len(sizes) - changed, "changed_frames": changed,
               "frame_pairs_sha256": canonical_hash(measurements), "ranges": output}
    if local is not None:
        if choice:
            start, end = choice["frames"]
            local["range"] = {"id": "L01", "start_frame": start, "end_frame_exclusive": end,
                              "start_s": float(clock[start]), "end_s": float(clock[end]),
                              "source_ranges": _clock_range(tables, start, end), "tile": choice["tile"],
                              "previews": {role: {**measurements[index], "time_s": float(clock[index]),
                                                  "assets": previews[index], "crop_assets": crops[index]}
                                           for role, index in choice["samples"].items()}}
        picture["local_comparison"] = local
    return picture


def compare(earlier: Path, submitted: Path, *, include_local: bool = True,
            timing_event_s: Fraction | None = None, timing_model: Path | None = None,
            timing_short_fallback: bool = False, include_soundtrack: bool = True) -> tuple[dict, dict[str, bytes]]:
    """Report actual decoded differences at equal relative native picture timestamps."""
    paths = earlier, submitted
    sources = tuple(_identity(path) for path in paths)
    tables = tuple(probe_frames(path, cap=MAX_FRAMES, timeout_s=30, max_pixels=MAX_PIXELS) for path in paths)
    status = _comparison_issue(tables) or "measured"
    assets: dict[str, bytes] = {}
    record = {"schema": SCHEMA, "status": status,
              "earlier": {**sources[0], **_reference(tables[0])}, "submitted": {**sources[1], **_reference(tables[1])},
              "interpretation": INTERPRETATION, "affects_overall_confidence": False}
    tracks = None
    if status == "measured":
        record["picture"] = _picture(paths, tables, assets, include_local=include_local)
        if "local_comparison" in record["picture"]:
            record["schema"] = SCHEMA_WITH_LOCAL
        tracks = [read_track(path, source, max_seconds=MAX_SECONDS, video=True, origin=table.pts[0] * table.tb)
                  for path, source, table in zip(paths, sources, tables, strict=True)]
        record["audio"] = compare_tracks(*tracks, assets)
    if timing_event_s is not None:
        from app.evidence import video_reference_timing

        record["schema"] = SCHEMA_WITH_TIMING
        record["timing"] = video_reference_timing.assess(
            paths, tables if status == "measured" else None, tracks,
            record.get("picture", {}).get("matching_frames", 0), timing_event_s, timing_model, assets, _image,
            short_fallback=timing_short_fallback,
        )
        if record["timing"].get("waveform_basis") == video_reference_timing.SHORT_WAVEFORM_BASIS:
            record["schema"] = SCHEMA_WITH_SHORT_TIMING
    if (include_soundtrack and tracks is not None and not record["picture"]["changed_frames"]
            and record["audio"]["status"] == "measured"):
        from app.evidence import video_reference_soundtrack

        timing_assets = {}
        try:
            alignment = video_reference_soundtrack.assess(submitted, tables[1], record["picture"], record["audio"], tracks, timing_assets)
        except DecodeError:
            alignment = {"status": "unassessed", "scope": video_reference_soundtrack.SCOPE, "affects_overall_confidence": False,
                         "summary": "The check could not retrieve the verified picture for the matched soundtrack excerpt."}
        else:
            assets.update(timing_assets)
        record.update(schema=SCHEMA_WITH_SOUNDTRACK, soundtrack_alignment=alignment)
    if sources != tuple(_identity(path) for path in paths):
        raise ValueError("One of the clips changed while the comparison was running.")
    return record, assets
