"""Measure picture and soundtrack differences between two clock-matched videos."""

from __future__ import annotations

import hashlib
import io
import os
from collections import Counter
from contextlib import closing
from fractions import Fraction
from itertools import pairwise, zip_longest
from pathlib import Path

import numpy as np
from PIL import Image

from app.core.canonical import canonical_hash
from app.detectors.audio.decode import SAMPLE_RATE, audio_stream_indices
from app.detectors.video.decode import DecodeError, FrameTable, decode_frames, probe_frames
from app.evidence import (
    video_reference_insertion,
    video_reference_local,
    video_reference_measured_interval,
    video_reference_photometric,
    video_reference_soundtrack_reencode,
    video_reference_trimmed,
)
from app.evidence.audio_reference import (
    _read_track,
    channel_timing,
    compare_tracks,
    marked_ranges,
    read_track,
)
from app.storage.files import VIDEO_MIMES, sniff

SCHEMA = "chipmunk.video-reference-comparison/1"
SCHEMA_WITH_LOCAL = "chipmunk.video-reference-comparison/2"
SCHEMA_WITH_TIMING = "chipmunk.video-reference-comparison/3"
SCHEMA_WITH_SHORT_TIMING = "chipmunk.video-reference-comparison/4"
SCHEMA_WITH_SOUNDTRACK = "chipmunk.video-reference-comparison/5"
SCHEMA_WITH_ADDITIONAL_SOUNDTRACK = "chipmunk.video-reference-comparison/6"
SCHEMA_WITH_MARKED_SOUND_FALLBACK = "chipmunk.video-reference-comparison/7"
SCHEMA_WITH_CHANNEL_TIMING = "chipmunk.video-reference-comparison/8"
SCHEMA_WITH_TRIM = "chipmunk.video-reference-comparison/9"
SCHEMA_WITH_TRIM_REENCODE = "chipmunk.video-reference-comparison/10"
SCHEMA_WITH_REENCODED_SOUNDTRACK = "chipmunk.video-reference-comparison/11"
SCHEMA_WITH_PICTURE_INSERTION = "chipmunk.video-reference-comparison/12"
SCHEMA_WITH_INSERTION_RETENTION = "chipmunk.video-reference-comparison/13"
MAX_COMPARED_SOUNDTRACKS = 2
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
ADDITIONAL_AUDIO_SCOPE = (
    "The second soundtrack is compared by track order after decoding to mono. The comparison requires matching "
    "clocks and supports up to 12 seconds of audio. Alternate mixes and ordinary edits can cause "
    "differences. This comparison contributes zero weight to the AI estimate."
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


def _image(assets, name, rgb, *, native=False):
    with Image.fromarray(rgb) as picture, io.BytesIO() as output:
        if not native:
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


def _clock_range(tables, start, end, earlier_offset=0):
    return {label: {"start_pts": table.pts[start + offset], "last_pts": table.pts[end + offset - 1],
                    "end_exclusive_pts": table.pts[end + offset] if end + offset < len(table) else None}
            for label, table, offset in zip(("earlier", "submitted"), tables, (earlier_offset, 0), strict=True)}


def _insertion_evidence(paths, tables, assets, located):
    start, stop = located["bounds"]
    witnesses = {}
    frame_ranges = ((start - 1, start), (start - 1, (start + stop) // 2, stop))
    for side, path, table, indices in zip(("earlier", "submitted"), paths, tables, frame_ranges, strict=True):
        roles = ("before", "after") if side == "earlier" else video_reference_insertion.EVIDENCE_ROLES
        selected = {}
        with closing(decode_frames(path, table, indices, timeout_s=30, max_pixels=MAX_PIXELS)) as frames:
            for frame in frames:
                if frame.index not in indices or frame.index in selected or not frame.pts_verified:
                    raise DecodeError("Video could not be decoded with verified insertion witnesses.")
                selected[frame.index] = frame
        if selected.keys() != set(indices):
            raise DecodeError("Video could not be decoded with every insertion witness.")
        for role, index in zip(roles, indices, strict=True):
            frame = selected[index]
            witnesses.setdefault(role, {})[side] = {
                "frame_index": index, "pts": frame.pts, "time_base": table.time_base,
                "time_s": float((frame.pts - table.pts[0]) * table.tb),
                "asset": _image(assets, f"insertion-{role}-{side}.png", frame.rgb),
            }
    return {"start_frame": start, "end_frame_exclusive": stop, "inserted_frames": stop - start,
            "start_s": tables[1].seconds(start), "end_s": tables[1].seconds(stop),
            "source_ticks": {"start_pts": tables[1].pts[start], "end_exclusive_pts": tables[1].pts[stop],
                             "time_base": tables[1].time_base},
            "earlier_join_frame": start, "plausible_start_frames": located["uncertain_starts"],
            "before_error_p50_p90": located["before_error_p50_p90"],
            "after_error_p50_p90": located["after_error_p50_p90"],
            "witnesses": witnesses, "audio_checked": False}


def _trim_offset(paths, tables):
    earlier, submitted = tables
    if not 32 <= len(submitted) < len(earlier) or any(table.rotation for table in tables):
        return None
    hashes = []
    for path, table in zip(paths, tables, strict=True):
        values = []
        with closing(decode_frames(path, table, None, timeout_s=40, max_pixels=MAX_PIXELS)) as frames:
            for frame in frames:
                if frame.index != len(values) or not frame.pts_verified:
                    raise DecodeError("Video could not be decoded with verified picture timestamps.")
                values.append(hashlib.sha256(frame.rgb).digest())
        if len(values) != len(table):
            raise DecodeError("Video could not be decoded with every recorded picture.")
        hashes.append(values)
    old, new = hashes
    old_count, new_count = Counter(old), Counter(new)
    unique_old = {value: i for i, value in enumerate(old) if old_count[value] == 1}
    votes = Counter(unique_old[value] - i for i, value in enumerate(new) if new_count[value] == 1 and value in unique_old)
    qualified = [(offset, count) for offset, count in votes.items()
                 if 0 <= offset <= len(old) - len(new) and count >= max(16, (len(new) + 1) // 2)]
    if len(qualified) != 1:
        return None
    offset, support = qualified[0]
    if [(pts - earlier.pts[offset]) * earlier.tb for pts in earlier.pts[offset:offset + len(new)]] != _elapsed(submitted):
        return None
    matches = np.asarray([a == b for a, b in zip(old[offset:offset + len(new)], new, strict=True)])
    unique = [same and new_count[value] == old_count[value] == 1 for same, value in zip(matches, new, strict=True)]
    flank = len(new) // 4
    if min(sum(unique[:flank]), sum(unique[-flank:])) < 3:
        return None
    ranges = marked_ranges(~matches)
    if len(ranges) > MAX_PRESENTED_RANGES or any(start == 0 or end == len(new) or not unique[start - 1]
                                               or not unique[end] for start, end in ranges):
        return None
    return offset, support


def _trimmed_pairs(paths, tables, offset):
    wanted = range(offset, offset + len(tables[1]))
    with (closing(decode_frames(paths[0], tables[0], wanted, timeout_s=40, max_pixels=MAX_PIXELS)) as old,
          closing(decode_frames(paths[1], tables[1], None, timeout_s=40, max_pixels=MAX_PIXELS)) as new):
        for index, (first, second) in enumerate(zip_longest(old, new)):
            if (first is None or second is None or first.index != index + offset or second.index != index
                    or not first.pts_verified or not second.pts_verified):
                raise DecodeError("Video could not be decoded on its aligned native clocks.")
            yield first, second


def _picture(paths, tables, assets, *, include_local, include_local_area=False, include_measured_interval=False,
             include_soundtrack_reencode=False, trim_offset=None, trim_reencoded=False):
    clock = _elapsed(tables[1])
    earlier_clock = _elapsed(tables[0]) if trim_offset is not None else None
    earlier_offset = trim_offset if trim_offset is not None else 0
    local_check = video_reference_local.eligible(paths, tables, clock) if include_local else None
    area_check = trim_reencoded or (
        include_local and include_local_area and video_reference_photometric.eligible(paths, tables, clock)
    )
    rects = video_reference_local.rectangles(tables[0].height, tables[0].width) if local_check or area_check else None
    measurements = []
    signals, background_errors, cells = [], [], []
    hd_profile = (video_reference_soundtrack_reencode.hd_report_profile(tables)
                  if include_soundtrack_reencode and (tables[1].width, tables[1].height)
                  == video_reference_soundtrack_reencode.HD_PICTURE_SIZE else None)
    timing_differences = [] if include_soundtrack_reencode else None
    fine_timing_differences = [] if hd_profile is not None else None
    pairs = _frame_pairs(paths, tables) if trim_offset is None else _trimmed_pairs(paths, tables, earlier_offset)
    for first, second in pairs:
        difference = first.rgb != second.rgb
        mask = difference[..., 0] | difference[..., 1] | difference[..., 2]
        count = int(np.count_nonzero(mask))
        measurements.append({"frame": second.index, "earlier_pts": first.pts, "submitted_pts": second.pts,
                             "earlier_rgb_sha256": hashlib.sha256(first.rgb).hexdigest(),
                             "submitted_rgb_sha256": hashlib.sha256(second.rgb).hexdigest(),
                             "changed_pixels": count, "difference_bounds": _bounds(mask) if count else None})
        if earlier_clock is not None:
            measurements[-1].update(earlier_frame=first.index, earlier_time_s=float(earlier_clock[first.index]))
        if timing_differences is not None:
            timing_differences.append(video_reference_soundtrack_reencode.picture_difference(first.rgb, second.rgb))
        if fine_timing_differences is not None:
            fine_timing_differences.append(video_reference_soundtrack_reencode.picture_difference(
                first.rgb, second.rgb, grid=video_reference_soundtrack_reencode.HD_GRID))
        if rects is not None:
            if area_check:
                signal, background, grid = video_reference_photometric.measurements(first.rgb, second.rgb, rects)
                cells.append(grid)
            else:
                signal, background = video_reference_local.measure(first.rgb, second.rgb, rects)
            signals.append(signal)
            background_errors.append(background)
    if len(measurements) != len(tables[1]):
        raise DecodeError("Video could not be decoded with every recorded picture.")

    sizes = [row["changed_pixels"] for row in measurements]
    ranges = [] if trim_reencoded else marked_ranges(np.asarray(sizes) > 0)
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
                       "source_ranges": _clock_range(tables, start, end, earlier_offset), "strongest_frame": measurements[strongest],
                       "preview_indices": examples})
        if earlier_clock is not None:
            output[-1].update(earlier_start_s=float(earlier_clock[start + earlier_offset]),
                              earlier_end_s=float(earlier_clock[end + earlier_offset]))

    local = choice = precise = None
    if rects is not None:
        if area_check:
            scope = video_reference_photometric.SCOPE
            status, choice = video_reference_photometric.assess(signals, background_errors, rects)
        else:
            dt, scope = local_check
            status, choice = video_reference_local.assess(signals, background_errors, dt, rects)
        local = {"status": status, "summary": video_reference_local.STATUS[status], "scope": scope}
        if area_check:
            local["basis"] = video_reference_photometric.BASIS
            if choice:
                if include_measured_interval and not trim_reencoded:
                    xs, ys, _, _ = rects
                    seed = np.flatnonzero((xs == choice["tile"]["x"]) & (ys == choice["tile"]["y"]))
                    refined = (video_reference_measured_interval.refine(np.asarray(signals)[:, int(seed[0])], choice["frames"])
                               if len(seed) == 1 else None)
                    if refined and refined != choice["frames"]:
                        start, end = refined
                        precise = {"frames": refined, "samples": {"before": start - 1, "during": (start + end) // 2, "after": end}}
                choice["tile"] = video_reference_photometric.expanded_region(np.array(cells), choice, constrained=False)
        if choice:
            needed.update(choice["samples"].values())
        if precise:
            needed.update(precise["samples"].values())

    previews, crops = {}, {}
    crop_indices = set(choice["samples"].values()) if choice else set()
    if precise:
        crop_indices.update(precise["samples"].values())
    for side, path, table in zip(("earlier", "submitted"), paths, tables, strict=True):
        offset = earlier_offset if side == "earlier" else 0
        with closing(decode_frames(path, table, sorted(index + offset for index in needed), timeout_s=40, max_pixels=MAX_PIXELS)) as frames:
            for frame in frames:
                index = frame.index - offset
                digest = hashlib.sha256(frame.rgb).hexdigest()
                if not frame.pts_verified or digest != measurements[index][f"{side}_rgb_sha256"]:
                    raise DecodeError("Video could not be decoded with consistent picture evidence.")
                previews.setdefault(index, {})[side] = _image(assets, f"{side}-frame-{frame.index:03d}.png", frame.rgb)
                if choice and index in crop_indices:
                    x, y, w, h = (choice["tile"][key] for key in ("x", "y", "w", "h"))
                    crops.setdefault(index, {})[side] = _image(
                        assets, f"{side}-local-crop-{frame.index:03d}.png", frame.rgb[y:y+h, x:x+w], native=area_check
                    )
    if set(previews) != needed or any(len(pictures) != 2 for pictures in previews.values()):
        raise DecodeError("Video could not be decoded with all requested evidence pictures.")

    for item in output:
        item["previews"] = {position: {**measurements[index], "assets": previews[index], "time_s": float(clock[index])}
                            for position, index in item.pop("preview_indices").items()}
    changed = sum(size > 0 for size in sizes)
    picture = {"status": "measured", "matching_frames": len(sizes) - changed, "changed_frames": changed,
               "frame_pairs_sha256": canonical_hash(measurements), "ranges": output}
    if timing_differences is not None and changed:
        comparison = video_reference_soundtrack_reencode.summarize(timing_differences)
        picture["soundtrack_picture_comparison"] = comparison
        if fine_timing_differences is not None and hd_profile is not None:
            fine = video_reference_soundtrack_reencode.summarize(fine_timing_differences)
            comparison.update(fine_grid=fine, within_bounds=comparison["within_bounds"] and fine["within_bounds"])
            if hd_profile["frames_per_second"] != 30:
                comparison["picture_profile"] = hd_profile
    if local is not None:
        for field, selected in (("range", choice), ("measured_interval", precise)):
            if selected is None:
                continue
            start, end = selected["frames"]
            region = {"id": "L01" if field == "range" else "L01-M", "start_frame": start, "end_frame_exclusive": end,
                      "start_s": float(clock[start]), "end_s": float(clock[end]),
                      "source_ranges": _clock_range(tables, start, end, earlier_offset), "tile": choice["tile"],
                      "previews": {role: {**measurements[index], "time_s": float(clock[index]),
                                          "assets": previews[index], "crop_assets": crops[index]}
                                   for role, index in selected["samples"].items()}}
            if earlier_clock is not None:
                region.update(earlier_start_s=float(earlier_clock[start + earlier_offset]),
                              earlier_end_s=float(earlier_clock[end + earlier_offset]))
            if field == "measured_interval":
                region["basis"] = video_reference_measured_interval.BASIS
            local[field] = region
        picture["local_comparison"] = local
    return picture


def _additional_soundtracks(paths, sources, tables, assets):
    try:
        left, right = map(audio_stream_indices, paths)
    except ValueError:
        return None
    if max(len(left), len(right)) < 2:
        return None
    coverage = {"inventory": {side: {"stream_indices": list(indices),
                                    "unassessed_stream_indices": list(indices[MAX_COMPARED_SOUNDTRACKS:])}
                              for side, indices in (("earlier", left), ("submitted", right))},
                "position": 2, "scope": ADDITIONAL_AUDIO_SCOPE,
                "stream_indices": {"earlier": left[1] if len(left) > 1 else None,
                                   "submitted": right[1] if len(right) > 1 else None}}
    if min(len(left), len(right)) < 2:
        return {**coverage, "status": "missing_counterpart"}
    tracks = [read_track(path, source, max_seconds=MAX_SECONDS, video=True, origin=table.pts[0] * table.tb, stream_position=1)
              for path, source, table in zip(paths, sources, tables, strict=True)]
    changes = compare_tracks(*tracks, assets, preview_prefix="track-2-", difference_waveform=True)
    if any(track[0] == "comparison_limit" for track in tracks):
        changes["status"] = "comparison_limit"
    for span in changes.get("ranges", []):
        span["id"] = "T2-" + span["id"]
    return {**coverage, **changes}


def compare(earlier: Path, submitted: Path, *, include_local: bool = True,
            include_local_area: bool = False,
            include_measured_interval: bool = False,
            timing_event_s: Fraction | None = None, timing_model: Path | None = None,
            timing_short_fallback: bool = False, include_soundtrack: bool = True,
            timing_local_sound: bool = False,
            include_additional_soundtrack: bool = True,
            include_channel_timing: bool = False, include_trimmed: bool = False,
            include_trim_reencoded: bool = False, include_soundtrack_reencode: bool = False,
            include_hd_soundtrack_reencode: bool = False,
            include_quiet_soundtrack: bool = False,
            include_picture_insertion: bool = False,
            verify_picture_insertion_retention: bool = True) -> tuple[dict, dict[str, bytes]]:
    """Report decoded differences on matching clocks or within a verified trimmed overlap."""
    paths = earlier, submitted
    sources = tuple(_identity(path) for path in paths)
    tables = tuple(probe_frames(path, cap=MAX_FRAMES, timeout_s=30, max_pixels=MAX_PIXELS) for path in paths)
    status = _comparison_issue(tables) or "measured"
    assets: dict[str, bytes] = {}
    record = {"schema": SCHEMA, "status": status,
              "earlier": {**sources[0], **_reference(tables[0])}, "submitted": {**sources[1], **_reference(tables[1])},
              "interpretation": INTERPRETATION, "affects_overall_confidence": False}
    if (include_picture_insertion and timing_event_s is None and status in {"picture_timelines_differ", "comparison_limit"}
            and (insertion := video_reference_insertion.locate(
                paths, tables, MAX_PIXELS, verify_retention=verify_picture_insertion_retention)) is not None):
        if insertion["bounds"] is not None:
            record.update(status="measured", interpretation=video_reference_insertion.INTERPRETATION,
                          picture_insertion=_insertion_evidence(paths, tables, assets, insertion))
        else:
            record.update(status=insertion["status"], interpretation=video_reference_insertion.RETENTION_INTERPRETATION)
        if sources != tuple(_identity(path) for path in paths):
            raise ValueError("One of the clips changed while the comparison was running.")
        if verify_picture_insertion_retention:
            record.update(schema=SCHEMA_WITH_INSERTION_RETENTION, insertion_retention=insertion["retention"])
        else:
            record["schema"] = SCHEMA_WITH_PICTURE_INSERTION
        return record, assets
    if include_trimmed and timing_event_s is None and status == "picture_timelines_differ":
        if trim := _trim_offset(paths, tables):
            offset, support = trim
            overlap = len(tables[1])
            record.update(schema=SCHEMA_WITH_TRIM, status="measured",
                          interpretation="This comparison measures decoded picture changes in a verified shared range after trimming. "
                                         "It covers the pictured overlap. Ordinary edits can produce these differences.",
                          trim_alignment={"earlier_frames": [offset, offset + overlap], "submitted_frames": [0, overlap],
                                          "unique_matching_frames": support, "audio_checked": False},
                          picture=_picture(paths, tables, assets, include_local=False, trim_offset=offset))
            return record, assets
        if include_trim_reencoded and video_reference_trimmed.eligible(paths, tables):
            coarse = [video_reference_trimmed.read_pictures(path, table)
                      for path, table in zip(paths, tables, strict=True)]
            votes = video_reference_trimmed.profile(*coarse)
            if alignment := video_reference_trimmed.decide(votes):
                offset = alignment["earlier_start"]
                overlap = len(tables[1])
                picture = _picture(paths, tables, assets, include_local=False, trim_offset=offset, trim_reencoded=True)
                picture["local_comparison"]["scope"] = video_reference_trimmed.SCOPE
                record.update(schema=SCHEMA_WITH_TRIM_REENCODE, status="measured",
                              interpretation=video_reference_trimmed.SCOPE,
                              trim_alignment={"basis": video_reference_trimmed.BASIS,
                                              "earlier_frames": [offset, offset + overlap],
                                              "submitted_frames": [0, overlap], "audio_checked": False,
                                              "motion_support": alignment["flanks"], "motion_margin": alignment["lead"]},
                              picture=picture)
                if sources != tuple(_identity(path) for path in paths):
                    raise ValueError("One of the clips changed while the comparison was running.")
                return record, assets
    tracks = decoded = None
    if status == "measured":
        compare_timing_pictures = (include_soundtrack and include_soundtrack_reencode and timing_event_s is None
                                   and video_reference_soundtrack_reencode.eligible(
                                       paths, tables, allow_hd=include_hd_soundtrack_reencode))
        record["picture"] = _picture(paths, tables, assets, include_local=include_local,
                                     include_local_area=include_local_area and timing_event_s is None,
                                     include_measured_interval=include_measured_interval,
                                     include_soundtrack_reencode=compare_timing_pictures)
        if "local_comparison" in record["picture"]:
            record["schema"] = SCHEMA_WITH_LOCAL
        decoded = [
            _read_track(path, source, max_seconds=MAX_SECONDS, video=True, origin=table.pts[0] * table.tb,
                        include_stereo=include_channel_timing and not record["picture"]["changed_frames"])
            for path, source, table in zip(paths, sources, tables, strict=True)
        ]
        tracks = [item[0] for item in decoded]
        record["audio"] = compare_tracks(*tracks, assets)
    use_marked_sound = False
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
        use_marked_sound = (timing_local_sound and status == "measured"
                            and record["timing"].get("reason") == "profile")
    reencoded_pictures = record.get("picture", {}).get("soundtrack_picture_comparison")
    if (include_soundtrack and tracks is not None and (not record["picture"]["changed_frames"] or reencoded_pictures)
            and record["audio"]["status"] == "measured"):
        from app.evidence import video_reference_soundtrack

        timing_assets = {}
        try:
            if reencoded_pictures:
                alignment = video_reference_soundtrack.assess_reencode(paths, tables, record["picture"],
                                                                       record["audio"], tracks, timing_assets,
                                                                       quiet=include_quiet_soundtrack)
            else:
                alignment = video_reference_soundtrack.assess(submitted, tables[1], record["picture"], record["audio"],
                                                             tracks, timing_assets)
        except DecodeError:
            alignment = {"status": "unassessed", "scope": (video_reference_soundtrack_reencode.scope(record["picture"]) if reencoded_pictures
                                                            else video_reference_soundtrack.SCOPE),
                         "affects_overall_confidence": False,
                         "summary": "The check could not retrieve the verified picture for the matched soundtrack excerpt."}
            if reencoded_pictures:
                alignment["basis"] = video_reference_soundtrack_reencode.BASIS
                alignment.update(video_reference_soundtrack_reencode.report_profile(record["picture"]))
        else:
            assets.update(timing_assets)
        record.update(schema=SCHEMA_WITH_SOUNDTRACK, soundtrack_alignment=alignment)
        if use_marked_sound and alignment["status"] != "local_shift" and record["audio"]["changed_samples"] > 0:
            local_sound = video_reference_timing.assess_local_sound(
                paths, tables, tracks, timing_event_s, timing_model, assets, _image,
            )
            if local_sound is not None:
                record["timing"] = local_sound
    elif reencoded_pictures:
        record["soundtrack_alignment"] = {
            "status": "unassessed", "basis": video_reference_soundtrack_reencode.BASIS,
            "scope": video_reference_soundtrack_reencode.scope(record["picture"]), "affects_overall_confidence": False,
            "summary": "Local soundtrack timing requires matching continuous sound clocks in both clips.",
            **video_reference_soundtrack_reencode.report_profile(record["picture"]),
        }
    if include_additional_soundtrack and status == "measured":
        additional = _additional_soundtracks(paths, sources, tables, assets)
        if additional is not None:
            record.update(schema=SCHEMA_WITH_ADDITIONAL_SOUNDTRACK, additional_soundtrack=additional,
                          interpretation=INTERPRETATION.replace("the first soundtrack", "the available soundtracks checked below"))
    if use_marked_sound:
        record["schema"] = SCHEMA_WITH_MARKED_SOUND_FALLBACK
        if record["timing"]["status"] == "unassessed":
            record["timing"]["summary"] = video_reference_timing.LOCAL_SOUND_UNASSESSED
    if (include_channel_timing and decoded is not None and not record["picture"]["changed_frames"]
            and record["audio"]["status"] == "measured" and record["audio"]["earlier"]["relative_start"] == "0"):
        limit = int(_elapsed(tables[1])[-1] * SAMPLE_RATE)
        if (channels := channel_timing(decoded, assets, sample_limit=limit)) is not None:
            channels["scope"] = (
                "The decoded pictures match. Only excerpts within the verified picture timeline are considered. "
                + channels["scope"]
            )
            channels["stream_indices"] = {
                side: item[3][0]["stream_index"] for side, item in zip(("earlier", "submitted"), decoded, strict=True)
            }
            record.update(schema=SCHEMA_WITH_CHANNEL_TIMING, channel_timing=channels)
    if reencoded_pictures:
        record["schema"] = SCHEMA_WITH_REENCODED_SOUNDTRACK
    if sources != tuple(_identity(path) for path in paths):
        raise ValueError("One of the clips changed while the comparison was running.")
    return record, assets
