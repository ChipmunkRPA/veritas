# SPDX-License-Identifier: Apache-2.0
"""Compare brief source-clock face views at a matched level of spatial detail."""

from __future__ import annotations

import json
import subprocess
import time
from collections import defaultdict
from contextlib import closing
from fractions import Fraction
from hashlib import sha256

import cv2
import numpy as np

from app.detectors.image.faces import detect_faces, iou
from app.detectors.video import face_appearance
from app.detectors.video.decode import DecodeError, _input
from app.evidence import face_matched_detail_binding as binding
from app.evidence import face_short_proposal_binding as short

_Y64, _X64 = np.mgrid[:64, :64]
_TRACE_MASK = ((_X64 - 32) / 22) ** 2 + ((_Y64 - 33) / 23) ** 2 <= 1
_Y112, _X112 = np.mgrid[:112, :112]
_CROP_MASK = ((_X112 - 56) / 33) ** 2 + ((_Y112 - 61) / 39) ** 2 <= 1


def _smooth(values, sigma):
    return cv2.GaussianBlur(values, (0, 0), sigma, borderType=cv2.BORDER_REFLECT_101) if sigma else values


def _detail(gray):
    return float(np.abs(gray - _smooth(gray, 1.35))[_CROP_MASK].mean())


def aligned_rgb(rgb, landmarks):
    return cv2.warpAffine(rgb, face_appearance.cpp_alignment(landmarks), (112, 112), flags=cv2.INTER_LINEAR)


def _match(crops, a, b):
    gray = {i: cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY).astype("float32") for i, crop in crops.items()}
    raw = {i: _detail(values) for i, values in gray.items()}
    target = min(
        float(np.median([raw[i] for i in h])) for h in (range(a - 3, a), range(a, b), range(b, b + 3))
    )
    if target < binding.DETAIL_FLOOR:
        return None
    normalized = {}
    readings = {}
    for index, crop in sorted(crops.items()):
        choices = []
        for sigma in binding.SIGMAS:
            level = _detail(_smooth(gray[index], sigma))
            choices.append((abs(np.log(max(level, 0.001) / target)), sigma, level))
        _, sigma, level = min(choices)
        if not binding.FRACTION * target <= level <= target / binding.FRACTION:
            return None
        normalized[index] = _smooth(crop, sigma)
        readings[index] = {
            "sigma_hex": sigma.hex(),
            "raw_detail_hex": raw[index].hex(),
            "matched_detail_hex": level.hex(),
            "matched_rgb_sha256": sha256(normalized[index]).hexdigest(),
        }
    return normalized, target, readings


def _trace_patch(rgb, box):
    x, y, w, h = box
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    bounded = cv2.getRectSubPix(
        gray, (max(12, round(w * 0.90)), max(12, round(h * 0.90))), (x + w * 0.50, y + h * 0.53)
    )
    crop = cv2.resize(bounded, (64, 64), interpolation=cv2.INTER_AREA).astype("float32")
    return float(np.abs(crop - _smooth(crop, 1.2))[_TRACE_MASK].mean()), _smooth(crop, 2.8)


def _coarse_difference(before, after):
    x, y = before[_TRACE_MASK], after[_TRACE_MASK]
    return float(np.abs((x - x.mean()) - (y - y.mean())).mean())


def _bridge_trace(frames, boxes):
    keys = np.asarray(sorted(boxes))
    coords = np.asarray([boxes[i].box() for i in keys], float)
    at = np.transpose([np.interp(range(binding.FRAME_COUNT), keys, col) for col in coords.T])
    trace = {}
    for i, rgb in frames:
        left = keys[keys <= i]
        right = keys[keys >= i]
        if (
            left.size
            and right.size
            and right[0] - left[-1] <= 20
            and iou(boxes[left[-1]], boxes[right[0]]) >= 0.48
        ):
            trace[i] = _trace_patch(rgb, at[i])
    return trace


def _shortlist(trace, allowed):
    ranked = []
    for a in short.EDGE_SCAN:
        for width in (3, 4):
            b = a + width
            if b not in short.EDGE_SCAN or not (
                allowed(a, b) and all(i in trace for i in range(a - 3, b + 3))
            ):
                continue
            left, center, right = (
                [trace[i][0] for i in group] for group in (range(a - 3, a), range(a, b), range(b, b + 3))
            )
            before, after = float(np.median(left)), float(np.median(right))
            floor = min(before, after)
            if floor < 1:
                continue
            drop = 1 - float(np.median(center)) / floor
            step = min((left[-1] - center[0]) / before, (right[0] - center[-1]) / after)
            if drop < 0.16 or step < 0.08:
                continue
            stable = _coarse_difference(trace[a - 1][1], trace[b][1])
            if stable <= 10:
                ranked.append((-min(step, drop), stable, a, b, drop, step))
    kept = []
    for row in sorted(ranked):
        _, _, a, b, _, _ = row
        if sum(abs(a - v[2]) <= 1 and abs(b - v[3]) <= 1 for v in kept) >= 2:
            continue
        kept.append(row)
        if len(kept) == binding.MAX_PAIRS:
            break
    return kept


def _response(crops, a, b, recognizer):
    values = {
        i: face_appearance.unit(recognizer.feature(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)).reshape(-1))
        for i, rgb in crops.items()
    }
    groups = (range(a - 3, a), range(a, a + 3), range(b - 3, b), range(b, b + 3))
    u, v, w, z = means = [np.median([values[i] for i in indices], axis=0) for indices in groups]
    distance = face_appearance.distance
    near, far, flanks = distance(u, v), distance(w, z), distance(u, z)
    if min(near, far) < 0.08 or flanks > 0.33:
        return None
    entry, exit = [
        jump
        - max(
            distance(values[i], center)
            for indices, center in zip(subgroups, centers, strict=True)
            for i in indices
        )
        for jump, subgroups, centers in ((near, groups[:2], means[:2]), (far, groups[2:], means[2:]))
    ]
    inside = distance(v, w)
    return min(entry, exit, min(near, far) - max(flanks, inside)), (near, far, flanks, inside, entry, exit)


def _view(face):
    return {
        "box": dict(zip(("x", "y", "w", "h"), face.box(), strict=True)),
        "landmarks_hex": [[float(x).hex(), float(y).hex()] for x, y in face.landmarks],
        "geometry": "observed",
    }


def _geometry(faces, a, b):
    flanks = (*range(a - 3, a), *range(b, b + 3))
    if any(i not in faces for i in flanks) or iou(faces[a - 1], faces[b]) < 0.48:
        return None
    rows = {i: _view(faces[i]) for i in flanks}
    for i in range(a, b):
        box, landmarks = binding.interpolate(rows[a - 1], rows[b], i, a, b)
        rows[i] = {"box": box, "landmarks_hex": landmarks, "geometry": "interpolated"}
    return rows


def inspect(inputs, detector, recognizer, deadline, prior, *, profile=binding.PROFILE_30):
    """Return a bounded optional face observation and its measured coverage."""
    table = inputs.table
    cuts = inputs.shot_boundaries
    if cuts is None:
        return None, binding.coverage("unassessed", "shot_context", profile=profile)

    def read(indices):
        expected = set(indices)
        with closing(inputs.store.iter_many(sorted(expected), timeout_s=deadline.remaining())) as frames:
            for i, rgb in frames:
                deadline.check("matched-detail source pictures")
                if i not in expected or not inputs.store.native_pts_verified(i):
                    raise DecodeError("The matched-detail source picture could not be verified")
                expected.remove(i)
                yield i, rgb
        if expected:
            raise DecodeError("The matched-detail source pictures could not be decoded")

    def seek(indexes, faces):
        for i, frame in read(indexes):
            observed = detect_faces(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), detector=detector, min_size=48)
            if len(observed) == 1:
                faces[i] = observed[0]

    def allowed(a, b):
        start, end = table.pts[a], table.pts[b]
        return all(abs(cut - index) > 1 for cut in cuts for index in (a, b)) and not any(
            2 * max(0, min(row["end_pts"], end) - max(row["start_pts"], start))
            >= max(row["end_pts"], end) - min(row["start_pts"], start)
            for row in prior
        )

    anchors = {}
    seek(short.ANCHOR_SAMPLES, anchors)
    if len(anchors) < 4:
        return None, binding.coverage("unassessed", "face_context", profile=profile)
    with closing(read(range(binding.FRAME_COUNT))) as frames:
        trace = _bridge_trace(frames, anchors)
    proposals = _shortlist(trace, allowed)
    flank_indices = {i for _, _, a, b, _, _ in proposals for i in (*range(a - 3, a), *range(b, b + 3))}
    if len(flank_indices | set(short.ANCHOR_SAMPLES)) > binding.MAX_FACE_READS:
        raise ValueError("Matched-detail face reads exceed the fixed schedule")
    seek(flank_indices - set(short.ANCHOR_SAMPLES), anchors)
    geometry, by_index = {}, defaultdict(list)
    for _, _, a, b, _, _ in proposals:
        if (views := _geometry(anchors, a, b)) is not None:
            geometry[a, b] = views
            for i in views:
                by_index[i].append((a, b))

    crops = defaultdict(dict)
    for i, native in read(by_index):
        digest = sha256(native).hexdigest()
        for pair in by_index[i]:
            row = geometry[pair][i]
            crop = aligned_rgb(native, binding.face_landmarks(row))
            row.update(
                index=i,
                pts=table.pts[i],
                decoded_rgb_sha256=digest,
                aligned_rgb_sha256=sha256(crop).hexdigest(),
            )
            crops[pair][i] = crop

    usable = 0
    outcomes = []
    for _, coarse, a, b, drop, step in proposals:
        deadline.check("matched-detail face crops")
        values = crops.get((a, b))
        if not values or (matched := _match(values, a, b)) is None:
            continue
        usable += 1
        normalized, target, frame_readings = matched
        if (result := _response(normalized, a, b, recognizer)) is None:
            continue
        response, readings = result
        if response > profile.point:
            rows = [{**geometry[a, b][i], **frame_readings[i]} for i in range(a - 3, b + 3)]
            outcomes.append(
                (
                    -response,
                    a,
                    b,
                    {
                        "method": profile.method,
                        "time_base": table.time_base,
                        "start_frame_index": a,
                        "end_frame_index": b,
                        "start_pts": table.pts[a],
                        "end_pts": table.pts[b],
                        "response_hex": response.hex(),
                        "readings_hex": dict(zip(binding.READINGS, (x.hex() for x in readings), strict=True)),
                        "proposal_hex": dict(
                            zip(binding.SUPPORT, (x.hex() for x in (drop, step, coarse)), strict=True)
                        ),
                        "target_detail_hex": target.hex(),
                        "views": rows,
                        "scene_cuts": list(cuts),
                    },
                )
            )
    deadline.check("matched-detail evidence")
    record = min(outcomes, key=lambda value: value[:3])[-1] if outcomes else None
    state = "localized" if record else "no_match" if usable else "unassessed"
    return record, binding.coverage(
        state, "face_context" if not usable else None, proposed=len(proposals), usable=usable, profile=profile
    )


def verify_crops(record, native_frames):
    """Repeat source alignment and the exact matching transformations without the scorer."""
    a, b = record["start_frame_index"], record["end_frame_index"]
    originals = {}
    for row in record["views"]:
        i = row["index"]
        pixels = native_frames[i]
        if sha256(pixels).hexdigest() != row["decoded_rgb_sha256"]:
            raise ValueError("Matched-detail frame differs from the original picture")
        crop = aligned_rgb(pixels, binding.face_landmarks(row))
        if sha256(crop).hexdigest() != row["aligned_rgb_sha256"]:
            raise ValueError("Matched-detail crop differs from its source alignment")
        originals[i] = crop
    match = _match(originals, a, b)
    if (
        match is None
        or match[1].hex() != record["target_detail_hex"]
        or any(any(row[k] != v for k, v in match[2][row["index"]].items()) for row in record["views"])
    ):
        raise ValueError("Original crops do not reproduce the matched detail readings")


def _verify_soundtrack(original, span, deadline, allowed_priming):
    from app.evidence.mp4_source_clock import aac_track_schedule

    block = binding.AAC_BLOCK
    try:
        rate, extent, durations = aac_track_schedule(original, span, block, allowed_priming=allowed_priming)
        priming = sum(durations) - extent
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ValueError("The soundtrack clock could not be checked within its time limit")
        read = subprocess.run([
            "ffprobe", "-v", "error", "-protocol_whitelist", "file", "-select_streams", "a",
            "-read_intervals", f"%+#{len(durations) + 1}", "-show_packets", "-show_streams", "-show_entries",
            "stream=index,codec_name,profile,channels,channel_layout,sample_rate,time_base,start_pts,duration_ts:"
            "packet=stream_index,pts,dts,duration:packet_side_data=side_data_type,skip_samples,discard_padding",
            "-of", "json", _input(original),
        ], capture_output=True, check=True, timeout=remaining, stdin=subprocess.DEVNULL)
        parsed = json.loads(read.stdout)
        stream, = parsed["streams"]
        fields = ("codec_name", "profile", "sample_rate", "time_base", "start_pts", "duration_ts")
        channels = stream["channels"]
        packets = parsed["packets"]
        if (tuple(stream[k] for k in fields) != ("aac", "LC", str(rate), f"1/{rate}", 0, extent)
                or (rate, channels, stream["channel_layout"]) not in binding.AAC_LAYOUTS
                or len(packets) != len(durations)):
            raise ValueError("AAC packets differ from the exact soundtrack clock")
        pts = -priming
        for i, (packet, duration) in enumerate(zip(packets, durations, strict=True)):
            packet_clock = tuple(packet[k] for k in ("stream_index", "pts", "dts", "duration"))
            padding = [{"side_data_type": "Skip Samples", "skip_samples": priming, "discard_padding": 0}] if i == 0 else []
            if packet_clock != (stream["index"], pts, pts, duration) or packet.get("side_data_list", []) != padding:
                raise ValueError("AAC packet timing differs from its presentation edit")
            pts += duration
        if pts != extent:
            raise ValueError("AAC packet clock ends at an unexpected sample")
    except (KeyError, OSError, TypeError, ValueError, subprocess.SubprocessError) as error:
        raise ValueError("Original soundtrack differs from the recorded matched-detail profile") from error
    clock = {"rate": rate, "channels": channels, "start_pts": 0, "end_pts": extent, "priming_samples": priming}
    if priming == binding.AAC_EXTENDED_PRIMING:
        clock["trailing_samples"] = block - durations[-1]
    return clock


def verify_profile(original, table, profile, *, timeout_s=4.0, allow_extended_aac=False):
    """Verify the exact source carrier for the recorded review profile."""
    from app.detectors.video.frames import probe as probe_container
    from app.evidence import mp4_source_clock

    strict = profile != binding.PROFILE_30
    deadline = time.monotonic() + timeout_s
    profile_options = {"include_sample_aspect_ratio": True, "timeout_s": timeout_s} if strict else {}
    clock_options = {"video_only": True} if profile == binding.PROFILE_HEVC25 else {}
    info = probe_container(original, include_codec_tag=True, **profile_options)
    encoding = tuple(info.get(key) for key in ("codec", "pix_fmt", "codec_tag"))
    if (
        not binding.eligible(table, encoding, "mp4", profile)
        or info.get("timeline_origin_s") != 0
        or (strict and info.get("sample_aspect_ratio") not in {None, "1:1"})
        or not mp4_source_clock.verified(original, table, profile.step, **clock_options)
    ):
        raise ValueError("Original video differs from the recorded matched-detail profile")
    if profile.audio_priming is not None:
        allowed = (profile.audio_priming,)
        if allow_extended_aac and profile == binding.PROFILE_HEVC25_AAC:
            allowed += (binding.AAC_EXTENDED_PRIMING,)
        return _verify_soundtrack(original, Fraction(binding.FRAME_COUNT * profile.step, profile.timescale), deadline, allowed)
    return None


def verify_coverage_source(original, table, coverage):
    """Replay an explicitly clock-bound scope even when the review selects no range."""
    profile = binding.PROFILES[coverage["method"]]
    if verify_profile(original, table, profile) != coverage.get(binding.SOUNDTRACK_CLOCK):
        raise ValueError("Matched-detail coverage soundtrack clock differs from the original")


def replay_source(original, table, record):
    """Verify the signed interval's MP4 clock, shot context and image transformations."""
    from app.detectors.video.decode import decode_frames, decode_stream
    from app.detectors.video.shots import ShotDetector

    if verify_profile(original, table, binding.PROFILES[record["method"]]) != record.get(binding.SOUNDTRACK_CLOCK):
        raise ValueError("Matched-detail soundtrack clock differs from the original")
    samples = [row["index"] for row in record["views"]]
    pixels = {}
    with closing(decode_frames(original, table, samples)) as frames:
        for frame in frames:
            if not frame.pts_verified or frame.pts != table.pts[frame.index]:
                raise ValueError("Matched-detail view does not have a verified native timestamp")
            pixels[frame.index] = frame.rgb
    if set(pixels) != set(samples):
        raise ValueError("Original video did not reproduce every matched-detail view")
    verify_crops(record, pixels)
    shots = ShotDetector(table)
    count = 0
    with closing(decode_stream(original, table)) as frames:
        for frame in frames:
            if frame.index != count or not frame.pts_verified or frame.pts != table.pts[count]:
                raise ValueError("Matched-detail shot context differs from the original picture clock")
            shots.update(frame.index, frame.rgb)
            count += 1
    if count != len(table) or [row["frame_index"] for row in shots.boundaries] != record["scene_cuts"]:
        raise ValueError("The original video does not reproduce the matched-detail shot context")
