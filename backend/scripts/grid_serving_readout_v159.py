"""Aggregate validated GRID construction and production records without running media."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from fractions import Fraction

from app.core.canonical import canonical_hash
from app.evidence.face_interval_binding import (
    APPEARANCE_METHOD,
    SEQUENCE_METHOD,
    validate_sequence_packet,
    validate_sequence_record,
)

FAMILIES = {
    "original": ("original",),
    "neural_redraw": (
        "musetalk_source_time",
        "musetalk_earlier_320ms",
        "liveportrait_source_time",
        "liveportrait_earlier_320ms",
    ),
    "resample_brightness": (
        "resample_musetalk",
        "resample_liveportrait",
        "brightness_musetalk",
        "brightness_liveportrait",
    ),
    "genuine_frame_retime": ("genuine_frame_retime",),
    "audio_delay": ("audio_delay_320ms",),
}
CONDITIONS = tuple(name for names in FAMILIES.values() for name in names)
FAMILY_OF = {name: family for family, names in FAMILIES.items() for name in names}
APPEARANCE = "vid.review.face_appearance"
TIMING = "vid.lipsync.landmark_xcorr"
RATE = 16_000
TICK_RATE = 25
HOP = RATE // TICK_RATE
MIN_PUBLIC_COMPONENTS = 5
ASSESSED = frozenset({"flagged", "measured_no_flag", "no_stable_offset"})
OUTCOMES = frozenset({"done", "failed", "cancelled", "lease_lost"})
STAGE_STATES = frozenset({"unstarted", "guard_failed", "completed"})
DETECTOR_STATES = {
    "ok": "successful",
    "skipped": "skipped",
    "failed": "failed",
    "timeout": "timed_out",
    "not_applicable": "not_applicable",
}
STAGE_ROUTES = ("provenance", "faces", "tracks", "detect", "second_pass")
ANALYSIS_STAGES = frozenset(
    {
        "ingest",
        "source_labels",
        "probe",
        "provenance",
        "provenance_gate",
        "decode",
        "shots",
        "faces",
        "tracks",
        "detect",
        "temporal",
        "second_pass",
        "condition",
        "fuse",
        "synthesise",
        "render",
        "publish",
    }
)
ANALYSIS_STAGE_STATES = frozenset({"done", "failed", "skipped", "timeout", "running", "pending"})
Span = tuple[Fraction, Fraction]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def decimal(value):
    require(type(value) in {int, float, str}, "A recorded rational number is required")
    return Fraction(str(value))


def merged(spans):
    union = []
    for a, b in sorted(spans):
        require(a < b, "A positive half-open interval is required")
        if union and a <= union[-1][1]:
            union[-1] = (union[-1][0], max(b, union[-1][1]))
        else:
            union.append((a, b))
    return union


def duration(spans):
    return sum((b - a for a, b in spans), Fraction())


def overlap(a, b):
    start, end = max(a[0], b[0]), min(a[1], b[1])
    return (start, end) if start < end else None


def bound_spans(spans, bounds):
    return [part for span in spans if (part := overlap(span, bounds)) is not None]


def interval_metrics(spans, bounds, truth=None):
    """Use observed bounds for predicted support and the full known edit for IoU."""
    union = merged(spans)
    clipped = bound_spans(union, bounds)
    flagged = duration(clipped)
    edit = duration(bound_spans(clipped, truth)) if truth else None
    return {
        "flagged_seconds": flagged,
        "overlap_seconds": edit,
        "outside_edit_seconds": flagged - edit if truth else None,
        "excluded_seconds": duration(union) - flagged,
        "truncated_spans": sum(overlap(span, bounds) != span for span in spans),
        "iou": edit / (flagged + duration([truth]) - edit) if truth else None,
    }


@dataclass
class Endpoint:
    state: str
    branch: str | None = None
    indices: set[int] = field(default_factory=set)
    measured: list[Span] = field(default_factory=list)
    spans: list[Span] = field(default_factory=list)
    windows: int = 0
    shown: int = 0
    shown_flag: int = 0
    bracket: bool = False
    during_changed: bool = False
    counts: Counter = field(default_factory=Counter)


@dataclass
class Clock:
    times: list[Fraction]
    rgb: list[str]
    table: dict
    sampled: dict
    file_sha: str

    @property
    def bounds(self):
        return self.times[0], self.times[-1]

    @property
    def source(self):
        return {key: self.table[key] for key in ("time_base", "width", "height")} | {
            "frame_count": len(self.times)
        }

    def native(self, index, pts, rgb):
        return (
            type(index) is int
            and 0 <= index < len(self.times)
            and self.table["pts"][index] == pts
            and self.rgb[index] == rgb
        )


def native_clock(case, row):
    proof = case["expected"]
    sampled = row.get("sampled_input") or {}
    if not all(sampled.get(key) is True for key in ("clock_exact", "sampled_rgb_exact")):
        return None
    table = sampled["probe_frame_table"]
    expected = [Fraction(ms, 1000) for ms in proof["picture_pts_ms"]]
    actual = [p * decimal(table["time_base"]) for p in table["pts"]]
    native = [p * decimal(case["source_video_time_base"]) for p in case["source_video_pts"]]
    require(
        actual == expected == native
        and len(actual) == len(proof["rgb_sha256"]) == 75
        and sorted(set(actual)) == actual
        and not table["truncated"]
        and not table["interpolated"],
        "Serving frame table differs from observed source clocks",
    )
    clock = Clock(actual, proof["rgb_sha256"], table, sampled, case["file_sha256"])
    require(
        sampled["sampled_frames"]
        and all(
            clock.native(frame["frame_index"], frame["pts"], frame["sha256"])
            for frame in sampled["sampled_frames"]
        ),
        "Sampled frame inputs differ from constructed pixels",
    )
    return clock


def stage_detector(detect, detector):
    found = [entry for entry in detect.get("detectors", []) if entry["detector_id"] == detector]
    require(len(found) <= 1, "Duplicate production detector entry")
    return found[0] if found else None


def initial_endpoint(entry):
    if entry is None:
        return Endpoint("unobserved")
    return Endpoint(
        {
            "ok": "no_support",
            "failed": "detector_failed",
            "timeout": "timed_out",
            "skipped": "skipped",
            "not_applicable": "not_applicable",
        }.get(entry["status"], "unobserved")
    )


def appearance_endpoint(entry, clock, evidence, changed, truth):
    result = initial_endpoint(entry)
    if entry is None:
        return result
    details = entry.get("details") or {}
    route = details.get("appearance_check_status")
    method = {"ok": APPEARANCE_METHOD, "not_applicable": SEQUENCE_METHOD}.get(route)
    if route == "not_applicable":
        result.branch = "fallback"
    elif route == "ok":
        result.branch = "appearance"
    if entry["status"] != "ok":
        return result
    windows = details.get("face_sequence_windows") or []
    selected = details.get("face_sequence")
    if clock is None or details.get("face_sequence_binding_version") != method or not method:
        result.state = "unverified_input" if clock is None else "unbound_evidence"
        result.counts["unbound_selected"] += int(selected is not None)
        return result
    try:
        require(selected is None or selected in windows, "Selected face range lacks measured coverage")
        for window in windows:
            require(window["method"] == method, "Returned face method differs from production route")
            validate_sequence_record(
                window, clock.source, table_pts=clock.table["pts"], require_flag=window == selected
            )
            for item in window["frame_inputs"]:
                index = item["frame_index"]
                require(
                    clock.native(index, item["pts"], item["decoded_rgb_sha256"]),
                    "Face sequence source pixels differ",
                )
                result.indices.add(index)
            tb = decimal(window["time_base"])
            span = window["start_pts"] * tb, window["end_pts"] * tb
            if window == selected:
                result.spans = [span]
    except (KeyError, TypeError, ValueError, IndexError, ZeroDivisionError):
        return Endpoint(
            "unbound_evidence", result.branch, counts=Counter(unbound_selected=int(selected is not None))
        )
    result.windows = len(windows)
    if not windows:
        return result
    result.state = "flagged" if selected is not None else "measured_no_flag"
    if selected is None:
        return result

    for packet in evidence.get("packets") or []:
        if packet.get("face_sequence") is None or canonical_hash(packet["face_sequence"]) != canonical_hash(
            selected
        ):
            continue
        try:
            validate_sequence_packet(packet, clock.source, table_pts=clock.table["pts"])
            frames = [frame["anchor"] for frame in packet["frames"]]
            require(
                all(
                    frame["original_sha256"] == clock.file_sha
                    and frame["time_base"] == clock.table["time_base"]
                    and clock.native(frame["frame_index"], frame["pts"], frame["decoded_rgb_sha256"])
                    for frame in frames
                ),
                "Delivered face anchors differ from the measured source",
            )
        except (KeyError, TypeError, ValueError, IndexError):
            result.counts["unbound_packets"] += 1
            continue
        before, during, after = (frame["frame_index"] for frame in frames)
        result.shown = result.shown_flag = 1
        result.bracket = truth is not None and (
            clock.times[before] < truth[0] <= clock.times[during] < truth[1] <= clock.times[after]
        )
        result.during_changed = during in changed
        break
    return result


def audio_clock(case, clock, proof):
    sampled = clock.sampled
    require(
        all(
            sampled.get(key) is True
            for key in ("audio_samples_equal", "audio_origin_equal", "audio_pcm_equal")
        ),
        "Serving audio comparison is unverified",
    )
    expected = case["expected"]
    audio = sampled["serving_audio"]
    mapping = audio["source_mapping"]
    origin = decimal(expected["audio_start_seconds"])
    samples = expected["audio_samples"]
    require(
        type(samples) is int
        and samples > 0
        and audio["samples"] == mapping["samples"] == samples
        and mapping["timeline_time_base"] == "1/16000"
        and mapping["sample_rate"] == RATE
        and mapping["source_sha256"] == clock.file_sha,
        "Serving audio metadata differs from construction",
    )
    position = 0
    for block in mapping["timeline"]:
        end = block["end_sample"]
        require(
            block["start_sample"] == position < end <= samples
            and Fraction(block["source_pts"], RATE) == origin + Fraction(position, RATE),
            "A contiguous decoded audio timeline is required",
        )
        position = end
    require(position == samples, "Audio source coverage is incomplete")
    bound = {
        "source_sha256": clock.file_sha,
        "pcm_sha256": mapping["pcm_sha256"],
        "mapping_sha256": canonical_hash(mapping),
        "frame_table_sha256": canonical_hash(clock.table),
        "frame_time_base": clock.table["time_base"],
        "samples": samples,
        "sample_rate": RATE,
        "audio_time_base": "1/16000",
        "tick_rate": TICK_RATE,
        "samples_per_tick": HOP,
        "lag_search_ticks": [-15, 15],
    }
    require(
        all(proof.get(key) == value for key, value in bound.items())
        and Fraction(proof["audio_origin_pts"], RATE) == origin,
        "Raw timing proof differs from served RGB/PCM clocks",
    )
    return origin, origin + Fraction(samples, RATE)


def timing_segments(track, origin, measured):
    segments = []
    for segment in track["segments"]:
        kind = segment["kind"]
        start, end = (origin + decimal(segment[key]) / TICK_RATE for key in ("start_tick", "end_tick"))
        require(
            kind in {"offset_step", "confidence_collapse"}
            and start < end
            and any(overlap((start, end), window) for window in measured),
            "Raw timing segment lacks compatible measured support",
        )
        segments.append((kind, (start, end)))
    return segments


def timing_windows(track, clock, origin, samples, entry):
    ticks, windows = {}, {}
    for frame in track["frame_inputs"]:
        tick, index = frame["tick"], frame["frame_index"]
        target = origin + Fraction(tick, TICK_RATE)
        require(
            type(tick) is int
            and 0 <= tick < samples // HOP
            and tick not in ticks
            and clock.native(index, frame["frame_pts"], frame["decoded_rgb_sha256"])
            and frame["source_audio_time_base"] == "1/16000"
            and Fraction(frame["source_audio_pts"], RATE) == target
            and abs(clock.times[index] - target) <= min(abs(pts - target) for pts in clock.times)
            and abs(clock.times[index] - target) <= Fraction(1, TICK_RATE),
            "Measured timing frame differs from the native picture or sample clock",
        )
        ticks[tick] = frame
    units = {unit["unit_ref"]: unit for unit in entry.get("unit_scores", [])}
    for window in track["windows"]:
        start, end = window["start_tick"], window["end_tick"]
        unit = units.get(window["unit_ref"]) or {}
        quality = unit.get("quality") or {}
        interval = {"start_sample": start * HOP, "end_sample": end * HOP}
        span = origin + Fraction(start, TICK_RATE), origin + Fraction(end, TICK_RATE)
        require(
            type(start) is int
            and type(end) is int
            and end - start == TICK_RATE
            and 15 <= start < end <= samples // HOP - 15
            and all(tick in ticks for tick in range(start, end))
            and window["audio_interval"] == interval
            and window["unit_ref"] not in windows
            and unit.get("raw_kind") == "score"
            and abs(decimal(unit["raw"]) - decimal(window["peak_correlation"])) <= Fraction(2, 1_000_000)
            and quality.get("offset_frames") == window["offset_ticks"]
            and all(
                abs(decimal(quality[key]) - actual) <= Fraction(2, 1_000_000)
                for key, actual in zip(("start_s", "end_s"), span, strict=True)
            ),
            "Raw timing window differs from the actual measurement or its support",
        )
        windows[window["unit_ref"]] = (window, span, [ticks[tick] for tick in range(start, end)])
    require(
        set(ticks) == {frame["tick"] for _, _, frames in windows.values() for frame in frames},
        "Timing ticks include positions that did not contribute to a measured window",
    )
    return windows


def timing_exhibits(entry, detect, clock, segments):
    expected = Counter((track, round(float(a), 6), round(float(b), 6)) for track, _, (a, b) in segments)
    actual = Counter()
    for exhibit in detect.get("exhibits") or []:
        if exhibit.get("kind") != "segments" or exhibit.get("id") not in entry.get("exhibit_ids", []):
            continue
        require(exhibit["created_by"] == TIMING and exhibit["lane"] == "lip_sync", "Timing exhibit differs")
        for row in exhibit["segments"]:
            require(
                row["channel"] == "av"
                and row["time_base"] == clock.table["time_base"]
                and row["start_pts"] in clock.table["pts"]
                and row["end_pts"] in clock.table["pts"]
                and row["start_pts"] <= row["end_pts"],
                "Raw timing exhibit lacks valid native anchors",
            )
            actual[row["track_id"], row["start_s"], row["end_s"]] += 1
    require(actual == expected, "Raw timing exhibits differ from detector segments")


def timing_packets(evidence, clock, proof, entry, windows, result):
    raw = canonical_hash(
        entry
        | {
            "run_id": None,
            "name": "",
            "family": "",
            "modality": "image",
            "hypotheses": [],
            "unit_scores": entry.get("unit_scores", [])[:500],
        }
    )
    seen = set()
    for packet in (evidence.get("av") or {}).get("packets") or []:
        ref = packet.get("source_record") or {}
        key = ref.get("track_id"), ref.get("unit_ref")
        if key not in windows:
            continue
        if ref.get("raw_output_sha256") != raw or key in seen:
            result.counts["unbound_packets"] += 1
            continue
        window, _, frames = windows[key]
        shown = packet["window"]
        anchor = packet["anchor"]
        measurement = packet["measurement"]
        flags = ("overlaps_offset_step", "overlaps_correlation_collapse")
        start, end = (window["audio_interval"][key] for key in ("start_sample", "end_sample"))
        origin = proof["audio_origin_pts"]
        source_ranges = []
        for block in clock.sampled["serving_audio"]["source_mapping"]["timeline"]:
            a, b = max(start, block["start_sample"]), min(end, block["end_sample"])
            if a < b:
                source_ranges.append(
                    {
                        "start_sample": a,
                        "end_sample": b,
                        "source_start_pts": origin + a,
                        "source_end_pts": origin + b,
                        "time_base": "1/16000",
                    }
                )
        if not (
            packet["semantics"] == "mouth_audio_correlation"
            and anchor["original_sha256"] == clock.file_sha
            and anchor["pcm_sha256"] == proof["pcm_sha256"]
            and anchor["sample_rate"] == RATE
            and anchor["source_timestamp_ranges"] == source_ranges
            and packet["interval"] == window["audio_interval"]
            and all(shown[key] == window[key] for key in ("start_tick", "end_tick"))
            and shown["frame_indices"] == [row["frame_index"] for row in frames]
            and canonical_hash(
                [{k: v for k, v in row.items() if k != "artifact_ref"} for row in shown["frame_inputs"]]
            )
            == canonical_hash(frames)
            and all(measurement[k] is window[k] for k in flags)
        ):
            result.counts["unbound_packets"] += 1
            continue
        seen.add(key)
        result.shown += 1
        result.shown_flag += any(window[k] for k in flags)


def timing_endpoint(entry, detect, case, clock, evidence):
    result = initial_endpoint(entry)
    if entry is None or entry["status"] != "ok":
        return result, None
    metrics = entry.get("metrics") or {}
    proof = (entry.get("details") or {}).get("av_input") or {}
    raw = sum(len(t.get("segments", [])) for t in proof.get("tracks", []))
    reported = sum(metrics.get(k, 0) for k in ("offset_steps", "correlation_collapses"))
    unbound = Counter(unbound_segments=max(raw, int(reported)))
    if clock is None or not all(
        clock.sampled.get(key) is True
        for key in ("audio_samples_equal", "audio_origin_equal", "audio_pcm_equal")
    ):
        return Endpoint("unverified_input", counts=unbound), None
    if (proof.get("version"), proof.get("status"), proof.get("method")) != (
        "av-correlation-input/1",
        "ready",
        "mouth_rms_correlation",
    ):
        state = "unbound_evidence" if raw or reported or metrics.get("windows", 0) else "no_support"
        return Endpoint(state, counts=unbound), None

    try:
        bounds = audio_clock(case, clock, proof)
        origin = bounds[0]
        comparison = overlap(bounds, clock.bounds)
        require(comparison is not None, "Decoded audio and observed pictures lack common support")
        windows, segments, readings = {}, [], {}
        for track in proof["tracks"]:
            tid = track["track_id"]
            require(tid not in readings, "Duplicate raw timing track")
            per_track = timing_windows(track, clock, origin, proof["samples"], entry)
            measured = [item[1] for item in per_track.values()]
            require(measured, "Attributed track lacks measured windows")
            track_segments = timing_segments(track, origin, measured)
            for ref, item in per_track.items():
                window, span, _ = item
                for flag, kind in (
                    ("overlaps_offset_step", "offset_step"),
                    ("overlaps_correlation_collapse", "confidence_collapse"),
                ):
                    require(
                        window[flag]
                        is any(
                            kind == name and overlap(span, edit) is not None for name, edit in track_segments
                        ),
                        "Raw timing window overlap marker differs from its segments",
                    )
                windows[tid, ref] = item
            segments.extend((tid, kind, span) for kind, span in track_segments)
            readings[tid] = track["reading"]
        counts = Counter(kind for _, kind, _ in segments)
        require(
            metrics.get("windows", 0) == len(windows)
            and metrics.get("speaking_tracks", 0) == len(readings)
            and metrics.get("offset_steps", 0) == counts["offset_step"]
            and metrics.get("correlation_collapses", 0) == counts["confidence_collapse"],
            "Raw timing counts differ from recorded detector metrics",
        )
        timing_exhibits(entry, detect, clock, segments)
    except (KeyError, TypeError, ValueError, IndexError, ZeroDivisionError):
        return Endpoint("unbound_evidence", counts=unbound), None

    if not windows:
        return result, comparison
    result.indices = {frame["frame_index"] for _, _, frames in windows.values() for frame in frames}
    result.measured = [span for _, span, _ in windows.values()]
    result.spans = [span for _, _, span in segments]
    result.counts = counts
    result.counts["tracks_without_stable_offset"] = sum(
        reading == "no_stable_offset" for reading in readings.values()
    )
    result.windows = len(windows)
    result.state = (
        "flagged"
        if segments
        else "no_stable_offset"
        if all(reading == "no_stable_offset" for reading in readings.values())
        else "measured_no_flag"
    )
    try:
        timing_packets(evidence, clock, proof, entry, windows, result)
    except (KeyError, TypeError, ValueError, IndexError):
        result.counts["unbound_packets"] += 1
    return result, comparison


def construction_truth(spec, case):
    expected = case["expected"]
    times = [Fraction(ms, 1000) for ms in expected["picture_pts_ms"]]
    proof = case["edit_proofs"]
    require(
        len(times) == 75 and [item["frame_index"] for item in proof] == spec["output_picture_indices"],
        "Construction must preserve the exact planned mask roster",
    )
    for item in proof:
        require(
            item["pts"] == case["source_video_pts"][item["frame_index"]]
            and type(item["changed_pixels"]) is int
            and item["changed_pixels"] >= 0,
            "Observed changed pixels require valid source-frame commitments",
        )
    changed = {item["frame_index"] for item in proof if item["changed_pixels"] > 0}
    require(
        case["changed_native_pixels"] == sum(item["changed_pixels"] for item in proof)
        and case["changed_video_frames"] == len(changed)
        and case["neural_provenance_verified"] is spec["planned_neural_provenance"]
        and case["observed_neural_pixel_change"] == (bool(changed) and spec["planned_neural_provenance"]),
        "Construction change counts and neural provenance must reconcile",
    )
    visual = None
    bounds = spec["edit_frames_half_open"]
    if bounds is not None:
        visual = tuple(times[index] for index in bounds)
        require(
            visual == tuple(map(decimal, case["source_time_bounds_seconds"])),
            "Construction edit bounds differ from observed source PTS",
        )
    audio = None
    delayed = spec["waveform_id"] == "delayed_audio_control"
    changes = [
        row
        for row in case["audio_value_mapping"]
        if row["output_samples_half_open"] != row["input_samples_half_open"]
    ]
    require(
        len(changes) == int(delayed)
        and type(case["changed_audio_samples"]) is int
        and 0 <= case["changed_audio_samples"] <= expected["audio_samples"]
        and (delayed or case["changed_audio_samples"] == 0),
        "Construction soundtrack change counts require exact copied support",
    )
    if delayed:
        change = changes[0]
        a, b = change["output_samples_half_open"]
        c, d = change["input_samples_half_open"]
        require(
            0 <= c < d <= expected["audio_samples"]
            and 0 <= a < b <= expected["audio_samples"]
            and a - c == b - d == 5120
            and case["changed_audio_samples"] <= b - a
            and change["time_base"] == "1/16000",
            "The audio delay requires its unchanged 320 ms value offset",
        )
        if case["changed_audio_samples"]:
            origin = decimal(expected["audio_start_seconds"])
            audio = (origin + Fraction(a, RATE), origin + Fraction(b, RATE))
    return changed, visual if changed else None, audio


def customer_outcome(row, complete):
    customer = row.get("customer_assessment") or {}
    actual = customer.get("direction")
    valid = actual in {"ai", "original", "inconclusive", "unavailable", "record"}
    direction = actual if complete and valid else "unavailable"
    if direction in {"ai", "original"} and not (
        customer.get("directional_conclusion") is True and customer.get("content_assessed") is True
    ):
        direction = "unavailable"
    counts = Counter({direction: 1})
    scope = customer.get("scope")
    if complete and scope in {"visual_content", "assessed_components", "whole_file"}:
        counts[f"scope_{scope}"] += 1
        for key in ("voice", "audio_picture_timing"):
            if key in (customer.get("excluded_channels") or []):
                counts[f"excluded_{key}"] += 1
    soundtrack = (customer.get("soundtrack") or {}).get("direction")
    if complete and soundtrack in {"ai", "original", "inconclusive", "unavailable"}:
        counts[f"separate_soundtrack_{soundtrack}"] += 1
    if valid:
        counts[f"engine_returned_{actual}"] += 1
    counts["assessed_content"] += int(complete and customer.get("content_assessed") is True)
    return direction, counts


def detector_coverage(row):
    counts = Counter()
    attempted_ids = {run["id"] for run in row.get("detector_runs", [])}
    for item in row.get("detector_inventory", []):
        stage = item["expected_stage"]
        if stage not in STAGE_ROUTES:
            counts["profile_not_scheduled"] += 1
            continue
        counts["expected"] += 1
        counts[f"{stage}_expected"] += 1
        if not item["runs"]:
            counts["unobserved"] += 1
            counts[f"{stage}_unobserved"] += 1
        for run in item["runs"]:
            status = DETECTOR_STATES.get(run["status"], "unobserved")
            counts[status] += 1
            counts[f"{stage}_{status}"] += 1
            if run.get("run_id") in attempted_ids:
                counts["attempted"] += 1
                counts[f"{stage}_attempted"] += 1
            if status == "skipped" and run.get("reason") in {"model not installed", "memory budget"}:
                counts["model_unavailable"] += 1
    return counts


def sampling_coverage(clock):
    counts = Counter(native_pictures=len(clock.times))
    sampled = clock.sampled
    plan = (sampled.get("probe_plan") or {}).get("legacy") or {}
    planned = set(plan.get("frames") or [])
    frames = sampled["sampled_frames"]
    decoded = {frame["frame_index"] for frame in frames}
    faces = sampled.get("sampled_face_counts")
    counts.update(legacy_planned=len(planned), legacy_decoded=len(planned & decoded))
    if isinstance(faces, list) and len(faces) == len(frames):
        face_indices = {
            frame["frame_index"]
            for frame, count in zip(frames, faces, strict=True)
            if type(count) is int and count > 0
        }
        counts.update(
            legacy_face_measured=len(planned & decoded), legacy_face_detected=len(planned & face_indices)
        )
    tracks = {
        frame["frame_index"]
        for exhibit in sampled.get("track_exhibits") or []
        for frame in exhibit["frames"]
        if type(frame["frame_index"]) is int
        and 0 <= frame["frame_index"] < len(clock.times)
        and clock.table["pts"][frame["frame_index"]] == frame["pts"]
    }
    counts["tracked_native_pictures"] = len(tracks)
    return counts


def grouped_rows(plan, construction, serving, states):
    sources = plan["sources"]
    pilots = {source["ordinal"]: source for source in sources if source["role"] == "pilot"}
    require(
        len(pilots) == 16
        and len(pilots) == sum(source["role"] == "pilot" for source in sources)
        and (construction.keys() | serving.keys() | states.keys()) <= pilots.keys()
        and all(value in STAGE_STATES for value in states.values()),
        "Only the complete fixed sixteen-source pilot may enter the readout",
    )
    for ordinal, source in pilots.items():
        specs = {spec["id"]: spec for spec in source["cases"]}
        require(
            len(specs) == len(source["cases"]) == len(CONDITIONS) and specs.keys() == set(CONDITIONS),
            "Exact fixed plan condition roster required",
        )
        built_rows = construction.get(ordinal, [])
        built = {row["condition"]: row for row in built_rows}
        served_rows = serving.get(ordinal, [])
        served = {row["condition"]: row for row in served_rows}
        state = states.get(ordinal, "completed" if ordinal in serving else "unstarted")
        require(
            len(built) == len(built_rows)
            and (ordinal not in construction or built.keys() == specs.keys())
            and len(served) == len(served_rows)
            and served.keys() <= built.keys()
            and (not served or state == "completed")
            and len({row["case_number"] for row in built_rows}) == len(built_rows),
            "Construction and serving roster do not match the fixed pilot",
        )
        for name, spec in specs.items():
            case, row = built.get(name), served.get(name)
            if case is not None:
                binding = {
                    "asset_index": ordinal,
                    "source_slot": source["source_slot"],
                    "component_index": source["component_index"],
                    "condition": name,
                }
                require(
                    all(case[key] == value for key, value in binding.items())
                    and case["study_role"] == "pilot"
                    and case["status"] in {"ready", "unavailable"}
                    and all(
                        case[key] == spec[key]
                        for key in ("planned_neural_provenance", "waveform_id", "edit_frames_half_open")
                    ),
                    "Case identity or planned intervention differs",
                )
                if row is not None:
                    require(
                        all(row[key] == value for key, value in binding.items())
                        and row["case_number"] == case["case_number"]
                        and (
                            row["status"] == "unavailable_input"
                            if case["status"] == "unavailable"
                            else row["status"] in OUTCOMES and row["input_file_sha256"] == case["file_sha256"]
                        ),
                        "Serving case was not joined to its actual constructed input",
                    )
            yield source["component_index"], spec, case, row, built.get("original"), state


def endpoint_metrics(endpoint, bounds, truth, changed):
    assessed = endpoint.state in ASSESSED
    info = {
        "state": endpoint.state,
        "assessed": assessed,
        "flagged": endpoint.state == "flagged",
        "delivered_flag": bool(endpoint.shown_flag),
        "windows": endpoint.windows,
        "displayed_windows": endpoint.shown,
        "displayed_flag_windows": endpoint.shown_flag,
        "omitted_windows": endpoint.windows - endpoint.shown,
        "verified_picture_inputs": len(endpoint.indices),
        "verified_changed_picture_inputs": len(endpoint.indices & changed),
        "observable_changed_pictures": len(changed),
        "measured_seconds": duration(bound_spans(merged(endpoint.measured), bounds)) if assessed else None,
    }
    if assessed:
        info.update(interval_metrics(endpoint.spans, bounds, truth))
    return info


def case_metrics(spec, case, row, original, stage):
    empty = Counter()
    planned_edits = {
        "appearance_planned_edit": bool(spec["output_picture_indices"]),
        "timing_planned_edit": spec["waveform_id"] == "delayed_audio_control",
    }
    if case is None or case["status"] != "ready":
        state = "construction_unobserved" if case is None else "unavailable_input"
        return {
            **planned_edits,
            "state": state,
            "effect": "unavailable",
            "customer": "unavailable",
            "customer_counts": Counter(unavailable=1),
            "appearance": {"state": state},
            "timing": {"state": state},
            "coverage": empty,
            "detectors": empty,
            "stages": empty,
        }

    changed, visual_truth, audio_truth = construction_truth(spec, case)
    effect = (
        "original"
        if spec["id"] == "original"
        else ("observable" if changed or audio_truth is not None else "zero_effect")
    )
    matched = original is not None and original["status"] == "ready"
    unchanged_pictures = matched and case["expected"]["rgb_sha256"] == original["expected"]["rgb_sha256"]
    unchanged_audio = matched and (
        all(
            case["expected"][key] == original["expected"][key]
            for key in ("audio_start_seconds", "audio_samples", "pcm_s16_sha256")
        )
    )
    complete = bool(
        row and row["status"] == "done" and row["analysis_status"] == "done" and row["result_hash"]
    )
    state = row["status"] if row is not None else "missing_case" if stage == "completed" else stage
    direction, customer = customer_outcome(row or {}, complete)
    normal_control = not spec["planned_neural_provenance"]
    result = {
        **planned_edits,
        "state": state,
        "effect": effect,
        "constructed": True,
        "complete": complete,
        "attempted": row is not None,
        "customer": direction,
        "customer_counts": customer,
        "neural_observable": case["observed_neural_pixel_change"],
        "non_neural": normal_control,
        "false_ai_attribution": normal_control and direction == "ai",
        "appearance_observable_edit": visual_truth is not None,
        "timing_observable_edit": audio_truth is not None,
        "unchanged_pictures": unchanged_pictures,
        "unchanged_audio": unchanged_audio,
        "coverage": empty,
        "detectors": empty,
        "stages": empty,
        "appearance": {"state": state, "observable_changed_pictures": len(changed)},
        "timing": {"state": state, "observable_changed_pictures": len(changed)},
    }
    if row is None:
        return result

    result["detectors"] = detector_coverage(row)
    result["stages"] = Counter(
        f"{item['stage']}_{item['status']}"
        for item in row.get("stages", [])
        if item["stage"] in ANALYSIS_STAGES and item["status"] in ANALYSIS_STAGE_STATES
    )
    clock = native_clock(case, row)
    if clock is not None:
        result["coverage"] = sampling_coverage(clock)
        result["visual_input_verified"] = True
    detect = (row.get("detector_stage_outputs") or {}).get("detect") or {}
    evidence = (row.get("evidence") or {}) if complete else {}
    appearance = appearance_endpoint(
        stage_detector(detect, APPEARANCE), clock, evidence, changed, visual_truth
    )
    timing, av_bounds = timing_endpoint(stage_detector(detect, TIMING), detect, case, clock, evidence)
    result["appearance"] = endpoint_metrics(
        appearance, clock.bounds if clock else None, visual_truth, changed
    )
    result["appearance"].update(
        branch=appearance.branch,
        bracketed_edit=appearance.bracket,
        displayed_changed_picture=appearance.during_changed,
        extra_counts=appearance.counts,
    )
    result["timing"] = endpoint_metrics(timing, av_bounds, audio_truth, changed)
    result["timing"]["extra_counts"] = timing.counts
    result["timing_input_verified"] = clock is not None and all(
        clock.sampled.get(key) is True
        for key in ("audio_samples_equal", "audio_origin_equal", "audio_pcm_equal")
    )
    result["raw_timing_binding_verified"] = av_bounds is not None
    if visual_truth and timing.state in ASSESSED:
        result["timing"]["descriptive_visual_iou"] = interval_metrics(timing.spans, av_bounds, visual_truth)[
            "iou"
        ]
    return result


def ratio(numerator, denominator):
    return {
        "numerator": numerator,
        "denominator": denominator,
        "rate": str(Fraction(numerator, denominator)) if denominator else None,
    }


def component_rate(rows, test, eligible):
    grouped = defaultdict(list)
    for component, row in rows:
        if eligible(row):
            grouped[component].append(bool(test(row)))
    rates = [Fraction(sum(values), len(values)) for values in grouped.values()]
    return {
        **ratio(sum(sum(values) for values in grouped.values()), sum(map(len, grouped.values()))),
        "contributing_components": len(rates),
        "component_weighted_rate": str(sum(rates, Fraction()) / len(rates))
        if len(rates) >= MIN_PUBLIC_COMPONENTS
        else None,
    }


def metrics_summary(rows, name):
    values = [row[name] for _, row in rows]
    assessed = [value for value in values if value.get("assessed")]
    counts = sum((value.get("extra_counts", Counter()) for value in values), Counter())
    result = {
        "states": dict(Counter(value["state"] for value in values)),
        "assessed": component_rate(rows, lambda row: row[name].get("assessed"), lambda row: True),
        "flagged": component_rate(
            rows, lambda row: row[name].get("flagged"), lambda row: row[name].get("assessed")
        ),
        "flag_delivered": component_rate(
            rows, lambda row: row[name].get("delivered_flag"), lambda row: row[name].get("flagged")
        ),
        "counts": dict(counts),
        "changed_picture_input_coverage": ratio(
            sum(value.get("verified_changed_picture_inputs", 0) for value in values),
            sum(value.get("observable_changed_pictures", 0) for value in values),
        ),
    }
    for key in (
        "windows",
        "displayed_windows",
        "displayed_flag_windows",
        "omitted_windows",
        "verified_picture_inputs",
        "truncated_spans",
    ):
        result[key] = sum(value.get(key, 0) for value in values)
    quantities = (
        "flagged_seconds",
        "overlap_seconds",
        "outside_edit_seconds",
        "excluded_seconds",
        "iou",
    )
    if name == "timing":
        quantities += ("measured_seconds", "descriptive_visual_iou")
    for key in quantities:
        by_component = defaultdict(list)
        for component, row in rows:
            value = row[name].get(key)
            if value is not None:
                by_component[component].append(value)
        measurements = [value for group in by_component.values() for value in group]
        result[key] = {
            "sum": str(sum(measurements, Fraction())) if len(by_component) >= MIN_PUBLIC_COMPONENTS else None,
            "denominator": len(measurements),
            "contributing_components": len(by_component),
            "component_weighted_mean": str(
                sum((sum(group, Fraction()) / len(group) for group in by_component.values()), Fraction())
                / len(by_component)
            )
            if len(by_component) >= MIN_PUBLIC_COMPONENTS
            else None,
        }
    hit = lambda row: (row[name].get("iou") or Fraction()) >= Fraction(1, 5)  # noqa: E731
    has_truth = lambda row: row[name].get("iou") is not None  # noqa: E731
    result["localized_assessed_iou_at_least_one_fifth"] = component_rate(rows, hit, has_truth)
    result["localized_planned_iou_at_least_one_fifth"] = component_rate(
        rows, hit, lambda row: row[f"{name}_planned_edit"]
    )
    result["observable_edits_assessed"] = component_rate(
        rows, has_truth, lambda row: row.get(f"{name}_observable_edit")
    )
    flag = lambda row: row[name].get("flagged")  # noqa: E731
    assessed_case = lambda row: row[name].get("assessed")  # noqa: E731
    result["ordinary_control_flags"] = component_rate(
        rows, flag, lambda row: row.get("non_neural") and row["effect"] != "original" and assessed_case(row)
    )
    result["unchanged_stream_flags"] = component_rate(
        rows,
        flag,
        lambda row: (
            row.get("unchanged_pictures")
            and (name == "appearance" or row.get("unchanged_audio"))
            and assessed_case(row)
        ),
    )
    if name == "appearance":
        for branch in ("appearance", "fallback"):
            selected = [value for value in values if value.get("branch") == branch]
            result[branch] = {
                "returned": len(selected),
                "assessed": sum(value.get("assessed", False) for value in selected),
                "flagged": sum(value.get("flagged", False) for value in selected),
            }
        result["displayed_changed_picture"] = sum(
            value.get("displayed_changed_picture", False) for value in assessed
        )
        result["bracketed_known_edit"] = sum(value.get("bracketed_edit", False) for value in assessed)
    return result


def summary(rows):
    values = [row for _, row in rows]
    counts = lambda key: dict(sum((row[key] for row in values), Counter()))  # noqa: E731
    complete = lambda row: row.get("complete")  # noqa: E731
    customer = {
        "counts": counts("customer_counts"),
        "directional": component_rate(rows, lambda row: row["customer"] in {"ai", "original"}, complete),
        "false_ai_attribution": component_rate(
            rows,
            lambda row: row.get("false_ai_attribution"),
            lambda row: row.get("non_neural") and complete(row),
        ),
        "false_ai_attribution_directional_only": component_rate(
            rows,
            lambda row: row.get("false_ai_attribution"),
            lambda row: row.get("non_neural") and row["customer"] in {"ai", "original"},
        ),
        "observable_neural": dict(Counter(row["customer"] for row in values if row.get("neural_observable"))),
    }
    appearance = metrics_summary(rows, "appearance")
    timing = metrics_summary(rows, "timing")
    sampling = counts("coverage")
    return {
        "planned": len(rows),
        "contributing_planned_components": len({component for component, _ in rows}),
        "constructed": sum(row.get("constructed", False) for row in values),
        "attempted": sum(row.get("attempted", False) for row in values),
        "completed": sum(row.get("complete", False) for row in values),
        "outcomes": dict(Counter(row["state"] for row in values)),
        "effects": dict(Counter(row["effect"] for row in values)),
        "visual_input_verified": sum(row.get("visual_input_verified", False) for row in values),
        "timing_input_verified": sum(row.get("timing_input_verified", False) for row in values),
        "raw_timing_binding_verified": sum(row.get("raw_timing_binding_verified", False) for row in values),
        "analysis_stages": counts("stages"),
        "detectors": counts("detectors"),
        "sampling": {
            "counts": sampling,
            "planned_legacy_decoded": ratio(
                sampling.get("legacy_decoded", 0), sampling.get("legacy_planned", 0)
            ),
            "decoded_legacy_with_detected_face": ratio(
                sampling.get("legacy_face_detected", 0), sampling.get("legacy_decoded", 0)
            ),
            "native_frames_tracked": ratio(
                sampling.get("tracked_native_pictures", 0), sampling.get("native_pictures", 0)
            ),
        },
        "customer_direction": customer,
        "appearance_review": appearance,
        "raw_timing_review": timing,
    }


def aggregate(plan, construction_by_ordinal, serving_by_ordinal, *, stage_states=None):
    """Return public pilot aggregates from caller-validated v143, v149 and v156 records.

    Inputs stay private. Stage states default to completed for supplied serving
    receipts and unstarted for absent receipts. Explicit guard failures and
    missing rows retain their places in the fixed planned denominator.
    """
    partitions = defaultdict(list)
    for component, spec, case, row, original, state in grouped_rows(
        plan, construction_by_ordinal, serving_by_ordinal, stage_states or {}
    ):
        name = spec["id"]
        observed = component, case_metrics(spec, case, row, original, state)
        partitions["overall"].append(observed)
        partitions[name].append(observed)
        partitions[FAMILY_OF[name] + "_family"].append(observed)

    def breakdown(names, suffix=""):
        groups = {name: partitions[name + suffix] for name in names}
        if any(len({component for component, _ in rows}) < MIN_PUBLIC_COMPONENTS for rows in groups.values()):
            return None
        return {name: summary(rows) for name, rows in groups.items()}

    conditions, families = breakdown(CONDITIONS), breakdown(FAMILIES, "_family")
    suppress_small_quantities(conditions, families)
    return {
        "schema": "chipmunk.grid-serving-readout/159",
        "cohort_role": "pilot",
        "profile": "full",
        "serving_receipt_version": 156,
        "minimum_public_components": MIN_PUBLIC_COMPONENTS,
        "overall": summary(partitions["overall"]),
        "conditions": conditions,
        "families": families,
    }


def suppress_small_quantities(conditions, families):
    """Remove continuous measures in a partition when its complement would disclose a small cell."""
    groups = [group for partition in (conditions, families) for group in (partition or {}).values()]
    for endpoint in ("appearance_review", "raw_timing_review"):
        summaries = [group[endpoint] for group in groups]
        quantities = [
            key
            for key, value in (summaries[0] if summaries else {}).items()
            if isinstance(value, dict) and "component_weighted_mean" in value
        ]
        for key in quantities:
            if any(0 < item[key]["contributing_components"] < MIN_PUBLIC_COMPONENTS for item in summaries):
                for item in summaries:
                    item[key]["sum"] = item[key]["component_weighted_mean"] = None
