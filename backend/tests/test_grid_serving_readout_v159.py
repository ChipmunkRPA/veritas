import json
from copy import deepcopy
from fractions import Fraction

import pytest

from app.core.canonical import canonical_hash
from app.evidence.face_interval_binding import APPEARANCE_REVIEW_POINT, SEQUENCE_REVIEW_POINT
from scripts import grid_serving_readout_v159 as readout


def sha(index):
    return f"{index:064x}"


def spec(name):
    original = name == "original"
    delay = name == "audio_delay_320ms"
    return {
        "id": name,
        "output_picture_indices": [] if original or delay else list(range(27, 47)),
        "edit_frames_half_open": None if original else [27, 47],
        "planned_neural_provenance": readout.FAMILY_OF[name] == "neural_redraw",
        "waveform_id": "delayed_audio_control" if delay else "original_pcm",
    }


def construct(name, ordinal=0, component=50):
    design = spec(name)
    change = name in {"musetalk_source_time", "brightness_musetalk"}
    delay = name == "audio_delay_320ms"
    masks = [
        {"frame_index": i, "pts": i * 3600, "changed_pixels": int(change and i == 30)}
        for i in design["output_picture_indices"]
    ]
    return {
        "status": "ready",
        "asset_index": ordinal,
        "source_slot": ordinal + 100,
        "component_index": component,
        "condition": name,
        "case_number": readout.CONDITIONS.index(name),
        "study_role": "pilot",
        "file_sha256": sha(2000 + readout.CONDITIONS.index(name)),
        "source_video_time_base": "1/90000",
        "source_video_pts": [i * 3600 for i in range(75)],
        "expected": {
            "picture_pts_ms": [i * 40 for i in range(75)],
            "rgb_sha256": [sha(300 + i if change and i == 30 else i) for i in range(75)],
            "audio_start_seconds": "0",
            "audio_samples": 51_200,
            "pcm_s16_sha256": sha(501 if delay else 500),
            "constructed_last_picture_duration_ms": 40,
        },
        "edit_proofs": masks,
        "changed_native_pixels": int(change),
        "changed_video_frames": int(change),
        "planned_neural_provenance": design["planned_neural_provenance"],
        "neural_provenance_verified": design["planned_neural_provenance"],
        "observed_neural_pixel_change": design["planned_neural_provenance"] and change,
        "waveform_id": design["waveform_id"],
        "edit_frames_half_open": design["edit_frames_half_open"],
        "source_time_bounds_seconds": None if name == "original" else ["27/25", "47/25"],
        "changed_audio_samples": 100 if delay else 0,
        "audio_value_mapping": [
            {
                "output_samples_half_open": [17_280, 30_080],
                "input_samples_half_open": [12_160, 24_960],
                "time_base": "1/16000",
            }
        ]
        if delay
        else [
            {
                "output_samples_half_open": [0, 51_200],
                "input_samples_half_open": [0, 51_200],
                "time_base": "1/16000",
            }
        ],
    }


def served(case, direction="inconclusive"):
    table = {
        "time_base": "1/1000",
        "pts": case["expected"]["picture_pts_ms"],
        "width": 360,
        "height": 288,
        "truncated": False,
        "interpolated": [],
    }
    mapping = {
        "samples": 51_200,
        "sample_rate": 16_000,
        "source_sha256": case["file_sha256"],
        "timeline_time_base": "1/16000",
        "timeline": [
            {"start_sample": 0, "end_sample": 20_000, "source_pts": 0},
            {"start_sample": 20_000, "end_sample": 51_200, "source_pts": 20_000},
        ],
        "pcm_sha256": sha(600),
    }
    frames = [
        {"frame_index": i, "pts": table["pts"][i], "sha256": case["expected"]["rgb_sha256"][i]}
        for i in (1, 10, 20, 30, 40, 50, 60)
    ]
    return {
        **{
            key: case[key]
            for key in ("asset_index", "source_slot", "component_index", "case_number", "condition")
        },
        "status": "done",
        "analysis_status": "done",
        "result_hash": sha(910),
        "input_file_sha256": case["file_sha256"],
        "customer_assessment": {
            "direction": direction,
            "content_assessed": True,
            "directional_conclusion": direction in {"ai", "original"},
            "scope": "visual_content",
            "excluded_channels": ["voice", "audio_picture_timing"],
            "soundtrack": {"direction": "inconclusive"},
        },
        "sampled_input": {
            "clock_exact": True,
            "sampled_rgb_exact": True,
            "audio_samples_equal": True,
            "audio_origin_equal": True,
            "audio_pcm_equal": True,
            "probe_frame_table": table,
            "probe_plan": {"legacy": {"frames": [1, 10, 20, 30, 40, 50, 60, 70]}},
            "sampled_frames": frames,
            "sampled_face_counts": [1, 0, 1, 2, 1, 1, 1],
            "track_exhibits": [
                {"frames": [{"frame_index": 30, "pts": 1200}, {"frame_index": 40, "pts": 1600}]},
                {"frames": [{"frame_index": 30, "pts": 1200}]},
            ],
            "serving_audio": {"samples": mapping["samples"], "source_mapping": mapping},
        },
        "detector_stage_outputs": {"detect": {"detectors": [], "exhibits": []}},
        "evidence": {},
    }


def window(clock, method, value):
    centers = {"before": [1, 10], "during": [20, 30, 40], "after": [50, 60]}
    indices = sorted({i + shift for group in centers.values() for i in group for shift in (-1, 0, 1)})
    box = {"x": 5, "y": 5, "w": 110, "h": 110}
    alignment = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    measures = (
        {"response": value, "middle_to_before": value, "middle_to_after": value, "flank_change": 0.0}
        if method == readout.APPEARANCE_METHOD
        else {"response": value, "inner_change": value, "outer_change": 0.0, "flank_change": 0.0}
    )
    return {
        "method": method,
        "track_id": "private-track",
        "time_base": "1/1000",
        "centers": centers,
        "start_pts": 800,
        "end_pts": 1600,
        "start_stream_s": 0.8,
        "end_stream_s": 1.6,
        **measures,
        "measurement_hex": {name: value.hex() for name, value in measures.items()},
        "frame_inputs": [
            {
                "frame_index": i,
                "pts": clock.table["pts"][i],
                "decoded_rgb_sha256": clock.rgb[i],
                "aligned_rgb_sha256": sha(800 + i),
                "alignment_side": 112,
                "aligned_size": 112,
                "bbox": box,
                "landmarks": [[10.0, 10.0]] * 5,
                "alignment_matrix": alignment,
                "alignment_matrix_hex": [[value.hex() for value in row] for row in alignment],
            }
            for i in indices
        ],
    }


def face_entry(record, route, *, select=True):
    return {
        "detector_id": readout.APPEARANCE,
        "status": "ok",
        "details": {
            "appearance_check_status": route,
            "face_sequence_binding_version": record["method"],
            "face_sequence_windows": [record],
            "face_sequence": record if select else None,
        },
    }


def face_packet(record, clock):
    inputs = {row["frame_index"]: row for row in record["frame_inputs"]}
    return {
        "face_sequence": record,
        "semantics": record["method"],
        "claim_scope": "sampled_window",
        "evidence_type": "anomaly",
        "window": {"frame_indices": list(inputs)},
        "frames": [
            {
                "interval_role": role,
                "score": None,
                "is_peak": False,
                "anchor": {**inputs[index], "original_sha256": clock.file_sha, "time_base": "1/1000"},
                "context_region": inputs[index]["bbox"],
            }
            for role, index in (("before", 10), ("during", 30), ("after", 50))
        ],
    }


def with_timing(row, case, *, segments=(), reading="no_stable_offset"):
    detect = row["detector_stage_outputs"]["detect"]
    clock = readout.native_clock(case, row)
    mapping = clock.sampled["serving_audio"]["source_mapping"]
    track_id = "private-timing-track"
    frames = [
        {
            "tick": i,
            "frame_index": i,
            "frame_pts": i * 40,
            "source_audio_pts": i * 640,
            "source_audio_time_base": "1/16000",
            "decoded_rgb_sha256": clock.rgb[i],
        }
        for i in range(15, 60)
    ]
    windows = [
        {
            "start_tick": a,
            "end_tick": b,
            "unit_ref": f"{track_id}@window={i}",
            "audio_interval": {"start_sample": a * 640, "end_sample": b * 640},
            "peak_correlation": 0.77,
            "offset_ticks": 2,
            "overlaps_offset_step": any(k == "offset_step" and max(a, c) < min(b, d) for k, c, d in segments),
            "overlaps_correlation_collapse": any(
                k == "confidence_collapse" and max(a, c) < min(b, d) for k, c, d in segments
            ),
        }
        for i, (a, b) in enumerate(((15, 40), (25, 50), (35, 60)))
    ]
    proof = {
        "version": "av-correlation-input/1",
        "status": "ready",
        "method": "mouth_rms_correlation",
        "source_sha256": case["file_sha256"],
        "pcm_sha256": mapping["pcm_sha256"],
        "mapping_sha256": canonical_hash(mapping),
        "frame_table_sha256": canonical_hash(clock.table),
        "frame_time_base": "1/1000",
        "audio_origin_pts": 0,
        "audio_time_base": "1/16000",
        "samples": mapping["samples"],
        "sample_rate": 16_000,
        "tick_rate": 25,
        "samples_per_tick": 640,
        "lag_search_ticks": [-15, 15],
        "tracks": [
            {
                "track_id": track_id,
                "frame_inputs": frames,
                "windows": windows,
                "reading": reading,
                "segments": [{"kind": k, "start_tick": a, "end_tick": b} for k, a, b in segments],
            }
        ],
    }
    entry = {
        "detector_id": readout.TIMING,
        "name": "private-name",
        "modality": "video",
        "family": "av-sync",
        "run_id": "private-run",
        "hypotheses": ["H2"],
        "status": "ok",
        "details": {"av_input": proof},
        "metrics": {
            "windows": 3,
            "speaking_tracks": 1,
            "offset_steps": sum(k == "offset_step" for k, _, _ in segments),
            "correlation_collapses": sum(k == "confidence_collapse" for k, _, _ in segments),
        },
        "exhibit_ids": ["private-exhibit"] if segments else [],
        "unit_scores": [
            {
                "unit_ref": w["unit_ref"],
                "raw_kind": "score",
                "raw": w["peak_correlation"],
                "quality": {
                    "offset_frames": w["offset_ticks"],
                    "start_s": w["start_tick"] / 25,
                    "end_s": w["end_tick"] / 25,
                },
            }
            for w in windows
        ],
    }
    detect["detectors"].append(entry)
    if segments:
        detect["exhibits"].append(
            {
                "kind": "segments",
                "id": "private-exhibit",
                "created_by": readout.TIMING,
                "lane": "lip_sync",
                "segments": [
                    {
                        "track_id": track_id,
                        "start_s": round(a / 25, 6),
                        "end_s": round(b / 25, 6),
                        "time_base": "1/1000",
                        "start_pts": int(a) * 40,
                        "end_pts": min(74, int(b - 1)) * 40,
                        "channel": "av",
                    }
                    for _, a, b in segments
                ],
            }
        )
    selected = windows[0]
    start, end = selected["audio_interval"].values()
    raw = canonical_hash(
        entry | {"run_id": None, "name": "", "family": "", "modality": "image", "hypotheses": []}
    )
    row["evidence"]["av"] = {
        "packets": [
            {
                "semantics": "mouth_audio_correlation",
                "source_record": {
                    "raw_output_sha256": raw,
                    "track_id": track_id,
                    "unit_ref": selected["unit_ref"],
                },
                "anchor": {
                    "original_sha256": clock.file_sha,
                    "pcm_sha256": proof["pcm_sha256"],
                    "sample_rate": 16_000,
                    "source_timestamp_ranges": [
                        {
                            "start_sample": start,
                            "end_sample": 20_000,
                            "source_start_pts": start,
                            "source_end_pts": 20_000,
                            "time_base": "1/16000",
                        },
                        {
                            "start_sample": 20_000,
                            "end_sample": end,
                            "source_start_pts": 20_000,
                            "source_end_pts": end,
                            "time_base": "1/16000",
                        },
                    ],
                },
                "interval": selected["audio_interval"],
                "window": {
                    "start_tick": 15,
                    "end_tick": 40,
                    "frame_indices": list(range(15, 40)),
                    "frame_inputs": [frame | {"artifact_ref": "private-file"} for frame in frames[:25]],
                },
                "measurement": {
                    "overlaps_offset_step": selected["overlaps_offset_step"],
                    "overlaps_correlation_collapse": selected["overlaps_correlation_collapse"],
                },
            }
        ]
    }
    return entry, detect, clock


def plan(components=16):
    return {
        "sources": [
            {
                "ordinal": i,
                "role": "pilot",
                "source_slot": 100 + i,
                "component_index": 50 + i % components,
                "cases": [spec(name) for name in readout.CONDITIONS],
            }
            for i in range(16)
        ]
        + [{"ordinal": 16, "role": "reserve"}]
    }


def test_appearance_point_is_strict_fallback_is_inclusive():
    case = construct("original")
    clock = readout.native_clock(case, served(case))
    record = window(clock, readout.APPEARANCE_METHOD, APPEARANCE_REVIEW_POINT)
    at_point = readout.appearance_endpoint(face_entry(record, "ok", select=False), clock, {}, set(), None)
    assert at_point.state == "measured_no_flag"
    assert at_point.spans == []
    invalid = readout.appearance_endpoint(face_entry(record, "ok"), clock, {}, set(), None)
    assert invalid.state == "unbound_evidence"
    fallback = window(clock, readout.SEQUENCE_METHOD, SEQUENCE_REVIEW_POINT)
    valid = readout.appearance_endpoint(face_entry(fallback, "not_applicable"), clock, {}, set(), None)
    assert valid.branch == "fallback"
    assert valid.spans == [(Fraction(4, 5), Fraction(8, 5))]


def test_only_selected_appearance_range_flags_and_delivered_anchors_are_bound():
    case = construct("musetalk_source_time")
    row = served(case)
    clock = readout.native_clock(case, row)
    record = window(clock, readout.APPEARANCE_METHOD, 0.4)
    entry = face_entry(record, "ok")
    entry["details"]["face_sequence_windows"].append(window(clock, readout.APPEARANCE_METHOD, 0.3))
    packet = face_packet(record, clock)
    truth = (Fraction(27, 25), Fraction(47, 25))
    measured = readout.appearance_endpoint(entry, clock, {"packets": [packet]}, {30}, truth)
    assert measured.state == "flagged"
    assert len(measured.spans) == 1
    assert measured.windows == 2
    assert measured.shown_flag == 1
    assert measured.bracket and measured.during_changed
    assert readout.endpoint_metrics(measured, clock.bounds, truth, {30})["iou"] == Fraction(13, 27)
    packet["frames"][1]["anchor"]["decoded_rgb_sha256"] = sha(999)
    unbound = readout.appearance_endpoint(entry, clock, {"packets": [packet]}, {30}, truth)
    assert unbound.state == "flagged"
    assert unbound.shown_flag == 0
    assert unbound.counts["unbound_packets"] == 1


def test_timing_raw_fractional_union_counts_beyond_delivered_window():
    case = construct("audio_delay_320ms")
    row = served(case)
    segments = (("offset_step", 26.5, 40.5), ("confidence_collapse", 35.0, 74.5))
    entry, detect, clock = with_timing(row, case, segments=segments, reading="offset_step_and_collapse")
    timing, bounds = readout.timing_endpoint(entry, detect, case, clock, row["evidence"])
    assert timing.state == "flagged"
    assert len(timing.spans) == 2
    assert timing.windows == 3
    assert timing.shown_flag == 1
    _, _, truth = readout.construction_truth(spec(case["condition"]), case)
    assert truth == (Fraction(27, 25), Fraction(47, 25))
    result = readout.endpoint_metrics(timing, bounds, truth, set())
    assert result["flagged_seconds"] == Fraction(19, 10)
    assert result["overlap_seconds"] == Fraction(4, 5)
    assert result["iou"] == Fraction(8, 19)
    assert result["excluded_seconds"] == Fraction(1, 50)
    assert result["truncated_spans"] == 1
    assert result["omitted_windows"] == 2
    assert result["verified_picture_inputs"] == 45
    # Presentation evidence cannot replace the raw segment count.
    entry["metrics"]["offset_steps"] = 2
    unbound, _ = readout.timing_endpoint(entry, detect, case, clock, row["evidence"])
    assert unbound.state == "unbound_evidence"
    assert unbound.counts["unbound_segments"] == 3


def test_no_stable_offset_silence_skips_and_unverified_audio_are_distinct():
    case = construct("original")
    row = served(case)
    entry, detect, clock = with_timing(row, case)
    result, _ = readout.timing_endpoint(entry, detect, case, clock, row["evidence"])
    assert result.state == "no_stable_offset"
    assert result.windows == 3
    entry["details"]["av_input"]["tracks"][0]["reading"] = "constant_offset"
    assert readout.timing_endpoint(entry, detect, case, clock, {})[0].state == "measured_no_flag"
    assert readout.timing_endpoint({"status": "ok"}, detect, case, clock, {})[0].state == "no_support"
    assert (
        readout.timing_endpoint({"status": "not_applicable"}, detect, case, clock, {})[0].state
        == "not_applicable"
    )
    clock.sampled["audio_pcm_equal"] = False
    assert readout.timing_endpoint(entry, detect, case, clock, {})[0].state == "unverified_input"


@pytest.mark.parametrize("change", ["hash", "ordinal", "number"])
def test_join_rejects_swapped_serving_records(change):
    cases = [construct(name) for name in readout.CONDITIONS]
    row = served(cases[0])
    key = {"hash": "input_file_sha256", "ordinal": "asset_index", "number": "case_number"}[change]
    row[key] = sha(999) if change == "hash" else 999
    with pytest.raises(ValueError, match="actual constructed input"):
        readout.aggregate(plan(), {0: cases}, {0: [row]})


def test_aggregate_preserves_planned_denominators_without_case_data():
    first = [construct(name) for name in readout.CONDITIONS]
    second = [construct(name, 1, 51) for name in readout.CONDITIONS]
    by_condition = {case["condition"]: case for case in first}
    original, delay = by_condition["original"], by_condition["audio_delay_320ms"]
    ordinary = by_condition["brightness_musetalk"]
    unavailable = by_condition["resample_musetalk"]
    unavailable.update(status="unavailable", reason="private reason should not appear")
    o = served(original, "original")
    o["customer_assessment"]["content_assessed"] = False
    f = served(ordinary, "ai")
    f["detector_inventory"] = [
        {
            "expected_stage": "detect",
            "runs": [{"status": "skipped", "reason": "memory budget", "run_id": "one"}],
        },
        {"expected_stage": "detect", "runs": []},
        {"expected_stage": None, "runs": []},
    ]
    f["detector_runs"] = [{"id": "one"}]
    f["stages"] = [{"stage": "decode", "status": "done"}]
    d = served(delay)
    clock = readout.native_clock(delay, d)
    record = window(clock, readout.APPEARANCE_METHOD, 0.4)
    d["detector_stage_outputs"]["detect"]["detectors"].append(face_entry(record, "ok"))
    d["evidence"]["packets"] = [face_packet(record, clock)]
    f["error"] = "private-error"
    s = readout.aggregate(plan(), {0: first, 1: second}, {0: [o, f, d]}, stage_states={1: "guard_failed"})
    assert s["overall"]["planned"] == 176
    assert s["overall"]["outcomes"] == {
        "done": 3,
        "missing_case": 7,
        "unavailable_input": 1,
        "guard_failed": 11,
        "construction_unobserved": 154,
    }
    assert s["overall"]["detectors"]["expected"] == 2
    assert s["overall"]["detectors"]["attempted"] == 1
    assert s["overall"]["detectors"]["model_unavailable"] == 1
    assert s["overall"]["sampling"]["native_frames_tracked"]["numerator"] == 6
    assert s["overall"]["sampling"]["planned_legacy_decoded"]["rate"] == "7/8"
    assert s["overall"]["customer_direction"]["false_ai_attribution"]["numerator"] == 1
    assert s["overall"]["customer_direction"]["counts"]["engine_returned_original"] == 1
    assert "original" not in s["overall"]["customer_direction"]["counts"]
    assert s["overall"]["appearance_review"]["unchanged_stream_flags"]["numerator"] == 1
    assert s["overall"]["appearance_review"]["ordinary_control_flags"]["numerator"] == 1
    assert s["overall"]["appearance_review"]["flag_delivered"]["numerator"] == 1
    assert s["overall"]["appearance_review"]["iou"]["denominator"] == 0
    assert s["conditions"]["audio_delay_320ms"]["raw_timing_review"]["assessed"]["numerator"] == 0
    public = json.dumps(s)
    for private in (
        "private-error",
        "private reason",
        "private-track",
        "source_slot",
        "component_index",
        "frame_index",
        original["file_sha256"],
    ):
        assert private not in public


def test_component_weighting_and_complete_partition_suppression():
    rows = [(0, {"hit": True}), (0, {"hit": True})] + [(i, {"hit": False}) for i in range(1, 5)]
    rate = readout.component_rate(rows, lambda row: row["hit"], lambda row: True)
    assert rate["rate"] == "1/3"
    assert rate["component_weighted_rate"] == "1/5"
    result = readout.aggregate(plan(4), {}, {})
    assert result["overall"]["planned"] == 176
    assert result["conditions"] is result["families"] is None
    assert result["overall"]["appearance_review"]["assessed"]["component_weighted_rate"] is None


def test_failed_analysis_can_retain_raw_review_without_a_customer_direction():
    case = construct("original")
    row = served(case, "ai")
    clock = readout.native_clock(case, row)
    record = window(clock, readout.APPEARANCE_METHOD, 0.5)
    row["detector_stage_outputs"]["detect"]["detectors"].append(face_entry(record, "ok"))
    row["evidence"]["packets"] = [face_packet(record, clock)]
    failed = deepcopy(row)
    failed.update(status="failed", analysis_status="failed", result_hash=None)
    metrics = readout.case_metrics(spec("original"), case, failed, case, "completed")
    assert metrics["customer"] == "unavailable"
    assert metrics["appearance"]["flagged"]
    assert not metrics["appearance"]["delivered_flag"]
    assert not metrics["false_ai_attribution"]
