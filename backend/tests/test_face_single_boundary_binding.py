from __future__ import annotations

import json
from fractions import Fraction

import pytest

from app.core.canonical import canonical_hash, canonical_json
from app.evidence import face_single_boundary_binding as binding


@pytest.fixture
def observation():
    source = {"sha256": "1" * 64, "width": 640, "height": 360, "frame_count": 100,
              "time_base": "1/25000", "timeline_origin_s": .2}
    clock = [5000 + index * 1000 for index in range(100)]
    box = {"x": 200, "y": 80, "w": 120, "h": 150}
    record = {"method": binding.METHOD, "coarse": 32, "boundary": 33, "pts": clock[33], "time_base": "1/25000",
              "time_s": 1.52, "scene": .91, "persistent": .46, "edge_supported": .38,
              "jump_excess": .38, "fine_margin": .22, "fine_scene": .95, "affects_overall_confidence": False,
              "source_frames": {}}
    record["readings_hex"] = {key: record[key].hex() for key in binding.READINGS}
    packet = {"id": binding.PACKET_ID, "semantics": binding.METHOD, binding.KEY: record,
              "effect": "context", "claim_scope": "sampled_window", "evidence_type": "anomaly",
              "affects_overall_confidence": False, "title": binding.TITLE, "observation": binding.OBSERVATION,
              "interpretation": binding.INTERPRETATION, "limitations": [binding.LIMITATION],
              "measurements": binding.measurements(record), "signal_refs": [], "finding_ids": [], "frames": [], "artifacts": {}}
    for role, index in zip(binding.ROLES, (32, 33, 43), strict=True):
        row = {"index": index, "pts": clock[index], "box": box, "eyes": [[233.5, 117.5], [288.5, 118.5]],
               "decoded_rgb_sha256": "2" * 64, "aligned_rgb_sha256": "3" * 64}
        row["eyes_hex"] = [[value.hex() for value in point] for point in row["eyes"]]
        record["source_frames"][role] = row
        refs = {name: f"{role}_{name}" for name in ("original", "context", "crop")}
        packet["artifacts"].update({value: {} for value in refs.values()})
        source_time = float(clock[index] * Fraction(source["time_base"]))
        packet["frames"].append({"id": f"{binding.PACKET_ID}.{role}", "interval_role": role, "score": None,
            "is_peak": False, "region_meaning": "detected_face_context", "context_region": box, "artifact_refs": refs,
            "anchor": {"frame_index": index, "pts": clock[index], "decoded_rgb_sha256": row["decoded_rgb_sha256"],
                       "original_sha256": source["sha256"], "time_base": source["time_base"], "stream_selector": "v:0",
                       "width": source["width"], "height": source["height"], "stream_time_s": source_time,
                       "playback_time_s": source_time - source["timeline_origin_s"]}})
    coverage = {"status": "localized", "reason": None, "frames_scanned": 100, "measured_face_frames": 49}
    source.update({binding.COMMITMENT: canonical_hash(record), binding.COVERAGE_COMMITMENT: canonical_hash(coverage)})
    return {"source": source, "coverage": {binding.COVERAGE: coverage}, "packets": [packet]}, clock


def test_valid_single_frame_and_offsets(observation):
    evidence, clock = observation
    binding.validate(evidence, canonical_hash, table_pts=clock)
    with pytest.raises(ValueError, match="short video clock"):
        binding.validate(evidence, canonical_hash, table_pts=clock, interpolated=(71,))


def test_retains_exact_alignment_and_boundary_margin_after_serialization(observation):
    evidence, clock = observation
    packet = evidence["packets"][0]
    record = packet[binding.KEY]
    actual_eye, actual_margin = 233.51234567891, .000000126
    record["source_frames"]["before"]["eyes"][0][0] = actual_eye
    record["source_frames"]["before"]["eyes_hex"][0][0] = actual_eye.hex()
    record["fine_margin"], record["readings_hex"]["fine_margin"] = actual_margin, actual_margin.hex()
    evidence["source"][binding.COMMITMENT] = canonical_hash(record)
    frozen = json.loads(canonical_json(evidence))
    binding.validate(frozen, canonical_hash, table_pts=clock)
    recorded = frozen["packets"][0][binding.KEY]
    assert binding.eyes(recorded["source_frames"]["before"])[0][0] == actual_eye
    assert binding.reading(recorded, "fine_margin") == actual_margin


@pytest.mark.parametrize(("case", "message"), [
    ("digests", "published commitments"),
    ("range", "single confidence-neutral"),
    ("role", "native frame"),
    ("coverage", "Invalid face transition coverage"),
    ("candidate_pts", "exact source position"),
])
def test_rejects_rewritten_boundary_evidence(observation, case, message):
    evidence, clock = observation
    packet = evidence["packets"][0]
    if case == "digests":
        evidence["source"].pop(binding.COMMITMENT)
    elif case == "range":
        packet["window"] = {"start_pts": 0, "end_pts": clock[33]}
    elif case == "role":
        packet["frames"][1]["interval_role"] = "during"
    elif case == "coverage":
        coverage = evidence["coverage"][binding.COVERAGE]
        coverage["measured_face_frames"] = 51
        evidence["source"][binding.COVERAGE_COMMITMENT] = canonical_hash(coverage)
    elif case == "candidate_pts":
        record = packet[binding.KEY]
        record["source_frames"]["after"]["pts"] += 1
        evidence["source"][binding.COMMITMENT] = canonical_hash(record)
    with pytest.raises(ValueError, match=message):
        binding.validate(evidence, canonical_hash, table_pts=clock)


def test_skipped_profile_has_no_visible_score():
    coverage = {"status": "unavailable", "reason": "profile", "frames_scanned": 0, "measured_face_frames": 0}
    document = {"source": {binding.COVERAGE_COMMITMENT: canonical_hash(coverage)}, "packets": [],
                "coverage": {binding.COVERAGE: coverage}}
    binding.validate(document, canonical_hash)
    assert "verified 25 fps clock" in binding.coverage_text(coverage)


@pytest.fixture
def sparse_observation(observation):
    evidence, clock = observation
    packet = evidence["packets"][0]
    record = packet[binding.KEY]
    record["method"] = binding.SPARSE_METHOD
    record["scene_context"] = {
        "arm": binding.MOTION_CONTEXT, "coarse": binding.MOTION_CONTEXT, "boundary": binding.COARSE_CONTEXT,
        "coarse_boxes": [{"x": 198, "y": 80, "w": 120, "h": 150}, {"x": 200, "y": 80, "w": 120, "h": 150}],
    }
    packet.update(semantics=record["method"], derivation={"selection": record["method"]},
                  observation=binding.observation(record), measurements=binding.measurements(record))
    coverage = evidence["coverage"][binding.COVERAGE]
    coverage["method"] = binding.SPARSE_METHOD
    evidence["source"].update({binding.COMMITMENT: canonical_hash(record), binding.COVERAGE_COMMITMENT: canonical_hash(coverage)})
    return evidence, clock


def test_sparse_coarse_context_keeps_exact_boundary_measurement(sparse_observation):
    evidence, clock = sparse_observation
    frozen = json.loads(canonical_json(evidence))
    binding.validate(frozen, canonical_hash, table_pts=clock)
    packet = frozen["packets"][0]
    assert "detail below the face" in packet["observation"]
    assert packet["measurements"][1]["label"] == "Surrounding picture similarity at the transition"
    assert [frame["anchor"]["frame_index"] for frame in packet["frames"]] == [32, 33, 43]
    assert packet["affects_overall_confidence"] is False


def test_incumbent_record_replays_with_original_and_sparse_coverage(observation):
    evidence, clock = observation
    frozen = json.loads(canonical_json(evidence))
    binding.validate(frozen, canonical_hash, table_pts=clock)
    coverage = frozen["coverage"][binding.COVERAGE]
    coverage["method"] = binding.SPARSE_METHOD
    frozen["source"][binding.COVERAGE_COMMITMENT] = canonical_hash(coverage)
    binding.validate(frozen, canonical_hash, table_pts=clock)
    assert frozen["source"][binding.COMMITMENT] == evidence["source"][binding.COMMITMENT]
    assert frozen["packets"][0]["semantics"] == binding.METHOD


def test_sparse_record_cannot_downgrade_coverage(sparse_observation):
    evidence, clock = sparse_observation
    coverage = evidence["coverage"][binding.COVERAGE]
    del coverage["method"]
    evidence["source"][binding.COVERAGE_COMMITMENT] = canonical_hash(coverage)
    with pytest.raises(ValueError, match="Sparse face transition requires matching coverage"):
        binding.validate(evidence, canonical_hash, table_pts=clock)


@pytest.mark.parametrize(("field", "legacy_value"), [
    ("semantics", binding.METHOD),
    ("derivation", {"selection": binding.METHOD}),
    ("observation", binding.OBSERVATION),
])
def test_sparse_packet_cannot_present_legacy_method(sparse_observation, field, legacy_value):
    evidence, clock = sparse_observation
    evidence["packets"][0][field] = legacy_value
    with pytest.raises(ValueError, match="single confidence-neutral boundary"):
        binding.validate(evidence, canonical_hash, table_pts=clock)


@pytest.mark.parametrize(("field", "invalid_value"), [
    ("boundary", binding.FINE_CONTEXT),
    ("coarse", binding.COARSE_CONTEXT),
    ("coarse_boxes", [{"x": -1, "y": 80, "w": 120, "h": 150}, {"x": 200, "y": 80, "w": 120, "h": 150}]),
])
def test_sparse_context_rejects_tampered_basis_and_geometry(sparse_observation, field, invalid_value):
    evidence, clock = sparse_observation
    record = evidence["packets"][0][binding.KEY]
    record["scene_context"][field] = invalid_value
    with pytest.raises(ValueError, match="published commitments"):
        binding.validate(evidence, canonical_hash, table_pts=clock)
    evidence["source"][binding.COMMITMENT] = canonical_hash(record)
    with pytest.raises(ValueError, match="Invalid sparse scene context"):
        binding.validate(evidence, canonical_hash, table_pts=clock)


def test_conflicting_context_abstention_requires_sparse_coverage():
    coverage = {"status": "unavailable", "reason": "conflicting_boundary_context", "frames_scanned": 100,
                "measured_face_frames": 49, "method": binding.SPARSE_METHOD}
    source = {"width": 640, "height": 360, "frame_count": 100, "time_base": "1/25000",
              binding.COVERAGE_COMMITMENT: canonical_hash(coverage)}
    evidence = {"source": source, "coverage": {binding.COVERAGE: coverage}, "packets": []}
    binding.validate(evidence, canonical_hash)
    del coverage["method"]
    source[binding.COVERAGE_COMMITMENT] = canonical_hash(coverage)
    with pytest.raises(ValueError, match="Invalid face transition coverage"):
        binding.validate(evidence, canonical_hash)
