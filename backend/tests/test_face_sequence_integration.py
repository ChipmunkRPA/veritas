from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from app.core.config import settings
from app.detectors.video import face_sequence
from app.detectors.video.adapters.face_sequence import (
    DETECTOR,
    FaceSequenceReviewDetector,
    SequenceInputs,
    record,
)
from app.detectors.video.decode import DecodedFrame, FrameTable
from app.detectors.video.frame_store import FrameStore
from app.engine.fusion.runtime import annotate
from app.engine.registry import get_registry
from app.evidence.face_interval_binding import (
    APPEARANCE_METHOD,
    SEQUENCE_METHOD,
    validate_packet,
    validate_sequence_record,
)
from app.evidence.image_packets import PacketIntegrityError
from app.evidence.video_packets import _sequence_observations
from scripts.score_face_sequence_candidates import sequence_windows as reference_windows


def table():
    return FrameTable(time_base="1/25", pts=tuple(range(125, 185)), key_frames=(), pict_types="I"*60,
                      width=96, height=64, coded_width=96, coded_height=64)


def sample():
    source_table = table()
    centers = [2, 10, 18, 26, 34, 42, 50]
    inputs = {i: {"frame_index": i, "pts": source_table.pts[i], "decoded_rgb_sha256": "a"*64,
                  "aligned_rgb_sha256": "b"*64, "bbox": {"x": 10, "y": 8, "w": 32, "h": 24},
                  "landmarks": [[12., 12.], [30., 12.], [20., 18.], [14., 24.], [28., 24.]],
                  "alignment_matrix": [[1., 0., 0.], [0., 1., 0.]], "alignment_side": 32, "aligned_size": 112}
              for i in sorted({i+k for i in centers for k in (-1, 0, 1)})}
    for item in inputs.values():
        item["alignment_matrix_hex"] = [[value.hex() for value in row] for row in item["alignment_matrix"]]
    value = record({"centers": centers, "track_id": "track:0", "start_s": source_table.seconds(18),
                    "end_s": source_table.seconds(34), "response": .02, "inner_change": .04,
                    "outer_change": .01, "flank_change": .04}, source_table, inputs)
    source = {"time_base": source_table.time_base, "frame_count": len(source_table), "width": 96, "height": 64}
    return source_table, value, source


def packet():
    source_table, value, source = sample()
    shown = [("before", 10), ("during", 26), ("after", 42)]
    body = {"semantics": SEQUENCE_METHOD, "evidence_type": "anomaly", "claim_scope": "sampled_window",
            "face_sequence": value, "window": {"frame_indices": [item["frame_index"] for item in value["frame_inputs"]]},
            "frames": [{"interval_role": role, "score": None, "is_peak": False,
                        "anchor": {"frame_index": i, "pts": source_table.pts[i], "decoded_rgb_sha256": "a"*64},
                        "context_region": {"x": 10, "y": 8, "w": 32, "h": 24}} for role, i in shown]}
    return source_table, body, source


def test_promoted_window_math_matches_frozen_reference_without_refitting():
    source_table = table()
    centers = {2, 10, 18, 26, 34, 42, 50}
    indices = {i+k for i in centers for k in (-1, 0, 1)}
    rng = np.random.default_rng(774)
    patches = {i: rng.normal(size=(64, 12)).astype(np.float32) for i in indices}
    tracks = {"track:0": [SimpleNamespace(frame_index=i) for i in sorted(indices)]}
    reference = reference_windows(centers, tracks, patches, source_table)
    actual = face_sequence.sequence_windows(centers, tracks, patches, source_table)
    assert len(actual) == len(reference) > 0
    assert [{k: value for k, value in item.items() if k != "flank_change"} for item in actual] == reference
    assert pytest.approx(.007326911389827726, abs=0) == face_sequence.REVIEW_POINT


def test_record_retains_native_triplets_and_nonzero_source_clock():
    source_table, value, source = sample()
    validate_sequence_record(value, source, table_pts=source_table.pts)
    assert value["start_stream_s"] == pytest.approx(5.72)
    assert value["end_stream_s"] == pytest.approx(6.36)
    assert len(value["frame_inputs"]) == 21
    assert [item["frame_index"] for item in value["frame_inputs"]][:3] == [1, 2, 3]


def test_exact_measurements_survive_canonical_rounding_without_changing_decision():
    import json

    from app.core.canonical import canonical_json

    source_table, value, source = sample()
    value.update(response=-0.02720938716083765, inner_change=0.023972488939762115,
                 outer_change=0.07032953202724457, flank_change=0.025669019669294357)
    value['measurement_hex'] = {k: float(value[k]).hex() for k in ('response', 'inner_change', 'outer_change', 'flank_change')}
    rounded = json.loads(canonical_json(value))
    # This real retained scalar case exceeds the old1.1e-6 approximate check
    # solely after independent six-decimal rounding of its four components.
    assert abs(rounded['response']-(rounded['inner_change']-.6*rounded['outer_change']-.35*rounded['flank_change'])) > .0000011
    validate_sequence_record(rounded, source, table_pts=source_table.pts, require_flag=False)
    with pytest.raises(ValueError):
        validate_sequence_record(rounded, source, table_pts=source_table.pts)
    rounded['measurement_hex']['response'] = .5.hex()
    with pytest.raises(ValueError, match='exact commitments'):
        validate_sequence_record(rounded, source, table_pts=source_table.pts, require_flag=False)


@pytest.mark.parametrize("damage", ["missing_neighbor", "wrong_pts", "interpolated", "wrong_range", "arithmetic", "nan",
                                    "bad_crop_hash", "bad_alignment", "unordered_centers", "misplaced_role", "frame_probability", "peak"])
def test_inconsistent_evidence_is_rejected(damage):
    source_table, body, source = packet()
    value = body["face_sequence"]
    interpolated = ()
    if damage == "missing_neighbor":
        value["frame_inputs"].pop()
    elif damage == "wrong_pts":
        value["frame_inputs"][0]["pts"] += 1
    elif damage == "interpolated":
        interpolated = (1,)
    elif damage == "wrong_range":
        value["start_stream_s"] += .1
    elif damage == "arithmetic":
        value["response"] += .1
    elif damage == "nan":
        value["inner_change"] = float("nan")
    elif damage == "bad_crop_hash":
        value["frame_inputs"][0]["aligned_rgb_sha256"] = "unknown"
    elif damage == "bad_alignment":
        value["frame_inputs"][0]["alignment_matrix"][0][0] = float("inf")
    elif damage == "unordered_centers":
        value["centers"]["during"].reverse()
    elif damage == "misplaced_role":
        body["frames"][0]["interval_role"] = "during"
    elif damage == "frame_probability":
        body["frames"][1]["score"] = .9
    elif damage == "peak":
        body["frames"][1]["is_peak"] = True
    with pytest.raises(ValueError):
        validate_packet(body, source, table_pts=source_table.pts, interpolated=interpolated)


def test_window_packet_has_no_frame_probability_or_peak():
    source_table, body, source = packet()
    validate_packet(body, source, table_pts=source_table.pts)
    assert all(frame["score"] is None and not frame["is_peak"] for frame in body["frames"])


def test_packet_selection_requires_the_exact_committed_input_pixels():
    source_table, value, source = sample()
    signal = {"id": DETECTOR+":global", "details": {"face_sequence_binding_version": SEQUENCE_METHOD,
               "face_sequence": value}, "status": "ok"}
    anchors = {item["frame_index"]: deepcopy(item) for item in value["frame_inputs"]}
    result = {"signals": [signal]}
    candidates, support = _sequence_observations(result, source_table, source, anchors)
    assert len(candidates) == 1 and len(support) == 21
    assert [item["interval_role"] for item in candidates[0]["rows"]] == ["before", "during", "after"]
    anchors[1]["decoded_rgb_sha256"] = "c"*64
    with pytest.raises(PacketIntegrityError, match="committed decoded source"):
        _sequence_observations(result, source_table, source, anchors)


def test_absent_and_failed_sequence_results_do_not_create_review_evidence():
    source_table, _, source = sample()
    for signal in ({"status": "failed", "details": {}}, {"status": "ok", "details": {
            "face_sequence_binding_version": SEQUENCE_METHOD, "face_sequence": None}}):
        assert _sequence_observations({"signals": [signal]}, source_table, source, {}) == ([], set())


def test_scope_and_budget_are_checked_before_model_acquisition():
    detector = FaceSequenceReviewDetector()
    source_table = table()
    for centers in ((), (2, 10, 18), tuple(range(65))):
        ctx = SimpleNamespace(outputs=SimpleNamespace(face_review=SequenceInputs(source_table, None, centers, ({},))))
        assert not detector.applicable(ctx).ok


def test_face_review_adapters_omit_unverified_decoder_timestamps(monkeypatch, tmp_path):
    from app.detectors.video.adapters import face_appearance as appearance_adapter
    from app.detectors.video.adapters import face_sequence as sequence_adapter

    source_table = table()
    centers = (2, 10, 18, 26, 34, 42, 50)
    verified_index = 18
    calls = []

    def decoder(_path, frame_table, indices, **_):
        for index in indices:
            rgb = np.full((frame_table.height, frame_table.width, 3), index, dtype=np.uint8)
            yield DecodedFrame(index, frame_table.pts[index], rgb, pts_verified=index == verified_index)

    def detect(bgr, **_):
        calls.append(int(bgr[0, 0, 0]))
        return []

    monkeypatch.setattr(sequence_adapter, "detect_faces", detect)
    monkeypatch.setattr(appearance_adapter, "detect_faces", detect)
    store = FrameStore(tmp_path / "not-read-by-fixture.mkv", source_table, decoder=decoder)
    inputs = SequenceInputs(source_table, store, centers, ({},))
    lease = SimpleNamespace(get=lambda _key: SimpleNamespace(obj=None, revision="fixture"))
    ctx = SimpleNamespace(outputs=SimpleNamespace(face_review=inputs), lease=lease, deadline=None)
    for detector, expected_calls in ((FaceSequenceReviewDetector(), 1),
                                     (appearance_adapter.FaceAppearanceReviewDetector(), 2)):
        calls.clear()
        result = detector.run(ctx)
        assert result.status == "not_applicable" and result.aggregate is None
        assert result.metrics["unverified_timestamp_frames"] == 20
        assert "verified decoder timestamps" in result.reason
        assert calls == [verified_index] * expected_calls


def test_sequence_requires_opt_in_and_keeps_legacy_available(monkeypatch):
    registry = get_registry()
    monkeypatch.setattr(settings, "video_face_review_mode", "legacy")
    assert not registry.decision(DETECTOR).enabled
    monkeypatch.setattr(settings, "video_face_review_mode", "sequence")
    assert registry.decision(DETECTOR).enabled
    entry = registry.manifest.detector(DETECTOR)
    assert not entry.hypotheses and entry.legacy_weight == 0
    assert all(registry.manifest.model(mid).licence.class_ == "commercial" for mid in entry.models)


def test_uncalibrated_review_response_has_zero_fusion_weight():
    signal = {"id": DETECTOR+":global", "detector_id": DETECTOR, "raw_score": .12, "raw_kind": "score",
              "effective_weight": 0, "calibrated_probability": None, "log_lr": None}
    [value] = annotate([signal], {}, artefact_detectors=["existing.classifier"])
    assert value["effective_weight"] == 0
    assert value["contribution_logit"] is None and value["contribution_pp"] is None
    assert value["calibrated_probability"] is None and value["log_lr"] is None


def test_unflagged_windows_retain_assessed_coverage_without_a_negative_verdict():
    source_table, value, source = sample()
    value.update(inner_change=.001, response=.001-.60*.01-.35*.04)
    value['measurement_hex'] = {k: float(value[k]).hex() for k in ('response', 'inner_change', 'outer_change', 'flank_change')}
    signal = {"id": DETECTOR+":global", "details": {"face_sequence_binding_version": SEQUENCE_METHOD,
               "face_sequence": None, "face_sequence_windows": [value]}, "status": "ok"}
    anchors = {item["frame_index"]: deepcopy(item) for item in value["frame_inputs"]}
    candidates, support = _sequence_observations({"signals": [signal]}, source_table, source, anchors)
    assert not candidates and support == set(anchors)
    assert signal["details"]["face_sequence"] is None


def test_small_sequence_check_does_not_depend_on_the_large_classifier_running(monkeypatch):
    from app.engine.detector import StageOutputs
    from app.engine.stages import video

    calls = []
    outputs = StageOutputs(faces=[[{}], [], [{}]], frames=[])
    monkeypatch.setattr(video, "_outputs", lambda _: outputs)
    monkeypatch.setattr(video.common, "face_stage_failure", lambda _: None)

    def run(_, modalities, **kwargs):
        calls.append(kwargs)
        return {"detectors": [{"detector_id": "vid.frames.dfdc_b7", "status": "skipped"}], "exhibits": [], "artefacts": []}

    monkeypatch.setattr(video.common, "run_detectors", run)
    frames = [SimpleNamespace(index=n, frame_index=i) for n, i in enumerate([2, 10, 18])]
    ctx = SimpleNamespace(profile="full", outputs={"tracks": {"exhibits": [{"kind": "face_track"}]}},
                          state={"frames": frames, "frame_store": SimpleNamespace(table=table()),
                                 "registry": SimpleNamespace(enabled=lambda *_: [DETECTOR])})
    video.detect(ctx)
    assert len(calls) == 2
    assert calls[-1]["outputs"].face_review.centers == (2, 18)


@pytest.mark.parametrize(("method", "observation"), [
    (SEQUENCE_METHOD, "after accounting for changes near its edges"),
    (APPEARANCE_METHOD, "differed from both surrounding time groups"),
])
def test_review_finding_is_explicit_and_linked_without_authorship_probability(monkeypatch, method, observation):
    from app.engine.stages import video

    _, sequence, _ = sample()
    sequence["method"] = method
    sid = DETECTOR+":global"
    generated = {"findings": [{"id": "F1", "evidence_ids": [sid, "face-series"]}], "template_set": "test"}
    monkeypatch.setattr(video.common, "synthesise_findings", lambda *_, **__: deepcopy(generated))
    ctx = SimpleNamespace(outputs={"fuse": {"signals": [{"id": sid, "detector_id": DETECTOR,
                          "status": "ok", "details": {"face_sequence": sequence}}]}})
    [finding] = video.synthesise(ctx)["findings"]
    assert finding["title"] == "Possible face-change range"
    assert "5.720 to 6.360 s" in finding["statement"]
    assert observation in finding["statement"]
    assert "overall confidence" in finding["statement"]
    assert finding["probability"] is None and finding["direction"] == "neutral"
    assert all(s["evidence_ids"] == [sid, "face-series"] for s in finding["sentences"])
