import pytest

from scripts import audio_measured_speaker_routing as study


def segment(speaker, start, end):
    return {"speaker": speaker, "start": start, "end": end}


def test_simultaneity_uses_distinct_track_ids_and_bounds_source_clock():
    base = [segment(1, 0., 3.01), segment(2, 0.5, 1.5), segment(2, 0.7, 1.2)]
    measured = study.measurement(base)
    assert measured == {"route": "single_interferer", "substantive_tracks": 2, "two_samples": 16000,
                        "three_or_more_samples": 0, "longest_two_samples": 16000}
    triple = study.measurement([*base, segment(3, .8, 1.2)])
    assert triple["route"] == "overlap_detected"
    assert triple["three_or_more_samples"] == 6400
    with pytest.raises(ValueError, match="source clock"):
        study.measurement([segment(1, 0, 3.02)])


def test_corroborated_route_preserves_ambiguity_and_rejects_added_voices():
    two = {"route": "single_interferer"}
    unknown = {"route": "unassessed"}
    three = {"route": "overlap_detected"}
    assert study.routes({"nominal": two, "sensitive": two})["corroborated"] == "single_interferer"
    assert study.routes({"nominal": two, "sensitive": unknown})["corroborated"] == "unassessed"
    assert study.routes({"nominal": two, "sensitive": three})["corroborated"] == "overlap_detected"


def test_replay_attributes_a_false_flag_to_the_measured_not_reference_route():
    manifest = {"rows": [{"id": 0, "panel": "p", "condition": "overlapping_interferers", "group": "private",
                          "label": 0, "decision": "inconclusive", "crossed_room_point": {"relation_unpaired": True}}]}
    actual = {"rows": [{"id": 0, "measurements": {
        "nominal": {"route": "single_interferer"}, "sensitive": {"route": "overlap_detected"}}}]}
    result = study.summarize(manifest, actual)
    measured = result["decisions"]["p"]["overlapping_interferers"]
    assert measured["nominal:relation_unpaired:added_genuine"] == 1
    assert "corroborated:relation_unpaired:added_genuine" not in measured
    assert "oracle:relation_unpaired:added_genuine" not in measured
