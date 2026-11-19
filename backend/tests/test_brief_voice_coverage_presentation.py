"""Coverage is a display of committed admissions, independent from authenticity."""
import copy
from collections import defaultdict
from types import SimpleNamespace

import pytest

from app.evidence import audio_brief_voice_binding as binding
from app.reports import engine
from app.reports.brief_voice_coverage import coverage


def fixture(schema=binding.SCHEMA):
    checks = [{"start_sample": start, "speech_samples": 6000 if i == 1 else 5999,
               "elevated": False if i == 1 else None} for i, start in enumerate(binding.GRID)]
    return ({"schema": schema, "status": "unassessed", "scheduled_windows": 23,
             "speech_windows_checked": 1, "window_checks": checks},
            {"samples": binding.SAMPLES, "sample_rate": binding.RATE, "timeline_time_base": "1/16000",
             "timeline": [{"start_sample": 0, "end_sample": binding.SAMPLES, "source_pts": 0}]})


@pytest.mark.parametrize("schema", [binding.SCHEMA, binding.IDENTICAL_STEREO_SCHEMA])
def test_recorded_windows_map_checked_union_gaps_and_uncovered_tail_without_mutation(schema):
    review, source = fixture(schema)
    original = copy.deepcopy(review)
    result = coverage(review, source)
    assert result["checked_samples"] == 12000
    assert result["intervals"] == [
        {"start_sample": 0, "end_sample": 8000, "status": "insufficient_speech"},
        {"start_sample": 8000, "end_sample": 20000, "status": "checked"},
        {"start_sample": 20000, "end_sample": 188000, "status": "insufficient_speech"},
        {"start_sample": 188000, "end_sample": 192000, "status": "outside_schedule"},
    ]
    assert len(result["windows"]) == 23
    assert result["windows"][1] == {"start_sample": 8000, "end_sample": 20000, "checked": True, "speech_samples": 6000}
    assert review == original
    review["window_checks"][2].update(speech_samples=12000, elevated=True)
    review["speech_windows_checked"] = 2
    result = coverage(review, source)
    assert result["checked_samples"] == 20000
    assert result["intervals"][1] == {"start_sample": 8000, "end_sample": 28000, "status": "checked"}


@pytest.mark.parametrize("mutation", ["skipped", "missing", "partial", "shifted", "invalid_admission", "wrong_count", "source_clock"])
def test_no_coverage_is_invented_for_unavailable_or_incomplete_recordings(mutation):
    review, source = fixture()
    if mutation == "skipped":
        review["status"] = "skipped"
    elif mutation == "missing":
        del review["window_checks"]
    elif mutation == "partial":
        review["window_checks"].pop()
    elif mutation == "shifted":
        review["window_checks"][0]["start_sample"] = 1
    elif mutation == "invalid_admission":
        review["window_checks"][0]["elevated"] = False
    elif mutation == "wrong_count":
        review["speech_windows_checked"] = 23
    else:
        source["timeline"][0]["source_pts"] = 1
    assert coverage(review, source) is None


@pytest.mark.parametrize("schema", [binding.SCHEMA, binding.IDENTICAL_STEREO_SCHEMA])
def test_standard_report_section_keeps_original_times_for_all_windows_and_tail(schema):
    review, source = fixture(schema)
    review.update(reason="No adjacent checked pair.", limitation=binding.LIMITATION, windows=[],
                  display_coverage=coverage(review, source))
    evidence = defaultdict(lambda: None, {"available": True, "kind": "audio", "title": "Audio evidence", "note": "Recorded coverage.",
                "version": "test", "sha256": "a" * 64, "coverage_note": "Source-clocked.", "coverage_details": [],
                "packets": [], "overview": [], "acoustic_contexts": [], "brief_voice_review": review, "source": source,
                "limitations": []})
    html = engine._env.get_template("sections/multimedia_evidence.html.j2").render(
        v={"evidence": evidence}, s=SimpleNamespace(report=SimpleNamespace(profile="standard")))
    assert "Where brief speech was checked" in html
    assert "0.750 of 12.000 source seconds" in html
    assert "Source PTS [8000, 20000) at 1/16000." in html
    assert "Source PTS [188000, 192000) at 1/16000." in html
    assert "Outside this check’s schedule" in html
    assert "limited or uncertain speech" in html
    assert "/brief_voice_review/window_checks" in html
    for start in binding.GRID:
        assert f"Source PTS [{start}, {start + binding.WIDTH}) at 1/16000." in html
    review.update(status="flagged", windows=[{"id": "B750-02", "start_sample": 8000, "end_sample": 20000,
        "start_s": 0.5, "end_s": 1.25, "waveform": None, "source_timestamp_ranges": [{
            "source_start_pts": 8000, "source_end_pts": 20000, "time_base": "1/16000"}]}])
    html = engine._env.get_template("sections/multimedia_evidence.html.j2").render(
        v={"evidence": evidence}, s=SimpleNamespace(report=SimpleNamespace(profile="standard")))
    assert "No usable local candidates were recorded" not in html
    review["display_coverage"] = None
    review["status"] = "skipped"
    review["windows"] = []
    html = engine._env.get_template("sections/multimedia_evidence.html.j2").render(
        v={"evidence": evidence}, s=SimpleNamespace(report=SimpleNamespace(profile="standard")))
    assert "No source-time coverage was recorded because this check did not complete." in html
    assert "All 23 scheduled windows" not in html
