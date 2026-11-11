from __future__ import annotations

from copy import deepcopy

import pytest

from scripts import audio_ami_stereo_transfer_v293 as study


def roster():
    return [{"session": session, "case": f"{session}-{index:02d}", "source_samples": 8_000_000,
             "source_range": {"start_sample": start, "end_sample": start + study.ami.SAMPLES, "time_base": "1/16000"},
             "source_timeline": study.stereo.TIMELINE, "channel_pcm_sha256": ["left", "right"],
             "stereo_wav_sha256": "stereo", "mixed_pcm_sha256": "mono"}
            for session in study.ami.SESSIONS for index, start in enumerate(study.ami.WINDOWS)]


def test_case_binding_preserves_both_sessions_and_exact_sample_coordinates():
    rows = roster()
    cases = study.bind_rows(rows)
    assert len(cases) == 24 and len({row["source_group"] for row in cases}) == 2
    assert all(row["lane"] == "genuine" and row["condition"] == row["session"] and row["wav_sha256"] == "stereo" for row in cases)
    altered = deepcopy(rows)
    altered[0]["source_range"]["start_sample"] += 1
    with pytest.raises(ValueError, match="Exact ordered AMI source intervals"):
        study.bind_rows(altered)
    with pytest.raises(ValueError, match="All 24"):
        study.bind_rows(rows[:-1])


def test_genuine_summary_retains_all_quiet_cases_without_synthetic_metrics():
    cases = study.bind_rows(roster())
    observed = [{"case": row["case"], "selected": None, "swap_invariant": True,
                 "admitted": {"mono": False, "ch0": False, "ch1": False}} for row in cases]
    assert not study.views_for(cases, observed)
    counts = study.previous.decision_report(cases, [], [])
    overall = study.genuine_summary(counts["all"]["all"])
    assert overall == {name: {"planned": 24, "assessed": 0, "false_ai": 0, "original": 0, "inconclusive": 0, "unassessed": 24}
                       for name in ("mono", "ai_only_rescue")}
    assert all(study.genuine_summary(counts["condition"][session])["mono"]["unassessed"] == 12 for session in study.ami.SESSIONS)
