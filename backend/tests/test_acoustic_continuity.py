"""Source binding and legacy compatibility for descriptive continuity leads."""

from copy import deepcopy

import numpy as np
import pytest
from test_audio_background import build, verify

from app.bench.audio_continuity import candidates as frozen_candidates
from app.detectors.audio.decode import read_wav
from app.evidence.audio_background_binding import validate_acoustic_context
from app.evidence.audio_continuity import candidates, select


def transition():
    time = np.arange(32000) / 16000
    return np.concatenate([0.15 * np.sin(2 * np.pi * 300 * time), 0.15 * np.sin(2 * np.pi * 3000 * time)])


def structural(fixture):
    document, _, _, source, mapping, packets = fixture
    validate_acoustic_context(document, source={**source, **mapping}, voice_packets=packets)


def test_new_transition_is_bound_to_exact_audio_and_surrounding_context():
    fixture = build(transition())
    doc, assets, pcm, *_ = fixture
    assert doc["schema"] == "chipmunk.acoustic-context/2"
    (event,) = [e for e in doc["events"] if e["category"] == "spectral_transition"]
    assert event["title"] == "Possible sound-content transition"
    assert event["change_sample"] == 32000
    assert event["interval"] == {"start_sample": 24000, "end_sample": 40000}
    assert event["evidence_type"] == "descriptive" and "not evidence of AI" in event["interpretation"]
    assert event["source_timestamp_ranges"][1]["source_start_pts"] == 64000
    assert len(event["observations"]) == 3
    np.testing.assert_array_equal(
        read_wav(assets[event["artifacts"]["selected_clip"]["sha256"]]), pcm[24000:40000]
    )
    structural(fixture)
    verify(fixture)


def test_old_acoustic_schema_remains_exactly_recomputable():
    fixture = build(transition(), version="mixed-acoustics-1")
    assert fixture[0]["schema"] == "chipmunk.acoustic-context/1"
    assert "continuity_windows" not in fixture[0] and fixture[0]["events"] == []
    structural(fixture)
    verify(fixture)


def test_production_measurement_matches_frozen_research_computation():
    rng = np.random.default_rng(37)
    for pcm in (transition(), rng.normal(0, 0.03, 64000), np.zeros(64000)):
        assert candidates(pcm) == frozen_candidates(pcm)


def test_additive_selection_retains_existing_descriptive_events():
    pcm = transition()
    pcm[32000:] *= 0.02
    fixture = build(pcm)
    assert {e["category"] for e in fixture[0]["events"]} == {"energy_change", "spectral_transition"}
    structural(fixture)
    verify(fixture)


@pytest.mark.parametrize("mutation", ["metric", "boundary", "missing", "source", "clip_bytes"])
def test_source_recompute_rejects_forged_transition_even_after_plausible_edits(mutation):
    fixture = build(transition())
    doc = fixture[0]
    event = next(e for e in doc["events"] if e["category"] == "spectral_transition")
    if mutation == "metric":
        event["observations"][0]["value"] = 0.8
        row = next(r for r in doc["continuity_windows"] if r["boundary_sample"] == event["change_sample"])
        row["spectral_distance"] = row["stationary_spectral_distance"] = 0.8
        structural(fixture)  # Plausible metadata still must match the recorded PCM.
    elif mutation == "boundary":
        event["change_sample"] += 4000
    elif mutation == "missing":
        doc["events"] = []
    elif mutation == "source":
        doc["source"] = deepcopy(doc["source"])
        doc["source"]["pcm_sha256"] = "f" * 64
    else:
        fixture[1][event["artifacts"]["selected_clip"]["sha256"]] = b"changed audio"
    with pytest.raises(ValueError, match="Acoustic context"):
        verify(fixture)


def test_transition_selection_is_bounded_and_distinct():
    rows = [dict(boundary_sample=i * 4000, stationary_spectral_distance=0.9) for i in range(1, 100)]
    selected = select(rows)
    assert len(selected) == 4
    assert all(
        abs(a["boundary_sample"] - b["boundary_sample"]) >= 16000
        for a in selected
        for b in selected
        if a is not b
    )
