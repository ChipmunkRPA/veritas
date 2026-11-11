from __future__ import annotations

import json

import numpy as np
import pytest

from scripts import audio_matched_background_panel_v248 as study


def fixture_sources():
    sources = {}
    for arm, level in zip(study.ARMS, (4000, 5000, 6000), strict=True):
        sources[arm] = []
        for i in range(study.TRIPLETS):
            qualified = i == 0 or i == 1 and arm != "genuine"
            pcm = (
                np.tile([level + i, -level - i], study.SAMPLES // 2).astype(np.float32) / 32768
                if qualified
                else None
            )
            group = "stock" if arm == "okur" else f"{arm}-group-{i:02d}"
            row = {"wav_sha256": f"{arm}-source-{i}"} if qualified else {}
            sources[arm].append(study.source_record(f"fixture-{arm}", i, group, row, pcm))
    return sources


def excerpt(band, slot, start):
    body = np.tile([420 + slot, -440, 300, -280], study.SAMPLES // 4).astype("<i2").tobytes()
    return {
        "environment": band,
        "slot": slot,
        "start_sample": start,
        "samples": study.SAMPLES,
        "pcm_sha256": study.digest(body),
        "qualification_receipt_sha256": "fixture",
    }, body


def fixture_bands():
    return {
        band: study.allocate([excerpt(band, slot, 0)]) for band, slot in zip(study.TAPES, (1, 2), strict=True)
    }


def test_recordings_interleave_before_reusing_any_cut():
    entries = [excerpt("kitchen", 1, 48_000), excerpt("kitchen", 1, 0), excerpt("kitchen", 2, 0)]
    assigned = study.allocate(entries)
    assert [(row["slot"], row["start_sample"]) for row, _ in assigned[:6]] == [
        (1, 0),
        (2, 0),
        (1, 48_000),
        (1, 0),
        (2, 0),
        (1, 48_000),
    ]
    assert study.tape_coverage({"kitchen": entries}, {"kitchen": assigned})["kitchen"] == {
        "provisional_excerpts_available": 3,
        "assigned_excerpts": 24,
        "assigned_distinct_recordings": 2,
        "assigned_distinct_cuts": 3,
        "assigned_distinct_excerpt_pcm": 2,
        "maximum_assignments_per_recording": 16,
    }


def test_pairing_preserves_exact_residual_and_missing_arm():
    sources = fixture_sources()
    outputs = list(study.cases(sources, fixture_bands()))
    assert len(outputs) == 360
    first = {(row["arm"], row["condition"]): (row, pcm) for row, pcm in outputs[:15]}
    for condition in study.CONDITIONS[1:]:
        target = int(condition.rsplit("_", 1)[1].removesuffix("db"))
        residuals, evidence = set(), []
        for arm in study.ARMS:
            row, payload = first[arm, condition]
            clean = first[arm, "matched_clean"][1]
            residual = np.frombuffer(payload, dtype="<i2").astype(np.int32) - np.frombuffer(
                clean, dtype="<i2"
            )
            residuals.add(residual.astype("<i2").tobytes())
            evidence.append(row["background"])
            assert abs(row["measured_snr_db"] - target) <= 0.1
            assert row["background"]["evidence_type"] == study.PROVISIONAL
        assert len(residuals) == 1 and all(row == evidence[0] for row in evidence)
        assert evidence[0]["quantized_residual_sha256"] == study.digest(next(iter(residuals)))
    rows = [row for row, _ in outputs]
    report = study.summarize(rows)
    assert report["pair_coverage"]["traffic_0db"] == {
        "planned_triplets": 24,
        "complete_triplets": 1,
        "genuine_alania": 1,
        "genuine_okur": 1,
        "alania_okur": 2,
    }
    assert study.source_coverage(sources)["okur"]["known_planned_voice_groups"] == 1
    with pytest.raises(ValueError, match="Every ordered"):
        study.summarize(rows[:-1])


@pytest.mark.parametrize("key", ("source_wav_sha256", "source_prefix_sha256"))
def test_previous_waveform_exposure_rejects_exact_repeat(key):
    sources = fixture_sources()
    row, _ = sources["genuine"][0]
    old = {"status": "prepared", "source_wav_sha256": "fresh", "source_prefix_sha256": "fresh"}
    study.check_exposure(sources, [old])
    with pytest.raises(ValueError, match="repeats a current or v217"):
        study.check_exposure(sources, [{**old, key: row[key]}])


def test_private_screen_receipt_stays_provisional(tmp_path, monkeypatch):
    packet_path = tmp_path / "private.json"
    monkeypatch.setattr(study, "PACKET", packet_path)
    row = excerpt("kitchen", 1, 0)[0]
    packet = {
        "schema": "chipmunk.audio-commons-slice-evidence/247",
        "background_result_sha256": "background",
        "evidence_type": study.PROVISIONAL,
        "human_listening_reviewed": False,
        "excerpts": [row],
    }
    packet_path.write_text(json.dumps(packet))
    aggregate = {
        "schema": "chipmunk.audio-commons-slice-screen-result/247",
        "status": "complete",
        "background_result_sha256": "background",
        "human_listening_reviewed": False,
        "private_admitted_slices_sha256": study.sha(packet_path),
    }
    assert study.evidence_packet({"screen": aggregate}, {"background": "background"}) == [row]
    packet_path.write_text(json.dumps({**packet, "human_listening_reviewed": True}))
    with pytest.raises(ValueError, match="evidence commitment"):
        study.evidence_packet({"screen": aggregate}, {"background": "background"})
    aggregate["private_admitted_slices_sha256"] = study.sha(packet_path)
    with pytest.raises(ValueError, match="machine-screened provisional"):
        study.evidence_packet({"screen": aggregate}, {"background": "background"})


def test_freeze_and_preparation_work_with_only_fixture_inputs(tmp_path, monkeypatch):
    trial, public = tmp_path / "fixture", tmp_path / "fixture-freeze.json"
    stage = trial / "bind-00"
    for key, path in {"TRIAL": trial, "PUBLIC": public, "STAGE": stage, "OUTPUT": stage / "output"}.items():
        monkeypatch.setattr(study, key, path)
    sources, assigned = fixture_sources(), fixture_bands()
    plan = {"foregrounds": {arm: [row for row, _ in items] for arm, items in sources.items()}}
    identity = {
        "foreground_coverage": study.source_coverage(sources),
        "background_coverage": study.tape_coverage(assigned, assigned),
    }
    monkeypatch.setattr(study, "context", lambda: (dict(identity), plan, sources, assigned))
    study.freeze()
    assert study.read(trial / "private-plan.json") == plan
    stage.mkdir()
    (stage / "resources.jsonl").touch()
    (stage / "worker.log").touch()
    study.worker()
    result = study.read(study.OUTPUT / "result.json")
    assert result["planned_cases"] == 360
    assert result["new_neural_calls"] == result["detector_calls"] == 0
    assert result["human_listening_reviewed"] is False
    assert len(list((study.OUTPUT / "prepared").glob("*.wav"))) == 25
    row = study.json_lines(study.OUTPUT / "private-cases.jsonl")[0]
    path, pcm = study.prepared_waveform(row)
    assert path.name == f"{row['case']}.wav" and study.digest(pcm) == row["pcm_sha256"]
    with pytest.raises(ValueError, match="row commitment"):
        study.prepared_waveform({**row, "pcm_sha256": "wrong"})
