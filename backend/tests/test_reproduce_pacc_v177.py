from __future__ import annotations

import json
from contextlib import nullcontext

import pytest

from scripts import reproduce_pacc_v177 as replay


def sample_plan():
    plan = []
    for pool, count in replay.source.POOLS.items():
        for n in range(count):
            family = "AMI" if pool == "AMI" else "VCTK"
            plan.append({"slot": f"source-{len(plan):02d}", "pool": pool, "family": family,
                         "source_context": f"{family}-{n}", "source_rate": 16000,
                         "members": {condition: "sample.wav" for condition in replay.source.CONDITIONS}})
    return plan


def test_frozen_public_scoring_sources_still_match():
    replay.verify_code()


def test_bad_publisher_pin_rejected_before_decoding(tmp_path, monkeypatch):
    archive = tmp_path / "pacc-p_sample.tar"
    archive.write_bytes(b"wrong")
    monkeypatch.setattr(replay.source, "SAMPLE_SIZE", 5)
    called = []
    monkeypatch.setattr(replay.source, "prepare", lambda *args: called.append(True))
    with pytest.raises(ValueError, match="Pinned publisher sample"):
        replay.source_files(tmp_path)
    assert not called


def test_qualification_preserves_all_planned_cases_and_is_private(tmp_path, monkeypatch):
    publisher, output = tmp_path / "publisher", tmp_path / "output"
    publisher.mkdir()
    output.mkdir()
    monkeypatch.setattr(replay, "verify_code", lambda: None)
    monkeypatch.setattr(replay, "source_plan", lambda _: sample_plan())
    monkeypatch.setattr(replay, "environment", lambda: {})
    monkeypatch.setattr(replay.tarfile, "open", lambda *args, **kwargs: nullcontext(None))
    monkeypatch.setattr(replay.source, "member", lambda *args: b"sample")
    calls = []

    def prepare(_incoming, outgoing, _raw, _rate):
        calls.append(outgoing.name)
        if len(calls) == 2:
            raise ValueError("decode unavailable")
        outgoing.write_bytes(b"wav")
        return {"status": "qualified", "wav_sha256": replay.source.sha(outgoing), "pcm_sha256": "pcm"}

    monkeypatch.setattr(replay.source, "prepare", prepare)
    replay.qualify(publisher, output)
    qualification = json.loads((output / "qualification.json").read_text())
    assert qualification["status_counts"] == {"qualified": 97, "unavailable": 1}
    assert qualification["neural_calls"] == 0 and not qualification["historical_catalogue_checked"]
    assert qualification["plan_sha256"] == replay.source.sha(output / "plan.private.json")
    assert qualification["observations_sha256"] == replay.source.sha(output / "qualification.private.json")
    assert len(calls) == 98 and output.stat().st_mode & 0o077 == 0
    with pytest.raises(ValueError, match="empty private directory"):
        replay.qualify(publisher, output)


def test_comparison_counts_unavailable_and_rejects_missing_case():
    plan = sample_plan()
    rows = [{"slot": item["slot"], "family": item["family"], "condition": condition,
             "decision": "unavailable", "route_status": "preparation_unavailable"}
            for item in plan for condition in replay.source.CONDITIONS]
    comparison = replay.compare(rows, plan)
    assert comparison["VCTK"]["by_condition"]["clean"]["directions"]["unavailable"] == 10
    assert comparison["AMI"]["paired_vs_clean"]["babble_snr00"]["planned_pairs"] == 4
    assert not comparison["VCTK"]["by_condition"]["clean"]["matches_reference"]
    with pytest.raises(ValueError, match="ordered result"):
        replay.compare(rows[:-1], plan)


def test_scoring_refuses_changed_qualification_before_loading_models(tmp_path, monkeypatch):
    monkeypatch.setattr(replay, "verify_code", lambda: None)
    (tmp_path / "plan.private.json").write_text("[]")
    (tmp_path / "qualification.private.json").write_text("[]")
    (tmp_path / "qualification.json").write_text(json.dumps({"source_sha256": replay.source.SAMPLE_SHA256,
        "planned_cases": 98, "plan_sha256": "wrong", "observations_sha256": "wrong"}))
    with pytest.raises(ValueError, match="Qualification commitments differ"):
        replay.run(tmp_path, tmp_path / "models")
    assert not (tmp_path / "scores.private.json").exists()
