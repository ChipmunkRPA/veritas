import hashlib
import types

import numpy as np
import pytest

from app.detectors.audio.decode import wav_bytes
from app.engine.boundary import Deadline
from scripts import audio_matched_edit_diagnostic_v174 as diagnostic


def test_prepared_layout_keeps_both_sides_and_rejects_unmatched_residual(monkeypatch):
    group = {
        "comparison_id": "comparison-002",
        "role": "development",
        "family": "ai",
        "lane": "clone",
        "genuine_host_slot": "host",
        "noise_slot": "noise",
        "planned_cases": 21,
    }
    rows = [
        {
            **row,
            "status": "constructed",
            "gain": 0.9,
            "shared_noise_pcm_sha256": None if row["condition"] == "clean" else row["condition"],
            "matched_host_pcm_sha256": "same_host_pcm",
        }
        for row in diagnostic.planned_rows([group])
    ]
    monkeypatch.setattr(diagnostic, "CASE_COUNT", 18)
    selected = diagnostic.select_matched_rows(rows, [group])
    assert len(selected) == 18
    assert len(list(diagnostic.pairs(selected))) == 9
    assert all(row["arm"].startswith("genuine_to_") for row in selected)
    rows[4]["shared_noise_pcm_sha256"] = "different_residual"
    with pytest.raises(ValueError, match="Matched PCM treatment differs"):
        diagnostic.select_matched_rows(rows, [group])


def test_reads_exact_production_pcm_and_rejects_out_of_scope_wav(tmp_path, monkeypatch):
    development = tmp_path / "development"
    development.mkdir()
    monkeypatch.setattr(diagnostic, "DEVELOPMENT", development)
    pcm = np.arange(48000, dtype=np.float32) / np.float32(32768) - 0.75
    data = wav_bytes(pcm)
    path = development / "comparison-001-03.wav"
    path.write_bytes(data)
    raw = (pcm * 32768).astype("<i2").tobytes()
    row = {
        "case": path.stem,
        "role": "development",
        "status": "constructed",
        "file": str(path),
        "wav_sha256": hashlib.sha256(data).hexdigest(),
        "production_f32_pcm_sha256": hashlib.sha256(pcm.astype("<f4").tobytes()).hexdigest(),
        "pcm_s16le_sha256": hashlib.sha256(raw).hexdigest(),
        "samples": len(pcm),
        "sample_rate": 16000,
    }
    production, exact = diagnostic.read_development_pcm(row)
    assert np.array_equal(production, pcm) and exact == raw
    with pytest.raises(ValueError, match="Signed16 PCM differs"):
        diagnostic.read_development_pcm({**row, "pcm_s16le_sha256": "incorrect"})
    with pytest.raises(ValueError, match="Only an admitted development"):
        diagnostic.read_development_pcm({**row, "role": "holdout"})
    outside = tmp_path / "out-of-scope.wav"
    path.rename(outside)
    path.symlink_to(outside)
    with pytest.raises(ValueError, match="Only a direct development file"):
        diagnostic.read_development_pcm(row)


@pytest.mark.parametrize(
    ("scores", "speech", "decision", "probe", "count"),
    [
        ([0.99999], [(0, 3)], "ai", "not_required_for_decision", 1),
        ([0.5], [(0, 3)], "inconclusive", "not_required_for_decision", 1),
        ([0.00001, 0.00001], [(0, 3)], "original", "computed", 2),
        ([0.00001, 0.5], [(0, 3)], "inconclusive", "computed", 2),
        ([], [], "unavailable", "not_run", 0),
    ],
)
def test_serving_band_runs_only_for_existing_original_veto(scores, speech, decision, probe, count):
    pcm = np.random.default_rng(174).normal(0.0, 0.05, 48000).astype(np.float32)
    predictions = iter(scores)
    views = []

    def predict(chunks):
        (chunk,) = chunks
        views.append(chunk)
        return [next(predictions)]

    result = diagnostic.score_voice(
        pcm,
        types.SimpleNamespace(predict=predict),
        types.SimpleNamespace(spans=lambda _: speech),
        Deadline(10),
    )
    assert result["decision"] == decision
    assert result["stability_probe"] == probe
    assert len(views) == count
    if count:
        assert np.array_equal(views[0], pcm)
    if count == 2:
        assert not np.array_equal(views[1], pcm)
    diagnostic.check_forwards(result, count)


def test_metrics_keep_pair_denominators_and_balance_hosts():
    def block(index, host, noise, outcomes):
        rows = []
        for i, (arm, (decision, repeat)) in enumerate(zip(diagnostic.ARMS, outcomes, strict=True)):
            rows.append(
                {
                    "case": f"comparison-{index:03}-{i:02}",
                    "comparison_id": str(index),
                    "genuine_host_slot": host,
                    "noise_slot": noise,
                    "lane": "clone",
                    "arm": f"{arm}_hard",
                    "join_ms": 0,
                    "condition": "snr_0db",
                    "route_status": "ok" if decision != "unavailable" else "not_applicable",
                    "decision": decision,
                    "repeat_abstention": "inconclusive" if decision == "ai" and repeat else decision,
                    "cues": {"exact": {"status": "measured", "spans": [[1, 2]] if repeat else []}},
                }
            )
        return rows

    rows = [
        *block(0, "shared", "A", [("ai", True), ("ai", True)]),
        *block(1, "shared", "B", [("ai", False), ("ai", False)]),
        *block(2, "other", "B", [("original", False), ("unavailable", False)]),
    ]
    summary = diagnostic.summary(rows, rows)
    slice_ = summary["by_donor_family_noise_condition_join_ms"]["clone"]["snr_0db"]["0"]
    assert slice_["planned_pairs"] == 3
    assert slice_["both_scored"] == 2
    assert slice_["ordinary"]["baseline"]["false_ai"] == 2
    assert slice_["ordinary"]["repeat_abstention"]["false_ai"] == 1
    assert slice_["synthetic"]["baseline"]["missed_ai"] == 1
    assert slice_["synthetic"]["repeat_abstention"]["missed_ai"] == 2
    assert slice_["synthetic"]["repeat_abstention"]["unavailable"] == 1
    assert slice_["synthetic"]["ai_to_inconclusive"] == 1
    assert summary["descriptive_host_balanced"]["overall"]["baseline"]["ordinary_false_ai_rate"] == 0.5
    omissions = summary["descriptive_leave_one_noise_parent_out"]
    assert [row["by_noise_condition"]["snr_0db"]["planned_pairs"] for row in omissions] == [2, 1]


def test_unavailable_construction_never_opens_wav_or_scores(monkeypatch):
    def never_called(*_):
        pytest.fail("Unavailable case cannot read or score")

    monkeypatch.setattr(diagnostic, "read_development_pcm", never_called)
    result = diagnostic.assess({"status": "unavailable"}, never_called)
    assert result["decision"] == result["repeat_abstention"] == "unavailable"
    assert result["cues"] == {}
    diagnostic.check_forwards(result, 0)
