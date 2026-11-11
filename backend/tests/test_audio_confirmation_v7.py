from types import SimpleNamespace

import numpy as np
import pytest

from scripts import score_audio_confirmation_v7 as runner
from scripts.audio_confirmation_v7 import arm_metrics, paired, summarize
from scripts.prepare_audio_confirmation_variants_v7 import mixtures


def scored(identifier, label, p, q, decision="inconclusive", group=None):
    return {
        "id": identifier,
        "base_id": identifier,
        "group": group or identifier,
        "condition": "clean",
        "label": label,
        "generator": "human" if label == 0 else "ElevenLabs",
        "dataset": "test",
        "background_id": None,
        "original_p": p,
        "candidate_p": q,
        "original_decision": decision,
        "candidate_decision": decision,
        "status": "not_applicable" if p is None else "ok",
        "probe": {"status": "not_required"},
    }


def test_zero_false_original_is_not_zero_missed_ai():
    rows = [scored("a", 1, 0.9, 0.9, "ai"), scored("b", 1, 0.6, 0.6), scored("c", 1, None, None)]
    result = arm_metrics(rows, "candidate_p", "candidate_decision")
    assert result["false_original"] == 0
    assert result["ai_without_ai_conclusion"] == 2
    assert result["ai_without_ai_conclusion_rate"] == pytest.approx(2 / 3)
    assert result["unassessed_inputs"] == 1


def test_bootstrap_is_source_weighted_connected_and_order_deterministic():
    rows = [
        scored("real-a", 0, 0.4, 0.2),
        scored("real-b", 0, 0.2, 0.1),
        scored("ai-a", 1, 0.7, 0.9, group="shared-prompt"),
        scored("ai-b", 1, 0.8, 0.9, group="shared-prompt"),
    ]
    duplicated = {**rows[0], "id": "second-condition"}
    a = paired(rows + [duplicated], draws=50)
    b = paired(list(reversed(rows + [duplicated])), draws=50)
    assert a["strata"] == {"0:human": 2, "1:ElevenLabs": 1}
    for key in (
        "brier_difference",
        "log_loss_difference",
        "brier_difference_ci95",
        "log_loss_difference_ci95",
    ):
        assert a[key] == pytest.approx(b[key])
    assert a["brier_difference"] == pytest.approx(paired(rows, draws=50)["brier_difference"])


def test_incomplete_panel_cannot_pass_even_with_better_probabilities():
    rows = [scored("real", 0, 0.7, 0.01), scored("ai", 1, 0.3, 0.99)]
    result = summarize(rows)
    assert result["gates"]["brier_gain"]
    assert not result["gates"]["complete_cases"]
    assert not result["passes"]


def test_all_nine_mixtures_keep_common_gain_snr_and_bounded_peak():
    time = np.arange(64000) / 16000
    clean = np.sin(2 * np.pi * 241 * time)
    backgrounds = {
        "noise": np.random.default_rng(1).normal(size=64000),
        "music": np.sin(2 * np.pi * 431 * time),
        "speech": np.sin(2 * np.pi * 101 * time),
    }
    values, gain = mixtures(clean, backgrounds)
    assert len(values) == 9
    assert 0 < gain <= 1
    assert all(v.shape == (64000,) and np.max(np.abs(v)) <= 0.980001 for v in values.values())
    for kind in backgrounds:
        for snr in (5, 15) if kind == "music" else (0, 5, 15):
            measured = 20 * np.log10(
                np.sqrt(np.mean(values["clean"].astype(float) ** 2))
                / np.sqrt(np.mean((values[f"{kind}_{snr}"].astype(float) - values["clean"]) ** 2))
            )
            assert measured == pytest.approx(snr, abs=1e-5)
    assert np.array_equal(
        values["clean"], (clean * (0.1 / np.sqrt(np.mean(clean**2))) * gain).astype(np.float32)
    )


def test_silent_fixed_background_stops_instead_of_selecting_another_offset():
    with pytest.raises(ValueError, match="Silent/invalid fixed background"):
        mixtures(
            np.ones(64000), {"noise": np.zeros(64000), "music": np.ones(64000), "speech": np.ones(64000)}
        )


@pytest.fixture
def head_and_bundle(monkeypatch):
    guards = {str(c): [{"maximum": 0.2} for _ in range(10)] for c in (0, 1)}
    bundle = {
        "parameters": {"slope": 1.0, "intercept": 0.0},
        "guards": {str(c): [0.2] * 10 for c in (0, 1)},
        "alpha": 0.1,
        "min_seconds": 2.5,
        "max_seconds": 4.0,
    }
    monkeypatch.setattr(runner.serving, "load", lambda: bundle)
    return {"fitted": {"transform": None, "coefficients": [1.0, 0.0]}, "guards": guards}


class Activity:
    def __init__(self, spans):
        self.value = spans

    def spans(self, _pcm):
        return self.value


def test_skipped_speech_cannot_become_original_or_run_voice_model(monkeypatch, head_and_bundle):
    def unexpected(*_args):
        raise AssertionError("A skipped source must not run the classifier")

    monkeypatch.setattr(runner, "capture", unexpected)
    result = runner.score_one(np.ones(64000, np.float32), None, Activity([]), head_and_bundle, [])
    assert result["status"] == "not_applicable"
    assert result["original_p"] is result["candidate_p"] is None
    assert result["original_decision"] == result["candidate_decision"] == "inconclusive"


def test_failed_original_probe_withholds_both_original_tendencies(monkeypatch, head_and_bundle):
    calls = []

    def capture(*_args):
        calls.append(True)
        if len(calls) == 2:
            raise ValueError("Deliberately unavailable stability response")
        return 0.01, np.zeros(1024, np.float32)

    monkeypatch.setattr(runner, "capture", capture)
    result = runner.score_one(np.ones(64000, np.float32), None, Activity([[0, 4]]), head_and_bundle, [])
    assert len(calls) == 2
    assert result["probe"]["status"] == "failed"
    assert result["original_decision"] == result["candidate_decision"] == "inconclusive"


def test_new_stability_uses_own_probability_mapping_and_shares_exact_probe(monkeypatch, head_and_bundle):
    values = iter([0.01, 0.001])
    monkeypatch.setattr(runner, "capture", lambda *_args: (next(values), np.zeros(1024, np.float32)))
    head_and_bundle["fitted"]["coefficients"] = [0.2, -1.5]
    result = runner.score_one(np.ones(64000, np.float32), None, Activity([[0, 4]]), head_and_bundle, [])
    assert result["original_decision"] == "inconclusive"
    assert result["candidate_decision"] == "original"
    assert result["probe_needed_by_service"] and result["probe_needed_by_candidate"]
    assert result["probe"]["p"] != result["probe"]["raw"]


@pytest.mark.parametrize("spans", [[], [[0, 1.9]], [[0, 2.0]], [[1, 3]], [[0, 4]]])
def test_source_windows_match_actual_production_adapter(spans):
    from app.detectors.video.adapters.audio import PellaVoiceDetector
    from app.detectors.video.adapters.common import AudioTrack

    pcm = np.ones(64000, np.float32)
    activity = Activity(spans)
    ctx = SimpleNamespace(
        profile="full",
        outputs=SimpleNamespace(audio=AudioTrack("analysed", pcm, len(pcm), max_windows=48)),
        lease=SimpleNamespace(get=lambda _key: SimpleNamespace(obj=activity)),
    )
    actual = PellaVoiceDetector()._windows(ctx)
    _, proposed = runner.speech_windows(pcm, activity)
    assert [(s, len(v)) for s, v in actual] == [(s, len(v)) for s, v in proposed]
