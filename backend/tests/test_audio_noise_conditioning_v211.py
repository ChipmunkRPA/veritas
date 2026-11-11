"""Exercise v211's fixed DSP and accounting without neural assets."""

import subprocess
import sys

import numpy as np
import pytest

from scripts import audio_noise_conditioning_v211 as study


def test_cold_import_is_model_and_settings_free():
    subprocess.run(
        [
            sys.executable,
            "-B",
            "-c",
            "from scripts import audio_noise_conditioning_v211 as v; v.PILOT.require_model_free()",
        ],
        check=True,
    )


def test_unity_overlap_add_retains_every_original_sample_and_impulse_position():
    rng = np.random.default_rng(16000)
    pcm = rng.normal(0, 0.1, 48001).astype(np.float32)
    pcm[[0, 2441, -1]] = [0.75, -0.5, 0.4]
    bins, padded = study.spectrum(pcm)
    output = study.reconstruct(bins, padded, len(pcm))
    assert output.dtype == np.float32
    assert np.array_equal(pcm, output)


def test_fixed_gain_is_bounded_and_conditioning_keeps_exact_non_aligned_length():
    x = np.arange(48121, dtype=np.float64)
    pcm = (0.2 * np.sin(2 * np.pi * 400 * x / study.RATE)).astype(np.float32)
    bins, _ = study.spectrum(pcm)
    mask = study.spectral_gain(np.abs(bins) ** 2)
    assert np.all((mask >= study.MIN_GAIN) & (mask <= 1))
    result = study.condition(pcm)
    repeated = study.condition(pcm)
    assert result.status == "ready"
    assert result.pcm.shape == pcm.shape and result.pcm.dtype == pcm.dtype
    assert np.array_equal(result.pcm, repeated.pcm)
    assert result.mask_sha256 == repeated.mask_sha256
    assert np.sqrt(np.mean(result.pcm**2)) < np.sqrt(np.mean(pcm**2))


def test_silence_constant_clipping_and_identity_are_explicitly_unavailable(monkeypatch):
    for level in (0, 0.1):
        result = study.condition(np.full(48000, level, dtype=np.float32))
        assert (result.status, result.reason, result.pcm) == ("unavailable", "silent_or_constant", None)
    with pytest.raises(ValueError, match="canonical"):
        study.condition(np.ones(47999, dtype=np.float64))
    with pytest.raises(ValueError, match="canonical"):
        study.condition(np.full(48000, np.nan, dtype=np.float32))
    rng = np.random.default_rng(22)
    pcm = rng.uniform(-0.25, 0.25, 48000).astype(np.float32)
    monkeypatch.setattr(study, "spectral_gain", np.ones_like)
    assert study.condition(pcm).reason == "identical_view"
    monkeypatch.setattr(
        study, "reconstruct", lambda _bins, _padded, size: np.full(size, 1.01, dtype=np.float32)
    )
    assert study.condition(pcm).reason == "conditioned_clipping"


def test_case_preserves_unchanged_window_direction_and_original_parity(monkeypatch):
    original = np.arange(48000, dtype=np.float32) / 48000
    conditioned = original * np.float32(0.75)
    monkeypatch.setattr(study, "waveform", lambda _: original)
    monkeypatch.setattr(
        study, "condition", lambda _: study.View("ready", pcm=conditioned, mask_sha256="mask")
    )
    case = {"case": "c", "source": "s", "condition": "river_0db", "origin": "synthetic"}
    baseline = {**case, "route_status": "ok", "decision": "inconclusive", "raw": 0.5, "windows": [[0, 48000]]}
    observed = []

    def score(values):
        observed.append(values)
        return 0.5 if values is original else 0.85

    def initial(raw):
        return "ai" if raw > 0.8 else "inconclusive"

    row = study.build_case(case, baseline, True, score, initial) | {"pella_forwards": 2}
    study.audit_case(row, case, baseline, True, initial)
    assert observed[0] is original and observed[1] is conditioned
    assert row["recorded_direction"] == "inconclusive"
    assert row["one_way_ai_rescue"] == "ai"
    assert row["window"] == baseline["windows"]
    with pytest.raises(ValueError, match="paired diagnostic"):
        study.audit_case(row | {"conditioned_gain_sha256": "different"}, case, baseline, True, initial)
    with pytest.raises(ValueError, match="Original-view numerical replay"):
        study.build_case(case, baseline, True, lambda _: 0.1, initial)
    rejected = baseline | {
        "route_status": "not_applicable",
        "decision": "unavailable",
        "raw": None,
        "windows": [],
    }
    skipped = study.build_case(case, rejected, False, score, initial) | {"pella_forwards": 0}
    study.audit_case(skipped, case, rejected, False, initial)
    assert len(observed) == 2 and skipped["one_way_ai_rescue"] == "unavailable"
    assert study.rescue("original", "ai") == "original"
    assert study.rescue("ai", "original") == "ai"


def test_ranks_use_common_assessed_inputs_and_exact_score_ties():
    rows = []
    for origin, base, candidate, before, after in (
        ("genuine", 0.4, 0.7, "inconclusive", "ai"),
        ("genuine", None, None, "unavailable", "unavailable"),
        ("synthetic", 0.4, 0.7, "inconclusive", "ai"),
        ("synthetic", 0.3, 0.8, "original", "ai"),
    ):
        rows.append(
            {
                "origin": origin,
                "recorded_raw": base,
                "conditioned_raw": candidate,
                "recorded_direction": before,
                "recorded_initial_direction": before,
                "conditioned_diagnostic_direction": after,
                "conditioned_status": "ready" if candidate is not None else "unavailable",
                "conditioned_reason": None if candidate is not None else "preparation_unavailable",
                "recorded_route_status": "ok" if base is not None else "preparation_unavailable",
                "one_way_ai_rescue": study.rescue(before, after),
            }
        )
    compared = study.comparison(rows)
    ranks = compared["raw_rank_on_common_assessed"]
    assert ranks["recorded"]["auc"] == 0.25
    assert ranks["conditioned"]["auc"] == 0.75
    assert ranks["conditioned"]["ties"] == 1
    assert ranks["conditioned"]["genuine_sources"] == 1
    genuine, synthetic = (compared["by_origin"][name] for name in study.ORIGINS)
    assert genuine["post_hoc_new_false_ai"] == 1
    assert genuine["post_hoc_one_way_ai_rescue_directions"]["unavailable"] == 1
    assert synthetic["post_hoc_ai_recoveries"] == 1
    assert synthetic["post_hoc_one_way_ai_rescue_directions"]["original"] == 1
