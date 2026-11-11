from __future__ import annotations

import hashlib

import numpy as np
import pytest

from scripts import audio_demand_crossover_v210 as study


def test_fixed_offsets_cross_the_original_strata_without_overlaps():
    assert study.offsets(0) == {"early": 4 * study.prior.RATE, "late": 148 * study.prior.RATE}
    assert study.offsets(23) == {"early": 142 * study.prior.RATE, "late": 286 * study.prior.RATE}
    starts = [offset for i in range(24) for offset in study.offsets(i).values()]
    assert len(set(starts)) == 48 and all(
        b - a >= 4 * study.prior.RATE for a, b in zip(sorted(starts), sorted(starts)[1:])
    )
    for bad in (-1, 24, True, 0.5):
        with pytest.raises(ValueError, match="fixed roster"):
            study.offsets(bad)


def test_pairing_retains_fixed_roster_order_and_rejects_swapped_classes():
    sources = [("genuine", {"slot": i}) for i in range(24)] + [("synthetic", {"slot": i}) for i in range(24)]
    assert [(left["slot"], right["slot"]) for left, right in study.paired_sources(sources)] == [
        (i, i) for i in range(24)
    ]
    sources[1] = ("synthetic", sources[1][1])
    with pytest.raises(ValueError, match="ordered v204"):
        study.paired_sources(sources)


def test_shared_noise_prefix_and_gain_matched_clean_control(monkeypatch):
    rate, n = study.prior.RATE, study.prior.MIN_SAMPLES
    x = np.arange(n + 8000, dtype=np.float64)
    genuine = np.rint(6500 * np.sin(x[:n] * 0.03)).astype(np.int16).astype(np.float32) / 32768
    synthetic = np.rint(4500 * np.sin(x * 0.04)).astype(np.int16).astype(np.float32) / 32768
    rng = np.random.default_rng(12)
    river = rng.integers(-9000, 9001, size=rate * 290, dtype=np.int16).astype("<i2").tobytes()
    g = {"slot": "genuine-00", "status": "qualified", "wav_sha256": "g"}
    s = {"slot": "synthetic-00", "status": "qualified", "wav_sha256": "s"}
    sources = [("genuine", g)] + [("genuine", {"status": "unavailable"}) for _ in range(23)]
    sources += [("synthetic", s)] + [("synthetic", {"status": "qualified"}) for _ in range(23)]
    monkeypatch.setattr(study.prior.genuine, "read_prepared_pcm", lambda _: genuine)
    monkeypatch.setattr(study.prior.synthetic, "read_pcm", lambda _: synthetic)
    output = list(study.cases(sources, river))
    assert len(output) == 144 and sum(record["status"] == "prepared" for record, _ in output) == 6
    records = {(row["origin"], row["condition"]): (row, pcm) for row, pcm in output[:6]}
    for origin in ("genuine", "synthetic"):
        clean, clean_pcm = records[origin, "matched_clean"]
        assert clean["samples"] == n and len(clean_pcm) == 2 * n
        for which in ("early", "late"):
            row, pcm = records[origin, f"river_{which}_0db"]
            assert row["samples"] == n and row["gain"] == clean["gain"] and abs(row["measured_snr_db"]) <= 0.1
            actual = np.frombuffer(pcm, dtype="<i2").astype(np.int32) - np.frombuffer(
                clean_pcm, dtype="<i2"
            ).astype(np.int32)
            assert (
                hashlib.sha256(actual.astype("<i2").tobytes()).hexdigest()
                == row["background_residual_sha256"]
            )
    for which in ("early", "late"):
        left = records["genuine", f"river_{which}_0db"][0]
        right = records["synthetic", f"river_{which}_0db"][0]
        start = study.offsets(0)[which]
        assert left["background_sample_offset"] == right["background_sample_offset"] == start
        assert (
            left["background_pcm_sha256"]
            == right["background_pcm_sha256"]
            == hashlib.sha256(river[2 * start : 2 * (start + n)]).hexdigest()
        )
    assert all(row["reason"] == "pair_unavailable" for row, _ in output[6:])


def test_short_noise_excerpt_is_rejected():
    clean = b"\x01\x00" * study.prior.MIN_SAMPLES
    with pytest.raises(ValueError, match="common prefix"):
        study.mix_foreground(clean, {"early": clean, "late": clean[:-2]})
