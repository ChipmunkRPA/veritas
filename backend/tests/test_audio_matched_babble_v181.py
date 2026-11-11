import io
import wave

import numpy as np
import pytest

from scripts import audio_matched_babble_v181 as lab


def tone(hz, gain=6000):
    clock = np.arange(lab.SAMPLES) / lab.RATE
    return np.rint(gain * np.sin(2 * np.pi * hz * clock)).astype("<i2").tobytes()


def pair():
    return {"pair": "pair-00", "component": "test-group", "pool": "pool-00"}


def test_pool_ranking_is_order_invariant_and_disjoint():
    candidates = [
        {
            "voice": f"vctk:p{voice}",
            "original_sha256": lab.digest(f"{voice}-{utterance}".encode()),
            "prior_pcm_sha256": lab.digest(f"pcm-{voice}-{utterance}".encode()),
        }
        for voice in range(30)
        for utterance in range(2)
    ]
    pools = lab.babble_pools(candidates)
    assert pools == lab.babble_pools(reversed(candidates))
    assert all(len(talkers) == 6 for talkers in pools.values())
    assert len({row["voice"] for talkers in pools.values() for row in talkers}) == 24
    assert lab.voice_id("P225") == lab.voice_id("vctk:p225")


def test_paired_babble_preserves_exact_shared_signed16_residual_and_labels(tmp_path):
    background = lab.make_babble([tone(180 + 43 * k, 3000 * (k + 1)) for k in range(6)])
    foregrounds = {"genuine": tone(421, 27000), "synthetic": tone(538, 18000)}
    rendered = list(lab.cases(pair(), foregrounds, background))
    assert len(rendered) == 6
    assert len({row["gain"] for row, _ in rendered}) == 1
    assert rendered[0][0]["gain"] < 1
    by_arm = {(row["arm"], row["condition"]): (row, payload) for row, payload in rendered}
    for arm in lab.ARMS:
        for condition, _ in lab.LEVELS:
            row, payload = by_arm[arm, condition]
            assert row["label"] == int(arm == "synthetic")
            assert row["ai_support_samples"] == ([0, lab.SAMPLES] if arm == "synthetic" else None)
            assert row["status"] == "constructed"
            receipt = lab.write_case(tmp_path / f"{row['case']}.wav", payload)
            assert receipt["samples"] == lab.SAMPLES
            assert receipt["pcm_s16le_sha256"] == lab.digest(payload)
    for condition, nominal in lab.LEVELS[1:]:
        residual = []
        for arm in lab.ARMS:
            row, mixed = by_arm[arm, condition]
            _, clean = by_arm[arm, "clean"]
            change = np.frombuffer(mixed, dtype="<i2").astype(np.int32) - np.frombuffer(clean, dtype="<i2")
            residual.append(change.tobytes())
            assert row["shared_noise_pcm_sha256"] == lab.digest(change.astype("<i2").tobytes())
        assert residual[0] == residual[1]
        assert by_arm["genuine", condition][0]["measured_snr_db"] == pytest.approx(nominal, abs=0.01)
        assert by_arm["synthetic", condition][0]["measured_snr_db"] < nominal


def test_missing_source_or_babble_keeps_all_planned_denominators():
    foregrounds = {"genuine": tone(330)}
    rows = list(lab.cases(pair(), foregrounds, None))
    assert len(rows) == 6
    assert [row["case"] for row, pcm in rows if pcm is not None] == ["pair-00-genuine-clean"]
    assert {row["reason"] for row, pcm in rows if pcm is None} == {"paired_foreground_unavailable"}
    rows = list(lab.cases(pair(), {**foregrounds, "synthetic": tone(720)}, None))
    assert sum(pcm is not None for _, pcm in rows) == 2
    assert {row["reason"] for row, pcm in rows if pcm is None} == {"fixed_babble_pool_unavailable"}
    with pytest.raises(lab.controls.UnsupportedControl, match="silent_or_constant"):
        lab.make_babble([tone(400)] * 5 + [bytes(2 * lab.SAMPLES)])


def test_production_prefix_rejects_protected_exact_pcm_and_holdout_paths(tmp_path, monkeypatch):
    bank = tmp_path / "sources"
    bank.mkdir()
    monkeypatch.setattr(lab, "VCTK", tmp_path)
    raw = tone(330)
    memory = io.BytesIO()
    with wave.open(memory, "wb") as wav:
        wav.setparams((1, 2, lab.RATE, lab.SAMPLES, "NONE", "not compressed"))
        wav.writeframes(raw)
    full = memory.getvalue()
    filename = bank / "talker.wav"
    filename.write_bytes(full)
    row = {"source_path": str(filename), "source_bytes": len(full), "original_sha256": lab.digest(full)}
    pcm, evidence, fingerprints = lab.production_prefix(row, "babble", set())
    assert pcm == raw
    assert evidence["source_range"]["source_start_pts"] == 0
    assert evidence["source_range"]["source_end_pts"] == lab.SAMPLES
    assert lab.float_pcm_sha(raw) in fingerprints
    with pytest.raises(ValueError, match="Known source byte or exact PCM overlap"):
        lab.production_prefix(row, "babble", {lab.float_pcm_sha(raw)})
    holdout = tmp_path / "holdout.wav"
    holdout.write_bytes(full)
    with pytest.raises(ValueError, match="admitted development bank"):
        lab.production_prefix({**row, "source_path": str(holdout)}, "babble", set())
