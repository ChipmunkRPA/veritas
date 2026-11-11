from __future__ import annotations

import os
import subprocess
import sys

import numpy as np
import pytest

from scripts import audio_fresh_shared_noise_v217 as panel


def test_import_stays_model_free_in_a_fresh_process():
    env = {**os.environ, "PYTHONPATH": str(panel.ROOT / "backend"), "PYTHONDONTWRITEBYTECODE": "1"}
    result = subprocess.run([sys.executable, "-B", "-c", "from scripts import audio_fresh_shared_noise_v217 as p; p.model_free()"],
                            env=env, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr


def test_exact_first_three_seconds_and_tape_offsets():
    source = np.arange(50000, dtype=np.int16).astype(np.float32) / 32768
    prefix = panel.signed_prefix(source)
    assert prefix == np.arange(48000, dtype=np.int16).astype("<i2").tobytes()
    assert panel.offsets(0) == 4 * 16000 and panel.offsets(15) == 94 * 16000
    with pytest.raises(ValueError, match="fixed source roster"):
        panel.offsets(16)
    with pytest.raises(ValueError, match="Three seconds"):
        panel.signed_prefix(source[:47999])
    source[10] = np.nan
    with pytest.raises(ValueError, match="finite source"):
        panel.signed_prefix(source)


def test_triplets_share_residuals_and_keep_missing_arm_counts():
    n = panel.shared.SAMPLES
    waves = {arm: (np.tile([amplitude, -amplitude], n // 2).astype(np.float32) / 32768)
             for arm, amplitude in (("genuine", 5000), ("alania", 6000), ("drifting", 7000))}
    rosters = {arm: [{"status": "unavailable", "slot": f"{arm}-{i:02d}"} for i in range(16)] for arm in panel.shared.ARMS}
    for arm in panel.shared.ARMS:
        rosters[arm][0].update(status="qualified", wav_sha256=f"{arm}-0")
    for arm in ("alania", "drifting"):
        rosters[arm][1].update(status="qualified", wav_sha256=f"{arm}-1")
    readers = {arm: (lambda row, name=arm: waves[name]) for arm in panel.shared.ARMS}
    samples = 100 * panel.shared.SAMPLE_RATE
    channels = {"kitchen": np.tile([1500, -1500], samples // 2).astype("<i2").tobytes(),
                "traffic": np.tile([250, 10, -140, 90], samples // 4).astype("<i2").tobytes()}
    outputs = list(panel.cases(rosters, channels, readers))
    assert len(outputs) == 240
    rows = [row for row, _ in outputs]
    result = panel.summary(rows)
    for condition in panel.CONDITIONS:
        coverage = result["pair_coverage"][condition]
        assert coverage == {"planned_triplets": 16, "complete_triplets": 1,
                            "genuine_alania": 1, "genuine_drifting": 1, "alania_drifting": 2}
    first = {(r["arm"], r["condition"]): (r, body) for r, body in outputs[:15]}
    for condition in panel.CONDITIONS[1:]:
        residuals, excerpts, offsets = set(), set(), set()
        for arm in panel.shared.ARMS:
            row, mixed = first[arm, condition]
            clean = first[arm, "matched_clean"][1]
            residual = np.frombuffer(mixed, dtype="<i2").astype(np.int32) - np.frombuffer(clean, dtype="<i2").astype(np.int32)
            residuals.add(residual.astype("<i2").tobytes())
            excerpts.add(row["noise_excerpt_sha256"])
            offsets.add(row["noise_sample_offset"])
            assert row["samples"] == n and abs(row["measured_snr_db"] - int(condition.rsplit("_", 1)[1][:-2])) <= .1
        assert len(residuals) == len(excerpts) == len(offsets) == 1
    assert result["by_arm_and_condition"]["genuine"]["matched_clean"] == {"prepared": 1, "unavailable": 15}
    assert result["by_arm_and_condition"]["alania"]["matched_clean"] == {"prepared": 2, "unavailable": 14}


def test_summary_rejects_omitted_and_shuffled_case():
    rosters = {arm: [{"status": "unavailable"} for _ in range(16)] for arm in panel.shared.ARMS}
    channels = {name: b"\0\0" * (100 * 16000) for name in panel.shared.TAPES}
    rows = [row for row, _ in panel.cases(rosters, channels, {arm: lambda _: None for arm in panel.shared.ARMS})]
    with pytest.raises(ValueError, match="Every planned"):
        panel.summary(rows[:-1])
    rows[0], rows[1] = rows[1], rows[0]
    with pytest.raises(ValueError, match="ordered triplet"):
        panel.summary(rows)


def test_wrong_publisher_record_is_rejected_before_tape_read(tmp_path, monkeypatch):
    record = tmp_path / "publisher.json"
    record.write_text("{}")
    monkeypatch.setattr(panel, "ASSET_RECORD", record)
    with pytest.raises(ValueError, match="Exact published DEMAND"):
        panel.publisher_assets()
