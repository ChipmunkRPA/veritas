from __future__ import annotations

import shutil

import numpy as np
import pytest

from scripts import audio_native_stereo_pairs_v277 as pairs


def authored():
    index = np.arange(pairs.SAMPLES)
    foreground = np.rint(np.sin(index * 0.021) * 8000).astype("<i2")
    first = np.column_stack([np.sin(index * 0.077) * 1800, np.cos(index * 0.091) * 3600])
    second = np.column_stack([np.cos(index * 0.15) * 2400, np.sin(index * 0.104) * 1600])
    return foreground.tobytes(), {"washing_room": first.astype("<i2"), "river": second.astype("<i2")}


def test_matched_pcm_residual_and_fixed_snr():
    foreground, tapes = authored()
    forward = pairs.mix(foreground, tapes, 0)
    swapped = pairs.mix(foreground, tapes, 1)
    assert list(forward) == list(pairs.conditions(tapes))
    clean = forward["matched_clean"][0].astype(np.int32)
    assert np.array_equal(clean[:, 0], clean[:, 1])
    assert len({record["common_gain"] for _, record in forward.values()}) == 1
    for condition, (payload, record) in forward.items():
        assert record["residual_pcm_sha256"] == pairs.stereo.digest((payload - clean).astype("<i2").tobytes())
        assert np.max(np.abs(payload.astype(np.int32)).sum(axis=1)) <= pairs.PEAK_LIMIT + 2
        if condition.endswith("_equal_gain"):
            assert record["snr_db"]["joint_db"] == pytest.approx(-5, abs=0.1)
        elif condition.endswith("_forced"):
            assert record["snr_db"]["channels_db"] == pytest.approx([5, -10], abs=0.1)
            assert swapped[condition][1]["snr_db"]["channels_db"] == pytest.approx([-10, 5], abs=0.1)


def test_unavailable_source_preserves_all_five_cases(monkeypatch):
    def must_not_open(*_args):
        pytest.fail("Unqualified source was opened")

    monkeypatch.setattr(pairs.sources, "exact_source", must_not_open)
    rows = [{"source": {"slot": "test", "group_id": "example", "lane": "genuine", "status": "unavailable"},
             "tape_start_sample": 12, "forced_quieter_channel": 1}]
    cases = list(pairs.planned_cases(rows, "development", {"washing_room": {}, "river": {}}))
    assert len(cases) == 5
    assert all(pcm is None and row["status"] == "unavailable" and row["reason"] == "foreground_unavailable"
               for row, pcm in cases)


def test_disjoint_role_check_and_seal(tmp_path, monkeypatch):
    def source(slot, group, voice):
        return {"source": {"slot": slot, "group_id": group, "voice": voice}}

    with pytest.raises(ValueError, match="source identities overlap"):
        pairs.verify_disjoint({"development": [source("a", "one", "shared")],
                               "holdout": [source("b", "two", "shared")]})
    monkeypatch.setattr(pairs, "DEVELOPMENT_SEAL", tmp_path / "not-admitted.json")
    with pytest.raises(ValueError, match="before opening holdout audio"):
        pairs.check_development_seal()
    with pytest.raises(ValueError, match="All 185"):
        pairs.dev_admission({"case_outcomes": {"prepared": 184, "failed": 1}, "planned_cases": 185}, None, None)


@pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="FFmpeg is required")
def test_full_authored_forced_case_verifies_saved_and_production_bytes(tmp_path):
    pcm, tapes = authored()
    payload, _ = pairs.mix(pcm, tapes, 1)["river_forced"]
    row = pairs.save_case(tmp_path, "fixture", payload)
    assert row["status"] == "prepared"
    assert row["source_timeline"] == [{"start_sample": 0, "end_sample": 48_000, "source_pts": 0}]
    assert row["channel_pcm_sha256"] == [pairs.stereo.digest(payload[:, index].tobytes()) for index in (0, 1)]
    assert row["mixed_pcm_sha256"] != row["channel_pcm_sha256"][0]
    with pytest.raises(ValueError, match="Preserve every"):
        pairs.save_case(tmp_path, "fixture", payload)
