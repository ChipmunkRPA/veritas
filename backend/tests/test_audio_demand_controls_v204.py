import numpy as np
import pytest

from scripts import audio_demand_controls_v204 as trial


@pytest.mark.parametrize("samples", [40000, 58400, 64000])
def test_equal_length_background_residuals_and_shared_gain(samples):
    index = np.arange(samples, dtype=np.float64)
    foreground = np.rint(22000 * np.sin(index / 24)).astype("<i2").tobytes()
    background = {
        "washing_room": np.rint(9800 * np.cos(index / 19)).astype("<i2").tobytes(),
        "river": np.rint(21000 * np.sin(index / 43)).astype("<i2").tobytes(),
    }
    prepared = trial.variants(foreground, background)
    assert tuple(prepared) == trial.CONDITIONS
    assert prepared["source_clean"] == (foreground, None, None, 1.0)
    control, _, _, gain = prepared["matched_clean"]
    assert 0 < gain < 1
    clean = np.frombuffer(control, dtype="<i2").astype(np.int32)
    for name in trial.TAPES:
        for level in trial.LEVELS:
            mixed, background, measured, gain_used = prepared[f"{name}_{level}db"]
            assert len(mixed) == len(background) == 2 * samples
            assert gain_used == gain
            assert measured == pytest.approx(level, abs=0.01)
            assert np.array_equal(
                np.frombuffer(mixed, dtype="<i2").astype(np.int32) - clean,
                np.frombuffer(background, dtype="<i2"),
            )


def test_wav_has_exact_variable_length_pcm(tmp_path):
    pcm = np.arange(-24000, 24000, dtype="<i2")
    assert trial.signed_pcm(pcm.astype(np.float32) / 32768) == pcm.tobytes()
    path = tmp_path / "test.wav"
    assert trial.save_wave(path, pcm.tobytes()) == trial.sha(path)
    assert trial.read_wave(path.read_bytes()) == pcm.tobytes()


def test_unavailable_input_retains_all_six_planned_conditions():
    cases = list(trial.cases([("genuine", {"status": "unavailable"})], {}))
    assert [row["condition"] for row, _ in cases] == list(trial.CONDITIONS)
    assert all(row["status"] == "unavailable" and pcm is None for row, pcm in cases)
