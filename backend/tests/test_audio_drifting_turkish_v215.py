from collections import Counter

import numpy as np
import pytest

from scripts import audio_drifting_turkish_v215 as lab


def sample_plan():
    words = (
        "sabah",
        "akşam",
        "gündüz",
        "gece",
        "evde",
        "köyde",
        "yolda",
        "işte",
        "yazın",
        "kışın",
        "güzün",
        "bugün",
        "yarın",
        "dün",
        "hemen",
        "sonra",
    )
    return lab.roster([f"Ilık çay ile İstanbul yoluna {word} çıktık." for word in words])


def test_fixed_stock_order_and_turkish_token_contract():
    plan = sample_plan()
    assert Counter(row["voice"] for row in plan) == {"studio": 6, "male": 5, "female": 5}
    assert [row["seed"] for row in plan] == list(range(215000, 215016))
    ids = lab.token_ids("Ilık çay ile İstanbul yoluna çıktık.")
    assert ids.dtype == np.int64
    assert ids[0, :21].tolist() == [1, 12, 1, 16, 1, 12, 1, 15, 1, 34, 1, 5, 1, 2, 1, 32, 1, 34, 1, 13, 1]
    assert np.all(ids[0, ::2] == 1)
    with pytest.raises(ValueError, match="One complete plain Turkish sentence"):
        lab.token_ids("Bu deneme iki cümle içerir. Biri yeter.")


def test_random_numbers_match_pinned_browser_fixture():
    random = lab.random_stream(215000)
    assert [random() for _ in range(3)] == [0.13883074722252786, 0.1491455784998834, 0.5807572358753532]
    noise, labels = lab.model_noise(215000, 1)
    expected = np.array(
        [
            0.09712132066488266,
            0.1321779489517212,
            -0.37585705518722534,
            -0.1233142614364624,
            -0.08819263428449631,
            0.24900494515895844,
        ],
        dtype=np.float32,
    )
    assert np.array_equal(noise.reshape(-1)[:6], expected)
    assert labels[0, :6].tolist() == [6, 1, 27, 44, 22, 9]


def test_fake_graphs_follow_the_pinned_pipeline_and_skip_out_of_range():
    plan = sample_plan()
    row = plan[0]
    config = {"duration_scales": {str(row["voice_id"]): 1.0}}
    count = lab.token_ids(row["phrase"]).size
    calls = []

    def fake_graph(name, feeds):
        calls.append(name)
        assert set(feeds) == set(lab.CONTRACTS[name][0])
        if name == "text_encoder":
            return [
                np.ones((1, lab.CHANNELS, count), dtype=np.float32),
                np.full((1, 1, count), np.log(3.1), dtype=np.float32),
            ]
        if name == "generator":
            assert feeds["z"].shape == (1, 100, 4 * count)
            assert feeds["cond"].shape == (1, 292, 4 * count)
            assert feeds["noise_labels"].shape == (1, 32)
            assert feeds["cfg_scale"].tolist() == [2.0]
            return [np.ones_like(feeds["z"])]
        return [np.zeros((1, 4 * count * lab.HOP), dtype=np.float32)]

    audio, frames = lab.render(row, config, fake_graph)
    assert calls == ["text_encoder", "generator", "vocoder"]
    assert audio.size == frames * lab.HOP and frames == 4 * count
    calls.clear()
    config["duration_scales"][str(row["voice_id"])] = 0.1
    with pytest.raises(lab.UnsupportedDuration, match="duration_outside_3_to_4"):
        lab.render(row, config, fake_graph)
    assert calls == ["text_encoder"]


def test_synthetic_fixture_uses_exact_native_and_production_clocks(tmp_path, monkeypatch):
    monkeypatch.setattr(lab, "OUTPUT", tmp_path)
    for name in ("native", "prepared"):
        (tmp_path / name).mkdir()
    slot = "synthetic-00"
    t = np.arange(330 * lab.HOP) / lab.NATIVE_RATE
    original = (0.2 * np.sin(2 * np.pi * 271 * t)).astype(np.float32)
    wav, receipt = lab.prepare_waveform(original, tmp_path / "native" / f"{slot}.wav")
    (tmp_path / "prepared" / f"{slot}.wav").write_bytes(wav)
    assert receipt["native_samples"] == len(original)
    assert receipt["native_sample_sha256"] == lab.digest(original.astype("<f4").tobytes())
    assert receipt["samples"] == len(original) * 2 // 3
    assert receipt["wav_range"] == receipt["source_range"] == lab.clock.sample_range(0, receipt["samples"])
    row = {**receipt, "slot": slot, "status": "qualified"}
    assert np.array_equal(lab.read_prepared_pcm(row), lab.read_wav(wav))
    with pytest.raises(ValueError, match="source-range"):
        lab.read_prepared_pcm({**row, "source_range": lab.clock.sample_range(0, receipt["samples"], 1)})


def test_aggregate_preserves_all_unavailable_slots():
    plan = sample_plan()
    rows = [
        {
            "slot": row["slot"],
            "status": "unavailable",
            "language": "tr",
            "label": 1,
            "voice": row["voice"],
            "reason": "predicted_native_duration_outside_3_to_4_seconds",
            "graph_calls": 1,
        }
        for row in plan
    ]
    totals = lab.summarize(rows, plan)
    assert totals["case_outcomes"] == {"unavailable": 16}
    assert totals["acoustic_generator_calls"] == 0
    assert totals["new_neural_calls"] == 16
    with pytest.raises(ValueError, match="Every planned source"):
        lab.summarize(rows[:-1], plan)
