import hashlib
import time

import numpy as np
import pytest

from scripts import audio_okur_turkish_v240 as study


def test_fixed_texts_have_distinct_valid_publisher_letter_ids():
    planned = study.plan()
    assert len(planned) == len({row["token_sha256"] for row in planned}) == 8
    assert len({row["seed"] for row in planned}) == 8
    assert all(study.token_ids(row["phrase"]).min() > 1 for row in planned)
    with pytest.raises(ValueError, match="already-normalized"):
        study.token_ids("saat 12 olduğunda parkın ışıkları yanacak.")


def test_timeline_uses_running_half_even_frames_and_space_ownership():
    durations = np.array([[9.5, 9, 10, 10, 10, 10, 10, 10.5]], dtype=np.float32)
    frames, timeline = study.timeline("abc def.", durations)
    assert frames == 79
    np.testing.assert_array_equal(timeline["cw"], [[0, 0, 0, 0, 1, 1, 1, 1]])
    np.testing.assert_array_equal(timeline["wstart"], [[0, 0, 0, 0, 4, 4, 4, 4]])
    assert (timeline["fw"][0, :38] == 0).all()
    assert (timeline["fw"][0, 38:] == 1).all()
    assert timeline["fp"][0, 37] == np.float32(37 / 38)
    assert timeline["fp"][0, 38] == 0


def test_seeded_noise_matches_known_publisher_algorithm_values():
    result = study.noise(0, 75)
    assert result.shape == (4, 1, 75, 64) and result.dtype == np.float32
    # Captured from the publisher's JavaScript arithmetic in Node.
    expected = np.array(
        [0.7871846556663513, 0.001630933489650488, 0.431439608335495, 0.5649690628051758], np.float32
    )
    np.testing.assert_array_equal(result.reshape(-1)[:4], expected)
    np.testing.assert_array_equal(result, study.noise(0, 75))
    assert not np.array_equal(result, study.noise(1, 75))


def test_render_does_not_sample_or_decode_out_of_range():
    item = study.plan()[0]
    calls = []

    def run(name, feeds):
        calls.append(name)
        length = feeds["ids"].size
        return np.zeros((1, length, study.HIDDEN), np.float32), np.full((1, length), 100, np.float32)

    frames, audio = study.render(item, run)
    assert frames > study.MAX_FRAMES and audio is None and calls == ["text"]


def test_render_uses_exact_graph_order_and_native_output():
    item = study.plan()[0]
    calls = []

    def run(name, feeds):
        calls.append(name)
        if name == "text":
            length = feeds["ids"].size
            durations = np.full((1, length), 80 / length, np.float32)
            return np.zeros((1, length, study.HIDDEN), np.float32), durations
        if name == "sample":
            assert feeds["noise"].shape == (4, 1, 80, study.LATENTS)
            return (np.zeros((1, study.LATENTS, 80), np.float32),)
        return (np.zeros((1, 80 * study.HOP), np.float32),)

    frames, audio = study.render(item, run)
    assert frames == 80 and audio.shape == (80 * study.HOP,)
    assert calls == ["text", "sample", "decoder"]


def test_native_and_production_pcm_preserve_complete_source_extent(tmp_path):
    count = 75 * study.HOP
    native = np.sin(np.arange(count, dtype=np.float64) * 2 * np.pi * 733 / study.NATIVE_RATE).astype(
        np.float32
    )
    wav, row = study.prepare_waveform(native * np.float32(0.25), tmp_path / "native.wav")
    assert row["native_samples"] == count and row["native_sample_rate"] == 48_000
    assert row["samples"] == 48_000 and row["sample_rate"] == 16_000
    assert hashlib.sha256(wav).hexdigest() == row["wav_sha256"]


def test_exact_repeat_required_and_unavailable_slots_are_kept():
    planned = study.plan()[:2]
    good = {
        "slot": planned[0]["slot"],
        "status": "qualified",
        "native_frames": 81,
        "graph_calls": 3,
        "pcm_sha256": "committed",
    }
    unavailable = {
        "slot": planned[1]["slot"],
        "status": "unavailable",
        "native_frames": 60,
        "graph_calls": 1,
        "reason": "predicted_duration_outside_three_to_four_seconds",
    }
    repeat = {**good, "slot": "synthetic-99"}
    result = study.summarize([good, unavailable], repeat, planned)
    assert result["new_neural_calls"] == 7 and result["exact_replays"] == 1
    assert result["case_outcomes"] == {"qualified": 1, "unavailable": 1}
    with pytest.raises(ValueError, match="replay"):
        study.summarize([good, unavailable], {**repeat, "pcm_sha256": "other"}, planned)


def test_signed_redirect_binds_host_path_linked_size_sha_and_revision():
    path = "/xet-bridge-us/" + "a" * 24 + "/" + "b" * 64
    url = "https://us.aws.cdn.hf.co" + path + "?signature=example"
    headers = {"X-Repo-Commit": study.MODEL_REV, "X-Linked-Etag": '"' + "b" * 64 + '"', "X-Linked-Size": "42"}
    assert study.signed_redirect(url, headers, 42, "b" * 64) == url
    for unsafe in (
        url.replace("cdn.hf.co", "example.com"),
        url.replace(path, "/unverified"),
        url.split("?")[0],
    ):
        with pytest.raises(study.AssetStop):
            study.signed_redirect(unsafe, headers, 42, "b" * 64)
    with pytest.raises(study.AssetStop):
        study.signed_redirect(url, headers, 42, "a" * 64)


def test_transport_keeps_unverified_bytes_out_of_final_path(tmp_path):
    class Response:
        status_code, headers = 200, {"Content-Length": "4"}

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def iter_content(self, **_):
            yield b"fake"

    class Session:
        def get(self, url, **kwargs):
            assert url == study.METADATA + "README.md" and not kwargs["allow_redirects"]
            return Response()

    destination = tmp_path / "README.md"
    with pytest.raises(study.AssetStop, match="asset_hash_or_extent_mismatch"):
        study.fetch_file(
            Session(), (study.METADATA + "README.md", 4, "0" * 64), destination, time.monotonic() + 5
        )
    assert not destination.exists() and destination.with_suffix(".md.partial").read_bytes() == b"fake"
