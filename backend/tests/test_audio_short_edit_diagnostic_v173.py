import hashlib
import json

import numpy as np
import pytest

from app.detectors.audio.decode import wav_bytes
from scripts import audio_short_edit_diagnostic_v173 as diagnostic


def planned_sources(development):
    return [
        {
            "slot": f"{lane}-{i}",
            "lane": lane,
            "status": "qualified",
            "group_id": f"{lane}-{i % diagnostic.GROUPS[lane]}",
            "file": str(development / f"{lane}-{i}.wav"),
            "wav_sha256": f"unique-{lane}-{i}",
        }
        for lane, count in diagnostic.FOREGROUNDS.items()
        for i in range(count)
    ]


def test_manifest_reads_only_development_and_rejects_holdout_file_paths(tmp_path, monkeypatch):
    development = tmp_path / "development"
    development.mkdir()
    monkeypatch.setattr(diagnostic, "DEVELOPMENT", development)
    rows = planned_sources(development)
    manifest = development / "private-manifest.json"

    def admit():
        payload = json.dumps({"role": "development", "scored": False, "foregrounds": rows}).encode()
        manifest.write_bytes(payload)
        return diagnostic.development_sources(hashlib.sha256(payload).hexdigest())

    paths = []
    read_bytes = type(manifest).read_bytes

    def track(path):
        paths.append(path)
        return read_bytes(path)

    monkeypatch.setattr(type(manifest), "read_bytes", track)
    assert admit() == rows
    rows[0]["file"] = str(tmp_path / "holdout" / "genuine-0.wav")
    with pytest.raises(ValueError, match="Only the direct admitted development WAV"):
        admit()
    assert paths == [manifest, manifest]


def test_wave_loader_rejects_holdout_symlink_before_reading_payload(tmp_path, monkeypatch):
    development = tmp_path / "development"
    holdout = tmp_path / "holdout"
    development.mkdir()
    holdout.mkdir()
    monkeypatch.setattr(diagnostic, "DEVELOPMENT", development)
    expected = development / "genuine-0.wav"
    private = holdout / expected.name
    private.write_bytes(b"not an allowed source")
    expected.symlink_to(private)

    def must_not_read(path):
        pytest.fail(f"Unexpected WAV read: {path.name}")

    monkeypatch.setattr(type(expected), "read_bytes", must_not_read)
    with pytest.raises(ValueError, match="Development WAV cannot resolve outside"):
        diagnostic.read_development_pcm({"slot": "genuine-0", "file": str(expected)})


def test_loader_verifies_exact_three_second_pcm_and_origin(tmp_path, monkeypatch):
    monkeypatch.setattr(diagnostic, "DEVELOPMENT", tmp_path)
    pcm = np.arange(48000, dtype="<f4") / np.float32(32768) - 0.75
    data = wav_bytes(pcm)
    path = tmp_path / "genuine-0.wav"
    path.write_bytes(data)
    row = {
        "slot": "genuine-0",
        "file": str(path),
        "wav_sha256": hashlib.sha256(data).hexdigest(),
        "prefix_pcm_sha256": hashlib.sha256(pcm.tobytes()).hexdigest(),
        "wav_range": {
            "start_sample": 0,
            "end_sample": 48000,
            "source_start_pts": 0,
            "source_end_pts": 48000,
            "time_base": "1/16000",
        },
    }
    assert diagnostic.read_development_pcm(row) == (pcm * 32768).astype("<i2").tobytes()
    row["wav_range"]["source_start_pts"] = 1
    with pytest.raises(ValueError, match="Exact qualified development PCM and sample origin"):
        diagnostic.read_development_pcm(row)


@pytest.mark.parametrize(
    ("direction", "cue_status", "spans", "expected"),
    [
        ("ai", "measured", ["inspection range"], "inconclusive"),
        ("ai", "measured", [], "ai"),
        ("ai", "ambiguous", [], "ai"),
        ("original", "measured", ["inspection range"], "original"),
        ("unavailable", "measured", ["inspection range"], "unavailable"),
    ],
)
def test_repeat_is_one_way_and_uses_only_measured_spans(direction, cue_status, spans, expected):
    assert diagnostic.repeat_direction(direction, {"cue": {"status": cue_status, "spans": spans}}) == expected


def test_planned_genuine_and_ai_family_counts_retain_unsupported_cases(tmp_path):
    sources = planned_sources(tmp_path)
    expected_cases = diagnostic.cases(sources)
    assert len(expected_cases) == 79
    silence = bytes(96000)

    def no_model(_pcm):
        pytest.fail("Unsupported construction cannot score")

    unsupported = diagnostic.assess_case(silence, "hard_copy", no_model)
    assert unsupported["construction"] == "unsupported"
    assert unsupported["reason"] == "zero_changed_samples"
    rows = []
    for slot, lane, arm in expected_cases:
        rows.append({"slot": slot, "lane": lane, "arm": arm, **unsupported})
    result = diagnostic.summary(rows, sources)
    assert result["genuine"]["hard_copy"]["planned"] == 14
    assert result["genuine"]["hard_copy"]["baseline_directions"] == {"unavailable": 14}
    assert result["clone"]["untouched"]["planned"] == 5
    assert result["expressive"]["untouched"]["planned"] == 12
    assert result["stock"]["untouched"]["planned"] == 6
    assert result["genuine"]["fade_50_ms"]["cues"]["exact"]["statuses"] == {"not_run": 14}
    with pytest.raises(ValueError, match="Every frozen case"):
        diagnostic.summary(rows[1:], sources)


def test_authored_copy_compares_baseline_and_repeat_on_the_same_pcm():
    rng = np.random.default_rng(173)
    values = rng.integers(-14000, 14000, size=48000, dtype=np.dtype("<i2"))
    seen = []

    def saved_score(pcm):
        seen.append((pcm * 32768).astype("<i2"))
        return {"status": "ok", "windows": [[0, 48000]], "decision": "ai", "symmetric": "ai"}

    observed = diagnostic.assess_case(values.tobytes(), "hard_copy", saved_score)
    assert len(seen) == 1
    np.testing.assert_array_equal(seen[0][17280:30080], values[12160:24960])
    assert observed["decision"] == observed["symmetric"] == "ai"
    assert observed["repeat_abstention"] == "inconclusive"
    assert observed["cues"]["exact"]["witnessed_copy"] is True
