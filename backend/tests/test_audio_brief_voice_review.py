from __future__ import annotations

import copy
import hashlib
import math
from types import SimpleNamespace

import numpy as np
import pytest

from app.core.canonical import canonical_json
from app.detectors.audio import brief_voice_review
from app.detectors.audio.decode import read_wav
from app.evidence import audio_brief_voice as evidence
from app.evidence import audio_brief_voice_binding as binding
from app.evidence.audio_packets import _chart


def test_scan_uses_exact_inputs_and_strict_point(monkeypatch):
    pcm = np.arange(binding.SAMPLES, dtype=np.float32)
    above = math.nextafter(binding.POINT, math.inf)
    scores = iter((binding.POINT, above, above, above))
    called = []

    def logit(_, chunk):
        called.append((int(chunk[0]), len(chunk)))
        return next(scores)

    monkeypatch.setattr(brief_voice_review.short, "voice_logit", logit)
    rows = brief_voice_review.scan(pcm, [(0, 30000 / binding.RATE)], None,
                                   SimpleNamespace(check=lambda _: None))
    checks = binding.raw_checks(rows)
    assert len(checks) == 23
    assert [row["start_sample"] for row in checks] == list(binding.GRID)
    assert called == [(0, 12000), (8000, 12000), (16000, 12000), (24000, 12000)]
    assert [row["elevated"] for row in checks[:5]] == [False, True, True, True, None]
    assert checks[3]["speech_samples"] == 6000
    assert binding.selected_indices(checks) == [1, 2, 3]
    assert binding.presentation(checks)["speech_windows_checked"] == 4

    rows[2]["logit_hex"] = binding.POINT.hex()
    isolated = binding.raw_checks(rows)
    assert binding.presentation(isolated)["status"] == "no_flag"


def test_no_speech_and_interrupted_scan_are_distinct():
    pcm = np.zeros(binding.SAMPLES, dtype=np.float32)
    deadline = SimpleNamespace(check=lambda _: None)
    checks = binding.raw_checks(brief_voice_review.scan(pcm, [], None, deadline))
    assert binding.presentation(checks)["status"] == "unassessed"
    assert binding.presentation([], reason=binding.TIMEOUT)["status"] == "skipped"
    with pytest.raises(ValueError, match="Incomplete"):
        binding.raw_checks(checks[:-1])

    for index in (0, 2):
        checks[index].update(speech_samples=6000, elevated=True)
        assert binding.presentation(checks)["status"] == "unassessed"
    checks[1].update(speech_samples=6000, elevated=False)
    assert binding.presentation(checks)["status"] == "no_flag"


def test_frozen_media_match_original_samples_and_reject_changed_windows(monkeypatch):
    pcm = (np.arange(binding.SAMPLES, dtype=np.int32) % 8192 - 4096).astype(np.float32) / 32768
    source = {"samples": len(pcm), "sample_rate": binding.RATE, "source_sample_rate": binding.RATE,
              "channels": 1, "source_channels": 1, "source_codec": "pcm_s16le", "source_container": "wav",
              "source_bit_depth": 16, "timeline_time_base": "1/16000",
              "pcm_sha256": hashlib.sha256(pcm.astype("<f4").tobytes()).hexdigest(),
              "timeline": [{"start_sample": 0, "end_sample": len(pcm), "source_pts": 0}]}
    monkeypatch.setattr(brief_voice_review.short, "voice_logit", lambda *_: binding.POINT + 1)
    rows = brief_voice_review.scan(pcm, [(0, 14000 / binding.RATE)], None,
                                   SimpleNamespace(check=lambda _: None))
    objects = {}

    def store_bytes(data):
        sha = hashlib.sha256(data).hexdigest()
        objects[sha] = data
        return sha

    def make_asset(data, media_type, **metadata):
        return {"sha256": store_bytes(data), "media_type": media_type, **metadata}

    def make_record():
        raw = canonical_json({"detector_id": binding.DETECTOR_ID, "status": "ok", "details": {
            "pcm_sha256": source["pcm_sha256"], binding.KEY: {"schema": binding.SCHEMA, "readings": rows},
        }})
        return {"raw_output_sha256": store_bytes(raw)}

    inputs = dict(store=SimpleNamespace(read_bytes=objects.__getitem__), source=source, pcm=pcm, mapping=source,
                  waveform=_chart)
    record = make_record()
    review = evidence.build(record, **inputs, asset=make_asset)
    assert review["affects_overall_confidence"] is False
    assert [window["id"] for window in review["windows"]] == ["B750-01", "B750-02"]
    for window in review["windows"]:
        start, end = window["start_sample"], window["end_sample"]
        assert end - start == 12000
        assert window["source_timestamp_ranges"][0]["source_end_pts"] == end
        clip = objects[window["artifacts"]["selected_clip"]["sha256"]]
        np.testing.assert_array_equal(read_wav(clip), pcm[start:end])
    evidence.verify(review, record, **inputs, read_asset=objects.__getitem__)
    evidence.verify_source(review, pcm=pcm, mapping=source, waveform=_chart)

    changed = copy.deepcopy(review)
    changed["windows"][0]["source_timestamp_ranges"][0]["source_end_pts"] += 1
    with pytest.raises(ValueError, match="exact source interval"):
        binding.validate(changed, source)
    rows[-1]["pcm_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="different source samples"):
        evidence.build(make_record(), **inputs, asset=make_asset)
    assert not binding.supported_source({**source, "source_container": "flac", "source_codec": "flac"})
