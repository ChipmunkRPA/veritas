import hashlib
import subprocess
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest

from app.core.config import settings
from app.detectors.audio.combined_adapter import CombinedSpeechDetector
from app.detectors.audio.decode import SAMPLE_RATE, wav_bytes
from app.detectors.video.adapters.audio import GustkingVoiceDetector, PellaVoiceDetector
from app.detectors.video.adapters.common import AudioTrack
from app.detectors.video.audio import voice_windows
from app.engine.boundary import Deadline
from app.engine.detector import Profile, StageOutputs
from app.engine.stages import audio, video
from app.evidence.audio_binding import (
    VOICE_CLOCK_LIMIT,
    current_voice_detection_output,
    exclude_recording_voice,
)


@pytest.mark.parametrize("shift", [0, 10240, -1024])
def test_real_pcm_clock_reaches_both_voice_pipelines(tmp_path, monkeypatch, shift):
    samples = SAMPLE_RATE * 3
    boundary = 24576
    pcm = (0.2 * np.sin(2 * np.pi * 557 * np.arange(samples) / SAMPLE_RATE)).astype(np.float32)
    original = tmp_path / "continuous.wav"
    original.write_bytes(wav_bytes(pcm))
    source = tmp_path / "source.mka"
    clock = rf"asetnsamples=n=2048:p=0,asetpts=PTS+if(gte(N\,{boundary})\,{shift}\,0)"
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", str(original), "-af", clock, "-c:a", "pcm_s16le",
                    str(source)], check=True, timeout=15)
    sha = hashlib.sha256(source.read_bytes()).hexdigest()
    breaks = (boundary,) if shift else ()

    def register(_db, _store, data, **meta):
        checksum = hashlib.sha256(data).hexdigest()
        return SimpleNamespace(id=checksum, sha256=checksum, bytes=len(data), **meta)

    monkeypatch.setattr(audio, "register_artefact", register)
    stage = SimpleNamespace(outputs={"ingest": {"sha256": sha}}, state={},
                            store=SimpleNamespace(path_for=lambda _: source), check=lambda *_: None, analysis_id="clock",
                            sessions=lambda: nullcontext(SimpleNamespace(commit=lambda: None)))
    decoded = audio.decode(stage)
    assert decoded["source_clock_break_count"] == len(breaks)
    standalone = stage.state["audio"]
    nested, mapping = video._audio(source, True, source_sha256=sha)
    assert standalone.clock_breaks == nested.clock_breaks == breaks
    assert wav_bytes(standalone.pcm) == wav_bytes(nested.pcm) == original.read_bytes()
    assert mapping["pcm_sha256"] == decoded["evidence_input"]["pcm_sha256"]

    gate_calls, voice_calls = [], []

    def spans(values):
        gate_calls.append(len(values))
        return [(0, len(values) / SAMPLE_RATE)]

    def predict(chunks):
        voice_calls.extend(chunks)
        return [0.000001 for _ in chunks]

    monkeypatch.setattr(settings, "audio_detector_mode", "candidate")
    models = {PellaVoiceDetector.MODEL_ID: SimpleNamespace(obj=SimpleNamespace(predict=predict), revision="test"),
              "silero.speech_activity": SimpleNamespace(obj=SimpleNamespace(spans=spans), revision="test")}
    ctx = SimpleNamespace(outputs=StageOutputs(audio=nested), profile=Profile.FULL, lease=models,
                          deadline=Deadline(30), exhibits=SimpleNamespace(series=lambda *a, **k: "series"))
    result = PellaVoiceDetector().run(ctx)
    assert result.status == "ok"
    screen = result.details["audio_screening"]
    if breaks:
        assert gate_calls == [boundary, samples - boundary]
        assert [len(chunk) for chunk in voice_calls] == gate_calls
        assert [u.quality["start_sample"] for u in result.unit_scores] == [0, boundary]
        assert screen["reason"] == VOICE_CLOCK_LIMIT
        assert screen["decision"] == "inconclusive" and not screen["scope_supported"]
        assert screen["probability_ai"] is None and screen["reference_estimate_ai"] is None
    else:
        assert gate_calls == [samples]
        assert screen["decision"] == "original" and screen["scope_supported"]
        assert screen["probability_ai"] is not None


def test_window_cap_ranks_continuous_pieces_together_and_can_keep_a_short_tail():
    pcm = np.repeat([1, 2, 3], [32000, 24000, 16000]).astype(np.float32)
    track = AudioTrack("analysed", pcm, len(pcm), max_windows=2, clock_breaks=(32000, 56000))
    ctx = SimpleNamespace(outputs=StageOutputs(audio=track), profile=Profile.FULL)
    windows = GustkingVoiceDetector._windows(ctx)
    assert [(round(start * SAMPLE_RATE), len(chunk)) for start, chunk in windows] == [(32000, 24000), (56000, 16000)]
    assert [int(chunk[0]) for _, chunk in windows] == [2, 3]
    [(tail, values)] = voice_windows(pcm[:32000], 2, clock_breaks=(15999,))
    assert tail == 15999 / SAMPLE_RATE
    np.testing.assert_array_equal(values, pcm[15999:32000])


def test_optional_four_second_check_and_fusion_leave_only_interval_evidence():
    pcm = np.ones(64000, np.float32)
    track = AudioTrack("analysed", pcm, len(pcm), clock_breaks=(32000,))
    ctx = SimpleNamespace(outputs=StageOutputs(audio=track))
    check = CombinedSpeechDetector()
    assert not check.applicable(ctx).ok
    assert check.run(ctx).reason == VOICE_CLOCK_LIMIT
    voice = {"family": "voice", "status": "ok", "aggregate": {"raw": 0.98}, "unit_scores": [{"raw": 0.98}]}
    visual = {"family": "learned-face", "status": "ok", "aggregate": {"raw": 0.75}}
    limited, kept = exclude_recording_voice([voice, visual])
    assert voice["status"] == "ok" and voice["unit_scores"]
    assert limited["status"] == "not_applicable" and limited["aggregate"] is None and not limited["unit_scores"]
    assert kept is visual
    assert not current_voice_detection_output({"detectors": [voice]})
    assert current_voice_detection_output({"detectors": [voice], "voice_clock_policy": "source-clock-split/1"})
