from types import SimpleNamespace

import numpy as np

from app.detectors.video.adapters.audio import GustkingVoiceDetector, PellaVoiceDetector, SoundEventsDetector
from app.detectors.video.adapters.common import AudioTrack


def test_new_speech_gate_does_not_change_existing_voice_detector_windows():
    pcm = np.ones(8 * 16000, np.float32) * .1
    track = AudioTrack('analysed', pcm, len(pcm), gate=((0, 8),))
    silence_gate = SimpleNamespace(spans=lambda value: [])
    ctx = SimpleNamespace(profile='full', outputs=SimpleNamespace(audio=track),
                          lease=SimpleNamespace(get=lambda key: SimpleNamespace(obj=silence_gate)))
    assert len(GustkingVoiceDetector()._windows(ctx)) == 3
    assert PellaVoiceDetector()._windows(ctx) == []
    assert track.gate == ((0, 8),)


def test_sound_categories_do_not_claim_an_authenticity_hypothesis():
    assert SoundEventsDetector.spec.hypotheses == ()
    assert SoundEventsDetector.spec.family == 'acoustic-context'
    assert SoundEventsDetector().applicable(SimpleNamespace(outputs=SimpleNamespace(
        audio=AudioTrack('silent', np.zeros(16000, np.float32), 16000)))).ok


def test_candidate_has_its_own_published_signal_and_does_not_replace_baseline_identity():
    from app.engine.stages import audio, video

    def entry(detector):
        return {'detector_id': detector, 'status': 'ok', 'family': 'voice',
                'aggregate': {'unit_ref': 'global', 'raw': .9, 'raw_kind': 'probability'},
                'unit_scores': [{'unit_ref': 'window:0', 'raw': .9, 'raw_kind': 'probability'}]}
    baseline, candidate = entry('aud.gustking_xlsr'), entry('aud.pella_v2')
    assert audio.sole_voice_entry([baseline, candidate]) is baseline
    assert video._published_ids(candidate, None, voice_id='baseline') == ['aud.pella_v2:global']
    assert video._published_ids(baseline, None, voice_id='baseline') == ['baseline']
