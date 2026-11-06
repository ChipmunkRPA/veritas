from types import SimpleNamespace

import numpy as np

from app.api.public_projection import project
from app.detectors.audio import calibration

OLD = 'The calibrated voice estimate and uncertainty check do not jointly support an 80% tendency.'
NEW = 'The calibrated voice estimate and uncertainty check do not jointly support a reliable tendency.'


def test_new_inconclusive_speech_explains_uncertainty_without_public_threshold_claim():
    pcm = np.zeros(64000, np.float32)
    screen = calibration.evaluate(.5, pcm, [(0., pcm)], None, SimpleNamespace(check=lambda *_: None))
    assert screen['decision'] == 'inconclusive' and screen['reason'] == NEW
    assert screen['probability_ai'] == calibration.probability(.5)


def test_historical_public_projection_preserves_uncertainty_and_source_quotations():
    source = {'details': {'audio_screening': {'decision': 'inconclusive', 'probability_ai': .53, 'reason': OLD}},
              'findings': [{'statement': OLD, 'sentences': [{'text': OLD, 'evidence_ids': ['speech']}]}],
              'source': {'text_original': OLD}, 'notes': OLD}
    public = project(source)
    assert public['details']['audio_screening']['reason'] == NEW
    assert public['details']['audio_screening']['decision'] == 'inconclusive'
    assert public['details']['audio_screening']['probability_ai'] == .53
    assert public['findings'][0]['statement'] == NEW
    assert public['findings'][0]['sentences'][0]['text'] == NEW
    assert public['source']['text_original'] == OLD and public['notes'] == OLD
    assert source['details']['audio_screening']['reason'] == OLD
