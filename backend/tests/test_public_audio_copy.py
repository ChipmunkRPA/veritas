"""Retired service claims disappear without rewriting source material or measured scores."""
from copy import deepcopy

from app.api.public_projection import neutral_text, project

OLD_REFERENCE = 'Frozen voice-specific calibration with an 80% tendency threshold and uncertainty checks.'
OLD_REASON = 'The calibrated voice estimate and uncertainty check do not jointly support an 80% tendency.'


def test_current_and_saved_voice_references_do_not_publish_the_old_service_threshold():
    original = {'signals': [{'reference': OLD_REFERENCE, 'reasoning': OLD_REASON, 'raw_score': 0.83}],
                'reason': 'The measured score is 83%.'}
    before = deepcopy(original)
    public = project(original)
    assert original == before
    assert '80%' not in public['signals'][0]['reference']
    assert '80%' not in public['signals'][0]['reasoning']
    assert 'duration limits' in public['signals'][0]['reference']
    assert public['signals'][0]['raw_score'] == 0.83
    assert public['reason'] == 'The measured score is 83%.'


def test_source_words_and_provenance_are_preserved_verbatim():
    source = {'content': OLD_REFERENCE, 'metadata': {'reference': OLD_REFERENCE},
              'signals': [{'category': 'provenance', 'reference': OLD_REFERENCE}]}
    assert project(source) == source


def test_neutral_text_preserves_unrelated_percentages():
    assert neutral_text('The source mentions 80% and 60% in its calculation.') == 'The source mentions 80% and 60% in its calculation.'
