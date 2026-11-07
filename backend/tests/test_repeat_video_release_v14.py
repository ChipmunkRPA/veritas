import pytest

from scripts.repeat_video_release_v14 import score_and_collect


def test_cleanup_does_not_replace_or_mutate_detector_response():
    response = {'status': 'not_applicable', 'prob': None}
    task = object()
    seen = []

    def score(actual):
        assert actual is task
        seen.append('score')
        return response

    actual = score_and_collect(score, task, lambda: seen.append('collect'))
    assert actual is response and actual == {'status': 'not_applicable', 'prob': None}
    assert seen == ['score', 'collect']


def test_failed_score_is_propagated_after_cleanup_never_made_an_original_vote():
    failure = ValueError('Native timeline unavailable')
    seen = []

    def score(_):
        raise failure

    with pytest.raises(ValueError) as caught:
        score_and_collect(score, object(), lambda: seen.append('collect'))
    assert caught.value is failure and seen == ['collect']


def test_cleanup_failure_is_not_a_successful_analysis():
    def collect():
        raise RuntimeError('Collection failed')

    with pytest.raises(RuntimeError):
        score_and_collect(lambda _: {'status': 'ok', 'prob': .1}, object(), collect)
