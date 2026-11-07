import pytest

from scripts.repeat_video_synthesis_v13 import correspondence


def panel(n):
    return [{'dataset': 'fixture', 'id': str(i), 'variant': 'original', 'sha256': str(i), 'label': i % 2,
             'split': 'test', 'status': 'ok', 'checks': {'a': {'status': 'ok', 'prob': .4}, 'b': {'status': 'not_applicable', 'prob': None}}}
            for i in range(n)]


def test_every_original_prefix_check_is_preserved():
    assert correspondence(panel(84), panel(288)) == 0.


@pytest.mark.parametrize('fault', ['omitted_old', 'omitted_new', 'source', 'status', 'applicability', 'probability', 'nonfinite'])
def test_missing_or_changed_prior_results_cannot_pass_correspondence(fault):
    a, b = panel(84), panel(288)
    if fault == 'omitted_old':
        a.pop()
    elif fault == 'omitted_new':
        b.pop()
    elif fault == 'source':
        b[83]['sha256'] = 'different'
    elif fault == 'status':
        b[83]['checks']['b']['status'] = 'skipped'
    elif fault == 'applicability':
        b[83]['checks']['b']['prob'] = .1
    elif fault == 'probability':
        b[83]['checks']['a']['prob'] = .4001
    else:
        b[83]['checks']['a']['prob'] = float('nan')
    with pytest.raises(ValueError):
        correspondence(a, b)
