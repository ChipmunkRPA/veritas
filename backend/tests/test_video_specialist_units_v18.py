"""Local frame truth, real clock and unavailable-readout regressions."""
from copy import deepcopy
from fractions import Fraction

import pytest

from scripts.prepare_video_specialists_v18 import frame_clock
from scripts.video_specialist_units_v18 import attach, planned


def fixture(rate=24, count=120, edit=None):
    table = {'truncated': False, 'interpolated': [], 'pts': [i * 512 for i in range(count)],
        'width': 640, 'height': 480, 'rotation': 0, 'time_base': f'1/{512 * rate}'}
    row = {'source_geometry': frame_clock(table, count, expected_fps=None), 'label': int(edit is not None),
        'known_edit': {'frames': list(edit), 'time_ns': [round(Fraction(i * 10**9, rate)) for i in edit]} if edit else None}
    return row, table


def test_whole_ai_label_does_not_spread_to_unchanged_sampled_frames():
    row, table = fixture(edit=(24, 48))
    result = planned(row, table)
    crops = {u['location']['sampled_frames'][0]: u for u in result['units'] if u['view'] == 'face-crop'}
    assert crops[12]['label'] == 0 and crops[24]['label'] == 1 and crops[36]['label'] == 1
    assert crops[48]['label'] == 0  # End boundary is exclusive.
    assert result['whole_clip_label'] == result['planned_scope_label'] == 1
    assert not result['whole_clip_originality_covered']


def test_background_context_edit_is_not_copied_to_middle_target_truth():
    row, table = fixture(edit=(12, 13))
    result = planned(row, table)
    unit = next(u for u in result['units'] if u['view'] == 'face-appearance')
    assert 12 in unit['location']['context_frames'] and 12 not in unit['location']['sampled_frames']
    assert unit['label'] == 0
    other = next(u for u in result['units'] if u['view'] == 'photometric' and u['location'] == unit['location'])
    assert other['label'] == 0 and other['target'] == unit['target']


def test_actual25fps_control_clock_is_not_interpreted_as24fps():
    row, table = fixture(rate=25, count=125)
    result = planned(row, table)
    unit = next(u for u in result['units'] if u['view'] == 'face-crop' and u['location']['sampled_frames'] == [25])
    assert unit['location']['start'] == 1_000_000_000
    assert unit['location']['end'] == 1_040_000_000
    assert all(u['label'] == 0 for u in result['units'])


def test_sparse_gap_and_short_context_are_explicit_not_an_original_verdict():
    row, table = fixture(count=12, edit=(1, 2))
    result = planned(row, table)
    assert result['whole_clip_label'] == 1 and result['planned_scope_label'] == 0
    assert result['unobserved_known_edit_frames'] == 1
    assert result['unsupported_views'] == ['face-appearance', 'photometric']
    assert result['whole_clip_originality_covered'] is False


def test_missing_readouts_keep_planned_locations_and_null_values():
    row, table = fixture()
    plan = planned(row, table)
    actual = attach(plan, {})
    assert len(actual) == len(plan['units'])
    assert all(u['status'] == 'unavailable' and u['raw'] is None and u['reason'] for u in actual)
    assert [u['location'] for u in actual] == [u['location'] for u in plan['units']]


@pytest.mark.parametrize('change', ['truth', 'location', 'nonfinite', 'failed_zero'])
def test_response_cannot_rewrite_truth_coordinates_or_missing_values(change):
    row, table = fixture()
    plan = planned(row, table)
    unit = plan['units'][0]
    response = {'location': deepcopy(unit['location']), 'status': 'ok', 'raw': .3}
    if change == 'truth':
        response['label'] = 1
    elif change == 'location':
        response['location']['sampled_frames'] = [1]
    elif change == 'nonfinite':
        response['raw'] = float('nan')
    else:
        response.update(status='failed', raw=0., reason='Unavailable native response')
    with pytest.raises(ValueError):
        attach(plan, {unit['id']: response})


def test_changed_pts_commitment_is_refused_before_making_units():
    row, table = fixture()
    row['source_geometry']['pts_table_sha256'] = 'a' * 64
    with pytest.raises(ValueError, match='source clock'):
        planned(row, table)
