"""Measured rational frame coordinates agree with the original native axis."""
from fractions import Fraction

import pytest

from scripts.prepare_video_scoring_units_v18 import plan_case
from scripts.prepare_video_specialists_v18 import frame_clock
from scripts.video_specialist_units_v18 import planned


@pytest.mark.parametrize('rate,count,label', [(24, 144, 0), (24, 144, 1), (25, 141, 0), (24, 136, 0)])
def test_rational_coordinates_are_exactly_the_same_as_native_pts(rate, count, label):
    table = {'pts': [i * 512 for i in range(count)], 'time_base': f'1/{512 * rate}',
        'truncated': False, 'interpolated': [], 'width': 640, 'height': 480, 'rotation': 0}
    edit = [30, 66]
    row = {'label': label, 'source_geometry': frame_clock(table, count, expected_fps=None),
        'known_edit': {'frames': edit, 'time_ns': [round(Fraction(i * 10**9, rate)) for i in edit]} if label else None}
    before, after = planned(row, table), plan_case(row)
    assert before['planned_scope_label'] == after['planned_scope_label']
    assert before['unobserved_known_edit_frames'] == after['unobserved_known_edit_frames']
    assert [(u['location'], u['label']) for u in before['units']] == [(u['location'], u['label']) for u in after['units']]
    assert all(u['view'] != 'photometric' for u in after['units'])


def test_nominal_rate_cannot_replace_the_measured_source_duration():
    row = {'label': 0, 'known_edit': None, 'source_geometry': {'native_frame_rate': '24', 'frame_count': 141,
        'width': 640, 'height': 480, 'duration_ns': 5_640_000_000}}
    with pytest.raises(ValueError, match='native frame clock'):
        plan_case(row)
