"""Actual frame differences and score-blind source role linking."""
import pytest

from scripts.audit_video_scoring_inputs_v18 import changed_frames, components


def row(group, role, hashes, component=None):
    return {'group': group, 'role': role, 'component': component or group,
        'nonuniform_model_input_rgb_sha256': hashes}


def test_unchanged_frames_inside_the_inserted_interval_are_not_ai_truth():
    assert changed_frames(['a', 'b', 'c', 'd', 'e'], ['a', 'x', 'c', 'y', 'e'], [1, 4]) == [1, 3]


@pytest.mark.parametrize('altered', [['x', 'b', 'c', 'd'], ['a', 'b', 'c', 'y'], ['a', 'b', 'c']])
def test_changes_outside_the_interval_or_wrong_extent_are_refused(altered):
    with pytest.raises(ValueError):
        changed_frames(['a', 'b', 'c', 'd'], altered, [1, 3])


def test_a_completely_unchanged_ai_source_is_refused():
    with pytest.raises(ValueError):
        changed_frames(['a', 'b'], ['a', 'b'], [0, 2])


def test_source_components_without_new_connections_keep_original_roles():
    rows = [row('a', 'specialist_cal', ['x']), row('b', 'confirmation', ['y'])]
    assigned, report = components(rows)
    assert report['components'] == 2 and not report['role_reassignment_required']
    assert assigned['a']['role'] == 'specialist_cal' and assigned['b']['role'] == 'confirmation'


def test_duplicate_nonuniform_source_frames_join_across_roles_before_scoring():
    rows = [row('a', 'specialist_cal', ['x']), row('b', 'confirmation', ['x']),
        row('c', 'fusion_fit', ['y']), row('d', 'fusion_cal', ['z'])]
    assigned, report = components(rows)
    assert report['components'] == 3 and report['components_crossing_original_roles'] == 1
    assert report['role_reassignment_required']
    assert assigned['a'] == assigned['b']


def test_transitive_frame_connections_preserve_one_connected_component():
    rows = [row('a', 'specialist_cal', ['x']), row('b', 'fusion_fit', ['x', 'y']),
        row('c', 'confirmation', ['y'])]
    assigned, report = components(rows)
    assert report['components'] == 1 and len({r['scoring_component'] for r in assigned.values()}) == 1


def test_original_known_source_connections_are_not_lost_without_shared_frames():
    assigned, report = components([row('a', 'specialist_cal', [], 'known'), row('b', 'specialist_cal', [], 'known')])
    assert report['components'] == 1 and assigned['a'] == assigned['b']
