"""Native proof review rejects changed clocks, truth, counters and role leakage."""
import copy

import pytest

from scripts.audit_video_native_bindings_v20 import check_input_roles, inspect


def unassessed():
    location = {'sampled_frames': [0], 'context_frames': []}
    planned = {'id': 'authored', 'sha256': 'a' * 64,
        'source_geometry': {'pts_table_sha256': 'b' * 64, 'native_frame_rate': '24'},
        'actual_changed_ai_source_frames': [],
        'units': [{'id': 'u', 'view': 'face-crop', 'label': 0, 'location': location}]}
    measured = {'id': 'authored', 'input_sha256': 'a' * 64, 'native_table_sha256': 'b' * 64,
        'frame_inputs': [{'frame_index': 0, 'source_rgb_sha256': 'c' * 64, 'pts': 0,
            'time_base': '1/12288', 'status': 'not_applicable'}],
        'native_repeats': [], 'native_input_counts': {'face_detection_inputs': 1,
            'crop_classifier_inputs': 0, 'appearance_inputs': 0, 'patch_feature_inputs': 0, 'native_repeat_inputs': 0},
        'units': [{'id': 'u', 'view': 'face-crop', 'location': location,
            'status': 'not_applicable', 'raw': None, 'reason': 'No visible face.'}]}
    return planned, measured


def test_unassessed_input_is_retained_without_an_invented_negative():
    planned, measured = unassessed()
    assert inspect(planned, measured, ['c' * 64])['face_detection_inputs'] == 1


@pytest.mark.parametrize('mutation', ['rgb', 'pts', 'count', 'label', 'unavailable_numeric'])
def test_incorrect_source_bindings_fail(mutation):
    planned, original = unassessed()
    measured = copy.deepcopy(original)
    if mutation == 'rgb':
        measured['frame_inputs'][0]['source_rgb_sha256'] = 'd' * 64
    elif mutation == 'pts':
        measured['frame_inputs'][0]['pts'] = 512
    elif mutation == 'count':
        measured['native_input_counts']['crop_classifier_inputs'] = 1
    elif mutation == 'label':
        planned['units'][0]['label'] = 1
    else:
        measured['units'][0]['raw'] = 0.
    with pytest.raises(ValueError):
        inspect(planned, measured, ['c' * 64])


def test_exact_eligible_inputs_may_repeat_within_one_role_but_cannot_cross_roles():
    frames = [{'crop_tensor_sha256': 'a' * 64}]
    seen = {}
    check_input_roles(frames, 'specialist_cal', seen)
    check_input_roles(frames, 'specialist_cal', seen)
    with pytest.raises(ValueError, match='crosses source roles'):
        check_input_roles(frames, 'confirmation', seen)
