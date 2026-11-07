"""Native repeat contracts, missing scope and local source-coordinate binding."""
from copy import deepcopy
import json

import numpy as np
import pytest

from scripts import run_image_supported_v11 as run


def test_frozen_policy_roundtrips_exactly_through_the_actual_json_format():
    assert json.loads(json.dumps(run.POLICY, allow_nan=False)) == run.POLICY


def test_native_png_requires_float32_and_bitwise_repeat():
    value = np.full((384, 384), .4, np.float32)
    assert run.repeat_check((value, None), (value.copy(), None), 'png')['map_bitwise_exact']
    changed = value.copy()
    changed[0, 0] = np.nextafter(changed[0, 0], np.float32(1))
    with pytest.raises(ValueError, match='bitwise'):
        run.repeat_check((value, None), (changed, None), 'png')
    with pytest.raises(ValueError, match='Invalid original model map'):
        run.verify_map(value.astype(np.float64), 'png')


def test_native_jpeg_tolerance_cannot_hide_changed_gate_side():
    value = np.full((512, 512), .4, np.float32)
    assert run.repeat_check((value, .7), (value + 1e-5, .70001), 'jpeg')['primary_difference'] < 1e-4
    with pytest.raises(ValueError, match='gate side'):
        run.repeat_check((value, .49999), (value, .50001), 'jpeg')


def test_inconclusive_ai_is_counted_as_unfound_and_never_as_genuine():
    rows = [{'label': 1, 'group': 'a', 'assessed': False, 'probability': None, 'flag': False},
        {'label': 0, 'group': 'b', 'assessed': False, 'probability': None, 'flag': False}]
    actual = run.metrics(rows)
    assert actual['ai_not_flagged_including_inconclusive'] == 1
    assert actual['inconclusive'] == 2 and actual['genuine'] == 1 and actual['ai'] == 1
    assert actual['source_class_balanced_brier'] is None


def test_local_summary_uses_local_truth_and_preserves_exact_source_box():
    box = {'kind': 'pixels', 'x': 20, 'y': 0, 'width': 20, 'height': 30,
        'source_width': 40, 'source_height': 30}
    document = {'files': [{'id': 'edited', 'group': 'original', 'label': 1, 'units': [
        {'id': 'quadrant.1', 'location': box, 'label': 0}]}]}
    result = {'unit_outcomes': [{'file_id': 'edited', 'unit_id': 'quadrant.1', 'location': box, 'probability': .1}]}
    actual = run.local_metrics(document, result, {'edited': {'format': 'png'}})['png|region']
    assert actual['genuine_units'] == 1 and actual['ai_units'] == 0
    revised = deepcopy(result)
    revised['unit_outcomes'][0]['location']['x'] = 0
    with pytest.raises(ValueError, match='exact source box'):
        run.local_metrics(document, revised, {'edited': {'format': 'png'}})


def test_confirmation_refuses_to_load_model_before_all_fits_are_frozen(tmp_path, monkeypatch):
    monkeypatch.setattr(run, 'check', lambda trial: ({'main': str(tmp_path)}, {}))
    from app.detectors.image import fused, splice

    def forbidden(*args, **kwargs):
        raise AssertionError('Model must not load before frozen fits')

    monkeypatch.setattr(fused, 'load_model', forbidden)
    monkeypatch.setattr(splice, 'load_model', forbidden)
    with pytest.raises(FileNotFoundError, match='fit-complete'):
        run.score(tmp_path, 'confirmation', 'jpeg')
    assert not list(tmp_path.iterdir())
