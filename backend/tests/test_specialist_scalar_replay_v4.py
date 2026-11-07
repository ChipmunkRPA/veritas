"""Authored regression fixtures; no real-model performance evidence."""
import copy

from test_specialist_calibration import fixture

from app.bench import specialists as engine
from scripts.reproduce_specialist_study_v4 import adverse


def audio_packet():
    manifest = fixture()
    manifest['files'] = [f for f in manifest['files'] if f['modality'] == 'audio']
    manifest['policy']['specialists'] = [s for s in manifest['policy']['specialists'] if s['modality'] == 'audio']
    return manifest


def test_unassessed_confirmation_is_preserved_and_missing_test_uses_assessed_file():
    manifest = audio_packet()
    first = next(f for f in manifest['files'] if f['role'] == 'confirmation')
    first['units'][0].update(status='not_applicable', raw=None, reason='No eligible speech')
    artifact = engine.train(manifest, min_groups_per_class=3)
    prior = copy.deepcopy(manifest)
    result = adverse(manifest, artifact, minimum=3)
    assert result['preserved_unassessed_confirmation_units'] == 1
    assert result['changed_assessed_confirmation_units'] == 31
    assert result['missing_check_test_status'] == 'passed_on_previously_assessed_file'
    assert result['missing_required_check_withheld'] is True
    assert manifest == prior


def test_all_inconclusive_is_not_reported_as_a_passed_missing_check():
    manifest = audio_packet()
    for file in manifest['files']:
        if file['role'] == 'confirmation':
            for unit in file['units']:
                unit.update(status='not_applicable', raw=None, reason='No eligible speech')
    result = adverse(manifest, engine.train(manifest, min_groups_per_class=3), minimum=3)
    assert result['changed_assessed_confirmation_units'] == 0
    assert result['preserved_unassessed_confirmation_units'] == 32
    assert result['missing_required_check_withheld'] is False
    assert result['missing_check_test_status'] == 'not_testable_no_assessed_confirmation'


def test_reflection_respects_declared_non_probability_raw_range():
    manifest = audio_packet()
    # The authored fixture uses [-10,10], with some genuine values below zero.
    # A probability-only reflection would exceed this declared response range.
    for file in manifest['files']:
        if file['role'] == 'confirmation':
            for unit in file['units']:
                unit['raw'] = -10.
    result = adverse(manifest, engine.train(manifest, min_groups_per_class=3), minimum=3)
    assert result['confirmation_not_fitted'] is True
