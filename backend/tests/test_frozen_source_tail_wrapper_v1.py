"""Immutable source selection and the actual YAML failure regression."""
from pathlib import Path

import pytest
import yaml

from scripts.replay_frozen_source_tail_v1 import SCIENCE_COMMIT, blob, safe_path, selected_files


@pytest.mark.parametrize('name', ['/private/tmp/x', '../x', 'backend/../../x', '--output=x', '', None])
def test_archive_selection_refuses_escaping_or_option_paths(name):
    with pytest.raises(ValueError):
        safe_path(name)


def test_selected_frozen_snapshot_contains_exact_old_harness_and_only_public_dependencies():
    names = selected_files()
    assert SCIENCE_COMMIT == 'fafafe88ab8db2bdd72d4612227bb396ab09f8ad'
    assert '.github/workflows/specialist-source-tail-v1-review.yml' in names
    assert 'docs/evaluation-panels/image-specialists-v10-manifests.json.gz' in names
    assert 'docs/evaluation-panels/writing-specialists-v5-scalars.json.gz' in names
    assert not any(name.startswith(('models/', 'backend/data/', 'output/')) for name in names)
    with pytest.raises(yaml.YAMLError):
        yaml.safe_load(blob('.github/workflows/specialist-source-tail-v1-review.yml'))


def test_corrected_workflow_parses_setup_as_a_script_and_runs_frozen_wrapper():
    root = Path(__file__).resolve().parents[2]
    value = yaml.safe_load((root / '.github/workflows/specialist-source-tail-v1-review.yml').read_text())
    steps = value['jobs']['scalar-refit']['steps']
    setup = next(step for step in steps if step.get('name') == 'Install only the pinned numerical libraries')
    assert isinstance(setup['run'], str) and '--only-binary=:all:' in setup['run']
    assert any('scripts.replay_frozen_source_tail_v1' in step.get('run', '') for step in steps)
    checkout = next(step for step in steps if str(step.get('uses', '')).startswith('actions/checkout@'))
    assert checkout['with']['fetch-depth'] == 0
