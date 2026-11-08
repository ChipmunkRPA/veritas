"""Completed public receipts retain failed checks and platform differences."""
import json
import shutil
from pathlib import Path

import pytest

from scripts import compare_public_visual_native_v1 as comparison

PUBLIC = Path(__file__).resolve().parents[2] / 'docs/evaluation-panels'
MAC = PUBLIC / 'public-visual-v1-mac-receipts'
LINUX = PUBLIC / 'public-visual-v1-linux-receipts'


def test_actual_completed_contracts_preserve_codec_disagreements():
    result = comparison.compare(MAC, LINUX)
    assert result['both_model_free_audits_completed'] is True
    assert result['cross_platform_groups_exact'] is False
    assert result['groups_with_changed_flags_or_misses'] == 2
    changed = [r for r in result['groups'] if any(k in r['differences'] for k in
               ('flagged', 'unmarked_flagged', 'marked_missed'))]
    assert all(r['kind'] == 'video' and r['alteration'] == 'h264' for r in changed)
    assert result['fresh_natural_media_cases'] == result['outside_human_reviewers'] == 0
    assert result['serving_changed'] is False


@pytest.mark.parametrize('damage', ['guard', 'model', 'omitted-case', 'incomplete-audit', 'admission', 'core-version'])
def test_incomplete_or_changed_science_cannot_be_reported_as_reproduction(tmp_path, damage):
    root = tmp_path / 'linux'
    shutil.copytree(LINUX, root)
    name = {'guard': 'profile/profile.json', 'model': 'result.json', 'omitted-case': 'result.json',
            'incomplete-audit': 'audit/audit.json', 'admission': 'result.json',
            'core-version': 'freeze.json'}[damage]
    path = root / name
    data = json.loads(path.read_text())
    if damage == 'guard':
        data['guard'] = 'memory_disk_or_time'
    elif damage == 'model':
        data['load_receipt']['checkpoint_sha256'] = '0' * 64
    elif damage == 'omitted-case':
        data['case_count'] -= 1
    elif damage == 'incomplete-audit':
        data['source_times_verified'] = False
    elif damage == 'admission':
        data['serving_admitted'] = True
    else:
        data['packages']['torch'] = '2.14.0+cpu'
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        comparison.compare(MAC, root)
