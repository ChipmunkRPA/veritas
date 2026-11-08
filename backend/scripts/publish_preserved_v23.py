"""Copy public-scalar-only completed research outputs into a replay package."""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

from scripts import run_preserved_video_v23 as study

ROOT = Path(__file__).resolve().parents[2]
PREFIX = 'docs/evaluation-panels/preserved-evidence-v23-'
EXTRA = ('backend/scripts/publish_preserved_v23.py', 'backend/scripts/reproduce_preserved_v23.py',
    'backend/scripts/replay_preserved_release_v23.py', 'backend/scripts/replay_frozen_specialist_release_v5.py',
    'backend/tests/test_preserved_replay_v23.py')


def publish(trial):
    frozen = study.check(trial)
    study.verify_fit(trial)
    study.read(trial / 'result.json')
    names = ['freeze.json', 'solver-audit.json', 'fit-complete.json', 'appearance-mapped-point.json', 'result.json']
    names += [arm + '-' + suffix + '.json' for arm in study.ARMS for suffix in ('artifact', 'point', 'evaluation')]
    outputs, expected = [], {}
    for name in names:
        value = study.read(trial / name)
        destination = PREFIX + name
        study.write(ROOT / destination, value)
        outputs.append(destination)
        expected[name] = value
    journal = trial / 'solver-audit.jsonl'
    data = [json.loads(line) for line in journal.read_text().splitlines()]
    expected['solver-audit-rows'] = data
    compressed = PREFIX + 'expected.json.gz'
    with (ROOT / compressed).open('xb') as stream:
        stream.write(gzip.compress(json.dumps(expected, separators=(',', ':'), allow_nan=False).encode(), mtime=0))
    outputs.append(compressed)
    audit_path = PREFIX + 'solver-audit.jsonl'
    with (ROOT / audit_path).open('xb') as stream:
        stream.write(journal.read_bytes())
    outputs.append(audit_path)
    profiles = {}
    for name in ('solver-audit', 'fit', 'evaluation'):
        profile = study.read(trial / (name + '-profile/profile.json'))
        study.legacy.check(profile['child_exit_code'] == 0 and profile['guard'] is None, 'All stages must complete without guard')
        profile['arguments'] = [profile['arguments'][0], '<private-scalar-trial>']
        profiles[name] = profile
    destination = PREFIX + 'resource-profiles.json'
    study.write(ROOT / destination, profiles)
    outputs.append(destination)
    source = {row['path']: row['sha256'] for row in frozen['files_sha256']}
    source.update({name: study.sha(ROOT / name) for name in outputs + list(EXTRA)})
    study.write(ROOT / (PREFIX + 'release.json'), {'schema': 'chipmunk.preserved-evidence-release/23',
        'study': 'preserved-evidence-v23', 'replay_module': 'scripts.reproduce_preserved_v23',
        'files_sha256': [{'path': name, 'sha256': value} for name, value in sorted(source.items())],
        'expected': compressed, 'freeze': PREFIX + 'freeze.json', 'absolute_limit': 1e-6,
        'serving_admitted': False, 'new_neural_forwards': 0,
        'limits': 'Public consumed-scalar optimization and combination replay; no raw media, face/voice features, customer files, fresh validation or outside human review.'})
    print(json.dumps({'published_files': len(outputs) + 1, 'exact_dependencies': len(source),
        'expected_compressed_bytes': (ROOT / compressed).stat().st_size, 'new_neural_forwards': 0}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trial', type=Path)
    args = parser.parse_args()
    publish(args.trial.resolve())
