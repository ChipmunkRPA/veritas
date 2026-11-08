"""Archive and replay one immutable reviewed checked-solver study, no retries."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path, PurePosixPath

from scripts.replay_frozen_specialist_release_v5 import validate

ROOT = Path(__file__).resolve().parents[2]
RELEASE = 'docs/evaluation-panels/preserved-evidence-v22-release.json'


def replay(commit, output):
    validate(commit, RELEASE)
    if output.exists():
        raise FileExistsError('Preserve the previous immutable replay')
    original = subprocess.check_output(['git', 'show', commit + ':' + RELEASE], cwd=ROOT)
    release = json.loads(original)
    if release.get('schema') != 'chipmunk.preserved-evidence-release/22' or release.get('replay_module') != 'scripts.reproduce_preserved_v22':
        raise ValueError('Only the declared original checked-solver replay is supported')
    names = {RELEASE, *[validate(commit, row['path']) for row in release['files_sha256']]}
    required = {'backend/scripts/reproduce_preserved_v22.py', release['freeze'], release['expected']}
    if not required <= names:
        raise ValueError('Complete numerical runner, freeze and expected outputs required')
    for name in list(names):
        if name.endswith('.py'):
            parent = PurePosixPath(name).parent
            while str(parent) not in ('.', 'backend'):
                candidate = str(parent / '__init__.py')
                if subprocess.run(['git', 'cat-file', '-e', commit + ':' + candidate], cwd=ROOT,
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0:
                    names.add(candidate)
                parent = parent.parent
    temp = Path(tempfile.mkdtemp(prefix='chipmunk-preserved-v22-immutable-', dir=output.parent))
    tree = temp / 'tree'
    tree.mkdir()
    try:
        archive = temp / 'source.tar'
        subprocess.run(['git', 'archive', '--format=tar', '--output=' + str(archive), commit, '--', *sorted(names)], cwd=ROOT, check=True)
        with tarfile.open(archive) as bundle:
            bundle.extractall(tree, filter='data')
        if (tree / RELEASE).read_bytes() != original:
            raise ValueError('Immutable release bytes changed')
        target = temp / 'numerical.json'
        job = subprocess.run([sys.executable, '-m', 'scripts.reproduce_preserved_v22', str(tree / RELEASE), str(target)],
            cwd=tree, env={**os.environ, 'PYTHONPATH': str(tree / 'backend'), 'OPENBLAS_NUM_THREADS': '1'},
            capture_output=True, text=True)
        if job.returncode:
            raise RuntimeError('Immutable fit/replay failed without retry:\n' + job.stderr[-10000:])
        receipt = {'schema': 'chipmunk.immutable-preserved-evidence-replay/22', 'science_commit': commit,
            'archived_files': len(names), 'numerical': json.loads(target.read_text()),
            'frozen_science_modified': False, 'new_neural_forwards': 0, 'serving_admitted': False}
        with output.open('x') as stream:
            json.dump(receipt, stream, indent=2, allow_nan=False)
            stream.write('\n')
    except BaseException as error:
        with (temp / 'immutable-replay-failure.json').open('x') as stream:
            json.dump({'error_type': type(error).__name__, 'error': str(error), 'science_commit': commit,
                'frozen_science_modified': False}, stream, indent=2)
        raise
    else:
        shutil.rmtree(temp)
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('commit')
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    replay(args.commit, args.output.resolve())
