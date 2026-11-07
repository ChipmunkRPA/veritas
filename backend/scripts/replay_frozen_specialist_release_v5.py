"""External model-free replay for immutable four-media specialist releases.

Archives only the published scalar release's declared files and package
initializers. Failed fitting remains a failed check; nothing retries or changes
the original study, observations, optimizer, tolerance or serving defaults.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[2]


def validate(commit, name):
    if not isinstance(commit, str) or len(commit) != 40 or any(c not in '0123456789abcdef' for c in commit):
        raise ValueError('Explicit immutable40-character commit required')
    if not isinstance(name, str) or not name or name.startswith('-'):
        raise ValueError('Explicit repository-relative source path required')
    path = PurePosixPath(name)
    if path.is_absolute() or '..' in path.parts or str(path) != name:
        raise ValueError('Source path escapes or changes its declared repository location')
    return name


def replay(commit, release_name, output):
    validate(commit, release_name)
    if output.exists():
        raise FileExistsError('Retain previous numerical replay receipts')

    def blob(name):
        return subprocess.check_output(['git', 'show', commit + ':' + validate(commit, name)], cwd=ROOT)

    release_bytes = blob(release_name)
    release = json.loads(release_bytes)
    if release.get('serving_admitted') is not False or release.get('modality') not in ('audio', 'video', 'image', 'text'):
        raise ValueError('Explicit research specialist release required')
    names = {release_name, *[validate(commit, name) for name in release['files_sha256']]}
    required = {'backend/scripts/reproduce_specialist_study_v4.py', release['manifests'], release['expected']}
    if not required <= names:
        raise ValueError('Pinned original numerical runner and complete planned/expected scalar packets required')
    for name in list(names):
        if name.endswith('.py'):
            parent = PurePosixPath(name).parent
            while str(parent) not in ('.', 'backend'):
                candidate = str(parent / '__init__.py')
                if subprocess.run(['git', 'cat-file', '-e', commit + ':' + candidate], cwd=ROOT,
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0:
                    names.add(candidate)
                parent = parent.parent
    with tempfile.TemporaryDirectory(prefix='chipmunk-frozen-specialist-release-') as temp:
        folder = Path(temp)
        archive, tree, numerical = folder / 'science.tar', folder / 'tree', folder / 'numerical.json'
        tree.mkdir()
        subprocess.run(['git', 'archive', '--format=tar', '--output=' + str(archive), commit,
            '--', *sorted(names)], cwd=ROOT, check=True)
        with tarfile.open(archive) as bundle:
            bundle.extractall(tree, filter='data')
        if (tree / release_name).read_bytes() != release_bytes:
            raise ValueError('Original release bytes changed during archival')
        process = subprocess.run([sys.executable, '-m', 'scripts.reproduce_specialist_study_v4',
            str(tree / release_name), str(numerical)], cwd=tree,
            env={**os.environ, 'PYTHONPATH': str(tree / 'backend'), 'OPENBLAS_NUM_THREADS': '1'},
            capture_output=True, text=True)
        if process.returncode:
            raise RuntimeError('Immutable numerical replay failed without retry; original diagnostic:\n' + process.stderr[-8000:])
        result = {'schema': 'chipmunk.immutable-specialist-release-replay/5', 'science_commit': commit,
            'release': release_name, 'release_sha256': hashlib.sha256(release_bytes).hexdigest(),
            'selected_repository_files': len(names), 'numerical': json.loads(numerical.read_text()),
            'frozen_science_modified': False, 'new_neural_forwards': 0, 'serving_admitted': False,
            'scope': 'Immutable consumed-source scalar replay, not native inference, fresh detection quality or outside human review.'}
    with output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('commit')
    parser.add_argument('release')
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    replay(args.commit, args.release, args.output)
