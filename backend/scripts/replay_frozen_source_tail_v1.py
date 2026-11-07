"""Replay the immutable study snapshot through a corrected external CI wrapper.

The original workflow was included in the study's byte commitments and has a
YAML setup error. Archive exactly the published study commit, preserving those
original bytes, and run its numerical replay. No frozen study code is edited.
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
SCIENCE_COMMIT = 'fafafe88ab8db2bdd72d4612227bb396ab09f8ad'
RELEASE = 'docs/evaluation-panels/specialist-source-tail-v1-release.json'


def safe_path(name):
    if not isinstance(name, str) or not name or name.startswith('-'):
        raise ValueError('Explicit repository-relative source path required')
    path = PurePosixPath(name)
    if path.is_absolute() or '..' in path.parts or str(path) != name:
        raise ValueError('Source path escapes or changes its declared repository location')
    return name


def blob(name):
    return subprocess.check_output(['git', 'show', SCIENCE_COMMIT + ':' + safe_path(name)], cwd=ROOT)


def selected_files():
    release = json.loads(blob(RELEASE))
    names = {RELEASE, *map(safe_path, release['files_sha256'])}
    frozen = json.loads(blob(release['freeze']))
    for prior in frozen['prior_releases'].values():
        name = safe_path(prior['path'])
        source = blob(name)
        if hashlib.sha256(source).hexdigest() != prior['sha256']:
            raise ValueError('Original source release differs from its frozen commitment')
        names.add(name)
        names.update(map(safe_path, json.loads(source)['files_sha256']))
    # Include real ancestor package initializers when present in the snapshot.
    for name in list(names):
        if name.endswith('.py'):
            parent = PurePosixPath(name).parent
            while str(parent) not in ('.', 'backend'):
                candidate = str(parent / '__init__.py')
                result = subprocess.run(['git', 'cat-file', '-e', SCIENCE_COMMIT + ':' + candidate],
                    cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                if result.returncode == 0:
                    names.add(candidate)
                parent = parent.parent
    return sorted(names)


def replay(output):
    output = output.resolve()
    if output.exists():
        raise FileExistsError('Retain prior replay receipts')
    names = selected_files()
    with tempfile.TemporaryDirectory(prefix='source-tail-frozen-snapshot-') as temporary:
        folder = Path(temporary)
        archive, tree, numerical = folder / 'science.tar', folder / 'tree', folder / 'numerical.json'
        tree.mkdir()
        subprocess.run(['git', 'archive', '--format=tar', '--output=' + str(archive), SCIENCE_COMMIT,
            '--', *names], cwd=ROOT, check=True)
        with tarfile.open(archive) as bundle:
            bundle.extractall(tree, filter='data')
        actual_release = tree / RELEASE
        if actual_release.read_bytes() != blob(RELEASE):
            raise ValueError('Archive changed the immutable published release')
        environment = {**os.environ, 'PYTHONPATH': str(tree / 'backend'), 'OPENBLAS_NUM_THREADS': '1'}
        subprocess.run([sys.executable, '-m', 'scripts.reproduce_specialist_source_tail_v1',
            str(actual_release), str(numerical)], cwd=tree, env=environment, check=True,
            stdout=subprocess.PIPE, text=True)
        receipt = {'schema': 'chipmunk.specialist-source-tail-frozen-archive-replay/1',
            'science_commit': SCIENCE_COMMIT, 'frozen_release_sha256': hashlib.sha256(blob(RELEASE)).hexdigest(),
            'selected_repository_files': len(names), 'archive_sha256': hashlib.sha256(archive.read_bytes()).hexdigest(),
            'original_numerical_receipt_sha256': hashlib.sha256(numerical.read_bytes()).hexdigest(),
            'numerical': json.loads(numerical.read_text()), 'frozen_study_source_modified': False,
            'raw_media_or_weights_loaded': False, 'serving_admitted': False,
            'scope': 'Corrected external wrapper executes the original immutable consumed-source numerical study; no fresh detection or outside human review.'}
    with output.open('x') as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    replay(args.output)
