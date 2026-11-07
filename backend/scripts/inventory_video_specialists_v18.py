"""Audit consumed video parents and plan connected roles without inference.

The private inventory contains source URLs/donor filenames. Its public receipt
contains only aggregate counts and manifest commitments. This is a source audit,
not a completed detector experiment or a claim of distinct people.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

PARENTS = ('video-v2/face-swap', 'video-v3/face-swap', 'video-v4/face-sources',
    'video-v5/sources', 'video-v6/sources', 'video-v17-sface/sources')
INTERVALS = ('video-v4/intervals', 'video-v5/intervals', 'video-v6/intervals', 'video-v17-sface/intervals')
ROLES = ('specialist_cal', 'fusion_fit', 'fusion_cal', 'confirmation')
REVISION = 'bca865e675852666a426368b2f5614abf1c1926a'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def stable(value):
    return hashlib.sha256(('chipmunk-video-specialists-v18\0' + value).encode()).hexdigest()


def connected(rows):
    """Union every known source/donor/byte connection before assigning roles."""
    parent = {}

    def root(value):
        parent.setdefault(value, value)
        if parent[value] != value:
            parent[value] = root(parent[value])
        return parent[value]

    for row in rows:
        group = ('group', row['group'])
        for key, value in [('donor', row.get('donor_filename')), ('video_url', row.get('source_video_url')),
                *[('bytes', row.get(kind, {}).get('sha256')) for kind in ('original', 'matched')]]:
            if value:
                parent[root((key, value))] = root(group)
        root(group)
    components = defaultdict(list)
    for group in sorted({row['group'] for row in rows}):
        components[root(('group', group))].append(group)
    return sorted((sorted(value) for value in components.values()), key=lambda members: stable('\0'.join(members)))


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def inventory(main, trial):
    source_root = main / 'backend/data/independent-evaluation'
    sources, intervals, commitments, checked = [], [], {}, {}

    def validate_media(root, item):
        path = (root / item['path']).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError('Parent media path escapes its declared research source')
        if path not in checked:
            if sha(path) != item['sha256']:
                raise ValueError('Original/derived media changed from acquisition receipt')
            checked[path] = {'sha256': item['sha256'], 'bytes': path.stat().st_size}

    for name in PARENTS:
        path = source_root / name / 'manifest.json'
        value = json.loads(path.read_text())
        if value['revision'] != REVISION:
            raise ValueError('Source acquisition revision changed')
        commitments[str(path.relative_to(main))] = sha(path)
        for row in value['items']:
            if row['status'] != 'ready' or row['label'] != int(row['donor_filename'] is not None):
                raise ValueError('Failed or inconsistent parent retained; do not silently replace')
            for kind in ('original', 'matched'):
                validate_media(path.parent, row[kind])
            sources.append({**row, 'acquisition_manifest': str(path.relative_to(main))})
    components = connected(sources)
    groups = defaultdict(list)
    for row in sources:
        groups[row['group']].append(row)
    if len(groups) != 162 or any(len(value) != 3 or {v['family'] for v in value}
            != {'original_videos', 'inswapper', 'uniface'} for value in groups.values()):
        raise ValueError('Complete consumed162 paired source groups required')
    assignments = {}
    for index, members in enumerate(components):
        role, component = ROLES[index % 4], stable('\0'.join(members))
        for member in members:
            assignments[member] = {'role': role, 'component': component}
    for name in INTERVALS:
        path = source_root / name / 'manifest.json'
        value = json.loads(path.read_text())
        if value['source_manifest_sha256'] not in commitments.values():
            raise ValueError('Interval data are not bound to included original acquisition')
        commitments[str(path.relative_to(main))] = sha(path)
        for row in value['items']:
            if row['group'] not in assignments or row['status'] != 'ready':
                raise ValueError('Missing or failed interval source')
            validate_media(path.parent, row)
            if row['fps'] != 24 or (row['label'] == 1 and not row['exact_frames_verified']):
                raise ValueError('Exact positive interval truth and source clock required')
            intervals.append({**row, **assignments[row['group']],
                'interval_manifest': str(path.relative_to(main))})
    existing = {row['group'] for row in intervals}
    counts = {}
    for role in ROLES:
        cases = [row for row in intervals if row['role'] == role]
        counts[role] = {'connected_components': len({a['component'] for a in assignments.values() if a['role'] == role}),
            'source_video_groups': sum(a['role'] == role for a in assignments.values()),
            'groups_with_existing_intervals': len({v['group'] for v in cases}),
            'groups_requiring_new_interval_preparation': sum(a['role'] == role and g not in existing for g, a in assignments.items()),
            'existing_positive_clips': sum(v['label'] for v in cases),
            'existing_genuine_clips': sum(v['label'] == 0 for v in cases),
            'existing_ordinary_edit_controls': sum(v['label'] == 0 and v['family'] != 'original_videos' for v in cases)}
    if any(value['connected_components'] < 25 for value in counts.values()):
        raise ValueError('Insufficient independent known-source components for the shared support floor')
    trial.mkdir(exist_ok=False)
    plan = {'schema': 'chipmunk.video-specialists-private-source-inventory/18',
        'parent_manifest_sha256': commitments, 'parents': sources, 'intervals': intervals,
        'assignments': assignments, 'role_assignment': 'Salted component hash; all known source URL, donor filename and parent/derived byte connections unioned before round-robin assignment. No detector response enters assignment.',
        'new_detector_forwards': 0, 'serving_admitted': False}
    write(trial / 'source-inventory.json', plan)
    receipt = {'schema': 'chipmunk.video-specialists-source-audit/18',
        'implementation_sha256': sha(Path(__file__)), 'private_inventory_sha256': sha(trial / 'source-inventory.json'),
        'acquisition_revision': REVISION, 'parent_manifest_sha256': commitments,
        'source_video_groups': len(groups), 'known_connected_components': len(components),
        'component_size_counts': dict(sorted(Counter(map(len, components)).items())),
        'parents': len(sources), 'retained_interval_clips': len(intervals),
        'retained_interval_groups': len(existing), 'media_bytes_checked': sum(row['bytes'] for row in checked.values()),
        'unique_media_paths_checked': len(checked), 'roles': counts,
        'minimum_groups_per_class': 25, 'new_detector_forwards': 0,
        'new_video_calibration_fitted': False, 'new_video_calibration_tested': False, 'serving_admitted': False,
        'weight_scope': 'Existing pinned commercially licensed face models only; any additional model requires its own weight-specific primary grant. This inventory loads no model.',
        'dataset_license': 'RTFS-10K publisher declares CC-BY-SA-4.0; source publisher labels/rights are not independently guaranteed.',
        'limits': 'All sources previously consumed in development. Known source-video/donor-file/byte disjointness does not prove distinct people, original licensors rights, or disjoint external model training. No new lip-generation, natural edit, fresh-source or authenticity-confidence result.'}
    write(trial / 'source-audit.json', receipt)
    print(json.dumps({k: receipt[k] for k in ('schema', 'source_video_groups', 'known_connected_components', 'retained_interval_groups', 'roles', 'new_detector_forwards')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('main', type=Path)
    parser.add_argument('trial', type=Path)
    args = parser.parse_args()
    inventory(args.main, args.trial)
