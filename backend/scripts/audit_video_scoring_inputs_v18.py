"""Decode prepared video sources before authenticity inference or calibration.

Check the actual changed-frame footprint, local labels, and nonuniform sampled
RGB duplicates across connected roles. No detector weights or model is loaded.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from scripts import inventory_video_specialists_v18 as inventory
from scripts import materialize_video_specialists_v18 as media

require = media.require
ROLES = ('specialist_cal', 'fusion_fit', 'fusion_cal', 'confirmation')
METHOD = {'source': 'All648 prepared consumed clips; reconstruct/verify exact input bytes and actual native clock.',
    'truth': 'AI source-frame presence requires publisher AI label, insertion into the known interval, and an actual decoded RGB change from the original. Ordinary controls remain genuine even when pixels change. No identified-face/pixel-authorship truth.',
    'connections': 'Seed original known source/donor/byte components. Join groups sharing an exactly equal nonuniform decoded RGB frame among predeclared model input frames. Uniform color frames are not identity/source links.',
    'roles': 'Preserve original roles if every refined component stays inside one role. If any component crosses roles, reassign all refined components in salted-hash order round-robin before any authenticity inference. No scores choose roles.',
    'local': 'Recompute sampled local labels from actual changed-frame indices; preserve full-clip truth and unobserved-edited-frame count separately.',
    'new_authenticity_forwards': 0, 'serving_admitted': False}
ROOT = Path(__file__).resolve().parents[2]
CODE = ('backend/scripts/audit_video_scoring_inputs_v18.py', 'backend/scripts/run_video_specialists_v18.py',
    'backend/scripts/prepare_video_scoring_units_v18.py', 'backend/scripts/materialize_video_specialists_v18.py',
    'backend/scripts/video_specialist_units_v18.py', 'backend/scripts/inventory_video_specialists_v18.py',
    'backend/scripts/prepare_video_specialists_v18.py', 'backend/scripts/prepare_face_interval_eval.py',
    'backend/scripts/prepare_video_upgrade_eval.py', 'backend/scripts/specialist_runtime_v4.py',
    'backend/app/detectors/video/decode.py', 'backend/tests/test_video_scoring_source_audit_v18.py',
    'backend/tests/test_video_scoring_plan_v18.py')


def identity(trial):
    return {'schema': 'chipmunk.video-source-frame-audit-freeze/18', 'method': METHOD,
        'code_sha256': {name: inventory.sha(ROOT / name) for name in CODE},
        'inputs_sha256': {name: inventory.sha(trial / name) for name in ('scoring-input-plan.json',
            'input-preparation.json', 'input-preparation-policy.json', 'input-plan.jsonl', 'source-inventory.json')},
        'packages': {name: importlib.metadata.version(name) for name in ('numpy', 'opencv-python-headless')},
        'binaries_sha256': {name: inventory.sha(Path(shutil.which(name))) for name in ('ffmpeg', 'ffprobe')},
        'new_neural_forwards': 0, 'serving_admitted': False}


def freeze(trial):
    inventory.write(trial / 'source-frame-audit-freeze-v1.json', identity(trial))
    print(json.dumps({'freeze_sha256': inventory.sha(trial / 'source-frame-audit-freeze-v1.json'), 'new_neural_forwards': 0}), flush=True)


def check(trial):
    require(json.loads((trial / 'source-frame-audit-freeze-v1.json').read_text()) == identity(trial),
        'Frozen source audit implementation, inputs or runtime changed')


def changed_frames(original, altered, interval):
    require(len(original) == len(altered), 'Paired AI/original native frame extents differ')
    a, b = interval
    changed = [i for i, (first, second) in enumerate(zip(original, altered, strict=True)) if first != second]
    require(changed and all(a <= i < b for i in changed), 'AI insertion changes pixels outside its interval or changes nothing')
    return changed


def components(files):
    parent, seen = {}, {}

    def root(key):
        parent.setdefault(key, key)
        if parent[key] != key:
            parent[key] = root(parent[key])
        return parent[key]

    def join(first, second):
        parent[root(first)] = root(second)

    original, connections = {}, 0
    for row in files:
        group = row['group']
        root(group)
        if row['component'] in original:
            join(group, original[row['component']])
        else:
            original[row['component']] = group
        for key in row['nonuniform_model_input_rgb_sha256']:
            if key in seen:
                other = seen[key]
                if root(group) != root(other):
                    connections += 1
                    join(group, other)
            else:
                seen[key] = group
    members, old_roles = defaultdict(set), defaultdict(set)
    for row in files:
        key = root(row['group'])
        members[key].add(row['group'])
        old_roles[key].add(row['role'])
    ordered = sorted(members, key=lambda k: inventory.stable('\0'.join(sorted(members[k]))))
    crossing = sum(len(old_roles[key]) > 1 for key in ordered)
    assignments = {}
    for index, key in enumerate(ordered):
        role = ROLES[index % 4] if crossing else next(iter(old_roles[key]))
        identity = inventory.stable('decoded-frame-component\0' + '\0'.join(sorted(members[key])))
        for group in members[key]:
            assignments[group] = {'role': role, 'scoring_component': identity}
    return assignments, {'components': len(ordered), 'additional_rgb_connections': connections,
        'components_crossing_original_roles': crossing, 'role_reassignment_required': bool(crossing)}


def audit(main, trial):
    from app.detectors.video.decode import FrameTable, decode_frames

    check(trial)
    original, grouped = media.inputs(main, trial)
    plan_path = trial / 'scoring-input-plan.json'
    plan = json.loads(plan_path.read_text())
    require(len(plan['files']) == 648 and plan['scoring_protocol_complete'] is False, 'Complete prescore geometry plan required')
    planned = {row['id']: row for row in plan['files']}
    journal = trial / 'source-frame-audit-v1.jsonl'
    require(not journal.exists() and not (trial / 'source-frame-audit-v1.json').exists(), 'Preserve previous source audit; no implicit resume')
    frames, files = {}, []
    counts = Counter()
    with journal.open('x') as stream:
        stream.write(json.dumps({'schema': 'chipmunk.video-source-frame-audit-journal/18',
            'method': METHOD, 'source_plan_sha256': inventory.sha(plan_path),
            'implementation_sha256': inventory.sha(Path(__file__))}) + '\n')
        for group in sorted(grouped, key=inventory.stable):
            with media.group_files(main, trial, original, grouped[group]) as cases:
                packets = []
                for case in cases:
                    row, path, table = case['planned'], case['path'], case['table']
                    current = planned[row['id']]
                    support = {i for unit in current['units'] for key in ('sampled_frames', 'context_frames') for i in unit['location'][key]}
                    native = FrameTable.from_json(table)
                    hashes, qualified = [], []
                    for frame in decode_frames(path, native, list(range(len(native))), timeout_s=120):
                        require(frame.index == len(hashes), 'Decoded frame sequence is incomplete or reordered')
                        digest = hashlib.sha256(frame.rgb.tobytes()).hexdigest()
                        hashes.append(digest)
                        if frame.index in support and (frame.rgb != frame.rgb[0, 0]).any():
                            qualified.append(digest)
                    require(len(hashes) == row['source_geometry']['frame_count'], 'Incomplete native source decoding')
                    frames[row['id']] = hashes
                    packets.append({'id': row['id'], 'group': group, 'role': row['role'], 'component': row['component'],
                        'input_sha256': row['sha256'], 'rgb_sha256': hashes,
                        'nonuniform_model_input_rgb_sha256': sorted(set(qualified)),
                        'native_pts_table_sha256': row['source_geometry']['pts_table_sha256']})
                reference = next(case['planned'] for case in cases if case['planned']['family'] == 'original_videos')
                for packet in packets:
                    row = planned[packet['id']]
                    actual = changed_frames(frames[reference['id']], frames[row['id']], row['known_edit']['frames']) if row['label'] else []
                    packet['actual_changed_ai_source_frames'] = actual
                    counts['declared_ai_frames'] += len(range(*row['known_edit']['frames'])) if row['label'] else 0
                    counts['actual_changed_ai_source_frames'] += len(actual)
                    counts['native_rgb_frames'] += len(packet['rgb_sha256'])
                    files.append(packet)
                    stream.write(json.dumps(packet, allow_nan=False) + '\n')
                stream.flush()
                counts['complete_clips'] += len(cases)
                print(json.dumps({'complete_source_clips': counts['complete_clips'], 'planned_source_clips': 648,
                    'new_authenticity_forwards': 0}), flush=True)
                for case in cases:
                    frames.pop(case['planned']['id'])
    require(counts['complete_clips'] == 648, 'Every planned source case required')
    check(trial)
    assignment, graph = components(files)
    by_id = {row['id']: row for row in files}
    scored_files, roles = [], Counter()
    for row in plan['files']:
        measured = by_id[row['id']]
        changed = set(measured['actual_changed_ai_source_frames'])
        units = [{**unit, 'label': int(bool(changed & set(unit['location']['sampled_frames'])))} for unit in row['units']]
        observed = {i for unit in units for i in unit['location']['sampled_frames']}
        current = {**row, 'preparation_role': row['role'], **assignment[row['group']], 'units': units,
            'actual_changed_ai_source_frames': sorted(changed), 'planned_scope_label': int(any(u['label'] for u in units)),
            'unobserved_known_edit_frames': len(changed - observed)}
        scored_files.append(current)
        roles[current['role']] += 1
        counts['clips_with_ai_outside_all_sampled_support'] += row['label'] == 1 and current['planned_scope_label'] == 0
    inventory.write(trial / 'scoring-input-plan-v2.json', {**plan, 'method': METHOD, 'files': scored_files,
        'source_frame_audit_journal_sha256': inventory.sha(journal), 'role_files': dict(roles),
        'refined_source_graph': graph, 'scoring_protocol_complete': False})
    receipt = {'schema': 'chipmunk.video-source-frame-audit/18', 'method': METHOD,
        'freeze_sha256': inventory.sha(trial / 'source-frame-audit-freeze-v1.json'),
        'implementation_sha256': inventory.sha(Path(__file__)), 'source_plan_sha256': inventory.sha(plan_path),
        'completed_source_plan_sha256': inventory.sha(trial / 'scoring-input-plan-v2.json'),
        'journal_sha256': inventory.sha(journal), **counts, **graph, 'roles': dict(roles),
        'new_neural_forwards': 0, 'serving_admitted': False,
        'limits': 'Actual RGB difference and publisher source labels, not identity/pixel authorship or detection quality. Source groups/people/pretraining may have unknown connections.'}
    inventory.write(trial / 'source-frame-audit-v1.json', receipt)
    print(json.dumps(receipt), flush=True)
