"""Source-bound video input preparation; schedule after heavy audio inference.

Retain existing clips and verify their current frame clocks. New intervals and
ordinary controls are virtual: create one group at a time, verify and commit its
bytes/clock, then delete only that temporary materialization. No model is loaded.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

from scripts import inventory_video_specialists_v18 as inventory

ROOT = Path(__file__).resolve().parents[2]


def require(value, reason):
    if not value:
        raise ValueError(reason)


def frame_clock(table, count, *, expected_fps=24):
    """Use measured PTS; inserted swaps must retain their exact 24 fps recipe.

    Older ordinary jump-cut controls can be 25 fps despite a nominal24 manifest
    field. Their measured clock is retained, never replaced with frame_index/24.
    """
    require(not table['truncated'] and not table['interpolated'] and len(table['pts']) == count,
        'Missing, inferred or truncated native frame clock')
    require(table['width'] == 640 and table['height'] == 480 and table['rotation'] == 0,
        'Native frame geometry differs from planned source')
    base = Fraction(table['time_base'])
    pts = table['pts']
    require(count >= 2 and pts[0] == 0, 'Native source clock has no supported zero-origin extent')
    step = (pts[1] - pts[0]) * base
    require(step > 0 and all((b - a) * base == step for a, b in zip(pts, pts[1:], strict=False)),
        'Native source frame clock is not constant')
    require(expected_fps is None or step == Fraction(1, expected_fps), 'Native timestamps differ from exact24fps recipe')
    return {'duration_ns': round(count * step * 10**9), 'frame_count': count, 'native_frame_rate': str(1 / step),
        'width': 640, 'height': 480, 'pts_table_sha256': hashlib.sha256(
            json.dumps(table, sort_keys=True, separators=(',', ':')).encode()).hexdigest()}


def truth(row):
    require(row['status'] == 'ready' and row['fps'] == 24 and type(row['label']) is int
        and row['label'] in (0, 1), 'Incomplete planned source/label')
    count = row['frame_count']
    require(type(count) is int and count >= 2, 'Invalid native source extent')
    if not row['label']:
        require(row['edit_s'] is None, 'Ordinary control incorrectly received AI truth')
        return None
    a, b = row['inserted_frames']
    require(all(type(v) is int for v in (a, b)) and 0 <= a < b <= count
        and row['exact_frames_verified'] and row['edit_s'] == [a / 24, b / 24],
        'Positive interval is not tied to verified exact source frames')
    return {'frames': [a, b], 'time_ns': [round(a * Fraction(10**9, 24)), round(b * Fraction(10**9, 24))]}


def helper():
    # Legacy source-preparation scripts import each other by script name.
    sys.path.insert(0, str(ROOT / 'backend/scripts'))
    from scripts import prepare_face_interval_eval

    return prepare_face_interval_eval


def ordinary(original, source, output):
    """Add a genuine ordinary-edit control to an older three-case group."""
    method = helper()
    choice = hashlib.sha256(original['group'].encode()).digest()
    family = ('jump-cut', 'lighting-change', 'recompression')[choice[2] % 3]
    start, end = original['inserted_frames']
    if family == 'jump-cut':
        transform = ['-vf', f"select='not(between(n,{start},{start+7}))',setpts=N/(24*TB)", *method.ENCODE]
    elif family == 'lighting-change':
        transform = ['-vf', f"eq=brightness=0.14:enable='between(n,{start},{end-1})'", *method.ENCODE]
    else:
        transform = ['-c:v', 'libx264', '-crf', '38', '-threads', '2']
    # Specify the intended output rate. Earlier preparation let FFmpeg choose
    # 25 fps for some controls, introducing duplicate frames after a24fps cut.
    method.run(['-i', str(source), *transform, '-r', '24', str(output)])
    original_hashes, changed_hashes = method.hashes(source), method.hashes(output)
    require(changed_hashes != original_hashes, 'Ordinary control contains no intended change')
    expected_count = len(original_hashes) - (8 if family == 'jump-cut' else 0)
    require(len(changed_hashes) == expected_count, 'Ordinary-control source extent changed unexpectedly')
    if family == 'jump-cut':
        require(changed_hashes == original_hashes[:start] + original_hashes[start + 8:],
            'Ordinary jump-cut no longer maps exactly to the genuine parent frames')
    return {**original, 'id': original['id'] + '-ordinary', 'path': output.name,
        'sha256': inventory.sha(output), 'family': family, 'frame_count': expected_count,
        'inserted_frames': None, 'edit_s': None, 'exact_frames_verified': family == 'jump-cut'}


def code():
    paths = ['backend/scripts/prepare_video_specialists_v18.py', 'backend/scripts/inventory_video_specialists_v18.py',
        'backend/scripts/prepare_face_interval_eval.py', 'backend/scripts/prepare_video_upgrade_eval.py',
        'backend/app/detectors/video/decode.py']
    return {name: inventory.sha(ROOT / name) for name in paths}


def audit_retained_clocks(main, trial):
    """Read existing production probe receipts; do not decode/run any model."""
    plan = json.loads((trial / 'source-inventory.json').read_text())
    source = main / 'backend/data/independent-evaluation'
    seen, counts, commitments, mismatches = set(), Counter(), {}, 0
    maximum_duration_error = Fraction(0)
    journals = {'video-v4': ('development.jsonl', 'test.jsonl'),
        'video-v5': ('current-development.jsonl', 'current-test.jsonl'),
        'video-v6': ('current-test.jsonl',), 'video-v17-sface': ('baseline.jsonl',)}
    for name, filenames in journals.items():
        path = source / name / 'intervals/manifest.json'
        expected = {row['id']: row for row in json.loads(path.read_text())['items']}
        for filename in filenames:
            journal = source / name / filename
            commitments[str(journal.relative_to(main))] = inventory.sha(journal)
            with journal.open() as stream:
                header = json.loads(next(stream))
                require(header['manifest_sha256'] == inventory.sha(path), 'Historical native receipt has different source input manifest')
                for line in stream:
                    row = json.loads(line)
                    require(row['id'] in expected and all(row.get(k) == v for k, v in expected[row['id']].items()),
                        'Historical probe receipt source metadata changed')
                    key = name, row['id']
                    require(key not in seen, 'Duplicate historical native frame table')
                    seen.add(key)
                    ordinary_control = row['label'] == 0 and row['family'] != 'original_videos'
                    geometry = frame_clock(row['table'], row['frame_count'], expected_fps=None if ordinary_control else 24)
                    truth(row)
                    counts[row['family'] + '/' + geometry['native_frame_rate']] += 1
                    nominal = row['fps']
                    actual = Fraction(geometry['native_frame_rate'])
                    mismatches += actual != nominal
                    maximum_duration_error = max(maximum_duration_error,
                        abs(Fraction(row['frame_count'], nominal) - Fraction(row['frame_count']) / actual))
    require(len(seen) == len(plan['intervals']) == 432, 'Every retained source/clock denominator is required')
    receipt = {'schema': 'chipmunk.video-specialists-retained-clock-audit/18',
        'implementation_sha256': inventory.sha(Path(__file__)), 'source_inventory_sha256': inventory.sha(trial / 'source-inventory.json'),
        'historical_native_journal_sha256': commitments, 'retained_clip_truth_bindings_verified': len(seen),
        'previously_measured_native_pts_tables_verified': len(seen), 'native_family_rate_counts': dict(sorted(counts.items())),
        'nominal_manifest_rate_mismatches': mismatches,
        'maximum_nominal_duration_error_ms': float(maximum_duration_error * 1000),
        'positive_intervals_require_actual24fps': True, 'ordinary_controls_keep_actual_source_clock': True,
        'new_decodes': 0, 'new_detector_forwards': 0, 'serving_admitted': False,
        'limits': 'Checks retained native production probe receipts and source bindings, not fresh decoding, detection quality, distinct people or independent provenance.'}
    inventory.write(trial / 'retained-clock-audit.json', receipt)
    print(json.dumps(receipt), flush=True)


def prepare(main, trial):
    from app.detectors.video.decode import probe_frames

    plan_path = trial / 'source-inventory.json'
    plan = json.loads(plan_path.read_text())
    audit = json.loads((trial / 'source-audit.json').read_text())
    require(audit['private_inventory_sha256'] == inventory.sha(plan_path), 'Consumed-source assignment inventory changed')
    for name, expected in plan['parent_manifest_sha256'].items():
        require(inventory.sha(main / name) == expected, 'Original/interval acquisition manifest changed')
    parents, existing = defaultdict(list), defaultdict(list)
    for row in plan['parents']:
        parents[row['group']].append(row)
    for row in plan['intervals']:
        existing[row['group']].append(row)
    policy = {'schema': 'chipmunk.video-specialists-input-preparation-policy/18', 'source_inventory_sha256': inventory.sha(plan_path),
        'implementation_sha256': code(), 'ffmpeg_version': subprocess.check_output(['ffmpeg', '-version'], text=True).splitlines()[0],
        'source_groups': 162, 'planned_clips': 648, 'new_detector_forwards': 0, 'serving_admitted': False,
        'limits': 'Consumed publisher-labeled sources. Source PTS and pixel-preserving controls are checked; external training overlap and distinct people are not established.'}
    inventory.write(trial / 'input-preparation-policy.json', policy)
    seen, counts, virtual_count = set(), defaultdict(int), 0
    path = trial / 'input-plan.jsonl'
    with path.open('x') as stream:
        stream.write(json.dumps({'schema': 'chipmunk.video-specialists-input-plan/18',
            'policy_sha256': inventory.sha(trial / 'input-preparation-policy.json')}) + '\n')
        for group in sorted(parents, key=inventory.stable):
            assignment = plan['assignments'][group]
            with tempfile.TemporaryDirectory(prefix='video-v18-group-', dir=trial) as temp:
                output = Path(temp)
                cases = []
                for row in existing[group]:
                    source = main / Path(row['interval_manifest']).parent / row['path']
                    require(inventory.sha(source) == row['sha256'], 'Retained source interval bytes changed')
                    cases.append((row, source, {'kind': 'retained', 'manifest': row['interval_manifest'], 'path': row['path']}))
                if not cases:
                    source_rows = sorted(parents[group], key=lambda r: (r['label'], r['family']))
                    require(len({r['acquisition_manifest'] for r in source_rows}) == 1, 'A virtual group crosses acquisition roots')
                    source_root = (main / source_rows[0]['acquisition_manifest']).parent
                    (output / 'normalized').mkdir()
                    generated = helper().prepare(source_rows, source_root, output, development=False, hard_controls=False)
                    require(len(generated) == 3 and all(row['status'] == 'ready' for row in generated),
                        'Unusable new interval group retained; no replacement source')
                    original = next(row for row in generated if row['family'] == 'original_videos')
                    generated.append(ordinary(original, output / original['path'], output / (original['id'] + '-ordinary.mp4')))
                    for row in generated:
                        cases.append((row, output / row['path'], {'kind': 'virtual-interval-group',
                            'source_manifest': source_rows[0]['acquisition_manifest'], 'group': group}))
                elif len(cases) == 3:
                    row, source, _ = next(case for case in cases if case[0]['family'] == 'original_videos')
                    control = ordinary(row, source, output / (row['id'] + '-ordinary.mp4'))
                    cases.append((control, output / control['path'], {'kind': 'virtual-ordinary-control',
                        'original_interval_manifest': row['interval_manifest'], 'original_id': row['id']}))
                require(len(cases) == 4 and sum(row['label'] for row, _, _ in cases) == 2,
                    'All planned genuine, ordinary and two-swap cases required; no silent source replacement')
                for row, source, recipe in sorted(cases, key=lambda case: case[0]['id']):
                    identity = inventory.stable(group + '/' + row['family'])
                    require(identity not in seen and row['group'] == group, 'Repeated/misattributed clip')
                    seen.add(identity)
                    ordinary_control = row['label'] == 0 and row['family'] != 'original_videos'
                    geometry = frame_clock(probe_frames(source, timeout_s=30).to_json(), row['frame_count'],
                        expected_fps=None if ordinary_control else 24)
                    packet = {'id': identity, 'source_id': row['id'], 'group': group, **assignment,
                        'family': row['family'], 'label': row['label'], 'sha256': row['sha256'],
                        'bytes': source.stat().st_size, 'source_geometry': geometry, 'manifest_nominal_fps': row['fps'], 'known_edit': truth(row),
                        'materialization': recipe}
                    stream.write(json.dumps(packet, allow_nan=False) + '\n')
                    stream.flush()
                    virtual_count += recipe['kind'] != 'retained'
                    counts[assignment['role']] += 1
            print(json.dumps({'prepared_clips': len(seen), 'planned_clips': 648, 'new_detector_forwards': 0}), flush=True)
    require(len(seen) == 648, 'Incomplete planned source denominator')
    receipt = {'schema': 'chipmunk.video-specialists-input-preparation/18',
        'policy_sha256': inventory.sha(trial / 'input-preparation-policy.json'), 'input_plan_sha256': inventory.sha(path),
        'source_groups': len(parents), 'clips': len(seen), 'virtual_clips': virtual_count,
        'roles': dict(counts), 'all_native_frame_clocks_verified': True, 'new_detector_forwards': 0, 'serving_admitted': False}
    inventory.write(trial / 'input-preparation.json', receipt)
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('main', type=Path)
    parser.add_argument('trial', type=Path)
    parser.add_argument('--audit-retained-clocks', action='store_true')
    args = parser.parse_args()
    if args.audit_retained_clocks:
        audit_retained_clocks(args.main, args.trial)
    else:
        prepare(args.main, args.trial)
