"""Publish scalar/time evidence only, after the complete private video review.

No clips, frame crops, face geometry, alignments, features or per-frame hashes.
This is numerical reproducibility, not independent human/source validation.
"""
from __future__ import annotations

import argparse
import copy
import gzip
import json
from pathlib import Path

from app.bench import specialists as engine
from scripts import inventory_video_specialists_v18 as source
from scripts import score_video_specialists_v20 as study

ROOT = Path(__file__).resolve().parents[2]
PREFIX = 'docs/evaluation-panels/video-face-specialists-v20-'


def assert_probabilities(actual, original):
    maximum = 0.
    for key in ('outcomes', 'unit_outcomes'):
        engine.check(len(actual[key]) == len(original[key]), 'All scalar outputs required')
        for a, b in zip(actual[key], original[key], strict=True):
            x, y = a['probability'], b['probability']
            if x is None or y is None:
                engine.check(x is y, 'Public packaging changed available evidence')
            else:
                maximum = max(maximum, abs(x - y))
    engine.check(maximum <= 1e-12, 'Public identity remapping changed fitted probabilities')
    return maximum


def public_manifest(private):
    current = copy.deepcopy(private)
    for index, file in enumerate(current['files']):
        file['id'] = 'source-input.' + str(index).zfill(4)
        file['group'] = 'component.' + engine.digest({'study': 'video-face-v20', 'private': file['group']})
        file['lineage'] = ['lineage.' + engine.digest({'study': 'video-face-v20', 'private': value}) for value in file['lineage']]
    engine.audit(current)
    return current


def compressed(path, value):
    with path.open('xb') as stream:
        stream.write(gzip.compress(json.dumps(value, separators=(',', ':'), allow_nan=False).encode(), mtime=0))


def release(trial):
    lock, _ = study.check(trial)
    study.verify_fit(trial)
    final = json.loads((trial / 'v20-result.json').read_text())
    binding = json.loads((trial / 'v20-full-native-binding-review.json').read_text())
    engine.check(binding['clips'] == 648 and binding['roles'] == list(engine.ROLES)
        and binding['exact_eligible_inputs_crossing_roles'] == 0 and binding['new_neural_forwards'] == 0,
        'Complete read-only source/input review required before publication')
    packets, expected, outputs, profiles, maxima = {}, {}, [], {}, {}
    for role in engine.ROLES:
        folder = role.replace('_', '-') + '-v20-profile'
        profile = json.loads((trial / folder / 'profile.json').read_text())
        engine.check(profile['child_exit_code'] == 0 and profile['guard'] is None, 'Every native role must complete without a guard')
        profile['arguments'] = ['native', '<private-video-trial>', '--role', role]
        profiles[role] = profile
        target = ROOT / (PREFIX + role + '-native.json')
        source.write(target, json.loads((trial / (role + '-v20-native.json')).read_text()))
        outputs.append(str(target.relative_to(ROOT)))
    for arm in study.ARMS:
        private = study.manifest(trial, arm, True)
        original = json.loads((trial / (arm + '-v20-confirmation.json')).read_text())
        current = public_manifest(private)
        artifact = engine.train(current, min_groups_per_class=25, l2=.001)
        result = engine.evaluate(current, artifact)
        maxima[arm] = assert_probabilities(result, original)
        packets[arm], expected[arm] = current, result
        for suffix, value in (('artifact', artifact), ('points', json.loads((trial / (arm + '-v20-points.json')).read_text()))):
            target = ROOT / (PREFIX + arm + '-' + suffix + '.json')
            source.write(target, value)
            outputs.append(str(target.relative_to(ROOT)))
    manifest_path, expected_path = PREFIX + 'manifests.json.gz', PREFIX + 'expected.json.gz'
    compressed(ROOT / manifest_path, packets)
    compressed(ROOT / expected_path, expected)
    outputs.extend((manifest_path, expected_path))
    for suffix, value in (('result', final), ('native-binding-review', binding),
            ('resource-profiles', profiles), ('fit-complete', json.loads((trial / 'v20-fit-complete.json').read_text())),
            ('training-native', json.loads((trial / 'v20-training-native.json').read_text()))):
        target = ROOT / (PREFIX + suffix + '.json')
        source.write(target, value)
        outputs.append(str(target.relative_to(ROOT)))
    dependencies = ['backend/app/bench/specialists.py', 'backend/scripts/reproduce_specialist_study_v4.py',
        'backend/scripts/release_video_scalars_v20.py', 'backend/scripts/audit_video_native_bindings_v20.py',
        'backend/tests/test_video_native_binding_review_v20.py', 'backend/tests/test_video_scalar_release_v20.py',
        'backend/scripts/score_video_specialists_v20.py', 'backend/scripts/measure_video_specialists_v18.py',
        'backend/scripts/specialist_calibration.py', 'backend/scripts/inventory_video_specialists_v18.py',
        'backend/scripts/audit_video_scoring_inputs_v18.py', 'backend/scripts/materialize_video_specialists_v18.py',
        'backend/scripts/prepare_video_specialists_v18.py', 'backend/scripts/prepare_face_interval_eval.py',
        'backend/scripts/prepare_video_upgrade_eval.py', 'backend/scripts/specialist_runtime_v4.py',
        'backend/scripts/specialist_runtime_v7.py', 'backend/scripts/run_video_specialists_v18.py',
        'backend/scripts/prepare_video_scoring_units_v18.py', 'backend/scripts/video_specialist_units_v18.py',
        'backend/app/detectors/video/decode.py', 'backend/app/detectors/video/face_sequence.py',
        'backend/app/detectors/video/face_appearance.py', 'backend/app/detectors/image/gend.py',
        'backend/app/detectors/image/faces.py', 'docs/video-face-specialists-v20-protocol.md',
        'docs/evaluation-panels/video-specialists-v18-source-frame-audit.json',
        'docs/evaluation-panels/video-face-specialists-v20-freeze.json']
    outputs.extend(dependencies)
    source.write(ROOT / (PREFIX + 'release.json'), {'schema': 'chipmunk.video-face-specialist-release/20',
        'study': 'video-face-specialists-v20', 'modality': 'video',
        'private_freeze_sha256': source.sha(trial / 'v20-freeze.json'),
        'files_sha256': {n: source.sha(ROOT / n) for n in outputs},
        'manifests': manifest_path, 'expected': expected_path,
        'manifest_sha256': {arm: engine.digest(m) for arm, m in packets.items()}, 'adverse_arm': 'all',
        'absolute_probability_limit': 1e-6, 'private_public_maximum_probability_difference': maxima,
        'serving_admitted': False, 'policy_sha256': engine.digest(lock['policy']),
        'limits': 'Consumed source groups. Scalar/head/file-probability replay only; no raw clip, crop, per-frame hash, alignment, face vector, source-authorship proof, full-app accuracy or outside human review.'})
    print(json.dumps({'arms': len(packets), 'manifest_bytes': (ROOT / manifest_path).stat().st_size,
        'expected_bytes': (ROOT / expected_path).stat().st_size,
        'private_public_maximum_probability_difference': maxima, 'serving_admitted': False}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trial', type=Path)
    args = parser.parse_args()
    release(args.trial.resolve())
