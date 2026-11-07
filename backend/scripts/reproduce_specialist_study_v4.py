"""Scalar-only specialist refit with explicit unassessed-evidence handling.

Version 3 remains byte-pinned by the earlier image study. This version preserves
unavailable observations in adverse checks and tests withholding on an actually
assessed confirmation file, rather than an already inconclusive one.
"""
from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
from pathlib import Path

from app.bench import specialists as engine

ROOT = Path(__file__).resolve().parents[2]


def adverse(manifest, artifact, *, minimum=25):
    altered = copy.deepcopy(manifest)
    specs = {s['id']: s for s in manifest['policy']['specialists']}
    changed, preserved = 0, 0
    for file in altered['files']:
        if file['role'] == 'confirmation':
            for unit in file['units']:
                if unit['status'] == 'ok':
                    low, high = specs[unit['specialist']]['raw_range']
                    unit['raw'] = low + high - unit['raw']
                    changed += 1
                else:
                    engine.check(unit['raw'] is None, 'Unassessed observation has a fabricated score')
                    preserved += 1
    engine.check(engine.train(altered, min_groups_per_class=minimum) == artifact,
        'Confirmation response changes entered fitted parameters')
    original = engine.evaluate(manifest, artifact)
    assessed = {r['id'] for r in original['outcomes'] if r['status'] == 'ok' and r['probability'] is not None}
    withheld = False
    if assessed:
        altered = copy.deepcopy(manifest)
        file = next(f for f in altered['files'] if f['id'] in assessed)
        unit = next(u for u in file['units'] if u['status'] == 'ok')
        unit.update(status='skipped', raw=None, reason='Adverse required observation removed from an assessed file')
        row = next(r for r in engine.evaluate(altered, artifact)['outcomes'] if r['id'] == file['id'])
        engine.check(row['status'] == 'inconclusive' and row['probability'] is None,
            'Missing evidence became an originality/AI confidence')
        withheld = True
    altered = copy.deepcopy(manifest)
    first = altered['files'][0]
    other = next(f for f in altered['files'] if f['role'] != first['role'])
    other['lineage'] = copy.deepcopy(first['lineage'])
    try:
        engine.audit(altered)
    except ValueError:
        pass
    else:
        raise ValueError('Cross-role source lineage accepted')
    return {'confirmation_not_fitted': True, 'changed_assessed_confirmation_units': changed,
        'preserved_unassessed_confirmation_units': preserved,
        'missing_required_check_withheld': withheld,
        'missing_check_test_status': 'passed_on_previously_assessed_file' if withheld else 'not_testable_no_assessed_confirmation',
        'cross_role_lineage_refused': True}


def reproduce(release_path, output):
    release = json.loads(release_path.read_text())
    for name, expected_hash in release['files_sha256'].items():
        path = (ROOT / name).resolve()
        engine.check(path.is_relative_to(ROOT.resolve()) and path.is_file(), 'Reviewed repository source path required')
        engine.check(hashlib.sha256(path.read_bytes()).hexdigest() == expected_hash, 'Reviewed source/packet bytes changed')
    packets = json.loads(gzip.decompress((ROOT / release['manifests']).read_bytes()))
    expected = json.loads(gzip.decompress((ROOT / release['expected']).read_bytes()))
    engine.check(set(packets) == set(expected) == set(release['manifest_sha256']), 'Complete frozen arms required')
    maximum, compared, adverse_receipt = 0., 0, None
    for arm, manifest in packets.items():
        engine.check(engine.digest(manifest) == release['manifest_sha256'][arm], 'Frozen source/observation manifest changed')
        engine.check({f['modality'] for f in manifest['files']} == {release['modality']}, 'Explicit modality scope required')
        artifact = engine.train(manifest, min_groups_per_class=25)
        result = engine.evaluate(manifest, artifact)
        for key in ('outcomes', 'unit_outcomes'):
            engine.check(len(result[key]) == len(expected[arm][key]), 'All planned outputs required')
            for actual, prior in zip(result[key], expected[arm][key], strict=True):
                engine.check({k: v for k, v in actual.items() if k != 'probability'} ==
                    {k: v for k, v in prior.items() if k != 'probability'}, 'Changed source location/context/truth/status binding')
                a, b = actual['probability'], prior['probability']
                if a is None or b is None:
                    engine.check(a is b, 'Changed assessment availability')
                else:
                    maximum = max(maximum, abs(a - b))
                    compared += 1
        if arm == release['adverse_arm']:
            adverse_receipt = adverse(manifest, artifact)
    engine.check(adverse_receipt is not None, 'Declared adverse arm was not tested')
    engine.check(maximum <= release['absolute_probability_limit'], 'Probability replay exceeded declared agreement limit')
    receipt = {'schema': 'chipmunk.specialist-real-scalar-reproduction/4', 'study': release['study'],
        'modality': release['modality'], 'arms': len(packets), 'probabilities_compared': compared,
        'max_absolute_probability_difference': maximum, 'absolute_probability_limit': release['absolute_probability_limit'],
        **adverse_receipt, 'new_neural_forwards': 0, 'raw_media_or_weights_loaded': False, 'serving_admitted': False,
        'scope': 'Retained-scalar numerical replay only; not native inference, source truth, fresh detection quality or outside human review.'}
    with output.open('x') as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('release', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    reproduce(args.release, args.output)
