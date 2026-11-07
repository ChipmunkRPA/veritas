"""Repeat published real-media scalar calibration studies without raw media."""
from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
from pathlib import Path

from app.bench import specialists as engine

ROOT = Path(__file__).resolve().parents[2]


def reproduce(release_path, output):
    release = json.loads(release_path.read_text())
    for name, expected in release['files_sha256'].items():
        path = (ROOT / name).resolve()
        engine.check(path.is_relative_to(ROOT.resolve()) and path.is_file(), 'Reviewed repository source path required')
        engine.check(hashlib.sha256(path.read_bytes()).hexdigest() == expected, 'Reviewed source/packet bytes changed')
    packets = json.loads(gzip.decompress((ROOT / release['manifests']).read_bytes()))
    expected = json.loads(gzip.decompress((ROOT / release['expected']).read_bytes()))
    engine.check(set(packets) == set(expected) == set(release['manifest_sha256']), 'Complete frozen arms required')
    maximum, compared = 0., 0
    for arm, manifest in packets.items():
        engine.check(engine.digest(manifest) == release['manifest_sha256'][arm], 'Frozen source/observation manifest changed')
        engine.check({f['modality'] for f in manifest['files']} == {release['modality']}, 'Explicit modality scope required')
        artifact = engine.train(manifest, min_groups_per_class=25)
        result = engine.evaluate(manifest, artifact)
        for key in ('outcomes', 'unit_outcomes'):
            engine.check(len(result[key]) == len(expected[arm][key]), 'All planned outputs required')
            for actual, prior in zip(result[key], expected[arm][key], strict=True):
                engine.check({k: v for k, v in actual.items() if k != 'probability'} ==
                    {k: v for k, v in prior.items() if k != 'probability'}, 'Changed location/context/truth/status binding')
                a, b = actual['probability'], prior['probability']
                if a is None or b is None:
                    engine.check(a is b, 'Changed assessment availability')
                else:
                    maximum = max(maximum, abs(a - b))
                    compared += 1
        if arm == release['adverse_arm']:
            altered = copy.deepcopy(manifest)
            for file in altered['files']:
                if file['role'] == 'confirmation':
                    for unit in file['units']:
                        unit['raw'] = 1 - unit['raw']
            engine.check(engine.train(altered) == artifact, 'Confirmation raw responses changed fitted parameters')
            altered = copy.deepcopy(manifest)
            file = next(f for f in altered['files'] if f['role'] == 'confirmation')
            file['units'][0].update(status='skipped', raw=None, reason='Adverse required-check replay')
            row = next(r for r in engine.evaluate(altered, artifact)['outcomes'] if r['id'] == file['id'])
            engine.check(row['status'] == 'inconclusive' and row['probability'] is None, 'Missing evidence became originality confidence')
            altered = copy.deepcopy(manifest)
            first = altered['files'][0]
            other = next(f for f in altered['files'] if f['role'] != first['role'])
            other['lineage'] = copy.deepcopy(first['lineage'])
            try:
                engine.audit(altered)
            except ValueError:
                pass
            else:
                raise ValueError('Cross-role real-source lineage accepted')
    engine.check(maximum <= release['absolute_probability_limit'], 'Probability replay exceeded predeclared agreement limit')
    receipt = {'schema': 'chipmunk.specialist-real-scalar-reproduction/3', 'study': release['study'],
        'modality': release['modality'], 'arms': len(packets), 'probabilities_compared': compared,
        'max_absolute_probability_difference': maximum, 'absolute_probability_limit': release['absolute_probability_limit'],
        'confirmation_not_fitted': True, 'missing_required_check_withheld': True, 'cross_role_lineage_refused': True,
        'new_neural_forwards': 0, 'raw_media_or_weights_loaded': False, 'serving_admitted': False,
        'scope': 'Real retained-scalar numerical replay only; not native inference, fresh source quality, training-overlap exclusion or outside human review.'}
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
