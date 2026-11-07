"""Review authored-score integration on another host; never detector accuracy."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from app.bench import specialists as engine

PUBLIC = Path(__file__).resolve().parents[2] / 'docs/evaluation-panels'
ABSOLUTE_PROBABILITY_LIMIT = 1e-6  # Fixed before the first separate-host run.


def review(manifest, artifact, actual, expected):
    engine.audit(manifest)
    engine.check(artifact['serving_admitted'] is False and actual['serving_admitted'] is False, 'Authored fixture must not be serving-admitted')
    engine.check(len(artifact['specialists']) == 16 and len(artifact['combiners']) == 4, 'Complete four-modality fixture required')
    maximum, compared = 0., 0
    for key in ('outcomes', 'unit_outcomes'):
        engine.check(len(actual[key]) == len(expected[key]), 'Incomplete authored confirmation outputs')
        for observed, recorded in zip(actual[key], expected[key], strict=True):
            engine.check({k: v for k, v in observed.items() if k != 'probability'} ==
                {k: v for k, v in recorded.items() if k != 'probability'}, 'Changed authored source/context/status/coordinate binding')
            a, b = observed['probability'], recorded['probability']
            engine.check(engine.finite(a) and engine.finite(b) and 0 <= a <= 1 and 0 <= b <= 1, 'Finite authored probabilities required')
            maximum = max(maximum, abs(a - b))
            compared += 1
    engine.check(maximum <= ABSOLUTE_PROBABILITY_LIMIT, 'Authored numerical difference exceeds the prospectively fixed tolerance')
    changed = copy.deepcopy(manifest)
    original = changed['files'][0]
    other = next(f for f in changed['files'] if f['role'] == 'fusion_fit')
    other['lineage'] = copy.deepcopy(original['lineage'])
    try:
        engine.audit(changed)
    except ValueError:
        pass
    else:
        raise ValueError('Cross-role source lineage was accepted')
    changed = copy.deepcopy(manifest)
    for file in changed['files']:
        if file['role'] == 'confirmation':
            for unit in file['units']:
                unit['raw'] *= -1
                unit['measurement_sha256'] = engine.digest({'changed_raw': unit['raw'], 'prior': unit['measurement_sha256']})
    same = engine.train(changed, min_groups_per_class=3)
    engine.check(same == artifact, 'Confirmation responses changed fitted/calibrated parameters')
    changed = copy.deepcopy(manifest)
    file = next(f for f in changed['files'] if f['role'] == 'confirmation' and f['label'] == 0)
    file['units'][0].update(status='skipped', raw=None, reason='Authored missing required check')
    row = next(r for r in engine.evaluate(changed, artifact)['outcomes'] if r['id'] == file['id'])
    engine.check(row['probability'] is None and row['status'] == 'inconclusive', 'Skipped evidence became an originality score')
    duplicate = copy.deepcopy(file)
    duplicate['units'] = copy.deepcopy(next(f for f in manifest['files'] if f['id'] == file['id'])['units'])
    before = engine._features(duplicate, artifact)
    extra = copy.deepcopy(duplicate['units'][0])
    extra['id'] = 'authored-duplicate'
    duplicate['units'].append(extra)
    engine.check(engine._features(duplicate, artifact) == before, 'Same-family duplicate amplified combined evidence')
    for modality in engine.MEDIA:
        neutral = copy.deepcopy(next(f for f in manifest['files'] if f['role'] == 'confirmation'
            and f['modality'] == modality and f['label'] == 1))
        genuine = copy.deepcopy(neutral)
        for unit_a, unit_b in zip(neutral['units'], genuine['units'], strict=True):
            if unit_a['specialist'].endswith('.second'):
                head = artifact['specialists'][unit_a['specialist'] + '|' + unit_a['context']]
                coefficient = head['coefficients'][0]
                engine.check(coefficient > 0, 'Applicable authored head required')
                unit_a['raw'] = head['center'][0] - head['scale'][0] * head['intercept'] / coefficient
                unit_b['raw'] = head['center'][0] + head['scale'][0] * (-2 - head['intercept']) / coefficient
        a, b = engine._features(neutral, artifact), engine._features(genuine, artifact)
        engine.check(a[0] > 0 and max(abs(x - y) for x, y in zip(a, b, strict=True)) <= 1e-12,
            'Unrelated genuine region reduced positive evidence elsewhere')
    changed = copy.deepcopy(manifest)
    for file in changed['files']:
        if file['role'] == 'specialist_cal':
            for unit in file['units']:
                unit['label'] = None
    unlabelled = engine.train(changed, min_groups_per_class=3)
    engine.check(all(m['status'] == 'insufficient_support' for m in unlabelled['specialists'].values()), 'File truth filled unknown local truth')
    return {'schema': 'chipmunk.specialist-authored-host-review/2', 'passed': True,
        'files': 128, 'observations': 512, 'confirmation_files': 32, 'confirmation_local_outputs': 128,
        'probabilities_compared': compared, 'max_absolute_probability_difference': maximum,
        'predeclared_absolute_limit': ABSOLUTE_PROBABILITY_LIMIT,
        'adverse_invariants_checked': ['cross_role_lineage_refused', 'confirmation_not_fitted',
            'missing_required_evidence_withheld', 'duplicate_family_not_amplified', 'unknown_unit_truth_not_copied',
            'unrelated_genuine_not_subtracted_in_all_four_media'],
        'real_media_evaluated': 0, 'new_neural_forwards': 0, 'serving_admitted': False,
        'scope': 'Authored-score four-media numerical integration only; not real-model accuracy or outside human review.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('artifact', type=Path)
    parser.add_argument('confirmation', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    manifest = json.loads((PUBLIC / 'specialist-calibration-v2-authored-manifest.json').read_text())
    expected = json.loads((PUBLIC / 'specialist-calibration-v2-authored-expected.json').read_text())
    result = review(manifest, json.loads(args.artifact.read_text()), json.loads(args.confirmation.read_text()), expected)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(result), flush=True)
