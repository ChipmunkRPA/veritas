"""Frozen, consumed-source operating comparison; no models or serving changes.

Prior image/text diagnostic results remain immutable. Each arm gets the same
genuine-source-tail calibration procedure, then all reserved development cases
are evaluated. This is new post-hoc development, not fresh confirmation.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.metadata
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from app.bench import specialists as engine
from scripts import reproduce_writing_specialists_v5 as writing

ROOT = Path(__file__).resolve().parents[2]
STUDIES = {'image-specialists-v10': 'global-image', 'writing-specialists-v5': 'baseline-global'}
METHOD = {'schema': 'chipmunk.specialist-source-tail-method/1', 'studies': STUDIES,
    'alpha': .1, 'minimum_assessed_genuine_groups': 25, 'specialist_minimum_groups_per_class': 25,
    'point': 'Only fusion_cal genuine files. One maximum among assessed files per connected source group; missing files/groups explicitly counted, never imputed. Strict greater than upper nearest rank ceil((n+1)*.9), capped at n.',
    'fit': 'Reproduce each prior frozen arm with unchanged source roles, default25-group/class monotone calibration, L2=.001 and50:50 reference prior. Fit every arm and point before reserved-role evaluation.',
    'comparison': 'Same calibration false-alarm target, not guaranteed or forced equal actual false-alarm rates. No point selected from reserved outcomes or ROC curves.',
    'gates': ['more_ai_files_found', 'no_additional_false_ai_files', 'no_additional_false_ai_groups',
        'no_ai_source_group_detection_loss', 'no_coverage_loss', 'lower_source_class_balanced_brier',
        'no_condition_false_alarm_increase', 'no_condition_ai_detection_loss', 'no_condition_coverage_loss'],
    'serving_admitted': False,
    'limits': 'All underlying source panels and their diagnostic outcomes were already consumed. This post-hoc development comparison does not create fresh data, validate customer probabilities or replace a direct comparison with the complete serving application. Raw media/weights are not loaded.'}


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def runtime():
    return {name: importlib.metadata.version(name) for name in ('numpy', 'scipy')}


def code():
    names = ('backend/scripts/specialist_source_tail_v1.py', 'backend/app/bench/specialists.py',
        'backend/scripts/reproduce_writing_specialists_v5.py', 'backend/tests/test_specialist_source_tail_v1.py',
        'backend/scripts/reproduce_specialist_source_tail_v1.py',
        '.github/workflows/specialist-source-tail-v1-review.yml', 'docs/specialist-source-tail-v1-protocol.md')
    return {name: sha(ROOT / name) for name in names}


def source_packets():
    packets, releases = {}, {}
    for study in STUDIES:
        path = ROOT / 'docs/evaluation-panels' / (study + '-release.json')
        release = json.loads(path.read_text())
        for name, expected in release['files_sha256'].items():
            source = (ROOT / name).resolve()
            engine.check(source.is_relative_to(ROOT.resolve()) and sha(source) == expected, 'Prior pinned source/packet changed')
        if study.startswith('image'):
            values = json.loads(gzip.decompress((ROOT / release['manifests']).read_bytes()))
        else:
            source = json.loads(gzip.decompress((ROOT / 'docs/evaluation-panels/writing-specialists-v5-scalars.json.gz').read_bytes()))
            values = {arm: writing.derive(source, arm) for arm in writing.ARMS}
        engine.check(set(values) == set(release['manifest_sha256']), 'Every prior frozen arm required')
        for arm, manifest in values.items():
            engine.check(engine.digest(manifest) == release['manifest_sha256'][arm], 'Prior source/role/measurement bindings changed')
            engine.audit(manifest)
        packets[study], releases[study] = values, {'path': str(path.relative_to(ROOT)), 'sha256': sha(path),
            'manifest_sha256': release['manifest_sha256']}
    return packets, releases


def predict(file, artifact):
    values, combined = engine._features(file, artifact), artifact['combiners'].get(engine.pattern(file))
    if (values is None or not combined or combined['response']['status'] != 'fitted'
            or combined.get('calibration', {}).get('status') != 'fitted'):
        return None
    eta = engine._logit(combined['calibration'], [engine._logit(combined['response'], values)])
    return 1 / (1 + math.exp(-max(-700., min(700., eta))))


def point(rows):
    """Metadata plus independently predicted calibration rows, no reserved use."""
    selected = [row for row in rows if row['role'] == 'fusion_cal' and row['label'] == 0]
    maxima, missing = {}, 0
    for row in selected:
        value = row['probability']
        if value is None:
            missing += 1
        else:
            engine.check(engine.finite(value) and 0 <= value <= 1, 'Invalid genuine calibration response')
            maxima[row['group']] = max(maxima.get(row['group'], value), value)
    total = len({r['group'] for r in selected})
    result = {'status': 'insufficient_support', 'threshold': None,
        'genuine_groups_total': total, 'genuine_groups_with_assessed_files': len(maxima),
        'genuine_groups_with_no_assessed_files': total - len(maxima), 'unassessed_genuine_files': missing,
        'scope': METHOD['point'], 'serving_admitted': False}
    if len(maxima) >= METHOD['minimum_assessed_genuine_groups']:
        values = sorted(maxima.values())
        rank = min(len(values), math.ceil((len(values) + 1) * (1 - METHOD['alpha'])))
        result.update(status='fitted', threshold=values[rank - 1], upper_rank=rank,
            calibration_genuine_groups_flagged=sum(value > values[rank - 1] for value in values))
    return result


def condition(file):
    if file['modality'] == 'image':
        result = file['id'].rsplit('-', 1)[-1]
        engine.check(result in ('native', 'jpeg75'), 'Declared image format required')
    else:
        prefix = file['group'] + '-'
        engine.check(file['id'].startswith(prefix), 'Text source/pattern identity required')
        result = file['id'][len(prefix):]
        engine.check(result in ('human', 'ai', 'human-ai', 'ai-human'), 'Declared text mixture pattern required')
    return result


def summarize(rows, operating):
    groups = {0: set(), 1: set()}
    loss, flagged = {0: defaultdict(list), 1: defaultdict(list)}, {0: set(), 1: set()}
    count = Counter({'planned': len(rows), 'genuine': 0, 'ai': 0, 'assessed': 0, 'inconclusive': 0,
        'false_ai_files': 0, 'ai_files_found': 0, 'ai_not_flagged_including_inconclusive': 0,
        'exact_operating_point_ties': 0})
    for row in rows:
        label, probability = row['label'], row['probability']
        groups[label].add(row['group'])
        count['ai' if label else 'genuine'] += 1
        usable = probability is not None
        count['assessed' if usable else 'inconclusive'] += 1
        flag = usable and operating['status'] == 'fitted' and probability > operating['threshold']
        if usable:
            loss[label][row['group']].append((probability - label)**2)
            count['exact_operating_point_ties'] += operating['status'] == 'fitted' and probability == operating['threshold']
        if flag:
            flagged[label].add(row['group'])
        count['ai_files_found' if flag else 'ai_not_flagged_including_inconclusive'] += int(label == 1)
        count['false_ai_files'] += int(label == 0 and flag)
    by_class = [sum(sum(v) / len(v) for v in entries.values()) / len(entries) if entries else None for entries in loss.values()]
    return {**count, 'genuine_groups': len(groups[0]), 'ai_groups': len(groups[1]),
        'false_ai_groups': len(flagged[0]), 'ai_groups_found': len(flagged[1]),
        'ai_groups_not_found': len(groups[1] - flagged[1]),
        'source_class_balanced_brier_on_assessed_files': sum(by_class) / 2 if all(value is not None for value in by_class) else None}


def gates(actual, reference):
    a, b = actual['all'], reference['all']
    conditions, old = actual['conditions'], reference['conditions']
    engine.check(set(conditions) == set(old), 'Same condition denominators required')
    return {'more_ai_files_found': a['ai_files_found'] > b['ai_files_found'],
        'no_additional_false_ai_files': a['false_ai_files'] <= b['false_ai_files'],
        'no_additional_false_ai_groups': a['false_ai_groups'] <= b['false_ai_groups'],
        'no_ai_source_group_detection_loss': a['ai_groups_found'] >= b['ai_groups_found'],
        'no_coverage_loss': a['assessed'] >= b['assessed'],
        'lower_source_class_balanced_brier': a['source_class_balanced_brier_on_assessed_files'] is not None
            and b['source_class_balanced_brier_on_assessed_files'] is not None
            and a['source_class_balanced_brier_on_assessed_files'] < b['source_class_balanced_brier_on_assessed_files'],
        'no_condition_false_alarm_increase': all(conditions[key]['false_ai_files'] <= old[key]['false_ai_files']
            and conditions[key]['false_ai_groups'] <= old[key]['false_ai_groups'] for key in conditions),
        'no_condition_ai_detection_loss': all(conditions[key]['ai_files_found'] >= old[key]['ai_files_found'] for key in conditions),
        'no_condition_coverage_loss': all(conditions[key]['assessed'] >= old[key]['assessed'] for key in conditions)}


def freeze(trial):
    packets, releases = source_packets()
    trial.mkdir(exist_ok=False)
    write(trial / 'freeze.json', {'schema': 'chipmunk.specialist-source-tail-freeze/1', 'method': METHOD,
        'code_sha256': code(), 'runtime': runtime(), 'prior_releases': releases,
        'manifest_sha256': {study: {arm: engine.digest(value) for arm, value in arms.items()} for study, arms in packets.items()},
        'new_neural_forwards': 0, 'new_calibration_fits': 0, 'new_reserved_role_evaluations': 0, 'serving_admitted': False})
    print(json.dumps({'freeze_sha256': sha(trial / 'freeze.json'), 'studies': len(packets), 'arms': sum(map(len, packets.values())), 'new_neural_forwards': 0}))


def check(trial):
    frozen = json.loads((trial / 'freeze.json').read_text())
    packets, releases = source_packets()
    engine.check(frozen['method'] == METHOD and frozen['code_sha256'] == code()
        and frozen['runtime'] == runtime() and frozen['prior_releases'] == releases, 'Frozen source-tail protocol/source/runtime changed')
    return packets


def fit(trial):
    packets, artifacts, points = check(trial), {}, {}
    for study, arms in packets.items():
        artifacts[study], points[study] = {}, {}
        for arm, manifest in arms.items():
            artifact = engine.train(manifest, min_groups_per_class=25, l2=.001)
            selected = [{**{key: file[key] for key in ('role', 'label', 'group')}, 'probability': predict(file, artifact)}
                for file in manifest['files'] if file['role'] == 'fusion_cal' and file['label'] == 0]
            operating = point(selected)
            name = study + '-' + arm
            write(trial / (name + '-artifact.json'), artifact)
            write(trial / (name + '-operating-point.json'), operating)
            artifacts[study][arm] = sha(trial / (name + '-artifact.json'))
            points[study][arm] = sha(trial / (name + '-operating-point.json'))
    write(trial / 'fit-complete.json', {'schema': 'chipmunk.specialist-source-tail-fit/1',
        'freeze_sha256': sha(trial / 'freeze.json'), 'artifacts': artifacts, 'operating_points': points,
        'new_reserved_role_evaluations': 0, 'serving_admitted': False})
    print(json.dumps({'all_eight_artifacts_and_points_frozen': True, 'fit_complete_sha256': sha(trial / 'fit-complete.json')}))


def review(trial):
    packets = check(trial)
    complete = json.loads((trial / 'fit-complete.json').read_text())
    engine.check(complete['freeze_sha256'] == sha(trial / 'freeze.json'), 'Fit is not bound to this frozen comparison')
    report = {'schema': 'chipmunk.specialist-source-tail-result/1', 'method': METHOD,
        'freeze_sha256': sha(trial / 'freeze.json'), 'fit_complete_sha256': sha(trial / 'fit-complete.json'),
        'studies': {}, 'new_neural_forwards': 0, 'serving_admitted': False}
    for study, arms in packets.items():
        observed = {}
        expected = json.loads(gzip.decompress((ROOT / 'docs/evaluation-panels' / (study + '-expected.json.gz')).read_bytes()))
        for arm, manifest in arms.items():
            name = study + '-' + arm
            for suffix, key in (('artifact', 'artifacts'), ('operating-point', 'operating_points')):
                engine.check(sha(trial / (name + '-' + suffix + '.json')) == complete[key][study][arm], 'Frozen artifact/point changed')
            artifact = json.loads((trial / (name + '-artifact.json')).read_text())
            operating = json.loads((trial / (name + '-operating-point.json')).read_text())
            result = engine.evaluate(manifest, artifact)
            difference, compared = writing.compare(result, expected[arm])
            metadata = {file['id']: file for file in manifest['files']}
            conditions = sorted({condition(metadata[row['id']]) for row in result['outcomes']})
            observed[arm] = {'all': summarize(result['outcomes'], operating), 'operating_point': operating,
                'prior_probability_replay': {'maximum_absolute_difference': difference, 'probabilities_compared': compared},
                'conditions': {value: summarize([r for r in result['outcomes'] if condition(metadata[r['id']]) == value], operating)
                    for value in conditions}}
        for arm, value in observed.items():
            values = gates(value, observed[STUDIES[study]])
            value.update(continuation_gates=values, continuation_eligible=arm != STUDIES[study] and all(values.values()))
        report['studies'][study] = observed
    write(trial / 'result.json', report)
    print(json.dumps({'schema': report['schema'], 'studies': len(report['studies']),
        'continuation_eligible': {study: [arm for arm, value in arms.items() if value['continuation_eligible']]
            for study, arms in report['studies'].items()}, 'new_neural_forwards': 0, 'serving_admitted': False}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('freeze', 'fit', 'review'))
    parser.add_argument('trial', type=Path)
    args = parser.parse_args()
    {'freeze': freeze, 'fit': fit, 'review': review}[args.stage](args.trial)
