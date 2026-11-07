"""Standalone arithmetic replay of the frozen consumed-video comparison.

Uses recorded calibration coefficients, not the project fitter. Independently
reduces probability mapping, ranks, losses, grouped uncertainty, outcomes and
requirements. Native replay receipts are bound and checked, not re-executed.
No app, neural-model or original evaluator imports; NumPy is the only dependency.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

FREEZE_SHA = "fe8aa9a67bcca54259b0d2e9bf7966b9723fc73ac1ebb4d891601559e3898d61"
CHECKS = {"vid.synth.frames.commfor384", "vid.synth.frames.dda"}
DATASETS = ("deepaction", "genbuster")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def same(actual, expected):
    """Arithmetic tolerance only; every outcome and requirement remains exact."""
    if isinstance(expected, dict):
        return isinstance(actual, dict) and actual.keys() == expected.keys() and all(same(actual[k], v) for k, v in expected.items())
    if isinstance(expected, list):
        return isinstance(actual, list) and len(actual) == len(expected) and all(same(a, b) for a, b in zip(actual, expected, strict=True))
    if type(expected) is float:
        return type(actual) in (int, float) and math.isfinite(actual) and math.isclose(actual, expected, rel_tol=1e-11, abs_tol=1e-12)
    return type(actual) is type(expected) and actual == expected


def probability(value):
    return type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1


def raw_margin(row, system):
    if row['status'] != 'ok':
        return None
    if system == 'candidate':
        return row['logit']
    p = min(1 - 1e-7, max(1e-7, row['score']))
    return math.log(p / (1 - p))


def mapped(margin, coef):
    require(len(coef) == 2 and all(type(v) in (int, float) and math.isfinite(v) for v in coef) and coef[0] >= 0,
            'Complete finite nonnegative-slope calibration required')
    if margin is None:
        return None
    z = min(20, max(-20, margin)) * coef[0] + coef[1]
    return 1 / (1 + math.exp(-z)) if z >= 0 else math.exp(z) / (1 + math.exp(z))


def decision(p):
    return 'inconclusive' if p is None else 'ai' if p >= .8 else 'original' if p <= .2 + 1e-12 else 'inconclusive'


def counts(rows):
    fields = ('planned', 'assessed', 'failed', 'inconclusive', 'correct_ai', 'correct_original', 'false_ai', 'false_original', 'ai_without_ai_conclusion')
    result = dict.fromkeys(fields, 0)
    for r in rows:
        d, y = r['decision'], r['label']
        flags = (True, r['status'] == 'ok', r['status'] == 'failed', d == 'inconclusive',
                 y == 1 and d == 'ai', y == 0 and d == 'original', y == 0 and d == 'ai',
                 y == 1 and d == 'original', y == 1 and d != 'ai')
        for field, flag in zip(fields, flags, strict=True):
            result[field] += int(flag)
    return result


def losses(y, p):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return -(y * np.log(p) + (1 - y) * np.log1p(-p))


def metrics(y, p):
    positives, negatives = p[y == 1], p[y == 0]
    require(len(positives) > 0 and len(negatives) > 0, 'Both development truth classes required')
    auc = sum(float(a > b) + .5 * float(a == b) for a in positives for b in negatives) / (len(positives) * len(negatives))
    # Tie groups enter together; this is non-interpolated average precision.
    ap = sum(float(np.sum((y == 1) & (p >= value))) / int(np.sum(p >= value)) * int(np.sum(positives == value))
             for value in set(positives)) / len(positives)
    return {'n': len(y), 'real': int(sum(y == 0)), 'ai': int(sum(y == 1)),
            'auc': auc, 'average_precision': ap, 'brier': float(np.mean((y - p)**2)), 'log_loss': float(np.mean(losses(y, p)))}


def uncertainty(groups, delta):
    means = np.asarray([np.mean([d for g, d in zip(groups, delta, strict=True) if g == name]) for name in sorted(set(groups))])
    draws = np.random.default_rng(901).choice(means, size=(2000, len(means)), replace=True)
    return {'mean': float(np.mean(means)), 'group_bootstrap_95': np.quantile(np.mean(draws, axis=1), [.025, .975]).tolist()}


def identity(row):
    return row['dataset'], row['id'], row['variant']


def journal(path, body, lock_sha, system):
    with path.open() as stream:
        header = json.loads(next(stream))
        rows = [json.loads(line) for line in stream]
    require(header['lock_sha256'] == lock_sha and header['system'] == system and len(rows) == 288,
            'Complete original source-bound journal required')
    for original, row in zip(body['rows'], rows, strict=True):
        require(type(row['label']) is int and all(row[k] == v for k, v in original.items() if k != 'status'), 'Changed source, truth, format or role')
        require(row['status'] in ('ok', 'not_applicable', 'unavailable'), 'Unknown or failed assessment')
        require((original['status'] == 'unavailable') == (row['status'] == 'unavailable'), 'Changed original availability')
        require(probability(row['score']) if row['status'] == 'ok' else row['score'] is None, 'Unassessed vote or invalid probability')
        if system == 'candidate' and row['status'] == 'ok':
            require(type(row['logit']) in (int, float) and math.isfinite(row['logit']), 'Finite candidate logit required')
            z = row['logit']
            p = 1 / (1 + math.exp(-z)) if z >= 0 else math.exp(z) / (1 + math.exp(z))
            require(abs(p - row['score']) <= 1e-7, 'Candidate sigmoid correspondence differs')
            support = row['support']
            require(support['tensor_shape'] == [1, 8, 3, 224, 224] and support['synthetic_padding_frames'] == 0
                    and len(support['frames']) == 8
                    and [f['native_index'] for f in support['frames']] == support['plan']['selected_indices'],
                    'Eight original supported pictures required')
        if system == 'baseline' and original['status'] == 'ready':
            require(set(row['checks']) == CHECKS, 'Both baseline check statuses required')
            values = []
            for check in row['checks'].values():
                require(check['status'] in ('ok', 'skipped', 'not_applicable'), 'A failed check cannot become missing')
                require(probability(check['prob']) if check['status'] == 'ok' else check['prob'] is None, 'Invalid baseline check vote')
                if check['status'] == 'ok':
                    values.append(check['prob'])
            require((row['status'] == 'ok') == bool(values), 'Baseline applicability differs')
            if values:
                require(math.isclose(sum(values) / len(values), row['score'], abs_tol=1e-12), 'Baseline mean differs')
    return rows


def review(freeze, baseline, candidate, summary, native, output):
    require(sha(freeze) == FREEZE_SHA, 'Published prospective freeze changed')
    public = json.loads(freeze.read_text())
    body, lock_sha = public['lock'], public['private_lock_sha256']
    require(body['recordings_scored'] == 0 and body['calibration_fitted'] is False and body['deployment_eligible'] is False,
            'Unscored prospective freeze required')
    runs = {name: journal(path, body, lock_sha, name) for name, path in [('baseline', baseline), ('candidate', candidate)]}
    reported = json.loads(summary.read_text())
    require(reported['lock_sha256'] == lock_sha and reported['baseline_sha256'] == sha(baseline)
            and reported['candidate_sha256'] == sha(candidate), 'Original comparison lineage required')
    reduced = {s: [] for s in runs}
    panels = {}
    for dataset in DATASETS:
        cal = [i for i, r in enumerate(body['rows']) if r['dataset'] == dataset and r['split'] == 'calibration'
               and all(rows[i]['status'] == 'ok' for rows in runs.values())]
        groups = {body['rows'][i]['group'] for i in cal}
        require({body['rows'][i]['label'] for i in cal} == {0, 1} and len(groups) >= 8, 'Frozen calibration support missing')
        fitted = reported['fitted_calibration'][dataset]
        require(fitted['common_rows'] == len(cal) and fitted['groups'] == len(groups), 'Calibration rows/groups differ')
        for system, rows in runs.items():
            for row in rows:
                if row['dataset'] != dataset or row['split'] != 'test':
                    continue
                p = mapped(raw_margin(row, system), fitted[system])
                reduced[system].append({**row, 'p': p, 'decision': decision(p)})
        for variant in ('original', 'matched'):
            source = {s: [r for r in rows if r['dataset'] == dataset and r['variant'] == variant] for s, rows in reduced.items()}
            require([identity(r) for r in source['baseline']] == [identity(r) for r in source['candidate']], 'Changed paired ordering')
            paired = [(a, b) for a, b in zip(source['baseline'], source['candidate'], strict=True) if a['status'] == b['status'] == 'ok']
            y = np.asarray([a['label'] for a, _ in paired])
            a, b = [np.asarray([r[i]['p'] for r in paired]) for i in (0, 1)]
            names = [r['group'] for r, _ in paired]
            panels[dataset + ':' + variant] = {'paired': len(paired), 'source_groups': len(set(names)),
                'baseline_metrics': metrics(y, a), 'candidate_metrics': metrics(y, b),
                'paired_brier_difference': uncertainty(names, (y - b)**2 - (y - a)**2),
                'paired_logloss_difference': uncertainty(names, losses(y, b) - losses(y, a)),
                **{s: counts(rows) for s, rows in source.items()}}
    totals = {s: counts(rows) for s, rows in reduced.items()}
    distinct = {s: len({(r['dataset'], r['id']) for r in rows if r['label'] == 1 and r['decision'] == 'ai'}) for s, rows in reduced.items()}
    gates = {
        'complete_matrix': all(len(rows) == 288 for rows in runs.values()),
        'no_failures': all(r['status'] != 'failed' for rows in runs.values() for r in rows),
        'calibration_support': True,
        'brier_gain': all(p['paired_brier_difference']['mean'] <= -.005 for p in panels.values()),
        'brier_interval': all(p['paired_brier_difference']['group_bootstrap_95'][1] < 0 for p in panels.values()),
        'wrong_directions': all(p['candidate'][name] <= p['baseline'][name] + 1 for p in [totals, *panels.values()] for name in ('false_ai', 'false_original')),
        'ai_usefulness': distinct['candidate'] >= distinct['baseline'] + 4 and all(p['candidate']['correct_ai'] >= p['baseline']['correct_ai'] - 2 for p in panels.values()),
        'original_retention': all(p['candidate']['correct_original'] >= p['baseline']['correct_original'] - 2 for p in [totals, *panels.values()]),
        'total_correct': sum(totals['candidate'][n] for n in ('correct_ai', 'correct_original')) > sum(totals['baseline'][n] for n in ('correct_ai', 'correct_original')),
        'scope_retention': all(a['status'] != 'ok' or b['status'] == 'ok' for a, b in zip(runs['baseline'], runs['candidate'], strict=True)),
    }
    for field, value in [('calibrated_development', reduced), ('comparisons', panels), ('totals', totals),
                         ('distinct_correct_ai_sources', distinct), ('gates_before_native_audit', gates)]:
        require(same(reported[field], value), 'Independent arithmetic differs: ' + field)
    require(reported['native_audit_pending'] is True and reported['advance_to_fresh_confirmation'] is False
            and reported['deployment_eligible'] is False, 'Pre-native scope changed')
    audited = json.loads(native.read_text())
    require(audited['lock_sha256'] == lock_sha and audited['summary_sha256'] == sha(summary)
            and [r['identity'] for r in audited['cases']] == body['native_cases'], 'Original complete native receipt required')
    require(all(type(r['passed']) is bool for r in audited['cases']), 'Native pass status must be explicit')
    gates['native_correspondence'] = len(audited['cases']) == 16 and all(r['passed'] for r in audited['cases'])
    require(audited['gates'] == gates and audited['advance_to_fresh_confirmation'] is all(gates.values())
            and audited['deployment_eligible'] is False, 'Native gates/scope differ')
    receipt = {'schema': 'chipmunk.video-demamba-independent-arithmetic/13', 'passed': True,
        'public_freeze_sha256': sha(freeze), 'baseline_sha256': sha(baseline), 'candidate_sha256': sha(candidate),
        'summary_sha256': sha(summary), 'native_sha256': sha(native), 'reviewer_sha256': sha(Path(__file__)),
        'planned_inputs_per_method': 288, 'development_outcomes_per_method': len(reduced['baseline']),
        'totals': totals, 'gates': gates, 'advance_to_fresh_confirmation': all(gates.values()), 'serving_admitted': False,
        'outside_human_reviewers': 0,
        'scope': 'Separate NumPy-only reduction of stored probabilities/calibration coefficients, ranks, losses, grouped bootstrap, all-source outcomes and requirements. No calibration refit, raw-media truth audit, native/model forward or outside human review. Arithmetic tolerance1e-11relative/1e-12absolute; directions and gates exact. Shared NumPy RNG/quantile, no original evaluator imports.'}
    with output.open('x') as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'passed': True, 'outcomes_per_method': len(reduced['baseline']), 'advance': all(gates.values())}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('freeze', 'baseline', 'candidate', 'summary', 'native', 'output'):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    review(args.freeze, args.baseline, args.candidate, args.summary, args.native, args.output)
