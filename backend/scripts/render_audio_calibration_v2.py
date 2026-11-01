"""Render completed, verified pilot results; never score or refit a detector."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import tempfile
from collections import Counter
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'backend/data/independent-evaluation/audio-v2'
OUTPUT = ROOT / 'output/evaluation'
SYSTEMS = {'old': 'Current check, refitted here', 'new': 'Candidate check, refitted here'}
RULES = {'plain': '80% rule only', 'guarded': '80% rule + abstention guard'}
GROUPS = {'all': 'All five conditions', 'clean': 'Clean speech', 'noise_5': 'Noise · 5 dB',
          'noise_15': 'Noise · 15 dB', 'music_5': 'Music · 5 dB', 'music_15': 'Music · 15 dB',
          'seconds_4': 'Four-second sources', 'seconds_8': 'Eight-second sources'}
METRICS = {'shared_brier': 'Brier on jointly assessed inputs', 'shared_log_loss': 'Log loss on jointly assessed inputs',
           'coverage': 'Decision coverage', 'false_ai_rate': 'Real incorrectly called AI',
           'false_original_rate': 'AI incorrectly called original'}


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def subset(rows, group):
    return [r for r in rows if group == 'all' or r['condition'] == group or f"seconds_{r['seconds']}" == group]


def load_complete():
    names = ['evaluation.json', 'manifest.json', 'calibration.json', 'predictions.json', 'scores.jsonl', 'run-lock.json']
    missing = [name for name in names if not (DATA / name).is_file()]
    if missing:
        raise ValueError('Results are incomplete; no report was written. Missing: ' + ', '.join(missing))
    report, manifest, calibration, rows = [json.loads((DATA / name).read_bytes()) for name in names[:4]]
    canonical = json.dumps({k: v for k, v in manifest.items() if k != 'sha256'}, ensure_ascii=False,
                           sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    checks = [report['schema'] == 'chipmunk.audio-pilot-report/2', report['deployment_eligible'] is False,
              manifest['sha256'] == hashlib.sha256(canonical).hexdigest(),
              report['manifest_sha256'] == manifest['sha256'] == calibration['manifest_sha256'],
              report['calibration_sha256'] == digest(DATA / 'calibration.json'),
              report['scores_sha256'] == digest(DATA / 'scores.jsonl'),
              calibration['run_lock_sha256'] == digest(DATA / 'run-lock.json'),
              calibration['test_scores_accessed'] is False, report['cohort'] == manifest['audit']]
    if not all(checks):
        raise ValueError('Pilot hashes, schema or cohort do not match')
    expected = {r['id'] for r in manifest['variants'] if r['split'] == 'test'}
    if len(rows) != len(expected) or {r['id'] for r in rows} != expected:
        raise ValueError('Predictions do not cover the complete test cohort')
    for group in GROUPS:
        chosen = subset(rows, group)
        for system in SYSTEMS:
            for rule in RULES:
                m = report['groups'][group][system][rule]
                decisions = Counter(r[f'{system}_{rule}'] for r in chosen)
                real = sum(r['label'] == 0 for r in chosen)
                false_ai = sum(r['label'] == 0 and r[f'{system}_{rule}'] == 'ai' for r in chosen)
                false_original = sum(r['label'] == 1 and r[f'{system}_{rule}'] == 'original' for r in chosen)
                if (m['total'], m['real'], m['ai'], m['false_ai_count'], m['false_original_count']) != (
                        len(chosen), real, len(chosen) - real, false_ai, false_original) or m['decisions'] != dict(decisions):
                    raise ValueError(f'Inconsistent decision counts: {group}/{system}/{rule}')
                if chosen and not math.isclose(m['coverage'], (decisions['ai'] + decisions['original']) / len(chosen), abs_tol=1e-12):
                    raise ValueError('Coverage does not match its denominator')
    return report, calibration, rows


def number(value, percent=False, signed=False):
    if value is None:
        return 'Not established'
    if percent:
        return f'{100 * value:+.1f} pp' if signed else f'{value:.1%}'
    return f'{value:+.4f}' if signed else f'{value:.4f}'


def fraction(count, total):
    return f'{count} / {total} ({count / total:.1%})' if total else 'No inputs'


def table(headers, rows):
    head = ''.join(f'<th scope="col">{escape(str(v))}</th>' for v in headers)
    body = ''.join('<tr>' + ''.join(f'<td>{escape(str(v))}</td>' for v in row) + '</tr>' for row in rows)
    return f'<div class="scroll"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def difference(report, rule, metric):
    values = report['shared_assessed'] if metric.startswith('shared_') else {s: report['groups']['all'][s][rule] for s in SYSTEMS}
    key = metric.removeprefix('shared_')
    a, b = values['old'].get(key), values['new'].get(key)
    return None if a is None or b is None else b - a


def assemble(report, calibration, predictions):
    audit, sections, export = report['cohort'], [], []
    test = audit['counts']['test']
    overall = report['groups']['all']
    new, old, plain = overall['new']['guarded'], overall['old']['guarded'], overall['new']['plain']
    remaining = []
    for condition in ('clean', 'noise_5', 'noise_15', 'music_5', 'music_15'):
        result = report['groups'][condition]['new']['guarded']
        if result['false_original_count']:
            remaining.append(f'{escape(GROUPS[condition])}: {result["false_original_count"]} of '
                             f'{result["ai"]} AI inputs ({number(result["false_original_rate"], True)})')
    error_note = ('<p><strong>AI speech still incorrectly cleared as original:</strong> '
                  + '; '.join(remaining) + '. These errors remain after the guard.</p>') if remaining else ''
    sections.append('<section><h2>What the candidate actually returned</h2>'
        f'<p>With the guard, the candidate gave a leaning conclusion on {number(new["coverage"], True)} of test inputs; '
        f'the current check did so on {number(old["coverage"], True)}. The candidate incorrectly called '
        f'{new["false_ai_count"]} of {new["real"]} real inputs AI, and {new["false_original_count"]} of {new["ai"]} AI inputs original.</p>'
        f'{error_note}'
        f'<p>Before the additional guard, candidate decision coverage was {number(plain["coverage"], True)}, with '
        f'{plain["false_ai_count"]} real inputs called AI and {plain["false_original_count"]} AI inputs called original. '
        'Withholding answers can reduce incorrect conclusions. It does not prove every remaining answer is correct.</p>'
        '<p class="small">The current-check comparator is refitted for this experiment. These rows are not the website’s combined confidence or a new estimate of live customer accuracy.</p></section>')
    sections.append('<section><h2>How to read this experiment</h2><p>The <strong>80% rule</strong> leans AI at estimated AI probability ≥ 80%, '
        'leans original at ≤ 20%, and is inconclusive in the middle. The additional guard withholds a conclusion unless the separately calibrated '
        f'class set supports only that direction. Its rank threshold is {number(calibration["alpha"], True)}; rank values are not customer confidence percentages.</p>'
        '<p><strong>False AI</strong> is real speech incorrectly called AI, divided by all real inputs. <strong>False original</strong> is AI speech incorrectly called original, '
        'divided by all AI inputs. <strong>Coverage</strong> is the share of all inputs receiving a leaning conclusion. Inconclusive includes unavailable checks, also shown separately.</p>'
        '<p class="small">Five background conditions reuse each source; do not count them as independent recordings. A 5 dB background is louder relative to speech than a 15 dB background. SNR is measured over the whole window.</p></section>')
    for group, title in GROUPS.items():
        chosen = subset(predictions, group)
        decision_rows, quality_rows = [], []
        for system, name in SYSTEMS.items():
            record = report['groups'][group][system]
            assessed = sum(r[system] is not None for r in chosen)
            for rule, rule_name in RULES.items():
                m = record[rule]
                d = m['decisions']
                decided = d.get('ai', 0) + d.get('original', 0)
                decision_rows.append([name, rule_name, fraction(m['false_ai_count'], m['real']),
                    fraction(m['false_original_count'], m['ai']), fraction(decided, m['total']),
                    d.get('inconclusive', 0), len(chosen) - assessed, number(m['decided_error'], True)])
                export.append(dict(record_type='system_result', group=group, system=name, rule=rule_name,
                    group_original_files=len({r['base_id'] for r in chosen}),
                    group_real_identities=len({r['speaker'] for r in chosen if r['label'] == 0}),
                    group_ai_identities=len({r['speaker'] for r in chosen if r['label'] == 1}),
                    total_inputs=m['total'], real_inputs=m['real'], ai_inputs=m['ai'], assessed_inputs=assessed,
                    unavailable_inputs=len(chosen) - assessed, leaning_ai=d.get('ai', 0), leaning_original=d.get('original', 0),
                    inconclusive_including_unavailable=d.get('inconclusive', 0), decision_coverage=m['coverage'],
                    false_ai_count=m['false_ai_count'], false_ai_rate=m['false_ai_rate'], false_original_count=m['false_original_count'],
                    false_original_rate=m['false_original_rate'], error_among_decisions=m['decided_error'],
                    raw_auc=record['raw_auc'], balanced_brier=m['brier'], balanced_log_loss=m['log_loss'], balanced_ece=m['ece']))
            m = record['plain']
            quality_rows.append([name, fraction(assessed, len(chosen)), number(record['raw_auc']), number(m['brier']), number(m['log_loss'])])
            for direction in ('false_ai', 'false_original'):
                bound = record[direction + '_speaker_event_bound']
                export.append(dict(record_type='identity_error_bound', group=group, system=name, rule=RULES['guarded'],
                    metric=direction, events=bound['events'], identities=bound['speakers'], ci95_high=bound['one_sided_95_upper'],
                    interval_kind='one-sided exact binomial descriptive upper bound'))
        sections.append(f'<section><h2>{escape(title)}</h2><p class="small">{len(chosen)} inputs from '
            f'{len({r["base_id"] for r in chosen})} originals. Both checks see the same inputs.</p>' + table(
            ['Check', 'Decision rule', 'Real called AI', 'AI called original', 'Decision coverage', 'Inconclusive', 'Unavailable (included)', 'Error among decisions'], decision_rows) +
            '<details><summary>Ranking and probability quality</summary><p class="small">AUC measures ranking, not confidence accuracy. '
            'Lower Brier and log loss indicate smaller probability errors. These metrics use assessed inputs and a balanced 50:50 class prior; '
            'the abstention guard does not change the probabilities. The paired comparison below uses jointly assessed inputs.</p>' +
            table(['Check', 'Assessed inputs', 'Raw AUC ↑', 'Brier ↓', 'Log loss ↓'], quality_rows) + '</details></section>')
    comparison_rows = []
    for rule in RULES:
        for metric, name in METRICS.items():
            interval = report['paired_cluster_bootstrap_new_minus_old'].get(f'{rule}:{metric}', {})
            bounds = interval.get('percentile_95') or [None, None]
            point = difference(report, rule, metric)
            percent = metric in {'coverage', 'false_ai_rate', 'false_original_rate'}
            formatted = 'Not established' if bounds[0] is None else ' to '.join(number(v, percent, True) for v in bounds)
            comparison_rows.append([RULES[rule], name, number(point, percent, True), formatted, interval.get('replicates', 0)])
            export.append(dict(record_type='paired_difference', group='all', system='candidate minus current', rule=RULES[rule],
                metric=metric, difference=point, ci95_low=bounds[0], ci95_high=bounds[1], bootstrap_replicates=interval.get('replicates', 0),
                interval_kind='paired stratified identity bootstrap percentile interval'))
    bounds_rows = []
    for system, name in SYSTEMS.items():
        m = report['shared_assessed'][system]
        export.append(dict(record_type='jointly_assessed_probability_result', group='all', system=name, total_inputs=m['total'],
            real_inputs=m['real'], ai_inputs=m['ai'], balanced_brier=m['brier'], balanced_log_loss=m['log_loss'], balanced_ece=m['ece']))
        for direction, label in [('false_ai', 'Real called AI'), ('false_original', 'AI called original')]:
            bound = overall[system][direction + '_speaker_event_bound']
            bounds_rows.append([name, label, f'{bound["events"]} / {bound["speakers"]}', number(bound['one_sided_95_upper'], True)])
    sections.append('<section><h2>Paired comparison and uncertainty</h2>'
        f'<p>{report["shared_assessed"]["old"]["total"]} inputs were assessed by both checks. Brier and log-loss differences use this common subset; '
        'coverage and error differences retain every test input. Values are candidate minus current. Negative error differences favor the candidate; '
        'positive coverage means more conclusions and must be considered alongside error counts.</p>' + table(
        ['Decision rule', 'Measure', 'Observed difference', 'Paired 95% interval', 'Replicates'], comparison_rows) +
        '<p class="small">Resampling carries each speaker or voice’s files and all background variants together. Intervals condition on this fixed fit and calibration; '
        'they exclude uncertainty from estimating those parameters and do not describe every environment. Zero observed errors can produce a misleading zero-width bootstrap interval.</p>'
        '<h3>Absolute error uncertainty after the guard</h3><p class="small">One event means an identity had at least one such error across its test inputs. '
        'The one-sided 95% exact binomial bounds describe this source population; they are not per-file customer risk estimates.</p>' + table(
        ['Check', 'Error direction', 'Affected identities / identities tested', 'One-sided 95% upper bound'], bounds_rows) + '</section>')
    split_rows = [[split.capitalize(), c['0']['files'], c['0']['speakers'], c['1']['files'], c['1']['speakers']]
                  for split, c in audit['counts'].items()]
    duration = audit['duration_counts']
    sections.append('<section><h2>Scope and next decision</h2>' + table(['Split', 'Real files', 'Real speakers', 'AI files', 'AI voices'], split_rows) +
        f'<p>Across all splits, {duration.get("4", 0)} original windows last four seconds and {duration.get("8", 0)} last eight seconds. '
        'Final inputs use production 16 kHz signed-16-bit decoding, speech gates and window aggregation. Both checks run on CPU.</p>'
        '<p>Source files, speaker identities, transcript strings, source identities and PCM hashes were checked against the previous experiment and across splits. '
        'Probability fitting and guard calibration use separate splits. Their parameters were frozen before test scoring.</p><ul>' +
        ''.join('<li>' + escape(text) + '</li>' for text in report['limitations']) +
        '<li>Four- and eight-second clips do not establish long-file, telephone, multilingual, video-soundtrack or partial-edit performance.</li>'
        '<li>The pilot guard’s 10% rank level does not meet a 5% release error target. Few test identities leave substantial upper bounds even with zero observed errors.</li></ul>'
        '<p><strong>Next gate:</strong> at least 100 new test originals per class across multiple recording and generation families, acceptable error bounds, '
        'and validation on the serving device. Any tuning after this report needs another prospectively frozen test. This pilot cannot justify removing the current check.</p></section>')
    evaluation_hash = digest(DATA / 'evaluation.json')
    sections.append('<section><h2>Reproducibility</h2><p class="hash">' + '<br>'.join(escape(label + ': ' + value) for label, value in [
        ('Manifest SHA-256', report['manifest_sha256']), ('Scores SHA-256', report['scores_sha256']),
        ('Calibration SHA-256', report['calibration_sha256']), ('Evaluation SHA-256', evaluation_hash)]) +
        '</p><p class="small">The renderer checked artifact hashes, complete test membership, counts and coverage before publishing. It does not run inference or alter calibration.</p></section>')
    for row in export:
        row.update(test_original_files=test['0']['files'] + test['1']['files'], real_test_identities=test['0']['speakers'],
                   ai_test_identities=test['1']['speakers'], deployment_eligible=False, evaluation_sha256=evaluation_hash)
    return ''.join(sections), export


STYLE = '''
:root{color-scheme:light;font-family:Inter,-apple-system,BlinkMacSystemFont,sans-serif;color:#293426;background:#f4f5ee}*{box-sizing:border-box}body{margin:0}main{max-width:1400px;margin:0 auto;padding:44px 32px 72px}.brand{font-size:12px;font-weight:750;letter-spacing:.12em;color:#526145}h1{font-size:38px;line-height:1.2;max-width:960px;margin:16px 0}h2{font-size:23px;margin:0 0 16px}h3{font-size:17px}p,li{line-height:1.7}p{max-width:1050px}.muted,.small{color:#63705d}.small{font-size:13px}.notice{border-left:5px solid #9b692d;background:#fff8e8;padding:18px 22px;margin:24px 0}.notice p{margin-bottom:0}.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}.card,section{background:#fff;border:1px solid #dce2d5;border-radius:12px;padding:24px}.card b{display:block;font-size:30px}.card span{font-size:14px;color:#63705d}section{margin:24px 0}.scroll{overflow-x:auto}table{width:100%;border-collapse:collapse;font-size:13px;font-variant-numeric:tabular-nums;white-space:nowrap}th,td{text-align:left;padding:12px 10px;border-bottom:1px solid #e4e8df;vertical-align:top}th{background:#edf1e7;color:#506046}tr:last-child td{border-bottom:0}td:first-child{font-weight:600}a{color:#49613a;text-underline-offset:3px}.hash{font:11px/1.7 monospace;overflow-wrap:anywhere}details{margin-top:20px;border-top:1px solid #e4e8df;padding-top:16px}summary{cursor:pointer;font-weight:650}.links{display:flex;flex-wrap:wrap;gap:16px;font-size:14px}footer{font-size:12px;color:#63705d;margin-top:24px}@media(max-width:800px){main{padding:28px 16px}.cards{grid-template-columns:1fr}h1{font-size:29px}section{padding:18px}}@media print{main{padding:0}section{break-inside:avoid}.scroll{overflow:visible}table{white-space:normal;font-size:10px}.links{display:none}}
'''


def atomic_write(path, content, encoding='utf-8'):
    with tempfile.NamedTemporaryFile('w', encoding=encoding, newline='', dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def render():
    report, calibration, predictions = load_complete()
    body, export = assemble(report, calibration, predictions)
    audit, test = report['cohort'], report['cohort']['counts']['test']
    page = '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
    page += '<title>Chipmunk Verify · Audio calibration pilot v2</title><style>' + STYLE + '</style></head><body><main>'
    page += '<div class="brand">CHIPMUNK VERIFY · INTERNAL VALIDATION</div><h1>Audio calibration: new sources, noise and music</h1>'
    page += '<p class="muted">A frozen local comparison of the current speech check and its candidate replacement.</p>'
    page += '<div class="notice"><strong>Decision: keep the candidate out of combined customer confidence.</strong><p>This pilot cannot qualify a replacement. '
    page += 'It tests one previously observed synthesis family, a small cohort and controlled backgrounds. No serving weights or installed detectors are changed by this report.</p></div>'
    page += '<div class="cards">' + ''.join(f'<div class="card"><b>{value}</b><span>{label}</span></div>' for value, label in [
        (audit['original_files'], 'Fresh originals across fit, calibration and test'),
        (test['0']['files'] + test['1']['files'], f"Held-out originals · {test['0']['files']} real / {test['1']['files']} AI"),
        (report['groups']['all']['new']['plain']['total'], 'Held-out inputs · five correlated conditions per source')]) + '</div>'
    page += '<p class="links"><a href="audio-calibration-v2.csv">Download measured results (CSV)</a><a href="audio-comparison.html">Previous audio experiment</a><a href="web-sample-results.html">Image, video and text tests</a></p>'
    page += body + '<footer>Chipmunk Verify · Local research record · CSV rates use fractions from 0 to 1; blank cells are not applicable or not established.</footer></main></body></html>'
    columns = list(dict.fromkeys(key for row in export for key in row))
    buffer = io.StringIO(newline='')
    writer = csv.DictWriter(buffer, fieldnames=columns)
    writer.writeheader()
    writer.writerows(export)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    atomic_write(OUTPUT / 'audio-calibration-v2.csv', buffer.getvalue(), 'utf-8-sig')
    atomic_write(OUTPUT / 'audio-calibration-v2.html', page)
    print(OUTPUT / 'audio-calibration-v2.html')
    print(OUTPUT / 'audio-calibration-v2.csv')


if __name__ == '__main__':
    try:
        render()
    except (ValueError, KeyError, TypeError, OSError) as error:
        raise SystemExit(str(error)) from None
