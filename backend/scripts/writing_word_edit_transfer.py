"""Test changed-word supervision against native writing-window localization."""

from __future__ import annotations

import argparse
import importlib.metadata
import math
import os
import re
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from statistics import median

from scripts import writing_interval_transfer as interval
from scripts import writing_revision_trajectory_transfer as source
from scripts import writing_temperature_transfer as short
from scripts.writing_focus_v3 import BASELINE_ASSETS, BASELINE_REVISION

SCHEMA = 'chipmunk.writing-word-edit-transfer/1'
WORDS = re.compile(r"[^\W_]+(?:[’'-][^\W_]+)*")
FAMILIES = tuple(interval.FAMILIES)
MODELS = ('context', 'word')
GROUPS = 20
WORD_MIN = 4
FEATURES = ('context_margin', 'sentence_margin', 'generated_margin', 'edited_margin', 'humanized_margin',
            'generated_minus_sentence', 'edited_minus_sentence', 'humanized_minus_sentence')
METHOD = {
    'fit': 'Retained first-stage report pairs. Sequence alignment labels only inserted or substituted destination words; '
           'all text outside the focus and all original words are ordinary. Unchanged words inside edited sentences are unlabeled.',
    'inference': 'Use complete three-sentence native context and read the contextual token head on each exact word. '
                 'Fit source-and-class-balanced ridge logistic heads with context only and with contextual word features. '
                 'Rank each sentence by the mean of its highest quarter of word responses.',
    'cutoff': 'Strictly exceed the second-highest source-group maximum on 42 dated genuine journal versions. '
              'A separate incumbent is the first six native context windows with response at least .5.',
    'held': 'Twenty fresh same-publisher report works, two generator edits each. Prefer 5 or 6 sentences, keep exact flanks. '
            'Exclude previous report groups, source 12-word overlap, and Qwen overlap before any new response. Also include '
            'retained genuine forum and USWDS revisions and previously studied eLife ordinary and generated controls.',
    'useful': 'One highlight must recover at least half of edited nonspace characters with at least .30 IoU.',
    'advance': 'At least four additional useful fresh edits across both generators, at least two more than the same-fit '
               'context arm, at most one newly flagged ordinary source in each held cohort, and at least 90% edited coverage.',
}


def held_reports(cache, previous):
    source.require(all(source.digest(cache / name) == sha for name, sha in source.METADATA.items()), 'Publisher metadata differs')
    source.require('released under Apache-2.0' in (cache / 'README.md').read_text(), 'Publisher terms differ')
    files = {f: interval.read_shard(cache, f, sha) for f, sha in (interval.FAMILIES | interval.EXCLUDED_SHARD).items()}
    gem, nano, qwen = files.values()
    excluded = {row['group'] for row in previous if row['cohort'] == 'report'}
    for ledger in short.known_sources():
        excluded.update(r['group'] for r in source.read(ledger)['rows'])
    historical = set()
    for group in gem.keys() | nano.keys() | qwen.keys():
        if group in qwen or source.sha(group) in excluded:
            for file in files.values():
                row = file.get(group, {}).get('v0')
                if row:
                    historical.update(interval.fingerprints(row['text']))
    order = sorted((k for k in (gem.keys() & nano.keys()) - qwen.keys() if source.sha(k) not in excluded),
                   key=lambda k: source.sha('chipmunk-changed-word-holdout/2026-11-13/' + k))
    rows, intake = [], Counter(eligible=len(order))
    for work in order:
        if intake['admitted'] == GROUPS:
            break
        intake['examined'] += 1
        group = source.sha(work)
        if gem[work]['v0']['text'] != nano[work]['v0']['text']:
            intake['different_original'] += 1
            continue
        try:
            base, gold, units = short.annotated(gem[work]['v0'])
            source.require(not gold, 'Original labels differ')
            edited = {f: short.annotated(files[f][work]['v1']) for f in FAMILIES}
            source.require(all(files[f][work]['v1']['edit_operation'] == 'polish' for f in FAMILIES), 'Unexpected publisher operation')
        except ValueError:
            intake['publisher_geometry'] += 1
            continue
        phrases = interval.fingerprints(base)
        if historical & phrases:
            intake['prior_or_duplicate_source_phrase'] += 1
            continue
        preferred = (2, 3) if int(group[0], 16) % 2 else (3, 2)
        choice = None
        for left, right in (preferred, (2, 2)):
            matches = {f: short.align(r, base, units, (left, right)) for f, r in edited.items()}
            common = set.intersection(*(set(m) for m in matches.values()))
            if common:
                target = min(common, key=lambda i: source.sha(f'{group}/{i}'))
                choice = left, right, target, matches
                break
        if choice is None:
            intake['no_shared_isolated_edit'] += 1
            continue
        left, right, target, matches = choice
        start, end = units[target-left].start, units[target+right].end
        specs = [('original', base[start:end], None, [start, end], gem[work]['v0'])]
        specs.extend((f, *matches[f][target], files[f][work]['v1']) for f in FAMILIES)
        try:
            group_rows = [short.record(text, 'report', group, family, truth, source_range=bounds,
                                      publisher_record=publisher['record_id'], target_index=left)
                          for family, text, truth, bounds, publisher in specs]
            source.require(all(len(r['spans']) == left + right + 1 and
                               (r['truth'] is None or r['truth'] == r['spans'][left]) for r in group_rows),
                           'Clipped geometry differs')
        except ValueError:
            intake['clipped_geometry'] += 1
            continue
        historical.update(phrases)
        rows.extend(group_rows)
        intake['admitted'] += 1
    source.require(intake['admitted'] == GROUPS, 'Incomplete fresh source intake')
    return rows, dict(intake)


def model_assets(snapshot):
    source.require(snapshot.name == BASELINE_REVISION and all(source.digest(snapshot / key) == sha for key, sha in BASELINE_ASSETS.items()),
                   'Unexpected pinned primary checkpoint')


def freeze(run, cache, short_roster, elife_roster, snapshot):
    model_assets(snapshot)
    previous = source.read(short_roster)
    fit = [dict(r, cohort='fit') for r in previous if r['cohort'] == 'report']
    calibration = [dict(r, cohort='calibration') for r in previous if r['cohort'] == 'development']
    held, intake = held_reports(cache, previous)
    held.extend(r for r in previous if r['cohort'] in {'forum', 'uswds'})
    for row in source.read(elife_roster):
        if row['cohort'] == 'elife':
            held.append(short.record(row['text'], 'elife', row['group'], row['family'], row['truth']))
    rows = [*fit, *calibration, *held]
    counts = Counter(r['cohort'] for r in rows)
    source.require(counts == {'fit': 72, 'calibration': 42, 'report': 60, 'forum': 16, 'uswds': 12, 'elife': 36}
                   and len({r['id'] for r in rows}) == len(rows), 'Unexpected complete source roles')
    partitions = [{r['group'] for r in part} for part in (fit, calibration, held)]
    source.require(not any(a & b for a, b in ((partitions[0], partitions[1]), (partitions[0], partitions[2]),
                                               (partitions[1], partitions[2]))), 'Source groups cross roles')
    source.seal(run / 'roster.json', rows)
    lock = {'schema': SCHEMA, 'method': METHOD, 'runner_sha256': source.digest(Path(__file__)),
            'roster_sha256': source.digest(run / 'roster.json'), 'checkpoint_revision': BASELINE_REVISION,
            'checkpoint_files': BASELINE_ASSETS, 'publisher_revision': source.REVISION, 'publisher_shards': interval.FAMILIES,
            'short_roster_sha256': source.digest(short_roster), 'elife_roster_sha256': source.digest(elife_roster),
            'counts': dict(counts), 'intake': intake, 'features': list(FEATURES),
            'runtime': {name: importlib.metadata.version(name) for name in ('torch', 'transformers', 'pysbd', 'numpy')}}
    source.seal(run / 'freeze.json', lock)
    print('frozen', lock['counts'], lock['intake'], flush=True)


def load(run):
    lock = source.read(run / 'freeze.json')
    source.require(lock['runner_sha256'] == source.digest(Path(__file__)) and
                   lock['roster_sha256'] == source.digest(run / 'roster.json'), 'Frozen runner or sources differ')
    return lock, source.read(run / 'roster.json')


def word_rows(clf, contents):
    import torch

    for start in range(0, len(contents), 6):
        batch = contents[start:start+6]
        encoded = clf.tok(batch, padding=True, truncation=False, return_offsets_mapping=True,
                          return_special_tokens_mask=True, return_tensors='pt')
        source.require(encoded['input_ids'].shape[1] <= 512, 'Incomplete native context')
        inputs, attention = (encoded[k] for k in ('input_ids', 'attention_mask'))
        mask = attention.bool() & ~encoded['special_tokens_mask'].bool()
        with torch.inference_mode():
            hidden = clf.model.model(inputs, attention_mask=attention)[0]
            token = clf.model.classifier(hidden).float()
            native = clf.model.classifier((hidden * attention.unsqueeze(-1)).sum(1) / attention.sum(1, keepdim=True)).float()
            reduced = (token * attention.unsqueeze(-1)).sum(1) / attention.sum(1, keepdim=True)
            source.require(float((native - reduced).abs().max()) < 3e-5 and torch.isfinite(token).all(),
                           'Native logit reduction differs')
            margin = torch.logsumexp(native[:, 1:], dim=-1) - native[:, 0]
            responses = (1 - torch.softmax(native, dim=-1)[:, 0]).tolist()
            channels = token[:, :, 1:] - token[:, :, :1]
        for index, text in enumerate(batch):
            offsets = encoded['offset_mapping'][index][mask[index]].tolist()
            yield text, (float(margin[index]), responses[index], list(zip(offsets, channels[index][mask[index]].tolist(), strict=True)))


def context_words(row, readings):
    from math import fsum

    output = []
    text, spans = row['text'], row['spans']
    for index, (a, b) in enumerate(spans):
        context = row['context'][index]
        start = 0 if index == 0 else spans[index-1][1] - spans[index-1][0] + 1
        source.require(context[start:start+b-a] == text[a:b], 'Native context does not bind its focus')
        margin, response, offsets = readings[context]
        words = []
        for word in WORDS.finditer(text[a:b]):
            x, y = word.span()
            overlaps = [(max(0, min(end, start+y) - max(first, start+x)), values) for (first, end), values in offsets]
            valid = [(length, values) for length, values in overlaps if length]
            total = sum(length for length, _ in valid)
            coverage = {p for (first, end), _ in offsets for p in range(max(first, start+x), min(end, start+y))}
            if len(coverage) != y-x:
                continue
            values = [fsum(length * vector[c] for length, vector in valid) / total for c in range(3)]
            words.append((a+x, a+y, word.group().casefold(), values))
        if len(words) < WORD_MIN:
            output.append((response, []))
            continue
        centers = [median(w[3][c] for w in words) for c in range(3)]
        sentence = median(max(w[3]) for w in words)
        output.append((response, [(x, y, word, [margin, sentence, *vec, *(v-c for v, c in zip(vec, centers, strict=True))])
                                  for x, y, word, vec in words]))
    return output


def score(run, snapshot, cohort):
    import torch

    from app.detectors.text.classifier import FourClassTextClassifier

    lock, rows = load(run)
    points_sha256 = None
    if cohort == 'held':
        points_file = run / 'points.json'
        source.require(source.read(points_file)['freeze_sha256'] == source.digest(run / 'freeze.json'),
                       'Lock development points before scoring held sources')
        points_sha256 = source.digest(points_file)
    source.require(all(os.environ.get(name) == '1' for name in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE')) and
                   all(importlib.metadata.version(k) == v for k, v in lock['runtime'].items()), 'Use the locked offline runtime')
    model_assets(snapshot)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    classifier = FourClassTextClassifier.from_snapshot(snapshot, 'cpu')
    rows = [r for r in rows if (r['cohort'] in {'fit', 'calibration'}) == (cohort == 'development')]
    texts = sorted({text for row in rows for text in row['context']}, key=source.sha)
    values = {}
    for start in range(0, len(texts), 60):
        values.update(word_rows(classifier, texts[start:start+60]))
        print(cohort, len(values), len(texts), flush=True)
    audit = classifier.predict(texts[::29], batch=6, check=lambda _: None)
    source.require(max(abs(values[t][1] - p) for t, p in zip(texts[::29], audit, strict=True)) < 3e-5,
                   'Independent native primary scores differ')
    source.seal(run / f'{cohort}.json', {'freeze_sha256': source.digest(run / 'freeze.json'), 'points_sha256': points_sha256,
                'inputs': len(texts),
                'rows': {r['id']: context_words(r, values) for r in rows}})


def fit_heads(rows, readings):
    import numpy as np

    originals = {(r['group'], r['target_index']): r for r in rows if r['family'] == 'original'}
    by_group = defaultdict(lambda: defaultdict(list))
    for row in rows:
        truth = row['truth']
        sentences = readings[row['id']]
        if truth is None:
            changed = set()
        else:
            original = originals[row['group'], row['target_index']]
            a, b = original['spans'][row['target_index']]
            before = [w.group().casefold() for w in WORDS.finditer(original['text'][a:b])]
            after = list(WORDS.finditer(row['text'][truth[0]:truth[1]]))
            blocks = SequenceMatcher(None, before, [w.group().casefold() for w in after], autojunk=False).get_opcodes()
            changed = {(truth[0] + after[i].start(), truth[0] + after[i].end()) for tag, _, _, first, last in blocks
                       if tag in {'insert', 'replace'} for i in range(first, last)}
        for _, words in sentences:
            for x, y, _, values in words:
                if truth is None or not (truth[0] <= x < y <= truth[1]):
                    by_group[row['group']][0].append(values)
                elif (x, y) in changed:
                    by_group[row['group']][1].append(values)
    source.require(len(by_group) == 24 and all(set(classes) == {0, 1} for classes in by_group.values()),
                   'Training group lacks changed or ordinary words')
    samples = [(values, label, 1 / (2 * len(by_group) * len(items))) for group in by_group.values()
               for label, items in group.items() for values in items]
    features = np.array([v for v, _, _ in samples], dtype=np.float64)
    y = np.array([label for _, label, _ in samples], dtype=np.float64)
    weights = np.array([w for _, _, w in samples], dtype=np.float64)
    models = {}
    for name, count in (('context', 2), ('word', len(FEATURES))):
        raw = features[:, :count]
        mean = weights @ raw
        scale = np.maximum(.05, np.sqrt(weights @ (raw - mean)**2))
        design = np.column_stack((np.ones(len(raw)), (raw - mean) / scale))
        coefs = np.zeros(count + 1)
        for _ in range(180):
            prediction = 1 / (1 + np.exp(-np.clip(design @ coefs, -40., 40.)))
            regularization = .10 * coefs
            regularization[0] = 0
            coefs -= .25 * (design.T @ (weights * (prediction - y)) + regularization)
        models[name] = {'means': mean.tolist(), 'scales': scale.tolist(), 'coefs': coefs.tolist()}
    return models, dict(Counter(map(int, y)))


def measure(sentences, models):
    from math import fsum

    scores = {name: [] for name in MODELS}
    for _, words in sentences:
        for name, params in models.items():
            means, scales, coefs = (params[k] for k in ('means', 'scales', 'coefs'))
            values = [coefs[0] + fsum(c * (v - m) / s for c, v, m, s in
                                    zip(coefs[1:], x[:len(means)], means, scales, strict=True)) for _, _, _, x in words]
            size = (len(values) + 3) // 4
            scores[name].append(round(fsum(sorted(values, reverse=True)[:size]) / size, 7) if size else None)
    return scores


def cutoff(run):
    _, rows = load(run)
    values = source.read(run / 'development.json')
    source.require(values['freeze_sha256'] == source.digest(run / 'freeze.json'), 'Development readings differ')
    readings = values['rows']
    models, labels = fit_heads([r for r in rows if r['cohort'] == 'fit'], readings)
    maxima = {name: defaultdict(list) for name in MODELS}
    for row in rows:
        if row['cohort'] != 'calibration':
            continue
        for name, scores in measure(readings[row['id']], models).items():
            maxima[name][row['group']].extend(v for v in scores if v is not None)
    by_group = {name: {group: max(items) for group, items in records.items() if items} for name, records in maxima.items()}
    source.require(all(len(groups) == 21 for groups in by_group.values()), 'Calibration source group unavailable')
    points = {name: sorted(groups.values())[-2] for name, groups in by_group.items()}
    source.seal(run / 'points.json', {'freeze_sha256': source.digest(run / 'freeze.json'), 'models': models,
                                    'points': points, 'maxima': by_group, 'fit_labels': labels})
    print('locked points', points, 'fit labels', labels, flush=True)


def evaluate(rows, points):
    counts = defaultdict(lambda: defaultdict(Counter))
    gains = {name: set() for name in MODELS}
    additional = {name: defaultdict(set) for name in MODELS}
    for row in rows:
        spans = row['spans']
        source.require(len(spans) == len(row['primary']) and all(len(v) == len(spans) for v in row['scores'].values()),
                       'Held positions and measurements differ')
        base = [[spans[max(0, i-1)][0], spans[min(len(spans)-1, i+1)][1]]
                for i, value in enumerate(row['primary']) if value >= .5][:6]
        options = {'primary': base}
        covered = {'primary': True}
        for name in MODELS:
            items = [(v, -i) for i, v in enumerate(row['scores'][name]) if v is not None]
            best = max(items, default=None)
            covered[name] = best is not None
            options[name] = [spans[-best[1]]] if best is not None and best[0] > points[name] else []
            options[f'plus_{name}'] = base + options[name]
            covered[f'plus_{name}'] = True
        found = {name: any(short.useful(span, row['truth'], set(row['spaces'])) for span in choices)
                 for name, choices in options.items()}
        for name, choices in options.items():
            counts[f"{row['cohort']}/{row['family']}"][name].update(
                files=1, assessed=covered[name], useful=found[name], flags=bool(choices),
                wrong_location=bool(choices) and row['truth'] is not None and not found[name])
        for name in MODELS:
            if found[name] and not found['primary'] and row['cohort'] == 'report':
                gains[name].add((row['group'], row['family']))
            if not row['truth'] and options[name] and not base:
                additional[name][row['cohort']].add(row['group'])
    summary = {key: {name: dict(tally) for name, tally in values.items()} for key, values in counts.items()}
    coverage = sum(summary[f'report/{family}']['word']['assessed'] for family in FAMILIES)
    pass_gate = (len(gains['word']) >= 4 and set(family for _, family in gains['word']) == set(FAMILIES)
                 and len(gains['word']) >= len(gains['context']) + 2 and coverage >= .90 * 2 * GROUPS
                 and all(len(groups) <= 1 for groups in additional['word'].values()))
    return {'counts': summary, 'new_report_locations': {n: len(x) for n, x in gains.items()},
            'new_ordinary_groups': {n: {k: len(g) for k, g in a.items()} for n, a in additional.items()},
            'passes': pass_gate}


def publish(run, output):
    lock, rows = load(run)
    held, points = (source.read(run / name) for name in ('held.json', 'points.json'))
    expected = source.digest(run / 'freeze.json')
    source.require(held['freeze_sha256'] == expected == points['freeze_sha256'] and
                   held['points_sha256'] == source.digest(run / 'points.json'), 'Held measurements or frozen points differ')
    values = []
    for row in rows:
        if row['cohort'] in {'fit', 'calibration'}:
            continue
        reading = held['rows'][row['id']]
        values.append({k: v for k, v in row.items() if k not in {'text', 'context', 'publisher_record'}} |
                      {'primary': [v for v, _ in reading], 'scores': measure(reading, points['models'])})
    report = {'schema': SCHEMA, 'freeze': lock, 'points': points, 'held_inputs': held['inputs'], 'rows': values,
              'summary': evaluate(values, points['points'])}
    source.seal(output, report)
    print(report['summary']['new_report_locations'], report['summary']['new_ordinary_groups'], report['summary']['passes'])


def review(path):
    report = source.read(path)
    source.require(report['schema'] == SCHEMA and report['freeze']['method'] == METHOD, 'Unexpected receipt')
    points = report['points']
    source.require(all(len(values) == 21 and sorted(values.values())[-2] == points['points'][name]
                       and all(math.isfinite(v) for v in values.values()) for name, values in points['maxima'].items()),
                   'Locked operating points differ')
    rows = report['rows']
    source.require(Counter(row['cohort'] for row in rows) == {'report': 60, 'forum': 16, 'uswds': 12, 'elife': 36}
                   and len({row['id'] for row in rows}) == len(rows), 'Incomplete held cohort')
    for row in rows:
        source.require(all(0 <= a < b <= row['length'] for a, b in row['spans']) and
                       (row['truth'] is None or row['truth'] in row['spans']) and
                       all(v is None or math.isfinite(v) for values in row['scores'].values() for v in values),
                       'Invalid held sentence geometry or reading')
    observed = evaluate(rows, points['points'])
    source.require(observed == report['summary'], 'Recomputed outcomes differ')
    print(source.serialize(observed), end='')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('freeze', 'score-development', 'cutoff', 'score-held', 'publish', 'review'))
    parser.add_argument('path', type=Path)
    parser.add_argument('--cache', type=Path)
    parser.add_argument('--short-roster', type=Path)
    parser.add_argument('--elife-roster', type=Path)
    parser.add_argument('--snapshot', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.phase == 'freeze':
        freeze(args.path, args.cache, args.short_roster, args.elife_roster, args.snapshot)
    elif args.phase.startswith('score-'):
        score(args.path, args.snapshot, args.phase.removeprefix('score-'))
    elif args.phase == 'cutoff':
        cutoff(args.path)
    elif args.phase == 'publish':
        publish(args.path, args.output)
    else:
        review(args.path)


if __name__ == '__main__':
    main()
