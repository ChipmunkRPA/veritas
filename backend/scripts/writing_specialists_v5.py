"""Private, source-bound real-text specialist calibration pilot; never serving."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import io
import json
import re
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

from app.bench import specialists as engine
from scripts import specialist_calibration as fitting
from scripts import writing_modernbert_v4 as models

SOURCE_SHA = '232c260f8caf7ea61a331f39fa730956862afbe0c5fc554720daa07d215a2f7d'
MODELS = ('baseline', 'candidate')
ROLES = engine.ROLES
AI_CLASSES = ('ChatGPT', 'Grok', 'Copilot', 'Gemini', 'DeepSeek', 'DeepAI', 'Qwen',
    'Claude', 'Llama', 'Gemma', 'Z.ai', 'Perplexity', 'Mistral AI', 'Poe', 'Nova', 'BLACKBOXAI', 'Pi')
PATTERNS = {'human': (0, 0), 'ai': (1, 1), 'human-ai': (0, 1), 'ai-human': (1, 0)}
ARMS = {'baseline-global': (('baseline',), False), 'baseline-context': (('baseline',), True),
    'combined-global': (MODELS, False), 'combined-context': (MODELS, True)}
METHOD = {
    'schema': 'chipmunk.writing-specialists-method/5',
    'question_groups': 128, 'groups_per_role': 32, 'minimum_groups_per_class': 25,
    'selection': 'Hash-ordered connected prompt/answer components, unchanged ASCII strings only; >=160 whitespace words per source; one human and one AI answer per component.',
    'passages': 'First60 words and next100 words, copied as exact contiguous source slices; no paraphrasing. Four two-passage compositions with explicit source provenance.',
    'routing': 'Measured whitespace-word count: <=75 short, >75 long; no generator, label, score or source identity in routing.',
    'combination': 'One dependency family per fixed detector; supervised nonnegative combiner, separate combination calibration; no independence claim.',
    'roles': list(ROLES), 'max_len': 512, 'batch': 4, 'threads': 4,
    'native_audit': 'First hash-ordered input for each training role/context/truth class, plus every tokenizer overflow; separate direct unbatched forward, abs difference <=1e-5.',
    'diagnostic_point': .5, 'diagnostic_tie_tolerance': 1e-12,
    'admission': 'Development pilot only: author IDs absent and external detector training overlap unknown. No serving admission regardless of numerical improvement.',
    'reference_prior': .5, 'serving_admitted': False,
    'resource_guard': {'tree_rss_gib': 6, 'minimum_available_mib': 1536, 'minimum_disk_mib': 1536, 'maximum_seconds': 1800},
}


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def text_sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def normalized(text):
    return ''.join(c for c in unicodedata.normalize('NFKC', text).casefold() if c.isalnum())


def source_components(rows):
    """Connect equal normalized prompts AND answers before selecting any rows."""
    parent = list(range(len(rows)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    seen = {}
    for index, row in enumerate(rows):
        for field in ('Question', 'Answer'):
            key = field, text_sha(normalized(row[field]))
            if key in seen:
                a, b = find(index), find(seen[key])
                parent[max(a, b)] = min(a, b)
            else:
                seen[key] = index
    grouped = defaultdict(list)
    for index in range(len(rows)):
        grouped[find(index)].append(index)
    return list(grouped.values())


def slices(text):
    words = list(re.finditer(r'\S+', text))
    if len(words) < 160:
        raise ValueError('Source needs160 unchanged words')
    spans = ((0, words[59].end()), (words[60].start(), words[159].end()))
    return [{'start': a, 'end': b, 'text': text[a:b], 'sha256': text_sha(text[a:b])} for a, b in spans]


def code():
    paths = {'adapter': __file__, 'engine': engine.__file__, 'fitting': fitting.__file__,
        'text_loader': str(Path(__file__).with_name('run_writing_modernbert_v4.py')),
        'text_coverage': str(Path(__file__).parents[1] / 'app/detectors/text/coverage.py'),
        'text_classifier': str(Path(__file__).parents[1] / 'app/detectors/text/classifier.py')}
    return {k: sha(v) for k, v in paths.items()}


def runtime():
    return {k: importlib.metadata.version(k) for k in
        ('torch', 'transformers', 'tokenizers', 'numpy', 'scipy', 'safetensors')}


def prepare(source, trial, baseline, candidate, license_record):
    if sha(source) != SOURCE_SHA:
        raise ValueError('Exact acquired publisher CSV required')
    # Latin1 is a byte-preserving CSV container, not an interpretation of the
    # ambiguous non-ASCII encoding. Only untouched ASCII strings are eligible.
    rows = list(csv.DictReader(io.StringIO(source.read_bytes().decode('latin1'), newline='')))
    if len(rows) != 66000 or set(rows[0]) != {'Question', 'Answer', 'Source'}:
        raise ValueError('Publisher source structure/count changed')
    counts = Counter(r['Source'] for r in rows)
    if len(counts) != 22 or set(counts.values()) != {3000}:
        raise ValueError('Publisher class counts changed')
    eligible = []
    components = source_components(rows)
    for component in components:
        allowed = [i for i in component if rows[i]['Source'] in (*AI_CLASSES, 'Human written')
            and all(rows[i][k].isascii() for k in ('Question', 'Answer', 'Source'))
            and len(rows[i]['Answer'].split()) >= 160]
        humans = [i for i in allowed if rows[i]['Source'] == 'Human written']
        ais = [i for i in allowed if rows[i]['Source'] in AI_CLASSES]
        if not humans or not ais:
            continue
        commitment = engine.digest(sorted({text_sha(normalized(rows[i]['Question'])) for i in component}))
        eligible.append((commitment, component, humans, ais))
    eligible.sort(key=lambda x: text_sha('writing-specialists-v5-component:' + x[0]))
    if len(eligible) < METHOD['question_groups']:
        raise ValueError('Insufficient eligible independent prompt components; do not lower support')
    chosen, observed = [], Counter()
    for index, (commitment, component, humans, ais) in enumerate(eligible[:METHOD['question_groups']]):
        # Round-robin generator preference is source metadata selection, never
        # a routing feature or detector-score-based choice. All17 are retained.
        preferred = AI_CLASSES[index % len(AI_CLASSES)]
        human = min(humans, key=lambda i: text_sha(rows[i]['Answer']))
        ai = min(ais, key=lambda i: (rows[i]['Source'] != preferred,
            text_sha(commitment + rows[i]['Source'] + rows[i]['Answer'])))
        role = ROLES[index % len(ROLES)]
        sources = []
        for truth, row_index in ((0, human), (1, ai)):
            row = rows[row_index]
            sources.append({'row': row_index, 'label': truth, 'generator': row['Source'],
                'question_sha256': text_sha(row['Question']), 'text_sha256': text_sha(row['Answer']),
                'text': row['Answer'], 'passages': slices(row['Answer'])})
        chosen.append({'id': commitment, 'role': role, 'component_rows': component, 'sources': sources})
        observed[role, rows[ai]['Source']] += 1
    trial.mkdir(exist_ok=False)
    plan = {'schema': 'chipmunk.writing-specialists-plan/5', 'source_sha256': SOURCE_SHA,
        'license_record_sha256': sha(license_record), 'method': METHOD, 'groups': chosen,
        'eligible_components': len(eligible), 'all_source_components': len(components),
        'source_records': len(rows), 'mixed_rows_used_for_truth': 0, 'model_forwards': 0,
        'known_limits': ['No participant IDs: prompt/duplicate disjointness is not author disjointness.',
            'Publisher source labels are not an independent authorship examination.',
            'External detector training overlap unknown; preliminary development only.',
            'Artificial concatenation boundaries and ASCII-only English are limited conditions.']}
    write(trial / 'plan.json', plan)
    from scripts.run_writing_modernbert_v4 import assets

    paths = {'baseline': baseline.resolve(), 'candidate': candidate.resolve()}
    assets_by_model = {'baseline': assets(baseline, models.BASELINE_ASSETS),
        'candidate': assets(candidate, models.CANDIDATE_ASSETS)}
    frozen = {'schema': 'chipmunk.writing-specialists-freeze/5', 'plan_sha256': sha(trial / 'plan.json'),
        'code_sha256': code(), 'runtime': runtime(), 'assets': assets_by_model,
        'model_paths': {k: str(v) for k, v in paths.items()}, 'method': METHOD, 'serving_admitted': False}
    write(trial / 'freeze.json', frozen)
    print(json.dumps({'groups': len(chosen), 'eligible_components': len(eligible),
        'roles': dict(Counter(g['role'] for g in chosen)), 'generators': len({s['generator']
            for g in chosen for s in g['sources'] if s['label']}),
        'freeze_sha256': sha(trial / 'freeze.json'), 'model_forwards': 0}), flush=True)


def check(trial):
    lock = json.loads((trial / 'freeze.json').read_text())
    if (lock['code_sha256'] != code() or lock['runtime'] != runtime() or lock['method'] != METHOD
            or lock['plan_sha256'] != sha(trial / 'plan.json')):
        raise ValueError('Prospective source/code/runtime freeze changed')
    return lock, json.loads((trial / 'plan.json').read_text())


def inputs(plan, confirmation):
    selected = {}
    for group in plan['groups']:
        if (group['role'] == 'confirmation') != confirmation:
            continue
        for source in group['sources']:
            for piece in source['passages']:
                key = piece['sha256']
                if key in selected:
                    raise ValueError('Repeated selected passage: rebuild connected source graph before scoring')
                selected[key] = {'text': piece['text'], 'group': group['id'], 'role': group['role'],
                    'context': 'short' if len(piece['text'].split()) <= 75 else 'long', 'label': source['label']}
    return dict(sorted(selected.items()))


def score(trial, name, confirmation):
    import torch

    from scripts.run_writing_modernbert_v4 import assets, direct_response, load_model, token_identity

    lock, plan = check(trial)
    selected = inputs(plan, confirmation)
    stage = 'confirmation' if confirmation else 'training'
    path = trial / f'{name}-{stage}.jsonl'
    if path.exists():
        raise FileExistsError('Retain every prior attempt; no implicit resume or overwrite')
    assets(Path(lock['model_paths'][name]), lock['assets'][name])
    torch.set_num_threads(METHOD['threads'])
    clf = load_model(name, lock['model_paths'])
    audit_cells = set()
    audit_count = 0
    maximum = 0.
    started = time.monotonic()
    with path.open('x') as stream:
        stream.write(json.dumps({'schema': 'chipmunk.writing-specialists-responses/5', 'model': name,
            'stage': stage, 'freeze_sha256': sha(trial / 'freeze.json'), 'planned': len(selected)}) + '\n')
        entries = list(selected.items())
        for start in range(0, len(entries), METHOD['batch']):
            batch = entries[start:start + METHOD['batch']]
            texts = [v['text'] for _, v in batch]
            identities = [token_identity(text, clf.tok) for text in texts]
            responses = clf.predict(texts, max_len=METHOD['max_len'], batch=METHOD['batch'], check=lambda _: None)
            if len(responses) != len(batch) or len(clf.last_coverage) != len(batch):
                raise ValueError('One complete native score per planned passage required')
            for (key, source), raw, coverage, actual in zip(batch, responses, identities, clf.last_coverage, strict=True):
                if any(coverage[k] != actual[k] for k in ('tokens', 'token_windows')):
                    raise ValueError('Native scored-token coverage differs')
                row = {'text_sha256': key, 'status': 'ok', 'raw': models.finite_score(raw), 'coverage': coverage}
                cell = source['role'], source['context'], source['label']
                if cell not in audit_cells or coverage['token_windows'] > 1:
                    delta = abs(direct_response(clf, name, source['text']) - raw)
                    if delta > 1e-5:
                        raise ValueError('Independent direct native forward disagrees')
                    maximum = max(maximum, delta)
                    audit_cells.add(cell)
                    audit_count += 1
                    row['direct_absolute_difference'] = delta
                stream.write(json.dumps(row, allow_nan=False) + '\n')
            stream.flush()
            print(json.dumps({'model': name, 'stage': stage, 'passages': min(start + len(batch), len(entries)),
                'planned': len(entries), 'elapsed_s': round(time.monotonic() - started, 2)}), flush=True)
    write(trial / f'{name}-{stage}-native.json', {'schema': 'chipmunk.writing-specialists-native/5',
        'freeze_sha256': sha(trial / 'freeze.json'), 'journal_sha256': sha(path), 'model': name, 'stage': stage,
        'complete_passages': len(selected), 'direct_audits': audit_count, 'max_absolute_difference': maximum,
        'source_texts_published': False, 'serving_admitted': False})


def profile(trial, name, confirmation):
    """Bound actual inference; retain guard failures and every partial journal."""
    import shutil
    import subprocess
    import sys

    import psutil

    check(trial)
    stage = 'confirmation' if confirmation else 'training'
    folder = trial / f'{name}-{stage}-profile'
    folder.mkdir(exist_ok=False)
    command = [sys.executable, '-m', 'scripts.writing_specialists_v5', 'score', str(trial), '--model', name]
    if confirmation:
        command.append('--confirmation')
    started, maximum, guard, sample_errors = time.monotonic(), 0, None, 0
    with (folder / 'worker.log').open('x') as log:
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        process = psutil.Process(child.pid)
        while child.poll() is None:
            tree, rss = [process], 0
            try:
                tree += process.children(recursive=True)
                rss = sum(p.memory_info().rss for p in tree)
                maximum = max(maximum, rss)
            except (psutil.NoSuchProcess, psutil.ZombieProcess):
                sample_errors += 1
            if (rss > 6 * 1024**3 or psutil.virtual_memory().available < 1536 * 1024**2
                    or shutil.disk_usage(trial).free < 1536 * 1024**2 or time.monotonic() - started > 1800):
                guard = 'memory_disk_or_time'
                for member in reversed(tree):
                    try:
                        member.terminate()
                    except psutil.NoSuchProcess:
                        sample_errors += 1
                break
            time.sleep(.25)
        try:
            status = child.wait(timeout=15)
        except subprocess.TimeoutExpired:
            child.kill()
            status = child.wait()
    value = {'stage': stage, 'model': name, 'child_exit_code': status, 'guard': guard,
        'sampled_tree_rss_lower_bound_bytes': maximum, 'sample_errors': sample_errors,
        'duration_s': time.monotonic() - started}
    write(folder / 'profile.json', value)
    print(json.dumps(value), flush=True)
    if status or guard:
        raise RuntimeError('Retained failed/guarded attempt; no silent restart')


def responses(trial, name, confirmation):
    stage = 'confirmation' if confirmation else 'training'
    path = trial / f'{name}-{stage}.jsonl'
    receipt = json.loads((trial / f'{name}-{stage}-native.json').read_text())
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if (rows[0]['freeze_sha256'] != sha(trial / 'freeze.json')
            or receipt['journal_sha256'] != sha(path) or receipt['complete_passages'] != len(rows) - 1):
        raise ValueError('Complete source-bound native inference receipt required')
    return {r['text_sha256']: r for r in rows[1:]}


def manifest(trial, arm, with_confirmation):
    lock, plan = check(trial)
    names, context = ARMS[arm]
    records = {}
    for name in names:
        records[name] = responses(trial, name, False)
        if with_confirmation:
            records[name].update(responses(trial, name, True))
    routing = engine.digest({'word_count': '<=75 short; >75 long', 'contextual': context})
    specs = [{'id': 'writing.' + name, 'modality': 'text', 'family': name, 'target': 'text.ai',
        'contexts': ['short', 'long'] if context else ['pooled'], 'routing_sha256': routing,
        'higher_means_ai': True, 'raw_range': [0., 1.], 'producer': {'kind': 'model',
            'implementation_sha256': engine.digest(lock['code_sha256']),
            'weight_sha256': [lock['assets'][name]['model.safetensors']],
            'commercial_license_record_sha256': plan['license_record_sha256'],
            'native_receipt_sha256': sha(trial / f'{name}-training-native.json')}} for name in names]
    files = []
    for group in plan['groups']:
        for pattern_name, truth in PATTERNS.items():
            pieces = [group['sources'][truth[k]]['passages'][k] for k in range(2)]
            text = pieces[0]['text'] + '\n\n' + pieces[1]['text']
            text_hash = text_sha(text)
            units = []
            for index, piece in enumerate(pieces):
                start = 0 if index == 0 else len(pieces[0]['text']) + 2
                provenance = {'source_sha256': SOURCE_SHA, 'row': group['sources'][truth[index]]['row'],
                    'source_text_sha256': group['sources'][truth[index]]['text_sha256'],
                    'source_span': [piece['start'], piece['end']], 'label': truth[index]}
                for name in names:
                    result = records[name].get(piece['sha256'])
                    unit = {'id': f'{name}.{index}', 'specialist': 'writing.' + name,
                        'context': ('short' if len(piece['text'].split()) <= 75 else 'long') if context else 'pooled',
                        'routing_sha256': routing, 'context_evidence_sha256': engine.digest({'words': len(piece['text'].split())}),
                        'measurement_sha256': engine.digest(result) if result else piece['sha256'],
                        'location': {'kind': 'text', 'unit': 'unicode_codepoints', 'start': start,
                            'end': start + len(piece['text']), 'extent': len(text), 'text_sha256': text_hash},
                        'status': 'ok' if result else 'unavailable', 'raw': result['raw'] if result else None,
                        'label': truth[index], 'label_scope': 'local_unit', 'label_provenance_sha256': engine.digest(provenance)}
                    if result is None:
                        unit['reason'] = 'Reserved confirmation passage has not been scored.'
                    units.append(unit)
            files.append({'id': group['id'] + '-' + pattern_name, 'group': group['id'], 'role': group['role'],
                'modality': 'text', 'source_sha256': engine.digest([s['text_sha256'] for s in group['sources']]),
                'input_sha256': text_hash, 'lineage': [s['text_sha256'] for s in group['sources']],
                'label': int(any(truth)), 'label_scope': 'planned_units',
                'label_provenance_sha256': engine.digest({'labels': truth, 'group': group['id']}),
                'source_geometry': {'codepoints': len(text), 'text_sha256': text_hash},
                'required_families': list(names), 'units': units})
    result = {'schema': engine.SCHEMA, 'policy': {'specialists': specs}, 'files': files}
    engine.audit(result)
    return result


def fit(trial):
    for arm in ARMS:
        document = manifest(trial, arm, False)
        write(trial / f'{arm}-manifest-before-confirmation.json', document)
        frozen = fitting.freeze(document, METHOD['minimum_groups_per_class'], .001)
        write(trial / f'{arm}-fit-freeze.json', frozen)
        artifact = fitting.fit(document, frozen)
        if any(h['status'] != 'fitted' for h in artifact['specialists'].values()) or any(
                c['response']['status'] != 'fitted' or c.get('calibration', {}).get('status') != 'fitted'
                for c in artifact['combiners'].values()):
            raise ValueError('Support floor failed; preserve artifact and stop before confirmation')
        write(trial / f'{arm}-artifact.json', artifact)
    write(trial / 'fit-complete.json', {'artifacts': {arm: sha(trial / f'{arm}-artifact.json') for arm in ARMS},
        'confirmation_forwards': 0, 'serving_admitted': False})


def summarize(trial):
    completion = json.loads((trial / 'fit-complete.json').read_text())
    report = {'schema': 'chipmunk.writing-specialists-result/5', 'arms': {}, 'method': METHOD,
        'freeze_sha256': sha(trial / 'freeze.json'), 'serving_admitted': False}
    for arm in ARMS:
        if completion['artifacts'][arm] != sha(trial / f'{arm}-artifact.json'):
            raise ValueError('Confirmation cannot change any fitted artifact')
        document = manifest(trial, arm, True)
        artifact = json.loads((trial / f'{arm}-artifact.json').read_text())
        result = engine.evaluate(document, artifact)
        write(trial / f'{arm}-confirmation.json', result)
        rows = result['outcomes']
        cells = {'false_ai_files': 0, 'missed_ai_files': 0, 'inconclusive_files': 0,
            'correct_ai_files': 0, 'correct_human_files': 0}
        for row in rows:
            p = row['probability']
            if p is None or abs(p - .5) <= METHOD['diagnostic_tie_tolerance']:
                cells['inconclusive_files'] += 1
            elif row['label']:
                cells['correct_ai_files' if p > .5 else 'missed_ai_files'] += 1
            else:
                cells['false_ai_files' if p > .5 else 'correct_human_files'] += 1
        local_truth = {(f['id'], u['id']): u['label'] for f in document['files'] for u in f['units']}
        local = result['unit_outcomes']
        by_context = {}
        for context in sorted({u['context'] for u in local}):
            chosen = [u for u in local if u['context'] == context]
            scored = [u for u in chosen if u['probability'] is not None]
            by_context[context] = {'planned_units': len(chosen), 'scored_units': len(scored),
                'brier': sum((u['probability'] - local_truth[u['file_id'], u['unit_id']])**2
                    for u in scored) / len(scored) if scored else None}
        report['arms'][arm] = {**result['summaries']['text'], **cells, 'local_contexts': by_context,
            'artifact_sha256': completion['artifacts'][arm]}
    report['limits'] = json.loads((trial / 'plan.json').read_text())['known_limits']
    report['interpretation'] = 'Internal50:50 diagnostic tendency only, no public threshold or validated customer probability. No automatic adoption; absent author IDs and unknown external training overlap require additional independent sources.'
    write(trial / 'result.json', report)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('prepare', 'score', 'profile', 'fit', 'summarize'))
    parser.add_argument('trial', type=Path)
    parser.add_argument('--source', type=Path)
    parser.add_argument('--baseline', type=Path)
    parser.add_argument('--candidate', type=Path)
    parser.add_argument('--license-record', type=Path)
    parser.add_argument('--model', choices=MODELS)
    parser.add_argument('--confirmation', action='store_true')
    args = parser.parse_args()
    if args.stage == 'prepare':
        prepare(args.source, args.trial, args.baseline, args.candidate, args.license_record)
    elif args.stage in ('score', 'profile'):
        if args.confirmation and not (args.trial / 'fit-complete.json').exists():
            raise ValueError('All four arms must be fitted and frozen before any confirmation inference')
        (profile if args.stage == 'profile' else score)(args.trial, args.model, args.confirmation)
    elif args.stage == 'fit':
        fit(args.trial)
    else:
        summarize(args.trial)
