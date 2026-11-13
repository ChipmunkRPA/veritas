"""Compare writing checks on pinned USWDS editorial changes and matched AI rewrites."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from tempfile import TemporaryDirectory

SOURCE_COMMIT = '37f277bbdce08fbcf9bd463fa0c4b33516ceed61'
SCHEMA = 'chipmunk.writing-uswds-revision-transfer/1'
SEEDS = {'granite': 862301, 'olmo': 975483}
PROMPT = ('Rewrite this text as one complete English sentence in different words. Preserve its meaning, names and facts. '
          'Use 12 to 35 words. Return the replacement alone, with no heading or explanation.\n\n{}')
NATIVE_CUTOFF = 0.5
PAIRED_LOG_ODDS = math.log(2)
PROTOCOL = {
    'primary': 'First six native three-sentence primary responses at or above the incumbent point.',
    'change_intersection': 'Keep a primary range only where it intersects a changed token in the exact paired draft.',
    'paired_direction': ('Same-count drafts with a changed native context and overlapping exact changed token. '
                         'Both independent detector odds must at least double.'),
    'draft_comparison': 'Exact changed tokens as factual version evidence. It does not attribute the edit to AI.',
    'useful': 'At least one proposal has 0.30 IoU and covers half the replaced sentence, ignoring whitespace.',
}


def sha(value):
    return hashlib.sha256(value.encode() if isinstance(value, str) else value).hexdigest()


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def write(path, value):
    with path.open('x', encoding='utf8') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sentences(raw):
    from lxml import html
    from markdown_it import MarkdownIt

    from app.detectors.text.extract import from_string
    from app.detectors.text.segment import segment

    text = re.sub(r'\A---\s*\n.*?\n---\s*\n', '', raw.decode('utf-8-sig'), count=1, flags=re.S)
    text = re.sub(r'{%.*?%}|{{.*?}}', ' ', text, flags=re.S)
    rendered = MarkdownIt('commonmark', {'html': True}).render(text)
    dom = html.fragment_fromstring(rendered, create_parent='div')
    prose = []
    for node in dom.xpath('.//p[not(ancestor::li or ancestor::blockquote)] | .//li[not(ancestor::li or ancestor::blockquote)]'):
        if node.xpath('.//code | .//img | .//script | .//style | .//blockquote | .//ul | .//ol | .//table'):
            continue
        text = ' '.join(node.text_content().split())
        if (len(text.split()) >= 5 and re.search(r'[.!?]["\u201d\u2019]?$', text)
                and not re.search(r'https?://|\b(?:USA-|Aria-|HTTP\w*:)\S*', text, re.I)):
            prose.append(text)
    return [unit.text for unit in segment(from_string('\n\n'.join(prose)))]


def qualifying(record, old, new):
    a, b = sentences(old), sentences(new)
    for tag, i, j, x, y in SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if tag != 'replace' or not 1 <= j - i <= 4 or not 1 <= y - x <= 4:
            continue
        if min(i, x, len(a) - j, len(b) - y) < 1 or a[i - 1] != b[x - 1] or a[j] != b[y]:
            continue
        before, after = ' '.join(a[i:j]), ' '.join(b[x:y])
        if min(len(before.split()), len(after.split())) < 10 or max(len(before.split()), len(after.split())) > 120:
            continue
        similarity = SequenceMatcher(None, re.findall(r'\w+', before.lower()), re.findall(r'\w+', after.lower()), autojunk=False).ratio()
        if not .35 <= similarity <= .95:
            continue
        left = a[i - 2:i] if min(i, x) >= 2 and a[i - 2] == b[x - 2] else a[i - 1:i]
        right = a[j:j + 2] if j + 1 < len(a) and y + 1 < len(b) and a[j + 1] == b[y + 1] else a[j:j + 1]
        target = next((k for k, value in enumerate(a[i:j]) if 10 <= len(value.split()) <= 40), None)
        if target is not None:
            yield {**record, 'before_blob': sha(old), 'after_blob': sha(new), 'before_focus': a[i:j], 'after_focus': b[x:y],
                   'left': left, 'right': right, 'target': target, 'similarity': round(similarity, 4)}


def frozen_inputs(run, generator):
    records = read(run / 'candidates.json')
    require(len(records) == 36 and all(row['date'] < '2020-01-01' for row in records), 'Wrong historical source intake')
    license_file = run / 'LICENSE.md'
    require('CC0 1.0 Universal public domain dedication' in license_file.read_text(), 'Missing CC0 terms')
    groups = defaultdict(list)
    for record in records:
        blobs = [(run / 'native' / sha(record['path'] + '@' + record[role])).read_bytes() for role in ('before', 'after')]
        groups[record['path']].extend(qualifying(record, *blobs))
    selected = []
    for path, rows in sorted(groups.items()):
        if not rows:
            continue
        row = max(rows, key=lambda r: (min(len(r['left']), len(r['right'])), -abs(r['similarity'] - .7), r['after']))
        selected.append({**row, 'group': sha(path)[:16]})
    require(len(selected) == 6 and len({row['after'] for row in selected}) >= 4, 'Expected six pages from at least four commits')
    proto = read(generator / 'protocol.json')
    models = {family: proto['weights'][family] for family in SEEDS}
    for family, model in models.items():
        require(model['license'] == 'Apache-2.0' and all(digest(generator / 'models' / family / file) == expected
                for file, expected in model['assets'].items()), 'A pinned generator or license differs')
    write(run / 'roster.json', selected)
    write(run / 'lock.json', {'schema': SCHEMA, 'source_commit': SOURCE_COMMIT, 'source_license': 'CC0-1.0',
          'license_sha256': digest(license_file), 'intake_sha256': digest(run / 'candidates.json'), 'roster_sha256': digest(run / 'roster.json'),
          'code_sha256': digest(Path(__file__)), 'models': models, 'seeds': SEEDS, 'prompt': PROMPT,
          'attempts': 2, 'native_cutoff': NATIVE_CUTOFF, 'paired_log_odds': PAIRED_LOG_ODDS, 'protocol': PROTOCOL})


def lock_of(run):
    lock = read(run / 'lock.json')
    require(lock['roster_sha256'] == digest(run / 'roster.json') and lock['code_sha256'] == digest(Path(__file__)), 'Roster or protocol changed')
    return lock


def generate(run, generator, family):
    import mlx.core as mx
    from mlx_lm import load, stream_generate
    from mlx_lm.sample_utils import make_sampler

    lock = lock_of(run)
    meta = lock['models'][family]
    mx.set_default_device(mx.gpu)
    mx.set_cache_limit(48 * 1024 * 1024)
    with TemporaryDirectory(prefix='uswds-model-', dir=run) as temp:
        for file, expected in meta['assets'].items():
            original = generator / 'models' / family / file
            require(digest(original) == expected, 'Generator bytes changed')
            output = Path(temp) / meta['loader_aliases'].get(file, file)
            if file == 'config.json' and meta['typed_config_overrides']:
                write(output, read(original) | meta['typed_config_overrides'])
            else:
                output.symlink_to(original.resolve())
        model, tokenizer = load(temp, tokenizer_config={'trust_remote_code': False})
    sample = make_sampler(temp=.65, top_p=.90)
    result = {}
    for i, row in enumerate(read(run / 'roster.json')):
        original = row['before_focus'][row['target']]
        chat = tokenizer.apply_chat_template([{'role': 'user', 'content': PROMPT.format(original)}], tokenize=False, add_generation_prompt=True)
        attempts = []
        for j in range(lock['attempts']):
            seed = SEEDS[family] + 2 * i + j
            mx.random.seed(seed)
            tokens = list(stream_generate(model, tokenizer, chat, sampler=sample, max_tokens=96))
            attempts.append({'text': ''.join(row.text for row in tokens), 'eos': tokens[-1].finish_reason == 'stop', 'seed': seed})
        result[row['group']] = {'input_sha256': sha(original), 'attempts': attempts}
        print(family, len(result), flush=True)
    write(run / f'{family}.json', {'lock_sha256': digest(run / 'lock.json'), 'rows': result})


def bind_case(row, family, value):
    from app.detectors.text.extract import from_string
    from app.detectors.text.segment import segment
    from app.evidence.document_revision import compare

    left, old, right = row['left'], row['before_focus'], row['right']
    target = len(left) + row['target']
    words = [*left, *value, *right]
    text = ' '.join(words)
    reference = ' '.join([*left, *old, *right])
    doc = from_string(text)
    units = segment(doc)
    require(doc.text == text and [u.text for u in units] == words, 'Native sentence boundary moved')
    native = []
    for i in range(len(units)):
        group = units[max(0, i - 1):i + 2]
        native.append({'text': ' '.join(unit.text for unit in group), 'span': [group[0].start, group[-1].end]})
    diff = compare(reference.encode(), text.encode(), '.txt')
    locations = [span for change in diff['changes'] if change['later'] for span in change['later']['changed_tokens']]
    require(bool(locations) == (family != 'before'), 'Native draft comparison did not bind the new wording')
    return {'group': row['group'], 'family': family, 'text': text, 'target': target,
            'truth': [units[target].start, units[target].end] if family in SEEDS else None,
            'native': native, 'locations': locations}


def cases(run):
    from scripts.writing_blind_rewrite_probe import valid_sentence

    lock_of(run)
    receipts = {family: read(run / f'{family}.json') for family in SEEDS}
    require(all(x['lock_sha256'] == digest(run / 'lock.json') for x in receipts.values()), 'Generator and frozen input disagree')
    out = []
    for row in read(run / 'roster.json'):
        for role in ('before', 'after'):
            out.append(bind_case(row, role, row[role + '_focus']))
        for family, receipt in receipts.items():
            original, result = row['before_focus'][row['target']], receipt['rows'][row['group']]
            require(result['input_sha256'] == sha(original), 'Generator input changed')
            for trial in result['attempts']:
                value = valid_sentence(trial, original, generated=True)
                if value:
                    after = list(row['before_focus'])
                    after[row['target']] = value
                    out.append(bind_case(row, family, after))
                    break
    require(all(sum(row['family'] == f for row in out) >= 3 for f in SEEDS), 'Too few valid rewrites across generator families')
    return out


def overlap(left, right):
    a, b = left
    x, y = right
    return max(a, x) < min(b, y)


def direction(value):
    score = min(max(float.fromhex(value), 1e-6), 1 - 1e-6)
    return math.log(score / (1 - score))


def proposal(row, earlier):
    windows = row['native']
    primary = [window for window in windows if float.fromhex(window['tropa']) >= NATIVE_CUTOFF][:6]
    baseline = [window['span'] for window in primary]
    intersect = [window['span'] for window in primary if any(overlap(window['span'], span) for span in row['locations'])]
    candidates = []
    if len(windows) == len(earlier['native']):
        for current, before in zip(windows, earlier['native'], strict=True):
            if (current['sha256'] != before['sha256'] and any(overlap(current['span'], span) for span in row['locations'])
                    and all(direction(current[key]) - direction(before[key]) >= PAIRED_LOG_ODDS for key in ('tropa', 'modernbert'))):
                candidates.append(current['span'])
    return {'primary': baseline, 'change_intersection': intersect, 'paired_direction': candidates, 'draft_comparison': row['locations']}


def results(rows):
    earlier = {row['group']: row for row in rows if row['family'] == 'before'}
    out = defaultdict(lambda: defaultdict(Counter))
    for row in rows:
        truth, whitespace = set(row['truth_indices']), set(row['spaces'])
        for method, ranges in proposal(row, earlier[row['group']]).items():
            spans = [ranges] if method == 'draft_comparison' else [[item] for item in ranges]
            positions = [{i for a, b in intervals for i in range(a, b)} - whitespace for intervals in spans]
            useful = any(2 * len(items & truth) >= len(truth) and 10 * len(items & truth) >= 3 * len(items | truth)
                         for items in positions) if truth else False
            selected = set().union(*positions)
            matched = len(selected & truth)
            stats = out[row['family']][method]
            stats.update(files=1, flags=bool(selected), useful=useful,
                         ordinary_characters=len(selected) - matched, labeled_characters=matched)
    return out


def score(run, first, second, output):
    import torch

    from scripts.score_text_upgrade import sequence_predictor
    from scripts.writing_focus_v3 import BASELINE_ASSETS
    from scripts.writing_paired_email_revision import SECONDARY

    frozen = lock_of(run)
    native = cases(run)
    texts = sorted({w['text'] for row in native for w in row['native']}, key=sha)
    torch.set_num_threads(2)
    reading = {}
    for mode, model, checks in (('tropa', first, BASELINE_ASSETS), ('modernbert', second, SECONDARY)):
        require(all(digest(model / filename) == expected for filename, expected in checks.items()), 'Pinned detector differs')
        predict = sequence_predictor(mode, model, 'cpu', 4)
        reading[mode] = {sha(text): value.hex() for text, value in zip(texts, predict(texts), strict=True)}
        del predict
        print(mode, len(texts), flush=True)
    rows = []
    for row in native:
        text, truth = row.pop('text'), row.pop('truth')
        for window in row['native']:
            digest_ = sha(window.pop('text'))
            window.update(sha256=digest_, **{mode: data[digest_] for mode, data in reading.items()})
        spaces = [i for i, char in enumerate(text) if char.isspace()]
        row.update(text_sha256=sha(text), characters=len(text), spaces=spaces,
                   truth_indices=[i for i in range(*truth) if not text[i].isspace()] if truth else [])
        rows.append(row)
    public_sources = [{k: row[k] for k in ('group', 'path', 'before', 'after', 'date', 'before_blob', 'after_blob')}
                      for row in read(run / 'roster.json')]
    write(output, {'schema': SCHEMA, 'freeze': frozen, 'freeze_sha256': digest(run / 'lock.json'),
                   'sources': public_sources, 'generation_sha256': {f: digest(run / f'{f}.json') for f in SEEDS},
                   'inputs': len(texts), 'rows': rows, 'results': results(rows)})


def review(file):
    ledger = read(file)
    require(ledger['schema'] == SCHEMA and ledger['freeze']['source_commit'] == SOURCE_COMMIT
            and ledger['freeze']['native_cutoff'] == NATIVE_CUTOFF and ledger['freeze']['paired_log_odds'] == PAIRED_LOG_ODDS
            and ledger['freeze']['protocol'] == PROTOCOL,
            'Unknown source and operating points')
    rows = ledger['rows']
    groups = {row['group'] for row in ledger['sources']}
    require(len(groups) == 6 and {(row['group'], row['family']) for row in rows if row['family'] in {'before', 'after'}}
            == {(group, role) for group in groups for role in ('before', 'after')}, 'Incomplete genuine controls')
    for row in rows:
        require(row['group'] in groups and all(0 <= a < b <= row['characters'] for a, b in row['locations'])
                and all(0 <= a < b <= row['characters'] and all(0 <= float.fromhex(w[key]) <= 1 for key in ('tropa', 'modernbert'))
                        for w in row['native'] for a, b in [w['span']]), 'Invalid score or source location')
    require(results(rows) == ledger['results'], 'Outcomes differ from the recorded inputs')
    print(json.dumps(ledger['results'], sort_keys=True, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('freeze', 'generate', 'score', 'review'))
    parser.add_argument('path', type=Path)
    parser.add_argument('--generator', type=Path)
    parser.add_argument('--family', choices=SEEDS)
    parser.add_argument('--first', type=Path)
    parser.add_argument('--second', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.stage == 'freeze':
        frozen_inputs(args.path, args.generator)
    elif args.stage == 'generate':
        generate(args.path, args.generator, args.family)
    elif args.stage == 'score':
        score(args.path, args.first, args.second, args.output)
    else:
        review(args.path)


if __name__ == '__main__':
    main()
