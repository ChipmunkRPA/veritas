# SPDX-License-Identifier: MIT
"""Test same-document counterfactual context and token losses at exact sentence positions."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import re
from dataclasses import dataclass
from itertools import accumulate
from pathlib import Path
from statistics import median

POLICY = {
    'scope_sentences': [12, 32], 'focus_tokens': [12, 72], 'prefix_limit_tokens': 96,
    'substitute_gap_sentences': 4, 'substitute_min_tokens': 20, 'minimum_complete_centers': 5,
    'calibrate': 'Use the second-highest maximum over distinct genuine works, with a strict comparison.',
    'geometry': 'Score an original sentence, report its exact Unicode range, cover at least half the labeled edit and 30% IoU on non-whitespace characters.',
    'primary': 'Require deficits both in how adjacent text helps the focus and how the focus helps predict its successor.',
    'comparison': 'Measure next-token and same-position token loss on the same focus tokens under the same native prefix.',
    'normalization': 'Use leave-one-out median and 1.4826 times median absolute deviation, with a floor of 0.15 nats.',
}
ARMS = ('left', 'right', 'two_sided', 'repeat', 'next_token')
BATCH = 5
RUNTIME = {'mlx': '0.32.4', 'mlx-lm': '0.31.3', 'transformers': '5.15.1'}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def write(path, result):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


@dataclass(frozen=True)
class Focus:
    index: int
    bounds: tuple[int, int]
    left_reference: int
    right_reference: int
    jobs: tuple[tuple[tuple[int, ...], tuple[int, ...], bool], ...]


def _match_prefix(native, substitute):
    width = min(len(native), len(substitute), POLICY['prefix_limit_tokens'])
    return native[-width:], substitute[-width:]


def _lexical(text):
    return {word for word in re.findall(r"[a-z]+", text.lower()) if len(word) > 3}


def _nearest(words, ids, candidate_indices, native_words, native_ids, width):
    def rank(index):
        tokens = sum(len(ids[index + offset]) for offset in range(width))
        pool = set().union(*(words[index + offset] for offset in range(width)))
        overlap = len(pool & native_words) / max(1, len(pool | native_words))
        return (abs(tokens - len(native_ids)) // 8, -overlap, abs(tokens - len(native_ids)), index)

    return min(candidate_indices, key=rank, default=None)


def clean_case(row):
    text = row['text']
    if sha(text.encode()) != row['sha256']:
        raise ValueError('Source case differs from its exact text digest')
    spans = [(item['start'], item['end']) for item in row['sentences']]
    if (any(not 0 <= a < b <= len(text) for a, b in spans)
            or any(second[0] < first[1] for first, second in zip(spans, spans[1:]))):
        raise ValueError('Native sentence boundaries are invalid')
    truth = row['truth']
    if (any(not 0 <= a < b <= len(text) for a, b in truth)
            or (row['role'] == 'edited') != bool(truth)):
        raise ValueError('Edit geometry differs from its source role')
    return spans


def focuses(row, tokenizer):
    spans = clean_case(row)
    lo, hi = POLICY['scope_sentences']
    if not lo <= len(spans) <= hi:
        return []
    text = row['text']
    sentences = [text[a:b].strip() for a, b in spans]
    words = [_lexical(s) for s in sentences]
    tokens = [tuple(tokenizer(' ' + s, add_special_tokens=False)['input_ids']) for s in sentences]
    separator = tuple(tokenizer('\n\n', add_special_tokens=False)['input_ids'])
    lows, highs = POLICY['focus_tokens']
    output = []
    for i in range(2, len(spans) - 1):
        if not (lows <= len(tokens[i]) <= highs and lows <= len(tokens[i + 1]) <= highs):
            continue
        remote = [j for j in range(len(spans) - 1) if j <= i - 5 or j >= i + 4]
        native_left = tokens[i - 2] + separator + tokens[i - 1]
        left_index = _nearest(words, tokens, remote, words[i - 2] | words[i - 1], native_left, 2)
        right_index = _nearest(words, tokens, remote, words[i], tokens[i], 1)
        if left_index is None or right_index is None:
            continue
        ref_left = tokens[left_index] + separator + tokens[left_index + 1]
        left, left_alt = _match_prefix(native_left, ref_left)
        actual, counter = _match_prefix(tokens[i], tokens[right_index])
        right, right_alt = _match_prefix(tokens[i - 1] + separator + actual, tokens[i - 1] + separator + counter)
        if min(len(left), len(right)) < POLICY['substitute_min_tokens']:
            continue
        jobs = ((left + separator, tokens[i], True), (left_alt + separator, tokens[i], False),
                (right + separator, tokens[i + 1], False), (right_alt + separator, tokens[i + 1], False))
        output.append(Focus(i, spans[i], left_index, right_index, jobs))
    return output


def batches(items, take):
    for offset in range(0, len(items), take):
        yield items[offset:offset + take]


def initialize(snapshot):
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm import load
    from transformers import AutoTokenizer

    from scripts.writing_token_curvature import verified_model

    if (not all(os.getenv(name) == '1' for name in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE'))
            or {name: importlib.metadata.version(name) for name in RUNTIME} != RUNTIME):
        raise ValueError('Use the pinned offline writing runtime')
    verified_model(snapshot)
    mx.set_default_device(mx.gpu)
    mx.set_cache_limit(256 * 1024 * 1024)
    model, _ = load(str(snapshot), tokenizer_config={'trust_remote_code': False})
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False)
    model.freeze()
    model.eval()
    nn.quantize(model, mode='affine', group_size=64, bits=4)
    model.apply(lambda value: value.astype(mx.float32) if value.dtype in (mx.float16, mx.bfloat16) else value)
    mx.eval(model.parameters())
    return mx, nn, model, tokenizer


def measure_batch(items, mx, nn, model):
    inputs = [prefix + target for prefix, target, _ in items]
    width = max(map(len, inputs))
    tokens = mx.array([list(ids) + [0] * (width - len(ids)) for ids in inputs])
    target_width = max(len(target) for _, target, _ in items)
    preceding, matching, labels, values = [], [], [], []
    for prefix, target, _ in items:
        low, high = len(prefix), len(prefix) + len(target)
        padding = [0] * (target_width - len(target))
        preceding.append(list(range(low - 1, high - 1)) + padding)
        matching.append(list(range(low, high)) + padding)
        labels.append(list(target) + padding)
        values.append([1.] * len(target) + padding)
    hidden = model.model(tokens)
    head = model.model.embed_tokens.as_linear if model.args.tie_word_embeddings else model.lm_head
    expected = mx.array(labels)
    mask = mx.array(values, dtype=mx.float32)

    def reduction(indices):
        selected = mx.take_along_axis(hidden, mx.array(indices)[..., None], axis=1)
        loss = nn.losses.cross_entropy(head(selected).astype(mx.float32), expected, reduction='none')
        return ((loss * mask).sum(axis=1) / mask.sum(axis=1)).tolist()

    next_token = reduction(preceding)
    repeat = reduction(matching) if items[0][2] else [None] * len(items)
    output = [(float(first), None if second is None else float(second)) for first, second in zip(next_token, repeat, strict=True)]
    if any(not math.isfinite(x) for a, b in output for x in (a, b) if x is not None):
        raise ValueError('Non-finite language model score')
    return output


def score(plan_path, snapshot, output, point_path=None):
    from scripts.audio_strong_traffic_view import network_free
    from scripts.writing_token_curvature import MODEL_ASSETS

    network_free()
    plan = read(plan_path)
    code = sha(Path(__file__).read_bytes())
    if point_path is not None:
        point = read(point_path)
        if (point['producer_sha256'] != code or point['policy'] != POLICY
                or set(point['calibration_groups']).intersection(case['group'] for case in plan['cases'])):
            raise ValueError('Assessment inputs must be independent of the pre-existing operating points')
    mx, nn, model, tokenizer = initialize(snapshot)
    spans = [focuses(row, tokenizer) for row in plan['cases']]
    unique = {job for rows in spans for item in rows for job in item.jobs}
    print(f'{len(plan["cases"])} cases, {sum(map(len, spans))} eligible sentences, {len(unique)} input views', flush=True)
    guard = {key: value for key, value in MODEL_ASSETS.items()}
    freeze = {'source_plan_sha256': sha(plan_path.read_bytes()), 'producer_sha256': code,
              'policy': POLICY, 'snapshot': snapshot.name, 'models_sha256': guard, 'runtime': RUNTIME, 'batch': BATCH,
              'point_sha256': sha(point_path.read_bytes()) if point_path is not None else None}
    write(output.with_suffix('.lock'), freeze)
    observations = {}
    all_sets = [sorted((job for job in unique if bool(job[2]) == reading), key=lambda item: (len(item[0]) + len(item[1]), item))
                for reading in (False, True)]
    for rows in all_sets:
        for group in batches(rows, BATCH):
            results = measure_batch(group, mx, nn, model)
            observations.update(zip(group, results, strict=True))
            if len(observations) % 50 < BATCH:
                print(f'{len(observations)}/{len(unique)} input views finished', flush=True)
    data = []
    for case, items in zip(plan['cases'], spans, strict=True):
        values = []
        prefix = list(accumulate((not char.isspace() for char in case['text']), initial=0))

        for item in items:
            (focus, repeat), (ref_focus, _), (successor, _), (ref_successor, _) = [observations[job] for job in item.jobs]
            values.append({'sentence': item.index, 'range': list(item.bounds), 'compact': [prefix[x] for x in item.bounds],
                           'ref_sentences': [item.left_reference, item.right_reference], 'tokens': len(item.jobs[0][1]),
                           'left_gain': ref_focus - focus, 'right_gain': ref_successor - successor,
                           'repeat_loss': repeat, 'next_token_loss': focus})
        data.append({key: case[key] for key in ('group', 'role', 'variant', 'sha256')}
                    | {'sentences': len(case['sentences']), 'truth': case['truth'],
                       'compact_truth': [[prefix[a], prefix[b]] for a, b in case['truth']], 'scores': values})
    write(output, {'freeze': freeze, 'observed': data, 'unique_inputs': len(unique)})


def relative(values, target):
    middle = median(values)
    scale = max(.15, 1.4826 * median(abs(x - middle) for x in values))
    return (target - middle) / scale


def candidates(case):
    rows = case['scores']
    if len(rows) < POLICY['minimum_complete_centers']:
        return {name: [] for name in ARMS}
    result = {name: [] for name in ARMS}
    for index, item in enumerate(rows):
        other = rows[:index] + rows[index + 1:]
        left, right = (-relative([row[field] for row in other], item[field]) for field in ('left_gain', 'right_gain'))
        repeat = relative([row['repeat_loss'] for row in other], item['repeat_loss'])
        next_token = -relative([row['next_token_loss'] for row in other], item['next_token_loss'])
        for arm, value in zip(ARMS, (left, right, min(left, right), repeat, next_token), strict=True):
            result[arm].append((value, item))
    return result


def fit(packet, output):
    groups = {name: {} for name in ARMS}
    for case in packet['observed']:
        if case['role'] != 'ordinary':
            raise ValueError('Only genuine original controls set the operating points')
        for arm, rows in candidates(case).items():
            if rows:
                maxima = groups[arm]
                maximum = max(value for value, _ in rows)
                maxima[case['group']] = max(maxima.get(case['group'], maximum), maximum)
    if min(map(len, groups.values())) < 8:
        raise ValueError('Fewer than eight independently scored genuine source groups')
    return {'arms': {arm: sorted(peak.values())[-2] for arm, peak in groups.items()},
            'calibration_groups': sorted({case['group'] for case in packet['observed']}),
            'maxima': groups, 'policy': POLICY, 'producer_sha256': packet['freeze']['producer_sha256'],
            'measured_sha256': output}


def overlap(a, b):
    common = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    return 2 * common >= (b[1] - b[0]) and 10 * common >= 3 * (max(a[1], b[1]) - min(a[0], b[0]))


def compare(packet, point):
    if (point['producer_sha256'] != packet['freeze']['producer_sha256'] or point['policy'] != POLICY
            or set(point['calibration_groups']).intersection(case['group'] for case in packet['observed'])):
        raise ValueError('Held data and calibration procedure are not independently bound')
    scored = []
    for row in packet['observed']:
        selected, before_point, reachable, selections = {}, {}, {}, {}
        for name, values in candidates(row).items():
            ranked = sorted(values, key=lambda pair: (-pair[0], pair[1]['sentence']))
            top = ranked[0] if ranked else None
            call = top[1] if top and top[0] > point['arms'][name] else None
            targets = row['compact_truth']
            selected[name] = [j for j, target in enumerate(targets) if call is not None and overlap(call['compact'], target)][:1]
            before_point[name] = [j for j, target in enumerate(targets) if top and overlap(top[1]['compact'], target)][:1]
            reachable[name] = [j for j, target in enumerate(targets) if any(overlap(hit['compact'], target) for _, hit in ranked)]
            selections[name] = {'range': call['range'], 'sentence': call['sentence'], 'compact': call['compact'],
                                'score': top[0]} if call is not None else None
        scored.append(row | {'matched': selected, 'top_before_point': before_point, 'reachable': reachable,
                              'selections': selections})
    return scored


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest='mode', required=True)
    for mode, names in (('score', ('plan', 'snapshot', 'output')), ('fit', ('measurements', 'output')),
                        ('compare', ('measurements', 'point', 'output'))):
        command = modes.add_parser(mode)
        for name in names:
            command.add_argument(name, type=Path)
        if mode == 'score':
            command.add_argument('--point', type=Path)
    args = parser.parse_args()
    if args.mode == 'score':
        score(args.plan, args.snapshot, args.output, args.point)
    elif args.mode == 'fit':
        write(args.output, fit(read(args.measurements), sha(args.measurements.read_bytes())))
    else:
        packet = read(args.measurements)
        if packet['freeze']['point_sha256'] != sha(args.point.read_bytes()):
            raise ValueError('Held scoring must follow the same frozen operating points')
        write(args.output, compare(packet, read(args.point)))
