"""Audit unused human-only study controls before classifier inference.

Private output retains the source inventory. Public output contains aggregates
only. No text, author identities, prompts or per-document labels are published.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import zipfile
from collections import Counter
from pathlib import Path

ARCHIVE_SHA = '65c7d38a96295453933a5620b46fc1baa54a54651b93dd7bed34d8f17262c641'
RAW_SHA = 'c799c1fe9dfce294dfda8c47fb57e77a357df6ec8e0dbca62b206b9dded28628'
SPAN_SHA = '0641d20f9326522061bee792cad5f8963faf26bf37268e491f901cb7c2ce5cfb'
PRIOR_SHA = 'e10769c188e69889c54f1116c5a4f8f03cbb58f4194a19a3a1ea9e969a92d28c'
RULES = {'selection': 'Every unused-author human_only=True row, no response-driven selection.',
         'history': 'Source order, publisher Python codepoint index semantics, only User insert/delete/retain; strict bounds, finite nondecreasing timestamps; complete exact final span text. Never clamp, repair, reorder or invent characters.',
         'overlap': 'All680 prior full texts, normalized exact match and five-word shingle comparisons; shared>=20 and (Jaccard>=0.5 or smaller-set containment>=0.8). Corpus source words and row identities remain private.',
         'interpretation': 'Publisher-controlled human-only condition and edit-log consistency, not verified absence of all off-platform assistance. Same task pool; no topic-disjoint or pretrained-training-disjoint claim.',
         'unresolved': 'Keep every planned control in inventory. Invalid history or prior near-duplicate is explicit unsupported source, never a human verdict, omitted record or a detector false alarm.',
         'classifier_forwards': 0, 'fitting_performed': False, 'serving_admitted': False}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def file_sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, body):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(body, stream, indent=2, allow_nan=False)
        stream.write('\n')


def tokens(text):
    return re.findall(r'\w+', text.casefold())


def normalized(text):
    return sha(' '.join(tokens(text)).encode())


def shingles(text):
    words = tokens(text)
    return {tuple(words[i:i + 5]) for i in range(len(words) - 4)}


def near(first, second):
    shared = len(first & second)
    return shared >= 20 and (shared / len(first | second) >= .5 or shared / min(len(first), len(second)) >= .8)


def replay(events):
    """Strict source-order replay under the publisher's codepoint semantics."""
    import math

    text, previous_time, identifiers = '', -math.inf, set()
    counts = Counter()
    for index, event in enumerate(events):
        if not isinstance(event, dict) or 'operationType' not in event or event.get('user') != 'User':
            raise ValueError(f'Non-user edit or AI interaction at event{index}')
        identifier, timestamp = event['id'], event['createdAt']
        if (type(identifier) is not int or identifier in identifiers or type(timestamp) not in (int, float)
                or not math.isfinite(timestamp) or timestamp < previous_time):
            raise ValueError(f'Invalid event identity/source clock at event{index}')
        identifiers.add(identifier)
        previous_time = timestamp
        start, size, operation = event['offset'], event['span'], event['operationType']
        if type(start) is not int or type(size) is not int or not 0 <= start <= len(text) or size < 0:
            raise ValueError(f'Out-of-bounds edit at event{index}')
        if operation == 'insert':
            if not isinstance(event['text'], str):
                raise ValueError(f'Missing inserted characters at event{index}')
            text = text[:start] + event['text'] + text[start:]
        elif operation == 'delete':
            if start + size > len(text):
                raise ValueError(f'Out-of-bounds delete at event{index}')
            text = text[:start] + text[start + size:]
        elif operation != 'retain':
            raise ValueError(f'Unknown edit operation at event{index}')
        elif start + size > len(text):
            raise ValueError(f'Out-of-bounds formatting retain at event{index}')
        counts[operation] += 1
    return text, dict(counts)


def audit(archive, spans, previous, private, public):
    import pyarrow.parquet as pq

    if private.exists() or public.exists():
        raise ValueError('New private inventory and public receipt required')
    for path, expected in ((archive, ARCHIVE_SHA), (spans, SPAN_SHA), (previous, PRIOR_SHA)):
        if file_sha(path) != expected:
            raise ValueError('Original source/archive/prior manifest changed')
    with zipfile.ZipFile(archive) as source:
        raw = source.read('original/test-00000-of-00001.parquet')
    if sha(raw) != RAW_SHA:
        raise ValueError('Exact institutional raw-history member required')
    histories = pq.read_table(io.BytesIO(raw)).to_pylist()
    annotations = pq.read_table(spans).to_pylist()
    prior = json.loads(previous.read_text())['cases']
    if len(histories) != len(annotations) or len(annotations) != 362 or len(prior) != 680:
        raise ValueError('Every original source/prior case required')
    for history, source in zip(histories, annotations, strict=True):
        if history['metadata'] != source['metadata']:
            raise ValueError('Raw history and span metadata differ')
    used = set()
    for case in prior:
        if case['label'] == 'mixed':
            source = annotations[int(case['id'].split(':')[1])]
            text = ''.join(s['text'] for s in source['data'])
            if text != case['text'] or sha(text.encode()) != case['text_sha256']:
                raise ValueError('Consumed author/source binding changed')
            used.add(source['metadata']['author'])
    old = [(normalized(c['text']), shingles(c['text'])) for c in prior]
    inventory, author_groups = [], set()
    for index, (history, source) in enumerate(zip(histories, annotations, strict=True)):
        metadata = source['metadata']
        if metadata['author'] in used or metadata['human_only'] is not True:
            continue
        text = ''.join(s['text'] for s in source['data'])
        norm, words = normalized(text), shingles(text)
        record = {'source_row': index, 'text_sha256': sha(text.encode()), 'task': metadata['task'],
                  'setting': metadata['setting'], 'author_group': sha(('writing-control-author:' + metadata['author']).encode()),
                  'history_events': len(history['data']), 'prior_exact_matches': sum(norm == h for h, _ in old),
                  'prior_near_matches': sum(near(words, s) for _, s in old), 'history_status': 'unsupported'}
        author_groups.add(record['author_group'])
        try:
            if not text.strip() or any(s['author'] != 'User' or s['queries'] for s in source['data']):
                raise ValueError('Human-only condition/span/query mismatch')
            observed, counts = replay([json.loads(e) for e in history['data']])
            if observed != text:
                raise ValueError('Strict history replay does not equal complete span text')
            record.update(history_status='exact', edit_counts=counts)
        except (ValueError, KeyError, TypeError) as error:
            record.update(reason=str(error), error_type=type(error).__name__)
        record['source_audit_eligible'] = record['history_status'] == 'exact' and not record['prior_exact_matches'] and not record['prior_near_matches']
        inventory.append(record)
    if len(used) != 40 or len(inventory) != 53 or len(author_groups) != 49:
        raise ValueError('Original unused human-control inventory changed')
    body = {'schema': 'chipmunk.writing-controls-private-source-audit/4', 'rules': RULES,
            'archive_sha256': ARCHIVE_SHA, 'raw_parquet_sha256': RAW_SHA, 'spans_sha256': SPAN_SHA,
            'prior_manifest_sha256': PRIOR_SHA, 'code_sha256': file_sha(Path(__file__)), 'records': inventory}
    write(private, body)
    summary = {'schema': 'chipmunk.writing-controls-public-source-audit/4', 'rules': RULES,
        'source_rows': 362, 'prior_documents': 680, 'consumed_authors': 40, 'unused_human_controls': len(inventory),
        'unused_human_authors': len(author_groups), 'history_exact': sum(r['history_status'] == 'exact' for r in inventory),
        'history_unsupported': sum(r['history_status'] != 'exact' for r in inventory),
        'controls_with_prior_exact_match': sum(bool(r['prior_exact_matches']) for r in inventory),
        'controls_with_prior_near_match': sum(bool(r['prior_near_matches']) for r in inventory),
        'source_audit_eligible': sum(r['source_audit_eligible'] for r in inventory),
        'tasks': dict(Counter(r['task'] for r in inventory)), 'settings': dict(Counter(r['setting'] for r in inventory)),
        'source_artifacts_sha256': {k: v for k, v in body.items() if k.endswith('_sha256')},
        'private_inventory_sha256': file_sha(private),
        'data_terms': 'Institutional CC-BY-SA4 metadata and bundled GPL2 prose conflict retained; source wording, author identities, edit logs, queries and per-document annotations remain private. No new training/redistribution clearance inferred.',
        'outside_human_reviewers': 0}
    write(public, summary)
    print(json.dumps({k: summary[k] for k in ('unused_human_controls', 'unused_human_authors', 'history_exact', 'history_unsupported', 'controls_with_prior_near_match', 'source_audit_eligible')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('archive', 'spans', 'previous', 'private', 'public'):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    audit(args.archive, args.spans, args.previous, args.private, args.public)
