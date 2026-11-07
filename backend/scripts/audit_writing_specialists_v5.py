"""Independent standard-library source/position audit; exports no source text."""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import io
import json
from collections import Counter
from pathlib import Path


def sha(data):
    return hashlib.sha256(data).hexdigest()


def verify(originals, plan):
    seen, positions = {}, 0
    for group in plan['groups']:
        for source in group['sources']:
            original = originals[source['row']]
            text = original['Answer']
            if (not all(original[k].isascii() for k in ('Question', 'Answer', 'Source'))
                    or text != source['text'] or sha(text.encode()) != source['text_sha256']
                    or sha(original['Question'].encode()) != source['question_sha256']
                    or original['Source'] != source['generator']
                    or source['label'] != int(original['Source'] != 'Human written')):
                raise ValueError('Original CSV row/text/truth binding failed')
            for piece in source['passages']:
                first, last = piece['start'], piece['end']
                if (type(first) is not int or type(last) is not int or not 0 <= first < last <= len(text)
                        or text[first:last] != piece['text'] or sha(piece['text'].encode()) != piece['sha256']):
                    raise ValueError('Original Unicode source slice binding failed')
                key = ''.join(c for c in piece['text'].casefold() if c.isalnum())
                if key in seen:
                    raise ValueError('Repeated selected normalized passage/source group')
                seen[key] = group['role']
                positions += 1
    return positions


def audit(source, plan_path, output):
    raw, plan_raw = source.read_bytes(), plan_path.read_bytes()
    plan = json.loads(plan_raw)
    if sha(raw) != plan['source_sha256'] or len(raw) != 121656267:
        raise ValueError('Exact original publisher bytes required')
    originals = list(csv.DictReader(io.StringIO(raw.decode('latin1'), newline='')))
    positions = verify(originals, plan)
    refusals = {}
    for name in ('wrong_source_row', 'wrong_original_span', 'copied_file_truth'):
        bad = copy.deepcopy(plan)
        item = bad['groups'][0]['sources'][0]
        if name == 'wrong_source_row':
            item['row'] = (item['row'] + 1) % len(originals)
        elif name == 'wrong_original_span':
            item['passages'][0]['start'] += 1
        else:
            item['label'] = 1 - item['label']
        try:
            verify(originals, bad)
        except ValueError:
            refusals[name] = True
        else:
            raise ValueError('Source/position/truth tampering was not refused')
    report = {'schema': 'chipmunk.writing-specialists-original-audit/5', 'source_sha256': sha(raw),
        'plan_sha256': sha(plan_raw), 'code_sha256': sha(Path(__file__).read_bytes()),
        'source_records': len(originals), 'source_groups': len(plan['groups']),
        'actual_original_source_slices_verified': positions,
        'roles': dict(Counter(g['role'] for g in plan['groups'])), 'adverse_bindings_refused': refusals,
        'source_text_published': False, 'participant_independence_established': False,
        'external_training_overlap_excluded': False, 'new_model_forwards': 0, 'serving_admitted': False}
    with output.open('x') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('plan', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    audit(args.source, args.plan, args.output)
