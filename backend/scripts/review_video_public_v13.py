"""Lossless public packet for the completed video comparison and resource receipts.

No media, pretrained weights or machine paths are included. The packet preserves
original UTF-8 bytes, including failures; replay uses the separate NumPy reviewer.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import tempfile
from pathlib import Path

MEMBERS = {
    'candidate.jsonl', 'object-release-repeat-1/baseline.jsonl',
    'object-release-repeat-1/summary.json', 'object-release-repeat-1/profile.json',
    'object-release-repeat-1/child-result.json', 'object-release-repeat-1/collection.jsonl',
    'object-release-repeat-1/native-release-1/native-review.json',
    'object-release-repeat-1/native-release-1/profile.json',
    'object-release-repeat-1/native-release-1/child-result.json',
}
MAX_BYTES = 50 * 1024**2
SCHEMA = 'chipmunk.video-public-development-packet/13'


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def validate(body):
    if body['schema'] != SCHEMA or body['serving_admitted'] is not False or set(body['members']) != MEMBERS:
        raise ValueError('Complete fixed public member set required')
    for member in body['members'].values():
        raw = member['utf8'].encode('utf-8')
        if member['bytes'] != len(raw) or member['sha256'] != digest(raw):
            raise ValueError('Lossless original member bytes required')
        if any(marker in raw for marker in (b'/Users/', b'/private/', b'/home/', b'file://')):
            raise ValueError('Private machine path found; preserve original privately and review before publication')
    root = 'object-release-repeat-1/'
    values = {name: json.loads(body['members'][root + name]['utf8'])
              for name in ('profile.json', 'child-result.json', 'summary.json')}
    child, profile, summary = (values[k] for k in ('child-result.json', 'profile.json', 'summary.json'))
    if (profile['guard'] is not None or profile['child_exit_code'] != 0 or child['completed'] is not True
            or child['planned_inputs'] != 288 or child['collection_attempts'] != 284
            or child['prior_inputs_reproduced'] != 84 or child['max_prior_raw_difference'] > 1e-5):
        raise ValueError('Successful complete guarded baseline required')
    for field, name in [('baseline_sha256', root + 'baseline.jsonl'), ('collection_sha256', root + 'collection.jsonl')]:
        if child[field] != body['members'][name]['sha256']:
            raise ValueError('Baseline/collection lineage differs')
    if (summary['baseline_sha256'] != child['baseline_sha256']
            or summary['candidate_sha256'] != body['members']['candidate.jsonl']['sha256']):
        raise ValueError('Original comparison lineage differs')
    native_root = root + 'native-release-1/'
    native_child = json.loads(body['members'][native_root + 'child-result.json']['utf8'])
    native_profile = json.loads(body['members'][native_root + 'profile.json']['utf8'])
    native = json.loads(body['members'][native_root + 'native-review.json']['utf8'])
    if (native_profile['guard'] is not None or native_profile['child_exit_code'] != 0
            or native_child['completed'] is not True or len(native['cases']) != 16
            or native_child['native_sha256'] != body['members'][native_root + 'native-review.json']['sha256']
            or native_child['native_passed'] is not native['gates']['native_correspondence']
            or native['summary_sha256'] != body['members'][root + 'summary.json']['sha256']):
        raise ValueError('Complete native receipt required; failure outcomes are retained')
    return body


def export(source, output):
    members = {}
    for name in sorted(MEMBERS):
        raw = (source / name).read_bytes()
        members[name] = {'bytes': len(raw), 'sha256': digest(raw), 'utf8': raw.decode('utf-8')}
    body = validate({'schema': SCHEMA, 'members': members, 'serving_admitted': False,
        'scope': 'Consumed development comparison. Original responses, fitted coefficients, summaries and resource/native receipts, without media or weights. Exact original UTF-8 members; no outcome redaction. Not fresh confirmation, source truth verification or outside human review.'})
    raw = (json.dumps(body, ensure_ascii=False, allow_nan=False, separators=(',', ':')) + '\n').encode()
    if len(raw) > MAX_BYTES:
        raise ValueError('Packet exceeds declared bound')
    with output.open('xb') as stream:
        stream.write(gzip.compress(raw, mtime=0))
    print(json.dumps({'members': len(members), 'expanded_bytes': len(raw), 'packet_sha256': digest(output.read_bytes())}))


def read(path):
    with gzip.open(path, 'rb') as stream:
        raw = stream.read(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES:
        raise ValueError('Oversized public packet')
    return validate(json.loads(raw))


def review(freeze, packet, output):
    from scripts.review_video_demamba_v13 import review as arithmetic

    body = read(packet)
    with tempfile.TemporaryDirectory(prefix='video-public-review-') as scratch:
        directory = Path(scratch)
        for name, member in body['members'].items():
            target = directory / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(member['utf8'].encode('utf-8'))
        root = directory / 'object-release-repeat-1'
        arithmetic(freeze, root / 'baseline.jsonl', directory / 'candidate.jsonl', root / 'summary.json',
                   root / 'native-release-1/native-review.json', output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('export', 'review'))
    parser.add_argument('input', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--freeze', type=Path)
    args = parser.parse_args()
    if args.mode == 'export':
        export(args.input, args.output)
    else:
        if args.freeze is None:
            parser.error('The original published prospective freeze is required')
        review(args.freeze, args.input, args.output)
