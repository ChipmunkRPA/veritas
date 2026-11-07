"""Apply the separate v16 vector/window arithmetic to all fresh v17 cases.

Private-only review. No image, real-face feature or per-frame digest export.
"""
import argparse
import json
from pathlib import Path

from scripts.audit_sface_vectors_v16 import TOLERANCE, digest, replay


def audit(trial, output):
    items = json.loads((trial / 'private-confirmation-panel.json').read_text())['items']
    records = [json.loads(line) for line in (trial / 'candidate.jsonl').read_text().splitlines()]
    by_id = {r['id']: r for r in records}
    if len(items) != 120 or len(records) != 120 or set(by_id) != {r['id'] for r in items}:
        raise ValueError('Exact120 fresh source-bound cases required')
    frames, retained, maximum = 0, 0, 0.
    for item in items:
        record = by_id[item['id']]
        packet = trial / 'private-features-confirmation' / (item['id'] + '.json')
        if (record['status'] == 'failed' or digest(Path(item['source_path'])) != item['sha256']
                or digest(packet) != record['private_feature_packet_sha256']
                or record['source_sha256'] != item['sha256']):
            raise ValueError('Frozen source/private packet binding changed')
        value = json.loads(packet.read_text())
        if value['source_sha256'] != item['sha256']:
            raise ValueError('Private vectors lost exact source association')
        windows, count = replay(item, value)
        actual = record['candidate_windows']
        if len(actual) != min(12, len(windows)):
            raise ValueError('Incomplete full allowed-window search')
        for window in actual:
            key = (window['track_id'], window['start_s'], window['end_s'], tuple(window['centers']))
            if key not in windows:
                raise ValueError('Window crossed source/track/clock boundaries')
            maximum = max(maximum, *(abs(window[k] - windows[key][k]) for k in
                ('response', 'middle_to_before', 'middle_to_after', 'flank_change')))
        if maximum > TOLERANCE:
            raise ValueError('Separate float64 window arithmetic differs')
        best = sorted((w['response'] for w in windows.values()), reverse=True)
        if any(abs(w['response'] - v) > TOLERANCE for w, v in zip(actual, best)):
            raise ValueError('Retained maxima differ from exhaustive search')
        frames += count
        retained += len(actual)
    result = {'schema': 'chipmunk.sface-private-vector-arithmetic/17', 'passed': True,
        'cases': len(items), 'native_vector_bytes_verified': frames,
        'retained_windows_verified': retained, 'all_allowed_source_track_cadence_windows_searched': True,
        'max_separate_component_difference': maximum, 'float64_arithmetic_tolerance': TOLERANCE,
        'native_profile_passed_cases': sum(bool(r.get('native_alignment_all_exact')) for r in records),
        'raw_vectors_exported': False, 'native_neural_inference_repeated': False, 'outside_human_review': False,
        'scope': 'Separate native-byte/source/track and prefix-sum/cosine review using v16 private arithmetic. No model rerun; no real-face vectors or per-frame biometric hashes in this aggregate.'}
    with output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trial', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    audit(args.trial, args.output)
