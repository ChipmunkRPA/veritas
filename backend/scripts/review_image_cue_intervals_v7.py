"""Separate component-count bootstrap arithmetic for the public image packet."""
import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def review(path, output):
    packet = json.loads(path.read_text())
    results = {}
    point = packet['gate']['raw_mean_logit']
    for fmt in ('native', 'jpeg75'):
        for label, metric in ((1, 'ai_gain_interval'), (0, 'false_alarm_change_interval')):
            grouped = defaultdict(list)
            for row in packet['items']:
                if row['role'] != 'development' or row['format'] != fmt or row['label'] != label:
                    continue
                baseline = row['baseline_status'] == 'done' and row['baseline_statistical_direction'] == 'ai'
                positive = row['candidate_status'] == 'ok' and row['candidate_mean_logit'] > point
                grouped[row['group']].append(int(positive and not baseline))
            names = sorted(grouped)
            gains = np.asarray([sum(grouped[name]) for name in names], np.int64)
            sizes = np.asarray([len(grouped[name]) for name in names], np.int64)
            generator = np.random.Generator(np.random.PCG64(20261107))
            draws = generator.integers(0, len(names), size=2000 * len(names)).reshape(2000, len(names))
            estimates = []
            for draw in draws:
                multiplicities = np.bincount(draw, minlength=len(names))
                estimates.append(float(np.dot(multiplicities, gains) / np.dot(multiplicities, sizes)))
            measured = {'planned_inputs': int(sizes.sum()), 'source_components': len(names),
                'paired_rate_difference': float(gains.sum() / sizes.sum()),
                'component_bootstrap_95_percentile': np.quantile(estimates, [.025, .975]).tolist()}
            if measured != packet['expected']['by_format'][fmt][metric]:
                raise ValueError('Separate count-weighted bootstrap differs from original published interval')
            results[fmt + ':' + metric] = measured
    result = {'passed': True, 'intervals_compared_exact': 4, 'replicates_each': 2000,
        'seed': 20261107, 'numpy': np.__version__, 'packet_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'values': results, 'scope': 'Separate NumPy count-weighted implementation of the declared paired component bootstrap. No pretrained inference, source pixel audit or outside human review.'}
    with output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('packet', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    review(args.packet, args.output)
