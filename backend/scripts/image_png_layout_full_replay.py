# SPDX-License-Identifier: Apache-2.0
"""Reproduce the baseline and assemble the PNG layout comparison receipt."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts import image_png_layout_guard_transfer as study
from scripts.review_image_png_layout_guard import audit


def full(directory):
    from scripts import run_image_smaug_v7, run_image_whole_fresh

    def held(run):
        lock, rows = study.prepared(run)
        return lock, [row for row in rows if 'prior_score' not in row]

    run_image_smaug_v7.ROOT = directory / 'checkout'
    run_image_whole_fresh.prepared = held
    run_image_whole_fresh.freeze_baseline(directory)
    run_image_whole_fresh.baseline(directory)


def export(directory):
    lock_path = directory / 'baseline-lock.json'
    lock = study.read(lock_path)
    for field, value in lock['baseline']['settings'].items():
        if isinstance(value, str) and value.startswith('/'):
            lock['baseline']['settings'][field] = '<local-path>'
    rows = []
    for response in map(json.loads, (directory / 'baseline.jsonl').read_text().splitlines()):
        name = response['id']
        packet = directory / 'baseline-results' / f'{name}.json'
        study.require(study.sha(packet) == response['retained_packet_sha256'], 'Retained Full response differs')
        rows.append({'id': name, 'sha256': response['input_sha256'], 'status': response['status'],
                     'direction': response['interpretation']['customer_direction'],
                     'result_sha256': response['canonical_result_sha256'], 'packet_sha256': study.sha(packet)})
    payload = study.read(directory / 'receipt.json')
    payload['baseline'] = {'lock_public': lock, 'lock_public_sha256': study.digest(study.encoded(lock)),
                           'lock_sha256': study.sha(lock_path), 'outcomes': rows}
    payload['summary'] = audit(payload)
    study.write(directory / 'full-receipt.json', payload)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('full', 'export'))
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    {'full': full, 'export': export}[args.phase](args.run.resolve())
