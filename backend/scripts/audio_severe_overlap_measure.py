# SPDX-License-Identifier: Apache-2.0
"""Evaluate time-aligned two-speaker estimates against private matched stems."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.audio_severe_overlap_separation import read_pcm, sha
from scripts.review_audio_severe_overlap_separation import RULE, summary


def power(signal):
    return float(np.dot(signal, signal))


def ratio(first, second):
    return float(10 * np.log10(max(first, 1e-14) / max(second, 1e-14)))


def si_sdr(estimate, source):
    estimate = estimate.astype(np.float64) - estimate.mean(dtype=np.float64)
    source = source.astype(np.float64) - source.mean(dtype=np.float64)
    if power(source) <= 1e-12:
        return None
    projected = source * (np.dot(source, estimate) / power(source))
    return ratio(power(projected), power(estimate - projected))


def assignment(tracks, foreground, interference):
    def total(index):
        values = si_sdr(tracks[index], foreground), si_sdr(tracks[1 - index], interference)
        return None if None in values else sum(values)

    orders = total(0), total(1)
    if None in orders:
        return None, None
    return int(orders[1] > orders[0]), abs(orders[1] - orders[0])


def target_components(estimate, foreground, interference):
    originals = np.column_stack((foreground, interference)).astype(np.float64)
    originals -= originals.mean(axis=0)
    level = np.linalg.norm(originals, axis=0)
    if min(level) <= 1e-8:
        return None, None
    frame = estimate.astype(np.float64) - estimate.mean(dtype=np.float64)
    gain, _, rank, _ = np.linalg.lstsq(originals / level, frame, rcond=1e-5)
    if rank != 2:
        return None, None
    target = originals[:, 0] * (gain[0] / level[0])
    return ratio(power(target), power(originals[:, 0])), ratio(power(target), power(frame - target))


def read_stem(pin, size):
    path = Path(pin['path'])
    if sha(path) != pin['sha256'] or pin['samples'] != size:
        raise ValueError('Stem checksum or clock differs from the source plan')
    result = np.load(path, allow_pickle=False)
    if result.shape != (size,) or result.dtype != '<f4' or not np.isfinite(result).all():
        raise ValueError('Expected one finite normalized float32 stem')
    return result


def reports(row, mixture, tracks, foreground, interference):
    from scripts.audio_overlap_reconstruction_views import mixture_phase

    views = {'direct': tracks, 'mixture_phase': mixture_phase(mixture, tracks)}
    report = {}
    intervals = row['expected_edits']
    for name, separated in views.items():
        if foreground is None:
            denominator = power(mixture.astype(np.float64))
            quieter = min(power(channel.astype(np.float64)) for channel in separated)
            report[name] = {'second_output_ratio_db': ratio(quieter, denominator)}
            continue
        untouched = np.ones(len(mixture), dtype=bool)
        for start, end in intervals:
            untouched[start:end] = False
        fixed, margin = (assignment(separated[:, untouched], foreground[untouched], interference[untouched])
                         if untouched.sum() >= RULE['minimum_assignment_samples'] else (None, None))
        detail = []
        if fixed is not None:
            # A fresh choice within an edit would leak its desired identity into the
            # measurements. Compare it only to count local disagreement.
            for start, end in intervals or [(0, len(mixture))]:
                window = slice(start, end)
                truth, other = foreground[window], interference[window]
                local, local_gap = assignment(separated[:, window], truth, other)
                baseline = si_sdr(mixture[window], truth)
                score = si_sdr(separated[fixed, window], truth)
                gain, output = target_components(separated[fixed, window], truth, other)
                detail.append({'start_sample': start, 'end_sample': end,
                               'input_ratio_db': ratio(power(truth.astype(np.float64)), power(other.astype(np.float64))),
                               'scale_invariant_gain_db': score - baseline if None not in (score, baseline) else None,
                               'target_gain_db': gain, 'target_vs_residual_db': output,
                               'assignment_disagrees': local != fixed if local is not None else None,
                               'local_assignment_margin_db': local_gap})
        report[name] = {'assignment_margin_db': margin, 'scored_ranges': detail}
    norm = np.linalg.norm(mixture.astype(np.float64))
    residual = np.linalg.norm(tracks.astype(np.float64).sum(axis=0) - mixture)
    return report, (float(residual / norm) if norm > 0 else 0.)


def run(manifest, outputs, destination):
    source = json.loads(manifest.read_bytes())
    measurements = outputs / 'measurements.jsonl'
    with measurements.open() as stream:
        header = json.loads(next(stream))
        ordered = [json.loads(line) for line in stream]
    if header['panel_sha256'] != sha(manifest) or len(ordered) != header['total'] or len(ordered) != len(source['rows']):
        raise ValueError('The separation inventory is incomplete or comes from another panel')
    rows = []
    for row, item in zip(source['rows'], ordered, strict=True):
        if (int(row['group']), row['role'], row['decoded_pcm']['sha256'], row['samples']) != (
                item['group'], item['role'], item['input_sha256'], item['samples']):
            raise ValueError('An output differs from the frozen input clock or order')
        minimum = {key: item.get(key) for key in ('group', 'role', 'status', 'exclusive_activity_frames', 'seconds')}
        minimum.update(family=row.get('family'), expected_edits=len(row['expected_edits']))
        if item['status'] == 'scored':
            path = outputs / f"{item['group']:03d}-{item['role']}.npy"
            if sha(path) != item['track_sha256']:
                raise ValueError('A derived track hash differs from local measurements')
            mixture = read_pcm(row)
            tracks = np.load(path, allow_pickle=False)
            target = read_stem(row['target'], len(mixture)) if row['target'] else None
            interference = read_stem(row['interference'], len(mixture))
            views, reconstruction = reports(row, mixture, tracks, target, interference)
            minimum.update(views=views, relative_reconstruction_error=reconstruction)
        rows.append(minimum)
    receipt = {'schema': 'chipmunk.severe-overlap-acoustic/1', 'method': RULE, 'upstream': header,
               'measurement_journal_sha256': sha(measurements), 'reader_sha256': sha(Path(__file__)),
               'rows': rows, 'summary': summary(rows)}
    with destination.open('x') as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    print(json.dumps(receipt['summary'], indent=2, sort_keys=True))


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    for arg in ('manifest', 'outputs', 'destination'):
        cli.add_argument(arg, type=Path)
    args = cli.parse_args()
    run(args.manifest, args.outputs, args.destination)
