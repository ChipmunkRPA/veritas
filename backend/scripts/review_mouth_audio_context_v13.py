"""Separate complete audio decode, scalar correspondence and outcome audit."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import subprocess
from pathlib import Path

import cv2
import numpy as np

from scripts import run_mouth_audio_context_v13 as run


def close(a, b):
    if not math.isfinite(a) or not math.isclose(a, b, rel_tol=0, abs_tol=1e-10):
        raise ValueError('Independent finite scalar arithmetic differs')


def correlate(x, y):
    mx, my = statistics.mean(x), statistics.mean(y)
    a, b = [v - mx for v in x], [v - my for v in y]
    norm = math.sqrt(math.fsum(v * v for v in a)) * math.sqrt(math.fsum(v * v for v in b))
    return (max(-1., min(1., math.fsum(v * w for v, w in zip(a, b, strict=True)) / norm)), True) if norm > 1e-12 else (0., False)


def features(window, audio, aperture):
    c, n = window['centers'], len(audio)
    middle = [i for i in range(8, n - 8) if c[2] - 1 <= i < c[-3] + 2]
    support = [i for i in range(8, n - 8) if c[0] - 1 <= i < c[-1] + 2]
    result = []
    for indices in (middle, support):
        run.require(len(indices) >= 4, 'Independent common support is insufficient')
        values = [correlate([aperture[i] for i in indices], [audio[i + lag] for i in indices]) for lag in (-8, -4, 0, 4, 8)]
        correlations = [v[0] for v in values]
        best = sorted(range(5), key=lambda j: (-correlations[j], abs((-8, -4, 0, 4, 8)[j]), (-8, -4, 0, 4, 8)[j]))[0]
        result += correlations + [(-8, -4, 0, 4, 8)[best] / 8, correlations[best] - correlations[2],
                  statistics.pstdev(audio[i] for i in indices), statistics.pstdev(aperture[i] for i in indices),
                  sum(v[1] for v in values) / 5]
    return result, [middle, support]


def margin(values, head):
    result = head['base_margin']
    for tree in head['trees']:
        index, seen = 0, set()
        while True:
            run.require(type(index) is int and 0 <= index < len(tree) and index not in seen, 'Invalid independent tree route')
            seen.add(index)
            node = tree[index]
            if node['is_leaf']:
                result += node['value']
                break
            index = node['left'] if values[node['feature_idx']] <= node['num_threshold'] else node['right']
    return result


def outcome(row, windows, point):
    selected = windows[0] if windows and windows[0]['response'] >= point else None
    truth_range = row.get('edit_time_range_s')
    overlap = 0.
    if selected and truth_range:
        overlap = max(0., min(selected['end_s'], truth_range[1]) - max(selected['start_s'], truth_range[0])) / (
            max(selected['end_s'], truth_range[1]) - min(selected['start_s'], truth_range[0]))
    spatial = 0.
    if selected and row['label']:
        box = selected['region']
        record = next((r for r in row.get('masks', []) if r['frame_index'] == box['frame_index']), None)
        if record:
            path = Path(row['mask_root']) / record['path']
            run.require(run.sha(path) == record['sha256'], 'Independent source mask changed')
            truth = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE) > 0
            region = np.zeros((box['height'], box['width']), np.uint8)
            cv2.fillConvexPoly(region, np.rint(box['polygon']).astype(np.int32), 1)
            total = int(np.count_nonzero(truth | (region > 0)))
            spatial = int(np.count_nonzero(truth & (region > 0))) / total if total else 0.
    return {**{k: row[k] for k in ('id', 'group', 'condition', 'label')}, 'assessed': bool(windows),
            'flagged': selected is not None, 'temporal_iou': overlap, 'localized': bool(row['label'] and overlap >= .2),
            'spatial_iou': spatial}


def summary(rows):
    ai, ordinary = [r for r in rows if r['label']], [r for r in rows if not r['label']]
    return {'planned': len(rows), 'ai_edits': len(ai), 'non_ai_controls': len(ordinary),
            'assessed': sum(r['assessed'] for r in rows), 'localized_ai': sum(r['localized'] for r in ai),
            'ai_without_flag': sum(not r['flagged'] for r in ai), 'non_ai_flags': sum(r['flagged'] for r in ordinary),
            'mean_ai_temporal_iou': sum(r['temporal_iou'] for r in ai) / len(ai) if ai else None,
            'mean_ai_spatial_iou': sum(r['spatial_iou'] for r in ai) / len(ai) if ai else None}


def gates(original, proposed, sources):
    a, b = summary(original), summary(proposed)
    first = [r for r, source in zip(proposed, sources, strict=True) if source['family'] == 'liveportrait']
    second = [r for r, source in zip(proposed, sources, strict=True) if source['family'] == 'musetalk']
    ordinary = {r['condition'] for r in original if not r['label']}
    return {'six_net_additional_locations': b['localized_ai'] >= a['localized_ai'] + 6,
            'first_family_six_locations': sum(r['localized'] for r in first) >= 6,
            'both_first_family_conditions': all(sum(r['localized'] for r in first if r['condition'] == c) >= 2
                                                for c in ('ai-redraw', 'ai-retime')),
            'retained_second_family': sum(r['localized'] for r in second) >= 13,
            'at_most_one_lost_location': sum(a['localized'] and not b['localized'] for a, b in zip(original, proposed, strict=True)) <= 1,
            'total_ordinary_flags': b['non_ai_flags'] <= a['non_ai_flags'] + 1,
            'condition_ordinary_flags': all(sum(r['flagged'] for r in proposed if r['condition'] == c)
                                          <= sum(r['flagged'] for r in original if r['condition'] == c) + 1 for c in ordinary),
            'spatial_iou_gain': b['mean_ai_spatial_iou'] >= a['mean_ai_spatial_iou'] + .02,
            'retained_temporal_iou': b['mean_ai_temporal_iou'] >= a['mean_ai_temporal_iou'],
            'retained_coverage': all(not a['assessed'] or b['assessed'] for a, b in zip(original, proposed, strict=True)),
            'complete_replays': len(sources) == 88 and all(r['status'] == 'ok' and r['native_input_replay']
                and all(h['baseline_replay_parity'] and h['prior_native_and_margin_parity'] for h in r['heads'].values()) for r in sources)}


def review(trial, output):
    parent, packet, rows = run.check(trial)
    run.check_fit(trial)
    records, joined, samples, frames, windows = {}, {}, 0, 0, 0
    for stage in run.COUNTS:
        joined[stage] = run.joined(trial, stage, rows[stage])
        records[stage] = [json.loads(line) for line in (trial / (stage + '.jsonl')).read_text().splitlines()]
        for source, record in zip(rows[stage], records[stage], strict=True):
            path = Path(source['mask_root']) / source['path']
            run.require(run.sha(path) == source['sha256'] == record['proof']['input_sha256'], 'Independent exact source file changed')
            encoded = subprocess.check_output(['ffmpeg', '-v', 'error', '-nostdin', '-i', str(path), '-map', '0:a:0',
                '-vn', '-ar', '16000', '-ac', '1', '-f', 's16le', 'pipe:1'], timeout=30)
            pcm = np.frombuffer(encoded, '<i2').astype(np.float32) / 32768
            run.require(len(pcm) == source['audio_samples'] == record['proof']['samples'] == 640 * source['frames']
                        and hashlib.sha256(pcm.astype('<f4').tobytes()).hexdigest() == source['pcm_sha256'] == record['proof']['pcm_sha256'],
                        'Independent full signed16 PCM differs')
            # ffprobe is separate from the primary frame-table adapter.
            probe = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-show_streams', '-show_frames',
                '-of', 'json', str(path)], timeout=30))
            stream = next(r for r in probe['streams'] if r['codec_type'] == 'video')
            actual = [r for r in probe['frames'] if r['media_type'] == 'video']
            proof = record['proof']
            run.require([int(r['pts']) for r in actual] == proof['pts'] and stream['time_base'] == proof['time_base']
                and stream['width'] == proof['width'] and stream['height'] == proof['height'], 'Independent native video clock/geometry differs')
            audio_stream = next(r for r in probe['streams'] if r['codec_type'] == 'audio')
            run.require(int(audio_stream.get('start_pts', 0)) == 0 and int(audio_stream['sample_rate']) == 16000
                and int(audio_stream['channels']) == 1, 'Independent original audio origin differs')
            audio = [math.log(max(math.sqrt(math.fsum(float(v) ** 2 for v in pcm[i:i + 640]) / 640), 1e-8))
                     for i in range(0, len(pcm), 640)]
            aperture = []
            for frame, trace in enumerate(source['geometry_trace']):
                run.require(trace['frame_index'] == frame, 'Independent native mesh order differs')
                p = [(r[0] * proof['width'], r[1] * proof['height']) for r in trace['points']]
                aperture.append(math.hypot(p[0][0] - p[1][0], p[0][1] - p[1][1]) /
                                math.hypot(p[2][0] - p[3][0], p[2][1] - p[3][1]))
            run.require(len(audio) == len(aperture) == len(record['audio']) == len(record['aperture']) == source['frames'],
                        'Complete independently measured temporal traces required')
            for a, b in zip(audio + aperture, record['audio'] + record['aperture'], strict=True):
                close(a, b)
            for original, recorded in zip(source['windows'], record['windows'], strict=True):
                values, supports = features(original, audio, aperture)
                run.require(supports == recorded['anchor_indices'], 'Independent common lag supports differ')
                for a, b in zip(values, recorded['features'], strict=True):
                    close(a, b)
                windows += 1
            samples += len(pcm)
            frames += len(audio)
    result, private = run.read(trial / 'result.json'), run.read(trial / 'private-result.json')
    absolute = run.absolute_rows(parent)
    absolute_point = packet['absolute_reference']['review_point']
    for name, selected in [('control', rows), ('candidate', joined)]:
        head = run.read(trial / (name + '-head.json'))
        point = math.nextafter(max(0., max(margin(w['features'], head) for r in selected['calibration'] for w in r['windows'])), math.inf)
        run.require(point == head['review_point'], 'Independent genuine-only point differs')
        standalone, combined = [], []
        for row, ref, actual, signal in zip(selected['development'], absolute, private[name]['scored'], private[name]['signals'], strict=True):
            ranked = sorted([{**w, 'response': margin(w['features'], head)} for w in row['windows']],
                            key=lambda w: (-w['response'], w['start_s'], w['track_id']))
            run.require(actual['id'] == row['id'] and len(actual['windows']) == len(ranked), 'All independently scored windows required')
            for left, right in zip(actual['windows'], ranked, strict=True):
                run.require({k: v for k, v in left.items() if k != 'response'} == {k: v for k, v in right.items() if k != 'response'},
                            'Independent source window differs')
                close(left['response'], right['response'])
            measured = outcome(row, ranked[:12], point)
            standalone.append(measured)
            old_flag = bool(ref['probe_windows'] and ref['probe_windows'][0]['response'] >= absolute_point)
            combined.append(packet['absolute_reference']['outcomes'][len(combined)] if old_flag else measured)
            expected = 'absolute' if old_flag else 'relative' if measured['flagged'] else None
            run.require(signal == {'id': row['id'], 'signal': expected}, 'Independent evidence priority differs')
        run.require(standalone == private[name]['standalone'] and combined == private[name]['combined'],
                    'Independent native-mask/time outcomes differ')
        for key, part in [('standalone', standalone), ('combined', combined)]:
            run.require(summary(part) == result['arms'][name][key], 'Independent complete aggregates differ')
        expected_gates = gates(packet['absolute_reference']['outcomes'], combined, rows['development'])
        if name == 'candidate':
            control = private['control']['combined']
            expected_gates.update(no_fewer_locations_than_visual_control=sum(r['localized'] for r in combined) >= sum(r['localized'] for r in control),
                                  no_more_ordinary_flags_than_visual_control=sum(not r['label'] and r['flagged'] for r in combined)
                                      <= sum(not r['label'] and r['flagged'] for r in control))
        run.require(expected_gates == result['arms'][name]['gates'], 'Independent unchanged advancement gates differ')
        for family in ('liveportrait', 'musetalk'):
            run.require(summary([r for r, source in zip(combined, rows['development'], strict=True) if source['family'] == family])
                        == result['arms'][name]['families'][family], 'Complete family result differs')
        for condition in result['conditions']:
            run.require(summary([r for r, source in zip(combined, rows['development'], strict=True) if source['condition'] == condition])
                        == result['conditions'][condition][name], 'Complete ordinary/AI condition result differs')
    run.require(windows == 6373 and result['passes'] == all(result['arms']['candidate']['gates'].values()), 'Complete decision required')
    run.write(output, {'schema': 'chipmunk.mouth-audio-context-complete-review/13', 'audio_tracks_redecoded': 272,
              'pcm_samples_replayed': samples, 'native_video_clocks_and_frame_geometry': frames,
              'new_window_feature_vectors_replayed': windows, 'raw_new_feature_values_replayed': windows * 20,
              'all_native_mask_time_outcomes_exact': True, 'both_complete_aggregates_exact': True,
              'result_sha256': run.sha(trial / 'result.json'), 'reviewer_sha256': run.sha(Path(__file__)),
              'new_neural_forwards': 0, 'new_head_fits': 0, 'outside_human_reviewers': 0})
    print({'audio_tracks': 272, 'windows': windows, 'samples': samples, 'all_outcomes_exact': True}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trial', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    review(args.trial, args.output)
