"""Within-track bounded face-appearance changes; not AI probabilities or identity matching."""
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

FLANK_PENALTY = .35
RULE_NAME = 'within-track-sface-bounded-change/1'


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def unit(vector):
    value = np.asarray(vector, np.float64).reshape(-1)
    norm = float(np.linalg.norm(value))
    if value.shape != (128,) or not np.isfinite(value).all() or norm <= 1e-12:
        raise ValueError('A finite nonzero native 128-value feature is required')
    return value / norm


def distance(a, b):
    an, bn = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if min(an, bn) <= 1e-12:
        raise ValueError('Degenerate pooled face appearance cannot support a window')
    return float(1 - np.clip(np.dot(a, b) / (an * bn), -1., 1.))


def windows(centers, tracks, features, table):
    """Same source triplets, track membership, cadence and windows as the current cue."""
    normalized = {i: unit(value) for i, value in features.items()}
    result = []
    for track_id, samples in tracks.items():
        tracked = {sample.frame_index for sample in samples}
        blocks = []
        for i in sorted(centers):
            group = (i - 1, i, i + 1)
            if all(k in tracked and k in normalized for k in group) and table.seconds(i + 1) - table.seconds(i - 1) <= .13:
                blocks.append((i, np.mean([normalized[k] for k in group], axis=0)))
        times = np.asarray([table.seconds(i) for i, _ in blocks], np.float64)
        for start in range(2, len(blocks) - 4):
            for end in range(start + 3, len(blocks) - 1):
                span = times[end - 1] - times[start]
                if span > 2.6:
                    break
                if span < .52 or np.max(np.diff(times[start - 2:end + 2])) > .55:
                    continue
                left, middle, right = [np.mean([value for _, value in part], axis=0) for part in
                    (blocks[start - 2:start], blocks[start:end], blocks[end:end + 2])]
                l, r, flanks = distance(middle, left), distance(middle, right), distance(left, right)
                response = min(l, r) - FLANK_PENALTY * flanks
                result.append({'track_id': track_id, 'start_s': float(times[start]), 'end_s': float(times[end - 1]),
                    'response': response, 'middle_to_before': l, 'middle_to_after': r, 'flank_change': flanks,
                    'centers': [i for i, _ in blocks[start - 2:end + 2]]})
    return sorted(result, key=lambda w: (-w['response'], w['start_s'], w['track_id']))[:12]


def selected(values, point, *, strict=False):
    if not values:
        return None
    top = values[0]
    if not math.isfinite(top['response']):
        raise ValueError('Nonfinite top window')
    eligible = top['response'] > point if strict else top['response'] >= point
    return [top['start_s'], top['end_s']] if eligible else None


def legacy_range(row):
    from app.detectors.video.decode import FrameTable
    from app.evidence.face_intervals import RULE, bind_tracks, localize

    points = sorted(row.get('points') or [], key=lambda p: p['frame_index'])
    if not points:
        return None
    table = FrameTable.from_json(row['table'])
    intervals = localize(bind_tracks(points, row['exhibits'], table), RULE)
    if intervals:
        return [intervals[0].during[0].time_s, intervals[0].during[-1].time_s]
    peak = max(range(len(points)), key=lambda i: points[i]['score'])
    return ([table.seconds(points[max(0, peak - 1)]['frame_index']),
             table.seconds(points[min(len(points) - 1, peak + 1)]['frame_index'])] if points[peak]['score'] >= .5 else None)


def operating_point(items, responses, current_point):
    by_id = {row['id']: row for row in responses}
    expected = [row for row in items if row['role'] == 'calibration']
    if len(expected) != 60 or len(by_id) != len(responses) or set(by_id) != {row['id'] for row in expected}:
        raise ValueError('Exact 60 planned genuine calibration cases required')
    if any(row['label'] != 0 for row in expected):
        raise ValueError('AI cases cannot choose this operating point')
    if any(row['status'] not in {'ok', 'not_applicable'} for row in responses):
        raise ValueError('Failed calibration case; no development admission')
    available = [row for row in expected if by_id[row['id']].get('candidate_windows')]
    if len(available) < 40 or len({row['group'] for row in available}) < 20:
        raise ValueError('Too little genuine tracked context to set an operating point')
    values = [w['response'] for row in responses for w in row.get('candidate_windows', [])]
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError('Complete finite calibration windows required')
    unflagged = [row for row in available if not (
        selected(by_id[row['id']].get('current_windows', []), current_point) or row['legacy_range_s'])]
    if len(unflagged) < 20:
        raise ValueError('At least twenty eligible currently unflagged genuine cases required')
    maxima = sorted((max(w['response'] for w in by_id[row['id']]['candidate_windows']) for row in unflagged), reverse=True)
    # A strict comparison to the second-largest top response adds at most one
    # genuine calibration review flag. No AI label/response enters selection.
    return {'method': 'strictly_greater_than_second_largest_currently_unflagged_genuine_case_maximum',
        'value': max(0., maxima[1]),
        'planned_genuine': 60, 'available_genuine': len(available), 'available_source_groups': len({r['group'] for r in available}),
        'eligible_currently_unflagged_genuine': len(unflagged), 'maximum_additional_genuine_calibration_flags': 1,
        'confidence_calibration': False, 'identity_match_point': False,
        'note': 'Inspection feature units, not an authenticity probability or public screening threshold.'}


def overlap(span, truth):
    if span is None or truth is None:
        return 0.
    union = max(span[1], truth[1]) - min(span[0], truth[0])
    return max(0., min(span[1], truth[1]) - max(span[0], truth[0])) / union if union > 0 else 0.


def outcomes(items, responses, point, current_point):
    by_id = {r['id']: r for r in responses}
    if len(by_id) != len(responses) or set(by_id) != {r['id'] for r in items}:
        raise ValueError('Exact planned development denominator required')
    result = []
    for item in items:
        response = by_id[item['id']]
        base = selected(response.get('current_windows', item.get('historical_current_windows', [])), current_point) or item['legacy_range_s']
        cue = selected(response.get('candidate_windows', []), point['value'], strict=True) if response['status'] == 'ok' else None
        candidate = cue or (base if response['status'] != 'ok' else item['legacy_range_s'])
        truth = item.get('edit_s')
        result.append({**{k: item[k] for k in ('id', 'group', 'family', 'label', 'role')},
            'known_edit_s': truth, 'current_range_s': base, 'candidate_range_s': candidate,
            'current_iou': overlap(base, truth), 'candidate_iou': overlap(candidate, truth),
            'current_localized': overlap(base, truth) >= .3, 'candidate_localized': overlap(candidate, truth) >= .3,
            'candidate_used': cue is not None, 'candidate_status': response['status'],
            'baseline_parity_passed': response.get('baseline_parity_passed', False),
            'native_alignment_all_exact': response.get('native_alignment_all_exact', True)})
    return result


def metrics(cases, arm):
    if arm not in {'current', 'candidate'}:
        raise ValueError('Unknown arm')
    ai, genuine = [r for r in cases if r['label']], [r for r in cases if not r['label']]
    hard = [r for r in genuine if r['family'] != 'original_videos']
    return {'planned': len(cases), 'source_groups': len({r['group'] for r in cases}), 'edits': len(ai), 'genuine': len(genuine),
        'localized_edits': sum(r[arm + '_localized'] for r in ai),
        'missed_localizations': sum(not r[arm + '_localized'] for r in ai),
        'genuine_review_flags': sum(r[arm + '_range_s'] is not None for r in genuine),
        'ordinary_edit_flags': sum(r[arm + '_range_s'] is not None for r in hard),
        'mean_edit_iou': float(np.mean([r[arm + '_iou'] for r in ai])) if ai else None,
        'status_counts': dict(Counter(r['candidate_status'] for r in cases)),
        'failed_cases': sum(r['candidate_status'] == 'failed' for r in cases)}
