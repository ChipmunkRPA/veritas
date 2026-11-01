"""Freeze the measured correlation inputs while they still exist in the AV run.

The record is an input/measurement commitment, not a manipulation-localisation
claim. Source ticks are integer 16 kHz units; 25 Hz frame choices are recorded
explicitly, including repeated VFR frame indices. No detector is rerun here.
"""
from __future__ import annotations

import hashlib
from dataclasses import asdict
from typing import Any

import numpy as np

from app.core.canonical import canonical_hash
from app.detectors.audio.decode import source_ranges
from app.detectors.video import lipsync, xcorr

VERSION = 'av-correlation-input/1'


def mouth_region(frame: dict, width: int, height: int) -> dict:
    """The exact integer RGB source rectangle used by mouth_patch, before resizing."""
    x, y, w, h = lipsync.mouth_box(frame)
    x0, y0 = int(np.clip(np.floor(x), 0, width - 1)), int(np.clip(np.floor(y), 0, height - 1))
    x1, y1 = int(np.clip(np.ceil(x + w), x0 + 1, width)), int(np.clip(np.ceil(y + h), y0 + 1, height))
    return {'x': x0, 'y': y0, 'w': x1 - x0, 'h': y1 - y0}


def capture_correlation(inputs: Any, grid: lipsync.TickGrid, speech: lipsync.Speech,
                        results: dict[str, lipsync.TrackSync], patches: dict,
                        apertures: dict, frames_by_track: dict, scores: dict[str, float]) -> dict:
    """Capture actual lag-search measurements and all of their source input ticks."""
    mapping = inputs.audio_mapping
    pcm = inputs.audio.pcm
    if not mapping or pcm is None or not hasattr(inputs.store, 'native_anchors'):
        return {'version': VERSION, 'status': 'unavailable', 'reason': 'The run has no native frame or audio commitments.'}
    ranges = source_ranges(mapping, 0, len(pcm))
    pcm_sha = hashlib.sha256(pcm.astype('<f4').tobytes()).hexdigest()
    if len(ranges) != 1 or pcm_sha != mapping.get('pcm_sha256'):
        raise ValueError('AV evidence input differs from its continuous source audio timeline')
    origin = ranges[0]['source_start_pts']
    if abs(grid.t0 - origin / lipsync.SR) > 1e-9:
        raise ValueError('AV tick grid differs from its recorded audio origin')
    anchors = {a['frame_index']: a for a in inputs.store.native_anchors()}
    tracks = []
    for tid, sync in results.items():
        ticks = sorted({k for window in sync.windows for k in window.window.ticks})
        frame_inputs = []
        for tick in ticks:
            index = grid.frames[tick]
            anchor = anchors.get(index)
            if anchor is None or anchor['pts'] != inputs.table.pts[index] or anchor['time_base'] != inputs.table.time_base:
                raise ValueError('AV measured frame has no matching native-pixel commitment')
            patch = np.ascontiguousarray(patches[tid][tick], dtype='<f4')
            frame_inputs.append({'tick': tick, 'frame_index': index, 'frame_pts': anchor['pts'],
                'source_audio_pts': origin + tick * lipsync.HOP, 'source_audio_time_base': f'1/{lipsync.SR}',
                'decoded_rgb_sha256': anchor['decoded_rgb_sha256'],
                'mouth_region': mouth_region(frames_by_track[tid][index], inputs.table.width, inputs.table.height),
                'mouth_patch_sha256': hashlib.sha256(patch.tobytes()).hexdigest(),
                'mouth_value': float(apertures[tid][tick])})
        windows = []
        for window, collapsed in zip(sync.windows, sync.collapsed, strict=True):
            start, end = window.start_tick, window.end_tick
            if not xcorr.full_search(window.window, speech.ticks) or window.distances.shape != (1, 31):
                raise ValueError('AV evidence requires the actual complete correlation lag search')
            profile = [1.0 - float(value) if np.isfinite(value) else None for value in window.distances[0]]
            # A scalar score alone does not demonstrate which lag won; keep all
            # candidates and the RMS inputs that each candidate actually used.
            audio_curve = [{'tick': k, 'sample_start': k * lipsync.HOP, 'sample_end': (k + 1) * lipsync.HOP,
                            'rms': float(speech.level[k]), 'speech': bool(speech.speech[k])}
                           for k in range(start - lipsync.VSHIFT, end + lipsync.VSHIFT)]
            windows.append({'unit_ref': f'{tid}@window={window.window.index}', 'window_index': window.window.index,
                'start_tick': start, 'end_tick': end, 'hop_ticks': window.window.hop,
                'audio_interval': {'start_sample': start * lipsync.HOP, 'end_sample': end * lipsync.HOP},
                'search_audio_interval': {'start_sample': (start - lipsync.VSHIFT) * lipsync.HOP,
                                          'end_sample': (end + lipsync.VSHIFT) * lipsync.HOP},
                'best_lag_audio_interval': {'start_sample': (start + window.offset) * lipsync.HOP,
                                            'end_sample': (end + window.offset) * lipsync.HOP},
                'offset_ticks': window.offset, 'peak_correlation': window.confidence,
                'lag_profile': [{'offset_ticks': offset, 'correlation': profile[offset + lipsync.VSHIFT]}
                                for offset in range(-lipsync.VSHIFT, lipsync.VSHIFT + 1)],
                'audio_curve': audio_curve, 'speech_fraction': window.window.speech_fraction,
                'median_face_px': window.window.face_px, 'low_correlation': bool(collapsed),
                'overlaps_offset_step': any(max(start, s.start_tick) < min(end, s.end_tick) for s in sync.steps),
                'overlaps_correlation_collapse': any(max(start, s.start_tick) < min(end, s.end_tick) for s in sync.collapses)})
        tracks.append({'track_id': tid, 'frame_inputs': frame_inputs, 'windows': windows,
            'speaker_attribution_score': float(scores[tid]), 'reading': sync.reading, 'baseline_offset_ticks': sync.baseline,
            'collapse_threshold': sync.collapse_threshold, 'minimum_trusted_peak': xcorr.MIN_PEAK,
            'segments': [asdict(segment) for segment in (*sync.steps, *sync.collapses)]})
    return {'version': VERSION, 'status': 'ready', 'method': 'mouth_rms_correlation',
            'source_sha256': mapping['source_sha256'], 'pcm_sha256': pcm_sha, 'samples': len(pcm), 'sample_rate': lipsync.SR,
            'mapping_sha256': canonical_hash(mapping), 'frame_table_sha256': canonical_hash(inputs.table.to_json()),
            'frame_time_base': inputs.table.time_base, 'audio_origin_pts': origin, 'audio_time_base': f'1/{lipsync.SR}',
            'tick_rate': 25, 'samples_per_tick': lipsync.HOP, 'lag_search_ticks': [-lipsync.VSHIFT, lipsync.VSHIFT],
            'patch_transform': {'version': 'mouth-rgb-grey-area/1', 'width': 32, 'height': 24,
                                'dtype': 'float32-le', 'colour': 'RGB-to-grayscale', 'resize': 'area',
                                'value': '(patch median - centre mean) / max(patch median, 1)'},
            'tracks': tracks}
