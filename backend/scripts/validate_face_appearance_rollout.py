"""Actual production adapter parity on all retained fresh v17 inputs.

Engineering replay of a completed panel, not another independent accuracy test.
No source imagery, private face vectors or customer uploads are exported.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def run(main, trial, output):
    import cv2
    import structlog
    import torch

    from scripts.run_sface_transition_v16 import configure

    configure(main, output.parent / 'isolated-state')
    from app.core.canonical import canonical_json
    from app.detectors.fakes import InMemoryExhibitWriter, InMemoryMedia
    from app.detectors.video.adapters.face_appearance import FaceAppearanceReviewDetector
    from app.detectors.video.adapters.face_sequence import SequenceInputs
    from app.detectors.video.decode import FrameTable
    from app.detectors.video.frame_store import FrameStore
    from app.engine.boundary import Deadline, InMemoryRunStore, run_detector
    from app.engine.detector import Profile, RunContext, StageOutputs
    from app.engine.models import get_model_manager
    from app.evidence.face_interval_binding import validate_sequence_record

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    cv2.setNumThreads(1)
    panel = json.loads((trial / 'private-confirmation-panel.json').read_text())['items']
    references = {r['id']: r for r in map(json.loads, (trial / 'candidate.jsonl').read_text().splitlines())}
    expected_cases = json.loads((trial / 'cases.json').read_text())
    if len(panel) != 120 or set(references) != {r['id'] for r in panel}:
        raise ValueError('Exact120 originally completed inputs required')
    manager, detector = get_model_manager(), FaceAppearanceReviewDetector()
    run_store = InMemoryRunStore()
    counts = {'cases': 0, 'direct_appearance_ranges': 0, 'fallback_cases': 0, 'no_window_cases': 0,
        'canonical_records_verified': 0, 'native_frame_bindings_verified': 0, 'retained_windows_verified': 0}
    maximum = 0.
    with output.open('x') as stream:
        stream.write(json.dumps({'schema': 'chipmunk.face-appearance-production-parity/1',
            'source_panel_sha256': sha(trial / 'private-confirmation-panel.json'),
            'reference_journal_sha256': sha(trial / 'candidate.jsonl'),
            'adapter_sha256': sha(Path(__file__).parents[1] / 'app/detectors/video/adapters/face_appearance.py'),
            'math_sha256': sha(Path(__file__).parents[1] / 'app/detectors/video/face_appearance.py'),
            'sface_component_tolerance': 1e-10, 'previous_fallback_tolerance': 2e-6,
            'runtime': 'CPU; torch2threads; OpenCV1thread; full real adapter boundary',
            'scope': __doc__}) + '\n')
        for item in panel:
            row, reference = item['legacy'], references[item['id']]
            path = Path(item['source_path'])
            if sha(path) != item['sha256'] or reference['source_sha256'] != item['sha256']:
                raise ValueError('Actual source bytes differ from original confirmation')
            table = FrameTable.from_json(row['table'])
            store = FrameStore(path, table, max_bytes=64 * 1024**2)
            inputs = SequenceInputs(table, store, tuple(r['frame_index'] for r in row['points']), tuple(row['exhibits']))
            context = RunContext(analysis_id=item['id'], media=InMemoryMedia(item['sha256'], 'video', path=path),
                profile=Profile.FULL, device='cpu', dtype=None, lease=None, outputs=StageOutputs(face_review=inputs),
                deadline=Deadline(120), exhibits=InMemoryExhibitWriter(), log=structlog.get_logger())
            actual = run_detector(detector, context, store=run_store, models=manager)
            appearance = bool(reference['candidate_windows'])
            expected = reference['candidate_windows'] if appearance else reference['current_windows']
            tolerance = 1e-10 if appearance else 2e-6
            windows = actual.details.get('face_sequence_windows', [])
            if len(windows) != len(expected) or actual.status != ('ok' if expected else 'not_applicable'):
                raise ValueError(f"Production window/status mismatch: {item['id']}")
            source = {'time_base': table.time_base, 'frame_count': len(table), 'width': table.width, 'height': table.height}
            anchors = {r['frame_index']: r for r in store.native_anchors()}
            for window, target in zip(windows, expected, strict=True):
                if ((window['track_id'], window['start_stream_s'], window['end_stream_s'], sum(window['centers'].values(), [])) !=
                        (target['track_id'], target['start_s'], target['end_s'], target['centers'])):
                    raise ValueError('Production selected a different source/track/time window')
                keys = (('response', 'middle_to_before', 'middle_to_after', 'flank_change') if appearance else
                        ('response', 'inner_change', 'outer_change', 'flank_change'))
                delta = max(abs(window[k] - target[k]) for k in keys)
                maximum = max(maximum, delta)
                if delta > tolerance:
                    raise ValueError('Production component differs from qualified native result')
                validate_sequence_record(json.loads(canonical_json(window)), source, table_pts=table.pts,
                    interpolated=table.interpolated, require_flag=False)
                counts['canonical_records_verified'] += 1
                counts['retained_windows_verified'] += 1
                for frame in window['frame_inputs']:
                    if frame['frame_index'] not in anchors or any(frame[k] != anchors[frame['frame_index']][k]
                            for k in ('pts', 'decoded_rgb_sha256')):
                        raise ValueError('Production input differs from actual native frame anchor')
                    counts['native_frame_bindings_verified'] += 1
            selected = actual.details.get('face_sequence')
            observed = [selected['start_stream_s'], selected['end_stream_s']] if selected else None
            fallback = item['legacy_range_s']
            combined = observed or fallback
            target_case = next(r for r in expected_cases if r['id'] == item['id'])
            if combined != target_case['candidate_range_s']:
                raise ValueError('Production plus declared legacy fallback differs from qualified final range')
            if selected:
                validate_sequence_record(json.loads(canonical_json(selected)), source, table_pts=table.pts,
                    interpolated=table.interpolated)
            if actual.aggregate is not None and actual.aggregate.raw_kind != 'score':
                raise ValueError('Inspection became a probability')
            counts['cases'] += 1
            counts['direct_appearance_ranges'] += bool(appearance and selected)
            counts['fallback_cases'] += not appearance
            counts['no_window_cases'] += not expected
            stream.write(json.dumps({'id': item['id'], 'passed': True, 'status': actual.status,
                'source_sha256': item['sha256'], 'selected_method': selected['method'] if selected else None,
                'combined_range_s': combined, 'canonical_source_binding': True, 'component_error': delta if windows else 0.}) + '\n')
            stream.flush()
            print(counts['cases'], '/120', item['id'], 'passed', flush=True)
        result = {'complete': counts['cases'] == 120, **counts, 'max_component_difference': maximum,
            'source_image_or_face_vector_exported': False, 'fresh_quality_reestimated': False}
        stream.write(json.dumps(result) + '\n')
        print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('main', type=Path)
    parser.add_argument('trial', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    run(args.main, args.trial, args.output)
