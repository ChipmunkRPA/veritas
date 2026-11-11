"""Qualified appearance ranges with current-check fallback on unavailable context."""
from __future__ import annotations

import hashlib
import time
from contextlib import closing

import cv2

from app.core.config import settings
from app.detectors.image.adapters.common import revisions
from app.detectors.image.faces import detect_faces
from app.detectors.video import face_appearance as method
from app.detectors.video import face_boundary
from app.detectors.video.adapters.common import CPU_ONLY, FULL_DEEP, video_spec
from app.detectors.video.adapters.face_sequence import FAMILY, FaceSequenceReviewDetector, SequenceInputs
from app.detectors.video.decode import DecodeError
from app.engine.detector import ActivationUse, DetectorResult, InputUnit, UnitScore
from app.evidence.face_intervals import bind_tracks

DETECTOR = "vid.review.face_appearance"
BOUNDARY_BUDGET_S = 8.0
COARSE_RESULT_RESERVE_S = 1.0


def record(window, table, inputs):
    centers = window['centers']
    roles = {'before': centers[:2], 'during': centers[2:-2], 'after': centers[-2:]}
    indices = sorted({i + k for i in centers for k in (-1, 0, 1)})
    return {'method': method.METHOD, 'track_id': window['track_id'], 'time_base': table.time_base,
        'start_pts': table.pts[roles['during'][0]], 'end_pts': table.pts[roles['during'][-1]],
        'start_stream_s': window['start_s'], 'end_stream_s': window['end_s'],
        **{k: window[k] for k in ('response', 'middle_to_before', 'middle_to_after', 'flank_change')},
        'measurement_hex': {k: float(window[k]).hex() for k in ('response', 'middle_to_before', 'middle_to_after', 'flank_change')},
        'centers': roles, 'frame_inputs': [inputs[i] for i in indices]}


def _measure(rgb, bgr, index, table, face, recognizer):
    try:
        detection = method.face_row(face.landmarks)
        detection[0, :4] = [face.x, face.y, face.w, face.h]
        detection[0, 14] = face.score
        crop = recognizer.alignCrop(bgr, detection)
        matrix = method.cpp_alignment(face.landmarks)
    except ValueError:
        return None
    feature = recognizer.feature(crop).reshape(-1).copy()
    method.unit(feature)  # Reject nonfinite or degenerate output.
    return feature, {
        'frame_index': index, 'pts': table.pts[index],
        'decoded_rgb_sha256': hashlib.sha256(rgb.tobytes()).hexdigest(),
        'bbox': {k: float(getattr(face, k)) for k in ('x', 'y', 'w', 'h')},
        'landmarks': [[float(x), float(y)] for x, y in face.landmarks],
        'alignment_matrix': matrix.tolist(), 'alignment_matrix_hex': [[float(v).hex() for v in r] for r in matrix],
        'alignment_side': 112, 'aligned_size': 112,
        'aligned_rgb_sha256': hashlib.sha256(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).tobytes()).hexdigest(),
    }


def _track_samples(bindings, inputs):
    observed = {'kind': 'regions', 'created_by': 'vid.faces.yunet', 'regions': [
        {'region': {'kind': 'bbox', 'frame_pts': round(inputs.table.seconds(i) * 1000), 'bbox': row['bbox']}}
        for i, row in bindings.items()]}
    return bind_tracks([{'frame_index': i, 'score': 0.} for i in bindings], [observed, *inputs.exhibits], inputs.table)


def _refine(coarse, inputs, yunet, recognizer, deadline):
    table = inputs.table
    times = face_boundary.schedule(table, coarse['start_pts'], coarse['end_pts'])
    remaining = deadline.remaining() if deadline is not None else None
    budget = BOUNDARY_BUDGET_S if remaining is None else min(BOUNDARY_BUDGET_S, remaining - COARSE_RESULT_RESERVE_S)
    if not times or budget <= 0:
        return None
    end = time.monotonic() + budget
    features, faces, bindings, decoded = {}, {}, {}, set()
    try:
        with closing(inputs.store.iter_many(times, timeout_s=budget)) as frames:
            for i, rgb in frames:
                if time.monotonic() >= end:
                    return None
                if deadline is not None:
                    deadline.check('native face timing')
                decoded.add(i)
                if not inputs.store.native_pts_verified(i):
                    continue
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                detected = detect_faces(bgr, detector=yunet, min_size=settings.face_min_size)
                if len(detected) != 1:
                    continue
                face = detected[0]
                measured = _measure(rgb, bgr, i, table, face, recognizer)
                if measured is None:
                    continue
                features[i], bindings[i] = measured
                faces[i] = face
    except (DecodeError, cv2.error, ValueError):
        return None
    if decoded != set(times) or time.monotonic() >= end:
        return None
    track_id = coarse['track_id']
    members = sorted(sample.frame_index for sample in _track_samples(bindings, inputs).get(track_id, ()))
    estimate = face_boundary.estimate(
        times, coarse['start_pts'] * table.tb, coarse['end_pts'] * table.tb,
        {i: features[i] for i in members}, faces,
    )
    if estimate is None or time.monotonic() >= end:
        return None
    first, stop = estimate.entry.first_new, estimate.exit.first_new
    return {
        'method': face_boundary.METHOD, 'time_base': table.time_base, 'track_id': track_id,
        'start_pts': table.pts[first], 'end_pts': table.pts[stop],
        'start_stream_s': table.seconds(first), 'end_stream_s': table.seconds(stop),
        'anchors': {'before': estimate.entry.frame_indices[1], 'during': estimate.during, 'after': stop},
        'scheduled_frame_indices': sorted(times),
        'frame_inputs': [bindings[i] for i in members],
        'edges': {name: {'frame_indices': list(edge.frame_indices),
                         'signed_distance_hex': [float(value).hex() for value in edge.signed_distances]}
                  for name, edge in (('entry', estimate.entry), ('exit', estimate.exit))},
    }


class FaceAppearanceReviewDetector(FaceSequenceReviewDetector):
    APPEARANCE = 'opencv.sface.2021dec'
    spec = video_spec(DETECTOR, 'Face appearance transition review', InputUnit.FRAME_BATCH, (), ('series',),
        family=FAMILY, model_ids=(FaceSequenceReviewDetector.YUNET, FaceSequenceReviewDetector.DINO, APPEARANCE),
        needs=('frames', 'faces'), profiles=FULL_DEEP, devices=CPU_ONLY, timeout_s=120., version='0.1.2')

    def activation_plan(self, ctx):
        return (*super().activation_plan(ctx), ActivationUse(self.APPEARANCE, 1, '112 px aligned face',
            'One native aligned face at a time; 128 float32 appearance values per retained frame',
            (f'At most 192 coarse face inputs and {face_boundary.MAX_FRAMES} optional timing inputs.',
             'Face vectors stay in working memory.')))

    def run(self, ctx):
        inputs: SequenceInputs = ctx.outputs.face_review
        table = inputs.table
        centers = {i for i in inputs.centers if 0 < i < len(table) - 1}
        scheduled = {i + k for i in centers for k in (-1, 0, 1)}
        yunet, recognizer = ctx.lease.get(self.YUNET).obj, ctx.lease.get(self.APPEARANCE).obj
        check = ctx.deadline.check if ctx.deadline is not None else lambda *_: None
        features, bindings, decoded = {}, {}, set()
        unverified = 0
        for i, rgb in inputs.store.iter_many(sorted(scheduled)):
            check('face appearance crops')
            decoded.add(i)
            if not inputs.store.native_pts_verified(i):
                unverified += 1
                continue
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            detected = detect_faces(bgr, detector=yunet, min_size=settings.face_min_size)
            if not detected:
                continue
            measured = _measure(rgb, bgr, i, table, detected[0], recognizer)
            if measured is not None:
                features[i], bindings[i] = measured
        if decoded != scheduled:
            return DetectorResult.failed('a scheduled native face input could not be decoded')
        tracks = _track_samples(bindings, inputs)
        windows = method.windows(centers, tracks, features, table)
        if not windows:
            fallback = super().run(ctx)
            fallback.details.update(appearance_check_status='not_applicable',
                appearance_check_reason='insufficient source-bound tracked context; the previous review check was retained')
            fallback.model_revisions = revisions(ctx, self.YUNET, self.DINO, self.APPEARANCE)
            return fallback
        highest = windows[0]
        selected = record(highest, table, bindings) if highest['response'] > method.REVIEW_POINT else None
        details = {'face_sequence_binding_version': method.METHOD, 'face_sequence': selected,
                   'face_sequence_windows': [record(w, table, bindings) for w in windows],
                   'input_frame_indices': sorted(features), 'review_only': True, 'appearance_check_status': 'ok'}
        if (settings.video_face_boundary_refinement and selected
                and (refined := _refine(selected, inputs, yunet, recognizer, ctx.deadline))):
            details['face_boundary'] = refined
        sid = f'{DETECTOR}:global'
        exhibit = ctx.exhibits.series([{'name': 'appearance_change', 'points': [
            ((w['start_s'] + w['end_s']) / 2, w['response']) for w in windows]}], x_unit='s',
            title='Face appearance transition measurements', media_ref='frame', source_signal_ids=[sid],
            caption='Overlapping sampled time windows; no individual-frame AI probability.')
        return DetectorResult(status='ok', aggregate=UnitScore('global', highest['response'], 'score'),
            observed=(f"A bounded face appearance change merits review from {highest['start_s']:.3f} to {highest['end_s']:.3f} s."
                if selected else 'The sampled face appearances supplied no additional range to review.'),
            reference='The middle time group differs from both neighboring groups, while the two neighbors resemble one another.',
            reasoning='Motion, expression, lighting, cuts and compression may cause similar changes. This inspection cue is not an AI or identity probability and does not change overall confidence.',
            metrics={'scheduled_frames': float(len(scheduled)), 'face_frames': float(len(features)), 'windows': float(len(windows)),
                     'unverified_timestamp_frames': float(unverified)},
            details=details,
            exhibit_ids=[exhibit], model_revisions=revisions(ctx, self.YUNET, self.DINO, self.APPEARANCE))
