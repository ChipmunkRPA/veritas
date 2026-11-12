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
from app.detectors.video import face_boundary, face_detail_repeat, face_return, face_return_fine_scene
from app.detectors.video.adapters.common import CPU_ONLY, FULL_DEEP, video_spec
from app.detectors.video.adapters.face_sequence import FAMILY, FaceSequenceReviewDetector, SequenceInputs
from app.detectors.video.decode import DecodeError
from app.engine.boundary import Deadline, DetectorTimeout
from app.engine.detector import ActivationUse, Applicability, DetectorResult, InputUnit, UnitScore
from app.evidence.face_intervals import bind_tracks
from app.evidence.face_return_binding import (
    DENSE_FRAME_PROFILES,
    FINE_SCENE_FRAME_COUNT,
    FINE_SCENE_METHOD,
    GUARDED_720P30_METHOD,
    GUARDED_CODEC_METHOD,
    GUARDED_METHODS,
    GUARDED_SCENE_QUARTILE_POINT,
    SECOND_HD_METHOD,
    SECOND_METHOD,
    SECOND_PRIOR_METHODS,
    observation,
)

DETECTOR = "vid.review.face_appearance"
BOUNDARY_BUDGET_S = 8.0
RETURN_BUDGET_S = 12.0
DETAIL_REPEAT_BUDGET_S = 4.0
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


def _inspect_return(details, inputs, yunet, recognizer, deadline, incumbent):
    coverage = {"status": "not_applicable", "reason": "native_scope", "scheduled_frames": 0, "measured_face_frames": 0}
    details["face_return_check"] = coverage
    table = inputs.table
    planned = face_return.schedule(table)
    hd = planned is None
    if hd:
        planned = face_return.guarded_720p30_schedule(table, inputs.encoding, getattr(inputs.store, "path", None))
    if planned is None or settings.face_min_size != 48:
        return
    remaining = deadline.remaining() if deadline is not None else None
    budget = RETURN_BUDGET_S if remaining is None else min(RETURN_BUDGET_S, remaining - COARSE_RESULT_RESERVE_S)
    stop = time.monotonic() + budget
    if hd:
        # Use the sparse pass for context only. Every published 720p30 range
        # comes from the tested dense pass and its stronger background guard.
        _return_pass({}, inputs, yunet, recognizer, deadline, incumbent, planned, coverage, face_return.METHOD, stop)
        if coverage["status"] == "localized":
            coverage.update(status="unassessed", reason="sparse_range_outside_profile")
        elif coverage["status"] == "no_distinct_range":
            coverage["sparse_context"] = coverage.copy()
            dense_coverage = {"measured_face_frames": 0}
            coverage["dense_fallback"] = dense_coverage
            dense_plan = planned[0], tuple(range(0, len(table), 2))
            context = [] if settings.video_face_second_return_review else None
            additional = {} if context is None else {"context": context}
            _return_pass(details, inputs, yunet, recognizer, deadline, incumbent, dense_plan, dense_coverage,
                         GUARDED_720P30_METHOD, stop, source_carrier="mp4", **additional)
            coverage.update(dense_coverage)
            if context and (primary := details.get("face_return")) is not None:
                _record_second_return(details, inputs, incumbent, primary, dense_plan, stop, context[0], SECOND_HD_METHOD)
        return
    dense = face_return.dense_schedule(table, inputs.encoding)
    path = getattr(inputs.store, "path", None)
    context = ([] if settings.video_face_second_return_review and dense is not None and len(table) == 120
               and path is not None and face_return.source_carrier(path) == "mp4" else None)
    _inspect_standard_return(details, inputs, yunet, recognizer, deadline, incumbent, planned, dense, coverage, stop, context)
    if context is not None:
        _inspect_second_return(details, inputs, yunet, recognizer, deadline, incumbent, dense, stop, context)


def _inspect_standard_return(details, inputs, yunet, recognizer, deadline, incumbent, planned, dense, coverage, stop, context):
    table = inputs.table
    _return_pass(details, inputs, yunet, recognizer, deadline, incumbent, planned, coverage, face_return.METHOD, stop)
    if time.monotonic() >= stop:
        return
    if dense is None:
        path = getattr(inputs.store, "path", None)
        if coverage["status"] == "no_distinct_range" and path is not None:
            guarded = face_return.guarded_dense_schedule(table, inputs.encoding, path)
            if guarded is not None:
                planned, carrier = guarded
                dense_coverage = {"measured_face_frames": 0}
                coverage["dense_fallback"] = dense_coverage
                _return_pass(details, inputs, yunet, recognizer, deadline, incumbent, planned, dense_coverage,
                             GUARDED_CODEC_METHOD, stop, source_carrier=carrier)
                if dense_coverage["status"] == "localized":
                    coverage["status"] = "localized"
        return
    dense_coverage = {}
    if coverage["status"] == "no_distinct_range":
        dense_coverage["measured_face_frames"] = 0
        coverage["dense_fallback"] = dense_coverage
        additional = {} if context is None else {"context": context}
        _return_pass(details, inputs, yunet, recognizer, deadline, incumbent, dense, dense_coverage,
                     DENSE_FRAME_PROFILES[len(table)], stop, **additional)
        if dense_coverage["status"] == "localized":
            coverage["status"] = "localized"
            return
    eligible = coverage["status"] == "no_distinct_range" or coverage.get("reason") == "insufficient_same_shot_context"
    if (len(table) != FINE_SCENE_FRAME_COUNT or not eligible or time.monotonic() >= stop
            or dense_coverage.get("reason") not in (None, "insufficient_same_shot_context")):
        return
    fine_coverage = {"measured_face_frames": 0}
    coverage["fine_scene_fallback"] = fine_coverage
    _return_pass(details, inputs, yunet, recognizer, deadline, incumbent, dense, fine_coverage,
                 FINE_SCENE_METHOD, stop)
    if fine_coverage["status"] == "localized":
        coverage.update(status="localized", reason=None)


def _inspect_second_return(details, inputs, yunet, recognizer, deadline, incumbent, planned, stop, context):
    primary = details.get("face_return")
    if not primary or primary["method"] not in SECOND_PRIOR_METHODS or time.monotonic() >= stop:
        return
    if not context:
        if primary["method"] != face_return.METHOD:
            return
        _return_pass({}, inputs, yunet, recognizer, deadline, incumbent, planned, {"measured_face_frames": 0},
                     DENSE_FRAME_PROFILES[len(inputs.table)], stop, context=context)
    if not context or time.monotonic() >= stop:
        return
    _record_second_return(details, inputs, incumbent, primary, planned, stop, context[0], SECOND_METHOD)


def _record_second_return(details, inputs, incumbent, primary, planned, stop, context, method):
    if time.monotonic() >= stop:
        return
    features, faces, scenes = context
    earlier = [item for item in (primary, incumbent, details.get("face_boundary")) if item]
    times, scheduled = planned
    chosen, quartiles = face_return.second(times, scheduled, features, inputs.table, earlier)
    if chosen is None or time.monotonic() >= stop:
        return
    boundary_scenes = {}
    for index in chosen.scene_frames:
        face = features[index].face
        boundary_scenes[index] = {**scenes[index], "face_box": list(face.box()) if face is not None else None}
    record = _return_record(inputs.table, scheduled, chosen, method, faces, boundary_scenes)
    details["face_return_secondary"] = {**record, "source_encoding": list(inputs.encoding), "source_carrier": "mp4",
        "scene_lower_quartiles": quartiles, "scene_lower_quartiles_hex": [value.hex() for value in quartiles]}


def _return_pass(details, inputs, yunet, recognizer, deadline, incumbent, planned, coverage, record_method, stop, *,
                 source_carrier=None, context=None):
    table = inputs.table
    times, scheduled = planned
    coverage.update(status="unassessed", reason="time_budget", scheduled_frames=len(scheduled))
    budget = stop - time.monotonic()
    if budget <= 0:
        return
    features, face_inputs, scene_inputs = {}, {}, {}
    fine = record_method == FINE_SCENE_METHOD
    guarded = record_method in GUARDED_METHODS
    fine_frames, fine_edges = {}, {}
    try:
        with closing(inputs.store.iter_many(scheduled, timeout_s=budget)) as frames:
            for index, rgb in frames:
                if time.monotonic() >= stop:
                    return
                if deadline is not None:
                    deadline.check("short face return")
                if not inputs.store.native_pts_verified(index):
                    coverage["reason"] = "native_timestamp_unverified"
                    return
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                faces = detect_faces(bgr, detector=yunet, min_size=48)
                if len(faces) > 1:
                    coverage["reason"] = "multiple_faces"
                    return
                face = faces[0] if faces else None
                observed = _measure(rgb, bgr, index, table, face, recognizer) if face is not None else None
                vector = None
                if observed is not None:
                    vector, face_inputs[index] = observed
                    vector = method.unit(vector)
                    coverage["measured_face_frames"] += 1
                scene_inputs[index] = {"frame_index": index, "pts": table.pts[index], "decoded_rgb_sha256":
                    face_inputs[index]["decoded_rgb_sha256"] if observed is not None else hashlib.sha256(rgb.tobytes()).hexdigest()}
                features[index] = face_return.Sample(face, vector, face_return.scene(rgb, face))
                if fine or guarded:
                    box = face.box() if face is not None else None
                    scene_inputs[index]["face_box"] = list(box) if box is not None else None
                if fine:
                    fine_frames[index] = face_return_fine_scene.frame(rgb, box)
    except (DecodeError, cv2.error, ValueError):
        coverage["reason"] = "native_measurement_unavailable"
        return
    if features.keys() != set(scheduled):
        coverage["reason"] = "incomplete_native_decode"
        return
    if time.monotonic() >= stop:
        return
    def compare_scenes(left, right):
        first, second = features[left], features[right]
        value = face_return.scene_correlation(first, second)
        if value is not None:
            return value
        coarse_tiles = int((first.scene[1] & second.scene[1]).sum())
        value, support = face_return_fine_scene.edge(fine_frames[left], fine_frames[right], coarse_tiles)
        if support is not None:
            fine_edges[left, right] = support
        return value

    if context is not None:
        context.append((features, face_inputs, scene_inputs))
    candidate = (face_return.highest(times, scheduled, features, compare_scenes=compare_scenes)
                 if fine else face_return.highest(times, scheduled, features))
    if time.monotonic() >= stop:
        return
    if candidate is None:
        coverage["reason"] = "insufficient_same_shot_context"
        return
    coverage.update(status="no_distinct_range", reason=None)
    if not face_return.distinct(candidate, table, incumbent):
        return
    if guarded:
        quartiles = face_return.scene_quartiles(candidate, features)
        if any(value is None or value < GUARDED_SCENE_QUARTILE_POINT for value in quartiles):
            coverage["reason"] = "surrounding_scene_changed"
            return
    if fine:
        selected = [fine_edges.get(pair) for pair in (candidate.scene_frames[:2], candidate.scene_frames[2:])]
        if not any(selected):
            return
    coverage["status"] = "localized"
    record = _return_record(table, scheduled, candidate, record_method, face_inputs, scene_inputs)
    if record_method != face_return.METHOD:
        record["source_encoding"] = list(inputs.encoding)
    if fine:
        record["fine_scene_edges"] = selected
    if guarded:
        record.update(source_carrier=source_carrier, scene_lower_quartiles=quartiles,
                      scene_lower_quartiles_hex=[value.hex() for value in quartiles])
    details["face_return"] = record


def _return_record(table, scheduled, candidate, record_method, face_inputs, scene_inputs):
    start, end = candidate.start, candidate.end
    return {
        "method": record_method, "time_base": table.time_base, "start_pts": table.pts[start], "end_pts": table.pts[end],
        "start_frame_index": start, "end_frame_index": end,
        "start_stream_s": table.seconds(start), "end_stream_s": table.seconds(end),
        "response": candidate.response, "response_hex": candidate.response.hex(),
        "scene_correlations": list(candidate.scene_correlations),
        "scene_correlations_hex": [value.hex() for value in candidate.scene_correlations],
        "anchors": candidate.anchors(), "scheduled_frame_indices": list(scheduled),
        "support_indices": {role: list(group) for role, group in zip(face_return.ROLES, candidate.support, strict=True)},
        "frame_inputs": [face_inputs[index] for group in candidate.support for index in group],
        "scene_frame_inputs": [scene_inputs[index] for index in candidate.scene_frames],
    }


def _with_return(result, inputs, yunet, recognizer, deadline):
    if not settings.video_face_short_return_review or result.status not in {'ok', 'not_applicable'}:
        return result
    incumbent = result.details.get('face_sequence')
    _inspect_return(result.details, inputs, yunet, recognizer, deadline, incumbent)
    selected = result.details.get('face_return')
    if selected is None:
        return result
    observed = (f"A brief face appearance change merits review from {selected['start_stream_s']:.3f} "
                f"to {selected['end_stream_s']:.3f} s.")
    if result.status == 'ok':
        result.observed = f"{result.observed} {observed}" if incumbent is not None else observed
        return result
    return DetectorResult(status='ok', aggregate=UnitScore('global', selected['response'], 'score'),
        observed=observed, reference='Sampled face appearances were compared with the surrounding time groups.',
        reasoning='Motion, expression, lighting and compression can also change appearance. This range carries zero weight in overall AI confidence.',
        details={**result.details, 'review_only': True}, model_revisions=result.model_revisions)


def _repeat_face_detail(result, inputs, recognizer, deadline):
    details = result.details
    coarse = details.get("face_sequence")
    if (not settings.video_face_detail_repeat_review or settings.face_min_size != 48 or coarse is None
            or not face_detail_repeat.supported(inputs.table, inputs.encoding)):
        return result
    coverage = {key: coarse[key] for key in ("start_pts", "end_pts", "time_base", "start_stream_s", "end_stream_s")}
    coverage["status"] = "unassessed"
    details["face_detail_repeat"] = coverage
    remaining = deadline.remaining() if deadline is not None else None
    budget = DETAIL_REPEAT_BUDGET_S if remaining is None else min(DETAIL_REPEAT_BUDGET_S, remaining - COARSE_RESULT_RESERVE_S)
    if budget <= 0:
        return result
    stop = deadline.nested(budget) if deadline is not None else Deadline(budget)
    table = inputs.table
    indices = sorted(item["frame_index"] for item in coarse["frame_inputs"])
    try:
        with closing(inputs.store.iter_many(indices, timeout_s=budget)) as frames:
            measured = face_detail_repeat.measure(
                coarse, table, inputs.encoding,
                ((index, table.pts[index] if inputs.store.native_pts_verified(index) else None, rgb)
                 for index, rgb in frames), recognizer, check=stop.check,
            )
    except (DecodeError, DetectorTimeout, cv2.error, ValueError):
        return result
    coverage.update(status=measured["status"], measurement=measured)
    if coverage["status"] == "withheld":
        details["face_sequence"] = None
        details.pop("face_boundary", None)
        start, end = coverage["start_stream_s"], coverage["end_stream_s"]
        result.observed = face_detail_repeat.observation("withheld", start, end)
        if short := details.get("face_return"):
            result.observed += " " + observation(short)
    elif coverage["status"] == "retained":
        result.observed += " The change remained when the same frames were checked at lower detail."
    return result


class FaceAppearanceReviewDetector(FaceSequenceReviewDetector):
    APPEARANCE = 'opencv.sface.2021dec'
    spec = video_spec(DETECTOR, 'Face appearance transition review', InputUnit.FRAME_BATCH, (), ('series',),
        family=FAMILY, model_ids=(FaceSequenceReviewDetector.YUNET, FaceSequenceReviewDetector.DINO, APPEARANCE),
        needs=('frames', 'faces'), profiles=FULL_DEEP, devices=CPU_ONLY, timeout_s=120., version='0.1.4')

    def applicable(self, ctx):
        incumbent = super().applicable(ctx)
        inputs = ctx.outputs.face_review
        if (not incumbent.ok and settings.video_face_short_return_review and settings.face_min_size == 48
                and isinstance(inputs, SequenceInputs) and (face_return.schedule(inputs.table) is not None
                     or face_return.guarded_720p30_schedule(inputs.table, inputs.encoding, getattr(inputs.store, "path", None)) is not None)):
            return Applicability(True)
        return incumbent

    def activation_plan(self, ctx):
        return_inputs = (" Up to 64 additional native inputs for the distinct return check, with up to 75 fallback inputs "
                         "on tested H.264 profiles or 60 on guarded 720p30 H.264, VP9 and HEVC profiles. The tested 120-frame "
                         "H.264 profile can use up to 60 further inputs for finer background detail, within the same budget."
                         if settings.video_face_short_return_review else "")
        return (*super().activation_plan(ctx), ActivationUse(self.APPEARANCE, 1, '112 px aligned face',
            'One native aligned face at a time; 128 float32 appearance values per retained frame',
            (f'At most 192 coarse face inputs and {face_boundary.MAX_FRAMES} optional timing inputs.{return_inputs}',
             'An enabled lower-detail repeat reads the same input frames from the selected coarse range.',
             'Face vectors stay in working memory.')))

    def run(self, ctx):
        inputs: SequenceInputs = ctx.outputs.face_review
        table = inputs.table
        yunet, recognizer = ctx.lease.get(self.YUNET).obj, ctx.lease.get(self.APPEARANCE).obj
        if settings.video_face_short_return_review and not (incumbent := super().applicable(ctx)).ok:
            result = DetectorResult.not_applicable(incumbent.reason or 'insufficient coarse face context')
            result.details['appearance_check_status'] = 'not_applicable'
            result.model_revisions = revisions(ctx, self.YUNET, self.DINO, self.APPEARANCE)
            return _with_return(result, inputs, yunet, recognizer, ctx.deadline)
        centers = {i for i in inputs.centers if 0 < i < len(table) - 1}
        scheduled = {i + k for i in centers for k in (-1, 0, 1)}
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
            return _with_return(fallback, inputs, yunet, recognizer, ctx.deadline)
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
        result = DetectorResult(status='ok', aggregate=UnitScore('global', highest['response'], 'score'),
            observed=(f"A bounded face appearance change merits review from {highest['start_s']:.3f} to {highest['end_s']:.3f} s."
                if selected else 'The sampled face appearances supplied no additional range to review.'),
            reference='The middle time group differs from both neighboring groups, while the two neighbors resemble one another.',
            reasoning='Motion, expression, lighting, cuts and compression may cause similar changes. This inspection cue is not an AI or identity probability and does not change overall confidence.',
            metrics={'scheduled_frames': float(len(scheduled)), 'face_frames': float(len(features)), 'windows': float(len(windows)),
                     'unverified_timestamp_frames': float(unverified)},
            details=details,
            exhibit_ids=[exhibit], model_revisions=revisions(ctx, self.YUNET, self.DINO, self.APPEARANCE))
        result = _with_return(result, inputs, yunet, recognizer, ctx.deadline)
        return _repeat_face_detail(result, inputs, recognizer, ctx.deadline)
