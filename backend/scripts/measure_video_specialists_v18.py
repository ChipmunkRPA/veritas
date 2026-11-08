"""Fixed-support video measurements with explicit unavailable observations.

Inputs use one visible detected face, without identifying a person. Local truth
describes sampled source-frame manipulation, never face/pixel authorship.
Pretrained models are unchanged; this module does not fit or admit anything.
"""
from __future__ import annotations

import hashlib
import math

import numpy as np


def require(value, reason):
    if not value:
        raise ValueError(reason)


def sha_array(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def overlap(first, second):
    x, y, w, h = first
    a, b, c, d = second
    intersection = max(0, min(x + w, a + c) - max(x, a)) * max(0, min(y + h, b + d) - max(y, b))
    union = w * h + c * d - intersection
    return intersection / union if union > 0 else 0.


def fixed_change(unit, values, *, appearance):
    """The existing cue's arithmetic on one predeclared three-middle window."""
    from app.detectors.video import face_appearance as identity
    from app.detectors.video import face_sequence as feature

    needed = {i for key in ('sampled_frames', 'context_frames') for i in unit['location'][key]}
    if not needed <= set(values):
        return None
    current = {i: identity.unit(values[i]) for i in needed} if appearance else values

    def blocks(centers):
        return np.mean([np.mean([current[i + k] for k in (-1, 0, 1)], axis=0) for i in centers], axis=0)

    left, middle, right = [blocks(unit[key]) for key in ('before_centers', 'middle_centers', 'after_centers')]
    if appearance:
        l, r, flank = identity.distance(middle, left), identity.distance(middle, right), identity.distance(left, right)
        raw = min(l, r) - identity.FLANK_PENALTY * flank
    else:
        l, r, flank = feature.distance(middle, left), feature.distance(middle, right), feature.distance(left, right)
        inside = min(float(l[feature.CORE].mean()), float(r[feature.CORE].mean()))
        outside = max(float(l[feature.RING].mean()), float(r[feature.RING].mean()))
        raw = inside - .60 * outside - .35 * float(flank[feature.CORE].mean())
    require(math.isfinite(raw) and -4 <= raw <= 4, 'Invalid fixed-window source response')
    return float(raw)


def supported_window(unit, boxes):
    indices = sorted({i for key in ('sampled_frames', 'context_frames') for i in unit['location'][key]})
    return set(indices) <= set(boxes) and all(overlap(boxes[a], boxes[b]) >= .1 for a, b in zip(indices, indices[1:], strict=False))


def measure(case, plan, yunet, gend_model, dino, recognizer, device, *, repeat_views=(), source_rgb_sha256=None):
    import cv2

    from app.detectors.image import gend
    from app.detectors.image.faces import detect_faces
    from app.detectors.video import face_appearance as appearance
    from app.detectors.video.decode import FrameTable, decode_frames
    from app.detectors.video.face_sequence import SIZE, patch_features

    table = FrameTable.from_json(case['table'])
    require(plan['id'] == case['planned']['id'] and plan['sha256'] == case['planned']['sha256'], 'Wrong prepared source case')
    scheduled = sorted({i for unit in plan['units'] for key in ('sampled_frames', 'context_frames') for i in unit['location'][key]})
    frames, boxes, crops, aligned, vectors, bindings = {}, {}, {}, {}, {}, []
    counters = {'face_detection_inputs': 0, 'crop_classifier_inputs': 0,
        'appearance_inputs': 0, 'patch_feature_inputs': 0, 'native_repeat_inputs': 0}
    repeats = []
    for frame in decode_frames(case['path'], table, scheduled, timeout_s=120):
        i, rgb = frame.index, frame.rgb
        require(i not in frames and i in scheduled, 'Duplicate or unscheduled native input')
        source_rgb = sha_array(rgb)
        if source_rgb_sha256 is not None:
            require(source_rgb == source_rgb_sha256[i], 'Native model input differs from the decoded source audit')
        frames[i] = True
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        counters['face_detection_inputs'] += 1
        faces = detect_faces(bgr, detector=yunet, min_size=48)
        proof = {'frame_index': i, 'pts': frame.pts, 'time_base': table.time_base,
            'source_rgb_sha256': source_rgb, 'detected_faces': len(faces), 'status': 'not_applicable'}
        bindings.append(proof)
        if len(faces) != 1:
            proof['reason'] = 'Single-visible-face input requirement not met; no identity/face-authorship assumption.'
            continue
        face = faces[0]
        transform = gend.align_transform(face.landmarks)
        if transform is None:
            proof['reason'] = 'Degenerate crop alignment; no invented negative response.'
            continue
        native_crop = gend.warp(rgb, *transform)
        aligned[i] = native_crop
        crops[i] = cv2.resize(native_crop, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
        boxes[i] = face.box()
        proof.update(status='ok', bbox=list(face.box()), landmarks=face.landmarks,
            crop_alignment_hex=[[float(value).hex() for value in row] for row in transform[0]],
            crop_alignment_side=int(transform[1]), crop_rgb_sha256=sha_array(native_crop),
            patch_crop_rgb_sha256=sha_array(crops[i]))
        try:
            detection = appearance.face_row(face.landmarks)
            detection[0, :4] = [face.x, face.y, face.w, face.h]
            detection[0, 14] = face.score
            matrix = appearance.cpp_alignment(face.landmarks)
        except ValueError as error:
            proof.update(appearance_status='not_applicable', appearance_reason=str(error))
            continue
        actual = recognizer.alignCrop(bgr, detection)
        independent = cv2.warpAffine(bgr, matrix, (112, 112), flags=cv2.INTER_LINEAR)
        require(np.array_equal(actual, independent), 'Native appearance alignment differs from its independent transcription')
        vector = recognizer.feature(actual).reshape(-1).copy()
        counters['appearance_inputs'] += 1
        # A model returning a degenerate/nonfinite vector is a failed native
        # contract, not a geometric quality exclusion.
        appearance.unit(vector)
        vectors[i] = vector
        proof.update(appearance_status='ok', appearance_crop_bgr_sha256=sha_array(actual),
            appearance_alignment_hex=[[float(value).hex() for value in row] for row in matrix],
            appearance_feature_sha256=sha_array(vector))
        if 'appearance' in repeat_views and 'appearance' not in repeats:
            repeated = recognizer.feature(actual).reshape(-1).copy()
            counters['native_repeat_inputs'] += 1
            require(np.array_equal(vector, repeated), 'Appearance native repeat changed bitwise')
            repeats.append('appearance')
    require(set(frames) == set(scheduled), 'A planned native frame was not decoded')
    by_frame = {row['frame_index']: row for row in bindings}
    crop_values = {}
    eligible = [i for i in plan['centers'] if i in aligned]
    for start in range(0, len(eligible), 8):
        indices = eligible[start:start + 8]
        batch = gend.to_batch([aligned[i] for i in indices])
        values = gend.logits(gend_model, batch, device)
        require(len(values) == len(indices) and all(math.isfinite(x) and -100 <= x <= 100 for x in values),
            'Invalid original crop classifier response')
        counters['crop_classifier_inputs'] += len(indices)
        for index, raw, tensor in zip(indices, values, batch.numpy(), strict=True):
            crop_values[index] = raw
            by_frame[index]['crop_tensor_sha256'] = sha_array(tensor)
        if 'crop' in repeat_views and 'crop' not in repeats:
            repeated = gend.logits(gend_model, batch[:1].clone(), device)[0]
            counters['native_repeat_inputs'] += 1
            require(abs(values[0] - repeated) <= 1e-4, 'Crop native batch/single-input repeat failed')
            repeats.append('crop')
    patches = patch_features(crops, dino)
    counters['patch_feature_inputs'] += len(crops)
    for i, value in patches.items():
        require(value.shape == (64, 384) and value.dtype == np.float32 and np.isfinite(value).all(), 'Invalid native patch feature')
        by_frame[i]['patch_feature_sha256'] = sha_array(value)
    if patches and 'features' in repeat_views:
        first = min(patches)
        repeated = patch_features({first: crops[first]}, dino)[first]
        counters['native_repeat_inputs'] += 1
        require(np.allclose(patches[first], repeated, atol=1e-4, rtol=1e-4), 'Patch native batch/single-input repeat failed')
        repeats.append('features')
    results = []
    for unit in plan['units']:
        view = unit['view']
        if view == 'face-crop':
            raw = crop_values.get(unit['location']['sampled_frames'][0])
        elif not supported_window(unit, boxes):
            raw = None
        else:
            raw = fixed_change(unit, vectors if view == 'face-appearance' else patches, appearance=view == 'face-appearance')
        results.append({'id': unit['id'], 'view': view, 'location': unit['location'],
            'status': 'ok' if raw is not None else 'not_applicable', 'raw': raw,
            **({} if raw is not None else {'reason': 'Planned single-face/alignment/geometric context unavailable; no originality verdict.'})})
    return {'id': plan['id'], 'input_sha256': plan['sha256'], 'native_table_sha256': plan['source_geometry']['pts_table_sha256'],
        'units': results, 'frame_inputs': bindings, 'native_input_counts': counters, 'native_repeats': repeats,
        'scope': 'Fixed single-visible-face source-frame evidence. No person identification, pixel-authorship, lip-sync or whole-video originality claim.'}
