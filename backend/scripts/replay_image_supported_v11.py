"""Replay actual production postprocessing from retained model responses.

No weights are loaded and no neural forward runs. PNG invokes the current
SpliceDetector with a checked cached prediction; JPEG invokes FusedDetector with
the original cached map and whole-image head. Raw maps survive selection gates.
"""
from __future__ import annotations

import hashlib
import io
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image

from scripts.prepare_image_supported_v11 import BRANCHES, require, route


def replay(data, pixels, branch, native, *, primary=None, boxes=None):
    require(branch in BRANCHES and route(data) == branch, 'Actual input bytes select the supported production branch')
    require(pixels.ndim == 3 and pixels.shape[2] == 3 and pixels.dtype == np.uint8,
        'Original RGB uint8 source pixels required')
    height, width = pixels.shape[:2]
    with Image.open(io.BytesIO(data)) as original:
        require(original.getexif().get(274, 1) == 1
            and np.array_equal(np.asarray(original.convert('RGB')), pixels), 'Cached inference is not bound to exact decoded source pixels')
    side = BRANCHES[branch]['raw_side']
    require(native.dtype == np.float32 and native.shape == (side, side) and np.isfinite(native).all()
        and native.min() >= 0 and native.max() <= 1, 'Invalid original native model map')
    if boxes is None:
        boxes = [(x0, y0, x1 - x0, y1 - y0) for y0, y1 in ((0, height // 2), (height // 2, height))
            for x0, x1 in ((0, width // 2), (width // 2, width))]
    from app.detectors.image import fused, localiser, splice
    from app.detectors.image.adapters.localisers import FusedDetector, SpliceDetector
    from scripts.localization_benchmark import component_masks

    def checked_image(image):
        require(image.size == (width, height) and np.array_equal(np.asarray(image.convert('RGB')), pixels),
            'Production postprocessing supplied another source image to the cached response')

    if branch == 'png':
        require(primary is None, 'PNG primary response is measured native area, not a borrowed image probability')
        detector, model, revision = SpliceDetector(), object(), splice.CHECKPOINT_SHA256

        def cached_png(_model, image):
            checked_image(image)
            return native

        context = SimpleNamespace(media=SimpleNamespace(original_bytes=lambda: data),
            lease=SimpleNamespace(get=lambda _identifier: SimpleNamespace(obj=model, revision=revision)))
        with patch.object(splice, 'predict', side_effect=cached_png) as cached:
            output = detector.localise(context, pixels)
            require(cached.call_count == 1, 'Expected exactly one cached PNG response request')
        primary = output.candidate_eligibility['score']
    else:
        require(not isinstance(primary, bool) and isinstance(primary, int | float)
            and np.isfinite(primary) and 0 <= primary <= 1, 'Original JPEG whole-image response required')
        detector, revision = FusedDetector(), fused.CHECKPOINT_SHA256
        calls = []

        def cached_jpeg(image):
            checked_image(image)
            calls.append(True)
            return fused.FusedPrediction(native, float(primary), (width, height))

        model = SimpleNamespace(predict=cached_jpeg)
        context = SimpleNamespace(media=SimpleNamespace(original_bytes=lambda: data),
            lease=SimpleNamespace(get=lambda _identifier: SimpleNamespace(obj=model, revision=revision)))
        output = detector.localise(context, pixels)
        require(len(calls) == 1, 'Expected exactly one cached JPEG response request')
    # Reconstruct selected connected pixels, never fill their enclosing boxes.
    recorded = np.rint(np.clip(output.anomaly_map.astype(np.float64), 0, 1) * 255) / 255.
    binary = (recorded >= detector.REGION_THRESHOLD) & output.reliable
    policy = {'connectivity': 8, 'minimum_region_fraction': localiser.MIN_REGION_FRACTION,
        'localiser_max_regions': localiser.MAX_REGIONS, 'per_map_display_max_regions': localiser.MAX_REGIONS}
    masks, facts = component_masks(binary, width=width, height=height, rule=policy, values=recorded)
    selected = masks['localiser_region_cap']
    eligible = output.candidate_eligibility['eligible']
    expected_regions = min(facts['qualifying_component_count'], localiser.MAX_REGIONS) if eligible else 0
    require(len(output.regions) == expected_regions, 'Component reconstruction differs from actual production method')
    if not eligible:
        selected = np.zeros_like(selected)
    source_selection = cv2.resize(selected.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST) > 0
    projected = cv2.resize(native, (width, height), interpolation=cv2.INTER_NEAREST)
    values = {'primary': float(primary), 'integrity': float(output.integrity_score)}
    for index, (x, y, w, h) in enumerate(boxes):
        require(all(type(v) is int for v in (x, y, w, h)) and min(w, h) > 0
            and 0 <= x < x + w <= width and 0 <= y < y + h <= height, 'Regional readout escapes source coordinates')
        values[f'quadrant.{index}'] = float(projected[y:y + h, x:x + w].mean(dtype=np.float64))
    require(all(np.isfinite(value) and 0 <= value <= 1 for value in values.values()), 'Nonfinite or out-of-range actual response')
    packet = {'schema': 'chipmunk.image-supported-postprocessing/11', 'format': branch,
        'input_sha256': hashlib.sha256(data).hexdigest(), 'rgb_sha256': hashlib.sha256(pixels.tobytes()).hexdigest(),
        'native_map_sha256': hashlib.sha256(native.tobytes()).hexdigest(), 'raw_values': values,
        'current_region_flag': bool(output.regions), 'current_region_count': len(output.regions),
        'current_regions': output.regions, 'current_gate': output.candidate_eligibility,
        'current_grid_shape': list(output.anomaly_map.shape), 'current_reliable_cells': int(output.reliable.sum()),
        'current_selected_source_pixels': int(source_selection.sum()), 'source_width': width, 'source_height': height,
        'new_neural_forwards': 0, 'weights_loaded': False,
        'scope': 'Actual localiser postprocessing/component selection only; excludes competing evidence packets, whole-generation checks and the full application verdict.'}
    return packet, source_selection
