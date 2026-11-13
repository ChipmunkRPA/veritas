# SPDX-License-Identifier: Apache-2.0
"""Compare a locally observed mouth event and matching speech with an earlier copy."""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import importlib.util
from contextlib import contextmanager
from fractions import Fraction

import numpy as np
from scipy.signal import correlate

SPEC = {
    "video_rate": 25,
    "sample_rate": 16000,
    "visual_offsets": [-2, 4],
    "visual_search_pictures": 10,
    "closure_max_gap": 0.055,
    "minimum_reopening": 0.032,
    "minimum_visual_separation_frames": 3,
    "maximum_visual_template_error": 0.045,
    "minimum_visual_margin": 0.004,
    "audio_offsets_samples": [640, 3200],
    "audio_search_samples": 7680,
    "minimum_audio_correlation": 0.90,
    "minimum_audio_margin": 0.07,
    "audio_alternative_exclusion_samples": 640,
    "review_offset_minimum_seconds": "3/25",
    "signed_convention": "Positive when the picture event moved later than its matching speech.",
}
SHORT_AUDIO_SPEC = SPEC | {
    "audio_offsets_samples": [0, 960],
    "audio_search_samples": 6400,
    "minimum_audio_correlation": 0.94,
    "minimum_audio_margin": 0.08,
}
SHORT_WAVEFORM_BASIS = "short_marked_excerpt"
LOCAL_SOUND_METHOD = "marked_local_sound_shift/1"
LOCAL_SOUND_SCOPE = (
    "The decoded pictures match. A marked mouth event anchors a distinctive sound excerpt that moved within "
    "the analyzed clip, with identical decoded sound before and after the change. Ordinary editing can cause "
    "this result. This evidence carries no AI confidence weight."
)
LOCAL_SOUND_UNASSESSED = (
    "The marked timing check needs matching upright 360 x 288 clips with 75 native pictures at 25 fps and continuous sound. "
    "For some short 720 x 480 pairs, it can also report a locally moved sound excerpt when every picture matches."
)


def aperture(series):
    return [None if item is None else float(np.median(item[:3])) for item in series]


def template(series, index):
    first, last = (index + v for v in SPEC["visual_offsets"])
    if first < 0 or last >= len(series):
        return None
    values = series[first : last + 1]
    if any(value is None for value in values):
        return None
    return np.asarray(values)


def visual_shift(original, submitted, anchor):
    a, b = aperture(original), aperture(submitted)
    reference = template(a, anchor)
    center = -SPEC["visual_offsets"][0]
    if (
        reference is None
        or reference[center] > SPEC["closure_max_gap"]
        or max(reference[center + 1 :]) - reference[center] < SPEC["minimum_reopening"]
    ):
        return None
    reference = reference - reference.mean()
    radius = SPEC["visual_search_pictures"]
    locations = []
    for i in range(max(0, anchor - radius), min(len(b), anchor + radius + 1)):
        value = template(b, i)
        if (
            value is None
            or value[center] > SPEC["closure_max_gap"]
            or max(value[center + 1 :]) - value[center] < SPEC["minimum_reopening"]
        ):
            continue
        err = float(np.sqrt(np.mean(((value - value.mean()) - reference) ** 2)))
        locations.append((err, i))
    if not locations:
        return None
    error, result = min(locations)
    other = min(
        (err for err, i in locations if abs(i - result) >= SPEC["minimum_visual_separation_frames"]),
        default=1.0,
    )
    if error > SPEC["maximum_visual_template_error"] or other - error < SPEC["minimum_visual_margin"]:
        return None
    return {
        "source_frame": anchor,
        "submitted_frame": result,
        "offset_pictures": result - anchor,
        "shape_error": error,
        "shape_margin": other - error,
    }


def audio_shift(original, submitted, anchor, *, spec=SPEC):
    rate = spec["sample_rate"]
    anchor_pcm = anchor * (rate // spec["video_rate"])
    start, stop = [anchor_pcm + offset for offset in spec["audio_offsets_samples"]]
    if not 0 <= start < stop <= len(original):
        return None
    wanted = original[start:stop].astype(np.float64)
    wanted -= wanted.mean()
    radius = spec["audio_search_samples"]
    low, high = max(0, start - radius), min(len(submitted) - len(wanted), start + radius)
    if low > high or np.linalg.norm(wanted) < 1:
        return None
    signal = submitted[low : high + len(wanted)].astype(np.float64)
    n = len(wanted)
    sums = np.r_[0, np.cumsum(signal)]
    sq = np.r_[0, np.cumsum(signal * signal)]
    denom = np.linalg.norm(wanted) * np.sqrt(
        np.maximum(0.0, sq[n:] - sq[:-n] - (sums[n:] - sums[:-n]) ** 2 / n)
    )
    scores = np.clip(
        correlate(signal, wanted, mode="valid", method="fft") / np.maximum(denom, 1e-12), -1.0, 1.0
    )
    selected = int(scores.argmax())
    best = float(scores[selected])
    alternative = np.abs(np.arange(len(scores)) - selected) >= spec["audio_alternative_exclusion_samples"]
    next_best = float(scores[alternative].max()) if alternative.any() else 0.0
    if best < spec["minimum_audio_correlation"] or best - next_best < spec["minimum_audio_margin"]:
        return None
    target = selected + low
    return {
        "source_samples": [start, stop],
        "submitted_samples": [target, target + n],
        "offset_samples": target - start,
        "correlation": best,
        "correlation_margin": best - next_best,
        "source_pcm_sha256": hashlib.sha256(original[start:stop].tobytes()).hexdigest(),
        "submitted_pcm_sha256": hashlib.sha256(submitted[target : target + n].tobytes()).hexdigest(),
    }


def compare(source_motion, submitted_motion, source_pcm, submitted_pcm, source_event, *, short_fallback=False):
    video = visual_shift(source_motion, submitted_motion, source_event)
    sound = audio_shift(source_pcm, submitted_pcm, source_event)
    basis = {}
    if video is not None and sound is None and short_fallback:
        sound = audio_shift(source_pcm, submitted_pcm, source_event, spec=SHORT_AUDIO_SPEC)
        if sound is not None:
            basis["waveform_basis"] = SHORT_WAVEFORM_BASIS
    if video is None or sound is None:
        return {"status": "unassessed", "visual": video, "sound": sound}
    v_rate, a_rate = SPEC["video_rate"], SPEC["sample_rate"]
    v_shift, a_shift = Fraction(video["offset_pictures"], v_rate), Fraction(sound["offset_samples"], a_rate)
    difference = v_shift - a_shift
    disposition = (
        "timing_difference"
        if abs(difference) >= Fraction(SPEC["review_offset_minimum_seconds"])
        else "no_large_timing_difference"
    )
    v = Fraction(video["submitted_frame"], v_rate)
    counterpart = Fraction(video["source_frame"], v_rate) + a_shift
    return {
        "status": disposition,
        "visual": video,
        "sound": sound,
        "picture_minus_audio_shift_s": str(difference),
        "event_time_range_s": [str(min(v, counterpart)), str(max(v, counterpart))],
        **basis,
    }


MODEL_SHA256 = "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff"
RUNTIME_VERSION = "0.10.32"
SCOPE = "One marked mouth event on verified matching native pictures and sound. Ordinary editing can cause timing differences. This evidence carries no AI confidence weight."
UNASSESSED = {
    "profile": "The marked timing check needs matching upright 360 x 288 clips with 75 native pictures at 25 fps and continuous sound.",
    "reference": "The clips share too few exact native pictures for this marked timing check.",
    "marker": "The marked time leaves too few native pictures or sound samples to compare.",
    "runtime": "The optional local timing check is unavailable.",
    "support": "The marked closure or matching speech did not have enough support for a timing estimate.",
}


def available(model):
    try:
        return bool(
            model is not None
            and model.is_file()
            and importlib.util.find_spec("mediapipe") is not None
            and importlib.metadata.version("mediapipe") == RUNTIME_VERSION
            and _model_bytes(model)
        )
    except (OSError, ImportError, ValueError):
        return False


def _model_bytes(model):
    payload = model.read_bytes()
    if hashlib.sha256(payload).hexdigest() != MODEL_SHA256:
        raise ValueError("The local timing task does not match the tested artifact")
    return payload


def _mesh(task, mp, rgb):
    faces = task.detect(
        mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
    ).face_landmarks
    if len(faces) != 1 or len(faces[0]) != 478:
        return None
    indices = (1, 13, 14, 33, 78, 82, 87, 133, 263, 308, 312, 317, 362)
    h, w = rgb.shape[:2]
    values = np.asarray([(faces[0][i].x * w, faces[0][i].y * h) for i in indices], dtype=np.float64)
    if not np.isfinite(values).all() or (values < 0).any() or (values >= [w, h]).any():
        return None
    p = dict(zip(indices, values, strict=True))
    left, right = (p[33] + p[133]) / 2, (p[362] + p[263]) / 2
    eye = right - left
    scale = float(np.linalg.norm(eye))
    if scale < 12:
        return None
    ex = eye / scale
    ey = np.array([-ex[1], ex[0]])
    gaps = [float((p[lower] - p[upper]) @ ey) for upper, lower in ((82, 87), (13, 14), (312, 317))]
    mouth_width = abs(float((p[308] - p[78]) @ ex))
    if mouth_width < 8 or min(gaps) < -1 or float((p[1] - (left + right) / 2) @ ey) <= 0:
        return None
    points = np.array([p[i] for i in (78, 82, 87, 13, 14, 308, 312, 317)])
    margin = max(4.0, mouth_width * 0.22)
    x0, y0 = np.maximum(0, np.floor(points.min(axis=0) - margin)).astype(int)
    x1, y1 = np.minimum((w, h), np.ceil(points.max(axis=0) + margin)).astype(int)
    return np.asarray([*(max(0.0, gap) for gap in gaps), mouth_width]) / scale, [
        int(x) for x in (x0, y0, x1, y1)
    ]


def _measure(path, table, wanted, task, mp):
    from contextlib import closing

    from app.detectors.video.decode import decode_frames

    series = [None] * len(table)
    pictures = {}
    with closing(decode_frames(path, table, wanted, timeout_s=20)) as stream:
        for frame in stream:
            if not frame.pts_verified or frame.pts != table.pts[frame.index]:
                raise ValueError("Marked timing picture lacks its exact native timestamp")
            value = _mesh(task, mp, frame.rgb)
            if value is not None:
                series[frame.index], box = value
                pictures[frame.index] = frame.rgb, box
    return series, pictures


def _assets(side, frame_index, table, pixels, sound, sample_range, assets, save_image):
    from app.detectors.audio.decode import wav_bytes
    from app.detectors.audio.waveform import envelope
    from app.evidence.audio_reference import _wav_asset

    rgb, box = pixels[frame_index]
    x0, y0, x1, y1 = box
    crop = rgb[y0:y1, x0:x1]
    start, stop = sample_range
    excerpt = sound[start:stop]
    as_float = excerpt.astype(np.float32) / 32768
    return {
        "frame_index": frame_index,
        "picture_time_s": float(table.pts[frame_index] * table.tb),
        "picture_pts": table.pts[frame_index],
        "picture_time_base": table.time_base,
        "native_picture_rgb_sha256": hashlib.sha256(rgb.tobytes()).hexdigest(),
        "crop_xyxy": box,
        "native_crop_rgb_sha256": hashlib.sha256(crop.tobytes()).hexdigest(),
        "pcm_start_s": start / 16000,
        "pcm_end_s": stop / 16000,
        "pcm_samples": [start, stop],
        "pcm_s16_sha256": hashlib.sha256(excerpt.tobytes()).hexdigest(),
        "crop": save_image(assets, f"timing-{side}-crop.png", crop),
        "speech": _wav_asset(assets, f"timing-{side}-speech.wav", wav_bytes(as_float)),
        "waveform": envelope(as_float),
    }


def local_sound_match(first, last, mark_sample):
    """Find an exact moved excerpt inside a short change with identical flanks."""
    if len(first) != len(last):
        return None
    changed = np.flatnonzero(first != last)
    if not changed.size:
        return None
    begin, end = int(changed[0]), int(changed[-1]) + 1
    rate = SPEC["sample_rate"]
    flank = 60 * rate // 1000
    if begin < flank or end > len(first) - flank or end - begin > 800 * rate // 1000:
        return None
    for spec in (SPEC, SHORT_AUDIO_SPEC):
        offsets = [mark_sample + offset for offset in spec["audio_offsets_samples"]]
        sound = audio_shift(first, last, 0, spec=spec | {"audio_offsets_samples": offsets})
        if sound is None or not Fraction(SPEC["review_offset_minimum_seconds"]) * rate <= abs(sound["offset_samples"]) <= 360 * rate // 1000:
            continue
        a, b = sound["source_samples"]
        c, d = sound["submitted_samples"]
        if begin <= c < d <= end and np.array_equal(first[a:b], last[c:d]):
            return sound, spec is SHORT_AUDIO_SPEC, [begin, end]
    return None


def assess_local_sound(paths, tables, tracks, earlier_time, model, assets, save_image):
    """Measure one local audio move in a short pair of identical pictures."""
    earlier, submitted = tables
    if not 48 <= len(earlier) <= 75 or any(
        table.rotation or table.pts[0] or (table.width, table.height) != (720, 480) for table in tables
    ):
        return None
    clock = [tick * earlier.tb for tick in earlier.pts]
    step = clock[1]
    if not Fraction(1, 25) <= step <= Fraction(2, 47) or any(
        t != i * step for i, t in enumerate(clock)
    ):
        return None
    if any(status != "measured" or info["relative_start"] != "0" or not 24000 <= info["samples"] <= 54400
           for status, _, info in tracks):
        return None
    anchor = min(range(len(clock)), key=lambda i: (abs(clock[i] - earlier_time), i))
    if not 2 <= anchor < len(clock) - 4 or abs(clock[anchor] - earlier_time) > step / 2:
        return None
    first, last = (np.rint(values * 32768).astype("<i2") for _, values, _ in tracks)
    match = local_sound_match(first, last, round(clock[anchor] * SPEC["sample_rate"]))
    if match is None or not available(model):
        return None
    sound, short, changed = match
    audio_end = max(sound["source_samples"][1], sound["submitted_samples"][1])
    if Fraction(audio_end, SPEC["sample_rate"]) > clock[-1]:
        return None
    try:
        with _landmarker(model) as (mp, task):
            wanted = range(max(0, anchor - 12), min(len(clock), anchor + 15))
            motion, pixels = _measure(paths[0], earlier, wanted, task, mp)
    except (AttributeError, OSError, ImportError, RuntimeError, ValueError):
        return None
    visual = visual_shift(motion, motion, anchor)
    if visual is None or visual["submitted_frame"] != anchor:
        return None
    shift = Fraction(sound["offset_samples"], SPEC["sample_rate"])
    direction = "later" if shift > 0 else "earlier"
    return {
        "status": "timing_difference",
        "method": LOCAL_SOUND_METHOD,
        "summary": f"Matching sound moved {float(abs(shift) * 1000):.1f} ms {direction} while the marked mouth event stayed in the same frame.",
        "scope": LOCAL_SOUND_SCOPE,
        "requested_earlier_time_s": str(earlier_time),
        "affects_overall_confidence": False,
        "offset_ms": float(-shift * 1000),
        "event_range_s": [sample / SPEC["sample_rate"] for sample in sound["submitted_samples"]],
        "changed_audio_samples": changed,
        "waveform_basis": SHORT_WAVEFORM_BASIS if short else "standard_marked_excerpt",
        "earlier": _assets("earlier", anchor, earlier, pixels, first, sound["source_samples"], assets, save_image),
        "submitted": _assets("submitted", anchor, submitted, pixels, last, sound["submitted_samples"], assets, save_image),
    }


@contextmanager
def _landmarker(model):
    mp = importlib.import_module("mediapipe")
    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_buffer=_model_bytes(model), delegate=mp.tasks.BaseOptions.Delegate.CPU
        ),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_faces=2,
    )
    with mp.tasks.vision.FaceLandmarker.create_from_options(options) as task:
        yield mp, task


def assess(paths, tables, tracks, matching_pictures, earlier_time, model, assets, save_image, *, short_fallback=False):
    def unavailable(reason):
        return {
            "status": "unassessed",
            "summary": UNASSESSED[reason],
            "reason": reason,
            "scope": SCOPE,
            "requested_earlier_time_s": str(earlier_time),
            "affects_overall_confidence": False,
        }

    if (
        tables is None
        or tracks is None
        or any(
            (table.width, table.height, len(table), table.rotation) != (360, 288, 75, 0)
            or table.interpolated
            or table.truncated
            or [p * table.tb for p in table.pts] != [Fraction(i, 25) for i in range(75)]
            for table in tables
        )
        or any(
            status != "measured" or info["relative_start"] != "0" or info["samples"] != 47648
            for status, _, info in tracks
        )
    ):
        return unavailable("profile")
    if matching_pictures < 57:
        return unavailable("reference")
    anchor = min(range(75), key=lambda i: (abs(Fraction(i, 25) - earlier_time), i))
    if not 2 <= anchor <= 70 or abs(Fraction(anchor, 25) - earlier_time) > Fraction(1, 50):
        return unavailable("marker")
    if not available(model):
        return unavailable("runtime")
    try:
        with _landmarker(model) as (mp, task):
            original, left = _measure(paths[0], tables[0], range(anchor - 2, anchor + 5), task, mp)
            lower, upper = max(0, anchor - 12), min(75, anchor + 15)
            submitted, right = _measure(paths[1], tables[1], range(lower, upper), task, mp)
    except (AttributeError, OSError, ImportError, RuntimeError, ValueError):
        return unavailable("runtime")
    first, last = (np.rint(values * 32768).astype("<i2") for _, values, _ in tracks)
    result = compare(original, submitted, first, last, anchor, short_fallback=short_fallback)
    if result["status"] == "unassessed":
        return unavailable("support")
    delta = Fraction(result["picture_minus_audio_shift_s"])
    value = abs(delta * 1000)
    if result["status"] == "timing_difference":
        direction = "later" if delta > 0 else "earlier"
        summary = (
            f"The marked mouth event moved {float(value):g} ms {direction} relative to the matching speech."
        )
    else:
        summary = (
            "The marked mouth event and its matching speech stayed close to their earlier relative timing."
        )
    visual, sound = result["visual"], result["sound"]
    basis = {}
    scope = SCOPE
    if result.get("waveform_basis") == SHORT_WAVEFORM_BASIS:
        basis["waveform_basis"] = SHORT_WAVEFORM_BASIS
        scope += " The timing check used a shorter speech excerpt beginning at the marked frame."
    return {
        "status": result["status"],
        "summary": summary,
        "scope": scope,
        **basis,
        "requested_earlier_time_s": str(earlier_time),
        "affects_overall_confidence": False,
        "offset_ms": float(delta * 1000),
        "event_range_s": [float(Fraction(x)) for x in result["event_time_range_s"]],
        "earlier": _assets(
            "earlier",
            visual["source_frame"],
            tables[0],
            left,
            first,
            sound["source_samples"],
            assets,
            save_image,
        ),
        "submitted": _assets(
            "submitted",
            visual["submitted_frame"],
            tables[1],
            right,
            last,
            sound["submitted_samples"],
            assets,
            save_image,
        ),
    }
