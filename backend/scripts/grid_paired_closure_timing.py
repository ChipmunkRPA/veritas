# SPDX-License-Identifier: Apache-2.0
"""Compare a locally observed mouth event and matching speech with an earlier copy."""

from __future__ import annotations

import hashlib
from fractions import Fraction

import numpy as np
from scipy.signal import correlate

SPEC = {
    "video_rate": 25, "sample_rate": 16000,
    "visual_offsets": [-2, 4], "visual_search_pictures": 10,
    "closure_max_gap": .055, "minimum_reopening": .032,
    "minimum_visual_separation_frames": 3, "maximum_visual_template_error": .045,
    "minimum_visual_margin": .004,
    "audio_offsets_samples": [640, 3200], "audio_search_samples": 7680,
    "minimum_audio_correlation": .90, "minimum_audio_margin": .07,
    "audio_alternative_exclusion_samples": 640,
    "review_offset_minimum_seconds": "3/25",
    "signed_convention": "Positive when the picture event moved later than its matching speech.",
}


def aperture(series):
    return [None if item is None else float(np.median(item[:3])) for item in series]


def template(series, index):
    first, last = (index+v for v in SPEC["visual_offsets"])
    if first < 0 or last >= len(series):
        return None
    values = series[first:last+1]
    if any(value is None for value in values):
        return None
    return np.asarray(values)


def visual_shift(original, submitted, anchor):
    a, b = aperture(original), aperture(submitted)
    reference = template(a,anchor)
    center = -SPEC["visual_offsets"][0]
    if reference is None or reference[center] > SPEC["closure_max_gap"] or max(reference[center+1:])-reference[center] < SPEC["minimum_reopening"]:
        return None
    reference = reference-reference.mean()
    radius = SPEC["visual_search_pictures"]
    locations = []
    for i in range(max(0,anchor-radius),min(len(b),anchor+radius+1)):
        value = template(b,i)
        if value is None or value[center]>SPEC["closure_max_gap"] or max(value[center+1:])-value[center]<SPEC["minimum_reopening"]:
            continue
        err = float(np.sqrt(np.mean(((value-value.mean())-reference)**2)))
        locations.append((err,i))
    if not locations:
        return None
    error, result = min(locations)
    other = min((err for err,i in locations if abs(i-result)>=SPEC["minimum_visual_separation_frames"]),default=1.)
    if error>SPEC["maximum_visual_template_error"] or other-error<SPEC["minimum_visual_margin"]:
        return None
    return {"source_frame":anchor,"submitted_frame":result,"offset_pictures":result-anchor,
            "shape_error":error,"shape_margin":other-error}


def audio_shift(original, submitted, anchor):
    rate = SPEC["sample_rate"]
    anchor_pcm = anchor*(rate//SPEC["video_rate"])
    start, stop = [anchor_pcm+offset for offset in SPEC["audio_offsets_samples"]]
    if not 0<=start<stop<=len(original):
        return None
    wanted = original[start:stop].astype(np.float64)
    wanted -= wanted.mean()
    radius = SPEC["audio_search_samples"]
    low, high = max(0,start-radius),min(len(submitted)-len(wanted),start+radius)
    if low>high or np.linalg.norm(wanted)<1:
        return None
    signal = submitted[low:high+len(wanted)].astype(np.float64)
    n = len(wanted)
    sums = np.r_[0,np.cumsum(signal)]
    sq = np.r_[0,np.cumsum(signal*signal)]
    denom = np.linalg.norm(wanted)*np.sqrt(np.maximum(0.,sq[n:]-sq[:-n]-(sums[n:]-sums[:-n])**2/n))
    scores = np.clip(correlate(signal,wanted,mode="valid",method="fft")/np.maximum(denom,1e-12),-1.,1.)
    selected = int(scores.argmax())
    best = float(scores[selected])
    alternative = np.abs(np.arange(len(scores))-selected)>=SPEC["audio_alternative_exclusion_samples"]
    next_best = float(scores[alternative].max()) if alternative.any() else 0.
    if best<SPEC["minimum_audio_correlation"] or best-next_best<SPEC["minimum_audio_margin"]:
        return None
    target = selected+low
    return {"source_samples":[start,stop],"submitted_samples":[target,target+n],"offset_samples":target-start,
            "correlation":best,"correlation_margin":best-next_best,
            "source_pcm_sha256":hashlib.sha256(original[start:stop].tobytes()).hexdigest(),
            "submitted_pcm_sha256":hashlib.sha256(submitted[target:target+n].tobytes()).hexdigest()}


def compare(source_motion, submitted_motion, source_pcm, submitted_pcm, source_event):
    video = visual_shift(source_motion,submitted_motion,source_event)
    sound = audio_shift(source_pcm,submitted_pcm,source_event)
    if video is None or sound is None:
        return {"status":"unassessed","visual":video,"sound":sound}
    v_rate, a_rate = SPEC["video_rate"],SPEC["sample_rate"]
    v_shift, a_shift = Fraction(video["offset_pictures"],v_rate),Fraction(sound["offset_samples"],a_rate)
    difference = v_shift-a_shift
    disposition = "timing_difference" if abs(difference)>=Fraction(SPEC["review_offset_minimum_seconds"]) else "no_large_timing_difference"
    v = Fraction(video["submitted_frame"],v_rate)
    counterpart = Fraction(video["source_frame"],v_rate)+a_shift
    return {"status":disposition,"visual":video,"sound":sound,"picture_minus_audio_shift_s":str(difference),
            "event_time_range_s":[str(min(v,counterpart)),str(max(v,counterpart))]}
