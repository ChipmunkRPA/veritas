# SPDX-License-Identifier: MIT
"""Bind local lip timing measurements to the frozen video and audio clocks."""

from __future__ import annotations

import re
from fractions import Fraction

from app.core.canonical import canonical_hash
from app.evidence import video_silent_mouth as silent
from app.evidence import video_viseme_ranked as ranked
from app.evidence import video_viseme_stereo as stereo
from app.evidence import video_viseme_timing as timing

KEY = "lip_timing_review"
COMMITMENT = "lip_timing_review_sha256"
SCHEMA = "chipmunk.video-lip-timing/1"
SCHEMA_WITH_APERTURE = "chipmunk.video-lip-timing/2"
SCHEMA_WITH_ARTICULATION = "chipmunk.video-lip-timing/3"
SCHEMA_WITH_STEREO = "chipmunk.video-lip-timing/4"
SCHEMA_WITH_SILENCE = "chipmunk.video-lip-timing/5"
PACKET_ID = "VLT01"
CLIP, WAVEFORM = "lip_timing_audio", "lip_timing_waveform"
REFERENCE = {"closure": {"median": .000578}, "rounded": {"median": .3973106}, "spread": {"median": .9265317}}
POINT = .9279881
ROLES = ("before", "during", "after")
LIMITATION = "Ordinary dubbing, recording delays and offscreen speech can produce similar timing patterns. This measurement carries zero weight in the AI assessment."
ARTICULATION_LIMITATION = (
    "Ordinary dubbing, offscreen speech and variation in how people form sounds can produce this pattern. "
    "This measurement carries zero weight in the AI assessment."
)
# Fit to unchanged Lombard Grid recordings, with a separate speaker
# calibration. The evaluation recipe reproduces these parameters.
ARTICULATION_POINT = .4560042
ARTICULATION_FIT = {
    "shape": False, "labels": ["closure", "open", "rounded", "spread", "lower_lip"],
    "midpoint": [.07946317, .3921596956, .0318890886], "scale": [.0762334333, .4103160824, .2172820689],
    "coefficients": [
        [-.4867539336, .4776827813, -.0567924924, .2493819306, -.0337309577],
        [-.7636353689, .4054574109, -.1158770341, .4915957125, -.3114714221],
        [-.2815556431, .7406398552, -.4076042161, .4424748409, -.8997715255],
    ],
    "bias": [-.566249122, -.5515752401, -.0883582224, -.2955922023, -.4240498172],
}
STATUS = {
    "matched": "A brief change in the relationship between spoken sounds and mouth opening was measured.",
    "no_match": "No qualifying local timing change was found within the measured source intervals.",
    "insufficient_events": "There were too few comparable speech and mouth observations for a timing result.",
    "ambiguous_change": "Several timings fit the observations, leaving the local change uncertain.",
    "incomplete_context": "The timing estimate lacked complete before, during, and after picture evidence.",
    "unsupported": "The timing trial needs a short, continuous shot and a supported mono soundtrack.",
    "stereo_unresolved": "A reliable timing comparison could not be made from the two soundtrack channels.",
    "assets": "The local timing trial's reference assets were unavailable.",
    "runtime": "The trial could not complete its measurements.",
    "provenance": "Trusted source credentials settled the provenance check. The local timing trial was skipped.",
}
_SHA = re.compile(r"[a-f0-9]{64}")
_UNSET = object()
AAC_PRESENTATION_FIELDS = ("source_codec", "source_time_base", "source_start_pts", "source_duration_pts")


def skipped(reason):
    return {"schema": SCHEMA, "status": reason, "observation": STATUS[reason], "affects_overall_confidence": False}


def presented_samples(mapping):
    """Keep AAC padding outside the timing comparison and its evidence."""
    samples = mapping.get("samples")
    if type(samples) is not int:
        return None
    rate = mapping.get("source_sample_rate")
    if rate == 48_000:
        if (mapping.get("source_codec") != "aac" or mapping.get("source_time_base") != "1/48000"
                or mapping.get("source_start_pts") != 0):
            return None
        duration = mapping.get("source_duration_pts")
        if type(duration) is not int:
            return None
        extent = duration // 3
        # An AAC-LC frame is 1024 native samples, at most 342 output samples.
        if not 0 <= samples - extent <= 342:
            return None
        samples = extent
    elif rate != timing.RATE:
        return None
    return samples if 20_000 <= samples <= 64_000 else None


def audio_origin(mapping, stereo_scope=None):
    extent = presented_samples(mapping)
    if mapping.get("timeline_time_base") != "1/16000" or extent is None:
        return None
    if stereo_scope is None and mapping.get("source_channels") != 1:
        return None
    if stereo_scope is not None and not stereo.qualifies(stereo_scope, mapping, extent):
        return None
    samples = mapping["samples"]
    clock = mapping.get("timeline") or []
    if not clock:
        return None
    first, cursor = clock[0]["source_pts"], 0
    if mapping.get("source_sample_rate") == 48_000 and first != 0:
        return None
    for block in clock:
        a, b, pts = (block[key] for key in ("start_sample", "end_sample", "source_pts"))
        if any(type(value) is not int for value in (a, b, pts)) or not cursor == a < b <= samples or pts != first + a:
            return None
        cursor = b
    return first if cursor == samples else None


def _witness(candidate, pictures, samples, origin, point):
    state = candidate["status"]
    if state != "measured" or candidate["score"] <= point:
        return state, None
    plates = timing.picture_evidence(pictures, candidate, samples)
    frames = plates["picture_brackets"]
    if not all(frames.values()):
        return "incomplete_context", None
    evidence = {}
    if disagreements := candidate.get("disagreements"):
        a, b = plates["picture_span"]
        qualifying = [item for item in disagreements if a <= pictures[item["frame"]]["sample"] < b]
        if not qualifying:
            return "incomplete_context", None
        strongest = min(qualifying, key=lambda item: (item["score"], item["sample"]))
        frame = pictures[strongest["frame"]]
        frames["during"] = frame
        evidence = {"articulation": {"mismatched_events": len(disagreements), "audio_sample": origin + strongest["sample"],
                                     "picture_frame": frame["index"], "picture_pts": frame["pts"]}}
    return "matched", {
        "audio_source_pts": [origin + tick for tick in candidate["span"]],
        "picture_source_samples": [origin + tick for tick in plates["picture_span"]],
        "baseline_lag_samples": candidate["baseline_lag"], "change_samples": candidate["change"],
        "picture_frames": {role: frame["index"] for role, frame in frames.items()},
        **evidence,
    }


def _validate_phones(events, samples, labels):
    last = -1
    for label, at, strength in events:
        if (label not in labels or type(at) is not int or not last < at < samples
                or (at - 200) % 320 or type(strength) not in (int, float) or not .35 <= strength <= 1):
            raise ValueError("Timing input has an invalid speech event")
        last = at


def measured(raw, *, timestamps, time_base, source, mapping, stereo_scope=None):
    origin = audio_origin(mapping, stereo_scope)
    if origin is None:
        raise ValueError("The mixed soundtrack has no supported continuous source clock")
    base = Fraction(time_base)
    rows, geometry = [], []
    articulation = "articulation_phones" in raw
    for i, (pts, row) in enumerate(zip(timestamps, raw["frames"], strict=True)):
        if row["index"] != i or row["pts"] != pts:
            raise ValueError("Timing input picture differs from the source clock")
        gap = row["gap_units"]
        if "shape" in row and (not articulation or gap is None):
            raise ValueError("Unexpected mouth shape without a qualified observation")
        if gap is not None:
            x, y, right, bottom = row["mouth_box"]
            if (type(gap) is not int or gap < 0
                    or any(type(value) is not int for value in (x, y, right, bottom))
                    or not 0 <= x < right <= source["width"] or not 0 <= y < bottom <= source["height"]
                    or not all(_SHA.fullmatch(row.get(key, "")) for key in ("rgb_sha256", "mouth_crop_sha256"))):
                raise ValueError("Timing input has an invalid mouth observation")
            if articulation:
                shape = row.get("shape")
                if (not isinstance(shape, list) or len(shape) != 4
                        or any(type(value) is not int or value < 0 for value in shape)
                        or shape[3] <= 0 or sorted(shape[:3])[1] != gap):
                    raise ValueError("Timing input has an invalid mouth shape observation")
        rows.append({**row, "sample": round(pts * base * timing.RATE) - origin})
        geometry.append({**rows[-1], "gap": None if gap is None else gap / timing.SCORE_UNITS})
    samples = presented_samples(mapping)
    _validate_phones(raw["phones"], samples, timing.PHONES)
    inputs = {"frames": rows, "phones": raw["phones"], "samples": samples}
    picture = {"frames": geometry}
    candidate = timing.hypotheses(timing.compatible(picture, raw["phones"], REFERENCE), samples)
    state, witness = _witness(candidate, geometry, samples, origin, POINT)
    basis = {}
    if "aperture_prediction" in raw:
        prediction = raw["aperture_prediction"]
        if (mapping["source_sample_rate"] != timing.RATE or not isinstance(prediction, list)
                or len(prediction) != (samples - 2 * ranked.CENTER) // ranked.STRIDE + 1
                or any(v is not None and (type(v) is not int or not -ranked.UNIT // 2 <= v <= ranked.UNIT * 3 // 2)
                       for v in prediction)):
            raise ValueError("Aperture estimates differ from the source clock or valid range")
        inputs["aperture_prediction"] = prediction
        basis = {"schema": SCHEMA_WITH_APERTURE, "measurement_basis": "speech_events"}
        if witness is None:
            extra = timing.hypotheses(ranked.scores(picture, prediction), samples)
            _, supported = _witness(extra, geometry, samples, origin, ranked.POINT)
            if supported is not None:
                candidate, state, witness = extra, "matched", supported
                basis["measurement_basis"] = "relative_mouth_opening"
    if articulation:
        from app.evidence import video_viseme_articulation as shape_check

        # The qualified presentation above supplies the unpadded 16 kHz
        # analysis extent for both supported source rates.
        events = raw["articulation_phones"]
        _validate_phones(events, samples, shape_check.PHONES)
        inputs["articulation_phones"] = events
        basis.setdefault("measurement_basis", "speech_events")
        basis["schema"] = SCHEMA_WITH_ARTICULATION
        if witness is None:
            extra = shape_check.locate(geometry, events, samples, ARTICULATION_FIT)
            disposition, supported = _witness(extra, geometry, samples, origin, ARTICULATION_POINT)
            if supported is not None:
                candidate, state, witness = extra, "matched", supported
                basis["measurement_basis"] = "local_articulation"
            elif disposition in {"measured", "no_supported_change"}:
                candidate, state = extra, disposition
    if witness is None and stereo_scope is None and "sound_levels" in raw:
        quiet = silent.record(raw["sound_levels"], geometry, samples, origin)
        if quiet:
            candidate, witness = quiet
            state = "matched"
            inputs["sound_levels"] = raw["sound_levels"]
            basis.update(schema=SCHEMA_WITH_SILENCE, measurement_basis=silent.KEY)
    # Canonical evidence rounds floats to six places. Keep seven-place inputs and fits as integers.
    if "score" in candidate:
        candidate["score_units"] = round(candidate.pop("score") * timing.SCORE_UNITS)
    note = skipped("no_match" if state in {"measured", "no_supported_change"} else state)
    if articulation and note["status"] in {"matched", "no_match"}:
        note["observation"] = ("A short mismatch between speech sounds and visible mouth opening was measured."
                               if basis["measurement_basis"] == "local_articulation" else note["observation"])
        if note["status"] == "no_match":
            note["observation"] = "No qualifying local speech and mouth difference was found in the measured ranges."
    if stereo_scope is not None:
        basis.update(schema=SCHEMA_WITH_STEREO, stereo_scope=stereo_scope)
    if basis.get("measurement_basis") == silent.KEY:
        note["observation"] = silent.observation(witness)
    return {**note, **basis,
            "pcm_sha256": mapping["pcm_sha256"], "audio_origin_pts": origin,
            "inputs": inputs, "measurement": candidate, "witness": witness}


def observation(record):
    witness = record["witness"]
    if silent.KEY in witness:
        return silent.observation(witness)
    lo, hi = [sample / timing.RATE for sample in witness["audio_source_pts"]]
    a, b = [sample / timing.RATE for sample in witness["picture_source_samples"]]
    soundtrack = " The excerpt and waveform show the analyzed mix of both soundtrack channels." if record.get("stereo_scope") else ""
    if anchor := witness.get("articulation"):
        count = anchor["mismatched_events"]
        return (f"Speech sounds and lip opening disagreed on {count} {'event' if count == 1 else 'events'} in this interval. "
                f"Audio source time {lo:.3f}–{hi:.3f} s. Corresponding picture time {a:.3f}–{b:.3f} s. "
                f"The strongest visible example is frame {anchor['picture_frame']}, compared with "
                f"the speech event at {anchor['audio_sample'] / timing.RATE:.3f} s." + soundtrack)
    direction = "later" if witness["change_samples"] > 0 else "earlier"
    change = abs(witness["change_samples"]) * 1000 / timing.RATE
    description = (f"Relative to the surrounding timing, the audio was approximately {change:.0f} ms {direction}. "
                   f"Audio source time {lo:.3f}–{hi:.3f} s. Corresponding picture time {a:.3f}–{b:.3f} s.")
    if record.get("measurement_basis") == "relative_mouth_opening":
        description += " Mouth opening and speech supported a different offset inside this interval than on both sides."
    return description + soundtrack


def validate(document, *, timestamps=None, committed=_UNSET):
    source = document["source"]
    record = document.get(KEY)
    packets = [packet for packet in document["packets"] if packet["id"] == PACKET_ID or KEY in packet]
    if (canonical_hash(record) if record is not None else None) != source.get(COMMITMENT):
        raise ValueError("Lip timing differs from the committed analysis")
    if committed is not _UNSET and record != committed:
        raise ValueError("Lip timing differs from the committed render")
    if record is None:
        if packets:
            raise ValueError("Uncommitted lip timing evidence")
        return
    state = record["status"]
    if (record.get("schema") not in {SCHEMA, SCHEMA_WITH_APERTURE, SCHEMA_WITH_ARTICULATION, SCHEMA_WITH_STEREO, SCHEMA_WITH_SILENCE}
            or state not in STATUS or record.get("affects_overall_confidence") is not False):
        raise ValueError("Invalid lip timing disposition")
    if "inputs" not in record:
        if record != skipped(state) or state not in {"unsupported", "stereo_unresolved", "assets", "runtime", "provenance"} or packets:
            raise ValueError("An unassessed lip timing review has unexpected details")
        return
    source_frames = record["inputs"]["frames"]
    actual_pts = [frame["pts"] for frame in source_frames]
    if (len(source_frames) != source["frame_count"] or not all(a < b for a, b in zip(actual_pts, actual_pts[1:]))
            or (timestamps is not None and list(timestamps) != actual_pts)):
        raise ValueError("Lip timing frames do not reproduce the source clock")
    audio = document["audio"]["source"]
    if measured(record["inputs"], timestamps=actual_pts, time_base=source["time_base"], source=source, mapping=audio,
                stereo_scope=record.get("stereo_scope")) != record:
        raise ValueError("Lip timing measurements do not reproduce")
    if len(packets) != (state == "matched"):
        raise ValueError("Lip timing packet does not match its disposition")
    if packets:
        [packet] = packets
        limitation = packet_limitation(record["witness"])
        if (packet["id"] != PACKET_ID or packet.get(KEY) != record["witness"] or packet["affects_overall_confidence"] is not False
                or packet["observation"] != observation(record) or limitation not in packet["limitations"]):
            raise ValueError("Lip timing packet differs from the measured interval")
        if silent.KEY in record["witness"] and (
            packet.get("title") != silent.TITLE or packet.get("interpretation") != silent.INTERPRETATION
            or packet.get("measurements") != silent.measurements(record["measurement"])
        ):
            raise ValueError("Near-silent soundtrack summary differs from the measured interval")
        frames = record["witness"]["picture_frames"]
        expected = [(role, frames[role]) for role in ROLES if frames[role] is not None]
        if [(frame.get("interval_role"), frame["anchor"]["frame_index"]) for frame in packet["frames"]] != expected:
            raise ValueError("Lip timing frame roles differ from the measured picture range")
        for frame in packet["frames"]:
            native = source_frames[frame["anchor"]["frame_index"]]
            x, y, right, bottom = native["mouth_box"]
            if (frame["anchor"]["decoded_rgb_sha256"] != native["rgb_sha256"]
                    or frame["context_region"] != {"x": x, "y": y, "w": right-x, "h": bottom-y}
                    or packet["artifacts"][frame["artifact_refs"]["crop"]]["width"] != right-x
                    or packet["artifacts"][frame["artifact_refs"]["crop"]]["height"] != bottom-y):
                raise ValueError("Lip timing mouth crop differs from its source region")
        clip, plot = (packet["artifacts"][key] for key in (CLIP, WAVEFORM))
        lo, hi = record["witness"]["audio_source_pts"]
        if ((clip["media_type"], clip["samples"], clip["sample_rate"], clip["channels"]) != ("audio/wav", hi-lo, timing.RATE, 1)
                or (plot["media_type"], plot["width"], plot["height"], plot.get("time_axis")) != ("image/png", 960, 280, "source_pts")):
            raise ValueError("Lip timing soundtrack assets do not match their exact extent")


def packet_limitation(witness):
    if silent.KEY in witness:
        return silent.LIMITATION
    return ARTICULATION_LIMITATION if "articulation" in witness else LIMITATION
