# SPDX-License-Identifier: MIT
"""Bind local lip timing measurements to the frozen video and audio clocks."""

from __future__ import annotations

import re
from fractions import Fraction

from app.core.canonical import canonical_hash
from app.evidence import video_lip_agreement as agreement
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
SCHEMA_WITH_MOTION_SOUND = "chipmunk.video-lip-timing/6"
SCHEMA_WITH_SHOTS = "chipmunk.video-lip-timing/7"
SCHEMA_WITH_ACTIVITY = "chipmunk.video-lip-timing/8"
ACTIVITY_KEY = "speech_activity_gap"
ACTIVITY_TITLE = "Reduced speech activity during visible mouth movement"
ACTIVITY_INTERPRETATION = ("The highlighted interval had lower measured speech activity than nearby portions of the "
                           "soundtrack while the visible mouth continued moving.")
ACTIVITY_LIMITATION = ("Ordinary speech pauses, offscreen voices, noise and mixing can affect this check. "
                       "The waveform and excerpt show a mix of the submitted channels. "
                       "This observation carries zero weight in the AI assessment.")
ACTIVITY_POINTS = {"pcm": 362686, "aac48": 379013}
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
MOTION_INTERPRETATION = (
    "The soundtrack and visible mouth movement agreed less in the highlighted interval than on both sides."
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
AAC_44100_PRESENTATION_FIELDS = ("source_container", "source_profile", agreement.AAC_44100_CLOCK)


def skipped(reason):
    return {"schema": SCHEMA, "status": reason, "observation": STATUS[reason], "affects_overall_confidence": False}


def presented_samples(mapping, *, max_samples=64_000):
    """Keep AAC padding outside the timing comparison and its evidence."""
    samples = mapping.get("samples")
    if type(samples) is not int:
        return None
    rate = mapping.get("source_sample_rate")
    if rate in (48_000, agreement.AAC_44100_RATE):
        if (mapping.get("source_codec") != "aac" or mapping.get("source_time_base") != f"1/{rate}"
                or mapping.get("source_start_pts") != 0):
            return None
        if rate == agreement.AAC_44100_RATE and (
            mapping.get("source_container") != stereo.CONTAINER or mapping.get("source_profile") != "LC"
            or mapping.get(agreement.AAC_44100_CLOCK) is not True
        ):
            return None
        duration = mapping.get("source_duration_pts")
        if type(duration) is not int or duration <= 0:
            return None
        extent = duration * timing.RATE // rate
        # Ceiling bounds the decoded tail by one native AAC-LC frame.
        padding = (1024 * timing.RATE + rate - 1) // rate
        if not 0 <= samples - extent <= padding:
            return None
        samples = extent
    elif rate != timing.RATE:
        return None
    return samples if 20_000 <= samples <= max_samples else None


def audio_origin(mapping, stereo_scope=None, *, max_samples=64_000):
    extent = presented_samples(mapping, max_samples=max_samples)
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
    if mapping.get("source_sample_rate") in (48_000, agreement.AAC_44100_RATE) and first != 0:
        return None
    for block in clock:
        a, b, pts = (block[key] for key in ("start_sample", "end_sample", "source_pts"))
        if any(type(value) is not int for value in (a, b, pts)) or not cursor == a < b <= samples or pts != first + a:
            return None
        cursor = b
    return first if cursor == samples else None


def activity_profile(timestamps, time_base, mapping):
    """Admit the continuous 30 fps stereo source and exact clock used by the panel."""
    if (not 40 <= len(timestamps) <= 120 or timestamps[0] != 0
            or any((b - a) * Fraction(time_base) != Fraction(1, 30) for a, b in zip(timestamps, timestamps[1:]))
            or mapping.get("source_channels") != 2 or mapping.get("audio_stream_indices")
            or mapping.get("timeline_time_base") != "1/16000"):
        return None
    count = presented_samples(mapping)
    if (count is None or count != round(len(timestamps) * timing.RATE / 30)
            or mapping.get("timeline") != [{"start_sample": 0, "end_sample": mapping["samples"], "source_pts": 0}]):
        return None
    carrier = mapping.get("source_codec"), mapping.get("source_sample_rate")
    if (carrier == ("pcm_s16le", 16_000) and mapping.get("source_time_base") == "1/16000"
            and mapping.get("source_start_pts") == 0):
        return "pcm"
    if (carrier == ("aac", 48_000) and mapping.get("source_profile") == "LC"
            and mapping.get("source_container") == stereo.CONTAINER):
        return "aac48"
    return None


def _activity(raw, inputs, mapping, profile):
    from app.bench.video_speech_activity_gap import BLOCK, scan

    traces = raw[ACTIVITY_KEY]
    if (not isinstance(traces, dict) or set(traces) != {"original", "reduced", "stereo_sha256"}
            or not _SHA.fullmatch(str(traces["stereo_sha256"])) or inputs["phones"]):
        raise ValueError("Invalid speech-activity measurement")
    values = []
    for name in ("original", "reduced"):
        trace = traces[name]
        if (not isinstance(trace, list) or len(trace) != inputs["samples"] // BLOCK
                or any(not isinstance(item, str) or len(item) > 28 for item in trace)):
            raise ValueError("Speech activity does not match the measured sample clock")
        try:
            values.append([float.fromhex(item) for item in trace])
        except (ValueError, OverflowError) as error:
            raise ValueError("Invalid speech activity value") from error
    result = scan(*values, inputs["frames"])["both"]
    chosen = result["reading"]
    if chosen and chosen["margin_millionths"] > ACTIVITY_POINTS[profile]:
        witness = {"audio_source_pts": chosen["span"], "picture_source_samples": chosen["span"],
                   "picture_frames": chosen["picture_frames"],
                   ACTIVITY_KEY: {"activity_millionths": chosen["activity_millionths"],
                                  "margin_millionths": chosen["margin_millionths"],
                                  "visible_mouth_frames": chosen["mouth_frames"]}}
        note = {**skipped("matched"), "observation": _activity_observation(witness)}
    else:
        witness = None
        if result["assessed"]:
            note = {**skipped("no_match"), "observation": "No qualifying short decline in speech activity was measured."}
        else:
            note = {**skipped("insufficient_events"),
                    "observation": "The soundtrack and picture did not provide enough comparable speech and mouth activity."}
    return {**note, "schema": SCHEMA_WITH_ACTIVITY, "measurement_basis": ACTIVITY_KEY,
            "inputs": {**inputs, ACTIVITY_KEY: traces}, "measurement": chosen, "witness": witness,
            "audio_origin_pts": 0, "pcm_sha256": mapping["pcm_sha256"]}


def _activity_observation(witness):
    start, end = (sample / timing.RATE for sample in witness["audio_source_pts"])
    return (f"Speech activity fell in both audio readings between {start:.3f} and {end:.3f} s, while mouth movement "
            f"continued across {witness[ACTIVITY_KEY]['visible_mouth_frames']} submitted frames.")


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


def sample_limit(record):
    return agreement.MAX_MULTISHOT_SAMPLES if record.get("schema") == SCHEMA_WITH_SHOTS else 64_000


def _measured_shots(raw, inputs, timestamps, time_base, mapping, origin, stereo_scope):
    if (not agreement.silence_carrier(mapping, origin) or stereo_scope is not None
            or mapping.get("audio_stream_indices") or inputs["phones"]):
        raise ValueError("The shot review requires one supported soundtrack")
    ranges, levels = (raw[key] for key in (silent.SHOT_KEY, "sound_levels"))
    samples, frames = inputs["samples"], inputs["frames"]
    times = agreement.shot_audio_ranges(timestamps, time_base, samples, ranges)
    # Validate the full measured sound record, even if every shot lacks a visible mouth.
    list(silent.quiet_ranges(levels, samples))
    coverage, findings, compared = [], [], 0
    for (start, stop), (lo, hi) in zip(ranges, times, strict=True):
        visible = frames[start:stop]
        packet_id = None
        if not agreement.eligible_shot(start, stop):
            state, reason = "outside_scope", "Shot length outside the checked range"
        elif sum(row["gap_units"] is not None for row in visible) < 20:
            state, reason = "coverage", "Insufficient views of mouth movement"
        else:
            compared += 1
            state, reason = "no_match", "No qualifying near-silent interval"
            if match := silent.record(levels, visible, samples, origin, shot=(lo, hi)):
                measurement, witness = match
                packet_id = f"VLT{len(findings) + 1:02d}"
                findings.append({"id": packet_id, "measurement": measurement, "witness": witness})
                state, reason = "matched", "Mouth movement during a near-silent interval"
        coverage.append({"frames": [start, stop], "audio_samples": [lo, hi], "status": state,
                         "observation": reason, "exhibit": packet_id})
    count = len(coverage)
    message = (f"Checked {compared} of {count} detected shots for near-silence while the mouth moved. "
               + (f"Found {len(findings)} qualifying {'interval' if len(findings) == 1 else 'intervals'}." if findings else
                  "No qualifying interval was found." if compared else "There were too few comparable observations."))
    return {"schema": SCHEMA_WITH_SHOTS, "status": "matched" if findings else "no_match" if compared else "insufficient_events",
            "observation": message, "affects_overall_confidence": False, "measurement_basis": silent.KEY,
            "pcm_sha256": mapping["pcm_sha256"], "audio_origin_pts": origin,
            "inputs": {**inputs, silent.SHOT_KEY: ranges, "sound_levels": levels}, "shot_coverage": coverage,
            "shot_findings": findings}


def exhibits(record):
    """Yield source-bound interval records for both historical and per-shot reviews."""
    if not record or record["status"] != "matched":
        return
    if record.get("schema") != SCHEMA_WITH_SHOTS:
        yield PACKET_ID, record
        return
    for shot in record["shot_findings"]:
        yield shot["id"], {**record, "measurement": shot["measurement"], "witness": shot["witness"]}


def measured(raw, *, timestamps, time_base, source, mapping, stereo_scope=None):
    limit = agreement.MAX_MULTISHOT_SAMPLES if silent.SHOT_KEY in raw else 64_000
    activity = ACTIVITY_KEY in raw
    profile = activity_profile(timestamps, time_base, mapping) if activity else None
    if activity and (profile is None or stereo_scope is not None
                     or not 320 <= source["width"] <= 1280 or not 240 <= source["height"] <= 720):
        raise ValueError("The speech-activity check requires a qualified stereo video source")
    origin = 0 if activity else audio_origin(mapping, stereo_scope, max_samples=limit)
    if origin is None or (mapping.get("source_sample_rate") == agreement.AAC_44100_RATE and silent.SHOT_KEY not in raw):
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
    samples = presented_samples(mapping, max_samples=limit)
    _validate_phones(raw["phones"], samples, timing.PHONES)
    inputs = {"frames": rows, "phones": raw["phones"], "samples": samples}
    if activity:
        return _activity(raw, inputs, mapping, profile)
    if silent.SHOT_KEY in raw:
        return _measured_shots(raw, inputs, timestamps, time_base, mapping, origin, stereo_scope)
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
    if witness is None and (motion := raw.get(agreement.KEY)) is not None:
        delivery = agreement.carrier(mapping, origin=origin, stereo_scope=stereo_scope)
        if delivery is None or mapping.get("audio_stream_indices") or not agreement.eligible_picture_clock(rows, time_base):
            raise ValueError("The local mouth and speech comparison uses an unsupported source clock")
        agreement._proposal(motion, rows, samples)
        inputs[agreement.KEY] = motion
        if motion["score_units"] > agreement.POINTS[delivery][0]:
            candidate = motion
            witness = agreement.record(motion, rows)
            state = "matched"
            basis.update(schema=SCHEMA_WITH_MOTION_SOUND, measurement_basis=agreement.KEY)
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
    if basis.get("measurement_basis") == agreement.KEY:
        note["observation"] = agreement.observation(witness)
    return {**note, **basis,
            "pcm_sha256": mapping["pcm_sha256"], "audio_origin_pts": origin,
            "inputs": inputs, "measurement": candidate, "witness": witness}


def observation(record):
    witness = record["witness"]
    if ACTIVITY_KEY in witness:
        return _activity_observation(witness)
    if silent.KEY in witness:
        return silent.observation(witness)
    if agreement.KEY in witness:
        return agreement.observation(witness)
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
    packets = [packet for packet in document["packets"] if re.fullmatch(r"VLT[0-9]{2}", packet["id"]) or KEY in packet]
    if (canonical_hash(record) if record is not None else None) != source.get(COMMITMENT):
        raise ValueError("Lip timing differs from the committed analysis")
    if committed is not _UNSET and record != committed:
        raise ValueError("Lip timing differs from the committed render")
    if record is None:
        if packets:
            raise ValueError("Uncommitted lip timing evidence")
        return
    state = record["status"]
    if (record.get("schema") not in {SCHEMA, SCHEMA_WITH_APERTURE, SCHEMA_WITH_ARTICULATION, SCHEMA_WITH_STEREO,
                                   SCHEMA_WITH_SILENCE, SCHEMA_WITH_MOTION_SOUND, SCHEMA_WITH_SHOTS, SCHEMA_WITH_ACTIVITY}
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
    intervals = list(exhibits(record))
    if len(packets) != len(intervals):
        raise ValueError("Lip timing packet does not match its disposition")
    for packet, (packet_id, record) in zip(packets, intervals, strict=True):
        limitation = packet_limitation(record["witness"])
        if (packet["id"] != packet_id or packet.get(KEY) != record["witness"] or packet["affects_overall_confidence"] is not False
                or packet["observation"] != observation(record) or limitation not in packet["limitations"]):
            raise ValueError("Lip timing packet differs from the measured interval")
        if silent.KEY in record["witness"] and (
            packet.get("title") != silent.TITLE or packet.get("interpretation") != silent.INTERPRETATION
            or packet.get("measurements") != silent.measurements(record["measurement"])
        ):
            raise ValueError("Near-silent soundtrack summary differs from the measured interval")
        if agreement.KEY in record["witness"] and (
            packet.get("title") != agreement.TITLE or packet.get("interpretation") != MOTION_INTERPRETATION
            or packet.get("measurements") != motion_measurements(record["witness"])
        ):
            raise ValueError("Local speech and mouth summary differs from the measured interval")
        if ACTIVITY_KEY in record["witness"] and (
            packet.get("title") != ACTIVITY_TITLE or packet.get("interpretation") != ACTIVITY_INTERPRETATION
            or packet.get("measurements") != activity_measurements(record["witness"])
        ):
            raise ValueError("Speech-activity summary differs from the measured interval")
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
    if ACTIVITY_KEY in witness:
        return ACTIVITY_LIMITATION
    if silent.KEY in witness:
        return silent.LIMITATION
    if agreement.KEY in witness:
        return agreement.LIMITATION
    return ARTICULATION_LIMITATION if "articulation" in witness else LIMITATION


def motion_measurements(witness):
    return [{"label": f"Sound and mouth agreement {where}", "value": round(value / agreement.UNITS, 4), "unit": "relative score"}
            for where, value in zip(("before", "within", "after"), witness[agreement.KEY]["agreement_units"], strict=True)]


def activity_measurements(witness):
    readings = witness[ACTIVITY_KEY]["activity_millionths"]
    return [{"label": f"Speech activity {where}, {audio}", "value": round(value / 1_000_000, 4),
             "unit": "relative score"}
            for where, pair in zip(("before", "within", "after"), readings, strict=True)
            for audio, value in zip(("original", "noise-reduced"), pair, strict=True)]
