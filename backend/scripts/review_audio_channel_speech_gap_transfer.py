"""Independently replay committed PCM, time ranges, and results without a model."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import wave
from collections import Counter, defaultdict
from pathlib import Path

RATE, WIDTH, SAMPLES = 16_000, 2, 48_000


def load(folder, name):
    return json.loads((folder / name).read_bytes())


def digest(blob):
    return hashlib.sha256(blob).hexdigest()


def bound(path, expected):
    blob = path.read_bytes()
    if digest(blob) != expected:
        raise ValueError(f"SHA-256 mismatch for {path.name}")
    return blob


def wav(path, channels, frames):
    with wave.open(str(path), "rb") as stream:
        if (stream.getframerate(), stream.getsampwidth(), stream.getnchannels(), stream.getnframes()) != (RATE, WIDTH, channels, frames):
            raise ValueError("Physical WAV format or duration differs")
        blob = stream.readframes(frames)
    if len(blob) != frames * WIDTH * channels:
        raise ValueError("Incomplete physical WAV")
    return blob


def ranges(items):
    previous = 0
    for start, end in items:
        if not previous <= start < end <= SAMPLES:
            raise ValueError("Recorded activity ranges overlap or exceed source")
        previous = end
    return items


def duration(items):
    return sum(end - start for start, end in items)


def subtract(interval, forbidden):
    cursor, end = interval
    for a, b in forbidden:
        if b <= cursor:
            continue
        if a >= end:
            break
        if a > cursor:
            yield cursor, a
        cursor = max(cursor, b)
        if cursor >= end:
            return
    if cursor < end:
        yield cursor, end


def overlap(a, b, references):
    return sum(max(0, min(b, stop) - max(a, start)) for start, stop in references)


def level(pcm):
    if not pcm:
        raise ValueError("Empty source interval")
    power = sum(x * x for x in pcm) / len(pcm) / 32768**2
    return round(10 * math.log10(power), 3) if power else -140


def channel_marks(channels, mono, spans, admitted, rule):
    if admitted:
        return [], []
    guard = rule["mono_support_guard_samples"]
    blocked = [(max(0, a - guard), min(SAMPLES, b + guard)) for a, b in spans[0]]
    marks, existing = [], []
    for i, channel in enumerate(channels):
        for item in spans[i + 1]:
            for a, b in subtract(item, blocked):
                values = channel[a:b]
                if b - a < rule["minimum_range_samples"] or (db := level(values)) < rule["minimum_channel_dbfs"]:
                    continue
                marks.append({"channel": i, "start_sample": a, "end_sample": b, "rms_dbfs": db,
                              "clip_pcm_sha256": digest(struct.pack(f"<{len(values)}h", *values))})
    for a in range(0, SAMPLES, RATE):
        b = a + RATE
        loud = max(level(channels[0][a:b]), level(channels[1][a:b]))
        mixed = level(mono[a:b])
        if loud >= -50 and (mixed == -140 or round(loud - mixed, 3) >= 12):
            existing.append({"start_sample": a, "end_sample": b})
    return marks, existing


def matched_reference(channels):
    if channels[0] != channels[1]:
        raise ValueError("Dry foreground is unequal between channels")
    power = [sum(x * x for x in channels[0][i:i + 160]) / (160 * 32768**2) for i in range(0, SAMPLES, 160)]
    floor = max(.002**2, .15**2 * max(power))
    return [(i * 160, (i + 1) * 160) for i, value in enumerate(power) if value >= floor]


def supports(marks, reference, rule):
    for item in marks:
        a, b = item["start_sample"], item["end_sample"]
        count = overlap(a, b, reference)
        if count >= rule["minimum_known_speech_samples"] and count >= (b - a) * rule["minimum_known_speech_fraction"]:
            return True
    return False


def verify(folder):
    public = load(folder, "aggregate.json")
    freeze = json.loads(bound(folder / "freeze.json", public["freeze_sha256"]))
    roster = json.loads(bound(folder / "roster.json", freeze["roster_sha256"]))
    observations = json.loads(bound(folder / "observations.json", public["observations_sha256"]))
    expected_evidence = json.loads(bound(folder / "private-evidence.json", public["private_evidence_sha256"]))
    if (freeze != public["freeze"] or freeze["views"] != len(roster) or observations["freeze_sha256"] != public["freeze_sha256"]
            or public["classifier_forwards"] != 0 or observations["classifier_forwards"] != 0
            or public["vad_forwards"] != observations["vad_forwards"] or observations["vad_forwards"] != 282 * len(roster)):
        raise ValueError("Invalid experiment binding or model count")
    rule = freeze["rule"]
    counts, known = defaultdict(Counter), {}
    native, synthetic, evidence = set(), set(), []
    quality = Counter()
    for index, (row, measured) in enumerate(zip(roster, observations["rows"], strict=True)):
        pcm_path = folder / "stereo" / f"{index:03}.wav"
        bound(pcm_path, row["wav_sha256"])
        stereo = wav(pcm_path, 2, SAMPLES)
        mono = bound(folder / "mono" / f"{index:03}.pcm", row["mono_sha256"])
        if (row["id"], measured["id"], measured["source_stereo_sha256"], measured["source_mono_sha256"]) != (
                index, index, digest(stereo), digest(mono)) or digest(stereo) != row["stereo_sha256"]:
            raise ValueError("Observation differs from exact source")
        interleaved = struct.unpack(f"<{2 * SAMPLES}h", stereo)
        channels = (interleaved[::2], interleaved[1::2])
        mixed = struct.unpack(f"<{SAMPLES}h", mono)
        spans = [ranges(items) for items in measured["spans"]]
        if len(spans) != 3:
            raise ValueError("Missing mono or physical channel activity")
        admitted = duration(spans[0]) >= SAMPLES / 2
        marks, baseline = channel_marks(channels, mixed, spans, admitted, rule)
        if (admitted, marks, baseline) != (measured["mono_assessed"], measured["marks"], measured["existing_attenuation"]):
            raise ValueError(f"Independent measurement differs for row {index}")
        panel, condition, family = row["panel"], row["condition"], row["family"]
        if panel == "native":
            reference = ranges(row["known_word_samples"])
        else:
            if condition == "matched_clean":
                known[row["foreground"]] = matched_reference(channels)
            reference = known[row["foreground"]]
        candidate_ok, baseline_ok = supports(marks, reference, rule), supports(baseline, reference, rule)
        added, unsupported = candidate_ok and not baseline_ok, bool(marks) and not candidate_ok
        counts[(panel, condition, family)].update(planned=1, mono_assessed=admitted, candidate_flags=bool(marks),
                                                baseline_flags=bool(baseline), candidate_supported=candidate_ok,
                                                baseline_supported=baseline_ok, additional_supported=added, unsupported_flags=unsupported,
                                                opportunities=not admitted and duration(reference) >= rule["minimum_known_speech_samples"])
        if panel == "native":
            quality.update(native_additions=added, native_unsupported=unsupported, native_quiet=condition == "quiet" and bool(marks))
            if added:
                native.add(row["group"])
        else:
            quality.update(control_clean=condition == "matched_clean" and bool(marks))
            if added and family != "genuine":
                synthetic.add((row["group"], family))
        for item in marks:
            evidence.append({**item, "id": index, "source_wav_sha256": row["wav_sha256"], "time_base": "1/16000",
                             "source_start_pts": row["source_start_sample"] + item["start_sample"],
                             "source_end_pts": row["source_start_sample"] + item["end_sample"], "mono_sha256": row["mono_sha256"]})
    groups = len({group for group, _ in synthetic})
    families = len({family for _, family in synthetic})
    checks = {"native_additions": quality["native_additions"] >= rule["minimum_native_additional_valid_windows"],
              "native_meetings": len(native) >= rule["minimum_native_meetings"],
              "native_unsupported": quality["native_unsupported"] <= rule["maximum_native_unsupported_flags"],
              "native_quiet": quality["native_quiet"] <= rule["maximum_quiet_flags"],
              "control_groups": groups >= rule["minimum_control_synthetic_added_groups"],
              "control_families": families >= rule["minimum_control_synthetic_families"],
              "control_clean": quality["control_clean"] <= rule["maximum_clean_flags"]}
    details = [{"panel": p, "condition": c, "family": f, **n} for (p, c, f), n in sorted(counts.items())]
    recomputed = (details, dict(quality), len(evidence), len(native), groups, families, checks, all(checks.values()))
    reported = tuple(public[key] for key in ("counts", "selection_counts", "source_intervals", "native_gained_meetings",
                                             "controlled_synthetic_gained_groups", "controlled_synthetic_gained_families", "checks", "advance"))
    if recomputed != reported or evidence != expected_evidence:
        raise ValueError("Independent counts, decision or source evidence differ")
    return {"verified_views": len(roster), "verified_source_intervals": len(evidence), "advance": all(checks.values()),
            "selection_counts": dict(quality)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.folder), sort_keys=True))
