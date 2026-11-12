"""Verify paired stereo samples, observed channel decisions and outcome arithmetic."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import struct
import wave
from collections import Counter, defaultdict
from pathlib import Path

from review_audio_harmonic_overlap import read, sha

COUNT, BLOCK = 48_000, 512


def unpack(body):
    return struct.unpack(f"<{COUNT * 2}h", body)


def gate(values, measured):
    mask = bytearray(COUNT)
    admitted = {}
    for view, spans in measured["spans"].items():
        previous = total = 0
        for a, b in spans:
            if not previous <= a < b <= COUNT:
                raise ValueError("Speech activity spans do not follow the original sample grid")
            total += b - a
            previous = b
            if view != "mono":
                mask[a:b] = b"\x01" * (b - a)
        admitted[view] = total * 2 >= COUNT
    active, quiet = [], []
    for start in range(0, COUNT - BLOCK + 1, BLOCK):
        block = sum(mask[start:start + BLOCK])
        if block == BLOCK:
            active.append(start)
        elif block == 0:
            quiet.append(start)
    selected, contrast = None, None
    if min(values) == -32768 or max(values) == 32767:
        reason = "rail_clipping"
    elif min(len(active), len(quiet)) < 8:
        reason = "insufficient_active_or_quiet_support"
    else:
        def median_power(starts, channel):
            return max(1, statistics.median(sum(x*x for x in values[2 * a + channel:2 * (a + BLOCK):2]) / BLOCK
                                            for a in starts))
        contrast = [10 * math.log10(median_power(active, i) / median_power(quiet, i)) for i in (0, 1)]
        if abs(contrast[0] - contrast[1]) >= 3:
            selected, reason = max(range(2), key=contrast.__getitem__), "selected"
        else:
            reason = "contrast_gap_below_minimum"
    got = measured["contrast_db"]
    if (selected, reason, len(active), len(quiet), admitted) != (
            measured["selected"], measured["reason"], measured["active_blocks"], measured["quiet_blocks"], measured["admitted"]
    ) or (contrast is None) != (got is None) or contrast and any(abs(a - b) > 1e-9 for a, b in zip(contrast, got, strict=True)):
        raise ValueError("Scalar stereo selection differs from the frozen evidence")
    return selected, admitted


def direction(value, calibration):
    raw = value["raw"]
    if not math.isfinite(raw) or not 0 <= raw <= 1:
        raise ValueError("Recorded voice response is outside its range")
    bounded = min(max(raw, 1e-6), 1 - 1e-6)
    fit = calibration["parameters"]
    response = 1 / (1 + math.exp(-(fit["slope"] * (math.log(bounded) - math.log1p(-bounded)) + fit["intercept"])))
    labels = []
    for label, statistic in enumerate((response, 1 - response)):
        guards = calibration["guards"][str(label)]
        if (1 + sum(x is None or x >= statistic for x in guards)) / (len(guards) + 1) > calibration["alpha"]:
            labels.append(label)
    calls = 1
    if response >= .8 and labels == [1]:
        result = "ai"
    elif response <= .2 + 1e-12 and labels == [0]:
        result = "inconclusive"
        if value["probe_status"] == "computed":
            calls += 1
            if value["stability_log_odds_change"] <= math.log(2):
                result = "original"
    else:
        result = "inconclusive"
    if value["decision"] != result:
        raise ValueError("Independent recalculation differs from the archived voice direction")
    return result, calls


def audit(folder, ami, calibration_path):
    result = read(folder, "aggregate.json")
    frozen = read(folder, "freeze.json", result["freeze_sha256"])
    roster = read(folder, "roster.json", frozen["roster_sha256"])
    gates = read(folder, "gate.json", result["gate_sha256"])
    scores = read(folder, "scores.json", result["scores_sha256"])
    stored_evidence = read(folder, "evidence-private.json", result["evidence_sha256"])
    calibration = read(calibration_path.parent, calibration_path.name, frozen["calibration_sha256"])
    if frozen != result["freeze"] or scores["gate_sha256"] != result["gate_sha256"] or any(
            phase["freeze_sha256"] != result["freeze_sha256"] for phase in (gates, scores)):
        raise ValueError("Report, source manifest and observations differ")
    media, reasons, cells, evidence = {}, Counter(), defaultdict(Counter), []
    native_checks = pair_checks = calls = 0
    for row, measured, score in zip(roster, gates["rows"], scores["rows"], strict=True):
        index = row["id"]
        stereo_file = folder / "wav" / f"{index:03}.wav"
        with stereo_file.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != row["wav_sha256"]:
                raise ValueError("Frozen stereo container differs")
        with wave.open(str(stereo_file), "rb") as stream:
            native = stream.readframes(COUNT)
        mono = (folder / "pcm" / f"{index:03}.pcm").read_bytes()
        if (sha(native), sha(mono), measured["id"], score["id"]) != (row["stereo_sha256"], row["pcm_sha256"], index, index):
            raise ValueError("View and assessment are bound to different samples")
        values = media[index] = unpack(native)
        if [sha(struct.pack(f"<{COUNT}h", *values[i::2])) for i in (0, 1)] != row["channels_sha256"]:
            raise ValueError("Measured physical microphone channel differs")
        treatment = row["treatment"]
        if treatment == "native":
            room = row["room"]
            for index_, mic in enumerate(("Array1-01", "Array1-05")):
                with wave.open(str(ami / f"{room}.{mic}.wav"), "rb") as source:
                    n = source.getnchannels()
                    source.setpos(row["native_source_start"])
                    raw = struct.unpack(f"<{COUNT * n}h", source.readframes(COUNT))
                if raw[::n] != values[index_::2]:
                    raise ValueError("Independent microphone sample cut differs")
            native_checks += 1
        elif treatment == "mixed":
            clean, background = (media[row[k]] for k in ("clean_id", "native_id"))
            noise = [round(x * row["scale"] * row["common_gain"]) for x in background]
            if tuple(a + b for a, b in zip(clean, noise, strict=True)) != values:
                raise ValueError("Independent paired mixture differs")
            measured_snr = 10 * math.log10(sum(x*x for x in clean[::2]) / sum(
                ((a + b) / 2)**2 for a, b in zip(noise[::2], noise[1::2], strict=True)))
            if abs(measured_snr - row["mono_snr_db"]) > 1e-9:
                raise ValueError("Measured level between foreground and background differs")
            pair_checks += 1
        selected, admission = gate(values, measured)
        reasons[measured["reason"]] += 1
        routes = {"mono": "mono"}
        if selected is not None:
            routes["selected"] = f"ch{selected}"
        outputs = {}
        if set(score) - {"id"} != {view for view, name in routes.items() if admission[name]}:
            raise ValueError("A score does not match the observed view admission")
        for view, value in score.items():
            if view != "id":
                outputs[view], count = direction(value, calibration)
                calls += count
        mono_direction = outputs.get("mono", "unassessed")
        channel_direction = outputs.get("selected", "unassessed")
        added = mono_direction in {"unassessed", "inconclusive"} and channel_direction == "ai"
        cell = cells[(treatment, row.get("room"), row.get("mode"), row["family"])]
        cell.update(planned=1, selected=int(selected is not None), assessed=int(mono_direction != "unassessed"),
                    baseline_ai=int(mono_direction == "ai"), baseline_original=int(mono_direction == "original"),
                    baseline_inconclusive=int(mono_direction == "inconclusive"), added_ai=int(added),
                    channel_admitted=int(channel_direction != "unassessed"))
        if added:
            evidence.append({"id": index, "view": selected, "start_sample": 0, "end_sample": COUNT, "time_base": "1/16000",
                             "stereo_wav_sha256": row["wav_sha256"], "selected_pcm_sha256": row["channels_sha256"][selected]})
    listed = {(x["treatment"], x["room"], x["mode"], x["family"]):
              {k: v for k, v in x.items() if k not in {"treatment", "room", "mode", "family"}} for x in result["cells"]}
    if (listed != cells or evidence != stored_evidence or reasons != result["selection_reasons"] or calls != scores["forwards"]
            or gates["forwards"] != len(roster) * 3 * math.ceil(COUNT / BLOCK)):
        raise ValueError("Independent decision, coverage or resource arithmetic differs")
    flagged = [roster[row["id"]] for row in evidence]
    gains = [row for row in flagged if row["family"] != "genuine"]
    false = [row for row in flagged if row["family"] == "genuine"]
    groups, families = ({row[field] for row in gains} for field in ("group", "family"))
    modes = {row["mode"] for row in gains if row["treatment"] == "mixed"}
    rule = frozen["rule"]
    coverage = all(cells[("mixed", room, mode, "genuine")]["assessed"] >= rule["minimum_assessed_genuine_per_room_condition"]
                   for room in frozen["rooms"] for mode in ("sequential", "simultaneous"))
    checks = {"gains": len(groups) >= rule["minimum_additional_synthetic_groups"],
              "families": len(families) >= rule["minimum_added_synthetic_families"],
              "conditions": len(modes) >= rule["minimum_conditions_with_gain"],
              "false": len(false) <= rule["maximum_additional_genuine_false_flags"],
              "clean_and_native": sum(row["treatment"] in {"clean", "native"} for row in false)
              <= rule["maximum_additional_clean_or_native_false_flags"], "coverage": coverage}
    if checks != result["checks"] or all(checks.values()) != result["advance"]:
        raise ValueError("The independent advancement result differs")
    return {"replayed_stereo_and_mono": len(roster), "exact_native_cuts": native_checks,
            "exact_mixes": pair_checks, "verified_classifier_forwards": calls, "verified_intervals": len(evidence)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ("folder", "ami", "calibration"):
        parser.add_argument(argument, type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.folder, args.ami, args.calibration), sort_keys=True))
