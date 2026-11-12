"""Verify exact physical channel commitments and replay agreement arithmetic offline."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import wave
from array import array
from collections import Counter, defaultdict
from pathlib import Path
from sys import byteorder


def digest(value):
    return hashlib.sha256(value.read_bytes() if isinstance(value, Path) else value).hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def channel_hashes(path):
    with wave.open(str(path), "rb") as stream:
        if stream.getparams()[:4] != (2, 2, 16000, 48000):
            raise ValueError("The stereo source clock differs")
        native = stream.readframes(48000)
    samples = array("h")
    samples.frombytes(native)
    return digest(native), [digest(samples[channel::2].tobytes()) for channel in (0, 1)]


def ai_support(raw, bundle):
    if not isinstance(raw, (int, float)) or not math.isfinite(raw) or not 0 <= raw <= 1:
        raise ValueError("Invalid response")
    raw = max(1e-6, min(raw, 1 - 1e-6))
    fit = bundle["parameters"]
    estimate = 1 / (1 + math.exp(-fit["slope"] * math.log(raw / (1 - raw)) - fit["intercept"]))
    valid_classes = []
    for label, score in (("0", estimate), ("1", 1 - estimate)):
        maxima = bundle["guards"][label]
        if len(maxima) < 9:
            return False
        rank = 1 + sum(value is None or value >= score for value in maxima)
        if rank > bundle["alpha"] * (len(maxima) + 1):
            valid_classes.append(label)
    return estimate >= .8 and valid_classes == ["1"]


def audit(source, output, calibration):
    if byteorder != "little":
        raise ValueError("Little-endian sample replay required")
    freeze, data, result = (read(output / f"{name}.json") for name in ("freeze", "measurements", "aggregate"))
    bundle = read(calibration)
    prior, gate, scores = (read(source / f"{name}.json") for name in ("roster", "gate", "scores"))
    lock_sha = digest(output / "freeze.json")
    if (freeze["source"] != {f"{name}_sha256": digest(source / f"{name}.json")
                            for name in ("aggregate", "freeze", "roster", "gate", "scores")}
            or digest(calibration) != freeze["calibration_sha256"]
            or data["freeze_sha256"] != lock_sha or result["freeze_sha256"] != lock_sha
            or result["freeze"] != freeze or result["measurements_sha256"] != digest(output / "measurements.json")):
        raise ValueError("The source, method or measurement commitment differs")

    cells, plans = defaultdict(Counter), Counter()
    cases = zip(prior, gate["rows"], scores["rows"], data["rows"], strict=True)
    compared = 0
    for row, eligibility, score, measured in cases:
        index = row["id"]
        wav = source / "wav" / f"{index:03}.wav"
        clock, channels = channel_hashes(wav)
        if (not index == eligibility["id"] == score["id"] == measured["id"]
                or digest(wav) != row["wav_sha256"] or clock != row["stereo_sha256"]
                or channels != row["channels_sha256"]
                or digest(source / "pcm" / f"{index:03}.pcm") != row["pcm_sha256"]):
            raise ValueError("Original channel or mono bytes differ")
        direction = score.get("mono", {}).get("decision", "unassessed")
        cell = cells[(row["treatment"], row.get("room"), row.get("mode"), row["family"])]
        cell.update(planned=1, assessed=direction != "unassessed", baseline_ai=direction == "ai")
        if direction != "ai":
            status = "no_mono_ai"
        elif channels[0] == channels[1]:
            status = "identical_channels"
        elif eligibility["admitted"]["ch0"] and eligibility["admitted"]["ch1"]:
            status = "compare"
        else:
            status = "speech_not_admitted_in_both"
        plans[status] += 1
        seen = measured["responses"]
        expected = [(f"ch{channel}", value) for channel, value in enumerate(channels)] if status == "compare" else []
        if measured["status"] != status or [(item["channel"], item["pcm_sha256"]) for item in seen] != expected:
            raise ValueError("Replayed physical-channel eligibility differs")
        if seen:
            support = [ai_support(item["raw"], bundle) for item in seen]
            if support != [item["supports_ai"] for item in seen]:
                raise ValueError("Replayed calibrated directions differ")
            compared += 1
            cell["compared"] += 1
            cell[f"ai_channels_{sum(support)}"] += 1
            if sum(support) == 0:
                cell["neither_withheld"] += 1
            if sum(support) < 2:
                cell["either_withheld"] += 1
    if (dict(plans) != freeze["planned"] or 2 * compared != result["forwards"] or 2 * compared != data["forwards"]
            or cells != {(cell["treatment"], cell["room"], cell["mode"], cell["family"]):
                         {key: count for key, count in cell.items() if key not in {"treatment", "room", "mode", "family"}}
                         for cell in result["cells"]}):
        raise ValueError("Replayed condition totals differ")
    for policy, recorded in result["results"].items():
        field = f"{policy}_withheld"
        false, missed = ({mode: sum(cell[field] for (t, _, m, f), cell in cells.items()
                                   if t == "mixed" and m == mode and (f == "genuine") == genuine)
                          for mode in ("sequential", "simultaneous")} for genuine in (True, False))
        controls = sum(cell[field] for (t, _, _, _), cell in cells.items() if t != "mixed")
        rule = freeze["policy"]
        advance = (min(false.values()) >= rule["minimum_false_flags_withheld_by_mode"]
                   and sum(missed.values()) <= rule["maximum_synthetic_flags_withheld"]
                   and controls <= rule["maximum_clean_or_native_flags_withheld"])
        if recorded != {"false_flags_withheld": false, "synthetic_flags_withheld": missed,
                        "controls_withheld": controls, "advance": advance}:
            raise ValueError("Replayed policy result differs")
    print(json.dumps({"verified_stereo_and_mono_inputs": len(prior), "verified_channel_outputs": 2 * compared,
                      "verified_policy_results": result["results"]}, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("calibration", type=Path)
    args = parser.parse_args()
    audit(args.source, args.output, args.calibration)
