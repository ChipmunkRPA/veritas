"""Recompute source coverage and decision changes on the frozen strong-traffic panel."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from array import array
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ARMS = ("incumbent", "ordinary_rescue", "reduced_rescue", "corroborated")
SYNTHETIC = {"clone", "expressive", "stock"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load(path):
    return json.loads(path.read_text())


def pcm(folder, row):
    source = folder / "pcm" / f"{row['id']:03}.pcm"
    require(source.stat().st_size == 96_000 and sha(source) == row["pcm_sha256"], "Exact prepared samples changed")
    values = array("h", source.read_bytes())
    if sys.byteorder != "little":
        values.byteswap()
    return values


def noise_error(clean, mixture, gain, level):
    fg_energy = sum(value * value for value in clean) * gain * gain
    residue_energy = sum((received - sample * gain) ** 2 for sample, received in zip(clean, mixture, strict=True))
    return abs(10 * math.log10(fg_energy / residue_energy) - level)


def readout(folder):
    freeze_path = folder / "freeze.json"
    base_path, candidate_path = (folder / f"{stem}.json" for stem in ("baseline", "candidate"))
    roster_path = folder / "roster.json"
    freeze, roster, base, scored = map(load, (freeze_path, roster_path, base_path, candidate_path))
    require(freeze["schema"] == "chipmunk.audio-strong-traffic-view-freeze/1"
            and sha(roster_path) == freeze["roster_sha256"] and sha(freeze_path) == base["freeze_sha256"]
            == scored["freeze_sha256"] and sha(base_path) == scored["baseline_sha256"]
            and len(roster) == freeze["planned"] == len(base["rows"]) == len(scored["rows"]) == 132
            and all(sha(ROOT / name) == digest for name, digest in freeze["code_sha256"].items()),
            "Frozen source, implementation or score commitment changed")
    clean = {r["group"]: r for r in roster if r["condition"] == "clean"}
    require(len(clean) == len({r["speaker"] for r in clean.values()}) == len(
        {r["source_pcm_sha256"] for r in clean.values()}) == 18
        and Counter(r["family"] for r in clean.values()) == {"genuine": 10, "clone": 3, "expressive": 3, "stock": 2},
        "Wrong count of separated reader or synthetic voice groups")
    tapes = sorted({r["tape"] for r in roster if r["tape"] is not None})
    require(len(tapes) == 3, "Wrong number of recorded backgrounds")
    source_groups = {slot: index + 1 for index, slot in enumerate(tapes)}
    planned = {(group, tape, level) for group in clean for tape in tapes for level in (-5, -10)}
    counts = defaultdict(lambda: {arm: Counter() for arm in ARMS})
    clean_pcm = {group: pcm(folder, row) for group, row in clean.items()}
    max_noise_error, admits, inconclusive, clean_flags = 0., 0, 0, 0
    correct, false = 0, 0
    gains, gain_tapes, false_groups, false_tapes, isolates = set(), set(), set(), set(), Counter()
    cutoff, rule = freeze["candidate_points"]["ai_above"], freeze["rule"]
    for index, (row, current, measured) in enumerate(zip(roster, base["rows"], scored["rows"], strict=True)):
        require(index == row["id"] == current["id"] == measured["id"]
                and current["pcm_sha256"] == row["pcm_sha256"], "Outcome reordered or attached to a different waveform")
        group, lane, condition, tape = (row[k] for k in ("group", "family", "condition", "tape"))
        if condition in {"traffic_5", "traffic_10"}:
            key = group, tape, row["snr_db"]
            require(key in planned and lane == clean[group]["family"]
                    and row["speaker"] == clean[group]["speaker"] and row["source_sha256"] == clean[group]["source_sha256"]
                    and row["source_pcm_sha256"] == clean[group]["source_pcm_sha256"], "Noise view left its source group")
            planned.remove(key)
            max_noise_error = max(max_noise_error, noise_error(clean_pcm[group], pcm(folder, row),
                                                            row["mix_gain"], row["snr_db"]))
        elif condition == "isolated":
            require(group == f"tape:{tape}" and lane == "background" and row["pcm_sha256"] == row["noise_pcm_sha256"],
                    "An isolated control differs from its exact background excerpt")
            pcm(folder, row)
            isolates[tape] += 1
        elif condition == "clean":
            require(row["source_pcm_sha256"] == row["pcm_sha256"] and tape is None, "Clean input differs from its source")
        else:
            raise ValueError("Unknown case condition")
        decision = current["decision"]
        require(decision in {"ai", "original", "inconclusive", "unassessed"}, "Unexpected incumbent decision")
        expected = {"incumbent": decision, **{arm: decision for arm in ARMS[1:]}}
        admits += decision != "unassessed"
        ordinary, reduced = measured["ordinary_candidate"], measured["reduced_candidate"]
        if decision == "inconclusive":
            inconclusive += 1
            require(all(type(p) is float and 0 <= p <= 1 for p in (ordinary, reduced))
                    and ordinary != cutoff and reduced != cutoff,
                    "Expected finite candidate scores distinct from the old strict diagnostic boundary")
            expected["ordinary_rescue"] = "ai" if ordinary > cutoff else decision
            expected["reduced_rescue"] = "ai" if reduced > cutoff else decision
            expected["corroborated"] = "ai" if reduced > cutoff and ordinary >= cutoff - rule["raw_margin"] else decision
        else:
            require(ordinary is None and reduced is None, "A candidate was computed outside its inconclusive route")
        require(all(measured[arm] == value for arm, value in expected.items() if arm != "incumbent"),
                "A comparison arm changed its frozen route")
        bucket = condition if tape is None else f"{condition}/recording_{source_groups[tape]}"
        for arm, actual in expected.items():
            for name in (condition, bucket) if bucket != condition else (condition,):
                counts[name][arm][f"{lane}:{actual}"] += 1
            counts["all"][arm][f"{lane}:{actual}"] += 1
        added = decision != expected["corroborated"]
        if added:
            clean_flags += condition == "clean"
            false += lane not in SYNTHETIC
            if lane not in SYNTHETIC:
                false_groups.add(group)
                if tape is not None:
                    false_tapes.add(tape)
            if lane in SYNTHETIC and tape is not None:
                correct += 1
                gains.add(group)
                gain_tapes.add(tape)
    require(not planned and all(isolates[tape] == 2 for tape in tapes) and max_noise_error < .10
            and base["speech_gate_forwards"] == 94 * len(roster) and scored["encoder_forwards"] == 2 * inconclusive,
            "Missing views, wrong nominal mixture levels or wrong fixed call count")
    min_genuine = min(sum(counts[f"traffic_5/recording_{slot}"]["incumbent"][f"genuine:{name}"]
                          for name in ("ai", "original", "inconclusive")) for slot in source_groups.values())
    checks = {
        "minimum_correct_additions": correct >= rule["minimum_correct_additions"],
        "minimum_gained_source_groups": len(gains) >= rule["minimum_gained_source_groups"],
        "minimum_traffic_recording_groups_with_gain": len(gain_tapes) >= rule["minimum_traffic_recording_groups_with_gain"],
        "maximum_new_false_flags": false <= rule["maximum_new_false_flags"],
        "maximum_new_clean_flags": clean_flags <= rule["maximum_new_clean_flags"],
        "minimum_genuine_assessed_minus5_per_traffic": min_genuine >= rule["minimum_genuine_assessed_minus5_per_traffic"],
    }
    return {"schema": "chipmunk.audio-strong-traffic-view-transfer/1",
            "frozen_rule": rule, "advanced": all(checks.values()), "checks": checks,
            "counts": counts, "genuine_source_groups": 10, "synthetic_source_groups": 8, "traffic_source_groups": 3,
            "views": len(roster), "assessed": admits, "incumbent_inconclusive": inconclusive,
            "correct_additions": correct, "gained_source_groups": len(gains), "gained_traffic_groups": len(gain_tapes),
            "new_false_flags": false, "new_false_source_groups": len(false_groups), "new_false_traffic_groups": len(false_tapes),
            "new_clean_flags": clean_flags, "minimum_genuine_assessed_minus5": min_genuine,
            "maximum_measured_noise_target_error_db": round(max_noise_error, 8),
            "calls": {"voice": base["voice_forwards"], "speech_gate": base["speech_gate_forwards"],
                      "encoder": scored["encoder_forwards"]},
            "commitments": {"freeze_sha256": sha(freeze_path), "baseline_sha256": sha(base_path),
                            "candidate_sha256": sha(candidate_path), **freeze["source_commitments"]}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    option = parser.add_mutually_exclusive_group(required=True)
    option.add_argument("--write", type=Path)
    option.add_argument("--published", type=Path)
    args = parser.parse_args()
    result = readout(args.folder)
    if args.write is not None:
        with args.write.open("x") as target:
            target.write(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    else:
        require(load(args.published) == result, "Published counts differ from the audited local records")
    print(json.dumps({key: result[key] for key in ("advanced", "views", "assessed", "correct_additions",
                                                  "new_false_flags", "checks")}, sort_keys=True))


if __name__ == "__main__":
    main()
