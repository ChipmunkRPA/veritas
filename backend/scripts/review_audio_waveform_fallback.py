"""Recompute the waveform fallback decision from a frozen local evidence journal."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONDITIONS = ("clean", "kitchen_plus5", "kitchen_minus5", "traffic_plus5", "traffic_minus5")
DELIVERIES = ("pcm", "aac_64k", "mp3_64k", "opus_48k")
LABELS = ("control", "copy")
METHODS = ("exact", "two_band", "waveform")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def exact_digest(value):
    return isinstance(value, str) and re.fullmatch(r"[a-f0-9]{64}", value) is not None


def overlap(a, b):
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))


def pairs(row, method):
    cue = row[method]
    if cue is None:
        return False, False, []
    result = cue["result"]
    matches, spans = [], result["spans"]
    require(result["status"] in ("measured", "unavailable", "ambiguous") and len(spans) <= 8,
            "Unexpected repetition result status or pair budget")
    require((result["reason"] is None) == (result["status"] == "measured")
            and (not spans or result["status"] == "measured"), "Pair availability conflicts with its status")
    for pair in spans:
        left, right = pair["first"], pair["second"]
        start, end, next_start, next_end = *left, *right
        require(all(type(value) is int for value in (*left, *right))
                and 0 <= start < end <= next_start < next_end <= row["samples"]
                and end - start == next_end - next_start >= 2400, "Bad native sample pair geometry")
        if method != "exact":
            require((end - start - 640) % 160 == 0, "Approximate range is off its comparison grid")
        if method == "waveform":
            require(.82 <= pair["minimum_similarity"] <= pair["median_similarity"] <= 1
                    and pair["windows"] == (end - start - 640) // 160 + 1, "Bad waveform response support")
        donor = row["donor"]
        if donor is None:
            continue
        expected = ([donor + 320, donor + 10_080], [28_320, 38_080])
        intersections = [overlap(a, b) for a, b in zip((left, right), expected, strict=True)]
        error = abs(next_start - start - (28_000 - donor))
        if error <= (0 if method == "exact" else 32) and min(intersections) >= 2400:
            iou = min(shared / (end - start + 9760 - shared) for shared in intersections)
            matches.append((iou, error))
    require(cue["correct"] is bool(matches), "Recorded copy support differs from exact interval arithmetic")
    return bool(spans), result["status"] == "measured", matches


def clock(row, source):
    samples, mapping = row["samples"], row["clock"]
    require(type(samples) is int and row["padding_samples"] == samples - 48_000
            and (48_000 <= samples < 49_024 if row["delivery"] == "aac_64k" else samples == 48_000)
            and mapping["start_sample"] == 0 and mapping["end_sample"] == samples
            and mapping["time_base"] == "1/16000" and type(mapping["source_start_pts"]) is int
            and mapping["source_end_pts"] - mapping["source_start_pts"] == samples,
            "Incomplete decoded sample-clock binding")
    require(exact_digest(row["encoded_sha256"]) and exact_digest(row["pcm_sha256"]),
            "Expected exact encoded and decoded digests")
    if row["condition"] == "clean":
        require(row["noise_offset"] is None and row["actual_snr_db"] is None, "Clean view used a noise source")
        if row["label"] == "control" and row["delivery"] == "pcm":
            # A shared anti-clipping scale can change a clean high-level original.
            return row["pcm_sha256"] == source["pcm_sha256"]
    else:
        db = row["actual_snr_db"]
        require(row["noise_offset"] == (100 + 5 * row["source"]) * 16_000
                and isinstance(db, (float, int)) and math.isfinite(db), "Noise offset or measured SNR differs")
        if row["label"] == "control":
            nominal = -5 if row["condition"].endswith("minus5") else 5
            require(abs(db - nominal) <= 1e-8, "Control noise gain differs")
    return None


def review(folder):
    freeze_path, journal = folder / "freeze.json", folder / "observations.jsonl"
    lock = json.loads(freeze_path.read_text())
    roster = lock["source_roster"]
    require(lock["schema"] == "chipmunk.audio-waveform-fallback-freeze/1"
            and len(roster) == 36 and lock["planned_views"] == 1440
            and {source["family"] for source in roster} == {"ami", "voxpopuli"}
            and len({source["pcm_sha256"] for source in roster}) == len(roster),
            "Unexpected source roster size, coverage, or duplicates")
    ami = [source for source in roster if source["family"] == "ami"]
    vox = [source for source in roster if source["family"] == "voxpopuli"]
    require(Counter((source["group"], source["annotated_voices"]) for source in ami)
            == Counter({(f"m{group:02}", voices): 4 for group in range(3) for voices in (2, 3)}),
            "AMI source roles or reference groups changed")
    require(len(vox) == len({source["speaker"] for source in vox}) == len({source["source_session"] for source in vox})
            == len({source["group"] for source in vox}) == 12
            and Counter(source["language"] for source in vox) == {"et": 2, "lt": 2, "sl": 2, "fi": 3, "nl": 3},
            "VoxPopuli speakers or original session dates overlap")
    require(all(sha(ROOT / name) == digest for name, digest in lock["code_sha256"].items()),
            "Frozen implementation digest differs")
    planned = {(i, condition, label, delivery) for i in range(len(roster))
               for condition in CONDITIONS for label in LABELS for delivery in DELIVERIES}
    counts = defaultdict(Counter)
    gains, false, known_donors, retained = defaultdict(set), set(), {}, []
    status = {method: Counter() for method in METHODS}
    native_matches = 0
    padding = defaultdict(set)
    with journal.open() as lines:
        for line in lines:
            row = json.loads(line)
            idx, family, group, cond, label, codec = (
                row[key] for key in ("source", "family", "group", "condition", "label", "delivery")
            )
            key = (idx, cond, label, codec)
            require(key in planned, "Unexpected or repeated codec view")
            planned.remove(key)
            source = roster[idx]
            require((family, group) == (source["family"], source["group"]),
                    "The observation changed its frozen source group")
            native_matches += clock(row, source) is True
            padding[codec].add(row["padding_samples"])
            donor = row["donor"]
            require((donor is None if label == "control" else type(donor) is int and donor in range(1600, 17601, 3200)),
                    "The authored donor is inconsistent with the view role or recipe")
            if donor is not None:
                require(known_donors.setdefault(idx, donor) == donor, "A matched source changed its authored donor")
            exact, two, added = (pairs(row, method) for method in METHODS)
            baseline_pair = exact[0] or two[0]
            require((row["waveform"] is None) == baseline_pair, "The supplemental route displaced earlier evidence")
            for method in METHODS:
                cue = row[method]
                status[method]["skipped_existing_pair" if cue is None else cue["result"]["status"]] += 1
            baseline_correct, add_correct = bool(exact[2] or two[2]), bool(added[2])
            tally = {
                "views": 1, "baseline_measured": int(exact[1] or two[1]),
                "extended_measured": int(exact[1] or two[1] or added[1]),
                "baseline_pairs": int(baseline_pair), "extended_pairs": int(baseline_pair or added[0]),
                "waveform_attempts": int(not baseline_pair),
                "baseline_correct": int(baseline_correct), "extended_correct": int(baseline_correct or add_correct),
                "added_correct": int(add_correct), "added_control_flags": int(label == "control" and added[0]),
                "added_wrong_only_copy": int(label == "copy" and added[0] and not add_correct),
            }
            for bucket in ("overall", f"family:{family}", f"condition:{cond}", f"codec:{codec}"):
                counts[f"{bucket}/{label}"].update(tally)
            if add_correct:
                gains[family].add(group)
                retained.append(max(added[2], key=lambda pair: (pair[0], -pair[1])))
            if tally["added_control_flags"]:
                false.add(group)
    require(not planned and native_matches >= 24, "Missing views or substantial mismatch of native controls")
    result = {
        "minimum_added_correct_views": counts["overall/copy"]["added_correct"],
        "minimum_groups_with_gain": sum(map(len, gains.values())),
        "minimum_ami_meetings_with_gain": len(gains["ami"]),
        "minimum_voxpopuli_speakers_with_gain": len(gains["voxpopuli"]),
        "maximum_added_control_flags": counts["overall/control"]["added_control_flags"],
        "maximum_groups_with_added_control_flags": len(false),
        "maximum_added_wrong_only_copy_views": counts["overall/copy"]["added_wrong_only_copy"],
    }
    policy = lock["method"]["policy"]
    checks = {name: result[name] >= limit if name.startswith("minimum") else result[name] <= limit
              for name, limit in policy.items()}
    return {"schema": "chipmunk.audio-waveform-fallback-transfer/1", "method": lock["method"],
            "freeze_sha256": sha(freeze_path), "observations_sha256": sha(journal),
            "source_groups": {"ami": 3, "voxpopuli": 12}, "source_windows": len(roster),
            "view_counts": {key: dict(tally) for key, tally in sorted(counts.items())},
            "method_availability": status, "advancement": checks, "observed_gate_values": result,
            "advanced": all(checks.values()), "native_clean_control_pcm_exact": native_matches,
            "decoder_padding_samples": {key: sorted(values) for key, values in padding.items()},
            "added_pair_minimum_side_iou_median": round(statistics.median(v[0] for v in retained), 6) if retained else None,
            "added_pair_lag_error_max_samples": max((value[1] for value in retained), default=None)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    options = parser.add_mutually_exclusive_group(required=True)
    options.add_argument("--write", type=Path)
    options.add_argument("--published", type=Path)
    args = parser.parse_args()
    result = review(args.folder)
    if args.write is not None:
        with args.write.open("x") as output:
            output.write(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    else:
        require(result == json.loads(args.published.read_text()), "Published summary differs from audited receipts")
    print(json.dumps({"advanced": result["advanced"], **result["observed_gate_values"]}, sort_keys=True))


if __name__ == "__main__":
    main()
