"""Independently audit Opus source clocks, decoded samples and localized outcomes."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import subprocess
import sys
import tempfile
from array import array
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

RATE = 16_000
MEDIA_FIELDS = ("id", "kind", "group", "family", "room", "samples", "truth", "pcm_sha256")
CODEC_POINTS = {"wav": 6.822523597593649, "aac": 7.049249651163794}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    return json.loads(gzip.decompress(path.read_bytes()) if path.suffix == ".gz" else path.read_bytes())


def require(ok, message):
    if not ok:
        raise ValueError(message)


def outputs(row, point):
    samples = row["samples"]
    windows = {(item["start_sample"], item["end_sample"]): item["direction"] for item in row["current_windows"]}
    last = samples - 3 * RATE
    sparse = {(start, start + 3 * RATE) for start in (0, last // 2, last)}
    dense = [(start, start + 3 * RATE) for start in range(0, last + 1, 24_000)] if samples <= 11 * RATE else []
    require(set(windows) == sparse.union(dense) and all(direction in ("ai", "original", "inconclusive", "unassessed")
                                                         for direction in windows.values()), "Incomplete baseline schedule")
    base = {(second[0], first[1]) for first, second in zip(dense, dense[1:])
            if windows[first] == windows[second] == "ai"}
    sparse_ai = [(a, b) for a, b in sparse if windows[a, b] == "ai"]
    if len(sparse_ai) >= 2 and all(windows[pair] != "original" for pair in sparse):
        base.update((a, b) for a, b in sparse_ai if not any(a <= p < q <= b for p, q in base))
    excerpts = row["excerpts"]
    extent = samples - 6 * RATE
    labeled_starts = sorted({extent, *range(0, extent + 1, 2 * RATE)})
    require(labeled_starts == [reading["start_sample"] for reading in excerpts], "Incomplete two-second excerpt schedule")
    proposed, elevated = set(), set()
    for excerpt in excerpts:
        counts = excerpt["speech_samples"]
        require(len(counts) == 5 and all(type(count) is int and 0 <= count <= 2 * RATE for count in counts),
                "Invalid speech duration")
        wanted = [(i * RATE, (i + 2) * RATE) for i, count in enumerate(counts) if count >= RATE]
        measured = [(value["start_sample"], value["end_sample"]) for value in excerpt["responses"]]
        require(wanted == measured, "Recorded score inputs disagree with exact admitted intervals")
        options = [(a + excerpt["start_sample"], b + excerpt["start_sample"], value["logit"])
                   for (a, b), value in zip(measured, excerpt["responses"], strict=True) if value["logit"] > point]
        if options:
            first = max(options, key=lambda value: (value[2], -value[0]))
            proposed.add(first[:2])
            elevated.update(value[:2] for value in options)
    extra = {(a, b) for a, b in proposed if any(0 < abs(other - a) <= RATE for other, _ in elevated)}
    return sorted(base), sorted(extra)


def matches(ranges, labels):
    adjacency = defaultdict(list)
    for label, (a, b) in enumerate(labels):
        for prediction, (p, q) in enumerate(ranges):
            intersection = min(q, b) - max(p, a)
            union = (q - p) + (b - a) - intersection
            if intersection * 2 >= (b - a) and intersection * 10 >= 3 * union:
                adjacency[label].append((-intersection / union, prediction))
    chosen = {}

    def augment(current, visited):
        for _, candidate in sorted(adjacency[current]):
            if candidate in visited:
                continue
            visited.add(candidate)
            if candidate not in chosen or augment(chosen[candidate], visited):
                chosen[candidate] = current
                return True
        return False

    for label in range(len(labels)):
        augment(label, set())
    return set(chosen.values())


def window_responses(row):
    responses = {}
    for excerpt in row["excerpts"]:
        for entry in excerpt["responses"]:
            start = excerpt["start_sample"] + entry["start_sample"]
            score = entry["logit"]
            require(type(score) in (int, float) and responses.setdefault(start, score) == score,
                    "Repeated windows give different stored responses on one carrier")
    return responses


def compare_receipts(result, wav, aac):
    known = [{(value["role"], value["id"]): value for value in source["rows"]} for source in (wav, aac)]
    cohort, geometry = defaultdict(Counter), defaultdict(Counter)
    response_pairs, logit_deltas = defaultdict(Counter), defaultdict(list)
    visited = set()
    for row in result["rows"]:
        identity = row["role"], row["id"]
        require(identity not in visited, "Duplicate Opus case")
        visited.add(identity)
        ordinary, compressed = (table[identity] for table in known)
        require(all(row[key] == ordinary[key] == compressed[key] for key in MEDIA_FIELDS), "Source roles changed by carrier")
        candidate_responses = window_responses(row)
        for previous, carrier in ((ordinary, "wav"), (compressed, "aac")):
            key = f"{row['role']}/{carrier}"
            prior_responses = window_responses(previous)
            paired_starts = candidate_responses.keys() & prior_responses.keys()
            response_pairs[key].update(scored_on_both=len(paired_starts),
                scored_on_opus_only=len(candidate_responses.keys() - prior_responses.keys()),
                scored_on_comparator_only=len(prior_responses.keys() - candidate_responses.keys()))
            logit_deltas[key].extend(abs(candidate_responses[start] - prior_responses[start]) for start in paired_starts)
        edited = row["truth"]
        references = {}
        if edited:
            last = row["samples"] - 6 * RATE
            starts = {*range(0, last + 1, 2 * RATE), last}
            grid = sorted({(start + k * RATE, start + (k + 2) * RATE) for start in starts for k in range(5)})
            possible = [interval for interval in grid if any(0 < abs(interval[0] - other) <= RATE for other, _ in grid)]
            geometry[row["role"]].update(labels=len(edited), shorter_than_600ms=sum(b-a < 9600 for a, b in edited),
                                         addressable_upper_bound=len(matches(possible, edited)))
            for recorded, tag in ((ordinary, "wav"), (compressed, "aac")):
                left, right = outputs(recorded, CODEC_POINTS[tag])
                references[tag] = matches(sorted(set(left + right)), edited)
        for name, point in (("lossless_point", CODEC_POINTS["wav"]), ("aac_point", CODEC_POINTS["aac"])):
            tally = cohort[f"{row['role']}/{name}"]
            before, after = outputs(row, point)
            kind = row["kind"]
            tally.update(planned=1, **{kind: 1, f"{kind}_assessed": any(x["responses"] for x in row["excerpts"]),
                f"{kind}_baseline_flagged": bool(before), f"{kind}_combined_flagged": bool(before or after),
                f"{kind}_added_flag": bool(after and not before)})
            if edited:
                initial = matches(before, edited)
                union = matches(sorted(set(before + after)), edited)
                added = matches(after, edited)
                gain = union - initial
                tally.update(labeled=len(edited), baseline_useful=len(initial), combined_useful=len(union),
                             added_useful=len(gain), placed_extra=len(added), wrongly_located_supplement=bool(after) and not added)
                for carrier, matched in references.items():
                    tally.update({f"{carrier}_useful": len(matched), f"{carrier}_retained_useful": len(matched & union)})
                if gain:
                    tally[f"gain_{row['family']}"] += len(gain)
    require(len(visited) == 96 and all(set(table) == visited for table in known), "A material case is absent")
    require(dict(cohort) == result["counts"], "Independent localization counts disagree")
    rule = result["plan"]["procedures"]["rule"]
    for point, result_check in result["advancement"].items():
        rows = [values for key, values in cohort.items() if key.endswith("/" + point)]
        tests = {
            "two_complete_roles": len(rows) == 2 and all(row["planned"] == 48 for row in rows),
            "useful_supplement": sum(row["added_useful"] for row in rows) >= rule["minimum_useful_same_carrier_additions"],
            "gain_in_both_roles": len(rows) == 2 and all(row["added_useful"] > 0 for row in rows),
            "location_retention": all(row[f"{fmt}_retained_useful"] >= row[f"{fmt}_useful"] * rule["minimum_useful_matched_retention"]
                                      for row in rows for fmt in ("wav", "aac")),
            "ordinary_flags": sum(row["ordinary_added_flag"] for row in rows) <= rule["maximum_new_ordinary_flags_vs_same_carrier"],
            "room_false_flags": sum(row["background_combined_flagged"] for row in rows) == rule["maximum_room_only_flags"],
        }
        require(result_check == {"tests": tests, "advances": all(tests.values())}, "Transfer advancement does not reproduce")
    return {"matched_cases": len(visited), "condition_cells": len(cohort), "reproduced_localization_counts": True,
            "geometry_upper_bound_before_admission": {role: dict(values) for role, values in geometry.items()},
            "paired_score_observations": {key: {**counts, "median_absolute_raw_logit_change": median(logit_deltas[key])}
                                          for key, counts in response_pairs.items()}}


def file_clock(path):
    data = json.loads(subprocess.check_output(
        ["ffprobe", "-v", "error", "-protocol_whitelist", "file", "-select_streams", "a:0", "-show_entries",
         "stream=sample_rate,channels,start_pts,time_base,codec_name:frame=pts,nb_samples",
         "-of", "json", str(path)], timeout=20))
    stream, = data["streams"]
    require(stream == {"codec_name": "opus", "sample_rate": "48000", "channels": 1,
                       "time_base": "1/48000", "start_pts": 0}, "Opus codec clock differs")
    position = 0
    for frame in data["frames"]:
        require(frame["pts"] == position and 0 < frame["nb_samples"] <= 960, "Decoded original frames skip or overlap")
        position += frame["nb_samples"]
    return position


def decode_sha(path, count):
    pcm = subprocess.check_output(
        ["ffmpeg", "-nostdin", "-v", "error", "-protocol_whitelist", "file", "-copyts", "-i", str(path),
         "-map", "0:a:0", "-af", "aformat=sample_fmts=s16:channel_layouts=mono,aresample=16000",
         "-c:a", "pcm_s16le", "-f", "s16le", "pipe:1"], timeout=20)
    require(len(pcm) == 2 * count, "Decoded samples differ from original presented duration")
    integers = array("h")
    integers.frombytes(pcm)
    if sys.byteorder == "big":
        integers.byteswap()
    floats = array("f", (value / 32768 for value in integers))
    if sys.byteorder == "big":
        floats.byteswap()
    return sha(floats.tobytes())


def verify_native(result, roots):
    with tempfile.TemporaryDirectory(prefix="opus-clock-verification-") as scratch:
        path = Path(scratch) / "clock.ogg"
        for row in result["rows"]:
            path.unlink(missing_ok=True)
            source = roots[row["role"]] / "pcm" / f"{row['pcm_sha256']}.pcm"
            require(sha(source.read_bytes()) == row["pcm_sha256"], "The archived comparison input has changed")
            subprocess.run(
                ["ffmpeg", "-nostdin", "-v", "error", "-f", "s16le", "-ar", "16000", "-ac", "1", "-i", str(source),
                 "-ar", "48000", "-c:a", "libopus", "-b:a", "64k", "-vbr", "off", "-frame_duration", "20",
                 "-application", "audio", "-fflags", "+bitexact", str(path)], check=True, capture_output=True, timeout=30)
            clock = row["clock"]
            require(sha(path.read_bytes()) == row["file_sha256"] and
                    file_clock(path) == 3 * row["samples"] == clock["presentation_end_exclusive_pts"]
                    and clock["end_granule"] == clock["pre_skip"] + 3 * row["samples"]
                    and clock["end_trim"] == 960 * clock["packets"] - clock["end_granule"]
                    and clock["pre_skip"] == 312 and clock["channels"] == 1
                    and decode_sha(path, row["samples"]) == row["decoded_pcm_sha256"], "Exact Opus fixture replay differs")
    return {"native_clocks": len(result["rows"]), "exact_opus_files": len(result["rows"]),
            "exact_decoded_pcm": len(result["rows"])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("receipt", "wav", "aac"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--native", nargs=2, type=Path, metavar=("FIRST", "SECOND"))
    args = parser.parse_args()
    result = read(args.receipt)
    require(result["plan"]["references"] == {"wav": sha(args.wav.read_bytes()), "aac": sha(args.aac.read_bytes())},
            "Comparators differ from the frozen reference digests")
    order = "reencoded", "transfer"
    evidence = compare_receipts(result, read(args.wav), read(args.aac))
    if args.native:
        evidence.update(verify_native(result, dict(zip(order, args.native, strict=True))))
    print(json.dumps(evidence, sort_keys=True))


if __name__ == "__main__":
    main()
