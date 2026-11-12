"""Rebuild six-second matched tracks and compare exact supported audio intervals."""

from __future__ import annotations

import argparse
import math
from array import array
from collections import Counter
from hashlib import sha256
from pathlib import Path

from scripts import audio_overlap_window_transfer as study


def decode(raw):
    values = array("h")
    values.frombytes(raw)
    return values


def exact_deliveries(pair, background):
    clean, edited, extra, before = pair
    extension = study.edits.exact_source(extra, clean["role"])
    sources = []
    for center in (clean, edited):
        raw = study.retained_case(center)
        n = before * 2
        sources.append(decode(extension[:n] + raw + extension[n:]))
    noise = decode(background)
    ratio = math.sqrt(sum(x*x for x in sources[0]) / sum(x*x for x in noise)) * 10**.25
    nuisance = [x * ratio for x in noise]
    peak = max(abs(v) + abs(n) for source in sources for v, n in zip(source, nuisance, strict=True))
    gain = min(1., 31129 / peak)
    mixed = [[s+n for s, n in zip(source, nuisance, strict=True)] for source in sources]
    return [array("h", (round(s * gain) for s in values)).tobytes() for values in [*sources, *mixed]]


def voiced(spans, start, end):
    amount = sum(max(0., min(end, b * 16000) - max(start, a * 16000)) for a, b in spans)
    return 2 * amount + 3.2e-5 >= end - start


def as_logit(raw):
    bounded = max(1e-6, min(1-1e-6, raw))
    return math.log(bounded) - math.log1p(-bounded)


def features(raw, observation):
    require = study.require
    spans = observation["speech_spans"]
    speech = [start for start in (0, 32_000) if voiced(spans, start, start + 64_000)]
    full, short = observation["full"], observation["short"]
    require([row["start"] for row in full] == speech and observation["current"] == ("inconclusive" if speech else "unassessed"),
            "Current window admission or out-of-scope direction differs")
    candidates = {arm: [] for arm in study.ARMS[1:-1]}
    for row in full:
        start, end = row["start"], row["end"]
        require(end == start + 64_000 and sha256(raw[start*2:end*2]).hexdigest() == row["pcm_sha256"]
                and 0 <= row["raw"] <= 1, "Incumbent native window differs")
        candidates["long_window"].append([as_logit(row["raw"]), [start, end]])
    require(len(short) == 5, "Short-window roster differs")
    for index, row in enumerate(short):
        start, end = 16_000 * index, 16_000 * (index+2)
        admitted = row["score"] is not None
        require((row["start"], row["end"]) == (start, end) and sha256(raw[start*2:end*2]).hexdigest() == row["pcm_sha256"]
                and admitted == voiced(spans, start, end), "Short window bytes or speech admission differ")
        if admitted:
            require(math.isfinite(row["score"]), "Invalid short score")
            arm = "short_even" if index % 2 == 0 else "short_odd"
            candidates[arm].append([row["score"], [start, end]])
            if index and short[index-1]["score"] is not None and voiced(spans, start, end-16_000):
                candidates["overlap"].append([min(short[index-1]["score"], row["score"]), [start, end-16_000]])
    selected = {arm: max(opts, key=lambda row: row[0], default=None) for arm, opts in candidates.items()}
    require(selected == observation["features"], "Independently reconstructed window score or interval differs")
    return selected


def measure(interval, truth):
    if interval is None or truth is None:
        return False
    a, b = interval
    c, d = truth
    common = len(range(max(a, c), min(b, d)))
    return common / (d-c) >= .5 and common / ((b-a)+(d-c)-common) >= .30


def verify(args):
    output = args.output
    lock, roster = study.checked(args)
    digest, read, require = study.digest, study.read, study.require
    result, point = (read(output / f"{file}.json") for file in ("result", "points"))
    require(digest(output / "freeze.json") == result["freeze_sha256"] == point["freeze_sha256"]
            and digest(output / "points.json") == result["points_sha256"], "Study or operating points differ")
    cuts = study.cuts(args)
    require([row for _, row in cuts] == lock["rooms"] and digest(args.annotations) == lock["annotations_sha256"], "Room cut lineage differs")
    ledgers, total = {}, Counter()
    for panel, role in study.PARTS.items():
        receipts_path = output / f"{panel}.json"
        require(digest(receipts_path) == result["ledgers_sha256"][panel], "Saved classifier receipts differ")
        if panel == "selection":
            require(digest(receipts_path) == point["selection_sha256"], "Calibration used a different selection roster")
        ledger = read(receipts_path)
        require(ledger["freeze_sha256"] == result["freeze_sha256"], "Model scoring source freeze differs")
        sources, commitments = study.source_pairs(role, 2)
        require(commitments == lock["source_receipts"][panel], "Source edit and speaker lineage differs")
        expected = []
        for index, pair in enumerate(sources):
            control, synthetic, extra, offset = pair
            room = 0 if panel == "selection" else (index + index // 2) % 2 + 1
            files = exact_deliveries(pair, cuts[room][0])
            left, right = (offset + value for value in study.edits.controls.TARGET_SAMPLES)
            for n, raw in enumerate(files):
                require(files[n ^ 1][:left*2] == raw[:left*2] and files[n ^ 1][right*2:] == raw[right*2:],
                        "Matched controls diverge outside their splice")
                truth = [offset + v for v in synthetic["label"]["ai_support_samples"]] if n % 2 else None
                expected.append((raw, truth, synthetic["lane"], synthetic["genuine_host_group"], room if n >= 2 else None))
        current = [row for row in roster if row["panel"] == panel]
        ledgers[panel] = []
        for row, observation, (raw, truth, family, host, room) in zip(current, ledger["rows"], expected, strict=True):
            require((truth, family, host, room) == (row["truth"], row["family"], row["host_group"], row["room_index"])
                    and ("partial_ai" if truth else "ordinary") == row["kind"] and ("clean" if room is None else "room") == row["condition"],
                    "Matched truth, room or source group differs")
            pcm_hash = sha256(raw).hexdigest()
            require(pcm_hash == digest(output / "pcm" / f"{row['id']:03}.pcm") == observation["input_sha256"] == row["source_pcm_sha256"]
                    and row["id"] == observation["id"], "Exact source PCM differs")
            selected = features(raw, observation)
            ledgers[panel].append((row, observation["current"], selected))
            total.update({"tracks": 1, "short_windows": len(observation["short"]), "full_windows": len(observation["full"]),
                          "short_forwards": sum(v["score"] is not None for v in observation["short"])})
        require(len(ledgers[panel]) == lock["planned"][panel] and ledger["classifier_forwards"] == sum(
                len(v["full"]) + sum(w["score"] is not None for w in v["short"]) for v in ledger["rows"]), "Model forward count differs")
    points = {}
    for arm in study.ARMS[1:-1]:
        negatives = [(scores[arm][0], row["host_group"]) for row, _, scores in ledgers["selection"] if row["truth"] is None and scores[arm] is not None]
        points[arm] = max(score for score, _ in negatives)
        require(point["points"][arm] == {"point": points[arm], "genuine_assessed": len(negatives),
                "genuine_groups": len({host for _, host in negatives})}, "Selection score bound differs")
    cells, no_point, matched_ranks, held_subgroups = 0, {}, {}, {}
    for panel, rows in ledgers.items():
        counters = {condition: {arm: Counter() for arm in study.ARMS} for condition in ("clean", "room")}
        pairs = {}
        for row, current, selected in rows:
            pairs.setdefault((row["condition"], row["host_group"]), {})[row["kind"]] = selected
            decisions = {"current": (current, None)}
            for arm, values in selected.items():
                if row["truth"] is not None and values and measure(values[1], row["truth"]):
                    no_point.setdefault(panel, Counter())[f"{row['condition']}/{arm}"] += 1
                state = "unassessed" if values is None else "ai" if values[0] > points[arm] else "inconclusive"
                decisions[arm] = state, values[1] if state == "ai" else None
            decisions["screening"] = decisions["overlap"] if current == "inconclusive" and decisions["overlap"][0] != "unassessed" else decisions["current"]
            for arm, (direction, support) in decisions.items():
                counts = counters[row["condition"]][arm]
                counts[f"{row['kind']}_{direction}"] += 1
                if panel == "confirmation":
                    key = "clean" if row["room_index"] is None else f"room_{row['room_index']}"
                    subgroup = held_subgroups.setdefault(key, {}).setdefault(arm, Counter())
                    subgroup[f"{row['kind']}_{direction}"] += 1
                if direction == "ai" and measure(support, row["truth"]):
                    counts["useful_location"] += 1
                    counts[f"useful_{row['family']}"] += 1
                    if panel == "confirmation":
                        subgroup[f"useful_{row['family']}"] += 1
        require(counters == result["results"][panel], "Exact outcome cells differ")
        cells += sum(len(cell) for cell in counters.values())
        matched_ranks[panel] = {}
        for (condition, _), pair in pairs.items():
            cell = matched_ranks[panel].setdefault(condition, {})
            for arm in study.ARMS[1:-1]:
                a, b = pair["ordinary"][arm], pair["partial_ai"][arm]
                ranking = "unassessed" if a is None or b is None else "above" if b[0] > a[0] else "below" if b[0] < a[0] else "tie"
                cell.setdefault(arm, Counter())[ranking] += 1
    held = result["results"]["confirmation"]
    noisy = held["room"]["screening"]
    conditions = {
        "noisy_useful_locations": noisy.get("useful_location", 0) >= 2,
        "noisy_two_families": sum(k.startswith("useful_") and k != "useful_location" for k in noisy) >= 2,
        "noisy_false_limits": noisy.get("ordinary_ai", 0) - held["room"]["long_window"].get("ordinary_ai", 0) <= 1,
        "clean_false_limits": held["clean"]["screening"].get("ordinary_ai", 0) <= held["clean"]["long_window"].get("ordinary_ai", 0),
    }
    require(conditions == result["criteria"] and result["advances"] == all(conditions.values()), "Independent advancement differs")
    require(total["tracks"] == 48 and total["short_windows"] == 240 and cells == 24, "Incomplete independent replay")
    out = {**total, "decision_cells": cells, "threshold_free_useful_locations": no_point, "matched_pair_ranks": matched_ranks,
           "held_subgroups": held_subgroups, "criteria": conditions,
           "result_sha256": digest(output / "result.json")}
    study.write(output / "audit.json", out)
    print(out, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("output", "annotations", "rooms"):
        parser.add_argument(name, type=Path)
    verify(parser.parse_args())
