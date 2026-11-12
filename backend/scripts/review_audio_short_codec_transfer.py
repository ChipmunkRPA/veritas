"""Independently check codec-study calibration, held metrics, and optional native bindings."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RECEIPTS = ROOT / "docs/evaluation-panels"
BASE = RECEIPTS / "audio-edge-window-grouped-2026-11-12.json"
RECEIPT = RECEIPTS / "audio-short-codec-transfer-2026-11-12.json"
CONFIRM_SOURCE = RECEIPTS / "audio-communication-replication-2026-11-12.json"
CONFIRM_SCHEMA = "chipmunk.audio-short-mp3-confirmation/1"
SCHEMA = "chipmunk.audio-short-codec-transfer/1"
GRID = ((0, 64000), (32000, 96000), (0, 32000), (16000, 48000), (32000, 64000), (48000, 80000), (64000, 96000))
ARMS = {"full": (0, 1), "combined": (2, 3, 4, 5, 6)}
CODECS = ("mp3", "opus")
BUDGETS = (2, 1)


def load(path):
    return json.loads(path.read_bytes())


def dump(path, data):
    path.write_text(json.dumps(data, sort_keys=True, indent=2, allow_nan=False) + "\n")


def rows_digest(rows):
    return hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def strongest(scores, arm):
    # A stable sort of indices preserves the earlier window on ties.
    ranked = sorted((i for i in ARMS[arm] if scores[i] is not None), key=lambda i: -scores[i])
    return (scores[ranked[0]], GRID[ranked[0]]) if ranked else (None, None)


def calibrate(rows, codec, arm, budget):
    responses = defaultdict(list)
    for row in rows:
        score, _ = strongest(row["scores"][codec], arm)
        if score is not None:
            responses[row["group"]].append(score)
    ranked = sorted(max(values) for values in responses.values())
    require(len(ranked) >= 12 and sum(v > ranked[-(budget + 1)] for v in ranked) == budget,
            "Codec point does not match its genuine-group allowance")
    return ranked[-(budget + 1)]


def held_metrics(rows, codec, arm, cutoff):
    totals = dict.fromkeys(("assessed", "flagged_edits", "flags_covering_half", "useful", "wrong_location",
                            "missed_edits", "false_files", "room_false_files"), 0)
    false_groups, useful_groups = set(), set()
    for row in rows:
        score, span = strongest(row["scores"][codec], arm)
        totals["assessed"] += score is not None
        flag = score is not None and score > cutoff
        if row["truth"] is None:
            if flag:
                false_groups.add(row["group"])
                totals["false_files"] += 1
                totals["room_false_files"] += row["condition"] == "background_only"
            continue
        totals["missed_edits"] += not flag
        if flag:
            totals["flagged_edits"] += 1
            left, right = row["truth"]
            a, b = span
            intersection = max(0, min(right, b) - max(left, a))
            union = right - left + b - a - intersection
            covers_half = 2 * intersection >= right - left
            useful = covers_half and 10 * intersection >= 3 * union
            totals["flags_covering_half"] += covers_half
            totals["useful"] += useful
            totals["wrong_location"] += not useful
            if useful:
                useful_groups.add(row["group"])
    return totals | {"false_groups": sorted(false_groups), "useful_groups": sorted(useful_groups)}


def reproduce(calibration, held, wav):
    points = {codec: {arm: {str(k): calibrate(calibration, codec, arm, k) for k in BUDGETS} for arm in ARMS}
              for codec in CODECS}
    points["wav"] = {arm: {str(k): wav["points"][arm][str(k)]["point"] for k in BUDGETS} for arm in ARMS}
    results = {codec: {arm: {str(k): held_metrics(held, codec, arm, point) for k, point in by_budget.items()}
                       for arm, by_budget in arms.items()} for codec, arms in points.items()}
    admission = {}
    for codec in CODECS:
        full, short = (results[codec][arm]["2"] for arm in ARMS)
        new_useful = set(short["useful_groups"]) - set(full["useful_groups"])
        new_false = set(short["false_groups"]) - set(full["false_groups"])
        admission[codec] = {"new_useful_groups": len(new_useful), "new_false_groups": len(new_false),
                            "passes": len(new_useful) >= 2 and len(new_false) <= 1}
    return points, results, admission


def inspect(receipt, wav):
    require(receipt["schema"] == SCHEMA and receipt["baseline"]["sha256"] == hashlib.sha256(BASE.read_bytes()).hexdigest(),
            "Original WAV study commitment changed")
    calibration, held = receipt["calibration"], receipt["held"]
    require(len(calibration) == 40 and len(held) == len(wav["held"]) == 50, "Study denominators changed")
    require(len({row["group"] for row in calibration}) == 16 and len({row["group"] for row in held}) == 14,
            "Source-group denominators changed")
    require(all(row["truth"] is None for row in calibration) and sum(row["truth"] is not None for row in held) == 12,
            "Label allocation changed")
    records = calibration + held
    require(receipt["response_rows"]["sha256"] == rows_digest(records), "Response rows changed")
    require(len({row["id"] for row in records}) == len(records), "Duplicate source record")
    for label, rows in (("calibration", calibration), ("held", held)):
        for index, row in enumerate(rows):
            require(row["id"] == f"{label}-{index:02d}" and row["group"].startswith(label + "-group-"),
                    "Source split or order changed")
            require(set(row["scores"]) == (set(CODECS) | ({"wav"} if label == "held" else set())),
                    "Missing response arm")
            for scores in row["scores"].values():
                require(len(scores) == len(GRID) and all(value is None or (
                    type(value) in (int, float) and math.isfinite(value)) for value in scores), "Invalid window response")
    groups = {}
    for row, source in zip(held, wav["held"], strict=True):
        group = groups.setdefault(source["group"], row["group"])
        require(group == row["group"] and row["scores"]["wav"] == source["scores"]
                and (row["truth"], row["family"], row["condition"]) == (source["truth"], source["family"], source["arm"]),
                "Paired held results no longer match the published WAV source row")
    require(len(set(groups.values())) == len(groups) == 14, "Held source groups were merged")
    actual = reproduce(calibration, held, wav)
    require(actual == (receipt["points"], receipt["results"], receipt["codec_admission"]), "Held comparison changed")
    require(receipt["lossless_exact_deliveries"] == 90 and receipt["native_window_bindings"] == 1260
            and receipt["confidence_weight"] == 0, "Unbound public scope")
    return actual


def native(run, held):
    from scripts import audio_short_codec_transfer as producer

    wav = load(BASE)
    split_rows = [load(run / f"{stage}.json") for stage in producer.STAGES]
    selected = [record for rows in split_rows for record in rows]
    original = list(producer.inputs(held))
    freeze = load(run / "freeze.json")
    require(freeze["inputs"]["sha256"] == producer.sha(run / "inputs.json")
            and freeze["held"]["sha256"] == producer.sha(held / "sources.json"), "Input freeze changed")
    checks = 0
    frozen_rows = load(run / "inputs.json")
    for row, frozen, (_, key, expected, source_pcm) in zip(selected, frozen_rows, original, strict=True):
        require({key: value for key, value in row.items() if key != "windows"} == frozen, "Scored row differs from its sealed input")
        require(row["id"] == key and row["original"]["sha256"] == producer.digest(source_pcm)
                and row["truth"] == expected["truth"] and row["condition"] == expected["arm"], "Source assignment changed")
        for codec, delivery in row["deliveries"].items():
            path = run / delivery["file"]
            pcm, clock = producer.decode_delivered(path, codec)
            require(delivery["bytes"]["sha256"] == producer.sha(path) and clock == delivery["clock"]
                    and delivery["pcm"]["sha256"] == producer.digest(pcm)
                    and (pcm == source_pcm) == (codec == "flac"), "Native file, PCM, or clock changed")
            if codec in CODECS:
                scored = row["windows"][codec]
                require(tuple((win["start"], win["end"]) for win in scored) == GRID and all(
                    win["pcm_sha256"] == producer.digest(pcm[win["start"]*2:win["end"]*2]) for win in scored),
                    "Scored range does not match exact decoded audio")
                checks += len(scored)

    def scrub(row):
        return {key: row[key] for key in ("id", "group", "truth", "family", "condition")} | {
            "scores": {codec: [value["score"] for value in row["windows"][codec]] for codec in CODECS}}

    calibration, held_scores = [[scrub(row) for row in rows] for rows in split_rows]
    for row, source in zip(held_scores, wav["held"], strict=True):
        row["scores"]["wav"] = source["scores"]
    points, results, admission = reproduce(calibration, held_scores, wav)
    require(load(run / "points.json") == {codec: points[codec] for codec in CODECS}, "Pre-inference point file changed")
    result = {"schema": SCHEMA, "baseline": {"sha256": producer.sha(BASE)}, "freeze": {"sha256": producer.sha(run / "freeze.json")},
              "calibration": calibration, "held": held_scores, "points": points, "results": results,
              "response_rows": {"sha256": rows_digest(calibration + held_scores)},
              "codec_admission": admission, "lossless_exact_deliveries": 90, "native_window_bindings": checks,
              "confidence_weight": 0}
    inspect(result, wav)
    print(f"Re-decoded {len(selected)*3} deliveries, compared 90 lossless files, and verified {checks} exact window bindings.")
    return result


def confirm_metrics(rows, cutoffs):
    metrics = {arm: held_metrics(rows, "mp3", arm, cutoffs[arm]) for arm in ARMS}
    full, short = (metrics[arm] for arm in ARMS)
    recovered = len(set(short["useful_groups"]) - set(full["useful_groups"]))
    new_false = len(set(short["false_groups"]) - set(full["false_groups"]))
    return metrics, {"new_useful_groups": recovered, "new_false_groups": new_false,
                     "passes": recovered >= 2 and new_false <= 1}


def confirm_inspect(receipt):
    require(receipt["schema"] == CONFIRM_SCHEMA
            and receipt["earlier_codec_receipt"]["sha256"] == hashlib.sha256(RECEIPT.read_bytes()).hexdigest()
            and receipt["source_receipt"]["sha256"] == hashlib.sha256(CONFIRM_SOURCE.read_bytes()).hexdigest(),
            "Confirmation source or earlier cutoff commitment changed")
    prior = load(RECEIPT)
    points = {arm: prior["points"]["mp3"][arm]["1"] for arm in ARMS}
    rows = receipt["held"]
    require(receipt["cutoffs"] == points and len(rows) == 52 and len({row["group"] for row in rows}) == 16
            and receipt["response_rows"]["sha256"] == rows_digest(rows), "Confirmation role or point changed")
    require(sum(row["truth"] is not None for row in rows) == 12 and receipt["native_window_bindings"] == 364
            and receipt["confidence_weight"] == 0, "Confirmation denominator or scope changed")
    for number, row in enumerate(rows):
        scores = row["scores"]["mp3"]
        require(row["id"] == f"case-{number:02d}" and len(scores) == len(GRID)
                and all(value is None or (type(value) in (int, float) and math.isfinite(value)) for value in scores),
                "Confirmation windows or file ordering changed")
        truth = row["truth"]
        require((truth is not None) == (row["condition"] == "synthetic")
                and (truth is None or (len(truth) == 2 and all(type(i) is int for i in truth)
                     and 0 <= truth[0] < truth[1] <= 96000)), "Invalid known edit interval")
    results, admission = confirm_metrics(rows, points)
    require(results == receipt["results"] and admission == receipt["admission"], "Confirmation outcomes changed")
    return results, admission


def confirm_native(run):
    from scripts import audio_short_codec_transfer as codec
    from scripts import audio_short_mp3_confirmation as producer

    selected, scored = (load(run / f"{name}.json") for name in ("inputs", "scores"))
    freeze = load(run / "freeze.json")
    require(freeze == producer.frozen(run, selected), "Confirmation freeze changed")
    rows, checks = [], 0
    for source, result, (prior, original) in zip(selected, scored, producer.source_cases(), strict=True):
        require({key: result[key] for key in source} == source and {key: source[key] for key in prior} == prior
                and prior["original"]["sha256"] == codec.digest(original), "Confirmation input assignment changed")
        item = source["delivery"]
        path = run / item["path"]
        pcm, clock = codec.decode_delivered(path, "mp3")
        require(item["bytes"]["sha256"] == codec.sha(path) and item["pcm"]["sha256"] == codec.digest(pcm)
                and item["clock"] == clock, "Confirmation native source or clock changed")
        windows = result["windows"]
        require(tuple((window["start"], window["end"]) for window in windows) == GRID and all(
            window["pcm_sha256"] == codec.digest(pcm[2*window["start"]:2*window["end"]]) for window in windows),
            "Confirmation window input changed")
        checks += len(windows)
        rows.append({key: prior[key] for key in ("id", "group", "truth", "family", "condition")} |
                    {"scores": {"mp3": [window["score"] for window in windows]}})
    results, admission = confirm_metrics(rows, freeze["cutoffs"])
    require(load(run / "results.json") == {"freeze": {"sha256": codec.sha(run / "freeze.json")}, "metrics": results},
            "Scorer and independent confirmation metrics differ")
    receipt = {key: freeze[key] for key in ("schema", "earlier_codec_receipt", "source_receipt", "cutoffs",
                                           "generator_profiles_previously_used")} | {
        "held": rows, "response_rows": {"sha256": rows_digest(rows)}, "results": results, "admission": admission,
        "freeze": {"sha256": codec.sha(run / "freeze.json")}, "native_window_bindings": checks, "confidence_weight": 0,
    }
    confirm_inspect(receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", nargs="?", type=Path, default=RECEIPT)
    parser.add_argument("--native", nargs=2, type=Path, metavar=("RUN", "HELD"))
    parser.add_argument("--confirmation", action="store_true", help="Review the separate MP3 confirmation receipt")
    parser.add_argument("--confirmation-native", type=Path, metavar="RUN")
    args = parser.parse_args()
    if args.confirmation or args.confirmation_native:
        require(args.receipt != RECEIPT and args.native is None, "Supply a separate confirmation receipt path")
        if args.confirmation_native:
            dump(args.receipt, confirm_native(args.confirmation_native))
        results, admission = confirm_inspect(load(args.receipt))
        print(json.dumps({"results": results, "admission": admission}, sort_keys=True))
        return
    if args.native:
        dump(args.receipt, native(*args.native))
    _, results, admission = inspect(load(args.receipt), load(BASE))
    for codec in CODECS:
        short, scope = results[codec]["combined"]["2"], admission[codec]
        print(f"{codec}: {short['useful']}/12 useful, {len(short['false_groups'])}/14 false groups, {scope}")


if __name__ == "__main__":
    main()
