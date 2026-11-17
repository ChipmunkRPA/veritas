"""Verify local soundtrack differences on paired short native video deliveries."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from app.bench.independent_locations import _matches
from scripts import video_lip_correspondence as data

SCHEMA = "chipmunk.video-short-native-soundtrack/1"
DELIVERIES = (("pcm", "pcm"), ("aac48", "aac48"), ("pcm", "aac48"))
CONTENTS = {"same_words", "different_words"}


def _measure(folder, items):
    from app.evidence import video_reference

    rows = []
    for earlier, submitted in DELIVERIES:
        ref = items[earlier, "original"]
        for treatment in data.ROLES:
            case = items[submitted, treatment]
            older, later = (folder / "clips" / f"{item['id']}.mov" for item in (ref, case))
            result, assets = video_reference.compare(older, later, include_local=False, include_additional_soundtrack=False,
                include_sound_difference=True, include_short_native_sound_difference=True)
            if (result["earlier"]["sha256"], result["submitted"]["sha256"]) != (ref["input_sha256"], case["input_sha256"]):
                raise ValueError("Earlier or submitted video differs from the published input roster")
            checked = result["sound_difference"]
            match = checked.get("range")
            if match and (result["picture"]["changed_frames"] or len(match["paired_pictures"]) != 3 or not assets):
                raise ValueError("The localized result lacks identical source frames or playable evidence")
            rows.append({"source": case["source_sha256"], "speaker": case["group"], "delivery": f"{earlier}-to-{submitted}",
                         "role": treatment, "truth": case["truth"], "earlier": ref["input_sha256"], "submitted": case["input_sha256"],
                         "status": checked["status"], "reason": checked.get("reason"), "profile": checked.get("picture_profile"),
                         "presentation": checked.get("audio_presentation"), "samples": match["samples"] if match else None})
    return rows


def totals(rows):
    outcomes = defaultdict(Counter)
    for row in rows:
        counts = outcomes[f"{row['delivery']}/{row['role']}"]
        marked, truth = row["samples"], row["truth"]
        useful = _matches([truth], [marked], .40, .40) if marked and row["role"] in CONTENTS else 0
        counts.update(files=1, localized=bool(marked), useful=useful, unassessed=row["status"] == "unassessed")
        if row["reason"]:
            counts[row["reason"]] += 1
    return {key: dict(value) for key, value in sorted(outcomes.items())}


def run(folder, receipt, output):
    delivered = json.loads((folder / "manifest.json").read_bytes())
    single = data.read(receipt)
    truth_by_file = {(row["picture_sha256"], row["delivery"], row["role"]): row["truth"] for row in single["rows"]}
    grouped = defaultdict(dict)
    for case in delivered["cases"]:
        _, _, delivery, role = case["id"].split("-", 3)
        key = case["source_sha256"], delivery, role
        grouped[key[0]][delivery, role] = case | {"truth": truth_by_file[key]}
    if len(truth_by_file) != len(delivered["cases"]):
        raise ValueError("The paired video panel differs from the single-upload source panel")
    rows = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        for index, values in enumerate(pool.map(lambda items: _measure(folder, items), grouped.values()), 1):
            rows.extend(values)
            print(f"Paired comparison {index}/{len(grouped)} recordings", flush=True)
    result = {"schema": SCHEMA, "single_upload_receipt_sha256": data.sha(receipt),
              "playable_manifest_sha256": data.sha(folder / "manifest.json"), "rows": rows, "summary": totals(rows)}
    data.save(output, result)
    print(json.dumps(verify(result), sort_keys=True), flush=True)


def verify(result):
    if result["schema"] != SCHEMA or result["summary"] != totals(result["rows"]):
        raise ValueError("Paired soundtrack outcomes differ from their recorded intervals")
    for row in result["rows"]:
        if row["samples"] is not None:
            start, end = row["samples"]
            counts = [stream["presented_samples"] for stream in row["presentation"].values()]
            if len(counts) != 2 or counts[0] != counts[1] or not 0 <= start < end <= counts[0]:
                raise ValueError("A soundtrack range exceeds the shared source presentation")
    return result["summary"]


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    steps = cli.add_subparsers(dest="step", required=True)
    collect = steps.add_parser("run")
    for key in ("folder", "receipt", "output"):
        collect.add_argument(key, type=Path)
    replay = steps.add_parser("verify")
    replay.add_argument("receipt", type=Path)
    args = cli.parse_args()
    if args.step == "run":
        run(args.folder, args.receipt, args.output)
    else:
        print(json.dumps(verify(data.read(args.receipt)), sort_keys=True))
