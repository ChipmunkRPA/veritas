# SPDX-License-Identifier: Apache-2.0
"""Bind the fixed candidate and compare it with sealed paid Full evidence."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

import cv2

from scripts import replay_video_native_background_panel as incumbent
from scripts import video_native_background_edges as candidate
from scripts import video_native_background_panel as panel

common, require = panel.common, panel.require
CANDIDATE_SHA256 = "ab7908fbaafe35a5e879190ada426401d7b248e8d37069db733b330a5df2eca6"


def candidate_lock(folder):
    actual = common.digest(Path(candidate.__file__))
    require(actual == CANDIDATE_SHA256, "Only the exact pre-held candidate is eligible")
    return {"schema": "chipmunk.video-native-background-candidate/1", "candidate_sha256": actual,
            "panel_freeze_sha256": common.digest(folder / "freeze.json"),
            "candidate_binding": "Decode the verified native output clock. Supply each measured incumbent coarse record and its face boxes. No truth, condition, file pairing, or reported scene location enters the method.",
            "comparison_worker_sha256": common.digest(Path(__file__)),
            "incumbent_worker_sha256": common.digest(Path(incumbent.__file__)),
            "advancement": panel.SPEC["advancement"]}


def observations(folder, roster):
    sealed = common.read(folder / "incumbent-seal.json")
    journal = folder / "private-incumbent.jsonl"
    require(sealed["status"] == "verified" and sealed["cases"] == len(roster)
            and sealed["journal_sha256"] == common.digest(journal)
            and sealed["lock_sha256"] == common.digest(folder / "incumbent-lock.json")
            and sealed["media_sha256"] == common.digest(folder / "media.json"), "Incumbent records are incomplete or changed")
    rows = [json.loads(line) for line in journal.read_text().splitlines()]
    require(len(rows) == len(roster) and common.read(folder / "incumbent-lock.json") == incumbent.commit(folder, roster),
            "Incumbent code or roster no longer matches the worker execution")
    for row, item in zip(rows, roster, strict=True):
        require(all(row[key] == item[key] for key in ("file", "sha256", "pair", "direction", "condition", "truth", "interval"))
                and row["paid_before_analysis"] and row["paid_event_idempotent"], "Incumbent receipt binds the wrong source or payment state")
    return rows


def frame_span(record, table):
    require(record["time_base"] == table.time_base, "Reported range clock differs")
    return [table.pts.index(record["start_pts"]), table.pts.index(record["end_pts"])]


def useful(ranges, truth):
    return bool(truth and any(panel.old.iou(span, truth) >= Fraction(1, 2) for span in ranges))


def score_case(folder, row, item):
    path = folder / "media" / item["file"]
    require(common.digest(path) == item["sha256"], "Delivered source bytes differ")
    table = panel.previous.prior.clock(path)
    require(table.to_json() == item["clock"], "Delivered source presentation table differs")
    coarse = row["coarse"]
    if coarse:
        for frame in coarse["frame_inputs"]:
            index = frame["frame_index"]
            require(frame["pts"] == table.pts[index] and frame["decoded_rgb_sha256"] == item["rgb_sha256"][index], "Coarse input crop frame differs")
    found = candidate.examine(path, coarse, table=table) if coarse else None
    if found:
        a,b = found["edge_frames"]
        require(found["method"] == candidate.METHOD and found["time_base"] == table.time_base
                and found["coarse_frames"] == frame_span(coarse,table)
                and found["edge_pts"] == [table.pts[a],table.pts[b]], "Native candidate timestamps differ")
        require([frame["index"] for frame in found["pictures"]] == [a-1,a,b-1,b], "Exact adjacent cut pictures required")
        for frame in found["pictures"]:
            i = frame["index"]
            require(frame["pts"] == table.pts[i] and frame["rgb_sha256"] == item["rgb_sha256"][i], "Native candidate picture differs")
        require(set(found["support_tiles"]) <= set(found["eligible_tiles"]), "Non-face evidence support differs")
    raw = {key: frame_span(row[key],table) for key in ("coarse", "boundary_refinement", "short_range") if row[key]}
    displayed = {key: frame_span(packet[key],table) for packet in row["displayed"] for key in packet}
    require(all(displayed.get(name, raw.get(key)) == raw.get(key)
                for name, key in (("face_sequence","coarse"),("face_boundary","boundary_refinement"),("face_return","short_range"))),
            "Displayed range differs from the measured lane")
    preserved = {key:span for key,span in displayed.items() if not found or key == "face_return"}
    labeled_cut = item["condition"] == "cross_cut"
    exact = bool(found and labeled_cut and all(abs(got-want) <= 1 for got,want in zip(found["edge_frames"],item["interval"],strict=True)))
    exact_both = exact and found["edge_frames"] == item["interval"]
    useful_coarse = useful([span for key,span in displayed.items() if key in ("face_sequence","face_boundary")], item["truth"])
    return {"source_sha256":item["sha256"], "condition":item["condition"], "source_pair":item["pair"], "direction":item["direction"],
            "authored":item["interval"], "true_swap":bool(item["truth"]), "incumbent":raw,
            "displayed":displayed, "counterfactual_displayed":preserved,
            "shot_boundaries":row["shot_boundaries"], "track_status":row["track_status"],"tracks":row["tracks"],
            "coarse_status":row["coarse_status"], "appearance_status":row["appearance_status"], "short_coverage":row["short_coverage"],
            "candidate":found, "cut_both_edges_within_one_picture":exact, "cut_both_edges_exact":exact_both,
            "useful_cut_removal":exact and "face_sequence" in displayed,
            "false_matched_background_cut":bool(found and not labeled_cut),
            "useful_swap_coarse":useful_coarse,
            "lost_useful_swap_range":bool(found and useful_coarse),
            "current_useful_swap":useful(displayed.values(),item["truth"]),
            "counterfactual_useful_swap":useful(preserved.values(),item["truth"])}


def summarize(rows):
    counts = defaultdict(Counter)
    for row in rows:
        out = counts[row["condition"]]
        out["cases"] += 1
        out["shot_transitions"] += bool(row["shot_boundaries"])
        out["coarse_review_measured"] += row["coarse_status"] == "ok"
        out["coarse_ranges"] += "coarse" in row["incumbent"]
        out["displayed_coarse_ranges"] += "face_sequence" in row["displayed"]
        out["displayed_short_ranges"] += "face_return" in row["displayed"]
        out["counterfactual_coarse_ranges"] += "face_sequence" in row["counterfactual_displayed"]
        out["counterfactual_short_ranges"] += "face_return" in row["counterfactual_displayed"]
        out["candidate_matched_background_pairs"] += row["candidate"] is not None
        for key in ("cut_both_edges_within_one_picture","cut_both_edges_exact","useful_cut_removal","false_matched_background_cut",
                    "useful_swap_coarse","lost_useful_swap_range","current_useful_swap","counterfactual_useful_swap"):
            out[key] += bool(row[key])
    distinct = {row["source_pair"] for row in rows if row["useful_cut_removal"]}
    expected = panel.SPEC["advancement"]
    gate = {"useful_cut_removals": counts["cross_cut"]["useful_cut_removal"] >= expected["min_useful_genuine_cut_removals"],
            "distinct_source_pairs": len(distinct) >= expected["min_distinct_source_pairs"],
            "useful_swap_coarse_coverage": sum(out["useful_swap_coarse"] for out in counts.values()) >= expected["min_coarse_useful_swap_assessment"],
            "useful_swap_preservation": sum(out["lost_useful_swap_range"] for out in counts.values()) == 0,
            "no_false_matched_background_cuts": sum(out["false_matched_background_cut"] for out in counts.values()) == 0}
    return {"held_counts": {key:dict(value) for key,value in counts.items()}, "useful_cut_source_pairs":len(distinct),
            "gate":gate,"advances":all(gate.values())}


def run(folder, archive):
    lock = panel.frozen(folder,archive)
    frozen = candidate_lock(folder)
    require(frozen == common.read(folder / "candidate-freeze.json"), "Candidate was not frozen against this panel")
    manifest = common.read(folder / "media.json")
    require(manifest["freeze_sha256"] == common.digest(folder / "freeze.json"), "Source manifest differs")
    roster = manifest["media"]
    rows = observations(folder,roster)
    journal = folder / "private-candidate.jsonl"
    results = []
    started = time.monotonic()
    with journal.open("x") as stream:
        for row, item in zip(rows,roster,strict=True):
            result = score_case(folder,row,item)
            stream.write(json.dumps(result,allow_nan=False,sort_keys=True)+"\n")
            stream.flush()
            results.append(result)
    require(len(results) == 64 and candidate_lock(folder) == frozen, "Incomplete or changed held experiment")
    output = {**summarize(results), "protocol": lock["spec"], "held_source_pairs":len(lock["pairs"]),
              "held_source_groups":len(lock["source_roster"]), "eligible_source_groups":lock["available"],
              "matching_distances": [distance for distance,_,_ in lock["pairs"]],
              "panel_freeze_sha256": frozen["panel_freeze_sha256"], "candidate_sha256":frozen["candidate_sha256"],
              "candidate_freeze_sha256":common.digest(folder/"candidate-freeze.json"),
              "incumbent_seal_sha256":common.digest(folder/"incumbent-seal.json"),
              "candidate_journal_sha256":common.digest(journal), "elapsed_s":time.monotonic()-started,
              "serving_change":False}
    common.write(folder / "aggregate.json",output)
    print(json.dumps(output,sort_keys=True),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze","run"))
    parser.add_argument("folder",type=Path)
    parser.add_argument("--archive",type=Path,default=panel.ARCHIVE)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    if args.phase == "freeze":
        panel.frozen(args.folder,args.archive)
        require(not (args.folder / "private-incumbent.jsonl").exists(), "Freeze candidate bindings before held analysis")
        common.write(args.folder / "candidate-freeze.json",candidate_lock(args.folder))
    else:
        run(args.folder,args.archive)


if __name__ == "__main__":
    main()
