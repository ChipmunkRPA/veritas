# SPDX-License-Identifier: Apache-2.0
"""Measure a second distinct face change on the same verified side of a video."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from scripts.review_video_brief_face_120_carrier import SHA, expected_tick, overlap, require
from scripts.video_face_primary_track_transfer import ROOT, measure, replay_pixels, sha

SCHEMA = "chipmunk.video-face-second-pane/1"
PARENT = ROOT / "docs/evaluation-panels/video-face-split-track-2026-11-13.json"
PROTOCOL = {
    "selection": "Retain the stronger first pane result and search a second range on that side only. Reuse the same frames, edges, pane motion guard and response point. Search five to nine pictures and require at least three pictures between intervals.",
    "advance": "At least four additional exact locations on three source recordings, at most one misplaced range and no additional ordinary control flags. Preserve every first range. This qualifies an opt-in local trial and carries zero confidence weight.",
}
SINGLE_PROTOCOL = "One edit from each original pair is retained at its original location. Use alternating left and right deliveries across groups. Compare against the unedited layout with a full re-encode of every case."


def prepare_single(bank, output, parent):
    import numpy as np

    from app.detectors.video.decode import FrameTable, decode_frames
    from scripts.video_face_720p30_transfer import authored, describe

    manifest = bank / "media.json"
    require(sha(manifest) == parent["media_sha256"], "The retained delivery roster differs")
    by_kind = {(r["group"], r["condition"]): r for r in json.loads(manifest.read_bytes())["rows"]}
    output.mkdir(mode=0o700)
    media = output / "media"
    media.mkdir(mode=0o700)

    def pictures(item):
        path = bank / "media" / item["file"]
        require(sha(path) == item["sha256"], "Delivery bytes differ")
        for frame in decode_frames(path, FrameTable.from_json(item["clock"]), range(120), timeout_s=35):
            require(frame.pts_verified and frame.pts == item["clock"]["pts"][frame.index],
                    "Delivered picture has no verified timestamp")
            yield frame.rgb

    prepared = []
    for group in sorted({item["group"] for item in by_kind.values()}):
        side = "left" if group % 2 else "right"
        edited = by_kind[group, f"swaps_{side}"]
        original = by_kind[group, "original"]
        for label, (start, end) in enumerate(edited["truth"], 1):
            condition = f"single_{label}_{side}"
            plain = pictures(original)
            # Swap pane positions in the unchanged view when the edit is on the right.
            if side == "right":
                plain = (np.roll(frame, 640, axis=1) for frame in plain)
            path = media / f"g{group}-{condition}.mp4"
            authored(path, pictures=(changed if start <= i < end else clean
                                     for i, (clean, changed) in enumerate(zip(plain, pictures(edited), strict=True))))
            prepared.append(describe(path, group, condition, [[start, end]]) | {"expected_pane": side})
        print("Prepared single-edit group", group, flush=True)
    (output / "media.json").write_text(json.dumps({"parent": sha(manifest), "rows": prepared}, indent=2) + "\n")


def checked_proposal(proposal, row):
    start, end = proposal["frames"]
    require(6 <= start < end <= 114 and end - start in (5, 6, 7, 8, 9)
            and proposal["pts"] == [expected_tick(i, row, "avc") for i in (start, end)],
            "The proposal boundaries do not match the native source")
    before, during, after = proposal["anchors"]
    require(before["frame_index"] == start - 1 and start <= during["frame_index"] < end and after["frame_index"] == end,
            "The evidence does not show before, during and after")
    for anchor in proposal["anchors"]:
        require(anchor["pts"] == expected_tick(anchor["frame_index"], row, "avc")
                and SHA.fullmatch(anchor["decoded_rgb_sha256"]),
                "The selected evidence has an invalid timestamp or pixel commitment")


def summarize(rows, prior):
    require(len(rows) == len(prior) == 64, "Incomplete source comparison")
    counts = Counter()
    groups = set()
    for row, previous in zip(rows, prior, strict=True):
        require(all(row[key] == previous[key] for key in (
            "group", "sha256", "condition", "truth", "expected_pane", "index", "time_base", "first_pts",
        )),
                "The measured source differs from the earlier pane study")
        first = previous["methods"]["split_guarded"]["proposal"]
        require(row["first"] == first, "The first range differs from the earlier readout")
        counts.update({"files": 1, "first_ranges": first is not None, "edited_files": bool(row["truth"]),
                       "ordinary_files": not row["truth"], "truth_intervals": len(row["truth"])})
        if first:
            checked_proposal(first, row)
            counts["first_exact"] += first["pane"] == row["expected_pane"] and first["frames"] in row["truth"]
        second = row["second"]
        if second is None:
            continue
        checked_proposal(second, row)
        require(first is not None and second["pane"] == first["pane"], "A second range changed the measured side")
        a, b = first["frames"]
        start, end = second["frames"]
        require(max(a - end, start - b) >= 3 and 5 <= end - start <= 9
                and len(second["pane_tiles"]) == 2 and all(type(i) is int and 0 <= i < 8 for i in second["pane_tiles"]),
                "The second range has incomplete separation or pane support")
        remaining = [target for target in row["truth"] if not overlap(first["frames"], target)]
        exact = second["frames"] in remaining and second["pane"] == row["expected_pane"]
        useful = second["pane"] == row["expected_pane"] and any(overlap(second["frames"], target) for target in remaining)
        counts.update({"second_ranges": 1, "exact_additions": exact, "useful_additions": useful,
                       "wrong_on_edited": bool(row["truth"]) and not useful, "ordinary_additions": not row["truth"]})
        if exact:
            groups.add(row["group"])
    counts["exact_source_groups"] = len(groups)
    return dict(counts) | {"advance": counts["exact_additions"] >= 4 and len(groups) >= 3
                           and counts["wrong_on_edited"] <= 1 and counts["ordinary_additions"] == 0}


def summarize_single(rows, parent):
    expected = [(g, f"single_{i}_{'left' if g % 2 else 'right'}") for g in range(1, 9) for i in (1, 2)]
    require([(r["group"], r["condition"]) for r in rows] == expected, "Missing single-edit cases")
    by_kind = {(r["group"], r["condition"]): r for r in parent["rows"]}
    counts = Counter(files=0, first_exact=0, first_ranges=0, extra_ranges=0)
    for row in rows:
        side = row["expected_pane"]
        original = by_kind[row["group"], f"swaps_{side}"]
        require(row["truth"] == [original["truth"][int(row["condition"].split("_")[1]) - 1]]
                and SHA.fullmatch(row["sha256"]), "Single-edit truth differs from the paired case")
        for record in (row["first"], row["second"]):
            if record:
                checked_proposal(record, row)
        first = row["first"]
        counts.update({"files": 1, "first_ranges": first is not None,
                       "first_exact": bool(first and first["pane"] == side and first["frames"] in row["truth"]),
                       "extra_ranges": row["second"] is not None})
    return dict(counts)


def run(bank, parent, *, single=False):
    import cv2

    from app.detectors.image.faces import load_yunet
    from scripts.video_face_split_track_transfer import combine

    manifest = bank / "media.json"
    prepared = json.loads(manifest.read_bytes())
    require((prepared["parent"] if single else sha(manifest)) == parent["media_sha256"],
            "The retained delivery roster differs")
    weights = ROOT / "models/weights"
    require(all(sha(weights / name) == expected for name, expected in parent["weights"].items()),
            "Pinned face assets differ")
    detector = load_yunet(weights / "face_detection_yunet_2023mar.onnx")
    recognizer = cv2.FaceRecognizerSF.create(str(weights / "face_recognition_sface_2021dec.onnx"), "")
    cv2.setNumThreads(1)
    rows = []
    for i, item in enumerate(prepared["rows"], 1):
        path = bank / "media" / item["file"]
        require(sha(path) == item["sha256"], "Delivered video differs")
        values, _ = measure(path, item, detector, recognizer, [], split_track=True, second_pane_range=True)
        first = combine(values, "_guarded")["proposal"]
        second = None
        if first and (extra := values[first["pane"] + "_guarded"].get("second")):
            second = {**extra, "pane": first["pane"]}
        rows.append({key: item[key] for key in ("sha256", "group", "condition", "truth", "expected_pane")}
                    | {"index": i, "first": first, "second": second, "time_base": item["clock"]["time_base"],
                       "first_pts": item["clock"]["pts"][0]})
        print(i, item["condition"], first and first["frames"], second and second["frames"], flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare-single", "run", "review", "pixels"))
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--bank", type=Path)
    parser.add_argument("--single", action="store_true", help="Use the single-edit control roster")
    args = parser.parse_args()
    parent = json.loads(PARENT.read_bytes())
    if args.action == "prepare-single":
        prepare_single(args.bank, args.receipt, parent)
        return
    headers = {"schema": SCHEMA, "protocol": SINGLE_PROTOCOL if args.single else PROTOCOL, "parent_sha256": sha(PARENT),
               "weights": parent["weights"]}
    summarize_rows = (lambda rows: summarize_single(rows, parent)) if args.single else (
        lambda rows: summarize(rows, parent["rows"]))
    if args.action == "run":
        rows = run(args.bank, parent, single=args.single)
        receipt = {**headers, "media_sha256": sha(args.bank / "media.json"), "rows": rows, "results": summarize_rows(rows)}
        with args.receipt.open("x") as handle:
            print(json.dumps(receipt, indent=2, sort_keys=True), file=handle)
    else:
        receipt = json.loads(args.receipt.read_bytes())
        require(all(receipt[key] == value for key, value in headers.items())
                and (SHA.fullmatch(receipt["media_sha256"]) if args.single else receipt["media_sha256"] == parent["media_sha256"])
                and receipt["results"] == summarize_rows(receipt["rows"]), "Decisions or freeze differ")
        if args.action == "pixels":
            manifest = args.bank / "media.json"
            require(sha(manifest) == receipt["media_sha256"], "Original delivery roster differs")
            items = json.loads(manifest.read_bytes())["rows"]
            paths = {item["sha256"]: args.bank / "media" / item["file"] for item in items}
            rows = [row | {"methods": {key: {"proposal": row[key]} for key in ("first", "second")}}
                    for row in receipt["rows"]]
            print(json.dumps(replay_pixels(rows, paths), sort_keys=True))
    print(json.dumps(receipt["results"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
