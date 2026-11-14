# SPDX-License-Identifier: Apache-2.0
"""Compare short face-change localization on opposite sides of a split-screen clip."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from scripts.review_video_brief_face_120_carrier import SHA, require
from scripts.video_face_primary_track_transfer import measure, replay_pixels, sha
from scripts.video_face_second_pane_transfer import PARENT, checked_proposal
from scripts.video_face_split_track_transfer import GROUPS, combine

SCHEMA = "chipmunk.video-face-opposite-pane/1"
SIDES = ("left", "right")
CASES = {
    "left_then_right": (0, 1), "right_then_left": (1, 0),
    "left_only": (0, None), "right_only": (None, 0),
    "ordinary": (None, None), "ordinary_cuts": (None, None),
}
PROTOCOL = (
    "Preserve both same-side selections. If there is no same-side follow-up, take the strongest "
    "guarded result already measured on the other side when at least three native frames separate "
    "the two ranges. Compare exact side and both boundaries; count wrong additional ranges on edits, "
    "unmodified pictures and single-swap controls separately."
)


def composition(group, case):
    index = GROUPS.index(group)
    peer = GROUPS[(index + 3) % len(GROUPS)]
    positions = CASES[case]
    if case in {"left_only", "right_only"}:
        positions = tuple(index % 2 if pos is not None else None for pos in positions)
    return zip(SIDES, (group, peer), positions, strict=True)


def prepare(bank, output, parent):
    import numpy as np

    from app.detectors.video.decode import FrameTable, decode_frames
    from scripts.video_face_720p30_transfer import authored, describe

    manifest = bank / "media.json"
    require(sha(manifest) == parent["media_sha256"], "Parent deliveries differ")
    inputs = {(r["group"], r["condition"]): r for r in json.loads(manifest.read_bytes())["rows"]}
    output.mkdir(mode=0o700)
    media = output / "media"
    media.mkdir(mode=0o700)

    def pictures(row):
        path = bank / "media" / row["file"]
        table = FrameTable.from_json(row["clock"])
        require(sha(path) == row["sha256"], "Parent clip differs")
        for frame in decode_frames(path, table, range(120), timeout_s=40):
            require(frame.pts_verified and frame.pts == table.pts[frame.index]
                    and hashlib.sha256(frame.rgb).hexdigest() == row["rgb_sha256"][frame.index],
                    "Parent picture differs")
            yield frame.rgb[:, :640]

    rows = []
    for group in GROUPS:
        for case in CASES:
            truth = []
            sources = []
            participants = []
            for side, actor, position in composition(group, case):
                participants.append(actor)
                original = inputs[actor, "original"]
                if position is None:
                    ordinary = inputs[actor, "ordinary_speaker_cut"] if case == "ordinary_cuts" else original
                    sources.append(pictures(ordinary))
                    continue
                edited = inputs[actor, "swaps_left"]
                start, end = edited["truth"][position]
                truth.append({"pane": side, "frames": [start, end]})
                sources.append(splice(pictures(original), pictures(edited), start, end))
            path = media / f"g{group}-{case}.mp4"
            authored(path, pictures=(np.concatenate(pair, axis=1) for pair in zip(*sources, strict=True)))
            rows.append(describe(path, group, case, truth) | {"participants": participants})
            print("Prepared", path.name, flush=True)
    (output / "media.json").write_text(json.dumps({"parent": sha(manifest), "rows": rows}, indent=2) + "\n")


def splice(original, edit, start, end):
    for i, (clean, changed) in enumerate(zip(original, edit, strict=True)):
        yield changed if start <= i < end else clean


def selections(methods):
    primary = combine(methods, "_guarded")["proposal"]
    if primary is None:
        return None, None, None
    side = primary["pane"]
    if same := methods[f"{side}_guarded"].get("second"):
        return primary, {**same, "pane": side}, None
    other = "right" if side == "left" else "left"
    opposite = methods[f"{other}_guarded"]["proposal"]
    if opposite:
        a, b = primary["frames"]
        start, end = opposite["frames"]
        if max(a - end, start - b) >= 3:
            return primary, None, {**opposite, "pane": other}
    return primary, None, None


def summarize(rows, parent):
    require([(row["group"], row["condition"]) for row in rows] == [(g, case) for g in GROUPS for case in CASES],
            "The comparison roster differs")
    originals = {(row["group"], row["condition"]): row for row in parent["rows"]}
    counts = {case: Counter(files=0, truth=0, baseline_exact=0, candidate_exact=0, baseline_extras=0,
                            candidate_extras=0, baseline_flagged_files=0, candidate_flagged_files=0)
              for case in CASES}
    gains = set()
    for row in rows:
        case, group = row["condition"], row["group"]
        expected = list(composition(group, case))
        require(row["participants"] == [actor for _, actor, _ in expected] and SHA.fullmatch(row["sha256"]),
                "Composition differs")
        truth = [{"pane": side, "frames": originals[actor, "swaps_left"]["truth"][pos]}
                 for side, actor, pos in expected if pos is not None]
        require(truth == row["truth"], "Known edit boundaries differ")
        methods = row["methods"]
        for side in SIDES:
            method = methods[f"{side}_guarded"]
            for key in ("proposal", "second"):
                if record := method.get(key):
                    checked_proposal(record, row)
                    require(len(record["pane_tiles"]) == 2
                            and all(type(n) is int and 0 <= n < 8 for n in record["pane_tiles"]),
                            "The measured side has incomplete off-face support")
        base, same, extra = selections(methods)
        require([base, same, extra] == [row["first"], row["same"], row["opposite"]], "Selection differs")
        counts[case].update(files=1, truth=len(truth))
        for arm, proposals in (("baseline", (base, same)), ("candidate", (base, same, extra))):
            found = [record for record in proposals if record is not None]
            n = sum({"pane": record["pane"], "frames": record["frames"]} in truth for record in found)
            counts[case].update({f"{arm}_exact": n, f"{arm}_extras": len(found) - n,
                                 f"{arm}_flagged_files": bool(found)})
        if extra and {"pane": extra["pane"], "frames": extra["frames"]} in truth:
            gains.add(row["participants"][SIDES.index(extra["pane"])])
    return {"by_case": counts, "additional_correct_people": sorted(gains)}


def run(bank, parent):
    import cv2

    from app.detectors.image.faces import load_yunet
    from scripts.video_face_primary_track_transfer import ROOT

    prepared = json.loads((bank / "media.json").read_bytes())
    weights = ROOT / "models/weights"
    require(prepared["parent"] == parent["media_sha256"]
            and all(sha(weights / name) == pin for name, pin in parent["weights"].items()),
            "Parent roster or face assets differ")
    cv2.setNumThreads(1)
    detector = load_yunet(weights / "face_detection_yunet_2023mar.onnx")
    recognizer = cv2.FaceRecognizerSF.create(str(weights / "face_recognition_sface_2021dec.onnx"), "")
    rows = []
    for item in prepared["rows"]:
        path = bank / "media" / item["file"]
        require(sha(path) == item["sha256"], "Composed delivery differs")
        values, _ = measure(path, item, detector, recognizer, [], split_track=True, second_pane_range=True)
        methods = {f"{side}_guarded": values[f"{side}_guarded"] for side in SIDES}
        first, same, opposite = selections(methods)
        rows.append({key: item[key] for key in ("sha256", "group", "condition", "truth", "participants")}
                    | {"methods": methods, "first": first, "same": same, "opposite": opposite,
                       "time_base": item["clock"]["time_base"], "first_pts": item["clock"]["pts"][0]})
        print(item["file"], *(record and (record["pane"], record["frames"])
                             for record in (first, same, opposite)), flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run", "review", "pixels"))
    parser.add_argument("path", type=Path)
    parser.add_argument("--bank", type=Path)
    args = parser.parse_args()
    parent = json.loads(PARENT.read_bytes())
    if args.action == "prepare":
        prepare(args.bank, args.path, parent)
        return
    header = {"schema": SCHEMA, "protocol": PROTOCOL, "parent_sha256": sha(PARENT), "weights": parent["weights"]}
    if args.action == "run":
        rows = run(args.bank, parent)
        receipt = {**header, "media_sha256": sha(args.bank / "media.json"), "rows": rows,
                   "results": summarize(rows, parent)}
        with args.path.open("x") as handle:
            print(json.dumps(receipt, indent=2, sort_keys=True), file=handle)
    else:
        receipt = json.loads(args.path.read_bytes())
        require(all(receipt[key] == value for key, value in header.items())
                and SHA.fullmatch(receipt["media_sha256"])
                and receipt["results"] == summarize(receipt["rows"], parent), "Receipt decisions differ")
        if args.action == "pixels":
            manifest = args.bank / "media.json"
            require(sha(manifest) == receipt["media_sha256"], "Composed deliveries differ")
            paths = {row["sha256"]: args.bank / "media" / row["file"] for row in json.loads(manifest.read_bytes())["rows"]}
            rows = [row | {"methods": {key: {"proposal": row[key]} for key in ("first", "same", "opposite")}}
                    for row in receipt["rows"]]
            print(json.dumps(replay_pixels(rows, paths), sort_keys=True))
    print(json.dumps(receipt["results"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
