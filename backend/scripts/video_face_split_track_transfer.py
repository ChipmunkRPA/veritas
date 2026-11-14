# SPDX-License-Identifier: Apache-2.0
"""Compare independent face panes on side-by-side camera layouts."""

import argparse
import hashlib
import json
from pathlib import Path

from scripts.review_video_brief_face_120_carrier import require
from scripts.video_face_position_track_transfer import GROUPS, load, write
from scripts.video_face_primary_track_transfer import measure, replay_pixels, review, sha

SCHEMA = "chipmunk.video-face-split-track/1"
ROLES = ("original", "swaps_left", "swaps_right", "swaps_resize", "speaker_resize",
         "ordinary_speaker_cut", "guest_return", "pane_exchange")
SWAPS = frozenset(ROLES[1:4])
ARMS = ("largest_face", "position_track", "split_track", "split_guarded")
GUARD = 5


def prepare(bank, output):
    import cv2
    import numpy as np

    from app.detectors.image.faces import detect_faces, load_yunet
    from app.detectors.video.decode import FrameTable, decode_frames
    from scripts.video_brief_face_120_carrier import ROOT, WEIGHTS
    from scripts.video_face_720p30_transfer import authored, describe

    output.mkdir(mode=0o700)
    media = output / "media"
    media.mkdir(mode=0o700)
    manifest = bank / "media.json"
    parents = {(r["group"], r["condition"]): r for r in load(manifest)["rows"]}
    weights = ROOT / "models/weights"
    require(all(sha(weights / name) == digest for name, digest in WEIGHTS.items()), "Face assets differ")
    detector = load_yunet(weights / "face_detection_yunet_2023mar.onnx")
    cv2.setNumThreads(1)

    def pictures(group, condition):
        row = parents[group, condition]
        path = bank / "media" / row["file"]
        require(sha(path) == row["sha256"], "Parent delivery differs")
        table = FrameTable.from_json(row["clock"])
        for frame in decode_frames(path, table, None, timeout_s=35):
            require(frame.pts_verified and frame.pts == table.pts[frame.index]
                    and hashlib.sha256(frame.rgb).hexdigest() == row["rgb_sha256"][frame.index],
                    "Native picture differs")
            yield frame.rgb

    def view(group, condition):
        # Crop around the ordinary full-frame face once. Apply the same crop to
        # edited and ordinary frames so the composition cannot reveal the label.
        original = pictures(group, "original")
        detections = []
        try:
            for i, frame in enumerate(original):
                if i in (4, 14, 24, 34, 44):
                    found = detect_faces(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), detector=detector, min_size=48)
                    if found:
                        detections.append(found[0].box())
                if i >= 44:
                    break
        finally:
            original.close()
        require(detections, f"Original group {group} has no usable face")
        x, y, w, h = map(int, np.median(detections, axis=0))
        side = min(650, max(350, int(max(w, h)*2.4)))
        x = max(0, min(1280-side, x+(w-side)//2))
        y = max(0, min(720-side, y+(h-side)//2))
        for frame in pictures(group, condition):
            crop = frame[y:y+side, x:x+side]
            yield cv2.resize(crop, (520, 520), interpolation=cv2.INTER_AREA)

    def compose(hosts, guests, *, role, guest_span, first_swap, place):
        for i, (host, guest, alternate) in enumerate(zip(hosts, *guests, strict=True)):
            active = first_swap[0]-GUARD <= i < first_swap[1]+GUARD
            scoped = guest_span[0] <= i < guest_span[1]
            if role == "guest_return" and scoped:
                guest = alternate
            if role == "pane_exchange" and scoped:
                host, guest = guest, host
            if role == "swaps_right":
                host, guest = guest, host
            out = np.full((720, 1280, 3), 23, np.uint8)
            for j, tile in enumerate((host, guest)):
                size = (440 if active == (j == place) else 520) if role in {"swaps_resize", "speaker_resize"} else 520
                shown = tile if size == 520 else cv2.resize(tile, (size, size), interpolation=cv2.INTER_AREA)
                y, x = (720-size)//2, (640-size)//2 + 640*j
                out[y:y+size, x:x+size] = shown
            yield out

    rows = []
    for group in GROUPS:
        guest = GROUPS[(group+2) % len(GROUPS)]
        substitute = GROUPS[(group+4) % len(GROUPS)]
        guests = tuple(list(view(g, "original")) for g in (guest, substitute))
        truth = parents[group, "two_swaps"]["truth"]
        for role in ROLES:
            parent = "two_swaps" if role in SWAPS else (
                "cross_recording_cuts" if role == "ordinary_speaker_cut" else "original")
            path = media / f"g{group}-{role}.mp4"
            authored(path, pictures=compose(view(group, parent), guests, role=role, guest_span=truth[1],
                                             first_swap=truth[0], place=group % 2))
            row = describe(path, group, role, truth if role in SWAPS else [])
            row["expected_pane"] = ("right" if role == "swaps_right" else "left") if role in SWAPS else None
            rows.append(row)
        print("prepared", group, flush=True)
    write(output / "media.json", {"schema": SCHEMA, "parent_manifest_sha256": sha(manifest),
                                   "source_indices": [row["index"] for row in load(bank / "freeze.json")["sources"]],
                                   "rows": rows})


def summarize(rows):
    require([(r["group"], r["condition"]) for r in rows] == [(i, role) for i in GROUPS for role in ROLES],
            "Unexpected layout roster")
    review(rows, ("single_face", "confirmed_track", "left", "right", "left_guarded", "right_guarded"))
    for row in rows:
        methods = row["methods"]
        require(methods["split_track"] == combine(methods, "")
                and methods["split_guarded"] == combine(methods, "_guarded"), "Selected face pane differs")
        for pane in ("left_guarded", "right_guarded"):
            lane = methods[pane]
            readings = lane["pane_changes"]
            require(readings == sorted(readings)
                    and len({r[0] for r in readings}) == len(readings)
                    and all(6 <= edge < 115 and type(tiles) is int and 0 <= tiles <= 36 for edge, tiles in readings),
                    "Invalid off-face tile counts")
            if proposal := lane["proposal"]:
                require(proposal["pane_tiles"] == [dict(readings).get(i) for i in proposal["frames"]]
                        and all(type(i) is int and i < 8 for i in proposal["pane_tiles"]),
                        "The selected interval does not pass the pane motion guard")
    counts = review(rows, ARMS)["split_screen"]
    for arm in ARMS:
        counts[arm]["with_measurable_pair"] = sum(row["methods"][arm]["complete_pair"] for row in rows)
        counts[arm]["flagged_controls"] = [
            row["index"] for row in rows if not row["truth"] and row["methods"][arm]["proposal"] is not None
        ]
    for arm in ("split_track", "split_guarded"):
        wrong_pane = [r["index"] for r in rows if r["truth"] and (proposal := r["methods"][arm]["proposal"])
                      and proposal["pane"] != r["expected_pane"]]
        exact_gain = [r["index"] for r in rows if r["truth"] and (proposal := r["methods"][arm]["proposal"])
                      and proposal["pane"] == r["expected_pane"] and proposal["frames"] in r["truth"]
                      and not ((current := r["methods"]["position_track"]["proposal"]) and current["frames"] in r["truth"])]
        result = counts[arm]
        result["wrong_pane"] = wrong_pane
        result["additional_exact_on_expected_pane"] = exact_gain
        result["additional_source_groups"] = sorted({rows[i-1]["group"] for i in exact_gain})
        result["new_controls_flagged"] = sorted(set(result["flagged_controls"])
                                               - set(counts["position_track"]["flagged_controls"]))
        result["eligible_for_local_trial"] = (len(exact_gain) >= 4 and len(result["additional_source_groups"]) >= 3
                                              and len(result["new_controls_flagged"]) <= 1 and not wrong_pane)
    return counts


def combine(values, suffix):
    records = [values[pane + suffix] for pane in ("left", "right")]
    panes = [(p, row["proposal"]) for p, row in zip(("left", "right"), records, strict=True) if row["proposal"]]
    primary = None
    if panes:
        pane, proposal = min(panes, key=lambda pair: (-pair[1]["response"], pair[1]["frames"], pair[0]))
        primary = {**proposal, "pane": pane}
    return {"measured": max(row["measured"] for row in records),
            "scheduled": max(row["scheduled"] for row in records),
            "complete_pair": any(row["complete_pair"] for row in records), "proposal": primary}


def run(bank, output):
    import cv2

    from app.detectors.image.faces import load_yunet
    from app.detectors.video import face_short_proposal
    from app.evidence import face_primary_track_binding
    from scripts.video_brief_face_120_carrier import ROOT, WEIGHTS

    manifest = bank / "media.json"
    prepared = load(manifest)
    require(prepared["schema"] == SCHEMA, "Input manifest differs")
    code = {"runner_sha256": Path(__file__), "measure_sha256": Path(measure.__code__.co_filename),
            "association_sha256": Path(face_primary_track_binding.__file__),
            "proposal_sha256": Path(face_short_proposal.__file__)}
    commitments = {name: sha(path) for name, path in code.items()}
    weights = ROOT / "models/weights"
    require(all(sha(weights / name) == digest for name, digest in WEIGHTS.items()), "Pinned face assets differ")
    detector = load_yunet(weights / "face_detection_yunet_2023mar.onnx")
    recognizer = cv2.FaceRecognizerSF.create(str(weights / "face_recognition_sface_2021dec.onnx"), "")
    cv2.setNumThreads(1)
    rows = []
    for index, item in enumerate(prepared["rows"], 1):
        path = bank / "media" / item["file"]
        require(sha(path) == item["sha256"], "Delivered bytes differ")
        values, crowded = measure(path, item, detector, recognizer, [], alternate_track=True, split_track=True)
        values["split_track"] = combine(values, "")
        values["split_guarded"] = combine(values, "_guarded")
        row = {k: item[k] for k in ("sha256", "group", "condition", "truth", "expected_pane")}
        row.update(index=index, first=None, prior=[], panel="split_screen", bank="source_separated",
                   multiple_faces=crowded, first_pts=item["clock"]["pts"][0], time_base=item["clock"]["time_base"],
                   methods=values)
        rows.append(row)
        print(index, row["group"], row["condition"], [k for k, v in values.items() if v["proposal"]], flush=True)
    result = summarize(rows)
    require(commitments == {name: sha(path) for name, path in code.items()}, "Scoring code changed during the run")
    write(output, {"schema": SCHEMA, "media_sha256": sha(manifest), "parent_sha256": prepared["parent_manifest_sha256"],
                   "source_indices": prepared["source_indices"], "weights": WEIGHTS, **commitments,
                   "rows": rows, "results": result})
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "run", "review", "pixels"))
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path, nargs="?")
    args = parser.parse_args()
    if args.phase == "prepare":
        prepare(args.input, args.output)
    elif args.phase == "run":
        run(args.input, args.output)
    else:
        receipt = load(args.input)
        result = summarize(receipt["rows"])
        require(receipt["schema"] == SCHEMA and result == receipt["results"], "Published outcomes differ")
        if args.phase == "pixels":
            manifest = args.output / "media.json"
            require(sha(manifest) == receipt["media_sha256"], "Original delivery manifest differs")
            paths = {r["sha256"]: args.output / "media" / r["file"] for r in load(manifest)["rows"]}
            result = replay_pixels(receipt["rows"], paths)
        print(json.dumps(result, indent=2, sort_keys=True))
