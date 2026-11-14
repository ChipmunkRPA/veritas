# SPDX-License-Identifier: Apache-2.0
"""Compare face tracking with a moving guest and a brief change of speaker layout."""

import argparse
import hashlib
import json
from pathlib import Path

from scripts.review_video_brief_face_120_carrier import require
from scripts.video_face_primary_track_transfer import METHODS, measure, replay_pixels, review, sha

SCHEMA = "chipmunk.video-face-position-track/1"
ARMS = (*METHODS, "position_track")
ROLES = ("original", "two_swaps", "tandem_two_swaps", "same_source_cuts", "cross_recording_cuts", "guest_cut")
GROUPS = tuple(range(1, 9))
MIN_EXACT_GAIN = 3
MIN_GROUP_GAIN = 2


def load(path):
    return json.loads(path.read_text())


def write(path, result):
    with path.open("x") as out:
        out.write(json.dumps(result, indent=2, allow_nan=False) + "\n")


def composite(pictures, tiles, *, left, enlarged):
    start, end = enlarged
    for i, (original, (small, large)) in enumerate(zip(pictures, tiles, strict=True)):
        out = original.copy()
        tile = large if start <= i < end else small
        h, w = tile.shape[:2]
        x = 14 if left else 1280 - w - 14
        out[14:14+h, x:x+w] = tile
        yield out


def prepare(bank, output):
    import cv2

    from app.detectors.image.faces import detect_faces, load_yunet
    from app.detectors.video.decode import FrameTable, decode_frames
    from scripts.video_brief_face_120_carrier import ROOT, WEIGHTS
    from scripts.video_face_720p30_transfer import authored, describe

    output.mkdir(mode=0o700)
    deliveries = output / "media"
    deliveries.mkdir(mode=0o700)
    manifest = bank / "media.json"
    parents = {(r["group"], r["condition"]): r for r in load(manifest)["rows"]}
    weights = ROOT / "models/weights"
    require(all(sha(weights / name) == digest for name, digest in WEIGHTS.items()), "Face assets differ")
    detector = load_yunet(weights / "face_detection_yunet_2023mar.onnx")
    cv2.setNumThreads(1)

    def pictures(group, role):
        row = parents[group, role]
        path = bank / "media" / row["file"]
        require(sha(path) == row["sha256"], "Parent delivery differs")
        table = FrameTable.from_json(row["clock"])
        for frame in decode_frames(path, table, None, timeout_s=35):
            i = frame.index
            require(frame.pts_verified and frame.pts == table.pts[i]
                    and hashlib.sha256(frame.rgb).hexdigest() == row["rgb_sha256"][i], "Native picture differs")
            yield frame.rgb

    def face_size(image):
        faces = detect_faces(cv2.cvtColor(image, cv2.COLOR_RGB2BGR), detector=detector, min_size=48)
        require(faces, "No face in a committed parent frame")
        return faces[0]

    def guest_tiles(group, width, height):
        for image in pictures(group, "original"):
            guest = face_size(image)
            y = max(0, guest.y - guest.h // 5)
            x = max(0, guest.x - guest.w // 4)
            crop = image[y:guest.y + guest.h * 6 // 5, x:guest.x + guest.w * 5 // 4]
            yield (cv2.resize(crop, (96, 144), interpolation=cv2.INTER_AREA),
                   cv2.resize(crop, (width, height), interpolation=cv2.INTER_AREA))

    rows, layouts = [], []
    for group in GROUPS:
        guest = GROUPS[(group + 2) % len(GROUPS)]
        substitute = GROUPS[(group + 4) % len(GROUPS)]
        parent = pictures(group, "original")
        try:
            main = face_size(next(parent))
        finally:
            parent.close()
        width = min(320, max(188, int(1.24 * main.w)))
        height = min(410, max(204, int(1.38 * main.h)))
        left = main.x + main.w / 2 > 640
        # Use the same picture-in-picture timing for all six deliveries of a recording.
        start, end = 61 + 2 * (group % 3), 108 + group % 3
        tiles = list(guest_tiles(guest, width, height))
        alternate = list(guest_tiles(substitute, width, height))
        require(len(tiles) == len(alternate) == 120, "Incomplete guest source")

        for role in ROLES:
            guest_only = role == "guest_cut"
            original_role = "original" if guest_only else role
            path = deliveries / f"g{group}-{role}.mp4"
            guest_pictures = [*tiles[:89], *alternate[89:96], *tiles[96:]] if guest_only else tiles
            authored(path, pictures=composite(pictures(group, original_role), guest_pictures,
                                              left=left, enlarged=(start, end)))
            rows.append(describe(path, group, role, parents[group, original_role]["truth"]))
        layouts.append({"group": group, "guest": guest, "second_guest": substitute,
                        "left": left, "enlarged": [start, end], "size": [width, height]})
        print("prepared", group, flush=True)
    write(output / "media.json", {"schema": SCHEMA, "parent_manifest_sha256": sha(manifest),
                                  "source_indices": [row["index"] for row in load(bank / "freeze.json")["sources"]],
                                  "layouts": layouts, "rows": rows})


def summarize(rows):
    require([(r["group"], r["condition"]) for r in rows] == [(i, role) for i in GROUPS for role in ROLES],
            "The study roster differs")
    counts = review(rows, ARMS)["moving_guest"]
    for arm in ARMS:
        counts[arm]["exact_clips"] = sum(
            row["methods"][arm]["proposal"] is not None
            and row["methods"][arm]["proposal"]["frames"] in row["truth"] for row in rows
        )
        counts[arm]["measurable_pairs"] = sum(row["methods"][arm]["complete_pair"] for row in rows)
        counts[arm]["ordinary_clips_flagged"] = sum(
            not row["truth"] and row["methods"][arm]["proposal"] is not None for row in rows
        )
    new_exact, lost_exact, suppressed, added_ordinary, added_cuts = [], [], [], [], []
    for row in rows:
        baseline = row["methods"]["single_face"]["proposal"] or row["methods"]["confirmed_track"]["proposal"]
        tracked = row["methods"]["position_track"]["proposal"]
        if baseline is not None:
            if baseline["frames"] in row["truth"]:
                if tracked is None or tracked["frames"] not in row["truth"]:
                    lost_exact.append(row["index"])
            elif not row["truth"] and tracked is None:
                suppressed.append(row["index"])
            continue
        if tracked is None:
            continue
        if tracked["frames"] in row["truth"]:
            new_exact.append(row["index"])
        elif not row["truth"]:
            added_ordinary.append(row["index"])
            if "cut" in row["condition"]:
                added_cuts.append(row["index"])
    gained_groups = sorted({rows[i - 1]["group"] for i in new_exact})
    counts["paired"] = {"new_exact": new_exact, "incumbent_exact_missing_from_candidate": lost_exact,
                         "new_exact_groups": gained_groups, "candidate_did_not_repeat_ordinary": suppressed,
                         "new_ordinary_flags": added_ordinary, "new_cut_flags": added_cuts}
    baseline, candidate = counts["confirmed_track"], counts["position_track"]
    counts["eligible_for_local_trial"] = (
        len(new_exact) >= MIN_EXACT_GAIN and len(gained_groups) >= MIN_GROUP_GAIN
        and len(added_ordinary) <= 1 and not added_cuts and candidate["misplaced"] <= baseline["misplaced"]
    )
    return counts


def run(directory, output):
    import cv2

    from app.detectors.image.faces import load_yunet
    from app.evidence import face_primary_track_binding
    from scripts.video_brief_face_120_carrier import ROOT, WEIGHTS

    manifest = directory / "media.json"
    saved = load(manifest)
    require(saved["schema"] == SCHEMA, "Unexpected fixture schema")
    weights = ROOT / "models/weights"
    require(all(sha(weights / name) == value for name, value in WEIGHTS.items()), "Face asset bytes differ")
    detector = load_yunet(weights / "face_detection_yunet_2023mar.onnx")
    recognizer = cv2.FaceRecognizerSF.create(str(weights / "face_recognition_sface_2021dec.onnx"), "")
    cv2.setNumThreads(1)
    rows = []
    for index, item in enumerate(saved["rows"], 1):
        path = directory / "media" / item["file"]
        require(sha(path) == item["sha256"], "Native delivery differs")
        methods, crowded = measure(path, item, detector, recognizer, [], alternate_track=True)
        row = {k: item[k] for k in ("group", "condition", "truth", "sha256")}
        row.update(index=index, panel="moving_guest", bank="held_layout", first=None, prior=[], multiple_faces=crowded,
                   time_base=item["clock"]["time_base"], first_pts=item["clock"]["pts"][0], methods=methods)
        rows.append(row)
        print(index, row["group"], row["condition"], [key for key, value in methods.items() if value["proposal"]],
              flush=True)
    result = summarize(rows)
    write(output, {"schema": SCHEMA, "media_manifest_sha256": sha(manifest),
                   "parent_manifest_sha256": saved["parent_manifest_sha256"], "layouts": saved["layouts"],
                   "source_indices": saved["source_indices"],
                   "runner_sha256": sha(Path(__file__)), "measurement_sha256": sha(Path(measure.__code__.co_filename)),
                   "method_sha256": sha(Path(face_primary_track_binding.__file__)),
                   "models": WEIGHTS, "rows": rows, "results": result})
    print(json.dumps(result, indent=2, sort_keys=True))


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
        data = load(args.input)
        result = summarize(data["rows"])
        require(data["schema"] == SCHEMA and result == data["results"], "Reported results differ")
        if args.phase == "pixels":
            manifest = args.output / "media.json"
            require(sha(manifest) == data["media_manifest_sha256"], "Provide the original fixture bank")
            result = replay_pixels(data["rows"], {row["sha256"]: args.output / "media" / row["file"]
                                                  for row in load(manifest)["rows"]})
        print(json.dumps(result, indent=2, sort_keys=True))
