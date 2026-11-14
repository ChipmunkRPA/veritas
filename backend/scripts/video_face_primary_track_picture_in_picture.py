# SPDX-License-Identifier: Apache-2.0
"""Challenge the dominant-face search with a persistent unrelated picture inset."""

import argparse
import hashlib
import json
from pathlib import Path

from scripts.review_video_brief_face_120_carrier import require
from scripts.video_face_primary_track_transfer import measure, replay_pixels, review, sha

GROUPS = (1, 2, 3, 5, 6, 7, 8)
ROLES = ("original", "two_swaps", "tandem_two_swaps", "same_source_cuts", "cross_recording_cuts")
SCHEMA = "chipmunk.video-face-primary-track-picture-in-picture/1"


def manifest(bank):
    path = bank / "media.json"
    return {row["file"]: row for row in json.loads(path.read_text())["rows"]}


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
    weights = ROOT / "models/weights"
    require(
        all(sha(weights / name) == digest for name, digest in WEIGHTS.items()), "Pinned face assets differ"
    )
    detector = load_yunet(weights / "face_detection_yunet_2023mar.onnx")
    cv2.setNumThreads(1)
    source = manifest(bank)

    def pictures(group, role, indices=None):
        reference = source[f"g{group}-{role}.mp4"]
        path = bank / "media" / reference["file"]
        require(sha(path) == reference["sha256"], "Authored parent differs")
        table = FrameTable.from_json(reference["clock"])
        for frame in decode_frames(path, table, indices, timeout_s=35):
            require(
                frame.pts_verified
                and frame.pts == table.pts[frame.index]
                and hashlib.sha256(frame.rgb).hexdigest() == reference["rgb_sha256"][frame.index],
                "An original picture differs from its commitment",
            )
            yield frame.rgb

    rows, source_map = [], []
    for ordinal, group in enumerate(GROUPS):
        other = GROUPS[(ordinal + 3) % len(GROUPS)]
        [guest] = pictures(other, "original", [35])
        [example] = pictures(group, "original", [35])
        found = detect_faces(cv2.cvtColor(guest, cv2.COLOR_RGB2BGR), detector=detector, min_size=48)
        main = detect_faces(cv2.cvtColor(example, cv2.COLOR_RGB2BGR), detector=detector, min_size=48)
        require(found and main, "A qualified source picture lacks a face")
        x, y, w, h = found[0].box()
        patch = guest[
            max(0, y - h // 5) : min(720, y + h + h // 5), max(0, x - w // 4) : min(1280, x + w + w // 4)
        ]
        tile = cv2.resize(patch, (96, 144), interpolation=cv2.INTER_AREA)
        position = (14, 1170) if main[0].x + main[0].w / 2 < 640 else (14, 14)
        y, x = position

        def composed(pictures, inset, top, left):
            for picture in pictures:
                picture = np.copy(picture)
                picture[top : top + inset.shape[0], left : left + inset.shape[1]] = inset
                yield picture

        for role in ROLES:
            path = media / f"g{group}-{role}.mp4"
            truth = source[f"g{group}-{role}.mp4"]["truth"]
            authored(path, pictures=composed(pictures(group, role), tile, y, x))
            rows.append(describe(path, group, role, truth))
        source_map.append({"group": group, "donor": other, "native_picture": 35, "inset": [x, y, 96, 144]})
        print("prepared", group, flush=True)
    (output / "media.json").write_text(
        json.dumps(
            {
                "schema": SCHEMA,
                "parent_manifest": sha(bank / "media.json"),
                "rows": rows,
                "source_map": source_map,
            },
            indent=2,
        )
        + "\n"
    )


def run(output, public):
    import cv2

    from app.detectors.image.faces import load_yunet
    from scripts.video_brief_face_120_carrier import ROOT, WEIGHTS

    cv2.setNumThreads(1)
    weights = ROOT / "models/weights"
    require(
        all(sha(weights / name) == digest for name, digest in WEIGHTS.items()), "Pinned face assets differ"
    )
    detector = load_yunet(weights / "face_detection_yunet_2023mar.onnx")
    recognizer = cv2.FaceRecognizerSF.create(str(weights / "face_recognition_sface_2021dec.onnx"), "")
    manifest = json.loads((output / "media.json").read_text())
    require(
        manifest["schema"] == SCHEMA and len(manifest["rows"]) == len(GROUPS) * len(ROLES),
        "Unexpected roster",
    )
    rows = []
    for index, item in enumerate(manifest["rows"], 1):
        path = output / "media" / item["file"]
        require(sha(path) == item["sha256"], "Qualified file differs")
        alternatives, crowded = measure(path, item, detector, recognizer, [])
        row = {key: item[key] for key in ("sha256", "group", "condition", "truth")}
        row.update(
            index=index,
            panel="picture_in_picture",
            bank="held_visual_layout",
            prior=[],
            first=None,
            multiple_faces=crowded,
            methods=alternatives,
            first_pts=item["clock"]["pts"][0],
            time_base=item["clock"]["time_base"],
        )
        rows.append(row)
        print(
            index,
            item["condition"],
            crowded,
            [k for k, v in alternatives.items() if v["proposal"]],
            flush=True,
        )
    result = review(rows)
    receipt = {
        "schema": SCHEMA,
        "parent_manifest": manifest["parent_manifest"],
        "models": WEIGHTS,
        "runner_sha256": sha(Path(__file__)),
        "native_manifest_sha256": sha(output / "media.json"),
        "rows": rows,
        "result": result,
    }
    with public.open("x") as result_file:
        json.dump(receipt, result_file, indent=2, allow_nan=False)
        result_file.write("\n")
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
        entry = json.loads(args.input.read_text())
        result = review(entry["rows"])
        require(entry["schema"] == SCHEMA and entry["result"] == result, "Reported decisions differ")
        if args.phase == "pixels":
            require(
                args.output and sha(args.output / "media.json") == entry["native_manifest_sha256"],
                "The original media directory is required",
            )
            paths = {r["sha256"]: args.output / "media" / r["file"] for r in manifest(args.output).values()}
            result = replay_pixels(entry["rows"], paths)
        print(json.dumps(result, indent=2, sort_keys=True))
