# SPDX-License-Identifier: Apache-2.0
"""Independently check the six-picture study counts and selected native crops."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from fractions import Fraction
from pathlib import Path


def read(path):
    return json.loads(path.read_bytes())


def digest(blob):
    return hashlib.sha256(blob).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def matches(a, b):
    common = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    return 3 * common >= a[1] - a[0] + b[1] - b[0]


def selection(candidates, point):
    result = []
    require(
        candidates == sorted(candidates, key=lambda c: (-c["response"], *c["frames"])),
        "Proposal rank changed",
    )
    for item in candidates:
        start, stop = item["frames"]
        anchors = item["anchors"]
        require(
            6 <= start < stop <= 114
            and stop - start in (5, 6, 7)
            and len(anchors) == 3
            and anchors[0]["frame_index"] == start - 1
            and anchors[2]["frame_index"] == stop
            and start <= anchors[1]["frame_index"] < stop
            and len(item["edges"]) == 2,
            "Native interval or roles differ",
        )
        for edge in item["edges"]:
            center = float.fromhex(edge["central_hex"])
            sectors = [float.fromhex(value) for value in edge["sectors_hex"]]
            require(
                3 <= center <= 255
                and 2 <= len(sectors) <= 4
                and sum(x >= max(6, center / 2) for x in sectors) < 2,
                "Near-head guard changed",
            )
        if (
            len(result) < 2
            and item["response"] > point
            and all(not matches(item["frames"], old["frames"]) for old in result)
        ):
            result.append(item)
    return result


def recount(rows, incumbent, arm, point):
    groups, totals = set(), Counter()
    picks = []
    for row in rows:
        values = selection(row["candidates"][arm], point)
        spans = [r["frames"] for r in values]
        target = [truth for truth in row["truth"] if truth[1] - truth[0] == 6]
        baseline = incumbent[row["sha256"]]
        new = [
            truth
            for truth in target
            if any(matches(truth, span) for span in spans)
            and not any(matches(truth, span) for span in baseline)
        ]
        if new:
            groups.add(row["group"])
        ordinary = not row["truth"]
        totals.update(
            files=1,
            target=len(target),
            added=len(new),
            exact=sum(span in spans for span in new),
            ordinary_files=ordinary,
            ordinary_flags=bool(ordinary and spans),
            cut_flags=bool(
                row["bank"] == "speaker_cuts"
                and row["condition"] in {"person_switch", "tandem_person_switch"}
                and spans
            ),
            wrong_edit_ranges=sum(not any(matches(span, truth) for truth in row["truth"]) for span in spans)
            if not ordinary
            else 0,
        )
        picks.extend((row, item) for item in values)
    return {**totals, "gained_groups": len(groups)}, picks


def audit(payload):
    require(
        payload["freeze"]["schema"] == "chipmunk.video-face-unrouted-short-transfer/1",
        "Unknown study version",
    )
    fixed = payload["point"]["points"]
    dev, held = payload["development"], payload["held"]
    require(
        len(dev) == len({r["sha256"] for r in dev}) == 61
        and len(held) == len({r["sha256"] for r in held}) == 56
        and not ({r["sha256"] for r in dev} & {r["sha256"] for r in held}),
        "Held source roster differs",
    )
    results = {"development": {}, "held": {}}
    picks = []
    for arm in ("dense", "shortlist"):
        point = max(
            [0.10, *(item["response"] for row in dev if not row["truth"] for item in row["candidates"][arm])]
        )
        require(point == fixed[arm], "Frozen development point differs")
        for bank, rows in (("development", dev), ("held", held)):
            counts, chosen = recount(rows, payload["baseline"][bank], arm, point)
            results[bank][arm] = counts
            if bank == "held" and arm == "shortlist":
                picks = chosen
    small, dense = results["held"]["shortlist"], results["held"]["dense"]
    gate = {
        "added": small["added"] >= 4,
        "groups": small["gained_groups"] >= 3,
        "exact": small["exact"] >= 2,
        "relative_to_dense": 4 * small["added"] >= 3 * dense["added"],
        "ordinary": small["ordinary_flags"] <= 1,
        "speaker_cuts": small["cut_flags"] == 0,
        "wrong_ranges": small["wrong_edit_ranges"] <= 1,
    }
    summary = {
        **results,
        "held_incumbent_complete": True,
        "gate": gate,
        "passed": all(gate.values()),
        "point": payload["point"],
    }
    require(summary == payload["result"], "Held result differs from saved counts")
    return summary, picks


def native(payload, picks, folder):
    import cv2

    from app.detectors.image.faces import detect_faces, load_yunet
    from app.detectors.video.decode import decode_frames, probe_frames
    from app.detectors.video.face_appearance import face_row
    from app.detectors.video.face_paired_edge import motion, reduced_rgb
    from app.evidence.face_paired_edge_binding import motion_inputs

    manifest = folder / "media.json"
    require(digest(manifest.read_bytes()) == payload["held_media_sha256"], "Native media manifest differs")
    rows = {row["sha256"]: row for row in read(manifest)["rows"]}
    weights = Path(__file__).resolve().parents[2] / "models/weights"
    require(
        {key: digest((weights / key).read_bytes()) for key in payload["freeze"]["weights"]}
        == payload["freeze"]["weights"],
        "Face crop assets changed",
    )
    cv2.setNumThreads(1)
    detector = load_yunet(weights / "face_detection_yunet_2023mar.onnx")
    aligner = cv2.FaceRecognizerSF.create(str(weights / "face_recognition_sface_2021dec.onnx"), "")
    checked = Counter()
    for row, selected in picks:
        commitment = rows[row["sha256"]]
        path = folder / commitment["file"]
        require(digest(path.read_bytes()) == row["sha256"], "Held source digest differs")
        table = probe_frames(path, timeout_s=15)
        require(
            table.to_json() == commitment["clock"]
            and all((pts - table.pts[0]) * table.tb == Fraction(i, 30) for i, pts in enumerate(table.pts)),
            "Held source clock differs",
        )
        require(
            [table.pts[i] for i in selected["frames"]] == selected["pts"],
            "Selected interval differs from native clock",
        )
        anchors = {item["frame_index"]: item for item in selected["anchors"]}
        boundaries = selected["frames"]
        planned = set(range(120))
        scene = {i for edge in boundaries for i in motion_inputs(edge, planned)}
        pixels, boxes, seen = {}, {}, set()
        for frame in decode_frames(path, table, sorted(scene | anchors.keys()), timeout_s=20):
            fingerprint = digest(frame.rgb.tobytes())
            require(
                frame.pts_verified
                and frame.pts == table.pts[frame.index]
                and fingerprint == commitment["rgb_sha256"][frame.index],
                "Selected original picture differs",
            )
            seen.add(frame.index)
            if frame.index in scene:
                pixels[frame.index] = reduced_rgb(frame.rgb)
            if frame.index not in anchors and frame.index + 1 not in boundaries:
                continue
            bgr = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)
            [face] = detect_faces(bgr, detector=detector, min_size=48)
            box = {key: float(getattr(face, key)) for key in ("x", "y", "w", "h")}
            if frame.index + 1 in boundaries:
                boxes[frame.index] = box
            if frame.index not in anchors:
                continue
            anchor = anchors[frame.index]
            require(
                fingerprint == anchor["decoded_rgb_sha256"]
                and frame.pts == anchor["pts"]
                and box == anchor["bbox"],
                "Saved original or measured face box differs",
            )
            detection = face_row(face.landmarks)
            detection[0, :4] = face.box()
            detection[0, 14] = face.score
            aligned = cv2.cvtColor(aligner.alignCrop(bgr, detection), cv2.COLOR_BGR2RGB)
            require(
                digest(aligned.tobytes()) == anchor["aligned_rgb_sha256"], "Held aligned face crop differs"
            )
            checked.update(original_pictures=1, aligned_crops=1)
        require(seen == scene | anchors.keys(), "A held picture was missing from source replay")
        for edge, expected in zip(boundaries, selected["edges"], strict=True):
            require(
                motion(edge, {i: pixels[i] for i in motion_inputs(edge, planned)}, boxes[edge - 1])
                == expected,
                "Near-head motion differs from the original video",
            )
            checked["motion_edges"] += 1
        checked["selected_ranges"] += 1
    return checked


def assemble(folder, development, destination):
    from scripts.video_face_unrouted_short_transfer import summary

    frozen = read(folder / "freeze.json")
    selection = read(folder / "development.json")
    held = read(folder / "held.json")
    measured = read(folder / "incumbent.json")
    previous = read(development / "receipt.json")
    point = read(folder / "point.json")
    require(
        digest((folder / "development.json").read_bytes()) == point["development_sha256"]
        and digest((folder / "media.json").read_bytes()) == held["media_sha256"] == measured["media_sha256"]
        and digest((development / "receipt.json").read_bytes()) == frozen["pins"]["development"]
        and digest((development / "media.json").read_bytes())
        == selection["media_sha256"]
        == frozen["pins"]["development_media"],
        "Study inputs changed",
    )
    result = summary(folder, development)
    require(result["held_incumbent_complete"], "The Full comparison is incomplete")
    payload = {
        "freeze": frozen,
        "development": selection["rows"],
        "held": held["rows"],
        "point": point,
        "baseline": {
            "development": {row["sha256"]: row["prior"] for row in previous["rows"]},
            "held": measured["rows"],
        },
        "held_media_sha256": held["media_sha256"],
        "result": result,
    }
    audit(payload)
    with destination.open("x") as file:
        json.dump(payload, file, sort_keys=True, indent=2, allow_nan=False)
        file.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--native", type=Path)
    parser.add_argument("--assemble", nargs=2, type=Path, metavar=("HELD_RUN", "DEVELOPMENT_RUN"))
    args = parser.parse_args()
    if args.assemble:
        assemble(*args.assemble, args.receipt)
    receipt = read(args.receipt)
    report, picks = audit(receipt)
    if args.native:
        report["native_replay"] = native(receipt, picks, args.native)
    print(json.dumps(report, sort_keys=True, indent=2))
