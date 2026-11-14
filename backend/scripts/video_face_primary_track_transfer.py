# SPDX-License-Identifier: Apache-2.0
"""Evaluate a dominant-face route on native short AVC deliveries."""

import argparse
import hashlib
import json
from collections import Counter
from fractions import Fraction
from pathlib import Path

from scripts.review_video_brief_face_120_carrier import native, overlap, require

ROOT = Path(__file__).resolve().parents[2]
CASES = ROOT / "docs/evaluation-panels"
DATA = ROOT / "backend/data/independent-evaluation"
SCHEMA = "chipmunk.video-face-primary-track/1"
METHODS = ("single_face", "largest_face", "confirmed_track")
DESIGN = {"required_new_groups": 2, "minimum_new_intervals": 3, "max_new_ordinary": 1, "max_new_cut": 0}
PARENTS = (
    "video-face-moving-shortlist-confirmation-2026-11-13.json",
    "video-face-moving-shortlist-development-2026-11-13.json",
)


def inputs(args):
    banks = (
        args.confirmation,
        args.previous,
        DATA / "video-face-two-returns-hd-2026-11-12",
        DATA / "video-face-hd-retry-scene-challenge-2026-11-12",
    )
    paths = {}
    for bank in banks:
        for row in json.loads((bank / "media.json").read_text())["rows"]:
            paths[row["sha256"]] = (bank / "media" / row["file"], row)
    selected = {}
    for book in PARENTS:
        parent = json.loads((CASES / book).read_text())
        for row in parent["rows"]:
            # The independent comparison includes the full set of controls from both studies.
            if row["sha256"] not in selected and (row["first"] is None or not row["truth"]):
                selected[row["sha256"]] = row | {"panel": parent["freeze"]["panel"]}
    return [(case, *paths[digest]) for digest, case in selected.items()]


def measure(path, item, detector, recognizer, earlier, *, alternate_track=False):
    import cv2

    from app.detectors.image.faces import detect_faces
    from app.detectors.video import face_appearance, face_paired_edge, face_primary_track, face_short_proposal
    from app.detectors.video.adapters.face_appearance import _measure
    from app.detectors.video.decode import FrameTable, decode_frames, probe_frames
    from app.detectors.video.face_return import source_carrier
    from app.evidence import face_primary_track_binding as track_binding
    from app.evidence import face_short_proposal_binding as proof

    table = probe_frames(path, cap=130, timeout_s=18)
    require(
        table == FrameTable.from_json(item["clock"])
        and proof.avc_source(table, proof.AVC_ENCODING, source_carrier(path)),
        "Native video profile differs",
    )
    reduced, detections, values = {}, {}, {}

    def verify(frame):
        i = frame.index
        require(
            frame.pts_verified
            and frame.pts == table.pts[i]
            and hashlib.sha256(frame.rgb).hexdigest() == item["rgb_sha256"][i],
            "Original decoded picture differs",
        )

    def detect(frame, indices):
        bgr = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)
        found = detect_faces(bgr, detector=detector, min_size=48)
        detections[frame.index] = found
        wanted = [selectors[name](frame.index, found) for name in indices]
        for face in {f.box(): f for f in wanted if f is not None}.values():
            reading = _measure(frame.rgb, bgr, frame.index, table, face, recognizer)
            if reading is not None:
                v, record = reading
                values[frame.index, face.box()] = face_appearance.unit(v), face, record

    def pick_single(_, found):
        return found[0] if len(found) == 1 else None

    def pick_largest(_, found):
        return found[0] if found else None

    anchors = {}
    alternate_anchors = {}

    def pick_tracked(index, found):
        return face_primary_track.follow(index, found, anchors)

    selectors = dict(zip(METHODS, (pick_single, pick_largest, pick_tracked), strict=True))
    if alternate_track:
        def pick_alternate(index, found):
            selected = track_binding.follow(index, [face.box() for face in found], alternate_anchors,
                                            position_only=True)
            return next((face for face in found if face.box() == selected), None)

        selectors["position_track"] = pick_alternate

    def check(*_):
        return None

    for frame in decode_frames(path, table, None, timeout_s=35):
        verify(frame)
        require(frame.index == len(reduced), "Native sequence is incomplete")
        reduced[frame.index] = face_paired_edge.reduced_rgb(frame.rgb)
        if frame.index in proof.ANCHOR_SAMPLES:
            # Both baseline routes use either the sole face or the top area face.
            detect(frame, ("single_face", "largest_face"))
    require(len(reduced) == 120, "Incomplete picture analysis")
    anchors = face_primary_track.find_anchors(detections)
    if alternate_track:
        alternate_anchors = track_binding.anchors({i: [face.box() for face in found] for i, found in detections.items()},
                                                  position_only=True)
        # The alternative can select a different face from the one with the largest area.
        additional = {i for i in proof.ANCHOR_SAMPLES if (face := pick_alternate(i, detections[i])) is not None
                      and (i, face.box()) not in values}
        if additional:
            for frame in decode_frames(path, table, sorted(additional), timeout_s=35):
                verify(frame)
                detect(frame, ("position_track",))
    plan = {}
    for name in selectors:
        boxes = {}
        for index in proof.ANCHOR_SAMPLES:
            face = selectors[name](index, detections[index])
            if face is not None and (reading := values.get((index, face.box()))):
                boxes[index] = reading[2]["bbox"]
        edges = face_short_proposal.moving_shortlist(reduced, boxes, check)
        plan[name] = (edges, proof.face_schedule(edges))
    requested = {i for _, schedule in plan.values() for i in schedule} - set(proof.ANCHOR_SAMPLES)
    if requested:
        read = set()
        for frame in decode_frames(path, table, sorted(requested), timeout_s=35):
            i = frame.index
            verify(frame)
            detect(frame, tuple(name for name, (_, schedule) in plan.items() if i in schedule))
            read.add(i)
        require(read == requested, "Not all scheduled pictures decoded")

    def excluded(start, end):
        return any(overlap((start, end), span) for span in earlier)

    out = {}
    for name, (edges, schedule) in plan.items():
        measured = {}
        for i in schedule:
            if (face := selectors[name](i, detections[i])) is not None and (
                reading := values.get((i, face.box()))
            ):
                measured[i] = reading
        selected = face_short_proposal.select(
            edges, measured, reduced, check, excluded, durations=proof.SECOND_AVC_DURATIONS
        )
        record = None
        if selected:
            (negative, start, end), support, _ = selected
            center = min(
                (i for i in measured if start <= i < end), key=lambda i: (abs(2 * i - start - end), i)
            )
            record = {
                "frames": [start, end],
                "response": -negative,
                "pts": [table.pts[start], table.pts[end]],
                "edges": [support[side]["frame_indices"] for side in ("entry", "exit")],
                "anchors": [
                    {k: measured[i][2][k] for k in ("frame_index", "pts", "decoded_rgb_sha256")}
                    for i in (start - 1, center, end)
                ],
            }
        out[name] = {"measured": len(measured), "scheduled": len(schedule), "proposal": record}
        if alternate_track:
            out[name]["complete_pair"] = face_short_proposal.has_measurable_pair(
                edges, measured, proof.SECOND_AVC_DURATIONS,
            )
    return out, sum(len(found) > 1 for found in detections.values())


def scan(args, rows):
    import cv2

    from app.detectors.image.faces import load_yunet
    from scripts.video_brief_face_120_carrier import WEIGHTS

    cv2.setNumThreads(1)
    weights = ROOT / "models/weights"
    require(
        all(sha(weights / name) == expected for name, expected in WEIGHTS.items()), "Face asset bytes differ"
    )
    detector = load_yunet(weights / "face_detection_yunet_2023mar.onnx")
    recognizer = cv2.FaceRecognizerSF.create(str(weights / "face_recognition_sface_2021dec.onnx"), "")
    with (args.run / "rows.jsonl").open("x") as output:
        for index, (case, source, item) in enumerate(rows, 1):
            require(sha(source) == item["sha256"] and item["truth"] == case["truth"], "Study source differs")
            methods, crowded = measure(source, item, detector, recognizer, case["prior"])
            if case["first"] is not None:
                # These ordinary controls already had a result, so the new route would never alter them.
                require(not item["truth"], "Only ordinary controls can already have a first result")
            frozen_candidate = (case["candidates"] or {}).get("moving_wider", {}).get("choice")
            replayed = methods["single_face"]["proposal"]
            if case["first"] is None:
                require(
                    (replayed and replayed["frames"]) == (frozen_candidate and frozen_candidate["frames"]),
                    "Historical single-face comparison did not reproduce",
                )
            record = {
                k: case[k]
                for k in (
                    "sha256",
                    "bank",
                    "panel",
                    "group",
                    "condition",
                    "truth",
                    "prior",
                    "first",
                    "time_base",
                    "first_pts",
                )
            }
            record.update(index=index, methods=methods, multiple_faces=crowded)
            output.write(json.dumps(record, allow_nan=False) + "\n")
            output.flush()
            print(
                index,
                case["panel"],
                case["condition"],
                [m for m, reading in methods.items() if reading["proposal"]],
                flush=True,
            )


def review(rows, methods=METHODS):
    require(
        [r["index"] for r in rows] == list(range(1, len(rows) + 1))
        and len({r["sha256"] for r in rows}) == len(rows),
        "Missing or duplicate evidence rows",
    )
    outcomes = {}
    for panel in sorted({r["panel"] for r in rows}):
        cases = [r for r in rows if r["panel"] == panel]
        counts = {
            "clips": len(cases),
            "ordinary": sum(not r["truth"] for r in cases),
            "edited": sum(bool(r["truth"]) for r in cases),
        }
        for arm in methods:
            totals, groups = Counter(), set()
            for item in cases:
                result = item["methods"][arm]
                require(0 <= result["measured"] <= result["scheduled"] <= 84, "Face input budget exceeded")
                if (proposal := result["proposal"]) is None or item["first"] is not None:
                    continue
                start, end = proposal["frames"]
                a, b, c = proposal["anchors"]
                require(
                    5 <= end - start <= 9
                    and proposal["response"] > 0.10
                    and [a["frame_index"], c["frame_index"]] == [start - 1, end]
                    and start <= b["frame_index"] < end
                    and proposal["edges"] == [list(range(i - 3, i + 3)) for i in (start, end)],
                    "Invalid boundaries",
                )
                points = [
                    *zip((start, end), proposal["pts"], strict=True),
                    *((a["frame_index"], a["pts"]) for a in proposal["anchors"]),
                ]
                require(
                    all(
                        (tick - item["first_pts"]) * Fraction(item["time_base"]) == Fraction(i, 30)
                        for i, tick in points
                    ),
                    "Wrong native timestamp",
                )
                existing = item["methods"]["single_face"]["proposal"] if arm != "single_face" else None
                if existing:
                    continue
                hits = [
                    span
                    for span in item["truth"]
                    if not any(overlap(span, old) for old in item["prior"]) and overlap(span, [start, end])
                ]
                if hits:
                    groups.add((item["bank"], item["group"]))
                totals["useful"] += len(hits)
                totals["exact"] += [start, end] in hits
                totals["misplaced"] += bool(item["truth"]) and not hits
                totals["ordinary_flags"] += not item["truth"]
                totals["cut_flags"] += not item["truth"] and (
                    "cut" in item["condition"] or "person_switch" in item["condition"]
                )
            counts[arm] = {
                k: totals[k] for k in ("useful", "exact", "misplaced", "ordinary_flags", "cut_flags")
            }
            counts[arm]["groups"] = len(groups)
        outcomes[panel] = counts
    return outcomes


def sha(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def replay_pixels(rows, paths):
    witnesses = 0
    for item in rows:
        selected = {
            anchor["frame_index"]: anchor
            for reading in item["methods"].values()
            if reading["proposal"]
            for anchor in reading["proposal"]["anchors"]
        }
        witnesses += native(
            paths[item["sha256"]],
            item | {"delivery_sha256": item["sha256"], "proposal": {"anchors": list(selected.values())}},
            "avc",
        )
    return {"clips": len(rows), "presentation_ticks": 120 * len(rows), "selected_original_frames": witnesses}


def freeze(args):
    from scripts.video_brief_face_120_carrier import WEIGHTS

    rows = inputs(args)
    method = ROOT / "backend/app/detectors/video/face_primary_track.py"
    return {
        "schema": SCHEMA,
        "methods": {p.name: sha(p) for p in (Path(__file__), method)},
        "parents": {p: sha(CASES / p) for p in PARENTS},
        "design": DESIGN,
        "weights": WEIGHTS,
        "inputs": [entry[0]["sha256"] for entry in rows],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "run", "review", "pixels"))
    parser.add_argument("run", type=Path)
    parser.add_argument(
        "--confirmation", default=Path("/private/tmp/chipmunk-moving-shortlist-confirm-v1"), type=Path
    )
    parser.add_argument("--previous", default=Path("/private/tmp/chipmunk-avc-distinct-fresh-v1"), type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.phase == "freeze":
        args.run.mkdir(mode=0o700)
        (args.run / "freeze.json").write_text(json.dumps(freeze(args), indent=2) + "\n")
        return
    if args.run.is_file():
        receipt = json.loads(args.run.read_text())
        frozen, rows = receipt["freeze"], receipt["rows"]
    else:
        frozen = json.loads((args.run / "freeze.json").read_text())
        if args.phase == "run":
            require(frozen == freeze(args), "The frozen code or roster differs")
            scan(args, inputs(args))
            return
        rows = [json.loads(line) for line in (args.run / "rows.jsonl").read_text().splitlines()]
    require(
        frozen["schema"] == SCHEMA
        and frozen["design"] == DESIGN
        and [r["sha256"] for r in rows] == frozen["inputs"],
        "Study definition differs",
    )
    result = review(rows)
    if args.run.is_file():
        require(receipt["result"] == result, "Public outcome differs")
    if args.phase == "pixels":
        paths = {case["sha256"]: path for case, path, _ in inputs(args)}
        result = replay_pixels(rows, paths)
    if args.output and args.phase == "review":
        with args.output.open("x") as output:
            json.dump({"freeze": frozen, "rows": rows, "result": result}, output, indent=2, allow_nan=False)
            output.write("\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
