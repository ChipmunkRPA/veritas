# SPDX-License-Identifier: Apache-2.0
"""Compare independent-edge and paired-edge scheduling for brief AVC face changes."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from scripts.review_video_brief_face_120_carrier import overlap, require

DESIGN = {
    "panel": "Retained 120-picture AVC comparison. This replay reuses sources already seen in prior studies.",
    "routing": "Keep all existing ranges and try the paired shortlist only if neither original short check found a range.",
    "minimum_added_locations": 3,
    "minimum_gained_groups": 2,
    "maximum_added_ordinary_flags": 1,
    "maximum_added_cross_recording_or_speaker_cut_flags": 0,
    "maximum_added_misplaced_ranges": 1,
}


def schedule_pairs(ranked, proof):
    strengths = {edge: score for score, edge in ranked}
    pairs = []
    for start, score in strengths.items():
        for length in proof.DURATIONS:
            end = start + length
            if end in strengths:
                other = strengths[end]
                pairs.append((-min(score, other), -max(score, other), start, end))

    kept = {}
    measured = set(proof.ANCHOR_SAMPLES)
    single_edges = ((edge,) for _, edge in sorted(ranked, key=lambda item: (-item[0], item[1])))
    for edges in [*(pair[2:] for pair in sorted(pairs)), *single_edges]:
        added = [edge for edge in edges if edge not in kept]
        if len(kept) + len(added) > proof.MAX_EDGES:
            continue
        pictures = {index for edge in added for index in proof.edge_frames(edge)}
        if len(measured | pictures) <= proof.MAX_FACE_READS:
            kept.update(dict.fromkeys(added))
            measured.update(pictures)
        if len(kept) == proof.MAX_EDGES:
            break
    return list(kept)


def freeze():
    from app.detectors.video import face_appearance
    from app.detectors.video.adapters import face_appearance as adapter
    from scripts import video_brief_face_120_carrier as carrier
    from scripts.video_common_sources_v89 import sha

    return {
        "schema": "chipmunk.video-brief-face-paired-shortlist/1",
        "design": DESIGN,
        "carrier": carrier.freeze(),
        "methods": {Path(path).name: sha(Path(path)) for path in (__file__, face_appearance.__file__)},
        "face_reader": sha(Path(adapter.__file__)),
    }


def scan(source, table, expected_pixels, detector, recognizer, exclusions, retained_short):
    import cv2
    import numpy as np

    from app.detectors.image.faces import detect_faces
    from app.detectors.video import face_appearance, face_paired_edge, face_short_proposal
    from app.detectors.video.adapters.face_appearance import _measure
    from app.detectors.video.decode import decode_frames
    from app.evidence import face_short_proposal_binding as proof

    reduced, faces = {}, {}
    anchor_indices = set(proof.ANCHOR_SAMPLES)

    def verify(frame):
        index = frame.index
        require(
            frame.pts_verified
            and frame.pts == table.pts[index]
            and hashlib.sha256(frame.rgb).hexdigest() == expected_pixels[index],
            "Decoded picture differs from the archived clock or native RGB",
        )

    def measure(frame):
        bgr = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)
        detected = detect_faces(bgr, detector=detector, min_size=48)
        if len(detected) == 1 and (
            result := _measure(frame.rgb, bgr, frame.index, table, detected[0], recognizer)
        ):
            vector, record = result
            faces[frame.index] = face_appearance.unit(vector), detected[0], record

    for frame in decode_frames(source, table, None, timeout_s=35):
        verify(frame)
        require(frame.index == len(reduced), "Native frames are missing or out of order")
        reduced[frame.index] = face_paired_edge.reduced_rgb(frame.rgb)
        if frame.index in anchor_indices:
            measure(frame)
    require(len(reduced) == len(expected_pixels) == 120, "Native pictures missing")

    boxes = [record["bbox"] for _, _, record in faces.values()]
    ranked = []
    if len(boxes) >= 4:
        box = {key: float(np.median([item[key] for item in boxes])) for key in ("x", "y", "w", "h")}
        for index in proof.EDGE_SCAN:
            if result := face_paired_edge.motion(index, reduced, box):
                ranked.append((float.fromhex(result["central_hex"]), index))
    edge_sets = {
        "current": face_short_proposal.shortlist(reduced, boxes, lambda *_: None),
        "paired": schedule_pairs(ranked, proof),
    }
    schedules = {name: set(proof.face_schedule(edges)) for name, edges in edge_sets.items()}
    require(
        all(len(edges) <= proof.MAX_EDGES for edges in edge_sets.values())
        and all(len(indices) <= proof.MAX_FACE_READS for indices in schedules.values()),
        "Exceeded the serving budget",
    )
    requested = set.union(*schedules.values()) - anchor_indices
    if requested:
        decoded = set()
        for frame in decode_frames(source, table, sorted(requested), timeout_s=35):
            verify(frame)
            decoded.add(frame.index)
            measure(frame)
        require(decoded == requested, "A scheduled face picture was not decoded")

    def excluded(start, end):
        return any(overlap((start, end), previous) for previous in exclusions)

    methods = {}
    for name, edges in edge_sets.items():
        inputs = {i: faces[i] for i in schedules[name] if i in faces}
        candidate = face_short_proposal.select(edges, inputs, reduced, lambda *_: None, excluded)
        result = None
        if candidate is not None and not retained_short:
            ((negative, start, end), _, _) = candidate
            middle = min((i for i in inputs if start <= i < end), key=lambda i: (abs(2 * i - start - end), i))
            result = {
                "frames": [start, end],
                "pts": [table.pts[start], table.pts[end]],
                "response": -negative,
                "anchors": [
                    {
                        key: inputs[i][2][key]
                        for key in ("frame_index", "pts", "decoded_rgb_sha256", "aligned_rgb_sha256")
                    }
                    for i in (start - 1, middle, end)
                ],
            }
        methods[name] = {
            "edges": edges,
            "scheduled": len(schedules[name]),
            "face_inputs": len(inputs),
            "proposal": result,
        }
    return methods, sorted(edge for _, edge in ranked)


def run(folder):
    import cv2

    from app.detectors.image.faces import load_yunet
    from app.detectors.video.decode import FrameTable, probe_frames
    from app.detectors.video.face_return import source_carrier
    from app.evidence.face_short_proposal_binding import AVC_ENCODING, avc_source
    from scripts import video_brief_face_120_carrier as carrier
    from scripts.video_common_sources_v89 import sha

    require(json.loads((folder / "freeze.json").read_text()) == freeze(), "Frozen code or manifests changed")
    cv2.setNumThreads(1)
    weights = carrier.ROOT / "models/weights"
    require(
        all(sha(weights / name) == digest for name, digest in carrier.WEIGHTS.items()), "Model bytes differ"
    )
    detector = load_yunet(weights / "face_detection_yunet_2023mar.onnx")
    recognizer = cv2.FaceRecognizerSF.create(str(weights / "face_recognition_sface_2021dec.onnx"), "")
    roster = list(carrier.roster())
    require(len(roster) == 56, "Retained replay panel is incomplete")
    with (folder / "rows.jsonl").open("x") as journal:
        for index, (path, row, bank, prior, retained) in enumerate(roster):
            require(sha(path) == row["sha256"], "Delivery bytes differ")
            table = probe_frames(path, cap=130, timeout_s=18)
            require(
                table == FrameTable.from_json(row["clock"])
                and avc_source(table, AVC_ENCODING, source_carrier(path)),
                "Unsupported native pictures or presentation clock",
            )
            methods, motion = scan(path, table, row["rgb_sha256"], detector, recognizer, prior, retained)
            item = {key: row[key] for key in ("sha256", "group", "condition", "truth")}
            item.update(
                index=index,
                bank=bank,
                prior=prior,
                retained_short=retained,
                time_base=table.time_base,
                first_pts=table.pts[0],
                methods=methods,
                motion_edges=motion,
            )
            journal.write(json.dumps(item, allow_nan=False) + "\n")
            journal.flush()
            print(
                index + 1,
                row["condition"],
                [
                    (arm, data["proposal"]["frames"] if data["proposal"] else None)
                    for arm, data in methods.items()
                ],
                flush=True,
            )


def summarize(rows):
    from fractions import Fraction

    require(len(rows) == 56 and [row["index"] for row in rows] == list(range(56)), "Incomplete replay")
    results, gains, diagnostics = Counter(), set(), Counter()
    for name in (
        "added_useful_locations",
        "added_ordinary_flags",
        "added_misplaced_ranges",
        "added_cross_recording_or_speaker_cut_flags",
        "exact_added_locations",
    ):
        results[name] = 0
    for row in rows:
        current, paired = (row["methods"][name]["proposal"] for name in ("current", "paired"))
        current_ranges = [*row["prior"], *([current["frames"]] if current else [])]
        fallback = paired if current is None else None
        results["different_edge_sets"] += set(row["methods"]["current"]["edges"]) != set(
            row["methods"]["paired"]["edges"]
        )
        results["current_short_proposals"] += current is not None
        results["paired_short_proposals"] += paired is not None
        if row["truth"]:
            results["edited_deliveries"] += 1
            base = [truth for truth in row["truth"] if any(overlap(truth, span) for span in current_ranges)]
            results["known_edits"] += len(row["truth"])
            results["current_useful_locations"] += len(base)
            for truth in row["truth"]:
                if truth in base:
                    continue
                if row["retained_short"] or current:
                    diagnostics["missed_edits_in_clips_with_an_existing_short_range"] += 1
                    continue
                diagnostics["eligible_missed_edits"] += 1
                for name, edges in (
                    ("motion", row["motion_edges"]),
                    *((name, method["edges"]) for name, method in row["methods"].items()),
                ):
                    if all(edge in edges for edge in truth):
                        diagnostics[name] += 1
            if fallback:
                span = fallback["frames"]
                added = [truth for truth in row["truth"] if truth not in base and overlap(truth, span)]
                if added:
                    results["added_useful_locations"] += len(added)
                    results["exact_added_locations"] += int(span in added)
                    gains.add((row["bank"], row["group"]))
                else:
                    results["added_misplaced_ranges"] += 1
        else:
            results["ordinary_deliveries"] += 1
            results["ordinary_deliveries_flagged_by_current"] += bool(current_ranges)
            if fallback:
                results["added_ordinary_flags"] += 1
                if row["condition"] in ("cross_recording_cuts", "person_switch", "tandem_person_switch"):
                    results["added_cross_recording_or_speaker_cut_flags"] += 1
        for name in ("current", "paired"):
            if proposal := row["methods"][name]["proposal"]:
                start, end = proposal["frames"]
                base = Fraction(row["time_base"])
                require(
                    all(
                        (pts - row["first_pts"]) * base == Fraction(index, 30)
                        for index, pts in zip((start, end), proposal["pts"], strict=True)
                    )
                    and [anchor["frame_index"] for anchor in proposal["anchors"]][::2] == [start - 1, end]
                    and start <= proposal["anchors"][1]["frame_index"] < end,
                    "Invalid source exhibit timeline",
                )
                for anchor in proposal["anchors"]:
                    require(
                        (anchor["pts"] - row["first_pts"]) * base == Fraction(anchor["frame_index"], 30),
                        "Exhibit timestamp differs from its original picture",
                    )
    results["gained_groups"] = len(gains)
    gates = {
        name: (
            results[name.removeprefix("minimum_")] >= point
            if name.startswith("minimum_")
            else results[name.removeprefix("maximum_")] <= point
        )
        for name, point in DESIGN.items()
        if name.startswith(("minimum_", "maximum_"))
    }
    return {
        "outcomes": dict(results),
        "missed_edit_diagnostics": dict(diagnostics),
        "gates": gates,
        "met_frozen_criteria": all(gates.values()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "run", "review"))
    parser.add_argument("path", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.phase == "freeze":
        args.path.mkdir(mode=0o700)
        (args.path / "freeze.json").write_text(json.dumps(freeze(), indent=2, allow_nan=False) + "\n")
    elif args.phase == "run":
        run(args.path)
    else:
        if args.path.is_dir():
            rows = [json.loads(line) for line in (args.path / "rows.jsonl").read_text().splitlines()]
            result = summarize(rows)
            if args.output:
                freeze_record = json.loads((args.path / "freeze.json").read_text())
                with args.output.open("x") as output:
                    json.dump(
                        {"freeze": freeze_record, "rows": rows, "result": result},
                        output,
                        indent=2,
                        allow_nan=False,
                    )
                    output.write("\n")
        else:
            receipt = json.loads(args.path.read_text())
            require(receipt["freeze"]["design"] == DESIGN, "Frozen design differs")
            result = summarize(receipt["rows"])
            require(result == receipt["result"], "Replay differs from the recorded outcome")
        print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
