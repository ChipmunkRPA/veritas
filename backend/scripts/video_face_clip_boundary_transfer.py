# SPDX-License-Identifier: Apache-2.0
"""Measure short native face-change ranges near the beginning and end of an upload."""

import argparse
import hashlib
import json
from collections import Counter
from fractions import Fraction
from pathlib import Path

from scripts.review_video_brief_face_120_carrier import SHA, overlap, require

SCHEMA = "chipmunk.video-face-clip-boundary/1"
POSITIONS = ((3, 9), (4, 11), (5, 13), (109, 115), (109, 116), (109, 117))
CASES = ("original", "early_swap", "late_swap", "early_light", "late_cut", "early_jump", "late_reencode")
READINGS = ("current", "boundary")
DESIGN = {"timeline": "Exactly 120 presented 720p pictures at 30 fps, upright 4:2:0 H.264 MP4.",
          "candidate": "Reserve whole pairs near the crop boundaries. Three actual native face samples on each side "
                       "of both adjacent-picture changes, using the fixed two-edge appearance and motion measurements.",
          "comparison": "Run the current stationary and moving short proposal first. Compare with a near-end supplement "
                        "using the same inputs. Keep the current proposal when it selected a range.",
          "coverage": "Counts use temporal IoU >= 0.5. Exact matches require both native frame indices. "
                      "Keep unmatched ordinary ranges and insufficient face coverage visible."}


def source_plan(prior, *, rate=30, groups=12, development=4):
    from app.detectors.video.decode import FrameTable
    from scripts.video_brief_face_avc_clock_confirmation import eligible_rate
    from scripts.video_brief_face_avc_confirmation import NATIVE, QUALIFIED
    from scripts.video_chroma_boundary_transfer import source_pin
    from scripts.video_common_sources_v89 import read
    from scripts.video_face_short_boundary_confirmation import metadata
    from scripts.video_native_sources_v92 import ARCHIVE

    catalog = metadata()
    qualified = {row["group"]: row["qualification"] for row in read(QUALIFIED)["assets"] if row["kind"] == "original_videos"}
    inventory = [read(path) for path in NATIVE.glob("group-*-complete.json")]
    by_index = {row["index"]: row for row in inventory}
    sources = read(prior)["sources"]
    seen = [by_index[row["index"]]["group"] for row in sources]
    urls = {catalog[group]["url"] for group in seen}
    donors = set().union(*(catalog[group]["donors"] for group in seen))
    components = {qualified[group]["component"] for group in seen}
    ordered = sorted(inventory, key=lambda row: hashlib.sha256(f"{SCHEMA}/{row['group']}".encode()).digest())
    output = []
    for slot in range(groups):
        early = POSITIONS[slot % 3]
        late = POSITIONS[3 + (slot // 3 + slot) % 3]
        offsets = (60 - early[0], 120 - late[1])
        families = ("swap_a", "swap_b") if slot % 2 == 0 else ("swap_b", "swap_a")
        for item in ordered:
            group = item["group"]
            details = catalog[group]
            qualification = qualified[group]
            if (details["url"] in urls or details["donors"] & donors or qualification["component"] in components
                    or "child" in details["tags"] or not qualification["eligible"] or qualification["historical_rgb_connection"]):
                continue
            cases = {case["condition"]: case for case in item["cases"]}
            table = FrameTable.from_json(cases["original"]["frame_table"])
            if (table.width, table.height) not in ((1280, 720), (1920, 1080)) or eligible_rate(table) != rate:
                continue
            if not all(offset + 120 <= len(table) and FrameTable.from_json(cases[family]["frame_table"]) == table
                       and set(range(offset + a, offset + b)) <= set(cases[family]["actual_changed_frames"])
                       for family, offset, (a, b) in zip(families, offsets, (early, late), strict=True)):
                continue
            output.append(source_pin(item["index"], ARCHIVE) | {"early": list(early), "late": list(late),
                          "families": list(families), "offsets": list(offsets), "split": "development" if slot < development else "held"})
            urls.add(details["url"])
            donors.update(details["donors"])
            components.add(qualification["component"])
            break
        require(len(output) == slot + 1, f"Missing a separate native source group for slot {slot+1}")
    return output


def prepare(output, split, *, rate=30):
    import cv2
    import numpy as np

    from scripts.video_chroma_boundary_transfer import common
    from scripts.video_common_sources_v89 import read, write
    from scripts.video_face_720p30_transfer import authored, describe, native
    from scripts.video_native_sources_v92 import ARCHIVE

    cv2.setNumThreads(1)
    plan = read(output / "freeze.json")
    bank = [row for row in plan["sources"] if row["split"] == split]
    folder = output / split
    folder.mkdir(mode=0o700)
    rows = []
    for slot, source in enumerate(bank):
        early, late = (range(*source[key]) for key in ("early", "late"))
        early_offset, late_offset = source["offsets"]
        original = native(ARCHIVE, common, source["index"], "original", range(late_offset, early_offset + 120))
        guest = bank[(slot+1) % len(bank)]

        def excerpt(source, family, positions, offset):
            measured = native(ARCHIVE, common, source["index"], family, [i+offset for i in positions])
            return {i: measured[i+offset] for i in positions}

        variants = {
            "original": {}, "early_swap": excerpt(source, source["families"][0], early, early_offset),
            "late_swap": excerpt(source, source["families"][1], late, late_offset),
            "early_light": {i: np.clip(original[i+early_offset].astype("int16") + 24, 0, 255).astype("uint8") for i in early},
            "early_jump": {i: original[i+early_offset+18] for i in early},
            "late_cut": excerpt(guest, "original", late, guest["offsets"][1]),
        }
        for condition in CASES:
            path = folder / f"{slot+1:02d}-{condition}.mp4"
            variant = variants[condition] if condition != "late_reencode" else None
            if variant is None:
                authored(path, source=folder / f"{slot+1:02d}-late_swap.mp4", crf=35)
            else:
                offset = early_offset if condition.startswith("early") else late_offset
                require(all(not np.array_equal(original[i+offset], frame) for i, frame in variant.items()), "Authored picture did not change")
                authored(path, pictures=(variant.get(i, original[i+offset]) for i in range(120)), rate=rate)
            truth = [source["early"]] if condition == "early_swap" else [source["late"]] if condition in {"late_swap", "late_reencode"} else []
            rows.append(describe(path, source["index"], condition, truth, rate=rate))
        print(f"Prepared {split} group {slot+1}/{len(bank)}", flush=True)
    write(folder / "media.json", rows)


def read_clip(path, row, detector, recognizer):
    import cv2

    from app.detectors.image.faces import detect_faces
    from app.detectors.video import face_appearance, face_paired_edge
    from app.detectors.video import face_short_proposal as proposal
    from app.detectors.video.adapters.face_appearance import _measure
    from app.detectors.video.decode import FrameTable, decode_frames
    from app.evidence import face_short_proposal_binding as proof

    table = FrameTable.from_json(row["clock"])
    frames, reduced, faces = {}, {}, {}
    for frame in decode_frames(path, table, None, timeout_s=45):
        index = frame.index
        require(frame.pts_verified and frame.pts == table.pts[index]
                and hashlib.sha256(frame.rgb).hexdigest() == row["rgb_sha256"][index], "Decoded picture or native clock differs")
        frames[index] = frame.rgb.copy()
        reduced[index] = face_paired_edge.reduced_rgb(frame.rgb)
    require(set(frames) == proof.PLAN, "The exact clip frame range is incomplete")

    def measure(indices):
        for i in indices:
            if i in faces:
                continue
            rgb = frames[i]
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            detected = detect_faces(bgr, detector=detector, min_size=48)
            result = _measure(rgb, bgr, i, table, detected[0], recognizer) if len(detected) == 1 else None
            faces[i] = (face_appearance.unit(result[0]), detected[0], result[1]) if result is not None else None

    def check(_):
        return None

    measure(proof.ANCHOR_SAMPLES)
    anchors = {i: faces[i][2]["bbox"] for i in proof.ANCHOR_SAMPLES if faces[i] is not None}

    def evaluate(edges, durations, *, boundary=False):
        schedule = proof.face_schedule(edges)
        measure(schedule)
        support = {i: faces[i] for i in schedule if faces[i] is not None}
        result = proposal.select(edges, support, reduced, check,
                                 lambda a, b: boundary and (a, b) not in proof.BOUNDARY_PAIRS, durations=durations)
        measured = len(anchors) >= 4 and (not edges or proposal.has_measurable_pair(edges, support, durations))
        if result is None:
            return {"selected": None, "measured": measured, "scheduled": len(schedule)}
        (_, a, b), _, _ = result
        middle = min((*range(a, a+3), *range(b-3, b)), key=lambda i: (abs(2*i-a-b), i))
        samples = [{"index": i, "pts": table.pts[i], "rgb_sha256": row["rgb_sha256"][i]} for i in (a-1, middle, b)]
        return {"selected": [a, b], "measured": True, "scheduled": len(schedule), "anchors": samples}

    stationary = proposal.shortlist(reduced, list(anchors.values()), check)
    prior = evaluate(stationary, proof.DURATIONS)
    if prior["selected"] is None:
        prior = evaluate(proposal.moving_shortlist(reduced, anchors, check), proof.SECOND_AVC_DURATIONS)
    edge = proposal.boundary_shortlist(reduced, list(anchors.values()), check)
    return {"current": prior, "boundary": evaluate(edge, proof.SECOND_AVC_DURATIONS, boundary=True)}


def score(output, split, *, rate=30, group_count=None):
    import cv2

    from app.detectors.image.faces import load_yunet
    from app.evidence import face_short_proposal_binding as proof
    from scripts.video_brief_face_120_carrier import ROOT, WEIGHTS
    from scripts.video_common_sources_v89 import read, sha, write

    cv2.setNumThreads(1)
    weights = ROOT / "models/weights"
    require(all(sha(weights / name) == digest for name, digest in WEIGHTS.items()), "Pinned face models differ")
    detector = load_yunet(weights / "face_detection_yunet_2023mar.onnx")
    recognizer = cv2.FaceRecognizerSF.create(str(weights / "face_recognition_sface_2021dec.onnx"), "")
    folder = output / split
    rows = []
    for row in read(folder / "media.json"):
        path = folder / row["file"]
        require(sha(path) == row["sha256"], "Fresh delivery bytes differ")
        readings = read_clip(path, row, detector, recognizer)
        clock = row["clock"]
        require(len(clock["pts"]) == len(proof.PLAN) and all(
            (tick - clock["pts"][0]) * Fraction(clock["time_base"]) == Fraction(i, rate)
            for i, tick in enumerate(clock["pts"])), "Unexpected clock")
        rows.append({key: row[key] for key in ("group", "condition", "truth", "sha256")} | {
            "first_pts": clock["pts"][0], "time_base": clock["time_base"], "readings": readings})
        print(split, row["file"], {key: result["selected"] for key, result in readings.items()}, flush=True)
    plan = read(output / "freeze.json")
    record = {"schema": SCHEMA, "method": DESIGN, "split": split, "weights": WEIGHTS, "rows": rows,
              "groups": [{key: row[key] for key in ("index", "early", "late", "receipt_sha256")} for row in plan["sources"] if row["split"] == split]}
    record["results"] = review(record, rate=rate, group_count=group_count)
    write(output / f"{split}.json", record)


def review(record, *, rate=30, group_count=None):
    require(record["schema"] == SCHEMA and record["method"] == DESIGN, "Unknown evaluation method")
    groups = record["groups"]
    require([(row["group"], row["condition"]) for row in record["rows"]]
            == [(group["index"], case) for group in groups for case in CASES], "Frozen case roster differs")
    grouped = {group["index"]: group for group in groups}
    expected = group_count if group_count is not None else {"development": 4, "held": 8}[record["split"]]
    require(len(grouped) == len(groups) == expected, "Source groups differ")
    counts = {case: Counter() for case in CASES}
    for row in record["rows"]:
        case = row["condition"]
        group = grouped[row["group"]]
        truth = ([group["early"]] if case == "early_swap" else [group["late"]] if case in {"late_swap", "late_reencode"} else [])
        require(row["truth"] == truth and SHA.fullmatch(row["sha256"]), "Known edit range or clip digest differs")
        outcomes = row["readings"]
        for name in READINGS:
            outcome = outcomes[name]
            require(12 <= outcome["scheduled"] <= 84, "Face sample budget exceeded")
            if selected := outcome["selected"]:
                a, b = selected
                base = Fraction(row["time_base"])
                anchor = outcome["anchors"]
                require(outcome["measured"] and 3 <= a < b <= 117 and 5 <= b-a <= 9 and
                        [item["index"] for item in anchor] == [a-1, min((*range(a, a+3), *range(b-3, b)), key=lambda i: (abs(2*i-a-b), i)), b]
                        and all((item["pts"] - row["first_pts"]) * base == Fraction(item["index"], rate)
                                and SHA.fullmatch(item["rgb_sha256"]) for item in anchor), "Evidence timestamps differ")
        prior = outcomes["current"]["selected"]
        supplement = outcomes["boundary"]["selected"] if prior is None else None
        expanded = prior or supplement
        counts[case].update(files=1, edits=len(truth), prior_measured=int(outcomes["current"]["measured"]),
                            boundary_measured=int(outcomes["boundary"]["measured"]), prior_flags=int(prior is not None),
                            expanded_flags=int(expanded is not None), prior_exact=int(prior in truth), expanded_exact=int(expanded in truth),
                            prior_useful=int(prior is not None and any(overlap(prior, item) for item in truth)),
                            expanded_useful=int(expanded is not None and any(overlap(expanded, item) for item in truth)))
    result = {key: dict(value) for key, value in counts.items()}
    require("results" not in record or record["results"] == result, "Held outcomes differ")
    return result


def pixels(output, split, *, rate=30, group_count=None):
    from scripts.review_video_brief_face_120_carrier import native

    folder = output / split
    record = json.loads((output / f"{split}.json").read_bytes())
    review(record, rate=rate, group_count=group_count)
    media = {(row["group"], row["condition"]): row for row in json.loads((folder / "media.json").read_bytes())}
    anchors = 0
    for row in record["rows"]:
        clip = media[row["group"], row["condition"]]
        require(clip["sha256"] == row["sha256"], "Independent replay input differs")
        selected = {}
        for reading in row["readings"].values():
            for item in reading.get("anchors", ()):
                bound = {"frame_index": item["index"], "pts": item["pts"], "decoded_rgb_sha256": item["rgb_sha256"]}
                require(selected.setdefault(item["index"], bound) == bound, "Two readings use different anchor pictures")
        anchors += native(folder / clip["file"], row | {"delivery_sha256": row["sha256"],
                          "proposal": {"anchors": list(selected.values())}}, "avc", rate=rate)
    return {"files": len(record["rows"]), "native_times_verified": 120 * len(record["rows"]), "verified_anchors": anchors}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "prepare", "score", "review", "pixels"))
    parser.add_argument("path", type=Path)
    parser.add_argument("--split", choices=("development", "held"))
    parser.add_argument("--previous", type=Path)
    args = parser.parse_args()
    if args.phase == "freeze":
        from scripts.video_common_sources_v89 import sha, write
        if args.previous is None:
            parser.error("--previous requires the earlier native source-selection freeze")
        write(args.path / "freeze.json", {"method": DESIGN, "previous_sha256": sha(args.previous), "sources": source_plan(args.previous)})
    elif args.phase == "review":
        print(json.dumps(review(json.loads(args.path.read_bytes())), sort_keys=True, indent=2))
    elif args.split is None:
        parser.error("--split is required to prepare, score or replay pixels")
    elif args.phase == "prepare":
        prepare(args.path, args.split)
    elif args.phase == "pixels":
        print(json.dumps(pixels(args.path, args.split), sort_keys=True))
    else:
        score(args.path, args.split)
