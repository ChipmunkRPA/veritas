# SPDX-License-Identifier: Apache-2.0
"""Measure synchronized changes in textured tiles outside a face-review region."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from scripts import video_face_paired_edge_transfer as frozen

ROOT = frozen.ROOT
GRID = frozen.DATA / "video-face-hd-retry-scene-challenge-2026-11-12"
CREMA = frozen.DATA / "video-face-near-head-transfer-2026-11-12-v2"
SCHEMA = "chipmunk.video-spatial-cut-context-transfer/1"
WIDTH, HEIGHT, TILE = 320, 180, 12
FLOORS = [0.10, 0.20, 0.30]
SPEC = {
    "canvas": [WIDTH, HEIGHT],
    "tile": TILE,
    "edge_neighborhood": 12,
    "durations": [5, 24],
    "face_padding_in_box_widths": 0.22,
    "maximum_face_detector_inputs": 3,
    "tile_search_radius": 3,
    "points": FLOORS,
    "minimum_tile_texture": 2.5,
    "minimum_role_tiles": 5,
    "minimum_signed_tile_change": 0.20,
    "minimum_temporary_texture_tiles": 4,
    "minimum_scored_sector_groups": 2,
    "stable_context_around_edges": True,
    "rule": "Outside-face texture changes at both edges; median neighboring motion and local tone normalized. Require stable flanks and at least 5 independently changing tiles across 2 spatial sectors.",
    "gate": {
        "minimum_cut_explanations": 8,
        "minimum_source_groups": 4,
        "maximum_wrong_swap_explanations": 0,
        "maximum_wrong_ordinary_explanations": 1,
        "minimum_cut_route_coverage": 0.8,
    },
}
read, save, sha, require = frozen.read, frozen.save, frozen.sha, frozen.require


def freeze(selection, held):
    files = {
        "method": Path(__file__),
        "selection_media": selection / "media.json",
        "selection_receipt": selection / "receipt.json",
        "held_media": held / "media.json",
        "held_receipt": held / "public.json",
        "grid_sources": GRID / "media.json",
        "crema_sources": CREMA / "media.json",
    }
    require(
        read(selection / "receipt.json")["media_sha256"] == sha(selection / "media.json"),
        "Selection input changed",
    )
    require(read(held / "public.json")["held_media_sha256"] == sha(held / "media.json"), "Held input changed")
    return {
        "schema": SCHEMA,
        "spec": SPEC,
        "pins": {key: sha(path) for key, path in files.items()},
        "yunet_sha256": frozen.WEIGHTS["face_detection_yunet_2023mar.onnx"],
    }


def frames(path, row, needed, face_indices, detector):
    import cv2
    import numpy as np

    from app.detectors.image.faces import detect_faces
    from app.detectors.video.decode import FrameTable, decode_frames
    from app.evidence.face_paired_edge_binding import supported

    require(sha(path) == row["sha256"], "Source delivery changed")
    table = FrameTable.from_json(row["clock"])
    require(supported(table, ("hevc", "yuv420p", "hvc1"), "mp4"), "Source frame clock or geometry changed")
    gray, boxes = {}, []
    for frame in decode_frames(path, table, needed, timeout_s=18):
        from hashlib import sha256

        require(
            frame.pts_verified
            and frame.pts == table.pts[frame.index]
            and sha256(frame.rgb).hexdigest() == row["rgb_sha256"][frame.index],
            "Decoded picture differs from the source ledger",
        )
        image = cv2.resize(frame.rgb, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
        gray[frame.index] = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        if frame.index in face_indices:
            faces = detect_faces(cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR), detector=detector, min_size=48)
            if len(faces) == 1:
                boxes.append(faces[0].box())
    require(set(gray) == set(needed), "Missing decoded native evidence")
    return table, gray, (np.median(boxes, axis=0) / 4 if len(boxes) >= 2 else None)


def layout(box):
    x, y, w, h = box
    cx, cy = x + w / 2, y + h / 2
    columns = range(TILE, WIDTH - TILE, TILE)
    rows = range(TILE, HEIGHT - TILE, TILE)
    return [
        (a, b, (a + TILE / 2 >= cx, b + TILE / 2 >= cy))
        for b in rows
        for a in columns
        if (a + TILE < x - 0.22 * w or a > x + 1.22 * w or b + TILE < y - 0.22 * h or b > y + 1.22 * h)
        and abs(a + TILE / 2 - cx) <= max(46.0, w * 1.45)
        and abs(b + TILE / 2 - cy) <= max(44.0, h * 1.45)
    ]


def local_change(reference, moving, a, b):
    import cv2

    template = reference[b : b + TILE, a : a + TILE]
    search = moving[b - 3 : b + TILE + 3, a - 3 : a + TILE + 3]
    if template.std() < SPEC["minimum_tile_texture"]:
        return None
    scores = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
    return max(0.0, 1.0 - float(scores.max()))


def edge_features(i, pictures, tiles):
    import numpy as np

    baseline = [(i - 5, i - 4), (i - 4, i - 3), (i + 3, i + 4), (i + 4, i + 5)]
    output = {}
    for k, (x, y, sector) in enumerate(tiles):
        context = [local_change(pictures[a], pictures[b], x, y) for a, b in baseline]
        change = local_change(pictures[i - 1], pictures[i], x, y)
        if change is not None and all(v is not None for v in context):
            residual = float(change - np.median(context))
            if residual >= SPEC["minimum_signed_tile_change"]:
                output[k] = residual, sector
    return output


def paired_edges(pictures, box, interval, check=lambda _: None):
    a, b = interval
    tiles = layout(box)
    start = range(max(6, a - 12), min(114, a + 12) + 1)
    end = range(max(6, b - 12), min(114, b + 12) + 1)
    scores = {}
    for i in sorted(set(start) | set(end)):
        check("outside-face cut context")
        scores[i] = edge_features(i, pictures, tiles)
    strongest = None
    for left in start:
        for right in end:
            if not 5 <= right - left <= 24 or 2 * (min(right, b) - max(left, a)) < min(right - left, b - a):
                continue
            first, last = scores[left], scores[right]
            common = sorted(first.keys() & last.keys())
            if (
                min(len(first), len(last)) < 5
                or len(common) < SPEC["minimum_temporary_texture_tiles"]
                or len({first[k][1] for k in common}) < SPEC["minimum_scored_sector_groups"]
            ):
                continue
            persistent = []
            for index in common:
                x, y, _ = tiles[index]
                stable = local_change(pictures[left - 1], pictures[right], x, y)
                if stable is not None and stable <= 0.16:
                    persistent.append((min(first[index][0], last[index][0]), index))
            if len(persistent) < 5 or len({first[k][1] for _, k in persistent}) < 2:
                continue
            response = sorted((v for v, _ in persistent), reverse=True)[4]
            rank = response, len(persistent), -left, -right
            if strongest is None or rank > strongest[0]:
                strongest = (
                    rank,
                    {
                        "interval": [left, right],
                        "score": response,
                        "shared_tiles": len(persistent),
                        "entry_tiles": len(first),
                        "exit_tiles": len(last),
                        "sectors": len({first[k][1] for _, k in persistent}),
                    },
                )
    return strongest[1] if strongest else None, len(tiles)


def evaluate(row, path, spans, authored, detector):
    if not spans:
        return {"status": "not_routed"}
    a, b = spans[0]
    if not 5 <= b - a <= 42:
        return {"status": "outside_range_scope"}
    roles = [max(0, a - 4), (a + b) // 2, min(119, b + 4)]
    needed = sorted(set(range(max(0, a - 17), min(120, b + 18))) | set(roles))
    table, pictures, box = frames(path, row, needed, roles, detector)
    if box is None:
        return {"status": "unassessed_face_context"}
    selected, tiles = paired_edges(pictures, box, spans[0])
    if selected is not None:
        start, end = selected["interval"]
        selected.update(
            pts=[table.pts[start], table.pts[end]],
            anchors=[
                {"index": i, "pts": table.pts[i], "sha256": row["rgb_sha256"][i]}
                for i in (start - 1, start, (start + end) // 2, end - 1, end)
            ],
        )
    return {
        "status": "measured" if tiles >= 5 else "unassessed_tiles",
        "coarse": spans[0],
        "original_ranges": spans,
        "authored_cut": authored,
        "tiles": tiles,
        "native_reads": len(needed),
        "result": selected,
    }


def run(folder, selection, held, phase):
    import cv2

    from app.detectors.image.faces import load_yunet

    require(read(folder / "freeze.json") == freeze(selection, held), "Frozen method or input roster changed")
    cv2.setNumThreads(1)
    weights = ROOT / "models/weights/face_detection_yunet_2023mar.onnx"
    require(sha(weights) == frozen.WEIGHTS[weights.name], "Existing face detector changed")
    detector = load_yunet(weights)
    if phase == "selection":
        source = selection
        baseline = {r["sha256"]: r["prior"] for r in read(source / "receipt.json")["rows"]}
        controls = {
            r["sha256"]: r["truth"][0]
            for r in read(CREMA / "media.json")["rows"]
            if r["condition"] in {"cutaway", "tandem_cutaway"}
        }
    else:
        require((folder / "point.json").is_file(), "Selection point must be frozen before held run")
        source = held
        baseline = read(source / "public.json")["baseline"]["held"]
        controls = {
            r["sha256"]: r["authored_span"]
            for r in read(GRID / "media.json")["rows"]
            if r["condition"] in {"person_switch", "tandem_person_switch"}
        }
    results = []
    for row in read(source / "media.json")["rows"]:
        observed = evaluate(
            row,
            source / row["file"],
            baseline[row["sha256"]],
            controls.get(row.get("input_h264_sha256", row.get("h264_sha256"))),
            detector,
        )
        results.append({**{k: row[k] for k in ("group", "bank", "condition", "sha256", "truth")}, **observed})
    save(folder / f"{phase}.json", {"freeze_sha256": sha(folder / "freeze.json"), "rows": results})
    if phase == "selection":
        scores = {point: tally(results, point) for point in FLOORS}
        eligible = [
            point
            for point, result in scores.items()
            if not result["wrong_swaps"] and not result["wrong_ordinary"]
        ]
        point = max(eligible, key=lambda p: (scores[p]["cut_exact"], p)) if eligible else 1.0
        save(
            folder / "point.json",
            {
                "selection_sha256": sha(folder / "selection.json"),
                "point": point,
                "selection": [{"point": p, "metrics": scores[p]} for p in FLOORS],
            },
        )


def tally(rows, point):
    summary, groups = Counter(), set()
    for row in rows:
        scope, authored, value = row["status"], row.get("authored_cut"), row.get("result")
        # Routed state is recorded independently of whether useful tiles could be measured.
        label = row["condition"] in {"person_switch", "tandem_person_switch", "cutaway", "tandem_cutaway"}
        flagged = bool(value and value["score"] >= point)
        exact = bool(
            flagged
            and authored
            and all(abs(a - b) <= 1 for a, b in zip(authored, value["interval"], strict=True))
        )
        if exact:
            groups.add((row["bank"], row["group"]))
        summary.update(
            files=1,
            routed=scope != "not_routed",
            measured=scope == "measured",
            cut_files=label,
            cut_routed=label and scope != "not_routed",
            cut_measured=label and scope == "measured",
            cut_exact=exact,
            cut_wrong=bool(flagged and label and not exact),
            wrong_swaps=bool(flagged and row["truth"]),
            wrong_ordinary=bool(flagged and not (row["truth"] or label)),
            swap_routed=bool(row["truth"] and scope != "not_routed"),
        )
    return {**dict(summary), "source_groups_with_cut_explanation": len(groups)}


def report(folder):
    lock, selection, held, point = (
        read(folder / f"{name}.json") for name in ("freeze", "selection", "held", "point")
    )
    require(
        selection["freeze_sha256"] == held["freeze_sha256"] == sha(folder / "freeze.json")
        and point["selection_sha256"] == sha(folder / "selection.json"),
        "Study records changed",
    )
    metrics = tally(held["rows"], point["point"])
    gate = {
        "enough_explanations": metrics["cut_exact"] >= 8,
        "enough_groups": metrics["source_groups_with_cut_explanation"] >= 4,
        "swap_preservation": metrics["wrong_swaps"] == 0,
        "ordinary_preservation": metrics["wrong_ordinary"] <= 1,
        "matched_edges": metrics["cut_wrong"] == 0,
        "coverage": metrics["cut_measured"] >= 0.8 * metrics["cut_routed"],
    }
    save(
        folder / "receipt.json",
        {
            "freeze": lock,
            "point": point,
            "held": held["rows"],
            "metrics": metrics,
            "gate": gate,
            "passed": all(gate.values()),
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "selection", "held", "report"))
    parser.add_argument("folder", type=Path)
    parser.add_argument("selection", type=Path)
    parser.add_argument("held", type=Path)
    args = parser.parse_args()
    if args.phase == "freeze":
        args.folder.mkdir(mode=0o700)
        save(args.folder / "freeze.json", freeze(args.selection, args.held))
    elif args.phase == "report":
        report(args.folder)
    else:
        run(args.folder, args.selection, args.held, args.phase)
