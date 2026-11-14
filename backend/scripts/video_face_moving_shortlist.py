# SPDX-License-Identifier: Apache-2.0
"""Compare fixed and moving face boxes for the first brief H.264 range."""

import argparse
import hashlib
import json
import os
from collections import Counter
from fractions import Fraction
from pathlib import Path

from scripts.review_video_brief_face_120_carrier import SHA, overlap, require

SCHEMA = "chipmunk.video-face-moving-shortlist/1"
ROOT = Path(__file__).resolve().parents[2]
CASES = ROOT / "docs/evaluation-panels"
PRIOR = "/private/tmp/chipmunk-avc-distinct-fresh-v1"
PLACEMENTS = (
    ((61, 67), (86, 94)),
    ((67, 75), (95, 101)),
    ((64, 70), (97, 105)),
    ((72, 80), (102, 108)),
    ((62, 68), (89, 97)),
    ((68, 76), (97, 103)),
    ((64, 70), (95, 103)),
    ((70, 78), (92, 98)),
)
ARMS = {
    "fixed_wider": (False, (5, 6, 7, 8, 9)),
    "moving": (True, (5, 6, 7)),
    "moving_wider": (True, (5, 6, 7, 8, 9)),
}
DESIGN = {
    "roles": [
        "original",
        "two_swaps",
        "tandem_two_swaps",
        "swap_then_jump",
        "light_then_swap",
        "same_source_cuts",
        "cross_recording_cuts",
    ],
    "min_gains": 3,
    "min_groups": 2,
    "max_ordinary_added": 1,
    "max_cut_added": 0,
    "max_wrong": 1,
    "route": "Only clips without an existing brief face range. Preserve all existing face ranges.",
    "point": 0.1,
    "max_edges": 16,
    "max_scheduled": 84,
}


def source_plan(prior):
    from app.detectors.video.decode import FrameTable
    from scripts.video_brief_face_avc_clock_confirmation import assignments, eligible_rate
    from scripts.video_brief_face_avc_confirmation import DATA, NATIVE, QUALIFIED
    from scripts.video_chroma_boundary_transfer import source_pin
    from scripts.video_common_sources_v89 import read
    from scripts.video_face_distinct_avc import EXCLUDE
    from scripts.video_face_short_boundary_confirmation import metadata
    from scripts.video_native_sources_v92 import ARCHIVE

    catalog = metadata()
    qualifications = {
        row["group"]: row["qualification"]
        for row in read(QUALIFIED)["assets"]
        if row["kind"] == "original_videos"
    }
    inventory = [read(path) for path in NATIVE.glob("group-*-complete.json")]
    prior_studies = [DATA / name / "freeze.json" for name in EXCLUDE] + [prior / "freeze.json"]
    excluded = {
        row["index"]
        for path in prior_studies
        for role in ("sources", "ordinary", "swaps")
        for row in read(path).get(role, ())
        if "index" in row
    }
    previous_keys = [row["group"] for row in inventory if row["index"] in excluded]
    urls = {catalog[key]["url"] for key in previous_keys}
    donors = set().union(*(catalog[key]["donors"] for key in previous_keys))
    components = {qualifications[key]["component"] for key in previous_keys}
    selected = []
    pool = sorted(inventory, key=lambda row: hashlib.sha256(f"{SCHEMA}/{row['group']}".encode()).digest())
    for slot, spans in enumerate(PLACEMENTS):
        rate = 30 if slot < 5 else Fraction(30000, 1001)
        families = ("swap_a", "swap_b") if slot % 2 == 0 else ("swap_b", "swap_a")
        for row in pool:
            index, key = row["index"], row["group"]
            meta, qualification = catalog[key], qualifications[key]
            if (
                index in excluded
                or meta["url"] in urls
                or meta["donors"] & donors
                or qualification["component"] in components
                or "child" in meta["tags"]
                or not qualification["eligible"]
                or qualification["historical_rgb_connection"]
            ):
                continue
            cases = {item["condition"]: item for item in row["cases"]}
            table = FrameTable.from_json(cases["original"]["frame_table"])
            if (table.width, table.height) not in ((1280, 720), (1920, 1080)) or eligible_rate(table) != rate:
                continue
            mapped = assignments(table)
            if any(
                FrameTable.from_json(cases[family]["frame_table"]) != table
                or not set(mapped[start:end]) <= set(cases[family]["actual_changed_frames"])
                for (start, end), family in zip(spans, families, strict=True)
            ):
                continue
            selected.append(
                {
                    **source_pin(index, ARCHIVE),
                    "placements": [list(span) for span in spans],
                    "families": list(families),
                    "assignments": mapped,
                    "native_rate": str(rate),
                }
            )
            excluded.add(index)
            urls.add(meta["url"])
            donors.update(meta["donors"])
            components.add(qualification["component"])
            break
        require(len(selected) == slot + 1, f"No disjoint source for slot {slot + 1}")
    return selected


def pin_inputs(panel, prior):
    from scripts import video_face_distinct_avc as distinct
    from scripts.video_brief_face_120_carrier import DATA
    from scripts.video_common_sources_v89 import sha

    prior_panels = [
        "video-face-distinct-avc-retained-2026-11-13.json",
        "video-face-distinct-avc-source-separated-2026-11-13.json",
    ]
    result = {name: sha(CASES / name) for name in prior_panels}
    result["prior_source_plan"] = sha(prior / "freeze.json")
    if panel == "development":
        result.update({folder: sha(DATA / folder / "media.json") for folder in distinct.PANELS})
        result["prior_media"] = sha(prior / "media.json")
    return result


def lock(panel, prior):
    from app.detectors.video import face_paired_edge, face_short_proposal
    from app.detectors.video.adapters import face_appearance
    from scripts import video_face_distinct_avc
    from scripts.video_brief_face_120_carrier import WEIGHTS
    from scripts.video_common_sources_v89 import sha

    paths = [
        Path(module.__file__)
        for module in (face_paired_edge, face_short_proposal, face_appearance, video_face_distinct_avc)
    ]
    return {
        "schema": SCHEMA,
        "panel": panel,
        "design": DESIGN,
        "weights": WEIGHTS,
        "inputs": pin_inputs(panel, prior),
        "methods": {path.name: sha(path) for path in [Path(__file__), *paths]},
        "sources": source_plan(prior) if panel == "confirmation" else [],
    }


def prepare(folder, frozen):
    from unittest.mock import patch

    from scripts import video_face_distinct_avc as distinct

    with patch.object(distinct, "fresh_plan", return_value=frozen["sources"]):
        distinct.prepare(folder, frozen)


def bank_rows(folder, panel, prior, scorer):
    from scripts import video_face_distinct_avc as distinct
    from scripts.video_brief_face_120_carrier import DATA
    from scripts.video_common_sources_v89 import read, sha

    if panel == "confirmation":
        media = read(folder / "media.json")
        require(media["freeze_sha256"] == sha(folder / "freeze.json"), "Delivery manifest is detached")
        for item in media["rows"]:
            path = folder / "media" / item["file"]
            require(sha(path) == item["sha256"], "Confirmation video changed")
            coarse, refined, first, other, bounded = distinct.stage_inputs(scorer, path, item)
            yield (
                path,
                item,
                "confirmation",
                first or bounded,
                [r for r in (coarse, refined, first or bounded, *other) if r],
            )
        return
    for name in ("retained", "source-separated"):
        ledger = read(CASES / f"video-face-distinct-avc-{name}-2026-11-13.json")
        homes = [DATA / bank for bank in distinct.PANELS] if name == "retained" else [prior]
        indexed = {
            row["sha256"]: (home / "media" / row["file"], row)
            for home in homes
            for row in read(home / "media.json")["rows"]
        }
        for row in ledger["rows"]:
            path, item = indexed[row["sha256"]]
            require(sha(path) == item["sha256"], "Retained video changed")
            require(item["truth"] == row["truth"], "Truth commitment differs")
            yield (
                path,
                item,
                row["bank"],
                row["first"],
                [r for r in (row["coarse"], row["refined"], row["first"], *row["secondary"]) if r],
            )


def measure(path, item, prior, detector, recognizer):
    import cv2

    from app.detectors.image.faces import detect_faces
    from app.detectors.video import face_appearance, face_paired_edge, face_short_proposal
    from app.detectors.video.adapters.face_appearance import _measure
    from app.detectors.video.decode import FrameTable, decode_frames, probe_frames
    from app.detectors.video.face_return import source_carrier
    from app.evidence import face_short_proposal_binding as proof

    table = probe_frames(path, cap=130, timeout_s=20)
    require(
        table == FrameTable.from_json(item["clock"])
        and proof.avc_source(table, proof.AVC_ENCODING, source_carrier(path)),
        "Video clock or profile differs",
    )
    frames, reduced, hashes = {}, {}, {}

    def read(frame, read_face):
        index = frame.index
        digest = hashlib.sha256(frame.rgb).hexdigest()
        require(
            frame.pts_verified and frame.pts == table.pts[index] and digest == item["rgb_sha256"][index],
            "Native frame differs from the saved source",
        )
        hashes[index] = digest
        if not read_face:
            return
        bgr = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)
        matches = detect_faces(bgr, detector=detector, min_size=48)
        if len(matches) == 1 and (result := _measure(frame.rgb, bgr, index, table, matches[0], recognizer)):
            vector, record = result
            frames[index] = face_appearance.unit(vector), matches[0], record

    for frame in decode_frames(path, table, None, timeout_s=40):
        read(frame, frame.index in proof.ANCHOR_SAMPLES)
        require(len(reduced) == frame.index, "Missing native picture")
        reduced[frame.index] = face_paired_edge.reduced_rgb(frame.rgb)
    require(len(reduced) == 120, "Native decode is incomplete")

    def check(*_):
        return None

    anchors = {index: record["bbox"] for index, (_, _, record) in frames.items()}
    if len(anchors) < 4:
        return None
    edges = {
        False: face_short_proposal.shortlist(reduced, list(anchors.values()), check),
        True: face_short_proposal.moving_shortlist(reduced, anchors, check),
    }
    schedules = {method: set(proof.face_schedule(values)) for method, values in edges.items()}
    need = set.union(*schedules.values()) - set(proof.ANCHOR_SAMPLES)
    if need:
        decoded = set()
        for frame in decode_frames(path, table, sorted(need), timeout_s=35):
            read(frame, True)
            decoded.add(frame.index)
        require(decoded == need, "Scheduled frames are missing")

    def excluded(start, stop):
        return any(overlap((start, stop), old) for old in prior)

    inputs = {
        method: {i: frames[i] for i in scheduled if i in frames} for method, scheduled in schedules.items()
    }
    require(
        face_short_proposal.select(edges[False], inputs[False], reduced, check, excluded) is None,
        "No-first baseline does not reproduce",
    )
    result = {}
    for name, (method, durations) in ARMS.items():
        selected = face_short_proposal.select(
            edges[method], inputs[method], reduced, check, excluded, durations=durations
        )
        choice = None
        if selected:
            (negative, start, end), support, _ = selected
            measured = inputs[method]
            middle = min(
                (i for i in measured if start <= i < end), key=lambda i: (abs(2 * i - start - end), i)
            )
            choice = {
                "frames": [start, end],
                "response": -negative,
                "pts": [table.pts[start], table.pts[end]],
                "edge_frames": [support[role]["frame_indices"] for role in ("entry", "exit")],
                "anchors": [
                    {"frame": i, "pts": table.pts[i], "rgb_sha256": hashes[i]}
                    for i in (start - 1, middle, end)
                ],
            }
        result[name] = {
            "edges": edges[method],
            "scheduled": len(schedules[method]),
            "measured": len(inputs[method]),
            "choice": choice,
        }
    return result


def configure(folder):
    from scripts import screen_video_native_background_challenge as worker
    from scripts.video_face_distinct_avc import BASE_FLAGS

    state = worker.configure(folder)
    os.environ.update(
        BASE_FLAGS, VERITAS_VIDEO_FACE_SECOND_AVC_REVIEW="false", VERITAS_VIDEO_FACE_MOVING_AVC_REVIEW="false"
    )
    return state


def run(folder, frozen, prior, state):
    import cv2

    from app.detectors.image.faces import load_yunet
    from scripts import screen_video_native_background_challenge as worker
    from scripts.video_common_sources_v89 import sha

    scorer = worker.scorer(state) if frozen["panel"] == "confirmation" else None
    cv2.setNumThreads(1)
    weights = ROOT / "models/weights"
    require(
        all(sha(weights / name) == expected for name, expected in frozen["weights"].items()),
        "Model bytes differ",
    )
    detector = load_yunet(weights / "face_detection_yunet_2023mar.onnx")
    recognizer = cv2.FaceRecognizerSF.create(str(weights / "face_recognition_sface_2021dec.onnx"), "")
    with (folder / "rows.jsonl").open("x") as output:
        for index, (path, item, bank, first, ranges) in enumerate(
            bank_rows(folder, frozen["panel"], prior, scorer), 1
        ):
            candidates = measure(path, item, ranges, detector, recognizer) if first is None else None
            row = {key: item[key] for key in ("group", "condition", "sha256", "truth")}
            row.update(
                index=index,
                bank=bank,
                first=first,
                prior=ranges,
                candidates=candidates,
                time_base=item["clock"]["time_base"],
                first_pts=item["clock"]["pts"][0],
            )
            output.write(json.dumps(row, allow_nan=False) + "\n")
            output.flush()
            print(
                index,
                row["condition"],
                [arm for arm, value in (candidates or {}).items() if value["choice"]],
                flush=True,
            )


def review(rows, panel):
    expected = 181 if panel == "development" else len(PLACEMENTS) * len(DESIGN["roles"])
    require(
        len(rows) == expected
        and [row["index"] for row in rows] == list(range(1, expected + 1))
        and len({(row["bank"], row["group"], row["condition"]) for row in rows}) == expected,
        "Missing or duplicate roster positions",
    )
    baseline = Counter()
    arms = {name: Counter() for name in ARMS}
    groups = {name: set() for name in ARMS}
    delivered = {}
    for row in rows:
        truth, prior = row["truth"], row["prior"]
        missed = [span for span in truth if not any(overlap(span, earlier) for earlier in prior)]
        require(
            SHA.fullmatch(row["sha256"]) and all(0 <= s < e < 120 for s, e in truth + prior),
            "Invalid source binding",
        )
        if row["sha256"] in delivered:
            original = delivered[row["sha256"]]
            require(
                panel == "development"
                and not truth
                and not original["truth"]
                and row["candidates"] == original["candidates"]
                and prior == original["prior"],
                "The same delivered bytes have inconsistent outcomes",
            )
            baseline["repeated_ordinary_deliveries"] += 1
            continue
        delivered[row["sha256"]] = row
        baseline["locations"] += len(truth) - len(missed)
        baseline["truth_spans"] += len(truth)
        baseline["ordinary"] += not truth
        baseline["ordinary_with_range"] += not truth and bool(prior)
        baseline["routed_edits"] += row["first"] is None and bool(truth)
        baseline["routed_ordinary"] += row["first"] is None and not truth
        candidates = row["candidates"]
        if candidates is None:
            continue
        require(row["first"] is None and set(candidates) == ARMS.keys(), "Invalid candidate routing")
        base = Fraction(row["time_base"])
        for name, method in candidates.items():
            require(
                len(method["edges"]) <= DESIGN["max_edges"]
                and 0 <= method["measured"] <= method["scheduled"] <= DESIGN["max_scheduled"],
                "Exceeded the measurement budget",
            )
            choice = method["choice"]
            if choice is None:
                continue
            start, stop = choice["frames"]
            anchors = choice["anchors"]
            require(
                choice["response"] > DESIGN["point"]
                and stop - start in ARMS[name][1]
                and {start, stop} <= set(method["edges"])
                and not any(overlap((start, stop), old) for old in prior)
                and [list(range(edge - 3, edge + 3)) for edge in (start, stop)] == choice["edge_frames"]
                and anchors[0]["frame"] == start - 1
                and start <= anchors[1]["frame"] < stop
                and anchors[2]["frame"] == stop,
                "Invalid chosen range",
            )
            require(
                all(
                    (tick - row["first_pts"]) * base == Fraction(i, 30)
                    for i, tick in zip((start, stop), choice["pts"], strict=True)
                )
                and all(
                    (a["pts"] - row["first_pts"]) * base == Fraction(a["frame"], 30)
                    and SHA.fullmatch(a["rgb_sha256"])
                    for a in anchors
                ),
                "Invalid native timestamp or picture",
            )
            gains = [span for span in missed if overlap(span, (start, stop))]
            count = arms[name]
            count["added"] += len(gains)
            count["exact"] += [start, stop] in gains
            count["wrong"] += bool(truth) and not gains
            count["ordinary_added"] += not truth
            count["cut_added"] += not truth and (
                "cut" in row["condition"] or "person_switch" in row["condition"]
            )
            if gains:
                groups[name].add((row["bank"], row["group"]))
    results = {}
    for name, count in arms.items():
        count["gained_groups"] = len(groups[name])
        results[name] = {
            key: count[key]
            for key in ("added", "exact", "wrong", "ordinary_added", "cut_added", "gained_groups")
        }
        results[name]["passes"] = (
            count["added"] >= DESIGN["min_gains"]
            and count["gained_groups"] >= DESIGN["min_groups"]
            and all(
                count[key] <= DESIGN[limit]
                for key, limit in (
                    ("wrong", "max_wrong"),
                    ("ordinary_added", "max_ordinary_added"),
                    ("cut_added", "max_cut_added"),
                )
            )
        )
    return {"baseline": dict(baseline), "arms": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "prepare", "run", "review"))
    parser.add_argument("path", type=Path)
    parser.add_argument("--panel", choices=("development", "confirmation"), default="development")
    parser.add_argument("--prior", type=Path, default=Path(PRIOR))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    folder = args.path
    if args.phase == "freeze":
        frozen = lock(args.panel, args.prior)
        folder.mkdir(mode=0o700)
        (folder / "freeze.json").write_text(json.dumps(frozen, indent=2) + "\n")
        return
    if folder.is_file():
        receipt = json.loads(folder.read_text())
        frozen, rows = receipt["freeze"], receipt["rows"]
    else:
        frozen = json.loads((folder / "freeze.json").read_text())
        if args.phase in {"prepare", "run"}:
            state = configure(folder) if args.phase == "run" else None
            require(frozen == lock(frozen["panel"], args.prior), "Frozen code or inputs changed")
            if args.phase == "prepare":
                prepare(folder, frozen)
            else:
                run(folder, frozen, args.prior, state)
            return
        rows = [json.loads(line) for line in (folder / "rows.jsonl").read_text().splitlines()]
    require(frozen["schema"] == SCHEMA and frozen["design"] == DESIGN, "Unknown protocol")
    result = review(rows, frozen["panel"])
    if folder.is_file():
        require(result == receipt["result"], "Public results differ")
    if args.output:
        with args.output.open("x") as output:
            del frozen["sources"]
            json.dump({"freeze": frozen, "rows": rows, "result": result}, output, indent=2, allow_nan=False)
            output.write("\n")
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
