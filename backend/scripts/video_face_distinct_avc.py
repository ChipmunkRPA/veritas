# SPDX-License-Identifier: Apache-2.0
"""Replay a second distinct face-change range on qualified four-second AVC files."""

import argparse
import hashlib
import json
import os
from collections import Counter
from fractions import Fraction
from pathlib import Path
from unittest.mock import patch

from scripts.review_video_brief_face_120_carrier import SHA, overlap, require

SCHEMA = "chipmunk.video-face-distinct-avc/1"
PANELS = (
    "video-face-two-returns-hd-2026-11-12",
    "video-face-hd-retry-scene-challenge-2026-11-12",
)
EXCLUDE = (*PANELS, "video-face-hd-sparse-retry-2026-11-12-v2", "video-face-two-returns-2026-11-12")
PLACEMENTS = (((63, 69), (93, 101)), ((69, 77), (96, 102)), ((66, 72), (99, 107)), ((71, 79), (91, 97)))
ROLES = (
    "original",
    "two_swaps",
    "tandem_two_swaps",
    "swap_then_jump",
    "light_then_swap",
    "same_source_cuts",
    "cross_recording_cuts",
)
BASE_FLAGS = {
    "VERITAS_VIDEO_FACE_SECOND_RETURN_REVIEW": "true",
    "VERITAS_VIDEO_FACE_DETAIL_REPEAT_REVIEW": "true",
    "VERITAS_VIDEO_FACE_PAIRED_EDGE_REFINEMENT": "true",
    "VERITAS_VIDEO_FACE_SPATIAL_CUT_CONTEXT": "true",
    "VERITAS_VIDEO_FACE_UNROUTED_SHORT_REVIEW": "false",
    "VERITAS_VIDEO_FACE_SHORT_CARRIER_REVIEW": "false",
    "VERITAS_VIDEO_FACE_SHORT_WINDOW_REVIEW": "false",
    "VERITAS_VIDEO_FACE_SHORT_AVC_REVIEW": "true",
}
DESIGN = {
    "gap_frames": 3,
    "point": 0.10,
    "minimum_added_locations": 3,
    "minimum_gained_groups": 2,
    "maximum_wrong_ranges": 1,
    "maximum_new_control_ranges": 1,
    "maximum_genuine_speaker_or_cut_ranges": 0,
    "durations": [list(range(5, 8)), list(range(5, 10))],
}
ARMS = ("five_to_seven", "five_to_nine")


def fresh_plan():
    from app.detectors.video.decode import FrameTable
    from scripts.video_brief_face_avc_clock_confirmation import assignments, eligible_rate
    from scripts.video_brief_face_avc_confirmation import DATA, NATIVE, QUALIFIED
    from scripts.video_chroma_boundary_transfer import source_pin
    from scripts.video_common_sources_v89 import read
    from scripts.video_face_short_boundary_confirmation import metadata
    from scripts.video_native_sources_v92 import ARCHIVE

    catalog = metadata()
    qualification = {
        row["group"]: row["qualification"]
        for row in read(QUALIFIED)["assets"]
        if row["kind"] == "original_videos"
    }
    inventory = [read(path) for path in sorted(NATIVE.glob("group-*-complete.json"))]
    used = set()
    for name in EXCLUDE:
        freeze = read(DATA / name / "freeze.json")
        for key in ("sources", "ordinary", "swaps"):
            used.update(row["index"] for row in freeze.get(key, ()) if "index" in row)
    keys = [row["group"] for row in inventory if row["index"] in used]
    urls = {catalog[key]["url"] for key in keys}
    donors = set().union(*(catalog[key]["donors"] for key in keys))
    components = {qualification[key]["component"] for key in keys}
    picked = []
    order = sorted(inventory, key=lambda row: hashlib.sha256(f"{SCHEMA}:{row['group']}".encode()).digest())
    for slot, spans in enumerate((*PLACEMENTS, *PLACEMENTS)):
        expected_rate = 30 if slot < 4 else Fraction(30000, 1001)
        for row in order:
            index, key = row["index"], row["group"]
            meta, qualify = catalog[key], qualification[key]
            if (
                index in used
                or meta["url"] in urls
                or meta["donors"] & donors
                or qualify["component"] in components
                or "child" in meta["tags"]
                or not qualify["eligible"]
                or qualify["historical_rgb_connection"]
            ):
                continue
            cases = {item["condition"]: item for item in row["cases"]}
            table = FrameTable.from_json(cases["original"]["frame_table"])
            if (table.width, table.height) not in ((1280, 720), (1920, 1080)) or eligible_rate(
                table
            ) != expected_rate:
                continue
            native = assignments(table)
            families = ("swap_a", "swap_b") if slot % 2 == 0 else ("swap_b", "swap_a")
            if any(
                FrameTable.from_json(cases[family]["frame_table"]) != table
                or not set(native[a:b]) <= set(cases[family]["actual_changed_frames"])
                for (a, b), family in zip(spans, families, strict=True)
            ):
                continue
            picked.append(
                {
                    **source_pin(index, ARCHIVE),
                    "placements": [list(span) for span in spans],
                    "families": list(families),
                    "assignments": native,
                    "native_rate": str(expected_rate),
                }
            )
            used.add(index)
            urls.add(meta["url"])
            donors.update(meta["donors"])
            components.add(qualify["component"])
            break
        require(len(picked) == slot + 1, f"No source for position {slot + 1} at {expected_rate} fps")
    return picked


def freeze(folder, panel):
    from app.detectors.video import face_paired_edge, face_short_proposal
    from app.detectors.video.adapters import face_appearance
    from scripts import video_brief_face_120_carrier as carrier
    from scripts.video_common_sources_v89 import sha, write

    folder.mkdir(mode=0o700)
    files = [
        Path(__file__),
        *(Path(m.__file__) for m in (face_paired_edge, face_short_proposal, face_appearance, carrier)),
    ]
    record = {
        "schema": SCHEMA,
        "panel": panel,
        "design": DESIGN,
        "models": carrier.WEIGHTS,
        "flags": BASE_FLAGS,
        "code": {path.name: sha(path) for path in files},
    }
    if panel == "retained":
        record["inputs"] = [
            {name: sha(carrier.DATA / bank / name) for name in ("media.json", "held.jsonl")}
            for bank in PANELS
        ]
    else:
        record["sources"] = fresh_plan()
    write(folder / "freeze.json", record)


def prepare(folder, frozen):
    import cv2
    import numpy as np

    from scripts import video_face_720p30_transfer as hd
    from scripts.video_brief_face_avc_clock_confirmation import native
    from scripts.video_common_sources_v89 import sha, write
    from scripts.video_face_two_returns_hd import recipe

    cv2.setNumThreads(1)
    require(frozen["sources"] == fresh_plan(), "Fresh source plan differs")
    media = folder / "media"
    media.mkdir(mode=0o700)
    rows = []
    for group, pin in enumerate(frozen["sources"], 1):
        spans = pin["placements"]
        host = native(pin, "original", range(120))
        first, second = (
            native(pin, role, range(*span)) for role, span in zip(pin["families"], spans, strict=True)
        )
        guest = native(
            frozen["sources"][(group - 1) ^ 1], "original", [i for span in spans for i in range(*span)]
        )
        for role in ROLES:
            path = media / f"g{group}-{role}.mp4"
            replacements, truth = recipe(host, first, second, guest, spans, role)
            if role == "tandem_two_swaps":
                hd.authored(path, source=media / f"g{group}-two_swaps.mp4", crf=35)
            else:
                require(
                    all(not np.array_equal(host[i], picture) for i, picture in replacements.items()),
                    "An authored edit did not change the source picture",
                )
                hd.authored(path, pictures=(replacements.get(i, host[i]) for i in range(120)))
            rows.append(hd.describe(path, group, role, truth))
        print("prepared", group, flush=True)
    write(folder / "media.json", {"freeze_sha256": sha(folder / "freeze.json"), "rows": rows})


def retained_inputs():
    from scripts import video_brief_face_120_carrier as carrier
    from scripts.video_common_sources_v89 import read
    from scripts.video_face_return_eligibility import interval

    for bank in PANELS:
        root = carrier.DATA / bank
        archived = {
            row["sha256"]: row for row in map(json.loads, (root / "held.jsonl").read_text().splitlines())
        }
        for row in read(root / "media.json")["rows"]:
            saved = archived[row["sha256"]]
            details = saved.get("baseline", saved.get("before"))
            pts = row["clock"]["pts"]
            coarse, refined, primary = [
                list(interval(details[key], pts)) if details[key] else None
                for key in ("coarse", "boundary_refinement", "short_range")
            ]
            secondary = (
                ([saved["second"]["range"]["frames"]] if saved["second"]["range"] else [])
                if "second" in saved
                else [list(interval(item, pts)) for item in saved["incumbent_second"]]
            )
            yield root / "media" / row["file"], row, bank, (coarse, refined, primary, secondary)


def stage_inputs(scorer, path, row):
    from app.detectors.video.adapters import face_appearance as adapter
    from scripts.video_face_return_eligibility import interval

    saved = []
    original = adapter._with_short_proposal

    def capture(result, *args):
        details = result.details
        pts = row["clock"]["pts"]
        coarse, refined, primary = [
            list(interval(details[key], pts)) if details.get(key) else None
            for key in ("face_sequence", "face_boundary", "face_return")
        ]
        secondary = (
            [list(interval(details["face_return_secondary"], pts))]
            if details.get("face_return_secondary")
            else []
        )
        final = original(result, *args)
        short = final.details.get("face_return")
        served_short = list(interval(short, pts)) if short and primary is None else None
        saved.append((coarse, refined, primary, secondary, served_short))
        return final

    with patch.object(adapter, "_with_short_proposal", capture):
        scorer(path, row)
    require(len(saved) <= 1, "The incumbent stage supplied duplicate readings")
    return saved[0] if saved else (None, None, None, [], None)


def measure(path, row, detector, recognizer, baseline):
    import cv2

    from app.detectors.image.faces import detect_faces
    from app.detectors.video import face_appearance, face_paired_edge, face_short_proposal
    from app.detectors.video.adapters.face_appearance import _measure
    from app.detectors.video.decode import FrameTable, decode_frames, probe_frames
    from app.detectors.video.face_return import source_carrier
    from app.evidence import face_short_proposal_binding as proof

    table = probe_frames(path, cap=130, timeout_s=20)
    require(
        table == FrameTable.from_json(row["clock"])
        and proof.avc_source(table, proof.AVC_ENCODING, source_carrier(path)),
        "The delivered source has a different clock or profile",
    )
    coarse, refined, primary, secondary = baseline[:4]
    reduced, faces = {}, {}

    def verify(frame):
        index = frame.index
        require(
            frame.pts_verified
            and frame.pts == table.pts[index]
            and hashlib.sha256(frame.rgb).hexdigest() == row["rgb_sha256"][index],
            "Native picture differs",
        )

    def face(frame):
        bgr = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)
        found = detect_faces(bgr, detector=detector, min_size=48)
        if len(found) == 1 and (sample := _measure(frame.rgb, bgr, frame.index, table, found[0], recognizer)):
            vector, record = sample
            faces[frame.index] = face_appearance.unit(vector), found[0], record

    for frame in decode_frames(path, table, None, timeout_s=35):
        verify(frame)
        require(frame.index == len(reduced), "Missing decoded picture")
        reduced[frame.index] = face_paired_edge.reduced_rgb(frame.rgb)
        if frame.index in proof.ANCHOR_SAMPLES:
            face(frame)
    require(len(reduced) == 120, "Incomplete native frame sequence")
    edges = face_short_proposal.shortlist(
        reduced, [record["bbox"] for _, _, record in faces.values()], lambda *_: None
    )
    schedule = proof.face_schedule(edges)
    requested = set(schedule) - set(proof.ANCHOR_SAMPLES)
    if requested:
        observed = set()
        for frame in decode_frames(path, table, sorted(requested), timeout_s=35):
            verify(frame)
            observed.add(frame.index)
            face(frame)
        require(observed == requested, "Missing scheduled face pictures")

    def pack(start, end, score):
        middle = min((i for i in faces if start <= i < end), key=lambda i: (abs(2 * i - start - end), i))
        return {
            "frames": [start, end],
            "response": score,
            "pts": [table.pts[start], table.pts[end]],
            "anchors": [
                {
                    key: faces[i][2][key]
                    for key in ("frame_index", "pts", "decoded_rgb_sha256", "aligned_rgb_sha256")
                }
                for i in (start - 1, middle, end)
            ],
        }

    def incumbent_excluded(start, end):
        return any(old and overlap((start, end), old) for old in (coarse, refined))

    def check(*_):
        return None

    selected = (
        face_short_proposal.select(edges, faces, reduced, check, incumbent_excluded)
        if primary is None
        else None
    )
    short = None
    if selected:
        (negative, start, end), _, _ = selected
        short = pack(start, end, -negative)
    if len(baseline) == 5:
        require(
            (short["frames"] if short else None) == baseline[4],
            "Production and replay disagree on the first bounded range",
        )
    first = primary or (short["frames"] if short else None)
    extras = dict.fromkeys(ARMS)
    if first and not secondary:
        measured_edges = face_paired_edge.appearance_edges(edges, faces, reduced, check)
        for arm, durations in zip(ARMS, DESIGN["durations"], strict=True):
            candidate = []
            for start, entry in measured_edges.items():
                for duration in durations:
                    end = start + duration
                    if end not in measured_edges or incumbent_excluded(start, end):
                        continue
                    if max(first[0] - end, start - first[1]) < DESIGN["gap_frames"]:
                        continue
                    score, _, _ = face_paired_edge.pair_response(entry, measured_edges[end])
                    if score > DESIGN["point"]:
                        candidate.append((-score, start, end))
            if candidate:
                negative, start, end = min(candidate)
                extras[arm] = pack(start, end, -negative)
    return {
        "coarse": coarse,
        "refined": refined,
        "first": first,
        "secondary": secondary,
        "bounded_first": short,
        "extras": extras,
        "edges": edges,
        "scheduled": len(schedule),
        "measured": len(faces),
        "time_base": table.time_base,
        "first_pts": table.pts[0],
    }


def run(folder, frozen):
    from scripts import screen_video_native_background_challenge as worker
    from scripts import video_brief_face_120_carrier as carrier
    from scripts.video_common_sources_v89 import read, sha

    state = worker.configure(folder)
    os.environ.update(BASE_FLAGS)

    import cv2

    from app.detectors.image.faces import load_yunet

    weights = carrier.ROOT / "models/weights"
    require(
        all(sha(weights / name) == digest for name, digest in frozen["models"].items()), "Model bytes differ"
    )
    detector = load_yunet(weights / "face_detection_yunet_2023mar.onnx")
    recognizer = cv2.FaceRecognizerSF.create(str(weights / "face_recognition_sface_2021dec.onnx"), "")
    cv2.setNumThreads(1)
    if frozen["panel"] == "fresh":
        media = read(folder / "media.json")
        require(media["freeze_sha256"] == sha(folder / "freeze.json"), "Delivery binding differs")
        scorer = worker.scorer(state)
        inputs = ((folder / "media" / row["file"], row, "source_separated", None) for row in media["rows"])
    else:
        require(
            frozen["inputs"]
            == [
                {name: sha(carrier.DATA / bank / name) for name in ("media.json", "held.jsonl")}
                for bank in PANELS
            ],
            "Retained input manifests differ",
        )
        inputs = retained_inputs()
    with (folder / "rows.jsonl").open("x") as output:
        for count, (path, row, bank, earlier) in enumerate(inputs, 1):
            require(sha(path) == row["sha256"], "The source bytes differ")
            if earlier is None:
                earlier = stage_inputs(scorer, path, row)
            measured = measure(path, row, detector, recognizer, earlier)
            item = {key: row[key] for key in ("sha256", "group", "condition", "truth")}
            item.update(index=count, bank=bank, **measured)
            output.write(json.dumps(item, allow_nan=False) + "\n")
            output.flush()
            print(
                count,
                row["condition"],
                measured["first"],
                [value["frames"] if value else None for value in measured["extras"].values()],
                flush=True,
            )


def review(rows, panel):
    expected = 125 if panel == "retained" else 8 * len(ROLES)
    require(
        len(rows) == expected
        and [row["index"] for row in rows] == list(range(1, expected + 1))
        and len({row["sha256"] for row in rows}) == expected,
        "Incomplete or duplicate study entries",
    )
    totals, outcomes = Counter(), {arm: Counter() for arm in ARMS}
    groups = {arm: set() for arm in ARMS}
    for row in rows:
        require(
            SHA.fullmatch(row["sha256"]) and 0 <= row["measured"] <= row["scheduled"] <= 84,
            "Invalid native source or input budget",
        )
        current = [item for item in (row["coarse"], row["refined"], row["first"], *row["secondary"]) if item]
        truth = row["truth"]
        hits = [span for span in truth if any(overlap(span, found) for found in current)]
        totals["edited"] += bool(truth)
        totals["ordinary"] += not truth
        totals["ordinary_with_current_ranges"] += not truth and bool(current)
        totals["two_edits_known"] += len(truth) == 2
        totals["two_edits_currently_located"] += len(hits) == 2
        totals["routed"] += row["first"] is not None and not row["secondary"]
        totals["current_useful"] += len(hits)
        totals["known_edits"] += len(truth)
        for arm, allowed in zip(ARMS, DESIGN["durations"], strict=True):
            result = row["extras"][arm]
            if result is None:
                continue
            start, end = result["frames"]
            first = row["first"]
            require(
                first is not None
                and not row["secondary"]
                and end - start in allowed
                and start in row["edges"]
                and end in row["edges"]
                and max(first[0] - end, start - first[1]) >= DESIGN["gap_frames"]
                and result["response"] > DESIGN["point"],
                "Invalid candidate conditions",
            )
            base = Fraction(row["time_base"])
            records = [(start, result["pts"][0]), (end, result["pts"][1])]
            anchors = result["anchors"]
            require(
                [anchors[0]["frame_index"], anchors[2]["frame_index"]] == [start - 1, end]
                and start <= anchors[1]["frame_index"] < end,
                "Wrong before/during/after roles",
            )
            records.extend((anchor["frame_index"], anchor["pts"]) for anchor in anchors)
            require(
                all((pts - row["first_pts"]) * base == Fraction(i, 30) for i, pts in records)
                and all(
                    SHA.fullmatch(anchor[key])
                    for anchor in anchors
                    for key in ("decoded_rgb_sha256", "aligned_rgb_sha256")
                ),
                "Bad native frame exhibits",
            )
            extra = [span for span in truth if span not in hits and overlap((start, end), span)]
            out = outcomes[arm]
            out["selected"] += 1
            out["added_locations"] += len(extra)
            out["wrong_ranges"] += bool(truth) and not extra
            out["new_control_ranges"] += not truth
            out["new_genuine_speaker_or_cut_ranges"] += not truth and (
                "cut" in row["condition"] or "person_switch" in row["condition"]
            )
            out["new_complete_dual"] += len(hits) < 2 and len(hits) + len(extra) == 2
            out["exact_additions"] += list((start, end)) in extra
            if extra:
                groups[arm].add((row["bank"], row["group"]))
    summary = {}
    for arm, out in outcomes.items():
        out["gained_groups"] = len(groups[arm])
        summary[arm] = {
            "counts": {
                key: out[key]
                for key in (
                    "selected",
                    "added_locations",
                    "gained_groups",
                    "wrong_ranges",
                    "new_control_ranges",
                    "new_genuine_speaker_or_cut_ranges",
                    "new_complete_dual",
                    "exact_additions",
                )
            },
            "gate": out["added_locations"] >= DESIGN["minimum_added_locations"]
            and out["gained_groups"] >= DESIGN["minimum_gained_groups"]
            and out["wrong_ranges"] <= DESIGN["maximum_wrong_ranges"]
            and out["new_control_ranges"] <= DESIGN["maximum_new_control_ranges"]
            and out["new_genuine_speaker_or_cut_ranges"] <= DESIGN["maximum_genuine_speaker_or_cut_ranges"],
        }
    return {"baseline": dict(totals), "candidates": summary}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "prepare", "run", "review"))
    parser.add_argument("path", type=Path)
    parser.add_argument("--panel", choices=("retained", "fresh"), default="retained")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.phase == "freeze":
        freeze(args.path, args.panel)
        return
    if args.path.is_file():
        require(args.phase == "review", "A public receipt is read-only")
        receipt = json.loads(args.path.read_text())
        frozen, rows = receipt["freeze"], receipt["rows"]
    else:
        frozen = json.loads((args.path / "freeze.json").read_text())
        require(
            hashlib.sha256(Path(__file__).read_bytes()).hexdigest() == frozen["code"][Path(__file__).name],
            "The native runner differs from the frozen procedure",
        )
        if args.phase == "prepare":
            prepare(args.path, frozen)
            return
        if args.phase == "run":
            run(args.path, frozen)
            return
        rows = [json.loads(line) for line in (args.path / "rows.jsonl").read_text().splitlines()]
    require(frozen["schema"] == SCHEMA and frozen["design"] == DESIGN, "Unknown experiment design")
    result = review(rows, frozen["panel"])
    if args.path.is_file():
        require(result == receipt["result"], "Public decisions differ from the recorded outcome")
    if args.output:
        with args.output.open("x") as output:
            json.dump(
                {
                    "freeze": {key: frozen[key] for key in frozen if key != "sources"},
                    "rows": rows,
                    "result": result,
                },
                output,
                indent=2,
                allow_nan=False,
            )
            output.write("\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
