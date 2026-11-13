# SPDX-License-Identifier: Apache-2.0
"""Compare the fixed short-face proposal on two overlapping native HEVC windows."""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import time
from fractions import Fraction
from pathlib import Path

import cv2

from scripts import video_face_paired_edge_hevc_confirmation as delivery
from scripts import video_face_paired_edge_transfer as earlier
from scripts import video_face_six_picture_transfer as corpus
from scripts import video_face_two_returns_hd as dual
from scripts.video_common_sources_v89 import read, require, sha, write
from scripts.video_native_sources_v92 import MEDIA as NATIVE_MEDIA
from scripts.video_native_sources_v92 import TRIAL as NATIVE_TRIAL

N = 240
STARTS = (0, 60, 120)
ROLES = (
    "genuine",
    "whole_loop",
    "jump",
    "light",
    "early_swap",
    "late_swap",
    "tandem_late",
    "two_swaps",
    "genuine_cut",
)
POSITIVES = ("early_swap", "late_swap", "tandem_late", "two_swaps")
GATE = {
    "minimum_recovered_late": 8,
    "minimum_groups": 4,
    "minimum_exact": 4,
    "max_new_ordinary": 1,
    "max_new_cut": 0,
    "minimum_measured": 0.75,
    "maximum_lost_first_window": 0,
    "minimum_temporal_iou": 0.5,
}
SPEC = {
    "profile": "Upright 240 native pictures, 1280 x 720, exact 30fps 8-bit 4:2:0 hvc1 HEVC MP4.",
    "windows": STARTS,
    "window_pictures": 120,
    "interventions": "Use continuous exact eight-second qualified originals. Independently verified publisher face edits replace "
    "six same-time pictures in one early range and one late range after the first window. Held positions vary by group.",
    "comparison": "Actual eight-second Full face stages with the existing optional checks. Compare a hypothetical single "
    "120-picture prefix at the same fixed point with three overlapping windows. The incumbent stays intact.",
    "output": "Keep at most one useful additional short proposal per 120-picture window. Deduplicate against prior ranges and "
    "against the earlier window at .5 temporal IoU. Use exact source PTS and source-bound full/crop anchor commitments.",
    "controls": "Original, whole-picture loop, same-recording jump, brightness edit and cross-recording hard cut on the same time bounds.",
    "max_source_bytes": 100 * 1024**2,
}


def plan():
    from app.detectors.video import face_paired_edge, face_short_proposal
    from app.evidence import face_short_proposal_binding

    _, common = dual.archive_context()
    parent = NATIVE_TRIAL / "qualified-plan.json"
    parent_sha = sha(parent)
    require(
        parent_sha == read(NATIVE_TRIAL / "summary.json")["qualified_private_plan_sha256"],
        "Publisher qualification changed",
    )
    assets = {}
    for asset in read(parent)["assets"]:
        if asset["kind"] in ("original_videos", "inswapper", "uniface"):
            assets.setdefault(asset["group"], {})[asset["kind"]] = asset
    meta = corpus.trial.prior.comparison.metadata()
    current = read(
        dual.ROOT / "backend/data/independent-evaluation/video-face-two-returns-hd-2026-11-12/freeze.json"
    )
    roster = []
    for p in sorted(common.DATA.glob("group-*-complete.json")):
        d = read(p)
        selected = assets[d["group"]]
        original = selected["original_videos"]
        pts = original["frame_table"]
        row = meta[d["group"]]
        if (
            "child" in row["tags"]
            or len(pts["pts"]) < N
            or not original["qualification"]["eligible"]
            or pts["width"] not in (1280, 1920)
            or any(
                (p - pts["pts"][0]) * Fraction(pts["time_base"]) != Fraction(i, 30)
                for i, p in enumerate(pts["pts"][:N])
            )
            or any(
                a["frame_table"]["pts"][:N] != pts["pts"][:N]
                or a["frame_table"]["time_base"] != pts["time_base"]
                or not a["qualification"]["eligible"]
                for a in selected.values()
            )
        ):
            continue
        roster.append(
            {
                "index": d["index"],
                "role": d["role"],
                "url": row["url"],
                "donors": sorted(row["donors"]),
                "source_sha256": sha(p),
                "source_group": d["group"],
                "original_component": original["qualification"]["component"],
            }
        )
    known = {r["index"] for r in current["sources"]}
    development = [r for r in roster if r["role"] == "selection"][:2]
    require(
        len({r["original_component"] for r in development}) == len(development)
        and not set(development[0]["donors"]) & set(development[1]["donors"]),
        "Development source pair overlaps",
    )
    held, seen_urls = [], {r["url"] for r in development}
    seen_donors = {key for r in development for key in r["donors"]}
    seen_components = {r["original_component"] for r in development}
    for row in sorted(roster, key=lambda r: (r["index"] in known, r["role"] != "confirmation", r["index"])):
        if (
            row in development
            or row["url"] in seen_urls
            or row["original_component"] in seen_components
            or seen_donors & set(row["donors"])
        ):
            continue
        held.append(row)
        seen_urls.add(row["url"])
        seen_donors.update(row["donors"])
        seen_components.add(row["original_component"])
        if len(held) == 6:
            break
    require(len(development) == 2 and len(held) == 6, "Need eight independent known source/donor groups")
    code = [
        Path(__file__),
        Path(face_paired_edge.__file__),
        Path(face_short_proposal.__file__),
        Path(face_short_proposal_binding.__file__),
    ]
    return {
        "schema": "chipmunk.video-face-short-prefix-transfer/1",
        "method": {**SPEC, "windows": list(STARTS)},
        "existing_point": face_short_proposal_binding.POINT,
        "native_parent_sha256": parent_sha,
        "gate": GATE,
        "code": {p.name: sha(p) for p in code},
        "weights": earlier.WEIGHTS,
        "opencv": cv2.__version__,
        "development": development,
        "held": held,
    }


def encoding():
    return {**delivery.ENCODING, "nb_frames": str(N)}


def probe_encoding(path):
    return delivery.ffprobe(path)


def write_video(path, pictures=None, source=None):
    opts = list(delivery.ENCODE)
    if source is not None:
        opts[opts.index("26")] = "35"
        subprocess.run(
            ["ffmpeg", "-nostdin", "-v", "error", "-threads", "1", "-i", str(source), *opts, str(path)],
            check=True,
            timeout=30,
        )
    else:
        command = [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-threads",
            "1",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            "1280x720",
            "-r",
            "30",
            "-i",
            "pipe:0",
            *opts,
            str(path),
        ]
        with subprocess.Popen(command, stdin=subprocess.PIPE) as proc:
            try:
                for rgb in pictures:
                    proc.stdin.write(memoryview(rgb).cast("B"))
                proc.stdin.close()
                require(proc.wait(timeout=30) == 0, "HEVC video creation failed")
            finally:
                if proc.poll() is None:
                    proc.kill()
    require(probe_encoding(path) == [encoding()], "Unexpected 240-picture delivery")


def clock(path):
    from app.detectors.video.decode import probe_frames

    table = probe_frames(path, timeout_s=15)
    require(
        len(table) == N
        and not (table.truncated or table.interpolated or table.rotation)
        and (table.width, table.height, table.coded_width, table.coded_height) == (1280, 720, 1280, 720)
        and all((p - table.pts[0]) * table.tb == Fraction(i, 30) for i, p in enumerate(table.pts)),
        "Unsupported native delivery clock",
    )
    return table


def positions(group):
    early = 69 + 2 * (group % 5)
    late = (128, 144, 168, 180, 198, 212)[(group - 1) % 6]
    return (early, early + 6), (late, late + 6)


def native(asset, indices, hashes):
    from app.detectors.video.decode import FrameTable, decode_frames

    path = NATIVE_MEDIA / f"asset-{asset['index']:03d}.mp4"
    require(sha(path) == asset["input_sha256"], "Publisher asset differs from its qualification")
    table = FrameTable.from_json(asset["frame_table"])
    result = {}
    for frame in decode_frames(path, table, sorted(indices), timeout_s=30):
        digest = hashlib.sha256(frame.rgb.tobytes()).hexdigest()
        require(
            frame.pts_verified
            and frame.pts == table.pts[frame.index]
            and digest == asset["rgb_sha256"][frame.index],
            "Publisher native clock or RGB commitment differs",
        )
        hashes.append(digest)
        result[frame.index] = cv2.resize(frame.rgb, (1280, 720), interpolation=cv2.INTER_AREA)
    require(result.keys() == set(indices), "Publisher native frame support incomplete")
    return result


def prepare_group(directory, member, other, group, assets):
    import numpy as np

    _, common = dual.archive_context()
    d = common.DATA / f"group-{member['index']:03d}-complete.json"
    require(sha(d) == member["source_sha256"], "Source freeze differs")
    donor_doc = common.DATA / f"group-{other['index']:03d}-complete.json"
    require(sha(donor_doc) == other["source_sha256"], "Control freeze differs")
    early, late = positions(group)
    a, b = early
    c, e = late
    source = assets[member["source_group"]]
    cutaway = assets[other["source_group"]]["original_videos"]
    hashes = {role: [] for role in ("original", "early", "late", "ordinary_cut")}
    host = native(source["original_videos"], range(N), hashes["original"])
    earlier_rgb = native(source["inswapper"], range(a, b), hashes["early"])
    later_rgb = native(source["uniface"], range(c, e), hashes["late"])
    guest = native(cutaway, range(c, e), hashes["ordinary_cut"])
    dark = {i: np.clip(host[i].astype(np.float32) * 0.78 + 8, 0, 255).astype(np.uint8) for i in range(c, e)}
    edits = {
        "genuine": ({}, []),
        "whole_loop": ({i: host[i - (c - a)] for i in range(c, e)}, []),
        "jump": ({i: host[i + 10] for i in range(c, e)}, []),
        "light": (dark, []),
        "early_swap": (earlier_rgb, [list(early)]),
        "late_swap": (later_rgb, [list(late)]),
        "tandem_late": (later_rgb, [list(late)]),
        "two_swaps": (earlier_rgb | later_rgb, [list(early), list(late)]),
        "genuine_cut": (guest, []),
    }
    result = []
    for role in ROLES:
        replacement, truth = edits[role]
        path = directory / f"{role}.mp4"
        if role == "tandem_late":
            write_video(path, source=directory / "late_swap.mp4")
        else:
            require(
                all(np.count_nonzero(np.any(host[i] != pic, axis=2)) >= 20 for i, pic in replacement.items()),
                "Authored source change must be visible on each selected picture",
            )
            write_video(path, (replacement.get(i, host[i]) for i in range(N)))
        table = clock(path)
        result.append(
            {
                "role": role,
                "truth": truth,
                "sha256": sha(path),
                "clock": table.to_json(),
                "authored_rgb_sha256": {
                    str(i): hashlib.sha256(frame.tobytes()).hexdigest() for i, frame in replacement.items()
                },
            }
        )
    write(directory / "source.json", hashes)
    return result


def iou(a, b):
    return max(0, min(a[1], b[1]) - max(a[0], b[0])) / max(1, max(a[1], b[1]) - min(a[0], b[0]))


def measure(path, receipt, detector, recognizer, baseline):
    from app.detectors.image.faces import detect_faces
    from app.detectors.video import face_paired_edge, face_short_proposal
    from app.detectors.video.adapters.face_appearance import _measure
    from app.detectors.video.decode import decode_frames
    from app.detectors.video.face_appearance import unit
    from app.evidence import face_short_proposal_binding as proposal

    table = clock(path)
    require(table.to_json() == receipt["clock"] and sha(path) == receipt["sha256"], "Source receipt changed")
    wanted = set(s + i for s in STARTS for i in proposal.ANCHOR_SAMPLES)
    pixels, values, hashes = {}, {}, {}

    def read_faces(frame):
        bgr = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)
        faces = detect_faces(bgr, detector=detector, min_size=48)
        if len(faces) == 1 and (
            measured := _measure(frame.rgb, bgr, frame.index, table, faces[0], recognizer)
        ):
            v, binding = measured
            values[frame.index] = (unit(v), faces[0], binding)

    def bound(frame):
        require(frame.pts_verified and frame.pts == table.pts[frame.index], "Native timestamps differ")
        return hashlib.sha256(frame.rgb.tobytes()).hexdigest()

    started = time.monotonic()
    for frame in decode_frames(path, table, None, timeout_s=25):
        hashes[frame.index] = bound(frame)
        pixels[frame.index] = face_paired_edge.reduced_rgb(frame.rgb)
        if frame.index in wanted:
            read_faces(frame)
    require(set(pixels) == set(range(N)), "Incomplete source native pictures")
    plans = {}
    for start in STARTS:
        reduced = {i: pixels[start + i] for i in range(120)}
        anchors = [values[start + i][2]["bbox"] for i in proposal.ANCHOR_SAMPLES if start + i in values]
        edges = face_short_proposal.shortlist(reduced, anchors, lambda _: None)
        plans[start] = (reduced, edges, len(anchors))
        wanted.update(start + i for i in proposal.face_schedule(edges))
    want = wanted - {s + i for s in STARTS for i in proposal.ANCHOR_SAMPLES}
    for frame in decode_frames(path, table, sorted(want), timeout_s=25):
        require(hashes[frame.index] == bound(frame), "Native RGB differs between measurement reads")
        read_faces(frame)
    ranges = list(baseline)
    records = []
    for start, (reduced, edges, anchors) in plans.items():
        seen = {i - start: values[i] for i in wanted if start <= i < start + 120 and i in values}
        selected = face_short_proposal.select(
            edges,
            seen,
            reduced,
            lambda _: None,
            lambda a, b, start=start: any(iou((a + start, b + start), prior) >= 0.5 for prior in ranges),
        )
        record = None
        if selected is not None:
            (response, a, b), _, _ = selected
            native = [
                start + i
                for i in (
                    a - 1,
                    min((i for i in seen if a <= i < b), key=lambda i: (abs(2 * i - a - b), i)),
                    b,
                )
            ]
            record = {
                "frames": [start + a, start + b],
                "response": -response,
                "anchors": [
                    {
                        k: values[i][2][k]
                        for k in ("frame_index", "pts", "decoded_rgb_sha256", "aligned_rgb_sha256", "bbox")
                    }
                    for i in native
                ],
            }
            ranges.append(record["frames"])
        records.append(
            {
                "window_start": start,
                "assessed": anchors >= 4,
                "anchors_observed": anchors,
                "edges": [start + i for i in edges],
                "match": record,
            }
        )
    return {
        "windows": records,
        "native_hashes": [hashes[i] for i in range(N)],
        "face_inputs": len(wanted),
        "elapsed": time.monotonic() - started,
    }


def run(folder, lock, role):
    from app.detectors.image.faces import load_yunet

    weights = earlier.ROOT / "models/weights"
    require({name: sha(weights / name) for name in earlier.WEIGHTS} == earlier.WEIGHTS, "Weights changed")
    detector = load_yunet(weights / "face_detection_yunet_2023mar.onnx")
    recognizer = cv2.FaceRecognizerSF.create(str(weights / "face_recognition_sface_2021dec.onnx"), "")
    assets = {}
    for asset in read(NATIVE_TRIAL / "qualified-plan.json")["assets"]:
        if asset["kind"] in ("original_videos", "inswapper", "uniface"):
            assets.setdefault(asset["group"], {})[asset["kind"]] = asset
    records = []
    for group, source in enumerate(lock[role], 1):
        local = folder / f"{role}-{group}"
        local.mkdir(mode=0o700)
        other = lock[role][group % len(lock[role])]
        cases = prepare_group(local, source, other, group, assets)
        for row in cases:
            path = local / (row["role"] + ".mp4")
            result = measure(path, row, detector, recognizer, [])
            records.append({"group": group, **row, **result})
            print(
                role,
                group,
                row["role"],
                [(v["match"] or {}).get("frames") for v in result["windows"]],
                round(result["elapsed"], 2),
                flush=True,
            )
        require(
            sum(p.stat().st_size for p in folder.glob("*/*.mp4")) <= SPEC["max_source_bytes"],
            "Local media budget exceeded",
        )
        write(local / "results.json", [r for r in records if r["group"] == group])
    write(folder / (role + ".json"), {"freeze_sha256": sha(folder / "freeze.json"), "rows": records})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "development", "held"))
    parser.add_argument("folder", type=Path)
    args = parser.parse_args()
    cv2.setNumThreads(2)
    lock = plan()
    if args.phase == "freeze":
        args.folder.mkdir(mode=0o700)
        write(args.folder / "freeze.json", lock)
        return
    require(read(args.folder / "freeze.json") == lock, "Frozen study differs")
    run(args.folder, lock, args.phase)


if __name__ == "__main__":
    main()
