# SPDX-License-Identifier: Apache-2.0
"""Transfer the frozen 120-picture face proposal to AVC MP4 and VP9 WebM."""

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

from scripts.video_common_sources_v89 import read, require, sha, write

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "backend/data/independent-evaluation"
BANKS = {"edits": "video-face-two-returns-hd-2026-11-12", "cuts": "video-face-hd-retry-scene-challenge-2026-11-12"}
ROLES = {"edits": {"original", "same_source_cuts", "cross_recording_cuts", "two_swaps", "tandem_two_swaps"},
         "cuts": {"person_switch", "tandem_person_switch"}}
SCHEMA = "chipmunk.video-brief-face-120-carrier/1"
WEIGHTS = {"face_detection_yunet_2023mar.onnx": "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
           "face_recognition_sface_2021dec.onnx": "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79"}
VP9 = ["-c:v", "libvpx-vp9", "-threads", "1", "-pix_fmt", "yuv420p", "-b:v", "0", "-crf", "30", "-deadline", "realtime",
       "-cpu-used", "6", "-lag-in-frames", "0", "-fps_mode", "passthrough"]
DESIGN = {"point": .10, "minimum_added_six_picture_ranges": 4, "minimum_distinct_gained_groups": 3,
          "minimum_exact_additions": 2, "maximum_new_ordinary_flags": 1, "maximum_new_speaker_or_cross_recording_cut_flags": 0,
          "maximum_wrong_edited_ranges": 1, "min_measured_files": 42,
          "windows": "Unchanged 5–7-picture two-edge shortlist with at most 84 face measurements and three frames per side.",
          "separation": "Use the frozen HEVC point. No new fit. Same retained source groups had prior app exposure. VP9 is an additional carrier, not an independent positive bank."}


def freeze():
    from app.detectors.video import face_paired_edge, face_short_proposal
    from app.evidence import face_short_proposal_binding as proof

    require(DESIGN["point"] == proof.POINT, "Production point changed")
    paths = [Path(__file__), Path(face_paired_edge.__file__), Path(face_short_proposal.__file__), Path(proof.__file__)]
    return {"schema": SCHEMA, "design": DESIGN, "weights": WEIGHTS, "vp9_encode": VP9,
            "methods": {p.name: sha(p) for p in paths},
            "banks": {name: {entry: sha(DATA / bank / entry) for entry in ("media.json", "held.jsonl")}
                      for name, bank in BANKS.items()}}


def matches(a, b):
    return 2 * (min(a[1], b[1]) - max(a[0], b[0])) >= max(a[1], b[1]) - min(a[0], b[0])


def roster():
    from scripts import video_face_return_eligibility as ranges

    for bank, folder in BANKS.items():
        home = DATA / folder
        measured = {row["sha256"]: row for row in map(json.loads, (home / "held.jsonl").read_text().splitlines())}
        for row in read(home / "media.json")["rows"]:
            if row["condition"] not in ROLES[bank]:
                continue
            saved = measured[row["sha256"]]
            details = saved["baseline"] if bank == "edits" else saved["before"]
            bounds = [list(ranges.interval(details[key], row["clock"]["pts"])) for key in ("coarse", "boundary_refinement", "short_range")
                      if details[key]]
            if bank == "edits":
                second = saved["second"]["range"]
                if second:
                    bounds.append(second["frames"])
            else:
                bounds.extend(list(ranges.interval(item, row["clock"]["pts"])) for item in saved["incumbent_second"])
            yield home / "media" / row["file"], row, bank, bounds, details["short_range"] is not None


def encode_webm(source, output):
    subprocess.run(["ffmpeg", "-nostdin", "-n", "-v", "error", "-threads", "1", "-i", str(source), "-map", "0:v:0",
                    "-an", "-sn", "-dn", *VP9, str(output)], check=True, timeout=30)


def scan(path, table, detector, recognizer, expected_pixels=None):
    import cv2

    from app.detectors.image.faces import detect_faces
    from app.detectors.video import face_appearance, face_paired_edge, face_short_proposal
    from app.detectors.video.adapters.face_appearance import _measure
    from app.detectors.video.decode import decode_frames
    from app.evidence import face_short_proposal_binding as proof

    reduced, measured, native = {}, {}, []

    def read_face(frame):
        bgr = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)
        observed = detect_faces(bgr, detector=detector, min_size=48)
        if len(observed) == 1 and (result := _measure(frame.rgb, bgr, frame.index, table, observed[0], recognizer)) is not None:
            vector, record = result
            measured[frame.index] = face_appearance.unit(vector), observed[0], record

    for frame in decode_frames(path, table, None, timeout_s=35):
        i = frame.index
        require(i == len(native) and frame.pts_verified and frame.pts == table.pts[i], "Invalid native picture clock")
        digest = hashlib.sha256(frame.rgb).hexdigest()
        require(expected_pixels is None or digest == expected_pixels[i], "Archived native picture differs")
        reduced[i] = face_paired_edge.reduced_rgb(frame.rgb)
        native.append(digest)
        if i in proof.ANCHOR_SAMPLES:
            read_face(frame)
    require(len(native) == 120, "Native pictures missing")
    anchor_count = len(measured)
    edges = face_short_proposal.shortlist(reduced, [record["bbox"] for _, _, record in measured.values()], lambda *_: None)
    schedule = proof.face_schedule(edges)
    requested = set(schedule) - set(proof.ANCHOR_SAMPLES)
    if requested:
        decoded = set()
        for frame in decode_frames(path, table, sorted(requested), timeout_s=35):
            i = frame.index
            require(frame.pts_verified and frame.pts == table.pts[i] and hashlib.sha256(frame.rgb).hexdigest() == native[i],
                    "Scheduled face picture differs")
            decoded.add(i)
            read_face(frame)
        require(decoded == requested, "Scheduled face pictures missing")

    def select(exclusions):
        candidate = face_short_proposal.select(edges, measured, reduced, lambda *_: None,
                                                lambda start, end: any(matches((start, end), prior) for prior in exclusions))
        if candidate is None:
            return None
        ((negative, start, end), _, _) = candidate
        middle = min((i for i in measured if start <= i < end), key=lambda i: (abs(2 * i - start - end), i))
        return {"frames": [start, end], "response": -negative, "pts": [table.pts[start], table.pts[end]],
                "anchors": [{key: measured[i][2][key] for key in ("frame_index", "pts", "decoded_rgb_sha256", "aligned_rgb_sha256")}
                            for i in (start - 1, middle, end)]}

    return select, {"available": anchor_count >= 4, "face_frames": len(measured), "scheduled": len(schedule),
                    "native_pts_sha256": hashlib.sha256(json.dumps(table.pts).encode()).hexdigest()}, native


def run(folder, carrier, limit, state):
    from scripts import screen_video_native_background_challenge as worker

    import cv2

    from app.detectors.image.faces import load_yunet
    from app.detectors.video.decode import FrameTable, probe_frames
    from app.detectors.video.face_return import source_carrier
    from app.evidence import scene_return_binding
    from scripts import video_face_hd_sparse_retry as incumbent
    from scripts import video_face_return_eligibility as ranges

    baseline = worker.scorer(state) if carrier == "vp9" else None
    if baseline:
        import torch

        torch.set_num_threads(1)
    cv2.setNumThreads(1)
    weights = ROOT / "models/weights"
    require(all(sha(weights / name) == digest for name, digest in WEIGHTS.items()), "Model bytes differ")
    detector = load_yunet(weights / "face_detection_yunet_2023mar.onnx")
    recognizer = cv2.FaceRecognizerSF.create(str(weights / "face_recognition_sface_2021dec.onnx"), "")
    out = folder / carrier
    out.mkdir(mode=0o700, exist_ok=True)
    rows = list(roster())
    require(len(rows) == 56, "Frozen panel is incomplete")
    with (folder / f"{carrier}.jsonl").open("x") as journal:
        for index, (source, row, bank, prior, retained_short) in enumerate(rows[:limit]):
            require(sha(source) == row["sha256"], "Source differs from frozen roster")
            path = source
            if carrier == "vp9":
                path = out / f"{index:02}.webm"
                encode_webm(source, path)
            table = probe_frames(path, cap=130, timeout_s=18)
            require((table.width, table.height, table.coded_width, table.coded_height, len(table)) == (1280, 720, 1280, 720, 120)
                    and not (table.rotation or table.truncated or table.interpolated), "Unsupported picture geometry")
            require((source_carrier(path) == "webm" and scene_return_binding.valid_webm_clock(table.time_base, table.pts, table.pts[0]))
                    if carrier == "vp9" else table == FrameTable.from_json(row["clock"]), "Native carrier clock differs")
            select, measured, native = scan(path, table, detector, recognizer, row["rgb_sha256"] if carrier == "avc" else None)
            if baseline:
                receipt = {"sha256": sha(path), "clock": table.to_json(), "rgb_sha256": native}
                actual = incumbent.test_file(baseline, path, receipt)
                detail = actual["before"]
                prior = [list(ranges.interval(detail[k], table.pts)) for k in ("coarse", "boundary_refinement", "short_range") if detail[k]]
                prior.extend(list(ranges.interval(item, table.pts)) for item in actual["incumbent_second"])
                retained_short = detail["short_range"] is not None
            proposal = select(prior) if not retained_short else None
            item = {"index": index, "source_sha256": row["sha256"], "delivery_sha256": sha(path), "bank": bank,
                    "group": row["group"], "condition": row["condition"], "truth": row["truth"], "prior": prior,
                    "retained_short": retained_short, "measurement": measured, "time_base": table.time_base,
                    "first_pts": table.pts[0], "proposal": proposal}
            journal.write(json.dumps(item, allow_nan=False) + "\n")
            journal.flush()
            print(carrier, index + 1, row["condition"], prior, proposal["frames"] if proposal else None, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "avc", "vp9"))
    parser.add_argument("run", type=Path)
    parser.add_argument("--limit", type=int, default=56)
    args = parser.parse_args()
    if args.phase == "freeze":
        args.run.mkdir(mode=0o700)
        write(args.run / "freeze.json", freeze())
        return
    from scripts import screen_video_native_background_challenge as worker

    state = worker.configure(args.run)
    os.environ.update(VERITAS_VIDEO_FACE_SECOND_RETURN_REVIEW="true", VERITAS_VIDEO_FACE_UNROUTED_SHORT_REVIEW="false",
                      VERITAS_VIDEO_FACE_SHORT_CARRIER_REVIEW="false")
    require(read(args.run / "freeze.json") == freeze(), "Frozen procedure or input manifest changed")
    run(args.run, args.phase, args.limit, state)


if __name__ == "__main__":
    main()
