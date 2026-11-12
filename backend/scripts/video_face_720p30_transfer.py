"""Test a source-separated 720p30 transfer of the serving short-face comparator."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from collections import Counter
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

from scripts import screen_video_native_background_challenge as worker
from scripts.review_video_lossy_sequence_transfer import read as load
from scripts.review_video_lossy_sequence_transfer import sha_file as hash_file

N = 120
SIZE = (1280, 720)
RATE = 30
SWAPS = {"swap_a": (72, 78), "swap_b": (87, 97)}
ORDINARY = ("original", "recompressed", "light", "jump", "cut")
CAMERA_SLOTS = (1, 3, 7)
SCHEMA = "chipmunk.video-face-720p30-transfer/1"
GATE = {"minimum_new_useful": 3, "minimum_new_groups": 2, "maximum_ordinary_additions": 1, "maximum_cut_additions": 0,
        "minimum_measured_share": .8, "minimum_useful_iou": .5}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def write(path, value):
    with path.open("x") as output:
        json.dump(value, output, allow_nan=False)
        output.write("\n")


def plan_sources():
    from scripts import video_face_codec_background_transfer as latest

    pool, histories = latest.cases(latest.ARCHIVE)
    latest_path = latest.PRIOR.parent.parent / "video-face-codec-background-transfer-2026-11-12/freeze.json"
    known = {row["index"] for row in load(latest_path)["sources"]}
    fresh = [row["index"] for row in pool if row["index"] not in known]
    require(len(fresh) == 4, "Require four source/donor-separated held groups")
    qualified = latest.ARCHIVE / "video-vcd2-unseen-controls-v287"
    pins = {}
    for slot in CAMERA_SLOTS:
        path = qualified / f"slot-{slot:02d}-attempt-01/qualification.json"
        pins[slot] = hash_file(path)
    return latest.ARCHIVE, latest.common, fresh, pins, [hash_file(path) for path in (*histories, latest_path)]


def relative(table):
    return [(tick - table.pts[0]) * table.tb for tick in table.pts]


def authored(path, *, pictures=None, source=None, crf=23):
    opts = ["-c:v", "libx264", "-preset", "fast", "-crf", str(crf), "-threads", "1", "-pix_fmt", "yuv420p",
            "-video_track_timescale", "30000", "-movflags", "+faststart", str(path)]
    base = ["ffmpeg", "-nostdin", "-v", "error", "-filter_threads", "1"]
    if source is not None:
        subprocess.run([*base, "-i", str(source), "-map", "0:v:0", "-an", *opts], check=True, timeout=45)
        return
    args = [*base, "-f", "rawvideo", "-pixel_format", "rgb24", "-video_size", "1280x720", "-framerate", "30", "-i", "pipe:0", *opts]
    with subprocess.Popen(args, stdin=subprocess.PIPE) as proc:
        try:
            for picture in pictures:
                proc.stdin.write(memoryview(picture).cast("B"))
            proc.stdin.close()
            require(proc.wait(timeout=45) == 0, "H.264 encode failed")
        finally:
            if proc.poll() is None:
                proc.kill()


def native(archive, common, index, kind, output_indices):
    import cv2

    case = next(row for row in load(common.DATA / f"group-{index:03d}-complete.json")["cases"] if row["condition"] == kind)
    path = archive / "video-native-inserts-v95" / f"group-{index:03d}" / case["file"]
    require(hash_file(path) == case["sha256"], "Archived video differs from its qualified digest")
    table = common.probe_frames(path, timeout_s=30)
    require(table.to_json() == case["frame_table"] and not (table.truncated or table.interpolated or table.rotation),
            "Archived frame clock differs from qualification")
    times = relative(table)
    samples = {min(range(len(times)), key=lambda i: (abs(times[i] - Fraction(n, RATE)), i)): n for n in output_indices}
    require(len(samples) == len(output_indices)
            and max(abs(times[i] - Fraction(n, RATE)) for i, n in samples.items()) <= Fraction(1, 100),
            "Authored frame assignments have too much retiming or reuse")
    if kind in SWAPS:
        require(set(samples) <= set(case["actual_changed_frames"]), "Authored swap frame was unchanged in native input")
    output = {}
    for frame in common.decode_frames(path, table, sorted(samples), timeout_s=40):
        require(frame.pts_verified and frame.pts == table.pts[frame.index]
                and hashlib.sha256(frame.rgb).hexdigest() == case["expected_rgb_sha256"][frame.index],
                "Archived native RGB picture differs from its qualification")
        output[samples[frame.index]] = cv2.resize(frame.rgb, SIZE, interpolation=cv2.INTER_AREA)
    require(set(output) == set(output_indices), "Incomplete source picture assignment")
    return output


def describe(path, group, condition, truth):
    from app.detectors.video.decode import decode_frames, probe_frames

    table = probe_frames(path, timeout_s=20)
    require((table.width, table.height) == SIZE and relative(table) == [Fraction(i, RATE) for i in range(N)]
            and not (table.rotation or table.interpolated or table.truncated), "Delivered source clock is not exact 720p30")
    hashes = []
    for frame in decode_frames(path, table, range(N), timeout_s=15):
        require(frame.index == len(hashes) and frame.pts_verified, "Native frame hash lacks a verified timestamp")
        hashes.append(hashlib.sha256(frame.rgb).hexdigest())
    require(len(hashes) == N, "Incomplete delivered pictures")
    return {"file": path.name, "group": group, "condition": condition, "truth": truth, "sha256": hash_file(path),
            "clock": table.to_json(), "rgb_sha256": hashes}


def prepare(run):
    import cv2
    import numpy as np

    cv2.setNumThreads(1)
    archive, common, groups, cameras, history = plan_sources()
    run.mkdir(mode=0o700)
    write(run / "freeze.json", {"schema": SCHEMA, "groups": groups, "camera_pins": cameras, "previous": history,
                               "swaps": SWAPS, "gate": GATE, "runner_sha256": hash_file(Path(__file__)),
                               "models": common.bindings()})
    rows = []
    for pair in range(2):
        codes = groups[pair * 2:pair * 2 + 2]
        base = [native(archive, common, index, "original", range(N)) for index in codes]
        for side, index in enumerate(codes):
            host, guest = base[side], base[1 - side]
            original = run / f"g{pair * 2 + side + 1}-original.mp4"
            group = pair * 2 + side + 1
            for kind in (*ORDINARY, *SWAPS):
                truth = SWAPS.get(kind)
                path = run / f"g{group}-{kind}.mp4"
                if kind == "recompressed":
                    authored(path, source=original, crf=35)
                else:
                    left, right = truth or SWAPS["swap_b"]
                    if truth:
                        altered = native(archive, common, index, kind, range(left, right))
                    elif kind == "cut":
                        altered = {i: guest[i] for i in range(left, right)}
                    elif kind == "jump":
                        altered = {i: host[i - 24] for i in range(left, right)}
                    elif kind == "light":
                        altered = {i: np.clip(host[i].astype(np.float32) * 1.20, 0, 255).astype("u1") for i in range(left, right)}
                    else:
                        altered = {}
                    require(all(not np.array_equal(rgb, host[i]) for i, rgb in altered.items()), "An authored picture did not change")
                    authored(path, pictures=(altered.get(i, host[i]) for i in range(N)))
                rows.append(describe(path, group, kind, truth))
            print(f"Prepared held face source {group}.", flush=True)
    for ordinal, slot in enumerate(CAMERA_SLOTS, len(groups) + 1):
        source = archive / "video-vcd2-unseen-controls-v287" / f"slot-{slot:02d}-attempt-01"
        path = run / f"g{ordinal}-camera.mp4"
        require(hash_file(source / "source.mkv") == load(source / "qualification.json")["sha256"], "Genuine camera source differs")
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", str(source / "source.mkv"), "-map", "0:v:0",
                        "-an", "-filter_threads", "1", "-vf", "fps=30,scale=1280:720:flags=area",
                        "-frames:v", str(N), "-c:v", "libx264", "-preset", "fast", "-threads", "1", "-crf", "23",
                        "-pix_fmt", "yuv420p", "-video_track_timescale", "30000", str(path)], check=True, timeout=45)
        rows.append(describe(path, ordinal, "camera", None))
    require(len(rows) == len({row["sha256"] for row in rows}) == 31, "Incomplete or duplicate held deliveries")
    write(run / "media.json", {"freeze_sha256": hash_file(run / "freeze.json"), "rows": rows})


def extra(path, row, baseline, models):
    from app.detectors.video import face_return
    from app.detectors.video.adapters.face_appearance import _return_pass
    from app.detectors.video.adapters.face_sequence import SequenceInputs
    from app.detectors.video.decode import FrameTable
    from app.detectors.video.frame_store import FrameStore
    from app.evidence.face_return_binding import GUARDED_CODEC_METHOD

    table = FrameTable.from_json(row["clock"])
    require(face_return.schedule(table) is None, "The current short review unexpectedly accepts 720p30")
    times, sparse = face_return.schedule(replace(table, width=1920, height=1080))
    inputs = SequenceInputs(table, FrameStore(path, table), (), (), ("h264", "yuv420p"))
    deadline = time.monotonic() + 12
    disposition, detail = {"measured_face_frames": 0}, {}
    _return_pass(detail, inputs, *models, None, baseline["coarse"], (times, sparse), disposition, face_return.METHOD, deadline)
    dense_detail, dense = {}, {"status": "not_run", "reason": "first_pass_disposition", "measured_face_frames": 0}
    if disposition["status"] == "no_distinct_range":
        _return_pass(dense_detail, inputs, *models, None, baseline["coarse"], (times, tuple(range(0, N, 2))), dense,
                     GUARDED_CODEC_METHOD, deadline, source_carrier="mp4")
    for selected in (detail.get("face_return"), dense_detail.get("face_return")):
        if selected:
            for item in selected["frame_inputs"] + selected["scene_frame_inputs"]:
                i = item["frame_index"]
                require(item["pts"] == table.pts[i] and item["decoded_rgb_sha256"] == row["rgb_sha256"][i], "Candidate pixels differ")
    return {"sparse": detail.get("face_return"), "sparse_coverage": disposition,
            "dense": dense_detail.get("face_return"), "dense_coverage": dense}


def iou(pair, truth):
    if pair is None:
        return 0.
    a, b = pair
    c, d = truth
    return max(0, min(b, d) - max(a, c)) / (max(b, d) - min(a, c))


def metrics(rows):
    summary = {}
    for arm in ("sparse", "dense"):
        positives = [row for row in rows if row["truth"]]
        ordinary = [row for row in rows if not row["truth"] and row["condition"] != "cut"]
        cuts = [row for row in rows if row["condition"] == "cut"]

        def extra_range(row, arm=arm):
            return row["sparse"] or (row["dense"] if arm == "dense" else None)

        useful = [row for row in positives if iou(row["baseline"], row["truth"]) >= .5]
        added = [row for row in positives if row not in useful and iou(extra_range(row), row["truth"]) >= .5]
        coverage = [sum(row["sparse_status"] in {"no_distinct_range", "localized"} for row in family)
                    for family in (positives, ordinary)]
        control = [sum(row["baseline"] is None and extra_range(row) is not None for row in family) for family in (ordinary, cuts)]
        groups = len({row["group"] for row in added})
        result = {"baseline_useful": len(useful), "combined_useful": len(useful) + len(added), "new_groups": groups,
                  "positives": len(positives), "ordinary": len(ordinary), "cuts": len(cuts), "new_ordinary": control[0],
                  "new_cut": control[1], "sparse_measured_swap": coverage[0], "sparse_measured_ordinary": coverage[1],
                  "baseline_control_flags": dict(Counter(r["condition"] for r in ordinary + cuts if r["baseline"] is not None))}
        result["passes"] = (len(added) >= GATE["minimum_new_useful"] and groups >= GATE["minimum_new_groups"]
                            and control[0] <= GATE["maximum_ordinary_additions"] and control[1] <= GATE["maximum_cut_additions"]
                            and min(coverage[0] / len(positives), coverage[1] / len(ordinary)) >= GATE["minimum_measured_share"])
        summary[arm] = result
    return summary


def score(run):
    state = worker.configure(run)
    incumbent = worker.scorer(state)
    import cv2

    _, common, _, _, _ = plan_sources()
    lock, media = load(run / "freeze.json"), load(run / "media.json")
    require(lock["runner_sha256"] == hash_file(Path(__file__)) and media["freeze_sha256"] == hash_file(run / "freeze.json"),
            "Frozen source or method changed")
    models = common.load_yunet(common.FACE_FILE), cv2.FaceRecognizerSF.create(str(common.APPEARANCE_FILE), "")
    private, public = [], []
    for row in media["rows"]:
        path = run / row["file"]
        baseline = incumbent(path, row)
        require(baseline["short_range"] is None, "Current profile returned a supplemental range")
        additions = extra(path, row, baseline, models)
        clocks = row["clock"]["pts"]
        prior = baseline["coarse"]
        public.append({k: row[k] for k in ("group", "condition", "truth")} | {
            "baseline": [clocks.index(prior[k]) for k in ("start_pts", "end_pts")] if prior else None,
            **{name: [additions[name][k] for k in ("start_frame_index", "end_frame_index")] if additions[name] else None
               for name in ("sparse", "dense")},
            **{f"{name}_status": additions[f"{name}_coverage"]["status"] for name in ("sparse", "dense")},
        })
        private.append({"file": row["file"], "baseline": baseline, "additions": additions})
        print(row["group"], row["condition"], {k: public[-1][k] for k in ("baseline", "sparse", "dense")}, flush=True)
    write(run / "private.json", private)
    result = metrics(public)
    write(run / "public.json", {"schema": SCHEMA, "gate": GATE, "rows": public, "results": result})
    print(result, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "score"))
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    (prepare if args.stage == "prepare" else score)(args.run)
