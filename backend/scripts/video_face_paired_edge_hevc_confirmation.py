# SPDX-License-Identifier: Apache-2.0
"""Transfer the fixed paired-edge face comparison to disjoint 720p30 HEVC deliveries."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

from scripts import screen_video_native_background_challenge as full
from scripts import video_face_720p30_transfer as native
from scripts import video_face_paired_edge_transfer as candidate

ROOT = candidate.ROOT
DATA = candidate.DATA
CONTROLS = DATA / "video-face-near-head-transfer-2026-11-12-v2"
CONTROLS_SELECTION = DATA / "video-face-scale-context-holdout-2026-11-12-v2"
SCHEMA = "chipmunk.video-face-paired-edge-hevc-confirmation/1"
POINT = .03176017255556951
GATE = {"min_added": 2, "min_groups": 2, "min_assessed_routed_share": .60, "max_ordinary_ranges": 0,
        "max_wrong_edit_ranges": 0}
ENCODING = {"codec_name": "hevc", "codec_tag_string": "hvc1", "pix_fmt": "yuv420p", "width": 1280,
            "height": 720, "r_frame_rate": "30/1", "nb_frames": "120"}
ENCODE = ("-map", "0:v:0", "-an", "-c:v", "libx265", "-preset", "ultrafast", "-crf", "26", "-threads", "1",
          "-x265-params", "pools=1:frame-threads=1:log-level=error", "-pix_fmt", "yuv420p", "-tag:v", "hvc1",
          "-video_track_timescale", "30000", "-movflags", "+faststart")
read, save, sha, require = candidate.read, candidate.save, candidate.sha, candidate.require


def lock_inputs(first, original):
    prior = read(first)
    require(prior["point"]["point"] == POINT and prior["freeze"]["code"] == candidate.bound_paths(),
            "Candidate or development point differs from the locked H.264 experiment")
    require(not prior["result"]["passed"], "Unexpected advancement status of initial H.264 gate")
    sources = read(original / "freeze.json")["groups"]
    sparse, challenge, dual = (DATA / candidate.SOURCES[role][0] / "freeze.json" for role in
                               ("selection_edits", "held_edits_and_cuts", "held_two_edits"))
    old_groups = {row["index"] for path in (sparse, dual) for row in read(path)["sources"]}
    old_groups.update(row["pin"]["index"] for row in read(challenge)["swaps"])
    prior_actors = {row["source_group"] for row in read(CONTROLS_SELECTION / "freeze.json")["sources"]}
    held_actors = {row["source_group"] for row in read(CONTROLS / "freeze.json")["crema_originals"]}
    require(len(set(sources)) == 4 and not (set(sources) & old_groups), "Repeated RTFS source group")
    require(len(held_actors) == 10 and not (held_actors & prior_actors), "Repeated CREMA-D actor group")
    paths = {"old_selection_edit_lock": sparse, "old_held_single_lock": challenge, "old_held_dual_lock": dual,
             "old_selection_control_lock": CONTROLS_SELECTION / "freeze.json", "held_rtfs_and_cameras_lock": original / "freeze.json",
             "held_rtfs_and_cameras_media": original / "media.json", "held_crema_lock": CONTROLS / "freeze.json",
             "held_crema_media": CONTROLS / "media.json", "original_experiment": first,
             "full_worker": Path(full.__file__), "confirmation_worker": Path(__file__)}
    for folder in (original, CONTROLS):
        require(read(folder / "media.json")["freeze_sha256"] == sha(folder / "freeze.json"), "Archived media ledger changed")
    return {"schema": SCHEMA, "pins": {name: sha(path) for name, path in paths.items()},
            "candidate_code": prior["freeze"]["code"], "weights": prior["freeze"]["weights"],
            "point": POINT, "gate": GATE, "encoding": ENCODING, "encoder_flags": ENCODE,
            "source_count": {"publisher_swaps": 4, "crema_actor_pairs": 5, "extra_camera_recordings": 3},
            "method": "Unchanged range search and selection point on HEVC encodes of the disjoint held source banks. "
                      "Use actual Full face ranges on each new delivery. A correct edit needs .5 temporal IoU."}


def ffprobe(path):
    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                            "stream=" + ",".join(ENCODING), "-of", "json", str(path)], check=True,
                           timeout=10, capture_output=True)
    return json.loads(probe.stdout)["streams"]


def prepare(run, first, original):
    run.mkdir(mode=0o700)
    save(run / "freeze.json", lock_inputs(first, original))
    rows = []
    for bank, folder in (("publisher_and_cameras", original), ("cutaways", CONTROLS)):
        for item in read(folder / "media.json")["rows"]:
            source = folder / "media" / item["file"] if folder == CONTROLS else folder / item["file"]
            require(sha(source) == item["sha256"], "Archived H.264 delivery changed")
            path = run / f"hevc-{source.name}"
            subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-threads", "1", "-i", str(source),
                            *ENCODE, str(path)], check=True, timeout=35)
            require(ffprobe(path) == [ENCODING], "Confirmation stream profile differs")
            truth = [item["truth"]] if bank == "publisher_and_cameras" and item["truth"] else []
            row = native.describe(path, item["group"], item["condition"], truth)
            require(row["clock"]["pts"] == item["clock"]["pts"], "Transfer changed authored native picture positions")
            rows.append({**row, "bank": bank, "input_h264_sha256": item["sha256"]})
    require(len(rows) == len({r["sha256"] for r in rows}) == 61, "Incomplete held confirmation bank")
    save(run / "media.json", {"freeze_sha256": sha(run / "freeze.json"), "rows": rows})
    print("Prepared 61 HEVC deliveries", flush=True)


def iou(pair, truth):
    return native.iou(pair, truth) if pair else 0.


def summarize(rows):
    conditions = {}
    for row in rows:
        label = row["bank"] + "/" + row["condition"]
        span, truth = row["selected"], row["truth"]
        best = max((iou(value, truth[0]) for value in row["prior"]), default=0.) if truth else 0.
        added = bool(truth and span and iou(span, truth[0]) >= .5 and best < .5)
        count = conditions.setdefault(label, {key: 0 for key in
            ("files", "routed", "assessed", "ordinary_prior", "ordinary_added_ranges", "new_ordinary_files", "wrong_edit_ranges",
             "useful_before", "useful_combined", "additional_useful", "exact_before", "exact_combined")})
        count["files"] += 1
        count["routed"] += bool(row["coarse"])
        count["assessed"] += row["status"] == "assessed"
        count["ordinary_prior"] += bool(not truth and row["prior"])
        count["ordinary_added_ranges"] += bool(not truth and span)
        count["new_ordinary_files"] += bool(not truth and span and not row["prior"])
        count["wrong_edit_ranges"] += bool(truth and span and iou(span, truth[0]) < .5)
        count["useful_before"] += best >= .5
        count["useful_combined"] += best >= .5 or added
        count["additional_useful"] += added
        count["exact_before"] += bool(truth and truth[0] in row["prior"])
        count["exact_combined"] += bool(truth and (truth[0] in row["prior"] or span == truth[0]))
    edits = [row for row in rows if row["truth"]]
    added = [row for row in edits if iou(row["selected"], row["truth"][0]) >= .5 and
             max((iou(value, row["truth"][0]) for value in row["prior"]), default=0.) < .5]
    groups = len({row["group"] for row in added})
    routed = sum(bool(row["coarse"]) for row in edits)
    assessed = sum(row["status"] == "assessed" for row in edits)
    gate = {"added": len(added) >= GATE["min_added"], "groups": groups >= GATE["min_groups"],
            "assessment": bool(routed and assessed >= GATE["min_assessed_routed_share"] * routed),
            "ordinary": sum(item["ordinary_added_ranges"] for item in conditions.values()) <= GATE["max_ordinary_ranges"],
            "wrong_edit": sum(item["wrong_edit_ranges"] for item in conditions.values()) <= GATE["max_wrong_edit_ranges"]}
    return {"counts": conditions, "gate": gate, "passed": all(gate.values()), "edits": len(edits), "routed_edits": routed,
            "assessed_edits": assessed, "added_groups": groups}


def score(run, first, original):
    frozen = read(run / "freeze.json")
    expected = json.loads(candidate.encode(lock_inputs(first, original)))
    media = read(run / "media.json")
    require(frozen == expected and media["freeze_sha256"] == sha(run / "freeze.json"), "Frozen transfer changed")
    state = full.configure(run)
    os.environ["VERITAS_VIDEO_FACE_SECOND_RETURN_REVIEW"] = "true"
    compare = full.scorer(state)
    import cv2

    from app.detectors.image.faces import load_yunet

    weights = ROOT / "models/weights"
    require({name: sha(weights / name) for name in candidate.WEIGHTS} == frozen["weights"], "Weights changed")
    detector = load_yunet(weights / "face_detection_yunet_2023mar.onnx")
    model = cv2.FaceRecognizerSF.create(str(weights / "face_recognition_sface_2021dec.onnx"), "")
    results = []
    for index, item in enumerate(media["rows"]):
        path = run / item["file"]
        result = compare(path, item)
        # The secondary brief-face check only accepts the H.264 delivery profiles.
        require(result["short_range"] is None and result["short_coverage"]["status"] == "not_applicable",
                "An unexpected 720p30 HEVC short check appeared in the comparator")
        spans = candidate.intervals(result, item["clock"]["pts"])
        coarse = spans[0] if result["coarse"] else None
        response = candidate.measure(path, item, {"id": index, "sha256": item["sha256"], "coarse": coarse}, detector, model)
        guarded = response["guarded"]
        chosen = guarded["frames"] if guarded and guarded["score"] > POINT else None
        row = {**{key: item[key] for key in ("group", "condition", "bank", "truth", "sha256", "input_h264_sha256")},
               "prior": spans, "coarse": coarse, "status": response["status"], "selected": chosen,
               "raw": response["raw"], "guarded": guarded, "native_reads": response["native_reads"]}
        results.append(row)
        print(index + 1, item["condition"], item["group"], coarse, response["status"], chosen, flush=True)
    save(run / "receipt.json", {"freeze": frozen, "media_sha256": sha(run / "media.json"), "rows": results,
                                "result": summarize(results)})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "score"))
    parser.add_argument("run", type=Path)
    parser.add_argument("original_h264_experiment", type=Path)
    parser.add_argument("source_study", type=Path)
    args = parser.parse_args()
    (prepare if args.phase == "prepare" else score)(args.run, args.original_h264_experiment, args.source_study)
