"""Replay partial face-change results on real files with a detector-independent manifest."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import subprocess
from fractions import Fraction
from pathlib import Path

from app.bench import independent
from app.bench.independent_inputs import INDEX_SCHEMA, verify_inputs
from app.bench.independent_locations import TRUTH_SCHEMA, count_events
from scripts import video_common_sources_v89 as io
from scripts import video_face_local_fit as fit

LICENSE = {"id": "CC BY-SA 4.0 (RTFS)", "evaluation_permitted": True, "redistribution_permitted": True}
DIMENSIONS = 640, 360
FRAMES = 100


def report(receipt):
    with gzip.open(receipt, "rt") as stream:
        return json.load(stream)


def correspondence(path, prediction):
    import numpy as np

    from app.detectors.video.face_detail_inputs import align

    anchors = prediction["anchors"]
    ordered = sorted(anchors.values(), key=lambda row: row["index"])
    frames = subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-threads", "1", "-i", str(path),
                             "-vf", "select=" + "+".join(f"eq(n\\,{row['index']})" for row in ordered),
                             "-pix_fmt", "rgb24", "-fps_mode", "passthrough", "-threads", "1", "-f", "rawvideo", "pipe:1"],
                            capture_output=True, timeout=20, check=True).stdout
    width, height = DIMENSIONS
    stride = 3 * width * height
    if len(frames) != stride * len(ordered):
        raise ValueError("Independent decoder did not return the selected original pictures")
    for i, item in enumerate(ordered):
        raw = frames[i*stride:(i+1)*stride]
        if hashlib.sha256(raw).hexdigest() != item["decoded_rgb_sha256"]:
            raise ValueError("Independent picture bytes differ from an evidence anchor")
        x, y, w, h = (item["box"][k] for k in "xywh")
        if not (0 <= x < x + w <= width and 0 <= y < y + h <= height):
            raise ValueError("Face crop exceeds the original picture")
        frame = np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3)
        cropped = align(frame, item["eyes"])
        if cropped is None or hashlib.sha256(cropped.tobytes()).hexdigest() != item["aligned_rgb_sha256"]:
            raise ValueError("Evidence crop does not come from the matching original frame")
    return len(ordered)


def picture_ticks(path):
    decoded = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                              "-show_entries", "stream=width,height,time_base,codec_name,pix_fmt:frame=best_effort_timestamp",
                              "-show_frames", "-show_streams", "-of", "json", str(path)],
                             capture_output=True, timeout=20, check=True)
    result = json.loads(decoded.stdout)
    stream, = result["streams"]
    ticks = [int(frame["best_effort_timestamp"]) for frame in result["frames"]]
    base = stream["time_base"]
    if ((stream["width"], stream["height"]) != DIMENSIONS or (stream["codec_name"], stream["pix_fmt"]) != ("h264", "yuv420p")
            or len(ticks) != FRAMES or any(t * Fraction(base) != Fraction(i, 25) for i, t in enumerate(ticks))):
        raise ValueError("A prepared delivery lacks its expected native clock or geometry")
    return base, ticks


def audit(receipt, prepared, output):
    """Open each native input and decode all three anchors for each selected proposal."""
    value = report(receipt)
    choice = value["lock"]["choices"][value["lock"]["selected"]]
    anchors, clock_reads = 0, 0
    for row in value["rows"]:
        path = prepared / row["file"]
        if io.sha(path) != row["input_sha256"]:
            raise ValueError("Prepared delivery does not match the comparison receipt")
        base, ticks = picture_ticks(path)
        clock_reads += len(ticks)
        candidate = row["candidate"][choice["arm"]]
        if row["existing"] or candidate is None or candidate["score"] <= choice["point"]:
            continue
        low, high = candidate["range"]
        before, middle, after = (candidate["anchors"][side] for side in ("before", "during", "after"))
        if (candidate["time_base"] != base or candidate["ticks"] != [ticks[low], ticks[high]]
                or not before["index"] < low <= middle["index"] < high == after["index"]
                or any(ticks[item["index"]] != item["pts"] for item in (before, middle, after))):
            raise ValueError("Evidence range or picture does not match its native time")
        anchors += correspondence(path, candidate)
    io.write(output, {"inputs": len(value["rows"]), "original_frame_clocks": clock_reads, "verified_anchors": anchors,
                      "receipt_sha256": io.sha(receipt)})


def split(receipt, prepared, publisher, output):
    """Export benchmark inputs, frame truth and two response sets."""
    value = report(receipt)
    prepared_manifest = io.read(prepared / "manifest.json")
    recorded = {row["file"]: row for row in prepared_manifest["rows"]}
    if len(recorded) != len(value["rows"]):
        raise ValueError("Prepared comparison manifest does not cover the same panel")
    choice = value["lock"]["choices"][value["lock"]["selected"]]
    truth, manifest_rows, files = [], [], []
    systems = {"current": [], "candidate": []}
    output.mkdir()
    root = prepared.parent.resolve()
    if publisher.parent.resolve() != root:
        raise ValueError("Place the prepared inputs and publisher files under the same review root")
    for row in value["rows"]:
        planned = recorded[row["file"]]
        if any(planned[key] != row[key] for key in ("input_sha256", "source_sha256", "group", "family",
                                                    "delivery", "truth", "edit_frames")):
            raise ValueError("Prepared comparison differs from the released truth or tested source bytes")
        name, source_id = row["file"], planned["source_index"]
        case_id = Path(name).stem
        family, group, region = row["family"], row["group"], row["truth"]
        manifest_rows.append({"id": case_id, "modality": "video", "group": group, "split": "confirmation",
                              "label": int(bool(region)), "family": family, "condition": f"{row['delivery']}/{family}",
                              "input_sha256": row["input_sha256"], "source_sha256": row["source_sha256"],
                              "license": LICENSE, "lineage": [group],
                              "localization_truth": {"unit": "frames", "source_units": FRAMES,
                                                       "target_units": region[1] - region[0] if region else 0}})
        key = {"id": case_id, "input_sha256": row["input_sha256"]}
        truth.append(key | {"extent": [FRAMES], "regions": [region] if region else []})
        files.append({"id": case_id, "input": (prepared / name).relative_to(root).as_posix(),
                      "source": (publisher / f"{source_id:02d}-ordinary.mp4").relative_to(root).as_posix()})
        for system, results in systems.items():
            enhanced = system == "candidate"
            marked = fit.selected(row, choice["arm"], choice["point"]) if enhanced else row["existing"]
            checked = row["existing_assessed"] or (enhanced and row["candidate"][choice["arm"]] is not None)
            if checked:
                results.append(key | {"status": "ok", "response": int(bool(marked)),
                                      "localization": {"status": "ok", "regions": [marked] if marked else []}})
            else:
                reason = "No eligible face track with the required same-scene support"
                results.append(key | {"status": "not_applicable", "response": None, "reason": reason,
                                      "localization": {"status": "not_applicable", "reason": reason}})
    manifest = {"schema": independent.SCHEMA, "purpose": "Locate temporary facial changes in fixed source coordinates.",
                "cases": manifest_rows}
    identity = independent.audit_manifest(manifest)["manifest_sha256"]
    index = {"schema": INDEX_SCHEMA, "manifest_sha256": identity, "cases": files}
    binding = {"manifest_sha256": identity, "split": "confirmation"}
    targets = binding | {"schema": TRUTH_SCHEMA, "cases": truth}
    for name, data in (("manifest", manifest), ("file-index", index), ("truth", targets),
                       ("input-verification", verify_inputs(manifest, index, root))):
        io.write(output / f"{name}.json", data)
    weights = sorted(pin for asset in value["confirmation"]["weights"].values() for pin in asset)
    for system, observations in systems.items():
        identity = independent.digest([system, value["lock_sha256"], value["confirmation"]["readout_sha256"]])
        responses = binding | {"system": system, "implementation_sha256": identity, "method_kind": "weighted",
                               "weight_sha256": weights, "score_kind": "raw", "response_units": "Reported range (1 or 0)",
                               "predictions": observations}
        events = count_events(manifest, targets, responses, minimum_iou=.50, minimum_recall=.50)
        io.write(output / f"{system}.json", responses)
        io.write(output / f"{system}-events.json", events)


def verify_worker(receipt, worker_path):
    """Recount the additions against the broader appearance and timing checks."""
    value, worker = report(receipt), io.read(worker_path)
    choice = value["lock"]["choices"][value["lock"]["selected"]]
    arm, point = choice["arm"], choice["point"]
    frozen = {row["file"]: row for row in value["rows"]}
    proposed = {name for name, row in frozen.items()
                if (candidate := row["candidate"][arm]) and candidate["score"] > point}
    reviewed = {row["file"]: row for row in worker["cases"]}
    other = reviewed.keys() - proposed
    if (not proposed <= reviewed.keys() or len(reviewed) != len(worker["cases"]) or len(other) != 1
            or any(frozen[name]["family"] != "face_blur" or reviewed[name]["added_range"] is not None for name in other)):
        raise ValueError("App replay must cover all proposals and the paired below-point blur control")

    profiles = {profile: {"replayed": 0, "additional_correct": 0, "ordinary_new_flags": 0, "edited_wrong_ranges": 0}
                for profile in ("direct", "tandem")}
    for name, row in reviewed.items():
        labeled = frozen[name]
        fields = ("input_sha256", "group", "family", "delivery", "truth")
        if any(row[key] != labeled[key] for key in fields):
            raise ValueError("App replay differs from the labeled input")
        observed = row["added_range"]
        prior = row["prior_ranges"]
        if any(len(bounds) != 2 or not 0 <= bounds[0] < bounds[1] <= FRAMES
               for bounds in prior + ([observed] if observed else [])):
            raise ValueError("App replay contains an invalid native frame range")
        if observed and (name not in proposed or observed != labeled["candidate"][arm]["range"]):
            raise ValueError("App returned a different range from the frozen candidate")
        useful = fit.overlap(observed, row["truth"]) >= .50
        new = useful and not any(fit.overlap(old, row["truth"]) >= .50 for old in prior)
        if row["new_useful_range"] != new:
            raise ValueError("App improvement differs from the labeled bounds")
        metrics = profiles[row["delivery"]]
        metrics["replayed"] += 1
        metrics["additional_correct"] += new
        metrics["ordinary_new_flags"] += observed is not None and row["truth"] is None
        metrics["edited_wrong_ranges"] += observed is not None and row["truth"] is not None and not useful
    summary = {name: sum(profile[name] for profile in profiles.values()) for name in profiles["direct"]}
    if profiles != worker["profiles"] or summary != worker["summary"]:
        raise ValueError("Stored worker summary does not reproduce the original frame ranges")
    print(json.dumps({"summary": summary, "profiles": profiles}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    for action, args in (("audit", ("receipt", "prepared", "output")),
                         ("export", ("receipt", "prepared", "publisher", "output")), ("verify-worker", ("receipt", "worker"))):
        sub = commands.add_parser(action)
        for name in args:
            sub.add_argument(name, type=Path)
    args = parser.parse_args()
    if args.action == "audit":
        audit(args.receipt, args.prepared, args.output)
    elif args.action == "verify-worker":
        verify_worker(args.receipt, args.worker)
    else:
        split(args.receipt, args.prepared.resolve(), args.publisher.resolve(), args.output)
