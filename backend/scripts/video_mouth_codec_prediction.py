# SPDX-License-Identifier: Apache-2.0
"""Compare local H.264 inter-picture prediction over short mouth redraws."""

from __future__ import annotations

import argparse
import subprocess
from collections import Counter
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np

from app.detectors.image.faces import load_yunet
from app.detectors.video.decode import decode_frames, probe_frames
from app.detectors.video.shots import ShotDetector
from scripts import video_mouth_luma_condition as prior
from scripts.video_common_sources_v89 import read, require, sha, write

SCHEMA = "chipmunk.video-mouth-codec-prediction/1"
ARMS = ("inter_coverage_loss", "vector_dispersion")
SPLITS = {"development": 56, "held": 92}
FLOOR = 0.04
SPEC = {
    "source": "One upright native H.264 file using I/P pictures only. H.264 encoder decisions, not decoded color residuals, optical flow, audio or a second file, supply the measured signal.",
    "region": "Classify global four-pixel block-center lattice samples using one native face geometry transform. Target mouth ellipse radii 19 and 12 in the eye plane. Reference ellipses at (-26,-12), (26,-12), and (0,-24) with radii (11,11), (11,11), (15,8). Require at least 12 target and reference lattice samples each.",
    "coverage_arm": "On P pictures with exported vectors and at least 50% reference coverage, subtract target inter-prediction coverage from reference coverage. A missing vector describes exported coverage, not residual magnitude or the identity of the prediction mode.",
    "dispersion_arm": "Require at least half the sampled target and reference sites to have a vector and at least 12 covered samples per region. Remove the median reference vector and convert to eye-plane length. Subtract the reference median log(1+distance/0.5) from the target median.",
    "range": "One best interval of 8, 12, 16 or 20 pictures, with five-picture flanks. Same observed shot, at least 80% measured in the middle and both flanks. Score the smaller positive rise against both flanks. Reuse the exact native interval selection and crop policy of the brightness-conditioned study.",
    "calibration": "The third-largest positive maximum from 56 ordinary GRID development H.264 deliveries sets each point with a fixed floor of 0.04. Require strict exceedance and native interval IoU of at least 0.50 for a correct edit location.",
    "held": "Use the previous 34 matching H.264 neural and untouched deliveries, plus 34 source-matched non-neural local edits, 12 conventional temporal-processing controls, and 12 separate codec/cut controls. New control encodes use the same carrier settings. Every relative presentation timestamp must stay exact. Sources have prior project exposure.",
    "gate": "At least four correct strict LivePortrait intervals across three groups and the strict MuseTalk interval; at least 70% measured coverage in each strict neural family and the ordinary bank; at most two ordinary flags, none on untouched recordings, conventional recompressions or hard cuts.",
}


def freeze(previous, exporter):
    for name, digest in prior.PREVIOUS.items():
        require(sha(previous / name) == digest, "Previous source roster differs")
    archive = read(previous / "freeze.json")
    stress = read(previous / "codec_stress_roster.json")
    require(stress["freeze_sha256"] == prior.PREVIOUS["freeze.json"], "Existing carrier roster differs")
    controls = {"resample_ordinary", "delayed_pixels_ordinary", "same_time_resample", "delayed_pixels"}
    rows = [r for r in archive["rows"] if r["split"] == "development" or r["role"] in controls
            or r["family"] in {"codec", "derived_ordinary"}]
    rows += [{**r, "split": "held", "previous_carrier": True} for r in stress["rows"]]
    require(Counter(r["split"] for r in rows) == SPLITS, "Unexpected source roles")
    return {"schema": SCHEMA, "protocol": SPEC, "rows": rows, "source_rosters": prior.PREVIOUS,
            "code": {p.name: sha(p) for p in (Path(__file__), Path(__file__).with_name("read_video_motion_vectors.c"),
                                              Path(prior.__file__))}, "exporter": sha(exporter),
            "face_weights": sha(prior.geometry.prior.WEIGHT), "ffmpeg": subprocess.check_output(
                ["ffmpeg", "-version"], text=True).splitlines()[0], "opencv": cv2.__version__, "numpy": np.__version__}


def prepare(work, previous, lock):
    folder = work / "transcodes"
    folder.mkdir(mode=0o700)
    archive = read(previous / "freeze.json")
    sources = {(r["family"], r["id"], r["role"]): r for r in archive["stress_sources"]}
    rows = []
    for i, row in enumerate(lock["rows"]):
        path = Path(row["file"])
        require(sha(path) == row["sha"], "Committed source differs")
        if row.get("previous_carrier"):
            original = sources[(row["family"], row["id"], row["role"])]
            source = Path(original["file"])
            require(sha(source) == row["source_sha"] == original["sha"], "Committed paired original differs")
            target = path
        else:
            source = path
            target = folder / f"carrier-{i:03d}.mp4"
        a = probe_frames(source, timeout_s=15)
        if target != path:
            subprocess.run([
                "ffmpeg", "-nostdin", "-v", "error", "-n", "-copyts", "-i", str(source), "-map", "0:v:0", "-an",
                "-fps_mode", "passthrough", "-enc_time_base", str(a.tb), "-frames:v", str(len(a)),
                "-video_track_timescale", str(a.tb.denominator), "-use_editlist", "0", "-c:v", "libx264", "-bf", "0",
                "-pix_fmt", "yuv420p", "-crf", "23", "-threads:v", "1", str(target),
            ], check=True, timeout=30)
        b = probe_frames(target, timeout_s=15)
        require([p * a.tb for p in a.pts] == [p * b.tb for p in b.pts], "Carrier clock differs")
        rows.append({**row, "carrier": str(target), "carrier_sha256": sha(target), "original": str(source),
                     "original_sha256": sha(source), "verified_frames": len(b)})
    write(work / "carriers.json", {"freeze_sha256": sha(work / "freeze.json"), "rows": rows})


def vector_rows(exporter, path, table):
    output = subprocess.check_output([str(exporter), str(path)], text=True, timeout=20)
    lines = iter(output.splitlines())
    kind, numerator, denominator = next(lines).split()
    base = Fraction(int(numerator), int(denominator))
    require(kind == "T" and base == table.tb, "Motion-vector time base differs")
    for index, expected in enumerate(table.pts):
        kind, pts, frame_type, width, height, count = next(lines).split()
        require(kind == "F" and int(pts) == expected and frame_type in {"I", "P"}
                and (int(width), int(height)) == (table.width, table.height), "Motion-vector frame table differs")
        data = np.empty((int(count), 8), dtype=np.int32)
        for block in data:
            prefix, *numbers = next(lines).split()
            require(prefix == "V", "Bad motion-vector record")
            block[:] = list(map(int, numbers))
        yield index, frame_type, data
    require(next(lines, None) is None, "Extra motion-vector pictures")


def block_plane(vectors, width, height):
    """Map nonoverlapping vector rectangles to their covered lattice sites."""
    if not len(vectors):
        return None
    x, y, w, h, mx, my, scale, source = vectors.T
    require(np.all(np.isin(w, (4, 8, 16))) and np.all(np.isin(h, (4, 8, 16))) and np.all(scale > 0)
            and np.all(source < 0), "Unexpected P-picture vector layout")
    left, top = x - w // 2, y - h // 2
    require(not np.any(left % 4) and not np.any(top % 4), "Vector grid differs")
    columns, rows = (width + 1) // 4, (height + 1) // 4
    plane = np.full((rows, columns, 2), np.nan)
    for dx in range(4):
        for dy in range(4):
            xx, yy = left // 4 + dx, top // 4 + dy
            keep = (dx * 4 < w) & (dy * 4 < h) & (xx >= 0) & (xx < columns) & (yy >= 0) & (yy < rows)
            indices = yy[keep] * columns + xx[keep]
            require(len(np.unique(indices)) == len(indices) and not np.isfinite(plane[yy[keep], xx[keep], 0]).any(),
                    "Overlapping vector blocks")
            plane[yy[keep], xx[keep], :] = np.column_stack((mx[keep] / scale[keep], my[keep] / scale[keep]))
    return plane


def measure(plane, located):
    scores = np.full(2, np.nan)
    _, matrix, mouth, _ = located
    y, x = np.mgrid[:plane.shape[0], :plane.shape[1]] * 4 + 2
    relative = np.stack((x, y), axis=-1) @ matrix[:, :2].T + matrix[:, 2] - mouth
    xx, yy = relative.transpose(2, 0, 1)
    target = (xx / 19) ** 2 + (yy / 12) ** 2 <= 1
    surround = np.zeros_like(target)
    for cx, cy, rx, ry in ((-26, -12, 11, 11), (26, -12, 11, 11), (0, -24, 15, 8)):
        surround |= ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1
    surround &= ~target
    covered = np.isfinite(plane[:, :, 0])
    total_target, total_ref = int(target.sum()), int(surround.sum())
    if min(total_target, total_ref) < 12:
        return scores
    covered_target, covered_ref = target & covered, surround & covered
    n_target, n_ref = int(covered_target.sum()), int(covered_ref.sum())
    if n_ref * 2 < total_ref:
        return scores
    scores[0] = n_ref / total_ref - n_target / total_target
    if min(n_target, n_ref) >= 12 and n_target * 2 >= total_target:
        ref, local = plane[covered_ref], plane[covered_target]
        offset = np.median(ref, axis=0)
        a, b = (np.log1p(np.linalg.norm((part - offset) @ matrix[:, :2].T, axis=1) / 0.5)
                for part in (local, ref))
        scores[1] = np.median(a) - np.median(b)
    return scores


def inspect(path, exporter, detector):
    table = probe_frames(path, timeout_s=15)
    require(not (table.truncated or table.interpolated or table.rotation) and 40 <= len(table) <= 100,
            "Expected bounded native clock")
    geometry = prior.geometry.prior.panel
    shots, anchors = [], []
    samples = np.full((len(table), len(ARMS)), np.nan)
    track = ShotDetector(table)
    frame_types = Counter()
    pictures = decode_frames(path, table, None, timeout_s=35)
    for frame, (index, frame_type, data) in zip(pictures, vector_rows(exporter, path, table), strict=True):
        require(frame.index == index and frame.pts_verified and frame.pts == table.pts[index],
                "Picture and decoder motion clock differ")
        track.update(index, frame.rgb)
        shots.append(len(track.boundaries))
        frame_types[frame_type] += 1
        located = geometry.prior.aligned_face(frame.rgb, detector)
        proof = None
        if located is not None:
            x0, y0, x1, y1 = geometry.native_bounds(located)
            proof = {"frame": index, "pts": frame.pts, "crop_xyxy": [x0, y0, x1, y1],
                     "rgb_sha256": geometry.raw_digest(frame.rgb),
                     "crop_sha256": geometry.raw_digest(frame.rgb[y0:y1, x0:x1])}
            plane = block_plane(data, table.width, table.height) if frame_type == "P" else None
            if plane is not None:
                samples[index] = measure(plane, located)
        anchors.append(proof)
    require(len(anchors) == len(table), "Incomplete picture decode")
    proposed = prior.ranges(table, np.asarray(shots), samples, anchors)
    reading = dict(zip(ARMS, proposed.values(), strict=True))
    return table, reading, dict(zip(ARMS, map(int, np.isfinite(samples).sum(axis=0)))), dict(frame_types)


def points(rows):
    return {arm: max(FLOOR, (sorted((r["arms"][arm]["score"] or 0 for r in rows), reverse=True) + [0] * 3)[2])
            for arm in ARMS}


def score(work, exporter, phase):
    inputs = work / "carriers.json"
    roster = read(inputs)
    require(roster["freeze_sha256"] == sha(work / "freeze.json"), "Carrier preparation differs")
    detector = load_yunet(prior.geometry.prior.WEIGHT)
    output = []
    for row in roster["rows"]:
        if row["split"] != phase:
            continue
        path = Path(row["carrier"])
        require(sha(path) == row["carrier_sha256"], "Prepared carrier bytes differ")
        table, arms, support, types = inspect(path, exporter, detector)
        truth = [str(table.pts[i] * table.tb) for i in row["edit"]] if row.get("edit") else row.get("native_truth")
        if truth:
            require(all(Fraction(value) in {p * table.tb for p in table.pts} for value in truth), "Edit clock differs")
        output.append({**row, "truth": truth, "arms": arms, "frame_support": support, "types": types,
                       "native_frames": len(table)})
        if len(output) % 8 == 0:
            print(phase, len(output), flush=True)
    require(len(output) == SPLITS[phase], "Incomplete held or development results")
    destination = work / f"{phase}.json"
    write(destination, {"carriers_sha256": sha(inputs), "rows": output})
    if phase == "development":
        write(work / "points.json", {"development_sha256": sha(destination), "points": points(output)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "prepare", *SPLITS))
    for key in ("work", "previous", "exporter"):
        parser.add_argument(key, type=Path)
    args = parser.parse_args()
    lock = freeze(args.previous, args.exporter)
    freeze_path = args.work / "freeze.json"
    if args.phase == "freeze":
        args.work.mkdir(mode=0o700)
        write(freeze_path, lock)
        return
    require(read(freeze_path) == lock, "Frozen procedure differs")
    if args.phase == "prepare":
        prepare(args.work, args.previous, lock)
        return
    if args.phase == "held":
        dev = args.work / "development.json"
        require(read(args.work / "points.json") == {"development_sha256": sha(dev), "points": points(read(dev)["rows"])},
                "Development points differ")
    cv2.setNumThreads(1)
    score(args.work, args.exporter, args.phase)


if __name__ == "__main__":
    main()
