# SPDX-License-Identifier: Apache-2.0
"""Evaluate local color-residual coupling on source-aligned mouth redraws."""

from __future__ import annotations

import argparse
import subprocess
from itertools import pairwise
from pathlib import Path

import cv2
import numpy as np

from app.detectors.image.faces import load_yunet
from app.detectors.video.decode import decode_frames, probe_frames
from app.detectors.video.shots import ShotDetector
from scripts import video_mouth_contour_phase as earlier
from scripts.video_common_sources_v89 import read, require, sha, write

ARMS = ("luma_control", "color_coupling", "conditioned_chroma")
FLOOR = dict.fromkeys(ARMS, 0.03)
SPEC = {
    "input": "One upright short upload, native RGB pixels, verified source picture clock and observed single-face geometry. No audio, earlier copy, edit mask or supplied range enters measurement.",
    "patches": "Eight sectors of 5 by 5 sample points around two mouth-centered elliptical rails, at 1.0 and 1.7 times the 15 by 8 normalized eye-unit radii. Gaussian band pass at 0.75 and 1.9 native pixels. Neighbor rails require luma residual RMS of at least 0.25 and within a factor of 3.",
    "arms": "Local-minus-outer change in log luma residual energy, luma/chroma regression coupling, or log chroma residual energy after separately regressing each chroma axis on luma. Values use RGB-decoder YCrCb conversion and do not establish sensor identity.",
    "range": "Scan 8, 12, 16 and 20 pictures with five-picture flanks in one observed shot. Each sector requires at least 80% observed pictures per window, with a same-direction change against each flank. At least three consistent sectors. Keep one strongest range, with separate arms and no selection using held labels.",
    "new_controls": "First committed original from each of 12 additional speaker groups. Add whole-picture H.264 4:2:0 CRF 28 and ordinary local chroma denoising in the central 16 pictures. Denoising uses a mouth-centered 26 by 15 elliptical feather and 35% of a 1.3-native-pixel chroma blur with two-picture end ramps. Preserve every original timestamp. These controls are held, never used to choose the points.",
    "decision": "Per arm, the larger of 0.03 and the third-largest positive ordinary development score. Flag when strictly larger. Correct range uses native time IoU at least 0.50.",
    "gate": "At least four source-event-aligned redraws localized across three groups, at least 70% coverage in each main class, at most two ordinary flags, and zero untouched, recompression or cut flags. Color arms must also exceed the luma comparator's useful locations without extra ordinary flags. No claim of generator-independent validation.",
}


def plan(lombard, fresh):
    lock = earlier.plan(lombard, fresh)
    control_groups = {}
    for row in lock["rows"]:
        if row["family"] == "fresh":
            control_groups.setdefault(row["group"], row)
    require(len(control_groups) == 12, "Expected 12 independently grouped native controls")
    lock.update(
        protocol=SPEC,
        tool_sha256=sha(Path(__file__)),
        input_planner_sha256=sha(Path(earlier.__file__)),
        geometry_sha256=sha(Path(earlier.panel.prior.__file__)),
        new_control_sources=list(control_groups.values()),
        ffmpeg=subprocess.check_output(["ffmpeg", "-version"], text=True).splitlines()[0],
    )
    return lock


def sample_locations():
    theta = (np.arange(8) + 0.5) * (np.pi / 4)
    x, y = np.meshgrid(np.arange(-2, 3), np.arange(-2, 3))
    patch = np.stack((x, y), axis=-1).reshape(25, 2)
    centers = np.asarray([1.0, 1.7])[:, None, None] * np.stack(
        (15 * np.cos(theta), 8 * np.sin(theta)), axis=-1
    )
    return centers[:, :, None, :] + patch


LOCATIONS = sample_locations()


def denoise(rgb, geometry, strength):
    _, matrix, mouth, _ = geometry
    x0, y0, x1, y1 = earlier.panel.native_bounds(geometry)
    if x1 - x0 < 12 or y1 - y0 < 12:
        return rgb
    x, y = np.meshgrid(np.arange(x0, x1), np.arange(y0, y1))
    relative = np.stack((x, y), axis=-1) @ matrix[:, :2].T + matrix[:, 2] - mouth
    radius = np.linalg.norm(relative / [26, 15], axis=-1)
    alpha = strength * np.clip((1 - radius) * 2, 0, 1)[..., None].astype(np.float32)
    region = rgb[y0:y1, x0:x1]
    color = cv2.cvtColor(region.astype(np.float32) / 255, cv2.COLOR_RGB2YCrCb)
    color[..., 1:] += alpha * (cv2.GaussianBlur(color[..., 1:], (0, 0), 1.3) - color[..., 1:])
    changed = np.clip(np.rint(255 * cv2.cvtColor(color, cv2.COLOR_YCrCb2RGB)), 0, 255).astype(np.uint8)
    output = rgb.copy()
    output[y0:y1, x0:x1] = np.where(alpha > 0, changed, region)
    return output


def native_time(table):
    return [t * table.tb for t in table.pts]


def prepare_controls(lock, run):
    cv2.setNumThreads(1)
    detector = load_yunet(earlier.WEIGHT)
    output = run / "controls"
    output.mkdir(mode=0o700)
    rows = []
    for row in lock["new_control_sources"]:
        source = Path(row["file"])
        require(sha(source) == row["sha"], "Frozen ordinary source differs")
        table = probe_frames(source, timeout_s=15)
        timeline = native_time(table)
        step = timeline[1] - timeline[0]
        require(
            not (table.truncated or table.interpolated or table.rotation)
            and table.tb.numerator == 1
            and timeline[0] == 0
            and step > 0
            and all(b - a == step for a, b in pairwise(timeline)),
            "Expected complete zero-origin source clocks for new ordinary controls",
        )
        base = ["ffmpeg", "-nostdin", "-v", "error", "-n", "-threads", "1"]
        tail = ["-an", "-threads:v", "1", "-video_track_timescale", str(table.tb.denominator)]
        recompressed = output / f"{row['id']:02d}-recompressed.mp4"
        subprocess.run(
            [
                *base,
                "-copyts",
                "-i",
                str(source),
                "-map",
                "0:v:0",
                "-fps_mode",
                "passthrough",
                "-c:v",
                "libx264",
                "-crf",
                "28",
                "-pix_fmt",
                "yuv420p",
                *tail,
                str(recompressed),
            ],
            check=True,
            timeout=35,
        )
        locally_filtered = output / f"{row['id']:02d}-denoised.mov"
        start, end = len(table) // 2 - 8, len(table) // 2 + 8
        pictures, changed = bytearray(), []
        for frame in decode_frames(source, table, None, timeout_s=25):
            require(
                frame.pts_verified and frame.pts == table.pts[frame.index], "Ordinary input clock differs"
            )
            rgb = frame.rgb
            if start <= frame.index < end:
                geometry = earlier.panel.prior.aligned_face(rgb, detector)
                if geometry is not None:
                    alpha = 0.35 * min(2, frame.index - start + 1, end - frame.index) / 2
                    rgb = denoise(rgb, geometry, alpha)
                    if np.any(rgb != frame.rgb):
                        changed.append(frame.index)
            pictures.extend(rgb.tobytes())
        require(len(pictures) == len(table) * table.width * table.height * 3, "Missing source pictures")
        subprocess.run(
            [
                *base,
                "-f",
                "rawvideo",
                "-pixel_format",
                "rgb24",
                "-video_size",
                f"{table.width}x{table.height}",
                "-framerate",
                str(1 / step),
                "-i",
                "pipe:0",
                "-c:v",
                "libx264rgb",
                "-qp",
                "0",
                "-bf",
                "0",
                "-preset",
                "veryfast",
                *tail,
                str(locally_filtered),
            ],
            input=pictures,
            check=True,
            timeout=45,
        )
        for role, path in (("reencode", recompressed), ("local_chroma_denoise", locally_filtered)):
            delivered = probe_frames(path, timeout_s=15)
            require(
                native_time(delivered) == timeline
                and (delivered.width, delivered.height) == (table.width, table.height),
                "New ordinary delivery did not preserve native clock and geometry",
            )
            rows.append(
                {
                    **row,
                    "family": "new_ordinary",
                    "role": role,
                    "file": str(path),
                    "sha": sha(path),
                    "source_sha256": row["sha"],
                    "ordinary_edit_frames": [start, end] if role == "local_chroma_denoise" else None,
                    "changed_pictures": changed if role == "local_chroma_denoise" else None,
                }
            )
        print("prepare", row["id"], len(changed), flush=True)
    write(run / "controls.json", {"freeze_sha256": sha(run / "freeze.json"), "rows": rows})


def frame_measure(rgb, geometry):
    _, matrix, mouth, _ = geometry
    inverse = cv2.invertAffineTransform(matrix)
    positions = (LOCATIONS + mouth) @ inverse[:, :2].T + inverse[:, 2]
    height, width = rgb.shape[:2]
    valid = ((positions >= 7) & (positions < np.asarray([width - 7, height - 7]))).all(axis=(0, 2, 3))
    if not valid.any():
        return np.full((8, len(ARMS)), np.nan)
    points = positions[:, valid]
    xmin, ymin = np.floor(points.min(axis=(0, 1, 2))).astype(int) - 7
    xmax, ymax = np.ceil(points.max(axis=(0, 1, 2))).astype(int) + 8
    color = cv2.cvtColor(rgb[ymin:ymax, xmin:xmax].astype(np.float32) / 255, cv2.COLOR_RGB2YCrCb)
    residual = cv2.GaussianBlur(color, (0, 0), 0.75) - cv2.GaussianBlur(color, (0, 0), 1.9)
    local = (points - [xmin, ymin]).reshape((-1, 25, 2)).astype(np.float32)
    samples = cv2.remap(residual * 255, local[..., 0], local[..., 1], cv2.INTER_LINEAR).reshape(
        (2, -1, 25, 3)
    )
    centered = samples - samples.mean(axis=2, keepdims=True)
    luma, chroma = centered[..., :1], centered[..., 1:]
    luma_energy = np.mean(luma**2, axis=2)
    chroma_energy = np.mean(chroma**2, axis=2)
    covariance = np.mean(luma * chroma, axis=2)
    explained = np.mean(covariance**2 / ((luma_energy + 0.01) * (chroma_energy + 0.01)), axis=-1)
    independent = np.maximum(0, chroma_energy - covariance**2 / (luma_energy + 0.01)).mean(axis=-1)
    values = np.stack(
        (np.log(np.sqrt(luma_energy[..., 0]) + 0.1), explained, np.log(np.sqrt(independent) + 0.1)), axis=-1
    )
    rms = np.sqrt(luma_energy[..., 0])
    matched = (rms.min(axis=0) >= 0.25) & (rms.max(axis=0) <= 3 * rms.min(axis=0))
    result = np.full((8, len(ARMS)), np.nan)
    result[np.flatnonzero(valid)[matched]] = (values[0] - values[1])[matched]
    return result


def observe(path, detector, cache):
    table = probe_frames(path, timeout_s=15)
    require(
        not (table.truncated or table.interpolated or table.rotation) and 40 <= len(table) <= 100,
        "Expected a short, verified upright picture table",
    )
    shots, measures, anchors = [], [], []
    track = ShotDetector(table)
    for frame in decode_frames(path, table, None, timeout_s=25):
        require(
            frame.index == len(shots) and frame.pts_verified and frame.pts == table.pts[frame.index],
            "Native picture clock differs",
        )
        track.update(frame.index, frame.rgb)
        shots.append(len(track.boundaries))
        digest = earlier.panel.raw_digest(frame.rgb)
        if digest not in cache:
            geometry = earlier.panel.prior.aligned_face(frame.rgb, detector)
            if geometry is None:
                cache[digest] = np.full((8, len(ARMS)), np.nan), None
            else:
                x0, y0, x1, y1 = earlier.panel.native_bounds(geometry)
                cache[digest] = (
                    frame_measure(frame.rgb, geometry),
                    (
                        [x0, y0, x1, y1],
                        earlier.panel.raw_digest(frame.rgb[y0:y1, x0:x1]),
                    ),
                )
        values, proof = cache[digest]
        measures.append(values)
        anchors.append(
            None
            if proof is None
            else {
                "frame": frame.index,
                "pts": frame.pts,
                "rgb_sha256": digest,
                "crop_xyxy": proof[0],
                "crop_sha256": proof[1],
            }
        )
    require(len(shots) == len(table), "Decoded frame inventory differs")
    return table, shots, np.asarray(measures), anchors


def select(table, shots, data, anchors):
    best = dict.fromkeys(ARMS)
    measured = dict.fromkeys(ARMS, False)
    summaries = {}
    for size in (5, 8, 12, 16, 20):
        windows = np.lib.stride_tricks.sliding_window_view(data, size, axis=0)
        support = np.isfinite(windows).sum(axis=-1) * 5 >= 4 * size
        values = np.ma.median(np.ma.masked_invalid(windows), axis=-1).filled(np.nan)
        summaries[size] = np.where(support, values, np.nan)
    for width in (8, 12, 16, 20):
        for a in range(5, len(data) - width - 4):
            b = a + width
            if shots[a - 5] != shots[b + 4] or not all(anchors[i] for i in (a - 1, (a + b) // 2, b)):
                continue
            during, before, after = summaries[width][a], summaries[5][a - 5], summaries[5][b]
            signed = np.stack(
                (np.minimum(during - before, during - after), np.minimum(before - during, after - during))
            )
            for i, arm in enumerate(ARMS):
                supported = np.flatnonzero(np.isfinite(signed[0, :, i]))
                if len(supported) < 3:
                    continue
                measured[arm] = True
                for sign, row in zip(("increase", "decrease"), signed, strict=True):
                    matching = np.flatnonzero(row[:, i] > 0)
                    if len(matching) < 3:
                        continue
                    score = float(np.median(np.maximum(0, row[supported, i])))
                    if score <= 0 or best[arm] is not None and score <= best[arm]["score"]:
                        continue
                    best[arm] = {
                        "score": score,
                        "frames": [a, b],
                        "seconds": [str(table.pts[j] * table.tb) for j in (a, b)],
                        "time_base": table.time_base,
                        "direction": sign,
                        "support_sectors": supported.tolist(),
                        "positive_sectors": matching.tolist(),
                        "anchors": [anchors[j] for j in (a - 1, (a + b) // 2, b)],
                    }
    return {a: {"measured": measured[a], "proposal": best[a]} for a in ARMS}


def points_for(rows):
    points = {}
    for arm in ARMS:
        scores = sorted(
            (r["arms"][arm]["proposal"]["score"] for r in rows if r["arms"][arm]["proposal"]), reverse=True
        )
        points[arm] = max(FLOOR[arm], scores[2] if len(scores) > 2 else 0)
    return points


def score(lock, run, phase):
    cv2.setNumThreads(1)
    detector = load_yunet(earlier.WEIGHT)
    result, groups = [], {}
    roster = lock["rows"]
    if phase == "held":
        controls = read(run / "controls.json")
        require(
            controls["freeze_sha256"] == sha(run / "freeze.json") and len(controls["rows"]) == 24,
            "Frozen ordinary test deliveries are incomplete",
        )
        roster = [*roster, *controls["rows"]]
    for row in roster:
        if row["split"] == phase:
            groups.setdefault((row["family"], row["id"]), []).append(row)
    for group in groups.values():
        cache = {}
        for row in group:
            path = Path(row["file"])
            require(sha(path) == row["sha"], "Frozen source bytes differ")
            inputs = observe(path, detector, cache)
            result.append(
                {
                    **row,
                    "picture_ticks": len(inputs[0]),
                    "geometry_frames": sum(a is not None for a in inputs[3]),
                    "authored_native_range": [str(inputs[0].pts[i] * inputs[0].tb) for i in row["edit"]]
                    if row.get("edit")
                    else None,
                    "arms": select(*inputs),
                }
            )
        print(phase, group[0]["family"], group[0]["id"], len(result), flush=True)
    path = run / f"{phase}.json"
    write(
        path,
        {
            "freeze_sha256": sha(run / "freeze.json"),
            "controls_sha256": sha(run / "controls.json") if phase == "held" else None,
            "rows": result,
        },
    )
    if phase == "development":
        points = points_for(result)
        write(run / "points.json", {"development_sha256": sha(path), "points": points})
        print(points, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "prepare", "development", "held"))
    parser.add_argument("run", type=Path)
    parser.add_argument("lombard", type=Path)
    parser.add_argument("fresh", type=Path)
    args = parser.parse_args()
    lock = plan(args.lombard, args.fresh)
    if args.phase == "freeze":
        args.run.mkdir(mode=0o700)
        write(args.run / "freeze.json", lock)
        return
    require(lock == read(args.run / "freeze.json"), "Frozen inputs or method changed")
    if args.phase == "prepare":
        prepare_controls(lock, args.run)
        return
    if args.phase == "held":
        dev = args.run / "development.json"
        require(
            read(args.run / "points.json")
            == {"development_sha256": sha(dev), "points": points_for(read(dev)["rows"])},
            "Held points were not committed before scoring",
        )
    score(lock, args.run, args.phase)


if __name__ == "__main__":
    main()
