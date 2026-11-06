"""Offline trial of inner-lip geometry for measuring known audio/video offsets."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mediapipe as mp
import numpy as np
from prepare_video_timing_upgrade import POLICY
from prepare_video_upgrade_eval import sha
from scipy.ndimage import gaussian_filter1d

from app.detectors.audio.decode import decode_audio, source_ranges
from app.detectors.video import lipsync, sampling
from app.detectors.video.decode import decode_frames, probe_frames

WEIGHT_SHA = "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff"
LAG = 12  # ±480 ms at 25 Hz
LIP = (78, 308, 13, 14, 82, 87, 312, 317)


def geometry(rgb, landmarks):
    height, width = rgb.shape[:2]
    points = np.array([(landmarks[k].x * width, landmarks[k].y * height) for k in LIP])
    axis = points[1] - points[0]
    mouth_width = float(np.linalg.norm(axis))
    if not 8 <= mouth_width <= width / 2 or not np.all(np.isfinite(points)):
        return None
    vertical = np.array([-axis[1], axis[0]]) / mouth_width
    separations = (points[3::2] - points[2::2]) @ vertical
    ratio = float(np.mean(np.abs(separations)) / mouth_width)
    lo, hi = points.min(axis=0), points.max(axis=0)
    pad = 0.18 * mouth_width
    x0, y0 = np.maximum(0, np.floor(lo - pad)).astype(int)
    x1, y1 = np.minimum([width, height], np.ceil(hi + pad)).astype(int)
    return ratio, [int(x0), int(y0), int(x1 - x0), int(y1 - y0)]


def features(path: Path, digest: str, weights: Path):
    table = probe_frames(path)
    sound = decode_audio(path, source_sha256=digest)
    ranges = source_ranges(sound.mapping, 0, len(sound.pcm))
    if len(ranges) != 1:
        return None
    origin = ranges[0]["source_start_pts"] / lipsync.SR
    grid = lipsync.tick_grid(table.times(), sampling.duration_s(table), len(sound.pcm), audio_start_s=origin)
    n = min(grid.ticks, 750)
    if n < 35:
        return None
    frames = sorted(set(grid.frames[:n]) - {-1})
    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(weights), delegate=mp.tasks.BaseOptions.Delegate.CPU),
        running_mode=mp.tasks.vision.RunningMode.VIDEO, num_faces=2,
    )
    seen = {}
    last_ms = -1
    with mp.tasks.vision.FaceLandmarker.create_from_options(options) as detector:
        for frame in decode_frames(path, table, frames):
            ms = max(last_ms + 1, round(table.seconds(frame.index) * 1000))
            last_ms = ms
            result = detector.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=frame.rgb), ms)
            if len(result.face_landmarks) == 1 and (measured := geometry(frame.rgb, result.face_landmarks[0])):
                seen[frame.index] = measured
    visual = np.array([seen.get(index, (np.nan, None))[0] for index in grid.frames[:n]])
    good = np.flatnonzero(np.isfinite(visual))
    if len(good) < n * .85 or np.max(np.diff(good), initial=0) > 3:
        return None
    visual = np.interp(np.arange(n), good, visual[good])
    if np.ptp(np.percentile(visual, (10, 90))) < .035:
        return None
    speech = lipsync.vad(sound.pcm)
    if float(np.percentile(speech.level_db, 90)) < -45:
        return None
    db = np.clip(speech.level_db, float(np.percentile(speech.level_db, 90)) - 32, 0.0)
    level = np.sqrt(speech.level)
    # Two amplitude representations and their onsets are kept as separate development alternatives.
    metrics = {"db": db, "level": level, "onset": np.gradient(db)}
    span = 30
    windows = [(LAG, n - LAG)]
    if n > 80:
        windows = [(a, a + span) for a in range(LAG, n - LAG - span + 1, 12)]
    profiles = []
    for start, end in windows:
        if end - start < 20:
            continue
        opening = gaussian_filter1d(visual, .8)
        curves = {}
        for key, value in metrics.items():
            a = np.gradient(opening) if key == "onset" else opening
            b = gaussian_filter1d(value, .8)
            curves[key] = [lipsync._pearson(a[start:end], b[start + lag:end + lag]) for lag in range(-LAG, LAG + 1)]
        anchors = []
        for tick in (start, (start + end - 1) // 2, end - 1):
            index = grid.frames[tick]
            if index in seen:
                anchors.append({"frame_index": index, "pts": table.pts[index], "time_s": table.seconds(index),
                                "mouth_bbox": seen[index][1], "aperture": seen[index][0]})
        profiles.append({"start_tick": start, "end_tick": end, "start_s": grid.seconds(start),
                         "end_s": grid.seconds(end), "frames": anchors, "curves": curves})
    return {"audio_origin_s": origin, "time_base": table.time_base, "profiles": profiles}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("weights", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--split", choices=("development", "test", "publisher-example"), required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    if manifest["policy"] != POLICY or sha(args.weights) != WEIGHT_SHA:
        raise ValueError("unexpected manifest or weights")
    with args.output.open("x") as out:
        out.write(json.dumps({"manifest_sha256": sha(args.manifest), "weights_sha256": WEIGHT_SHA,
                              "script_sha256": sha(Path(__file__)), "split": args.split}) + "\n")
        for row in manifest["items"]:
            if row["split"] != args.split:
                continue
            path = args.manifest.parent / row["path"]
            if sha(path) != row["sha256"]:
                raise ValueError(f"source bytes changed: {row['id']}")
            measured = features(path, row["sha256"], args.weights)
            out.write(json.dumps({**row, "measurement": measured}, allow_nan=False) + "\n")
            out.flush()
            print(row["id"], bool(measured and measured["profiles"]), flush=True)


if __name__ == "__main__":
    main()
