"""Separate non-neural replay of retained RGB, masks, messages and source frame times."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy import ndimage

from scripts import run_public_visual_watermark_v1 as frozen


def independent(logits, packed):
    mask = logits > 0
    labels, count = ndimage.label(mask, structure=np.ones((3, 3), int))
    regions = []
    for i in range(1, count + 1):
        y, x = np.nonzero(labels == i)
        if len(y) >= 500:
            regions.append(
                {
                    "x": int(x.min()),
                    "y": int(y.min()),
                    "w": int(x.max() - x.min() + 1),
                    "h": int(y.max() - y.min() + 1),
                    "pixels": len(y),
                }
            )
    bits = np.unpackbits(packed, axis=1, count=65536, bitorder="little").reshape(32, 256, 256)
    decoded = (
        None
        if not mask.any()
        else [int(np.count_nonzero(b[mask]) > np.count_nonzero(mask) / 2) for b in bits]
    )
    return mask, regions, decoded


def audit(trial, output):
    frozen.check(trial)
    m = frozen.method
    profile = m.read(trial / "profile/profile.json")
    result = m.read(trial / "result.json")
    rows = m.read(trial / "cases-private.json")["cases"]
    m.require(
        profile["child_exit_code"] == 0
        and profile["guard"] is None
        and len(rows) == 168
        and result["generator_forwards"] == 18
        and result["detector_forwards"] == 173,
        "Complete native study required",
    )
    output.mkdir(exist_ok=False)
    authored = m.authored()
    bases = {}
    videos = {}
    pixels_replayed = 0
    bits_replayed = 0
    expected_images = {
        (name, state, transform)
        for name in authored
        for state in frozen.STATES
        for transform in frozen.TRANSFORMS
    }
    m.require(
        {(r["carrier"], r["state"], r["transform"]) for r in rows if r["kind"] == "image"} == expected_images,
        "Complete72 unique image cases required",
    )
    expected_video = {
        (name, state, container, i)
        for name in ["gradient", "waves", "texture"]
        for state in ["unmarked", "partial"]
        for container in ["raw", "h264"]
        for i in range(8)
    }
    m.require(
        {
            (r["carrier"], r["clip_state"], r["container"], r["anchor"]["frame_index"])
            for r in rows
            if r["kind"] == "video"
        }
        == expected_video,
        "Complete96 unique video frame cases required",
    )
    for row in rows:
        path = Path(row["private_array"])
        m.require(m.digest(path) == row["array_sha256"], "Native case changed")
        with np.load(path, allow_pickle=False) as z:
            rgb, logits, packed = z["pixels"], z["presence_logits"], z["binary_bit_maps"]
        m.require(
            rgb.dtype == np.uint8
            and rgb.shape == (256, 256, 3)
            and hashlib.sha256(rgb.tobytes()).hexdigest() == row["source_rgb_sha256"],
            "Native RGB identity changed",
        )
        m.require(
            logits.dtype == np.float32
            and logits.shape == (256, 256)
            and np.isfinite(logits).all()
            and packed.dtype == np.uint8
            and packed.shape == (32, 8192),
            "Retained native geometry/type changed",
        )
        base_path = Path(row["base_array"])
        m.require(m.digest(base_path) == row["base_sha256"], "Retained source/marked arrays changed")
        if str(base_path) not in bases:
            with np.load(base_path, allow_pickle=False) as z:
                base = {k: z[k] for k in z.files}
            bases[str(base_path)] = base
            u, p = base["unmarked"], base["partial"]
            if u.ndim == 3:
                m.require(np.array_equal(u, authored[row["carrier"]]), "Authored source changed")
                outside = np.ones((256, 256), bool)
                outside[64:192, 64:192] = False
                m.require(
                    np.array_equal(p[outside], u[outside])
                    and np.array_equal(p[64:192, 64:192], base["marked"][64:192, 64:192]),
                    "Partial image changed outside truth",
                )
                ordinary = u.copy()
                ordinary[64:192, 64:192] = np.rot90(u[64:192, 64:192])
                m.require(np.array_equal(base["ordinary_edit"], ordinary), "Ordinary edit recipe changed")
            else:
                original = np.stack([np.roll(authored[row["carrier"]], i * 3, axis=1) for i in range(8)])
                m.require(np.array_equal(u, original), "Authored video source changed")
                outside = np.ones((8, 256, 256), bool)
                outside[2:6, 64:192, 64:192] = False
                m.require(
                    np.array_equal(p[outside], u[outside]), "Partial video changed outside source time/region"
                )
        base = bases[str(base_path)]
        if row["kind"] == "image":
            m.require(
                np.array_equal(rgb, frozen.transform(base[row["state"]], row["transform"])),
                "Image codec/resize input replay differs",
            )
        elif row["container"] == "raw":
            m.require(
                np.array_equal(rgb, base[row["clip_state"]][row["anchor"]["frame_index"]]),
                "Raw frame source differs",
            )
        else:
            video = Path(row["video_path"])
            if str(video) not in videos:
                data = subprocess.run(
                    [
                        "ffmpeg",
                        "-v",
                        "error",
                        "-nostdin",
                        "-i",
                        str(video),
                        "-fps_mode",
                        "passthrough",
                        "-f",
                        "rawvideo",
                        "-pix_fmt",
                        "rgb24",
                        "pipe:1",
                    ],
                    capture_output=True,
                    check=True,
                    timeout=30,
                ).stdout
                probe = json.loads(
                    subprocess.run(
                        [
                            "ffprobe",
                            "-v",
                            "error",
                            "-select_streams",
                            "v:0",
                            "-show_streams",
                            "-show_frames",
                            "-of",
                            "json",
                            str(video),
                        ],
                        capture_output=True,
                        text=True,
                        check=True,
                        timeout=30,
                    ).stdout
                )
                m.require(
                    len(data) == 8 * 256 * 256 * 3 and len(probe["frames"]) == 8,
                    "Codec replay changed frame count",
                )
                videos[str(video)] = (np.frombuffer(data, np.uint8).reshape(8, 256, 256, 3), probe)
            frames, probe = videos[str(video)]
            anchor = row["anchor"]
            i = anchor["frame_index"]
            pts = int(probe["frames"][i]["best_effort_timestamp"])
            tb = probe["streams"][0]["time_base"]
            m.require(
                np.array_equal(rgb, frames[i])
                and anchor["source_pts"] == pts
                and anchor["time_base"] == tb
                and Fraction(pts) * Fraction(tb) == Fraction(i, 2),
                "Decoded frame/source PTS replay differs",
            )
        mask, regions, decoded = independent(logits, packed)
        e = row["evidence"]

        def key(r):
            return (r["y"], r["x"], r["w"], r["h"], r["pixels"])

        m.require(
            sorted(regions, key=key) == sorted(e["regions"], key=key)
            and int(mask.sum()) == e["selected_pixels"]
            and float(mask.mean()) == e["selected_fraction"]
            and decoded == e["decoded_message"]
            and bool(regions) == e["flagged_for_inspection"],
            "Independent pixel/region/message replay differs",
        )
        m.require(
            e["provider_authenticated"] is e["used_in_confidence"] is e["early_stop_eligible"] is False,
            "Unsupported provider/AI/early-stop meaning acquired",
        )
        truth = frozen.truth(row["state"])
        inter = int(np.count_nonzero(mask & truth))
        union = int(np.count_nonzero(mask | truth))
        actual = {
            "iou": inter / union if truth.any() else None,
            "coverage": inter / int(truth.sum()) if truth.any() else None,
            "pixels_selected_outside": int(mask.sum()) - inter,
        }
        m.require(actual == row["overlap"], "Known spatial truth/overlap replay differs")
        pixels_replayed += mask.size
        bits_replayed += 32 * mask.size
    m.require(frozen.aggregate(rows) == result["groups"], "Aggregate denominators differ")
    receipt = {
        "schema": "chipmunk.public-visual-watermark-source-audit/1",
        "cases": len(rows),
        "presence_pixels_replayed": pixels_replayed,
        "binary_bit_predictions_replayed": bits_replayed,
        "codec_clips_redecoded": len(videos),
        "codec_frames_redecoded": len(videos) * 8,
        "source_times_verified": True,
        "neural_forwards": 0,
        "provider_requests": 0,
        "outside_human_reviewers": 0,
        "raw_continuous_bit_logits": "Unused and not retained; hard readout replays every retained binary bit.",
    }
    m.write(output / "audit.json", receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("trial", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    audit(a.trial, a.output)
