"""Measure spatial support for temporary luma changes in archived video flags."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from app.detectors.video.decode import decode_frames, probe_frames
from scripts import video_saved_landmark_support_v254 as saved
from scripts.video_appearance_corrected_v112 import scoped

ROOT, PANEL = saved.webcam.ROOT, saved.webcam.PANEL
NAME = "video-luma-context-v278"
PUBLIC = PANEL / f"{NAME}-freeze.json"
SAVED = PANEL / "video-saved-landmark-support-v254-complete.json"
SAVED_SHA = "7402d598c251158a6d4ec4b42b5e6b059f739e62acf8c46d015011aceab66021"
ROLES = ("before", "during", "after")
STRIDE = 16
SHIFT = 12.0
CODE = (
    f"backend/scripts/{NAME.replace('-', '_')}.py",
    "backend/scripts/video_saved_landmark_support_v254.py",
    "backend/scripts/video_webcam_confound_v224.py",
    "backend/app/detectors/video/decode.py",
    "backend/app/evidence/face_interval_binding.py",
    f"backend/tests/test_{NAME.replace('-', '_')}.py",
    f"docs/{NAME}-method.md",
)
read, sha, require = saved.read, saved.sha, saved.require


def luma(rgb):
    return rgb.astype(np.float64) @ np.array([0.299, 0.587, 0.114])


def in_box(x, y, box, margin):
    left, top, width, height = (box[key] for key in ("x", "y", "w", "h"))
    return (
        (x >= left - margin * width)
        & (x < left + (1 + margin) * width)
        & (y >= top - margin * height)
        & (y < top + (1 + margin) * height)
    )


def temporary_delta(before, during, after):
    a, b = during - before, during - after
    magnitude = np.minimum(np.abs(a), np.abs(b))
    return np.where(np.sign(a) == np.sign(b), np.sign(a) * magnitude, 0.0)


def measure(sampled):
    """Compare corresponding nonface sample sites outside every expanded face box."""
    require(tuple(sampled) == ROLES, "Exact three chronological role sets required")
    if min(len(sampled[role]) for role in ROLES) < 3:
        return {"status": "unavailable", "reason": "insufficient_role_or_context_support"}
    sites = np.logical_and.reduce([row["context"] for role in ROLES for row in sampled[role]])
    if np.count_nonzero(sites) < 128:
        return {"status": "unavailable", "reason": "insufficient_role_or_context_support"}
    context = [np.median([row["luma"][sites] for row in sampled[role]], axis=0) for role in ROLES]
    face = [np.median([row["face"] for row in sampled[role]]) for role in ROLES]
    field = temporary_delta(*context)
    face_delta = float(temporary_delta(*face))
    affected = np.abs(field) >= SHIFT
    return {
        "status": "measured",
        "context_sites": int(sites.sum()),
        "minimum_face_sites": min(row["face_sites"] for role in ROLES for row in sampled[role]),
        "face_shift": face_delta,
        "context_median_absolute_shift": float(np.median(np.abs(field))),
        "context_affected_fraction": float(np.mean(affected)),
        "context_affected_in_face_direction_fraction": float(
            np.mean(affected & (np.sign(field) == np.sign(face_delta)))
        )
        if abs(face_delta) >= SHIFT
        else None,
    }


def sample_frame(rgb, box):
    h, w = rgb.shape[:2]
    ys, xs = np.ogrid[STRIDE // 2 : h : STRIDE, STRIDE // 2 : w : STRIDE]
    reduced = rgb[STRIDE // 2 :: STRIDE, STRIDE // 2 :: STRIDE]
    inside = in_box(xs, ys, box, 0)
    face_sites = int(inside.sum())
    if face_sites < 4:
        return None
    values = luma(reduced)
    return {
        "luma": values,
        "context": ~in_box(xs, ys, box, 0.5),
        "face": np.median(values[inside]),
        "face_sites": face_sites,
    }


def measure_source(path, record, expected):
    role_map = saved.archived.role_indices(record)
    if role_map is None:
        return {"status": "unavailable", "reason": "overlapping_or_unordered_role_frames"}
    by_index = {index: role for role in ROLES for index in role_map[role]}
    table = probe_frames(path, timeout_s=60)
    require(
        list(table.pts) == expected["pts"]
        and table.tb == Fraction(expected["time_base"])
        and (table.width, table.height) == expected["size"]
        and not (table.rotation or table.truncated or table.interpolated),
        "Current decoder differs from the saved source geometry or clock",
    )
    rows = {row["frame_index"]: row for row in record["frame_inputs"]}
    sampled = {role: [] for role in ROLES}
    seen = []
    for frame in decode_frames(path, table, list(by_index), timeout_s=90):
        row = rows[frame.index]
        digest = hashlib.sha256(frame.rgb.tobytes()).hexdigest()
        require(
            frame.pts == row["pts"] and digest == row["decoded_rgb_sha256"] == expected["rgb"][frame.index],
            "Source RGB or frame time differs from the retained analysis",
        )
        sampled[by_index[frame.index]].append(sample_frame(frame.rgb, row["bbox"]))
        seen.append(frame.index)
    require(seen == list(by_index), "Complete native context decode required")
    if any(row is None for role in ROLES for row in sampled[role]):
        return {
            "status": "unavailable",
            "reason": "insufficient_face_grid_support",
            "decoded_frames": len(seen),
        }
    return {**measure(sampled), "decoded_frames": len(seen)}


def protocol():
    require(sha(SAVED) == SAVED_SHA, "Exact prior cohort and coverage required")
    return {
        "schema": "chipmunk.video-luma-context-freeze/278",
        "saved_context_sha256": SAVED_SHA,
        "code_sha256": {path: sha(ROOT / path) for path in CODE},
        "decoder_sha256": {name: sha(Path(shutil.which(name)).resolve()) for name in ("ffmpeg", "ffprobe")},
        "grid_stride": STRIDE,
        "minimum_luma_shift_code_values": SHIFT,
        "minimum_face_sites_per_frame": 4,
        "minimum_common_nonface_sites": 128,
        "numpy_version": np.__version__,
        "selected_flags_only": True,
        "model_calls_permitted": False,
        "serving_change_permitted": False,
    }


def readout():
    require(read(PUBLIC) == protocol(), "Exact prospective readout freeze required")
    swap_rows = read(saved.swap.MANIFEST)["cases"]
    webcam_rows = read(saved.webcam.TRIAL / "private-plan.json").values()
    paths = {row["sha256"]: saved.swap.ARCHIVE / f"{row['condition']}.mp4" for row in swap_rows}
    paths.update(
        {
            row["sha256"]: saved.webcam.raw.PREPARATION
            / f"slot-{row['slot']:02d}-attempt-01"
            / f"{row['role']}.mp4"
            for row in webcam_rows
        }
    )
    records = scoped(saved.windows, {"support": SimpleNamespace(measure=lambda row: row)})

    def windows(db, load, digest, case, expected):
        result = records(db, load, digest, case, expected)
        if not any(flag for flag, _ in result):
            return [(False, None) for _ in result]
        path = paths[digest]
        require(
            path.resolve(strict=True) == path and sha(path) == digest, "Exact direct source video required"
        )
        return [(flag, measure_source(path, row, expected) if flag else None) for flag, row in result]

    cases = [
        case
        for source in (saved.swap_cases, saved.webcam_cases)
        for case in scoped(source, {"windows": windows})()
    ]
    summaries = {}
    for cohort, prior in read(SAVED)["cohorts"].items():
        selected = [row for label, rows in cases if label == cohort for flag, row in rows if flag]
        require(len(selected) == prior["selected"]["windows"], "Retained selection coverage differs")
        assessed = [row for row in selected if row["status"] == "measured"]
        summaries[cohort] = {
            "source_groups": prior["source_groups"],
            "selected_windows": len(selected),
            "measured": len(assessed),
            "unavailable": dict(Counter(row["reason"] for row in selected if row["status"] != "measured")),
            "decoded_frame_observations": sum(row.get("decoded_frames", 0) for row in selected),
            "with_face_shift_at_least_12": sum(
                row["context_affected_in_face_direction_fraction"] is not None for row in assessed
            ),
            "ranges": {
                key: saved.bounds(row[key] for row in assessed if row[key] is not None)
                for key in (
                    "context_sites",
                    "minimum_face_sites",
                    "face_shift",
                    "context_median_absolute_shift",
                    "context_affected_fraction",
                    "context_affected_in_face_direction_fraction",
                )
            },
        }
    return {
        "schema": "chipmunk.video-luma-context-result/278",
        "freeze_sha256": sha(PUBLIC),
        "cohorts": summaries,
        "model_calls": 0,
        "serving_changed": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "readout"))
    if parser.parse_args().command == "freeze":
        with PUBLIC.open("x") as file:
            json.dump(protocol(), file, indent=2, allow_nan=False)
            file.write("\n")
    else:
        saved.edge.enforce_read_only()
        print(json.dumps(readout(), indent=2, allow_nan=False))
