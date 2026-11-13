# SPDX-License-Identifier: Apache-2.0
"""Measure released lip timing and frozen research checks on qualified Lombard deliveries."""

from __future__ import annotations

import argparse
from collections import Counter
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from app.detectors.audio.decode import decode_audio
from app.detectors.image.faces import detect_faces, load_yunet
from app.detectors.video import lipsync, sampling
from app.detectors.video.adapters.common import AudioTrack
from app.detectors.video.adapters.lipsync import AVInputs, source_audio_origin
from app.detectors.video.adapters.lipsync_xcorr import LandmarkXcorrDetector
from app.detectors.video.frame_store import FrameStore
from scripts import build_lombard_neural_mouth as build
from scripts import review_lombard_neural_mouth as review
from scripts import video_lipsync_local_contrast_transfer as contrast
from scripts import video_mouth_temporal_detail as detail
from scripts.video_common_sources_v89 import read, require, sha, write

BASE = detail.panel.DATA
TIMING = BASE / "video-lipsync-local-contrast-2026-11-12"
VISUAL = BASE / "video-mouth-temporal-detail-2026-11-12"
WEIGHT = detail.panel.prior.WEIGHT


def frozen(run):
    fitted = read(TIMING / "head.json")
    point = read(TIMING / "point.json")
    visual = read(VISUAL / "aggregate.json")
    require(
        fitted["freeze_sha256"] == sha(TIMING / "freeze.json")
        and point["head_sha256"] == sha(TIMING / "head.json")
        and visual["runtime"]["calibration_sha256"] == sha(VISUAL / "calibration.json"),
        "Previous development points differ",
    )
    bound = (
        Path(__file__),
        Path(contrast.__file__),
        Path(detail.__file__),
        Path(lipsync.__file__),
        WEIGHT,
        TIMING / "head.json",
        TIMING / "point.json",
        VISUAL / "calibration.json",
    )
    return {
        "fixture_freeze_sha256": sha(run / "freeze.json"),
        "files": {p.name: sha(p) for p in bound},
        "points": {"timing": point["value"], **{k: visual["points"][k] for k in detail.ARMS}},
        "interpretation": "The actual released lip-timing adapter with directly observed one-face track and native frames, "
        "and unchanged former research timing and visual points. Group independent. Local source-aligned redraws are controls "
        "for timing but targets for visual alteration. A real timing mismatch is not an AI finding. No refitting.",
    }


class Exhibits:
    def __init__(self):
        self.ranges = []
        self.count = 0

    def series(self, *_args, **_kwargs):
        self.count += 1
        return f"series:{self.count}"

    def segments(self, segments, **_kwargs):
        self.ranges.extend(segments)
        return "segments:1"


def inspect(path, row, digest, detector, points, fitted):
    require(sha(path) == digest, "Verified media bytes differ")
    table, _ = build.table_for(path, row)
    build.verified_audio(path, row)
    audio = decode_audio(path, source_sha256=digest, max_seconds=8)
    require(
        audio.pcm is not None and len(audio.pcm) == row["audio"]["samples"],
        "Released audio input loses samples",
    )
    frames = []
    store = FrameStore(path, table, max_bytes=110 * 1024**2)
    for i, rgb in store.iter_many(range(len(table))):
        require(store.native_pts_verified(i), "Released mouth adapter input lacks a native PTS")
        faces = detect_faces(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), detector=detector, min_size=48)
        if len(faces) != 1:
            continue
        [face] = faces
        frames.append(
            {
                "frame_index": i,
                "pts": table.pts[i],
                "landmarks": face.landmarks,
                "bbox": dict(zip(("x", "y", "w", "h"), face.box(), strict=True)),
            }
        )
    av = AVInputs(
        table,
        ({"track_id": "track:0"},) if frames else (),
        {"track:0": frames},
        AudioTrack("analysed", audio.pcm, len(audio.pcm)),
        store,
        audio.mapping,
    )
    origin = source_audio_origin(av)
    check = LandmarkXcorrDetector()
    exhibits = Exhibits()
    context = SimpleNamespace(outputs=SimpleNamespace(av=av), deadline=None, exhibits=exhibits)
    applicability = check.applicable(context)
    outcome = check.run(context) if applicability.ok else None
    tracks = outcome.details.get("av_input", {}).get("tracks", []) if outcome else []
    current = {
        "status": outcome.status if outcome else "not_applicable",
        "measured_windows": int(outcome.metrics.get("windows", 0)) if outcome else 0,
        "ranges": exhibits.ranges,
        "reason": (outcome.reason or outcome.observed) if outcome else applicability.reason,
        "timing_segments": [segment for track in tracks for segment in track["segments"]],
    }
    by_frame = {f["frame_index"]: f for f in frames}
    grid = lipsync.tick_grid(table.times(), sampling.duration_s(table), len(audio.pcm), audio_start_s=0)
    patches = {
        i: lipsync.mouth_patch(store.get(f), by_frame[f]) for i, f in enumerate(grid.frames) if f in by_frame
    }
    pcm = np.clip(np.rint(audio.pcm * 32768), -32768, 32767).astype("<i2")
    timing = contrast.candidate(
        contrast.visual_features(patches, grid.ticks), *contrast.acoustic_features(pcm), fitted
    )
    best = timing["best"]
    visual = detail.select(
        *detail.edge_observations(path, {"file_sha256": digest, "expected": None}, detector, {}, {})
    )
    return {
        "native_face_pictures": len(frames),
        "current": current,
        "released_audio_origin": origin,
        "timing_candidate": {
            "measured": timing["measured"],
            "range": best if best and best["score"] > points["timing"] else None,
        },
        "visual_candidates": {
            k: {"measured": v is not None, "range": v if v and v["score"] > points[k] else None}
            for k, v in visual.items()
        },
    }


def overlap(first, second):
    a, b = map(Fraction, first)
    c, d = map(Fraction, second)
    return max(0, min(b, d) - max(a, c)) / (max(b, d) - min(a, c))


def score(run, split):
    lock = read(run / "detector-freeze.json")
    require(lock == frozen(run), "Detector method differs from its freeze")
    native = read(run / f"review-{split}.json")
    require(
        native["freeze_sha256"] == sha(run / "freeze.json"), "Complete independent native replay required"
    )
    construction = {(c["ordinal"], c["role"]): c for c in review.partitions(run, split)}
    cv2.setNumThreads(1)
    detector = load_yunet(WEIGHT)
    fitted = read(TIMING / "head.json")
    fixtures = {r["source"]["ordinal"]: r for r in read(run / "freeze.json")["sources"]}
    results = []
    for receipt in native["cases"]:
        k, role = receipt["ordinal"], receipt["role"]
        if receipt["status"] != "authored":
            continue
        row = fixtures[k]
        case = construction[k, role]
        path = run / "deliveries" / f"{case['token']}.mov"
        result = inspect(path, row, receipt["file_sha256"], detector, lock["points"], fitted)
        observation = receipt["observation"]
        interval = receipt["authored_seconds"]
        for v in result["visual_candidates"].values():
            v["right_interval"] = bool(
                interval and v["range"] and overlap(v["range"]["source_clock_s"], interval) >= Fraction(1, 2)
            )
        if observation["status"] == "observed_delayed_control":
            expected = -observation["visual_picture_shift"]
            candidate = result["timing_candidate"]
            measured = candidate["range"]
            candidate["right_interval"] = bool(
                measured
                and abs(measured["relative_lag_ticks"] - expected) <= 2
                and overlap([Fraction(i, 25) for i in measured["extent_ticks"]], interval) >= Fraction(2, 5)
            )
            result["current"]["right_interval"] = any(
                r["kind"] == "offset_step"
                and abs(r["offset"] - r["baseline"] - expected) <= 2
                and overlap([Fraction(str(r[s])) / 25 for s in ("start_tick", "end_tick")], interval)
                >= Fraction(2, 5)
                for r in result["current"]["timing_segments"]
            )
        results.append(
            {
                "ordinal": k,
                "group": receipt["group"],
                "role": role,
                "observation": observation["status"],
                **result,
            }
        )
        print(split, k, role, observation["status"], result["current"]["measured_windows"], flush=True)
    summary = Counter((r["role"], r["observation"]) for r in results)
    write(
        run / f"detector-{split}.json",
        {
            "native_review_sha256": sha(run / f"review-{split}.json"),
            "detector_freeze_sha256": sha(run / "detector-freeze.json"),
            "counts": [[*k, v] for k, v in sorted(summary.items())],
            "cases": results,
        },
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("freeze", "score"))
    p.add_argument("run", type=Path)
    p.add_argument("--split", choices=("development", "held"), default="development")
    a = p.parse_args()
    if a.phase == "freeze":
        write(a.run / "detector-freeze.json", frozen(a.run))
    else:
        score(a.run, a.split)


if __name__ == "__main__":
    main()
