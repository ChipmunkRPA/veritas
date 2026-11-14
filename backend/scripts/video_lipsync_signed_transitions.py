# SPDX-License-Identifier: Apache-2.0
"""Compare short timing hypotheses using the signed direction of mouth motion."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np

from app.detectors.image.faces import load_yunet
from app.detectors.video import lipsync, xcorr
from app.detectors.video.adapters.lipsync_xcorr import _mouth_status, _track_sync
from scripts import review_video_lipsync_signed_transitions as rule
from scripts import video_lipsync_window_schedule as parent
from scripts.video_common_sources_v89 import read, require, sha, write

LAGS = np.arange(-rule.MAX_LAG, rule.MAX_LAG + 1)
LAG_CHANGES = np.abs(LAGS[np.newaxis, :] - LAGS[:, np.newaxis])
ALLOWED = (rule.MIN_LAG_CHANGE <= LAG_CHANGES) & (LAG_CHANGES <= rule.MAX_LAG_CHANGE)


def freeze(actors, conversations):
    base = parent.freeze(actors, conversations)
    later = [{**row, "bank": "later_excerpt", "offset_s": row["offset_s"] + 4}
             for row in base["sources"] if row["bank"] == "conversation"]
    return {"schema": rule.SCHEMA, "design": rule.DESIGN, "sources": [*base["sources"], *later],
            "source_plans": base["plans"], "incumbent": base["implementation_sha256"],
            "bindings": {Path(p).name: sha(p) for p in (__file__, rule.__file__, parent.__file__, parent.FACE_FILE)},
            "runtime": {"opencv": cv2.__version__, "numpy": np.__version__,
                        "ffmpeg": subprocess.check_output(["ffmpeg", "-version"], text=True).splitlines()[0]}}


def variants(pcm, grid, group):
    """Use the prior exact audio controls plus matching changes in the picture grid."""
    identity = tuple(range(grid.ticks))
    proposals = []
    for role, wave, truth, lag in parent.variants(pcm, grid.ticks, group):
        yield role, wave, truth, lag, identity
        if truth:
            proposals.append((role, wave, truth, lag))
    for role, wave, (a, b), lag in proposals:
        indices = tuple(i - lag if a <= i * rule.HOP < b else i for i in identity)
        require(all(0 <= i < grid.ticks for i in indices), "A synchronized control leaves the source grid")
        yield role.replace("local_", "matched_picture_"), wave, None, None, indices


def _profiles(aperture, speech, grid):
    """Measure signed increase/decrease and the old level proxy at the same lags."""
    frames = np.asarray(grid.frames)
    motion = np.r_[np.nan, np.diff(aperture)]
    motion[1:][frames[1:] == frames[:-1]] = np.nan
    # Compression reduces sensitivity to near-silent low-level noise without
    # converting raw dB-floor crossings into large transient events.
    change = np.r_[np.nan, np.diff(np.log1p(100 * speech.level))]
    rows = {}
    for width in {rule.FLANK, *rule.WIDTHS}:
        for start in range(rule.MAX_LAG + 1, min(grid.ticks, speech.ticks - rule.MAX_LAG) - width + 1):
            end = start + width
            window = aperture[start:end]
            if (not np.isfinite(window).all() or np.std(window) <= lipsync.CORRELATION_VARIATION_MIN
                    or np.mean(speech.speech[start:end]) < .35):
                continue
            indices = start + LAGS[:, None] + np.arange(width)[None, :]
            profiles = {"aperture_level": _correlate(window, speech.level[indices])}
            velocity = motion[start:end]
            if np.isfinite(velocity).all() and np.std(velocity) >= .006:
                profiles["signed_transitions"] = _correlate(velocity, change[indices])
            rows[start, end] = profiles
    return rows


def _correlate(visual, audio):
    left = visual - visual.mean()
    right = audio - audio.mean(axis=1, keepdims=True)
    norm = np.linalg.norm(left) * np.linalg.norm(right, axis=1)
    numerator = (left * right).sum(axis=1)
    value = np.divide(numerator, norm, out=np.zeros_like(norm), where=norm > 1e-10)
    return np.round(np.clip(value, -1., 1.), 6)


def _matrix(left, middle, right):
    flank = np.minimum(left[:, None] - left, right[:, None] - right)
    center = middle - middle[:, None]
    peaks = (left[:, None] >= rule.MIN_CORRELATION) & (right[:, None] >= rule.MIN_CORRELATION) & (
        middle >= rule.MIN_CORRELATION)
    return np.where(ALLOWED & peaks & (flank >= rule.MIN_FLANK_GAIN) & (center > 0),
                    np.minimum(center, 2 * flank).round(6), 0.)


def candidate(aperture, speech, grid):
    result = {name: {"brackets": 0, "best": None} for name in rule.ARMS}
    rows = _profiles(aperture, speech, grid)
    for width in rule.WIDTHS:
        for start in range(grid.ticks):
            spans = rule.brackets(start, width)
            profiles = [rows.get(span, {}) for span in spans]
            common = [name for name in rule.SIGNALS if all(name in profile for profile in profiles)]
            vectors = {name: [profile[name] for profile in profiles] for name in common}
            matrices = {name: _matrix(*data) for name, data in vectors.items()}
            if len(matrices) == len(rule.SIGNALS):
                matrices["agreement"] = np.minimum(*matrices.values())
            for name, matrix in matrices.items():
                arm = result[name]
                arm["brackets"] += 1
                i, j = np.unravel_index(np.argmax(matrix), matrix.shape)
                best = float(matrix[i, j])
                if best <= (arm["best"] or {}).get("score", 0):
                    continue
                included = rule.SIGNALS if name == "agreement" else (name,)
                correlations = {key: [[float(row[i]), float(row[j])] for row in vectors[key]] for key in included}
                reference, changed = int(LAGS[i]), int(LAGS[j])
                require(rule.scores(correlations, reference, changed).get(name) == best, "Correlation arithmetic differs")
                arm["best"] = {"ticks": [start, start + width], "reference_lag": reference, "changed_lag": changed,
                               "correlations": correlations, "score": best}
    return result


def current(faces, patches, aperture, speech, grid):
    plan = lipsync.track_windows("track:0", faces, grid, speech, hop=xcorr.HOP_TICKS, max_windows=xcorr.MAX_WINDOWS)
    windows = [w for w in plan if xcorr.full_search(w, speech.ticks)
               and _mouth_status(w, patches, aperture) == "measurable"]
    if not windows or lipsync.speaker_score(windows, patches, speech.level) < lipsync.ATTRIBUTION_MIN:
        return {"starts": [], "ranges": []}
    scored = _track_sync(windows, aperture, speech.level)
    return {"starts": [window.start_tick for window in scored.windows],
            "ranges": [[s.kind, round(2 * s.start_tick), round(2 * s.end_tick),
                        None if s.offset is None else s.offset - s.baseline] for s in (*scored.steps, *scored.collapses)]}


def witnesses(best, grid, commitments, indices, wave, origin):
    out = {}
    start, stop = best["ticks"]
    for role, (a, b) in zip(rule.ROLES, rule.brackets(start, stop - start), strict=True):
        video = [{"tick": tick, "source_tick": indices[tick], **commitments[grid.frames[tick]]}
                 for tick in (a, (a + b - 1) // 2, b - 1)]
        audio = {}
        for key, field in (("reference", "reference_lag"), ("changed", "changed_lag")):
            lag = best[field]
            lo, hi = (a + lag - 1) * rule.HOP, (b + lag) * rule.HOP
            audio[key] = {"samples": [lo, hi], "time": [str(origin + Fraction(i, rule.RATE)) for i in (lo, hi)],
                          "sha256": hashlib.sha256(wave[lo:hi].tobytes()).hexdigest()}
        out[role] = {"video": video, "audio": audio}
    return out


def score(lock, actors, conversations, split, points):
    require(lock == freeze(actors, conversations), "Source plan or method changed after freeze")
    require(sha(parent.FACE_FILE) == parent.FACE_SHA, "The face geometry asset differs from the pinned weights")
    cv2.setNumThreads(1)
    face_model = load_yunet(parent.FACE_FILE)
    rows = []
    with TemporaryDirectory(prefix="signed-video-transitions-") as scratch:
        for plan in lock["sources"]:
            if plan["split"] != split:
                continue
            bank, group = plan["bank"], plan["group"]
            row = {"bank": bank, "group": group, "source_sha256": plan["sha256"]}
            rows.append(row)
            try:
                observed = {**plan, "bank": "conversation"} if bank == "later_excerpt" else plan
                grid, table, origin, pcm, _, faces, patches, frames = parent._observe(observed, actors, conversations,
                                                                                   face_model, Path(scratch))
            except (RuntimeError, ValueError, subprocess.SubprocessError) as error:
                row["unavailable"] = type(error).__name__
                continue
            if grid.ticks < 55:
                row["unavailable"] = "short_input"
                continue
            row.update(samples=len(pcm), ticks=grid.ticks, frame_count=len(table), time_base=table.time_base,
                       origin=str(origin), cases=[])
            for role, wave, truth, lag, indices in variants(pcm, grid, group):
                picture = replace(grid, frames=tuple(grid.frames[i] for i in indices))
                crops = {i: patches[source] for i, source in enumerate(indices) if source in patches}
                aperture = {tick: xcorr.aperture(patch) for tick, patch in crops.items()}
                vad = lipsync.vad(wave.astype(np.float64) / 32768.)
                candidate_rows = candidate(np.array([aperture.get(tick, np.nan) for tick in range(grid.ticks)]), vad, picture)
                if points is not None:
                    for name, arm in candidate_rows.items():
                        if (best := arm["best"]) and best["score"] >= points[name]:
                            best["witness"] = witnesses(best, picture, frames, indices, wave, origin)
                row["cases"].append({"role": role, "truth": truth, "lag": lag,
                                     "pcm_sha256": hashlib.sha256(wave.tobytes()).hexdigest(),
                                     "current": current(faces, crops, aperture, vad, picture), "candidate": candidate_rows})
            print(f"{bank} {group} measured {grid.ticks} ticks", flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "development", "held"))
    parser.add_argument("actors", type=Path)
    parser.add_argument("conversations", type=Path)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    freeze_file, development, receipt = (args.run / name for name in ("freeze.json", "development.json", "receipt.json"))
    if args.phase == "freeze":
        args.run.mkdir(mode=0o700)
        write(freeze_file, freeze(args.actors, args.conversations))
        return
    lock = read(freeze_file)
    if args.phase == "development":
        rows = score(lock, args.actors, args.conversations, "development", None)
        points = rule.thresholds(rows)
        write(development, {"freeze_sha256": sha(freeze_file), "rows": rows, "points": points})
        print(json.dumps(points), flush=True)
        return
    fit = read(development)
    require(fit["freeze_sha256"] == sha(freeze_file), "Development points differ from the frozen design")
    rows = score(lock, args.actors, args.conversations, "held", fit["points"])
    by_group = {(row["bank"], row["group"]): row for row in fit["rows"] + rows}
    document = {"freeze": lock, "points": fit["points"],
                "rows": [by_group[s["bank"], s["group"]] for s in lock["sources"]]}
    document = json.loads(json.dumps(document, allow_nan=False))
    document["outcome"] = rule.review(document)
    write(receipt, document)
    print(json.dumps(document["outcome"], sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
