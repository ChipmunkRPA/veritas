# SPDX-License-Identifier: Apache-2.0
"""Compare short AV schedules on source-bound, paired deliveries."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.review_video_lipsync_window_schedule import ARMS, CASES, SCHEMA, review
from scripts.video_common_sources_v89 import read, require, sha, write

ROOT = Path(__file__).resolve().parents[2]
CONVERSATION = ROOT / "docs/evaluation-panels/video-mouth-hd-hold-2026-11-12.json"
ACTORS = ROOT / "docs/evaluation-panels/video-lipsync-local-delay-transfer-2026-11-11.json"
FACE_FILE = ROOT / "models/weights/face_detection_yunet_2023mar.onnx"
FACE_SHA = "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"
WIDTHS = (16, 22)
EXCERPT_TICKS = 88
LAGS = (5, 7)
FADE = 160


def freeze(actor_dir, conversation_dir):
    parent = read(actor_dir / "intake-receipt.json")
    reference = read(ACTORS)
    require(sha(actor_dir / "intake-receipt.json") == reference["source"]["publisher_byte_receipt_sha256"],
            "Archived actor intake receipt differs")
    actors = [{"bank": "acted", "group": i, "file": row["file"], "sha256": row["sha256"], "split": row["split"],
               "offset_s": None} for i, row in enumerate(parent["rows"], 1)]
    conversation = [{"bank": "conversation", "group": i, "file": row["name"] + ".webm",
                     "sha256": row["source_sha256"], "offset_s": row["offset_s"] + 12, "split": "held"}
                    for i, row in enumerate(read(CONVERSATION)["source_plan"]["rows"], 1)]
    for folder, rows in ((actor_dir, actors), (conversation_dir, conversation)):
        require(all(sha(folder / row["file"]) == row["sha256"] for row in rows),
                "A frozen original recording differs from its commitment")
    return {"schema": SCHEMA, "plans": {"acted_receipt_sha256": sha(actor_dir / "intake-receipt.json"),
                                        "conversation_receipt_sha256": sha(CONVERSATION)},
            "sources": actors + conversation,
            "method_sha256": sha(__file__),
            "review_sha256": sha(Path(__file__).with_name("review_video_lipsync_window_schedule.py")),
            "implementation_sha256": {name: sha(ROOT / "backend/app/detectors/video" / name)
                                      for name in ("lipsync.py", "xcorr.py", "adapters/lipsync_xcorr.py")},
            "face_weights_sha256": FACE_SHA, "arms": {key: list(value) for key, value in ARMS.items()},
            "confirmations": []}


def variants(pcm, ticks, group):
    """Place distinct known slips using only recording duration and group."""
    import numpy as np

    from app.detectors.video.lipsync import HOP, SR
    from scripts.video_lipsync_local_delay_transfer import shift

    start = (ticks - WIDTHS[0]) // 2 + (group % 5) - 2
    width = WIDTHS[group % 2]
    end = start + width
    left, right = start * HOP, end * HOP
    delta = LAGS[group % 2]
    changed_span = (left, right)
    scale = np.ones(right - left)
    scale[:FADE] = np.arange(FADE) / FADE
    scale[-FADE:] = scale[:FADE][::-1]
    source = pcm[left:right]

    def change(value):
        changed = pcm.copy()
        changed[left:right] = np.clip(np.rint(source * (1 - scale) + value * scale), -32768, 32767).astype("<i2")
        return changed

    yield "original", pcm, None, None
    yield "gain", change(source * .55), None, None
    filtered = np.convolve(source, np.ones(7) / 7, mode="same")
    yield "lowpass", change(filtered), None, None
    rng = np.random.default_rng(83141 + group)
    noise = rng.normal(0, 1, right - left)
    yield "noise", change(source + .70 * np.std(source.astype(np.float64)) * noise), None, None
    for sign, suffix in ((-1, "early"), (1, "late")):
        signed = sign * delta
        moved = shift(pcm, signed * HOP)
        yield "global_" + suffix, moved, None, None
        local = change(moved[left:right])
        require(np.count_nonzero(local[left:right] != source) >= SR // 20, "Local intervention has too few changed samples")
        yield "local_" + suffix, local, changed_span, signed


def _assess(faces, patches, pcm, grid):
    import numpy as np

    from app.detectors.video import lipsync, xcorr
    from app.detectors.video.adapters.lipsync_xcorr import _mouth_status, _track_sync

    speech = lipsync.vad(pcm)
    aperture = {tick: xcorr.aperture(patch) for tick, patch in patches.items()}
    rows = {}
    for name, (hop, margin) in ARMS.items():
        plan = lipsync.track_windows("track:0", faces, grid, speech, hop=hop, max_windows=240)
        windows = [w for w in plan if w.start_tick >= margin and w.ticks.stop + margin <= speech.ticks
                   and _mouth_status(w, patches, aperture) == "measurable"]
        if not windows or lipsync.speaker_score(windows, patches, speech.level) < lipsync.ATTRIBUTION_MIN:
            rows[name] = {"starts": [], "ranges": []}
            continue
        if margin == lipsync.VSHIFT:
            scored = _track_sync(windows, aperture, speech.level)
        else:
            scores = []
            for window in windows:
                lo = lipsync.VSHIFT - margin
                hi = lipsync.VSHIFT + margin + 1
                values = np.asarray([aperture[tick] for tick in window.ticks])
                correlation = xcorr.correlation_profile(values, speech.level, window.start_tick)[lo:hi]
                winner = int(np.argmax(correlation))
                peak = float(correlation[winner])
                distance = np.full((1, 2 * lipsync.VSHIFT + 1), np.inf)
                distance[0, lo:hi] = 1.0 - correlation
                scores.append(lipsync.WindowSync(window, (window.start_tick,), distance, winner - margin, 1.0 - peak, peak))
            gaps = set(range(windows[0].index, windows[-1].index)) - {w.index for w in windows}
            scored = lipsync.track_sync(scores, xcorr.SETTINGS, unmeasured_indices=sorted(gaps))
        rows[name] = {"starts": [w.start_tick for w in scored.windows],
                      "ranges": [[segment.kind, round(2 * segment.start_tick), round(2 * segment.end_tick),
                                  None if segment.offset is None else segment.offset - segment.baseline]
                                 for segment in (*scored.steps, *scored.collapses)]}
    return rows


def _observe(row, actor_dir, conversation_dir, detector, work):
    from scripts.video_lipsync_local_delay_transfer import inputs

    if row["bank"] == "acted":
        return inputs(actor_dir, row, detector)
    path = work / "excerpt.mov"
    source = conversation_dir / row["file"]
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-ss", str(row["offset_s"]), "-i", str(source),
                    "-map", "0:v:0", "-map", "0:a:0", "-t", str(EXCERPT_TICKS / 25),
                    "-vf", "fps=25,scale=480:360:force_original_aspect_ratio=decrease,pad=480:360:(ow-iw)/2:(oh-ih)/2,setsar=1",
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p", "-threads", "1",
                    "-c:a", "pcm_s16le", "-ar", "16000", "-ac", "1", "-y", str(path)],
                   check=True, timeout=30, capture_output=True)
    return inputs(work, {"file": path.name, "sha256": sha(path)}, detector)


def score(lock, actor_dir, conversation_dir):
    import cv2
    import numpy as np

    from app.detectors.image.faces import load_yunet
    from app.detectors.video.lipsync import HOP

    require(lock == freeze(actor_dir, conversation_dir), "The experiment changed since the source freeze")
    require(sha(FACE_FILE) == FACE_SHA, "The geometry asset differs from the frozen weights")
    cv2.setNumThreads(2)
    detector = load_yunet(FACE_FILE)
    with TemporaryDirectory(prefix="lip-window-") as scratch:
        for row in lock["sources"]:
            try:
                grid, table, origin, pcm, _, faces, patches, frames = _observe(
                    row, actor_dir, conversation_dir, detector, Path(scratch))
            except (RuntimeError, ValueError, subprocess.CalledProcessError) as exc:
                lock["confirmations"].append({"bank": row["bank"], "group": row["group"],
                                              "unavailable": type(exc).__name__})
                continue
            if min(grid.ticks, len(pcm) // HOP) < 55:
                lock["confirmations"].append({"bank": row["bank"], "group": row["group"], "unavailable": "duration"})
                continue
            cases = []
            for role, wave, truth, lag in variants(pcm, grid.ticks, row["group"]):
                cases.append({"case": role, "truth_samples": truth, "truth_lag_ticks": lag,
                              "pcm_sha256": hashlib.sha256(wave.tobytes()).hexdigest(),
                              "arms": _assess(faces, patches, wave.astype(np.float32) / 32768, grid)})
            require({case["case"] for case in cases} == set(CASES), "Incomplete paired deliveries")
            lock["confirmations"].append({"bank": row["bank"], "group": row["group"], "pcm_samples": len(pcm),
                                          "source_video_time_base": table.time_base, "audio_origin_s": str(origin),
                                          "verified_pictures": len(frames), "available_mouth_ticks": len(patches),
                                          "cases": cases})
            print(row["bank"], row["group"], grid.ticks, flush=True)
    return lock


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("phase", choices=("freeze", "score"))
    cli.add_argument("actor_dir", type=Path)
    cli.add_argument("conversation_dir", type=Path)
    cli.add_argument("run_dir", type=Path)
    args = cli.parse_args()
    lock = args.run_dir / "freeze.json"
    if args.phase == "freeze":
        args.run_dir.mkdir(mode=0o700, exist_ok=False)
        write(lock, freeze(args.actor_dir, args.conversation_dir))
    else:
        measured = score(read(lock), args.actor_dir, args.conversation_dir)
        measured["outcome"] = review(measured)
        write(args.run_dir / "receipt.json", measured)
        print(json.dumps(measured["outcome"], sort_keys=True))


if __name__ == "__main__":
    main()
