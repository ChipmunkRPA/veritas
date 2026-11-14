# SPDX-License-Identifier: Apache-2.0
"""Compare shorter repeated-audio nominations on a pinned short-video panel."""
from __future__ import annotations

import argparse
import json
import subprocess
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np

from app.detectors.image.faces import load_yunet
from app.evidence import audio_similarity
from scripts import review_video_adjacent_sound as review
from scripts import video_adjacent_sound_witness as method
from scripts import video_lipsync_window_schedule as parent
from scripts.video_common_sources_v89 import read, require, sha, write

DESIGN = {
    "input": "Same verified picture and decoded sound clock as the prior 26-source comparison. All groups are already exposed.",
    "comparison": "Current two-band audio calls and adjacent local-band audio calls, each with the same mouth measurement. "
                  "This compares nominations, independently of the incumbent app's previously recorded timing result.",
    "new_controls": "Move the corresponding mouth crops and sound together in both directions. "
                    "Add the same independent speech-band bed under each of two adjacent segments of ordinary audio.",
    "truth": "Both witnesses reach 0.50 temporal IoU. The first or second range may contain the reused samples. "
             "The match cannot determine which changed, or the full edit interval.",
    "rule": "At least four added useful pairs across at least three held groups, both banks represented, "
            "no lost useful pairs, at most two additional ordinary flags and zero synchronized or global flags.",
}


def freeze(actors, speech):
    old = parent.freeze(actors, speech)
    return {"schema": review.SCHEMA, "sources": old["sources"], "plans": old["plans"], "design": DESIGN,
            "bindings": {Path(path).name: sha(path) for path in (__file__, method.__file__, review.__file__,
                         parent.__file__, audio_similarity.__file__, parent.FACE_FILE)},
            "runtime": {"opencv": cv2.__version__, "numpy": np.__version__,
                        "ffmpeg": subprocess.check_output(["ffmpeg", "-version"], text=True).splitlines()[0]}}


def known_repeat(role, edited, lag):
    width = abs(lag) * method.PICTURE_HOP
    adjunct = edited[0] if role.endswith("late") else edited[1]
    return [[adjunct - width, adjunct], [adjunct, adjunct + width]]


def variants(pcm, ticks, group, native_frames, patches):
    moved = []
    for role, wave, span, lag in parent.variants(pcm, ticks, group):
        truth = known_repeat(role, span, lag) if span else None
        yield role, wave, truth, native_frames, patches
        if truth:
            moved.append((role, wave, span, lag, truth))
    for role, wave, (a, b), lag, _ in moved:
        start, end = a // method.PICTURE_HOP, min(len(native_frames), b // method.PICTURE_HOP)
        indices = [i - lag if start <= i < end else i for i in range(len(native_frames))]
        yield ("matched_picture_" + role.removeprefix("local_"), wave, None,
               [native_frames[i] if 0 <= i < len(native_frames) else -1 for i in indices],
               {i: patches[source] for i, source in enumerate(indices) if source in patches})
    a, b = moved[1][-1]
    rng = np.random.default_rng(1422 + group)
    profile = np.hanning(a[1] - a[0]) * rng.normal(0, 1, a[1] - a[0])
    clip = pcm.astype(np.float64)
    rms = min(np.std(clip[slice(*span)]) for span in (a, b))
    for span in (a, b):
        clip[slice(*span)] += profile * (.8 * rms / max(1e-5, np.std(profile)))
    yield "repeated_bed", np.clip(np.rint(clip), -32768, 32767).astype("<i2"), None, native_frames, patches


def _hash(values):
    return sha256(values.tobytes()).hexdigest()


def assess(wave, frames, patches):
    current = audio_similarity.find_similar_spans(wave.tobytes())
    current_pairs = [(span.first, span.second) for span in current.spans] if current.status == "measured" else None
    wave_f = wave.astype(np.float64) / 32768
    proposed = method.adjacent_repeats(wave_f)
    speech = method.lipsync.vad(wave_f).speech
    measured = len(patches) >= .75 * len(frames)

    def corroborate(pairs):
        if pairs is None or not measured:
            return None
        result = []
        for pair in pairs:
            if min(np.mean(speech[a // method.PICTURE_HOP:(b + method.PICTURE_HOP - 1) // method.PICTURE_HOP])
                   for a, b in pair) < .4:
                continue
            if observation := method.visual_support(pair, frames, patches):
                result.append({"ranges": pair, "pcm_sha256": [_hash(wave[a:b]) for a, b in pair], **observation})
        return result

    return {"existing_audio": current_pairs, "adjacent_audio": proposed,
            "existing_mouth": corroborate(current_pairs), "adjacent_mouth": corroborate(proposed)}


def score(lock, actors, speech):
    require(lock == freeze(actors, speech), "Source plan, detector or experiment changed after freeze")
    require(sha(parent.FACE_FILE) == parent.FACE_SHA, "Face detector differs from its pinned weights")
    cv2.setNumThreads(1)
    detector = load_yunet(parent.FACE_FILE)
    rows = []
    with TemporaryDirectory(prefix="adjacent-sound-") as scratch:
        for row in lock["sources"]:
            result = {"bank": row["bank"], "group": row["group"], "source_sha256": row["sha256"]}
            rows.append(result)
            try:
                grid, table, origin, pcm, _, _, patches, frames = parent._observe(row, actors, speech, detector, Path(scratch))
            except (RuntimeError, ValueError, subprocess.CalledProcessError) as error:
                result["unavailable"] = type(error).__name__
                continue
            if min(grid.ticks, len(pcm) // method.PICTURE_HOP) < 55:
                result["unavailable"] = "duration"
                continue
            result.update(samples=len(pcm), picture_pts_sha256=sha256(repr(table.pts).encode()).hexdigest(),
                          picture_time_base=table.time_base, audio_origin_s=str(origin), observed_mouth_ticks=len(patches),
                          last_verified_frame=max(frames), cases=[])
            for role, wave, truth, frame_grid, visual in variants(pcm, grid.ticks, row["group"], grid.frames, patches):
                result["cases"].append({"id": role, "truth_samples": truth, "pcm_sha256": _hash(wave),
                                        "arms": assess(wave, frame_grid, visual)})
            print(row["bank"], row["group"], flush=True)
    return json.loads(json.dumps({"freeze": lock, "rows": rows}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "score"))
    parser.add_argument("actor_sources", type=Path)
    parser.add_argument("conversation_sources", type=Path)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    if args.phase == "freeze":
        args.run.mkdir(mode=0o700)
        write(args.run / "freeze.json", freeze(args.actor_sources, args.conversation_sources))
    else:
        document = score(read(args.run / "freeze.json"), args.actor_sources, args.conversation_sources)
        document["outcome"] = review.review(document)
        write(args.run / "receipt.json", document)
        print(json.dumps(document["outcome"], sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
