# SPDX-License-Identifier: Apache-2.0
"""Author word-timed conventional mouth edits and verify their native evidence."""

from __future__ import annotations

import argparse
import hashlib
import subprocess
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np

from app.detectors.image.faces import load_yunet
from app.detectors.video.decode import decode_frames, probe_frames
from app.evidence.video_reference import compare
from scripts import grid_reserve_word_clocks as clocks
from scripts import review_video_lipsync_local_delay_transfer as audio
from scripts import video_mouth_reconstruction_transfer as geometry
from scripts.video_common_sources_v89 import read, require, sha, write

SHIFT = 8
CONTROL_TYPES = ("ordinary_mouth_retime", "ordinary_mouth_brightness")
SPEC = {
    "schema": "chipmunk.grid-word-timed-ordinary-controls/1",
    "sources": "First qualified literal-label source and first qualified remapped-label source.",
    "word": "Select the word whose midpoint is closest to 1.35 seconds among published intervals of "
            "at least 0.14 seconds with endpoints in 0.62–2.46 seconds.",
    "picture_support": "Use the first native source picture at or after each word endpoint. "
                       "Original driver pictures are exactly eight native ticks earlier for the mouth-only retime.",
    "geometry": "Commercially permitted face landmarks register the original donor eye positions to the target eye positions. "
                "A fixed ellipse feathers an eye-normalized mouth region. The other control brightens that same area.",
    "roles": "Conventional controls. These are neither neural redraws nor genuine originals.",
}


def selection(records):
    picked = [next(item for item in records if item["literal_label"] == literal) for literal in (True, False)]
    plans = []
    for item in picked:
        candidates = []
        for ordinal, word in enumerate(item["words"]):
            first, last = map(Fraction, word["source_seconds"])
            if Fraction(62, 100) <= first < last <= Fraction(246, 100) and last-first >= Fraction(14, 100):
                candidates.append((abs((first+last)/2 - Fraction(135, 100)), ordinal, word))
        require(candidates, "A fixed reserve source has no eligible word interval")
        _, _, selected = min(candidates)
        plans.append({"ordinal": item["ordinal"], "source_sha256": item["source_sha256"],
                      "mapped_word_seconds": selected["source_seconds"], "word_samples": selected["source_pcm_samples"],
                      "output_frames_half_open": selected["source_native_picture_indices"]})
    return plans


def mouth_alpha(shape, transform, center):
    y, x = np.mgrid[:shape[0], :shape[1]]
    xref = transform[0,0]*x + transform[0,1]*y + transform[0,2] - center[0]
    yref = transform[1,0]*x + transform[1,1]*y + transform[1,2] - center[1]
    radius = np.sqrt((xref/16.)**2 + (yref/9.)**2)
    return np.clip((1-radius)/.25, 0, 1).astype(np.float32)


def original_pair(originals, derived, index):
    target = originals[index]
    _, target_matrix, mouth, shape = derived[index]
    _, donor_matrix, _, _ = derived[index-SHIFT]
    xform = cv2.invertAffineTransform(target_matrix) @ np.vstack((donor_matrix, [0,0,1]))
    shifted = cv2.warpAffine(originals[index-SHIFT], xform, shape[::-1], flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    brighter = np.clip(target.astype(np.int16) + 12, 0, 255).astype(np.uint8)
    alpha = mouth_alpha(shape, target_matrix, mouth)[..., None]
    results = [np.uint8(np.rint(target.astype(np.float32)*(1-alpha) + candidate.astype(np.float32)*alpha))
               for candidate in (shifted, brighter)]
    require(all(np.any(result != target) and np.array_equal(result[alpha[..., 0] == 0], target[alpha[..., 0] == 0])
                for result in results), "Word control lacks exact local changed pixels")
    return dict(zip(CONTROL_TYPES, results, strict=True))


def encode(path, pictures, wav):
    height, width = pictures[0].shape[:2]
    command = ["ffmpeg", "-nostdin", "-v", "error", "-n", "-f", "rawvideo", "-pixel_format", "rgb24",
               "-video_size", f"{width}x{height}", "-framerate", "25", "-i", "pipe:0", "-i", str(wav),
               "-map", "0:v:0", "-map", "1:a:0", "-c:v", "ffv1", "-pix_fmt", "bgr0", "-c:a", "pcm_s16le", str(path)]
    subprocess.run(command, input=b"".join(frame.tobytes() for frame in pictures), check=True, timeout=45)


def verify(path, original, native, table, pcm, origin, interval, authored):
    delivered = probe_frames(path, timeout_s=15)
    source_clock = [value*table.tb for value in table.pts]
    require((delivered.width, delivered.height) == (table.width, table.height)
            and [value*delivered.tb for value in delivered.pts] == source_clock, "Delivery changed a native picture timestamp")
    edited = []
    for frame in decode_frames(path, delivered, None, timeout_s=30):
        require(frame.pts_verified and frame.pts == delivered.pts[frame.index]
                and np.array_equal(frame.rgb, authored[frame.index]), "Control delivery differs from the authored pictures")
        mask = np.any(frame.rgb != native[frame.index], axis=2)
        if mask.any():
            edited.append({"frame": frame.index, "source_pts": frame.pts, "changed_pixels": int(mask.sum()),
                           "exact_changed_mask_sha256": hashlib.sha256(mask.tobytes()).hexdigest()})
    require([row["frame"] for row in edited] == list(range(*interval)), "Source changes fall outside the planned word pictures")
    decoded, start, discarded = audio.source_pcm(path, sha(path))
    require(not discarded and start == origin and np.array_equal(decoded, pcm), "Control delivery changed the source soundtrack")
    report, assets = compare(original, path)
    [evidence] = report["picture"]["ranges"]
    require([evidence["start_frame"], evidence["end_frame_exclusive"]] == interval
            and set(evidence["previews"]) == {"before", "during", "after"} and len(assets) == 6
            and report["audio"]["changed_samples"] == 0 and not report["affects_overall_confidence"],
            "The application's independent native comparison differs from the exact authoring")
    return {
        "control_sha256": sha(path), "edited_frame_count": len(edited), "native_pictures": len(delivered),
        "unchanged_pcm_samples": len(decoded), "exact_changed_masks": edited,
        "comparison_seconds": [evidence["start_s"], evidence["end_s"]],
        "comparison_before_during_after_frames": {role: item["frame"] for role, item in evidence["previews"].items()},
    }


def build(input_folder, output):
    qualification = read(input_folder / "qualification.json")
    path = input_folder / "private-word-clocks.json"
    require(sha(path) == qualification["private_word_clocks_sha256"], "Qualified word records differ")
    timings = read(path)
    sources, plan_sha = clocks.plan()
    require(timings["plan_sha256"] == plan_sha and timings["protocol"] == clocks.SPEC, "Source or timing protocol differs")
    plans = selection(timings["records"])
    admitted = read(geometry.ROOT / "docs/evaluation-panels/native-mouth-renderer-freeze-v118.json")["face_model_record"]
    weight_sha = sha(geometry.WEIGHT)
    require(admitted["licence"]["class_"] == "commercial"
            and admitted["files"][str(geometry.WEIGHT.relative_to(geometry.ROOT))] == weight_sha,
            "The exact commercially qualified face geometry weight is required")
    output.mkdir(mode=0o700)
    write(output / "freeze.json", {"plans": plans, "scope": SPEC, "words_sha256": sha(path),
                                  "implementation_sha256": sha(Path(__file__)), "face_weight_sha256": weight_sha})
    cv2.setNumThreads(1)
    detector = load_yunet(geometry.WEIGHT)
    results = []
    for plan in plans:
        source = sources[plan["ordinal"]]["input"]
        table, pictures, pcm, origin = clocks.exact_native(source)
        original = Path(source["source_file"])
        begin, end = plan["output_frames_half_open"]
        needed = {*range(begin,end), *range(begin-SHIFT,end-SHIFT)}
        derived = {index: geometry.aligned_face(pictures[index], detector) for index in sorted(needed)}
        require(all(observed is not None for observed in derived.values()), "The fixed control lacks observed mouth geometry")
        replacements = {index: original_pair(pictures, derived, index) for index in range(begin,end)}
        for arm in CONTROL_TYPES:
            authored = [replacements[index][arm] if index in replacements else pixels for index, pixels in enumerate(pictures)]
            dest = output / f"{plan['ordinal']:02}-{arm}.mkv"
            encode(dest, authored, Path(source["wav_file"]))
            result = verify(dest, original, pictures, table, pcm, origin, [begin,end], authored)
            results.append({"scope": arm, "private_source_ordinal": plan["ordinal"], **result})
            print("verified", plan["ordinal"], arm, flush=True)
    write(output / "controls.json", {"status": "verified", "controls": results, "freeze_sha256": sha(output / "freeze.json")})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("word_qualification", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    build(args.word_qualification, args.output)


if __name__ == "__main__":
    main()
