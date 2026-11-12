# SPDX-License-Identifier: Apache-2.0
"""Replay the source recipes, selected crops, and compression stability outcomes."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts import video_face_compression_stability as study

WEIGHT = study.ROOT / "models/weights/face_recognition_sface_2021dec.onnx"
WEIGHT_SHA256 = "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79"
POINT = .11243900750879454


def require(ok, message):
    if not ok:
        raise ValueError(message)


def pixels(array):
    return hashlib.sha256(array.tobytes()).hexdigest()


def authoring(folder, archive):
    import cv2
    import numpy as np

    cv2.setNumThreads(1)
    panel = study.panel()
    freeze = study.frozen(folder)
    manifest = study.read(folder / "media.json")
    require(manifest["freeze_sha256"] == study.digest(folder / "freeze.json"), "Media lost its source freeze")
    cases = {(row["role"], row["group"], row["condition"], row["treatment"]): row for row in manifest["cases"]}
    counts = Counter()
    with TemporaryDirectory(dir=folder, prefix="verify-compression-") as temp:
        tmp = Path(temp)
        for entry in freeze["sources"]:
            start, end = entry["span"]
            raw = list(panel.earlier.source_pictures(archive, entry["pin"], "original", range(120)).values())
            replacement = list(panel.earlier.source_pictures(archive, entry["pin"], entry["variant"], range(start, end)).values())
            require(len(raw) == 120 and len(replacement) == end - start, "Native source is incomplete")
            counts["original_source_pictures"] += len(raw)
            counts["publisher_source_pictures"] += len(replacement)
            changed = [i for i, frame in enumerate(replacement, start) if not np.array_equal(frame, raw[i])]
            swapped = raw[:start] + replacement + raw[end:]
            for content, frames in (("original", raw), ("swap", swapped)):
                direct = tmp / "first.mp4"
                output = tmp / "second.mp4"
                panel.old.encoded.encode(direct, frames=iter(frames), crf=23)
                for label in ("direct23", "direct36", "tandem23", "tandem36"):
                    case = cases.pop((entry["role"], entry["group"], content, label))
                    if label == "direct36":
                        panel.old.encoded.encode(output, frames=iter(frames), crf=36)
                    elif label.startswith("tandem"):
                        panel.old.encoded.encode(output, source=direct, crf=int(label.removeprefix("tandem")))
                    path = folder / "media" / case["file"]
                    expected = direct if label == "direct23" else output
                    require(case["native_changed_indices"] == (changed if content == "swap" else [])
                            and case["truth"] == ([start, end] if content == "swap" else None)
                            and study.digest(expected) == study.digest(path) == case["sha256"],
                            "Rendered output, treatment or exact native labels differ")
                    counts["byte_identical_mp4s"] += 1
                    if expected == output:
                        output.unlink()
                direct.unlink()
            print(f"Independently rebuilt {entry['role']} source {entry['group']:02d}", flush=True)
    require(not cases and counts["byte_identical_mp4s"] == len(manifest["cases"]), "Authoring replay is incomplete")
    receipt = {**counts, "media_sha256": study.digest(folder / "media.json")}
    study.write(folder / "independent-authoring.json", receipt)


def cosine_distance(left, right):
    dot = float(left @ right)
    norm = math.sqrt(float((left @ left) * (right @ right)))
    return 1. - max(-1., min(1., dot / norm))


def score(centers, embeddings):
    before, during, after = (sum(sum(embeddings[i + k] for k in (-1, 0, 1)) for i in centers[role])
                             / (3 * len(centers[role])) for role in ("before", "during", "after"))
    return min(cosine_distance(during, before), cosine_distance(during, after)) - .35 * cosine_distance(before, after)


def verify_range(folder, item, record, repeat, recognizer):
    import cv2
    import numpy as np

    from app.detectors.video.decode import FrameTable, decode_frames

    table = FrameTable.from_json(item["clock"])
    inputs = {entry["frame_index"]: entry for entry in record["frame_inputs"]}
    reduced_hash = {entry["index"]: entry["low_rgb_sha256"] for entry in repeat["inputs"]}
    idx = sorted(inputs)
    require(set(idx) == set(reduced_hash) and set(idx) == {
        i + k for centers in record["centers"].values() for i in centers for k in (-1, 0, 1)
    }, "Measured face inputs do not match their complete window roles")
    native_features, reduced_features = {}, {}
    for frame, index in zip(decode_frames(folder / "media" / item["file"], table, idx, timeout_s=30), idx, strict=True):
        source = inputs[index]
        require(frame.index == index and frame.pts_verified and frame.pts == source["pts"] == table.pts[index]
                and pixels(frame.rgb) == source["decoded_rgb_sha256"] == item["rgb_sha256"][index],
                "The native frame or PTS differs from the stored selected evidence")
        affine = np.array([[float.fromhex(cell) for cell in row] for row in source["alignment_matrix_hex"]])
        face = cv2.warpAffine(frame.rgb, affine, (112, 112), flags=cv2.INTER_LINEAR)
        reduced = cv2.resize(cv2.resize(face, (56, 56), interpolation=cv2.INTER_AREA), (112, 112), interpolation=cv2.INTER_LINEAR)
        require(pixels(face) == source["aligned_rgb_sha256"] and pixels(reduced) == reduced_hash[index],
                "Original aligned face or low-pass input differs")
        for values, rgb in ((native_features, face), (reduced_features, reduced)):
            vector = recognizer.feature(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)).reshape(-1).astype(np.float64)
            values[index] = vector / math.sqrt(float(vector @ vector))
    raw, low = (score(record["centers"], vector) for vector in (native_features, reduced_features))
    require(abs(raw - record["response"]) <= 2e-6 and abs(raw - repeat["original_response"]) <= 2e-6
            and abs(low - repeat["response"]) <= 2e-6 and repeat["point"] == POINT
            and (low <= POINT) == repeat["suppressed"], "Replayed responses or suppression decision differ")
    return len(idx)


def span(result, clock):
    if result is None:
        return None
    require(result["time_base"] == clock["time_base"], "Selected clock differs")
    ticks = clock["pts"]
    a, b = ticks.index(result["start_pts"]), ticks.index(result["end_pts"])
    require(a < b, "Invalid temporal range")
    return a, b


def useful(ranges, truth):
    if not truth:
        return False
    a, b = truth
    return any(2 * max(0, min(end, b) - max(start, a)) >= max(end, b) - min(start, a)
               for start, end in (x for x in ranges if x is not None))


def outcomes(folder, publication):
    import cv2

    cv2.setNumThreads(1)
    require(study.digest(WEIGHT) == WEIGHT_SHA256, "The admitted CPU weights differ")
    recognizer = cv2.FaceRecognizerSF.create(str(WEIGHT), "", cv2.dnn.DNN_BACKEND_OPENCV, cv2.dnn.DNN_TARGET_CPU)
    manifest = study.read(folder / "media.json")
    author = study.read(folder / "independent-authoring.json")
    seal = study.read(folder / "score-seal.json")
    journal = folder / "results.jsonl"
    require(author["media_sha256"] == seal["media_sha256"] == study.digest(folder / "media.json")
            and seal["journal_sha256"] == study.digest(journal), "Study receipts are detached from exact source")
    counts, groups = defaultdict(Counter), defaultdict(set)
    measured_pictures = 0
    measured_ranges = 0
    rows = [json.loads(line) for line in journal.read_text().splitlines()]
    require(len(rows) == seal["cases"] == author["byte_identical_mp4s"] == len(manifest["cases"]), "Incomplete held panel")
    for row, item in zip(rows, manifest["cases"], strict=True):
        require(all(row[k] == item[k] for k in ("file", "sha256", "condition", "role", "group", "treatment", "truth")),
                "Outcome differs from the source commitment")
        current = row["incumbent"]
        coarse, refinement, short = (span(current[key], item["clock"]) for key in ("coarse", "boundary_refinement", "short_range"))
        repeat = row["repeat"]
        require(bool(coarse) == bool(row["coarse_record"]) == bool(repeat), "Selected coarse range was not measured")
        suppressed = bool(repeat and repeat["suppressed"])
        if repeat:
            measured_pictures += verify_range(folder, item, row["coarse_record"], repeat, recognizer)
            measured_ranges += 1
        before = (coarse, refinement, short)
        after = (short,) if suppressed else before
        truth = row["truth"]
        key = f"{row['role']}/{row['condition']}/{row['treatment']}"
        false_removed = not truth and any(before) and not any(after)
        lost = useful(before, truth) and not useful(after, truth)
        counts[key].update({
            "files": 1, "current_flags": any(before), "coarse_flags": coarse is not None,
            "current_useful": useful(before, truth), "candidate_flags": any(after),
            "candidate_useful": useful(after, truth), "suppressed_coarse": suppressed,
            "false_removed": false_removed, "useful_losses": lost,
            "useful_coarse": useful((coarse, refinement), truth),
            "suppressed_useful_coarse": suppressed and useful((coarse, refinement), truth),
        })
        if false_removed:
            groups[row["role"]].add(row["group"])
    held = sum((value for key, value in counts.items() if key.startswith("held/")), Counter())
    gate = study.SPEC["gate"]
    advances = (held["false_removed"] >= gate["minimum_false_removals"]
                and len(groups["held"]) >= gate["minimum_distinct_held_sources"]
                and held["suppressed_useful_coarse"] <= gate["maximum_lost_useful_held_swaps"]
                and held["useful_losses"] <= gate["maximum_lost_useful_held_swaps"]
                and held["useful_coarse"] >= gate["minimum_held_true_coarse_support"])
    result = {"method": study.SPEC, "score_seal": seal, "independent_authoring": author,
              "outcomes": counts, "removed_false_groups": {role: len(value) for role, value in groups.items()},
              "verified_face_picture_occurrences": measured_pictures, "verified_coarse_ranges": measured_ranges,
              "advances": advances, "serving_change": False}
    study.write(publication, result)
    print(json.dumps({"held": held, "removed_false_held_groups": len(groups["held"]), "advances": advances,
                      "verified_coarse_ranges": measured_ranges, "verified_face_picture_occurrences": measured_pictures},
                     sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("authoring", "outcomes"))
    parser.add_argument("folder", type=Path)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--publication", type=Path)
    args = parser.parse_args()
    if args.phase == "authoring":
        authoring(args.folder, args.archive or study.panel().ARCHIVE)
    else:
        outcomes(args.folder, args.publication or args.folder / "result.json")


if __name__ == "__main__":
    main()
