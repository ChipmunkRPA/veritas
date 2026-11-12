# SPDX-License-Identifier: Apache-2.0
"""Evaluate appearance stability on matched single and tandem H.264 encodes."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from unittest.mock import patch

from scripts import screen_video_native_background_challenge as worker

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "backend/data/independent-evaluation"
DISCOVERY = DATA / "video-face-return-eligibility-2026-11-12"
HELD = DATA / "video-native-background-challenge-2026-11-12"
DISCOVERY_GROUPS = (0, 1, 3, 4)
TREATMENTS = {"direct23": (None, 23), "direct36": (None, 36),
              "tandem23": ("direct23", 23), "tandem36": ("direct23", 36)}
SPEC = {
    "schema": "chipmunk.video-face-compression-stability/1",
    "scope": "Silent upright 640x360 25 fps H.264 yuv420p; 120 native pictures. Four exposed genuine compression alerts and matching swaps; twelve distinct held source groups with originals and publisher swaps. Held sources had prior exposure at CRF 23.",
    "treatments": TREATMENTS,
    "candidate": "For a selected coarse range, measure the same aligned 112-pixel faces again after 112→56 area downsample and 56→112 linear upsample. Keep the exact same coarse centers, triplets, face alignment, track and response calculation. A low-pass response at or below the current point suppresses that coarse range and its dependent refinement. Retain all previously measured short ranges. Missing repeat evidence keeps the coarse range.",
    "gate": {"minimum_false_removals": 2, "minimum_distinct_held_sources": 2,
             "maximum_lost_useful_held_swaps": 0, "minimum_held_true_coarse_support": 4,
             "temporal_iou": "1/2"},
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def panel():
    from scripts import video_native_background_challenge as source

    return source


def freeze():
    from scripts.replay_video_native_background_panel import APP, BINDINGS

    discovery_manifest, held_manifest = (read(folder / "media.json")["media"] for folder in (DISCOVERY, HELD))
    discovery_lock, held_lock = (read(folder / "freeze.json") for folder in (DISCOVERY, HELD))
    known = [json.loads(line) for line in (DISCOVERY / "results.jsonl").read_text().splitlines()]
    alerts = {row["group"] for row in known if row["condition"] == "recompressed" and row["incumbent"]["coarse"]}
    if alerts != set(DISCOVERY_GROUPS):
        raise ValueError("The four exposed compression false alerts must be present")
    sources = []
    for group in DISCOVERY_GROUPS:
        rows = {row["condition"]: row for row in discovery_manifest if row["group"] == group}
        swap = rows["ten_odd"]
        sources.append({"role": "development", "group": group, "pin": discovery_lock["roster"][group],
                        "span": swap["truth"], "variant": swap["publisher_variant"],
                        "reference_original": rows["original"]["sha256"], "reference_swap": swap["sha256"],
                        "reference_tandem": rows["recompressed"]["sha256"]})
    roster = {pin["index"]: pin for pin in held_lock["roster"]}
    for pair, match in enumerate(held_lock["pairs"]):
        for direction, index in enumerate(match["groups"]):
            rows = {row["condition"]: row for row in held_manifest
                    if row["source_pair"] == pair and row["direction"] == direction}
            swap = rows["long_swap"]
            sources.append({"role": "held", "group": pair * 2 + direction, "pin": roster[index],
                            "span": swap["truth"], "variant": ("swap_a", "swap_b")[direction],
                            "reference_original": rows["original"]["sha256"], "reference_swap": swap["sha256"]})
    if len(sources) != 16 or len({row["pin"]["index"] for row in sources}) != 16:
        raise ValueError("The held and development groups must be distinct")
    return json.loads(json.dumps({
        "spec": SPEC, "sources": sources,
        "parents": {folder.name: {name: digest(folder / name) for name in names} for folder, names in (
            (DISCOVERY, ("freeze.json", "media.json", "score-seal.json", "results.jsonl")),
            (HELD, ("freeze.json", "media.json")),
        )},
        "implementation": {name: digest(APP / name) for name in BINDINGS},
        "worker": digest(Path(worker.__file__)), "script": digest(Path(__file__)),
    }))


def frozen(folder):
    lock = read(folder / "freeze.json")
    if lock != freeze():
        raise ValueError("Source, method, or serving files differ from the freeze")
    return lock


def prepare(folder, archive):
    import cv2
    import numpy as np

    cv2.setNumThreads(1)
    source = panel()
    lock = frozen(folder)
    out = folder / "media"
    out.mkdir(mode=0o700)
    cases = []
    for group in lock["sources"]:
        start, stop = group["span"]
        original = source.earlier.source_pictures(archive, group["pin"], "original", range(120))
        swap = source.earlier.source_pictures(archive, group["pin"], group["variant"], range(start, stop))
        # Archived long swaps may include native pictures where the generator leaves the host unchanged.
        changed = [i for i in swap if not np.array_equal(swap[i], original[i])]
        if len(changed) < .9 * (stop - start):
            raise ValueError("The pinned publisher interval lacks sufficient exact changed pictures")
        for content in ("original", "swap"):
            truth = group["span"] if content == "swap" else None
            deliveries = {}
            for treatment, (input_treatment, crf) in TREATMENTS.items():
                path = out / f"{group['role']}-{group['group']:02d}-{content}-{treatment}.mp4"
                if input_treatment:
                    source.old.encoded.encode(path, source=deliveries[input_treatment], crf=crf)
                else:
                    native = (swap.get(i, image) if truth else image for i, image in original.items())
                    source.old.encoded.encode(path, frames=native, crf=crf)
                deliveries[treatment] = path
                row = source.old.describe(path, group["group"], group["role"], content, truth)
                if treatment == "direct23" and row["sha256"] != group[f"reference_{content}"]:
                    raise ValueError("Rebuilt CRF 23 delivery differs from its earlier authenticated original")
                if (content == "original" and treatment == "tandem36" and group["role"] == "development"
                        and row["sha256"] != group["reference_tandem"]):
                    raise ValueError("Rebuilt tandem CRF 36 differs from the exposed alert")
                cases.append({**row, "role": group["role"], "treatment": treatment,
                              "input_treatment": input_treatment, "crf": crf,
                              "native_changed_indices": changed if truth else []})
        print(f"Prepared {group['role']} source {group['group']:02d}", flush=True)
    if len(cases) != 128 or len({row["sha256"] for row in cases}) != len(cases):
        raise ValueError("Expected 128 distinct matched encodes")
    write(folder / "media.json", {"freeze_sha256": digest(folder / "freeze.json"), "cases": cases})


def aligned_features(path, clock, record, recognizer):
    import cv2
    import numpy as np

    from app.detectors.video import face_appearance as method
    from app.detectors.video.decode import FrameTable, decode_frames

    table = FrameTable.from_json(clock)
    inputs = {row["frame_index"]: row for row in record["frame_inputs"]}
    aligned, low, hashes = {}, {}, []
    indices = sorted(inputs)
    for frame, i in zip(decode_frames(path, table, indices, timeout_s=20), indices, strict=True):
        row = inputs[i]
        if (not frame.pts_verified or frame.index != i or frame.pts != row["pts"]
                or hashlib.sha256(frame.rgb.tobytes()).hexdigest() != row["decoded_rgb_sha256"]):
            raise ValueError("Repeated face input differs from its native picture")
        bgr = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)
        crop = recognizer.alignCrop(bgr, method.face_row(row["landmarks"]))
        if hashlib.sha256(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).tobytes()).hexdigest() != row["aligned_rgb_sha256"]:
            raise ValueError("Repeated face alignment differs")
        smaller = cv2.resize(crop, (56, 56), interpolation=cv2.INTER_AREA)
        reduced = cv2.resize(smaller, (112, 112), interpolation=cv2.INTER_LINEAR)
        aligned[i] = method.unit(recognizer.feature(crop).reshape(-1))
        low[i] = method.unit(recognizer.feature(reduced).reshape(-1))
        hashes.append({"index": i, "low_rgb_sha256": hashlib.sha256(reduced[:, :, ::-1].tobytes()).hexdigest()})

    def measure(vectors):
        groups = []
        for role in ("before", "during", "after"):
            blocks = [np.mean([vectors[i + j] for j in (-1, 0, 1)], axis=0) for i in record["centers"][role]]
            groups.append(np.mean(blocks, axis=0))
        a, b, c = groups
        return min(method.distance(b, a), method.distance(b, c)) - method.FLANK_PENALTY * method.distance(a, c)

    before, after = measure(aligned), measure(low)
    if abs(before - record["response"]) > 2e-6:
        raise ValueError("Repeat at original resolution disagrees with the selected coarse response")
    return {"response": after, "original_response": before, "suppressed": after <= method.REVIEW_POINT,
            "point": method.REVIEW_POINT, "inputs": hashes}


def score(folder, state):
    from app.detectors.video.adapters import face_appearance as adapter

    frozen(folder)
    manifest = read(folder / "media.json")
    if manifest["freeze_sha256"] != digest(folder / "freeze.json"):
        raise ValueError("Media is detached from the frozen protocol")
    run = worker.scorer(state)
    measure, record = adapter._measure, adapter.record
    captures = {}

    def remember_face(rgb, bgr, index, table, face, recognizer):
        captures["recognizer"] = recognizer
        return measure(rgb, bgr, index, table, face, recognizer)

    def remember_window(window, table, inputs):
        row = record(window, table, inputs)
        captures.setdefault("first", row)
        return row

    journal = folder / "results.jsonl"
    started = time.monotonic()
    with journal.open("x") as stream, patch.object(adapter, "_measure", remember_face), patch.object(adapter, "record", remember_window):
        for count, delivery in enumerate(manifest["cases"], 1):
            captures.clear()
            path = folder / "media" / delivery["file"]
            incumbent = run(path, delivery)
            selected = incumbent["coarse"]
            review = None
            selected_record = None
            if selected:
                selected_record = captures["first"]
                if selected_record["start_pts"] != selected["start_pts"] or selected_record["end_pts"] != selected["end_pts"]:
                    raise ValueError("The measured window differs from the selected coarse result")
                review = aligned_features(path, delivery["clock"], selected_record, captures["recognizer"])
            keys = ("file", "sha256", "condition", "role", "group", "treatment", "truth")
            output = {**{key: delivery[key] for key in keys}, "incumbent": incumbent, "repeat": review,
                      "coarse_record": selected_record}
            stream.write(json.dumps(output, sort_keys=True, allow_nan=False) + "\n")
            stream.flush()
            if count % 8 == 0:
                print(f"Measured {count}/{len(manifest['cases'])} exact Full stage runs", flush=True)
    write(folder / "score-seal.json", {"cases": count, "journal_sha256": digest(journal),
          "media_sha256": digest(folder / "media.json"), "elapsed_s": time.monotonic() - started})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "prepare", "score"))
    parser.add_argument("folder", type=Path)
    parser.add_argument("--archive", type=Path)
    args = parser.parse_args()
    state = worker.configure(args.folder) if args.phase == "score" else None
    if args.phase == "freeze":
        args.folder.mkdir(mode=0o700)
        write(args.folder / "freeze.json", freeze())
    elif args.phase == "prepare":
        prepare(args.folder, args.archive or panel().ARCHIVE)
    else:
        score(args.folder, state)


if __name__ == "__main__":
    main()
