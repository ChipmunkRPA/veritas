# SPDX-License-Identifier: Apache-2.0
"""Test fixed-band soundtrack witnesses on held speakers and encoded deliveries."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from scripts import review_video_adjacent_sound as replay
from scripts.audio_vocal_event_mixtures import read, sha
from scripts.audio_vocal_event_mixtures import save as write

require = replay.require

ROOT = Path(__file__).resolve().parents[2]
PANELS = ROOT / "docs/evaluation-panels"
DEVELOPMENT = PANELS / "video-lipsync-repeat-public-speech-sources-2026-11-13.json"
PREVIOUS = PANELS / "video-adjacent-sound-delivery-2026-11-14.json"
HELD = PANELS / "video-local-alignment-2026-11-14.json"
ARMS = ("existing", "three_band", "same_two_bands")
CARRIERS = ("PCM", "AAC")
SCHEMA = "chipmunk.video-repeated-speech-band-transfer/1"
WINDOW, HOP = 63, 640
BANDS = ((220, 900), (900, 2100), (2100, 4100))
NOISY = (*replay.EDITS, "matched_picture_early", "matched_picture_late", "original")
CONDITIONS = {"clean": (*replay.EDITS, *replay.CONTROLS), "competing": NOISY, "train": NOISY}
GATE = {"new_aac_noisy": 3, "new_aac_sources": 2, "max_aac_new_controls": 1,
        "max_pcm_new_controls": 1, "max_new_synchronized_or_constant": 0}


def plan():
    development = read(DEVELOPMENT)["sources"]
    selected = {row["id"]: row for row in read(PREVIOUS)["freeze"]["sources"]}
    held = [row for row in read(HELD)["freeze"]["sources"] if row["split"] == "held"]
    known = [{"id": row["id"], "file": row["id"] + ".webm", "sha256": row["sha256"],
              "page": row["page"], "license": row["license"], "attribution": row["attribution"],
              "split": "comparison", "offsets": [selected[row["id"]]["offset"]]} for row in development]
    holdout = [row | {"id": row["file"].removesuffix(".webm"), "offsets": [50, 62]} for row in held]
    require(not {row["sha256"] for row in known}.intersection(row["sha256"] for row in holdout),
            "Comparison and held recordings overlap")
    return {"schema": SCHEMA, "sources": known + holdout, "bands_hz": [list(band) for band in BANDS],
            "bindings": {file.name: sha(file) for file in (DEVELOPMENT, PREVIOUS, HELD)},
            "gate": GATE, "method": "The fixed prior 60 ms search, speech gate and native mouth comparison are shared. "
                      "Compare three-band agreement and two fixed agreeing bands as alternatives to the existing two bands. "
                      "Re-encode shifted-picture controls and 48 kbps AAC audio. Use 0.50 overlap for both repeated intervals. "
                      "An accepted variant would supplement existing evidence, without assigning an edit direction or AI attribution."}


def _case_keys():
    return [(scene, role, carrier) for scene, roles in CONDITIONS.items() for role in roles for carrier in CARRIERS]


def _truth(role, group):
    if role not in replay.EDITS:
        return None
    start = ((WINDOW - 16) // 2 + (group % 5) - 2) * HOP
    width, lag = ((16, 5), (22, 7))[group % 2]
    hinge = start if role == "local_late" else start + width * HOP
    return [[hinge - lag * HOP, hinge], [hinge, hinge + lag * HOP]]


def recount(document):
    freeze = plan()
    require(document["freeze"] == freeze, "Frozen source roster changed")
    expected = [(index, source, offset) for index, source in enumerate(freeze["sources"], 301) for offset in source["offsets"]]
    tallies, contrasts, gains = defaultdict(Counter), defaultdict(Counter), defaultdict(set)
    for (group, source, offset), result in zip(expected, document["rows"], strict=True):
        require(result["source"] == source["id"] and result["sha256"] == source["sha256"]
                and result["offset"] == offset and result["group"] == group, "Unexpected source measurement")
        require([(c["scene"], c["role"], c["carrier"]) for c in result["cases"]] == _case_keys(), "Missing or reordered delivery")
        pictures = result["pictures"]
        require(len(pictures) == WINDOW and all(replay.SHA.fullmatch(p) for p in pictures), "Source picture commitments differ")
        control_pictures = result["retimed_pictures"]
        require(set(control_pictures) == {"matched_picture_early", "matched_picture_late"}
                and all(len(pics) == WINDOW and pics != pictures and all(replay.SHA.fullmatch(p) for p in pics)
                        for pics in control_pictures.values()), "Retimed controls lack separately decoded pictures")
        split = source["split"]
        for case in result["cases"]:
            scene, role, carrier = (case[k] for k in ("scene", "role", "carrier"))
            truth = _truth(role, group)
            require(case["truth"] == truth and replay.SHA.fullmatch(case["pcm_sha256"]) and set(case["arms"]) == set(ARMS),
                    "Delivery clock or arm binding differs")
            counts = tallies[split, carrier, scene]
            counts["edits" if truth else "controls"] += 1
            found, alerts = {}, {}
            for arm in ARMS:
                records = case["arms"][arm]
                counts[f"{arm}_assessed_{'edits' if truth else 'controls'}"] += records is not None
                for item in records or []:
                    pair = item["ranges"]
                    replay._validate_pair(pair, WINDOW * HOP)
                    require(len(item["frame_pairs"]) >= 3 and all(len(indices) == 2 and all(type(i) is int and 0 <= i < WINDOW
                            for i in indices) for indices in item["frame_pairs"]), "Invalid mouth witness frame")
                    require(len(item["pcm_sha256"]) == 2 and all(replay.SHA.fullmatch(digest) for digest in item["pcm_sha256"]),
                            "Missing source audio commitments")
                found[arm] = any(replay.useful(item["ranges"], truth) for item in records or [])
                alerts[arm] = bool(records)
                counts[f"{arm}_useful"] += found[arm]
                counts[f"{arm}_wrong_edits"] += bool(truth) and alerts[arm] and not found[arm]
                counts[f"{arm}_ordinary_flags"] += not truth and alerts[arm]
            for arm in ARMS[1:]:
                counts[f"{arm}_union_useful"] += found["existing"] or found[arm]
                counts[f"{arm}_union_flags"] += not truth and (alerts["existing"] or alerts[arm])
                if split != "held":
                    continue
                stats = contrasts[arm]
                if truth and found[arm] and not found["existing"]:
                    stats[f"new_{carrier.lower()}_{'noisy' if scene != 'clean' else 'clean'}"] += 1
                    if carrier == "AAC" and scene != "clean":
                        gains[arm].add(source["id"])
                if not truth and alerts[arm] and not alerts["existing"]:
                    stats[f"new_{carrier.lower()}_controls"] += 1
                    if role.startswith(("global_", "matched_picture_")):
                        stats["new_synchronized_or_constant"] += 1
    summary = {}
    for arm in ARMS[1:]:
        counts = contrasts[arm]
        result = {key: counts[key] for key in ("new_aac_noisy", "new_aac_clean", "new_pcm_noisy", "new_pcm_clean",
                                               "new_aac_controls", "new_pcm_controls", "new_synchronized_or_constant")}
        result["new_aac_sources"] = len(gains[arm])
        result["passes"] = (result["new_aac_noisy"] >= GATE["new_aac_noisy"] and result["new_aac_sources"] >= GATE["new_aac_sources"]
                            and result["new_aac_controls"] <= GATE["max_aac_new_controls"]
                            and result["new_pcm_controls"] <= GATE["max_pcm_new_controls"]
                            and result["new_synchronized_or_constant"] <= GATE["max_new_synchronized_or_constant"])
        summary[arm] = result
    return {"cells": {"/".join(key): dict(value) for key, value in tallies.items()}, "summary": summary}


def run(args, freeze):
    from tempfile import TemporaryDirectory

    import cv2

    from scripts import video_adjacent_sound_delivery as delivery
    from scripts import video_repeated_speech_bands as method
    from scripts.audio_reference_variable_window import read_noise

    base = delivery.video
    prior = delivery.transfer
    require(method.BANDS == BANDS and method.ARMS == ARMS
            and sha(delivery.parent.FACE_FILE) == delivery.parent.FACE_SHA, "Pinned band implementation or face weights differ")
    inputs = []
    cv2.setNumThreads(1)
    detector = base.load_yunet(delivery.parent.FACE_FILE)
    train = read_noise(args.sounds)["train"]
    with TemporaryDirectory(prefix="video-repeat-bands-") as scratch:
        temp = Path(scratch)
        for group, source in enumerate(freeze["sources"], 301):
            folder = args.held if source["split"] == "held" else args.comparison
            path = folder / source["file"]
            require(sha(path) == source["sha256"], "Pinned publisher recording differs")
            for offset in source["offsets"]:
                picture = temp / f"{group}-{offset}.mov"
                prepared = delivery.speech.prepare(path, picture, offset)
                require(prepared is not None, "Source did not provide a complete continuous 2.52 s excerpt")
                pixels, pcm = prepared[:2]
                table, _, patches, _, frames = base.video_inputs(picture, detector)
                require(len(pcm) == WINDOW * HOP, "The decoded audio differs from the expected sample count")
                inputs.append((source, group, offset, picture, table, pixels, pcm, patches, frames))
        rows = []
        for index, (source, group, offset, picture, table, pixels, pcm, patches, frames) in enumerate(inputs):
            possible_donors = [row for row in inputs if row[0]["split"] == source["split"] and row[1] != group]
            donor = possible_donors[index % len(possible_donors)][6]
            cases, changed_pictures = {}, {}
            for role, wave, truth, frame_map, mouth in prior.variants(pcm, len(table), group, tuple(range(len(table))), patches):
                tracks, observed_clock = picture, table
                observed_frames = frame_map
                if role.startswith("matched_picture_"):
                    tracks = temp / f"{group}-{offset}-{role}.mov"
                    base.picture(pixels[list(frame_map)], tracks)
                    observed_clock, _, mouth, _, retimed = base.video_inputs(tracks, detector)
                    observed_frames = tuple(range(len(observed_clock)))
                    changed_pictures[role] = [item["rgb_sha256"] for item in retimed]
                conditions = {"clean": wave}
                if role in NOISY:
                    conditions["competing"] = delivery._with_competing_voice(wave, pcm, donor)
                    conditions["train"] = delivery._with_competing_voice(wave, pcm, train[index % len(train)][:len(pcm)])
                require(json.loads(json.dumps(truth)) == _truth(role, group), "Authored repeat differs from its frozen interval")
                for scene, signal in conditions.items():
                    for carrier, samples in (("PCM", signal), ("AAC", delivery._decode_aac(signal, tracks, temp, observed_clock))):
                        measures = (method.measure(samples, observed_frames, mouth) if len(mouth) >= .75 * WINDOW
                                    else dict.fromkeys(ARMS))
                        for entries in measures.values():
                            for item in entries or []:
                                item["pcm_sha256"] = [prior._hash(samples[a:b]) for a, b in item["ranges"]]
                        cases[scene, role, carrier] = {"scene": scene, "role": role, "carrier": carrier, "truth": truth,
                                                       "pcm_sha256": prior._hash(samples), "arms": measures}
            rows.append({"source": source["id"], "sha256": source["sha256"], "offset": offset, "group": group,
                         "pictures": [item["rgb_sha256"] for item in frames], "retimed_pictures": changed_pictures,
                         "cases": [cases[key] for key in _case_keys()]})
            print(f"Measured {index + 1}/{len(inputs)} source excerpts", flush=True)
            del pixels
    doc = json.loads(json.dumps({"freeze": freeze, "rows": rows}, allow_nan=False))
    doc["results"] = recount(doc)
    write(args.output, doc)
    print(json.dumps(doc["results"]["summary"], sort_keys=True), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "run", "review"))
    parser.add_argument("output", type=Path)
    parser.add_argument("--freeze", type=Path)
    parser.add_argument("--comparison", type=Path)
    parser.add_argument("--held", type=Path)
    parser.add_argument("--sounds", type=Path)
    args = parser.parse_args()
    if args.phase == "review":
        doc = read(args.output)
        observed = recount(doc)
        require(doc["results"] == observed, "Saved results do not recount")
        print(json.dumps(observed, indent=2, sort_keys=True))
        return
    if args.output.exists():
        parser.error("Use a new output file")
    method = Path(__file__).with_name("video_repeated_speech_bands.py")
    freeze = plan()
    design = {"freeze": freeze, "runner_sha256": sha(Path(__file__)), "method_sha256": sha(method)}
    if args.phase == "freeze":
        write(args.output, design)
    else:
        if None in (args.freeze, args.comparison, args.held, args.sounds):
            parser.error("Specify the design lock, both video banks, and the noise recordings")
        require(read(args.freeze) == design, "Pre-run lock differs")
        run(args, freeze)


if __name__ == "__main__":
    main()
