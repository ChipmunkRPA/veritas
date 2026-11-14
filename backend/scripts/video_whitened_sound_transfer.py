# SPDX-License-Identifier: Apache-2.0
"""Replay background-downweighted soundtrack repeats on encoded local timing cases."""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts import review_video_adjacent_sound as replay
from scripts import video_repeated_speech_band_transfer as previous
from scripts.audio_vocal_event_mixtures import read, save, sha

SCHEMA = "chipmunk.video-whitened-sound-transfer/1"
ARMS = ("prior", "fixed", "downweighted")
N, HOP = previous.WINDOW, previous.HOP
CARRIERS = previous.CARRIERS
ROLES = previous.CONDITIONS | {"rain": previous.NOISY}
GATE = {"added_noisy_aac": 3, "sources": 2, "max_added_control_flags_per_carrier": 1,
        "max_added_matched_or_constant": 0}
require = replay.require


def plan():
    sources = previous.plan()["sources"]
    reserved = [[56, 66], [56, 66], [66, 76], [66, 76], [56, 66]]
    held = [source | {"offsets": offsets} for source, offsets in zip(sources[4:], reserved, strict=True)]
    return {"schema": SCHEMA, "sources": sources[:4] + held, "gate": GATE,
            "parent_sha256": sha(previous.PANELS / "video-repeated-speech-bands-2026-11-14.json"),
            "method_sha256": sha(Path(__file__).with_name("video_whitened_sound_witness.py")),
            "method": "Compare prior two-band repeat support, a lag contrast on the median of three fixed bands, and that same contrast after one clip-wide background downweighting filter. The exact same activity and source-mouth rules apply to each. Shifted-picture controls are actually retimed, encoded and decoded. Use 0.50 overlap on both sound ranges. Compare each new arm as a supplement to the prior arm."}


def keys():
    return [(scene, role, carrier) for scene, roles in ROLES.items() for role in roles for carrier in CARRIERS]


def review(receipt):
    freeze = plan()
    require(receipt["freeze"] == freeze, "Source plan or method changed")
    expected = [(i, row, offset) for i, row in enumerate(freeze["sources"], 501) for offset in row["offsets"]]
    counts, comparisons, gains = defaultdict(Counter), defaultdict(Counter), defaultdict(set)
    for (group, source, offset), row in zip(expected, receipt["rows"], strict=True):
        require((row["group"], row["file"], row["source_sha256"], row["offset"]) == (group, source["file"], source["sha256"], offset)
                and [(r["scene"], r["role"], r["carrier"]) for r in row["cases"]] == keys(), "Incomplete matched case roster")
        picture = row["pictures"]
        require(picture is None or (set(picture) == {"original", "matched_picture_early", "matched_picture_late"}
                and all(len(frames) == N and all(replay.SHA.fullmatch(item) for item in frames) for frames in picture.values())
                and all(picture[role] != picture["original"] for role in picture if role != "original")),
                "Synchronized controls need their own verified picture inputs")
        for case in row["cases"]:
            scene, role, carrier = (case[k] for k in ("scene", "role", "carrier"))
            truth = previous._truth(role, group)
            pcm = case["pcm_sha256"]
            require(case["truth"] == truth and (pcm is None if picture is None else replay.SHA.fullmatch(pcm))
                    and set(case["arms"]) == set(ARMS) and (picture is not None or all(x is None for x in case["arms"].values())),
                    "Known sound intervals or expected checks changed")
            cell = counts[source["split"], carrier, scene]
            kind = "edits" if truth else "controls"
            cell[kind] += 1
            found, alerts = {}, {}
            for arm, records in case["arms"].items():
                cell[f"{arm}_assessed_{kind}"] += records is not None
                for record in records or []:
                    replay._validate_pair(record["ranges"], N * HOP)
                    score = record["score"]
                    require((score is None if arm == "prior" else math.isfinite(score) and .53 <= score <= 1.000001)
                            and len(record["frame_pairs"]) >= 3
                            and all(len(pair) == 2 and all(type(i) is int and 0 <= i < N for i in pair) for pair in record["frame_pairs"])
                            and len(record["audio_sha256"]) == 2 and all(replay.SHA.fullmatch(x) for x in record["audio_sha256"]),
                            "Repeat needs distinct picture pairs and a measured waveform witness")
                found[arm] = any(replay.useful(record["ranges"], truth) for record in records or [])
                alerts[arm] = bool(records)
                cell[f"{arm}_useful"] += found[arm]
                cell[f"{arm}_wrong_edits"] += bool(truth) and alerts[arm] and not found[arm]
                cell[f"{arm}_control_flags"] += not truth and alerts[arm]
            if source["split"] != "held":
                continue
            for arm, reference in (("fixed", "prior"), ("downweighted", "prior"), ("downweighted_vs_fixed", "fixed")):
                candidate = arm.split("_vs_", 1)[0]
                stat = comparisons[arm]
                stat[f"{carrier}_{scene}_added_locations"] += found[candidate] and not found[reference]
                stat[f"{carrier}_{scene}_locations_lost"] += found[reference] and not found[candidate]
                if carrier == "AAC" and scene != "clean" and found[candidate] and not found[reference]:
                    stat["added_noisy_aac"] += 1
                    gains[arm].add(source["file"])
                if not truth and alerts[candidate] and not alerts[reference]:
                    stat[f"{carrier}_added_controls"] += 1
                    stat["added_matched_or_constant"] += role.startswith(("global_", "matched_picture_"))
    outcome = {}
    for arm, stat in comparisons.items():
        outcome[arm] = dict(stat) | {"sources": len(gains[arm]), "advances": arm in ARMS and
            stat["added_noisy_aac"] >= GATE["added_noisy_aac"] and len(gains[arm]) >= GATE["sources"]
            and max(stat[f"{c}_added_controls"] for c in CARRIERS) <= GATE["max_added_control_flags_per_carrier"]
            and stat["added_matched_or_constant"] <= GATE["max_added_matched_or_constant"]}
    return {"counts": {"/".join(k): dict(v) for k, v in counts.items()}, "comparisons": outcome}


def run(freeze, args):
    import cv2

    from scripts import video_adjacent_sound_delivery as delivery
    from scripts import video_whitened_sound_witness as method
    from scripts.audio_reference_variable_window import read_noise

    base, edits = delivery.video, delivery.transfer
    require(method.ARMS == ARMS and sha(delivery.parent.FACE_FILE) == delivery.parent.FACE_SHA, "Method or pinned face weights differ")
    cv2.setNumThreads(1)
    detector = base.load_yunet(delivery.parent.FACE_FILE)
    noise = read_noise(args.sounds)
    with TemporaryDirectory(prefix="video-whitened-sound-") as scratch:
        folder = Path(scratch)
        clips = []
        for group, source in enumerate(freeze["sources"], 501):
            path = (args.held if source["split"] == "held" else args.comparison) / source["file"]
            require(sha(path) == source["sha256"], "Source bytes differ")
            for offset in source["offsets"]:
                picture = folder / f"{group}-{offset}.mov"
                prepared = delivery.speech.prepare(path, picture, offset)
                if prepared is None:
                    print(f"Continuous inputs unavailable at {source['file']} + {offset} s", flush=True)
                    clips.append((source, group, offset, picture, None, None))
                else:
                    require(len(prepared[1]) == N * HOP, "Excerpts require a fixed audio extent")
                    clips.append((source, group, offset, picture, prepared[0], prepared[1]))
        rows = []
        for index, (source, group, offset, picture, pixels, pcm) in enumerate(clips):
            row = {"file": source["file"], "source_sha256": source["sha256"], "group": group, "offset": offset}
            if pcm is None:
                rows.append(row | {"pictures": None, "cases": [
                    {"scene": scene, "role": role, "carrier": carrier, "truth": previous._truth(role, group),
                     "pcm_sha256": None, "arms": dict.fromkeys(ARMS)} for scene, role, carrier in keys()]})
                continue
            same_split = [clip[-1] for clip in clips if clip[0]["split"] == source["split"] and clip[1] != group and clip[-1] is not None]
            beds = {scene: sounds[index % len(sounds)][:len(pcm)] for scene, sounds in noise.items()}
            beds["competing"] = same_split[index % len(same_split)]
            table, _, patches, _, frames = base.video_inputs(picture, detector)
            pictures = {"original": [row["rgb_sha256"] for row in frames]}
            cases = {}
            for role, wave, truth, mapping, mouth in edits.variants(pcm, N, group, tuple(range(N)), patches):
                movie, clock = picture, table
                if role.startswith("matched_picture_"):
                    movie = folder / f"{group}-{offset}-{role}.mov"
                    base.picture(pixels[list(mapping)], movie)
                    clock, _, mouth, _, changed = base.video_inputs(movie, detector)
                    pictures[role] = [row["rgb_sha256"] for row in changed]
                for scene, roles in ROLES.items():
                    if role not in roles:
                        continue
                    signal = wave if scene == "clean" else delivery._with_competing_voice(wave, pcm, beds[scene])
                    for carrier, samples in (("PCM", signal), ("AAC", delivery._decode_aac(signal, movie, folder, clock))):
                        results = method.measure(samples, tuple(range(N)), mouth) if len(mouth) >= .75 * N else dict.fromkeys(ARMS)
                        for records in results.values():
                            for item in records or []:
                                item["audio_sha256"] = [edits._hash(samples[a:b]) for a, b in item["ranges"]]
                        cases[scene, role, carrier] = {"scene": scene, "role": role, "carrier": carrier, "truth": truth,
                                                       "pcm_sha256": edits._hash(samples), "arms": results}
            rows.append(row | {"pictures": pictures, "cases": [cases[key] for key in keys()]})
            print(f"Measured {index + 1}/{len(clips)} excerpts", flush=True)
    receipt = json.loads(json.dumps({"freeze": freeze, "rows": rows}, allow_nan=False))
    receipt["results"] = review(receipt)
    save(args.output, receipt)
    print(json.dumps(receipt["results"]["comparisons"], indent=2, sort_keys=True), flush=True)


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
        receipt = read(args.output)
        results = review(receipt)
        require(results == receipt["results"], "Saved counts do not replay")
        print(json.dumps(results, indent=2, sort_keys=True))
        return
    freeze = plan()
    lock = {"plan": freeze, "runner_sha256": sha(Path(__file__))}
    if args.output.exists():
        parser.error("Specify a new output path")
    if args.phase == "freeze":
        save(args.output, lock)
        return
    if not all((args.freeze, args.comparison, args.held, args.sounds)):
        parser.error("Specify the frozen method, both source folders, and the noise folder")
    require(read(args.freeze) == lock, "The method changed after freeze")
    run(freeze, args)


if __name__ == "__main__":
    main()
