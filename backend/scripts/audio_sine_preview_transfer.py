"""Compare previously fixed voice localization windows on the matched SINE preview."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
from collections import Counter
from pathlib import Path

from scripts import audio_had_positive_transfer as had
from scripts import audio_source_isolation_diagnostic as prior

INTAKE = prior.PANELS / "audio-sine-preview-2026-11-13.json"
SCHEMA = "chipmunk.audio-sine-preview-transfer/1"
ROLES = ("real", "infill", "cut_paste")
METHOD = {
    **had.METHOD,
    "decode": "Production mono signed 16-bit decode from 24 kHz to 16 kHz. Require a continuous zero-origin analysis clock. Map centisecond edit times exactly between the two sample rates.",
    "scope": "All ten publisher preview triplets. Retain every annotated edit, including ones too short for either comparison window. One audiobook reader, one generator, two edit mechanisms and no artificial interference or fitting.",
    "groups": "Compare each positive score against the real control in its same publisher triplet. Report raw flags and overlap separately for each role.",
}


def bindings():
    native = prior.read(INTAKE)
    parent = prior.read(had.PARENT)
    points = {"stock": parent["freeze"]["prior_point"], "dense": parent["points"]["mixture/dense"]["point"]}
    prior.check(points == had.POINTS and native["sample_rate"] == 24_000 and native["matched_triplets"] == 10
                and native["source_reader_ids"] == ["100"], "Unexpected source or points")
    rows = []
    for group, sources in native["cases"].items():
        for role in ROLES:
            source = sources[role]
            truth = None
            if role != "real":
                a, b, label = source["publisher_intervals"][1]
                prior.check(label == "F" and a % 3 == b % 3 == 0, "Unexpected publisher edit timing")
                truth = [a * 2 // 3, b * 2 // 3]
            rows.append((group, role, source, truth))
    hashes = {"intake_sha256": prior.path_sha(INTAKE), "parent_sha256": prior.path_sha(had.PARENT)}
    return rows, hashes


def best(items):
    return max((item for item in items if item["score"] is not None),
               key=lambda item: item["score"], default=None)


def summarize(rows):
    summary = {key: {role: Counter() for role in ROLES} for key in had.POINTS}
    controls = {row["group"]: row for row in rows if row["role"] == "real"}
    for row in rows:
        truth = row["truth"]
        for key, point in had.POINTS.items():
            counts = summary[key][row["role"]]
            items = row["routes"][key]
            winner = best(items)
            counts["planned"] += 1
            counts["assessed"] += winner is not None
            counts["flags"] += winner is not None and winner["score"] > point
            if truth is None:
                continue
            counts["geometrically_reachable"] += any(prior.useful(item["span"], truth) for item in items)
            counts["admitted_reachable"] += any(item["score"] is not None and prior.useful(item["span"], truth)
                                                for item in items)
            useful = winner is not None and prior.useful(winner["span"], truth)
            counts["best_useful_before_point"] += useful
            counts["useful_flags"] += useful and winner["score"] > point
            counts["misplaced_flags"] += winner is not None and winner["score"] > point and not useful
            control = best(controls[row["group"]]["routes"][key])
            paired = winner is not None and control is not None
            counts["paired_assessed"] += paired
            counts["above_matched_real"] += paired and winner["score"] > control["score"]
    return summary


def review(result):
    sources, hashes = bindings()
    prior.check(result["schema"] == SCHEMA and result["method"] == METHOD and result["points"] == had.POINTS
                and result["inputs"] == hashes and result["rates"] == [24_000, had.RATE], "Frozen inputs differ")
    for (group, role, source, truth), case in zip(sources, result["rows"], strict=True):
        prior.check(case["group"] == group and case["role"] == role and case["truth"] == truth
                    and case["wav_sha256"] == source["wav_sha256"], "Native case differs")
        prior.check(abs(3 * case["samples"] - 2 * source["samples"]) <= 3, "Decoded length differs")
        expected = had.slots(case["samples"])
        prior.check(case["routes"].keys() == expected.keys(), "Missing route")
        for key, items in case["routes"].items():
            prior.check([item["span"] for item in items] == expected[key], "Window grid differs")
            prior.check(all(item["score"] is None or math.isfinite(item["score"]) for item in items), "Nonfinite score")
    summary = summarize(result["rows"])
    prior.check(summary == result["summary"], "Outcome counts differ")
    return summary


def score(sources_dir, weights):
    import numpy as np

    from app.detectors.audio.calibration import logit
    from app.detectors.audio.content import window_in_speech
    from app.detectors.audio.decode import decode_audio
    from app.detectors.audio.upgraded import SpeechActivity
    from scripts.audio_strong_traffic_view import network_free
    from scripts.prepare_audio_specialists_v15 import ASSETS
    from scripts.run_audio_specialists_v15 import load_voice

    sources, hashes = bindings()
    network_free()
    model = load_voice({"weights": str(weights), "assets": ASSETS})
    activity = SpeechActivity(weights / "silero_vad.jit")
    capture = prior.FrameCapture(model)
    forwards = Counter()
    rows = []
    decoders = set()
    rate = had.RATE
    width = 4 * rate
    for group, role, source, truth in sources:
        decoded = decode_audio(sources_dir / source["filename"], source_sha256=source["wav_sha256"],
                               max_seconds=10, inspect_stereo=False)
        pcm, mapping = decoded.pcm, decoded.mapping
        clock = [{"start_sample": 0, "end_sample": len(pcm), "source_pts": 0}]
        prior.check(mapping["source_sample_rate"] == 24_000 and mapping["timeline"] == clock
                    and mapping["timeline_time_base"] == f"1/{rate}", "Unexpected decode clock")
        decoders.add(mapping["decoder"])
        speech = activity.spans(pcm)
        routes = {}
        contexts = {}
        for key, spans in had.slots(len(pcm)).items():
            items = []
            for a, b in spans:
                response = None
                if window_in_speech(speech, a / rate, b / rate):
                    if key == "stock":
                        response = logit(model.predict([pcm[a:b]])[0])
                        forwards[key] += 1
                    else:
                        values = []
                        for offset in (0, len(pcm) - width):
                            if offset <= a < b <= offset + width:
                                if offset not in contexts:
                                    frames = capture.forward(pcm[offset:offset + width])
                                    contexts[offset] = offset + 280 + np.arange(len(frames)) * 320, frames
                                    forwards[key] += 1
                                centers, frames = contexts[offset]
                                selected = frames[(a <= centers) & (centers < b)]
                                if len(selected) >= 40:
                                    values.append(float(np.median(selected)))
                        response = min(values) if values else None
                items.append({"span": [a, b], "score": response})
            routes[key] = items
        rows.append({"group": group, "role": role, "wav_sha256": source["wav_sha256"], "truth": truth,
                     "samples": len(pcm), "analysed_pcm_sha256": mapping["pcm_sha256"], "routes": routes})
        print(len(rows), "/", len(sources), role, "scored", flush=True)
    return {"schema": SCHEMA, "method": METHOD, "inputs": hashes, "rates": [24_000, rate], "points": had.POINTS,
            "assets": ASSETS, "decoders": sorted(decoders), "forwards": dict(forwards), "rows": rows,
            "runtime": {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "torch", "transformers")},
            "summary": summarize(rows)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("score", "review"))
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--sources", type=Path)
    parser.add_argument("--weights", type=Path, default=prior.ROOT / "models/weights/audio-upgrade")
    args = parser.parse_args()
    if args.phase == "score":
        if args.sources is None:
            parser.error("--sources is required for scoring")
        result = score(args.sources, args.weights)
        review(result)
        prior.write(args.receipt, result)
    else:
        result = prior.read(args.receipt)
    print(json.dumps(review(result), sort_keys=True))


if __name__ == "__main__":
    main()
