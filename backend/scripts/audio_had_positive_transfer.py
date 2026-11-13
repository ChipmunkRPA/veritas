"""Measure a frozen local voice comparator on the native HAD intake."""

from __future__ import annotations

import argparse
import importlib.metadata
import io
import json
import math
import wave
from collections import Counter
from pathlib import Path

from scripts import audio_source_isolation_diagnostic as prior

SCHEMA = "chipmunk.audio-had-positive-transfer/1"
INTAKE = prior.PANELS / "audio-had-native-intake-2026-11-13.json"
PARENT = prior.PANELS / "audio-source-isolation-diagnostic-2026-11-13.json"
POINTS = {"stock": 6.400866488004889, "dense": 7.024354934692383}
INTAKE_SHA = "db7d08d89fdb5df66be0d884ec0f76a38892b05dfcf417dcf2045fabc2a6bda3"
PARENT_SHA = "b431596eb48f1c7339427873e45a4d4f583dc455ff50be3624873d2053f0daf9"
RATE = prior.RATE
METHOD = {
    "stock": "Native two-second windows at source seconds 1 and 3. Omit a window when it cannot fit completely.",
    "dense": "Native one-second windows every half second. Keep fully contained windows from four-second contexts at the start and end. Apply the previous dense affine frame reduction, keeping the lower of qualifying context medians.",
    "admission": "Existing half-window speech gate. No score or annotation determines the windows.",
    "decisions": "Strict exceedance of each previously fixed point. Report the strongest measured window, including below the point. Useful requires 0.30 IoU and half of the publisher-marked edit covered.",
    "scope": "Publisher-positive-only Mandarin test prefix. No new fitting, artificial backgrounds, false-alarm estimate or serving change.",
}


def slots(samples):
    return {"stock": [[a, a + 2 * RATE] for a in (RATE, 3 * RATE) if a + 2 * RATE <= samples],
            "dense": [[a, a + RATE] for a in range(0, samples - RATE + 1, RATE // 2)]}


def bindings():
    prior.check(prior.path_sha(INTAKE) == INTAKE_SHA and prior.path_sha(PARENT) == PARENT_SHA,
                "Frozen source receipt changed")
    intake = prior.read(INTAKE)
    parent = prior.read(PARENT)
    expected_points = {"stock": parent["freeze"]["prior_point"], "dense": parent["points"]["mixture/dense"]["point"]}
    prior.check(expected_points == POINTS and intake["eligible_positives"] == 8, "Frozen source or points differ")
    rows = [row for row in intake["rows"] if row["eligible"]]
    return rows, {"intake_sha256": INTAKE_SHA, "parent_sha256": PARENT_SHA}


def summarize(rows):
    counts = {key: Counter() for key in POINTS}
    for row in rows:
        for key, values in row["routes"].items():
            current = counts[key]
            current["planned"] += 1
            current["geometrically_reachable"] += any(prior.useful(item["span"], row["truth"]) for item in values)
            qualified = [item for item in values if item["score"] is not None]
            current["assessed"] += bool(qualified)
            current["admitted_reachable"] += any(prior.useful(item["span"], row["truth"]) for item in qualified)
            if not qualified:
                continue
            best = max(qualified, key=lambda item: item["score"])
            useful = prior.useful(best["span"], row["truth"])
            flag = best["score"] > POINTS[key]
            current["useful_before_point"] += useful
            current["flags"] += flag
            current["useful"] += flag and useful
            current["misplaced"] += flag and not useful
    return counts


def review(ledger):
    sources, commitments = bindings()
    prior.check(ledger["schema"] == SCHEMA and ledger["method"] == METHOD and ledger["points"] == POINTS
                and ledger["inputs"] == commitments, "Method or source bindings differ")
    cases = ledger["rows"]
    prior.check(len(cases) == len(sources), "Missing native case")
    for source, case in zip(sources, cases, strict=True):
        expected_truth = next([a, b] for a, b, kind in source["labels"] if kind == "F")
        prior.check(case["id"] == source["id"] and case["pcm_sha256"] == source["pcm_sha256"]
                    and case["truth"] == expected_truth, "Native case truth or source differs")
        expected_slots = slots(source["samples"])
        prior.check(case["routes"].keys() == expected_slots.keys(), "Wrong native route")
        for key, values in case["routes"].items():
            prior.check([item["span"] for item in values] == expected_slots[key], "Wrong native window layout")
            for item in values:
                prior.check(item["score"] is None or type(item["score"]) in {int, float} and math.isfinite(item["score"]),
                            "Nonfinite source score")
    actual = summarize(cases)
    prior.check(actual == ledger["summary"], "Outcome counts differ")
    return actual


def score(source_dir, weights):
    import numpy as np
    import torch

    from app.detectors.audio.calibration import logit
    from app.detectors.audio.content import window_in_speech
    from app.detectors.audio.upgraded import SpeechActivity
    from scripts.audio_strong_traffic_view import network_free
    from scripts.prepare_audio_specialists_v15 import ASSETS
    from scripts.run_audio_specialists_v15 import load_voice

    torch.set_num_threads(2)
    network_free()
    model = load_voice({"weights": str(weights), "assets": ASSETS})
    activity = SpeechActivity(weights / "silero_vad.jit")
    capture = prior.FrameCapture(model)
    sources, commitments = bindings()
    rows = []
    forwards = Counter()
    for row in sources:
        original = (source_dir / f'{row["id"]}.wav').read_bytes()
        prior.check(prior.digest(original) == row["wav_sha256"], "Native WAV changed")
        with wave.open(io.BytesIO(original), "rb") as stream:
            raw = stream.readframes(stream.getnframes())
        prior.check(len(raw) == 2 * row["samples"] and prior.digest(raw) == row["pcm_sha256"], "Native PCM changed")
        pcm = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768
        speech = activity.spans(pcm)
        routes = {}
        width = 4 * RATE
        contexts = {}
        for key, values in slots(len(pcm)).items():
            routed = []
            for a, b in values:
                response = None
                if window_in_speech(speech, a / RATE, b / RATE):
                    if key == "stock":
                        response = logit(model.predict([pcm[a:b]])[0])
                        forwards[key] += 1
                    else:
                        scores = []
                        for offset in sorted({0, len(pcm) - width}):
                            if not offset <= a < b <= offset + width:
                                continue
                            if offset not in contexts:
                                frames = capture.forward(pcm[offset:offset + width])
                                contexts[offset] = offset + 280 + np.arange(len(frames)) * 320, frames
                                forwards[key] += 1
                            centers, frames = contexts[offset]
                            selected = frames[(a <= centers) & (centers < b)]
                            if len(selected) >= 40:
                                scores.append(float(np.median(selected)))
                        response = min(scores) if scores else None
                routed.append({"span": [a, b], "score": response})
            routes[key] = routed
        rows.append({"id": row["id"], "pcm_sha256": row["pcm_sha256"],
                     "truth": next([a, b] for a, b, kind in row["labels"] if kind == "F"), "routes": routes})
        print(row["id"], "scored", flush=True)
    return {"schema": SCHEMA, "method": METHOD, "inputs": commitments, "points": POINTS, "assets": ASSETS,
            "runtime": {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "torch", "transformers")},
            "forwards": dict(forwards), "rows": rows, "summary": summarize(rows)}


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
        ledger = score(args.sources, args.weights)
        review(ledger)
        prior.write(args.receipt, ledger)
    else:
        ledger = prior.read(args.receipt)
    print(json.dumps(review(ledger), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
