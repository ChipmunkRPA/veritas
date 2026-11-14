"""Measure short-window admission after a longer-window coverage skip."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
from collections import Counter
from functools import partial
from pathlib import Path

from scripts import audio_source_isolation_diagnostic as prior
from scripts import review_audio_stock_voice_transfer as stock

BANK = "audio-remote-neighbor-replication-2026-11-12"
PARENT_SHA256 = "3d3c27c859e60c86fcb583e017b3e4fda4397ef81110c42b8c1773ec7edd1599"
RATE, SAMPLES = stock.RATE, 6 * stock.RATE
CLIP, DESTINATION = 26400, 35200
METHOD = {
    "input": "From each retained six-second matched source group, take the same 1.65 s region in all four foreground and original-mix versions. Center on the known edit subject to source bounds, then put the excerpt at 2.200 s in a six-second file with silence outside it. Room-only controls use the centered excerpt at the same destination. This is a bounded sparse-speech test, with labels used to prepare fixtures before inference only.",
    "comparison": "Hold the model, raw flag point, source clock and short-window admission rule fixed. The current path requires one admitted four-second window. The proposed path permits the independent two-second check when the primary explicitly reports a speech-admission skip. Record every skipped window and distinguish checked, elevated, and useful exact-time coverage. The overall result is outside this comparison.",
    "sources": "Replay the previously bound three-generator panel. Each of its twelve groups contains a synthetic edit, an untouched file and same-speaker and other-speaker human edits. Eight additional deliveries comprise foreground and mixed routes for the four isolated room inputs. Controlled excerpt placement and earlier project source exposure apply.",
}


def parent():
    path = prior.PANELS / f"{BANK}.json"
    prior.check(prior.path_sha(path) == PARENT_SHA256, "Published parent selection changed")
    return prior.read(path)


def inputs(folder):
    import numpy as np

    pins = parent()["commitments"]
    prior.check(all(prior.path_sha(folder / f"{key}.json") == pins[key] for key in ("roster", "freeze")),
                "Native parent selection changed")
    rows = prior.read(folder / "roster.json")
    positives = {row["group"]: row["truth"] for row in rows if row["truth"]}
    for index, row in enumerate(rows):
        room = row["arm"] == "background_only"
        gold = positives.get(row["group"])
        midpoint = sum(gold) // 2 if gold else SAMPLES // 2
        begin = min(SAMPLES - CLIP, max(0, midpoint - CLIP // 2))
        native = tuple(prior.source_data(folder, row))
        prior.check(room == (gold is None), "A matched comparison lost its source group")
        for view, raw in (("foreground", native[1]), ("mixture", native[0])):
            pcm = np.zeros(SAMPLES, dtype="<i2")
            pcm[DESTINATION:DESTINATION + CLIP] = np.frombuffer(raw, dtype="<i2")[begin:begin + CLIP]
            truth = [sample - begin + DESTINATION for sample in gold] if row["truth"] else None
            yield {"id": index, "group": row["group"], "kind": row["arm"],
                   "family": row["family"] if row["truth"] else None, "view": view, "source_start_sample": begin,
                   "truth": truth, "pcm_sha256": prior.digest(pcm.tobytes())}, pcm


def observe(rows, weights, fixtures):
    import numpy as np

    from app.detectors.audio.content import window_in_speech
    from app.detectors.audio.short_voice_review import scan
    from app.detectors.audio.upgraded import SpeechActivity
    from app.detectors.video.audio import voice_windows
    from app.engine.boundary import Deadline
    from app.engine.detector import Profile
    from app.engine.profiles import SAMPLING
    from scripts.audio_strong_traffic_view import network_free
    from scripts.prepare_audio_specialists_v15 import ASSETS
    from scripts.run_audio_specialists_v15 import load_voice

    network_free()
    model = load_voice({"weights": weights, "assets": ASSETS})
    activity = SpeechActivity(weights / "silero_vad.jit")
    profile = SAMPLING[Profile.FULL]
    if fixtures is not None:
        from app.detectors.audio.decode import wav_bytes

        fixtures.mkdir(exist_ok=False)
    receipts = []
    for row, pcm in rows:
        values = pcm.astype(np.float32) / 32768
        spans = activity.spans(values)
        windows = voice_windows(values, profile.voice_hop_s, keep=partial(window_in_speech, spans))
        measured = scan(values, spans, model, Deadline(None))
        response = {item["start_sample"]: item["logit"] for item in measured["responses"]}
        receipts.append(row | {"primary_admitted": len(windows), "speech_samples": measured["speech_counts"],
                              "scores": [response.get(start) for start, _ in stock.WINDOWS]})
        if fixtures is not None:
            (fixtures / f"{row['id']:02}-{row['view']}.wav").write_bytes(wav_bytes(values))
        if len(receipts) % 16 == 0:
            print(f"Measured {len(receipts)} sparse-speech deliveries", flush=True)
    return receipts


def review(receipt):
    rows = receipt["observations"]
    prior.check(receipt["method"] == METHOD and receipt["stock_point"] == stock.SHORT_POINT
                and receipt["input_parent"] == PARENT_SHA256 and len(rows) == 104, "Evaluation inputs changed")
    totals = {view: {arm: Counter() for arm in ("current", "rescue")} for view in ("foreground", "mixture")}
    prior.check([(row["id"], row["view"]) for row in rows] == [(i, view) for i in range(52) for view in totals],
                "Incomplete matched bank")
    for row in rows:
        truth = row["truth"]
        prior.check(row["view"] in totals and row["primary_admitted"] in (0, 1, 2) and stock.sha256(row["pcm_sha256"])
                    and (truth is None or DESTINATION <= truth[0] < truth[1] <= DESTINATION + CLIP), "Invalid assessed range")
        prior.check(len(row["speech_samples"]) == len(stock.WINDOWS) == len(row["scores"])
                    and all(type(samples) is int and 0 <= samples <= 2 * RATE and
                            ((samples < RATE and score is None) or (samples >= RATE and type(score) in (float, int) and math.isfinite(score)))
                            for samples, score in zip(row["speech_samples"], row["scores"], strict=True)), "Speech admission changed")
        best = stock.strongest(row, range(len(stock.WINDOWS)))
        strata = "edited" if truth else "background" if row["kind"] == "background_only" else "ordinary"
        for arm, counts in totals[row["view"]].items():
            counts[f"planned_{strata}"] += 1
            if (arm == "current" and not row["primary_admitted"]) or best is None:
                counts[f"unassessed_{strata}"] += 1
            else:
                counts[f"checked_{strata}"] += 1
                score, negative_index = best
                if score > stock.SHORT_POINT:
                    counts[f"{strata}_flagged"] += 1
                    if truth and stock.useful(stock.WINDOWS[-negative_index], truth):
                        counts["usefully_located"] += 1
    return totals


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("score", "review", "audit"))
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--bank", type=Path, default=prior.DATA / BANK)
    parser.add_argument("--weights", type=Path, default=prior.ROOT / "models/weights/audio-upgrade")
    parser.add_argument("--fixtures", type=Path)
    args = parser.parse_args()
    if args.phase == "score":
        payload = {"method": METHOD, "stock_point": stock.SHORT_POINT, "input_parent": PARENT_SHA256,
                   "runtime": {name: importlib.metadata.version(name) for name in ("numpy", "torch", "transformers")},
                   "observations": observe(inputs(args.bank), args.weights, args.fixtures)}
        payload["results"] = review(payload)
        prior.write(args.receipt, payload)
    else:
        payload = prior.read(args.receipt)
    if args.phase == "audit":
        for (row, _), observed in zip(inputs(args.bank), payload["observations"], strict=True):
            prior.check(all(value == observed[key] for key, value in row.items()), "Native receipt differs")
    results = review(payload)
    prior.check(payload["results"] == results, "Reported results differ from the recorded responses")
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
