# SPDX-License-Identifier: Apache-2.0
"""Replay the existing short-window points against original-leaning mixed speech."""
from __future__ import annotations

import argparse
import math
from collections import defaultdict
from pathlib import Path

from scripts import audio_vocal_event_mixtures as pilot

SOURCE = pilot.PUBLIC / "audio-vocal-event-mixtures-2026-11-12.json"
WINDOWS = pilot.WINDOWS[1:]
SCHEMA = "chipmunk.audio-vocal-event-voice-conflict/1"
POINTS = {"wav": 6.822523597593649, "mp3": 7.085599896496718}
require = pilot.require


def cohort():
    result = pilot.read(SOURCE)
    cases = {case["id"]: case for case in result["cases"]}
    rows = [{"id": row["id"], "carrier": row["carrier"], "group": case["group"],
             "truth": case["speech_source"]["truth"], "source": case["carriers"][row["carrier"]]}
            for row in result["measurements"] if row["voice"]["decision"] == "original" for case in [cases[row["id"]]]]
    require(len(rows) == 58 and sum(row["truth"] is not None for row in rows) == 5, "Pilot cohort changed")
    return rows


def lock():
    return {
        "schema": SCHEMA, "pilot_sha256": pilot.sha(SOURCE), "runner_sha256": pilot.sha(Path(__file__)),
        "points": POINTS, "windows": [list(window) for window in WINDOWS], "cohort_sha256": pilot.commitment(cohort()),
        "selection": "Highest admitted two-second logit, earlier start on ties. Strict excess withholds the original tendency.",
        "criterion": "Locate false-original responses from at least two voice groups, with at most one ordinary group affected per carrier.",
        "purpose": "Retrospective diagnostic on the complete already scored original-leaning pilot cohort. No serving gate.",
    }


def selected(row):
    values = row["windows"]
    require(len(values) == len(WINDOWS) and all(
        value is None or type(value) in {int, float} and math.isfinite(value) for value in values),
        "Invalid short-window response")
    order = sorted((index for index, value in enumerate(values) if value is not None), key=lambda index: -values[index])
    return WINDOWS[order[0]] if order and values[order[0]] > POINTS[row["carrier"]] else None


def metrics(rows, manifest):
    require(len(rows) == len(manifest), "Missing diagnostic cases")
    results = defaultdict(lambda: {"ordinary_files": 0, "wrong_original_files": 0, "unassessed_files": 0,
                                   "ordinary_withheld": 0, "wrong_original_withheld": 0, "usefully_located": 0})
    actual_groups, ordinary_groups = defaultdict(set), defaultdict(set)
    for row, source in zip(rows, manifest, strict=True):
        require((row["id"], row["carrier"], row["pcm_sha256"]) == (source["id"], source["carrier"], source["source"]["pcm"]),
                "A response is not bound to the pilot case")
        carrier, truth = row["carrier"], source["truth"]
        count = results[carrier]
        count["ordinary_files"] += truth is None
        count["wrong_original_files"] += truth is not None
        count["unassessed_files"] += all(value is None for value in row["windows"])
        span = selected(row)
        if span is None:
            continue
        if truth is None:
            count["ordinary_withheld"] += 1
            ordinary_groups[carrier].add(source["group"])
            continue
        count["wrong_original_withheld"] += 1
        overlap = max(0, min(span[1], truth[1]) - max(span[0], truth[0]))
        union = span[1] - span[0] + truth[1] - truth[0] - overlap
        if 2 * overlap >= truth[1] - truth[0] and 10 * overlap >= 3 * union:
            count["usefully_located"] += 1
            actual_groups[carrier].add(source["group"])
    for carrier, totals in results.items():
        totals["ordinary_groups_withheld"] = sorted(ordinary_groups[carrier])
        totals["usefully_located_groups"] = sorted(actual_groups[carrier])
    return {"carriers": dict(results), "follow_up_criterion_met":
            len(set().union(*actual_groups.values())) >= 2 and all(len(groups) <= 1 for groups in ordinary_groups.values())}


def score(run, clips, weights):
    from app.detectors.audio import calibration
    from app.detectors.audio.content import window_in_speech
    from app.detectors.audio.decode import decode_audio
    from app.detectors.audio.upgraded import SpeechActivity
    from app.evidence.audio_short_review_binding import MP3_REVIEW_POINT, REVIEW_POINT
    from scripts.run_audio_specialists_v15 import load_voice

    frozen = pilot.read(run / "freeze.json")
    require(frozen == lock() and POINTS == {"wav": REVIEW_POINT, "mp3": MP3_REVIEW_POINT},
            "The diagnostic freeze or existing points changed")
    assets = {key: value for key, value in pilot.read(SOURCE)["freeze"]["assets"].items() if key not in pilot.SOUND_ASSETS}
    voice = load_voice({"weights": str(weights), "assets": assets})
    vad = SpeechActivity(weights / "silero_vad.jit")
    manifest, rows = cohort(), []
    for index, source in enumerate(manifest, 1):
        carrier = source["carrier"]
        path = clips / f"{source['id']}.{carrier}"
        require(pilot.sha(path) == source["source"]["bytes"], "Original pilot carrier changed")
        decoded = decode_audio(path, source_sha256=source["source"]["bytes"], max_seconds=5)
        pcm = decoded.pcm
        pcm_sha = pilot.digest((pcm * 32768).astype("<i2").tobytes())
        require(pcm_sha == source["source"]["pcm"] and decoded.mapping["timeline"] ==
                [{"start_sample": 0, "end_sample": pilot.SAMPLES, "source_pts": pilot.CARRIERS[carrier]}],
                "Original pilot PCM or source clock changed")
        spans = vad.spans(pcm)
        values = [calibration.logit(voice.predict([pcm[a:b]])[0])
                  if window_in_speech(spans, a / pilot.RATE, b / pilot.RATE) else None for a, b in WINDOWS]
        rows.append({"id": source["id"], "carrier": carrier, "pcm_sha256": pcm_sha, "windows": values})
        if index % 12 == 0:
            print(f"Measured {index}/58 original-leaning inputs.", flush=True)
    result = {"freeze": frozen, "rows": rows, "response_sha256": pilot.commitment(rows), "summary": metrics(rows, manifest)}
    pilot.save(run / "result.json", result)


def review(path):
    data = pilot.read(path)
    require(data["freeze"] == lock() and data["response_sha256"] == pilot.commitment(data["rows"])
            and data["summary"] == metrics(data["rows"], cohort()), "The diagnostic counts or commitments changed")
    print(data["summary"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "score", "review"))
    parser.add_argument("path", type=Path)
    parser.add_argument("--clips", type=Path, default=pilot.BASE / "audio-vocal-event-mixtures-2026-11-12")
    parser.add_argument("--weights", type=Path, default=pilot.WEIGHTS)
    args = parser.parse_args()
    if args.phase == "freeze":
        args.path.mkdir(mode=0o700)
        pilot.save(args.path / "freeze.json", lock())
    elif args.phase == "score":
        score(args.path, args.clips, args.weights)
    else:
        review(args.path)


if __name__ == "__main__":
    main()
