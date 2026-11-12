# SPDX-License-Identifier: Apache-2.0
"""Test an independent speech check on the original-leaning vocal-event pilot."""
from __future__ import annotations

import argparse
import math
from collections import Counter, defaultdict
from pathlib import Path

from scripts import audio_vocal_event_voice_conflict as prior

pilot, require = prior.pilot, prior.require
CALIBRATION = pilot.PUBLIC / "audio-disagreement-transfer-2026-11-12.json"
PREPROCESSOR = Path(__file__).with_name("audio_spectra_background_transfer.py")
POINT = 0.49009108543395996
WINDOW = (8000, 56000)


def lock():
    return {
        "schema": "chipmunk.audio-vocal-event-second-opinion/1",
        "pilot_sha256": pilot.sha(prior.SOURCE),
        "cohort_sha256": pilot.commitment(prior.cohort()),
        "calibration_sha256": pilot.sha(CALIBRATION),
        "calibration_points_sha256": pilot.read(CALIBRATION)["commitments"]["points"],
        "preprocessor_sha256": pilot.sha(PREPROCESSOR),
        "runner_sha256": pilot.sha(Path(__file__)),
        "point": POINT,
        "window": list(WINDOW),
        "decision": "Strict excess would withhold an original speech tendency.",
        "purpose": "Consumed pilot diagnostic. The central window does not localize the edit.",
    }


def summarize(rows, manifest):
    totals, groups = defaultdict(Counter), defaultdict(set)
    for row, source in zip(rows, manifest, strict=True):
        carrier = source["carrier"]
        require((row["id"], row["carrier"], row["pcm_sha256"])
                == (source["id"], carrier, source["source"]["pcm"]), "Score source differs")
        require(type(row["score"]) in {float, int} and math.isfinite(row["score"]), "Invalid score")
        role = "edited" if source["truth"] else "ordinary"
        totals[carrier][f"{role}_files"] += 1
        totals[carrier][f"{role}_withheld"] += int(row["score"] > POINT)
        if row["score"] > POINT:
            groups[carrier, role].add(source["group"])
    return {carrier: {**totals[carrier],
                      "edited_groups_withheld": len(groups[carrier, "edited"]),
                      "ordinary_groups_withheld": len(groups[carrier, "ordinary"])}
            for carrier in pilot.CARRIERS}


def score(run, clips, assets, calibration):
    from app.detectors.audio.decode import decode_audio
    from scripts import audio_spectra_background_transfer as independent

    frozen = pilot.read(run / "freeze.json")
    require(frozen == lock() and pilot.sha(calibration) == frozen["calibration_points_sha256"]
            and pilot.read(calibration)["points"]["independent"] == POINT, "Freeze or original calibration differs")
    session = independent.start_session(assets)
    manifest, rows = prior.cohort(), []
    for source in manifest:
        carrier = source["carrier"]
        path = clips / f"{source['id']}.{carrier}"
        pin = source["source"]
        require(pilot.sha(path) == pin["bytes"], "Original pilot carrier changed")
        decoded = decode_audio(path, source_sha256=pin["bytes"], max_seconds=5)
        pcm = (decoded.pcm * 32768).astype("<i2")
        require(pilot.digest(pcm.tobytes()) == pin["pcm"] and decoded.mapping["timeline"] ==
                [{"start_sample": 0, "end_sample": pilot.SAMPLES, "source_pts": pilot.CARRIERS[carrier]}],
                "Original pilot PCM or source clock changed")
        value, graph = independent.candidate_score(session, pcm[slice(*WINDOW)].tobytes())
        rows.append({"id": source["id"], "carrier": carrier, "pcm_sha256": pin["pcm"],
                     "score": value, "input_sha256": graph})
    result = {"freeze": frozen, "rows": rows, "response_sha256": pilot.commitment(rows),
              "summary": summarize(rows, manifest)}
    with (run / "result.json").open("xb") as output:
        output.write(pilot.encoded(result))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "score", "review"))
    parser.add_argument("path", type=Path)
    parser.add_argument("--clips", type=Path, default=pilot.BASE / "audio-vocal-event-mixtures-2026-11-12")
    parser.add_argument("--assets", type=Path)
    parser.add_argument("--calibration", type=Path,
                        default=pilot.BASE / "audio-disagreement-transfer-2026-11-12/points.json")
    args = parser.parse_args()
    if args.phase == "freeze":
        args.path.mkdir(mode=0o700)
        pilot.save(args.path / "freeze.json", lock())
    elif args.phase == "score":
        if args.assets is None:
            parser.error("score requires --assets")
        score(args.path, args.clips, args.assets, args.calibration)
    else:
        result = pilot.read(args.path)
        require(result["freeze"] == lock() and result["response_sha256"] == pilot.commitment(result["rows"])
                and result["summary"] == summarize(result["rows"], prior.cohort()), "Response commitment differs")
        print(result["summary"])


if __name__ == "__main__":
    main()
