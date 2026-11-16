# SPDX-License-Identifier: MIT
"""Run the current short voice review on the frozen duration-transfer holdout."""
from __future__ import annotations

from argparse import ArgumentParser
from collections import Counter
from pathlib import Path

from app.evidence.audio_short_review_binding import GRID
from scripts.audio_short_edit_duration_transfer import BASE, EARLIER_SINE, ROOM, SOURCE, compatible, read_gz


def capture(plan, weights, output):
    from app.detectors.audio import short_voice_review
    from scripts.prepare_audio_specialists_v15 import ASSETS
    from scripts.run_audio_specialists_v15 import load_voice

    class Deadline:
        def check(self, stage):
            pass

    frozen = SOURCE.read(plan / "freeze.json")
    fresh = next(iter(frozen["rosters"].values()))["fresh"]
    activity = ROOM.runtime(weights / "silero_vad.jit")
    model = load_voice({"weights": weights, "assets": ASSETS})
    rows = []
    for case in fresh:
        values = BASE.pcm(plan / "fresh", case)
        result = short_voice_review.scan(values, activity.spans(values), model, Deadline(), codec="pcm_s16le")
        selected = result["selected"]
        rows.append(case | {
            "selected": [selected["start_sample"], selected["end_sample"]] if selected else None,
            "assessed": bool(result["responses"]),
        })
        if len(rows) % 16 == 0:
            print(f"Measured {len(rows)} of {len(fresh)} files", flush=True)
    SOURCE.write(output, {"assets": ASSETS, "cases": rows})


def review(frozen, measured):
    first, *others = [fold["fresh"] for fold in frozen["rosters"].values()]
    assets = read_gz(EARLIER_SINE)["current"]["assets"]
    SOURCE.check(all(first == other for other in others) and measured["assets"] == assets, "Serving basis differs")
    totals = {}
    for baseline, row in zip(first, measured["cases"], strict=True):
        SOURCE.check(baseline == {key: row[key] for key in baseline}, "Serving reading uses another source")
        selected, assessed = row["selected"], row["assessed"]
        SOURCE.check(type(assessed) is bool, "Missing serving coverage")
        if selected is not None:
            a, b = selected
            SOURCE.check(assessed and (a, b) in GRID, "Serving short range differs from its grid")
        tally = totals.setdefault(row["condition"] + "/" + row["kind"],
                                  Counter(deliveries=0, unassessed=0, flags=0, useful=0, truth_fits_grid=0))
        tally.update(deliveries=1, unassessed=int(not assessed), flags=int(selected is not None),
                     useful=int(row["truth"] is not None and compatible(selected, row["truth"])),
                     truth_fits_grid=int(row["truth"] is not None and any(compatible(window, row["truth"]) for window in GRID)))
    return totals


def main():
    import json

    cli = ArgumentParser(description=__doc__)
    sub = cli.add_subparsers(dest="phase", required=True)
    native = sub.add_parser("capture")
    native.add_argument("plan", type=Path)
    native.add_argument("weights", type=Path)
    native.add_argument("output", type=Path)
    portable = sub.add_parser("review")
    portable.add_argument("duration_receipt", type=Path)
    portable.add_argument("serving_receipt", type=Path)
    args = cli.parse_args()
    if args.phase == "capture":
        capture(args.plan, args.weights, args.output)
    else:
        print(json.dumps(review(read_gz(args.duration_receipt)["freeze"], SOURCE.read(args.serving_receipt)),
                         sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
