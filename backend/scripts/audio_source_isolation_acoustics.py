"""Measure blind foreground recovery without using detector decisions."""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from statistics import median

from scripts import audio_source_isolation_diagnostic as trial

SCHEMA = "chipmunk.audio-source-isolation-acoustics/1"


def readout(rows):
    result = {}
    for scope, subset in (("all", rows), ("edits", [row for row in rows if row["truth"] is not None])):
        result[scope] = counts = {}
        for interval in ("whole", "exact_edit"):
            cohort = [row for row in subset if interval in row["si_sdr_db"]]
            counts[interval] = {}
            for method in ("blind", "ideal_mask"):
                gains = []
                wins = 0
                for row in cohort:
                    view = row["si_sdr_db"][interval]
                    estimate = (max(view[method], default=None) if method == "blind" else view[method])
                    if estimate is not None:
                        gain = estimate - view["mixture"]
                        gains.append(gain)
                        wins += estimate >= 0. and gain >= 3.
                counts[interval][method] = {"planned": len(cohort), "measured": len(gains), "recovered": wins,
                                            "median_gain_db": round(median(gains), 5) if gains else None}
    return result


def measure(lock, args):
    import numpy as np
    import torch

    from scripts.audio_separation_v14 import SPEC, load

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    separator = load(args.separator)
    records = []
    _, rows = trial.freeze(args)
    for row in (row for row in rows if row["role"] == "assessment"):
        files = trial.source_data(trial.DATA / row["bank"], row)
        mixed, fg, bg = [np.frombuffer(file, "<i2").astype(np.float32) / 32768. for file in files]
        estimates, agreement = trial.separate(mixed, separator)
        ideal = trial.ideal_mask(mixed, fg, bg)
        scores = {}
        if fg.any():
            regions = {"whole": (0, trial.SAMPLES)}
            if row["truth"] is not None:
                regions["exact_edit"] = row["truth"]
            for name, (a, b) in regions.items():
                scores[name] = {"mixture": round(trial.acoustic_quality(fg[a:b], mixed[a:b]), 7),
                                "ideal_mask": round(trial.acoustic_quality(fg[a:b], ideal[a:b]), 7),
                                "blind": [round(value, 7) for _, estimate in estimates
                                          if (value := trial.acoustic_quality(fg[a:b], estimate[a:b])) is not None]}
        records.append({"input_sha256": row["pcm_sha256"], "kind": row["kind"], "family": row["family"],
                        "truth": row["truth"], "stream_correlations": [round(a, 8) for a in agreement], "si_sdr_db": scores})
    return {"schema": SCHEMA, "freeze_sha256": trial.path_sha(lock), "runner_sha256": trial.path_sha(Path(__file__)),
            "separator": SPEC, "scope": "Offline acoustic comparison. Blind recovery uses hindsight to identify the best anonymous stream.",
            "rows": records, "summary": readout(records)}


def review(path):
    receipt = trial.read(path)
    rows = receipt["rows"]
    trial.check(receipt["schema"] == SCHEMA and len({row["input_sha256"] for row in rows}) == len(rows) == 52,
                "Expected the fixed 52 clips")
    trial.check(Counter(row["family"] for row in rows if row["truth"] is not None) == {"clone": 4, "expressive": 4, "stock": 4},
                "Generated source families differ")
    for row in rows:
        truth = row["truth"]
        if truth is not None:
            trial.check(0 <= truth[0] < truth[1] <= trial.SAMPLES, "Invalid edit span")
        valid = set(row["si_sdr_db"])
        expected = ({"whole", "exact_edit"} if truth is not None else
                    set() if row["kind"] == "background_only" else {"whole"})
        trial.check(valid == expected, "Changed assessment scope")
        agreement = row["stream_correlations"]
        trial.check(len(agreement) == 2 and all(math.isfinite(v) and -1 <= v <= 1 for v in agreement), "Invalid stream match")
        for view in row["si_sdr_db"].values():
            trial.check(set(view) == {"mixture", "ideal_mask", "blind"} and len(view["blind"]) <= 2 and
                        all(math.isfinite(x) for x in (view["mixture"], view["ideal_mask"], *view["blind"])),
                        "Invalid acoustic response")
    trial.check(readout(rows) == receipt["summary"], "Acoustic readout differs")
    return receipt["summary"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("score", "review"))
    parser.add_argument("path", type=Path)
    parser.add_argument("--freeze", type=Path)
    parser.add_argument("--voice", type=Path, default=trial.ROOT / "models/weights/audio-upgrade")
    parser.add_argument("--separator", type=Path, default=trial.ROOT / "models/research/separation-v14-convtasnet")
    args = parser.parse_args()
    if args.phase == "score":
        if args.freeze is None:
            parser.error("--freeze is required to measure native waveforms")
        lock, _ = trial.freeze(args)
        trial.check(trial.read(args.freeze) == lock, "Input comparison freeze differs")
        trial.write(args.path, measure(args.freeze, args))
    print(json.dumps(review(args.path), sort_keys=True))


if __name__ == "__main__":
    main()
