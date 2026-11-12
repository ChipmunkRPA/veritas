"""Replay periodic-excitation sample clocks and false-alarm arithmetic independently."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


def read(path):
    return json.loads(path.read_bytes())


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def useful(span, truth):
    if span is None or truth is None:
        return False
    start, end = span
    low, high = truth
    common = len(range(max(start, low), min(end, high)))
    union = end-start + high-low-common
    return common / union >= .30 and common / (high-low) >= .50


def verify(run, parent, meeting):
    receipt = read(run / "result.json")
    for name, expected in receipt["commitments"].items():
        require(sha(run / f"{name}.json") == expected, f"Changed {name} ledger")
    for folder, key in ((parent, "parent"), (meeting, "meeting")):
        for name, expected in receipt[key].items():
            require(sha(folder / name) == expected, f"Changed source parent {name}")
    fit = read(run / "fit.json")
    require(fit["freeze_sha256"] == sha(run / "freeze.json"), "Fitted point is not bound to the freeze")
    observed = read(run / "observations.json")
    matched = {r["id"]: r for r in read(parent / "inputs.json")["rows"]}
    ami = {r["id"]: r for r in read(meeting / "roster.json")}
    values = [row for row in fit["selection"] if row["score"] is not None]
    require(set(row["id"] for row in fit["selection"]) ==
            {key for key, row in matched.items() if row["role"] == "confidence_cal" and row["truth"] is None},
            "Genuine point includes missing or different sources")
    require(max(row["score"] for row in values) == fit["point"], "Genuine maximum point differs")
    require(next(row for row in values if row["id"] == fit["point_witness"])["score"] == fit["point"],
            "Genuine point witness differs")
    require(len(observed) == len({(r["panel"], r["id"]) for r in observed}) == 216, "Observation roster size differs")
    candidate = "periodic_excitation"
    by_cell = {}
    for row in observed:
        if row["panel"] == "matched":
            source = matched[row["id"]]
            pcm = parent / source["path"]
            commitment, extent, role = source["pcm_sha256"], 64_000, source["role"]
        else:
            require(row["panel"] == "meeting", "Unknown recording panel")
            source = ami[row["id"]]
            pcm = meeting / "pcm" / f"{row['id']:03}.pcm"
            commitment, extent, role = source["source_pcm_sha256"], 96_000, source["panel"]
        require(role == row["role"] and source["truth"] == row["truth"] and source["condition"] == row["condition"]
                and commitment == row["pcm_sha256"] == sha(pcm) and pcm.stat().st_size == 2*extent,
                "Observation is disconnected from its exact source role, sample clock or edit")
        bucket = by_cell.setdefault("/".join(row[key] for key in ("panel", "role", "condition")), [])
        bucket.append(row)
        for arm, item in row["checks"].items():
            span = item["range"]
            if span is not None:
                start, end = span
                width = 32_000 if arm == "incumbent_two_second" else 9_600
                require(type(start) is int and type(end) is int and 0 <= start < end <= extent and end-start == width
                        and start % 3_200 == 0, "Selected range differs from the fixed sample clock")
            require(item["located"] is useful(span, source["truth"]), "Localized decision does not match truth overlap")
            if arm != candidate:
                continue
            score = item["score"]
            require((score is not None) == (span is not None) and (score is None or math.isfinite(score))
                    and item["flag"] is (score is not None and score > fit["point"]), "Candidate threshold or missing range differs")
            support = item["range_support"]
            require(type(item["covered_cells"]) is int and 0 <= item["covered_cells"] <= extent//3_200, "Covered cell count differs")
            require(not support if span is None else (2 <= len(support) <= 3 and len(support) == len({tuple(x) for x in support})
                    and all(start <= lo < hi <= end and lo % 3_200 == 0 and hi-lo == 3_200 for lo, hi in support)),
                    "Selected range contains invalid supported subcells")

    require(set(by_cell) == receipt["cells"].keys(), "Outcome cells differ from the observations")
    for key, rows in by_cell.items():
        cell = receipt["cells"][key]
        for arm in rows[0]["checks"]:
            predicted = [row for row in rows if row["checks"][arm]["flag"]]
            false = [row for row in predicted if row["truth"] is None]
            require(cell[arm]["useful"] == sum(row["checks"][arm]["located"] for row in predicted)
                    and cell[arm]["genuine_false_flags"] == len(false)
                    and cell[arm]["genuine_false_groups"] == len({row["group"] for row in false}), "Fixed-point outcomes differ")
        for budget, comparison in cell.get("equal_false_groups", {}).items():
            for arm, outcome in comparison.items():
                negative = {}
                for row in rows:
                    score = row["checks"][arm].get("score")
                    if score is not None and row["truth"] is None:
                        negative[row["group"]] = max(negative.get(row["group"], -math.inf), score)
                threshold = sorted(negative.values())[-int(budget)-1]
                chosen = [row for row in rows if row["checks"][arm].get("score") is not None and row["checks"][arm]["score"] > threshold]
                require(outcome["genuine_false_groups"] == len({r["group"] for r in chosen if r["truth"] is None})
                        and outcome["useful"] == sum(r["checks"][arm]["located"] for r in chosen), "Equal-alarm outcomes differ")
    print(f"Independently verified {len(observed)} exact PCM commitments, source ranges, and {len(by_cell)} outcome cells.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("parent", type=Path)
    parser.add_argument("meeting", type=Path)
    args = parser.parse_args()
    verify(args.run, args.parent, args.meeting)
