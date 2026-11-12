"""Independently verify local spectrogram decisions and the AMI overlap diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def load(path):
    return json.loads(path.read_bytes())


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def within(predicted, truth):
    if truth is None:
        return False
    common = len(range(max(predicted[0], truth[0]), min(predicted[1], truth[1])))
    width, support = predicted[1]-predicted[0], truth[1]-truth[0]
    return common / (width+support-common) >= .30 and common / support >= .50


def choose(rows, arm, budget):
    controls = {r["group"] for r in rows if r["truth"] is None}
    by_group = sorted((max(r["checks"][arm]["score"] for r in rows if r["truth"] is None and r["group"] == g)
                       for g in controls), reverse=True)
    return [r for r in rows if r["checks"][arm]["score"] > by_group[budget]]


def verify(run, parent, meeting):
    result = load(run / "result.json")
    for name, expected in result["commitments"].items():
        require(sha(run / f"{name}.json") == expected, f"Changed {name} commitment")
    for source, key in ((parent, "parent"), (meeting, "meeting")):
        for name, expected in result[key].items():
            require(sha(source / name) == expected, f"Changed source parent {name}")
    seal = load(run / "points.json")
    require(seal["freeze_sha256"] == sha(run / "freeze.json") and seal["heads_sha256"] == sha(run / "heads.pt"),
            "Fitted checkpoint differs")
    roster = {r["id"]: r for r in load(parent / "inputs.json")["rows"]}
    room = {r["id"]: r for r in load(meeting / "roster.json")}
    observations = load(run / "observations.json")
    require(len(observations) == len({(r["panel"], r["id"]) for r in observations}) == 216, "Observation roster differs")
    buckets = {}
    for row in observations:
        if row["panel"] == "matched":
            source, extent = roster[row["id"]], 64_000
            pcm = parent / source["path"]
            role, commitment = source["role"], source["pcm_sha256"]
        else:
            source, extent = room[row["id"]], 96_000
            pcm = meeting / "pcm" / f"{row['id']:03}.pcm"
            role, commitment = source["panel"], source["source_pcm_sha256"]
        require(role == row["role"] and source["truth"] == row["truth"] and source["condition"] == row["condition"]
                and commitment == row["pcm_sha256"] == sha(pcm) and pcm.stat().st_size == 2*extent, "PCM or source role differs")
        for arm, check in row["checks"].items():
            start, stop = check["range"]
            require(type(start) is int and type(stop) is int and 0 <= start < stop <= extent and start % 3200 == 0
                    and stop-start == (32_000 if arm == "incumbent_two_second" else 9_600), "Location differs from the fixed clock")
            require(check["located"] is within(check["range"], row["truth"]), "Truth overlap differs")
            if arm in seal["points"]:
                require(check["flag"] is (check["score"] > seal["points"][arm]), "Fixed decision point differs")
        buckets.setdefault("/".join(row[k] for k in ("panel", "role", "condition")), []).append(row)
    for key, rows in buckets.items():
        cell = result["cells"][key]
        for arm in rows[0]["checks"]:
            flagged = [r for r in rows if r["checks"][arm]["flag"]]
            require(cell[arm]["genuine_false_flags"] == sum(r["truth"] is None for r in flagged)
                    and cell[arm]["useful"] == sum(r["checks"][arm]["located"] for r in flagged), "Fixed results differ")
        for budget, arms in cell.get("equal_false_groups", {}).items():
            for arm, report in arms.items():
                selected = choose(rows, arm, int(budget))
                require(report["useful"] == sum(r["checks"][arm]["located"] for r in selected)
                        and report["genuine_false_groups"] == len({r["group"] for r in selected if r["truth"] is None}), "Equal-alarm results differ")
    loud = buckets["meeting/confirmation/room"]
    useful, false = set(), set()
    for arm, budget in (("spectrogram_local", 1), ("incumbent_two_second", 2)):
        selected = choose(loud, arm, budget)
        hits = [r for r in selected if r["checks"][arm]["located"]]
        useful.update(r["id"] for r in hits)
        false.update(r["group"] for r in selected if r["truth"] is None)
        print(arm, "useful sample spans and exact edit bounds", [(r["checks"][arm]["range"], r["truth"]) for r in hits])
    print(f"Verified {len(observations)} sources and sample-clock decisions. Post-hoc AMI union: {len(useful)} useful, {len(false)} false groups.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "parent", "meeting"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    verify(args.run, args.parent, args.meeting)
