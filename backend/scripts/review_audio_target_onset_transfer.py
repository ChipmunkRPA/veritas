"""Verify frozen target-onset score arithmetic and source-role separation without inference."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from statistics import median


def read(path):
    return json.loads(path.read_bytes())


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify(run):
    result = read(run / "result.json")
    for key, checksum in result["commitments"].items():
        require(sha(run / f"{key}.json") == checksum, f"Changed {key} ledger")
    roster, frozen = (read(run / f"{key}.json") for key in ("roster", "freeze"))
    require(frozen["roster_sha256"] == sha(run / "roster.json"), "Freeze is disconnected from roster")
    chosen = read(run / "points.json")
    require(chosen["selection_sha256"] == sha(run / "selection.json"), "Points are disconnected from selection")
    selection, confirmation = (read(run / f"{key}.json") for key in ("selection", "confirmation"))
    require(selection["points_sha256"] is None and confirmation["points_sha256"] == sha(run / "points.json")
            and selection["freeze_sha256"] == confirmation["freeze_sha256"] == sha(run / "freeze.json"),
            "Score role is disconnected from its freeze or points")
    arms = "mono", "onset_steered"
    stages = {"selection": selection, "confirmation": confirmation}
    tables, source_groups, rooms = {}, {}, {}
    for role, stage in stages.items():
        actual = {row["id"]: row for row in stage["rows"]}
        role_rows = [row for row in roster if row["role"] == role]
        require(len(actual) == len(stage["rows"]) == len(role_rows), "Unexpected role roster size")
        table = tables[role] = []
        source_groups[role] = {row["group"] for row in role_rows if row["source"] == "center"}
        rooms[role] = {row["room"] for row in role_rows if row["source"] == "center"}
        for row in role_rows:
            observation = actual[row["id"]]
            require(observation["pcm_sha256"] == row["pcm_sha256"], "Response joined to a different query")
            for arm in arms:
                path = run / "pcm" / f"{row['id']:03}-{arm}.pcm"
                require(path.stat().st_size == 80_000 and sha(path) == row["pcm_sha256"][arm], "Exact query changed")
            if observation["candidate_input"] == "mono":
                require(observation["mono"] == observation["onset_steered"], "Mono fallback changed score")
            else:
                require(row["source"] == "center" and row["target"] is not None and observation["mono"] is not None
                        and observation["onset_steered"] is not None, "Steered score lacks required observation")
            table.append((row, observation))
    require(not rooms["selection"] & rooms["confirmation"] and not source_groups["selection"] & source_groups["confirmation"],
            "Source or room roles overlap")

    def flags(entries, arm, cutoff):
        return Counter(row["truth"] for row, obs in entries if obs[arm] is not None and obs[arm] > cutoff)

    point = {}
    for arm in arms:
        genuine = [(row["source"], obs[arm]) for row, obs in tables["selection"] if row["truth"] == "genuine" and obs[arm] is not None]
        controls = max(value for kind, value in genuine if kind != "center")
        mixed = sorted(value for kind, value in genuine if kind == "center")
        point[arm] = max(mixed[-3], controls)
    require(chosen["points"] == point, "Operating points do not replay")
    held_noisy = [(row, obs) for row, obs in tables["confirmation"] if row["source"] == "center"]
    for arm in arms:
        count = flags(held_noisy, arm, point[arm])
        require(all(count[kind] == result["confirmation_noisy_flags"][kind][arm] for kind in ("genuine", "synthetic")),
                "Frozen held totals do not replay")
        scores = sorted(obs[arm] for row, obs in held_noisy if row["truth"] == "genuine" and obs[arm] is not None)
        for budget in (0, 1, 2, 4):
            count = flags(held_noisy, arm, scores[-budget-1])
            require(result["confirmation_equal_false_alarm"][str(budget)][arm] ==
                    {"false_alarms": count["genuine"], "ai_flags": count["synthetic"]}, "Equal-alarm totals do not replay")
    for room, aggregate in result["confirmation_acoustics"].items():
        records = [row["acoustics"] for row, _ in held_noisy if row["room"] == room and row["target"] is not None]
        require(len(records) == aggregate["accepted_onset"] and all(median(row[key] for row in records) == value
                for key, value in aggregate.items() if key != "accepted_onset"), "Held room medians do not replay")
    print(f"Independently verified {len(roster)} rows, {len(roster)*2} PCM hashes, frozen points, held totals and room medians.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    verify(parser.parse_args().run)
