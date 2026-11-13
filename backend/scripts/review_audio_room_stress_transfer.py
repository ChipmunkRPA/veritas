"""Verify the paired room stress receipt with Python's standard library."""
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
import sys

WINDOWS = ((16000, 48000), (48000, 80000))
NOISY = ("snr_5", "snr_0", "snr_-5", "snr_-10")
STAGES = ("dry", "early_room", "full_room", *NOISY)
POINTS = {"mono": 5.356589748054317, "earlier_either": 6.760727907655432, "directional": 7.068065580780201}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(value):
    return hashlib.sha256((json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()).hexdigest()


def overlaps(truth, window):
    a, b = truth
    c, d = window
    inter = max(0, min(b, d) - max(a, c))
    union = max(b, d) - min(a, c)
    return 2 * inter >= b - a and 10 * inter >= 3 * union


def pick(row, arm):
    options = []
    for track in row["responses"][arm]:
        if track is not None:
            require(len(track) == 2, "Invalid window count")
            for index, value in enumerate(track):
                if value is not None:
                    require(type(value) in (int, float) and math.isfinite(value), "Invalid score")
                    options.append((value, -index))
    return max(options, default=None)


def evaluate(rows, arm, point):
    statuses = {kind: [] for kind in ("located", "ordinary", "background", "wrong", "unassessed")}
    locations = {}
    for row in rows:
        best = pick(row, arm)
        group = row["group"]
        if best is None:
            statuses["unassessed"].append(group)
        elif best[0] > point:
            window = WINDOWS[-best[1]]
            if row["truth"]:
                if any(overlaps(truth, window) for truth in row["truth"]):
                    statuses["located"].append(group)
                    locations[group] = {"truth": row["truth"], "window": list(window), "family": row["family"], "room": row["room"]}
                else:
                    statuses["wrong"].append(group)
            else:
                statuses["ordinary"].append(group)
                if row["kind"] == "background_only":
                    statuses["background"].append(group)
    return {key: sorted(values) for key, values in statuses.items()}, locations


def verify(receipt):
    freeze = receipt["freeze"]
    require(receipt["schema"] == freeze["schema"] == "chipmunk.audio-room-stress/1", "Wrong receipt")
    require(sha(freeze) == receipt["freeze_sha256"] == receipt["point"]["freeze_sha256"], "Freeze changed")
    require(freeze["windows"] == [list(x) for x in WINDOWS] and freeze["prior_points"] == POINTS
            and freeze["levels_db"] == [5, 0, -5, -10], "Method differs")
    fit = receipt["development"]
    require(sha(fit) == receipt["point"]["development_sha256"], "Development ledger changed")
    admitted = defaultdict(list)
    for row in fit["rows"]:
        require(row["role"] == "development" and row["kind"] == "genuine" and not row["truth"]
                and row["stage"] in NOISY and set(row["responses"]) == {"auxiva"}, "Invalid development delivery")
        if (choice := pick(row, "auxiva")) is not None:
            admitted[row["group"]].append(choice[0])
    require(len(admitted) == 8 and len(fit["rows"]) == 32 and len({(row["group"], row["stage"]) for row in fit["rows"]}) == 32,
            "Development roster differs")
    point = max(max(scores) for scores in admitted.values())
    require(point == receipt["point"]["point"], "Development point changed")
    rows = receipt["assessment"]["rows"]
    require(len(rows) == 111 and all(row["role"] == "assessment" and row["group"] not in admitted for row in rows),
            "Assessment roster overlaps development")
    require(len({(row["group"], row["stage"]) for row in rows}) == 111, "Repeated assessment delivery")
    grouped = defaultdict(list)
    for row in rows:
        require(row["room"] in {"stairway", "aula_carolina"} and row["stage"] in (*STAGES, "background_only"), "Bad cohort")
        require(bool(row["truth"]) == bool(row["family"]) == (row["kind"] == "publisher_edit")
                and all(0 <= a < b <= 96000 for a, b in row["truth"]), "Bad labels")
        grouped[row["group"]].append(row)
    kinds = Counter()
    for paired in grouped.values():
        first = paired[0]
        kinds[first["kind"]] += 1
        stages = ("background_only",) if first["kind"] == "background_only" else STAGES
        require({row["stage"] for row in paired} == set(stages), "Missing paired stage")
        require(all([row[key] for key in ("truth", "family", "kind", "room")]
                    == [first[key] for key in ("truth", "family", "kind", "room")] for row in paired), "Paired source changed")
    require(kinds == {"publisher_edit": 7, "genuine": 8, "background_only": 6}, "Group counts changed")
    result, passes, flags, scores = {}, defaultdict(dict), defaultdict(set), defaultdict(dict)
    for stage in (*STAGES, "background_only"):
        subset = [row for row in rows if row["stage"] == stage]
        mono, ranges = evaluate(subset, "mono", POINTS["mono"])
        result[stage] = {"mono": mono, "mono_ranges": ranges}
        for row in subset:
            if row["truth"]:
                eligible = [value for window, value in zip(WINDOWS, row["responses"]["mono"][0], strict=True)
                            if value is not None and any(overlaps(truth, window) for truth in row["truth"])]
                if eligible:
                    scores[stage][row["group"]] = max(eligible)
        if stage not in (*NOISY, "background_only"):
            continue
        for arm, cutoff in (("auxiva_prior", POINTS["earlier_either"]), ("auxiva_development", point)):
            status, located = evaluate(subset, "auxiva", cutoff)
            additional = sorted(set(status["located"]) - set(mono["located"]))
            extra = sorted(set(status["ordinary"]) - set(mono["ordinary"]))
            status.update(additions={group: located[group] for group in additional}, added_ordinary=extra)
            result[stage][arm] = status
            flags[arm].update(extra)
            passes[arm][stage] = len(additional) >= 2 and len({located[group]["family"] for group in additional}) >= 2
    decision = {}
    for arm, states in passes.items():
        background = set(result["background_only"][arm]["added_ordinary"])
        adjacent = any(states[a] and states[b] for a, b in zip(NOISY, NOISY[1:]))
        decision[arm] = {"passes_adjacent_levels": adjacent, "added_background_only_groups": len(background),
                         "added_genuine_groups": len(flags[arm] - background),
                         "passes": adjacent and len(flags[arm]) <= 1 and not background}
    changes = {}
    for a, b in zip(STAGES, STAGES[1:]):
        first, last = (set(result[x]["mono"]["located"]) for x in (a, b))
        differences = [scores[b][g] - scores[a][g] for g in scores[a].keys() & scores[b].keys()]
        changes[f"{a}_to_{b}"] = {"paired_truth_window_scores": len(differences),
                                  "median_log_odds_change": round(median(differences), 6) if differences else None,
                                  "lost_locations": sorted(first - last), "gained_locations": sorted(last - first)}
    summary = {"stages": result, "paired_transitions": changes, "decision": decision}
    require(summary == receipt["summary"], "Public readout differs")
    return summary


if __name__ == "__main__":
    print(json.dumps(verify(json.loads(Path(sys.argv[1]).read_bytes())), indent=2))
