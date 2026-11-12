"""Verify isolated-sentence scope, selection and published counts without model imports."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from scripts import review_writing_agreement_revision_transfer as prior

ARMS = ("current", "native_agreement", "focus_agreement", "either_agreement")
SYNTHETIC = {"granite", "olmo"}


def indices(text, ranges):
    return {i for start, stop in ranges for i in range(start, stop) if not text[i].isspace()}


def reconstruct(run, parent, elife_source):
    prior.review(parent)
    prior.review_elife(parent / "elife", elife_source)
    prior.review_olmo(parent / "olmo")
    aggregate = prior.read(run / "aggregate.json")
    lock = prior.read(run / "freeze.json")
    prior.check(prior.sha(run / "freeze.json") == aggregate["freeze_sha256"], "Freeze commitment differs")
    isolated = {}
    for arm in ("current", "second"):
        file = run / f"{arm}-focus.json"
        packet = prior.read(file)
        prior.check(prior.sha(file) == aggregate["score_receipts"][arm]["sha256"]
                    and packet["freeze_sha256"] == aggregate["freeze_sha256"], "Score artifact differs")
        isolated[arm] = {digest: entry["score"] for digest, entry in packet["responses"].items()}

    text_controls = {entry["case"]: entry["text"] for entry in
                     (json.loads(line) for line in (elife_source / "private-controls.jsonl").read_text().splitlines())}
    summary = defaultdict(list)
    counted, windows, used = 0, 0, set()
    transitions = {arm: defaultdict(int) for arm in ("current", "second")}
    diagnostic = defaultdict(lambda: {arm: defaultdict(int) for arm in ("current", "second")})
    for panel in (parent, parent / "olmo", parent / "elife"):
        is_science = panel.name == "elife"
        raw = {arm: prior.read(panel / f"{arm}.json")["responses"] for arm in ("current", "second")}
        source = {} if is_science else {(c["group"], c["role"]): c for c in prior.read(panel / "cases.json")["cases"]}
        for row in prior.read(panel / "private-readout.json"):
            case = None if is_science else source[row["group"], row["role"]]
            if case is None:
                ranges = row["source_paragraphs"]
                text = text_controls[row["case"]][ranges[0][0]:ranges[-1][1]]
                role, publisher, target = row["arm"], "elife", None
            else:
                text, role, publisher, target = (case[key] for key in ("text", "role", "publisher", "target"))
            check = {arm: set() for arm in ARMS}
            useful = set()
            for n, record in enumerate(row["windows"]):
                windows += 1
                spans = record["spans"]
                focus = spans[1 if n else 0]
                sentence = text[slice(*focus)]
                focus_key = prior.sha(sentence) if len(sentence.split()) >= 12 else None
                ordinary_scores = {arm: raw[arm][record["sha256"]]["score"] for arm in raw}
                local_scores = {arm: isolated[arm][focus_key] for arm in isolated} if focus_key else {}
                if focus_key:
                    used.add(focus_key)
                    counted += 1
                    for arm, local in local_scores.items():
                        states = ["positive" if score >= .5 else "negative" for score in (ordinary_scores[arm], local)]
                        transitions[arm]["_to_".join(states)] += 1
                if role in SYNTHETIC:
                    known = indices(text, [case["focus"]])
                    shown = indices(text, spans)
                    if 10 * len(known & shown) >= 3 * len(known | shown):
                        useful.add(n)
                    if target == n:
                        for arm, original in ordinary_scores.items():
                            local = local_scores.get(arm)
                            cells = diagnostic[role][arm]
                            cells["cases"] += 1
                            cells["native_positive"] += original >= .5
                            cells["focus_positive"] += local is not None and local >= .5
                            cells["focus_above_native"] += local is not None and local > original
                            cells["focus_unavailable"] += local is None
                if ordinary_scores["current"] < .5 or len(check["current"]) == 6:
                    continue
                check["current"].add(n)
                if ordinary_scores["second"] >= .5:
                    check["native_agreement"].add(n)
                if local_scores.get("second", 0) >= .5:
                    check["focus_agreement"].add(n)
            check["either_agreement"] = check["native_agreement"] | check["focus_agreement"]
            source_key = (publisher, row["group"])
            row_stats = (source_key, target, check, useful)
            if role in SYNTHETIC:
                summary[role].append(row_stats)
                summary["all_synthetic"].append(row_stats)
            else:
                scope = "elife_human" if is_science else "forum_human"
                summary[scope].append(row_stats)
                summary["all_human"].append(row_stats)
                if not is_science:
                    summary[f"forum_{role}"].append(row_stats)
    prior.check(used == set(lock["inputs"]) and all(used == set(packet) for packet in isolated.values())
                and counted == aggregate["eligible_occurrences"] and windows == aggregate["native_window_occurrences"],
                "Focus input roster or exact occurrence count differs")
    prior.check(aggregate["scope_transitions"] == transitions and aggregate["synthetic_target_diagnostic"] == diagnostic,
                "Context-to-focus transition counts differ")
    for scope, rows in summary.items():
        outcome = aggregate["outcomes"][scope]
        prior.check(len(rows) == outcome["files"], "Scope size differs")
        for arm in ARMS:
            relevant = [(group, target, marked, useful) for group, target, marked, useful in rows if marked[arm]]
            cells = {"flagged_files": len(relevant), "flagged_groups": len({row[0] for row in relevant})}
            if scope in SYNTHETIC | {"all_synthetic"}:
                correct = sum(bool(row[2][arm] & row[3]) for row in relevant)
                cells.update(useful=correct, wrong_location=len(relevant) - correct, unflagged=len(rows) - len(relevant),
                             exact_focus=sum(target in marked[arm] for _, target, marked, _ in rows),
                             current_useful_lost=sum(bool(marked["current"] & useful) and not (marked[arm] & useful)
                                                     for _, _, marked, useful in rows))
            prior.check(cells == outcome["outcomes"][arm], "One scope and arm differs")
    return {"scoped_cases": len(summary["all_human"]) + len(summary["all_synthetic"]),
            "scope_arm_counts": len(summary) * len(ARMS), "verified_native_windows": windows,
            "verified_eligible_foci": counted, "verified_distinct_inputs": len(used)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("parent", type=Path)
    parser.add_argument("elife_source", type=Path)
    args = parser.parse_args()
    print(json.dumps(reconstruct(args.run, args.parent, args.elife_source), sort_keys=True))


if __name__ == "__main__":
    main()
