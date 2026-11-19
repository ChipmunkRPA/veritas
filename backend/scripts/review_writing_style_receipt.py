"""Recount a portable short-passage style evaluation without source prose or weights."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import re
import struct
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

SCHEMA = "chipmunk.writing-style-localization-scalar/1"
ARMS = ("own_left", "foil_left", "own_right", "foil_right", "cue_only")
GENERATED = ("same_family_rewrite", "independent_family_rewrite")
ROLES = {"earlier": 14, "later": 14, GENERATED[0]: 7, GENERATED[1]: 1}
METHODS = ("styles", "cue", "incumbent")
FACTS = ("fact_preserving", "fact_changing")
POLICY = {"minimum_complete_foci": 4, "scored_positions_per_focus": 16,
          "minimum_target_nonspace_recall": [1, 2], "minimum_nonspace_iou": [3, 10]}
HEX = re.compile(r"[0-9a-f]{64}\Z")
COMMAND = re.compile(r"C[0-9]{3}\Z")
MAX_RECEIPT = 2_000_000


def require(condition, message):
    if not condition:
        raise ValueError(message)


def reject_duplicate_fields(items):
    result = {}
    for key, value in items:
        require(key not in result, "A JSON field is duplicated")
        result[key] = value
    return result


def digest(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()


def bounds(value, length):
    return (isinstance(value, list) and len(value) == 2 and all(type(pos) is int for pos in value)
            and 0 <= value[0] < value[1] <= length)


def parse_score(encoded):
    if type(encoded) is not str or len(encoded) > 32:
        raise ValueError("Invalid score encoding")
    score = float.fromhex(encoded)
    require(math.isfinite(score) and score <= 0 and score.hex() == encoded, "Invalid finite score")
    return score


def winners(foci, scores):
    left = [own - foil for own, foil, _, _, _ in scores]
    right = [own - foil for _, _, own, foil, _ in scores]
    median_left, median_right = median(left), median(right)
    margins = [min(median_left - a, median_right - b) for a, b in zip(left, right, strict=True)]
    cue = [-row[-1] for row in scores]

    def chosen(values, positive=False):
        high = max(values)
        return foci[values.index(high)]["source"] if values.count(high) == 1 and (not positive or high > 0) else None

    return {"styles": chosen(margins, positive=True), "cue": chosen(cue)}


def overlap(case, selected):
    target = case["target"]
    a, b = selected
    x, y = target
    counts = case["nonspace"]
    common = counts[min(b, y)] - counts[max(a, x)] if max(a, x) < min(b, y) else 0
    selected_chars, target_chars = counts[b] - counts[a], counts[y] - counts[x]
    return common > 0 and 2 * common >= target_chars and 10 * common >= 3 * (selected_chars + target_chars - common)


def unpack(cases):
    rows = {}
    scores_by_command = defaultdict(list)
    total_foci = 0
    for case in cases:
        require(isinstance(case, dict) and set(case) == {"work", "role", "characters", "assessed", "target", "factual",
                "focuses", "selections", "nonspace_prefix"}, "Invalid case fields")
        ident = (case["work"], case["role"])
        require(type(ident[0]) is int and 1 <= ident[0] <= 14 and ident[1] in ROLES and ident not in rows,
                "Missing or duplicated anonymous work role")
        total = case["characters"]
        require(type(total) is int and 0 < total <= 50_000 and bounds(case["target"], total), "Invalid source extent")
        factual = case["factual"]
        require(factual in FACTS if ident[1] in GENERATED else factual is None, "Invalid factual split")
        foci = case["focuses"]
        require(isinstance(foci, list) and len(foci) <= 64 and case["assessed"] is (len(foci) >= 4),
                "Incomplete focus coverage")
        selections = case["selections"]
        require(isinstance(selections, dict) and set(selections) == set(METHODS), "Invalid reading comparison")
        incumbent = selections["incumbent"]
        require(isinstance(incumbent, list) and all(bounds(x, total) for x in incumbent), "Invalid incumbent intervals")
        score_rows = []
        eligible = set()
        current_end = 0
        for focus in foci:
            require(isinstance(focus, dict) and set(focus) == {"source", "scored_source_ranges", "local_example_ranges", "scores"},
                    "Invalid focus fields")
            source = focus["source"]
            require(bounds(source, total) and current_end <= source[0], "Overlapping or out-of-order focus intervals")
            current_end = source[1]
            eligible.add(tuple(source))
            scored = focus["scored_source_ranges"]
            require(isinstance(scored, list) and len(scored) == 16 and all(bounds(x, total) and source[0] <= x[0] < x[1] <= source[1] for x in scored),
                    "Scored position falls outside its source focus")
            examples = focus["local_example_ranges"]
            require(isinstance(examples, list) and len(examples) == 2 and all(bounds(x, total) for x in examples)
                    and examples[0][1] <= source[0] < source[1] <= examples[1][0], "Invalid local example source intervals")
            reading = focus["scores"]
            require(isinstance(reading, list) and len(reading) == len(ARMS), "Missing comparison scores")
            values = []
            for command, encoded in reading:
                require(isinstance(command, str) and COMMAND.fullmatch(command), "Invalid anonymous command ordinal")
                value = parse_score(encoded)
                scores_by_command[command].append(struct.pack(">d", value))
                values.append(value)
            score_rows.append(values)
        recomputed = winners(foci, score_rows) if case["assessed"] else {"styles": None, "cue": None}
        require(all(selections[k] == v for k, v in recomputed.items()), "A saved rank does not reproduce")
        necessary = {0, total, *case["target"], *(point for interval in incumbent for point in interval),
                     *(point for interval in eligible for point in interval)}
        prefix = case["nonspace_prefix"]
        require(isinstance(prefix, list) and len(prefix) == len(necessary), "Missing non-whitespace count witness")
        counts = {}
        previous_pos = previous_count = 0
        for pos, count in prefix:
            require(type(pos) is int and type(count) is int and pos in necessary and pos not in counts and pos >= previous_pos
                    and 0 <= count - previous_count <= pos - previous_pos, "Invalid non-whitespace count witness")
            counts[pos] = count
            previous_pos, previous_count = pos, count
        require(counts[case["target"][1]] > counts[case["target"][0]], "The target marks no readable writing")
        rows[ident] = {"target": case["target"], "assessed": case["assessed"], "eligible": eligible,
                       "selections": selections, "nonspace": counts, "factual": factual,
                       "examples": {tuple(x["source"]): x["local_example_ranges"] for x in foci}}
        total_foci += len(foci)
    require(Counter(role for _, role in rows) == ROLES and all((w, "earlier") in rows and (w, "later") in rows for w, _ in rows),
            "Natural comparison pairs or rewrite-family counts differ")
    require(len(rows) == 36 and total_foci == 141, "Source geometry count changed")
    multiplicity = Counter(len(items) for items in scores_by_command.values())
    require(multiplicity == {1: 155, 2: 95, 3: 120}, "Duplicate-input multiplicities changed")
    require(set(scores_by_command) == {f"C{i:03d}" for i in range(1, 371)}
            and sum(k * v for k, v in multiplicity.items()) == 705, "Command aggregate count changed")
    duplicate_groups = [values for values in scores_by_command.values() if len(values) > 1]
    bitwise = sum(len(set(v)) == 1 for v in duplicate_groups)
    pack_decoded = [[struct.unpack(">d", v)[0] for v in vs] for vs in duplicate_groups]
    largest = max(max(decoded) - min(decoded) for decoded in pack_decoded)
    return rows, {"all_occurrences": 705, "distinct_commands": 370, "repeated_command_groups": len(duplicate_groups),
                  "repeated_command_groups_with_identical_binary64_scores": bitwise, "largest_score_difference": largest}


def recount(cases):
    rows, repeated = unpack(cases)

    def chosen(case, method):
        value = case["selections"][method]
        return value if method == "incumbent" else [] if value is None else [value]

    def assessment(case, method):
        return method == "incumbent" or case["assessed"]

    def eligible_at_target(case, method):
        return assessment(case, method) and (method == "incumbent" or tuple(case["target"]) in case["eligible"])

    edited = {role: {} for role in GENERATED}
    diagnostics = {role: {} for role in GENERATED}
    natural = {}
    factual = {role: defaultdict(Counter) for role in GENERATED}
    for method in METHODS:
        hist = Counter()
        for work, role in rows:
            if role != "earlier":
                continue
            first, second = rows[work, "earlier"], rows[work, "later"]
            original, revised = chosen(first, method), chosen(second, method)
            start, end = assessment(first, method), assessment(second, method)
            first_hit = any(overlap(first, region) for region in original)
            second_hit = any(overlap(second, region) for region in revised)
            hist.update(known_pairs=1, earlier_assessed=start, later_assessed=end, both_assessed=start and end,
                        earlier_selected=bool(original), later_selected=bool(revised),
                        selected_only_after=start and end and not original and bool(revised), selected_both=bool(original and revised),
                        earlier_edit_scope_selected=first_hit, actual_edit_scope_selected=second_hit,
                        edit_scope_selected_only_after=start and end and second_hit and not first_hit,
                        edit_scope_selected_before_and_after=first_hit and second_hit)
        natural[method] = {key: int(value) for key, value in hist.items()}
        for label in GENERATED:
            stats, misses = Counter(), Counter()
            for (work, role), case in rows.items():
                if role != label:
                    continue
                original_case = rows[work, "earlier"]
                proposed = chosen(case, method)
                assessed, eligible = assessment(case, method), eligible_at_target(case, method)
                paired = eligible and eligible_at_target(original_case, method)
                correct = any(overlap(case, p) for p in proposed)
                original_match = any(overlap(original_case, p) for p in chosen(original_case, method))
                stats.update(known_edits=1, assessed=assessed, uncovered_files=not assessed, eligible_focus=eligible,
                             matched_original_focus_eligible=paired, selected=bool(proposed), correct_locations=correct,
                             wrong_locations=sum(not overlap(case, p) for p in proposed), matching_earlier_target=original_match,
                             correct_and_earlier_target_selected=correct and original_match,
                             correct_with_eligible_earlier_target_without_earlier_selection=correct and paired and not original_match)
                if method != "incumbent":
                    if not assessed:
                        outcome = "file_has_too_few_foci"
                    elif not eligible:
                        outcome = "target_outside_focus_geometry"
                    elif correct:
                        outcome = "correct"
                    elif not proposed:
                        outcome = "eligible_no_selected_rank"
                    else:
                        a, b = case["target"]
                        near = any(lo <= a < b <= hi for lo, hi in case["examples"][tuple(proposed[0])])
                        outcome = "wrong_rank_target_inside_local_example" if near else "wrong_rank_target_outside_local_examples"
                    misses[outcome] += 1
                facts = factual[label][case["factual"]]
                facts[method + "_correct"] += correct
                if method == "styles":
                    facts.update(files=1, eligible_for_exact_paired_geometry=paired)
            edited[label][method] = {key: int(value) for key, value in stats.items()}
            if method != "incumbent":
                require(sum(misses.values()) == ROLES[label], "Miss diagnostics do not cover the roster")
                diagnostics[label][method] = dict(misses)
    return {"cases": 36, "foci": 141, "prompt_scores": 705, "edited": edited, "natural_revisions": natural,
            "coverage_and_miss_diagnostics": diagnostics,
            "factual_sensitivity": {label:{fact:dict(counts) for fact,counts in parts.items()} for label,parts in factual.items()},
            "repeated_command_replay": repeated}


def review(path):
    open_file = gzip.open if path.suffix == ".gz" else Path.open
    with open_file(path, "rb") as stream:
        data = stream.read(MAX_RECEIPT + 1)
    require(len(data) <= MAX_RECEIPT, "Research receipt exceeds its size bound")
    receipt = json.loads(data, object_pairs_hook=reject_duplicate_fields)
    require(isinstance(receipt, dict) and set(receipt) == {"schema", "rule", "arm_order", "commitments", "rows", "rows_sha256", "closure"},
            "Unknown research receipt fields")
    require(receipt["schema"] == SCHEMA and receipt["rule"] == POLICY and receipt["arm_order"] == list(ARMS),
            "Unknown frozen measurement rules")
    commitments = receipt["commitments"]
    require(isinstance(commitments, dict) and set(commitments) == {"frozen_input", "frozen_evaluator", "frozen_rules", "private_output", "independent_recount"}
            and all(isinstance(value, str) and HEX.fullmatch(value) for value in commitments.values()), "Invalid experiment commitments")
    require(receipt["rows_sha256"] == digest(receipt["rows"]), "Anonymized rows differ from their commitment")
    result = recount(receipt["rows"])
    require(result == receipt["closure"], "Recorded outcome differs from the independent arithmetic")
    return result


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("receipt", type=Path)
    try:
        print(json.dumps(review(cli.parse_args().receipt), sort_keys=True, indent=2))
    except (KeyError, TypeError, ValueError, OSError) as error:
        raise SystemExit(f"Could not verify research receipt: {error}") from None
