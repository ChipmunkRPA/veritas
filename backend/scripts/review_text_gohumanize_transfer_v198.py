"""Independently replay v196 projection, accounting and metrics from committed records."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PANEL = ROOT / "docs/evaluation-panels"
BASE = ROOT / "backend/data/independent-evaluation"
INTAKE = BASE / "text-gohumanize-intake-v193"
TRIAL = BASE / "text-gohumanize-transfer-v196"
INTAKE_FREEZE = PANEL / "text-gohumanize-preparation-freeze-v193.json"
INTAKE_RESULT = PANEL / "text-gohumanize-intake-complete-v193.json"
TRANSFER_FREEZE = PANEL / "text-gohumanize-transfer-freeze-v196.json"
REFERENCE = PANEL / "text-format-response-freeze-v94.json"
EXPECTED = PANEL / "text-gohumanize-transfer-complete-v196.json"
ARMS = ("human", "full_ai", "short_ai", "short_format", "long_ai", "long_format")
METHODS = ("baseline", "canonical_current", "canonical_union", "baseline_plus_canonical_union")
GENERATOR_PAIRS = {"gpt-4o-mini": 39, "deepseek-chat": 15}
BATCHES, PAIRS_PER_BATCH, ARTICLE_GROUPS = 6, 9, 30
GIB = 1024**3
NONSPACE = re.compile(r"\S+")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def text_digest(text):
    return digest(text.encode("utf-8", "surrogatepass"))


def direct(path):
    require(path.is_file() and not path.is_symlink(), f"Missing or non-direct reviewer input: {path.name}")
    return path.read_bytes()


def bound_file(path, expected):
    data = direct(path)
    require(digest(data) == expected, f"Reviewer input commitment changed: {path.name}")
    return data


def json_lines(data):
    return [json.loads(line) for line in data.splitlines()]


def exact_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def paragraph_ranges(raw):
    """Recognize a blank line separated by ASCII horizontal whitespace."""
    start = cursor = 0
    while (first := raw.find("\n", cursor)) != -1:
        second = first + 1
        while second < len(raw) and raw[second] in " \t":
            second += 1
        if raw[second : second + 2] == "\r\n":
            second += 1
        if raw[second : second + 1] == "\n":
            stop = first - int(first > start and raw[first - 1] == "\r")
            yield start, stop
            start = cursor = second + 1
        else:
            cursor = first + 1
    yield start, len(raw)


@dataclass(frozen=True)
class View:
    text: str
    positions: tuple[int | None, ...]


def view(raw, *, paragraphs):
    # Emit tokens and their original indices together. Inserted spaces and paragraph breaks have no source position.
    chunks, mapped = [], []
    source_ranges = paragraph_ranges(raw) if paragraphs else ((0, len(raw)),)
    for start, stop in source_ranges:
        tokens = list(NONSPACE.finditer(raw, start, stop))
        if chunks and tokens:
            chunks.append("\n\n")
            mapped.extend((None, None))
        for ordinal, match in enumerate(tokens):
            if ordinal:
                chunks.append(" ")
                mapped.append(None)
            chunks.append(match.group())
            mapped.extend(range(match.start(), match.end()))
    text = "".join(chunks)
    expected = [(i, c) for i, c in enumerate(raw) if not c.isspace()]
    observed = [(i, c) for i, c in zip(mapped, text, strict=True) if i is not None]
    require(
        expected and expected == observed and raw.split() == text.split(),
        "Original Unicode characters or token boundaries were not preserved",
    )
    return View(text, tuple(mapped))


def span(value, length):
    require(
        isinstance(value, list)
        and len(value) == 2
        and all(type(i) is int for i in value)
        and 0 <= value[0] < value[1] <= length,
        "Invalid original Unicode source range",
    )
    return value


def original_positions(row):
    raw = row["text"]
    require(isinstance(raw, str) and text_digest(raw) == row["text_sha256"], "Prepared Unicode bytes differ")
    universe = {i for i, c in enumerate(raw) if not c.isspace()}
    ranges = row["ai_spans"]
    require(universe and isinstance(ranges, list), "Missing original Unicode truth")
    ranges = [span(bounds, len(raw)) for bounds in ranges]
    require(
        all(a[1] <= b[0] for a, b in zip(ranges, ranges[1:])), "Original labels overlap or run out of order"
    )
    ai = universe.intersection(i for first, last in ranges for i in range(first, last))
    require(
        bool(ai) == row["arm"].endswith("_ai") and (row["arm"] != "full_ai" or ai == universe),
        "Source truth disagrees with the planned arm",
    )
    return sorted(universe), ai


def check_pair(group, roster):
    require(
        tuple(row["arm"] for row in group) == ARMS
        and all(
            row["case"] == f"{roster['pair']}-{row['arm']}"
            and all(row[key] == roster[key] for key in ("pair", "article_group", "agency", "generator"))
            for row in group
        ),
        "Frozen article roster or arm order changed",
    )
    human, ai = group[:2]
    require(
        human["status"] == ai["status"] == "constructed"
        and human["text_sha256"] == roster["human_sha256"]
        and ai["text_sha256"] == roster["full_ai_sha256"],
        "Base passages do not match intake",
    )
    host, donor = human["text"], ai["text"]
    require(human["ai_spans"] == [] and ai["ai_spans"] == [[0, len(donor)]], "Base passage label changed")
    for insert, control in (group[2:4], group[4:6]):
        require(insert["status"] == control["status"], "Matched construction/control availability differs")
        if insert["status"] == "unavailable":
            require(
                all(
                    row[key] is None
                    for row in (insert, control)
                    for key in (
                        "text",
                        "text_sha256",
                        "ai_spans",
                        "host_range",
                        "donor_range",
                        "format_range",
                    )
                ),
                "Unavailable arms supplied original text or labels",
            )
            continue
        require(insert["status"] == "constructed", "Unknown preparation disposition")
        left, right = span(insert["host_range"], len(host))
        first, last = span(insert["donor_range"], len(donor))
        replacement = donor[first:last]
        wrap, end = span(control["format_range"], len(host))
        require(
            insert["text"] == host[:left] + replacement + host[right:]
            and insert["ai_spans"] == [[left, left + len(replacement)]]
            and control["host_range"] == [left, right]
            and control["donor_range"] is None
            and left <= wrap < end <= right
            and host[wrap:end] == " "
            and control["text"] == host[:wrap] + "\n" + host[end:]
            and control["ai_spans"] == [],
            "Prepared replacement or all-human formatting control differs at its original offsets",
        )


def check_roster(rows, roster, source):
    pairs = BATCHES * PAIRS_PER_BATCH
    require(
        len(roster) == pairs
        and len(rows) == pairs * len(ARMS)
        and len({row["case"] for row in rows}) == len(rows)
        and [row["pair"] for row in roster] == [f"pair-{i:02d}" for i in range(pairs)]
        and Counter(row["generator"] for row in roster) == GENERATOR_PAIRS
        and Counter(row["agency"] for row in roster) == source["pairs_by_agency"]
        and len({row["article_group"] for row in roster}) == ARTICLE_GROUPS,
        "Complete unique frozen article-pair denominator required",
    )
    counts = {arm: {"constructed": 0, "unavailable": 0} for arm in ARMS}
    grouped = {name: {arm: {"constructed": 0, "unavailable": 0} for arm in ARMS} for name in GENERATOR_PAIRS}
    present = {arm: set() for arm in ARMS}
    for ordinal, record in enumerate(roster):
        first = ordinal * len(ARMS)
        group = rows[first : first + len(ARMS)]
        check_pair(group, record)
        for row in group:
            counts[row["arm"]][row["status"]] += 1
            grouped[row["generator"]][row["arm"]][row["status"]] += 1
            if row["status"] == "constructed":
                present[row["arm"]].add(row["article_group"])
    require(
        counts == source["by_arm"]
        and grouped == source["by_generator"]
        and {arm: len(groups) for arm, groups in present.items()}
        == source["article_groups_with_constructed_case_by_arm"]
        and len(rows) == source["planned_cases"]
        and sum(row["constructed"] for row in counts.values()) == source["constructed_cases"]
        and sum(row["unavailable"] for row in counts.values()) == source["unavailable_cases"],
        "Rebuilt roster counts disagree with the published preparation receipt",
    )


@dataclass(frozen=True)
class Mask:
    selected: frozenset[int]
    observed: frozenset[int]


def project(rendered, proposals, point):
    if proposals is None:
        return None
    require(isinstance(proposals, list), "A recorded response must be a list or explicitly unavailable")
    selected, observed = set(), set()
    for row in proposals:
        start, end, score = row["start"], row["end"], row["score"]
        require(
            type(start) is type(end) is int
            and 0 <= start <= end <= len(rendered.text)
            and type(score) in (int, float)
            and math.isfinite(score)
            and 0 <= score <= 1,
            "Proposal is not bounded by the independent Unicode view",
        )
        actual = {i for i in rendered.positions[start:end] if i is not None}
        observed.update(actual)
        if score >= point:
            selected.update(actual)
    return Mask(frozenset(selected), frozenset(observed))


def combine(left, right):
    if left is None or right is None:
        return None
    return Mask(left.selected | right.selected, left.observed | right.observed)


def measure(universe, ai, mask):
    human = set(universe) - ai
    truth = {"ai_characters": len(ai), "human_characters": len(human)}
    if mask is None:
        return {"status": "unassessed", **truth}
    require(mask.selected <= mask.observed <= set(universe), "Mask exceeds observed original characters")
    found = mask.selected & ai
    union = ai | mask.selected
    iou = len(found) / len(union) if union else 1.0
    # Compress only the ordered eligible positions. Unlabelled whitespace never splits a true interval.
    lengths, hits = [], []
    on = False
    for index in universe:
        if index not in ai:
            on = False
            continue
        if not on:
            lengths.append(0)
            hits.append(0)
            on = True
        lengths[-1] += 1
        hits[-1] += index in found
    return {
        "status": "ok",
        **truth,
        "located_ai_characters": len(found),
        "highlighted_human_characters": len(mask.selected & human),
        "missed_ai_characters": len(ai - found),
        "unscored_ai_characters": len(ai - mask.observed),
        "unscored_human_characters": len(human - mask.observed),
        "character_iou": iou,
        "useful_ai_location": bool(ai) and iou >= 0.3,
        "known_ai_intervals": len(lengths),
        "recovered_ai_intervals": sum(2 * hit >= size for hit, size in zip(hits, lengths, strict=True)),
        "genuine_document_flagged": not ai and bool(mask.selected),
        "ai_document_with_no_ai_located": bool(ai) and not found,
    }


def measurements(row, response, cuts):
    if row["status"] == "unavailable":
        require(
            response == {"case": row["case"], "status": "preparation_unavailable"},
            "Unavailable preparation must have no saved scores",
        )
        return {method: {"status": "preparation_unavailable"} for method in METHODS}
    universe, ai = original_positions(row)
    primary = view(row["text"], paragraphs=True)
    canonical = view(row["text"], paragraphs=False)
    if response["status"] == "unassessed":
        require(response.get("reason") == "extraction_correspondence", "Unknown unassessed case reason")
        masks = (None,) * len(METHODS)
    else:
        require(response["status"] == "measured", "Unknown response disposition")
        shared = primary.text == canonical.text
        proofs = response["native_proofs"]
        recorded = [response[key] for key in ("primary_current", "canonical_current", "canonical_interval")]
        require(
            response["canonical_shared_current"] is shared
            and len(proofs) == len(recorded)
            and all(
                (proof is None) == (proposals is None)
                for proof, proposals in zip(proofs, recorded, strict=True)
            )
            and (not shared or recorded[0] == recorded[1] and proofs[0] == proofs[1]),
            "Saved view sharing or absent response proof disagrees",
        )
        baseline = project(primary, recorded[0], cuts["current"])
        current = project(canonical, recorded[1], cuts["current"])
        interval = project(canonical, recorded[2], cuts["interval"])
        canonical_union = combine(current, interval)
        masks = (baseline, current, canonical_union, combine(baseline, canonical_union))
    return {method: measure(universe, ai, mask) for method, mask in zip(METHODS, masks, strict=True)}


def aggregate(rows, responses, cuts):
    require(
        [row["case"] for row in rows] == [row["case"] for row in responses],
        "Every planned case, including unavailable cases, requires one ordered response",
    )
    counts, flagged = defaultdict(Counter), defaultdict(set)
    for row, response in zip(rows, responses, strict=True):
        for method, item in measurements(row, response, cuts).items():
            for generator in ("all", row["generator"]):
                key = generator, row["arm"], method
                tally = counts[key]
                tally[item["status"]] += 1
                for field in item.keys() - {"status"}:
                    tally[field] += item[field]
                if item.get("genuine_document_flagged"):
                    flagged[key].add(row["article_group"])
    report = {}
    for (generator, arm, method), tally in sorted(counts.items()):
        values = dict(tally)
        numerator = values.pop("character_iou", 0)
        values.update(
            planned=tally["ok"] + tally["unassessed"] + tally["preparation_unavailable"],
            mean_character_iou_assessed=numerator / tally["ok"] if tally["ok"] else None,
            genuine_article_groups_flagged=len(flagged[generator, arm, method]),
        )
        report.setdefault(generator, {}).setdefault(arm, {})[method] = values
    return report


def verify_resources(profile, resources):
    require(resources, "Missing actual resource journal observations")
    last = 0
    for index, row in enumerate(resources):
        elapsed = row["elapsed_s"]
        require(
            type(elapsed) in (int, float)
            and math.isfinite(elapsed)
            and last <= elapsed <= 900
            and all(
                type(row[key]) is int and row[key] >= 0
                for key in (
                    "tree_rss_bytes",
                    "ram_available_bytes",
                    "disk_free_bytes",
                    "archive_free_disk_bytes",
                )
            )
            and row["violations"] == []
            and row["tree_rss_bytes"] <= 6 * GIB
            and row["ram_available_bytes"] >= (6 if index == 0 else 1.5) * GIB
            and min(row["disk_free_bytes"], row["archive_free_disk_bytes"])
            >= (3 if index == 0 else 1.5) * GIB,
            "Recorded resource sample violates the frozen execution procedure",
        )
        last = elapsed
    require(
        type(profile["duration_s"]) in (int, float)
        and last <= profile["duration_s"] <= 900
        and profile["minimum_available_ram_bytes"] == min(row["ram_available_bytes"] for row in resources)
        and profile["sampled_tree_rss_lower_bound_bytes"] == max(row["tree_rss_bytes"] for row in resources),
        "Resource profile extrema or duration disagree with the journal",
    )


def verify_batch(batch, rows, freeze_hash, script_hash):
    stage = TRIAL / f"extract-{batch:02d}"
    result_data = direct(stage / "output/result.json")
    profile_data = direct(stage / "profile.json")
    result, profile = json.loads(result_data), json.loads(profile_data)
    counts = result["actual_forwards"]
    require(
        result["status"] == "complete"
        and result["batch"] == batch
        and result["planned_cases"] == len(rows) == len(ARMS) * PAIRS_PER_BATCH
        and result["freeze_sha256"] == freeze_hash
        and counts.keys() == {"encoder", "current_classifier"}
        and all(type(value) is int and value >= 0 for value in counts.values())
        and result["new_neural_calls"] == profile["new_neural_calls"] == sum(counts.values()),
        "Frozen worker result, planned denominator or forward counts differ",
    )
    require(
        profile["schema"] == "chipmunk.video-appearance-resource-profile/108"
        and profile["command"] == "extract"
        and profile["batch"] == batch
        and profile["child_started"] is True
        and profile["child_exit_code"] == 0
        and profile["guard"] is None
        and profile["sample_errors"] == 0
        and profile["error_types"] == profile["resource_limit_observations"] == []
        and profile["runtime_sha256"] == script_hash
        and profile["worker_receipt_sha256"] == digest(result_data),
        "Exact successful worker resource profile required",
    )
    verify_resources(
        profile, json_lines(bound_file(stage / "resources.jsonl", profile["resource_journal_sha256"]))
    )
    responses = json_lines(bound_file(stage / "output/private-responses.jsonl", result["journal_sha256"]))
    require(
        [row["case"] for row in rows] == [row["case"] for row in responses],
        "A guarded batch omitted, duplicated or reordered planned cases",
    )
    return responses, counts, digest(profile_data)


def load_plan():
    frozen_data, intake_data = direct(TRANSFER_FREEZE), direct(INTAKE_FREEZE)
    frozen, intake = json.loads(frozen_data), json.loads(intake_data)
    freeze_hash = digest(frozen_data)
    require(
        direct(TRIAL / "freeze.json") == frozen_data
        and direct(INTAKE / "freeze.json") == intake_data
        and frozen["schema"] == "chipmunk.text-gohumanize-transfer-freeze/196"
        and (frozen["batches"], frozen["pairs_per_batch"], tuple(frozen["methods"]))
        == (BATCHES, PAIRS_PER_BATCH, METHODS)
        and intake["schema"] == "chipmunk.text-gohumanize-preparation-freeze/193"
        and tuple(intake["arms"]) == ARMS
        and intake["planned_pairs"] == BATCHES * PAIRS_PER_BATCH
        and intake["article_groups"] == ARTICLE_GROUPS,
        "Exact intake and transfer freezes required",
    )
    source = json.loads(bound_file(INTAKE_RESULT, frozen["preparation_receipt_sha256"]))
    require(
        source["schema"] == "chipmunk.text-gohumanize-preparation-result/193"
        and source["status"] == "complete"
        and source["freeze_sha256"] == digest(intake_data)
        and source["private_cases_sha256"] == frozen["private_cases_sha256"]
        and source["planned_pairs"] == BATCHES * PAIRS_PER_BATCH
        and source["article_groups"] == ARTICLE_GROUPS,
        "Preparation result differs from the frozen transfer source",
    )
    reference = json.loads(bound_file(REFERENCE, frozen["reference_sha256"]))
    require(
        reference["schema"] == "chipmunk.text-format-response-freeze/94",
        "Expected frozen v7 operating points",
    )
    for lock in (frozen, intake, reference):
        for relative, expected in lock["code_sha256"].items():
            path = ROOT / relative
            require(path.resolve().is_relative_to(ROOT), "Frozen code path escaped the repository")
            bound_file(path, expected)
    points = json.loads(
        bound_file(BASE / "writing-spans-v7/points.json", reference["prior_artifacts_sha256"]["points.json"])
    )
    cuts = {name: points[name]["cut"] for name in ("current", "interval")}
    require(
        all(type(value) in (int, float) and math.isfinite(value) and value >= 0 for value in cuts.values()),
        "Invalid frozen research points",
    )
    roster = json.loads(bound_file(INTAKE / "private-roster.json", intake["private_roster_sha256"]))
    rows = json_lines(bound_file(INTAKE / "preparation/private-cases.jsonl", frozen["private_cases_sha256"]))
    check_roster(rows, roster, source)
    return rows, cuts, frozen, freeze_hash


def review(expected):
    rows, cuts, frozen, freeze_hash = load_plan()
    responses, profiles, totals = [], [], Counter()
    width = PAIRS_PER_BATCH * len(ARMS)
    script_hash = frozen["code_sha256"]["backend/scripts/text_gohumanize_transfer_v196.py"]
    for batch in range(BATCHES):
        batch_responses, counts, profile_hash = verify_batch(
            batch, rows[batch * width : (batch + 1) * width], freeze_hash, script_hash
        )
        responses.extend(batch_responses)
        profiles.append(profile_hash)
        totals.update(counts)
    replay = {
        "schema": "chipmunk.text-gohumanize-transfer-readout/196",
        "status": "complete",
        "freeze_sha256": freeze_hash,
        "actual_forwards": dict(totals),
        "profile_sha256": profiles,
        "source_pairs": BATCHES * PAIRS_PER_BATCH,
        "article_groups": ARTICLE_GROUPS,
        "scope": frozen["scope"],
        "results": aggregate(rows, responses, cuts),
    }
    expected_data = direct(expected)
    require(
        exact_json(replay) == exact_json(json.loads(expected_data)),
        "Independent replay differs from the published v196 readout",
    )
    return {
        "schema": "chipmunk.text-gohumanize-transfer-review/198",
        "status": "complete",
        "published_readout_sha256": digest(expected_data),
        "transfer_freeze_sha256": freeze_hash,
        "verified_batch_count": len(profiles),
        "verified_case_count": len(rows),
        "verified_source_pairs": BATCHES * PAIRS_PER_BATCH,
        "verified_article_groups": ARTICLE_GROUPS,
        "methods": list(METHODS),
        "exact_readout_match": True,
        "new_model_calls": 0,
        "source_text_published": False,
        "scope": "Independent Unicode and arithmetic replay over committed model responses; no tokenizer or model rerun.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected", type=Path, default=EXPECTED, help="Existing public v196 aggregate JSON")
    args = parser.parse_args()
    print(json.dumps(review(args.expected), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
