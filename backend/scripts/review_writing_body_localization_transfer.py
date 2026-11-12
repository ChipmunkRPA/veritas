"""Independently reassemble writing body, exact model inputs and evidence selections."""

from __future__ import annotations

import argparse
import hashlib
import math
from collections import Counter
from pathlib import Path

from app.detectors.text.extract import extract
from app.detectors.text.segment import segment
from scripts import writing_archive_author_intake as archive
from scripts import writing_archive_prospective_intake as prose
from scripts import writing_body_localization_transfer as study
from scripts.video_appearance_native_v108 import read, require, write
from scripts.writing_blind_rewrite_probe import valid_sentence, words
from scripts.writing_focus_v3 import digest

FAMILIES = ("qwen", "smol2")


def sha(value):
    return hashlib.sha256(value.encode()).hexdigest()


def verify_source(row, repository):
    raw = archive.git(repository, "show", f"{row['commit']}:{row['path']}")
    require(sha(raw) == row["archive_blob_sha256"] and archive.rights_record(
        repository, row["commit"], archive.SOURCES["kubernetes"]) is not None, "Archive text or source rights differ")
    position, start, stop = row["focus"]
    bounds = [part["source_span"] for part in row["paragraphs"]]
    require(len(bounds) >= 4 and all(bounds[i - 1][1] < bounds[i][0] for i in range(1, len(bounds))),
            "Missing or unordered body blocks")
    for index, part in enumerate(row["paragraphs"]):
        lo, hi = part["source_span"]
        normalized, mapping = prose.visible(raw, lo, hi)
        require(0 <= lo < hi <= len(raw) and normalized == part["text"]
                and sha(raw[lo:hi]) == part["source_sha256"], "Archive paragraph mapping differs")
        if index == position:
            require([mapping[start][0], mapping[stop - 1][1]] == row["target_raw_span"]
                    and normalized[start:stop] == row["before"]["focus_text"], "Archive edit source differs")
    require(sha(row["quote"]) == row["quote_sha256"], "Genuine quote differs")
    return len(bounds)


def measurement(row):
    value = row["score"]
    cov = row["coverage"]
    require(type(value) is float and math.isfinite(value) and 0 <= value <= 1
            and cov["token_windows"] == 1 and 0 < cov["tokens"] <= 512
            and not cov.get("omitted_tokens", 0) and cov.get("covered_tokens", cov["tokens"]) == cov["tokens"],
            "Model input lacks complete coverage")
    return value


def logodds(value):
    p = min(max(value, 1e-7), 1 - 1e-7)
    return math.log(p / (1 - p))


def ordinal_median(numbers):
    points = sorted(numbers)
    center = len(points) // 2
    return points[center] if len(points) % 2 else (points[center - 1] + points[center]) / 2


def positions(text, spans):
    marked = set()
    for start, stop in spans:
        require(0 <= start < stop <= len(text), "Displayed span outside decoded text")
        marked.update(i for i in range(start, stop) if not text[i].isspace())
    return marked


def verify_fixture(run, row, source, replacement, expected_selection, output, first, second, inventory):
    location = run / "fixtures" / row["file"]
    doc = extract(location, "." + row["format"])
    require(digest(location) == row["file_sha256"] == output["file_sha256"] and sha(doc.text) == row["text_sha256"],
            "Fixture file, decoded text or outcome differs")
    units = segment(doc)
    roles = [part.get("quote_role", "body") for part in doc.source_map]
    require(roles == row["roles"] == output["roles"] and [list(bounds) for bounds in doc.paragraphs] == row["paragraphs"],
            "Paragraph role or spans differ")
    body = [part["text"] for part in source["paragraphs"]]
    para, lo, hi = source["focus"]
    truth = []
    if replacement:
        body[para] = body[para][:lo] + replacement + body[para][hi:]
        offset = sum(len(part) + 2 for part in body[:para]) + lo
        truth = [offset, offset + len(replacement)]
    require(row["truth"] == output["truth"] == (truth or None), "Exact edited sentence range differs")
    real_body = [doc.text[a:b] for i, (a, b) in enumerate(doc.paragraphs) if roles[i] == "body"]
    require(real_body == body, "Article body was modified outside the exact known edit")
    quote = [doc.text[a:b] for i, (a, b) in enumerate(doc.paragraphs) if roles[i] == "quoted"]
    require(quote == (["> " + source["quote"]] if row["format"] == "eml" else []), "Genuine quote mapping differs")
    if row["format"] == "eml":
        metadata = doc.meta["email"]
        require(metadata["body_content_type"] == "text/plain" and len(metadata["attachments"]) == 1
                and all(part["mime_part"] == metadata["body_part"] and part.get("source_ranges") for part in doc.source_map),
                "Decoded MIME source ranges or role binding differs")

    scores, eligible, spans, single = [], [], [], []
    for i, unit in enumerate(units):
        bounds = [[part.start, part.end] for part in units[max(0, i - 1):i + 2]]
        key = sha(" ".join(doc.text[a:b] for a, b in bounds))
        exact = sha(doc.text[unit.start:unit.end])
        native = {"sha256": key, "spans": bounds, "paragraph": unit.paragraph, "focus": [unit.start, unit.end]}
        require(row["windows"][i] == native and row["isolated"][i] == exact, "Native window or single-sentence input differs")
        scores.append(measurement(first[key]))
        spans.append(bounds)
        single.append(exact)
        inventory[0].add(key)
        if (0 < i < len(units) - 1 and roles[unit.paragraph] == "body"
                and unit.paragraph == units[i - 1].paragraph == units[i + 1].paragraph
                and 12 <= len(words(unit.text)) <= 40
                and 6 <= len(words(units[i - 1].text)) <= 65 and 6 <= len(words(units[i + 1].text)) <= 65):
            eligible.append(i)
    require(eligible == row["eligible"] and len(units) == len(row["windows"]), "Exact focus coverage differs")
    current = [spans[i] for i, value in enumerate(scores) if value >= .50][:6]
    options = {}
    for i in eligible:
        key_group = single[i - 1:i + 2]
        inventory[0].update(key_group)
        native = logodds(scores[i])
        comparison = ordinal_median([logodds(scores[j]) for j in eligible if units[j].paragraph == units[i].paragraph])
        a, b, c = map(lambda key: logodds(measurement(first[key])), key_group)
        delta = min(b - a, b - c)
        if delta > 0 and native > study.POINT and native >= comparison:
            options[i] = delta
    largest = max(options.values(), default=0)
    winners = [i for i, value in options.items() if value == largest]
    winner = winners[0] if len(winners) == 1 else None
    require(winner == expected_selection, "Independent shortlisting disagrees")
    proposed, confirmed = [], []
    if winner is not None:
        key = single[winner]
        inventory[1].add(key)
        proposed = [[[units[winner].start, units[winner].end]]]
        if measurement(second[key]) >= .50:
            confirmed = proposed
    seen = [[set(range(a, b)), roles[i]] for i, (a, b) in enumerate(doc.paragraphs)]
    require(output["body_paragraphs"] == len(body)
            and output["eligible_body_paragraphs"] == len({units[i].paragraph for i in eligible})
            and output["target_eligible"] is any([units[i].start, units[i].end] == truth for i in eligible),
            "Output coverage flags differ")
    target = positions(doc.text, [truth]) if truth else set()
    for arm, regions in zip(study.ARMS, (current, proposed, confirmed, current + confirmed), strict=True):
        hits = [positions(doc.text, ranges) for ranges in regions]
        merged = set().union(*hits)
        useful = any(len(target & marked) * 10 >= 3 * len(target | marked)
                     and len(target & marked) * 2 >= len(target) for marked in hits) if target else False
        expect = {"flag": bool(regions), "useful": useful, "marked": regions,
                  "quote_flag": any(label == "quoted" and bool(chars & merged) for chars, label in seen),
                  "outside_target_paragraph": bool(target) and any(i != para and bool(chars & merged)
                                                                 for i, (chars, _) in enumerate(seen))}
        require(expect == output["arms"][arm], "Independent result or misplaced highlight differs")
    return len(units)


def independent_summary(outcomes, aggregate):
    groups = {}
    for row in outcomes:
        scope = row["scope"]
        for key in (scope, scope + "/" + row["format"], "/".join((scope, row["format"], row["kind"]))):
            groups.setdefault(key, []).append(row)
    require(groups.keys() == aggregate["counts"].keys(), "Published scope inventory differs")
    for key, rows in groups.items():
        controls = [row for row in rows if row["kind"] == "human"]
        mixed = [row for row in rows if row["kind"] != "human"]
        figures = aggregate["counts"][key]
        expected = {"human": len(controls), "mixed": len(mixed),
                    "target_eligible": len([row for row in mixed if row["target_eligible"]]),
                    "body_paragraphs": sum(row["body_paragraphs"] for row in rows),
                    "eligible_body_paragraphs": sum(row["eligible_body_paragraphs"] for row in rows)}
        require({key: figures[key] for key in expected} == expected, "Source coverage counts differ")
        for arm in study.ARMS:
            flags = [r for r in controls if r["arms"][arm]["flag"]]
            selected = [r for r in mixed if r["arms"][arm]["flag"]]
            found = [r for r in selected if r["arms"][arm]["useful"]]
            count = {"human_flagged": len(flags), "human_flagged_groups": len({r["group"] for r in flags}),
                     "useful": len(found), "wrong_only": len(selected) - len(found), "no_flag": len(mixed) - len(selected),
                     "outside_target_paragraph": len([r for r in mixed if r["arms"][arm]["outside_target_paragraph"]]),
                     "quote_flags": len([r for r in rows if r["arms"][arm]["quote_flag"]])}
            require(count == figures["arms"][arm], "Outcome group counts differ")
    fresh = groups["fresh"]
    flagged = {arm: {r["group"] for r in fresh if r["kind"] == "human" and r["arms"][arm]["flag"]}
               for arm in study.ARMS}
    grouped = {}
    for arm in study.ARMS:
        positive = {r["group"] for r in fresh if r["kind"] != "human" and r["arms"][arm]["useful"]}
        grouped[arm] = {"any_useful_positive_work_groups": len(positive), "any_flagged_genuine_work_groups": len(flagged[arm]),
                        "new_genuine_groups_relative_to_any_incumbent_format_flag": len(flagged[arm] - flagged["current"]),
                        "cleared_genuine_groups_across_both_formats": len(flagged["current"] - flagged[arm])}
    return grouped


def audit(run):
    names = ("freeze.json", "originals.json", "sources.json", "inputs.json", "primary.json",
             "shortlist.json", "independent.json", "outcomes.json", "aggregate.json")
    files = {name: run / name for name in names}
    commitments = {name: digest(path) for name, path in files.items()}
    lock, primary, shortlist, second = (read(files[name]) for name in ("inputs.json", "primary.json",
                                                                     "shortlist.json", "independent.json"))
    require(lock["freeze_sha256"] == commitments["freeze.json"]
            and primary["binding"] == shortlist["input_sha256"] == commitments["inputs.json"]
            and shortlist["primary_sha256"] == commitments["primary.json"]
            and second["binding"] == commitments["shortlist.json"], "Experiment hash chain differs")
    previous = read(study.PRIOR_PUBLIC)
    prior_inputs = study.PRIOR / "input-lock.json"
    require(digest(prior_inputs) == previous["audit"]["commitments"]["input-lock.json"], "Old fixture packet differs")
    prior_lock = read(prior_inputs)
    prior_response = study.PRIOR / "responses.json"
    require(digest(prior_response) == previous["audit"]["commitments"]["responses.json"], "Old measured input packet differs")
    batch = {}
    for family in FAMILIES:
        fresh = run / f"generated-{family}.json"
        old = study.PRIOR / f"generated-{family}.json"
        require(digest(fresh) == lock["generated_sha256"][family]
                and digest(old) == prior_lock["generation_sha256"][family], "Generation input provenance differs")
        batch["fresh", family] = read(fresh)["results"]
        batch["exposed_rewrite_transfer", family] = read(old)["results"]
    repository = study.ARCHIVE / archive.SOURCES["kubernetes"]["archive"]
    sources = {row["group"]: row for row in read(files["sources.json"])["works"]}
    old_sources = {row["group"] for row in read(study.PRIOR / "originals.json")}
    require(all((row["group"] not in old_sources) == (row["scope"] == "fresh") for row in sources.values()),
            "Source exposure strata differ")
    paragraphs = sum(verify_source(row, repository) for row in sources.values())
    outcomes = read(files["outcomes.json"])
    first_scores, second_scores = primary["responses"], second["responses"]
    by_scope = {"fresh": (set(), set()), "exposed_rewrite_transfer": (set(), set())}
    span_bindings = 0
    paired_formats = Counter()
    for row, selection, result in zip(lock["rows"], shortlist["selected"], outcomes, strict=True):
        require(all(row[key] == result[key] for key in ("group", "kind", "format", "scope")), "Outcome keys differ")
        item = sources[row["group"]]
        replacement = None
        if row["kind"] != "human":
            raw = batch[row["scope"], row["kind"]][row["group"]]
            source = item["before"]["focus_text"]
            require(raw["input_sha256"] == sha(source), "Generator source input differs")
            replacement = valid_sentence(raw, source, generated=True)
            require(replacement and sha(replacement) != item["known_genuine_alternate_sha256"], "Rewrite eligibility differs")
        span_bindings += verify_fixture(run, row, item, replacement, selection, result, first_scores, second_scores,
                                        by_scope[row["scope"]])
        paired_formats[row["group"], row["kind"]] += 1
    require(all(n == 2 for n in paired_formats.values()) and len(paired_formats) == len(outcomes) // 2,
            "Article and MIME formats are not paired")
    inventory = tuple(by_scope["fresh"][i] | by_scope["exposed_rewrite_transfer"][i] for i in (0, 1))
    require(inventory[0] == set(lock["inputs"]) == first_scores.keys()
            and inventory[1] == set(shortlist["inputs"]) == second_scores.keys(), "Distinct model input inventory differs")
    require(primary["forwards"] == (len(inventory[0]) + 3) // 4 and second["forwards"] == (len(inventory[1]) + 3) // 4,
            "Unexpected CPU forward counts")
    old_keys = read(prior_response)["responses"].keys()
    return {"source_groups": len(sources), "verified_source_paragraphs": paragraphs, "fixtures": len(outcomes),
            "exact_context_span_bindings": span_bindings, "distinct_primary_inputs": len(inventory[0]),
            "distinct_independent_inputs": len(inventory[1]), "primary_forwards": primary["forwards"],
            "independent_forwards": second["forwards"], "commitments": commitments,
            "primary_exact_context_matches_to_previous_excerpt_responses": {
                scope: len(both[0] & old_keys) for scope, both in by_scope.items()},
            "fresh_work_level_transitions": independent_summary(outcomes, read(files["aggregate.json"])),
            "verified": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    result = audit(args.run)
    write(args.run / "audit.json", result)
    print(result, flush=True)


if __name__ == "__main__":
    main()
