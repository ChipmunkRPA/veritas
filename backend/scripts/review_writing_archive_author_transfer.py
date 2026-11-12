"""Replay archived writing inputs, exact offsets and decisions without another fit."""

from __future__ import annotations

import argparse
import hashlib
import math
from collections import Counter
from pathlib import Path

from app.detectors.text.extract import from_string
from app.detectors.text.segment import segment
from scripts import writing_archive_author_intake as archive
from scripts.video_appearance_native_v108 import read, require, write
from scripts.writing_archive_author_transfer import ARMS, PEP_SCORES, POINT
from scripts.writing_blind_rewrite_probe import valid_sentence, words
from scripts.writing_focus_v3 import digest

FAMILIES = ("qwen", "smol2")


def sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def source_proof(row, repository):
    provenance = row["source"]
    if "commit" in provenance:
        commits = [provenance["commit"]] * 2
        blobs = [provenance["blob_sha256"]] * 2
        notices = [provenance["license_sha256"]] * 2
        require(row["recorded_authorship"] == "same_bylined_work_matched_passages"
                and row["before"]["raw_span"][1] < row["after"]["raw_span"][0], "Control passages overlap")
    else:
        commits = [provenance[key] for key in ("before", "after")]
        blobs, notices = [provenance[key] for key in ("blob_sha256", "license_sha256")]
        require(row["recorded_authorship"] == "same_recorded_contributor" and archive.git(
            repository, "show", "-s", "--format=%P", commits[1]).strip() == commits[0], "Revision is not a Git parent pair")
        proofs = [archive.blame(repository, commit, provenance["path"], row[key]["witness"][2]["lines"])
                  for commit, key in zip(commits, ("before", "after"), strict=True)]
        require(proofs == provenance["focus_line_attribution"] and any(record["commit"] == commits[1] for record in proofs[1])
                and all(not record["boundary"] and (record["identity"][0] == provenance["editor"][0]
                                                   or record["identity"][1] == provenance["editor"][1])
                        for proof in proofs for record in proof), "Same-contributor claim differs")
    bindings = 0
    for label, commit, blob, notice in zip(("before", "after"), commits, blobs, notices, strict=True):
        raw = archive.git(repository, "show", f"{commit}:{provenance['path']}")
        rights = archive.git(repository, "show", f"{commit}:LICENSE")
        require(sha(raw) == blob and sha(rights) == notice and rights.startswith("Attribution 4.0 International"),
                "Source blob or license differs")
        if "byline_span" in provenance:
            lo, hi = provenance["byline_span"]
            require(sha(raw[lo:hi]) == provenance["byline_sha256"], "Author witness differs")
        control = row[label]
        witness = control["witness"]
        require(control["text"] == " ".join(item["text"] for item in witness) and len(witness) == 5,
                "Source witness does not reconstruct the fixture")
        for item in witness:
            lo, hi = item["raw_span"]
            lines = [raw.count("\n", 0, lo) + 1, raw.count("\n", 0, hi - 1) + 1]
            require(0 <= lo < hi <= len(raw) and sha(raw[lo:hi]) == item["raw_sha256"] and lines == item["lines"],
                    "Raw archive witness span or line range differs")
            bindings += 1
    return bindings


def permitted_variants(row, generators):
    first, second = row["before"], row["after"]
    result = {"before": first["text"], "after": second["text"]}
    origin = first["focus_text"]
    begin, end = first["focus"]
    units = [entry.text for entry in segment(from_string(first["text"]))]
    require(units[2] == origin and first["text"][begin:end] == origin, "Invalid generator source")
    for family, packet in generators.items():
        attempt = packet["results"][row["group"]]
        require(attempt["input_sha256"] == sha(origin), "Generation input differs")
        rewritten = valid_sentence(attempt, origin, generated=True)
        if rewritten is None or rewritten == second["focus_text"]:
            continue
        changed = first["text"][:begin] + rewritten + first["text"][end:]
        doc = from_string(changed)
        expected = [*units[:2], rewritten, *units[3:]]
        if doc.text == changed and [entry.text for entry in segment(doc)] == expected:
            result[family] = changed
    return result


def positions(ranges, text):
    return {index for lo, hi in ranges for index in range(lo, hi) if not text[index].isspace()}


def assess(text, locked, outputs, probabilities, input_seen):
    doc = from_string(text)
    units = segment(doc)
    require(doc.text == text and sha(text) == locked["text_sha256"] == outputs["text_sha256"]
            and len(units) == 5, "Frozen fixture changed")
    windows, scores, focus = [], [], [(unit.start, unit.end) for unit in units]
    require(list(focus[2]) == locked["focus"] == outputs["focus"], "Expected known changed center differs")
    for i, raw in enumerate(locked["windows"]):
        indexes = range(max(0, i - 1), min(5, i + 2))
        spans = [list(focus[j]) for j in indexes]
        exact = sha(" ".join(text[a:b] for a, b in spans))
        require(raw == {"spans": spans, "sha256": exact, "focus": list(focus[i])}, "Exact input windows differ")
        input_seen.add(exact)
        row = probabilities[exact]
        value, cov = row["score"], row["coverage"]
        require(type(value) is float and math.isfinite(value) and 0 <= value <= 1
                and cov["token_windows"] == 1 and 0 < cov["tokens"] <= 512 and cov.get("omitted_tokens", 0) == 0
                and cov.get("covered_tokens", cov["tokens"]) == cov["tokens"], "Incomplete native score receipt")
        windows.append(spans)
        scores.append(value)
    eligible = [i for i in (1, 2, 3) if 12 <= len(words(units[i].text)) <= 40]
    require(eligible == locked["centers"], "Sentence eligibility differs")
    current = [window for window, p in zip(windows, scores, strict=True) if p >= 0.5][:6]
    clipped = {i: min(1 - 1e-7, max(1e-7, scores[i])) for i in eligible}
    highest = max(clipped.values(), default=-1)
    winners = [i for i, value in clipped.items() if value == highest]
    ranked = len(winners) == 1 and winners[0] == 2
    approved = len(winners) == 1 and math.log(highest / (1 - highest)) > POINT
    matched = [[list(focus[winners[0]])]] if approved else []
    require(outputs["ranked_exact_center_first"] is ranked, "Top ranked location differs")
    truth = positions([focus[2]], text)
    for arm, selections in zip(ARMS, (current, matched, current + matched), strict=True):
        passes = False
        for spans in selections:
            guess = positions(spans, text)
            intersection = len(guess.intersection(truth))
            passes |= intersection / len(guess.union(truth)) >= .30 and intersection / len(truth) >= .50
        require(outputs["arms"][arm] == {"flag": bool(selections), "useful": passes, "spans": selections},
                "Independent localization decision differs")
    return len(windows)


def grouped_results(results, aggregate):
    generated = [row for row in results if row["role"] in FAMILIES]
    genuine = [row for row in results if row["role"] in {"before", "after"}]
    positives = {arm: {row["group"] for row in generated if row["arms"][arm]["useful"]} for arm in ARMS}
    false_groups = {arm: {row["group"] for row in genuine if row["arms"][arm]["flag"]} for arm in ARMS}
    grouped = {"groups_with_admitted_generation": len({row["group"] for row in generated})}
    for arm in ARMS:
        gains = {family: {row["group"] for row in generated if row["role"] == family and row["arms"][arm]["useful"]
                          and not row["arms"]["current"]["useful"]} for family in FAMILIES}
        lost = {row["group"] for row in generated if row["arms"]["current"]["useful"] and not row["arms"][arm]["useful"]}
        real_added = {row["group"] for row in genuine if row["arms"][arm]["flag"] and not row["arms"]["current"]["flag"]}
        later_added = {row["group"] for row in genuine if row["role"] == "after" and row["arms"][arm]["flag"]
                       and not row["arms"]["current"]["flag"]}
        n_gains = len(gains["qwen"] | gains["smol2"])
        grouped[arm] = {"any_useful_generated_groups": len(positives[arm]),
                        "any_false_flag_genuine_groups": len(false_groups[arm]),
                        "new_false_flag_groups_at_either_real_passage": len(real_added),
                        "new_false_flag_groups_relative_to_any_incumbent_real_flag": len(false_groups[arm] - false_groups["current"]),
                        "new_useful_groups": n_gains, "lost_useful_groups": len(lost),
                        "useful_gain_groups_by_generator": {family: len(gain) for family, gain in gains.items()}}
        if arm != "current":
            decision = aggregate["decision"][arm]
            advance = (all(aggregate["admission"][family].get("admitted", 0) >= 6 and gains[family] for family in FAMILIES)
                       and n_gains >= 3 and not lost and len(later_added) <= 1)
            require(decision == {"new_useful_groups": n_gains, "lost_useful_groups": len(lost),
                                  "new_later_false_flag_groups": len(later_added),
                                  "new_useful_by_generator": grouped[arm]["useful_gain_groups_by_generator"],
                                  "advance": advance}, "Frozen group decision differs")
    return grouped


def audit(run, historical):
    paths = {name: run / name for name in ("source-freeze.json", "source-receipt.json", "originals.json",
                                           "freeze.json", "input-lock.json", "responses.json", "outcomes.json", "aggregate.json")}
    freeze, inputs, model = (read(paths[name]) for name in ("freeze.json", "input-lock.json", "responses.json"))
    require(freeze["source_sha256"] == digest(paths["originals.json"])
            and freeze["source_receipt_sha256"] == digest(paths["source-receipt.json"])
            and inputs["freeze_sha256"] == digest(paths["freeze.json"])
            and model["binding"] == digest(paths["input-lock.json"]), "Private experiment bindings differ")
    samples = read(paths["originals.json"])
    results = read(paths["outcomes.json"])
    raw = model["responses"]
    require(model["forwards"] == (len(raw) + 3) // 4, "Model forward count differs")
    generators = {}
    for family in FAMILIES:
        path = run / f"generated-{family}.json"
        value = read(path)
        require(inputs["generation_sha256"][family] == digest(path)
                and value["freeze_sha256"] == digest(paths["freeze.json"])
                and set(value["results"]) == {row["group"] for row in samples}, "Generator packet differs")
        generators[family] = value
    fixture = {(row["group"], row["role"]): row for row in inputs["rows"]}
    emitted = {(row["group"], row["role"]): row for row in results}
    require(len(fixture) == len(inputs["rows"]) == len(results) == len(emitted) and fixture.keys() == emitted.keys(),
            "Fixture inventory differs")
    corpus, observed, raw_links, input_links = set(), set(), 0, 0
    repository = historical / archive.SOURCES["kubernetes"]["archive"]
    for row in samples:
        raw_links += source_proof(row, repository)
        for role, text in permitted_variants(row, generators).items():
            key = row["group"], role
            require(key in fixture and key not in corpus, "Eligible fixture missing or repeated")
            require(all(emitted[key][field] == fixture[key][field] for field in ("archive", "genre", "recorded_authorship")),
                    "Source outcome stratum differs")
            corpus.add(key)
            input_links += assess(text, fixture[key], emitted[key], raw, observed)
    require(corpus == fixture.keys() and observed == raw.keys() == set(inputs["input_hashes"])
            and not observed.intersection(read(PEP_SCORES / "calibration.json")["responses"]), "Held input coverage differs")
    aggregate = read(paths["aggregate.json"])
    strata = {"overall": results}
    for name in aggregate["strata"]:
        strata[name] = [row for row in results if "/".join(
            row[field] for field in ("archive", "genre", "recorded_authorship")) == name]
    require(sum(map(len, strata.values())) == 2 * len(results), "Reported source strata differ")
    for key, group in strata.items():
        stats = aggregate["overall"] if key == "overall" else aggregate["strata"][key]
        for role in ("before", "after", *FAMILIES):
            items = [row for row in group if row["role"] == role]
            reported = stats[role]
            require(reported["files"] == len(items) and reported["ranked_exact_center_first"] == sum(
                row["ranked_exact_center_first"] for row in items), "Per-role total differs")
            for arm in ARMS:
                counts = Counter("useful" if row["arms"][arm]["useful"] else
                                 "wrong_only" if row["arms"][arm]["flag"] else "no_flag" for row in items)
                require(reported[arm] == dict(counts) | {"flag": len(items) - counts["no_flag"]}, "Aggregate counts differ")
    return {"source_groups": len(samples), "archive_span_checks": raw_links, "held_fixtures": len(corpus),
            "exact_context_span_bindings": input_links, "distinct_native_inputs": len(observed),
            "score_forwards": model["forwards"], "prior_fit_input_overlap": 0,
            "grouped_results": grouped_results(results, aggregate),
            "commitments": {name: digest(path) for name, path in paths.items()}, "verified": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("historical", type=Path)
    args = parser.parse_args()
    output = audit(args.run, args.historical)
    write(args.run / "audit.json", output)
    print(output, flush=True)


if __name__ == "__main__":
    main()
