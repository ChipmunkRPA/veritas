"""Verify exact percentage disagreements on source-held GAO highlights and ordinary writing."""

from __future__ import annotations

import argparse
import re
from collections import Counter
from dataclasses import asdict
from decimal import Decimal
from pathlib import Path

from app.core.canonical import canonical_hash
from app.detectors.text.extract import from_string
from app.detectors.text.segment import segment
from app.evidence import document_percentage_consistency as check
from scripts import writing_revision_trajectory_transfer as source

SCHEMA = "chipmunk.writing-percentage-consistency-transfer/1"
PACK = "671287085feaa971c501be2e2db7d50a36ca2a9adba0642c9540f4add35b341e"
INDEX = source.ROOT / "docs/evaluation-panels"
AUTHORED = {
    "changed_value", "changed_value_numeral", "exact_repeat", "equivalent_numeral", "different_year", "different_cohort",
    "different_denominator", "different_unit", "different_citation", "quotation", "approximation", "range", "same_paragraph", "natural",
}
POSITIVE = {"changed_value", "changed_value_numeral"}
ARCHIVE_FILES = (
    "writing-gao-archive-transfer", "writing-gao-exact-excerpt", "writing-gao-word-paragraphs", "writing-gao-word-boundary-deletions",
)
ROSTERS = {"short_a": ("217eb06cc145d060cfdb1242d28219d1a0a9817a3278744bb57bdcc45a739ee9", 166),
           "short_b": ("dfb6a52072e32af9acf75b42f84b671850d402a0ba534f414e4d8fa31aa6ad58", 60),
           "long": ("417edacc01afbc6248f1065f7803f900f9e01d8fcebbb1cc19eeff5883347429", 140),
           "f1000": ("9b07d5ee0afa19051f11063e5e1615a72222eeae4042661df3b653d42145e44c", 36),
           "elife": ("597573e182a68e0edf5800a5c470fee23aa5bc79592b514f6a8c87811258a61c", 45)}
METHOD = {
    "sample": "24 previously unused GAO test reports in salted source order, with an eligible percentage sentence in a complete highlight, and 128 additional unused reports with no sentence filtering.",
    "evidence": "Require complete sentences in separate source blocks, identical normalized wording, punctuation, unit, qualifiers, citations and numeric values except one percentage numeral. Both exact Unicode locations must match the annotated changed delivery when triggered.",
    "controls": "Unmodified source highlights, unchanged/equivalent repeated values, different year, cohort, denominator, unit and citation, quotation, approximation, ranges and repetitions within a single paragraph. Also recount archived genuine journal, forum, USWDS versions, and genuine originals matching AI-edit banks.",
    "evaluation": "All scenarios and source rules fixed before test. Apply the same candidate as for local serving. For deployment, at least 44 of 48 authored differences must give both exact values, with no findings on constructed confounds and no unexplained control flags. Do not use these observations for AI-authorship confidence.",
}


def measure(text):
    doc = from_string(text)
    commitment = {"method": "plain", "chars": len(doc.text), "extracted_text_sha256": source.sha(doc.text),
                  "source_map_sha256": canonical_hash(doc.source_map)}
    sentences = [asdict(row) for row in segment(doc)]
    found = check.build(commitment, doc.text, doc.source_map, sentences)
    check.verify_source({"source": commitment, check.KEY: found}, doc.text, doc.source_map, sentences)
    groups = [[[r["quantity_start"], r["quantity_end"]] for r in group] for group in found["groups"]]
    return {"text_sha256": commitment["extracted_text_sha256"], "characters": len(doc.text), "positions": groups,
            "checked_sentences": found["checked_sentences"], "matches": found["different_claims"]}, doc.text


def select(pack):
    source.require(source.digest(pack) == PACK, "Pinned GAO pack differs")
    archive = source.read(pack)["reports"]
    excluded = {name for title in ARCHIVE_FILES for name in re.findall(r"GAO-\d+-\d+(?:-SP|-T|-R)?",
                (INDEX / f"{title}-2026-11-13.json").read_text().upper())}
    ordered = [name for name in sorted(archive, key=lambda n: source.sha(f"{SCHEMA}/{n}"))
               if name not in excluded and archive[name]["split"] == "test"]
    selected = []
    for name in ordered:
        text = archive[name]["text"]
        doc = from_string(text)
        matching = [(line.text, hit) for line in segment(doc) if (hit := check.percentage(line.text)) is not None
                    and hit[2] <= 100 and text.count(line.text) == 1]
        if len(text) <= 5900 and matching:
            selected.append((name, text, *matching[0]))
        if len(selected) >= 24:
            break
    random_controls = [(name, archive[name]["text"]) for name in ordered if name not in {r[0] for r in selected}][:128]
    source.require(len(selected) == 24 and len(random_controls) == 128, "Unexpected GAO source coverage")
    return selected, random_controls


def fixtures(text, claim, hit):
    _, (a, b), value = hit
    digits = claim[a:b]
    number = str(value + (Decimal(1) if value < 99 else Decimal(-1)))
    fractional = format(value, "f")
    equivalent = f"{fractional}0" if "." in fractional else f"{fractional}.0"
    changed = claim[:a] + number + claim[b:]
    different_unit = re.sub(r"^(?:%|percent(?:age points?)?)", " basis points", claim[b:].lstrip(), count=1, flags=re.I)
    year = re.search(r"\b(?:19|20)\d{2}\b", claim)
    next_year = (claim[:year.start()] + str(int(year.group()) + 1) + claim[year.end():] if year
                 else claim[:-1] + " during fiscal year 2015.")
    # The value differs for each confound. Exact framing must prevent a match.
    def with_changed(amount_in_context):
        return amount_in_context.replace(digits + claim[b:b+4], number + claim[b:b+4], 1)

    yield "changed_value", text + "\n\n" + changed, (claim, changed)
    fractional_changed = claim[:a] + number + ("0" if "." in number else ".0") + claim[b:]
    yield "changed_value_numeral", text + "\n\n" + fractional_changed, (claim, fractional_changed)
    yield "exact_repeat", text + "\n\n" + claim, None
    yield "equivalent_numeral", text + "\n\n" + claim[:a] + equivalent + claim[b:], None
    yield "different_year", text + "\n\n" + with_changed(next_year), None
    yield "different_cohort", text + "\n\n" + changed[:-1] + " among the rural cohort.", None
    yield "different_denominator", text + "\n\n" + changed[:-1] + " as a share of the applications only.", None
    yield "different_unit", text + "\n\n" + claim[:a] + number + different_unit, None
    yield "different_citation", text + "\n\n" + changed[:-1] + " (see appendix B).", None
    yield "quotation", text + "\n\n> " + changed, None
    yield "approximation", text + "\n\n" + claim[:a] + "approximately " + number + claim[b:], None
    yield "range", text + "\n\n" + claim[:a] + digits + "–" + number + claim[b:], None
    yield "same_paragraph", text + " " + changed, None
    yield "natural", text, None


def expected_positions(text, exact):
    if exact is None:
        return []
    output = []
    for sentence in exact:
        source.require(text.count(sentence) == 1, "Repeated authored sentence unexpectedly collides")
        offset = text.index(sentence)
        _, positions, _ = check.percentage(sentence)
        output.append([offset + positions[0], offset + positions[1]])
    return [output]


def public_cases(pack):
    positive, ordinary = select(pack)
    result = []
    for name, text, claim, hit in positive:
        for role, material, truth in fixtures(text, claim, hit):
            found, actual = measure(material)
            result.append({"source": name, "role": role, "expected": expected_positions(actual, truth), **found})
    for name, text in ordinary:
        found, _ = measure(text)
        result.append({"source": name, "role": "untouched_sample", "expected": [], **found})
    return result


def prior_cases(paths):
    output = []
    for cohort, path in paths.items():
        expected, count = ROSTERS[cohort]
        source.require(source.digest(path) == expected, f"Pinned native file differs: {cohort}")
        rows = source.read(path)
        if cohort == "short_a":
            rows = [r for r in rows if r["cohort"] != "fit"]
        elif cohort == "short_b":
            rows = [r for r in rows if r["cohort"] == "report"]
        source.require(len(rows) == count, f"Assessment case count differs: {cohort}")
        for row in rows:
            names = {"gemini-2.5-flash", "gpt-5.4-nano", "granite", "olmo", "qwen"}
            variant = row.get("family", row.get("variant"))
            found, _ = measure(row["text"])
            output.append({"source": str(row["group"]), "role": "known_ai_edit" if variant in names else "ordinary_revision",
                           "panel": cohort + "/" + row.get("cohort", "articles"), "expected": [], **found})
    return output


def summarize(rows):
    counts = {}
    for row in rows:
        role = row["role"]
        val = counts.setdefault(role, Counter())
        val.update(files=1, eligible=bool(row["checked_sentences"]), flagged=bool(row["matches"]),
                   useful=bool(row["expected"] and row["positions"] == row["expected"]))
        source.require(all(0 <= lo < hi <= row["characters"] for group in row["positions"] for lo, hi in group), "Invalid source positions")
    source.require({k: counts[k]["files"] for k in sorted(AUTHORED)} == dict.fromkeys(AUTHORED, 24)
                   and counts["untouched_sample"]["files"] == 128 and counts["known_ai_edit"]["files"] == 255
                   and counts["ordinary_revision"]["files"] == 192,
                   "Incorrect test roster")
    eligible = (sum(counts[k]["useful"] for k in POSITIVE) >= 44
                and not any(v["flagged"] for k, v in counts.items() if k not in POSITIVE | {"known_ai_edit"}))
    return {"counts": counts, "narrow_opt_in_candidate": eligible}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("pack", type=Path)
    cli.add_argument("output", type=Path)
    for panel in ROSTERS:
        cli.add_argument(f"--{panel.replace('_', '-')}", type=Path, required=True)
    args = cli.parse_args()
    paths = {k: getattr(args, k) for k in ROSTERS}
    rows = public_cases(args.pack) + prior_cases(paths)
    result = {"schema": SCHEMA, "method": METHOD, "implementation_sha256": source.digest(Path(check.__file__)),
              "pack_sha256": PACK, "results": summarize(rows), "rows": rows}
    source.seal(args.output, result)
    print(source.serialize(result["results"]), end="")


if __name__ == "__main__":
    main()
