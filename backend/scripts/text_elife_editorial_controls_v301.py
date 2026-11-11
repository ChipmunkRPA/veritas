"""Admit fresh article groups and construct matched ordinary editorial controls."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from itertools import chain
from pathlib import Path
from xml.etree import ElementTree as ET

from scripts import text_elife_source_intake_v272 as july
from scripts import text_elife_transfer_intake_v286 as transfer
from scripts import text_elife_transfer_rewrites_v289 as generation
from scripts.video_appearance_corrected_v112 import scoped

NAME = "text-elife-editorial-controls-v301"
ROOT = transfer.ROOT
DATA = transfer.DATA.with_name(NAME)
ARCHIVES = tuple(
    f"https://elifesciences.org/archive/2022/{month}" for month in ("january", "february", "march", "april")
)
WORKS = (
    ("76246", "2022-01-25", "Juan C Sánchez-Arcila, Kirk DC Jensen"),
    ("76105", "2022-01-21", "Jason Laurich, Anna M O'Brien"),
    ("76320", "2022-01-20", "Xingyun Wang, Richard Naud"),
    ("76231", "2022-01-20", "Tomohiro Sawa, Takaaki Akaike"),
    ("77309", "2022-02-28", "Giovanna M Collu, Marek Mlodzik"),
    ("77180", "2022-02-25", "Toshiko Tanaka, Luigi Ferrucci"),
    ("77202", "2022-02-22", "Jian Liu, Siri Caspersen, Jean WH Yong"),
    ("77041", "2022-02-21", "Rebecca G Wells"),
    ("76801", "2022-02-17", "César Augusto Ruiz Agudelo, Ángela María Cortes Gómez"),
    ("76475", "2022-02-09", "Bruce Stillman"),
    ("76632", "2022-02-08", "Chinyere Kemet, Emily Hill, Hui Feng"),
    ("76818", "2022-02-07", "Young J Yoon, Gary J Bassell"),
    ("77751", "2022-03-28", "Elizabeth H Aitken, Stephen J Rogerson"),
    ("77978", "2022-03-24", "Tessa AC Garrud, Jonathan H Jaggar"),
    ("77749", "2022-03-18", "Na Yang, Payel Sen"),
    ("77550", "2022-03-17", "Gaowei Chen, Zijun Chen, Yingjie Zhu"),
    ("77544", "2022-03-15", "Alessandro Benedetto, Martina Poletti"),
    ("77539", "2022-03-08", "Carol L Wagner, Bruce W Hollis"),
    ("77349", "2022-03-08", "Yuuki Y Watanabe, Christian Rutz"),
    ("78693", "2022-04-28", "Brandon L Pierce"),
    ("78930", "2022-04-27", "Orit Nafcha, Uri Hertz"),
    ("78327", "2022-04-25", "Polytimi Frangou, William T Clarke"),
    ("78549", "2022-04-21", "Avner Schlessinger, Massimiliano Bonomi"),
    ("78373", "2022-04-11", "Martina M Yordanova, Pavel V Baranov"),
    ("78246", "2022-04-05", "Xiaodan Lin, Dong Ren"),
    ("78248", "2022-04-04", "Pierre A Coulombe"),
)
ROLES = ("development", "confirmation")
ARMS = ("original", "copyedit", "human_paragraph_splice")
SALT = "chipmunk-elife-v301-split-v1:"
MIN_GROUPS_PER_ROLE = 7
MIN_EDITORIAL_GROUPS_PER_ROLE = 4
REWRITE_RESULT = generation.PANEL / "text-elife-transfer-rewrites-complete-v289.json"
REWRITE_SHA = "d4e13615a3f080ed03d159a2c4219f2cc67e907559c7929db0fb6a6b78da8580"
SCORE_RESULT = generation.PANEL / "text-elife-relative-paragraph-v292-complete.json"
SCORE_SHA = "3c0ce5847abbf0d37812a574e8daef0b3d7e2c08973016e5d9d5f5e76d35c877"
COPYEDITS = {
    "in order to": "to",
    "for example": "for instance",
    "for instance": "for example",
    "in addition": "additionally",
    "however": "nevertheless",
    "therefore": "thus",
    "although": "though",
    "amongst": "among",
    "whilst": "while",
    "towards": "toward",
    "approximately": "about",
    "furthermore": "moreover",
    "moreover": "furthermore",
}
COPYEDIT_CONTEXT = {"however": r"(?=,)", "in addition": r"(?=,)", "approximately": r"(?= \d)"}
COPYEDIT_PATTERN = re.compile(
    r"(?<![\w'-])(?:"
    + "|".join(re.escape(word) + COPYEDIT_CONTEXT.get(word, "") for word in COPYEDITS)
    + r")(?![\w'-])",
    re.I,
)
require, sha, checked, json_bytes = transfer.require, transfer.sha, transfer.checked, july.json_bytes


def partitions(works):
    ids = sorted(work[0] for work in works)
    require(len(ids) == len(set(ids)) and len(ids) % 2 == 0, "Distinct, balanced source roster required")
    ranked = sorted(ids, key=lambda value: (sha(f"{SALT}10.7554/eLife.{value}".encode()), value))
    halfway = len(ranked) // 2
    return dict(zip(ROLES, (ranked[:halfway], ranked[halfway:]), strict=True))


def protocol():
    old = json.loads(checked(generation.INTAKE, generation.INTAKE_SHA))
    earlier = json.loads(checked(transfer.JULY, transfer.JULY_SHA))
    used = {row["source_id"] for record in (old, earlier) for row in record["works"]}
    require(
        len(WORKS) == 26
        and not used.intersection(work[0] for work in WORKS)
        and old["publisher_revision"] == earlier["publisher_revision"] == transfer.REVISION
        and sha(Path(july.__file__).read_bytes()) == earlier["code_sha256"],
        "Exact prior publisher corpus and independent source IDs required",
    )
    checked(REWRITE_RESULT, REWRITE_SHA)
    checked(SCORE_RESULT, SCORE_SHA)
    roles = {source: role for role, sources in partitions(WORKS).items() for source in sources}
    files = (
        Path(__file__),
        Path(july.__file__),
        Path(transfer.__file__),
        ROOT / f"docs/{NAME}-method.md",
        ROOT / "backend/tests/test_text_elife_editorial_controls_v301.py",
    )
    return {
        "schema": "chipmunk.text-elife-editorial-roster/301",
        "publisher_revision": transfer.REVISION,
        "publisher_archives": list(ARCHIVES),
        "known_exposure_receipts_sha256": {
            "v272": transfer.JULY_SHA,
            "v273": transfer.REWRITES_SHA,
            "v286": generation.INTAKE_SHA,
            "v289": REWRITE_SHA,
            "v292": SCORE_SHA,
        },
        "split_salt": SALT,
        "minimum_admitted_groups_per_role": MIN_GROUPS_PER_ROLE,
        "minimum_editorial_groups_per_role": MIN_EDITORIAL_GROUPS_PER_ROLE,
        "maximum_bytes_per_xml": july.MAX_XML,
        "control_arms": list(ARMS),
        "code_sha256": {str(path.relative_to(ROOT)): sha(path.read_bytes()) for path in files},
        "works": [
            {"source_id": source, "date": date, "authors": names.split(", "), "role": roles[source]}
            for source, date, names in WORKS
        ],
    }


def project_body(directory, record):
    raw = july.prior.text.read_file(directory / transfer.filename(record["source_id"]), july.MAX_XML)
    require(sha(raw) == record["xml_sha256"], "Exact admitted publisher bytes required")
    projection = list(july.body_paragraphs(ET.fromstring(raw).find("body")))
    require(
        sha(json_bytes(projection)) == record["body_projection_sha256"], "Publisher body projection changed"
    )
    return projection


def historical(earlier_directory, transfer_directory):
    authors, related, texts = transfer.earlier_july(earlier_directory)
    v273_hash, v273 = transfer.earlier_rewrites()
    v286 = json.loads(checked(generation.INTAKE, generation.INTAKE_SHA))
    known_bodies = []
    for row in v286["works"]:
        if row.get("body_projection_sha256"):
            known_bodies.append("\n\n".join(filter(None, project_body(transfer_directory, row))))
            related.update(row["related_research_dois"])
    authors.update(transfer.name_key(name) for _, _, names in transfer.WORKS for name in names.split(", "))
    require(len(known_bodies) == 34, "Exact licensed v286 publisher projections required")
    v289 = json.loads(checked(REWRITE_RESULT, REWRITE_SHA))
    raw = checked(generation.TRIAL / "private-cases.jsonl", v289["private_cases_sha256"])
    rows = list(july.prior.json_lines(raw))
    rewrites = list(
        july.prior.validated_texts(
            row for row in rows if row["arm"] == "paragraph_ai" and row["status"] == "constructed"
        )
    )
    require(len(rows) == 78 and len(rewrites) == 22, "Exact v289 and v292 writing inputs required")
    packets = (
        ("v272_publisher_bodies", transfer.JULY_SHA, len(texts), texts),
        ("v273_constructed_rewrites", v273_hash, len(v273), v273),
        ("v286_licensed_publisher_bodies", generation.INTAKE_SHA, len(known_bodies), known_bodies),
        ("v289_v292_constructed_rewrites", v289["private_cases_sha256"], len(rewrites), rewrites),
    )
    return authors, related, packets


def copyedit(paragraph):
    pieces, changes, cursor, size = [], [], 0, 0
    for match in COPYEDIT_PATTERN.finditer(paragraph):
        source = match[0]
        if source != source.lower() and source != source.capitalize():
            continue
        unchanged = paragraph[cursor : match.start()]
        replacement = COPYEDITS[source.lower()]
        if source[0].isupper():
            replacement = replacement.capitalize()
        size += len(unchanged)
        changes.append([size, size + len(replacement)])
        size += len(replacement)
        pieces.extend((unchanged, replacement))
        cursor = match.end()
        if len(changes) == 4:
            break
    return "".join((*pieces, paragraph[cursor:])), changes


def paragraph_offsets(text):
    offset, result = 0, []
    for paragraph in text.split("\n\n"):
        result.append([offset, offset + len(paragraph)])
        offset += len(paragraph) + 2
    return result


def splice_control(original, arm, replacement, changes, reason=None, donor_body_slot=None):
    start, end = original["edit_target_codepoints"]
    source = original["text"]
    edges = original["paragraph_codepoints"]
    require(
        [start, end] in edges[1:-1]
        and sha(source.encode()) == original["text_sha256"]
        and edges == paragraph_offsets(source),
        "Exact intact Unicode paragraphs and fixed interior target required",
    )
    row = {
        "case": f"{original['group']}-{arm}",
        "article_group": original["group"],
        "role": original["role"],
        "arm": arm,
        "source_target": [start, end],
        "source_sha256": original["text_sha256"],
        "donor_body_slot": donor_body_slot,
    }
    if reason is not None:
        return {
            **row,
            "status": "unavailable",
            "reason": reason,
            "text": None,
            "text_sha256": None,
            "paragraph_codepoints": None,
            "ordinary_spans": None,
            "ai_spans": None,
        }
    require(
        replacement.strip() == replacement
        and replacement
        and "\n" not in replacement
        and (replacement != source[start:end]) == bool(changes)
        and all(0 <= left < right <= len(replacement) for left, right in changes),
        "Expected one intact replacement paragraph and exact ordinary edit coordinates",
    )
    text = source[:start] + replacement + source[end:]
    return {
        **row,
        "status": "constructed",
        "reason": None,
        "text": text,
        "text_sha256": sha(text.encode()),
        "paragraph_codepoints": paragraph_offsets(text),
        "ordinary_spans": [[start + a, start + b] for a, b in changes],
        "ai_spans": [],
    }


def controls(original, projection):
    start, end = original["edit_target_codepoints"]
    first, stop = original["body_slots"]
    target = original["text"][start:end]
    require(
        "\n\n".join(projection[first:stop]) == original["text"],
        "Control donor and original must share the exact publisher body",
    )
    edited, changes = copyedit(target)
    target_words = len(target.split())
    donors = [
        (abs(len(part.split()) - target_words), index, part)
        for index, part in enumerate(projection)
        if part
        and not first <= index < stop
        and part not in original["text"].split("\n\n")
        and 24 <= len(part.split()) <= 120
        and 3 * target_words <= 4 * len(part.split()) <= 5 * target_words
    ]
    _, index, donor = min(donors, default=(None, None, None))
    return [
        splice_control(original, "original", target, []),
        splice_control(original, "copyedit", edited, changes, None if changes else "no_fixed_copyedit"),
        splice_control(
            original,
            "human_paragraph_splice",
            donor,
            [[0, len(donor)]] if donor else [],
            None if donor else "no_length_matched_source_donor",
            index,
        ),
    ]


def run(lock, directory, earlier_directory, transfer_directory, research_cache, gohumanize):
    require(lock == protocol(), "Use the exact pre-acquisition source roster")
    roles = {row["source_id"]: row["role"] for row in lock["works"]}
    authors, related, history = historical(earlier_directory, transfer_directory)
    inspect = scoped(july.inspect, {"metadata": transfer.metadata})
    inspected = [inspect(directory, work, transfer.REVISION) for work in WORKS]
    prepared, fingerprints = {}, {}
    for record, original, signatures in inspected:
        source = record["source_id"]
        record["role"] = roles[source]
        if original is not None:
            original["role"] = roles[source]
            cases = controls(original, project_body(directory, record))
            prepared[original["group"]] = cases
            fingerprints[source] = signatures.union(
                *(
                    july.prior.text.fingerprints(case["text"])
                    for case in cases
                    if case["status"] == "constructed"
                )
            )
    exposures, old, duplicate = july.overlap(
        chain(july.exposure_packets(research_cache, gohumanize), history), fingerprints
    )
    originals = transfer.selected_groups(inspected, old, duplicate, authors, related)
    for record, _, _ in inspected:
        if record["status"].startswith("prior_july_"):
            record["status"] = record["status"].replace("prior_july_", "prior_publisher_", 1)
    cases = [case for original in originals for case in prepared[original["group"]]]
    counts = Counter(row["role"] for row in originals)
    planned = Counter(roles.values())
    editorial_counts = Counter(
        role
        for role, _ in {
            (row["role"], row["article_group"])
            for row in cases
            if row["status"] == "constructed" and row["arm"] != "original"
        }
    )
    original_bytes = b"".join(map(json_bytes, originals))
    case_bytes = b"".join(map(json_bytes, cases))
    return (
        original_bytes,
        case_bytes,
        {
            "schema": "chipmunk.text-elife-editorial-controls/301",
            "status": "complete"
            if all(
                counts[role] >= MIN_GROUPS_PER_ROLE
                and editorial_counts[role] >= MIN_EDITORIAL_GROUPS_PER_ROLE
                for role in ROLES
            )
            else "complete_below_target",
            "roster_sha256": sha(json_bytes(lock)),
            "publisher_revision": transfer.REVISION,
            "publisher_terms_url": july.TERMS,
            "planned_works": len(WORKS),
            "admitted_groups_by_role": {role: counts[role] for role in ROLES},
            "editorial_groups_by_role": {role: editorial_counts[role] for role in ROLES},
            "source_outcomes_by_role": {
                role: dict(Counter(row["status"] for row, _, _ in inspected if row["role"] == role))
                for role in ROLES
            },
            "control_outcomes_by_role": {
                role: {
                    arm: dict(
                        Counter(
                            row["reason"] or row["status"]
                            for row in cases
                            if row["role"] == role and row["arm"] == arm
                        ),
                        source_not_admitted=planned[role] - counts[role],
                    )
                    for arm in ARMS
                }
                for role in ROLES
            },
            "works": [record for record, _, _ in inspected],
            "exposure_packets": exposures,
            "private_originals_sha256": sha(original_bytes),
            "private_controls_sha256": sha(case_bytes),
            "generator_calls": 0,
            "detector_calls": 0,
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="mode", required=True)
    commands.add_parser("roster")
    commands.add_parser("urls").add_argument("roster", type=Path)
    intake = commands.add_parser("intake")
    for argument in ("roster", "xml_dir", "july_xml_dir", "transfer_xml_dir", "gohumanize"):
        intake.add_argument(argument, type=Path)
    intake.add_argument("--research-cache", type=Path, default=Path("/private/tmp"))
    intake.add_argument("--retain", action="store_true")
    args = parser.parse_args()
    if args.mode == "roster":
        print(json_bytes(protocol()).decode(), end="")
        return
    lock = json.loads(args.roster.read_bytes())
    require(lock == protocol(), "Use the exact pre-acquisition source roster")
    if args.mode == "urls":
        for work in lock["works"]:
            name = transfer.filename(work["source_id"])
            print(
                f"{name}\thttps://raw.githubusercontent.com/elifesciences/elife-article-xml/{transfer.REVISION}/articles/{name}"
            )
        return
    for folder in (args.xml_dir, args.july_xml_dir, args.transfer_xml_dir):
        require(folder.is_dir() and not folder.is_symlink(), "Expected direct owner-only XML directories")
        july.prior.archive.safe_cache(folder)
    originals, cases, receipt = run(
        lock, args.xml_dir, args.july_xml_dir, args.transfer_xml_dir, args.research_cache, args.gohumanize
    )
    if args.retain:
        july.prior.archive.safe_cache(DATA)
        for name, body in (
            ("private-originals.jsonl", originals),
            ("private-controls.jsonl", cases),
            ("aggregate.json", json_bytes(receipt)),
        ):
            july.prior.archive.retain(DATA / name, body)
    print(json_bytes(receipt).decode(), end="")


if __name__ == "__main__":
    main()
