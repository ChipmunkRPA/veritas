# SPDX-License-Identifier: Apache-2.0
"""Check date observations on Federal Register notices and matched mutations."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from calendar import monthrange
from collections import Counter, defaultdict
from pathlib import Path

from scripts.audio_native_alignment_panel import digest, read, save

SCHEMA = "chipmunk.document-calendar-panel/1"
WEEK = ("Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday")
MONTH = ("January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December")
LOOKUP = {name.casefold(): index for index, name in enumerate(MONTH, 1)}
FORM = re.compile(
    r"\b(Sunday|Monday|Tuesday|Wednesday|Thursday|Friday|Saturday)[, ]+"
    r"([A-Za-z]+) ([0-9]{1,2})(?:st|nd|rd|th)?[, ]+(19[0-9]{2}|20[0-9]{2})\b", re.I,
)


def weekday(year, month, day):
    """Gregorian weekday formula independent of the app's datetime method."""
    if not 1 <= day <= monthrange(year, month)[1]:
        return None
    adjusted = year - int(month < 3)
    offsets = (0, 3, 2, 5, 0, 3, 5, 1, 4, 6, 2, 4)
    return WEEK[(adjusted + adjusted // 4 - adjusted // 100 + adjusted // 400 + offsets[month - 1] + day) % 7]


def _sources(directory):
    from lxml import etree

    for path in sorted(directory.glob("FR-2019-*.xml")):
        parser = etree.XMLParser(resolve_entities=False, load_dtd=False, no_network=True)
        document = etree.parse(str(path), parser)
        if document.getroot().tag != "FEDREG":
            raise ValueError("Expected the original daily Federal Register XML")
        parts = {}
        for node in document.iter("NOTICE"):
            label = re.search(r"FR Doc\. (\d{4}-\d{5})", node.findtext("FRDOC") or "")
            section = node.find("PREAMB/DATES")
            if label is None or section is None:
                continue
            paragraphs = []
            for element in section.iterfind("./P"):
                body = " ".join("".join(element.itertext()).split())
                if body:
                    paragraphs.append(body)
            text = "\n\n".join(paragraphs)
            if len(text) >= 40:
                if label[1] in parts:
                    raise ValueError("The notice must appear once in each original issue")
                parts[label[1]] = text
        yield path, parts


def _cases(text, match):
    weekday_word, month, numeric, year = match.groups()
    original = weekday_word.capitalize()
    day, year_num, month_num = int(numeric), int(year), LOOKUP[month.casefold()]
    swapped = WEEK[(WEEK.index(original) + 2) % 7]
    changed = day + 1 if day < monthrange(year_num, month_num)[1] else day - 1
    chronology = (year_num, month_num, day)
    for role, replaced, fields, claim in (
        ("weekday_changed", f"{swapped}, {month} {day}, {year}", chronology, swapped),
        ("day_changed", f"{original}, {month} {changed}, {year}", (year_num, month_num, changed), original),
        ("impossible_date", f"{original}, February 30, {year}", (year_num, 2, 30), original),
        ("day_first_control", f"{original} {day} {month} {year}", chronology, original),
        ("year_omitted", f"{original}, {month} {day}", None, original),
    ):
        expectation = weekday(*fields) if fields else None
        start = match.start()
        yield role, text[:start] + replaced + text[match.end():], {
            "span": [start, start + len(replaced)], "stated": claim, "date": fields,
            "weekday": expectation, "mismatch": bool(fields and expectation != claim),
        }


def _measure(text):
    from app.detectors.text.extract import from_string
    from app.evidence import document_calendar as check

    doc = from_string(text)
    if doc.text != text:
        raise ValueError("Staged source offsets must survive normal extraction")
    source = {"method": "plain", "chars": len(text)}
    result = check.build(source, doc.text, doc.source_map)
    check.validate({"source": source, check.KEY: result})
    check.verify_source({"source": source, check.KEY: result}, doc.text, doc.source_map)
    rows = [{
        "span": [row["start"], row["end"]], "stated": row["stated_weekday"], "weekday": row["expected_weekday"],
        "reason": row["reason"],
    } for row in result["locations"]]
    return {"text_sha256": hashlib.sha256(text.encode()).hexdigest(), "checked": result["checked_dates"],
            "disagreements": result["disagreements"], "findings": rows}


def totals(rows):
    counts = defaultdict(Counter)
    for row in rows:
        prior = row["truth"]
        treatment = row["role"]
        found = row["measurement"]
        if prior:
            fields = prior["date"]
            expected = weekday(*fields) if fields else None
            reason = "weekday_mismatch" if expected else "invalid_date"
            if expected != prior["weekday"] or bool(fields and expected != prior["stated"]) != prior["mismatch"]:
                raise ValueError("The expected day no longer follows the indicated Gregorian date")
            truth = {"span": prior["span"], "weekday": expected, "stated": prior["stated"], "reason": reason}
            target = truth in found["findings"] if prior["mismatch"] else any(
                item["span"] == prior["span"] for item in found["findings"])
        else:
            target = False
        counts[treatment].update(files=1, dates=found["checked"], observations=found["disagreements"],
                                 files_with_observations=bool(found["disagreements"]), agreed_exactly=target == bool(prior and prior["mismatch"]))
    return dict(counts)


def measure(directory, output, examples=None):
    from app.evidence import document_calendar

    roster, all_sources, positives = [], {}, {}
    for file, notices in _sources(directory):
        all_sources[file.name] = {"sha256": digest(file), "url": f"https://www.govinfo.gov/bulkdata/FR/{file.name[3:7]}/{file.name[8:10]}/{file.name}"}
        for name, text in notices.items():
            base = {"issue": file.name, "document": name}
            normal = _measure(text)
            roster.append(base | {"role": "original", "truth": None, "measurement": normal})
            if name in positives:
                raise ValueError("A selected agency notice appears twice")
            for match in FORM.finditer(text):
                if (month_num := LOOKUP.get(match[2].casefold())) is None:
                    continue
                if weekday(int(match[4]), month_num, int(match[3])) == match[1].capitalize():
                    positives[name] = (base, text, match)
                    break
    selected = sorted(positives, key=lambda label: hashlib.sha256(f"{SCHEMA}/{label}".encode()).hexdigest())[:90]
    if len(selected) < 45 or len(all_sources) < 50:
        raise ValueError("Too few independent notices with explicit weekday-date pairs")
    for index, label in enumerate(selected):
        base, text, match = positives[label]
        for role, modified, truth in _cases(text, match):
            roster.append(base | {"role": role, "truth": truth, "measurement": _measure(modified)})
            if examples and index == 0 and role == "weekday_changed":
                examples.mkdir(mode=0o700)
                (examples / "calendar-mismatch.txt").write_text(modified, encoding="utf-8")
                (examples / "untouched.txt").write_text(text, encoding="utf-8")
    summary = totals(roster)
    value = {"schema": SCHEMA, "issues": all_sources, "selected_notices": selected, "cases": roster, "totals": summary,
             "implementation_sha256": digest(Path(document_calendar.__file__)), "recipe_sha256": digest(Path(__file__))}
    save(output, value)
    print(json.dumps(summary, sort_keys=True))


def review(receipt):
    package = read(receipt)
    if package["schema"] != SCHEMA or totals(package["cases"]) != package["totals"]:
        raise ValueError("The published calendar outcomes could not be reproduced")
    print(json.dumps(package["totals"], sort_keys=True))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    actions = cli.add_subparsers(dest="action", required=True)
    probe = actions.add_parser("measure")
    probe.add_argument("directory", type=Path)
    probe.add_argument("output", type=Path)
    probe.add_argument("--examples", type=Path)
    actions.add_parser("review").add_argument("receipt", type=Path)
    opts = vars(cli.parse_args())
    globals()[opts.pop("action")](**opts)
