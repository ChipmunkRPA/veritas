# SPDX-License-Identifier: Apache-2.0
"""Verify which text reaches writing checks when an article ends with a reference list."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from scripts.audio_vocal_event_mixtures import read, save, sha

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "backend/data/independent-evaluation"
SCHEMA = "chipmunk.writing-bibliography-scope-transfer/1"
PUBLISHERS = ("elife", "gates", "wellcome")
FORMATS = ("txt", "docx", "pdf")
ROLES = ("body", "bibliography", "followed_by_prose", "underfilled", "heading_with_prose", "one_nonentry")
GROUPS = 4
ENTRIES = 6
LICENSE = "https://creativecommons.org/licenses/by/4.0/"
DESIGN = {
    "sources": "Four dated article sources per publisher from the retained JATS archives. Pin article bytes and DOI. "
               "Use their initial body paragraphs and six of their labeled bibliography records, rendered from the same JATS fields.",
    "matched_controls": "Render text, Word and native PDF without dropping source characters. Test actual references, "
                        "an ordinary appendix after references, too few entries, a heading followed by prose and a mixed list.",
    "utility": "The entire qualified reference section should be excluded in the bibliography case only. "
               "On those cases, the retained normalized text and all source-map offsets must equal the body control.",
}
APPENDIX = ("Appendix", "This section describes the protocol used in the field. The team kept the \n"
            "calibration forms available so that later readers could reproduce the measurements.")


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def flatten(node):
    return " ".join("".join(node.itertext()).split()) if node is not None else ""


def latin(text):
    return text.translate(str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"', "–": "-", "—": "-", "−": "-"}))


def supported_pdf_text(text):
    return all(32 <= ord(char) < 127 or 160 <= ord(char) <= 255 for char in text)


def fields(node):
    authors = node.xpath("./person-group[@person-group-type='author']/name")
    if not authors:
        return None
    surname, initials = (flatten(authors[0].find(field)) for field in ("surname", "given-names"))
    year, title, source = (flatten(node.find(field)) for field in ("year", "article-title", "source"))
    doi = flatten(node.find("pub-id[@pub-id-type='doi']"))
    if not surname or not initials or not re.fullmatch(r"(?:18|19|20)\d{2}", year) or not (title and source and doi.startswith("10.")):
        return None
    name = re.sub(r"[^A-Z]", "", initials)[:4]
    if not name:
        return None
    other = " et al." if len(authors) > 1 or node.xpath("./person-group/etal") else ""
    entry = latin(f"{surname}, {name}.{other} ({year}). {title}. {source}. doi:{doi}.")
    return entry if 32 <= len(entry) <= 350 and supported_pdf_text(entry) else None


def article(path, publisher):
    from lxml import etree

    parser = etree.XMLParser(resolve_entities=False, no_network=True)
    try:
        tree = etree.parse(str(path), parser)
    except etree.XMLSyntaxError:
        return None
    meta = tree.find("./front/article-meta")
    if meta is None:
        return None
    doi = meta.xpath("string(./article-id[@pub-id-type='doi'])")
    license_urls = meta.xpath("./permissions/license/@*[local-name()='href']")
    if not doi or not any(url.replace("http:", "https:", 1).rstrip("/") == LICENSE.rstrip("/") for url in license_urls):
        return None
    years = {int(value) for value in meta.xpath("./pub-date/year/text()") if value.isdigit()}
    if not years or min(years) < 2000 or max(years) > 2021:
        return None
    body = [latin(flatten(p)) for p in tree.xpath("./body//p[not(ancestor::fig|ancestor::table-wrap|ancestor::boxed-text|ancestor::supplementary-material)]")]
    body = [part for part in body if 180 <= len(part) <= 650 and supported_pdf_text(part)][:2]
    bibliography = [entry for node in tree.xpath("./back/ref-list/ref/*[self::element-citation or self::mixed-citation]")
                    if (entry := fields(node)) is not None][:ENTRIES]
    if len(body) != 2 or len(bibliography) != ENTRIES:
        return None
    return {"publisher": publisher, "source_file": path.name, "xml_sha256": sha(path),
            "doi": doi, "year": min(years), "body_sha256": digest("\n\n".join(body)),
            "entries_sha256": digest("\n\n".join(bibliography))}, body, bibliography


def cases(elife, journals, excluded=()):
    seen = set(excluded)
    for publisher in PUBLISHERS:
        base = elife if publisher == "elife" else journals
        pattern = "*-v2.xml" if publisher == "elife" else f"{publisher}-*-after.xml"
        matched = 0
        for source in sorted(base.glob(pattern), key=lambda path: digest(f"{SCHEMA}/{path.name}")):
            row = article(source, publisher)
            if row and row[0]["doi"] not in seen:
                seen.add(row[0]["doi"])
                matched += 1
                yield row
                if matched == GROUPS:
                    break
        if matched != GROUPS:
            raise ValueError(f"Missing the required article groups for {publisher}")


def render(path, paragraphs):
    if path.suffix == ".txt":
        path.write_text("\n\n".join(paragraphs), encoding="utf-8")
    elif path.suffix == ".docx":
        from docx import Document

        file = Document()
        for paragraph in paragraphs:
            file.add_paragraph(paragraph)
        file.save(path)
    else:
        import textwrap

        import pymupdf

        with pymupdf.open() as file:
            page = file.new_page(width=612, height=792)
            position = 64
            for paragraph in paragraphs:
                lines = textwrap.wrap(paragraph, width=86, break_long_words=False, break_on_hyphens=False)
                height = 15 * len(lines)
                if position + height > 730:
                    page = file.new_page(width=612, height=792)
                    position = 64
                for line in lines:
                    page.insert_text((44, position), line, fontname="helv", fontsize=10)
                    position += 15
                position += 19
            file.save(path, no_new_id=True)


def deliverables(body, references):
    heading = ["References"]
    full = body + heading + references
    return {"body": body, "bibliography": full, "followed_by_prose": full + list(APPENDIX),
            "underfilled": body + heading + references[:3], "heading_with_prose": body + heading + body,
            "one_nonentry": body + heading + [*references[:3], APPENDIX[1], *references[3:]]}


def measure(output, elife, journals, excluded=()):
    from app.detectors.text.extract import extract
    from app.evidence import document_bibliography as rules

    output.mkdir(mode=0o700)
    observed = []
    for index, (case, body, entries) in enumerate(cases(elife, journals, excluded)):
        for ext in FORMATS:
            plain = None
            for role, paragraphs in deliverables(body, entries).items():
                path = output / f"{index:02}-{role}.{ext}"
                render(path, paragraphs)
                original = extract(path, path.suffix, allow_ocr=False)
                body_scope = rules.body_view(original)
                if role == "body":
                    plain = original
                assert plain is not None
                record = body_scope.meta.get(rules.KEY)
                if record:
                    rules.validate({"source": {"method": original.meta["method"], "chars": len(body_scope.text)},
                                    "coverage": {rules.KEY: record}})
                observed.append({"source": case, "format": ext, "role": role, "section": record,
                                 "body_matches": body_scope.text == plain.text and body_scope.source_map == plain.source_map,
                                 "other_text_preserved": body_scope.text == original.text and body_scope.source_map == original.source_map,
                                 "input_sha256": original.meta["extracted_text_sha256"],
                                 "writing_sha256": body_scope.meta["extracted_text_sha256"]})
    return observed


def run(args):
    from unittest.mock import patch

    from app.evidence import document_bibliography as rules

    parent = read(args.previous) if args.previous else None
    excluded = sorted({row["source"]["doi"] for row in parent["rows"]}) if parent else []
    if args.initial_rule:
        author_pattern = re.compile(r"^[^\W\d_][\w’'.-]{1,45}(?:,\s{0,2}|\s{1,2})[A-Z]{1,4}\b")
        with patch.object(rules, "_AUTHOR", author_pattern):
            rows = measure(args.path, args.elife, args.journals, excluded)
    else:
        rows = measure(args.path, args.elife, args.journals, excluded)
    receipt = {"schema": SCHEMA, "design": DESIGN, "rows": rows, "results": review(rows)}
    if args.previous:
        receipt["separation"] = {"previous_sha256": sha(args.previous), "excluded_dois": excluded}
    save(args.path / "receipt.json", receipt)
    return receipt


def review(rows):
    expected = {(ext, role) for ext in FORMATS for role in ROLES}
    counts = defaultdict(Counter)
    groups = defaultdict(set)
    identities = defaultdict(set)
    for row in rows:
        case, ext, role = (row[k] for k in ("source", "format", "role"))
        key = ext, role
        identities[case["publisher"]].add(case["doi"])
        assert key not in groups[case["doi"]]
        groups[case["doi"]].add(key)
        item = counts[ext, role]
        item.update(cases=1, excluded=int(row["section"] is not None), body_matches=int(row["body_matches"]),
                    original_preserved=int(row["other_text_preserved"]))
        if row["section"]:
            assert row["section"]["entry_count"] == ENTRIES and row["input_sha256"] != row["writing_sha256"]
    assert set(identities) == set(PUBLISHERS) and all(len(group) == GROUPS for group in identities.values())
    assert len(groups) == len(PUBLISHERS) * GROUPS and all(group == expected for group in groups.values())
    return {"/".join(key): dict(value) for key, value in sorted(counts.items())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("measure", "review"))
    parser.add_argument("path", type=Path)
    parser.add_argument("--elife", type=Path, default=DATA / "writing-elife-body-replication-2026-11-11/publisher")
    parser.add_argument("--journals", type=Path, default=DATA / "writing-journal-body-transfer-2026-11-11/publisher")
    parser.add_argument("--previous", type=Path)
    parser.add_argument("--initial-rule", action="store_true")
    args = parser.parse_args()
    if args.phase == "measure":
        receipt = run(args)
    else:
        receipt = read(args.path)
        assert receipt["schema"] == SCHEMA and receipt["design"] == DESIGN and review(receipt["rows"]) == receipt["results"]
        if args.previous:
            previous = read(args.previous)
            assert review(previous["rows"]) == previous["results"]
            excluded = sorted({row["source"]["doi"] for row in previous["rows"]})
            assert receipt["separation"] == {"previous_sha256": sha(args.previous), "excluded_dois": excluded}
            assert {row["source"]["doi"] for row in receipt["rows"]}.isdisjoint(excluded)
    print(json.dumps(receipt["results"], indent=2))


if __name__ == "__main__":
    main()
