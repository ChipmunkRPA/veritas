"""Admit a fixed set of publisher JATS originals for passage evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from xml.etree import ElementTree as ET

from scripts import text_goce_source_admission_v243 as prior
from scripts import text_goce_span_v249 as goce
from scripts import text_opai_bench_intake_v257 as opai

NAME = "text-elife-source-intake-v272"
DATA = prior.DATA / "independent-evaluation" / NAME
SOURCE = "https://github.com/elifesciences/elife-article-xml"
ARCHIVE = "https://elifesciences.org/archive/2022/july"
TERMS = "https://elifesciences.org/terms"
LICENSE = "https://creativecommons.org/licenses/by/4.0/"
# DOI suffix, July 2022 publication day, and ordered author surnames.
WORKS = (
    ("81306", 28, ("Carmichael",)),
    ("81108", 27, ("Romet-Lemonne",)),
    ("81334", 26, ("Weinreich",)),
    ("80891", 21, ("Pradeep", "Arratia")),
    ("80609", 20, ("Sebastian",)),
    ("80890", 15, ("Tonnelé", "Baud")),
    ("80718", 11, ("Fallon", "English")),
    ("80699", 7, ("Wang Chau", "Sugimura")),
    ("80392", 1, ("Pantalacci",)),
)
MAX_XML = 256_000
LLMTRACE_SHA = "6d4431a85a70916ba7af5c19f062eed6c277183d7b4bb5c6cea42f520e717f48"
OPAI_RECEIPT_SHA = "ad4193bb430e5c3c68f973e4a8edd80ad6c1c98a841b95cf575c6ef768ed6369"
XLINK = "{http://www.w3.org/1999/xlink}href"
INLINE = {"bold", "italic", "monospace", "overline", "sc", "strike", "sub", "sup", "underline"}
NON_PROSE = {"title", "fig", "fig-group", "table-wrap", "table-wrap-group"}
CITATIONS = {"bibr", "fig", "table", "fn", "supplementary-material"}
QUOTED = re.compile(r"""["“”]|(?<!\w)[‘'][^’'\n]{2,}[’'](?!\w)""")
CONTACT = re.compile(r"(?:https?://|www\.|[\w.+-]+@[\w.-]+\.[a-z]{2,})", re.I)
EMPTY_CITATION = re.compile(r"\(\s*[;,:\s]*\)")
EMPTY_CITATION_AT_STOP = re.compile(r"\s*\(\s*[;,:\s]*\)(?=[.,;:!?]|$)")
require, sha = prior.text.require, prior.text.sha
json_bytes = prior.prior.json_bytes


class Rejected(ValueError):
    """A fixed work is unavailable or fails an admission check."""


def verify(condition, reason):
    if not condition:
        raise Rejected(reason)


def normalized(element):
    return " ".join("".join(element.itertext()).split()) if element is not None else ""


def canonical_license(uri):
    if uri in (LICENSE, LICENSE.rstrip("/"), LICENSE.replace("https://", "http://")):
        return LICENSE
    return None


def metadata(article, work):
    source_id, day, surnames = work
    doi = f"10.7554/eLife.{source_id}"
    meta = article.find("front/article-meta")
    verify(
        article.tag == "article" and article.get("article-type") == "article-commentary" and meta is not None,
        "wrong_article_type",
    )
    kind = meta.find("article-categories/subj-group[@subj-group-type='display-channel']/subject")
    doi_element = meta.find("article-id[@pub-id-type='doi']")
    verify(
        normalized(kind) == "Insight" and normalized(doi_element).casefold() == doi.casefold(), "wrong_work"
    )
    verify(
        normalized(meta.find("volume")) == "11" and normalized(meta.find("elocation-id")) == f"e{source_id}",
        "wrong_work",
    )

    dates = set()
    for publication in meta.findall("pub-date"):
        year, month, date = (publication.findtext(key, "") for key in ("year", "month", "day"))
        if all(part.isdecimal() for part in (year, month, date)):
            dates.add((int(year), int(month), int(date)))
    verify(dates == {(2022, 7, day)}, "publication_date_mismatch")

    names = meta.findall("contrib-group/contrib[@contrib-type='author']/name")
    actual = tuple(normalized(name.find("surname")) for name in names)
    given = tuple(normalized(name.find("given-names")) for name in names)
    verify(actual == surnames and all(given), "author_mismatch")
    title = normalized(meta.find("title-group/article-title"))
    verify(bool(title), "missing_title")

    licenses = meta.findall("permissions/license")
    verify(len(licenses) == 1, "missing_or_ambiguous_license")
    license_element = licenses[0]
    notices = " ".join(normalized(line) for line in license_element.findall("license-p"))
    verify(canonical_license(license_element.get(XLINK)) == LICENSE and notices, "license_not_cc_by_4")
    links = {link.get(XLINK) for link in license_element.iter("ext-link")}
    verify(all(canonical_license(link) == LICENSE for link in links), "conflicting_license_link")
    return {
        "doi": doi,
        "title": title,
        "authors": [f"{first} {last}" for first, last in zip(given, actual, strict=True)],
        "publication_date": f"2022-07-{day:02d}",
        "license_url": LICENSE,
        "license_notice_sha256": sha(notices.encode()),
    }


def prose(element):
    if element.get("specific-use"):
        return None
    chunks = [element.text or ""]
    for child in element:
        if child.tag in INLINE:
            value = prose(child)
            if value is None:
                return None
            chunks.append(value)
        elif child.tag != "xref" or child.get("ref-type") not in CITATIONS:
            return None
        chunks.append(child.tail or "")
    return "".join(chunks)


def body_paragraphs(parent):
    """Yield normalized body paragraphs and gaps at excluded prose blocks."""
    for child in parent:
        if child.tag == "sec" and not child.get("specific-use"):
            yield from body_paragraphs(child)
        elif child.tag == "p":
            raw = prose(child)
            clean = (
                prior.text.canonical(EMPTY_CITATION.sub("", EMPTY_CITATION_AT_STOP.sub("", raw)))
                if raw
                else ""
            )
            yield clean if clean and not (QUOTED.search(clean) or CONTACT.search(clean)) else None
        elif child.tag not in NON_PROSE:
            yield None


def excerpt(paragraphs, source_id):
    for first in range(len(paragraphs)):
        for count in range(5, 9):
            window = paragraphs[first : first + count]
            if len(window) != count or any(part is None for part in window):
                break
            words = [len(part.split()) for part in window]
            if sum(words) > 500:
                break
            targets = [i for i in range(1, count - 1) if 24 <= words[i] <= 120]
            if sum(words) < 220 or not targets:
                continue
            edges, position = [], 0
            for part in window:
                edges.append([position, position + len(part)])
                position += len(part) + 2
            target = targets[int(sha(f"v272:{source_id}".encode()), 16) % len(targets)]
            text = "\n\n".join(window)
            return {
                "group": f"elife-{source_id}",
                "text": text,
                "text_sha256": sha(text.encode()),
                "body_slots": [first, first + count],
                "paragraph_codepoints": edges,
                "edit_target_codepoints": edges[target],
            }
    raise Rejected("no_eligible_excerpt")


def inspect(path, work, revision):
    source_id = work[0]
    file = f"elife-{source_id}-v1.xml"
    record = {
        "source_id": source_id,
        "article_url": f"https://elifesciences.org/articles/{source_id}",
        "xml_url": f"{SOURCE}/blob/{revision}/articles/{file}",
    }
    try:
        source = path / file
        verify(source.is_file() and not source.is_symlink(), "missing_direct_xml")
        with source.open("rb") as stream:
            raw = stream.read(MAX_XML + 1)
        verify(0 < len(raw) <= MAX_XML, "invalid_xml_size")
        record["xml_sha256"] = sha(raw)
        verify(not re.search(r"<!ENTITY\s", raw.decode("utf-8"), re.I), "xml_entity_declaration")
        article = ET.fromstring(raw)
        record.update(metadata(article, work))
        body = article.find("body")
        verify(body is not None, "missing_body")
        paragraphs = list(body_paragraphs(body))
        record["body_projection_sha256"] = sha(json_bytes(paragraphs))
        selection = excerpt(paragraphs, source_id)
        fragments = prior.text.fingerprints(selection["text"])
        for part in paragraphs:
            if part is not None:
                fragments.update(prior.text.fingerprints(part))
        return record, selection, fragments
    except Rejected as error:
        record["status"] = str(error)
    except (ET.ParseError, UnicodeDecodeError):
        record["status"] = "invalid_xml"
    except OSError:
        record["status"] = "unreadable_xml"
    return record, None, set()


def exposure_packets(cache, gohumanize_path):
    index_path = cache / "veritas-wikipedia-goce-rights-v243-attempt01/private-exposure-index.json"
    raw = prior.checked_bytes(index_path, opai.EXPOSURE_SHA)
    index = json.loads(raw)
    for (name, source_hash, texts), recorded in zip(
        prior.exposure_packets(cache, gohumanize_path), index["packets"], strict=True
    ):
        require(
            name == recorded["packet"] and source_hash == recorded["source_sha256"],
            "Expected exact v243 exposure inventory",
        )
        yield name, source_hash, recorded["text_entries"], texts

    opai_receipt = json.loads(
        prior.checked_bytes(prior.PANELS / "text-opai-bench-intake-complete-v257.json", OPAI_RECEIPT_SHA)
    )
    opai_hash = opai_receipt["private_cases_sha256"]
    opai_raw = prior.checked_bytes(opai.DATA / "private-cases.jsonl", opai_hash)
    yield (
        "opai_v257_v260",
        opai_hash,
        opai_receipt["selected_cases"],
        prior.validated_texts(prior.json_lines(opai_raw)),
    )

    goce_rows, evidence = goce.inputs(cache)
    yield "goce_v249", evidence["v243_private_commitment_sha256"], 4, prior.validated_texts(goce_rows)
    yield "llmtrace_classification_english_archive", LLMTRACE_SHA, None, llmtrace_texts()


def llmtrace_texts():
    path = prior.DATA / "text-upgrade/sources/llmtrace-test.jsonl"
    require(path.is_file() and not path.is_symlink(), "Expected direct LLMTrace archive")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for raw in stream:
            digest.update(raw)
            row = json.loads(raw)
            if row["lang"] == "eng":
                yield row["text"]
    require(digest.hexdigest() == LLMTRACE_SHA, "LLMTrace exposure bytes changed")


def overlap(packets, by_work):
    lookup = defaultdict(set)
    for source_id, fragments in by_work.items():
        for fragment in fragments:
            lookup[fragment].add(source_id)
    external, rows = set(), []
    for name, source_hash, expected_count, texts in packets:
        matched, count = set(), 0
        for value in texts:
            count += 1
            if lookup:
                for hit in prior.text.fingerprints(value) & lookup.keys():
                    matched.update(lookup[hit])
        require(count > 0 and expected_count in (None, count), "Writing exposure text count differs")
        rows.append(
            {
                "packet": name,
                "source_sha256": source_hash,
                "text_entries": count,
                "matched_sources": len(matched),
            }
        )
        external.update(matched)
    internal = {source_id for group in lookup.values() if len(group) > 1 for source_id in group}
    return rows, external, internal


def run(directory, revision, packets):
    require(re.fullmatch(r"[0-9a-f]{40}", revision), "Use a full publisher Git commit")
    inspections = [inspect(directory, work, revision) for work in WORKS]
    witnesses = {
        row["source_id"]: fragments for row, selection, fragments in inspections if selection is not None
    }
    exposure, external, internal = overlap(packets, witnesses)
    selected = []
    for row, selection, _ in inspections:
        if selection is None:
            continue
        source_id = row["source_id"]
        if source_id in external or source_id in internal:
            row["status"] = (
                "historical_fragment_overlap" if source_id in external else "internal_fragment_overlap"
            )
        else:
            row.update({key: value for key, value in selection.items() if key != "text"})
            row["status"] = "admitted"
            selected.append(selection)
    journal = b"".join(map(json_bytes, selected))
    method = prior.ROOT / f"docs/{NAME}-method.md"
    records = [row for row, _, _ in inspections]
    return journal, {
        "schema": "chipmunk.text-elife-source-intake/272",
        "status": "complete_intake" if selected else "no_sources_admitted",
        "publisher_revision": revision,
        "archive_url": ARCHIVE,
        "publisher_terms_url": TERMS,
        "planned_works": len(WORKS),
        "admitted_works": len(selected),
        "dispositions": dict(Counter(row["status"] for row in records)),
        "works": records,
        "exposure_packets": exposure,
        "private_originals_sha256": sha(journal),
        "method_sha256": sha(method.read_bytes()),
        "code_sha256": sha(Path(__file__).read_bytes()),
        "new_neural_calls": 0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("xml_dir", type=Path)
    parser.add_argument("publisher_revision")
    parser.add_argument("gohumanize_source", type=Path)
    parser.add_argument("--research-cache", type=Path, default=Path("/private/tmp"))
    parser.add_argument("--retain", action="store_true")
    args = parser.parse_args()
    require(args.xml_dir.is_dir() and not args.xml_dir.is_symlink(), "Expected a direct XML input directory")
    prior.archive.safe_cache(args.xml_dir)
    journal, receipt = run(
        args.xml_dir, args.publisher_revision, exposure_packets(args.research_cache, args.gohumanize_source)
    )
    if args.retain:
        prior.archive.safe_cache(DATA)
        prior.archive.retain(DATA / "private-originals.jsonl", journal)
        prior.archive.retain(DATA / "aggregate.json", json_bytes(receipt))
    print(json.dumps(receipt, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
