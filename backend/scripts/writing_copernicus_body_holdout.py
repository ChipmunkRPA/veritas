"""Collect distinct Copernicus article-body revisions before evaluating writing detectors."""
from __future__ import annotations

import argparse
import os
import re
from collections import Counter
from datetime import date
from pathlib import Path
from time import sleep
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pymupdf
from lxml import etree

from scripts.prepare_writing_option_order_transfer import benchmark_texts
from scripts.text_elife_source_intake_v272 import overlap
from scripts.text_short_insert_intake_v203 import fingerprints
from scripts.writing_agreement_revision_sources import read, require, save
from scripts.writing_blind_rewrite_probe import distance
from scripts.writing_focus_v3 import digest
from scripts.writing_publisher_revision_holdout import PUBLISHERS, frozen, licensed, metadata
from scripts.writing_publisher_revision_holdout import SCHEMA as PUBLISHER_SCHEMA
from scripts.writing_reference_revision_sources import AUTHOR_YEAR, excerpt, text_hash

SCHEMA = "chipmunk.writing-copernicus-body-holdout/1"
PARENT = Path(__file__).resolve().parents[2] / "docs/evaluation-panels/writing-paired-revision-replication-2026-11-11.json"
SCOUT_DOI = "10.5194/essd-2017-16"
POOL_SIZE = 20
CAPS = {"before": 2_000_000, "after": 600_000}
EARLIER = re.compile(r"10\.5194/([a-z]+)-([0-9]{4})-([0-9]+)")
LATER = re.compile(r"10\.5194/([a-z]+)-([0-9]+)-([0-9]+)-([0-9]{4})")
INTRO = re.compile(r"(?:1\.?\s+)?introduction[.:]?", re.I)
END = re.compile(r"(?:\d+\.?\s+)?(?:references|bibliography|author contributions|acknowledg(?:e)?ments|"
                 r"data and code availability|competing interests|supplementary material)\b", re.I)
NUMBERED = re.compile(r"\d+(?:\.\d+)*\.?\s+[A-Za-z]")
CAPTION = re.compile(r"(?:fig(?:ure)?\.?|table|appendix)\s+[A-Z]?\d+[.:\s]", re.I)
CC = re.compile(r"(?:cc\s*[-–]?\s*by|creative commons attribution)\s+(3\.0|4\.0)", re.I)
COPYRIGHT = re.compile(r"copyright|©|author\(s\)", re.I)
HYPHEN = re.compile(r"(?<=\w)[-‐‑–]\s*(?=\w)")
PLAN = {
    "schema": SCHEMA, "publisher": "copernicus", "pool_size": POOL_SIZE, "max_bytes": CAPS,
    "geometry": "Use one five-sentence source-ordered body excerpt per work with two exact common sentences on each "
                "side and a changed center of 12–40 words. Require a material lexical edit after removing citation "
                "and hyphenation artifacts. Keep source PDF page and line bounding boxes and final body offsets.",
    "source": "Use only existing DOI-linked Copernicus roster metadata, excluding the two scored abstract works "
              "and the source-only PDF scout. Require both per-version commercial CC BY terms and matching original "
              "author names. Freeze the first 20 eligible works in existing committed order before accessing bodies.",
    "extraction": "Read horizontal single-column discussion manuscript text after an exact Introduction heading. "
                  "Omit running headers, line-number gutters, separate headings, captions and unsupported content. "
                  "Match final publisher JATS body prose, keeping source names and dates private.",
}


def normalized_terms(terms):
    return {value.rsplit("/", 1)[-1] for value in terms}


def roster(base):
    release = read(PARENT)["holdout"]["source"]
    require(digest(base / "source-receipt.json") == release["source_receipt_sha256"], "Prior source receipt differs")
    prior = read(base / "source-receipt.json")
    cache = base / "publisher/copernicus"
    pool, binding = frozen(base, "copernicus")
    require(read(cache / "catalog-freeze.json") == binding
            and digest(cache / "catalog-freeze.json") == release["catalog_commitments"]["copernicus"]["sha256"],
            "Prior Copernicus catalog differs")
    excluded = {row[key] for row in read(base / "originals.json") for key in ("before_doi", "after_doi")} | {SCOUT_DOI}
    records = {}
    for i, (key, (earlier, counterpart)) in enumerate(pool.items()):
        path = cache / f"{i:03}.json"
        receipt = prior["publishers"]["copernicus"]["records"][key]
        require(digest(path) == receipt["counterpart_sha256"] and receipt["ordinal"] == i, "Prior DOI bytes differ")
        later = read(path)["message"]
        before, after = earlier["DOI"], later["DOI"]
        require(after == counterpart, "Publisher DOI link differs")
        m, n = EARLIER.fullmatch(before), LATER.fullmatch(after)
        if {before, after} & excluded or not m or not n or m[1] != n[1]:
            continue
        try:
            a, authors = metadata(earlier, PUBLISHERS["copernicus"])
            b, new_authors = metadata(later, PUBLISHERS["copernicus"])
        except (KeyError, TypeError, ValueError):
            continue
        if not a < b < date(2020, 1, 1) or authors != new_authors:
            continue
        prefix = f"https://{m[1]}.copernicus.org"
        slug, final = before.split("/")[1], after.split("/")[1]
        records[text_hash(before)[:16]] = {
            "before_doi": before, "after_doi": after, "before_date": a.isoformat(), "after_date": b.isoformat(),
            "authors": [list(value) for value in authors], "before_license": licensed(earlier), "after_license": licensed(later),
            "before_url": f"{prefix}/preprints/{slug}/{slug}.pdf",
            "after_url": f"{prefix}/articles/{n[2]}/{n[3]}/{n[4]}/{final}.xml",
            "metadata_counterpart_sha256": digest(path),
        }
        if len(records) == POOL_SIZE:
            break
    require(len(records) == POOL_SIZE, "Fewer than 20 independently bound article pairs")
    return records


def source_plan(base):
    return {**PLAN, "records": roster(base), "previous_public_sha256": digest(PARENT),
            "previous_source_sha256": digest(base / "source-receipt.json"), "pymupdf": pymupdf.__version__}


def fetch_asset(path, url, limit):
    if path.is_file():
        return {"sha256": digest(path), "bytes": path.stat().st_size}
    request = Request(url, headers={"User-Agent": "Chipmunk-Verify-public-revision-research/1.0"})
    try:
        with urlopen(request, timeout=20) as response:
            require(response.geturl().startswith("https://") and "copernicus.org/" in response.geturl(),
                    "Publisher download redirected outside its archive")
            if int(response.headers.get("Content-Length", "0")) > limit:
                return {"omitted": "size_limit"}
            data = response.read(limit + 1)
    except HTTPError as error:
        if error.code in (403, 404):
            return {"omitted": f"http_{error.code}"}
        raise
    if len(data) > limit:
        return {"omitted": "size_limit"}
    require(data.startswith(b"%PDF") if path.suffix == ".pdf" else data.lstrip().startswith(b"<?xml"),
            "Publisher download format differs")
    with path.open("xb") as stream:
        stream.write(data)
    return {"sha256": digest(path), "bytes": len(data)}


def fetch(run, base):
    plan = read(run / "source-plan.json")
    require(plan == source_plan(base), "Frozen body sources differ")
    cache = run / "publisher"
    cache.mkdir(exist_ok=True)
    for group, row in plan["records"].items():
        path = cache / f"{group}.json"
        if not path.exists():
            assets = {}
            for side, extension in (("before", "pdf"), ("after", "xml")):
                assets[side] = fetch_asset(cache / f"{group}-{side}.{extension}", row[f"{side}_url"], CAPS[side])
                sleep(.2)
            save(path, assets)
        print(group, {side: asset.get("bytes", asset.get("omitted")) for side, asset in read(path).items()}, flush=True)


def pdf_lines(page):
    lines = []
    for block in page.get_text("dict", sort=True, flags=pymupdf.TEXTFLAGS_DICT & ~pymupdf.TEXT_PRESERVE_IMAGES)["blocks"]:
        for line in block.get("lines", []):
            if abs(line["dir"][0] - 1) > .01 or abs(line["dir"][1]) > .01:
                continue
            spans = [span for span in line["spans"] if span["text"].strip()]
            text = " ".join("".join(span["text"] for span in line["spans"]).split())
            if not spans or not text or text.isdecimal():
                continue
            bbox = line["bbox"]
            heading = all(span["flags"] & pymupdf.TEXT_FONT_BOLD for span in spans)
            lines.append((bbox, text, heading))
    return sorted(lines, key=lambda value: (value[0][1], value[0][0]))


def pdf_sections(path, record):
    sections, paragraph, section = [], [], []

    def end_paragraph():
        if paragraph:
            section.append(paragraph[:])
            paragraph.clear()

    def end_section():
        end_paragraph()
        if section:
            sections.append(section[:])
            section.clear()

    with pymupdf.open(path) as document:
        front = " ".join(document[0].get_text().split())
        observed_terms = set(CC.findall(front))
        expected_date = date.fromisoformat(record["before_date"])
        timestamp = f"{expected_date.day} {expected_date:%B %Y}"
        require(record["before_doi"] in front.lower() and timestamp in front and bool(COPYRIGHT.search(front))
                and observed_terms & normalized_terms(record["before_license"]), "Discussion PDF identity or terms differ")
        started, finished = False, False
        for page in document:
            if finished:
                break
            previous = None
            for bbox, text, bold in pdf_lines(page):
                x0, y0, x1, y1 = bbox
                if (not .12 * page.rect.height <= y0 < y1 <= .96 * page.rect.height
                        or not .045 * page.rect.width <= x0 < x1 <= .99 * page.rect.width):
                    continue
                if not started:
                    started = bool(INTRO.fullmatch(text))
                    continue
                if END.fullmatch(text) or (bold and END.match(text)):
                    finished = True
                    break
                if ((bold and NUMBERED.match(text)) or CAPTION.match(text) or "©" in text
                        or any(ch in text for ch in ('"', '“', '”', '«', '»'))
                        or len(text.split()) < 2):
                    end_section()
                    previous = None
                    continue
                if previous is not None:
                    prior_x, prior_y, prior_end_x, prior_end_y = previous
                    if y0 < prior_y + 2:
                        end_section()
                        previous = None
                        continue
                    if y0 - prior_y > 2.6 * (prior_end_y - prior_y):
                        end_paragraph()
                    elif abs(x0 - prior_x) > .08 * page.rect.width and prior_end_x < .8 * page.rect.width:
                        end_section()
                paragraph.append({"page": page.number + 1, "bbox": [round(value, 3) for value in bbox], "text": text})
                previous = bbox
            end_paragraph()
        end_section()
    require(started and sections, "No supported single-column Introduction body")
    body = etree.Element("body")
    anchors = []
    for section_lines in sections:
        block = etree.SubElement(body, "sec")
        cursor, positions = 0, []
        for lines in section_lines:
            p = etree.SubElement(block, "p")
            p.text = " ".join(line["text"] for line in lines)
            for line in lines:
                stop = cursor + len(line["text"])
                positions.append({**line, "block_span": [cursor, stop]})
                cursor = stop + 1
            cursor += 1
        normalized = "\n\n".join(element.text for element in block)
        require(all(normalized[a:b] == line["text"] for line in positions for a, b in [line["block_span"]]),
                "PDF line does not map to its logical paragraph")
        anchors.append((text_hash(normalized), positions))
    return body, anchors


def final_body(path, record):
    root = etree.fromstring(path.read_bytes(), parser=etree.XMLParser(resolve_entities=False, no_network=True))
    meta, body = root.find("front/article-meta"), root.find("body")
    require(meta is not None and body is not None and meta.findtext("article-id[@pub-id-type='doi']") == record["after_doi"],
            "Final body DOI differs")
    stamp = date.fromisoformat(record["after_date"])
    date_parts = [(node.findtext("year"), node.findtext("month"), node.findtext("day")) for node in meta.findall("pub-date")]
    require(any(year == str(stamp.year) and month in {str(stamp.month), stamp.strftime("%m"), stamp.strftime("%B")}
                and day in {str(stamp.day), stamp.strftime("%d")} for year, month, day in date_parts), "Final body date differs")
    license_urls = {value for node in meta.findall("permissions/license") for element in node.iter()
                    for value in element.attrib.values() if "creativecommons.org/licenses/by/" in value}
    observed = normalized_terms({value.rstrip("/") for value in license_urls})
    require(observed & normalized_terms(record["after_license"]), "Final XML license differs")
    return body


def material_change(before, after):
    clean = []
    for row in (before, after):
        focus = HYPHEN.sub("", AUTHOR_YEAR.sub("", row["focus_text"]))
        clean.append(" ".join(re.findall(r"[^\W\d_]+", focus.casefold())))
    widths = [len(text.split()) for text in clean]
    return min(widths) >= 2 and distance(*clean) * max(widths) >= 2 - 1e-12


def collect(run, base, histories, f1000):
    lock_path = run / "source-plan.json"
    plan = read(lock_path)
    require(plan == source_plan(base) and histories, "Body source plan or exposure roster differs")
    cache = run / "publisher"
    candidates, signatures, receipts, dispositions = {}, {}, {}, Counter()
    for group, record in plan["records"].items():
        path = cache / f"{group}.json"
        assets = read(path)
        receipts[group] = {"asset_receipt_sha256": digest(path), **assets}
        if any("omitted" in data for data in assets.values()):
            dispositions["download_omitted"] += 1
            continue
        before, after = cache / f"{group}-before.pdf", cache / f"{group}-after.xml"
        require(all(digest(path) == assets[side]["sha256"] and path.stat().st_size == assets[side]["bytes"]
                    for side, path in (("before", before), ("after", after))), "An official asset differs from its receipt")
        try:
            old, positions = pdf_sections(before, record)
            new = final_body(after, record)
            edits = [(a, b) for a, b in excerpt(old, new, expanded=True) if material_change(a, b)]
        except (ValueError, etree.XMLSyntaxError) as error:
            dispositions["metadata_or_layout_rejected"] += 1
            print(f"{group} excluded: {error}", flush=True)
            continue
        if not edits:
            dispositions["no_exact_body_edit"] += 1
            continue
        candidates[group] = []
        for i, (earlier, later) in enumerate(edits):
            block_hash, lines = positions[earlier["block"]]
            require(block_hash == earlier["block_sha256"], "PDF line and normalized body block differ")
            start, stop = earlier["block_span"]
            focus_a, focus_b = earlier["focus"]
            earlier["pdf_lines"] = [{key: line[key] for key in ("page", "bbox", "block_span")}
                                    for line in lines if max(start + focus_a, line["block_span"][0]) <
                                    min(start + focus_b, line["block_span"][1])]
            require(earlier["pdf_lines"] and stop >= start + focus_b, "Exact PDF focus has no native line locator")
            candidates[group].append({"group": group, "publisher": "copernicus", "before": earlier, "after": later, **record})
            signatures[group, i] = fingerprints(earlier["text"]) | fingerprints(later["text"])
    prior = [*read(base / "originals.json"), *read(f1000)]
    packets = [(name, digest(path), None, benchmark_texts(path)) for name, path in histories]
    packets.append(("f1000-body", digest(f1000), None,
                    (row[side]["text"] for row in read(f1000) for side in ("before", "after"))))
    packets.append(("publisher-abstract-pilot", digest(base / "originals.json"), None,
                    (row[side]["text"] for row in read(base / "originals.json") for side in ("before", "after"))))
    exposure, excluded, _ = overlap(packets, signatures)
    used, authors, selected = set(), set(), []
    for row in prior:
        authors.update(" ".join(value) for value in row.get("authors", []))
        authors.update(value.casefold() for value in row.get("attribution", []))
    for group, rows in candidates.items():
        for i, row in enumerate(rows):
            identity = group, i
            contributors = {" ".join(value) for value in row["authors"]}
            if identity in excluded or used & signatures[identity] or authors & contributors:
                continue
            selected.append(row)
            used.update(signatures[identity])
            authors.update(contributors)
            dispositions["selected"] += 1
            break
        else:
            dispositions["author_or_text_overlap"] += 1
    return selected, {"schema": PUBLISHER_SCHEMA, "body_method_schema": SCHEMA, "source_plan_sha256": digest(lock_path),
                      "source_code_sha256": digest(Path(__file__)), "assets": receipts, "history": exposure,
                      "dispositions": dict(dispositions)}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("phase", choices=("freeze", "fetch", "prepare", "verify"))
    cli.add_argument("run", type=Path)
    cli.add_argument("base", type=Path)
    cli.add_argument("--f1000", type=Path)
    cli.add_argument("--history", action="append", default=[])
    args = cli.parse_args()
    os.umask(0o077)
    if args.phase == "freeze":
        save(args.run / "source-plan.json", source_plan(args.base))
    elif args.phase == "fetch":
        fetch(args.run, args.base)
    else:
        require(args.f1000, "Supply the earlier F1000 source roster")
        histories = [(name, Path(path)) for name, path in (entry.split("=", 1) for entry in args.history)]
        rows, receipt = collect(args.run, args.base, histories, args.f1000)
        originals, record = args.run / "originals.json", args.run / "source-receipt.json"
        if args.phase == "prepare":
            save(originals, rows)
            save(record, {**receipt, "originals_sha256": digest(originals)})
        else:
            require(read(originals) == rows and read(record) == {**receipt, "originals_sha256": digest(originals)},
                    "Body source replay differs")
        print(receipt["dispositions"], flush=True)


if __name__ == "__main__":
    main()
