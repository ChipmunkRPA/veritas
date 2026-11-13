"""Rebuild the published Word study's publisher-source and ordinary draft sides."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import xml.etree.ElementTree as ET
from io import BytesIO
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

ROOT = Path(__file__).resolve().parents[2]
PANELS = ROOT / "docs/evaluation-panels"
WORD_RECEIPT = PANELS / "writing-word-boundary-transfer-2026-11-12.json"
SOURCE_RECEIPT = PANELS / "writing-semantic-continuity-transfer-2026-11-12.json"
JOURNAL = "https://journals.plos.org/plosbiology/article/file"
DOI = re.compile(r"10\.1371/journal\.pbio\.(\d{7})")
INLINE = {"bold", "italic", "monospace", "sup", "sub", "underline"}
X_LINK = "{http://www.w3.org/1999/xlink}href"
LANGUAGE = "{http://www.w3.org/XML/1998/namespace}lang"
WORD = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
XML_SPACE = "{http://www.w3.org/XML/1998/namespace}space"
MAX_XML = 900_000


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(value.encode() if isinstance(value, str) else value).hexdigest()


def bounded_read(path):
    with path.open("rb") as stream:
        material = stream.read(MAX_XML + 1)
    require(0 < len(material) <= MAX_XML, f"{path.name} exceeds the publisher XML limit or is empty")
    return material


class PublisherRedirect(HTTPRedirectHandler):
    def __init__(self, group):
        self.group = group

    def redirect_request(self, request, fp, code, message, headers, newurl):
        target = urlsplit(newurl)
        require(request.full_url.startswith(JOURNAL + "?") and target.scheme == "https"
                and re.fullmatch(r"corpus-prod\.[0-9a-f]+\.r2\.cloudflarestorage\.com", target.netloc)
                and target.path == f"/10.1371/journal.pbio.{self.group}/1/pbio.{self.group}.xml",
                "Publisher XML redirect differs from the recorded delivery")
        return super().redirect_request(request, fp, code, message, headers, newurl)


def publisher_xml(work, directory, *, download):
    url = f"{JOURNAL}?id={work['doi']}&type=manuscript"
    require(url == work["url"], "Publisher download URL differs from the pinned PLOS Biology path")
    path = directory / f"{work['group']}.xml"
    if path.is_file():
        material = bounded_read(path)
    else:
        require(download, f"Missing {path.name}. Place the exact publisher XML in the cache or use --download")
        request = Request(url, headers={"User-Agent": "Chipmunk-Verify-public-source-replay/1"})
        with build_opener(PublisherRedirect(work["group"])).open(request, timeout=25) as response:
            material = response.read(MAX_XML + 1)
        require(0 < len(material) <= MAX_XML, f"{path.name} exceeds the publisher XML limit or is empty")
    require(digest(material) == work["xml_sha256"], f"{path.name} does not match the pinned publisher XML")
    if not path.exists():
        with path.open("xb") as stream:
            stream.write(material)
    return material


def body_blocks(body):
    """Reconstruct the archived native prose blocks, omitting bibliographic references."""
    for parent in body.iter():
        previous = None
        for child in list(parent):
            if child.tag == "xref" and child.get("ref-type") == "bibr":
                tail = child.tail or ""
                if previous is None:
                    parent.text = (parent.text or "") + tail
                else:
                    previous.tail = (previous.tail or "") + tail
                parent.remove(child)
            else:
                previous = child

    def sections(section):
        yield section
        for child in section.findall("sec"):
            yield from sections(child)

    blocks = []
    for section in sections(body):
        paragraphs = []
        for item in section:
            if item.tag == "p" and all(node.tag in INLINE for node in list(item.iter())[1:]):
                value = " ".join("".join(item.itertext()).split())
                if value and not any(char in value for char in ('"', "“", "”", "«", "»")):
                    paragraphs.append(value)
                    continue
            if paragraphs:
                blocks.append("\n\n".join(paragraphs))
                paragraphs = []
        if paragraphs:
            blocks.append("\n\n".join(paragraphs))
    return blocks


def extract(work, material):
    # The exact known bytes are checked before parsing. The public JATS file's
    # external DTD is ignored by ElementTree. Reject entity declarations as well.
    require(b"<!ENTITY" not in material, "Unexpected publisher XML entity declaration")
    article = ET.fromstring(material)
    meta = article.find("front/article-meta")
    require(article.tag == "article" and article.get(LANGUAGE) == "en" and article.get("article-type") == "research-article"
            and meta is not None and meta.findtext("article-id[@pub-id-type='doi']") == work["doi"],
            f"{work['group']} has different source identity")
    licenses = meta.findall("permissions/license")
    require(len(licenses) == 1 and licenses[0].get(X_LINK) == work["license"]
            and work["license"].removeprefix("http://").removeprefix("https://").rstrip("/")
            == "creativecommons.org/licenses/by/4.0", f"{work['group']} has different license")
    online = meta.find("pub-date[@pub-type='epub']")
    require(online is not None and "-".join(online.findtext(key, "").zfill(2) for key in ("year", "month", "day")) == work["date"],
            f"{work['group']} has different date")
    authors = [" ".join(" ".join(name.itertext()).split())
               for name in meta.findall("contrib-group/contrib[@contrib-type='author']/name")]
    require(authors == work["authors"], f"{work['group']} has different attribution")
    body = article.find("body")
    require(body is not None, "No publisher article body")
    blocks = body_blocks(body)
    original = excerpt(blocks, work, "source_extent")
    donor = excerpt(blocks, work["donor"], "extent")
    return original, donor


def excerpt(blocks, locator, extent_field):
    index = locator["source_block"]
    require(type(index) is int and 0 <= index < len(blocks), "Published source block number is out of bounds")
    block = blocks[index]
    require(digest(block) == locator["source_block_sha256"], "Published source block differs")
    start, end = locator[extent_field]
    require(0 <= start < end <= len(block), "Published excerpt is out of bounds")
    return block[start:end]


def reconstructed_text(work, original, donor, native):
    source, ordinary = native
    start, end = work["target"]
    require([start, end] in source["spans"], "Published target is not a retained sentence")
    index = source["spans"].index([start, end])
    split_index = index + (int(digest(work["doi"]), 16) % 2)
    require(0 < split_index < len(source["spans"]), "Native boundary falls outside the excerpt")
    variants = (original, original[:start] + donor + original[end:])
    drafts = []
    for text, row in zip(variants, native, strict=True):
        require(digest(text) == row["text_sha256"], "Reconstructed text differs from the published semantic input")
        for (lo, hi), sha in zip(row["spans"], row["sentence_sha256"], strict=True):
            require(0 <= lo < hi <= len(text) and digest(text[lo:hi]) == sha, "Published sentence commitment differs")
        a, b = row["spans"][split_index - 1][1], row["spans"][split_index][0]
        require(text[a:b] == " ", "Expected one inter-sentence space for native Word split")
        drafts.append(text[:a] + "\n\n" + text[b:])
    return tuple(drafts)


def verify_side(text, source):
    left, right = text.split("\n\n")
    bounds = ((0, len(left)), (len(left) + 2, len(left) + 2 + len(right)))
    anchors, sentences = [], []
    for start, end in source["spans"]:
        containing = [(n + 1, n, start - a, end - a) for n, (a, b) in enumerate(bounds) if a <= start < end <= b]
        require(len(containing) == 1, "Saved sentence position crosses a Word paragraph")
        anchors.append(list(containing[0]))
        sentences.append(text[start:end])
    require(len(sentences) == 3 and anchors == source["word"], "Saved Word paragraph coordinates differ")
    require(digest(" ".join(sentences)) == source["context"] and digest(sentences[1]) == source["center"],
            "Reconstructed model input differs from the Word study")


def docx(text):
    """Write a deterministic minimal native package with the reconstructed body."""
    namespaces = "http://schemas.openxmlformats.org/package/2006"
    types = (f'<Types xmlns="{namespaces}/content-types">'
             '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
             '<Override PartName="/word/document.xml" '
             'ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/></Types>')
    rels = (f'<Relationships xmlns="{namespaces}/relationships"><Relationship Id="rId1" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
            'Target="word/document.xml"/></Relationships>')
    document = ET.Element(f"{{{WORD}}}document")
    body = ET.SubElement(document, f"{{{WORD}}}body")
    for paragraph in text.split("\n\n"):
        p = ET.SubElement(body, f"{{{WORD}}}p")
        run = ET.SubElement(p, f"{{{WORD}}}r")
        ET.SubElement(run, f"{{{WORD}}}t", {XML_SPACE: "preserve"}).text = paragraph
    output = BytesIO()
    with ZipFile(output, "w") as archive:
        for name, material in (("[Content_Types].xml", types), ("_rels/.rels", rels),
                               ("word/document.xml", ET.tostring(document, encoding="utf-8"))):
            archive.writestr(ZipInfo(name), material, compress_type=ZIP_DEFLATED)
    return output.getvalue()


def build(cache, native_dir=None, *, download=False):
    word_raw, semantic_raw = WORD_RECEIPT.read_bytes(), SOURCE_RECEIPT.read_bytes()
    word, semantic = json.loads(word_raw), json.loads(semantic_raw)
    require(word["schema"] == "chipmunk.writing-word-boundary-transfer/1", "Unknown Word study receipt")
    sources = {row["group"]: row for row in semantic["publisher"]}
    require(len(sources) == len(word["plos_sources"]) == 10, "Public source group count differs")
    source_rows = {(row["group"], row["role"]): row for row in semantic["cases"] if row["panel"] == "plos"}
    cache.mkdir(parents=True, exist_ok=True)
    if native_dir is not None:
        native_dir.mkdir(parents=True, exist_ok=True)
    source_count = ordinary_count = missing_generated_count = 0
    local_sources = []
    for work in word["plos_sources"]:
        match = DOI.fullmatch(work["doi"])
        require(match is not None and match[1] in sources, "Unexpected DOI in public source roster")
        group = match[1]
        source = sources[group]
        require({name: source[name] for name in work} == work, "Word and semantic publisher commitments differ")
        original, donor = extract(source, publisher_xml(source, cache, download=download))
        pair = tuple(source_rows[group, role] for role in ("original", "ordinary"))
        before, after = reconstructed_text(source, original, donor, pair)
        related = [row for row in word["cases"] if row["panel"] == "plos" and row["group"] == group]
        require(sum(row["role"] == "ordinary" for row in related) == 1, "Public group lacks one ordinary draft")
        for row in related:
            verify_side(before, row["geometry"]["earlier"])
            source_count += 1
            if row["role"] == "ordinary":
                verify_side(after, row["geometry"]["later"])
                ordinary_count += 1
            else:
                missing_generated_count += 1
        if native_dir is not None:
            files = {}
            for role, text in (("earlier", before), ("ordinary", after)):
                target = native_dir / f"{group}-{role}.docx"
                generated = docx(text)
                if target.exists():
                    require(target.read_bytes() == generated, f"{target.name} already exists with different bytes")
                else:
                    with target.open("xb") as stream:
                        stream.write(generated)
                files[role] = {"filename": target.name, "sha256": digest(generated)}
            local_sources.append({"doi": work["doi"], "authors": work["authors"], "license": work["license"],
                                  "changes": "Extracted article excerpt, omitted citations, added paragraph split. Ordinary copy replaces one sentence with other prose from the same article.",
                                  "drafts": files})
    require((source_count, ordinary_count, missing_generated_count) == (27, 10, 17), "Frozen Word panel counts differ")
    return {
        "schema": "chipmunk.writing-word-public-source-replay/1",
        "word_receipt_sha256": digest(word_raw), "semantic_receipt_sha256": digest(semantic_raw),
        "publisher_xmls_verified": len(sources), "earlier_case_sides_reconstructed": source_count,
        "ordinary_case_sides_reconstructed": ordinary_count, "generated_case_sides_not_public": missing_generated_count,
        "word_sentence_ranges_verified": 3 * (source_count + ordinary_count),
        "model_input_hashes_verified": 2 * (source_count + ordinary_count),
        "native_drafts": local_sources,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache", type=Path, help="local publisher XML cache")
    parser.add_argument("--download", action="store_true", help="download missing pinned XMLs from PLOS Biology")
    parser.add_argument("--native-dir", type=Path, help="optionally write locally rebuilt DOCX pairs")
    args = parser.parse_args()
    try:
        result = build(args.cache, args.native_dir, download=args.download)
    except (KeyError, OSError, TypeError, ValueError, ET.ParseError) as error:
        print(f"Source replay failed. {error}", file=sys.stderr)
        raise SystemExit(1) from None
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
