# SPDX-License-Identifier: Apache-2.0
"""Build the local release pack from the verified economic study intake."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime
from pathlib import Path

from lxml import html

from app.detectors.text.extract import from_string
from app.evidence import document_economic_archive as archive
from scripts import writing_economic_paraphrase_holdout as intake
from scripts.writing_multi_passage_panel import read

RESERVATION_SHA256 = "9e57697adda63c8ecc541bb1dbbe7c0e50917b2fca4cb25a1fbc2e273b1f1617"
CENSUS_DATE = '//meta[@property="article:published_time"]/@content'
BEA_RELEASE_DATE = '//*[contains(@class, "field--name-field-release-date")]'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def release_identity(address, publisher, tree):
    [canonical] = tree.xpath('//link[@rel="canonical"]/@href')
    if canonical != address or not archive.permitted_url(publisher, address):
        raise ValueError("The retained page does not confirm its official address")
    if publisher == "census":
        [published] = tree.xpath(CENSUS_DATE)
        title = tree.xpath("string(//h1)")
        return " ".join(title.split()), published[:10]
    title = tree.xpath("string(//title)").removesuffix(" | U.S. Bureau of Economic Analysis (BEA)")
    if "/blog/" in address:
        return " ".join(title.split()), address.split("/")[5]
    [row] = tree.xpath(BEA_RELEASE_DATE)
    [printed] = re.findall(r"\b[A-Z]+ \d{1,2}, 20\d{2}\b", row.text_content(), re.I)
    return " ".join(title.split()), datetime.strptime(printed.title(), "%B %d, %Y").date().isoformat()


def build(folder, output):
    reservation_path = folder / "reservation.json.gz"
    if sha(reservation_path.read_bytes()) != RESERVATION_SHA256:
        raise ValueError("Use the verified economic study reservation")
    references = []
    for row in read(reservation_path)["sources"]:
        address, source_hash = row["document"], row["source_sha256"]
        raw = (folder / "publisher" / f"{source_hash}.html").read_bytes()
        _, publisher, parts = intake.bodies(address, raw)
        text = from_string(parts).text
        if sha(raw) != source_hash or sha(text.encode()) != row["normalized_body_sha256"]:
            raise ValueError("A source differs from the reviewed federal editorial text")
        title, published_date = release_identity(address, publisher, html.fromstring(raw))
        if "©" in text or re.search(r"\b(?:all rights reserved|copyright(?:ed)?|used with permission)\b", text, re.I):
            raise ValueError("A retained body includes an indication of reserved rights")
        reference = {"id": sha(address.encode()), "publisher": publisher, "title": title, "source_url": address,
                     "published_date": published_date, "source_record_sha256": source_hash,
                     "transcript_sha256": sha(text.encode()), "text": text}
        archive.validate_reference(reference)
        references.append(reference)
    data = json.dumps({"schema": archive.SCHEMA, "references": references}, sort_keys=True,
                      separators=(",", ":"), ensure_ascii=False).encode() + b"\n"
    with output.open("xb") as stream:
        stream.write(data)
    print(f"{len(references)} release records · SHA-256 {sha(data)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("intake", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    build(args.intake, args.output)
