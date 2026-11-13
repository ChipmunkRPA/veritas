"""Replay genuine article excerpts from the recorded first-party publisher XML."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from itertools import combinations
from pathlib import Path
from tempfile import TemporaryDirectory
from time import sleep
from urllib.error import URLError
from urllib.request import HTTPRedirectHandler, Request, build_opener

from lxml.etree import XMLSyntaxError

from app.detectors.text.extract import from_string, text_hash
from app.detectors.text.segment import segment
from scripts.review_writing_long_position import read, verify
from scripts.text_short_insert_intake_v203 import fingerprints
from scripts.writing_journal_body_holdout import JOURNALS
from scripts.writing_journal_body_holdout import article as journal_article
from scripts.writing_reference_revision_sources import article as f1000_article
from scripts.writing_reference_revision_sources import blocks

PUBLISHERS = {
    "f1000": "f1000research.com",
    "gates": "gatesopenresearch.org",
    "wellcome": "wellcomeopenresearch.org",
}
VERSIONS = {"before": 1, "after": 2}
MAX_XML_BYTES = 2_500_000


class NoRedirect(HTTPRedirectHandler):
    """Use only the exact recorded version URL from the supported publisher."""

    def redirect_request(self, request, fp, code, msg, headers, new_url):
        raise ValueError("Publisher XML URL redirected")


def check(condition, message):
    if not condition:
        raise ValueError(message)


def source_paths(source):
    group, publisher = source["group"], source["publisher"]
    check(publisher in PUBLISHERS, "Unsupported publisher")
    check(bool(re.fullmatch(r"[0-9]+|(?:gates|wellcome)-[0-9]+", group)), "Invalid work identifier")
    paths = {}
    article_id = None
    for version, number in VERSIONS.items():
        url = source["urls"][version]
        match = re.fullmatch(
            rf"https://{re.escape(PUBLISHERS[publisher])}/articles/([0-9]+-[0-9]+)/v{number}/xml", url
        )
        check(match and article_id in (None, match[1]), f"{group}: invalid publisher version URL")
        article_id = match[1]
        paths[version] = (url, f"{group}-{version}.xml")
    return paths


def fetch(url, path, offline):
    if path.is_file():
        check(path.stat().st_size <= MAX_XML_BYTES, "Cached XML exceeded the size bound")
        return
    check(not offline, f"Cached XML is unavailable for {path.name}")
    request = Request(
        url,
        headers={
            "User-Agent": "ChipmunkVerify-Source-Replay/1.0",
            "Accept": "application/xml,text/xml;q=0.9",
        },
    )
    opener = build_opener(NoRedirect)
    with opener.open(request, timeout=20) as response:
        raw = response.read(MAX_XML_BYTES + 1)
        encoding = response.headers.get("Content-Encoding", "identity").lower()
    check(0 < len(raw) <= MAX_XML_BYTES, "Publisher XML exceeded the size bound")
    if encoding == "gzip":
        with gzip.GzipFile(fileobj=io.BytesIO(raw)) as body:
            raw = body.read(MAX_XML_BYTES + 1)
    else:
        check(encoding == "identity", "Unsupported publisher encoding")
    check(0 < len(raw) <= MAX_XML_BYTES, "Decoded publisher XML exceeded the size bound")
    with path.open("xb") as output:
        output.write(raw)
    sleep(0.20)


def verify_case(text, case, focal_range):
    doc = from_string(text)
    check(doc.text == text and text_hash(text) == case["text_sha256"], "Normalized article excerpt differs")
    units = segment(doc)
    positions = [[unit.start, unit.end, unit.paragraph, unit.n_words] for unit in units]
    check(positions == case["positions"] and len(text) == case["characters"], "Sentence geometry differs")
    check(
        [sum(not c.isspace() for c in unit.text) for unit in units] == case["nonspace"],
        "Sentence size differs",
    )
    match = [i for i, unit in enumerate(units) if [unit.start, unit.end] == focal_range]
    check(
        len(match) == 1 and 0 < match[0] < len(units) - 1,
        "The original synthetic target is not an interior sentence",
    )
    index = match[0]
    check(
        len({unit.paragraph for unit in units[index - 1 : index + 2]}) == 1,
        "Target neighbors cross a paragraph",
    )

    for i, record in enumerate(case["native_current"]):
        content = " ".join(unit.text for unit in units[max(0, i - 1) : i + 2])
        check(text_hash(content) == record["input_sha256"], "Current check input differs")
    input_count = len(case["native_current"])
    candidates = {}
    for i in range(1, len(units) - 1):
        a, middle, z = units[i - 1 : i + 2]
        if (
            a.paragraph == middle.paragraph == z.paragraph
            and 12 <= middle.n_words <= 65
            and all(7 <= neighbor.n_words <= 80 for neighbor in (a, z))
            and all(unit.text.endswith((".", "?", "!")) for unit in (a, middle, z))
        ):
            candidates[i] = ((a.text, middle.text), (a.text, middle.text, z.text), (middle.text, z.text))
    check(candidates.keys() == {row["idx"] for row in case["review"]}, "Eligible review positions differ")
    for row in case["review"]:
        for words, recorded in zip(candidates[row["idx"]], row["responses"], strict=True):
            if recorded is not None:
                check(text_hash(" ".join(words)) == recorded["input_sha256"], "Position review input differs")
                input_count += 1
    return input_count, (units[index - 1].text, units[index + 1].text)


def verify_version(source, version, case, path):
    publisher = source["publisher"]
    index = VERSIONS[version] - 1
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    doi = source["doi"][index]
    if publisher == "f1000":
        body, authors, published = f1000_article(path, doi)
    else:
        body, authors, published = journal_article(path, doi, JOURNALS[publisher][1])
    check(
        source["license"] == "CC-BY-4.0"
        and published == source["dates"][index]
        and authors == source["authors"],
        "Source license, publication date or byline differs",
    )
    check(date(*published) < date(2020, 1, 1), "Source publication date falls outside the freeze")
    runs = blocks(body, expanded=True)
    view = source["views"][version]
    block = view["block"]
    check(type(block) is int and 0 <= block < len(runs), "Publisher body block unavailable")
    run = runs[block]
    check(text_hash(run) == view["block_sha256"], "Publisher body block differs")
    paragraphs = run.split("\n\n")
    start, end = view["paragraph_range"]
    focal_paragraph = view["body_paragraph"]
    check(
        0 <= start <= focal_paragraph < end <= len(paragraphs) and 3 <= end - start <= 5,
        "Publisher paragraph range differs",
    )
    excerpt = "\n\n".join(paragraphs[start:end])
    check(text_hash(excerpt) == view["text_sha256"], "Publisher paragraph excerpt differs")
    count, neighbors = verify_case(excerpt, case, view["focus"])
    origin = sum(len(paragraph) + 2 for paragraph in paragraphs[start:focal_paragraph])
    check(
        origin <= view["focus"][0] < view["focus"][1] <= origin + len(paragraphs[focal_paragraph]),
        "Focal sentence lies outside its recorded body paragraph",
    )
    return digest, count, neighbors, excerpt


def fetch_group(source, cache, offline):
    for url, filename in source_paths(source).values():
        fetch(url, cache / filename, offline)


def verify_group(source, cases, cache):
    versions = {}
    for version, (_, filename) in source_paths(source).items():
        versions[version] = verify_version(source, version, cases[source["group"], version], cache / filename)
    before, after = (versions[v] for v in VERSIONS)
    check(
        before[2] == after[2]
        and before[3] != after[3]
        and date(*source["dates"][0]) < date(*source["dates"][1]),
        "Publisher versions do not preserve neighbors across a genuine revision",
    )
    return {
        "group": source["group"],
        "publisher": source["publisher"],
        "versions": [
            {
                "version": v,
                "observed_sha256": versions[v][0],
                "frozen_sha256": source["source_sha256"][v],
                "matches_frozen_bytes": versions[v][0] == source["source_sha256"][v],
                "verified_input_occurrences": versions[v][1],
            }
            for v in VERSIONS
        ],
        "authors": {name.casefold() for name in source["authors"]},
        "fingerprints": fingerprints(before[3]) | fingerprints(after[3]),
    }


def replay(receipt, cache, offline):
    verify(receipt)
    cases = {(case["group"], case["variant"]): case for case in receipt["cases"]}

    def lane(publisher):
        ready, failures = [], []
        for source in receipt["sources"]:
            if source["publisher"] != publisher:
                continue
            try:
                fetch_group(source, cache, offline)
                ready.append(source)
            except (OSError, URLError, ValueError) as error:
                failures.append({"group": source["group"], "reason": str(error)})
        return ready, failures

    results, failures = [], []
    with ThreadPoolExecutor(max_workers=len(PUBLISHERS)) as pool:
        for ready, lane_failures in pool.map(lane, PUBLISHERS):
            failures.extend(lane_failures)
            # Sentence segmentation uses one reusable parser. Keep native checks in this thread.
            for source in ready:
                try:
                    results.append(verify_group(source, cases, cache))
                except (OSError, ValueError, XMLSyntaxError) as error:
                    failures.append({"group": source["group"], "reason": str(error)})
    cross_source_pairs = 0
    for left, right in combinations(results, 2):
        check(
            not left["authors"] & right["authors"] and not left["fingerprints"] & right["fingerprints"],
            "Selected publisher sources overlap by author or exact 12-word fragment",
        )
        cross_source_pairs += 1
    versions = [v for row in results for v in row["versions"]]
    counts = Counter(
        "exact_xml" if row["matches_frozen_bytes"] else "changed_xml_same_body" for row in versions
    )
    return {
        "schema": "chipmunk.long-writing-publisher-source-replay/1",
        "verified_versions": len(versions),
        "verified_genuine_input_occurrences": sum(row["verified_input_occurrences"] for row in versions),
        "verified_cross_source_pairs": cross_source_pairs,
        "publisher_version_counts": dict(counts),
        "expected_versions": len(receipt["sources"]) * 2,
        "groups": [{key: row[key] for key in ("group", "publisher", "versions")} for row in results],
        "failures": failures,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    parser.add_argument(
        "--cache", type=Path, help="Retain publisher XML in this directory for offline replay"
    )
    parser.add_argument(
        "--offline", action="store_true", help="Use the supplied cache without fetching any sources"
    )
    parser.add_argument(
        "--strict-xml", action="store_true", help="Require historical XML bytes as well as exact body text"
    )
    args = parser.parse_args()
    if args.offline and args.cache is None:
        parser.error("--offline requires --cache")
    receipt = read(args.receipt)
    with TemporaryDirectory(prefix="chipmunk-writing-publisher-") as temp:
        cache = args.cache or Path(temp)
        cache.mkdir(parents=True, exist_ok=True)
        result = replay(receipt, cache, args.offline)
    print(json.dumps(result, sort_keys=True, indent=2))
    return int(
        bool(result["failures"])
        or args.strict_xml
        and result["publisher_version_counts"].get("exact_xml") != result["expected_versions"]
    )


if __name__ == "__main__":
    raise SystemExit(main())
