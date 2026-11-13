"""Reconstruct historical source-separated F1000Research passage pairs."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from difflib import SequenceMatcher
from itertools import combinations
from pathlib import Path
from urllib.request import Request, build_opener

from lxml.etree import XMLSyntaxError

from app.detectors.text.extract import from_string, text_hash
from app.detectors.text.segment import segment
from scripts.text_short_insert_intake_v203 import fingerprints
from scripts.verify_writing_long_position_sources import NoRedirect, fetch
from scripts.writing_blind_rewrite_probe import words
from scripts.writing_reference_revision_sources import article, blocks

ROOT = Path(__file__).resolve().parents[2]
PUBLIC = ROOT / "docs/evaluation-panels"
CATALOG_QUERY = (
    "https://api.crossref.org/journals/2046-1402/works?"
    "filter=from-pub-date%3A2018-03-01%2Cuntil-pub-date%3A2018-04-30%2Ctype%3Ajournal-article"
    "&rows=300&select=DOI%2Cpublished%2Clink%2Clicense%2Ccontainer-title%2Ctype"
)
LICENSE = "https://creativecommons.org/licenses/by/4.0"
DOI = re.compile(r"10\.12688/f1000research\.(\d+)\.2")
XML = re.compile(r"https://f1000research\.com/articles/\d+-\d+/v2/xml")
SCHEMA = "chipmunk.f1000-long-revision-intake/1"
SIDES = ("before", "after")
LIMIT = 28
LONG_STUDY = "writing-long-position-transfer-2026-11-12.json"
HISTORY = (
    LONG_STUDY,
    "writing-position-qwen-admission-2026-11-12.json",
    "writing-position-confirmation-2026-11-12.json",
    "writing-clipped-rank-transfer-2026-11-12.json",
    "writing-source-conditioned-2026-11-12.json",
    "writing-expression-ore-transfer-2026-11-12.json",
    "writing-expression-plos-transfer-2026-11-12.json",
)


def read(path):
    return json.loads(path.read_bytes())


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def get_catalog(path, offline):
    max_bytes = 2_000_000
    if path.is_file():
        if not 0 < path.stat().st_size <= max_bytes:
            raise ValueError("Cached Crossref catalogue exceeded the size bound")
        return
    if offline:
        raise ValueError("Cached Crossref catalogue is unavailable")
    request = Request(
        CATALOG_QUERY,
        headers={"Accept": "application/json", "User-Agent": "ChipmunkVerify-Source-Replay/1.0"},
    )
    with build_opener(NoRedirect).open(request, timeout=20) as response:
        data = response.read(max_bytes + 1)
    if not 0 < len(data) <= max_bytes:
        raise ValueError("Crossref catalogue exceeded the size bound")
    with path.open("xb") as output:
        output.write(data)


def history():
    authors = set()
    works = set()
    pins = []
    for name in HISTORY:
        path = PUBLIC / name
        pins.append({"file": name, "sha256": digest(path)})
        values = read(path)["sources"]
        for row in values.values() if isinstance(values, dict) else values:
            if isinstance(row.get("group"), str) and row["group"].isdigit():
                works.add(row["group"])
            works.update(re.findall(r"10\.12688/f1000research\.(\d+)\.", json.dumps(row), re.I))
            names = row.get("authors") or row.get("attribution") or row.get("byline") or []
            authors.update(name.casefold() for name in names if isinstance(name, str))
    return works, authors, pins


def catalog_items(catalog):
    message = read(catalog)["message"]
    if len(message["items"]) != message["total-results"] or len(message["items"]) > 300:
        raise ValueError("The Crossref query did not return a complete catalogue")
    items = sorted(message["items"], key=lambda row: row["DOI"].lower())
    if len({row["DOI"].lower() for row in items}) != len(items):
        raise ValueError("Crossref returned a duplicate DOI")
    return items


def roster(items, excluded):
    eligible = {}
    for item in items:
        doi = item["DOI"].lower()
        match = DOI.fullmatch(doi)
        if not match or match[1] in excluded or item["container-title"] != ["F1000Research"]:
            continue
        if not any(
            row["URL"].replace("http:", "https:").rstrip("/") == LICENSE for row in item.get("license", [])
        ):
            continue
        urls = [
            row["URL"]
            for row in item.get("link", [])
            if row["content-type"] == "application/xml" and XML.fullmatch(row["URL"])
        ]
        if len(urls) == 1:
            eligible[match[1]] = {
                "doi": [doi[:-1] + "1", doi],
                "urls": {"before": urls[0].replace("/v2/", "/v1/"), "after": urls[0]},
            }
    order = sorted(eligible, key=lambda group: text_hash(f"long-writing-f1000-2018:{group}"))
    return {group: eligible[group] for group in order[:LIMIT]}


def path_for(cache, group, side):
    return cache / f"{group}-{side}.xml"


def get_sources(candidates, cache, offline):
    def pair(item):
        group, info = item
        for side in SIDES:
            path = path_for(cache, group, side)
            fetch(info["urls"][side], path, offline)
        return group

    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(pair, candidates.items()))


def views(body):
    """Index exact flanks and a bounded three-to-five-paragraph native excerpt."""
    index = defaultdict(list)
    for block_index, value in enumerate(blocks(body, expanded=True)):
        if len(value) > 90_000:
            continue
        paragraphs = value.split("\n\n")
        if len(paragraphs) < 3:
            continue
        units = [segment(from_string(paragraph)) for paragraph in paragraphs]
        block_hash = text_hash(value)
        for i, sentences in enumerate(units):
            for j in range(1, len(sentences) - 1):
                a, focus, z = sentences[j - 1 : j + 2]
                if not 12 <= focus.n_words <= 40 or any(not 7 <= u.n_words <= 65 for u in (a, z)):
                    continue
                if any(not u.text.endswith((".", "?", "!")) for u in (a, focus, z)):
                    continue
                choices = []
                for width in range(3, 6):
                    for start in range(max(0, i - width + 1), min(i + 1, len(paragraphs) - width + 1)):
                        end = start + width
                        sample = [unit for paragraph in units[start:end] for unit in paragraph]
                        if not 9 <= len(sample) <= 25 or not 160 <= sum(u.n_words for u in sample) <= 680:
                            continue
                        text = "\n\n".join(paragraphs[start:end])
                        if len(text) > 5_200 or from_string(text).text != text:
                            continue
                        offset = sum(len(p) + 2 for p in paragraphs[start:i])
                        position = [offset + focus.start, offset + focus.end]
                        choices.append((width, abs(i - start - width // 2), start, end, text, position))
                    if choices:
                        break
                if choices:
                    _, _, start, end, text, position = min(choices)
                    index[a.text, z.text].append(
                        {
                            "text": text,
                            "target_text": focus.text,
                            "flanks": [a.text, z.text],
                            "target": position,
                            "block": block_index,
                            "block_sha256": block_hash,
                            "paragraph_range": [start, end],
                        }
                    )
    return index


def material(old, revised):
    a, b = words(old), words(revised)
    difference = SequenceMatcher(None, a, b, autojunk=False)
    edit = sum(max(j - i, v - u) for tag, i, j, u, v in difference.get_opcodes() if tag != "equal")
    return edit, difference.ratio()


def select_pair(before, after, seen):
    options = []
    for flank in before.keys() & after.keys():
        if len(before[flank]) != 1 or len(after[flank]) != 1:
            continue
        a, b = before[flank][0], after[flank][0]
        body_changes, ratio = material(a["text"], b["text"])
        if body_changes < 2 or ratio < 0.80:
            continue
        target_changes, target_ratio = material(a["target_text"], b["target_text"])
        if target_changes and target_ratio < 0.45:
            continue
        marks = fingerprints(a["text"]) | fingerprints(b["text"])
        if not marks or seen & marks:
            continue
        options.append((target_changes < 2, text_hash(a["text"] + "\0" + b["text"]), a, b, marks))
    return min(options, key=lambda row: (row[0], row[1], row[2]["block"], row[2]["target"]), default=None)


def public_view(view):
    text = view["text"]
    native = segment(from_string(text))
    match = [i for i, unit in enumerate(native) if [unit.start, unit.end] == view["target"]]
    if len(match) != 1:
        raise ValueError("Native target position is missing or ambiguous")
    i = match[0]
    if (
        i < 1
        or i >= len(native) - 1
        or native[i].text != view["target_text"]
        or [native[i - 1].text, native[i + 1].text] != view["flanks"]
    ):
        raise ValueError("Native segmentation differs from the source target and its flanks")
    inputs = [" ".join(u.text for u in native[max(0, i - 1) : i + 2]) for i in range(len(native))]
    contexts = []
    for index in range(1, len(native) - 1):
        a, m, z = native[index - 1 : index + 2]
        if not (
            a.paragraph == m.paragraph == z.paragraph
            and 12 <= m.n_words <= 65
            and all(7 <= u.n_words <= 80 for u in (a, z))
            and all(u.text.endswith((".", "?", "!")) for u in (a, m, z))
        ):
            continue
        contexts.append(
            {
                "focus": [m.start, m.end],
                "inputs_sha256": [
                    text_hash(" ".join((a.text, m.text))),
                    text_hash(inputs[index]),
                    text_hash(" ".join((m.text, z.text))),
                ],
            }
        )
    if not any(row["focus"] == view["target"] for row in contexts):
        raise ValueError("The target falls outside the frozen longer-writing geometry")
    return {key: view[key] for key in ("target", "block", "block_sha256", "paragraph_range")} | {
        "text_sha256": text_hash(text),
        "immediate_flanks_sha256": [text_hash(value) for value in view["flanks"]],
        "positions": [[unit.start, unit.end, unit.paragraph, unit.n_words] for unit in native],
        "native_input_sha256": [text_hash(value) for value in inputs],
        "position_contexts": contexts,
    }


def long_study_coverage(rows):
    prior = read(PUBLIC / LONG_STUDY)
    prior_inputs = {entry["input_sha256"] for case in prior["cases"] for entry in case["native_current"]}
    prior_inputs.update(
        entry["input_sha256"]
        for case in prior["cases"]
        for focus in case["review"]
        for entry in focus["responses"]
        if entry is not None
    )
    views = [view for row in rows for view in row["views"].values()]
    contexts = [context for view in views for context in view["position_contexts"]]
    inputs = {sha for view in views for sha in view["native_input_sha256"]}
    inputs.update(sha for context in contexts for sha in context["inputs_sha256"])
    return {
        "prior_unique_measured_inputs": len(prior_inputs),
        "unique_planned_inputs": len(inputs),
        "shared_input_hashes": len(prior_inputs & inputs),
        "geometry_eligible_targets": len(views),
        "geometry_eligible_centers": len(contexts),
        "frozen_comparison_points": {
            "short_passage": prior["point"]["original_point"],
            "long_document": prior["point"]["operating_point"],
        },
    }


def prepare(catalog, cache, offline):
    excluded, old_authors, pins = history()
    items = catalog_items(catalog)
    candidates = roster(items, excluded)
    get_sources(candidates, cache, offline)
    rows = []
    acquired = {}
    admitted_signatures = []
    counts = Counter()
    seen = set()
    for group, entry in candidates.items():
        paths = [path_for(cache, group, side) for side in SIDES]
        acquired[group] = {
            side: {"sha256": digest(path), "bytes": path.stat().st_size}
            for side, path in zip(SIDES, paths, strict=True)
        }
        try:
            (before, people, d1), (after, others, d2) = [
                article(path, doi) for path, doi in zip(paths, entry["doi"], strict=True)
            ]
        except (XMLSyntaxError, ValueError, AttributeError, TypeError):
            counts["metadata_rejected"] += 1
            continue
        byline = {name.casefold() for name in people}
        if people != others or byline & old_authors or not date(*d1) < date(*d2) < date(2018, 5, 1):
            counts["byline_or_dates_rejected"] += 1
            continue
        result = select_pair(views(before), views(after), seen)
        if result is None:
            counts["no_exact_paired_excerpt"] += 1
            continue
        target_unchanged, _, a, b, marks = result
        try:
            projections = [public_view(view) for view in (a, b)]
        except ValueError:
            counts["segment_rejected"] += 1
            continue
        old_authors.update(byline)
        seen.update(marks)
        admitted_signatures.append((byline, marks))
        counts["admitted"] += 1
        counts["genuine_center_changed" if not target_unchanged else "genuine_change_elsewhere_or_minor"] += 1
        rows.append(
            {
                "group": group,
                **entry,
                "authors": people,
                "dates": [d1, d2],
                "license": LICENSE + "/",
                "views": dict(zip(SIDES, projections, strict=True)),
                "genuine_target_materially_revised": not target_unchanged,
            }
        )
    compared = 0
    for (people, phrases), (others, other_phrases) in combinations(admitted_signatures, 2):
        if people & others or phrases & other_phrases:
            raise ValueError("Admitted sources share a byline or a 12-word source fragment")
        compared += 1
    return {
        "schema": SCHEMA,
        "catalog_url": CATALOG_QUERY,
        "catalog_items": len(items),
        "catalog_items_sha256": text_hash(json.dumps(items, sort_keys=True, separators=(",", ":"))),
        "prior_source_ledgers": pins,
        "candidate_work_groups": len(candidates),
        "acquired_publisher_files": acquired,
        "compared_admitted_pairs": compared,
        "verified_native_inputs": sum(
            len(view["native_input_sha256"]) for row in rows for view in row["views"].values()
        ),
        "long_study_coverage": long_study_coverage(rows),
        "counts": dict(counts),
        "sources": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument(
        "--expected", type=Path, help="Verify against a previously frozen source-only receipt"
    )
    args = parser.parse_args()
    args.cache.mkdir(parents=True, exist_ok=True)
    catalog = args.cache / "catalog.json"
    get_catalog(catalog, args.offline)
    receipt = prepare(catalog, args.cache, args.offline)
    if args.expected and receipt != read(args.expected):
        raise ValueError("Reconstructed catalogue, source bytes or passage positions differ")
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
