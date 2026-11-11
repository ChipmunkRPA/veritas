"""Validate an offline, source-grouped OpAI-Bench reports intake."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

from scripts import text_goce_source_admission_v243 as exposure

NAME = "text-opai-bench-intake-v257"
DATA = exposure.DATA / "independent-evaluation" / NAME
REVISION = "072c03c7051409346aaba05168d183cf6d863ee9"
FILES = {
    "qwen3-8b": (35, 5_919_856, "23e2b29270910b2a54d1ae2859e9e2b1e89cee1dd179f16d2ea2b39c7e60af25"),
    "gpt-5.4-nano": (300, 53_589_036, "912d90f2f22b2ea97bdf69930f6891584112381f5b811aedcac390bbdf1da064"),
}
VERSIONS = {"v0": "none", "v1": "polish", "v4": "compress"}
MAX_GROUPS = 12
EXPOSURE_SHA = "4f95c3715166cda4f83b8d17355b516b1ed0474c2e65018ea34c2a85e6e5c489"
AUTHORITY_RECEIPT_SHA = "d56f54e50f87b719e5b2a360e75da04c7534b440cf672dcdb2037c3db25ec9db"
AUTHORITY_RECORD_BYTES = 1_709_758
AUTHORITY = {
    "schema": "chipmunk.govreport-upstream-receipt/257",
    "revision": "32feeaede49fed993aef070bc4da09263fd0429a",
    "publisher_sha256": "ee7a981a768c7f0c68d635e4a3cbfac03b95e45284181232d6c7021d511ea729",
    "publisher_bytes": 709_026_557,
    "publisher_rows": 11_005,
    "matched_ids": 30,
    "private_upstream_packet_sha256": "c3e0df301d7405c50752aa2755f51f2693747e90a764636b9cab1c55b5aa0e1e",
}
TAG = re.compile(r"<AI_Start>|</AI_End>")
ID = re.compile(r"GAO_[A-Za-z0-9_-]+")
require, sha = exposure.text.require, exposure.text.sha


def decode_tags(tagged):
    """Return unmodified Unicode text and half-open character intervals."""
    chunks, spans = [], []
    pos = cursor = 0
    start = None
    for tag in TAG.finditer(tagged):
        segment = tagged[cursor : tag.start()]
        chunks.append(segment)
        pos += len(segment)
        cursor = tag.end()
        if tag.group() == "<AI_Start>":
            require(start is None, "Nested opening span marker")
            start = pos
        else:
            require(start is not None and start < pos, "Unmatched or empty closing span marker")
            spans.append([start, pos])
            start = None
    require(start is None, "Unclosed span marker")
    chunks.append(tagged[cursor:])
    return "".join(chunks), spans


def record(row):
    version = row["version"]
    raw, spans = decode_tags(row["text_tagged"])
    declared = json.loads(row["ai_spans_char"])
    require(
        raw
        and raw == row["text"]
        and type(declared) is list
        and all(
            type(span) is list and len(span) == 2 and all(type(edge) is int for edge in span)
            for span in declared
        )
        and spans == declared,
        "Tagged content, full untagged text and published character intervals must agree exactly",
    )
    edited = sum(end - start for start, end in spans)
    require(
        row["edit_operation"] == VERSIONS[version]
        and row["version_index"] == version[1:]
        and (edited == 0 if version == "v0" else 0 < edited < len(raw)),
        "Expected a clean source or a partially edited variant",
    )
    return {
        "record_id": row["record_id"],
        "generator": row["generator"] if edited else None,
        "version": version,
        "edit_operation": row["edit_operation"],
        "text": raw,
        "text_sha256": sha(raw.encode()),
        "ai_spans": spans,
    }


def inventory(directory, generator, spec):
    expected_groups, expected_bytes, expected_digest = spec
    path = directory / f"reports_{generator}.csv"
    require(path.is_file() and not path.is_symlink(), "Expected a direct publisher CSV")
    selected, seen_records = defaultdict(dict), set()
    with path.open("rb") as binary:
        require(
            binary.seek(0, 2) == expected_bytes
            and binary.seek(0) == 0
            and hashlib.file_digest(binary, "sha256").hexdigest() == expected_digest,
            "Pinned publisher file changed",
        )
        binary.seek(0)
        with io.TextIOWrapper(binary, encoding="utf-8", newline="") as text:
            for row in csv.DictReader(text, strict=True):
                require(
                    row["generator"] == generator
                    and row["split"] == "test"
                    and row["domain"] == "reports"
                    and row["subset"] == "main"
                    and row["ablation_type"] == ""
                    and ID.fullmatch(row["id"])
                    and row["record_id"] not in seen_records,
                    "Unexpected source or duplicate publisher record",
                )
                seen_records.add(row["record_id"])
                version = row["version"]
                if version in VERSIONS:
                    group = selected[row["id"]]
                    require(version not in group, "Duplicate selected version in a report group")
                    group[version] = {
                        k: row[k]
                        for k in (
                            "record_id",
                            "document_hash_id",
                            "generator",
                            "edit_operation",
                            "version",
                            "version_index",
                            "text",
                            "text_tagged",
                            "ai_spans_char",
                        )
                    }
    require(len(selected) == expected_groups, "Publisher report inventory differs")
    return selected


def common_groups(inventories):
    first, second = inventories.values()
    common = sorted(first.keys() & second.keys())
    require(len(common) == 30, "Expected 30 common publisher report IDs")
    groups = []
    for source_id in common:
        groups_by_generator = [panel[source_id] for panel in inventories.values()]
        require(
            all(set(rows) == VERSIONS.keys() for rows in groups_by_generator), "Incomplete common version set"
        )
        hashes = {row["document_hash_id"] for rows in groups_by_generator for row in rows.values()}
        require(len(hashes) == 1 and all(hashes), "Publisher source group identity differs")
        versions = [[record(rows[version]) for version in VERSIONS] for rows in groups_by_generator]
        seed, other_seed = (rows[0] for rows in versions)
        require(seed["text"] == other_seed["text"], "Publishers disagree on the unchanged source")
        rewrites = [case for rows in versions for case in rows[1:]]
        require(
            all(retained_prose_matches(seed["text"], case) for case in rewrites),
            "Untagged prose must occur in the same order in its unchanged source",
        )
        groups.append(
            {
                "source_id": source_id,
                "publisher_group": next(iter(hashes)),
                "second_seed_record_id": other_seed["record_id"],
                "cases": [seed, *rewrites],
            }
        )
    require(
        len({group["publisher_group"] for group in groups})
        == len({g["cases"][0]["text_sha256"] for g in groups})
        == 30,
        "Common source groups must have distinct publisher identities and source texts",
    )
    return groups


def retained_prose_matches(source, rewrite):
    source = exposure.text.canonical(source)
    boundaries = [0, *(edge for span in rewrite["ai_spans"] for edge in span), len(rewrite["text"])]
    cursor = 0
    for start, end in zip(boundaries[::2], boundaries[1::2], strict=True):
        segment = exposure.text.canonical(rewrite["text"][start:end])
        if not segment:
            continue
        match = source.find(segment, cursor)
        if match < 0:
            return False
        cursor = match + len(segment)
    return True


def known_fragments(cache, gohumanize):
    index, fragments = exposure.build_exposure(exposure.exposure_packets(cache, gohumanize))
    require(
        len(index["packets"]) == exposure.PACKET_COUNT
        and index["text_entries"] == exposure.TEXT_COUNT
        and sha(exposure.prior.json_bytes(index)) == EXPOSURE_SHA,
        "Exact previously retained v243 exposure index required",
    )
    return index, fragments


def select(groups, exposure_fragments):
    selected, signatures = [], {}
    dispositions = Counter()
    for group in sorted(groups, key=lambda group: sha(f"v257:{group['source_id']}".encode())):
        fragments = [exposure.text.fingerprints(case["text"]) for case in group["cases"]]
        family = set.union(*fragments)
        if fragments[0] & exposure_fragments:
            dispositions["seed_fragment_match"] += 1
        if family & exposure_fragments:
            dispositions["family_fragment_match"] += 1
            continue
        dispositions["novel_groups"] += 1
        if len(selected) < MAX_GROUPS:
            selected.append(group)
            signatures[group["source_id"]] = family
    pairwise_shared = sum(bool(a & b) for a, b in combinations(signatures.values(), 2))
    return selected, {
        "known_seed_fragment_matches": dispositions["seed_fragment_match"],
        "known_family_fragment_matches": dispositions["family_fragment_match"],
        "novel_groups": dispositions["novel_groups"],
        "selected_source_pairs_sharing_fragment": pairwise_shared,
        "selected_source_pairs_compared": len(selected) * (len(selected) - 1) // 2,
    }


def bind_source_records(groups, records):
    expected = {group["source_id"]: group["cases"][0]["text"] for group in groups}
    seen = {}
    for row in records:
        source_id = f"GAO_{row['id']}"
        upstream = " ".join(
            paragraph
            for section in row["highlight"]
            if section["section_title"] != "What GAO Recommends"
            for paragraph in section["paragraphs"]
        )
        require(
            source_id not in seen
            and source_id in expected
            and exposure.text.canonical(upstream) == exposure.text.canonical(expected[source_id]),
            "Publisher seed must match the independent GovReport source excerpt",
        )
        seen[source_id] = row["url"]
    require(seen.keys() == expected.keys(), "Every common source requires exactly one GovReport record")
    return seen


def source_authority(groups):
    receipt = exposure.prior.private_bytes(DATA / "govreport-source-receipt.json")
    source = exposure.text.read_file(DATA / "govreport-30-source-records.jsonl", AUTHORITY_RECORD_BYTES)
    require(
        sha(receipt) == AUTHORITY_RECEIPT_SHA
        and json.loads(receipt) == AUTHORITY
        and len(source) == AUTHORITY_RECORD_BYTES
        and sha(source) == AUTHORITY["private_upstream_packet_sha256"],
        "Exact independent upstream source receipt and retained packet required",
    )
    return bind_source_records(groups, exposure.json_lines(source))


def scoring_rows(selected):
    for ordinal, group in enumerate(selected, start=1):
        article_group = f"report-{ordinal:02d}"
        for case in group["cases"]:
            generator = case["generator"] or "shared_human"
            arm = f"{case['edit_operation']}_ai" if case["generator"] else "human"
            yield {
                "case": f"{article_group}-{generator}-{arm}",
                "article_group": article_group,
                "generator": generator,
                "arm": arm,
                "status": "constructed",
                "text": case["text"],
                "text_sha256": case["text_sha256"],
                "ai_spans": case["ai_spans"],
            }


def run(cache, gohumanize):
    csv.field_size_limit(max(spec[1] for spec in FILES.values()))
    inventories = {key: inventory(DATA, key, spec) for key, spec in FILES.items()}
    groups = common_groups(inventories)
    witnessed = source_authority(groups)
    index, prior = known_fragments(cache, gohumanize)
    selected, dispositions = select(groups, prior)
    plan = {
        "schema": "chipmunk.text-opai-bench-private-plan/257",
        "publisher_revision": REVISION,
        "groups": [
            {
                "article_group": f"report-{ordinal:02d}",
                "source_id": group["source_id"],
                "publisher_group": group["publisher_group"],
                "govreport_source_url": witnessed[group["source_id"]],
                "publisher_seed_record_ids": [group["cases"][0]["record_id"], group["second_seed_record_id"]],
                "publisher_rewrite_record_ids": [case["record_id"] for case in group["cases"][1:]],
            }
            for ordinal, group in enumerate(selected, start=1)
        ],
    }
    common_cases = [case for group in groups for case in group["cases"]]
    selected_cases = [case for group in selected for case in group["cases"]]
    journal = b"".join(map(exposure.prior.json_bytes, scoring_rows(selected)))
    return (
        plan,
        journal,
        {
            "schema": "chipmunk.text-opai-bench-intake/257",
            "status": "complete_intake",
            "publisher_revision": REVISION,
            "publisher_files": {key: {"bytes": spec[1], "sha256": spec[2]} for key, spec in FILES.items()},
            "publisher_report_groups_by_generator": {key: len(panel) for key, panel in inventories.items()},
            "common_report_groups": len(groups),
            "publisher_rows_validated": len(common_cases) + len(groups),
            "rewrites_with_untagged_prose_bound_to_seed": sum(
                case["version"] != "v0" for case in common_cases
            ),
            "common_cases_with_non_ascii_text": sum(not case["text"].isascii() for case in common_cases),
            "known_exposure_packets": len(index["packets"]),
            "known_exposure_texts": index["text_entries"],
            "known_exposure_fragments": index["unique_fragments"],
            "known_exposure_index_sha256": EXPOSURE_SHA,
            **dispositions,
            "selected_groups": len(selected),
            "selected_human_seeds": sum(case["version"] == "v0" for case in selected_cases),
            "selected_rewrite_counts_by_generator": dict(
                Counter(case["generator"] for case in selected_cases if case["generator"])
            ),
            "selected_cases": len(selected_cases),
            "independent_source_receipt_sha256": AUTHORITY_RECEIPT_SHA,
            "common_seeds_with_exact_normalized_govreport_match": len(witnessed),
            "selected_seeds_with_exact_normalized_govreport_match": sum(
                group["source_id"] in witnessed for group in selected
            ),
            "direct_individual_gao_report_verification": False,
            "private_plan_sha256": sha(exposure.prior.json_bytes(plan)),
            "private_cases_sha256": sha(journal),
            "new_neural_calls": 0,
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gohumanize_source", type=Path)
    parser.add_argument("--research-cache", type=Path, default=Path("/private/tmp"))
    parser.add_argument(
        "--retain", action="store_true", help="Retain the exact private case journal and attribution."
    )
    args = parser.parse_args()
    plan, journal, result = run(args.research_cache, args.gohumanize_source)
    if args.retain:
        exposure.archive.safe_cache(DATA)
        exposure.archive.retain(DATA / "private-plan.json", exposure.prior.json_bytes(plan))
        exposure.archive.retain(DATA / "private-cases.jsonl", journal)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
