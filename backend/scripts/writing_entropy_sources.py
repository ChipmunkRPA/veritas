# SPDX-License-Identifier: Apache-2.0
"""Prepare source-separated article originals and annotated rewrites for entropy testing."""

from __future__ import annotations

import argparse
import gzip
import re
from collections import Counter, defaultdict
from pathlib import Path

from scripts import writing_local_baseline_transfer as paired
from scripts import writing_multi_passage_panel as panel
from scripts import writing_profile_sources as profile
from scripts import writing_report_sources as gao
from scripts.writing_short_sources import SHARDS

SCHEMA = "chipmunk.writing-entropy-sources/1"
PUBLIC = Path(__file__).resolve().parents[2] / "docs/evaluation-panels"
LIMIT_PER_GENRE = 30
LIMIT_PER_REPORT_GENERATOR = 24


def ordered(value):
    return panel.digest(f"{SCHEMA}/{value}".encode())


def original_layout(work):
    return work | {"variants": [version for version in work["variants"] if version["layout"] == "published"]}


def exposures(exact=None):
    source_ids, digests, commitments = set(), set(), {}
    paths = sorted(PUBLIC.iterdir()) if exact is None else sorted(PUBLIC / name for name in exact)
    for path in paths:
        if not path.name.startswith(("writing-", "text-", "document-gao-")) or path.suffix not in {".gz", ".json"}:
            continue
        if path.name.startswith("writing-entropy-prominence-"):
            continue
        data = path.read_bytes()
        commitments[path.name] = panel.digest(data)
        raw = gzip.decompress(data) if path.suffix == ".gz" else data
        source_ids.update(match.decode().upper() for match in re.findall(
            rb"(?:GAO_|gao\.gov/products/)([a-z]+-[0-9a-z-]+)", raw, re.I))
        digests.update(match.decode() for match in re.findall(rb"\b[0-9a-f]{64}\b", raw))
    if exact is not None and commitments != exact:
        raise ValueError("Earlier source exposure receipts differ from the frozen list")
    return source_ids, digests, commitments


def novel(work, source_ids, digests):
    return (work["document"].removeprefix("GAO_").upper() not in source_ids
            and work["group_sha256"] not in digests
            and all(version["native_text_sha256"] not in digests for version in work["variants"]))


def mixed_sources(path, source_ids, digests):
    import pyarrow.parquet as pq

    if panel.digest(path.read_bytes()) != profile.PUBLISHER_SHA:
        raise ValueError("Mixed-source input differs from the pinned publisher file")
    candidates = defaultdict(list)
    for row in pq.read_table(path, filters=[("source_dataset", "in", sorted(panel.GENRES))]).to_pylist():
        try:
            group = panel.digest(panel._source_id(row).encode())
        except AssertionError:
            continue
        if group not in digests:
            candidates[group].append(row)
    results, genres, families = [], Counter(), Counter()
    for group in sorted(candidates, key=ordered):
        for row in sorted(candidates[group], key=lambda row: (families[row["generator_backend"]], ordered(row["id"]))):
            genre = panel.GENRES[row["source_dataset"]]
            if genres[genre] >= LIMIT_PER_GENRE or row["split"] != "train":
                continue
            panel._validate(row)
            versions = paired.variants(row)
            if versions is None:
                continue
            work = {key: row[key] for key in ("source_reference", "source_license", "model_identity_status",
                                             "generator_backend", "requested_model")}
            work.update(document=row["id"], group_sha256=group, genre=genre, split="test", publisher_split=row["split"],
                        variants=versions)
            if novel(work, source_ids, digests):
                results.append(original_layout(work))
                genres[genre] += 1
                families[row["generator_backend"]] += 1
                break
    return results


def report_sources(folder, pack, source_ids, digests):
    raw = pack.read_bytes()
    if panel.digest(raw) != gao.GAO_SHA256:
        raise ValueError("Official report pack differs from the pinned revision")
    archive = panel.read(pack)["reports"]
    selected, families = [], Counter()
    for role, split, family, _, digest in SHARDS:
        if role == "validation":
            continue
        file = folder / "default" / split / f"reports_{family}.csv"
        pairs = gao.test_pairs(file, family, split, digest)
        for key in sorted(pairs, key=ordered):
            if families[family] >= LIMIT_PER_REPORT_GENERATOR:
                break
            identifier = key.removeprefix("GAO_").upper()
            if identifier in source_ids or panel.digest(key.encode()) in digests or set(pairs[key]) != {"v0", "v1"}:
                continue
            work, _ = gao.verified_gao_work(key, pairs[key], archive.get(identifier), family, split, max_edits=4,
                                            changed_only=True)
            if work is not None and novel(work, source_ids, digests):
                selected.append(original_layout(work | {"generator_backend": family}))
                families[family] += 1
                source_ids.add(identifier)
        del pairs
    return selected


def prepare(args):
    exact = panel.read(args.reference)["exposed"] if args.reference else None
    source_ids, digests, previous = exposures(exact)
    development = [original_layout(work) for path in (args.older_mixed, args.older_reports)
                   for work in panel.read(path)["works"]]
    groups = {role: {work["group_sha256"] for work in development if work["split"] == role}
              for role in ("validation", "test")}
    if groups["validation"] & groups["test"]:
        raise ValueError("Older fitting and selection work groups overlap")
    digests.update(group for values in groups.values() for group in values)
    mixed = mixed_sources(args.mixed, source_ids, digests)
    reports = report_sources(args.reports, args.gao, source_ids, digests)
    fresh = mixed + reports
    if len({work["group_sha256"] for work in fresh}) != len(fresh):
        raise ValueError("Fresh source groups overlap")
    for cohort, samples in (("older", development), ("fresh", fresh)):
        counts = Counter((work["split"], work["genre"]) for work in samples)
        print(cohort, len(samples), dict(counts), flush=True)
        panel.save(args.output / f"{cohort}-plan.json.gz", {"schema": SCHEMA, "cohort": cohort, "works": samples,
                   "exposed": previous, "mixed_publisher_sha256": profile.PUBLISHER_SHA, "report_archive_sha256": gao.GAO_SHA256})


def add_transfer(older_path, fresh_path, transfer_path, output):
    older, fresh, additional = (panel.read(path) for path in (older_path, fresh_path, transfer_path))
    used = older["works"] + fresh["works"]
    groups = {work["group_sha256"] for work in used}
    texts = {version["native_text_sha256"] for work in used for version in work["variants"]}
    transfer = [original_layout(work) for work in additional["works"]]
    if not transfer or any(work["split"] != "test" or work["group_sha256"] in groups
                           or any(row["native_text_sha256"] in texts for row in work["variants"]) for work in transfer):
        raise ValueError("The transfer cases overlap with the calibration, selection or new reports")
    works = [work | {"evaluation_panel": "fresh_reports"} for work in fresh["works"]]
    works.extend(work | {"evaluation_panel": "earlier_layout_transfer"} for work in transfer)
    panel.save(output, fresh | {"transfer_plan_sha256": panel.digest(transfer_path.read_bytes()), "works": works})
    print("held", dict(Counter(work["evaluation_panel"] for work in works)), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("mixed", "reports", "gao", "older_mixed", "older_reports", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--reference", type=Path)
    prepare(parser.parse_args())
