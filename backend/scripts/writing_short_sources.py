"""Select fresh, source-matched reports for short passage localization."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from scripts import writing_multi_passage_panel as panel
from scripts import writing_report_sources as reports

SCHEMA = "chipmunk.writing-short-passage-sources/1"
MAX_EDITS = 4
PUBLIC = Path(__file__).resolve().parents[2] / "docs/evaluation-panels"
PREVIOUS = PUBLIC / "writing-report-confirmation-2026-11-16.json.gz"
SHARDS = (
    ("validation", "train", "gpt-5.4", 32, "a205bd7102225997b9e2245589cdf416e42499677b31f76382e28bd1b0b9fa68"),
    ("test", "test", "gpt-5.4", 24, "9e28d157d336a343106487c8acab242ee41829f49ceb748dcb5f9fd6a9f9fb33"),
    ("test", "train", "gemini-2.5-flash", 36, "c4af2313124b86f5003f6635bf451960f6cbd015a849fb33cd35935c89cf778e"),
    ("test", "train", "gpt-5.4", 28, "a205bd7102225997b9e2245589cdf416e42499677b31f76382e28bd1b0b9fa68"),
)


def collect(shards, archive, output, *, receipt=None):
    raw = archive.read_bytes()
    if panel.digest(raw) != reports.GAO_SHA256:
        raise ValueError("The GAO highlights archive differs from the verified publisher pack")
    authority = json.loads(raw)["reports"]
    previous_sha = panel.digest(PREVIOUS.read_bytes())
    commitments = None
    if receipt is not None:
        commitments = panel.read(receipt)["held_metadata"]["earlier_evaluation_digests"].copy()
        if commitments.pop(PREVIOUS.name, None) != previous_sha:
            raise ValueError("The previous report comparison differs from the study receipt")
    seen, digests, exposure_files = reports.exposed_reports(PUBLIC, commitments=commitments)
    for work in panel.read(PREVIOUS)["works"]:
        if work["document"].startswith("GAO_"):
            seen.add(work["document"][4:].upper())
        digests.add(work["group_sha256"])
        digests.update(variant["native_text_sha256"] for variant in work["variants"])
    exposure_files[PREVIOUS.name] = previous_sha
    results, intake, included = [], {}, {}
    for role, split, family, limit, expected in SHARDS:
        path = shards / f"default/{split}/reports_{family}.csv"
        entries = reports.test_pairs(path, family, split, expected)
        key = f"{role}/{split}/{family}"
        included[key] = expected
        rejection = Counter()
        count = 0
        for document in sorted(entries, key=lambda item: hashlib.sha256(f"{SCHEMA}/{key}/{item}".encode()).digest()):
            if count >= limit:
                break
            identifier = document.removeprefix("GAO_").upper()
            versions = entries[document]
            if set(versions) != {"v0", "v1"}:
                rejection["incomplete_pair"] += 1
                continue
            seed, rewrite = versions["v0"], versions["v1"]
            source_hashes = {panel.digest(value.encode()) for value in
                             (document, identifier, seed["text"], rewrite["text"], seed["document_hash_id"])}
            if identifier in seen or source_hashes & digests:
                rejection["earlier_source"] += 1
                continue
            work, reason = reports.verified_gao_work(
                document, versions, authority.get(identifier), family, split, max_edits=MAX_EDITS, changed_only=True,
            )
            if work is None:
                rejection[reason] += 1
                continue
            work["split"] = role
            if any(variant["native_text_sha256"] in digests for variant in work["variants"]):
                rejection["earlier_text"] += 1
                continue
            seen.add(identifier)
            digests.update(source_hashes)
            results.append(work)
            count += 1
        intake[key] = {"admitted": count, "examined_rejections": dict(rejection)}
        print(key, count, flush=True)
    panel.save(output, {"schema": SCHEMA, "publisher_revision": reports.OPAI_REVISION, "max_changed_sentences": MAX_EDITS,
                        "publisher_files": included, "govreport_pack_sha256": reports.GAO_SHA256,
                        "earlier_evaluation_digests": exposure_files, "intake": intake, "works": results})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("shards", "govreport", "output"):
        parser.add_argument(key, type=Path)
    parser.add_argument("--receipt", type=Path, help="Reuse the exact earlier exposure list from a saved comparison")
    args = parser.parse_args()
    collect(args.shards, args.govreport, args.output, receipt=args.receipt)
