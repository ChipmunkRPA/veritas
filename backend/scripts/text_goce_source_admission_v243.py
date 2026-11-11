"""Verify and record two reviewed historical copy-edit controls."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

if __package__:
    from . import text_goce_source_followup_v235 as followup
    from . import text_gohumanize_intake_v193 as gohumanize
else:
    import text_goce_source_followup_v235 as followup
    import text_gohumanize_intake_v193 as gohumanize

prior = followup.prior
archive, intake, text = prior.archive, prior.intake, prior.text
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "backend/data"
PANELS = ROOT / "docs/evaluation-panels"
FOLLOWUP_PUBLIC_SHA = "1b4677aeb85dbb0dfecf07c96f4ad44685f67f3db46110e002e736cfd24f284e"
REVIEW_METHOD = "assistant_assisted_source_inspection"
LICENSE_URL = "https://creativecommons.org/licenses/by-sa/3.0/"
REVIEWED = {(1, 0, 0): (5, 6), (2, 0, 0): (30, 31)}
HELD = {
    (0, 0, 0): "earlier_prose_boundary_unseen",
    (0, 1, 0): "template_in_retouch",
    (2, 1, 0): "earlier_prose_boundary_unseen",
    (2, 1, 1): "earlier_prose_boundary_unseen",
    (3, 0, 0): "credited_revision_words_in_full_context",
}
KEY_FIELDS = ("page_id", "before_revision", "after_revision", "window_sha256")
MANIFESTS = {
    "manifest.json": ("cases", "e10769c188e69889c54f1116c5a4f8f03cbb58f4194a19a3a1ea9e969a92d28c"),
    "article-confirmation-manifest.json": (
        "cases",
        "1aa08df98620fbe9d251f978ee6f61042cd7aa9d2f98937ec0f35f16a3b48d41",
    ),
    "email-originals-v2.json": (
        "originals",
        "5442c89d86f7c8d3a052df7f39ed6df8fbb6f0f32dd7fb286cff240ee2745f93",
    ),
    "email-manifest-explicit.json": (
        "cases",
        "0d9847ca82940beadcf14660d33d981d230950dff57f58f2d770f6067acf2329",
    ),
    "email-manifest-v2.json": ("cases", "593b70240923772b883e72a1ff9b14d59f6e8a2e8e8993c11658d3dfced01438"),
    "email-originals.json": ("originals", "92f89bf520197aa2e6e9703814fe0ee5169720f2defcee9be5b464d24474ef82"),
}
EMAIL_JOURNALS = {
    "email-qwen-explicit.jsonl": "2eda831c9fa9f2db5e2b3c2cf6a80a0b8e55cddbfd0aecb31903675fb27817fe",
    "email-smol2-explicit.jsonl": "3921dcdccc1aaa732408210776ee18c17c6d634a3889f6ff209104d3ad3724e7",
    "email-qwen-mlx.jsonl": "ac66fda8bd896fafe1102fd6f8e2a1c9fff3c5a6cc3789ca9448a28be4e9cff3",
    "email-smol2-mlx.jsonl": "86c0f4b58bbfd4166a82586ac61fc58a595170b67992661ada3812022d208d1e",
}
WIKINEWS_RENDERS = {
    "archive-v205": "7409f1ddcff8714fb5fc0ba05a6c1c3cbd0d479ee97cc7b0762b62fc07b3f7c5",
    "direct-v209": "de8a8afd0022cd0b679c5d985017b7f9ee6f584f96d986130daa82471e649ef5",
}
AITDNA_SPANS_SHA = "0641d20f9326522061bee792cad5f8963faf26bf37268e491f901cb7c2ce5cfb"
PACKET_COUNT, TEXT_COUNT = 20, 6364


def read_json(path):
    return json.loads(prior.private_bytes(path))


def checked_bytes(path, expected):
    raw = text.read_file(path, 24_000_000)
    text.require(text.sha(raw) == expected, "A fixed writing exposure packet changed")
    return raw


def json_lines(raw):
    return (json.loads(line) for line in raw.splitlines())


def digest_hashes(hashes):
    digest = hashlib.sha256()
    for item in hashes:
        digest.update(bytes.fromhex(item))
    return digest.hexdigest()


def validated_texts(rows):
    for row in rows:
        value = row["text"]
        text.require(
            isinstance(value, str) and text.sha(value.encode()) == row["text_sha256"], "Text hash differs"
        )
        yield value


def wikinews_exposure(folder, expected):
    raw = [prior.private_bytes(path) for path in sorted(folder.glob("render-*.json"))]
    text.require(digest_hashes(map(text.sha, raw)) == expected, "A fixed Wikinews render inventory changed")
    for body in raw:
        packet = json.loads(body)
        page = packet["parse"]
        yield intake.prose.article_prose(packet, page["pageid"], page["revid"])[0]


def email_generation(raw, originals, original_sha):
    rows = json_lines(raw)
    header = next(rows)
    text.require(header["originals_sha256"] == original_sha, "Federal generation original manifest differs")
    for row in rows:
        text.require(
            row["input_text_sha256"] == originals[row["id"]], "Federal generation source hash differs"
        )
        if row["text"] is not None:
            yield from validated_texts((row,))


def exposure_packets(cache, gohumanize_source):
    independent = DATA / "independent-evaluation"
    for label, (private, public, key) in gohumanize.HISTORY.items():
        expected = read_json(PANELS / public)[key]
        yield (
            label,
            expected,
            gohumanize.historical_texts(label, checked_bytes(independent / private, expected)),
        )
    expected = read_json(PANELS / "text-layout-methods-freeze-v101.json")["input_packet_sha256"]
    raw = checked_bytes(independent / "text-format-audit-v93/inputs.jsonl", expected)
    yield (
        "text_layout_v93_v101",
        expected,
        (item["text"] for row in json_lines(raw) for item in row["effective"]),
    )
    expected = read_json(PANELS / "text-gohumanize-intake-complete-v193.json")["private_cases_sha256"]
    raw = checked_bytes(independent / "text-gohumanize-intake-v193/preparation/private-cases.jsonl", expected)
    yield (
        "gohumanize_prepared_v193_v196",
        expected,
        validated_texts(row for row in json_lines(raw) if row["text"] is not None),
    )
    upgrade = DATA / "text-upgrade"
    for name, (key, expected) in MANIFESTS.items():
        rows = json.loads(checked_bytes(upgrade / name, expected))[key]
        yield name, expected, validated_texts(rows)
    key, original_sha = MANIFESTS["email-originals-v2.json"]
    originals = {
        row["id"]: row["text_sha256"]
        for row in json.loads(checked_bytes(upgrade / "email-originals-v2.json", original_sha))[key]
    }
    for name, expected in EMAIL_JOURNALS.items():
        yield (
            name,
            expected,
            email_generation(checked_bytes(upgrade / name, expected), originals, original_sha),
        )
    expected = gohumanize.ASSETS["test.jsonl"]
    raw = checked_bytes(gohumanize_source, expected)
    yield (
        "gohumanize_publisher_test",
        expected,
        (row[key] for row in json_lines(raw) for key in ("input", "output")),
    )
    spans = upgrade / "sources/ait-span.parquet"
    checked_bytes(spans, AITDNA_SPANS_SHA)
    from pyarrow.parquet import ParquetFile

    batches = ParquetFile(spans).iter_batches(batch_size=16, columns=["data"])
    yield (
        "aitdna_span_texts_v2_v4",
        AITDNA_SPANS_SHA,
        ("".join(part["text"] for part in row["data"]) for batch in batches for row in batch.to_pylist()),
    )
    for suffix, expected in WIKINEWS_RENDERS.items():
        yield (
            f"wikinews_{suffix}",
            expected,
            wikinews_exposure(cache / f"veritas-wikinews-{suffix}", expected),
        )
    raw = checked_bytes(cache / "veritas-iterater-train-v203.json", text.TRAIN_SHA256)
    yield (
        "iterater_v203_publisher_train",
        text.TRAIN_SHA256,
        (row[key] for row in json_lines(raw) for key in ("before_revision", "after_revision")),
    )


def build_exposure(packets):
    receipts, combined = [], set()
    for name, source_hash, texts in packets:
        fragments, count = set(), 0
        for value in texts:
            fragments.update(text.fingerprints(value))
            count += 1
        text.require(count > 0 and fragments, "An expected writing packet supplied no fragments")
        receipts.append(
            {
                "packet": name,
                "source_sha256": source_hash,
                "text_entries": count,
                "unique_fragments": len(fragments),
                "fragments_sha256": digest_hashes(sorted(fragments)),
            }
        )
        combined.update(fragments)
    return {
        "schema": "chipmunk.text-goce-private-exposure-index/243",
        "normalization": "v203.fingerprints",
        "text_entries": sum(row["text_entries"] for row in receipts),
        "unique_fragments": len(combined),
        "combined_fragments_sha256": digest_hashes(sorted(combined)),
        "packets": receipts,
    }, combined


def source_evidence(cache):
    groups, packets, prior_hash = followup.verified_prior(
        cache / "veritas-wikipedia-goce-v228", cache / "veritas-wikipedia-goce-rights-v233-attempt01"
    )
    plan = followup.private_plan(groups, packets, prior_hash)
    path = cache / "veritas-wikipedia-goce-rights-v235-attempt01"
    text.require(path.is_dir(), "The completed v235 cache is required")
    archive.safe_cache(path)
    ledger = followup.PriorLedger(intake.OfficialArchive(path, offline=True))
    result = followup.collect(ledger, plan)
    public = checked_bytes(PANELS / "text-goce-source-followup-complete-v235.json", FOLLOWUP_PUBLIC_SHA)
    text.require(
        json.loads(public) == result
        and prior.private_bytes(path / "aggregate.json") == prior.json_bytes(result)
        and prior.private_bytes(path / "private-plan.json") == prior.json_bytes(plan)
        and ledger.source.requests == 0,
        "Published v235 receipt and private evidence did not replay together",
    )
    return groups, [*packets, *ledger.packets], result


def source_body(rows, revision):
    row = rows[revision]
    text.require(prior.visible_wikitext(row), "Source review requires visible historical wikitext")
    return text.sha(row["slots"]["main"]["content"].encode())


def source_citation(pair, location, packets, introduction, commitments):
    page, before, after = pair["page_id"], pair["before_revision"], pair["after_revision"]
    known, history, attribution = {}, {}, None
    for purpose, query, packet in packets:
        if purpose == "article_attribution":
            attribution = next(info for info in prior.page_rows(packet) if info["pageid"] == page)
        elif purpose == "exact_article_wikitext" and before in map(int, query["revids"].split("|")):
            known.update({row["revid"]: row for row in prior.revisions_on_page(packet, page, 0)})
        elif purpose == "article_parent_wikitext" and query["pageids"] == page:
            for row in prior.revisions_on_page(packet, page, 0):
                text.require(history.get(row["revid"], row) == row, "Overlapping source bodies differ")
                history[row["revid"]] = row
    text.require(
        attribution is not None and known[after]["parentid"] == before, "Exact source attribution is missing"
    )
    chain = sorted(history.values(), key=lambda row: (row["timestamp"], row["revid"]), reverse=True)
    start = next(i for i, row in enumerate(chain) if row["revid"] == before)
    child_idx, parent_idx = introduction
    retained = chain[start : start + parent_idx + 1]
    text.require(
        len(retained) == parent_idx + 1
        and all(a["parentid"] == b["revid"] for a, b in zip(retained, retained[1:])),
        "The reviewed source introduction chain differs",
    )
    introduced, older = retained[child_idx], retained[parent_idx]
    text.require(
        intake.safe_revision(introduced) and older["revid"] == introduced["parentid"],
        "The reviewed introduction does not bind a registered source edit",
    )
    return {
        **location,
        "title": attribution["title"],
        "canonical_url": attribution["canonicalurl"],
        "revision_urls": [
            f"https://en.wikipedia.org/w/index.php?oldid={revision}" for revision in (before, after)
        ],
        "contributor_history_url": f"https://en.wikipedia.org/w/index.php?curid={page}&action=history",
        "source_license_url": LICENSE_URL,
        "source_policy": intake.SOURCE_POLICY,
        "attribution_note": "Historical Wikipedia excerpts attributed to the contributors in the revision history.",
        "modification_note": "The historical versions are excerpted with display formatting removed.",
        "source_scope": "Original main-body prose with exact offsets in both archived revisions.",
        "before_after_render_sha256": pair["render_sha256"],
        "before_after_wikitext_sha256": [source_body(known, revision) for revision in (before, after)],
        "introduction_revision_urls": [
            f"https://en.wikipedia.org/w/index.php?oldid={row['revid']}" for row in (older, introduced)
        ],
        "introduction_wikitext_sha256": [source_body(history, row["revid"]) for row in (older, introduced)],
        "evidence_commitments": commitments,
    }


def review_records(groups, packets, evidence, fragments):
    approved, held, citations, planned = [], [], [], []
    seen = set()
    for group_index, pairs in enumerate(groups.values()):
        planned.append({"pairs": pairs})
        for pair_index, pair in enumerate(pairs):
            for window_index, window in enumerate(pair["windows"]):
                selector = group_index, pair_index, window_index
                seen.add(selector)
                key = intake.prose.review_key(pair, window)
                location = dict(zip(KEY_FIELDS, key, strict=True))
                if selector in REVIEWED:
                    both_sides = text.fingerprints(window["before"]["excerpt"]) | text.fingerprints(
                        window["after"]["excerpt"]
                    )
                    text.require(
                        not both_sides & fragments, "A reviewed source has a known project fragment match"
                    )
                    citations.append(source_citation(pair, location, packets, REVIEWED[selector], evidence))
                    approved.append(
                        {
                            **location,
                            "decision": "approved",
                            "source_license": intake.SOURCE_LICENSE,
                            "source_policy": intake.SOURCE_POLICY,
                            "story_component": text.sha(f"en.wikipedia.org:page:{key[0]}".encode()),
                            "checks": dict.fromkeys(intake.RIGHTS_CHECKS, True),
                        }
                    )
                else:
                    text.require(selector in HELD, "The frozen source review denominator differs")
                    held.append({**location, "decision": "rights_unresolved", "reason": HELD[selector]})
    text.require(seen == set(REVIEWED) | set(HELD), "The frozen source review denominator differs")
    return (
        {"review_method": REVIEW_METHOD, "external_human_review": False, "approved": approved, "held": held},
        citations,
        planned,
    )


def apply_review(planned, review, fragments):
    result = intake.rights_review(planned, review["approved"], fragments)
    text.require(
        result["rights_admitted_pages"] == len(REVIEWED)
        and result["pending_windows"] == len(HELD)
        and not result["rights_exclusions"],
        "The admitted or held source scope differs",
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("preflight", "admit", "replay"))
    parser.add_argument("research_cache", type=Path)
    parser.add_argument("gohumanize_source", type=Path)
    parser.add_argument("output_cache", nargs="?", type=Path)
    args = parser.parse_args()
    if (args.mode == "preflight") == bool(args.output_cache):
        parser.error("Supply an output cache for admission or replay only")
    groups, packets, evidence = source_evidence(args.research_cache)
    exposure, fragments = build_exposure(exposure_packets(args.research_cache, args.gohumanize_source))
    text.require(
        len(exposure["packets"]) == PACKET_COUNT and exposure["text_entries"] == TEXT_COUNT,
        "The fixed historical exposure denominator differs",
    )
    evidence_commitments = {
        "v233_public_sha256": evidence["v233_public_sha256"],
        "v233_private_commitment_sha256": evidence["v233_private_commitment_sha256"],
        "v235_public_sha256": FOLLOWUP_PUBLIC_SHA,
        "v235_private_commitment_sha256": evidence["private_evidence_commitment_sha256"],
    }
    review, citations, planned = review_records(groups, packets, evidence_commitments, fragments)
    private = {
        "private-attribution.json": citations,
        "private-exposure-index.json": exposure,
        "private-review.json": review,
    }
    aggregate = {
        "schema": "chipmunk.text-goce-source-admission/243",
        "status": "verified_preflight",
        "review_method": REVIEW_METHOD,
        "external_human_review": False,
        "ready_witnesses": len(REVIEWED),
        "held_witnesses": len(HELD),
        "held_reasons": dict(Counter(row["reason"] for row in review["held"])),
        "known_exposure_packets": PACKET_COUNT,
        "known_exposure_text_entries": TEXT_COUNT,
        "known_exposure_unique_fragments": len(fragments),
        "reviewed_witnesses_with_known_fragment_matches": 0,
        "rights_admitted_pages": 0,
        "private_artifacts_commitment_sha256": text.sha(prior.json_bytes(private)),
        "model_calls": 0,
    }
    if args.mode != "preflight":
        cache = args.output_cache
        replay = args.mode == "replay"
        text.require(
            cache.is_dir() if replay else not cache.exists(), "Use a fresh admission cache or replay it"
        )
        archive.safe_cache(cache)
        aggregate.update(apply_review(planned, review, fragments), status="complete")
        private["aggregate.json"] = aggregate
        for name, value in private.items():
            body = prior.json_bytes(value)
            if replay:
                text.require(prior.private_bytes(cache / name) == body, "Private admission replay differs")
            else:
                archive.retain(cache / name, body)
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
