"""Retry the frozen BG-REAL roster through the observed publisher transport."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from urllib.parse import urljoin, urlsplit

from scripts import image_bg_real_acquisition_v221 as original
from scripts.video_appearance_corrected_v112 import scoped

ROOT, DATA, PUBLIC = original.ROOT, original.DATA, original.PUBLIC
PREVIOUS = DATA / "image-bg-real-acquisition-v221"
PREVIOUS_FREEZE = PUBLIC / "image-bg-real-acquisition-freeze-v221.json"
PREVIOUS_FREEZE_SHA = "6ad677b87095a16da6745ac04c33801a47087a81d10dc7a0da2c52c4854e1cd1"
FAILED_RECEIPTS = {
    "plan.json": "61668d30878fc8ebf72aa2a9030e0ba8f2349baf802de0f1c1b4a0cd23dee2f2",
    "acquired.json": "9e4db67da8326593fc868b6cd3401ef62614b55920a6b195890a4815da3d985a",
    "attempts.jsonl": "4a9d4b174e094f77330500ee3b8b85c8441daa0023e21e49be96fcb9ac31de91",
    "qualified-private.json": "ade10c31c42e3ccc477e06b1383fbe2c6ee2438b87500ecd775cdfa66c4060ea",
    "summary.json": "d33a4a70ce4ce88a36ba3778178e01df6a3b628d14bcf54f0f11f360090b1054",
}
AWS = "us.aws.cdn.hf.co"
AWS_PATH = re.compile(r"/xet-bridge-us/[0-9a-f]{24}/[0-9a-f]{64}")
SHA256 = re.compile(r"[0-9a-f]{64}")
CODE = (
    "backend/scripts/image_bg_real_transport_v225.py",
    "backend/tests/test_image_bg_real_transport_v225.py",
    "docs/image-bg-real-transport-v225-method.md",
    "backend/scripts/video_appearance_corrected_v112.py",
)
read, write, sha, require, check = (
    original.read,
    original.write,
    original.sha,
    original.require,
    original.check,
)


def aws_route(current, location, asset, linked_sha, linked_size):
    """Return the allowed signed route and a query-free private proof."""
    target = urlsplit(location)
    check(
        current == original.download_url(asset)
        and target.scheme == "https"
        and target.netloc == AWS
        and AWS_PATH.fullmatch(target.path) is not None
        and bool(target.query)
        and not target.fragment,
        "untrusted_aws_asset_route",
    )
    check(isinstance(linked_sha, str) and SHA256.fullmatch(linked_sha), "aws_publisher_sha256_required")
    check(
        isinstance(linked_size, str)
        and linked_size.isascii()
        and linked_size.isdecimal()
        and 0 < int(linked_size) <= original.MAX_BYTES,
        "aws_publisher_size_required",
    )
    return location, {"path": target.path, "linked_size": int(linked_size), "linked_sha256": linked_sha}


def download(session, asset, expected_sha, deadline):
    linked_sha, linked_size, proof = None, None, None

    def record_commitment(headers, *, allow_etag=False):
        nonlocal linked_sha, linked_size
        digest = original.commitment(headers, allow_etag=allow_etag)
        if headers.get("X-Linked-Etag") is not None:
            linked_sha = digest
        linked_size = headers.get("X-Linked-Size")
        return digest

    def follow(current, location, name):
        nonlocal proof
        check(urlsplit(current).hostname != AWS, "aws_asset_must_be_terminal")
        try:
            target = urlsplit(urljoin(current, location)) if location else None
        except ValueError:
            return original.redirect_to(current, location, name)
        if target is not None and target.hostname == AWS:
            url, proof = aws_route(current, location, name, linked_sha, linked_size)
            return url
        return original.redirect_to(current, location, name)

    body, receipt = scoped(
        original.download,
        {"commitment": record_commitment, "redirect_to": follow},
    )(session, asset, expected_sha, deadline)
    if proof is not None:
        require(
            receipt["final_host"] == AWS
            and receipt["bytes"] == proof["linked_size"]
            and receipt["publisher_asset_commitment"] == receipt["sha256"] == proof["linked_sha256"],
            "AWS receipt differs from its pinned publisher commitment",
        )
        receipt["aws_transport"] = proof
    return body, receipt


def verify_file(cache, case, role, recorded):
    if recorded.get("final_host") == AWS:
        proof = recorded.get("aws_transport")
        require(
            isinstance(proof, dict)
            and set(proof) == {"path", "linked_size", "linked_sha256"}
            and isinstance(proof["path"], str)
            and AWS_PATH.fullmatch(proof["path"]) is not None
            and type(proof["linked_size"]) is int
            and proof["linked_size"] == recorded["bytes"]
            and isinstance(proof["linked_sha256"], str)
            and SHA256.fullmatch(proof["linked_sha256"]) is not None
            and proof["linked_sha256"] == recorded["publisher_asset_commitment"] == recorded["sha256"],
            "Retained AWS publisher transport proof differs",
        )
    else:
        require("aws_transport" not in recorded, "Unexpected AWS transport proof")
    return scoped(original.verify_file, {"CDN": original.CDN | {AWS}})(cache, case, role, recorded)


def failed_context():
    require(sha(PREVIOUS_FREEZE) == PREVIOUS_FREEZE_SHA, "Published v221 freeze changed")
    require(
        all(sha(PREVIOUS / name) == digest for name, digest in FAILED_RECEIPTS.items()),
        "Preserve the exact original v221 failed receipts",
    )
    freeze = read(PREVIOUS_FREEZE)
    plan = read(PREVIOUS / "plan.json")
    result = read(PREVIOUS / "summary.json")
    require(
        freeze["private_plan_sha256"] == FAILED_RECEIPTS["plan.json"]
        and {k: v for k, v in freeze.items() if k != "private_plan_sha256"}
        == {k: v for k, v in plan.items() if k not in {"roster_dir", "private_cache"}}
        and result["attempted"] == 36
        and result["qualified"] == result["neural_calls"] == 0
        and result["status_counts"] == {"untrusted_asset_redirect": 36}
        and result["acquisition_journal_sha256"] == FAILED_RECEIPTS["attempts.jsonl"],
        "Exact original redirect-failure scope required",
    )
    attempts = [json.loads(line) for line in (PREVIOUS / "attempts.jsonl").read_text().splitlines()]
    require(
        [row["ordinal"] for row in attempts] == list(range(36))
        and all(
            set(row["files"]) == set(original.ASSETS)
            and all(
                f["status"] == "rejected" and f["reason"] == "untrusted_asset_redirect"
                for f in row["files"].values()
            )
            for row in attempts
        ),
        "All 108 original asset failures must be accounted for",
    )
    return plan


def separate(first, second):
    require(
        not first.is_relative_to(second) and not second.is_relative_to(first),
        "Use new paths separate from the original failed study and cache",
    )


def bindings(roster_dir, cache):
    failed = failed_context()
    require(roster_dir == Path(failed["roster_dir"]), "Original frozen roster path required")
    separate(cache, Path(failed["private_cache"]))
    separate(cache, PREVIOUS)
    base, cases, source_keys, exposure = original.bindings(roster_dir, cache)
    require(
        {k: v for k, v in base.items() if k != "private_cache"}
        == {k: v for k, v in failed.items() if k != "private_cache"},
        "Only the transport and private output destinations may change",
    )
    return (
        {
            **base,
            "schema": "chipmunk.image-bg-real-acquisition-plan/225",
            "prior_v221_public_freeze_sha256": PREVIOUS_FREEZE_SHA,
            "prior_v221_failed_receipts_sha256": FAILED_RECEIPTS,
            "successor_code_commitments": {name: sha(ROOT / name) for name in CODE},
            "aws_transport": {
                "host": AWS,
                "path_regex": AWS_PATH.pattern,
                "requires_signed_query": True,
                "requires_linked_sha256_and_size": True,
                "terminal_only": True,
            },
        },
        cases,
        source_keys,
        exposure,
    )


def prepare(trial, cache):
    failed = failed_context()
    separate(trial, PREVIOUS)
    separate(trial, Path(failed["private_cache"]))
    scoped(original.prepare, {"bindings": bindings})(Path(failed["roster_dir"]), trial, cache)


def lease(trial, expected):
    separate(trial, PREVIOUS)
    separate(trial, Path(failed_context()["private_cache"]))
    return scoped(original.lease, {"bindings": bindings})(trial, expected)


def inspector():
    return scoped(original.inspect_case, {"verify_file": verify_file})


def acquire(trial, expected):
    scoped(
        original.acquire,
        {
            "lease": lease,
            "fetch_case": scoped(original.fetch_case, {"download": download}),
            "inspect_case": inspector(),
        },
    )(trial, expected)


def seal(trial, expected):
    scoped(original.seal, {"lease": lease, "inspect_case": inspector(), "print": lambda *_a, **_kw: None})(
        trial, expected
    )
    base = trial / "summary.json"
    rows = (json.loads(line) for line in (trial / "attempts.jsonl").read_text().splitlines())
    hosts = Counter(
        file["final_host"] for row in rows for file in row["files"].values() if file["status"] == "saved"
    )
    result = {
        **read(base),
        "schema": "chipmunk.image-bg-real-pixel-qualification/225",
        "transport_base_summary_sha256": sha(base),
        "prior_v221_public_freeze_sha256": PREVIOUS_FREEZE_SHA,
        "prior_v221_failed_summary_sha256": FAILED_RECEIPTS["summary.json"],
        "saved_assets_by_host": dict(hosts),
    }
    write(trial / "summary-v225.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "acquire", "seal"):
        command = commands.add_parser(name)
        command.add_argument("trial", type=Path)
        if name == "prepare":
            command.add_argument("cache", type=Path)
        else:
            command.add_argument("plan_sha256")
    args = parser.parse_args()
    trial = args.trial.resolve()
    if args.command == "prepare":
        prepare(trial, args.cache.resolve())
    else:
        (acquire if args.command == "acquire" else seal)(trial, args.plan_sha256)


if __name__ == "__main__":
    main()
