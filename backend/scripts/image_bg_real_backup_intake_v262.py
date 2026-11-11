"""Qualify the twelve unopened BG-REAL pairs as a separate local panel."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from scripts import image_bg_real_acquisition_v221 as original
from scripts import image_bg_real_transport_v225 as transport
from scripts.video_appearance_corrected_v112 import scoped

ROOT, DATA, PANELS = original.ROOT, original.DATA, original.PUBLIC
NAME = "image-bg-real-backup-intake-v262"
TRIAL = DATA / NAME
PUBLIC = PANELS / f"{NAME}-freeze.json"
PREVIOUS = DATA / "image-bg-real-acquisition-v225"
PREVIOUS_SCORE = DATA / "image-bg-real-jpeg-v229"
QUALIFICATION = PANELS / "image-bg-real-pixel-qualification-complete-v225.json"
SCORE = PANELS / "image-bg-real-jpeg-complete-v229.json"
RECEIPTS = {
    QUALIFICATION: "3a46ad593be6a095f25e5998f7c51d9d592de6d413915348b57271111ec50306",
    SCORE: "55a6c1aed3448056f4046c768102bf032064635c6a3a625cb98017a7356cb13d",
    PANELS
    / "image-bg-real-transport-freeze-v225.json": "d3c239bfad6a332ef027dd991ffc2fa1ee016d1c788f245853e29c88735863c4",
}
CODE = (
    "backend/scripts/image_bg_real_backup_intake_v262.py",
    "backend/tests/test_image_bg_real_backup_intake_v262.py",
    f"docs/{NAME}-method.md",
)
PAIR_COUNT = 12
LIMITS = SimpleNamespace(PRIMARY=PAIR_COUNT, BACKUP=0)
read, write, sha, require = original.read, original.write, original.sha, original.require
enough = scoped(original.enough, {"roster": LIMITS})


def backup_cases(cases):
    require(
        len(cases) == 24 + PAIR_COUNT
        and [row["ordinal"] for row in cases] == list(range(len(cases)))
        and [row["tier"] for row in cases] == ["primary"] * 24 + ["backup"] * PAIR_COUNT,
        "Exact unopened v218 tail required",
    )
    return [
        {**row, "v218_ordinal": row["ordinal"], "ordinal": index, "tier": "primary"}
        for index, row in enumerate(cases[24:])
    ]


def prior_evidence(cases, source_keys, exposure):
    require(all(sha(path) == digest for path, digest in RECEIPTS.items()), "Published image receipts changed")
    qual, scored = read(QUALIFICATION), read(SCORE)
    require(
        sha(PREVIOUS / "summary-v225.json") == RECEIPTS[QUALIFICATION]
        and sha(PREVIOUS / "plan.json") == qual["plan_sha256"]
        and sha(PREVIOUS / "attempts.jsonl") == qual["acquisition_journal_sha256"]
        and sha(PREVIOUS / "qualified-private.json") == qual["qualified_private_sha256"]
        and qual["qualified"] == qual["attempted"] == 24
        and qual["backup_attempted"] == 0
        and qual["unattempted_backups"] == PAIR_COUNT
        and scored["qualification_sha256"] == RECEIPTS[QUALIFICATION]
        and sha(PREVIOUS_SCORE / "freeze.json") == scored["freeze_sha256"],
        "Exact v225 qualification and scored v229 exposure required",
    )
    sealed = read(PREVIOUS / "qualified-private.json")
    old_cache = Path(sealed["cache"])
    previous = sealed["qualified"]
    require(
        old_cache == Path(read(PREVIOUS / "plan.json")["private_cache"])
        and old_cache == old_cache.resolve()
        and old_cache.is_dir()
        and [entry["source"] for entry in previous] == cases[:24]
        and all(not (old_cache / f"{n:02d}").exists() for n in range(24, 24 + PAIR_COUNT))
        and all(set(entry["measurement"]["fingerprints"]) == {"original", "edited"} for entry in previous),
        "Preserve the consumed primary pairs and unopened backups",
    )
    inputs = PREVIOUS_SCORE / "inputs.json"
    require(
        sha(inputs) == read(PREVIOUS_SCORE / "freeze.json")["private_input_bindings_sha256"],
        "Exact v229 JPEG exposure index required",
    )
    rows = read(inputs)["sources"]
    views = [view for row in rows for quality in row["inputs"].values() for view in quality.values()]
    require(
        [row["source_id"] for row in rows] == [row["recipient"] for row in cases[:24]] and len(views) == 240,
        "Exact twenty-four v229 sources and JPEG views required",
    )
    fingerprints = [value for entry in previous for value in entry["measurement"]["fingerprints"].values()]
    old_keys = set().union(*(original.roster.source_keys(row) for row in cases[:24]))
    graph, encoded, rgb = exposure
    return (
        old_cache,
        source_keys | old_keys,
        (
            [*graph, *fingerprints],
            encoded
            | {row["encoded_sha256"] for row in fingerprints}
            | {row["input_sha256"] for row in views},
            rgb | {row["rgb_sha256"] for row in fingerprints} | {row["decoded_rgb_sha256"] for row in views},
        ),
    )


def bindings(roster_dir, cache):
    base, old_cases, source_keys, exposure = transport.bindings(roster_dir, cache)
    old_cache, source_keys, exposure = prior_evidence(old_cases, source_keys, exposure)
    require(
        cache == old_cache.parent / "image-bg-real-backup-media-v262"
        and cache.parent.parent.parent.is_mount()
        and cache == cache.resolve(),
        "Exact new private Seagate cache required",
    )
    cases = backup_cases(old_cases)
    require(
        all(not original.roster.source_keys(case) & source_keys for case in cases),
        "A selected backup shares an already observed source identity",
    )
    plan = {
        **base,
        "schema": "chipmunk.image-bg-real-backup-intake-plan/262",
        "primary": PAIR_COUNT,
        "backup": 0,
        "v218_backup_count": PAIR_COUNT,
        "source_receipts": {path.name: digest for path, digest in RECEIPTS.items()},
        "intake_code_commitments": {name: sha(ROOT / name) for name in CODE},
        "publisher_split_counts": dict(Counter(row["publisher_split"] for row in cases)),
        "prior_v225_png_fingerprints": 48,
        "prior_v229_jpeg_views": 240,
        "maximum_encoded_bytes": PAIR_COUNT * len(original.ASSETS) * original.MAX_BYTES,
        "publisher_mask_commitment_required": True,
    }
    return plan, cases, source_keys, exposure


def frozen_view(plan, digest):
    return {
        **{key: value for key, value in plan.items() if key not in {"roster_dir", "private_cache"}},
        "private_plan_sha256": digest,
    }


def preflight():
    roster_dir = Path(transport.failed_context()["roster_dir"])
    cache = Path(read(PREVIOUS / "plan.json")["private_cache"]).parent / "image-bg-real-backup-media-v262"
    require(
        not TRIAL.exists() and not PUBLIC.exists() and not cache.exists(),
        "Preserve any previous intake attempt",
    )
    return bindings(roster_dir, cache)[0]


def freeze():
    plan = preflight()
    TRIAL.mkdir(mode=0o700)
    private = TRIAL / "plan.json"
    write(private, plan)
    write(PUBLIC, frozen_view(plan, sha(private)))
    return {"status": "frozen", "planned_pairs": PAIR_COUNT, "freeze_sha256": sha(PUBLIC)}


def lease(trial, expected):
    require(
        trial == TRIAL and read(PUBLIC) == frozen_view(read(trial / "plan.json"), expected),
        "Exact prospective v262 freeze required",
    )
    return scoped(original.lease, {"bindings": bindings})(trial, expected)


def inspect_case(cache, case, files, source_keys, exposure):
    row, exact = transport.inspector()(cache, case, files, source_keys, exposure)
    if exact is not None and row["measured"]["semantic_mask_binding"] != "publisher_asset_commitment":
        return {**row, "pair_failure": "semantic_mask_publisher_commitment_missing", "measured": None}, None
    return row, exact


def acquire(expected):
    scoped(
        original.acquire,
        {
            "lease": lease,
            "fetch_case": scoped(original.fetch_case, {"download": transport.download}),
            "inspect_case": inspect_case,
            "enough": enough,
        },
    )(TRIAL, expected)


def seal(expected):
    scoped(original.seal, {"lease": lease, "inspect_case": inspect_case, "enough": enough, "roster": LIMITS})(
        TRIAL, expected
    )


def readout(expected):
    cache, cases, source_keys, exposure = lease(TRIAL, expected)
    journal, private = TRIAL / "attempts.jsonl", TRIAL / "qualified-private.json"
    journal_sha = sha(journal)
    sealed = read(private)
    summary = read(TRIAL / "summary.json")
    rows = [json.loads(line) for line in journal.read_text().splitlines()]
    require(
        len(rows) == PAIR_COUNT
        and read(TRIAL / "acquired.json")
        == {"plan_sha256": expected, "journal_sha256": journal_sha, "attempted": PAIR_COUNT}
        and sealed["plan_sha256"] == summary["plan_sha256"] == expected
        and sealed["acquisition_journal_sha256"] == summary["acquisition_journal_sha256"] == journal_sha
        and Path(sealed["cache"]) == cache
        and summary["qualified_private_sha256"] == sha(private),
        "Exact sealed acquisition and qualification required",
    )
    checked = [
        inspect_case(cache, case, row["files"], source_keys, exposure)
        for case, row in zip(cases, rows, strict=True)
    ]
    require([row for row, _ in checked] == rows, "Source and pixel readback differs from the saved journal")
    decisions = original.ordered_decisions(rows)
    outcomes = [{"ordinal": index, "status": decisions[index]} for index in range(PAIR_COUNT)]
    qualified = [index for index in range(PAIR_COUNT) if decisions[index] == "qualified"]
    require(sealed["attempt_statuses"] == outcomes, "Saved exposure or qualification decisions differ")
    for index, entry in zip(qualified, sealed["qualified"], strict=True):
        case, (row, exact) = cases[index], checked[index]
        relative = f"{index:02d}/exact_truth.png"
        truth = cache / relative
        require(
            entry["source"] == case
            and entry["measurement"] == row
            and entry["truth_file"] == relative
            and truth.is_file()
            and truth == truth.resolve()
            and sha(truth) == entry["truth_png_sha256"]
            and np.array_equal(original.png_pixels(truth.read_bytes(), case["size"])[:, :, 0] > 0, exact),
            "Saved exact changed-pixel mask differs",
        )
    return {
        "schema": "chipmunk.image-bg-real-backup-intake-result/262",
        "status": "complete",
        "freeze_sha256": sha(PUBLIC),
        "planned_pairs": PAIR_COUNT,
        "attempted_pairs": len(rows),
        "qualified_pairs": len(qualified),
        "target_shortfall": PAIR_COUNT - len(qualified),
        "statuses": dict(Counter(decisions.values())),
        "qualified_by_publisher_split": dict(Counter(cases[index]["publisher_split"] for index in qualified)),
        "saved_asset_counts": dict(Counter(file["status"] for row in rows for file in row["files"].values())),
        "qualified_publisher_bound_masks": len(qualified),
        "acquisition_journal_sha256": journal_sha,
        "qualified_private_sha256": sha(private),
        "neural_calls": 0,
        "detector_calls": 0,
        "serving_admitted": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "freeze", "acquire", "seal", "readout"))
    command = parser.parse_args().command
    if command == "preflight":
        plan = preflight()
        result = {
            "status": "ready_for_freeze",
            "planned_pairs": PAIR_COUNT,
            "publisher_split_counts": plan["publisher_split_counts"],
            "maximum_encoded_bytes": plan["maximum_encoded_bytes"],
            "model_calls": 0,
        }
    elif command == "freeze":
        result = freeze()
    else:
        expected = read(PUBLIC)["private_plan_sha256"]
        result = {"acquire": acquire, "seal": seal, "readout": readout}[command](expected)
    if result is not None:
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
