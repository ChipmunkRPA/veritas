"""Prospective passage evidence comparison on two admitted human copy-edits."""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

from scripts import text_goce_source_admission_v243 as admission
from scripts import text_gohumanize_transfer_v196 as transfer

source, layout = admission.prior, transfer.layout
intake = source.intake
ROOT, PANEL = transfer.ROOT, transfer.PANEL
TRIAL = ROOT / "backend/data/independent-evaluation/text-goce-span-v249"
PUBLIC = PANEL / "text-goce-span-freeze-v249.json"
CACHE = Path("/private/tmp")
MODULE = "scripts.text_goce_span_v249"
ADMISSION_SHA = "c31b9ef25fc1b8cb43d7bd476f1e110d6ea960e44d917e2ef9d6aea331e2bc33"
TRANSFER_SHA = "eff16a6244be6dcb3ee8b200b4fc55e1b157794372bf63300b5729003a45ea53"
TRANSFER_FREEZE_SHA = "bcdea8d8dd81bd5f5dc934c546d60ea03ee256b29a557d67996146281fad5475"
ARMS = ("clean_human", "human_retouch")
REGIONS = ("replacement_locus", "target_padding", "outside_target")
SCOPE = "Two distinct archived human copy-edits, four related inputs and no new synthetic positives."
require, read, sha = transfer.require, transfer.read, transfer.sha


def key(record):
    return tuple(record[name] for name in admission.KEY_FIELDS)


def exact_admission(cache):
    admitted = cache / "veritas-wikipedia-goce-rights-v243-attempt01"
    require(admitted.is_dir(), "The completed private v243 cache is required")
    source.archive.safe_cache(admitted)
    public = PANEL / "text-goce-source-admission-complete-v243.json"
    result = read(public)
    private = {
        name: json.loads(source.private_bytes(admitted / name))
        for name in ("private-attribution.json", "private-exposure-index.json", "private-review.json")
    }
    require(
        sha(public) == ADMISSION_SHA
        and result == read(admitted / "aggregate.json")
        and result["status"] == "complete"
        and result["rights_admitted_pages"] == result["reviewed_windows"] == 2
        and result["pending_windows"] == 5
        and result["rights_exclusions"] == {}
        and source.text.sha(source.json_bytes(private)) == result["private_artifacts_commitment_sha256"],
        "Exact published and private v243 admission required",
    )
    review = private["private-review.json"]
    approved = {key(row): row for row in review["approved"]}
    held = {key(row) for row in review["held"]}
    citations = {key(row): row for row in private["private-attribution.json"]}
    require(
        len(approved) == len(citations) == 2
        and len({row["story_component"] for row in approved.values()}) == 2
        and len(held) == 5
        and approved.keys() == citations.keys()
        and not approved.keys() & held
        and all(row["decision"] == "approved" for row in approved.values()),
        "Two distinct admitted and attributed sources required",
    )
    return approved, held, citations, result["private_artifacts_commitment_sha256"]


def window_row(witness, ordinal, arm):
    raw = witness["excerpt"]
    origin, end = witness["article_codepoints"]
    target = [n - origin for n in witness["target_codepoints"]]
    locus = [n - origin for n in witness["retouch_codepoints"]]
    require(
        all(type(n) is int for n in (origin, end, *target, *locus))
        and origin >= 0
        and end - origin == len(raw)
        and 0 <= target[0] <= locus[0] < locus[1] <= target[1] <= len(raw)
        and 8 <= witness["target_words"] <= 16
        and witness["target_words"] == len(raw[target[0] : target[1]].split()),
        "Exact excerpt, target and lexical replacement coordinates required",
    )
    group = f"source-{ordinal:02d}"
    return {
        "case": f"{group}-{arm}",
        "article_group": group,
        "arm": arm,
        "status": "constructed",
        "text": raw,
        "text_sha256": transfer.text_sha(raw),
        "ai_spans": [],
        "article_offset": origin,
        "target": target,
        "locus": locus,
    }


def shared_context(before, after):
    """Bind matching positions outside the exact lexical replacement."""
    a, b = before["text"], after["text"]
    a0, a1 = before["locus"]
    b0, b1 = after["locus"]
    positions = []
    for original, edited, ao, bo in ((a[:a0], b[:b0], 0, 0), (a[a1:], b[b1:], a1, b1)):
        mapping = transfer.correspondence(original, edited)
        positions.extend((p + ao, i + bo) for i, p in enumerate(mapping) if p is not None)
    require(positions, "An exact paired unchanged context is required")
    return positions


def inputs(cache=CACHE):
    approved, held, citations, admission_hash = exact_admission(cache)
    groups, roster_hash = source.frozen_candidates(cache / "veritas-wikipedia-goce-v228")
    witnesses = {
        intake.prose.review_key(pair, window): (pair, window)
        for pairs in groups.values()
        for pair in pairs
        for window in pair["windows"]
    }
    require(witnesses.keys() == approved.keys() | held, "Exact admitted and pending source keys required")
    result = []
    for ordinal, scope in enumerate(approved, 1):
        pair, window = witnesses[scope]
        require(
            pair["render_sha256"] == citations[scope]["before_after_render_sha256"],
            "Source render and admission attribution differ",
        )
        both = tuple(
            window_row(window[side], ordinal, arm)
            for side, arm in zip(("before", "after"), ARMS, strict=True)
        )
        shared_context(*both)
        require(both[0]["text"] != both[1]["text"], "Copy-edit control must contain an actual revision")
        result.extend(both)
    return result, {"v243_private_commitment_sha256": admission_hash, "v228_roster_sha256": roster_hash}


def historical_positive(rows):
    public = PANEL / "text-gohumanize-transfer-complete-v196.json"
    prior_freeze = PANEL / "text-gohumanize-transfer-freeze-v196.json"
    require(
        sha(public) == TRANSFER_SHA and sha(prior_freeze) == TRANSFER_FREEZE_SHA,
        "Exact v196 reference required",
    )
    aggregate = read(public)
    require(
        aggregate["freeze_sha256"] == TRANSFER_FREEZE_SHA and aggregate["status"] == "complete",
        "Complete v196 reference required",
    )
    prior = [row for row in transfer.inputs() if row["arm"] == "short_ai"]
    measured = [row for row in prior if row["status"] == "constructed"]
    positives = set().union(*(source.text.fingerprints(row["text"]) for row in measured))
    require(
        len(prior) == 54
        and len(measured) == 23
        and all(not (source.text.fingerprints(row["text"]) & positives) for row in rows),
        "Disjoint previously assessed positive comparison required",
    )
    lengths = [len(row["text"].split()) for row in measured]
    comparisons = aggregate["results"]["all"]["short_ai"]
    require(
        all(
            comparisons[method]["planned"] == len(prior)
            and comparisons[method]["ok"] == len(measured)
            and comparisons[method]["preparation_unavailable"] == len(prior) - len(measured)
            for method in layout.METHODS
        ),
        "Exact historical positive denominators required",
    )
    return {
        "public_result_sha256": TRANSFER_SHA,
        "planned": len(prior),
        "assessed": len(measured),
        "preparation_unavailable": len(prior) - len(measured),
        "article_groups_assessed": len({row["article_group"] for row in measured}),
        "assessed_word_range": [min(lengths), max(lengths)],
        "useful_ai_locations": {
            method: comparisons[method]["useful_ai_location"] for method in layout.METHODS
        },
        "scope": "Previously scored v196 short AI inserts in longer passages, separate from the new human controls.",
    }


def prospective():
    rows, evidence = inputs()
    pinned_runtime = transfer.identity()
    earlier = read(PANEL / "text-gohumanize-transfer-freeze-v196.json")
    require(
        pinned_runtime == earlier, "Unchanged native v196 model, calibration and method identities required"
    )
    from scripts import writing_modernbert_v4 as models

    asset_paths = read(transfer.historical.PRIOR.parent / "writing-specialists-v5/freeze.json")["model_paths"]
    require(
        all(
            (Path(asset_paths[role]) / name).is_file()
            for role, names in (("baseline", models.BASELINE_ASSETS), ("candidate", models.CANDIDATE_ASSETS))
            for name in names
        ),
        "Both already pinned native checkpoints must be available locally",
    )
    reference = historical_positive(rows)
    paths = (
        Path(__file__),
        ROOT / "backend/tests/test_text_goce_span_v249.py",
        ROOT / "docs/text-goce-span-v249-method.md",
        ROOT / "backend/scripts/text_goce_source_admission_v243.py",
        ROOT / "backend/scripts/text_goce_intake_v228.py",
        ROOT / "backend/scripts/text_goce_source_evidence_v233.py",
        ROOT / "backend/app/models.yaml",
    )
    shared_views = 0
    for row in rows:
        # This is the existing text-length floor. The native worker still
        # records any unavailable extraction or incomplete scoring coverage.
        require(len(row["text"].split()) >= 15, "Exact excerpt falls below the local text length floor")
        transfer.labels_for(row)
        primary, canonical = transfer.views(row["text"])
        shared_views += primary["text"] == canonical["text"]
    return rows, {
        "schema": "chipmunk.text-goce-span-freeze/249",
        "admission_public_sha256": ADMISSION_SHA,
        "private_source_commitment_sha256": source.text.sha(source.json_bytes(evidence)),
        "private_inputs_sha256": source.text.sha(source.json_bytes(rows)),
        "native_reference_sha256": TRANSFER_FREEZE_SHA,
        "code_sha256": {str(path.relative_to(ROOT)): sha(path) for path in paths},
        "source_groups": 2,
        "planned_inputs": len(rows),
        "inputs_with_identical_effective_views": shared_views,
        "word_range": [
            min(len(row["text"].split()) for row in rows),
            max(len(row["text"].split()) for row in rows),
        ],
        "methods": list(layout.METHODS),
        "position_regions": list(REGIONS),
        "historical_positive_reference": reference,
        "scope": SCOPE,
    }


def freeze():
    require(not TRIAL.exists() and not PUBLIC.exists(), "Use a fresh prospective trial")
    _, lock = prospective()
    TRIAL.mkdir(mode=0o700)
    transfer.write(TRIAL / "freeze.json", lock)
    transfer.write(PUBLIC, lock)
    return {
        "status": "frozen",
        "freeze_sha256": sha(PUBLIC),
        "planned_inputs": lock["planned_inputs"],
        "new_neural_calls": 0,
    }


def verify():
    rows, identity = prospective()
    require(read(PUBLIC) == read(TRIAL / "freeze.json") == identity, "Exact prospective v249 freeze required")
    return rows


def position_masks(row, response, points):
    if response["status"] != "measured":
        require(response["status"] == "unassessed", "Unknown worker disposition")
        return dict.fromkeys(layout.METHODS)
    primary, canonical = transfer.views(row["text"])

    def project(key, view, model):
        proposals = response[key]
        return layout.project(view, proposals, points[model]["cut"]) if proposals is not None else None

    return layout.compare(
        project("primary_current", primary, "current"),
        project("canonical_current", canonical, "current"),
        project("canonical_interval", canonical, "interval"),
    )


def private_positions(row, methods):
    return {
        "article_offset": row["article_offset"],
        "replacement_locus": row["locus"],
        "target": row["target"],
        "methods": {
            method: {"flagged": sorted(evidence.mask), "covered": sorted(evidence.covered)}
            if evidence is not None
            else None
            for method, evidence in methods.items()
        },
    }


def summarise(rows, responses, points):
    arm_fields = ("planned", "fully_assessed", "partially_assessed", "unassessed", "false_flagged_windows")
    region_fields = tuple(
        f"{region}_{suffix}"
        for region in REGIONS
        for suffix in ("characters", "covered_characters", "false_flagged_characters")
    )
    pair_states = (
        "neither_false_flagged",
        "before_only_false_flagged",
        "retouch_only_false_flagged",
        "both_false_flagged",
    )
    pair_fields = (
        "planned_pairs",
        "fully_assessed_pairs",
        "partially_assessed_pairs",
        "unassessed_pairs",
        "known_unchanged_characters",
        "jointly_covered_unchanged_characters",
        "unchanged_characters_with_new_false_flag",
        "unchanged_characters_with_removed_false_flag",
        *pair_states,
    )
    counts = {
        method: {arm: Counter(dict.fromkeys((*arm_fields, *region_fields), 0)) for arm in ARMS}
        for method in layout.METHODS
    }
    comparisons = {method: Counter(dict.fromkeys(pair_fields, 0)) for method in layout.METHODS}
    require(
        [r["case"] for r in rows] == [r["case"] for r in responses],
        "Complete ordered admitted responses required",
    )
    masks = []
    known = []
    for row, response in zip(rows, responses, strict=True):
        regions = {name: set() for name in REGIONS}
        labels = transfer.labels_for(row)
        known.append(set(labels))
        for n in labels:
            name = "outside_target"
            if row["locus"][0] <= n < row["locus"][1]:
                name = "replacement_locus"
            elif row["target"][0] <= n < row["target"][1]:
                name = "target_padding"
            regions[name].add(n)
        methods = position_masks(row, response, points)
        require(
            response["positions"] == private_positions(row, methods), "Exact source-position replay differs"
        )
        masks.append(methods)
        for method, evidence in methods.items():
            table = counts[method][row["arm"]]
            table["planned"] += 1
            for name, indices in regions.items():
                table[f"{name}_characters"] += len(indices)
            if evidence is None:
                table["unassessed"] += 1
                continue
            values = layout.measure(labels, evidence)
            if values["unscored_human_characters"]:
                table["partially_assessed"] += 1
            else:
                table["fully_assessed"] += 1
            table["false_flagged_windows"] += values["genuine_document_flagged"]
            for name, indices in regions.items():
                table[f"{name}_covered_characters"] += len(evidence.covered & indices)
                table[f"{name}_false_flagged_characters"] += len(evidence.mask & indices)
    for i in range(0, len(rows), 2):
        a, b = rows[i : i + 2]
        require(
            a["article_group"] == b["article_group"] and (a["arm"], b["arm"]) == ARMS,
            "Exact paired arm order required",
        )
        shared = shared_context(a, b)
        for method in layout.METHODS:
            x, y = masks[i][method], masks[i + 1][method]
            table = comparisons[method]
            table["planned_pairs"] += 1
            table["known_unchanged_characters"] += len(shared)
            if x is None or y is None:
                table["unassessed_pairs"] += 1
                continue
            if x.covered == known[i] and y.covered == known[i + 1]:
                table["fully_assessed_pairs"] += 1
                table[pair_states[int(bool(x.mask)) + 2 * int(bool(y.mask))]] += 1
            else:
                table["partially_assessed_pairs"] += 1
            for left, right in shared:
                if left in x.covered and right in y.covered:
                    table["jointly_covered_unchanged_characters"] += 1
                    table["unchanged_characters_with_new_false_flag"] += (
                        left not in x.mask and right in y.mask
                    )
                    table["unchanged_characters_with_removed_false_flag"] += (
                        left in x.mask and right not in y.mask
                    )
    return {"by_arm_and_method": counts, "paired_false_flags_and_shared_context": comparisons}


def worker(output):
    rows = verify()
    stage = TRIAL / "extract-00"
    require(
        output == stage / "output"
        and (stage / "resources.jsonl").is_file()
        and (stage / "worker.log").is_file()
        and not (stage / "profile.json").exists(),
        "Run inference through the owned guard",
    )
    from scripts import writing_span_heads_v7 as heads
    from scripts.writing_span_native_v7 import Native

    assets = transfer.historical.PRIOR
    head = heads.restore("interval", read(assets / "interval-head.json")["state"])
    points, maps = read(assets / "points.json"), read(assets / "calibrators.json")
    native = Native(with_current=True)
    output.mkdir(mode=0o700)
    journal = output / "private-responses.jsonl"
    with journal.open("x") as stream:
        for i, row in enumerate(rows):
            response = transfer.infer(native, head, maps, row, i == 0)
            positions = private_positions(row, position_masks(row, response, points))
            stream.write(
                json.dumps({"case": row["case"], **response, "positions": positions}, allow_nan=False) + "\n"
            )
            stream.flush()
    native.unchanged()
    transfer.write(
        output / "result.json",
        {
            "status": "complete",
            "freeze_sha256": sha(PUBLIC),
            "journal_sha256": sha(journal),
            "planned_inputs": len(rows),
            "actual_forwards": native.calls,
            "new_neural_calls": sum(native.calls.values()),
        },
    )


def run():
    verify()
    expected = TRIAL / "extract-00/output"

    def launch(argv, **kwargs):
        require(
            argv[3:] == ["scripts.video_appearance_native_v108", "extract", str(expected), "0"],
            "Exact owned v249 command required",
        )
        return subprocess.Popen([*argv[:3], MODULE, "_worker", str(expected)], **kwargs)

    transfer.scoped(
        transfer.runtime.run,
        {
            "TRIAL": TRIAL,
            "verify": verify,
            "prerequisites": lambda *_: True,
            "__file__": __file__,
            "subprocess": SimpleNamespace(
                Popen=launch, STDOUT=subprocess.STDOUT, TimeoutExpired=subprocess.TimeoutExpired
            ),
        },
    )("extract", 0)


def readout():
    rows = verify()
    stage = TRIAL / "extract-00"
    journal = stage / "output/private-responses.jsonl"
    result = read(stage / "output/result.json")
    transfer.qualification_profile(stage, result["new_neural_calls"], {}, Path(__file__).name, "extract", 0)
    require(
        result["planned_inputs"] == len(rows)
        and result["freeze_sha256"] == sha(PUBLIC)
        and result["journal_sha256"] == sha(journal),
        "Complete guarded v249 responses required",
    )
    points = read(transfer.historical.PRIOR / "points.json")
    return {
        "schema": "chipmunk.text-goce-span-readout/249",
        "status": "complete",
        "scope": SCOPE,
        "freeze_sha256": sha(PUBLIC),
        "resource_profile_sha256": sha(stage / "profile.json"),
        "source_groups": 2,
        "planned_inputs": len(rows),
        "actual_forwards": result["actual_forwards"],
        "historical_positive_reference": read(PUBLIC)["historical_positive_reference"],
        **summarise(rows, transfer.records(journal), points),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "freeze", "run", "readout", "_worker"))
    parser.add_argument("output", type=Path, nargs="?")
    args = parser.parse_args()
    if (args.command == "_worker") != (args.output is not None):
        parser.error("The private worker requires its exact guarded output directory")
    if args.command == "_worker":
        worker(args.output)
    elif args.command == "run":
        run()
    else:
        value = (
            {"status": "verified_preflight", **prospective()[1], "new_neural_calls": 0}
            if args.command == "preflight"
            else {"freeze": freeze, "readout": readout}[args.command]()
        )
        print(json.dumps(value, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
