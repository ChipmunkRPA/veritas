"""Measure the existing passage check and the saved interval head on sealed eLife rewrites."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

from scripts import text_elife_rewrites_v273 as generation
from scripts import text_gohumanize_transfer_v196 as prior
from scripts import text_opai_bench_score_v260 as opai

ROOT, PANEL = prior.ROOT, prior.PANEL
NAME = "text-elife-passage-score-v275"
MODULE = "scripts.text_elife_passage_score_v275"
TRIAL = opai.BASE / NAME
PUBLIC = PANEL / "text-elife-passage-score-freeze-v275.json"
GENERATED = PANEL / "text-elife-rewrites-complete-v273.json"
CASES = generation.TRIAL / "private-cases.jsonl"
GENERATED_SHA = "5437db6e274c3fc87eab1aa6424e7ed17f4d2a84a243dde84045fc16e40e7b7f"
OPAI_FREEZE_SHA = "67f3d2f204d3da6503be9edf6ba2be5f0b3f8e20605fa148379c21db90473c7c"
ROLES = ((None, "human"), ("qwen", "paragraph_ai"), ("smol2", "paragraph_ai"))
EXPECTED = {"qwen": {"constructed": 5}, "smol2": {"constructed": 1, "outside_word_bound": 4}}
GROUPS_PER_BATCH = 3
BATCHES = 2
WIDTH = len(ROLES) * GROUPS_PER_BATCH
METHODS = ("current", "interval")
read, write, sha, require, scoped = prior.read, prior.write, prior.sha, prior.require, prior.scoped


def validate(rows, originals):
    require(
        len(rows) == len(originals) * len(ROLES) == 15, "Expected every sealed control and rewrite attempt"
    )
    dispositions = defaultdict(Counter)
    for offset, original in enumerate(originals):
        start = offset * len(ROLES)
        control, *mixed = rows[start : start + len(ROLES)]
        group = original["group"]
        require(
            all(row["article_group"] == group for row in (control, *mixed))
            and tuple((row["generator"], row["arm"]) for row in (control, *mixed)) == ROLES
            and control["case"] == f"{group}-human"
            and control["status"] == "constructed"
            and control["text"] == original["text"]
            and control["ai_spans"] == [],
            "The sealed control, generator order and original must agree",
        )
        prior.labels_for(control)
        source = original["text"]
        a, b = original["edit_target_codepoints"]
        for row in mixed:
            family = row["generator"]
            require(
                row["case"] == f"{group}-{family}"
                and row["source_sha256"] == original["text_sha256"]
                and row["source_target"] == [a, b],
                "Generated case differs from the sealed source or original target",
            )
            if row["status"] == "unavailable":
                require(
                    row["text"] is row["text_sha256"] is row["ai_spans"] is None
                    and row["reason"] == "outside_word_bound",
                    "Unavailable rewrite must have no input or truth",
                )
            else:
                require(
                    row["status"] == "constructed" and row["reason"] is None, "Unknown rewrite disposition"
                )
                labels = prior.labels_for(row)
                require(len(row["ai_spans"]) == 1, "Expected one known generated interval")
                left, right = row["ai_spans"][0]
                edited = row["text"]
                require(
                    left == a
                    and edited[:left] == source[:a]
                    and edited[right:] == source[b:]
                    and edited[left:right] != source[a:b]
                    and 0 < sum(labels.values()) < len(labels),
                    "Generated interval must preserve the exact Unicode prefix and suffix",
                )
            dispositions[family][row["reason"] or "constructed"] += 1
    require(dispositions == EXPECTED, "Expected the sealed six mixed cases and four unavailable rewrites")
    return rows


def inputs():
    receipt = read(GENERATED)
    require(
        sha(GENERATED) == GENERATED_SHA == sha(generation.TRIAL / "result.json")
        and receipt["schema"] == "chipmunk.text-elife-rewrites-result/273"
        and receipt["status"] == "complete"
        and receipt["source_groups"] == receipt["human_controls"] == len(generation.GROUPS) == 5
        and receipt["planned_rewrites"] == 10
        and receipt["generation_dispositions"] == EXPECTED
        and receipt["freeze_sha256"] == sha(generation.FREEZE) == sha(generation.TRIAL / "freeze.json")
        and receipt["private_cases_sha256"] == sha(CASES),
        "Exact sealed v273 cohort required",
    )
    return validate(generation.records(CASES), generation.inputs())


def identity(rows):
    require(
        sha(opai.PUBLIC) == OPAI_FREEZE_SHA
        and prior.identity() == read(prior.PUBLIC)
        and read(opai.PUBLIC)["native_reference_sha256"] == sha(prior.PUBLIC),
        "Exact v260 passage runtime, calibration maps and operating points required",
    )
    paths = (
        Path(__file__),
        ROOT / "backend/tests/test_text_elife_passage_score_v275.py",
        ROOT / f"docs/{NAME}-method.md",
    )
    return {
        "schema": "chipmunk.text-elife-passage-score-freeze/275",
        "v273_result_sha256": GENERATED_SHA,
        "private_cases_sha256": sha(CASES),
        "v260_freeze_sha256": OPAI_FREEZE_SHA,
        "native_reference_sha256": sha(prior.PUBLIC),
        "research_points_sha256": sha(prior.historical.PRIOR / "points.json"),
        "code_sha256": {str(path.relative_to(ROOT)): sha(path) for path in paths},
        "source_groups": len(rows) // len(ROLES),
        "planned_inputs": len(rows),
        "constructible_inputs": sum(row["status"] == "constructed" for row in rows),
        "batches": BATCHES,
        "batch_max_inputs": WIDTH,
        "methods": list(METHODS),
        "serving_change_permitted": False,
    }


def freeze():
    require(not TRIAL.exists() and not PUBLIC.exists(), "Preserve earlier attempts")
    locked = identity(inputs())
    TRIAL.mkdir(mode=0o700)
    write(TRIAL / "freeze.json", locked)
    write(PUBLIC, locked)
    return {"status": "frozen", "planned_inputs": locked["planned_inputs"], "new_neural_calls": 0}


def verify():
    rows = inputs()
    require(
        identity(rows) == read(PUBLIC) == read(TRIAL / "freeze.json"), "Exact prospective freeze required"
    )
    return rows


worker = scoped(
    prior.worker,
    {
        "TRIAL": TRIAL,
        "PUBLIC": PUBLIC,
        "verify": verify,
        "ARMS": ROLES,
        "PAIRS_PER_BATCH": GROUPS_PER_BATCH,
    },
)


def run(batch):
    require(
        batch == 0 or (TRIAL / "extract-00/profile.json").is_file(), "Run the two fixed batches sequentially"
    )
    scoped(
        prior.run,
        {
            "TRIAL": TRIAL,
            "verify": verify,
            "BATCHES": BATCHES,
            "MODULE": MODULE,
            "__file__": __file__,
        },
    )(batch)


def position_scores(view, proposals):
    result = {}
    for proposal in proposals:
        raw = proposal["raw"]
        require(
            type(raw) in (int, float) and math.isfinite(raw) and 0 <= raw <= 1,
            "Finite bounded raw proposal response required",
        )
        for pos in view["source_positions"][proposal["start"] : proposal["end"]]:
            if pos is not None:
                result[pos] = max(raw, result.get(pos, 0.0))
    return result


def rank(scores, common, labels):
    groups = defaultdict(Counter)
    for pos in common:
        groups[scores[pos]][labels[pos]] += 1
    ai = sum(group[1] for group in groups.values())
    human = len(common) - ai
    if not ai or not human:
        return None
    lower = wins = 0
    for _, group in sorted(groups.items()):
        wins += group[1] * (2 * lower + group[0])
        lower += group[0]
    return wins / (2 * ai * human)


def assessments(case, response, points):
    if case["status"] == "unavailable":
        require(
            response == {"case": case["case"], "status": "preparation_unavailable"},
            "Unavailable rewrites must have no prediction",
        )
        return {}, [None, None], [None, None]
    labels = prior.labels_for(case)
    if response["status"] == "unassessed":
        require(response["reason"] == "extraction_correspondence", "Unknown unassessed disposition")
        return labels, [None, None], [None, None]
    require(response["status"] == "measured", "Unexpected response disposition")
    views = prior.views(case["text"])
    proposals = [response["primary_current"], response["canonical_interval"]]
    proofs = [response["native_proofs"][0], response["native_proofs"][2]]
    evidence, scores = [], []
    for method, view, rows, proof in zip(METHODS, views, proposals, proofs, strict=True):
        require((rows is None) == (proof is None), "Proposal and native proof availability differs")
        if rows is None:
            evidence.append(None)
            scores.append({})
        else:
            mask = prior.layout.project(view, rows, points[method]["cut"])
            raw = position_scores(view, rows)
            require(
                mask.covered == raw.keys() and mask.covered <= labels.keys(),
                "Exact source positions required",
            )
            evidence.append(mask if mask.covered else None)
            scores.append(raw)
    common = scores[0].keys() & scores[1].keys()
    return labels, evidence, [rank(raw, common, labels) for raw in scores]


def summarize(rows, responses, points):
    fields = (
        "planned",
        "preparation_unavailable",
        "unassessed",
        "partially_assessed",
        "fully_assessed",
        "complete_direction_matches",
        "genuine_flagged_inputs",
        "positive_inputs_with_overlap",
        "useful_locations",
        "recovered_intervals",
        "known_ai_characters",
        "known_human_characters",
        "covered_ai_characters",
        "covered_human_characters",
        "located_ai_characters",
        "highlighted_human_characters",
    )
    counts = defaultdict(lambda: Counter(dict.fromkeys(fields, 0)))
    rankings = defaultdict(list)
    for row, response in zip(rows, responses, strict=True):
        require(row["case"] == response["case"], "Exact ordered cohort responses required")
        labels, evidence, scores = assessments(row, response, points)
        ai = sum(labels.values())
        human = len(labels) - ai
        family = row["generator"] or "shared_human"
        for method, mask, score in zip(METHODS, evidence, scores, strict=True):
            counter = counts[family, method]
            counter["planned"] += 1
            counter["known_ai_characters"] += ai
            counter["known_human_characters"] += human
            if not labels:
                counter["preparation_unavailable"] += 1
            elif mask is None:
                counter["unassessed"] += 1
            else:
                result = prior.layout.measure(labels, mask)
                full = mask.covered == labels.keys()
                overlap = result["located_ai_characters"] > 0
                false_review = result["genuine_document_flagged"]
                counter["fully_assessed"] += full
                counter["partially_assessed"] += not full
                counter["complete_direction_matches"] += full and (overlap if ai else not false_review)
                counter["genuine_flagged_inputs"] += false_review
                counter["positive_inputs_with_overlap"] += overlap
                counter["useful_locations"] += result["useful_ai_location"]
                counter["recovered_intervals"] += result["recovered_ai_intervals"]
                counter["covered_ai_characters"] += ai - result["unscored_ai_characters"]
                counter["covered_human_characters"] += human - result["unscored_human_characters"]
                counter["located_ai_characters"] += result["located_ai_characters"]
                counter["highlighted_human_characters"] += result["highlighted_human_characters"]
            if score is not None:
                rankings[family, method].append(score)
    tables = {}
    for (family, method), counter in counts.items():
        scores = rankings[family, method]
        tables.setdefault(family, {})[method] = {
            **counter,
            "raw_common_position_rank_inputs": len(scores),
            "mean_raw_common_position_rank": mean(scores) if scores else None,
        }
    return tables


def readout():
    rows = verify()
    responses, profiles, calls = scoped(
        opai.observations, {"TRIAL": TRIAL, "PUBLIC": PUBLIC, "WIDTH": WIDTH, "__file__": __file__}
    )(rows)
    return {
        "schema": "chipmunk.text-elife-passage-score-result/275",
        "status": "complete",
        "freeze_sha256": sha(PUBLIC),
        "planned_inputs": len(rows),
        "source_groups": len(rows) // len(ROLES),
        "actual_forwards": calls,
        "resource_profile_sha256": profiles,
        "by_generator_and_method": summarize(rows, responses, read(prior.historical.PRIOR / "points.json")),
        "serving_changed": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for action in ("preflight", "freeze", "readout"):
        commands.add_parser(action)
    for action in ("run", "_worker"):
        command = commands.add_parser(action)
        command.add_argument("batch", type=int, choices=range(BATCHES))
        if action == "_worker":
            command.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.command == "run":
        run(args.batch)
    elif args.command == "_worker":
        worker(args.batch, args.output)
    else:
        action = {"preflight": lambda: identity(inputs()), "freeze": freeze, "readout": readout}[args.command]
        print(json.dumps(action(), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
