"""Replay PEP source bindings and decisions at the earlier phrase-study points."""

from __future__ import annotations

import argparse
import math
from collections import Counter, defaultdict
from pathlib import Path

from app.detectors.text.extract import text_hash
from scripts import review_writing_phrase_context_stability as reviewer
from scripts import writing_pep_phrase_replication as replication
from scripts import writing_pep_revision_intake as pep
from scripts import writing_phrase_context_stability as trial
from scripts.video_appearance_native_v108 import read, require, write
from scripts.writing_focus_v3 import digest


def archive_bindings(sources):
    result = []
    for row in sources["holdout"]:
        source = row["source"]
        raw = pep.git(replication.ARCHIVE, "show", f"{source['commit']}:{source['path']}")
        notice = raw[slice(*source["notice_span"])]
        require(text_hash(raw) == source["blob_sha256"] and text_hash(notice) == source["notice_sha256"]
                and pep.LICENSE.search(notice), "Source or reuse notice changed")
        require(row["before"]["raw_span"][1] < row["after"]["raw_span"][0], "Source passages overlap")
        ranges = []
        for side in ("before", "after"):
            passage = row[side]
            for anchor in passage["witness"]:
                a, b = anchor["raw_span"]
                require(0 <= a < b <= source["notice_span"][0] and text_hash(raw[a:b]) == anchor["raw_sha256"]
                        and anchor["lines"] == [raw.count("\n", 0, a) + 1, raw.count("\n", 0, b - 1) + 1],
                        "A source sentence range or line does not replay")
                ranges.append({"range": [a, b], "sha256": anchor["raw_sha256"]})
            focus = passage["text"][slice(*passage["focus"])]
            require(focus == passage["focus_text"] == passage["witness"][2]["text"]
                    and " ".join(part["text"] for part in passage["witness"]) == passage["text"],
                    "Archived source and scored passage differ")
        result.append({"group": row["group"], **source, "sentence_ranges": ranges})
    return result


def verify_points(points):
    prior = replication.PRIOR
    fixtures = read(prior / "fixtures.json")["fixtures"]
    calibration, inputs, _ = reviewer.evaluate_phase(prior, fixtures, "calibration")
    for index, arm in enumerate(trial.ARMS[1:-1]):
        groups = defaultdict(lambda: -math.inf)
        for row, _, candidates, _ in calibration:
            groups[row["group"]] = max(groups[row["group"]], max(scores[index] for _, scores in candidates))
        require(len(groups) == 11 and math.isclose(sorted(groups.values())[-2], points[arm], rel_tol=1e-12, abs_tol=1e-12),
                "Original operating points do not replay")
    return inputs


def compare_cases(held, result, points):
    counts, false_groups = defaultdict(Counter), defaultdict(set)
    added = set()
    positions = 0
    for archived, (row, baseline, candidates, count) in zip(result["cases"], held, strict=True):
        require((archived["group"], archived["kind"], archived["fixture_sha256"], archived["target"], archived["candidate_count"])
                == (row["group"], row["kind"], row["sha256"], row["span"], count), "Held record misbound")
        positions += count
        proposed = {"current": [region for region, _ in baseline]}
        chosen = {"current": [region for region, score in baseline if score >= .5][:6]}
        for index, arm in enumerate(trial.ARMS[1:-1]):
            best = max((scores[index] for _, scores in candidates), default=-math.inf)
            winners = [[region] for region, scores in candidates if scores[index] == best]
            chosen[arm] = winners if len(winners) == 1 and best > points[arm] else []
            proposed[arm] = [[region] for region, _ in candidates]
        joined = trial.ARMS[-1]
        chosen[joined] = chosen["current"] + chosen["local_contrast"]
        proposed[joined] = proposed["current"] + proposed["local_contrast"]
        kind = row["kind"]
        generated = kind in trial.SEEDS
        counts[kind]["cases"] += 1
        for arm in trial.ARMS:
            useful = generated and reviewer.overlap_good(row["text"], row["span"], chosen[arm])
            possible = generated and reviewer.overlap_good(row["text"], row["span"], proposed[arm])
            require(archived["outcomes"][arm] == {"spans": chosen[arm], "useful": useful, "geometry_allows_useful": possible},
                    "Exact location or disposition differs")
            counts[kind].update({f"{arm}_flag": int(bool(chosen[arm])), f"{arm}_useful": int(useful),
                                 f"{arm}_possible": int(possible)})
            if chosen[arm] and not generated:
                false_groups[arm].add(row["group"])
        if archived["outcomes"][joined]["useful"] and not archived["outcomes"]["current"]["useful"]:
            added.add((row["group"], kind))
    extra = len(false_groups[trial.ARMS[-1]] - false_groups["current"])
    advance = (len({work for work, _ in added}) >= 2 and {kind for _, kind in added} == set(trial.SEEDS)
               and extra <= 1 and all(counts[family]["cases"] >= 6 for family in trial.SEEDS))
    require(result["counts"] == dict(counts) and result["false_groups"] == {arm: len(false_groups[arm]) for arm in trial.ARMS}
            and result["added_useful_groups"] == sorted(map(list, added)) and result["new_false_groups"] == extra
            and result["advance"] == advance, "Aggregate arithmetic or frozen gate differs")
    return positions


def review(run):
    sources = read(run / "sources.json")
    require(replication.qualify() == sources, "Source-only qualifying replay differs")
    frozen = read(run / "freeze.json")
    points = replication.prior_points()
    require(frozen["replication_schema"] == replication.SCHEMA and frozen["frozen_points"] == points
            and frozen["method"] == trial.METHOD and frozen["replication_method"] == replication.METHOD
            and frozen["generation_mode"] == "sentence_fragment"
            and frozen["source_sha256"] == digest(run / "sources.json")
            and frozen["prior_receipt_sha256"] == digest(replication.PUBLIC_PRIOR)
            and frozen["impl_sha256"] == digest(Path(trial.__file__))
            and frozen["replication_sha256"] == digest(Path(replication.__file__)), "Original trial or replication freeze differs")
    fixtures = read(run / "fixtures.json")
    require(trial.build(run) == fixtures, "Exact fixture selection differs")
    held, inputs, forwards = reviewer.evaluate_phase(run, fixtures["fixtures"], "holdout")
    result = read(run / "result.json")
    positions = compare_cases(held, result, points)
    calibration_inputs = verify_points(points)
    witnesses = archive_bindings(sources)
    return {"schema": replication.SCHEMA + "-audit", "method": frozen["replication_method"],
            "archive_commit": pep.ANCHOR, "intake_paths": sources["intake_paths"],
            "source_qualification": sources["source_qualification"], "prior_corpus": sources["prior_corpus"],
            "prior_receipt_sha256": frozen["prior_receipt_sha256"], "points": points,
            "source_witnesses": witnesses, "source_sentence_ranges_verified": sum(len(row["sentence_ranges"]) for row in witnesses),
            "held_works": len(witnesses), "calibration_inputs_replayed": calibration_inputs, "held_inputs": inputs,
            "held_classifier_forwards": forwards, "held_candidate_positions_verified": positions,
            "held_dispositions_verified": len(held) * len(trial.ARMS), "admission": fixtures["counts"], "result": result,
            "retrospective_at_matched_false_group_budgets": reviewer.retrospective(held),
            "reviewer_sha256": digest(Path(__file__)), "commitments": {name: digest(run / name) for name in (
                "freeze.json", "sources.json", "fixtures.json", "generate-qwen.json", "generate-smol2.json", "holdout-responses.json", "result.json")}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = review(args.run)
    write(args.output, result)
    print(f"Verified {result['held_dispositions_verified']} held dispositions and {result['source_sentence_ranges_verified']} archive ranges.", flush=True)


if __name__ == "__main__":
    main()
