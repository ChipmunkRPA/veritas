"""Evaluate per-sentence draft coverage without requiring all edits to align."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

from scripts.writing_paired_scale_transfer import POINT, cases, increases, measure, sha

SCHEMA = "chipmunk.writing-independent-anchor-transfer/1"
METHOD = {
    "baseline": "Use the frozen same-domain paired assessment and require its exact source hashes and target rosters.",
    "candidate": "Keep all existing readings. If a draft is outside the current scope, map separate sentences "
                 "between globally unique pairs of unchanged neighbors within their respective paragraphs. "
                 "Refuse reordered or overlapping neighborhoods. Diff only within each mapped sentence.",
    "rules": "On the additional regions compare the existing both-check context odds rule with the historical "
             "two-scale rule. Each arm retains every incumbent outcome. Count exact labeled sentences and "
             "distinct flagged ordinary works. Wrong and unmatched locations remain in the denominator.",
    "adoption": "Only trial an arm locally with at least six added exact edits across three held works "
                "and both generators, at most two additional ordinary works, and no more than two other "
                "highlighted locations whose authorship is unassigned. Adopt at most one arm, preferring the "
                "one with more additional exact edits then lower ordinary and unknown flags.",
}


def prepare(base, root, directory, domain):
    from app.evidence import document_revision as draft
    from app.evidence import document_revision_writing as paired

    *inputs, metadata = cases(root, directory, domain)
    if metadata != base["sources"] or len(inputs) != len(base["rows"]):
        raise ValueError("The source bank differs from the previous frozen receipt")
    rows, examples = [], []
    for (bank, earlier, later, truth, variant), incumbent in zip(inputs, base["rows"], strict=True):
        if (bank, [sha(earlier), sha(later)], truth, variant) != (
                incumbent["bank"], incumbent["source_sha256"], incumbent["truth"], incumbent["variant"]):
            raise ValueError("The source pair differs from its previous commitment")
        row = {key: incumbent[key] for key in ("bank", "source_sha256", "truth", "scope", "ranges", "readings")}
        row["independent"] = []
        candidates = []
        if incumbent["scope"] != "current":
            comparison, before, after = draft.compare_with_text(earlier.encode(), later.encode(), ".txt")
            candidates = paired.independently_anchored_revisions(comparison, before, after)
            for candidate in candidates:
                row["independent"].append({
                    side: {k: v for k, v in candidate.geometry[side].items() if k != "text"}
                    for side in ("earlier", "later")
                } | {"wording_changes": candidate.geometry["wording_changes"]})
        rows.append(row)
        examples.append(candidates)
    return rows, examples


def summarize(rows):
    counts = Counter()
    groups = {arm: {"works": set(), "families": set(), "ordinary": set()} for arm in ("context", "two_scale")}
    ordinary_assessed = set()
    for row in rows:
        truth = {tuple(label["later"]): label for label in row["truth"]}
        generated = {i for i, label in truth.items() if label["family"] != "ordinary"}
        counts["generated_total"] += len(generated)
        if row["scope"] == "current":
            for extent, reading in zip(row["ranges"], row["readings"], strict=True):
                if reading.get("context"):
                    key = tuple(extent["later"])
                    label = truth[key]
                    if key in generated:
                        counts["incumbent_assessed"] += 1
                        counts["incumbent_exact"] += min(increases(reading["context"])) >= math.log(2)
                    else:
                        ordinary_assessed.add(label["group"])
            if row["independent"]:
                raise ValueError("An incumbent reading cannot be replaced")
        for evidence in row["independent"]:
            key = tuple(evidence["later"]["context_spans"][1])
            reading = evidence["reading"]
            label = truth.get(key)
            if key != (evidence["later"]["start"], evidence["later"]["end"]):
                raise ValueError("The later sentence does not match its context")
            if not reading["context"]:
                continue
            context = increases(reading["context"])
            center = increases(reading["center"]) if reading["center"] else []
            counts["added_assessed"] += key in generated
            if label and label["family"] == "ordinary":
                ordinary_assessed.add(label["group"])
            for arm, marked in (("context", min(context) >= math.log(2)),
                                ("two_scale", bool(center) and min(context + center) > POINT)):
                if not marked:
                    continue
                if key in generated:
                    counts[f"{arm}/exact"] += 1
                    groups[arm]["works"].add(label["group"])
                    groups[arm]["families"].add(label["family"])
                elif label:
                    groups[arm]["ordinary"].add(label["group"])
                else:
                    counts[f"{arm}/unassigned"] += 1
    result = dict(counts) | {"ordinary_works_assessed": len(ordinary_assessed)}
    for arm, covered in groups.items():
        exact = counts[f"{arm}/exact"]
        unknown = counts[f"{arm}/unassigned"]
        works, families, ordinary = (len(covered[key]) for key in ("works", "families", "ordinary"))
        result[arm] = {"additional_exact_edits": exact, "additional_edited_works": works,
                       "generator_families": sorted(covered["families"]), "additional_ordinary_works": ordinary,
                       "unassigned_flags": unknown,
                       "passes_local_trial": exact >= 6 and works >= 3 and families == 2 and ordinary <= 2 and unknown <= 2}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "score", "review"))
    parser.add_argument("file", type=Path)
    parser.add_argument("--prior", type=Path)
    parser.add_argument("--root", type=Path, default=Path("data/independent-evaluation"))
    parser.add_argument("--news", type=Path)
    parser.add_argument("--abstracts", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.action == "review":
        packet = json.loads(args.file.read_text())
        summary = summarize(packet["rows"])
        if packet["schema"] != SCHEMA or packet["method"] != METHOD or summary != packet["summary"]:
            raise ValueError("The receipt does not replay")
        print(json.dumps(summary, indent=2), flush=True)
        return
    if not args.prior or (args.news is None) == (args.abstracts is None):
        parser.error("freeze and score require the previous receipt and either --news or --abstracts")
    source = args.prior.read_text()
    rows, inputs = prepare(json.loads(source), args.root, args.news or args.abstracts, "news" if args.news else "abstracts")
    packet = {"schema": SCHEMA, "method": METHOD, "previous_sha256": sha(source),
              "runner_sha256": sha(Path(__file__).read_text()), "rows": rows}
    if args.action == "freeze":
        print(Counter((row["bank"], row["scope"], len(row["independent"])) for row in rows), flush=True)
        target = args.file
    else:
        if packet != json.loads(args.file.read_text()) or args.output is None:
            raise ValueError("The frozen independent comparisons differ or no output was specified")
        import torch

        torch.set_num_threads(2)
        for row, readings in zip(rows, measure(inputs), strict=True):
            for evidence, reading in zip(row["independent"], readings, strict=True):
                evidence["reading"] = reading
        packet["summary"] = summarize(rows)
        print(json.dumps(packet["summary"], indent=2), flush=True)
        target = args.output
    with target.open("x") as stream:
        json.dump(packet, stream, separators=(",", ":"))
        stream.write("\n")


if __name__ == "__main__":
    main()
