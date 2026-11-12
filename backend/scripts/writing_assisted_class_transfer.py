"""Measure AI-assisted class responses on source-separated retained writing."""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

from scripts.writing_focus_v3 import BASELINE_ASSETS, digest, text_sha

ROOT = Path(__file__).resolve().parents[2]
PUBLIC = ROOT / "docs/evaluation-panels"
BASE = ROOT / "backend/data/independent-evaluation"
BLOG = BASE / "writing-archive-prospective-transfer-2026-11-12"
JOURNAL = BASE / "writing-journal-body-transfer-2026-11-11"
BLOG_RECEIPT = PUBLIC / "writing-archived-passage-transfer-2026-11-12.json"
JOURNAL_RECEIPT = PUBLIC / "writing-journal-body-transfer-2026-11-11.json"
OUTPUT = PUBLIC / "writing-assisted-class-transfer-2026-11-12.json"
SCHEMA = "chipmunk.writing-assisted-class-transfer/1"
FAMILIES = {"qwen", "smol2"}
ARMS = ("total_top", "assisted_top", "assisted_relative")
METHOD = {
    "selection": "First 16 technical blog groups by SHA-256 of the retained group ID. Only ordinary before/after views set the points.",
    "held": "All Wellcome and Gates held groups, measured as single files. Both actual ordinary body versions and admitted local rewrites.",
    "responses": "Current 1-human, assisted=edited+humanized, and assisted log odds minus the median assisted log odds of the exact native contexts in that same file.",
    "current": "First six native contexts whose current raw response is at least 0.50, in source order.",
    "alternates": "Choose the unique highest context per view. Require strictly above its selection-only ordinary-group order statistic matching the incumbent number of ordinary groups flagged. Tied boundary values abstain.",
    "outcome": "Useful measured context requires 0.30 non-whitespace IoU and 0.50 recall. Report exact winning focus and incorrect locations separately.",
    "advance": "The assisted-only measured-context arm must add at least three useful works across both generators versus the current measured context, lose at most one existing useful work, and add at most one ordinary flagged work. No confidence or serving changes.",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_bytes())


def write(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def prep():
    from app.detectors.text.extract import from_string
    from app.detectors.text.segment import segment
    from scripts.writing_archive_author_transfer import inputs

    retained = read(BLOG_RECEIPT)["audit"]["commitments"]
    later = read(JOURNAL_RECEIPT)["commitments"]
    for filename in ("input-lock.json", "originals.json", "responses.json"):
        require(digest(BLOG / filename) == retained[filename], "Retained technical blog commitment differs")
    for filename in ("scored/cases.json", "scored/current.json", "scored/scoring-freeze.json"):
        require(digest(JOURNAL / filename) == later[filename], "Retained journal commitment differs")
    archived, _, _ = inputs(BLOG)
    public_rows = read(BLOG / "input-lock.json")["rows"]
    require([{k: v for k, v in row.items() if k != "text"} for row in archived] == public_rows,
            "Reconstructed archived inputs differ")
    selected = set(sorted({row["group"] for row in archived if row["archive"] == "kubernetes"}, key=text_sha)[:16])
    groups = {"fit": [row for row in archived if row["group"] in selected],
              "held": read(JOURNAL / "scored/cases.json")}
    baselines = {"fit": read(BLOG / "responses.json")["responses"],
                 "held": read(JOURNAL / "scored/current.json")["responses"]}
    rows, values = {}, {}
    for phase, original in groups.items():
        rows[phase], values[phase] = [], {}
        for row in original:
            text = row["text"]
            doc = from_string(text)
            units = segment(doc)
            require(doc.text == text and text_sha(text) == row["text_sha256"]
                    and len(units) == len(row["windows"]), "Source text or native segmentation differs")
            windows = []
            for i, witness in enumerate(row["windows"]):
                spans = [[unit.start, unit.end] for unit in units[max(0, i - 1):i + 2]]
                exact = " ".join(text[a:b] for a, b in spans)
                pin = witness["sha256"]
                baseline = baselines[phase][pin]
                require(spans == witness["spans"] and text_sha(exact) == pin
                        and baseline["coverage"]["token_windows"] == 1, "Exact previous model input differs")
                values[phase][pin] = {"input": exact, "baseline": baseline}
                windows.append({"input": pin, "spans": spans, "focus": [units[i].start, units[i].end]})
            rows[phase].append({"group": row["group"], "publisher": row.get("publisher", "kubernetes"), "role": row["role"],
                                "text": text, "truth": row["focus"] if row["role"] in FAMILIES else None, "windows": windows})
    require(len(selected) == 16 and len({r["group"] for r in rows["held"]}) == 21
            and not selected & {r["group"] for r in rows["held"]}
            and not values["fit"].keys() & values["held"].keys(), "Source and model input splits overlap")
    return rows, values


def freeze(run):
    rows, values = prep()
    write(run / "cases.json", rows)
    write(run / "inputs.json", values)
    write(run / "freeze.json", {"schema": SCHEMA, "method": METHOD, "implementation": digest(Path(__file__)),
                                "cases_sha256": digest(run / "cases.json"), "inputs_sha256": digest(run / "inputs.json"),
                                "assets": BASELINE_ASSETS,
                                "source_receipts": {p.name: digest(p) for p in (BLOG_RECEIPT, JOURNAL_RECEIPT)}})
    print({phase: {"cases": len(rows[phase]), "unique_inputs": len(values[phase])} for phase in rows}, flush=True)


def check(run):
    lock = read(run / "freeze.json")
    require(lock["schema"] == SCHEMA and lock["method"] == METHOD and lock["implementation"] == digest(Path(__file__))
            and lock["cases_sha256"] == digest(run / "cases.json") and lock["inputs_sha256"] == digest(run / "inputs.json"),
            "Research freeze differs")
    return lock


def score(run, phase, snapshot):
    require(os.environ.get("HF_HUB_OFFLINE") == os.environ.get("TRANSFORMERS_OFFLINE") == "1", "Use offline inference")
    lock = check(run)
    require(all(digest(snapshot / name) == pin for name, pin in lock["assets"].items()), "Pinned checkpoint differs")
    if phase == "held":
        verify_points(run)
    import torch

    from app.detectors.text.classifier import FourClassTextClassifier

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    classifier = FourClassTextClassifier.from_snapshot(snapshot, torch.device("cpu"))
    inputs = read(run / "inputs.json")[phase]
    keys = sorted(inputs)
    results = {}
    max_difference = 0.
    for start in range(0, len(keys), 4):
        chunk = keys[start:start + 4]
        data = classifier.tok([inputs[key]["input"] for key in chunk], padding=True,
                              return_tensors="pt", return_special_tokens_mask=True)
        special = data.pop("special_tokens_mask")
        valid = data["attention_mask"].bool()
        counts = (valid & ~special.bool()).sum(dim=1).tolist()
        require(data["input_ids"].shape[1] <= 512, "A supposedly complete context would overflow")
        with torch.inference_mode():
            logits = classifier.model(data["input_ids"], attention_mask=data["attention_mask"])["logits"]
            probabilities = logits.softmax(dim=-1)
        for key, count, token_ids, mask, values in zip(chunk, counts, data["input_ids"], valid, probabilities.tolist(), strict=True):
            expected = inputs[key]["baseline"]
            delta = abs((1 - values[0]) - expected["score"])
            max_difference = max(max_difference, delta)
            require(count == expected["coverage"]["tokens"] > 0 and delta <= 1e-5
                    and len(values) == 4 and all(math.isfinite(p) and 0 <= p <= 1 for p in values)
                    and abs(sum(values) - 1) < 1e-6, "Native model response differs from the committed current score")
            results[key] = {"p": values, "tokens": count, "token_ids_sha256": text_sha(json.dumps(token_ids[mask].tolist()))}
        if start % 80 == 0:
            print(phase, len(results), "/", len(keys), flush=True)
    write(run / f"{phase}.json", {"inputs_sha256": lock["inputs_sha256"], "measurements": results,
                                "forwards": (len(keys) + 3) // 4, "max_current_response_difference": max_difference})


def maximum(row, measured):
    p = [measured[w["input"]]["p"] for w in row["windows"]]
    assisted = [min(1 - 1e-8, max(1e-8, values[2] + values[3])) for values in p]
    odds = [math.log(value / (1 - value)) for value in assisted]
    prior = median(odds)
    return {"total_top": [1 - values[0] for values in p], "assisted_top": assisted,
            "assisted_relative": [value - prior for value in odds]}


def points(run):
    rows = read(run / "cases.json")["fit"]
    measured = read(run / "fit.json")["measurements"]
    groups = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row["role"] not in FAMILIES:
            for arm, scores in maximum(row, measured).items():
                groups[row["group"]][arm].append(max(scores))
    allowance = sum(max(row["total_top"]) >= .5 for row in groups.values())
    require(allowance < len(groups), "No ordinary selection group remains for a finite cutoff")
    limits = {arm: sorted((max(group[arm]) for group in groups.values()), reverse=True)[allowance] for arm in ARMS}
    return {"ordinary_group_count": len(groups), "ordinary_flags_allowed": allowance, "limits": limits,
            "fit_sha256": digest(run / "fit.json"), "source_freeze_sha256": digest(run / "freeze.json")}


def verify_points(run):
    current = points(run)
    require(read(run / "points.json") == current, "Frozen ordinary-only operating points differ")
    return current


def useful(text, truth, spans):
    if not truth:
        return False
    actual = {i for a, b in spans for i in range(a, b) if not text[i].isspace()}
    expected = {i for i in range(*truth) if not text[i].isspace()}
    overlap = len(actual & expected)
    return 10 * overlap >= 3 * len(actual | expected) and 2 * overlap >= len(expected)


def outcomes(rows, measured, limits):
    observations = []
    for row in rows:
        windows = row["windows"]
        scores = maximum(row, measured)
        first = [i for i, value in enumerate(scores["total_top"]) if value >= .5][:6]
        selected = {"current": first}
        for arm in ARMS:
            values = scores[arm]
            top = max(values)
            selected[arm] = [values.index(top)] if top > limits[arm] and values.count(top) == 1 else []
        observations.append({"group": text_sha(row["group"])[:16], "publisher": row["publisher"], "role": row["role"],
                             "measures": {arm: {"flag": bool(indices), "useful": any(
                                 useful(row["text"], row["truth"], windows[i]["spans"]) for i in indices),
                                 "exact_focus": any(windows[i]["focus"] == row["truth"] for i in indices)}
                                 for arm, indices in selected.items()}})
    return observations


def summarize(rows):
    metrics = {}
    current_false_groups = {row["group"] for row in rows if row["role"] not in FAMILIES and row["measures"]["current"]["flag"]}
    for arm in ("current", *ARMS):
        hit_groups, false_groups, new_false_groups, lost_groups = set(), set(), set(), set()
        gains = defaultdict(set)
        counts = Counter()
        for row in rows:
            measured, prior = row["measures"][arm], row["measures"]["current"]
            group = row["group"]
            if row["role"] in FAMILIES:
                counts.update(generated=1, useful=measured["useful"], exact_focus=measured["exact_focus"],
                              incorrect_only=measured["flag"] and not measured["useful"])
                if measured["useful"]:
                    hit_groups.add(group)
                    if not prior["useful"]:
                        gains[row["role"]].add(group)
                elif prior["useful"]:
                    lost_groups.add(group)
            else:
                counts.update(ordinary=1, ordinary_flags=measured["flag"])
                if measured["flag"]:
                    false_groups.add(group)
                    if not prior["flag"]:
                        new_false_groups.add(group)
        metrics[arm] = dict(counts) | {"hit_groups": len(hit_groups), "false_groups": len(false_groups),
                                     "gained_groups": len(set.union(set(), *gains.values())),
                                     "lost_groups": len(lost_groups), "added_false_groups": len(false_groups - current_false_groups),
                                     "groups_with_new_ordinary_flags": len(new_false_groups),
                                     "gains_by_generator": {kind: len(groups) for kind, groups in gains.items()}}
    return metrics


def publish(run):
    lock = check(run)
    point = verify_points(run)
    rows, values = prep()
    require(read(run / "cases.json") == rows and read(run / "inputs.json") == values, "Native source replay differs")
    result = {"schema": SCHEMA, "method": METHOD, "freeze_sha256": digest(run / "freeze.json"), "points": point}
    for phase in ("fit", "held"):
        recorded = read(run / f"{phase}.json")
        require(recorded["inputs_sha256"] == lock["inputs_sha256"] and recorded["measurements"].keys() == values[phase].keys(),
                "Complete response receipt required")
        observed = outcomes(rows[phase], recorded["measurements"], point["limits"])
        result[phase] = {"rows": observed, "metrics": summarize(observed), "forwards": recorded["forwards"],
                         "max_current_response_difference": recorded["max_current_response_difference"]}
    candidate = result["held"]["metrics"]["assisted_top"]
    result["advance"] = (candidate["gained_groups"] >= 3 and set(candidate["gains_by_generator"]) == FAMILIES
                         and candidate["lost_groups"] <= 1 and candidate["added_false_groups"] <= 1)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "fit", "points", "held", "publish", "verify"))
    parser.add_argument("run", type=Path)
    parser.add_argument("--snapshot", type=Path)
    args = parser.parse_args()
    if args.phase == "freeze":
        freeze(args.run)
    elif args.phase in ("fit", "held"):
        score(args.run, args.phase, args.snapshot)
    elif args.phase == "points":
        check(args.run)
        write(args.run / "points.json", points(args.run))
    else:
        result = publish(args.run)
        if args.phase == "publish":
            write(OUTPUT, result)
        else:
            require(read(OUTPUT) == result, "Published result differs from native replay")
        print(result["held"]["metrics"], flush=True)


if __name__ == "__main__":
    main()
