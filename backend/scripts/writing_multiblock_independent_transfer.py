"""Compare independent exact-sentence and omission reads on multi-paragraph documents."""
from __future__ import annotations

import argparse
import math
from collections import defaultdict
from pathlib import Path

from scripts.video_appearance_native_v108 import read, require, write
from scripts.writing_blind_rewrite_probe import text_hash, words
from scripts.writing_body_localization_transfer import SECONDARY_ASSETS, observe, span_positions
from scripts.writing_focus_v3 import BASELINE_ASSETS, digest

SCHEMA = "chipmunk.writing-multiblock-independent-transfer/1"
INPUT_FILES = ("inputs.json", "primary.json", "sources.json")
PARENT_RECEIPT = Path(__file__).resolve().parents[2] / "docs/evaluation-panels/writing-multiparagraph-localization-transfer-2026-11-12.json"
NAMES = ("primary_focus", "independent_focus", "omit_center", "combined")
FALSE_BUDGET = 2
PLAN = {
    "scope": "Retained English plain-text articles from the earlier multi-paragraph trial. Fit using the 12 earlier works and hold out the ten additional works. Add one new matched ordinary same-work substitution per original when exact segmentation and length agree.",
    "candidate": "An interior native sentence with 12–65 words and adjacent sentences of 8–80 words. Native contexts may cross a paragraph break. Omit first and last sentences. Use the existing exact three-sentence input and read the isolated center and flanks and the left-right input with the center omitted.",
    "ordinary_splice": "Take the closest-length eligible sentence from the disjoint genuine same-article excerpt already extracted for the MIME quote. Require 12–65 words, within half to twice target word length, distinct from every original sentence and complete sentence punctuation. Resolve ties by donor source order. Verify that only one native sentence changed.",
    "reference": "First six exact native primary contexts scoring at least .50. Use each complete measured context for overlap, including if it crosses a paragraph break.",
    "features": ["native primary log odds", "isolated independent log odds",
                 "independent full three-sentence log odds minus its center-omitted log odds",
                 "isolated independent log odds above the stronger isolated neighbor"],
    "fit": "Standardize with fitting observations only, then fit balanced L2 logistic regression with C=1. Positive observations are exact known generated centers. Negative observations are all eligible centers in ordinary originals and the exact center of each ordinary substitution. Compare the three unfitted raw scores and the combined fit.",
    "point": "Each arm selects its unique top focus and requires strict excess of the third-largest group maximum across fitting ordinary originals and ordinary substitutions. A group alerts if either ordinary file alerts.",
    "overlap": "A useful result covers at least half the known edited nonwhitespace characters and has at least .30 intersection over union. Separate false ordinary groups, ordinary changed centers selected, wrong locations and unassessed.",
    "advance": "Use the combined rule as a supplemental cue only if it adds two held useful locations in distinct work groups and across both generators, without more than one new ordinary work group. Require complete measured geometry on at least 12 of 14 known edits. Compare against the earlier held baseline and preserve its useful locations by union.",
}


def checked_assets(path, assets):
    require(all(digest(path / name) == pin for name, pin in assets.items()), "Checkpoint assets differ from the audited pins")


def key_input(values, text):
    pin = text_hash(text)
    values[pin] = text
    return pin


def cases(parent):
    from app.detectors.text.extract import from_string
    from app.detectors.text.segment import segment

    incoming = read(parent / "inputs.json")
    cached_primary = read(parent / "primary.json")["responses"]
    work = {row["group"]: row for row in read(parent / "sources.json")["works"]}
    results, primary, independent = [], {}, {}
    for base in incoming["rows"]:
        if base["format"] != "txt":
            continue
        path = parent / "fixtures" / base["file"]
        original = path.read_bytes().decode("utf-8")
        require(digest(path) == base["file_sha256"] and text_hash(original) == base["text_sha256"], "Source fixture differs")
        samples = [(base["kind"], original, base["truth"], None)]
        if base["kind"] == "human":
            source = work[base["group"]]
            a, start, end = source["focus"]
            lo = sum(len(row["text"]) + 2 for row in source["paragraphs"][:a]) + start
            hi = lo + end - start
            quote = source["quote"]
            require(text_hash(quote) == source["quote_sha256"], "Ordinary donor source differs")
            originals = [sentence.text for sentence in segment(from_string(original))]
            source_length = len(words(original[lo:hi]))
            donor = [(abs(len(words(sentence.text)) - source_length), sentence.start, sentence.end)
                     for sentence in segment(from_string(quote))
                     if (sentence.text not in original and sentence.text.endswith((".", "?", "!"))
                         and max(12, source_length / 2) <= len(words(sentence.text)) <= min(65, source_length * 2))]
            if donor:
                _, a, b = min(donor)
                replacement = quote[a:b]
                altered = original[:lo] + replacement + original[hi:]
                focus = next(i for i, item in enumerate(base["windows"]) if item["focus"] == [lo, hi])
                expected = [*originals[:focus], replacement, *originals[focus + 1:]]
                if [sentence.text for sentence in segment(from_string(altered))] == expected:
                    samples.append(("splice", altered, [lo, lo + len(replacement)],
                                    {"quote_sha256": text_hash(quote), "donor_span": [a, b], "donor_sha256": text_hash(replacement)}))
        for kind, body, truth, donor in samples:
            doc = from_string(body)
            native = segment(doc)
            require(doc.text == body and len(native) == len(base["windows"]), "Native source geometry differs")
            require(truth is None or any([unit.start, unit.end] == truth for unit in native), "Known changed center is not an exact sentence")
            windows, eligible = [], []
            counts = [len(words(item.text)) for item in native]
            for i, unit in enumerate(native):
                group = native[max(0, i - 1):i + 2]
                focus = [unit.start, unit.end]
                spans = [[item.start, item.end] for item in group]
                value = " ".join(item.text for item in group)
                pin = key_input(primary, value)
                row = {"focus": focus, "spans": spans, "primary": pin}
                if kind != "splice":
                    prior = base["windows"][i]
                    require((prior["focus"], prior["spans"], prior["sha256"]) == (focus, spans, pin)
                            and cached_primary[pin]["coverage"]["token_windows"] == 1,
                            "Primary context differs from its earlier complete native response")
                windows.append(row)
                if 0 < i < len(native) - 1 and 12 <= counts[i] <= 65 and all(8 <= counts[j] <= 80 for j in (i - 1, i + 1)):
                    left, center, right = (part.text for part in group)
                    row["independent"] = {label: key_input(independent, text) for label, text in (
                        ("left", left), ("center", center), ("right", right), ("full", value), ("omitted", f"{left} {right}"))}
                    eligible.append(i)
            results.append({"group": base["group"], "partition": "held" if base["scope"] == "fresh" else "fit",
                            "kind": kind, "text": body, "text_sha256": text_hash(body), "truth": truth,
                            "source_sha256": base["file_sha256"], "donor": donor, "windows": windows, "eligible": eligible})
    groups = {partition: {row["group"] for row in results if row["partition"] == partition} for partition in ("fit", "held")}
    require(len(groups["fit"]) == 12 and len(groups["held"]) == 10 and not groups["fit"] & groups["held"],
            "The sealed work groups differ")
    return results, {pin: value for pin, value in primary.items() if pin not in cached_primary}, independent


def freeze(parent, run, primary, independent):
    checked_assets(primary, BASELINE_ASSETS)
    checked_assets(independent, SECONDARY_ASSETS)
    parent_hashes = {name: digest(parent / name) for name in INPUT_FILES}
    published = read(PARENT_RECEIPT)["audit"]["commitments"]
    require(all(published[name] == value for name, value in parent_hashes.items()), "Parent input differs from the published audit")
    records, missing, second = cases(parent)
    run.mkdir(mode=0o700)
    write(run / "cases.json", records)
    write(run / "inputs.json", {"primary_missing": missing, "independent": second})
    write(run / "freeze.json", {"schema": SCHEMA, "plan": PLAN, "runner_sha256": digest(Path(__file__)),
          "parent": parent_hashes, "parent_receipt_sha256": digest(PARENT_RECEIPT),
          "cases_sha256": digest(run / "cases.json"), "inputs_sha256": digest(run / "inputs.json"),
          "primary": BASELINE_ASSETS, "independent": SECONDARY_ASSETS})
    print(f"Froze {len(records)} files, {len(missing)} new primary inputs and {len(second)} independent inputs", flush=True)


def checked(parent, run):
    lock = read(run / "freeze.json")
    require(lock["schema"] == SCHEMA and lock["plan"] == PLAN and lock["runner_sha256"] == digest(Path(__file__)),
            "Frozen comparison differs from the running method")
    require(lock["parent"] == {name: digest(parent / name) for name in INPUT_FILES}
            and lock["cases_sha256"] == digest(run / "cases.json") and lock["inputs_sha256"] == digest(run / "inputs.json"),
            "Frozen source or input changed")
    return read(run / "cases.json"), read(run / "inputs.json")


def log_odds(p):
    p = max(1e-7, min(1 - 1e-7, p))
    return math.log(p) - math.log1p(-p)


def measure(geometry, primary, independent):
    first = primary[geometry["primary"]]
    other = {label: independent[pin] for label, pin in geometry["independent"].items()}
    require(all(entry["coverage"]["token_windows"] == 1 and entry["coverage"]["tokens"] > 0
                for entry in [first, *other.values()]), "A planned reading was not complete")
    p = {label: log_odds(entry["score"]) for label, entry in other.items()}
    return [log_odds(first["score"]), p["center"], p["full"] - p["omitted"], p["center"] - max(p["left"], p["right"])]


def fit(rows):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    x, y = [], []
    for case in rows:
        if case["partition"] != "fit":
            continue
        for window in case["windows"]:
            is_target = window["focus"] == case["truth"]
            if case["kind"] == "human" or is_target:
                x.append(window["features"])
                y.append(int(is_target and case["kind"] != "splice"))
    scaler = StandardScaler().fit(x)
    clf = LogisticRegression(class_weight="balanced", C=1, solver="lbfgs", max_iter=1000,
                             random_state=0).fit(scaler.transform(x), y)
    return {"means": scaler.mean_.tolist(), "scales": scaler.scale_.tolist(),
            "coefficients": clf.coef_[0].tolist(), "intercept": float(clf.intercept_[0]),
            "rows": len(y), "positives": sum(y)}


def ranks(row, model):
    combined = model["intercept"] + math.fsum(
        weight * ((value - mean) / scale)
        for value, mean, scale, weight in zip(row, model["means"], model["scales"], model["coefficients"], strict=True))
    return dict(zip(NAMES, (*row[:3], combined), strict=True))


def unique_top(case, arm, model):
    order = sorted(((ranks(row["features"], model)[arm], i) for i, row in enumerate(case["windows"])), reverse=True)
    return order[0] if order and (len(order) == 1 or order[0][0] > order[1][0]) else None


def points(rows, model):
    result = {}
    for arm in NAMES:
        by_group = defaultdict(list)
        for case in rows:
            if case["partition"] == "fit" and case["kind"] in {"human", "splice"} and (top := unique_top(case, arm, model)):
                by_group[case["group"]].append(top[0])
        require(len(by_group) >= FALSE_BUDGET + 1, "Too few complete fitting groups")
        result[arm] = sorted((max(values) for values in by_group.values()), reverse=True)[FALSE_BUDGET]
    return result


def useful(overlap):
    shared, union, truth = overlap
    return truth > 0 and 10 * shared >= 3 * union and 2 * shared >= truth


def outcome(case, arm, point, model):
    if arm == "current":
        return {"assessed": True, "flag": bool(case["current"]),
                "useful": any(useful(overlap) for overlap in case["current"])}
    picked = unique_top(case, arm, model)
    flag = bool(picked and picked[0] > point)
    return {"assessed": bool(case["windows"]), "flag": flag,
            "useful": flag and useful(case["windows"][picked[1]]["overlap"])}


def aggregate(rows, model, points):
    result = {}
    for partition in ("fit", "held"):
        cases = [row for row in rows if row["partition"] == partition]
        all_outcomes = {arm: [outcome(case, arm, points.get(arm), model) for case in cases] for arm in ("current", *NAMES)}
        baseline = all_outcomes["current"]
        for arm, values in all_outcomes.items():
            human_groups = {c["group"] for c, x in zip(cases, values, strict=True)
                            if c["kind"] in {"human", "splice"} and x["flag"]}
            base_groups = {c["group"] for c, x in zip(cases, baseline, strict=True)
                           if c["kind"] in {"human", "splice"} and x["flag"]}
            gains = [(c["group"], c["kind"]) for c, x, y in zip(cases, values, baseline, strict=True)
                     if c["kind"] in {"qwen", "smol2"} and x["useful"] and not y["useful"]]
            metrics = {"ordinary_groups_flagged": len(human_groups), "added_ordinary_groups": len(human_groups - base_groups),
                       "added_useful_groups": len({group for group, _ in gains}),
                       "added_useful_families": sorted({family for _, family in gains})}
            for kind in ("human", "splice", "qwen", "smol2"):
                subset = [(case, x) for case, x in zip(cases, values, strict=True) if case["kind"] == kind]
                metrics[kind] = {"files": len(subset), "flags": sum(x["flag"] for _, x in subset),
                                 "useful": sum(x["useful"] for _, x in subset),
                                 "unassessed": sum(not x["assessed"] for _, x in subset),
                                 "eligible_truth": sum(any(useful(row["overlap"]) for row in case["windows"]) for case, _ in subset),
                                 "wrong_only": sum(x["flag"] and not x["useful"] for _, x in subset) if kind in {"qwen", "smol2"} else 0}
            result[f"{partition}/{arm}"] = metrics
    return result


def overlap(text, truth, spans):
    chars = span_positions(text, spans)
    return [len(truth & chars), len(truth | chars), len(truth)]


def publish(parent, run, destination):
    source, inputs = checked(parent, run)
    older = read(parent / "primary.json")["responses"]
    primary = read(run / "primary.json")
    independent = read(run / "independent.json")
    for packet, name in ((primary, "primary_missing"), (independent, "independent")):
        require(packet["binding"] == digest(run / "inputs.json") and packet["responses"].keys() == inputs[name].keys(),
                "Saved responses differ from exact sealed input set")
    all_primary = older | primary["responses"]
    rows = []
    for case in source:
        text = case["text"]
        truth = span_positions(text, [case["truth"]]) if case["truth"] else set()

        native = [overlap(text, truth, window["spans"])
                  for window in case["windows"] if all_primary[window["primary"]]["score"] >= .5][:6]
        windows = [{"focus": case["windows"][i]["focus"], "overlap": overlap(text, truth, [case["windows"][i]["focus"]]),
                    "features": measure(case["windows"][i], all_primary, independent["responses"])} for i in case["eligible"]]
        rows.append({field: case[field] for field in ("partition", "kind", "text_sha256", "source_sha256", "truth", "donor")}
                    | {"group": text_hash(case["group"])[:16], "current": native, "windows": windows})
    model = fit(rows)
    cutoffs = points(rows, model)
    results = aggregate(rows, model, cutoffs)
    selected = results["held/combined"]
    advance = (selected["added_useful_groups"] >= 2 and set(selected["added_useful_families"]) == {"qwen", "smol2"}
               and selected["added_ordinary_groups"] <= 1
               and sum(selected[family]["eligible_truth"] for family in ("qwen", "smol2")) >= 12)
    receipt = {"schema": SCHEMA, "plan": PLAN, "freeze_sha256": digest(run / "freeze.json"),
               "private_commitments": {name: digest(run / name) for name in ("cases.json", "inputs.json", "primary.json", "independent.json")},
               "forwards": {"primary": primary["forwards"], "independent": independent["forwards"]},
               "counts": {"primary_cached": len({w["primary"] for c in source for w in c["windows"]} & older.keys()),
                          "primary_new": len(primary["responses"]), "independent_new": len(independent["responses"])},
               "model": model, "points": cutoffs, "outcomes": results, "advanced": advance, "cases": rows}
    write(destination, receipt)
    print({key: value for key, value in results.items() if key.startswith("held/")}, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "primary", "independent", "publish"))
    parser.add_argument("parent", type=Path)
    parser.add_argument("run", type=Path)
    parser.add_argument("--primary", type=Path)
    parser.add_argument("--independent", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.phase == "freeze":
        freeze(args.parent, args.run, args.primary, args.independent)
    elif args.phase in {"primary", "independent"}:
        second = args.phase == "independent"
        _, inputs = checked(args.parent, args.run)
        checked_assets(args.independent if second else args.primary, SECONDARY_ASSETS if second else BASELINE_ASSETS)
        observe(args.independent if second else args.primary, inputs["independent" if second else "primary_missing"],
                args.run / f"{args.phase}.json", digest(args.run / "inputs.json"), secondary=second)
    else:
        publish(args.parent, args.run, args.output)


if __name__ == "__main__":
    main()
