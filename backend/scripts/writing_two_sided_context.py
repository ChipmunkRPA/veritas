"""Compare native sentence contexts in a frozen, local writing evaluation."""
from __future__ import annotations

import argparse
import math
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from time import monotonic

from app.detectors.text.extract import from_string
from app.detectors.text.segment import segment
from scripts.writing_focus_v3 import BASELINE_ASSETS, digest, text_sha

SCHEMA = "chipmunk.writing-two-sided-context/1"
BASELINE_POINT = .5
MIN_FOCUS_WORDS = 12
LIMIT = 6
BATCH_SIZE = 8
ARMS = ("current_window", "current_focus", "fitted_window", "two_sided")
CALIBRATION_FEATURES = ("full_log_odds", "weaker_side_log_odds", "weaker_focus_contrast")
RULES = {
    "split": "Separate consumed fit and calibration source groups. Freeze their data and prospective holdout before scoring.",
    "current_window": "Current raw three-sentence response, including edge windows. Keep the first six qualifying foci and show full contexts.",
    "current_focus": "Same current raw selected foci, displayed as their exact sentences.",
    "fitted_window": "Matched fit on the unchanged full-window response with the same eligible interior foci as the new arm.",
    "two_sided": "Add the weaker two-sentence response and the weaker focus contrast against the corresponding individual flank.",
    "fit": "Balanced by source and class, standardized L2 logistic regression with C=.5. One strict point above every human calibration group maximum.",
    "geometry": "First six source-ordered qualifiers. A useful range overlaps the known rewrite with at least .30 non-whitespace IoU.",
    "comparison": "One fixed point per fit, no fresh recalibration. Same classifier and exact native token coverage throughout.",
    "advance": "At least two extra useful edits over current, no extra human source groups flagged, and at least one useful-location gain over matched fitted window.",
}
ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Focus:
    position: tuple[int, int]
    full: tuple[tuple[int, int], ...]
    pairs: tuple[tuple[tuple[int, int], ...], ...] | None


def geometry(text: str) -> list[Focus]:
    sentences = segment(from_string(text))
    boxes = [(s.start, s.end) for s in sentences]
    output = []
    for i, sentence in enumerate(sentences):
        full = tuple(boxes[max(0, i - 1):i + 2])
        left, right = i - 1, i + 1
        pairs = None
        if (left >= 0 and right < len(sentences) and sentence.n_words >= MIN_FOCUS_WORDS
                and sentences[left].paragraph == sentence.paragraph == sentences[right].paragraph):
            pairs = (tuple(boxes[left:i + 1]), tuple(boxes[i:right + 1]), (boxes[left],), (boxes[right],))
        output.append(Focus(boxes[i], full, pairs))
    return output


def native_input(text: str, ranges: tuple[tuple[int, int], ...]) -> str:
    return " ".join(text[a:b] for a, b in ranges)


def log_odds(response: float) -> float:
    if not math.isfinite(response) or not 0 <= response <= 1:
        raise ValueError("Invalid native response")
    safe = min(max(response, 1e-6), 1 - 1e-6)
    return math.log(safe / (1 - safe))


def features(full: float, left: float, right: float, left_only: float, right_only: float) -> list[float]:
    p, l, r, ll, rr = map(log_odds, (full, left, right, left_only, right_only))
    return [p, min(l, r), min(l - ll, r - rr)]


def evaluate_regions(text: str, ranges, ai_ranges, human_ranges) -> dict[str, bool]:
    def chars(edges):
        return {index for a, b in edges for index in range(a, b) if not text[index].isspace()}

    covered, ai, human = map(chars, (ranges, ai_ranges, human_ranges))
    overlap = len(covered & ai)
    return {
        "flagged": bool(covered),
        "useful": bool(ai and 10 * overlap >= 3 * len(covered | ai)),
        "wrong_location": bool(ai and covered and not overlap),
        "ordinary_edit_overlap": bool(covered & human),
    }


def open_classifier(snapshot: Path):
    import torch

    from app.detectors.text.classifier import FourClassTextClassifier

    for file, expected in BASELINE_ASSETS.items():
        if digest(snapshot / file) != expected:
            raise ValueError(f"Local classifier asset differs from the pinned digest: {file}")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    return FourClassTextClassifier.from_snapshot(snapshot, "cpu")


def native_responses(classifier, texts, checkpoint):
    import torch

    table = {}
    with torch.inference_mode():
        for start in range(0, len(texts), BATCH_SIZE):
            checkpoint()
            batch = texts[start:start+BATCH_SIZE]
            enc = classifier.tok(batch, padding=True, truncation=False, return_tensors="pt")
            mask = enc["attention_mask"]
            ids = enc["input_ids"]
            if ids.shape[1] > 512:
                raise ValueError("A native context exceeded its complete-token budget")
            actual = ids[mask != 0].tolist()
            unpadded = [ids[i][mask[i] != 0].tolist() for i in range(len(batch))]
            if len(actual) != sum(map(len, unpadded)) or any(
                    classifier.tok(text, truncation=False, verbose=False)["input_ids"] != toks
                    for text, toks in zip(batch, unpadded, strict=True)):
                raise ValueError("Native tokens differ from the complete source inputs")
            logits = classifier.model(input_ids=ids, attention_mask=mask)["logits"].float()
            if logits.shape != (len(batch), 4) or not torch.isfinite(logits).all():
                raise ValueError("Invalid native classifier logits")
            values = (1 - logits.softmax(dim=-1)[:, 0]).tolist()
            for text, value, toks in zip(batch, values, unpadded, strict=True):
                table[text] = (value, text_sha(",".join(map(str, toks))))
    return table


def score_cases(classifier, cases, checkpoint):
    geometries = {case["case"]: geometry(case["text"]) for case in cases if case["status"] == "constructed"}
    contexts = {}
    for case in cases:
        for focus in geometries.get(case["case"], ()):
            for view in (focus.full, *(focus.pairs or ())):
                contexts[native_input(case["text"], view)] = None
    native = native_responses(classifier, list(contexts), checkpoint)
    print(f"{len(cases)} planned cases, {len(contexts)} unique complete classifier inputs", flush=True)
    for case in cases:
        rows = []
        for focus in geometries.get(case["case"], ()):
            views = (focus.full, *(focus.pairs or ()))
            values = [native[native_input(case["text"], view)] for view in views]
            rows.append({"focus": list(focus.position), "full": [list(pair) for pair in focus.full],
                         "views": views, "context_input_sha256": [text_sha(native_input(case["text"], v)) for v in views],
                         "token_sha256": [s for _, s in values], "responses": [p for p, _ in values]})
        yield rows


def fitted_value(vector, coefficients):
    return coefficients["intercept"] + sum(
        weight * ((value - mean) / scale)
        for value, mean, scale, weight in zip(vector, coefficients["mean"], coefficients["scale"], coefficients["coefficient"],
                                              strict=True)
    )


def fit_selectors(fit, cal):
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    collected = [(case["group"], int(row["focus"] in case["ai_spans"]), features(*row["responses"]))
                 for case, rows in fit for row in rows if len(row["responses"]) == 5]
    populations = Counter((group, label) for group, label, _ in collected)
    group_count = {label: sum(y == label for _, y in populations) for label in (0, 1)}
    if group_count[1] < 3 or group_count[0] < 4:
        raise ValueError("Fit requires at least three independent positive and four negative groups")
    labels = [y for _, y, _ in collected]
    weights = np.array([len(collected) / (2 * group_count[y] * populations[group, y]) for group, y, _ in collected])
    choices = {}
    for arm, width in (("fitted_window", 1), ("two_sided", 3)):
        design = np.asarray([values[:width] for _, _, values in collected])
        scaler = StandardScaler().fit(design, sample_weight=weights)
        fit_model = LogisticRegression(C=.5, solver="lbfgs", max_iter=200, tol=1e-9)
        fit_model.fit(scaler.transform(design), labels, sample_weight=weights)
        info = {"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(), "coefficient": fit_model.coef_[0].tolist(),
                "intercept": float(fit_model.intercept_[0])}
        groups = defaultdict(list)
        for case, rows in cal:
            if case["status"] != "constructed" or case["ai_spans"]:
                continue
            for row in rows:
                if len(row["responses"]) == 5:
                    groups[case["group"]].append(fitted_value(features(*row["responses"])[:width], info))
        if len(groups) < 4:
            raise ValueError("Calibration requires at least four fully assessed genuine groups")
        choices[arm] = {**info, "point": max(max(values) for values in groups.values())}
    return {"arms": choices, "fit_counts": dict(Counter(labels)), "fit_source_groups": group_count, "calibration_groups": len(groups),
            "calibration_unassessed_groups": len({case["group"] for case, _ in cal}) - len(groups)}


def selected_ranges(rows, fitted):
    base = [row for row in rows if row["responses"][0] >= BASELINE_POINT][:LIMIT]
    ranges = {"current_window": [pair for row in base for pair in row["full"]], "current_focus": [row["focus"] for row in base]}
    for arm, width in (("fitted_window", 1), ("two_sided", 3)):
        params = fitted["arms"][arm]
        ranges[arm] = [row["focus"] for row in rows if len(row["responses"]) == 5
                       and fitted_value(features(*row["responses"])[:width], params) > params["point"]][:LIMIT]
    return ranges


def aggregate(scored, fitted):
    results = defaultdict(lambda: {arm: Counter() for arm in ARMS})
    gain = defaultdict(lambda: {arm: Counter() for arm in ARMS})
    human_groups = {arm: set() for arm in ARMS}
    positive_groups = {arm: set() for arm in ARMS}
    coverage = Counter()
    for case, rows in scored:
        group, source = case["group"], case["arm"]
        if case["status"] != "constructed":
            for arm in ARMS:
                results[source][arm]["unavailable_generation"] += 1
            continue
        eligible = sum(len(row["responses"]) == 5 for row in rows)
        coverage.update(verified_decoded_bodies=1, current_contexts=len(rows), eligible_foci=eligible,
                        native_context_inputs=len(rows) + 4 * eligible)
        coverage["exact_ai_targets_eligible"] += sum(row["focus"] in case["ai_spans"] and len(row["responses"]) == 5 for row in rows)
        marks = selected_ranges(rows, fitted)
        bits = {}
        for arm, regions in marks.items():
            result = evaluate_regions(case["text"], regions, case["ai_spans"], case["ordinary_spans"])
            assessed = bool(rows) if arm.startswith("current_") else bool(eligible)
            results[source][arm].update(assessed=int(assessed), unassessed=int(not assessed),
                                        missed_if_assessed=int(bool(case["ai_spans"]) and assessed and not result["useful"]),
                                        **{key: int(value) for key, value in result.items()})
            if result["flagged"] and not case["ai_spans"]:
                human_groups[arm].add(group)
            if result["useful"]:
                positive_groups[arm].add(group)
            bits[arm] = result["useful"] if case["ai_spans"] else result["flagged"]
            gain[source][arm][f"{int(bits['current_window'])}->{int(bits[arm])}" if assessed else "unassessed"] += 1
    return {"counts": results, "paired_vs_current": gain, "coverage": coverage,
            "human_source_flags": {arm: len(groups) for arm, groups in human_groups.items()},
            "useful_source_groups": {arm: len(groups) for arm, groups in positive_groups.items()}}


def read_cases(path):
    import json

    from app.detectors.text.extract import extract

    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    if len({r["case"] for r in rows}) != len(rows):
        raise ValueError("Duplicate evaluation case ID")
    for row in rows:
        if row["status"] != "constructed":
            continue
        body = row["text"]
        if row["text_sha256"] != text_sha(body):
            raise ValueError("Decoded source body commitment changed")
        file = row.get("fixture")
        if file:
            fixture = Path(file)
            decoded = extract(fixture, fixture.suffix)
            if decoded.text != body or decoded.paragraphs != from_string(body).paragraphs:
                raise ValueError("Actual decoded source body differs from the frozen text or paragraphs")
        known = row["ai_spans"] + row["ordinary_spans"]
        actual = {tuple(x.position) for x in geometry(body)}
        if any(tuple(x) not in actual for x in known):
            raise ValueError("An edit no longer maps to an exact source sentence")
    return rows


def write(path, data):
    import json

    with path.open("x") as stream:
        json.dump(data, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def main():
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "run"))
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--snapshot", type=Path)
    args = parser.parse_args()
    os.umask(0o077)
    run = args.run_dir
    files = {role: run / f"private-{role}.jsonl" for role in ("fit", "calibration", "holdout")}
    policy = {"schema": SCHEMA, "rules": RULES, "features": CALIBRATION_FEATURES, "native_baseline_point": BASELINE_POINT,
              "source_sha256": digest(run / "source.json"), "scorer_sha256": digest(Path(__file__)), "classifier_assets": BASELINE_ASSETS,
              "code_sha256": {key: digest(ROOT / key) for key in ("app/detectors/text/extract.py", "app/detectors/text/segment.py",
                                                                "app/detectors/text/classifier.py", "app/detectors/text/pipeline.py")},
              "case_sha256": {role: digest(path) for role, path in files.items()}}
    lock = run / "frozen.json"
    if args.stage == "freeze":
        write(lock, policy)
        print(digest(lock))
        return
    if args.snapshot is None or json.loads(lock.read_text()) != json.loads(json.dumps(policy)):
        parser.error("A verified freeze and local snapshot are required")
    import psutil

    if psutil.virtual_memory().available < 3.75 * 2**30:
        raise ValueError("Initial classifier memory reserve is unavailable")
    start = monotonic()
    lowest_ram = psutil.virtual_memory().available

    def checkpoint():
        nonlocal lowest_ram
        lowest_ram = min(lowest_ram, psutil.virtual_memory().available)
        if lowest_ram < 1.25 * 2**30 or monotonic() - start > 180:
            raise ValueError("Local CPU run resource limit reached")

    cases = {role: read_cases(path) for role, path in files.items()}
    groups = {role: {case["group"] for case in rows} for role, rows in cases.items()}
    if any(groups[a] & groups[b] for a, b in (("fit", "calibration"), ("fit", "holdout"), ("calibration", "holdout"))):
        raise ValueError("Frozen source groups overlap")
    classifier = open_classifier(args.snapshot)
    forwards = [0]
    hook = classifier.model.register_forward_pre_hook(lambda *_: forwards.__setitem__(0, forwards[0] + 1))
    try:
        fit, cal = (list(zip(cases[role], score_cases(classifier, cases[role], checkpoint), strict=True))
                    for role in ("fit", "calibration"))
        fitted = fit_selectors(fit, cal)
        write(run / "fitted.json", fitted)
        fit_forwards = forwards[0]
        hold = list(zip(cases["holdout"], score_cases(classifier, cases["holdout"], checkpoint), strict=True))
    finally:
        hook.remove()
    rows = {case["case"]: values for case, values in hold}
    write(run / "private-native.json", rows)
    output = aggregate(hold, fitted)
    source = json.loads((run / "source.json").read_text())
    output.update(schema=SCHEMA, method=RULES, fitted=fitted,
                  source_groups=len(groups["holdout"]), known_prior_texts=sum(x["text_entries"] for x in source["history"]),
                  model_forward_calls={"fit_and_calibration": fit_forwards, "holdout": forwards[0] - fit_forwards},
                  elapsed_seconds=round(monotonic() - start, 3), minimum_available_bytes=lowest_ram,
                  commitments={"freeze": digest(lock), "native": digest(run / "private-native.json"), "fit": digest(run / "fitted.json")})
    write(run / "aggregate.json", output)


if __name__ == "__main__":
    main()
