"""Fit a small mixed-writing interval ranker and replay a source-disjoint report test."""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import math
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

from scripts import writing_revision_trajectory_transfer as source

ROOT = Path(__file__).resolve().parents[2]
PANELS = ROOT / "docs/evaluation-panels"
PRIOR = PANELS / "writing-revision-trajectory-2026-11-13.json"
WORKER = PANELS / "writing-revision-worker-complete-2026-11-13.json"
SCHEMA = "chipmunk.writing-interval-transfer/1"
FAMILIES = {"gemini-2.5-flash": source.SHARDS["gemini-2.5-flash"],
            "gpt-5.4-nano": "912d90f2f22b2ea97bdf69930f6891584112381f5b811aedcac390bbdf1da064"}
EXCLUDED_SHARD = {"qwen3-8b": source.SHARDS["qwen3-8b"]}
WANTED = 20
MIN_CHARS = 60
MAX_SENTENCES = 12
METHODS = ("primary_interval", "combined_interval")
FEATURE_NAMES = (
    "mean primary context odds", "mean primary odds above the file median",
    "primary odds above the strongest available two-sentence flank",
    "mean isolated independent odds", "mean isolated odds above the file median",
    "isolated odds above the strongest available two-sentence flank",
    "log fraction of the document-wide sentence count",
)
FEATURES = {"primary_interval": (0, 1, 2, 6), "combined_interval": (0, 1, 2, 3, 4, 5, 6)}
PROTOCOL = {
    "fit": "Already consumed text-changed OpAI report variants and their originals. Train one positive-versus-negative interval ranker per arm with equal source and within-source class weights.",
    "source": "Fresh same-domain works are selected in a salted ID order from the first 48 eligible-by-ID shared Gemini and Nano works. Exclude all Qwen test works and all earlier roster IDs before source geometry checks.",
    "interval": "All same-paragraph runs of one to twelve native sentences, at least 60 non-whitespace characters, no more than 80% of the document characters, and at least one available same-paragraph flank. Choose one highest response with a deterministic position tie-break.",
    "supervision": "Positive intervals have at least 30% IoU and 50% recall over all annotated non-whitespace edit characters.",
    "fit_parameters": "Source and class-weighted logistic ranker, standardized features, 120 fixed gradient steps of .35, ridge .10 on each non-intercept coefficient.",
    "point": "The second-highest score on the 16 separate genuine report files. Select only a higher value.",
    "comparison": "Unchanged first six native primary windows at .50 and, on previously scored works only, actual published worker packets.",
    "advance": "Three additional strict locations on distinct new works across both generator families, at most two genuine works flagged by the candidate, and at least 90% candidate coverage of changed versions. Review-only research response with zero authorship confidence weight.",
}


def eligible(row):
    ranges = row["ranges"]
    total = sum(span["characters"] for span in ranges)
    for start, first in enumerate(ranges):
        inside = 0
        for end in range(start + 1, min(len(ranges), start + MAX_SENTENCES) + 1):
            last = ranges[end - 1]
            if last["paragraph"] != first["paragraph"]:
                break
            inside += last["characters"]
            if inside * 5 > total * 4:
                break
            if inside < MIN_CHARS:
                continue
            left = [i for i in range(max(0, start - 2), start) if ranges[i]["paragraph"] == first["paragraph"]]
            right = [i for i in range(end, min(len(ranges), end + 2)) if ranges[i]["paragraph"] == first["paragraph"]]
            if left or right:
                yield start, end, inside, left, right


def candidates(row):
    streams = [[source.odds(value) for _, value in row[key]] for key in ("primary", "independent")]
    medians = [median(values) for values in streams]
    n = len(row["ranges"])
    for start, end, count, left, right in eligible(row):
        features = []
        for values, center in zip(streams, medians, strict=True):
            mean = math.fsum(values[start:end]) / (end - start)
            flanks = [math.fsum(values[i] for i in indices) / len(indices) for indices in (left, right) if indices]
            features.extend((mean, mean - center, mean - max(flanks)))
        features.append(math.log((end - start) / n))
        yield (start, end, count), features


def sigmoid(value):
    return 1 / (1 + math.exp(-max(-40., min(40., value))))


def fit_parameters(rows, arm):
    by_source = defaultdict(lambda: defaultdict(list))
    columns = FEATURES[arm]
    for row in rows:
        truth = row["truth_characters"]
        for (start, end, count), features in candidates(row):
            overlap = sum(span["ai_characters"] for span in row["ranges"][start:end])
            useful = overlap > 0 and 2 * overlap >= truth and 10 * overlap >= 3 * (truth + count - overlap)
            by_source[row["group"]][useful].append([features[i] for i in columns])
    samples = []
    for arms in by_source.values():
        for label, samples_in_class in sorted(arms.items()):
            weight = 1 / (len(by_source) * len(arms) * len(samples_in_class))
            samples.extend((values, label, weight) for values in samples_in_class)
    source.require(any(label for _, label, _ in samples), "No useful development intervals")
    means = [math.fsum(v[i] * w for v, _, w in samples) for i in range(len(columns))]
    scales = [max(.1, math.sqrt(math.fsum(w * (v[i] - means[i]) ** 2 for v, _, w in samples))) for i in range(len(columns))]
    normalized = [([1., *((x - mean) / scale for x, mean, scale in zip(v, means, scales, strict=True))], label, w)
                  for v, label, w in samples]
    coefs = [0.] * (len(columns) + 1)
    for _ in range(120):
        errors = [(v, w * (sigmoid(math.fsum(x * c for x, c in zip(v, coefs, strict=True))) - label))
                  for v, label, w in normalized]
        for i in range(len(coefs)):
            coefs[i] -= .35 * (math.fsum(v[i] * error for v, error in errors) + (.1 * coefs[i] if i else 0))
    return {"means": means, "scales": scales, "coefs": coefs, "fitted_intervals": len(samples)}


def rank(row, arm, parameters):
    means, scales, coefs = (parameters[key] for key in ("means", "scales", "coefs"))
    choices = []
    for (start, end, _), features in candidates(row):
        features = [features[i] for i in FEATURES[arm]]
        response = coefs[0] + math.fsum(c * (v - m) / s for c, v, m, s in zip(coefs[1:], features, means, scales, strict=True))
        choices.append((response, start, end))
    return min(choices, key=lambda x: (-x[0], x[1], x[2]), default=None)


def split_changed(rows):
    originals = {row["group"]: row["text_sha256"] for row in rows if row["family"] == "original"}
    return [row for row in rows if row["family"] == "original" or row["text_sha256"] != originals[row["group"]]]


def freeze():
    retained = source.read(PRIOR)
    fit = [r for r in retained["rows"] if r["role"] == "fit"]
    dev = split_changed([r for r in retained["rows"] if r["role"] == "held"])
    source.require(len(fit) == 16 and len(dev) == 125 and all(r["family"] == "original" for r in fit), "Wrong development roles")
    parameters = {arm: fit_parameters(dev, arm) for arm in METHODS}
    maxima = {arm: {r["group"]: rank(r, arm, params)[0] for r in fit} for arm, params in parameters.items()}
    return {"schema": SCHEMA, "protocol": PROTOCOL, "features": list(FEATURE_NAMES), "runner_sha256": source.digest(Path(__file__)),
            "prior_sha256": source.digest(PRIOR), "worker_sha256": source.digest(WORKER),
            "parameters": parameters, "genuine_maxima": maxima,
            "points": {arm: sorted(values.values())[-2] for arm, values in maxima.items()},
            "publisher_revision": source.REVISION, "metadata": source.METADATA, "shards": FAMILIES | EXCLUDED_SHARD,
            "native_source_sha256": source.digest(Path(source.__file__))}


def read_shard(cache, family, expected):
    path = cache / f"reports_{family}.csv"
    source.require(source.digest(path) == expected, "Pinned publisher file differs")
    csv.field_size_limit(1024 * 1024)
    found = defaultdict(dict)
    with path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream, strict=True):
            source.require((row["generator"], row["domain"], row["split"], row["subset"]) ==
                           (family, "reports", "test", "main"), "Incorrect publisher role")
            if row["version"] not in {"v0", *source.STAGES}:
                continue
            source.require(re.fullmatch(r"GAO_[A-Z]+-[0-9A-Z-]+", row["id"])
                           and row["version"] not in found[row["id"]], "Unknown or duplicated publisher source")
            found[row["id"]][row["version"]] = row
    source.require(all(set(rows) == {"v0", *source.STAGES} for rows in found.values()), "Incomplete source trajectory")
    return found


def fingerprints(text):
    words = re.findall(r"\w+", text.casefold())
    return {tuple(words[i:i + 12]) for i in range(len(words) - 11)}


def select_sources(cache):
    source.require(all(source.digest(cache / name) == pin for name, pin in source.METADATA.items()), "Publisher metadata differs")
    source.require("released under Apache-2.0" in (cache / "README.md").read_text(), "Unexpected dataset terms")
    metadata = {row["name"]: row.get("sha256") for row in source.read(cache / "croissant.json")["distribution"]}
    for family, digest in (FAMILIES | EXCLUDED_SHARD).items():
        source.require(metadata[f"default/test/reports_{family}.csv"] == digest, "Publisher shard metadata differs")
    families = {family: read_shard(cache, family, expected) for family, expected in FAMILIES.items()}
    excluded = read_shard(cache, *next(iter(EXCLUDED_SHARD.items())))
    first, second = families.values()
    prior = source.read(PRIOR)
    prior_groups = {r["group"] for r in prior["rows"]}
    history = set()
    for group in first.keys() | excluded.keys():
        if source.sha(group) in prior_groups or group in excluded:
            records = (first.get(group, {}), second.get(group, {}), excluded.get(group, {}))
            history.update(*(fingerprints(r["text"]) for record in records for r in record.values()))
    new = (group for group in first.keys() & second.keys() if group not in excluded and source.sha(group) not in prior_groups)
    order = sorted(new, key=lambda value: source.sha(f"chipmunk-report-interval-held-2026-11-13/{value}"))[:48]
    dispositions = Counter()
    roster = []
    for group in order:
        if dispositions["admitted"] == WANTED:
            break
        dispositions["examined"] += 1
        original, copy = first[group]["v0"], second[group]["v0"]
        if original["text"] != copy["text"]:
            dispositions["different_publisher_originals"] += 1
            continue
        rows = [("original", "v0", original)]
        rows.extend((family, version, family_rows[group][version]) for family, family_rows in families.items()
                    for version in source.STAGES)
        try:
            geometries = [source.source_geometry(row) for _, _, row in rows]
        except ValueError:
            dispositions["publisher_annotations_disagree"] += 1
            continue
        if any(item is None for item in geometries):
            dispositions["unsupported_geometry"] += 1
            continue
        if geometries[0]["truth_characters"] or not all(g["truth_characters"] for g in geometries[1:]):
            dispositions["incomplete_truth"] += 1
            continue
        identity = set().union(*(fingerprints(g["text"]) for g in geometries))
        if history & identity:
            dispositions["prior_or_other_selected_phrase"] += 1
            continue
        history.update(identity)
        for (family, version, row), geometry in zip(rows, geometries, strict=True):
            roster.append({"group": source.sha(group), "source": source.normalize_title_source(group),
                           "family": family, "version": version, "record_id": row["record_id"], **geometry})
        dispositions["admitted"] += 1
    source.require(dispositions["admitted"] >= 12, "Insufficient complete fresh source groups")
    return roster, dict(dispositions)


def native_score(run, snapshot, arm):
    import torch

    from scripts.score_text_upgrade import sequence_predictor
    from scripts.writing_body_localization_transfer import SECONDARY_ASSETS
    from scripts.writing_focus_v3 import BASELINE_ASSETS

    source.require(os.getenv("HF_HUB_OFFLINE") == os.getenv("TRANSFORMERS_OFFLINE") == "1", "Use offline model settings")
    lock = source.read(run / "freeze.json")
    source.require(lock["runner_sha256"] == source.digest(Path(__file__))
                   and lock["prior_sha256"] == source.digest(PRIOR), "Frozen method differs")
    coverage = source.read(run / "source-receipt.json")
    source.require(coverage["freeze_sha256"] == source.digest(run / "freeze.json")
                   and coverage["roster_sha256"] == source.digest(run / "roster.json"), "Fresh roster differs")
    assets, field, classifier = (BASELINE_ASSETS, "context", "tropa") if arm == "primary" else (SECONDARY_ASSETS, "isolated", "modernbert")
    source.require(all(source.digest(snapshot / name) == expected for name, expected in assets.items()), "Model assets differ")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    roster = source.read(run / "roster.json")
    texts = sorted({text for row in roster for text in row[field]}, key=source.sha)
    predict = sequence_predictor(classifier, snapshot, "cpu", 4, max_len=512)
    responses = {}
    for start in range(0, len(texts), 80):
        batch = texts[start:start + 80]
        values = predict(batch)
        source.require(all(math.isfinite(v) and 0 <= v <= 1 for v in values), "Invalid model response")
        responses.update(zip(batch, values, strict=True))
        print(arm, len(responses), len(texts), flush=True)
    source.seal(run / f"{arm}.json", {
        "source_receipt_sha256": source.digest(run / "source-receipt.json"), "model_assets": assets,
        "runtime": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "pysbd")},
        "inputs": len(texts), "rows": {r["record_id"]: [[source.sha(t), responses[t]] for t in r[field]] for r in roster},
    })


def proposals(row, lock):
    output = {}
    for method, coefficients in lock["parameters"].items():
        choice = rank(row, method, coefficients)
        output[method] = ({"assessed": choice is not None, "response": choice[0] if choice else None,
                           "interval": list(choice[1:]) if choice and choice[0] > lock["points"][method] else None})
    return output


def report_rows(rows, lock, worker=None):
    originals = {row["group"]: row["text_sha256"] for row in rows if row["family"] == "original"}
    stats = defaultdict(lambda: defaultdict(Counter))
    additions = defaultdict(set)
    families = defaultdict(set)
    decisions = []
    for row in rows:
        n = len(row["ranges"])
        primary = [(max(0, i - 1), min(n, i + 2)) for i, (_, p) in enumerate(row["primary"]) if p >= .5][:6]
        selections = {"primary": primary}
        novel = proposals(row, lock)
        for method, record in novel.items():
            selections[method] = [record["interval"]] if record["interval"] else []
            selections[f"primary_plus_{method}"] = primary + selections[method]
        label = "original" if row["family"] == "original" else (
            "identical" if row["text_sha256"] == originals[row["group"]] else "changed")
        outcomes = {key: source.outcome(row, ranges) for key, ranges in selections.items()}
        if worker is not None:
            from scripts.review_writing_revision_worker_transfer import metrics, sentence_indices
            plates = [[sentence_indices(row, *span) for span in packet["spans"]]
                      for packet in worker[row["record_id"]]["packets"]]
            measured = metrics(row, plates)
            outcomes["full_worker"] = dict(zip(
                ("flag", "useful_document_location", "useful_individual_span", "gold_spans_half_covered",
                 "highlighted_ai_characters", "highlighted_ordinary_characters"), measured.values(), strict=True))
        for method, outcome in outcomes.items():
            base = outcomes["full_worker" if worker is not None else "primary"]
            if label == "changed" and method in METHODS and outcome["useful_document_location"] and not base["useful_document_location"]:
                additions[method].add(row["group"])
                families[method].add(row["family"])
            for stratum in (label, f"{row['family']}/{row['version']}") if label == "changed" else (label,):
                stats[stratum][method].update(files=1, assessed=novel.get(method, {"assessed": True})["assessed"],
                    **{key: value for key, value in outcome.items() if key != "ranges"})
        decisions.append({"record_id": row["record_id"], "stratum": label, "proposals": novel})
    verdict = {}
    for method in METHODS:
        genuine = stats["original"][method]
        changed = stats["changed"][method]
        verdict[method] = {"new_source_groups": len(additions[method]), "new_families": sorted(families[method]),
                           "genuine_flags": genuine["flag"],
                           "met": (len(additions[method]) >= 3 and families[method] == set(FAMILIES) and genuine["flag"] <= 2
                                   and changed["assessed"] >= .9 * changed["files"])}
    return {"counts": stats, "verdict": verdict, "decisions": decisions}


def reviewed_rows(run):
    roster = source.read(run / "roster.json")
    readout = {name: source.read(run / f"{name}.json") for name in ("primary", "independent")}
    source.require(all(packet["source_receipt_sha256"] == source.digest(run / "source-receipt.json")
                       for packet in readout.values()), "Score packet uses another source receipt")
    result = []
    for row in roster:
        slim = {key: value for key, value in row.items() if key not in ("text", "isolated", "context")}
        for arm, field in (("primary", "context"), ("independent", "isolated")):
            values = readout[arm]["rows"][row["record_id"]]
            source.require([text_hash for text_hash, _ in values] == [source.sha(text) for text in row[field]], "Source model input changed")
            slim[arm] = values
        result.append(slim)
    return result, {arm: {key: packet[key] for key in ("runtime", "inputs", "model_assets")} for arm, packet in readout.items()}


def evaluation(lock, rows, intake, checkpoints):
    dev = source.read(PRIOR)["rows"]
    previous = {r["record_id"]: r for r in source.read(WORKER)["rows"]}
    return {"schema": SCHEMA, "freeze": lock, "intake": intake, "scoring": checkpoints,
            "fresh": report_rows(rows, lock), "development": report_rows([r for r in dev if r["role"] == "held"], lock, previous),
            "rows": rows}


def review(public, cache=None):
    value = source.read(public)
    lock = freeze()
    source.require(value["schema"] == SCHEMA and value["freeze"] == lock, "Published fit or source commitments differ")
    rows = value["rows"]
    by_group = defaultdict(list)
    for row in rows:
        by_group[row["group"]].append((row["family"], row["version"]))
        source.require(len(row["primary"]) == len(row["independent"]) == len(row["ranges"])
                       and sum(span["ai_characters"] for span in row["ranges"]) == row["truth_characters"]
                       and sum(row["gold_characters"]) == row["truth_characters"], "Malformed native range or truth")
        for arm in ("primary", "independent"):
            source.require(all(re.fullmatch(r"[0-9a-f]{64}", h) and math.isfinite(v) and 0 <= v <= 1 for h, v in row[arm]),
                           "Malformed native response")
    expected_roles = {("original", "v0"), *((family, version) for family in FAMILIES for version in source.STAGES)}
    source.require(all(len(roles) == len(expected_roles) and set(roles) == expected_roles for roles in by_group.values())
                   and len(by_group) == value["intake"]["dispositions"]["admitted"], "Incomplete held source roster")
    source.require(value == evaluation(lock, rows, value["intake"], value["scoring"]), "Frozen outcomes differ from replay")
    if cache is not None:
        native, counts = select_sources(cache)
        source.require(counts == value["intake"]["dispositions"] and source.sha(source.serialize(native)) == value["intake"]["roster_sha256"],
                       "Exact publisher roster differs")
        for a, b in zip(native, rows, strict=True):
            source.require({k: v for k, v in a.items() if k not in {"text", "isolated", "context"}} ==
                           {k: v for k, v in b.items() if k not in {"primary", "independent"}}, "Publisher positions differ")
            for arm, field in (("primary", "context"), ("independent", "isolated")):
                source.require([h for h, _ in b[arm]] == [source.sha(text) for text in a[field]], "Exact model input differs")
    return value["fresh"]["verdict"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "intake", "primary", "independent", "report", "review"))
    parser.add_argument("path", type=Path)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.action == "freeze":
        source.seal(args.path / "freeze.json", freeze())
    elif args.action == "intake":
        frozen = source.read(args.path / "freeze.json")
        source.require(frozen == freeze(), "Freeze development before held intake")
        roster, counts = select_sources(args.cache)
        source.seal(args.path / "roster.json", roster)
        source.seal(args.path / "source-receipt.json", {"dispositions": counts,
                    "roster_sha256": source.digest(args.path / "roster.json"), "freeze_sha256": source.digest(args.path / "freeze.json")})
        print(counts)
    elif args.action in {"primary", "independent"}:
        native_score(args.path, args.snapshot, args.action)
    elif args.action == "report":
        locked = source.read(args.path / "freeze.json")
        source.require(locked == freeze(), "Fit no longer reproduces")
        rows, scoring = reviewed_rows(args.path)
        receipt = evaluation(locked, rows, source.read(args.path / "source-receipt.json"), scoring)
        source.seal(args.output, receipt)
        print(receipt["fresh"]["verdict"])
    else:
        print(review(args.path, args.cache))


if __name__ == "__main__":
    os.umask(0o077)
    main()
