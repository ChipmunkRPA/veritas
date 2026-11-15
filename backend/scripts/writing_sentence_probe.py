"""Train and test sentence-local heads on native offsets from matched rewrites."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

from scripts import writing_context_pool_transfer as pool
from scripts import writing_editorial_worker_audit as editorial
from scripts import writing_focus_inside_context as holdout
from scripts.writing_interval_transfer import fingerprints

SCHEMA = "chipmunk.writing-sentence-probe/1"
EDITORIAL = pool.PANEL_DIR / "writing-editorial-worker-audit-2026-11-14.json"
MODES = ("absolute", "paired")
REPORT_PINS = (
    "23e2b29270910b2a54d1ae2859e9e2b1e89cee1dd179f16d2ea2b39c7e60af25",
    "9e28d157d336a343106487c8acab242ee41829f49ceb748dcb5f9fd6a9f9fb33",
)
PLAN = {
    "fit": {"news": 96, "abstracts": 96},
    "calibration": {"news": 24, "abstracts": 24, "essays": 24},
    "held": {"news": 24, "abstracts": 24, "essays": 24},
}
METHOD = {
    "inputs": "Use one source-native five-sentence English paragraph per work. Only one interior sentence changes. "
              "Test against a genuine control. Check publisher character tags and exact normalized source positions.",
    "cohorts": "Use Gemini 2.5 Flash and GPT-5.4 Nano for development only. Use Qwen3-8B and full GPT-5.4 for the held "
               "roster. Exclude prior and admitted source ids and normalized 12-word overlap. Preserve shortfalls.",
    "encoder": "Freeze the serving encoder. Read a five-sentence excerpt once, validating exact non-whitespace token "
               "coverage. Average token vectors for each sentence, then concatenate the target mean with its difference "
               "from the average of the adjacent sentences.",
    "heads": "Fit an ordinary class-balanced ridge logistic head, and a separate no-intercept logistic head on the "
             "differences between matched revised and original focus vectors. Use C=.05 and .1 as the minimum feature "
             "scale. The earlier copy is used only during training.",
    "points": "Set two internal points from genuine calibration maxima only. The first permits at most ten percent "
              "calibration flags. The second matches the native context proxy's genuine flag count. Select one unique "
              "interior sentence strictly beyond the point.",
    "baseline": "Use the serving classifier on all five native neighboring contexts. Score at the existing .50 research "
                "point. Useful overlap is at least 30 percent intersection over union and coverage of half the edit. "
                "Compare sentence cards from the earlier actual paid worker on the 74 pinned journal controls and rewrites.",
}
SHA256 = re.compile(r"[a-f0-9]{64}")


def write(path, payload):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def exposure_paths():
    return sorted(p for p in pool.PANEL_DIR.glob("writing-*-2026-11-14.json")
                  if not p.name.startswith("writing-sentence-probe-"))


def source_tables(archive, test):
    development = {domain: [pool.shard(archive / domain, domain, family) for family in pool.GENERATORS]
                   for domain in pool.DOMAINS}
    assessment = {domain: [holdout.shard(test / f"{domain}_{family}.csv", domain, family, pin)
                           for family, pin in zip(holdout.FAMILIES, holdout.PINS[domain], strict=True)]
                  for domain in pool.DOMAINS}
    return development, assessment


def editorial_rows(upload_dir):
    previous = holdout.read(EDITORIAL)["rows"]
    files = {pool.file_hash(path): path for path in upload_dir.glob("*/*") if path.is_file()}
    for row in previous:
        path = files.get(row["text_sha256"])
        if path is None:
            raise ValueError("A committed editorial source file is missing")
        text = path.read_text(encoding="utf-8")
        if len(text) != row["length"] or pool.sha(text) != row["text_sha256"]:
            raise ValueError("Journal source changed")
        yield {"role": "editorial", "domain": "journal", "family": row["role"], "group": pool.sha(row["group"]),
               "truth": row["focus"] if row["role"] in {"qwen", "smol2"} else None, "spans": row["ranges"],
               "sha256": row["text_sha256"], "text": text, "spaces": row["spaces"]}


def freeze(archive, test, uploads):
    development, assessment = source_tables(archive, test)
    exposed = [path for path in exposure_paths()]
    prior_ids = {digest for path in exposed for digest in SHA256.findall(path.read_text())}
    preserved = set()
    for collection in (development, assessment):
        for tables in collection.values():
            for table in tables:
                for work, source in table.items():
                    if pool.sha(work) in prior_ids and "v0" in source:
                        preserved.update(fingerprints(source["v0"]["text"]))
    rows, counts = [], Counter()
    for role, domains in PLAN.items():
        families = holdout.FAMILIES if role == "held" else pool.GENERATORS
        collection = assessment if role == "held" else development
        for domain, cap in domains.items():
            for family, table in zip(families, collection[domain], strict=True):
                count = 0
                for work in sorted(table, key=lambda key: pool.sha(f"{SCHEMA}/{role}/{family}/{key}")):
                    if count == cap // 2:
                        break
                    group = pool.sha(work)
                    source = table[work]
                    if group in prior_ids or "v0" not in source:
                        continue
                    phrases = fingerprints(source["v0"]["text"])
                    if phrases & preserved:
                        continue
                    try:
                        position = count % 3 + 1
                        pair = holdout.excerpt_options(work, source).get(position)
                    except (KeyError, ValueError):
                        continue
                    if pair is None:
                        continue
                    for (text, spans, variant), label in zip(pair, ("original", family), strict=True):
                        if role == "calibration" and label != "original":
                            continue
                        rows.append({"role": role, "domain": domain, "family": label, "pair_family": family, "group": group,
                                     "sha256": pool.sha(text), "text": text, "spans": spans, "variant": variant,
                                     "truth": spans[position] if variant else None,
                                     "spaces": [i for i, c in enumerate(text) if c.isspace()]})
                    count += 1
                    prior_ids.add(group)
                    preserved.update(phrases)
                counts["/".join((role, domain, family))] = count
                print(role, domain, family, count, flush=True)
    rows.extend(editorial_rows(uploads))
    if sum(count for key, count in counts.items() if key.startswith("fit/")) < 100:
        raise ValueError("Too few unique fitting works")
    return {"schema": SCHEMA, "method": METHOD, "counts": counts, "editorial_sha256": pool.file_hash(EDITORIAL),
            "model": pool.BASELINE_ASSETS, "revision": pool.BASELINE_REVISION,
            "exposure": {path.name: pool.file_hash(path) for path in exposed}, "rows": rows}


def extend_reports(roster, directory):
    """Add untouched-domain work groups before any fitting or held inference."""
    prior_ids = {digest for path in exposure_paths() for digest in SHA256.findall(path.read_text())}
    prior_ids.update(row["group"] for row in roster["rows"])
    used_phrases = {phrase for row in roster["rows"] for phrase in fingerprints(row["text"])}
    for family, pin in zip(holdout.FAMILIES, REPORT_PINS, strict=True):
        table = holdout.shard(directory / f"reports_{family}.csv", "reports", family, pin)
        for work, versions in table.items():
            if pool.sha(work) in prior_ids and "v0" in versions:
                used_phrases.update(fingerprints(versions["v0"]["text"]))
        positions = Counter()
        for work in sorted(table, key=lambda key: pool.sha(f"{SCHEMA}/held/reports/{family}/{key}")):
            if sum(positions.values()) == 24:
                break
            versions = table[work]
            group = pool.sha(work)
            if group in prior_ids or "v0" not in versions:
                continue
            phrases = fingerprints(versions["v0"]["text"])
            if phrases & used_phrases:
                continue
            try:
                options = holdout.excerpt_options(work, versions)
            except (KeyError, ValueError):
                continue
            if not options:
                continue
            position = min(options, key=lambda index: (positions[index], index))
            for (text, spans, variant), label in zip(options[position], ("original", family), strict=True):
                roster["rows"].append({"role": "held", "domain": "reports", "family": label, "pair_family": family,
                                       "group": group, "sha256": pool.sha(text), "text": text, "spans": spans,
                                       "variant": variant, "truth": spans[position] if variant else None,
                                       "spaces": [i for i, c in enumerate(text) if c.isspace()]})
            positions[position] += 1
            used_phrases.update(phrases)
            prior_ids.add(group)
        roster["counts"][f"held/reports/{family}"] = sum(positions.values())
        print("held reports", family, dict(positions), flush=True)
    return roster | {"reports_sha256": dict(zip(holdout.FAMILIES, REPORT_PINS, strict=True))}


def token_groups(text, spans, offsets, valid):
    groups = []
    for start, end in spans:
        expected = {i for i in range(start, end) if not text[i].isspace()}
        covered, indices = set(), []
        for i, ((a, b), used) in enumerate(zip(offsets, valid, strict=True)):
            if not used or a >= end or b <= start:
                continue
            chars = {j for j in range(a, b) if not text[j].isspace()}
            if chars - expected:
                return None
            if chars:
                covered.update(chars)
                indices.append(i)
        if covered != expected or len(indices) < 4:
            return None
        groups.append(indices)
    return groups


def extract_features(rows, clf):
    import numpy as np
    import torch

    for start in range(0, len(rows), 8):
        batch = rows[start:start + 8]
        texts = [row["text"] for row in batch]
        encoded = clf.tok(texts, return_offsets_mapping=True, padding=True, return_tensors="pt", truncation=False)
        offsets = encoded.pop("offset_mapping").tolist()
        mask = encoded["attention_mask"].tolist()
        if encoded["input_ids"].shape[1] > 512:
            for row in batch:
                yield from extract_features([row], clf) if len(batch) > 1 else (None,)
            continue
        with torch.inference_mode():
            tokens = {key: value.to(clf.device) for key, value in encoded.items()
                      if key in {"input_ids", "attention_mask"}}
            vectors = clf.model.model(**tokens).last_hidden_state.float().cpu().numpy()
        for row, hidden, coords, active in zip(batch, vectors, offsets, mask, strict=True):
            groups = token_groups(row["text"], row["spans"], coords, active)
            if groups is None:
                yield None
                continue
            means = np.stack([hidden[indices].mean(0) for indices in groups])
            delta = means[1:4] - (means[:3] + means[2:]) * .5
            yield np.concatenate((means[1:4], delta), axis=1)


def fit(rows, features):
    import numpy as np
    from sklearn.linear_model import LogisticRegression

    groups = defaultdict(list)
    for row, matrix in zip(rows, features, strict=True):
        if matrix is not None:
            groups[row["group"]].append((row, matrix))
    complete = [pair for pair in groups.values() if len(pair) == 2]
    if len(complete) < 100:
        raise ValueError("The head needs at least 100 complete training groups")
    all_vectors, labels, weights, positives = [], [], [], []
    for (base, original), (target, edited) in complete:
        assert base["truth"] is None and target["truth"] in target["spans"][1:4]
        index = target["spans"][1:4].index(target["truth"])
        all_vectors.extend([*original, *edited])
        labels.extend([False] * 3 + [j == index for j in range(3)])
        weights.extend([.6] * 3 + [3. if j == index else .6 for j in range(3)])
        positives.append(edited[index] - original[index])
    x = np.asarray(all_vectors)
    means = np.mean(x, axis=0)
    scales = np.maximum(np.std(x, axis=0), .1)
    delta = np.asarray(positives) / scales
    absolute = LogisticRegression(C=.05, max_iter=400).fit((x - means) / scales, labels, sample_weight=weights)
    paired = LogisticRegression(C=.05, max_iter=400, fit_intercept=False).fit(np.vstack((delta, -delta)), [1] * len(delta) + [0] * len(delta))
    return {"works": len(complete), "means": means.tolist(), "scales": scales.tolist(),
            "heads": {name: {"weights": clf.coef_[0].tolist(), "bias": float(clf.intercept_[0])}
                      for name, clf in zip(MODES, (absolute, paired), strict=True)}}


def measure(rows, features, parameters, clf):
    import numpy as np

    # Reuse native three-sentence inputs for the incumbent comparison.
    for start in range(0, len(rows), 16):
        batch = rows[start:start + 16]
        inputs = [text for row in batch for text, _ in list(pool.inputs(row))[:5]]
        baseline = clf.predict(inputs, batch=8)
        for index, row in enumerate(batch):
            matrix = features[start + index]
            scores = None
            if matrix is not None:
                normalized = (matrix - parameters["means"]) / parameters["scales"]
                scores = {name: [round(float(x), 7) for x in normalized @ head["weights"] + head["bias"]]
                          for name, head in parameters["heads"].items()}
            yield {key: value for key, value in row.items() if key != "text"} | {
                "baseline": [round(float(x), 7) for x in baseline[index * 5:(index + 1) * 5]], "scores": scores}
        if (start + 16) % 64 == 0:
            print(f"Measured {min(start + 16, len(rows))} of {len(rows)}", flush=True)


def limits(rows):
    qualified = [row for row in rows if row["scores"] is not None]
    if len(qualified) < 50:
        raise ValueError("Too few ordinary calibration files")
    baseline = sum(max(row["baseline"]) >= .50 for row in qualified)
    output = {}
    for name in MODES:
        maxima = sorted((max(row["scores"][name]) for row in qualified), reverse=True)
        output[name] = {"conservative": maxima[len(maxima) // 10], "matched": maxima[min(baseline, len(maxima) - 1)]}
    return output


def selected(row, name, cutoff):
    if row["scores"] is None:
        return None
    scores = row["scores"][name]
    highest = max(scores)
    return row["spans"][scores.index(highest) + 1] if highest > cutoff and scores.count(highest) == 1 else None


def report(rows, points):
    table = {}
    actual = {(pool.sha(row["group"]), row["role"]): row for row in holdout.read(EDITORIAL)["rows"]}
    for row in rows:
        ordinary = row["truth"] is None
        role = row["family"] if row["role"] == "editorial" else ("ordinary" if ordinary else "rewritten")
        cell = table.setdefault("/".join((row["domain"], role)), Counter())
        cell.update(total=1, assessed=row["scores"] is not None)
        source = row["spans"]
        ranges = [span for j, score in enumerate(row["baseline"]) if score >= .50
                  for span in [(source[max(0, j-1)][0], source[min(4, j+1)][-1])]]
        truth = row["truth"]
        space = set(row["spaces"])
        if row["role"] == "editorial":
            earlier = actual[row["group"], row["family"]]
            if row["sha256"] != earlier["text_sha256"]:
                raise ValueError("The writing worker input does not match")
            ranges = [span for card in earlier["cards"] for span in card["spans"]]
        baseline = bool(ranges) if ordinary else editorial.covers(ranges, truth, space)
        cell["baseline"] += baseline
        for head, rules in points.items():
            for policy, cutoff in rules.items():
                name = "/".join((head, policy))
                span = selected(row, head, cutoff)
                flag = bool(span) if ordinary else span == truth
                cell[name] += flag
                cell[name + "/combined"] += flag or baseline
                cell[name + "/wrong_location"] += bool(span and not ordinary and span != truth)
    return table


def review(record):
    if record["schema"] != SCHEMA or record["method"] != METHOD or record["editorial_sha256"] != pool.file_hash(EDITORIAL):
        raise ValueError("Evaluation contract differs")
    if (record["model"] != pool.BASELINE_ASSETS or record["revision"] != pool.BASELINE_REVISION
            or record["reports_sha256"] != dict(zip(holdout.FAMILIES, REPORT_PINS, strict=True))
            or any(pool.file_hash(pool.PANEL_DIR / name) != digest for name, digest in record["exposure"].items())):
        raise ValueError("Recorded model or source commitments differ")
    delivered = defaultdict(list)
    for row in [*record["calibration"], *record["held"]]:
        spans = row["spans"]
        scores = row["scores"]
        if (len(spans) != 5 or any(not 0 <= a < b for a, b in spans)
                or any(a[1] > b[0] for a, b in zip(spans, spans[1:]))
                or row["truth"] is not None and row["truth"] not in spans[1:4]
                or len(row["baseline"]) != 5 or any(not 0 <= x <= 1 for x in row["baseline"])
                or not SHA256.fullmatch(row["sha256"]) or not SHA256.fullmatch(row["group"])
                or scores is not None and (set(scores) != set(MODES)
                    or any(len(x) != 3 or not all(math.isfinite(y) for y in x) for x in scores.values()))):
            raise ValueError("A sentence range, source binding or scoring record is invalid")
        if row["role"] != "editorial":
            delivered[row["group"]].append(row)
    for group in delivered.values():
        if (len({(r["role"], r["domain"], r["pair_family"]) for r in group}) != 1
                or sum(r["truth"] is None for r in group) != 1
                or len(group) != (1 if group[0]["role"] == "calibration" else 2)
                or len({r["sha256"] for r in group}) != len(group)):
            raise ValueError("A benchmark source group has repeated or mismatched deliveries")
    expected = {key: count for key, count in record["counts"].items() if not key.startswith("fit/") and count}
    counted = Counter("/".join((group[0]["role"], group[0]["domain"], group[0]["pair_family"])) for group in delivered.values())
    if counted != expected:
        raise ValueError("The results are missing source groups")
    points = limits(record["calibration"])
    if points != record["points"] or summarize(record["held"], points) != record["results"]:
        raise ValueError("Evaluation recount differs")
    return record["results"]


def summarize(rows, points):
    return {"by_domain": report(rows, points),
            "by_generator": {family: report([r for r in rows if r.get("pair_family") == family], points)
                             for family in holdout.FAMILIES}}


def run(path, target, parameters, mode):
    import torch

    from app.detectors.text.classifier import FourClassTextClassifier

    roster = holdout.read(path)
    if roster["schema"] != SCHEMA or roster["method"] != METHOD:
        raise ValueError("Frozen roster differs")
    if any(pool.file_hash(pool.SNAPSHOT / name) != digest for name, digest in pool.BASELINE_ASSETS.items()):
        raise ValueError("The model must match the pinned serving weights")
    torch.set_num_threads(3)
    clf = FourClassTextClassifier.from_snapshot(pool.SNAPSHOT, "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Encoder loaded on {clf.device}", flush=True)
    if mode == "develop":
        training = [row for row in roster["rows"] if row["role"] == "fit"]
        params = fit(training, list(extract_features(training, clf)))
        print(f"Fitted {params['works']} complete source groups", flush=True)
        calibration = [row for row in roster["rows"] if row["role"] == "calibration"]
        measured = list(measure(calibration, list(extract_features(calibration, clf)), params, clf))
        write(target, {"roster_sha256": pool.file_hash(path), "parameters": params, "calibration": measured,
                       "points": limits(measured)})
    else:
        dev = holdout.read(parameters)
        if dev["roster_sha256"] != pool.file_hash(path) or limits(dev["calibration"]) != dev["points"]:
            raise ValueError("Operating points or levels differ from development")
        rows = [row for row in roster["rows"] if row["role"] in {"held", "editorial"}]
        measured = list(measure(rows, list(extract_features(rows, clf)), dev["parameters"], clf))
        results = summarize(measured, dev["points"])
        result = {key: value for key, value in roster.items() if key != "rows"} | {
            "roster_sha256": dev["roster_sha256"], "parameters_sha256": pool.file_hash(parameters),
            "fit_works": dev["parameters"]["works"], "points": dev["points"], "calibration": dev["calibration"],
            "held": measured, "results": results}
        review(result)
        write(target, result)
        print(json.dumps(results, sort_keys=True), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "extend-reports", "develop", "held", "review"))
    parser.add_argument("roster", type=Path)
    for name in ("archive", "test", "uploads", "reports", "output", "development"):
        parser.add_argument(f"--{name}", type=Path)
    args = parser.parse_args()
    if args.action == "freeze":
        if not (args.archive and args.test and args.uploads):
            parser.error("Freeze requires the development shards, held shards and retained journal inputs.")
        write(args.roster, freeze(args.archive, args.test, args.uploads))
    elif args.action == "extend-reports":
        if not (args.reports and args.output):
            parser.error("Provide the two report shards and a new output path.")
        write(args.output, extend_reports(holdout.read(args.roster), args.reports))
    elif args.action == "review":
        print(json.dumps(review(holdout.read(args.roster)), sort_keys=True))
    else:
        if not args.output or args.action == "held" and not args.development:
            parser.error("Inference needs a new output path. Held inference also needs the frozen development record.")
        run(args.roster, args.output, args.development, args.action)
