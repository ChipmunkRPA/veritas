# SPDX-License-Identifier: Apache-2.0
"""Compare local sentence-transition scores on short writing revisions."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import re
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

from app.bench.writing_entropy_prominence import compact_range, matched
from scripts.verify_writing_entropy_prominence import matching

SCHEMA = "chipmunk.writing-transition-deviation/1"
REVISION = "c9745ed1d9f207416be6d2e6f8de32d1f16199bf"
ASSETS = {
    "config.json": "953f9c0d463486b10a6871cc2fd59f223b2c70184f49815e7efbcab5d8908b41",
    "model.safetensors": "53aa51172d142c89d9012cce15ae4d6cc0ca6895895114379cacb4fab128d9db",
    "special_tokens_map.json": "303df45a03609e4ead04bc3dc1536d0ab19b5358db685b6f3da123d05ec200e3",
    "tokenizer.json": "be50c3628f2bf5bb5e3a7f17b1f74611b2561a3a27eeab05e5aa30f411572037",
    "tokenizer_config.json": "acb92769e8195aabd29b7b2137a9e6d6e25c476a4f15aa4355c233426c61576b",
    "vocab.txt": "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3",
}
PINS = {
    "fit": "e5a610922707cf242db38395bfa7db9215814eed8667a463aa788708d534ec35",
    "calibration": "9cdff3ef7c23b7929efe5777d2791c55afe9cc988f1d96943e053e82b81039bb",
    "held": "393d9792f642f098e32f7432ee765076e054be828b6a86567c0fa08bc574373c",
    "gao_rewrites": "710d36ef4b31438567ab4186b194d9e493c4fa7975eefea9b8e00d48452edb0e",
    "economic_rewrites": "0eb90145de1102f4732edfea7275f5d53219b40a01a063885c19bb70caa1b76c",
}
COMPARATORS = {
    "held": "52feb197d0a2a801fbd51e70fe12655afb52c80ead52f3a0b8fe47cbb2c2bd49",
    "gao_rewrites": "f7599805e3d53a2c26c06d7696011b474740a54f240601b2b87d20cc84256a0e",
    "economic_rewrites": "db5ca53b463875b76f5de54710932e6359dff7d4c698c2ee934ba82bcf95d013",
}
ARMS = ("semantic", "surface", "joint")
SCOPES = ("paragraph", "document")
RADIUS = (1, 2, 3)
FIT_GROUPS_PER_STRATUM = 24
QUANTILE = 0.95
SCORE_UNITS = 100_000
MAX_RANGES = 6
WORD = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)?")
FUNCTION = frozenset({
    "a", "an", "the", "this", "that", "these", "those", "it", "its", "they", "them", "their", "you", "your", "his",
    "her", "he", "she", "i", "we", "our", "us", "who", "which", "where", "what", "when", "if", "although", "because",
    "while", "but", "and", "or", "yet", "is", "are", "was", "were", "been", "be", "being", "has", "have", "had",
    "does", "do", "did", "could", "would", "might", "may", "can", "should", "shall", "must", "about", "from", "between",
    "within", "in", "into", "via", "per", "to", "with", "without", "of", "at", "on", "under", "over", "for", "by",
    "as", "not", "than", "then", "there", "here", "further", "also", "however", "therefore",
})


def file_hash(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def text_hash(text):
    return hashlib.sha256(text.encode()).hexdigest()


def read(path):
    return json.loads(gzip.decompress(path.read_bytes()))


def save(path, data):
    with path.open("xb") as stream:
        stream.write(gzip.compress(json.dumps(data, separators=(",", ":"), sort_keys=True, allow_nan=False).encode(), mtime=0))


def require(path, digest):
    if file_hash(path) != digest:
        raise ValueError(f"{path.name} differs from its pinned bytes")


def style(text):
    words = WORD.findall(text)
    lowered = [word.casefold() for word in words]
    n = max(len(words), 1)
    return (
        math.log(n), sum(len(word) for word in words) / n, len(set(lowered)) / n,
        sum(word in FUNCTION for word in lowered) / n, sum(word.isupper() for word in words) / n,
        sum(c.isdigit() for c in text) / n, sum(c in ",;:" for c in text) / n,
        sum(c in "–—-()/“”" for c in text) / n,
    )


def text(case):
    source = case["text"]
    if text_hash(source) != case["native_text_sha256"]:
        raise ValueError("Source text differs from its exact commitment")
    return [source[a:b] for a, b, *_ in case["sentences"]]


def features(case, vectors, scope):
    import numpy as np

    values = text(case)
    geometry = case["sentences"]
    if len(values) < 8:
        return {}
    dense = np.asarray([vectors[item] for item in values], dtype=np.float64)
    semantic = dense @ dense.T
    raw = np.asarray([style(item) for item in values])
    centers = np.median(raw, axis=0)
    dispersion = np.maximum(np.median(np.abs(raw - centers), axis=0) * 1.4826, [.15, .2, .08, .05, .05, .05, .05, .05])
    normalized = np.clip((raw - centers) / dispersion, -6., 6.)
    surface = -np.abs(normalized[:, None, :] - normalized[None, :, :]).mean(axis=2)

    def neighbors(matrix, index, floor):
        result = []
        for radius in RADIUS:
            background = [matrix[a, a + radius] for a in range(len(values) - radius)
                          if a != index and a + radius != index
                          and (scope == "document" or geometry[a][4] == geometry[a + radius][4])]
            left, right = index - radius, index + radius
            if len(background) < 4:
                return None
            middle = np.median(background)
            spread = max(np.median(np.abs(background - middle)) * 1.4826, floor)
            earlier, later = ((matrix[index, column] - middle) / spread for column in (left, right))
            earlier, later = np.clip([earlier, later], -6., 6.)
            result.extend(((earlier + later) / 2, (abs(earlier) + abs(later)) / 2, abs(earlier - later)))
        return result

    result = {}
    for i in range(max(RADIUS), len(values) - max(RADIUS)):
        if (not 8 <= geometry[i][5] <= 100
                or scope == "paragraph" and geometry[i - max(RADIUS)][4] != geometry[i + max(RADIUS)][4]):
            continue
        sem, surf = neighbors(semantic, i, .04), neighbors(surface, i, .15)
        if sem is None or surf is None:
            continue
        surface_features = [*surf, *normalized[i], *np.abs(normalized[i])]
        result[i] = dict(semantic=sem, surface=surface_features, joint=[*sem, *surface_features])
    return result


def encode(works, folder, device):
    import torch
    import torch.nn.functional as functional
    from transformers import AutoModel, AutoTokenizer

    for filename, sha256 in ASSETS.items():
        require(folder / filename, sha256)
    torch.set_num_threads(3)
    tokenizer = AutoTokenizer.from_pretrained(folder, local_files_only=True, trust_remote_code=False)
    model = AutoModel.from_pretrained(folder, local_files_only=True, trust_remote_code=False,
                                     use_safetensors=True, attn_implementation="eager").eval().to(device)
    values = sorted({sentence for work in works for case in work["cases"] for sentence in text(case)},
                    key=lambda item: (len(item), item))
    vectors = {}
    with torch.inference_mode():
        for offset in range(0, len(values), 64):
            batch = values[offset:offset + 64]
            enc = tokenizer(batch, padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
            output = model(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1)
            encoded = functional.normalize((output * mask).sum(dim=1) / mask.sum(dim=1), dim=1).cpu().numpy()
            vectors.update(zip(batch, encoded, strict=True))
    print(f"Encoded {len(vectors):,} exact sentences from {len(works)} groups", flush=True)
    return vectors


def choose_fit(works):
    totals = Counter()
    answer = []
    for work in sorted(works, key=lambda item: text_hash(SCHEMA + item["group"])):
        stratum = work["generator"], work["operation"]
        if totals[stratum] < FIT_GROUPS_PER_STRATUM:
            answer.append(work)
            totals[stratum] += 1
    if len(totals) != 8 or any(total != FIT_GROUPS_PER_STRATUM for total in totals.values()):
        raise ValueError("Fit data do not cover the eight preselected strata")
    return answer


def fit_rows(works, vectors, scope):
    observations, weights = [], []
    for work in works:
        old, new = work["cases"]
        if (old["kind"], new["kind"]) != ("ordinary", "mixed"):
            raise ValueError("Each fit group needs one earlier version and one labeled edit")
        earlier, revised = (features(case, vectors, scope) for case in (old, new))
        alignment = SequenceMatcher(None, text(old), text(new), autojunk=False)
        pairs = []
        truth = [row["compact"] for row in new["truth"]]
        for tag, a, b, c, d in alignment.get_opcodes():
            if tag != "replace" or b - a != d - c:
                continue
            for i, j in zip(range(a, b), range(c, d), strict=True):
                if i in earlier and j in revised and matched([compact_range(new["sentences"], (j, j + 1))], truth):
                    pairs.append((earlier[i], revised[j]))
        if pairs:
            observations.extend(pairs)
            weights.extend([1. / len(pairs)] * len(pairs))
    if len(observations) < 160:
        raise ValueError("Not enough aligned fit locations")
    return observations, weights


def fit_models(observations, weights):
    import numpy as np
    from sklearn.linear_model import LogisticRegression

    fit = {}
    for arm in ARMS:
        negative, positive = (np.asarray([row[side][arm] for row in observations]) for side in (0, 1))
        scale = np.maximum(np.std(np.concatenate((negative, positive)), axis=0), 1e-6)
        movement = (positive - negative) / scale
        model = LogisticRegression(C=1., solver="liblinear", fit_intercept=False, random_state=0)
        target = np.r_[np.zeros(len(weights)), np.ones(len(weights))]
        model.fit(np.concatenate((-movement, movement)), target, sample_weight=weights * 2)
        fit[arm] = (model.coef_[0] / scale).tolist()
    return fit


def score(works, vectors, weights, scope, prior=None):
    result = []
    incumbent = {(row["group"], row["native_sha256"]): row for row in prior} if prior else {}
    for work in works:
        for case in work["cases"]:
            measured = features(case, vectors, scope)
            truth = [row["compact"] for row in case["truth"]]
            row = {"group": work["group"], "native_sha256": case["native_text_sha256"], "kind": case["kind"],
                   "genre": work["genre"], "generator": work["generator"], "operation": work["operation"],
                   "prior_project_exposure": work["prior_project_exposure"],
                   "sentences": case["sentences"], "extent": case["nonspace_characters"], "truth": truth,
                   "scores": {arm: [[index, round(SCORE_UNITS * math.fsum(a * b for a, b in zip(values[arm], weights[arm], strict=True)))]
                                    for index, values in measured.items()] for arm in ARMS}}
            if prior:
                reference = incumbent.pop((row["group"], row["native_sha256"]))
                if any(row[key] != reference[key] for key in ("sentences", "kind", "truth", "extent")):
                    raise ValueError("The serving comparison uses different source positions")
                row.update(incumbent=reference["incumbent"], incumbent_assessed=reference["incumbent_assessed"])
            result.append(row)
    if incumbent:
        raise ValueError("The trial omitted retained serving readings")
    return result


def points(rows):
    answer = {}
    for arm in ARMS:
        maxima = sorted(max(score for _, score in row["scores"][arm]) for row in rows
                        if row["kind"] == "ordinary" and row["scores"][arm])
        if len(maxima) < 64:
            raise ValueError("Fewer than 64 calibration controls have usable sentence neighborhoods")
        answer[arm] = maxima[math.ceil(len(maxima) * QUANTILE) - 1]
    return answer


def tally(rows, operating, *, alternate=False):
    totals = {}
    for arm in ("incumbent", *ARMS, "incumbent_plus_joint"):
        summary = Counter()
        for row in rows:
            proposed = row.get("incumbent", []) if arm.startswith("incumbent") else []
            assessed = row.get("incumbent_assessed", False) if arm.startswith("incumbent") else False
            if arm != "incumbent":
                name = "joint" if arm == "incumbent_plus_joint" else arm
                scores = row["scores"][name]
                ranked = sorted(scores, key=lambda item: (-item[-1], item[0]))
                proposed = sorted({tuple(pair) for pair in proposed} | {
                    tuple(compact_range(row["sentences"], (i, i + 1)))
                    for i, value in ranked[:MAX_RANGES] if value > operating[name]
                })
                assessed |= bool(scores)
                if row["kind"] == "mixed" and ranked:
                    viable = [compact_range(row["sentences"], (i, i + 1)) for i, _ in ranked]
                    summary.update(rank_assessed=1, top_location=bool(matched(viable[:1], row["truth"])))
                    summary["uniform_expectation_millionths"] += round(1_000_000 * sum(
                        bool(matched([location], row["truth"])) for location in viable) / len(viable))
            summary.update(files=1, assessed=assessed)
            if row["kind"] == "ordinary":
                summary.update(ordinary=1, ordinary_alerts=bool(proposed))
            else:
                hits = matching(proposed, row["truth"]) if alternate else len(matched(proposed, row["truth"]))
                summary.update(rewrites=1, known=len(row["truth"]), located=hits,
                               missed=len(row["truth"]) - hits, misplaced=len(proposed) - hits)
        totals[arm] = dict(summary)
    return totals


def develop(fit, calibration, encoder, output, device, scope):
    require(fit, PINS["fit"])
    require(calibration, PINS["calibration"])
    selected = choose_fit(read(fit)["works"])
    dev = read(calibration)["works"]
    fit_groups = {row["group"] for row in selected}
    dev_groups = {row["group"] for row in dev}
    if fit_groups & dev_groups:
        raise ValueError("Fit and calibration source groups overlap")
    vectors = encode(selected + dev, encoder, device)
    pairs, weights = fit_rows(selected, vectors, scope)
    model = fit_models(pairs, weights)
    ordinary = score(dev, vectors, model, scope)
    operating = points(ordinary)
    save(output, {"schema": SCHEMA, "implementation_sha256": file_hash(Path(__file__)), "encoder_revision": REVISION,
                  "assets": ASSETS, "pins": PINS, "comparator_pins": COMPARATORS, "device": device, "scope": scope,
                  "fit_groups": sorted(fit_groups), "calibration_groups": sorted(dev_groups), "fit_pairs": len(pairs),
                  "coefficients": model, "points": operating, "calibration": ordinary})
    print(f"Froze {len(pairs)} matched edits and {len(dev)} source groups at points {operating}", flush=True)


def apply(freeze, plan, comparator, encoder, output, panel):
    require(plan, PINS[panel])
    require(comparator, COMPARATORS[panel])
    frozen = read(freeze)
    if (frozen["schema"] != SCHEMA or frozen["scope"] not in SCOPES or frozen["assets"] != ASSETS
            or frozen["points"] != points(frozen["calibration"])
            or frozen["implementation_sha256"] != file_hash(Path(__file__))):
        raise ValueError("The operating settings differ from the frozen comparison")
    works, reference = read(plan)["works"], read(comparator)
    if {work["group"] for work in works} & set(frozen["fit_groups"] + frozen["calibration_groups"]):
        raise ValueError("Evaluation includes a calibrated source group")
    vectors = encode(works, encoder, frozen["device"])
    cases = score(works, vectors, frozen["coefficients"], frozen["scope"], reference.get("cases", reference.get("readings")))
    totals = tally(cases, frozen["points"])
    save(output, {"schema": SCHEMA, "panel": panel, "plan_sha256": PINS[panel], "comparator_sha256": COMPARATORS[panel],
                  "freeze_sha256": file_hash(freeze), "cases": cases, "results": totals})
    print(panel, totals, flush=True)


def review_scope(record):
    frozen = record["frozen"]
    operating = points(frozen["calibration"])
    if frozen["schema"] != SCHEMA or operating != frozen["points"] or set(record["panels"]) != COMPARATORS.keys():
        raise ValueError("Recorded operating points or evaluation panels differ")
    for name, panel in record["panels"].items():
        rows = panel["cases"]
        if ({row["group"] for row in rows} & set(frozen["fit_groups"] + frozen["calibration_groups"])
                or len({(row["group"], row["native_sha256"]) for row in rows}) != len(rows)):
            raise ValueError("Evaluation includes a reused or duplicate input")
        for row in rows:
            for readings in row["scores"].values():
                if (sorted({index for index, _ in readings}) != [index for index, _ in readings]
                        or any(not 0 <= index < len(row["sentences"]) or type(value) is not int for index, value in readings)):
                    raise ValueError("A sentence reading has invalid or duplicate source coordinates")
        if tally(rows, operating, alternate=True) != panel["results"]:
            raise ValueError("Independent location counting differs")
        print(frozen["scope"], name, panel["results"], flush=True)


def review(receipt):
    record = read(receipt)
    if scopes := record.get("scopes"):
        if set(scopes) != set(SCOPES) or any(value["frozen"]["scope"] != key for key, value in scopes.items()):
            raise ValueError("Both distinct measurement scopes are required")
        for value in scopes.values():
            review_scope(value)
    else:
        review_scope(record)


def publish(freeze, held, gao, economic, output):
    frozen = read(freeze)
    panels = {name: read(path) for name, path in zip(COMPARATORS, (held, gao, economic), strict=True)}
    expected = file_hash(freeze)
    if any(panel["freeze_sha256"] != expected or key != panel["panel"] for key, panel in panels.items()):
        raise ValueError("At least one panel used a different freeze")
    save(output, {"schema": SCHEMA, "frozen": frozen, "panels": panels})
    review(output)


def bundle(paragraph, document, output):
    scopes = {name: read(path) for name, path in zip(SCOPES, (paragraph, document), strict=True)}
    save(output, {"schema": SCHEMA, "scopes": scopes})
    review(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    for method, paths in {
        develop: ("fit", "calibration", "encoder", "output"),
        apply: ("freeze", "plan", "comparator", "encoder", "output"),
        publish: ("freeze", "held", "gao", "economic", "output"),
        bundle: ("paragraph", "document", "output"),
        review: ("receipt",),
    }.items():
        cli = subs.add_parser(method.__name__)
        cli.set_defaults(method=method)
        for field in paths:
            cli.add_argument(field, type=Path)
        if method is develop:
            cli.add_argument("--device", choices=("cpu", "mps"), default="mps")
            cli.add_argument("--scope", choices=SCOPES, required=True)
        if method is apply:
            cli.add_argument("--panel", choices=tuple(COMPARATORS), required=True)
    args = vars(parser.parse_args())
    args.pop("command")
    args.pop("method")(**args)
