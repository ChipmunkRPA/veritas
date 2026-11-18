# SPDX-License-Identifier: Apache-2.0
"""Test paraphrase-inversion style embeddings for standalone rewrite localization."""

from __future__ import annotations

import argparse
import math
from collections import Counter
from pathlib import Path
from statistics import median

from app.bench.writing_entropy_prominence import compact_range, matched
from scripts.verify_writing_entropy_prominence import matching
from scripts.writing_matched_lexical import fit_pairs, usable
from scripts.writing_transition_deviation import (
    COMPARATORS,
    PINS,
    file_hash,
    read,
    require,
    save,
    text,
    text_hash,
)

SCHEMA = "chipmunk.writing-style-prototypes/1"
REVISION = "ab7852825ad1be40112bbfdaf1f9e2315a1727b7"
ASSETS = {
    "README.md": "41cf1fbb8527b6ac016f04482520c210f846138a8e4edce5bb3fd5cff1a8c9a6",
    "config.json": "ebfd5ca1bc3e15c998d7d1d43589cf0bcbcee381f4d1831104e93a415ab3f81f",
    "model.safetensors": "9dc936bd31f349d874a79880549a5a9a9ca0a7ead64bebb83904cac2666cd5b9",
    "special_tokens_map.json": "06e405a36dfe4b9604f484f6a1e619af1a7f7d09e34a8555eb0b77b66318067f",
    "tokenizer.json": "727009a8214ddfa5af1deedf1006d4d06e8e51e54aa5f03566263d4e19bfcdce",
    "tokenizer_config.json": "4ea8648e4067230942ac6b32dd61328675bc99ae0a9f4cc665c30a95f0f68712",
}
ARMS = ("style", "document_relative")
SETTINGS = {"fit_groups_per_stratum": 24, "shrinkage": .5, "quantile": .95, "min_other_sentences": 3,
            "model_tokens": 256, "batch_size": 32, "max_ranges": 6, "scale": 100_000}


def source_groups(works):
    # Draw balanced source works by identifier before running the encoder.
    strata, selected = Counter(), []
    for row in sorted(works, key=lambda work: text_hash(SCHEMA + work["group"])):
        bucket = row["generator"], row["operation"]
        if strata[bucket] < SETTINGS["fit_groups_per_stratum"]:
            strata[bucket] += 1
            selected.append(row)
    if len(strata) != 8 or set(strata.values()) != {SETTINGS["fit_groups_per_stratum"]}:
        raise ValueError("The balanced fit strata are incomplete")
    return selected


def encode(sentences, folder, device):
    import torch
    from transformers import AutoModel, AutoTokenizer

    for filename, digest in ASSETS.items():
        require(folder / filename, digest)
    torch.set_num_threads(3)
    tokenizer = AutoTokenizer.from_pretrained(folder, local_files_only=True, trust_remote_code=False)
    model = AutoModel.from_pretrained(folder, local_files_only=True, trust_remote_code=False,
                                     use_safetensors=True, attn_implementation="eager").eval().to(device)
    sorted_sentences = sorted(set(sentences), key=lambda sentence: (len(sentence), sentence))
    vectors = {}
    size = SETTINGS["batch_size"]
    with torch.inference_mode():
        for start in range(0, len(sorted_sentences), size):
            batch = sorted_sentences[start:start + size]
            inputs = tokenizer(batch, padding=True, truncation=True, max_length=SETTINGS["model_tokens"],
                               return_tensors="pt").to(device)
            mask = inputs["attention_mask"].unsqueeze(-1)
            hidden = model(**inputs).last_hidden_state
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1)
            vectors.update(zip(batch, pooled.float().cpu().numpy(), strict=True))
            if start % (size * 32) == 0:
                print(f"Embedded {start + len(batch)} of {len(sorted_sentences)} study sentences", flush=True)
    return vectors


def fit(pairs, vectors):
    import numpy as np

    sides = np.asarray([[vectors[sentence] for sentence in sentences] for sentences in zip(*pairs, strict=True)], dtype=float)
    centers = sides.mean(axis=1)
    mean = centers.mean(axis=0)
    pooled = ((sides - centers[:, None, :]) ** 2).mean(axis=(0, 1))
    mix = SETTINGS["shrinkage"]
    scale = np.sqrt(np.maximum((1 - mix) * pooled + mix * np.median(pooled), 1e-8))
    direction = (centers[1] - centers[0]) / scale
    direction /= np.linalg.norm(direction)
    return {"mean": mean.tolist(), "scale": scale.tolist(), "direction": direction.tolist()}


def measured(works, vectors, model, *, calibration=False, incumbent=None, panel=None):
    import numpy as np

    mean, scale, direction = (np.asarray(model[key]) for key in ("mean", "scale", "direction"))
    prior = {(row["group"], row["native_sha256"]): row for row in incumbent or []}
    result = []
    for work in works:
        for index, case in enumerate(work["cases"]):
            kind = case["kind"]
            if calibration and kind != "ordinary":
                continue
            raw = []
            for i, sentence in enumerate(text(case)):
                if sentence in vectors:
                    value = (vectors[sentence] - mean) / scale
                    raw.append((i, float(value @ direction / max(np.linalg.norm(value), 1e-8))))
            scale_units = SETTINGS["scale"]
            style = [[i, round(scale_units * value)] for i, value in raw]
            relative = [[i, round(scale_units * (value - median(other for j, other in raw if j != i)))] for i, value in raw
                        if len(raw) > SETTINGS["min_other_sentences"]]
            role = "human_substitution" if panel == "economic_rewrites" and index > 0 and kind == "ordinary" else kind
            row = {"group": work["group"], "kind": kind, "role": role, "generator": work["generator"],
                   "native_sha256": case["native_text_sha256"], "sentences": case["sentences"],
                   "extent": case["nonspace_characters"], "truth": [item["compact"] for item in case["truth"]],
                   "scores": {"style": style, "document_relative": relative}}
            if incumbent is not None:
                old = prior.pop((row["group"], row["native_sha256"]))
                if any(row[key] != old[key] for key in ("kind", "sentences", "extent", "truth")):
                    raise ValueError("The deployed baseline scored different document positions")
                row.update(incumbent=old["incumbent"], incumbent_assessed=old["incumbent_assessed"])
            result.append(row)
    if prior:
        raise ValueError("Some deployed baseline cases were omitted")
    return result


def operating_points(calibration):
    points = {}
    for arm in ARMS:
        peaks = sorted(max(score for _, score in row["scores"][arm]) for row in calibration if row["scores"][arm])
        if len(peaks) < 80:
            raise ValueError("Insufficient ordinary calibration documents")
        points[arm] = peaks[math.ceil(SETTINGS["quantile"] * len(peaks)) - 1]
    return points


def tally(rows, points, match):
    totals = {arm: Counter() for arm in ("incumbent", *ARMS, "incumbent_plus_style", "incumbent_plus_relative")}
    for row in rows:
        baseline = {tuple(span) for span in row["incumbent"]}
        proposals = {"incumbent": (baseline, row["incumbent_assessed"])}
        for arm, complement in zip(ARMS, ("incumbent_plus_style", "incumbent_plus_relative"), strict=True):
            ordered = sorted(row["scores"][arm], key=lambda item: (-item[1], item[0]))[:SETTINGS["max_ranges"]]
            ranges = {tuple(compact_range(row["sentences"], (i, i + 1))) for i, value in ordered if value > points[arm]}
            proposals[arm] = ranges, bool(ordered)
            proposals[complement] = baseline | ranges, row["incumbent_assessed"] or bool(ordered)
        for arm, (ranges, assessed) in proposals.items():
            total = totals[arm]
            total.update(cases=1, assessed=assessed)
            if row["kind"] == "ordinary":
                total.update(ordinary=1, ordinary_alerts=bool(ranges))
                if row["role"] == "human_substitution":
                    total.update(human_substitutions=1, human_substitution_alerts=bool(ranges))
            else:
                hits = match(sorted(ranges), row["truth"])
                total.update(known=len(row["truth"]), correct=hits, missed=len(row["truth"]) - hits,
                             misplaced=len(ranges) - hits)
    return {arm: dict(total) for arm, total in totals.items()}


def develop(fit_plan, calibration, encoder, output, device):
    require(fit_plan, PINS["fit"])
    require(calibration, PINS["calibration"])
    fit_works, ordinary = source_groups(read(fit_plan)["works"]), read(calibration)["works"]
    first, second = [{row["group"] for row in works} for works in (fit_works, ordinary)]
    if first & second:
        raise ValueError("Calibration must use different training sources")
    pairs = [(original, edited) for _, original, edited in fit_pairs({"works": fit_works})]
    if len(pairs) < 400:
        raise ValueError("Insufficient exact aligned rewrites for the fit")
    ordinary_sentences = {sentence for work in ordinary for case in work["cases"] if case["kind"] == "ordinary"
                          for sentence in text(case) if usable(sentence)}
    vectors = encode(ordinary_sentences | {item for pair in pairs for item in pair}, encoder, device)
    model = fit(pairs, vectors)
    readings = measured(ordinary, vectors, model, calibration=True)
    save(output, {"schema": SCHEMA, "revision": REVISION, "assets": ASSETS, "settings": SETTINGS,
                  "implementation_sha256": file_hash(Path(__file__)), "device": device, "inputs": PINS,
                  "comparators": COMPARATORS, "fit_groups": sorted(first), "calibration_groups": sorted(second),
                  "fit_pairs": len(pairs), "model": model, "calibration": readings, "points": operating_points(readings)})
    print(f"Froze the fit on {len(pairs)} pairs and two operating points", flush=True)


def _validate_freeze(row):
    expected = (SCHEMA, REVISION, ASSETS, SETTINGS, PINS, COMPARATORS, operating_points(row["calibration"]))
    if tuple(row[key] for key in ("schema", "revision", "assets", "settings", "inputs", "comparators", "points")) != expected:
        raise ValueError("The archived settings differ from the prospective comparison")


def evaluate(frozen, encoder, output, inputs):
    sealed = read(frozen)
    _validate_freeze(sealed)
    if sealed["implementation_sha256"] != file_hash(Path(__file__)):
        raise ValueError("This implementation differs from the frozen comparison")
    used = set(sealed["fit_groups"] + sealed["calibration_groups"])
    panels = {}
    for panel, (plan, comparator) in zip(COMPARATORS, zip(inputs[::2], inputs[1::2], strict=True), strict=True):
        require(plan, PINS[panel])
        require(comparator, COMPARATORS[panel])
        works, baseline = read(plan)["works"], read(comparator)
        if used & {row["group"] for row in works}:
            raise ValueError("A held source overlaps fit or calibration")
        panels[panel] = (works, baseline.get("cases", baseline.get("readings")))
    vectors = encode((sentence for works, _ in panels.values() for work in works
                      for case in work["cases"] for sentence in text(case) if usable(sentence)), encoder, sealed["device"])
    result = {}
    for name, (works, baseline) in panels.items():
        rows = measured(works, vectors, sealed["model"], incumbent=baseline, panel=name)
        totals = tally(rows, sealed["points"], lambda p, t: len(matched(p, t)))
        result[name] = {"cases": rows, "results": totals}
    save(output, {"frozen": sealed, "panels": result})
    review(output)


def review(path):
    record = read(path)
    seal = record["frozen"]
    _validate_freeze(seal)
    used = set(seal["fit_groups"] + seal["calibration_groups"])
    if set(record["panels"]) != COMPARATORS.keys():
        raise ValueError("The saved evaluation omits one of the planned held panels")
    for name, panel in record["panels"].items():
        rows = panel["cases"]
        if ({row["group"] for row in rows} & used
                or len({(row["group"], row["native_sha256"]) for row in rows}) != len(rows)
                or tally(rows, seal["points"], matching) != panel["results"]):
            raise ValueError("An independent tally disagrees or a source was reused")
        print(name, panel["results"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    dev = modes.add_parser("develop")
    for name in ("fit_plan", "calibration", "encoder", "output"):
        dev.add_argument(name, type=Path)
    dev.add_argument("--device", choices=("mps", "cpu"), required=True)
    held = modes.add_parser("evaluate")
    for name in ("frozen", "encoder", "output"):
        held.add_argument(name, type=Path)
    held.add_argument("inputs", nargs=6, type=Path, metavar="PLAN_OR_BASELINE")
    modes.add_parser("review").add_argument("path", type=Path)
    args = vars(parser.parse_args())
    {"develop": develop, "evaluate": evaluate, "review": review}[args.pop("mode")](**args)
