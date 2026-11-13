"""Evaluate entropy-conditioned token predictability on exact mixed-writing locations."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from bisect import bisect_right
from collections import defaultdict
from pathlib import Path
from statistics import fmean, pstdev

from scripts.review_writing_long_position import baseline, read, select, useful

SCHEMA = "chipmunk.writing-entropy-conditioned/1"
ROOT = Path(__file__).resolve().parents[2]
PANELS = ROOT / "docs/evaluation-panels"
PREVIOUS = {
    "development": PANELS / "writing-long-position-transfer-2026-11-12.json",
    "f1000": PANELS / "writing-f1000-position-transfer-2026-11-13.json",
    "elife": PANELS / "writing-elife-long-position-transfer-2026-11-13.json",
}
SOURCE_ROSTERS = (PANELS / "writing-f1000-revision-spans-2026-11-13.json",)
ENTROPY_BOUNDS = [1, 2, 3, 4, 6]
MIN_BIN_TOKENS = 100
POSITION_POINT = 3.6821095168341915
MIN_FOCUS_TOKENS = 12
MAX_PARAGRAPH_TOKENS = 1024
METAL_CACHE_LIMIT = 96 * 1024 * 1024
RUNTIME = {"mlx": "0.32.4", "mlx-lm": "0.31.3", "transformers": "5.15.1", "pysbd": "0.3.4"}


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def sha(data):
    return hashlib.sha256(data.encode()).hexdigest()


def seal(path, data):
    with path.open("x") as out:
        json.dump(data, out, indent=2, sort_keys=True, allow_nan=False)
        out.write("\n")


def case_id(panel, row):
    return "/".join((panel, row["group"], row["variant"]))


def cases(files=None):
    result = {}
    for panel, public in PREVIOUS.items():
        prior = {case_id(panel, r): r for r in read(public)["cases"]
                 if panel != "development" or r["role"] == "calibration"}
        native = {} if files is None else {case_id(panel, r): r for r in read(files[panel])}
        for key, previous in prior.items():
            if files is None:
                result[key] = previous
                continue
            row = native[key]
            if sha(row["text"]) != previous["text_sha256"] or row["truth"] != previous["truth"]:
                raise ValueError(f"Source or authored range changed: {key}")
            result[key] = {**previous, "text": row["text"]}
    return result


def groups_for_fit(rows):
    groups = {row["group"] for key, row in rows.items() if key.startswith("development/")}
    ordered = sorted(groups, key=lambda group: sha(f"entropy-conditioning/2026-11-13/{group}"))
    if len(ordered) != 14:
        raise ValueError("Expected fourteen prior genuine development groups")
    return {"token_moments": ordered[:7], "decision_point": ordered[7:]}


def freeze(run, files, snapshot):
    from scripts.writing_token_curvature import MODEL_ASSETS, REVISION, verified_model

    verified_model(snapshot)
    rows = cases(files)
    groups = groups_for_fit(rows)
    roles = defaultdict(set)
    for key, row in rows.items():
        roles[key.split("/", 1)[0]].add(row["group"])
    if sum(map(len, roles.values())) != len(set().union(*roles.values())):
        raise ValueError("Panel work identifiers overlap")
    development_hashes = {row["text_sha256"] for key, row in rows.items() if key.startswith("development/")}
    if any(row["text_sha256"] in development_hashes for key, row in rows.items()
           if not key.startswith("development/")):
        raise ValueError("Genuine development excerpts overlap held excerpts")
    seal(run / "freeze.json", {
        "schema": SCHEMA, "script_sha256": digest(Path(__file__)), "runtime": RUNTIME,
        "model_revision": REVISION, "model_assets": MODEL_ASSETS,
        "case_sha256": {panel: digest(path) for panel, path in files.items()},
        "prior_sha256": {panel: digest(path) for panel, path in PREVIOUS.items()},
        "source_roster_sha256": {path.name: digest(path) for path in SOURCE_ROSTERS},
        "groups": groups, "entropy_bounds_nats": ENTROPY_BOUNDS, "minimum_bin_tokens": MIN_BIN_TOKENS,
        "minimum_focus_tokens": MIN_FOCUS_TOKENS, "maximum_paragraph_tokens": MAX_PARAGRAPH_TOKENS,
        "metal_cache_limit_bytes": METAL_CACHE_LIMIT,
        "position_point": POSITION_POINT, "advance": "Two additional exact locations in at least two works "
        "and no more than one added ordinary work per held panel. Research transfer only.",
    })


def native_geometry(row):
    from app.detectors.text.extract import from_string
    from app.detectors.text.segment import segment

    body = from_string(row["text"])
    sentences = segment(body)
    geometry = [[s.start, s.end, s.paragraph, s.n_words] for s in sentences]
    if body.text != row["text"] or geometry != row["positions"]:
        raise ValueError(f"Native source positions changed: {row['group']}/{row['variant']}")
    return body, sentences


def score(run, files, snapshot, cohort):
    from functools import lru_cache

    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm import load
    from transformers import AutoTokenizer

    from scripts.writing_token_curvature import target_token_indices, verified_model

    locked = read(run / "freeze.json")
    if (locked["script_sha256"] != digest(Path(__file__))
            or locked["case_sha256"] != {panel: digest(path) for panel, path in files.items()}
            or locked["prior_sha256"] != {panel: digest(path) for panel, path in PREVIOUS.items()}):
        raise ValueError("Locked study inputs changed")
    if (any(os.getenv(k) != "1" for k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"))
            or any(importlib.metadata.version(p) != v for p, v in RUNTIME.items())):
        raise ValueError("Expected the pinned offline runtime")
    verified_model(snapshot)
    mx.set_default_device(mx.gpu)
    mx.set_cache_limit(METAL_CACHE_LIMIT)
    model, _ = load(str(snapshot), tokenizer_config={"trust_remote_code": False})
    nn.quantize(model, mode="affine", group_size=64, bits=4)
    mx.eval(model.parameters())
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False)

    @lru_cache(maxsize=512)
    def measure(paragraph, spans):
        encoded = tokenizer(paragraph, add_special_tokens=False, truncation=False, return_offsets_mapping=True)
        tokens, offsets = encoded["input_ids"], encoded["offset_mapping"]
        if len(tokens) > MAX_PARAGRAPH_TOKENS:
            return {span: None for span in spans}
        positions = {}
        for span in spans:
            try:
                indices = target_token_indices(paragraph, offsets, span)
            except ValueError:
                positions[span] = []
            else:
                positions[span] = indices if len(indices) >= MIN_FOCUS_TOKENS else []
        selected = sorted({idx for indices in positions.values() for idx in indices})
        if not selected:
            return {span: None for span in spans}
        hidden = model.model(mx.array([tokens]))[0]
        mx.eval(hidden)
        head = model.model.embed_tokens.as_linear if model.args.tie_word_embeddings else model.lm_head
        by_index = {}
        for start in range(0, len(selected), 16):
            batch = selected[start:start + 16]
            logits = head(hidden[mx.array([i - 1 for i in batch])]).astype(mx.float32)
            shifted = logits - mx.max(logits, axis=-1, keepdims=True)
            logp = shifted - mx.logsumexp(shifted, axis=-1, keepdims=True)
            probs = mx.exp(logp)
            expected = mx.sum(probs * logp, axis=-1)
            variance = mx.sum(probs * mx.square(logp - expected[:, None]), axis=-1)
            observed = logp[mx.arange(len(batch)), mx.array([tokens[i] for i in batch])]
            scalars = mx.stack([-expected, observed - expected, variance], axis=-1).tolist()
            if any(not all(math.isfinite(x) for x in item) or item[2] <= 0 for item in scalars):
                raise ValueError("Nonfinite or degenerate token moments")
            by_index.update(zip(batch, scalars, strict=True))
        return {span: {"token_sha256": sha(json.dumps([tokens[i] for i in ids])),
                       "values": [by_index[i] for i in ids]} if ids else None for span, ids in positions.items()}

    rows = cases(files)
    selected_rows = {key: row for key, row in rows.items() if key.startswith("development/") == (cohort == "fit")}
    result = {}
    for key, row in selected_rows.items():
        body, sentences = native_geometry(row)
        by_paragraph = defaultdict(list)
        for focus in row["review"]:
            sentence = sentences[focus["idx"]]
            by_paragraph[sentence.paragraph].append(sentence)
        readings = {}
        for paragraph, focus_sentences in by_paragraph.items():
            a, b = body.paragraphs[paragraph]
            spans = tuple((s.start - a, s.end - a) for s in focus_sentences)
            values = measure(body.text[a:b], spans)
            readings.update((str(s.idx), values[span]) for s, span in zip(focus_sentences, spans, strict=True))
        result[key] = readings
        print(key, sum(x is not None for x in readings.values()), "/", len(readings), flush=True)
    seal(run / f"{cohort}-private.json", {"freeze_sha256": digest(run / "freeze.json"), "readings": result,
                                          "paragraph_batches": measure.cache_info().misses})


def fit_bins(token_values):
    buckets = [[] for _ in range(len(ENTROPY_BOUNDS) + 1)]
    for entropy, excess, _ in token_values:
        buckets[bisect_right(ENTROPY_BOUNDS, entropy)].append(excess)
    pooled = [x for group in buckets for x in group]
    if len(pooled) < MIN_BIN_TOKENS:
        raise ValueError("Insufficient genuine tokens to fit")
    return [{"count": len(group), "mean": fmean(values), "sd": max(0.25, pstdev(values))}
            for group in buckets for values in [group if len(group) >= MIN_BIN_TOKENS else pooled]]


def reduce_tokens(values, bins):
    raw = math.fsum(r for _, r, _ in values) / math.sqrt(math.fsum(v for _, _, v in values))
    calibrated = []
    for entropy, residual, _ in values:
        group = bins[bisect_right(ENTROPY_BOUNDS, entropy)]
        calibrated.append((residual - group["mean"]) / group["sd"])
    return {"raw": raw, "conditioned": math.fsum(calibrated) / math.sqrt(len(values))}


def winner(readings, method, point):
    ordered = sorted(((r[method], int(i)) for i, r in readings.items() if r is not None), reverse=True)
    if not ordered or ordered[0][0] <= point or (len(ordered) > 1 and ordered[0][0] == ordered[1][0]):
        return None
    return ordered[0][1]


def evaluate(rows, measurements, decision_groups):
    methods = ("raw", "conditioned")
    group_scores = {m: defaultdict(list) for m in methods}
    for key, readings in measurements.items():
        row = rows[key]
        if key.startswith("development/") and row["group"] in decision_groups:
            for method in methods:
                group_scores[method][row["group"]].extend(r[method] for r in readings.values() if r is not None)
    if any(set(groups) != set(decision_groups) or any(not v for v in groups.values())
           for groups in group_scores.values()):
        raise ValueError("Incomplete genuine point fit")
    maxima = {m: {g: max(vals) for g, vals in scores.items()} for m, scores in group_scores.items()}
    points = {m: sorted(scores.values(), reverse=True)[1] for m, scores in maxima.items()}
    outcomes = {}
    for key, readings in measurements.items():
        row = rows[key]
        if key.startswith("development/"):
            continue
        current = baseline(row)
        position = select(row, POSITION_POINT)
        candidate = {m: winner(readings, m, points[m]) for m in methods}
        outcomes[key] = {"current": current, "position": position, **candidate,
                         "current_hit": any(useful(row, i) for i in current),
                         **{f"{name}_hit": idx is not None and row["positions"][idx][:2] == row["truth"]
                            for name, idx in {"position": position, **candidate}.items()}}
    summary = {}
    for panel in PREVIOUS.keys() - {"development"}:
        items = [(key, val) for key, val in outcomes.items() if key.startswith(panel + "/")]
        positives = [(k, r) for k, r in items if rows[k]["truth"] is not None]
        ordinary = [(k, r) for k, r in items if rows[k]["truth"] is None]
        base_group = {rows[k]["group"] for k, r in ordinary if r["current"] or r["position"] is not None}
        panel_result = {"edits": len(positives), "ordinary_files": len(ordinary),
                        "ordinary_works": len({rows[k]["group"] for k, _ in ordinary}),
                        "current_hits": sum(r["current_hit"] for _, r in positives),
                        "current_plus_position_hits": sum(r["current_hit"] or r["position_hit"] for _, r in positives),
                        "current_plus_position_flagged_works": len(base_group),
                        "unmeasured_targets": sum(not any(val is not None and rows[k]["positions"][int(i)][:2]
                                                           == rows[k]["truth"] for i, val in measurements[k].items())
                                                  for k, _ in positives)}
        for m in methods:
            flagged = {rows[k]["group"] for k, r in ordinary if r[m] is not None}
            additions = [k for k, r in positives if r[f"{m}_hit"] and not r["current_hit"] and not r["position_hit"]]
            panel_result[m] = {"hit_count": sum(r[f"{m}_hit"] for _, r in positives),
                               "wrong_location_count": sum(r[m] is not None and not r[f"{m}_hit"] for _, r in positives),
                               "ordinary_file_flags": sum(r[m] is not None for _, r in ordinary),
                               "ordinary_work_flags": sorted(flagged),
                               "added_ordinary_works": sorted(flagged - base_group), "added_locations": additions}
        summary[panel] = panel_result
    gains = [key for panel in summary.values() for key in panel["conditioned"]["added_locations"]]
    advance = len({key.rsplit("/", 1)[0] for key in gains}) >= 2 and all(
        len(panel["conditioned"]["added_ordinary_works"]) <= 1 for panel in summary.values())
    return {"genuine_group_maxima": maxima, "points": points, "outcomes": outcomes, "summary": summary, "advance": advance}


def fit(run):
    locked = read(run / "freeze.json")
    data = read(run / "fit-private.json")
    if data["freeze_sha256"] != digest(run / "freeze.json"):
        raise ValueError("Genuine scores are not bound to the frozen run")
    rows = cases()
    fit_groups = set(locked["groups"]["token_moments"])
    tokens = [token for key, readings in data["readings"].items() if rows[key]["group"] in fit_groups
              for reading in readings.values() if reading is not None for token in reading["values"]]
    bins = fit_bins(tokens)
    seal(run / "fit.json", {"freeze_sha256": data["freeze_sha256"], "readings_sha256": digest(run / "fit-private.json"),
                            "bins": bins})


def publish(run, target):
    locked = read(run / "freeze.json")
    fitted = read(run / "fit.json")
    fit_data = read(run / "fit-private.json")
    held = read(run / "held-private.json")
    freeze_digest = digest(run / "freeze.json")
    if ({v["freeze_sha256"] for v in (fit_data, held, fitted)} != {freeze_digest}
            or fitted["readings_sha256"] != digest(run / "fit-private.json")):
        raise ValueError("Unbound measurement or fit")
    measurements = {}
    token_commitments = {}
    for key, readings in (fit_data["readings"] | held["readings"]).items():
        measurements[key] = {i: None if r is None else reduce_tokens(r["values"], fitted["bins"]) for i, r in readings.items()}
        token_commitments[key] = {i: [len(r["values"]), r["token_sha256"]] for i, r in readings.items() if r is not None}
    outcome = evaluate(cases(), measurements, locked["groups"]["decision_point"])
    seal(target, {"schema": SCHEMA, "freeze": locked, "freeze_sha256": freeze_digest, "bins": fitted["bins"],
                  "readings": measurements, "token_commitments": token_commitments, "results": outcome,
                  "private_commitments": {name: digest(run / name) for name in ("fit-private.json", "held-private.json", "fit.json")},
                  "paragraph_batches": {"fit": fit_data["paragraph_batches"], "held": held["paragraph_batches"]}})


def verify(path):
    receipt = read(path)
    rows = cases()
    readings = receipt["readings"]
    lock_json = json.dumps(receipt["freeze"], indent=2, sort_keys=True, allow_nan=False) + "\n"
    if (receipt["schema"] != SCHEMA or set(readings) != set(rows)
            or receipt["freeze_sha256"] != sha(lock_json)
            or receipt["freeze"]["script_sha256"] != digest(Path(__file__))
            or receipt["freeze"]["groups"] != groups_for_fit(rows)
            or receipt["freeze"]["prior_sha256"] != {panel: digest(p) for panel, p in PREVIOUS.items()}
            or receipt["freeze"]["source_roster_sha256"] != {path.name: digest(path) for path in SOURCE_ROSTERS}
            or receipt["freeze"]["entropy_bounds_nats"] != ENTROPY_BOUNDS
            or receipt["freeze"]["position_point"] != POSITION_POINT):
        raise ValueError("Missing rows or changed prior")
    for key, focus_rows in readings.items():
        indices = {str(row["idx"]) for row in rows[key]["review"]}
        bindings = receipt["token_commitments"][key]
        if set(focus_rows) != indices or set(bindings) != {idx for idx, r in focus_rows.items() if r is not None}:
            raise ValueError(f"Missing or extra source focus: {key}")
        for idx, (count, digest_hex) in bindings.items():
            if not MIN_FOCUS_TOKENS <= count <= MAX_PARAGRAPH_TOKENS or len(bytes.fromhex(digest_hex)) != 32:
                raise ValueError(f"Invalid focus token commitment: {key}/{idx}")
            if set(focus_rows[idx]) != {"raw", "conditioned"} or not all(map(math.isfinite, focus_rows[idx].values())):
                raise ValueError(f"Invalid focus scores: {key}/{idx}")
    recomputed = evaluate(rows, readings, receipt["freeze"]["groups"]["decision_point"])
    if recomputed != receipt["results"]:
        raise ValueError("Decision arithmetic differs from receipt")
    print(json.dumps({k: recomputed[k] for k in ("points", "summary", "advance")}, indent=2, sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "score-fit", "fit", "score-held", "publish", "verify"))
    parser.add_argument("path", type=Path)
    parser.add_argument("--development", type=Path)
    parser.add_argument("--f1000", type=Path)
    parser.add_argument("--elife", type=Path)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--publication", type=Path)
    args = parser.parse_args()
    files = {key: getattr(args, key) for key in PREVIOUS}
    if args.phase in ("freeze", "score-fit", "score-held") and (not args.snapshot or not all(files.values())):
        parser.error("Native execution needs all three case files and the pinned snapshot")
    if args.phase == "freeze":
        freeze(args.path, files, args.snapshot)
    elif args.phase == "score-fit":
        score(args.path, files, args.snapshot, "fit")
    elif args.phase == "fit":
        fit(args.path)
    elif args.phase == "score-held":
        if not (args.path / "fit.json").exists():
            parser.error("Freeze the genuine-reference fit before held inference")
        score(args.path, files, args.snapshot, "held")
    elif args.phase == "publish":
        if not args.publication:
            parser.error("Supply --publication")
        publish(args.path, args.publication)
    else:
        verify(args.path)


if __name__ == "__main__":
    main()
