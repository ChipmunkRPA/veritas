# SPDX-License-Identifier: MIT
"""Measure whether short speech-edit localization transfers after multi-generator fitting."""

from __future__ import annotations

import argparse
import math
from collections import Counter, defaultdict
from pathlib import Path

from scripts import audio_clean_teacher_transfer as prior

SOURCE, base, features, panels = prior.SOURCE, prior.base, prior.features, prior.panels
SCHEMA = "chipmunk.audio-joint-generator-transfer/1"
GENERATOR_A, GENERATOR_B = base.sources.FAMILIES
ARMS = (GENERATOR_A, GENERATOR_B, "balanced", "both")
METHOD = {
    "fit": "Keep the existing speech encoder, temporal head, losses, learning rate and epochs. Compare both "
           "single-generator models with one rendering per fitting prompt alternating by reader, and with both "
           "renderings per prompt. Only the last fit increases updates. Fit readers and timing labels remain fixed.",
    "readout": "Existing speech-admitted 500, 750 and 1000 ms proposals and content plus boundary reading.",
    "calibration": "Use the same 16 ordinary prompt groups for every arm, including dry and noisy genuine recordings, "
                   "both resampling controls and isolated backgrounds. Set the strict or 90th-percentile point "
                   "using the maximum foreground response per prompt and the highest isolated response as a floor.",
    "evaluation": "Use the fixed points on the same held recordings and count a time as useful if it covers at least "
                  "half the labeled range and reaches 0.30 temporal intersection over union. Compare with the frozen "
                  "current two-second result and with the current result followed by each candidate.",
}


def fit_plan(plan, arm):
    if arm in base.sources.FAMILIES:
        return plan, arm, dict.fromkeys((arm,), 48)
    rows_by_group = defaultdict(list)
    for row in plan["rows"]:
        if row["role"] == "fit":
            rows_by_group[row["group"]].append(row)
    readers = defaultdict(list)
    for case in plan["sources"]:
        if case["role"] == "fit":
            readers[case["reader_sha256"]].append(case)
    sources, rows, counts = [], [], Counter()
    for _, prompts in sorted(readers.items()):
        assert len(prompts) == 8
        for index, case in enumerate(sorted(prompts, key=lambda item: item["group"])):
            families = base.sources.FAMILIES if arm == "both" else (base.sources.FAMILIES[index % 2],)
            for family in families:
                group = len(sources)
                sources.append(case | {"group": group})
                counts[family] += 1
                for row in rows_by_group[case["group"]]:
                    kind = row["kind"]
                    if "/" in kind:
                        name, variant = kind.split("/", 1)
                        if variant != family:
                            continue
                        kind = f"{name}/joint"
                    rows.append(row | {"group": group, "kind": kind})
    expected = 48 if arm == "both" else 24
    assert len(readers) == 6 and counts == dict.fromkeys(base.sources.FAMILIES, expected)
    return {**plan, "sources": sources, "rows": rows}, "joint", dict(counts)


def ordinary_calibration(plan):
    rows = [row for row in plan["rows"] if row["role"] == "calibration" and not row["kind"].startswith("edited/")]
    assert len(rows) == 144 and len({row["group"] for row in rows}) == 16
    return rows


def points(rows):
    maxima, background = {}, []
    for row in rows:
        assert not row["truth"]
        if row["best"] is None:
            continue
        score = row["best"]["score"]
        if row["kind"] == "background":
            background.append(score)
        else:
            maxima[row["group"]] = max(score, maxima.get(row["group"], -math.inf))
    assert len(maxima) == 16 and background
    sorted_maxima = sorted(maxima.values())
    return {"strict": max(sorted_maxima[-1], max(background)),
            "wider": max(sorted_maxima[math.ceil(.90 * len(maxima)) - 1], max(background))}


def fit(args):
    import torch
    from safetensors.torch import save_file

    plan, digest = base.load_plan(args.data)
    ledger, feature_of = features.feature_reader(args.cache, plan, digest)
    activity = prior.runtime(args.speech)
    freeze = {"schema": SCHEMA, "method": METHOD, "source_sha256": digest,
              "feature_ledger_sha256": SOURCE.path_sha(args.cache / "features.json"), "encoder": ledger["encoder"],
              "code_sha256": SOURCE.path_sha(Path(__file__)),
              "serving_comparisons_sha256": [SOURCE.path_sha(path) for path in args.anchor]}
    args.output.mkdir(mode=0o700)
    prior.save(args.output / "freeze.json", freeze)
    rows = ordinary_calibration(plan)
    heads = []
    for arm in ARMS:
        training, family, counts = fit_plan(plan, arm)
        model, center, scale, losses = base.train(args.data, training, family, None, feature_of=feature_of)
        weights = args.output / f"{arm}.safetensors"
        save_file(model.state_dict() | {"feature_mean": torch.from_numpy(center),
                                        "feature_scale": torch.from_numpy(scale)}, str(weights))
        measured = base.score_rows(model, (center, scale), args.data, rows, None, activity, feature_of=feature_of)
        calibration = prior._attach(rows, measured)
        heads.append({"arm": arm, "family_counts": counts, "file": weights.name, "sha256": SOURCE.path_sha(weights),
                      "losses": losses, "calibration": calibration, "points": points(calibration)})
        print(f"Fit and calibrated {arm} with {counts}", flush=True)
    prior.save(args.output / "heads.json", heads)


def matches(source, anchor):
    return all(source[key] == anchor[key] for key in ("panel", "group", "kind", "room", "pcm_sha256", "truth"))


def assess(args):
    import torch

    plan, digest = base.load_plan(args.data)
    ledger, feature_of = features.feature_reader(args.cache, plan, digest)
    freeze = prior.read(args.fit / "freeze.json")
    heads = prior.read(args.fit / "heads.json")
    assert freeze["schema"] == SCHEMA and freeze["method"] == METHOD and freeze["source_sha256"] == digest
    assert freeze["encoder"] == ledger["encoder"] and freeze["code_sha256"] == SOURCE.path_sha(Path(__file__))
    assert freeze["feature_ledger_sha256"] == SOURCE.path_sha(args.cache / "features.json")
    assert [SOURCE.path_sha(path) for path in args.anchor] == freeze["serving_comparisons_sha256"]
    held, fresh = map(prior.read, args.anchor)
    attached = {(row["panel"], row["group"], row["kind"], row["pcm_sha256"]): row
                for packet in (held, fresh) for row in packet["rows"]}
    assert len(attached) == 559
    activity = prior.runtime(args.speech)
    encode = features.frozen_encoder(args.encoder)
    models = [(head["arm"], *features.load_head(args.fit, {"trained_family": head["arm"], "weights_sha256": head["sha256"]}))
              for head in heads]
    known = {row["pcm_sha256"]: row for row in plan["rows"]}
    rows, cache = [], {}

    def inspect(source, pcm):
        digest = source["pcm_sha256"]
        anchor = attached.pop((source["panel"], source["group"], source["kind"], digest))
        assert matches(source, anchor)
        if digest not in cache:
            raw = feature_of(args.data, known[digest]) if digest in known else encode(pcm)
            speech = activity.spans(pcm)
            cache[digest] = {}
            for arm, model, (center, scale) in models:
                x = torch.from_numpy(((raw - center[:, None]) / scale[:, None])[None])
                with torch.inference_mode():
                    local, boundary = model(x)[0].numpy()
                cache[digest][arm] = base.best_interval(local, boundary, speech)
        rows.append({key: anchor[key] for key in ("panel", "group", "kind", "room", "pcm_sha256", "truth", "current")}
                    | {"candidates": cache[digest]})
        if len(rows) % 50 == 0:
            print(f"Compared {len(rows)} retained recordings", flush=True)

    primary = SOURCE.read(panels.PRIMARY)
    retained = prior.read(panels.SINE)
    for row, pcm, _ in panels.inputs(args, plan, primary, retained):
        inspect(row, pcm)

    for data, receipt in ((args.recent_sine, args.recent_sine_receipt), (args.recent_bbc, args.recent_bbc_receipt),
                          (args.confirmation, args.anchor[1])):
        packet = prior.read(receipt)
        original_freeze = packet.get("source_manifest", packet["freeze"])
        assert prior.read(data / "freeze.json") == original_freeze
        for row in packet["rows"]:
            inspect(row, base.pcm(data, row))
    assert not attached
    result = {"freeze": freeze, "heads": heads, "rows": rows}
    result["results"] = review(result)
    prior.save(args.output, result)


def review(document):
    assert document["freeze"]["schema"] == SCHEMA and document["freeze"]["method"] == METHOD
    groups, results = defaultdict(list), {}
    for row in document["rows"]:
        groups[row["panel"]].append(row)
    for label, rows in sorted(groups.items()):
        data = [row | {"candidates": {"fitted/" + key: value for key, value in row["candidates"].items()}}
                for row in rows]
        summary = {"current": prior._summarize(data, "current", "fitted", None)}
        for head in document["heads"]:
            assert points(head["calibration"]) == head["points"]
            arm = head["arm"]
            summary[arm] = {name: {"alone": prior._summarize(data, arm, "fitted", point),
                                  "fallback": prior._summarize(data, arm, "fitted", point, fallback=True)}
                            for name, point in head["points"].items()}
        results[label] = summary
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="phase", required=True)
    for phase in ("fit", "assess", "review"):
        cmd = commands.add_parser(phase)
        cmd.add_argument("output", type=Path)
        if phase != "review":
            for name in ("data", "cache", "speech"):
                cmd.add_argument(f"--{name}", required=True, type=Path)
            cmd.add_argument("--anchor", type=Path, nargs=2, required=True)
        if phase == "assess":
            for name in ("fit", "encoder", "retained", "fresh", "stereo", "recent_sine", "recent_sine_receipt",
                         "recent_bbc", "recent_bbc_receipt", "confirmation"):
                cmd.add_argument("--" + name.replace("_", "-"), required=True, type=Path)
    args = parser.parse_args()
    if args.phase == "fit":
        fit(args)
    elif args.phase == "assess":
        assess(args)
    else:
        receipt = prior.read(args.output)
        assert review(receipt) == receipt["results"]
        print(f"Verified every cutoff and outcome on {len(receipt['rows'])} held deliveries.")


if __name__ == "__main__":
    main()
