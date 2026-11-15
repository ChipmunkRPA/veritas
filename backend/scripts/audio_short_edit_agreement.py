# SPDX-License-Identifier: MIT
"""Compare ways to combine short speech-edit checks at the same source clock."""

import argparse
import gzip
import json
from bisect import bisect_left
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

from scripts import audio_competing_speech_ranking as ranking
from scripts import audio_short_edit_layer_ablation as layers
from scripts import audio_temporal_boundary_transfer as boundary
from scripts.review_audio_stock_voice_transfer import useful

SOURCE = layers.SOURCE
detector = layers.detector
PANELS = ranking.PANELS
BASE = PANELS / "audio-short-edit-detector-training-2026-11-14.json"
LAYER_FIT = PANELS / "audio-short-edit-layer-ablation-fit-2026-11-15.json"
RANK_FIT = PANELS / "audio-competing-speech-ranking-2026-11-15.json"
BOUNDARY_FIT = PANELS / "audio-temporal-boundary-fit-2026-11-15.json"
LAYER_TRANSFER = PANELS / "audio-short-edit-layer-ablation-transfer-2026-11-15.json"
RANK_TRANSFER = PANELS / "audio-competing-speech-ranking-transfer-2026-11-15.json"
BOUNDARY_TRANSFER = PANELS / "audio-temporal-boundary-transfer-2026-11-15.json"
RANDOM_LAYERS = PANELS / "audio-short-edit-layer-ablation-random-2026-11-15.json"
RANDOM_EXTRA = PANELS / "audio-short-edit-agreement-random-readings-2026-11-15.json"
READERS = ("early_layer", "boundary", "matched_rank")
LEVELS = ("strict", "wider")
POLICIES = ("incumbent", *(f"{view}/{level}" for level in LEVELS
                          for view in (*READERS, "union", "agreement", "joint")))
FAMILIES = detector.sources.FAMILIES
BACKGROUND = {"background", "background_only", "isolated"}
METHOD = {
    "views": "Keep the earlier layer-4, explicit temporal boundary, and matched-clock rank head. "
             "Match windows only between different views of the same fit direction. Both directions can supply a transfer interval.",
    "union": "Preserve the current app interval and add every view's strongest interval over its original operating point.",
    "agreement": "Preserve the current app interval. Two positive views must overlap by at least half of each interval. "
                 "Report the shorter original proposal, with the earlier interval breaking equal-length ties.",
    "joint": "Rank each response against group maxima on the original calibration controls. A pair takes the weaker rank. "
             "Use the largest colocated pair rank across each file, rank calibration files with their own group excluded, "
             "and set the strict and wider points to the highest and second-highest foreground group scores, "
             "each bounded by the highest isolated-background score. Keep exact ties below the point.",
    "outcomes": "Count useful locations, ordinary and isolated-background flags, unassessed files and ordinary control "
                "intervals that independently locate the corresponding true edit on its source clock.",
}


def fits():
    return {
        "early_layer": SOURCE.read(LAYER_FIT)["layer4"]["cases"],
        "boundary": SOURCE.read(BOUNDARY_FIT)["cases"],
        "matched_rank": [item for item in SOURCE.read(RANK_FIT)["cases"] if item["arm"] == "clock"],
    }


def verified_readings(rows, scores, point):
    detector.verdicts(rows, scores, point)
    return [item["best"] for item in scores]


def source_geometry(base, fitted):
    plan = base["sources"]
    detector.validate_plan(plan)
    SOURCE.check(all({case["trained_family"] for case in cases} == set(FAMILIES)
                     and {case["held_family"] for case in cases} == set(FAMILIES)
                     for cases in fitted.values()), "Missing trained or held generator role")
    established = [row for row in plan["rows"] if row["role"] == "held" and row["condition"] == "competing_speech"]
    baseline = {}
    for row, current in zip(established, base["incumbent"]["results"], strict=True):
        SOURCE.check(row["pcm_sha256"] == current["pcm_sha256"], "App check read a different recording")
        baseline[row["pcm_sha256"]] = current
    calibration, held, original_points = {}, {}, {}
    for trained, tested in (FAMILIES, FAMILIES[::-1]):
        calibrators = detector.relevant(plan, "calibration", trained, calibration=True)
        test = detector.relevant(plan, "held", tested)
        calibration[trained] = [row | {"views": {}} for row in calibrators]
        held[trained] = [row | {"views": {}} for row in test]
        original_points[trained] = {}
        for name, cases in fitted.items():
            case, = [item for item in cases if item["trained_family"] == trained and item["held_family"] == tested]
            points = detector.scores_at_points(calibrators, case["calibration_scores"])
            SOURCE.check(points == case["points"], "Original operating points differ")
            original_points[trained][name] = points
            for rows, scored in ((calibration[trained], case["calibration_scores"]), (held[trained], case["held_scores"])):
                for row, item in zip(rows, verified_readings(rows, scored, points["strict"]), strict=True):
                    row["views"][name] = item
    return calibration, held, original_points, baseline


def overlap(first, second):
    left, right = first["range"], second["range"]
    return 2 * (min(left[1], right[1]) - max(left[0], right[0])) >= max(left[1] - left[0], right[1] - right[0])


def proposal_pairs(views):
    for (first_name, first), (second_name, second) in combinations(views.items(), 2):
        if first is None or second is None or not overlap(first, second):
            continue
        chosen = min((first["range"], second["range"]), key=lambda span: (span[1] - span[0], span))
        yield first_name, second_name, chosen


def group_reference(rows):
    maxima = {view: {} for view in READERS}
    for row in rows:
        for view, item in row["views"].items():
            if item is not None:
                previous = maxima[view].get(row["group"], item["score"])
                maxima[view][row["group"]] = max(previous, item["score"])
    return maxima


def ranked_pairs(views, reference, *, exclude=None):
    ranks = {}
    for name, values in reference.items():
        if views[name] is not None:
            distribution = sorted(value for group, value in values.items() if group != exclude)
            # Keep one rank beyond the largest reference value and leave zero
            # for responses below or equal to every reference group maximum.
            ranks[name] = 10000 * bisect_left(distribution, views[name]["score"]) // (len(distribution) + 1)
    return [(min(ranks[first], ranks[second]), proposal) for first, second, proposal in proposal_pairs(views)]


def joint_cutoffs(rows, reference):
    ordinary, isolated = {}, []
    for row in rows:
        pairs = ranked_pairs(row["views"], reference, exclude=row["group"])
        score = max((rank for rank, _ in pairs), default=0)
        if row["kind"] == "background":
            isolated.append(score)
        else:
            ordinary[row["group"]] = max(score, ordinary.get(row["group"], 0))
    ranked = sorted(ordinary.values(), reverse=True)
    SOURCE.check(len(ranked) >= 2 and isolated, "Missing ordinary or isolated calibration groups")
    return {label: max(point, max(isolated)) for label, point in zip(LEVELS, ranked, strict=False)}


def decisions(views, original, reference, joint):
    policies = {}
    pairs = list(proposal_pairs(views))
    joint_pairs = ranked_pairs(views, reference)
    for label in LEVELS:
        flags = {name for name, item in views.items() if item is not None and item["score"] > original[name][label]}
        for name in READERS:
            policies[f"{name}/{label}"] = [views[name]["range"]] if name in flags else []
        policies[f"union/{label}"] = [views[name]["range"] for name in views if name in flags]
        policies[f"agreement/{label}"] = [span for first, second, span in pairs if {first, second} <= flags]
        supported = [(rank, span) for rank, span in joint_pairs if rank > joint[label]]
        # The cutoff is fitted to the strongest pair per submission.
        policies[f"joint/{label}"] = [min(supported, key=lambda pair: (-pair[0], pair[1]))[1]] if supported else []
    return policies


def project(row, incumbent, assessed, fit_views, references, points, joint):
    flagged = defaultdict(set)
    coverage = set()
    for family, views in fit_views.items():
        available = {name for name, item in views.items() if item is not None}
        coverage.update(available)
        if available:
            coverage.add("union")
        if len(available) >= 2:
            coverage.update(("agreement", "joint"))
        for arm, spans in decisions(views, points[family], references[family], joint[family]).items():
            flagged[arm].update(tuple(span) for span in spans)
    return {key: row[key] for key in ("group", "kind", "condition", "pcm_sha256", "truth")} | {
        "incumbent": incumbent, "incumbent_assessed": assessed, "assessed_supplements": sorted(coverage),
        "additions": {arm: [list(span) for span in sorted(spans)] for arm, spans in flagged.items()},
    }


def primary(held, baseline, references, points, joint):
    for family, rows in held.items():
        for row in rows:
            if row["condition"] == "competing_speech":
                current = baseline[row["pcm_sha256"]]
                yield project(row, current["selected"], current["assessed"], {family: row["views"]}, references, points, joint)


def transfer_views(layer_rows, edge_rows, rank_rows, points):
    for layer, edges, rank in zip(layer_rows, edge_rows, rank_rows, strict=True):
        witness = {key: value for key, value in layer.items() if key != "heads"}
        SOURCE.check(all({key: value for key, value in row.items() if key != "heads"} == witness for row in (edges, rank)),
                     "Checks were measured on different PCM or source roles")
        views = {}
        for family in FAMILIES:
            items = {"early_layer": layer["heads"][f"layer4/{family}"],
                     "boundary": edges["heads"][family], "matched_rank": rank["heads"][f"clock-{family}"]}
            for name, item in items.items():
                verified_readings([layer], [{"pcm_sha256": layer["pcm_sha256"], "best": item}], points[family][name]["strict"])
            views[family] = items
        yield witness | {"kind": witness["arm"]}, views


def transferred(points, references, joint):
    recorded = SOURCE.read(LAYER_TRANSFER)
    layer_rows = recorded["rows"]
    archive = [SOURCE.read(BOUNDARY_TRANSFER), SOURCE.read(RANK_TRANSFER)]
    SOURCE.check(recorded["fit_sha256"] == SOURCE.path_sha(LAYER_FIT)
                 and [item["fit_sha256"] for item in archive] == [SOURCE.path_sha(BOUNDARY_FIT), SOURCE.path_sha(RANK_FIT)],
                 "Transfer fit commitments differ")
    layer_points = {f"layer4/{family}": points[family]["early_layer"] for family in FAMILIES}
    SOURCE.check(all(recorded["points"][name] == value for name, value in layer_points.items())
                 and archive[0]["points"] == {family: points[family]["boundary"] for family in FAMILIES}
                 and all(archive[1]["points"][f"clock-{family}"] == points[family]["matched_rank"] for family in FAMILIES),
                 "Transfer research points differ")
    # Replay the existing source and role checks from the prior comparison.
    ranking.transfer_scores(layer_rows, recorded["points"])
    rooms = SOURCE.read(ranking.ROOM_BASELINE)
    stock = SOURCE.read(ranking.STOCK_BASELINE)
    current = {row["pcm_sha256"]: (row["incumbent"]["range"], row["incumbent"]["assessed"]) for row in rooms["rows"]}
    for row in stock["rows"]:
        response = ranking.stock.strongest(row, range(len(stock["windows"])))
        span = stock["windows"][-response[1]] if response and response[0] > stock["points"]["existing_short"] else None
        current[row["pcm_sha256"]] = span, response is not None
    for row, views in transfer_views(layer_rows, archive[0]["rows"], archive[1]["rows"], points):
        yield row["bank"], project(row, *current.pop(row["pcm_sha256"]), views, references, points, joint)
    SOURCE.check(not current, "The transfer comparison omitted recordings")


def random_clock(points, references, joint, path):
    layer_bank, extra = SOURCE.read(RANDOM_LAYERS), SOURCE.read(path)
    SOURCE.check(extra["schema"] == "chipmunk.audio-short-edit-agreement-random/1"
                 and extra["fit_sha256"] == {"boundary": SOURCE.path_sha(BOUNDARY_FIT), "matched_rank": SOURCE.path_sha(RANK_FIT)}
                 and layer_bank["fit_sha256"] == SOURCE.path_sha(LAYER_FIT), "Random clock fit commitments differ")
    census = SOURCE.read(layers.RANDOM_ROSTER)
    SOURCE.check(extra["source_sha256"] == SOURCE.path_sha(layers.RANDOM_ROSTER), "Random clock roster differs")
    SOURCE.check(all(layer_bank["points"][f"layer4/{family}"] == points[family]["early_layer"] for family in FAMILIES),
                 "Random-clock research points differ")
    layers.random_results(layer_bank["rows"], layer_bank["points"])
    extra_rows = extra["rows"]
    words = [{key: value for key, value in row.items() if key != "heads"} for row in extra_rows]
    edges = [row | {"heads": item["heads"]["boundary"]} for row, item in zip(words, extra_rows, strict=True)]
    ranks = [row | {"heads": item["heads"]["matched_rank"]} for row, item in zip(words, extra_rows, strict=True)]
    for native, (row, views) in zip(census["rows"], transfer_views(layer_bank["rows"], edges, ranks, points), strict=True):
        span, _, flagged = layers.random_stock.selected(native, "existing_short")
        yield project(row, list(span) if flagged else None, span is not None, views, references, points, joint)


def summarize(rows):
    counts = defaultdict(Counter)
    false_groups = defaultdict(set)
    unique_pcm = defaultdict(set)
    ordinary_intervals = defaultdict(list)
    for row in rows:
        if row["truth"] is None:
            for arm in POLICIES:
                base = [row["incumbent"]] if row["incumbent"] else []
                ordinary_intervals[arm, row["condition"], row["group"]].extend(base + row["additions"].get(arm, []))
    for row in rows:
        kind = "edit" if row["truth"] is not None else "background" if row["kind"] in BACKGROUND else "ordinary"
        for arm in POLICIES:
            base = [row["incumbent"]] if row["incumbent"] else []
            calls = base + row["additions"].get(arm, [])
            key = row["condition"], arm, kind
            cell = counts[key]
            measured = row["incumbent_assessed"] or arm.split("/", 1)[0] in row["assessed_supplements"]
            cell.update(files=1, flagged=bool(calls), unassessed=not measured)
            unique_pcm[key].add(row["pcm_sha256"])
            if kind == "edit":
                located = any(useful(span, row["truth"]) for span in calls)
                before = any(useful(span, row["truth"]) for span in base)
                collision = any(useful(span, row["truth"]) for span in ordinary_intervals[arm, row["condition"], row["group"]])
                cell.update(useful=located, added_useful=located and not before,
                            useful_without_matched_control=located and not collision,
                            added_useful_without_matched_control=located and not before and not collision)
            elif calls:
                false_groups[key].add(row["group"])
    output = {}
    for key, cell in sorted(counts.items()):
        cell["distinct_pcm"] = len(unique_pcm[key])
        if key[-1] == "ordinary":
            cell["ordinary_groups"] = len(false_groups[key])
        output["/".join(key)] = dict(cell)
    return output


def evaluate(random_path, freeze_path=None):
    base, fitted = SOURCE.read(BASE), fits()
    calibration, held, points, baseline = source_geometry(base, fitted)
    references = {family: group_reference(rows) for family, rows in calibration.items()}
    joint = {family: joint_cutoffs(rows, references[family]) for family, rows in calibration.items()}
    saved_reference = {family: {name: {str(group): score for group, score in maxima.items()} for name, maxima in views.items()}
                       for family, views in references.items()}
    freeze = {"schema": "chipmunk.audio-short-edit-agreement/1", "method": METHOD, "rank_reference": saved_reference,
              "points": points, "joint_points": joint,
              "sources": {path.name: SOURCE.path_sha(path) for path in (
                  BASE, LAYER_FIT, RANK_FIT, BOUNDARY_FIT, LAYER_TRANSFER, RANK_TRANSFER, BOUNDARY_TRANSFER,
                  RANDOM_LAYERS, layers.RANDOM_ROSTER, ranking.ROOM_BASELINE, ranking.STOCK_BASELINE)}
              | {RANDOM_EXTRA.name: SOURCE.path_sha(random_path)}}
    if freeze_path:
        SOURCE.write(freeze_path, freeze)
    measurements = defaultdict(list, primary=list(primary(held, baseline, references, points, joint)))
    for bank, reading in transferred(points, references, joint):
        measurements[bank].append(reading)
    measurements["random_clock"] = list(random_clock(points, references, joint, random_path))
    return {"freeze": freeze, "rows": dict(measurements), "outcomes": {name: summarize(rows) for name, rows in measurements.items()}}


def capture_random(args):
    import numpy as np

    from app.detectors.audio.upgraded import SpeechActivity

    expected = SOURCE.read(layers.RANDOM_ROSTER)
    freeze = args.media / "placements-freeze.json"
    SOURCE.check(expected["native_freeze_sha256"] == SOURCE.path_sha(freeze), "Random-clock source manifest differs")
    native = SOURCE.read(freeze)
    with np.load(args.media / "placements-native.npz", allow_pickle=False) as bank:
        signal = bank["native"]
    SOURCE.check(signal.shape == (len(expected["rows"]), detector.SAMPLES) and signal.dtype == np.dtype("<i2"),
                 "Random-clock PCM geometry differs")
    SOURCE.check(SOURCE.path_sha(args.speech) == layers.frozen.SPEECH, "Speech activity asset differs")
    torch = layers.projection.runtime(3)
    heads = {"boundary": {}, "matched_rank": {}}
    fitted = fits()
    for name, root, factory in (("boundary", args.boundary, boundary.boundary_model), ("matched_rank", args.ranking, detector.model)):
        for item in fitted[name]:
            family = item["trained_family"]
            name_in_store = family if name == "boundary" else f"clock-{family}"
            heads[name][name_in_store] = layers.frozen.load_head(root, item | {"trained_family": name_in_store}, model_factory=factory)
    encoder, activity = layers.frozen.frozen_encoder(args.encoder), SpeechActivity(args.speech)
    rows = []
    with torch.inference_mode():
        for pcm, commitment, row in zip(signal, native["native_manifest_rows"], expected["rows"], strict=True):
            SOURCE.check(all(row[key] == value for key, value in commitment.items())
                         and SOURCE.digest(pcm.tobytes()) == row["pcm_sha256"], "Random-clock recording differs")
            samples = pcm.astype(np.float32) / 32768.
            features, speech = encoder(samples), activity.spans(samples)
            readings = {}
            for name, variants in heads.items():
                current = {}
                for fit, (model, (mean, scale)) in variants.items():
                    local, edge = model(torch.from_numpy(((features - mean[:, None]) / scale[:, None])[None]))[0].numpy()
                    current[fit] = detector.best_interval(local, edge, speech)
                readings[name] = current
            rows.append(layers.random_identity(row) | {"heads": readings})
            if len(rows) % 24 == 0:
                print(f"Verified and measured {len(rows)} random-clock recordings.", flush=True)
    SOURCE.write(args.output, {"schema": "chipmunk.audio-short-edit-agreement-random/1",
                               "source_sha256": SOURCE.path_sha(layers.RANDOM_ROSTER),
                               "fit_sha256": {"boundary": SOURCE.path_sha(BOUNDARY_FIT), "matched_rank": SOURCE.path_sha(RANK_FIT)}, "rows": rows})


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    sub = cli.add_subparsers(dest="phase", required=True)
    for phase, fields in (("capture-random", ("media", "boundary", "ranking", "encoder", "speech", "output")),
                          ("measure", ("receipt",)), ("review", ("receipt",))):
        command = sub.add_parser(phase)
        for field in fields:
            command.add_argument(field, type=Path)
        if phase != "capture-random":
            command.add_argument("--random", type=Path, default=RANDOM_EXTRA)
    args = cli.parse_args()
    if args.phase == "capture-random":
        capture_random(args)
    else:
        receipt = evaluate(args.random, args.receipt.with_suffix(".freeze.json") if args.phase == "measure" else None)
        if args.phase == "measure":
            encoded = (json.dumps(receipt, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
            with args.receipt.open("xb") as output:
                output.write(gzip.compress(encoded, mtime=0))
        else:
            with gzip.open(args.receipt, "rt", encoding="utf-8") as published:
                SOURCE.check(json.load(published) == receipt, "The combined audio findings do not reproduce")
        print(json.dumps(receipt["outcomes"], indent=2, sort_keys=True))
