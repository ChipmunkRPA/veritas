# SPDX-License-Identifier: MIT
"""Test background-only supervision for localization of short synthetic speech edits."""

from __future__ import annotations

import argparse
import gzip
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from scripts import audio_short_edit_room_training as room

BASE = room.base
SOURCE = room.SOURCE
REFERENCE = room.PHONEME.with_name("audio-short-edit-room-training-2026-11-15.json.gz")
SCHEMA = "chipmunk.audio-short-edit-background-training/1"
ARMS = ("continued", "background_negative", "background_margin")
ORIGINAL = "paired_centered"
EPOCHS, BATCH, SEED = 6, 8, 123941
METHOD = {
    "fit": "Start both generator directions from their exact frozen centered room weights. Keep normalization, four matched "
           "deliveries and the original paired objective. Continue for six epochs with the same minibatches and AdamW schedule. "
           "Compare continuation alone, a fractional penalty for positive content or edge scores on the exact isolated "
           "background, and the same penalty plus a margin separating synthetic content from the background's same-clock "
           "response and strongest 500 millisecond response anywhere. Background stems are used only in training.",
    "calibration": "Use original disjoint calibration readers and all ordinary conditions. Preserve the maximum and 95th "
                   "percentile points with the isolated background floor. Confirm on the other staged generator and two "
                   "stock generators, followed by the archived separate four-room transfer, with points fixed before inference.",
    "comparison": "Compare to the original centered fit, unchanged serving short windows, and equal continuation without isolated "
                  "background supervision. Score exact-sample overlap, same-clock genuine collisions, changed false alarms, and "
                  "rank the candidate edit above equivalent genuine controls independent of the cutoff.",
}


def reference():
    with gzip.open(REFERENCE, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def contract():
    return {"schema": SCHEMA, "method": METHOD, "reference_sha256": SOURCE.path_sha(REFERENCE),
            "seed": SEED, "epochs": EPOCHS, "batch": BATCH,
            "learning_rate": .0008, "weight_decay": .001, "background_penalty": .45, "background_margin": .35}


def previous(record, family, section):
    return next(row for row in record[section]["trials"] if row["family"] == family and row["arm"] == ORIGINAL)


def check_plan(inputs, known):
    plan = SOURCE.read(inputs / "freeze.json")
    room.validate(plan)
    SOURCE.check(plan == known["plan"], "Native room plan differs from the published reference")
    return plan


def prepare_fit(plan, family, extract, mean, scale):
    import numpy as np
    import torch

    features, intervals, boundaries = room.training_input(plan, family, ORIGINAL, extract)
    groups = [group for group in plan["groups"] if group["role"] == "fit"]
    stems = [group["observations"]["competing"]["background"] for group in groups]
    SOURCE.check(len(set(stems)) == len(groups) == len(features) == 48, "Fitting background pool differs")
    background = np.stack([extract(digest, True)[0] for digest in stems])
    return tuple(torch.from_numpy(value) for value in ((features - mean[:, None]) / scale[:, None], intervals, boundaries,
                                                        (background - mean[:, None]) / scale[:, None]))


def prior_loss(values, edges, edit, boundaries):
    import torch
    from torch.nn import functional as F

    target = torch.zeros_like(values)
    target[:, :, -1:, :] = edit
    borders = torch.zeros_like(edges)
    borders[:, :, -1:, :] = boundaries
    score = F.binary_cross_entropy_with_logits(values, target, pos_weight=torch.tensor(8.))
    score += .25 * F.binary_cross_entropy_with_logits(edges, borders, pos_weight=torch.tensor(20.))
    local = (values * edit).sum(dim=-1) / edit.sum(dim=-1)
    ranks = local[:, :, -1:] - local[:, :, 1:3]
    score += .35 * F.softplus(-ranks).mean()
    score += .12 * F.smooth_l1_loss(ranks[:, 1:], ranks[:, :1].expand_as(ranks[:, 1:]))
    return score, local[:, -1, -1]


def background_loss(model, stem, edit, synthetic, *, relative):
    from torch.nn import functional as F

    speech_logit, edge_logit = model(stem).unbind(dim=1)
    loss = .45 * (F.softplus(speech_logit).mean() + .25 * F.softplus(edge_logit).mean())
    if relative:
        in_range = edit[:, 0, 0, :]
        same_clock = (speech_logit * in_range).sum(dim=-1) / in_range.sum(dim=-1)
        highest = F.avg_pool1d(speech_logit[:, None, :], kernel_size=50, stride=10)[:, 0].max(dim=-1).values
        loss += .35 * (F.softplus(1. + same_clock - synthetic) + F.softplus(1. + highest - synthetic)).mean() * .5
    return loss


def continue_fit(model, data, arm):
    import torch

    samples, intervals, endpoints, background = data
    generator = torch.Generator().manual_seed(SEED)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0008, weight_decay=.001)
    trace = []
    for epoch in range(EPOCHS):
        total = 0.
        for selection in torch.randperm(len(samples), generator=generator).split(BATCH):
            batch = samples[selection]
            regions = intervals[selection, None, None, :]
            borders = endpoints[selection, None, None, :]
            content, edges = model(batch.flatten(0, 2)).reshape(*batch.shape[:3], 2, BASE.FRAMES).unbind(dim=3)
            loss, noisy_synthetic = prior_loss(content, edges, regions, borders)
            if arm != "continued":
                loss += background_loss(model, background[selection], regions, noisy_synthetic, relative=arm == "background_margin")
            SOURCE.check(math.isfinite(float(loss.detach())), "Short-edit optimization diverged")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(selection)
        trace.append(round(total / len(samples), 8))
        print(f"{arm}: epoch {epoch+1}/{EPOCHS}, mean objective {trace[-1]:.5f}", flush=True)
    return model.eval(), trace


def fit(args):
    import torch
    from safetensors.torch import save_file

    known = reference()
    plan = check_plan(args.inputs, known)
    SOURCE.check(SOURCE.read(args.initial / "fit.json") == known["fit"], "Initial fit receipt differs")
    extract = room.feature_inputs(args.inputs, room.runtime(args.speech))
    args.output.mkdir(mode=0o700)
    lock = {"contract": contract(), "plan_sha256": room.plan_hash(plan), "trials": []}
    for family in room.FAMILIES:
        parent = previous(known, family, "fit")
        cases = list(room.trial_cases(plan, "calibration", family))
        initial, mean, scale = room.load_trial(args.initial, parent)
        features = prepare_fit(plan, family, extract, mean, scale)
        for arm in ARMS:
            print(f"Fitting the {family} fold / {arm}", flush=True)
            model = BASE.model()
            model.load_state_dict(initial.state_dict(), strict=True)
            trained, loss = continue_fit(model, features, arm)
            path = args.output / f"{family}-{arm}.safetensors"
            save_file({**trained.state_dict(), "feature_mean": torch.from_numpy(mean), "feature_scale": torch.from_numpy(scale)}, str(path))
            calibration = room.score_rows(trained, mean, scale, cases, ORIGINAL, extract)
            lock["trials"].append({"family": family, "arm": arm, "initial_sha256": parent["weights_sha256"],
                "weights_sha256": SOURCE.path_sha(path), "loss": loss, "calibration": calibration,
                "points": BASE.scores_at_points(cases, calibration)})
    SOURCE.write(args.output / "fit.json", lock)
    print("Sealed all six fits and their ordinary-only points", flush=True)


def check_fit(lock, known):
    plan = known["plan"]
    SOURCE.check(lock["contract"] == contract() and lock["plan_sha256"] == room.plan_hash(plan), "Fitting contract changed")
    SOURCE.check([(row["family"], row["arm"]) for row in lock["trials"]] ==
                 [(family, arm) for family in room.FAMILIES for arm in ARMS], "Fitting inventory changed")
    for row in lock["trials"]:
        cases = list(room.trial_cases(plan, "calibration", row["family"]))
        SOURCE.check(row["initial_sha256"] == previous(known, row["family"], "fit")["weights_sha256"]
                     and len(row["loss"]) == EPOCHS and all(math.isfinite(value) for value in row["loss"])
                     and row["points"] == BASE.scores_at_points(cases, row["calibration"]), "Unsealed training or calibration")
        BASE.verdicts(cases, row["calibration"], row["points"]["strict"])


def measure(args):
    known = reference()
    plan = check_plan(args.inputs, known)
    lock = SOURCE.read(args.fit / "fit.json")
    check_fit(lock, known)
    transfer = SOURCE.read(args.recorded / "freeze.json")
    SOURCE.check(transfer["plan_sha256"] == room.plan_hash(plan)
                 and transfer["transfer_sha256"] == SOURCE.path_sha(room.ROOM_TRANSFER), "Recorded-room lineage differs")
    used = {digest for row in plan["groups"] for view in row["observations"].values() for digest in view.values()}
    transfer_cases = list(room.transfer_cases())
    SOURCE.check(not any(case["pcm_sha256"] in used for case in transfer_cases), "Recorded room PCM overlaps a primary role")
    speech = room.runtime(args.speech)
    main = room.feature_inputs(args.inputs, speech)
    extra = room.feature_inputs(args.recorded, speech)
    args.output.mkdir(mode=0o700)
    witness = {"fit_sha256": room.plan_hash(lock), "transfer_sha256": transfer["transfer_sha256"]}
    SOURCE.write(args.output / "held-basis.json", witness)
    trials = []
    for row in lock["trials"]:
        model, mean, scale = room.load_trial(args.fit, row)
        family, arm = row["family"], row["arm"]
        cases = list(room.trial_cases(plan, "held", family))
        scores = room.score_rows(model, mean, scale, cases, ORIGINAL, main)
        recorded = room.score_rows(model, mean, scale, transfer_cases, ORIGINAL, extra)
        trials.append({"family": family, "arm": arm, "primary": scores, "recorded": recorded})
        print(f"Measured {family} / {arm} over {len(scores)} main and {len(recorded)} separate room deliveries", flush=True)
    SOURCE.write(args.output / "held.json", {**witness, "trials": trials})


def matched_rank(cases, observations):
    """Rank files' strongest proposals using only scores from matched deliveries."""
    observations = zip(cases, observations, strict=True)
    ordinary = defaultdict(list)
    edits = []
    for case, reading in observations:
        tag = room.case_key(case)[:3]
        best = reading["best"]
        score = best["score"] if best is not None else None
        if case["truth"] is not None:
            edits.append((tag, score))
        elif case["kind"] != "background":
            ordinary[tag].append(score)
    counts = defaultdict(Counter)
    for tag, score in edits:
        panel, _, condition = tag
        summary = counts[f"{panel}/{condition}"]
        summary["edits"] += 1
        if score is not None and ordinary[tag] and all(value is not None for value in ordinary[tag]):
            summary["comparable"] += 1
            summary["edit_above_every_control"] += score > max(ordinary[tag])
    return {name: dict(summary) for name, summary in sorted(counts.items())}


def evaluate(cases, observations, point):
    locations = BASE.verdicts(cases, observations, point)
    outcomes = room.summarize(cases, [reading["best"] for reading in observations], locations)
    return locations, outcomes


def archive_held(known, family, kind):
    section = "held" if kind == "primary" else "transfer"
    return previous(known, family, section)["scores"]


def report(lock, observed):
    known = reference()
    plan = known["plan"]
    room.validate(plan)
    check_fit(lock, known)
    SOURCE.check(observed["fit_sha256"] == room.plan_hash(lock)
                 and observed["transfer_sha256"] == SOURCE.path_sha(room.ROOM_TRANSFER), "Held run does not use the sealed fits")
    SOURCE.check([(row["family"], row["arm"]) for row in observed["trials"]] ==
                 [(row["family"], row["arm"]) for row in lock["trials"]], "Held trial inventory differs")
    transfer_cases = list(room.transfer_cases())
    app = room.current_readings(plan, known["current"])
    result = {"ranks": {}, "points": {}, "relative": {}, "fallback": {}, "current": {}}
    current_recorded = [row["incumbent"] for row in SOURCE.read(room.ROOM_TRANSFER)["rows"]]
    result["current"]["recorded"] = room.summarize(transfer_cases,
        [True if row["assessed"] else None for row in current_recorded], [row["range"] for row in current_recorded])
    for family in room.FAMILIES:
        cases = {"primary": list(room.trial_cases(plan, "held", family)), "recorded": transfer_cases}
        paired = [index for index, row in enumerate(cases["primary"]) if row["condition"] == "competing"]
        app_cases = [cases["primary"][index] for index in paired]
        app_choices = [app[room.case_key(row)][0] for row in app_cases]
        app_assessed = [app[room.case_key(row)][1] for row in app_cases]
        result["current"][family] = room.summarize(app_cases, [True if value else None for value in app_assessed], app_choices)
        parent = previous(known, family, "fit")
        saved = {kind: archive_held(known, family, kind) for kind in cases}
        fit_rows = [row for row in lock["trials"] if row["family"] == family]
        held_rows = [row for row in observed["trials"] if row["family"] == family]
        comparison_points = {}
        for fitted, held in zip([parent, *fit_rows], [saved, *held_rows], strict=True):
            arm = fitted["arm"]
            for kind, samples in cases.items():
                raw = held[kind]
                results_key = f"{family}/{arm}/{kind}"
                result["ranks"][results_key] = matched_rank(samples, raw)
                for level, point in fitted["points"].items():
                    key = f"{results_key}/{level}"
                    locations, result["points"][key] = evaluate(samples, raw, point)
                    comparison_points[arm, kind, level] = locations
                    if arm != ORIGINAL:
                        result["relative"][key] = {"frozen": room.differences(samples,
                            comparison_points[ORIGINAL, kind, level], locations)}
                        if arm != "continued":
                            result["relative"][key]["continuation"] = room.differences(samples,
                                comparison_points["continued", kind, level], locations)
                    if kind == "primary":
                        scoped = [locations[i] for i in paired]
                        merged = [before if before is not None else new for before, new in zip(app_choices, scoped, strict=True)]
                        measured = [True if had or raw[i]["best"] is not None else None for i, had in zip(paired, app_assessed, strict=True)]
                        result["fallback"][key] = room.summarize(app_cases, measured, merged)
    return result


def combine(args):
    fitted, measured = SOURCE.read(args.fit), SOURCE.read(args.held)
    summary = report(fitted, measured)
    payload = {"fit": fitted, "held": measured, "result": summary}
    packet = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    with args.output.open("xb") as stream:
        stream.write(gzip.compress(packet, mtime=0))


def review(args):
    with gzip.open(args.receipt, "rt", encoding="utf-8") as stream:
        payload = json.load(stream)
    summary = report(payload["fit"], payload["held"])
    SOURCE.check(summary == payload["result"], "Recorded comparison differs from independent replay")
    print(f"Verified {len(payload['fit']['trials'])} sealed fits and {len(summary['points'])} outcome groups.")
    for key, rows in summary["points"].items():
        for condition, count in rows.items():
            if condition.endswith("/competing"):
                print(key, condition, count)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    modes = cli.add_subparsers(dest="command", required=True)
    for name, fields in (("fit", ("inputs", "initial", "speech", "output")),
                         ("measure", ("inputs", "fit", "recorded", "speech", "output")),
                         ("combine", ("fit", "held", "output")), ("review", ("receipt",))):
        parser = modes.add_parser(name)
        for field in fields:
            parser.add_argument(field, type=Path)
    args = cli.parse_args()
    {"fit": fit, "measure": measure, "combine": combine, "review": review}[args.command](args)


if __name__ == "__main__":
    main()
