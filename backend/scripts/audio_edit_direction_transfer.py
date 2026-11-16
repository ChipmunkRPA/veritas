# SPDX-License-Identifier: MIT
"""Compare entry and exit evidence for short speech edits under competing voices."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from scripts import audio_clean_teacher_transfer as prior

base, features, panels, SOURCE = prior.base, prior.features, prior.panels, prior.SOURCE
SCHEMA = "chipmunk.audio-edit-direction-transfer/1"
FEATURES = ("unsigned", "signed")
READOUTS = ("mean", "both")
METHOD = {
    "fit": "Use the earlier content model with independent entry and exit heads. Both train on the same "
           "48 paired dry and noisy prompts per direction, including ordinary splices and resampling. "
           "Only the respective boundary is positive within 30 ms. The two controls use the same parameter "
           "counts and losses, reading absolute or signed feature changes across 80 ms.",
    "selection": "Use the same speech support and timing grid. Within 40 ms of the proposed start and end, "
                 "take the best respective entry and exit readings. Add 0.15 times their mean or minimum "
                 "to the mean content reading. Calibrate each score on ordinary files and isolated backgrounds.",
    "transfer": "Fit each direction on one generator and assess the other. Use both fits separately for the "
                "external speakers and rooms. A useful range covers at least half of a labeled edit and "
                "has 0.30 intersection over union. Preserve the app's existing range when testing a fallback.",
}


def model(feature):
    import torch
    from torch import nn
    from torch.nn import functional as F

    class BoundaryHeads(nn.Module):
        def __init__(self):
            super().__init__()
            trunk = base.model()[:-1]
            self.trunk = trunk
            self.content = nn.Conv1d(16, 1, 1)
            self.boundaries = nn.Conv1d(16, 2, 1)

        def forward(self, tensor):
            state = self.trunk(tensor)
            padded = F.pad(state, (4, 4), mode="replicate")
            change = padded[..., 8:] - padded[..., :-8]
            bounds = self.boundaries(change if feature == "signed" else change.abs())
            return torch.cat((self.content(state), bounds), dim=1)

    return BoundaryHeads()


def train(data, plan, family, feature, feature_of):
    import numpy as np
    import torch
    from torch.nn import functional as F

    values, region, _ = base.fit_input(data, plan, family, feature_of)
    axes = (0, 1, 2, 4)
    center = values.mean(axis=axes, dtype=np.float64).astype(np.float32)
    scale = np.maximum(.001, values.std(axis=axes, dtype=np.float64)).astype(np.float32)
    values = torch.from_numpy((values - center[:, None]) / scale[:, None])
    clock = np.arange(base.FRAMES) * base.HOP
    boundaries = np.array([np.abs(clock - np.array(source["truth"])[:, None]) <= 480
                           for source in plan["sources"] if source["role"] == "fit"], dtype=np.float32)
    region, boundaries = torch.from_numpy(region), torch.from_numpy(boundaries)
    torch.manual_seed(base.SEED)
    classifier = model(feature).train()
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=.002, weight_decay=.001)
    losses = []
    for _ in range(base.EPOCHS):
        batch_losses = []
        for indices in torch.randperm(len(values)).split(6):
            batch = values[indices]
            prediction = classifier(batch.flatten(0, 2)).reshape(*batch.shape[:3], 3, base.FRAMES)
            content = prediction[..., 0, :]
            edges = prediction[..., 1:, :]
            interval = region[indices, None, None, :]
            labels = torch.zeros_like(content)
            labels[:, :, -1:, :] = interval
            boundary_labels = torch.zeros_like(edges)
            boundary_labels[:, :, -1, :, :] = boundaries[indices, None, :, :]
            content_loss = F.binary_cross_entropy_with_logits(content, labels, pos_weight=torch.tensor(8.))
            boundary_loss = F.binary_cross_entropy_with_logits(edges, boundary_labels, pos_weight=torch.tensor(40.))
            means = (content * interval).sum(dim=-1) / interval.sum(dim=-1)
            margins = means[:, :, -1:] - means[:, :, 1:3]
            loss = (content_loss + .25 * boundary_loss + .35 * F.softplus(-margins).mean()
                    + .12 * F.smooth_l1_loss(margins[:, 0, :], margins[:, 1, :]))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.detach()))
        losses.append(round(sum(batch_losses) / len(batch_losses), 8))
    return classifier.eval(), center, scale, losses


def eligible(spans):
    from app.detectors.audio.content import window_in_speech

    return [(start, start + length) for length in base.LENGTHS
            for start in range(0, base.FRAMES - length + 1, base.STRIDE)
            if window_in_speech(spans, start * base.HOP / base.RATE, (start + length) * base.HOP / base.RATE, fraction=.5)]


def pick(logits, ranges):
    import numpy as np

    if not ranges:
        return dict.fromkeys(READOUTS)
    content, enter, leave = logits
    if not np.isfinite(logits).all():
        raise ValueError("Nonfinite speech edit responses")
    best = dict.fromkeys(READOUTS)
    entry = {start: float(max(enter[max(0, start-4):min(base.FRAMES, start+5)])) for start, _ in ranges}
    exit_ = {end: float(max(leave[max(0, end-4):min(base.FRAMES, end+5)])) for _, end in ranges}
    for a, b in ranges:
        center = float(content[a:b].mean())
        for name, value in zip(READOUTS, ((entry[a] + exit_[b]) / 2, min(entry[a], exit_[b])), strict=True):
            candidate = (center + .15 * value, -a, -b)
            if best[name] is None or candidate > best[name]:
                best[name] = candidate
    return {key: {"score": round(score, 8), "range": [-a * base.HOP, -b * base.HOP]}
            for key, (score, a, b) in best.items()}


def measure(head, raw, ranges):
    import torch

    _, classifier, center, scale = head
    with torch.inference_mode():
        values = torch.from_numpy(((raw - center[:, None]) / scale[:, None])[None])
        return pick(classifier(values)[0].numpy(), ranges)


def fit(args):
    import torch
    from safetensors.torch import save_file

    plan, digest = base.load_plan(args.data)
    ledger, feature_of = features.feature_reader(args.cache, plan, digest)
    activity = prior.runtime(args.speech)
    freeze = {"schema": SCHEMA, "method": METHOD, "features": FEATURES, "readouts": READOUTS,
              "source_sha256": digest, "encoder": ledger["encoder"], "code_sha256": SOURCE.path_sha(Path(__file__)),
              "feature_ledger_sha256": SOURCE.path_sha(args.cache / "features.json"),
              "held_sha256": {"mixed": SOURCE.path_sha(args.mixed), "fresh": SOURCE.path_sha(args.fresh / "freeze.json"),
                              "room": SOURCE.path_sha(args.room / "freeze.json")}}
    args.output.mkdir(mode=0o700)
    prior.save(args.output / "freeze.json", freeze)
    heads = []
    for family, unknown in (base.sources.FAMILIES, base.sources.FAMILIES[::-1]):
        ordinary = prior._rows(plan, "calibration", family, calibration=True)
        spans = {row["pcm_sha256"]: eligible(activity.spans(base.pcm(args.data, row))) for row in ordinary}
        for mode in FEATURES:
            trained, center, scale, losses = train(args.data, plan, family, mode, feature_of)
            path = args.output / f"{family}-{mode}.safetensors"
            values = trained.state_dict() | {"feature_mean": torch.from_numpy(center), "feature_scale": torch.from_numpy(scale)}
            save_file(values, str(path))
            measured = [measure((None, trained, center, scale), feature_of(args.data, row), spans[row["pcm_sha256"]])
                        for row in ordinary]
            calibration = {name: prior._attach(ordinary, [{"best": case[name]} for case in measured]) for name in READOUTS}
            points = {name: base.scores_at_points(ordinary, results) for name, results in calibration.items()}
            heads.append({"trained_family": family, "held_family": unknown, "feature": mode, "file": path.name,
                          "sha256": SOURCE.path_sha(path), "losses": losses, "calibration": calibration, "points": points})
            print(f"Frozen {family} {mode} for {unknown} and the external test panels.", flush=True)
    prior.save(args.output / "heads.json", heads)


def load_head(path, seal):
    from safetensors.torch import load_file

    path = path / seal["file"]
    assert SOURCE.path_sha(path) == seal["sha256"]
    params = load_file(str(path))
    center, scale = (params.pop(key).numpy() for key in ("feature_mean", "feature_scale"))
    classifier = model(seal["feature"]).eval()
    classifier.load_state_dict(params, strict=True)
    return seal, classifier, center, scale


def independent_inputs(path, receipt, expected):
    freeze_file = path / "freeze.json"
    parent = prior.read(receipt)
    assert SOURCE.path_sha(freeze_file) == expected and prior.read(freeze_file) == parent["freeze"]
    for row in parent["rows"]:
        yield row, base.pcm(path, row), row["current"]


def score(args):
    plan, digest = base.load_plan(args.data)
    ledger, feature_of = features.feature_reader(args.cache, plan, digest)
    freeze = prior.read(args.fit / "freeze.json")
    seals = prior.read(args.fit / "heads.json")
    assert freeze["schema"] == SCHEMA and freeze["method"] == METHOD and freeze["source_sha256"] == digest
    assert freeze["encoder"] == ledger["encoder"] and SOURCE.path_sha(Path(__file__)) == freeze["code_sha256"]
    assert SOURCE.path_sha(args.cache / "features.json") == freeze["feature_ledger_sha256"]
    assert SOURCE.path_sha(args.mixed) == freeze["held_sha256"]["mixed"]
    combined = prior.read(args.mixed)
    exact = {(row["panel"], row["pcm_sha256"]): row["current"] for row in combined["rows"]}
    heads = [load_head(args.fit, seal) for seal in seals]
    activity = prior.runtime(args.speech)
    encode = features.frozen_encoder(args.encoder)
    known = {row["pcm_sha256"]: row for row in plan["rows"]}
    rows, cached = [], {}

    def scan(inputs):
        for row, wave, saved in inputs:
            digest = row["pcm_sha256"]
            if digest not in cached:
                raw = feature_of(args.data, known[digest]) if digest in known else encode(wave)
                support = eligible(activity.spans(wave))
                cached[digest] = {f"{seal['trained_family']}/{seal['feature']}_{rule}": result
                                  for seal, *rest in heads
                                  for rule, result in measure((seal, *rest), raw, support).items()}
            current = exact.get((row["panel"], digest), saved)
            assert current is not None and (saved is None or saved == current)
            rows.append({key: row[key] for key in ("pcm_sha256", "panel", "room", "group", "kind", "truth")}
                        | {"current": current, "candidates": cached[digest]})
            if len(rows) % 40 == 0:
                print(f"Measured {len(rows)} held recordings.", flush=True)

    scan(panels.inputs(args, plan, SOURCE.read(panels.PRIMARY), prior.read(panels.SINE)))
    assert {(row["panel"], row["pcm_sha256"]) for row in rows} == exact.keys()
    for name in ("fresh", "room"):
        path = args.followup if name == "fresh" else args.room
        scan(independent_inputs(path, getattr(args, f"{name}_receipt"), freeze["held_sha256"][name]))
    packet = {"freeze": freeze, "heads": seals, "rows": rows}
    packet["comparison"] = compare(packet)
    prior.save(args.output, packet)
    print(f"Recounted {len(rows)} held recordings.", flush=True)


def compare(packet):
    assert packet["freeze"]["schema"] == SCHEMA and packet["freeze"]["method"] == METHOD
    groups = defaultdict(list)
    for row in packet["rows"]:
        groups[row["panel"]].append(row)
    result = {}
    for head in packet["heads"]:
        family, other, mode = (head[key] for key in ("trained_family", "held_family", "feature"))
        for readout in READOUTS:
            calibration = head["calibration"][readout]
            assert base.scores_at_points([row | {"truth": None} for row in calibration], calibration) == head["points"][readout]
            arm = f"{mode}_{readout}"
            for panel, observed in sorted(groups.items()):
                if panel == "phoneme":
                    observed = [row for row in observed if row["kind"] in
                                ("untouched", "genuine_splice", "background", f"edited/{other}", f"resampled/{other}")]
                counts = result.setdefault(panel, {}).setdefault(family, {})
                counts["current"] = prior._summarize(observed, "current", family, None)
                counts[arm] = {name: {"alone": prior._summarize(observed, arm, family, point),
                                      "fallback": prior._summarize(observed, arm, family, point, fallback=True)}
                               for name, point in head["points"][readout].items()}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    phases = parser.add_subparsers(dest="phase", required=True)
    fit_parser = phases.add_parser("fit")
    for name in ("data", "cache", "speech", "mixed", "fresh", "room", "output"):
        fit_parser.add_argument(name, type=Path)
    score_parser = phases.add_parser("score")
    for name in ("data", "cache", "speech", "encoder", "mixed", "retained", "fresh", "stereo",
                 "followup", "fresh_receipt", "room", "room_receipt", "fit", "output"):
        score_parser.add_argument(name, type=Path)
    review_parser = phases.add_parser("review")
    review_parser.add_argument("packet", type=Path)
    args = parser.parse_args()
    if args.phase == "fit":
        fit(args)
    elif args.phase == "score":
        score(args)
    else:
        data = prior.read(args.packet)
        result = compare(data)
        assert result == data["comparison"]
        print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
