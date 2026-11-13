# SPDX-License-Identifier: Apache-2.0
"""Test context-only token disagreement as a native JPEG edit localization cue."""

from __future__ import annotations

import argparse
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from scripts import image_three_class_patch_fresh as fresh
from scripts import image_three_class_patch_probe as prior

SCHEMA = "chipmunk.image-masked-context-transfer/1"
ARMS = ("masked_context", "joint_context")
FLOORS = (.02, .10)
DESIGN = {
    "source_roles": "Eight ordinary-only calibration groups and twelve source-disjoint confirmation groups, from previously assessed projects.",
    "profile": "Exact matched JPEG75 native pixels, no resize. Pinned Apache-2.0 feature encoder and frozen three-class weights.",
    "mask": "Four interleaved row/column parity phases. Each token hidden in one forward and visible in the other three phases.",
    "context_score": "One minus cosine of context-only and normally visible final token, clipped to [0,2].",
    "joint_score": "Equal-weight average of median/MAD-standardized contextual score and frozen three-class log odds, clipped at ±8 per field.",
    "spread_floors": list(FLOORS),
    "proposal": prior.DESIGN["geometry"],
    "operating_point": "Strict exceedance of second-largest ordinary calibration photo maximum; no held recalibration.",
    "comparison": "Unchanged three-class head at its fixed operating point and aligned JPEG luma on identical bytes.",
    "useful": prior.DESIGN["useful"],
    "advance": {"extra_feathered_over_luma": 3, "editor_families": 2, "extra_ordinary_groups": 2, "misplaced": 4},
    "confidence_weight": 0,
}
read, write, require, sha = prior.read, prior.write, prior.require, prior.sha


def normalized(field, floor):
    middle = np.median(field)
    spread = max(floor, 1.4826 * np.median(np.abs(field - middle)))
    return np.clip((field - middle) / spread, -8, 8)


def extractor(checkpoint, head):
    import torch
    from transformers import Dinov2Model

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    model = Dinov2Model.from_pretrained(checkpoint, local_files_only=True, use_safetensors=True).cpu().eval()
    require(model.config.patch_size == prior.PATCH and model.config.use_mask_token,
            "Pinned model must support native patch masking")
    side = prior.TILES // prior.PATCH
    y, x = torch.meshgrid(torch.arange(side), torch.arange(side), indexing="ij")
    phases = (y % 2) * 2 + x % 2
    phases = [(phases == n).flatten() for n in range(4)]
    require(torch.stack(phases).sum(0).eq(1).all().item(), "Every token must be hidden in one phase")
    mean = torch.tensor([.485, .456, .406])[None, :, None, None]
    std = torch.tensor([.229, .224, .225])[None, :, None, None]
    counters = Counter()

    def normalize_batch(tiles):
        return (torch.from_numpy(tiles).permute(0, 3, 1, 2).float() / 255 - mean) / std

    @torch.inference_mode()
    def intervention_control():
        y, x = torch.meshgrid(torch.arange(prior.TILES), torch.arange(prior.TILES), indexing="ij")
        original = torch.stack([(x + c*y) % 211 for c in (1, 2, 3)], dim=-1).numpy()[None].astype(np.uint8)
        edited = original.copy()
        edited[:, 2*prior.PATCH:3*prior.PATCH, 4*prior.PATCH:5*prior.PATCH, :] = 249
        tensor = normalize_batch(np.concatenate((original, edited)))
        mask = phases[0].expand(2, -1)
        hidden = model(pixel_values=tensor, bool_masked_pos=mask).last_hidden_state[:, 1:]
        shown = model(pixel_values=tensor).last_hidden_state[:, 1:]
        j = 2*side + 4
        hidden_delta = float((hidden[0] - hidden[1]).abs().max())
        visible_delta = float((shown[0, j] - shown[1, j]).abs().max())
        require(hidden_delta < 1e-6 and visible_delta > 1e-3,
                "Masked tokens should lose access to their own source pixels")
        return {"max_hidden_delta": hidden_delta, "visible_patch_delta": visible_delta, "control_batches": 2}

    @torch.inference_mode()
    def measure(body):
        rgb = prior.base.decode(body)
        tiles, layout = prior.split_tiles(rgb)
        surprise, response = [], []
        for first in range(0, len(tiles), prior.BATCH):
            tensor = normalize_batch(tiles[first:first + prior.BATCH])
            output = model(pixel_values=tensor, output_hidden_states=True)
            visible = torch.nn.functional.normalize(output.last_hidden_state[:, 1:], dim=-1)
            tokens = torch.cat([torch.nn.functional.normalize(output.hidden_states[layer][:, 1:], dim=-1)
                                for layer in prior.LAYERS], dim=-1).numpy()
            errors = torch.zeros(visible.shape[:2], dtype=visible.dtype)
            for phase in phases:
                masked = model(pixel_values=tensor, bool_masked_pos=phase.expand(len(tensor), -1)).last_hidden_state[:, 1:]
                hidden = torch.nn.functional.normalize(masked[:, phase], dim=-1)
                errors[:, phase] = (1 - (hidden * visible[:, phase]).sum(dim=-1)).clamp(0, 2)
            surprise.append(errors[..., None].numpy())
            response.append(prior.score_map(tokens, head)[..., None])
            counters.update(visible_batches=1, masked_batches=len(phases), tiles=len(tensor))
        fields = [prior.stitch(np.concatenate(values), layout, rgb.shape)[..., 0] for values in (surprise, response)]
        require(all(np.isfinite(field).all() for field in fields), "Nonfinite native patch score")
        context, frozen = fields
        joint = (normalized(context, FLOORS[0]) + normalized(frozen, FLOORS[1])) / 2
        counters.update(deliveries=1)
        return {"masked_context": context, "joint_context": joint, "three_class": frozen}, rgb

    return measure, intervention_control(), counters


def inputs(args, lock):
    args.later = args.later_f5
    prior_cal = read(args.prior_probe / "calibration.json")
    cal = prior.views(args, lock["profile"], "calibration")
    yield "calibration", ((old, row, body, mask) for old, (row, body, mask) in zip(prior_cal, cal, strict=True))
    plan, known = fresh.planned(args)
    require(plan == read(args.output_fresh / "plan.json"), "New transfer intake changed")
    held = read(args.output_fresh / "measurements.json")
    views = (view for item, values, mask in fresh.groups(args.output_fresh, plan, known)
             for view in fresh.views(item, values, mask))
    yield "confirmation", ((old, row, body, mask) for old, (row, body, mask, _) in zip(held, views, strict=True))


def points(rows):
    maxima = {arm: defaultdict(lambda: -np.inf) for arm in ARMS}
    for row in rows:
        for arm, groups in maxima.items():
            score = row["arms"][arm]["score"]
            if score is not None:
                groups[row["source"]] = max(groups[row["source"]], score)
    require(all(len(groups) == 8 for groups in maxima.values()), "Eight ordinary calibration groups required")
    return {arm: sorted(groups.values())[-2] for arm, groups in maxima.items()}


def assess(fields, truth, ordinary_points, incumbent):
    result = {arm: prior.proposal(field) for arm, field in fields.items()}
    for arm, hit in result.items():
        hit["overlap"] = prior.overlap(truth, hit["box"])
        hit["useful"] = prior.base.geometry.useful(hit["overlap"])
        if ordinary_points is not None:
            hit["flagged"] = bool(hit["box"]) and hit["score"] > ordinary_points[arm]
    for arm in ("three_class", "current_luma"):
        if arm in incumbent:
            measured = result.get(arm)
            saved = incumbent[arm]
            if measured is not None:
                require(measured["box"] == saved["box"] and abs(measured["score"] - saved["score"]) < 1e-8,
                        "Normal forward no longer reproduces the frozen head")
            result[arm] = saved
    return result


def summarize(rows):
    by_stratum, ordinary, useful = defaultdict(Counter), defaultdict(set), defaultdict(set)
    wrong = Counter()
    for row in rows:
        role, family, source = (row[key] for key in ("role", "family", "source"))
        positive = role in prior.HELD_POSITIVE
        for arm, hit in row["arms"].items():
            found = hit["flagged"] and hit["useful"]
            by_stratum[f"{family}/{role}/{arm}"].update(
                files=1, flags=hit["flagged"], useful=found, useful_before_point=hit["useful"],
                misplaced=positive and hit["flagged"] and not found)
            if positive and role == "feathered" and found:
                useful[arm].add((family, source))
            if positive and hit["flagged"] and not found:
                wrong[arm] += 1
            if not positive and hit["flagged"]:
                ordinary[arm].add(source)
    result = {}
    limits = DESIGN["advance"]
    for arm in ARMS:
        added = useful[arm] - useful["current_luma"]
        actual = {"extra_feathered_over_luma": len(added), "editor_families": len({family for family, _ in added}),
                  "extra_ordinary_groups": len(ordinary[arm] - ordinary["current_luma"]), "misplaced": wrong[arm]}
        result[arm] = actual | {
            "extra_feathered_over_frozen_head": len(useful[arm] - useful["three_class"]),
            "lost_feathered_vs_frozen_head": len(useful["three_class"] - useful[arm]),
            "ordinary_groups": len(ordinary[arm]),
            "passes": (actual["extra_feathered_over_luma"] >= limits["extra_feathered_over_luma"]
                       and actual["editor_families"] >= limits["editor_families"]
                       and actual["extra_ordinary_groups"] <= limits["extra_ordinary_groups"]
                       and actual["misplaced"] <= limits["misplaced"]),
        }
    return {"strata": by_stratum, "outcome": result}


def run(args, lock, head, old_point):
    read_fields, controls, counters = extractor(args.checkpoint, head)
    rows, active = {}, None
    for phase, deliveries in inputs(args, lock):
        rows[phase] = []
        if phase == "confirmation":
            active = points(rows["calibration"]) | {"three_class": old_point}
            write(args.output / "points.json", active)
            print("frozen points", active, flush=True)
        for saved, key, body, truth in deliveries:
            require(all(saved[k] == v for k, v in key.items()), "Stored JPEG or mask commitment differs")
            fields, rgb = read_fields(body)
            require(truth.shape == rgb.shape[:2], "Source geometry differs")
            selected = assess(fields, truth, active, saved["arms"])
            rows[phase].append(key | {"arms": selected})
        write(args.output / f"{phase}.json", rows[phase])
        print(phase, len(rows[phase]), flush=True)
    require([len(rows[k]) for k in ("calibration", "confirmation")] == [40, 96], "Roster changed")
    return summarize(rows["confirmation"]) | {"counts": dict(counters), "intervention_control": controls, "points": active}


def review(args, lock, result):
    reports = {phase: read(args.output / f"{phase}.json") for phase in ("calibration", "confirmation")}
    counted, boxes = 0, 0
    limits = points(reports["calibration"]) | {"three_class": result["points"]["three_class"]}
    require(limits == result["points"] == read(args.output / "points.json"), "Operating point differs")
    for phase, deliveries in inputs(args, lock):
        for row, (saved, key, _, truth) in zip(reports[phase], deliveries, strict=True):
            require(all(row[k] == saved[k] == v for k, v in key.items()), "Exact source binding differs")
            counted += 1
            for arm, hit in row["arms"].items():
                actual = prior.overlap(truth, hit["box"])
                require(actual == hit["overlap"] and hit["useful"] == prior.base.geometry.useful(actual),
                        "Native overlap differs")
                if phase == "confirmation":
                    require(hit["flagged"] == (bool(hit["box"]) and (arm == "current_luma" or hit["score"] > limits[arm])),
                            "Operating point flag differs")
                boxes += 1
    require(all(result[key] == value for key, value in summarize(reports["confirmation"]).items()), "Reported counts differ")
    return {"exact_inputs_reproduced": counted, "native_overlap_and_selection_checks": boxes,
            "neural_inference_repeated": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "run", "review"))
    for key in ("archive", "prior-probe", "residual", "early-f5", "later-f5", "checkpoint", "output-fresh", "output"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    parser.add_argument("--publication", type=Path)
    args = parser.parse_args()
    args.exclude = []
    cv2.setNumThreads(1)
    lock, heads, old_points = fresh.previous(args)
    cal = args.prior_probe / "calibration.json"
    confirm = args.output_fresh / "measurements.json"
    calibration_sha = fresh.roster.sha(cal)
    confirmation_sha = fresh.roster.sha(confirm)
    require(calibration_sha == read(args.prior_probe / "result.json")["commitments"]["calibration.json"]
            and confirmation_sha == read(args.output_fresh / "results.json")["measurements_sha256"],
            "Earlier exact panel measurement differs")
    design = {"schema": SCHEMA, "design": DESIGN, "prior_probe": fresh.PREVIOUS, "checkpoint": prior.REV,
              "prior_confirmation_sha256": confirmation_sha, "prior_calibration_sha256": calibration_sha,
              "runner_sha256": fresh.roster.sha(Path(__file__))}
    freeze = args.output / "freeze.json"
    if args.stage == "freeze":
        args.output.mkdir(mode=0o700)
        write(freeze, design)
        return
    require(read(freeze) == design, "Frozen method changed")
    if args.stage == "run":
        start = time.monotonic()
        result = run(args, lock, heads["three_class"], old_points["three_class"])
        write(args.output / "result.json", design | result | {"seconds": round(time.monotonic() - start, 2),
              "journals": {key: fresh.roster.sha(args.output / f"{key}.json") for key in ("points", "calibration", "confirmation")}})
        print(result["outcome"], flush=True)
    else:
        require(args.publication is not None, "Aggregate output path required")
        result = read(args.output / "result.json")
        require(all(fresh.roster.sha(args.output / f"{key}.json") == digest for key, digest in result["journals"].items()),
                "Result journal changed")
        write(args.publication, result | {"audit": review(args, lock, result)})


if __name__ == "__main__":
    main()
