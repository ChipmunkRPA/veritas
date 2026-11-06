"""Single-model source-matched ranking plus frozen-teacher retention; research only."""

from __future__ import annotations

import json

import numpy as np
import torch

from scripts.mouth_learner_v8 import fit_positions, frozen_identity, sha_tensor
from scripts.mouth_probe_v6 import source_weights

METHOD = "mouth-source-paired-retention/9.0"
SEED, EPOCHS, BATCH = 2718, 16, 32
POSITIVES = {"ai-redraw", "ai-retime"}
NEGATIVES = ("original", "local-light", "local-resample")


def make_pairs(rows):
    lookup = {}
    for i, row in enumerate(rows):
        key = row["group"], row["frame_index"], row["condition"]
        if key in lookup:
            raise ValueError("Repeated actor/frame/condition in immutable fitting inputs")
        lookup[key] = i
    pairs = []
    for i, row in enumerate(rows):
        if row["role"] != "fit" or row["condition"] not in POSITIVES:
            continue
        if row["family"] != "liveportrait" or row["label"] != 1:
            raise ValueError("Pair positive has changed generation provenance")
        for condition in NEGATIVES:
            index = lookup.get((row["group"], row["frame_index"], condition))
            if index is None:
                raise ValueError("Matched ordinary source/frame is missing")
            negative = rows[index]
            if (
                negative["role"] != "fit"
                or negative["label"] != 0
                or any(row[k] != negative[k] for k in ("source_sha256", "original_rgb_sha256"))
            ):
                raise ValueError("Pair roles/labels/native source identities differ")
            pairs.append((i, index))
    if not pairs:
        raise ValueError("No source-matched fitting pairs")
    return pairs


def teacher_eligible(row):
    return row["role"] == "fit" and row["condition"] not in POSITIVES


def teacher_targets(teacher, tokens, rows, bindings):
    fit_positions(rows, bindings)
    offsets = [
        v["offset"]
        for row, binding in zip(rows, bindings, strict=True)
        if teacher_eligible(row)
        for v in binding["views"]
    ]
    targets = torch.zeros(len(tokens), dtype=torch.float32)
    eligible = torch.zeros(len(tokens), dtype=torch.bool)
    device = next(teacher.parameters()).device
    teacher.eval().requires_grad_(False)
    with torch.no_grad():
        for start in range(0, len(offsets), BATCH):
            selected = offsets[start : start + BATCH]
            values = teacher(tokens[selected].to(device)).detach().cpu()
            if not torch.isfinite(values).all():
                raise ValueError("Nonfinite frozen teacher target")
            targets[selected] = values
            eligible[selected] = True
    return targets, eligible


def optimize(learner, teacher, tokens, rows, bindings, *, epochs=EPOCHS, batch=BATCH):
    fit, _ = fit_positions(rows, bindings)
    pairs = make_pairs(rows)
    targets, eligible = teacher_targets(teacher, tokens, rows, bindings)
    teacher_identity = {k: sha_tensor(p) for k, p in teacher.named_parameters()}
    prefix_identity = frozen_identity(learner)
    initial = {k: sha_tensor(p) for k, p in learner.named_parameters() if p.requires_grad}
    fit_rows = [rows[i] for i in fit]
    weights = torch.tensor(source_weights(fit_rows), dtype=torch.float32)
    labels = torch.tensor([r["label"] for r in fit_rows], dtype=torch.float32)
    device = next(learner.parameters()).device
    blocks = [p for name, p in learner.named_parameters() if p.requires_grad and name.startswith("backbone.")]
    optimizer = torch.optim.AdamW(
        [{"params": blocks, "lr": 5e-5}, {"params": list(learner.head.parameters()), "lr": 1e-4}],
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
    )
    learner.eval()
    history = []
    for epoch in range(epochs):
        order = torch.randperm(len(fit), generator=torch.Generator().manual_seed(SEED + epoch))
        pair_order = torch.randperm(len(pairs), generator=torch.Generator().manual_seed(SEED + 1000 + epoch))
        terms = np.zeros(4, np.float64)
        steps = 0
        pair_draws = 0
        for start in range(0, len(fit), batch):
            selected = order[start : start + batch]
            indices = [fit[int(i)] for i in selected]
            offsets = [bindings[i]["views"][(epoch + i) % 4]["offset"] for i in indices]
            z = learner(tokens[offsets].to(device).detach())
            w = weights[selected].to(device)
            bce = (
                w
                * torch.nn.functional.binary_cross_entropy_with_logits(
                    z, labels[selected].to(device), reduction="none"
                )
            ).mean()
            chosen_pairs = [int(pair_order[(steps * batch + k) % len(pairs)]) for k in range(batch)]
            plus = [bindings[pairs[k][0]]["views"][(epoch + k) % 4]["offset"] for k in chosen_pairs]
            minus = [bindings[pairs[k][1]]["views"][(epoch + k) % 4]["offset"] for k in chosen_pairs]
            paired = learner(tokens[plus + minus].to(device).detach())
            ranking = torch.nn.functional.softplus(0.75 - (paired[:batch] - paired[batch:])).mean()
            mask = eligible[offsets].to(device)
            retention = (
                (w[mask] * (z[mask] - targets[offsets].to(device)[mask]).square()).mean()
                if bool(mask.any())
                else z.sum() * 0
            )
            loss = bce + 0.5 * ranking + 0.25 * retention
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite source-paired objective")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_([p for p in learner.parameters() if p.requires_grad], 1.0)
            if not torch.isfinite(norm):
                raise ValueError("Nonfinite source-paired gradient")
            optimizer.step()
            terms += [float(x.detach().cpu()) for x in (loss, bce, ranking, retention)]
            steps += 1
            pair_draws += batch
        entry = {
            "epoch": epoch + 1,
            **dict(
                zip(
                    ("total_loss", "weighted_bce", "pair_softplus", "weighted_retention_mse"),
                    (terms / steps).tolist(),
                    strict=True,
                )
            ),
            "steps": steps,
            "pair_draws": pair_draws,
        }
        history.append(entry)
        print(json.dumps(entry), flush=True)
    if (
        frozen_identity(learner) != prefix_identity
        or {k: sha_tensor(p) for k, p in teacher.named_parameters()} != teacher_identity
    ):
        raise ValueError("A frozen prefix or teacher parameter changed")
    changed = [k for k, p in learner.named_parameters() if p.requires_grad and sha_tensor(p) != initial[k]]
    if any(not any(k.startswith(f"backbone.encoder.layer.{layer}.") for k in changed) for layer in (10, 11)):
        raise ValueError("Both declared representation blocks must learn")
    return {
        "history": history,
        "frozen_parameter_identity": prefix_identity,
        "teacher_parameter_identity": [{"name": k, "sha256": v} for k, v in teacher_identity.items()],
        "teacher_target_f32_sha256": sha_tensor(targets),
        "teacher_eligible_mask_sha256": sha_tensor(eligible),
        "teacher_targets": int(eligible.sum()),
        "pairs": [{"positive_id": rows[p]["id"], "negative_id": rows[n]["id"]} for p, n in pairs],
        "unique_pairs": len(pairs),
        "fit_rows": len(fit),
        "trainable_parameters": {k: p.numel() for k, p in learner.named_parameters() if p.requires_grad},
        "changed_trainable_parameters": changed,
    }
