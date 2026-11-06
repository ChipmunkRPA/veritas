"""Train the final two small vision blocks on source-separated mouth examples."""

from __future__ import annotations

import hashlib
import math

import cv2
import numpy as np
import torch

from scripts.mouth_probe_v6 import preprocessed, source_weights

METHOD = "mouth-two-block-learning/8.0"
SEED, EPOCHS, BATCH = 1729, 12, 32


def sha_tensor(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def view(chip, index):
    if index not in (0, 1, 2, 3):
        raise ValueError("Unknown frozen fitting view")
    result = chip
    if index >= 2:
        ok, encoded = cv2.imencode(
            ".jpg", cv2.cvtColor(result, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 75]
        )
        if not ok:
            raise ValueError("Could not create fixed JPEG75 fitting view")
        result = cv2.cvtColor(cv2.imdecode(encoded, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    return np.ascontiguousarray(result[:, ::-1] if index % 2 else result)


class MouthLearner(torch.nn.Module):
    def __init__(self, backbone):
        super().__init__()
        if len(backbone.encoder.layer) != 12:
            raise ValueError("The predeclared twelve-block backbone is required")
        self.backbone = backbone.eval().requires_grad_(False)
        for layer in self.backbone.encoder.layer[10:]:
            layer.requires_grad_(True)
        self.backbone.layernorm.requires_grad_(True)
        self.head = torch.nn.Linear(backbone.config.hidden_size * 2, 1)
        generator = torch.Generator(device="cpu").manual_seed(SEED)
        with torch.no_grad():
            self.head.weight.copy_(torch.randn(self.head.weight.shape, generator=generator) * 0.01)
            self.head.bias.zero_()
        self.eval()

    def prefix(self, pixels):
        with torch.no_grad():
            values = self.backbone.embeddings(pixels)
            for layer in self.backbone.encoder.layer[:10]:
                values = layer(values)
        return values

    def representation(self, values):
        for layer in self.backbone.encoder.layer[10:]:
            values = layer(values)
        return pool(self.backbone.layernorm(values))

    def forward(self, tokens):
        return self.head(self.representation(tokens)).squeeze(-1)


def pool(tokens):
    return torch.cat(
        (
            torch.nn.functional.normalize(tokens[:, 0], dim=-1),
            torch.nn.functional.normalize(tokens[:, 1:].mean(dim=1), dim=-1),
        ),
        dim=-1,
    )


def frozen_identity(learner):
    return {k: sha_tensor(p) for k, p in learner.named_parameters() if not p.requires_grad}


def fit_positions(rows, bindings):
    if len(rows) != len(bindings) or [r["id"] for r in rows] != [r["id"] for r in bindings]:
        raise ValueError("Cache and source identities differ")
    fit = [i for i, r in enumerate(rows) if r["role"] == "fit"]
    point = [i for i, r in enumerate(rows) if r["role"] == "point"]
    if not fit or not point or any(r["role"] not in ("fit", "point") for r in rows):
        raise ValueError("Explicit fitting and point roles required")
    if {rows[i]["group"] for i in fit} & {rows[i]["group"] for i in point}:
        raise ValueError("Source actors overlap across learning and point selection")
    for i, binding in enumerate(bindings):
        expected = list(range(4)) if rows[i]["role"] == "fit" else [0]
        if [v["view"] for v in binding["views"]] != expected:
            raise ValueError("Fitting views or native-only point inputs changed")
    return fit, point


def optimize(learner, tokens, rows, bindings, *, epochs=EPOCHS, batch=BATCH):
    fit, _ = fit_positions(rows, bindings)
    fit_rows = [rows[i] for i in fit]
    weights = torch.tensor(source_weights(fit_rows), dtype=torch.float32)
    labels = torch.tensor([r["label"] for r in fit_rows], dtype=torch.float32)
    device = next(learner.parameters()).device
    initial_frozen = frozen_identity(learner)
    initial_trainable = {k: sha_tensor(p) for k, p in learner.named_parameters() if p.requires_grad}
    blocks = [p for name, p in learner.named_parameters() if p.requires_grad and name.startswith("backbone.")]
    head = list(learner.head.parameters())
    optimizer = torch.optim.AdamW(
        [{"params": blocks, "lr": 1e-4}, {"params": head, "lr": 5e-4}],
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
    )
    history = []
    learner.eval()
    for epoch in range(epochs):
        order = torch.randperm(len(fit), generator=torch.Generator().manual_seed(SEED + epoch))
        total = 0.0
        for start in range(0, len(fit), batch):
            selected = order[start : start + batch]
            offsets = [bindings[fit[int(i)]]["views"][(epoch + fit[int(i)]) % 4]["offset"] for i in selected]
            x = tokens[offsets].to(device).detach()
            y, w = labels[selected].to(device), weights[selected].to(device)
            optimizer.zero_grad(set_to_none=True)
            response = learner(x)
            loss = (
                w * torch.nn.functional.binary_cross_entropy_with_logits(response, y, reduction="none")
            ).mean()
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite learning loss")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_([p for p in learner.parameters() if p.requires_grad], 1.0)
            if not torch.isfinite(norm):
                raise ValueError("Nonfinite learned gradient")
            optimizer.step()
            total += float(loss.detach().cpu()) * len(selected)
        history.append({"epoch": epoch + 1, "weighted_bce": total / len(fit)})
        print(__import__("json").dumps(history[-1]), flush=True)
    if frozen_identity(learner) != initial_frozen:
        raise ValueError("A frozen prefix parameter changed during learning")
    final = {k: sha_tensor(p) for k, p in learner.named_parameters() if p.requires_grad}
    changed = [k for k in final if final[k] != initial_trainable[k]]
    if not any(k.startswith("backbone.encoder.layer.10.") for k in changed) or not any(
        k.startswith("backbone.encoder.layer.11.") for k in changed
    ):
        raise ValueError("Both declared representation blocks must actually learn")
    return {
        "history": history,
        "frozen_parameter_identity": initial_frozen,
        "trainable_parameters": {k: p.numel() for k, p in learner.named_parameters() if p.requires_grad},
        "changed_trainable_parameters": changed,
        "fit_rows": len(fit),
    }


def linear_head(learner):
    return {
        "coef": learner.head.weight.detach().cpu().double().numpy()[0].tolist(),
        "intercept": float(learner.head.bias.detach().cpu()[0]),
        "mean": [0.0] * learner.head.in_features,
        "scale": [1.0] * learner.head.in_features,
    }


def choose_point(rows, responses):
    if len(rows) != len(responses) or not np.isfinite(responses).all():
        raise ValueError("Complete finite point-selection responses required")
    controls = [
        float(v) for r, v in zip(rows, responses, strict=True) if r["role"] == "point" and r["label"] == 0
    ]
    if not controls:
        raise ValueError("Separate non-AI point-selection inputs required")
    return math.nextafter(max(0.0, max(controls)), math.inf)


def features(chips, dino):
    pixels = preprocessed(chips)
    with torch.inference_mode():
        tokens = dino.model(pixel_values=torch.from_numpy(pixels).to(dino.device)).last_hidden_state.float()
        values = pool(tokens).cpu().numpy()
    if values.shape != (len(chips), 768) or not np.isfinite(values).all():
        raise ValueError("Invalid learned mouth representation")
    return values, [hashlib.sha256(p.tobytes()).hexdigest() for p in pixels]
