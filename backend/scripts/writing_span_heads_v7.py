"""Small supervised token and interval heads; frozen external encoder stays separate."""

from __future__ import annotations

import math

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
from torch.nn import functional as F

from scripts.writing_span_geometry_v7 import absolute_interval, require

WIDTH, LATENT, QUERIES, SEED = 768, 64, 8, 20261108


class TokenHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.output = nn.Linear(WIDTH, 1)

    def forward(self, features, positions, padding=None):
        return self.output(features).squeeze(-1)


class IntervalHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.project = nn.Linear(WIDTH, LATENT)
        self.queries = nn.Parameter(torch.randn(QUERIES, LATENT) * 0.02)
        layer = nn.TransformerDecoderLayer(LATENT, 4, 128, dropout=0.0, batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(layer, 2)
        self.foreground = nn.Linear(LATENT, 2)
        self.bounds = nn.Sequential(nn.Linear(LATENT, LATENT), nn.GELU(), nn.Linear(LATENT, 2))

    def forward(self, features, positions, padding=None):
        frequencies = torch.arange(1, LATENT // 2 + 1, dtype=features.dtype, device=features.device)
        angle = positions.unsqueeze(-1) * frequencies * (2 * math.pi)
        encoding = torch.cat([angle.sin(), angle.cos()], -1)
        memory = self.project(features) + encoding
        query = self.queries.unsqueeze(0).expand(len(features), -1, -1)
        latent = self.decoder(query, memory, memory_key_padding_mask=padding)
        center, width = self.bounds(latent).sigmoid().unbind(-1)
        spans = torch.stack([(center - width / 2).clamp(0, 1), (center + width / 2).clamp(0, 1)], -1)
        return {"logits": self.foreground(latent), "spans": spans}


def build(kind):
    require(kind in ("token", "interval"), "Explicit head kind required")
    torch.manual_seed(SEED + int(kind == "interval"))
    return {"token": TokenHead, "interval": IntervalHead}[kind]()


def pair_iou(a, b, generalized=False):
    left = torch.maximum(a[:, None, 0], b[None, :, 0])
    right = torch.minimum(a[:, None, 1], b[None, :, 1])
    overlap = (right - left).clamp(min=0)
    union = (a[:, 1] - a[:, 0])[:, None] + (b[:, 1] - b[:, 0])[None, :] - overlap
    value = overlap / union.clamp(min=1e-12)
    if generalized:
        enclosing = torch.maximum(a[:, None, 1], b[None, :, 1]) - torch.minimum(a[:, None, 0], b[None, :, 0])
        value = value - (enclosing - union) / enclosing.clamp(min=1e-12)
    return value


def interval_loss(output, targets):
    logits, spans = output["logits"], output["spans"]
    require(len(logits) == len(targets), "Every training document needs targets")
    classes = torch.ones(logits.shape[:2], dtype=torch.long, device=logits.device)
    predictions, references = [], []
    for i, truth in enumerate(targets):
        require(
            truth.ndim == 2 and truth.shape[1] == 2 and len(truth) <= QUERIES,
            "Finite bounded interval target set required",
        )
        require(
            bool(torch.isfinite(truth).all())
            and bool((truth >= 0).all())
            and bool((truth <= 1).all())
            and bool((truth[:, 0] < truth[:, 1]).all()),
            "Training spans must be valid source intervals",
        )
        if not len(truth):
            continue
        cost = (
            -logits[i].softmax(-1)[:, 0, None]
            + 5 * torch.cdist(spans[i], truth, p=1)
            - 2 * pair_iou(spans[i], truth, True)
        )
        a, b = linear_sum_assignment(cost.detach().cpu().numpy())
        classes[i, a] = 0
        predictions.append(spans[i, a])
        references.append(truth[b])
    loss = F.cross_entropy(logits.flatten(0, 1), classes.flatten(), weight=logits.new_tensor([1.0, 0.1]))
    if predictions:
        a, b = torch.cat(predictions), torch.cat(references)
        loss = loss + 5 * F.l1_loss(a, b) + 2 * (1 - pair_iou(a, b, True).diag()).mean()
    return loss


def batch(cases):
    length = max(len(r["features"]) for r in cases)
    features = torch.zeros((len(cases), length, WIDTH))
    positions = torch.zeros((len(cases), length))
    padding = torch.ones((len(cases), length), dtype=torch.bool)
    labels, weights = torch.zeros((len(cases), length)), torch.zeros((len(cases), length))
    targets = []
    for i, r in enumerate(cases):
        n = len(r["features"])
        require(n > 0 and r["features"].shape == (n, WIDTH), "All native token features required")
        features[i, :n] = torch.from_numpy(r["features"])
        positions[i, :n] = torch.from_numpy(r["positions"])
        padding[i, :n] = False
        if "targets" in r:
            labels[i, :n] = torch.from_numpy(r["targets"])
            weights[i, :n] = torch.from_numpy(r["weights"])
            targets.append(
                torch.tensor(r["truth_spans"], dtype=torch.float32).reshape(-1, 2) / len(r["text"])
            )
    return features, positions, padding, labels, weights, targets


def fit(cases, kind):
    require(cases and all(r["role"] == "fit" for r in cases), "Only fitting-role features enter the learner")
    require(len({r["group"] for r in cases}) == 32, "All32 fitting groups required")
    model = build(kind)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.003 if kind == "token" else 0.0005, weight_decay=0.001
    )
    epochs = 40 if kind == "token" else 60
    generator = torch.Generator().manual_seed(SEED)
    history = []
    for epoch in range(epochs):
        order = torch.randperm(len(cases), generator=generator).tolist()
        total = 0.0
        for start in range(0, len(order), 8):
            part = [cases[i] for i in order[start : start + 8]]
            f, pos, pad, y, w, target = batch(part)
            output = model(f, pos, pad)
            if kind == "token":
                loss = (
                    (F.binary_cross_entropy_with_logits(output, y, reduction="none") * w).sum(1)
                    / w.sum(1).clamp(min=1)
                ).mean()
            else:
                loss = interval_loss(output, target)
            require(bool(torch.isfinite(loss)), "Training objective is non-finite")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            total += float(loss.detach()) * len(part)
        history.append(total / len(cases))
        if (epoch + 1) % 10 == 0:
            print({"head": kind, "epoch": epoch + 1, "fit_loss": history[-1]}, flush=True)
    model.eval().requires_grad_(False)
    return model, history


@torch.inference_mode()
def propose(model, kind, case):
    f, pos, pad, _, _, _ = batch([case])
    output = model(f, pos, pad)
    if kind == "token":
        scores = output[0].sigmoid().tolist()
        return [
            {"start": a, "end": b, "raw": float(q)}
            for (a, b), q in zip(case["offsets"], scores, strict=True)
            if b > a and any(not c.isspace() for c in case["text"][a:b])
        ]
    scores = output["logits"][0].softmax(-1)[:, 0].tolist()
    return [
        {"start": a, "end": b, "raw": float(q)}
        for span, q in zip(output["spans"][0].tolist(), scores, strict=True)
        for a, b in [absolute_interval(span, len(case["text"]))]
    ]


def save_state(model):
    return {
        k: {"shape": list(v.shape), "values": v.detach().float().flatten().tolist()}
        for k, v in model.state_dict().items()
    }


def restore(kind, state):
    model = build(kind)
    template = model.state_dict()
    require(set(template) == set(state), "Fitted head fields differ")
    tensors = {}
    for key, value in state.items():
        require(value["shape"] == list(template[key].shape), "Fitted head shape differs")
        array = np.asarray(value["values"], np.float32)
        require(array.size == template[key].numel() and np.isfinite(array).all(), "Invalid fitted values")
        tensors[key] = torch.from_numpy(array.reshape(value["shape"]).copy())
    model.load_state_dict(tensors, strict=True)
    return model.eval().requires_grad_(False)
