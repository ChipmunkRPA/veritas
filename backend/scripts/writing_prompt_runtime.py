# SPDX-License-Identifier: MIT
"""Fit and measure two short prompts while keeping the causal model frozen."""

from __future__ import annotations

import importlib.metadata
import os
import random
from dataclasses import dataclass

from scripts import writing_prompt_contrast as study
from scripts.writing_token_curvature import MODEL_ASSETS, REVISION, verified_model

RUNTIME = {"mlx": "0.32.4", "mlx-lm": "0.31.3", "transformers": "5.22.1"}


@dataclass(frozen=True)
class SentenceInput:
    tokens: tuple[int, ...]
    first: int
    stop: int

    @property
    def count(self):
        return self.stop - self.first


def encode(tokenizer, row, center):
    text, bounds = study.native_spans(row)
    left, start, end = bounds[center - 1][0], *bounds[center]
    material = text[left:end]
    reading = tokenizer(material, add_special_tokens=False, truncation=False, return_offsets_mapping=True)
    offsets = reading["offset_mapping"]
    pairs = [(a + left, b + left) for a, b in offsets]
    selection = [i for i, (a, b) in enumerate(pairs) if b > start and a < end and b > a]
    if not selection:
        return None
    first, stop = selection[0], selection[-1] + 1
    tokens = tuple(reading["input_ids"])
    if (not 0 < first < stop <= len(tokens) <= study.MAX_INPUT_TOKENS
            or not 8 <= stop - first <= study.MAX_FOCUS_TOKENS or selection != list(range(first, stop))):
        return None
    selected = pairs[first:stop]
    covered = {pos for a, b in selected for pos in range(a, b)}
    if (any(not char.isspace() and start + position not in covered for position, char in enumerate(text[start:end]))
            or any(not text[pos].isspace() for pos in covered if pos < start)):
        return None
    return SentenceInput(tokens, first, stop)


def initialize(snapshot):
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm import load
    from transformers import AutoTokenizer

    study.require(all(os.getenv(name) == "1" for name in ("TRANSFORMERS_OFFLINE", "HF_HUB_OFFLINE")),
                  "Use offline settings for the pinned weights")
    study.require({name: importlib.metadata.version(name) for name in RUNTIME} == RUNTIME, "The MLX runtime differs")
    verified_model(snapshot)
    mx.set_default_device(mx.gpu)
    mx.set_cache_limit(256 * 1024 * 1024)
    model, _ = load(str(snapshot), tokenizer_config={"trust_remote_code": False})
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False)
    model.freeze()
    model.eval()
    nn.quantize(model, mode="affine", group_size=64, bits=4)
    model.apply(lambda value: value.astype(mx.float32) if value.dtype in (mx.float16, mx.bfloat16) else value)
    mx.eval(model.parameters())
    return mx, nn, model, tokenizer


def batch(inputs, mx):
    length = max(len(item.tokens) for item in inputs)
    focus_length = max(item.count for item in inputs)
    # All padding sits after the source text, outside the causal context of the scored tokens.
    ids = [list(item.tokens) + [0] * (length - len(item.tokens)) for item in inputs]
    previous = [list(range(item.first - 1, item.stop - 1)) + [0] * (focus_length - item.count) for item in inputs]
    labels = [list(item.tokens[item.first:item.stop]) + [0] * (focus_length - item.count) for item in inputs]
    weights = [[1.] * item.count + [0.] * (focus_length - item.count) for item in inputs]
    return mx.array(ids), mx.array(previous), mx.array(labels), mx.array(weights, dtype=mx.float32)


def cost(model, mx, nn, prefix, package):
    ids, previous, labels, weights = package
    backbone = model.model
    embed = backbone.embed_tokens(ids)
    context = mx.broadcast_to(prefix.astype(embed.dtype), (ids.shape[0], *prefix.shape))
    hidden = backbone(ids, input_embeddings=mx.concatenate((context, embed), axis=1))
    source_states = mx.take_along_axis(hidden, (previous[..., None] + len(prefix)), axis=1)
    head = backbone.embed_tokens.as_linear if model.args.tie_word_embeddings else model.lm_head
    logits = head(source_states).astype(mx.float32)
    token_cost = nn.losses.cross_entropy(logits, labels, reduction="none")
    return (token_cost * weights).sum(axis=1) / weights.sum(axis=1)


def embeddings(model, tokenizer, mx, text):
    tokens = tokenizer(text, add_special_tokens=False)["input_ids"]
    return model.model.embed_tokens(mx.array(tokens))


def fitting_pairs(source, tokenizer):
    by_work = {}
    for row in source["rows"]:
        if row["role"] != "fit":
            continue
        by_work.setdefault(row["group"], {})[row["label"]] = row
    pairs = []
    for group, items in sorted(by_work.items()):
        target = items["edited"]["target"]
        human, synthetic = (encode(tokenizer, items[label], target) for label in ("ordinary", "edited"))
        if human and synthetic:
            pairs.append((group, human, synthetic))
    return pairs, len(by_work)


def train(plan, snapshot, weights, output):
    import mlx.optimizers as optim

    mx, nn, model, tokenizer = initialize(snapshot)
    fit, planned = fitting_pairs(plan, tokenizer)
    study.require(len(fit) >= 32, "Too few paired sources for learning")
    neutral = embeddings(model, tokenizer, mx, "The following is an excerpt from a plain English article presented for review.\n")
    initial = neutral[:study.PREFIX_TOKEN_COUNT].astype(mx.float32)
    study.require(len(initial) == study.PREFIX_TOKEN_COUNT, "Prefix initializer is too short")
    prompts = mx.stack((initial, initial))
    optimizer = optim.Adam(learning_rate=study.LEARNING_RATE)
    rng = random.Random(study.SEED)

    def objective(parameters, package, split):
        human = cost(model, mx, nn, parameters[0], package)
        artificial = cost(model, mx, nn, parameters[1], package)
        margin = study.HINGE
        human_loss = human[:split] + mx.maximum(0., margin + human[:split] - human[split:])
        artificial_loss = artificial[split:] + mx.maximum(0., margin + artificial[split:] - artificial[:split])
        return .5 * (human_loss.mean() + artificial_loss.mean())

    compute = mx.value_and_grad(objective)
    losses = []
    for epoch in range(study.EPOCHS):
        rng.shuffle(fit)
        for start in range(0, len(fit), study.BATCH):
            group = fit[start:start + study.BATCH]
            package = batch([row[i] for i in (1, 2) for row in group], mx)
            loss, gradient = compute(prompts, package, len(group))
            norm = mx.linalg.norm(gradient)
            study.require(bool(mx.isfinite(norm).item()), f"Nonfinite gradient at step {len(losses) + 1}")
            gradient = gradient / mx.maximum(1., norm / 5.)
            prompts = optimizer.apply_gradients({"prompts": gradient}, {"prompts": prompts})["prompts"]
            mx.eval(prompts, optimizer.state, loss)
            value = float(loss.item())
            study.require(value == value and abs(value) < 100, "Prompt fitting became unstable")
            losses.append(round(value, 7))
            if len(losses) == 1 or len(losses) % 8 == 0:
                print(f"Prompt training step {len(losses)} (epoch {epoch + 1}, loss {value:.3f})", flush=True)
    mx.save_safetensors(str(weights), {"prompts": prompts})
    study.save(output, {"schema": study.SCHEMA, "model_revision": REVISION, "runtime": RUNTIME, "model_assets": MODEL_ASSETS,
                        "optimized_source_groups": sorted(item[0] for item in fit), "planned_source_groups": planned,
                        "losses": losses, "prefix_tokens": study.PREFIX_TOKEN_COUNT, "epochs": study.EPOCHS,
                        "batch_size": study.BATCH, "hinge": study.HINGE, "learning_rate": study.LEARNING_RATE,
                        "quantization": "four-bit affine, group 64, float32 scales and activations",
                        "gradient_norm_cap": 5.,
                        "seed": study.SEED, "weights_sha256": study.file_hash(weights)})


def score(plan, snapshot, weights, output, training):
    mx, nn, model, tokenizer = initialize(snapshot)
    provenance = study.read(training)
    study.require(study.file_hash(weights) == provenance["weights_sha256"], "Learned prefix differs")
    current_fit_groups = {item[0] for item in fitting_pairs(plan, tokenizer)[0]}
    study.require(sorted(current_fit_groups) == provenance["optimized_source_groups"], "The fitted works differ")
    prompts = mx.load(str(weights))["prompts"]
    expected_shape = (2, study.PREFIX_TOKEN_COUNT, model.args.hidden_size)
    study.require(prompts.shape == expected_shape, "Learned prefix geometry differs")
    fixed = tuple(embeddings(model, tokenizer, mx, text) for text in study.FIXED)
    null = mx.zeros((0, model.args.hidden_size), dtype=mx.float32)
    results = {}
    for role in ("calibration", "held"):
        rows = [row for row in plan["rows"] if row["role"] == role]
        unique = {}
        index = {}
        for row in rows:
            readings = [encode(tokenizer, row, center) for center in study.CENTERS]
            for item in readings:
                if item is not None:
                    unique.setdefault(item, None)
            index[row["sha256"]] = readings
        values = list(unique)
        for start in range(0, len(values), 8):
            active = values[start:start+8]
            package = batch(active, mx)
            free, plain_h, plain_a, fitted_h, fitted_a = (
                cost(model, mx, nn, prefix, package) for prefix in (null, *fixed, prompts[0], prompts[1])
            )
            matrix = mx.stack((-free, plain_h - plain_a, fitted_h - fitted_a), axis=1).tolist()
            unique.update(zip(active, matrix, strict=True))
            if start % 160 == 0:
                print(f"Prompt scores {role} {min(start+8,len(values))}/{len(values)}", flush=True)
        for digest, items in index.items():
            results[digest] = {arm: [round(unique[item][i], 8) if item is not None else None for item in items]
                               for i, arm in enumerate(("unconditioned", "fixed_prompt", "learned_prompt"))}
    study.save(output, {"schema": study.SCHEMA, "training": provenance, "rows": results})
