# SPDX-License-Identifier: Apache-2.0
"""Fit and measure a compact writing localizer with sentence and changed-word labels."""

from __future__ import annotations

import argparse
import importlib.metadata
import math
import os
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import NamedTuple

from scripts import writing_masked_word_reader as pinned
from scripts import writing_multi_passage_panel as data
from scripts import writing_token_supervision_data as source

SCHEMA = "chipmunk.writing-token-supervision/1"
MAX_TOKENS = 320
CONTEXT = 24
BATCH = 12
EPOCHS = 3
SEED = 731
UNITS = 100_000
ARMS = ("frozen", "tuned")
VIEWS = ("sentence", "words", "joint")
QUANTILES = {"strict": .95, "medium": .90, "wider": .80}
REGIONS = 6


class Word(NamedTuple):
    token: int
    start: int
    stop: int
    target: int


class Sentence(NamedTuple):
    index: int
    start: int
    stop: int
    target: int
    words: tuple[Word, ...]


class Window(NamedTuple):
    ids: tuple[int, ...]
    sentences: tuple[Sentence, ...]


def configure(device):
    import torch

    torch.set_num_threads(3)
    torch.set_num_interop_threads(1)
    if device == "mps" and not torch.backends.mps.is_available():
        raise ValueError("The Apple accelerator is unavailable for this comparison")
    if any(os.environ.get(name) != "1" for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")):
        raise ValueError("Run with the reviewed local assets in offline mode")
    return {name: importlib.metadata.version(name) for name in ("torch", "transformers", "tokenizers", "safetensors")} | {"device": device}


def initialization(checkpoint, *, preload):
    import torch
    from tokenizers import Tokenizer, models, normalizers, pre_tokenizers
    from transformers import BertConfig, BertModel

    components = pinned.ASSETS if preload else {k: v for k, v in pinned.ASSETS.items() if k != "pytorch_model.bin"}
    if {key: pinned.hash_file(checkpoint / key) for key in components} != components:
        raise ValueError("The initialization differs from the pinned Apache-2.0 assets")
    config = BertConfig.from_json_file(str(checkpoint / "config.json"))
    encoder = BertModel(config, add_pooling_layer=False)
    if preload:
        tensors = torch.load(checkpoint / "pytorch_model.bin", weights_only=True, map_location="cpu")
        encoder.load_state_dict({key.removeprefix("bert."): value for key, value in tensors.items()
                                 if key.startswith("bert.") and not key.startswith("bert.pooler.")}, strict=True)
    model = torch.nn.ModuleDict({"encoder": encoder, "heads": torch.nn.Linear(config.hidden_size, 2)})
    vocabulary = {word: index for index, word in enumerate((checkpoint / "vocab.txt").read_text().splitlines())}
    if vocabulary["[PAD]"] != 0 or len(vocabulary) != config.vocab_size:
        raise ValueError("The source vocabulary is outside the verified configuration")
    tokenizer = Tokenizer(models.WordPiece(vocabulary, unk_token="[UNK]"))
    tokenizer.normalizer = normalizers.BertNormalizer(lowercase=True)
    tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    return model, tokenizer, tuple(vocabulary[key] for key in ("[CLS]", "[SEP]", "[UNK]"))


def target_fraction(a, b, ranges):
    return sum(max(0, min(b, y) - max(a, x)) for x, y in ranges) / (b-a)


def windows(case, tokenizer, tokens):
    text = case["text"]
    prefix = data._prefix(text)
    if data.digest(text.encode()) != case["native_text_sha256"] or prefix[-1] != case["nonspace_characters"]:
        raise ValueError("The input document differs from its verified native source positions")
    encoded = tokenizer.encode(text, add_special_tokens=False)
    ids, offsets = encoded.ids, encoded.offsets
    if not offsets:
        return []
    if any(not 0 <= a < b <= len(text) for a, b in offsets) or any(a[1] > b[0] for a, b in zip(offsets, offsets[1:])):
        raise ValueError("Word pieces disagree with native Unicode positions")
    starts, stops = zip(*offsets, strict=True)
    cls, sep, unk = tokens
    truth = [value["compact"] for value in case["truth"]]
    exact_words = {tuple(span) for span in case["changed_words"]}
    focus_limit = MAX_TOKENS - 2 - 2 * CONTEXT
    sentences = []
    for index, (a, b, left, right, _, word_count) in enumerate(case["sentences"]):
        scope = pinned.token_bounds(starts, stops, a, b)
        if (left, right) != (prefix[a], prefix[b]) or left >= right:
            raise ValueError("The native sentence disagrees with its nonspace positions")
        if scope is None or scope[1] - scope[0] > focus_limit or word_count < 5:
            continue
        fraction = target_fraction(left, right, truth)
        label = int(fraction >= .5) if fraction == 0 or fraction >= .5 else -1
        words = []
        for match in source.WORD.finditer(text, a, b):
            lo, hi = match.span()
            part = pinned.token_bounds(starts, stops, lo, hi)
            if part is None or starts[part[0]] != lo or stops[part[1] - 1] != hi or unk in ids[part[0]:part[1]]:
                continue
            bounds = (prefix[lo], prefix[hi])
            word_label = 1 if bounds in exact_words else (-1 if target_fraction(*bounds, truth) else 0)
            words.append(Word(part[0], *bounds, word_label))
        if words:
            sentences.append(Sentence(index, *scope, label, tuple(words)))
    groups = []
    for sentence in sentences:
        if not groups or sentence.stop - groups[-1][0].start > focus_limit:
            groups.append([])
        groups[-1].append(sentence)
    output = []
    for group in groups:
        lo, hi = max(0, group[0].start - CONTEXT), min(len(ids), group[-1].stop + CONTEXT)
        shift = 1 - lo
        placed = tuple(Sentence(s.index, s.start + shift, s.stop + shift, s.target,
                                tuple(Word(w.token + shift, w.start, w.stop, w.target) for w in s.words)) for s in group)
        output.append(Window((cls, *ids[lo:hi], sep), placed))
    return output


def batches(all_windows, *, rng=None):
    buckets = defaultdict(list)
    for w in all_windows:
        buckets[(len(w.ids) + 31) // 32].append(w)
    work = []
    for bucket in buckets.values():
        if rng:
            rng.shuffle(bucket)
        work.extend(bucket[i:i + BATCH] for i in range(0, len(bucket), BATCH))
    if rng:
        rng.shuffle(work)
    return work


def forward(model, windows, device):
    import torch
    import torch.nn.functional as functional

    width = max(len(w.ids) for w in windows)
    inputs = [(*w.ids, *((0,) * (width - len(w.ids)))) for w in windows]
    ids = torch.tensor(inputs, dtype=torch.long, device=device)
    states = model["encoder"](input_ids=ids, attention_mask=ids.ne(0)).last_hidden_state
    logits = model["heads"](states).float()
    totals = functional.pad(logits[:, :, 0].cumsum(dim=1), (1, 0))
    sentences = [(i, s) for i, w in enumerate(windows) for s in w.sentences]
    positions = torch.tensor([(i, s.start, s.stop) for i, s in sentences], dtype=torch.long, device=device)
    row, lo, hi = positions.unbind(dim=1)
    sentence_scores = (totals[row, hi] - totals[row, lo]) / (hi - lo)
    positions = torch.tensor([(i, w.token) for i, s in sentences for w in s.words], dtype=torch.long, device=device)
    row, at = positions.unbind(dim=1)
    return sentences, sentence_scores, logits[row, at, 1]


def loss(logits, labels, upweight, device):
    import torch
    import torch.nn.functional as functional

    targets = torch.tensor(labels, dtype=torch.float32, device=device)
    weights = targets.ge(0) * (1 + (upweight - 1) * targets.eq(1))
    terms = functional.binary_cross_entropy_with_logits(logits, targets.clamp_min(0), reduction="none")
    return (terms * weights).sum() / weights.sum().clamp_min(1)


def labeled_counts(examples):
    sentences = [s for w in examples for s in w.sentences]
    return Counter(s.target for s in sentences), Counter(w.target for s in sentences for w in s.words)


def fit(plan, checkpoint, output, device):
    import torch
    from safetensors.torch import save_file

    runtime = configure(device)
    frozen = {role: pinned.hash_file(plan / f"{role}.json.gz") for role in ("fit", "calibration", "held")}
    prepared = data.read(plan / "fit.json.gz")
    if prepared["schema"] != source.SCHEMA or prepared["role"] != "fit":
        raise ValueError("Supply the sealed source-training roster")
    output.mkdir(mode=0o700)
    runs = {}
    examples = None
    for arm in ARMS:
        torch.manual_seed(SEED)
        model, tokenizer, tokens = initialization(checkpoint, preload=True)
        if examples is None:
            examples = [w for case in prepared["works"] for delivery in case["cases"] for w in windows(delivery, tokenizer, tokens)]
        if not examples:
            raise ValueError("The sampled documents contain no supported sentence positions")
        model.to(device)
        encoder = model["encoder"]
        if arm == "frozen":
            encoder.requires_grad_(False).eval()
        parameters = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW([
            {"params": encoder.parameters(), "lr": 3e-5}, {"params": model["heads"].parameters(), "lr": 3e-4},
        ], weight_decay=.01)
        history = []
        for epoch in range(EPOCHS):
            pending = batches(examples, rng=random.Random(SEED + epoch))
            total = torch.zeros((), device=device)
            for step, batch in enumerate(pending, 1):
                optimizer.zero_grad(set_to_none=True)
                rows, sentence_scores, word_scores = forward(model, batch, device)
                active = loss(sentence_scores, [s.target for _, s in rows], 3, device)
                lexical = loss(word_scores, [w.target for _, s in rows for w in s.words], 6, device)
                objective = active + .65 * lexical
                objective.backward()
                torch.nn.utils.clip_grad_norm_(parameters, 1.)
                optimizer.step()
                total += objective.detach()
                if step % 120 == 0:
                    print(f"{arm} epoch {epoch + 1} update {step}/{len(pending)}", flush=True)
            mean = float(total.cpu()) / len(pending)
            if not math.isfinite(mean):
                raise ValueError("A nonfinite loss prevents using this fit")
            history.append(round(mean, 7))
            print(f"{arm} epoch {epoch + 1} of {EPOCHS}, {len(pending)} updates, loss {mean:.5f}", flush=True)
        path = output / f"{arm}.safetensors"
        save_file({key: value.detach().cpu().contiguous() for key, value in model.state_dict().items()}, path)
        runs[arm] = {"weights_sha256": pinned.hash_file(path), "loss": history,
                     "trainable_parameters": sum(p.numel() for p in parameters)}
        del model, optimizer, encoder, parameters
    sentence_counts, word_counts = labeled_counts(examples)
    data.save(output / "fit.json", {"schema": SCHEMA, "plans": frozen, "runtime": runtime, "seed": SEED,
                                    "initialization": pinned.ASSETS, "revision": pinned.REVISION, "windows": len(examples),
                                    "fit_works": len(prepared["works"]),
                                    "training_labels": {"sentence": sentence_counts, "word": word_counts}, "runs": runs})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    train = commands.add_parser("fit")
    for name in ("plan", "checkpoint", "output"):
        train.add_argument(name, type=Path)
    train.add_argument("--device", required=True, choices=("cpu", "mps"))
    args = vars(parser.parse_args())
    args.pop("action")
    fit(**args)


if __name__ == "__main__":
    main()
