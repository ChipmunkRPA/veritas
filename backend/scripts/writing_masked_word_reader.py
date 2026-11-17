# SPDX-License-Identifier: Apache-2.0
"""Measure the same masked words with and without surrounding sentences."""

from __future__ import annotations

import hashlib
import math
import re
from bisect import bisect_left, bisect_right
from collections import defaultdict
from pathlib import Path

REVISION = "387825ce42dbb39b87911cdf8e383ee3b25184f8"
ASSETS = {
    "config.json": "61716a972c73e9cc1d6f1165e27e577c1f4012a3cc480e431f427a78d5071ece",
    "pytorch_model.bin": "01671831dfec6818302153919efbfd6ee4ad91dd9fba7f3649bdd6cb7b0f16e4",
    "vocab.txt": "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3",
}
MAX_DOCUMENT = 2048
MAX_FOCUS = 128
NEIGHBOR_TOKENS = 24
WORDS_PER_SENTENCE = 8
MIN_WORDS = 5
MAX_WORD_PIECES = 5
BATCH = 96
UNITS = 100_000
WORD = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)*", re.UNICODE)


def hash_file(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def token_bounds(starts, ends, begin, end):
    """Return only tokens contained in the requested character interval."""
    first, stop = bisect_right(ends, begin), bisect_left(starts, end)
    if first >= stop or starts[first] < begin or ends[stop - 1] > end:
        return None
    return first, stop


def locate(tokenizer, text, sentences, unknown):
    encoded = tokenizer.encode(text, add_special_tokens=False)
    ids, offsets = encoded.ids, encoded.offsets
    if len(ids) > MAX_DOCUMENT:
        return len(ids), {}, {}, []
    starts, ends = zip(*offsets, strict=True) if offsets else ([], [])
    if (not all(0 <= a < b <= len(text) for a, b in offsets)
            or any(b > c for (_, b), (c, _) in zip(offsets, offsets[1:]))):
        raise ValueError("The masked model tokenizer does not preserve ordered character positions")
    by_sentence = {}
    for index, (begin, end, *_) in enumerate(sentences):
        scope = token_bounds(starts, ends, begin, end)
        if scope is None or scope[1] - scope[0] > MAX_FOCUS:
            continue
        words = []
        for match in WORD.finditer(text, begin, end):
            start, stop = match.span()
            part = token_bounds(starts, ends, start, stop)
            if not part:
                continue
            first, last = part
            if (starts[first] == start and ends[last - 1] == stop and 1 <= last - first <= MAX_WORD_PIECES
                    and unknown not in ids[first:last]):
                words.append((first, last))
        if len(words) < MIN_WORDS:
            continue
        sample_count = min(WORDS_PER_SENTENCE, len(words))
        selected = [words[((2*i+1)*len(words)) // (2*sample_count)] for i in range(sample_count)]
        by_sentence[index] = (scope, selected)
    # Each view keeps the same sequence positions. The isolated view masks only attention to the adjoining text.
    plans, commitments = defaultdict(list), {}
    for index, ((focus, end), words) in by_sentence.items():
        first, stop = max(0, focus - NEIGHBOR_TOKENS), min(len(ids), end + NEIGHBOR_TOKENS)
        frames = []
        for word_index, (a, b) in enumerate(words):
            frames.append([starts[a], ends[b - 1]])
            for target in range(a, b):
                plans[stop - first].append((index, word_index, first, stop, focus, end, target, b))
        commitments[index] = (end - focus, len(words), sum(b - a for a, b in words), frames)
    return len(ids), plans, commitments, ids


def read_checkpoint(folder, device):
    import torch
    from tokenizers import Tokenizer, models, normalizers, pre_tokenizers
    from transformers import BertConfig, BertForMaskedLM

    if {name: hash_file(folder / name) for name in ASSETS} != ASSETS:
        raise ValueError("The masked checkpoint differs from the pinned publisher assets")
    torch.set_num_threads(3)
    torch.set_num_interop_threads(1)
    config = BertConfig.from_json_file(str(folder / "config.json"))
    model = BertForMaskedLM(config)
    # Load only tensors from the pinned official checkpoint, with no remote or checkpoint-defined code.
    state = torch.load(folder / "pytorch_model.bin", map_location="cpu", weights_only=True)
    state = {key: tensor for key, tensor in state.items() if not key.startswith(("bert.pooler.", "cls.seq_relationship."))}
    model.load_state_dict(state, strict=True)
    model.eval().to(device)
    vocabulary = {token: index for index, token in enumerate((folder / "vocab.txt").read_text().splitlines())}
    tokenizer = Tokenizer(models.WordPiece(vocabulary, unk_token="[UNK]"))
    tokenizer.normalizer = normalizers.BertNormalizer(lowercase=True)
    tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    if len(vocabulary) != config.vocab_size:
        raise ValueError("The source vocabulary differs from the masked model configuration")
    start_token, stop_token, mask, unknown = (vocabulary[name] for name in ("[CLS]", "[SEP]", "[MASK]", "[UNK]"))

    def measure(text, sentences):
        n_tokens, plans, commitments, tokens = locate(tokenizer, text, sentences, unknown)
        readings = defaultdict(list)
        with torch.inference_mode():
            # Round only sequence padding, to make batches inexpensive across different sentence widths.
            buckets = defaultdict(list)
            for width, items in plans.items():
                buckets[(width + 2 + 15) // 16 * 16].extend(items)
            for width, items in sorted(buckets.items()):
                for start in range(0, len(items), BATCH // 2):
                    subset = items[start:start + BATCH // 2]
                    inputs, attends, positions, labels = [], [], [], []
                    for _, _, first, stop, focus, end, target, word_end in subset:
                        original = [start_token, *tokens[first:stop], stop_token]
                        original[target - first + 1:word_end - first + 1] = [mask] * (word_end - target)
                        pad = [0] * (width - len(original))
                        inputs.extend([original + pad] * 2)
                        attends.append([1] * len(original) + pad)
                        attends.append([1] + [int(focus <= idx < end) for idx in range(first, stop)] + [1] + pad)
                        positions.extend([target - first + 1] * 2)
                        labels.extend([tokens[target]] * 2)
                    tensor = torch.tensor(inputs, dtype=torch.long, device=device)
                    attention = torch.tensor(attends, dtype=torch.long, device=device)
                    row = torch.arange(len(inputs), device=device)
                    hidden = model.bert(input_ids=tensor, attention_mask=attention).last_hidden_state
                    logits = model.cls(hidden[row, torch.tensor(positions, device=device)]).float()
                    target_logits = logits[row, torch.tensor(labels, device=device)]
                    losses = (torch.logsumexp(logits, dim=-1) - target_logits).reshape(-1, 2).cpu().tolist()
                    for (sentence, word, *rest), pair in zip(subset, losses, strict=True):
                        readings[sentence, word].append(pair)
        result = []
        for sentence, (focus_tokens, words, pieces, offsets) in sorted(commitments.items()):
            by_word = [readings[sentence, index] for index in range(words)]
            mean = [math.fsum(math.fsum(pair[i] for pair in word) / len(word) for word in by_word) / words
                    for i in (0, 1)]
            if any(not math.isfinite(value) or value < 0 for value in mean):
                raise ValueError("The masked model produced an invalid log loss")
            result.append([sentence, focus_tokens, words, pieces, round(mean[0]*UNITS), round(mean[1]*UNITS), offsets])
        return n_tokens, result

    return measure
