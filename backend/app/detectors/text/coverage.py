"""Complete classifier coverage using non-overlapping native tokenizer windows.

A long sentence is not a model context window. Keep all overflow windows, score
in bounded micro-batches, then weight by content-token count within each input.
This changes long-input inference; it is not a calibrated probability mapping.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

from app.core import deadline


def classify_complete(texts: list[str], tokenizer: Any, device: Any, predict: Callable, *,
                      max_len: int, batch: int, check: Callable[[str], None] = deadline.check,
                      ) -> tuple[list[float], list[dict[str, int]]]:
    if batch < 1 or max_len <= tokenizer.num_special_tokens_to_add(pair=False):
        raise ValueError("classifier batch must be positive and token limit must allow content")
    probabilities: list[float] = []
    coverage: list[dict[str, int]] = []
    for start in range(0, len(texts), batch):
        check("text tokenisation")
        inputs = texts[start:start + batch]
        # stride=0 avoids counting overlapped context as additional evidence. The native tokenizer preserves special
        # tokens, padding and truncation-side behaviour; overflow windows retain the complete original token sequence.
        encoded = tokenizer(inputs, padding=True, truncation=True, max_length=max_len, stride=0,
                            return_overflowing_tokens=True, return_special_tokens_mask=True, return_tensors="pt")
        mapping = encoded.pop("overflow_to_sample_mapping").tolist()
        special = encoded.pop("special_tokens_mask")
        counts = ((encoded["attention_mask"] != 0) & (special == 0)).sum(dim=1).tolist()
        weighted, denominators = [0.0] * len(inputs), [0] * len(inputs)
        records = [{"tokens": 0, "token_windows": 0} for _ in inputs]
        for offset in range(0, len(mapping), batch):
            check("text classifier token windows")
            mini = {key: value[offset:offset + batch].to(device) for key, value in encoded.items()}
            values = predict(mini).detach().float().cpu().reshape(-1).tolist()
            check("completed text classifier token windows")
            if len(values) != len(mapping[offset:offset + batch]):
                raise ValueError("classifier returned the wrong number of window scores")
            for i, value in enumerate(values, offset):
                sample, count = mapping[i], counts[i]
                if not 0 <= value <= 1:
                    raise ValueError("classifier returned a non-finite or invalid probability")
                weight = max(count, 1)  # preserve the legacy one-window result for an empty input
                weighted[sample] += value * weight
                denominators[sample] += weight
                records[sample]["tokens"] += count
                records[sample]["token_windows"] += 1
        if any(count == 0 for count in denominators):
            raise ValueError("tokenizer did not return a window for each input")
        probabilities.extend(value / count for value, count in zip(weighted, denominators, strict=True))
        coverage.extend(records)
    return probabilities, coverage
