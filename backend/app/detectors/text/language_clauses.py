# SPDX-License-Identifier: Apache-2.0
"""Find a bounded within-sentence language witness on the original text."""

from __future__ import annotations

from app.detectors.text.language_scope import MIN_PASSAGE_LETTERS, WORDS, _recognised

NARROW_WORDS = 12
WIDE_WORDS = 20
STRIDE = 4
MAX_WINDOWS = 96


def _windows(sentences):
    all_windows = []
    for sentence in sentences:
        spans = [word.span() for word in WORDS.finditer(sentence.text)]
        if len(spans) < WIDE_WORDS:
            continue
        last = len(spans) - NARROW_WORDS
        starts = list(range(0, last + 1, STRIDE))
        if starts[-1] != last:
            starts.append(last)
        all_windows.extend((sentence, spans, start) for start in starts)
    total = len(all_windows)
    if total <= MAX_WINDOWS:
        return all_windows
    return [all_windows[i * (total - 1) // (MAX_WINDOWS - 1)] for i in range(MAX_WINDOWS)]


def _range(spans, start, width):
    return spans[start][0], spans[start + width - 1][1]


def candidates(sentences, *, single=False):
    """Return the first corroborated range, optionally retaining its narrower comparison."""
    narrow = None
    wide_readings = {}
    for sentence, spans, start in _windows(sentences):
        left, right = _range(spans, start, NARROW_WORDS)
        excerpt = sentence.text[left:right]
        if sum(char.isalpha() for char in excerpt) < MIN_PASSAGE_LETTERS:
            continue
        language = _recognised(excerpt)
        if language in (None, "en"):
            continue
        location = {"start": sentence.start + left, "end": sentence.start + right, "sentence": sentence.idx,
                    "language": language}
        if single and narrow is None:
            narrow = location
        far = len(spans) - WIDE_WORDS
        padding = WIDE_WORDS - NARROW_WORDS
        options = sorted({max(0, start - padding), min(start, far)})
        for offset in options:
            key = sentence.idx, offset
            if key not in wide_readings:
                a, b = _range(spans, offset, WIDE_WORDS)
                wide_readings[key] = _recognised(sentence.text[a:b])
            if wide_readings[key] == language:
                a, b = _range(spans, offset, WIDE_WORDS)
                return {"narrow": narrow, "paired": {**location, "context_start": sentence.start + a,
                                                     "context_end": sentence.start + b}}
    return {"narrow": narrow, "paired": None}
