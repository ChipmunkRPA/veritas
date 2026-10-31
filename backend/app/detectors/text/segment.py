"""Sentence segmentation with character offsets (pysbd), paragraph-aware.

This file imports pysbd and nothing of the application: `pipeline.segment_within` runs it as a script in a child process
(`python -I segment.py`), which reads the paragraphs and the text from its input and prints the offsets of the sentences as
JSON (N-14b; S2-01). The time of pysbd grows with the sentences times the characters of a paragraph, inside one call that
holds the interpreter of whoever runs it; `spans` is that part, and a request's document passes through it in the child
only. `segment` is for a caller that brings its own text (the benchmark, the tests)."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pysbd

if TYPE_CHECKING:
    from collections.abc import Sequence

    from app.detectors.text.extract import Document

_seg = pysbd.Segmenter(language="en", clean=False, char_span=True)
_WORD = re.compile(r"\w[\w'’-]*", re.UNICODE)  # unicode-aware: CJK characters count as words
MAX_UNITS = 1500  # beyond this, adjacent sentences are merged so cost stays bounded for huge documents


_CJK = re.compile(r"[぀-ヿ㐀-䶿一-鿿가-힯]")


def _count_words(t: str) -> int:
    """Words for space-delimited scripts; for CJK scripts (no spaces) count characters / 1.5 as a proxy."""
    return len(_WORD.findall(t)) + int(len(_CJK.findall(t)) / 1.5)


@dataclass
class Sentence:
    idx: int
    start: int
    end: int
    text: str
    paragraph: int
    n_words: int


def spans(text: str, paragraphs: Sequence[Sequence[int]]) -> list[tuple[int, int, int]]:
    """(start, end, paragraph) of each sentence in `text`, without the whitespace around it."""
    out: list[tuple[int, int, int]] = []
    for pi, (ps, pe) in enumerate(paragraphs):
        ptxt = text[ps:pe]
        try:
            found = [(sp.start, sp.end) for sp in _seg.segment(ptxt)]
        except Exception:
            found = []
        if not found:
            found = [(0, len(ptxt))]
        for a, b in found:
            s, e = ps + a, ps + b
            # trim whitespace
            while s < e and text[s].isspace():
                s += 1
            while e > s and text[e - 1].isspace():
                e -= 1
            if e <= s:
                continue
            out.append((s, e, pi))
    return out


def sentences(doc: Document, found: Sequence[Sequence[int]]) -> list[Sentence]:
    """The sentences of `doc` at the offsets `spans` found, each built from the document's own text."""
    out = [Sentence(i, s, e, doc.text[s:e], pi, _count_words(doc.text[s:e])) for i, (s, e, pi) in enumerate(found)]
    if len(out) > MAX_UNITS:
        k = -(-len(out) // MAX_UNITS)  # sentences per merged unit
        merged: list[Sentence] = []
        for i in range(0, len(out), k):
            grp = out[i : i + k]
            if grp[-1].paragraph != grp[0].paragraph:  # never merge across paragraphs
                grp = [x for x in grp if x.paragraph == grp[0].paragraph] or grp[:1]
            t = doc.text[grp[0].start : grp[-1].end]
            merged.append(Sentence(len(merged), grp[0].start, grp[-1].end, t, grp[0].paragraph, sum(x.n_words for x in grp)))
            # re-add any sentences dropped by the paragraph rule
            rest = out[i + len(grp) : i + k]
            for x in rest:
                merged.append(Sentence(len(merged), x.start, x.end, x.text, x.paragraph, x.n_words))
        out = merged
    return out


def segment(doc: Document) -> list[Sentence]:
    return sentences(doc, spans(doc.text, doc.paragraphs))


def _child() -> int:
    """`python -I segment.py`: the input is one line of JSON, the paragraphs, and after it the text in UTF-8 (a lone
    surrogate of a damaged file passes as it is); the output is the offsets of the sentences as JSON."""
    import json
    import sys

    head, _, body = sys.stdin.buffer.read().partition(b"\n")
    sys.stdout.write(json.dumps(spans(body.decode("utf-8", "surrogatepass"), json.loads(head))))
    return 0


if __name__ == "__main__":
    raise SystemExit(_child())
