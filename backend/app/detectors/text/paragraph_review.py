"""Bounded optional paragraph-only responses from the already loaded text classifier."""

from __future__ import annotations

from app.detectors.text.extract import text_hash

VERSION = "isolated-paragraph/1"
MAX_PARAGRAPHS = 2
MAX_TOKENS = 512


def record(classifier, doc, sentences, current, deadline) -> dict | None:
    if doc.meta.get("method") != "plain" or len(doc.paragraphs) < 4:
        return None
    remaining = deadline.remaining()
    if remaining is not None and remaining < 10:
        return None
    ranked = {}
    for sentence, window in zip(sentences, current, strict=True):
        index = sentence.paragraph
        if 0 < index < len(doc.paragraphs) - 1 and window["score"] >= 0.5:
            ranked[index] = max(ranked.get(index, 0.0), window["score"])
    spans, inputs = [], []
    for index in sorted(ranked, key=lambda i: (-ranked[i], i)):
        start, end = doc.paragraphs[index]
        text = doc.text[start:end]
        if 18 <= len(text.split()) <= 145 and len(text) <= 1200:
            spans.append({"start": start, "end": end})
            inputs.append(text)
        if len(inputs) == MAX_PARAGRAPHS:
            break
    if not inputs:
        return None
    eligible = [(span, text) for span, text in zip(spans, inputs, strict=True)
                if len(classifier.tok(text, truncation=False, verbose=False)["input_ids"]) <= MAX_TOKENS]
    if not eligible:
        return None
    scores = classifier.predict([text for _, text in eligible], max_len=MAX_TOKENS,
                                batch=MAX_PARAGRAPHS, check=deadline.check)
    recorded = []
    for (span, text), score, coverage in zip(eligible, scores, classifier.last_coverage, strict=True):
        tokens = coverage["tokens"]
        if (tokens <= 0 or coverage["token_windows"] != 1 or coverage.get("omitted_tokens", 0) != 0
                or coverage.get("covered_tokens", tokens) != tokens):
            continue
        recorded.append({"spans": [span], "input_sha256": text_hash(text), "score": float(score),
                         "score_exact_hex": float(score).hex(), "token_coverage": {
                             **coverage, "covered_tokens": tokens, "omitted_tokens": 0,
                         }})
    return {"version": VERSION, "inputs": recorded} if recorded else None
