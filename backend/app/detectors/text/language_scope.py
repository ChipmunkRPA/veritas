"""Gate English authorship checks using the bundled language identifier."""

from __future__ import annotations

import re
from collections.abc import Iterator

from app.detectors.text import extract
from app.detectors.text.segment import Sentence

KEY = "writing_language_scope"
ENGLISH = "english"
LIMITATION = "English writing could not be confirmed for statistical analysis"
ASSESSMENT_SCOPE = "Writing-language coverage"
MESSAGES = {
    "unsupported": ("Writing not assessed: English required",
                    "The extracted writing was identified as a language outside the current English coverage. "
                    "Automated authorship checks were skipped."),
    "mixed": ("Writing not assessed: mixed languages",
              "The language check found both English and another language in the extracted text. The writing checks use "
              "neighbouring sentences as context, so automated authorship checks were skipped."),
    "unknown": ("Writing not assessed: language unclear",
                "The language of the extracted writing could not be established reliably enough for the English "
                "authorship checks. Automated authorship checks were skipped."),
}
LANGUAGE_POINT = .99
MIN_PASSAGE_LETTERS = 45
MAX_LOCAL_PASSAGES = 48
MAX_PASSAGE_CHARS = 4_096
WORDS = re.compile(r"[^\W\d_]+", flags=re.UNICODE)


def _recognised(text: str) -> str | None:
    reading = extract._ocr_language(text)
    confidence = reading.get("confidence")
    return reading.get("code") if confidence is not None and confidence >= LANGUAGE_POINT else None


def _local_passages(sentences: list[Sentence]) -> Iterator[str]:
    pending: list[str] = []
    letters = words = 0
    previous_paragraph = None
    for sentence in sentences:
        if previous_paragraph != sentence.paragraph:
            pending.clear()
            letters = words = 0
            previous_paragraph = sentence.paragraph
        pending.append(sentence.text)
        letters += sum(c.isalpha() for c in sentence.text)
        words += sum(1 for _ in WORDS.finditer(sentence.text))
        if letters < MIN_PASSAGE_LETTERS or words < 6:
            continue
        yield " ".join(pending)[:MAX_PASSAGE_CHARS]
        pending.clear()
        letters = words = 0


def classify(text: str, sentences: list[Sentence]) -> str:
    """Sample passages across the file to look for both English and other languages."""
    # Mixed text can evade sampling, especially when switches are short. The
    # whole-file reading complements the evenly spaced local readings.
    whole = _recognised(text)
    english = whole == "en"
    other = whole not in (None, "en")
    passages = list(_local_passages(sentences))
    n = len(passages)
    if n > MAX_LOCAL_PASSAGES:
        passages = [passages[i * (n - 1) // (MAX_LOCAL_PASSAGES - 1)] for i in range(MAX_LOCAL_PASSAGES)]
    for passage in passages:
        code = _recognised(passage)
        english |= code == "en"
        other |= code not in (None, "en")
        if english and other:
            return "mixed"

    if other:
        return "unsupported"
    return ENGLISH if whole == "en" else "unknown"
