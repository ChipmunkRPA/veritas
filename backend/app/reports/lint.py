"""Report lints (ARCHITECTURE §6.2 `lint.py`, §6.3 rule 7; PRD RP-01, RP-02; UX §7.2, §7.3 rule 10; BUILD_SPEC R-3).

- **Vocabulary** (RP-02). The banned terms "proof", "proves", "fake", "real", "certain" and "guaranteed", as whole words in any
  case, in extracted report text. Quoted snapshot values are exempt (RP-02 "outside quoted metadata"): the filename, case
  fields, detector and signal names, observed and reference values, metadata values, annotation bodies and the analyst's
  rationale are removed before the scan, so a method name such as "universal fake detector" or a metadata field does not count.
- **Lay summary** (RP-01 (1)). At most 300 words and a Flesch-Kincaid grade of at most 9. Words are letter runs (with
  apostrophes) and numerals; a numeral counts one syllable; syllables are counted by vowel groups after removing a silent final
  `e` / `es` / `ed`; sentences end at `.`, `!` or `?` followed by white space or the end.
"""
from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

BANNED = ("proof", "proves", "fake", "real", "certain", "guaranteed")
LAY_SUMMARY_MAX_WORDS = 300
LAY_SUMMARY_MAX_GRADE = 9.0
_BANNED = re.compile(r"(?<![\w-])(" + "|".join(BANNED) + r")(?![\w-])", re.IGNORECASE)
_WORD = re.compile(r"[A-Za-z]+(?:['’][A-Za-z]+)*|\d+(?:[.,]\d+)*")
_SENTENCE_END = re.compile(r"[.!?](?:\s|$)")
_SILENT = re.compile(r"(?:[^laeiouy]es|ed|[^laeiouy]e)$")
_VOWELS = re.compile(r"[aeiouy]{1,2}")


def _collapse(text: str) -> str:
    return " ".join(text.split())


def vocabulary(text: str, quoted: Iterable[str] = ()) -> list[str]:
    """Banned terms found in `text` after removing the quoted values (longest first), lower-cased, in order of appearance."""
    scanned = _collapse(text)
    for value in sorted({_collapse(q) for q in quoted if q and q.strip()}, key=len, reverse=True):
        scanned = scanned.replace(value, " ")
    return [m.group(1).lower() for m in _BANNED.finditer(scanned)]


def contexts(text: str, quoted: Iterable[str] = (), width: int = 50) -> list[str]:
    """Each banned-term hit with up to `width` characters either side, for tracing a lint failure to its sentence."""
    scanned = _collapse(text)
    for value in sorted({_collapse(q) for q in quoted if q and q.strip()}, key=len, reverse=True):
        scanned = scanned.replace(value, " ")
    return [scanned[max(0, m.start() - width):m.end() + width] for m in _BANNED.finditer(scanned)]


INTERNAL_PATTERNS = (  # DR P0-3: vocabulary of the build, not of the report
    r"\b(?:docs|backend|frontend|scripts|tests)/[\w./<>-]+",  # repository paths
    r"\bapp\.[a-z_][\w.]*(?::\w+)?",  # Python module paths such as an adapter's module:Class (L-7)
    r"\b[\w-]+\.(?:py|md|tsx?|ya?ml)\b",  # source and documentation files
    r"\bralph-loop\b",  # the build loop
    # N-19d (design review DS-19): the id of a validation slice, such as image/q>=90/side>1024/face<120 (a content type such
    # as image/jpeg is not one: a slice names a bucket)
    r"\b(?:image|video|audio|document)/(?:[\w<>=.-]+/)*(?:(?:q|side|face|res|dur|words)(?:[<>]=?\d|\d+-\d|-unknown)|noface|lossless)[\w<>=./-]*",
    r"\bD-\d{2}\b|\b(?:BUILD_SPEC|PRD|ADR)\b|\bP[0-2]-\d{1,2}\b|\badmission A\d+[a-z]?\b",  # decision, spec, review and admission ids
    r"\((?:[A-Z]-\d{1,2}[a-z]?)(?:[,;][^)]*)?\)",  # ticket references such as "(B-5, admission A4)"
    # L-7: a bare ticket id such as J-3c1, H-3 or V-5a. Client labels stay client wording: C-n is a crop of the evidence (D-26) and
    # L-01 a limitation of the report, written with two digits (N-19d: the tenth limitation, L-10, was read as a ticket).
    r"(?<![\w-])(?!C-|L-\d\d(?![\w-]))[A-Z]-[1-9]\d?(?:[a-z]\d?)?(?![\w-])",
    r"\b(?:ML|IF|VF|UX|DS) (?:§\s?\d[\d.]*|[A-Z]\d{1,2}\b)|\bfit ∪ cal\b",  # L-7: research-document sections and codes, the build's split algebra
    # raw enums: snake_case values, hypothesis keys such as H1_whole_content_synthesis, finding keys such as M4.no_claim
    r"\b(?:[a-z][a-z0-9]*|[A-Z]\d+)(?:_[a-z0-9]+)+\b|\b[A-Z]\d+\.[a-z][a-z0-9_]*\b",
    # N-19q (finding 10): the build's words for parts of the fusion model and a band key in words
    r"\bsub-fusers?\b|\bfuser artefacts?\b|\blane estimate\b|\blikely authentic\b",
)
_INTERNAL = re.compile("|".join(f"(?:{p})" for p in INTERNAL_PATTERNS))
# a detector id such as img.hf.ateeqq or specimen.codec_audio is a technical reference the report may print, even where a narrow
# column wraps it; a file name (.md, .py …) and a module path (app.…) are not
_DETECTOR_ID = re.compile(r"(?<![\w.])(?!app\.)[a-z][a-z0-9]*\.(?!(?:py|md|tsx?|ya?ml)\b)[a-z0-9_]+(?:\.[a-z0-9_]+)*\b")


def internal_terms(text: str, quoted: Iterable[str] = ()) -> list[str]:
    """Internal vocabulary in client text (DR P0-3; BUILD_SPEC J-8, L-7): repository and module paths, file names, the build loop's
    name, decision, spec, review and ticket ids (in brackets or bare), research-document sections, the build's split algebra and
    raw enums. Quoted values and detector ids are removed first. The matches, in order."""
    scanned = _collapse(text)
    for value in sorted({_collapse(q) for q in quoted if q and q.strip()}, key=len, reverse=True):
        scanned = scanned.replace(value, " ")
        if "_" in value or "/" in value:
            # A PDF may wrap a published identifier in the middle of a token.
            pattern = r"(?<![\w/])" + r"\s*".join(re.escape(c) for c in value if not c.isspace()) + r"(?![\w/])"
            scanned = re.sub(pattern, " ", scanned)
    return [m.group(0) for m in _INTERNAL.finditer(_DETECTOR_ID.sub(" ", scanned))]


def syllables(word: str) -> int:
    w = word.lower()
    if not w[:1].isalpha():
        return 1
    if len(w) <= 3:
        return 1
    w = _SILENT.sub("", w)
    w = w[1:] if w.startswith("y") else w
    return max(1, len(_VOWELS.findall(w)))


def words(text: str) -> list[str]:
    return _WORD.findall(text)


def sentences(text: str) -> int:
    stripped = text.strip()
    if not stripped:
        return 0
    return len(_SENTENCE_END.findall(stripped)) + (0 if stripped[-1] in ".!?" else 1)


def fk_grade(text: str) -> float:
    """Flesch-Kincaid grade level: 0.39 · words / sentences + 11.8 · syllables / words − 15.59."""
    ws = words(text)
    n = sentences(text)
    if not ws or n == 0:
        return 0.0
    return 0.39 * len(ws) / n + 11.8 * sum(syllables(w) for w in ws) / len(ws) - 15.59


@dataclass(frozen=True)
class LaySummary:
    words: int
    grade: float

    @property
    def ok(self) -> bool:
        return self.words <= LAY_SUMMARY_MAX_WORDS and self.grade <= LAY_SUMMARY_MAX_GRADE


def lay_summary(text: str) -> LaySummary:
    return LaySummary(words=len(words(text)), grade=round(fk_grade(text), 2))
