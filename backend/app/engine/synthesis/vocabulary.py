"""Banned-term lint for generated findings and their templates (PRD RP-02 ∪ IF §6.2 ∪ UX §7.2; D-16; ARCHITECTURE §4.6).

- **Scope.** Titles, sentences, statements and dissent notes of generated findings, and the literal text of the templates.
- **Exempt.** Quoted material (“…” or "…": detector observations, metadata, card text) and exact third-party names passed
  as `exempt` (detector names such as "dima806 deepfake_vs_real_image_detection" are method names, UX §7.2).
- **Allowed phrases.** "not established" (D-23 requires it), "not confirmed" (UX §7.7 legend), and "genuine" only for a
  known-label population ("genuine originals", "genuine photos", …; UX §7.2).
"""
from __future__ import annotations

import re
from collections.abc import Iterable

BANNED: tuple[tuple[str, str], ...] = (
    ("proof / proves", r"\bproofs?\b|\bprov(?:e|es|ed|en|ing)\b"),
    ("fake", r"\bfakes?\b"),
    ("deepfake", r"\bdeepfakes?\b"),
    ("real", r"\breal\b"),
    ("genuine as a conclusion", r"\bgenuine\b(?!\s+(?:originals?|photos?|images?|videos?|recordings?|documents?|test))"),
    ("authentic", r"\bauthentic\b"),
    ("certain", r"\bcertain(?:ly|ty)?\b"),
    ("guaranteed", r"\bguarantee[ds]?\b"),
    ("tampered", r"\btampered\b"),
    ("confirms", r"(?<!not )\bconfirm(?:s|ed|ing)?\b"),
    ("determines", r"\bdetermin(?:es|ed|ing)\b"),
    ("establishes", r"(?<!not )\bestablish(?:es|ed|ing)?\b"),
    ("match", r"\bmatch(?:es|ed|ing)?\b"),
    ("definitely / clearly", r"\bdefinitely\b|\bclearly\b"),
    ("100 % / 0 %", r"\b100\s?%|(?<![\d.])0\s?%"),
    ("AI detected", r"\bAI detected\b"),
    ("the edited area / fake region", r"\bthe edited area\b|\bfake region\b"),
    ("weak or limited evidence", r"\b(?:weak|limited) evidence\b"),
    ("anthropomorphism", r"\bthe (?:model|detector|classifier) (?:thinks|believes|sees)\b"),
    ("clean / safe / verified / trusted", r"\b(?:clean|safe|verified|trusted)\b"),
    ("no manipulation", r"\bno manipulation\b"),
    ("critical / high risk", r"\bcritical\b|\bhigh risk\b"),
    ("confidence as a percentage", r"\bconfidence\s+\d"),
)
_COMPILED = tuple((label, re.compile(pattern, re.IGNORECASE)) for label, pattern in BANNED)
_QUOTED = re.compile(r"“[^”]*”|\"[^\"]*\"")
_JINJA = re.compile(r"\{\{.*?\}\}|\{%.*?%\}|\{#.*?#\}", re.DOTALL)


def violations(text: str, *, exempt: Iterable[str] = ()) -> list[str]:
    """Labels of the banned terms in `text`, after removing quoted material and exempt names."""
    body = _QUOTED.sub(" ", text)
    for term in sorted({t for t in exempt if t}, key=len, reverse=True):
        body = body.replace(term, " ")
    return [label for label, pattern in _COMPILED if pattern.search(body)]


def template_text(source: str) -> str:
    """The literal prose of a template: expressions, statements and comments removed."""
    return _JINJA.sub(" ", source)
