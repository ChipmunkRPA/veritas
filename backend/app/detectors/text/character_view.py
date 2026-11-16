"""Derive a second English reading with reversible, limited character changes."""

from __future__ import annotations

import re

from app.evidence.document_invisible_characters import IN_WORD, LOOKALIKES, _candidates, _lookalikes

SCHEMA = "english-writing-character-view/1"
_TOKEN = re.compile(r"\S+")


def _protected(token: str) -> bool:
    return "@" in token or token.casefold().lstrip("([<{‘\"'").startswith(("https://", "http://", "www."))


def derive(text: str) -> tuple[str, list[tuple[int, str]]]:
    """Return the alternate reading and source offsets of its exact substitutions."""
    if not any(ch in IN_WORD or ord(ch) in LOOKALIKES for ch in text if not ch.isascii()):
        return text, []
    excluded = {i for match in _TOKEN.finditer(text) if _protected(match[0]) for i in range(match.start(), match.end())}
    omitted = {pos for pos, ch in _candidates(text) if ch in IN_WORD and pos not in excluded}
    offsets = [pos for pos in range(len(text)) if pos not in omitted]
    intermediate = "".join(text[pos] for pos in offsets)
    substitute = {offsets[pos]: LOOKALIKES[ord(ch)] for pos, ch in _lookalikes(intermediate) if offsets[pos] not in excluded}
    changes = sorted([(pos, "") for pos in omitted] + list(substitute.items()))
    return "".join(substitute.get(pos, text[pos]) for pos in offsets), changes
