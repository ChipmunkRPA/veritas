"""Preserve RTF structure while removing directly hidden text runs."""

import re

from striprtf.striprtf import PATTERN, destinations, specialchars

_HIDE_WORD = re.compile(r"(?<!\\)(?:\\\\)*\\v(?![a-z])")
_LINK_FIELD = re.compile(r"((?<!\\)(?:\\\\)*\{\\field)(?=\{)", re.IGNORECASE)
_BOUNDARIES = {"par", "sect", "page", "line", "row", "cell", "nestcell", "\r", "\n"}


def visible_runs(rtf: str) -> str:
    """Apply direct visibility and retain only saved field display text."""
    # A delimiter after the control word stops the decoder from appending
    # a hyperlink's destination. The ordinary saved field result stays intact.
    rtf = _LINK_FIELD.sub(r"\1 ", rtf)
    if not _HIDE_WORD.search(rtf):
        return rtf

    parts = []
    kept = 0
    hidden = ignored = False
    stack = []
    for token in PATTERN.finditer(rtf):
        word, argument, hex_byte, symbol, brace, text = token.groups()
        if brace == "{":
            stack.append((hidden, ignored))
        elif brace == "}":
            hidden, ignored = stack.pop()
        elif word in destinations or symbol == "*":
            ignored = True
        elif not ignored:
            if word == "v":
                hidden = argument is None or int(argument) != 0
            elif word == "plain":
                hidden = False
            elif hidden and (text is not None or hex_byte or word == "u"
                             or word in specialchars and word not in _BOUNDARIES
                             or symbol in specialchars and symbol not in _BOUNDARIES):
                if kept < token.start():
                    parts.append(rtf[kept:token.start()])
                kept = token.end()
    parts.append(rtf[kept:])
    return "".join(parts)
