"""Template rendering for findings (ARCHITECTURE §4.6; EA-07). Template set `tpl-2.17` (`engine.TEMPLATE_SET`).

- Templates live in `templates/<family>/<scope>.j2`; a missing scope falls back to `<family>/global.j2`. Shared sentences
  are macros in `templates/_macros.j2`.
- **One sentence per rendered line.** Each line starts with the marker `⟦id|id…⟧` written by the `cite(...)` global, so
  every sentence carries at least one evidence id. A line without a marker fails rendering (`SynthesisError`).
- `StrictUndefined`: a template that reads a missing value fails instead of printing an empty slot.
"""
from __future__ import annotations

import re
from collections.abc import Iterable
from functools import cache
from pathlib import Path
from typing import Any

import jinja2

TEMPLATES = Path(__file__).parent / "templates"
_LINE = re.compile(r"^⟦([^⟧]+)⟧\s*(\S.*)$")


class SynthesisError(ValueError):
    """A template produced text that is not a sentence with evidence ids."""


def _flatten(ids: Iterable[Any]) -> list[str]:
    out: list[str] = []
    for item in ids:
        if isinstance(item, list | tuple):
            out += _flatten(item)
        elif item:
            out.append(str(item))
    return out


def cite(*ids: Any) -> str:
    flat = list(dict.fromkeys(_flatten(ids)))
    if not flat:
        raise SynthesisError("a sentence needs at least one evidence id")
    if any("|" in i or "⟧" in i for i in flat):
        raise SynthesisError(f"evidence ids may not contain '|' or '⟧': {flat}")
    return "⟦" + "|".join(flat) + "⟧"


@cache
def environment() -> jinja2.Environment:
    env = jinja2.Environment(loader=jinja2.FileSystemLoader(str(TEMPLATES)), undefined=jinja2.StrictUndefined,
                             autoescape=False, trim_blocks=True, lstrip_blocks=True)
    env.globals["cite"] = cite
    return env


def template_name(family: str, scope: str) -> str:
    name = f"{family}/{scope}.j2"
    return name if (TEMPLATES / name).is_file() else f"{family}/global.j2"


def render(name: str, **context: Any) -> list[dict[str, Any]]:
    """Render a template into `[{text, evidence_ids}]`."""
    text = environment().get_template(name).render(**context)
    sentences = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        match = _LINE.match(line)
        if match is None:
            raise SynthesisError(f"template {name} produced text without evidence ids: {line[:80]!r}")
        sentences.append({"text": re.sub(r"\s+", " ", match.group(2)).strip(),
                          "evidence_ids": list(dict.fromkeys(i for i in match.group(1).split("|") if i))})
    if not sentences:
        raise SynthesisError(f"template {name} produced no sentences")
    return sentences
