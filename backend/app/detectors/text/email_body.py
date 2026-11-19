# SPDX-License-Identifier: Apache-2.0
"""Select the saved email body without promoting attachments to writing input."""

from __future__ import annotations

SELECTION_KEY = "body_selection"
SELECTION_POLICY = "readable-email-body/1"
READER_KEY = "email_body_reader"


def _candidates(part, path="1", alternative=None):
    if part.is_attachment() or part.get_filename() is not None:
        return
    content_type = part.get_content_type()
    if content_type in {"text/plain", "text/html"}:
        yield path, part, alternative
        return
    if part.get_content_maintype() != "multipart" or not part.is_multipart():
        return
    children = list(enumerate(part.iter_parts(), 1))
    if content_type == "multipart/related":
        start = part.get_param("start")
        roots = [(index, child) for index, child in children if start and child["Content-ID"] == start]
        if start and len(roots) != 1:
            alternative = None
        for index, child in (roots or children)[:1]:
            yield from _candidates(child, f"{path}.{index}", alternative)
        return
    for index, child in children:
        yield from _candidates(child, f"{path}.{index}", path if content_type == "multipart/alternative" else None)


def select(message):
    """Return the selected MIME path and part, plus whether it differs from legacy selection."""
    candidates = list(_candidates(message))
    plain = next((row for row in candidates if row[1].get_content_type() == "text/plain"), None)
    html = [row for row in candidates if row[1].get_content_type() == "text/html"]
    selected = plain or (html[0] if html else None)
    if plain is not None and plain[2] is not None:
        alternatives = [row for row in html if row[2] == plain[2]]
        if len(alternatives) == 1:
            raw = plain[1].get_content()
            if isinstance(raw, str) and not raw.strip():
                selected = alternatives[0]
    if selected is None:
        return None, None, False
    path, part, _ = selected
    return path, part, part is not message.get_body(preferencelist=("plain", "html"))
