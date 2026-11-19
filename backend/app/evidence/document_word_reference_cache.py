# SPDX-License-Identifier: Apache-2.0
"""Compare saved Word REF text against an exact local bookmark without updating fields."""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from pathlib import Path

from docx.oxml.ns import qn
from docx.text.paragraph import Paragraph
from docx.text.run import Run

from app.detectors.text import word_saved_fields
from app.detectors.text.strict_word import open_document
from app.evidence import document_word_reference_cache_binding as binding
from app.storage.files import check_docx_safe

_RUN, _PROPERTIES, _TEXT = (qn(f"w:{name}") for name in ("r", "rPr", "t"))
_SIMPLE, _CODE, _MARKER = (qn(f"w:{name}") for name in ("fldSimple", "instrText", "fldChar"))
_BOOKMARK_START, _BOOKMARK_END = (qn(f"w:{name}") for name in ("bookmarkStart", "bookmarkEnd"))
_ID, _NAME, _INSTRUCTION, _MARKER_TYPE, _LOCKED = (qn(f"w:{name}") for name in ("id", "name", "instr", "fldCharType", "fldLock"))
_BREAKS_READING = ".//w:ins | .//w:del | .//w:moveFrom | .//w:moveTo | .//w:pPrChange | .//w:rPrChange"
_VISIBLE_RUN = "./w:rPr/w:vanish | ./w:rPr/w:specVanish | ./w:rPr/w:webHidden"
_REFERENCE = re.compile(r'\s*REF\s+(?:"([_A-Za-z][_A-Za-z0-9]{0,39})"|([_A-Za-z][_A-Za-z0-9]{0,39}))(?:\s+(.+?))?\s*', re.I)
_SWITCH = re.compile(r"\\h|\\\*\s+(?:MERGEFORMAT|CHARFORMAT)", re.I)
_RESERVED_NAMES = {"control", "data", "embed", "htmlcontrol", "subscriber"}
_UNLOCKED = {"0", "false", "off"}
_MAX_FILE_BYTES = 25 * 1024 * 1024
_MAX_RUNS = 32


def _target_name(code):
    if len(code) > 1024 or (matched := _REFERENCE.fullmatch(code)) is None:
        return None
    name = matched[1] or matched[2]
    if name.casefold() in _RESERVED_NAMES:
        return None
    rest = (matched[3] or "").strip()
    switches = set()
    while rest:
        switch = _SWITCH.match(rest)
        if switch is None or (switch.end() < len(rest) and not rest[switch.end()].isspace()):
            return None
        keyword = switch[0].casefold().split()[0]
        if keyword in switches:
            return None
        switches.add(keyword)
        rest = rest[switch.end():].strip()
    return name


def _plain_run(paragraph, run):
    if run.tag != _RUN or any(child.tag not in {_PROPERTIES, _TEXT} for child in run):
        return None
    if run.xpath(_VISIBLE_RUN) or not word_saved_fields.style_allows_visible_runs(Run(run, paragraph).style):
        return None
    value = "".join(child.text or "" for child in run.findall(_TEXT))
    return value if not value or value.isprintable() else None


def _saved_runs(paragraph, runs):
    if not 1 <= len(runs) <= _MAX_RUNS:
        return None
    values = [_plain_run(paragraph, run) for run in runs]
    if None in values:
        return None
    text = "".join(values)
    return text if 0 < len(text) <= binding.MAX_VALUE_CHARS and text.strip() else None


def _field_marker(run, name):
    if run.tag == _RUN and len(run) == len(run.findall(_PROPERTIES)) + 1:
        marker = run.find(_MARKER)
        if marker is not None and marker.get(_MARKER_TYPE) == name:
            return marker
    return None


def _reference(paragraph, nodes, index):
    node = nodes[index]
    if node.tag == _SIMPLE:
        if (node.get(_LOCKED, "false") in _UNLOCKED and (target := _target_name(node.get(_INSTRUCTION, "")))
                and (value := _saved_runs(paragraph, list(node))) is not None):
            return index + 1, target, value, "simple"
        return None
    begin = _field_marker(node, "begin")
    if begin is None or begin.get(_LOCKED, "false") not in _UNLOCKED:
        return None
    code = []
    cursor = index + 1
    while cursor < len(nodes) and len(code) < 8:
        run = nodes[cursor]
        pieces = [child for child in run if child.tag != _PROPERTIES]
        if run.tag != _RUN or not pieces or any(child.tag != _CODE for child in pieces):
            break
        if len(code) + len(pieces) > 8:
            return None
        code.extend(child.text or "" for child in pieces)
        cursor += 1
    if (not code or cursor >= len(nodes) or _field_marker(nodes[cursor], "separate") is None
            or not (target := _target_name("".join(code)))):
        return None
    first = cursor + 1
    for end in range(first, min(len(nodes), first + _MAX_RUNS + 1)):
        run = nodes[end]
        if _field_marker(run, "end") is not None:
            if (value := _saved_runs(paragraph, nodes[first:end])) is not None:
                return end + 1, target, value, "complex"
            return None
        if run.tag != _RUN or run.find(_MARKER) is not None:
            return None
    return None


def _address(value, offset, body_index, child_index):
    first = offset + len(value) - len(value.lstrip())
    text = value.strip()
    return {"body_index": body_index, "child_index": child_index, "range": [first, first + len(text)], "text": text}


def _spanning_fields(body):
    """Exclude all blocks inside a field that crosses paragraph boundaries."""
    blocked = set()
    depth = 0
    for block in body:
        before = depth
        for marker in block.iter(_MARKER):
            match marker.get(_MARKER_TYPE):
                case "begin":
                    depth += 1
                case "end" if depth:
                    depth -= 1
                case "separate" if depth:
                    pass
                case _:
                    return None
        if before or depth:
            blocked.add(block)
    return blocked if depth == 0 else None


def _paragraph(paragraph, body_index, names, starts, ends):
    nodes = list(paragraph._p)
    index, offset = 0, 0
    bookmark = None
    targets, reference = {}, None
    allowed_metadata = {qn("w:pPr"), qn("w:proofErr")}
    while index < len(nodes):
        item = nodes[index]
        if item.tag == _BOOKMARK_START:
            name, identifier = item.get(_NAME, ""), item.get(_ID)
            if (bookmark is not None or not name or identifier is None or names[name.casefold()] != 1
                    or starts[identifier] != 1 or ends[identifier] != 1
                    or any(qn(f"w:{key}") in item.attrib for key in ("colFirst", "colLast"))):
                return {}, None
            bookmark = name, identifier, index, offset
        elif item.tag == _BOOKMARK_END:
            if bookmark is None or item.get(_ID) != bookmark[1]:
                return {}, None
            name, _, first, start = bookmark
            if (value := _saved_runs(paragraph, nodes[first + 1:index])) is None:
                return {}, None
            targets[name] = _address(value, start, body_index, first)
            bookmark = None
        elif item.tag in {_SIMPLE, _RUN}:
            if (value := _plain_run(paragraph, item)) is not None:
                offset += len(value)
            else:
                if bookmark is not None or reference is not None or (_reference_data := _reference(paragraph, nodes, index)) is None:
                    return {}, None
                following, name, value, shape = _reference_data
                reference = name, shape, _address(value, offset, body_index, index)
                offset += len(value)
                index = following - 1
        elif item.tag not in allowed_metadata:
            return {}, None
        if offset > binding.MAX_PARAGRAPH_CHARS:
            return {}, None
        index += 1
    return (targets, reference) if bookmark is None else ({}, None)


def capture(document):
    body = document.element.body
    if (len(body) > binding.MAX_BODY_BLOCKS or body.xpath(_BREAKS_READING)
            or not word_saved_fields.defaults_allow_visible_runs(document) or (blocked := _spanning_fields(body)) is None):
        return None
    starts = Counter(node.get(_ID) for node in body.iter(_BOOKMARK_START))
    if not starts:
        return None
    names = Counter(node.get(_NAME, "").casefold() for node in body.iter(_BOOKMARK_START))
    ends = Counter(node.get(_ID) for node in body.iter(_BOOKMARK_END))
    targets, refs = {}, []
    for index, block in enumerate(body):
        if block.tag != qn("w:p") or block in blocked or len(block) > binding.MAX_CHILDREN:
            continue
        paragraph = Paragraph(block, document)
        if word_saved_fields.paragraph_allows_visible_runs(paragraph):
            found, reference = _paragraph(paragraph, index, names, starts, ends)
            targets.update(found)
            if reference:
                refs.append(reference)

    findings = []
    checked = 0
    for name, shape, field in refs:
        target = targets.get(name)
        if target is None or target["body_index"] == field["body_index"]:
            continue
        checked += 1
        if binding.plain_text(field["text"]) != binding.plain_text(target["text"]):
            findings.append({"field": field, "target": target, "record_shape": shape})
    if not findings:
        return None
    return {"schema": binding.SCHEMA, "checked": checked, "mismatches": len(findings),
            "findings": findings[:binding.MAX_FINDINGS], "authorship_confidence_contribution": 0}


def inspect(path: Path, *, source_sha256: str):
    if not 0 < path.stat().st_size <= _MAX_FILE_BYTES:
        return None
    with path.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != source_sha256:
            raise ValueError("Submitted Word file differs from the cross-reference source")
    check_docx_safe(path)
    return capture(open_document(path))
