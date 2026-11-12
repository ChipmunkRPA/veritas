"""Compare two reviewer-supplied text versions without a model or persistence."""

from __future__ import annotations

import hashlib
import re
from difflib import SequenceMatcher
from email import policy
from email.parser import BytesParser
from pathlib import Path
from tempfile import TemporaryDirectory

from app.detectors.text.extract import _normalise, extract, text_hash

SCHEMA = "chipmunk.document-revision-comparison/1"
TOKENS = re.compile(r"\w+(?:[’'-]\w+)*|[^\w\s]")
MAX_BYTES = 1024 * 1024
MAX_CHARS = 30_000
MAX_TOKENS = 2_000
ROLES = ("body", "quoted", "signature")
SCOPE = "Compares words and punctuation in the decoded text. Email body, quoted text and signatures are mapped separately."
INTERPRETATION = "This comparison locates changes between the supplied versions. The revision sequence and authorship require independent evidence."


def tokens(text, offsets):
    return [(item.group(), item.start() + start, item.end() + start)
            for start, end in offsets for item in TOKENS.finditer(text[start:end])]


def differences(earlier, later):
    """Build half-open source ranges for paired changes and their exact token extents."""
    edits = []
    for action, a, b, c, d in SequenceMatcher(None, [row[0] for row in earlier], [row[0] for row in later],
                                               autojunk=False).get_opcodes():
        if action == "equal":
            continue
        if edits and a - edits[-1][1] <= 2 and c - edits[-1][3] <= 2:
            old_a, _, old_c, _, changed_before, changed_after = edits[-1]
            edits[-1] = (old_a, b, old_c, d, [*changed_before, *earlier[a:b]], [*changed_after, *later[c:d]])
        else:
            edits.append((a, b, c, d, earlier[a:b], later[c:d]))

    def bounds(rows, start, stop):
        if start < stop:
            return [rows[start][1], rows[stop - 1][2]]
        point = rows[start][1] if start < len(rows) else (rows[-1][2] if rows else None)
        return [point, point] if point is not None else None

    return [{"earlier": bounds(earlier, a, b), "later": bounds(later, c, d),
             "earlier_changed_tokens": [[u, v] for _, u, v in old], "later_changed_tokens": [[u, v] for _, u, v in new]}
            for a, b, c, d, old, new in edits]


def _part_id(message, target, parent="1"):
    if message is target:
        return parent
    for i, child in enumerate(message.iter_parts(), 1):
        if found := _part_id(child, target, f"{parent}.{i}"):
            return found
    return None


def _source(path: Path, data: bytes, extension: str):
    if extension == ".eml":
        message = BytesParser(policy=policy.default).parsebytes(data)
        if not any(field in message for field in ("From", "To", "Subject", "Date", "Message-ID", "Content-Type", "MIME-Version")):
            raise ValueError("Choose an .eml file with email headers and a plain-text body.")
        body = message.get_body(preferencelist=("plain", "html"))
        if body is None or body.get_content_type() != "text/plain":
            raise ValueError("Choose an email with a readable plain-text body.")
        if body.get_filename() is not None or body.get_content_disposition() == "attachment":
            raise ValueError("The selected plain-text part is an attachment. A plain-text message body is required.")
        mime_part = _part_id(message, body)
        decoded = body.get_content()
    else:
        mime_part = None
    path.write_bytes(data)
    doc = extract(path, extension)
    if extension == ".eml":
        if not mime_part or doc.meta["email"]["body_part"] != mime_part:
            raise ValueError("The selected email body did not match the source map.")
    else:
        decoded = data.decode(doc.meta["encoding"], errors="strict")
    if not isinstance(decoded, str) or "\ufffd" in decoded or "\x00" in decoded:
        raise ValueError("The source text could not be decoded without replacement characters.")
    if len(decoded) > MAX_CHARS or not doc.source_map:
        raise ValueError(f"Each version needs readable text and at most {MAX_CHARS:,} decoded characters.")
    offsets = {role: [] for role in ROLES}
    previous = None
    for block in doc.source_map:
        role = block.get("quote_role", "body")
        ranges = block.get("source_ranges")
        if role not in offsets or not ranges or (role != previous and offsets[role]):
            raise ValueError("Each text, quote and signature section needs a contiguous source role.")
        a, b = ranges[0][2], ranges[-1][3]
        if (not 0 <= a < b <= len(decoded) or block.get("mime_part") != mime_part
                or _normalise(decoded[a:b])[0] != doc.text[block["start"]:block["end"]]):
            raise ValueError("Decoded text positions could not be mapped to the original.")
        offsets[role].append((a, b))
        previous = role
    role_tokens = {role: tokens(decoded, ranges) for role, ranges in offsets.items()}
    if sum(len(rows) for rows in role_tokens.values()) > MAX_TOKENS:
        raise ValueError(f"Each version can contain up to {MAX_TOKENS:,} words and punctuation tokens.")
    record = {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data), "decoded_text_sha256": text_hash(decoded),
              "decoded_characters": len(decoded), "mime_part": mime_part}
    return decoded, role_tokens, record


def _excerpt(text: str, extent: list[int] | None, changed_tokens: list[list[int]]) -> dict | None:
    if extent is None:
        return None
    start, end = extent
    return {"start": start, "end": end, "text": text[start:end], "changed_tokens": changed_tokens}


def compare(earlier: bytes, later: bytes, extension: str) -> dict:
    if extension not in {".eml", ".txt"}:
        raise ValueError("Choose two .eml files or two .txt files.")
    if any(not 0 < len(item) <= MAX_BYTES for item in (earlier, later)):
        raise ValueError("Each version must be nonempty and at most 1 MiB.")
    with TemporaryDirectory(prefix="document-revision-") as staging:
        directory = Path(staging)
        old_text, old_tokens, old_source = _source(directory / f"earlier{extension}", earlier, extension)
        new_text, new_tokens, new_source = _source(directory / f"later{extension}", later, extension)
    changes = []
    for role in ROLES:
        for change in differences(old_tokens[role], new_tokens[role]):
            before = _excerpt(old_text, change["earlier"], change["earlier_changed_tokens"])
            after = _excerpt(new_text, change["later"], change["later_changed_tokens"])
            changes.append({"id": f"C{len(changes) + 1:02d}", "role": role, "earlier": before, "later": after})
    return {"schema": SCHEMA, "format": "email" if extension == ".eml" else "plain_text",
            "coordinate_space": "decoded_source_unicode_codepoint", "earlier": old_source, "later": new_source,
            "changes": changes, "scope": SCOPE, "interpretation": INTERPRETATION}
