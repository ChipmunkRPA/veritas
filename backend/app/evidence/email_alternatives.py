"""Bounded, passive comparison of two decoded email body alternatives."""

from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher
from email import policy
from email.parser import BytesParser
from hashlib import sha256
from html import unescape
from html.parser import HTMLParser

from app.core.canonical import canonical_hash

MAX_FILE = 1_000_000
MAX_BODY = 64_000
MAX_TOKENS = 1200
TOKEN = re.compile(r"(?P<url>https?://[^\s<>]+)|(?P<word>[^\W_]+(?:[’\x27-][^\W_]+)*)", re.I)
KEY = "email_body_alternatives"
COMMITMENT = "email_body_alternatives_sha256"
SCHEMA = "chipmunk.email-body-alternatives/1"
MAX_DIFFERENCES = 6
MAX_EXCERPT_WORDS = 48
MAX_EXCERPT_CHARS = 600
MAX_FRAGMENTS = 96
SIDES = ("plain", "html")
PIN = re.compile(r"[a-f0-9]{64}")
GRAMMAR = frozenset(
    [
        "a",
        "an",
        "as",
        "at",
        "be",
        "but",
        "by",
        "can",
        "could",
        "did",
        "do",
        "for",
        "from",
        "had",
        "has",
        "have",
        "he",
        "how",
        "i",
        "if",
        "in",
        "is",
        "it",
        "may",
        "might",
        "of",
        "on",
        "or",
        "our",
        "she",
        "so",
        "than",
        "that",
        "the",
        "their",
        "them",
        "then",
        "there",
        "these",
        "they",
        "this",
        "to",
        "us",
        "was",
        "we",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "will",
        "with",
        "would",
        "you",
        "your",
    ]
)
OPT_OUT = re.compile(
    r"^\s*(?:to )?unsubscribe from (?:these|this) (?:notifications|mailing list),? (?:visit|use) (?:https?://\S+|\{SERVICE-OPT-OUT-LINK\}\.?)\s*$",
    re.I,
)
QUOTE = re.compile(r"^\s*(?:>|On .{1,300} wrote:\s*$|-----Original Message-----)", re.I)
SAFE_INLINE = {"strong", "b", "em", "i", "span", "a", "small", "u"}
SAFE_BLOCK = {"p", "div", "li", "ul", "ol", "body", "html", "head", "title", "blockquote", "address"}
VOID = {"br", "hr", "img", "meta"}
IGNORE = {"head", "title", "blockquote", "address"}


def sha(text):
    return sha256(text.encode()).hexdigest()


def body_pair(data):
    if not 0 < len(data) <= MAX_FILE:
        raise ValueError("file outside bounded scope")
    try:
        message = BytesParser(policy=policy.default).parsebytes(data)
    except RecursionError as error:
        raise ValueError("MIME nesting exceeds the supported scope") from error
    if message.defects or any(
        len(message.get_all(name, [])) > 1 for name in ("Content-Type", "Content-Disposition")
    ):
        raise ValueError("malformed MIME tree")
    root = "1"
    container = message
    if container.get_content_type() == "multipart/mixed":
        children = list(container.iter_parts())
        if not children or any(x.get_content_disposition() != "attachment" for x in children[1:]):
            raise ValueError("ambiguous mixed body")
        root = "1.1"
        container = children[0]
    if (
        container.get_content_type() != "multipart/alternative"
        or container.defects
        or container.get_content_disposition() not in (None, "inline")
    ):
        raise ValueError("no unique direct alternative body")
    parts = list(container.iter_parts())
    if [p.get_content_type() for p in parts] != ["text/plain", "text/html"] or message.get_body(
        preferencelist=("plain", "html")
    ) is not parts[0]:
        raise ValueError("unexpected alternative body parts")
    for p in parts:
        if (
            p.defects
            or p.get_filename() is not None
            or p.get_content_disposition() not in (None, "inline")
            or any(
                len(p.get_all(name, [])) > 1
                for name in ("Content-Type", "Content-Disposition", "Content-Transfer-Encoding")
            )
            or p.get("Content-Transfer-Encoding", "7bit").lower()
            not in ("7bit", "8bit", "base64", "quoted-printable", "binary")
        ):
            raise ValueError("unexpected body disposition")
        if p.get_param("format") not in (None, "fixed") or p.get_content_charset() not in (
            "utf-8",
            "us-ascii",
            "iso-8859-1",
            "windows-1252",
        ):
            raise ValueError("unsupported body format or charset")
    output = {}
    for i, (name, part) in enumerate(zip(("plain", "html"), parts, strict=True), 1):
        # Decode only each admitted leaf. A decode defect added by the parser invalidates it.
        raw = part.get_payload(decode=True)
        if part.defects or raw is None or len(raw) > MAX_BODY:
            raise ValueError("malformed or large MIME part")
        text = raw.decode(part.get_content_charset(), errors="strict")
        if "\ufffd" in text or "\0" in text or text != part.get_content():
            raise ValueError("unclear decoded characters")
        output[name] = (f"{root}.{i}", text)
    return output


class Visible(HTMLParser):
    def __init__(self, text):
        super().__init__(convert_charrefs=False)
        self.text = text
        self.lines = [0] + [m.end() for m in re.finditer("\n", text)]
        self.stack = []
        self.out = []
        self.open_anchor = None
        self.anchors = []
        self.trailer = False

    def emit(self, char, rng):
        self.out.append((char, rng))

    def source_offset(self):
        line, col = self.getpos()
        return self.lines[line - 1] + col

    def visible(self):
        return not IGNORE.intersection(self.stack)

    def block(self):
        if self.visible():
            self.emit("\n", None)

    def handle_starttag(self, tag, attrs):
        if len(self.stack) >= 32 or tag not in SAFE_INLINE | SAFE_BLOCK | VOID:
            raise ValueError("unsupported HTML element")
        values = dict(attrs)
        allowed = {
            "a": {"href", "title"},
            "html": {"lang", "dir"},
            "meta": {"charset"},
            "img": {"src", "alt", "width", "height"},
            "ol": {"start"},
            "li": {"value"},
        }.get(tag, set())
        if len(values) != len(attrs) or not set(values) <= allowed:
            raise ValueError("opaque HTML attributes")
        if tag == "img" and values.get("alt") != "":
            raise ValueError("HTML image text is outside scope")
        if tag in SAFE_BLOCK | {"br", "hr"}:
            self.block()
        if tag in {"blockquote", "address"}:
            self.trailer = True
        if tag == "a":
            if self.open_anchor is not None:
                raise ValueError("nested links outside scope")
            self.open_anchor = (len(self.out), values.get("href") or "")
        if tag not in VOID:
            self.stack.append(tag)

    def handle_startendtag(self, tag, attrs):
        if tag not in VOID:
            raise ValueError("unexpected self-closing element")
        self.handle_starttag(tag, attrs)

    def handle_endtag(self, tag):
        if not self.stack or self.stack.pop() != tag:
            raise ValueError("unbalanced HTML element")
        if tag == "a":
            start, href = self.open_anchor
            self.anchors.append((start, len(self.out), href))
            self.open_anchor = None
        if tag in SAFE_BLOCK:
            self.block()

    def handle_data(self, text):
        if self.visible():
            if self.trailer and text.strip():
                raise ValueError("interleaved HTML quotation or signature")
            i = self.source_offset()
            for j, char in enumerate(text):
                self.emit(char, (i + j, i + j + 1))

    def entity(self, body):
        i = self.source_offset()
        value = unescape(body)
        if not self.visible():
            return
        if self.trailer and value.strip():
            raise ValueError("interleaved HTML quotation or signature")
        if self.text[i : i + len(body)] != body or len(value) != 1 or value == body or value == "\ufffd":
            raise ValueError("unsupported HTML entity")
        self.emit(value, (i, i + len(body)))

    def handle_entityref(self, name):
        self.entity("&" + name + ";")

    def handle_charref(self, name):
        self.entity("&#" + name + ";")

    def handle_pi(self, data):
        raise ValueError("unsupported HTML instruction")

    def handle_decl(self, decl):
        if decl.casefold() != "doctype html":
            raise ValueError("unsupported HTML declaration")


def strip_footer(chars, anchors):
    visible = "".join(c for c, _ in chars)
    lines = list(re.finditer(r"[^\r\n]+", visible))
    if not lines:
        return chars
    start, stop = lines[-1].span()
    candidate = visible[start:stop]
    for a, b, href in anchors:
        label = visible[a:b].strip().casefold()
        if (
            start <= a < b <= stop
            and label in {"preferences", "email preferences", "unsubscribe", "manage preferences"}
            and re.fullmatch(r"https?://[^\s<>]+", href, re.I)
        ):
            candidate = visible[start:a] + "{SERVICE-OPT-OUT-LINK}" + visible[b:stop]
            break
    return chars[:start] + chars[stop:] if OPT_OUT.fullmatch(candidate) else chars


def read_tokens(name, text):
    if name == "html":
        mapper = Visible(text)
        mapper.feed(text)
        mapper.close()
        if mapper.stack:
            raise ValueError("unbalanced HTML tail")
        chars = mapper.out
    else:
        chars = []
        pos = 0
        trailer = None
        for line in text.splitlines(keepends=True):
            stripped = line.lstrip()
            if stripped.strip() == "--" or (QUOTE.match(line) and not stripped.startswith(">")):
                trailer = "marked"
            elif stripped.startswith(">") and trailer is None:
                trailer = "quoted_lines"
            elif trailer == "quoted_lines" and stripped and not stripped.startswith(">"):
                raise ValueError("interleaved plain-text quotation")
            if trailer is None:
                chars.extend((c, (pos + i, pos + i + 1)) for i, c in enumerate(line))
            pos += len(line)
    chars = strip_footer(chars, mapper.anchors if name == "html" else ())
    surface = "".join(c for c, _ in chars)
    tokens = []
    for m in TOKEN.finditer(surface):
        if m.lastgroup == "url":
            continue
        key = unicodedata.normalize("NFKC", m.group().replace("’", "'")).casefold()
        ranges = []
        for _, span in chars[m.start() : m.end()]:
            if span is None:
                raise ValueError("unbound lexical token")
            if ranges and ranges[-1][1] == span[0]:
                ranges[-1][1] = span[1]
            else:
                ranges.append(list(span))
        tokens.append({"key": key, "source_ranges": ranges, "text": m.group()})
    if not 12 <= len(tokens) <= MAX_TOKENS:
        raise ValueError("body lexical scope")
    return tokens


def keys(tokens):
    return [x["key"] for x in tokens]


def contains(haystack, needle):
    return any(haystack[i : i + len(needle)] == needle for i in range(len(haystack) - len(needle) + 1))


def material(values, other):
    return (
        len(values) >= 5
        and len({value for value in values if value not in GRAMMAR}) >= 3
        and not contains(other, values)
    )


def locate(data):
    try:
        parts = body_pair(data)
        p, h = [read_tokens(name, text) for name, (_, text) in parts.items()]
    except (LookupError, UnicodeError, ValueError) as error:
        return {"status": "unsupported", "reason": str(error)}
    pkeys, hkeys = keys(p), keys(h)
    matcher = SequenceMatcher(None, pkeys, hkeys, autojunk=False)
    if 2 * matcher.ratio() < 1:
        return {"status": "unsupported", "reason": "insufficient common reply content"}
    # Common one- and two-token fragments inside a replacement do not split it.
    anchors = [entry for entry in matcher.get_matching_blocks() if entry.size >= 3]
    anchors.append(matcher.get_matching_blocks()[-1])
    a = c = 0
    found = []
    for b, d, length in anchors:
        if not (a < b or c < d):
            a, c = b + length, d + length
            continue
        left, right = pkeys[a:b], hkeys[c:d]
        relevant = {}
        if material(left, hkeys):
            relevant["plain"] = p[a:b]
        if material(right, pkeys):
            relevant["html"] = h[c:d]
        if relevant:
            found.append(relevant)
        a, c = b + length, d + length
    return {
        "status": "difference" if found else "equivalent",
        "parts": {
            name: {"id": mid, "sha256": sha(text), "characters": len(text)}
            for name, (mid, text) in parts.items()
        },
        "differences": found,
    }


def _excerpt(tokens):
    """Expose mapped, passive words only. Source spans can cross markup boundaries."""
    words, ranges = [], []
    for token in tokens[:MAX_EXCERPT_WORDS]:
        if (
            sum(map(len, words)) + len(words) + len(token["text"]) > MAX_EXCERPT_CHARS
            or len(ranges) + len(token["source_ranges"]) > MAX_FRAGMENTS
        ):
            break
        words.append(token["text"])
        ranges.extend(token["source_ranges"])
    if not words:
        return None
    return {
        "text": " ".join(words),
        "source_ranges": ranges,
        "token_count": len(words),
        "excerpt_truncated": len(words) != len(tokens),
    }


def review(data):
    """Build the frozen, confidence-neutral result without decoded HTML or URL destinations."""
    outcome = locate(data)
    result = {"schema": SCHEMA, "status": outcome["status"], "authorship_confidence_contribution": 0}
    if "parts" in outcome:
        ranges = []
        for diff in outcome["differences"][:MAX_DIFFERENCES]:
            excerpts = {side: _excerpt(tokens) for side, tokens in diff.items()}
            if all(excerpts.values()):
                ranges.append({"id": f"EML{len(ranges) + 1:02}", **excerpts})
        result.update(
            parts=outcome["parts"],
            differences=ranges,
            omitted_differences=len(outcome["differences"]) - len(ranges),
        )
    return result


def result_review(result):
    return ((result.get("manifest") or {}).get("sampling") or {}).get(KEY)


def validate_set(evidence, expected_hash):
    """Check the public shape and its independently signed result commitment."""
    coverage = evidence.get("coverage") or {}
    if KEY not in coverage and expected_hash is None:
        return None
    record = coverage.get(KEY)
    if (
        not isinstance(record, dict)
        or not isinstance(expected_hash, str)
        or canonical_hash(record) != expected_hash
        or evidence["source"].get("method") != "email"
        or record.get("schema") != SCHEMA
        or type(record.get("authorship_confidence_contribution")) is not int
        or record["authorship_confidence_contribution"] != 0
    ):
        raise ValueError("Email alternative evidence differs from the frozen result")
    common = {"schema", "status", "authorship_confidence_contribution"}
    status = record["status"]
    if status == "unsupported" and set(record) == common:
        return record
    if (
        status not in {"difference", "equivalent"}
        or set(record) != common | {"parts", "differences", "omitted_differences"}
        or not isinstance(record["parts"], dict)
        or set(record["parts"]) != set(SIDES)
        or not isinstance(record["differences"], list)
        or len(record["differences"]) > MAX_DIFFERENCES
        or type(record["omitted_differences"]) is not int
        or not 0 <= record["omitted_differences"] <= MAX_TOKENS
        or (status == "difference") != bool(record["differences"] or record["omitted_differences"])
    ):
        raise ValueError("Invalid email alternative review shape")
    selected = (coverage.get("email") or {}).get("body_part")
    if (
        selected not in {"1.1", "1.1.1"}
        or (coverage.get("email") or {}).get("body_content_type") != "text/plain"
    ):
        raise ValueError("Email alternative review must preserve the selected plain body")
    for index, side in enumerate(SIDES, 1):
        part = record["parts"][side]
        if (
            set(part) != {"id", "sha256", "characters"}
            or part["id"] != selected[:-1] + str(index)
            or not PIN.fullmatch(part["sha256"])
            or type(part["characters"]) is not int
            or not 0 < part["characters"] <= MAX_BODY
        ):
            raise ValueError("Invalid decoded MIME part binding")
    offsets = dict.fromkeys(SIDES, 0)
    for index, change in enumerate(record["differences"], 1):
        sides = set(change) - {"id"}
        if not sides or not sides <= set(SIDES) or change.get("id") != f"EML{index:02}":
            raise ValueError("Invalid email alternative location")
        for side in sides:
            excerpt = change[side]
            words = excerpt["text"].split(" ")
            ranges = excerpt["source_ranges"]
            if (
                set(excerpt) != {"text", "source_ranges", "token_count", "excerpt_truncated"}
                or type(excerpt["excerpt_truncated"]) is not bool
                or type(excerpt["token_count"]) is not int
                or excerpt["token_count"] != len(words)
                or not 1 <= len(words) <= MAX_EXCERPT_WORDS
                or len(excerpt["text"]) > MAX_EXCERPT_CHARS
                or not 1 <= len(ranges) <= MAX_FRAGMENTS
                or any(not (match := TOKEN.fullmatch(word)) or match.lastgroup != "word" for word in words)
            ):
                raise ValueError("Email alternative excerpts must be bounded passive text")
            for start, end in ranges:
                if (
                    type(start) is not int
                    or type(end) is not int
                    or not offsets[side] <= start < end <= record["parts"][side]["characters"]
                ):
                    raise ValueError("Invalid decoded MIME source range")
                offsets[side] = end
    return record


def verify_source(evidence, raw):
    record = (evidence.get("coverage") or {}).get(KEY)
    if record is not None and canonical_hash(review(raw)) != canonical_hash(record):
        raise ValueError("Email alternative text or coordinates differ from the original MIME parts")
    return record
