"""Local source binding for a named BEA or Census release."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import date
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path

from app.evidence.document_archive import ArchiveUnavailable
from app.evidence.document_gao_archive import HASH, comparison_tokens, token_stream, token_values

SCHEMA = "chipmunk.economic-release-pack/1"
MAX_BYTES = 2 * 1024 * 1024
ARCHIVES = {
    "bea": ("U.S. Bureau of Economic Analysis", "https://www.bea.gov/help/faq/145",
            re.compile(r"https://www\.bea\.gov/news/(?:20\d{2}/|blog/20\d{2}-\d{2}-\d{2}/)[a-z0-9-]{1,220}")),
    "census": ("U.S. Census Bureau", "https://www.census.gov/about/policies/citation.html",
               re.compile(r"https://www\.census\.gov/newsroom/press-releases/20\d{2}/[a-z0-9-]{1,180}\.html")),
}
SCOPE = ("This check compares the marked text with a locally retained copy of the cited federal release. "
         "Ordinary editing, transcription and AI assistance can all change wording. "
         "The comparison contributes zero weight to the overall AI assessment.")
HEADER = re.compile(r"\A(?P<title>[^\n\r]{5,280})\n{1,2}Source: (?P<url>[^\s]{1,250})\n{1,2}(?=\S)")
BOUNDARY_TERMS = 12
MIN_EXACT_PERCENT = 60
MAX_TEXT_CHARS = 20_000
MAX_COMPARED_TERMS = 800
MAX_EDIT_TERMS = 96
MAX_EDITS = 80


def permitted_url(publisher, url):
    return publisher in ARCHIVES and isinstance(url, str) and ARCHIVES[publisher][2].fullmatch(url) is not None


def _digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def validate_reference(reference):
    if (set(reference) != {"id", "publisher", "title", "source_url", "published_date", "source_record_sha256",
                           "transcript_sha256", "text"}
            or not permitted_url(reference["publisher"], reference["source_url"])
            or reference["id"] != _digest(reference["source_url"])
            or not isinstance(reference["title"], str) or not 5 <= len(reference["title"]) <= 280
            or any(char in reference["title"] for char in "\r\n")
            or date.fromisoformat(reference["published_date"]).isoformat() != reference["published_date"]
            or not HASH.fullmatch(reference["source_record_sha256"])
            or not 0 < len(reference["text"]) <= 20_000
            or _digest(reference["text"]) != reference["transcript_sha256"]):
        raise ValueError("The retained release does not match its source identity")


@lru_cache(maxsize=2)
def library(path: Path, sha256: str):
    try:
        with path.open("rb") as stream:
            raw = stream.read(MAX_BYTES + 1)
        if not 0 < len(raw) <= MAX_BYTES or hashlib.sha256(raw).hexdigest() != sha256:
            raise ValueError("Release archive digest differs")
        pack = json.loads(raw)
        if set(pack) != {"schema", "references"} or pack["schema"] != SCHEMA or not 1 <= len(pack["references"]) <= 128:
            raise ValueError("Unknown release archive format")
        releases = {}
        for item in pack["references"]:
            validate_reference(item)
            url = item["source_url"]
            if url in releases:
                raise ValueError("Multiple versions of one URL require separate attribution")
            releases[url] = item
        return releases
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise ArchiveUnavailable("The local economic release archive could not be verified.") from error


def _title_key(title):
    return " ".join(token_values(token_stream(title)))


def select(text, releases):
    header = HEADER.match(text)
    if header is None or (ref := releases.get(header["url"])) is None or _title_key(ref["title"]) != _title_key(header["title"]):
        return None
    return ref


def _unique_index(stream, needle):
    occurrences = [i for i in range(len(stream) - len(needle) + 1) if stream[i:i + len(needle)] == needle]
    return occurrences[0] if len(occurrences) == 1 else None


def _span(stream, start, end):
    point = stream[start][1] if start < len(stream) else stream[-1][2]
    return [point, stream[end - 1][2] if start < end else point]


def align(text, reference):
    """Bind both ends to a unique official excerpt and compare the full interior."""
    if len(text) > MAX_TEXT_CHARS or select(text, {reference["source_url"]: reference}) is None:
        return None
    start = HEADER.match(text).end()
    query = [(token, lo + start, hi + start) for token, lo, hi in comparison_tokens(text[start:])]
    if not query or len(query) > MAX_COMPARED_TERMS:
        return None
    words = token_values(query)
    official = token_stream(reference["text"])
    earlier = token_values(official)
    n = BOUNDARY_TERMS
    first, last = _unique_index(earlier, words[:n]), _unique_index(earlier, words[-n:])
    if first is None or last is None or not first + n <= last <= first + MAX_COMPARED_TERMS - n:
        return None
    earlier_stop = last + n
    changes = []
    aligned_size = 2 * n
    source_middle, query_middle = earlier[first + n:last], words[n:-n]
    for tag, a, b, c, d in SequenceMatcher(None, source_middle, query_middle, autojunk=False).get_opcodes():
        if tag == "equal":
            aligned_size += d - c
        elif max(b - a, d - c) > MAX_EDIT_TERMS:
            return None
        else:
            changes.append((_span(query, n + c, n + d), _span(official, first + n + a, first + n + b)))
    if len(changes) > MAX_EDITS or aligned_size * 100 < MIN_EXACT_PERCENT * max(len(query), earlier_stop - first):
        return None
    return _span(query, 0, len(query)), _span(official, first, earlier_stop), changes
