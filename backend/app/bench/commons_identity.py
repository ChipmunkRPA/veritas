"""Conservative, pure Commons metadata screening for a research acquisition queue.

No network, media decoding, author authentication or automatic dataset admission.
Artist links are attribution assertions, not proof of creator ownership. Legacy
hashes deliberately preserve the existing fetcher's spelling/entity behaviour.
"""
from __future__ import annotations

import hashlib
import html
import re
import unicodedata
from collections.abc import Mapping
from datetime import UTC, datetime
from html.parser import HTMLParser
from urllib.parse import parse_qsl, unquote, urlparse, urlsplit

POLICY = "commons-camera-metadata/1"
MAX_FILE_BYTES = 10_000_000
MIN_SIDE = 1024
MAX_PIXELS = 50_000_000
_TAG = re.compile(r"<[^>]+>")
_HEX_SHA1 = re.compile(r"[0-9a-fA-F]{40}")
_BASE36_SHA1 = re.compile(r"[0-9a-zA-Z]{31}")
_CAMERA_KEYS = frozenset({"Make", "Model", "DateTimeOriginal", "Software"})
_AI_SOFTWARE = re.compile(
    r"\b(?:topaz\s+(?:photo|gigapixel|denoise|sharpen)(?:\s+\w+)?\s+ai|gigapixel\s+ai|denoise\s+ai|sharpen\s+ai|"
    r"adobe\s+firefly|stable\s+diffusion|midjourney|dall[ -]?e|generative\s+(?:fill|erase|expand)|"
    r"luminar\s+ai|ai\s+(?:enhanc\w*|upscal\w*|denois\w*))\b", re.IGNORECASE)
_EDITING_SOFTWARE = re.compile(r"\b(?:photoshop|lightroom|gimp|darktable|capture\s+one|affinity\s+photo)\b", re.IGNORECASE)
LIMITATIONS = [
    "Metadata eligibility is for a research acquisition queue only; visual and byte-integrity review remain required.",
    "Own-work and creator links are publisher assertions, not authenticated camera capture or authorship.",
    "Camera make/model does not identify a physical device; editing and upstream training overlap remain possible.",
    "Artist-only identity parsing is incomplete; structured-data aliases and external author identities need separate review.",
]


def _text(value: str) -> str:
    return unicodedata.normalize("NFC", " ".join(value.split()))


def normalize_username(value: str) -> str | None:
    """MediaWiki-style first-letter capitalization, preserving all later letter case."""
    if not isinstance(value, str):
        return None
    value = _text(value.replace("_", " "))
    if (not value or any(c in value for c in ":/#[]|<>%\ufffd")
            or any(unicodedata.category(c).startswith("C") for c in value)):
        return None
    return value[0].upper() + value[1:]


def legacy_group_key(value: str) -> str | None:
    """Exactly fetch.group_key(normalise='text', namespace='wikimedia/commons'), without row fallback."""
    if not isinstance(value, str) or not value.strip():
        return None
    normalized = " ".join(value.split()).lower()
    return "wikimedia/commons:" + hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def _user_link(href: str) -> tuple[str | None, str | None]:
    """Explicit Commons User: links, including the exact generated missing-page form."""
    # urlparse removes some controls; reject them before they can change URL identity.
    if any(unicodedata.category(c).startswith("C") for c in href):
        return None, "invalid_author_link"
    try:
        parsed = urlparse(href)
        if parsed.scheme and parsed.scheme.lower() not in {"http", "https"}:
            return None, None
        if parsed.netloc:
            # The authority must not contain credentials or even an empty/default port.
            if parsed.netloc.lower() != "commons.wikimedia.org":
                return None, None
        elif not href.startswith(("/wiki/", "/w/index.php?")):
            return None, None
        if parsed.params:
            return None, "noncanonical_author_link"
        if parsed.path == "/w/index.php":
            # urlparse strips even an empty ';' suffix; preserve the literal URL path.
            if urlsplit(href).path != "/w/index.php" or "#" in href:
                return None, "noncanonical_author_link"
            pairs = parse_qsl(parsed.query, keep_blank_values=True, strict_parsing=True, errors="strict")
            query = dict(pairs)
            if (len(pairs) != 3 or len(query) != 3 or set(query) != {"title", "action", "redlink"}
                    or query["action"] != "edit" or query["redlink"] != "1"):
                return None, "noncanonical_author_link"
            title = query["title"]
            if not title.startswith("User:"):
                return None, "unsupported_author_namespace"
            # parse_qsl already percent-decodes once. A remaining '%' is not decoded again.
            raw = title.removeprefix("User:")
        else:
            path = unquote(parsed.path, errors="strict")
            if not path.startswith("/wiki/User:"):
                if path.startswith("/wiki/User_") or path.startswith("/wiki/User_talk:"):
                    return None, "unsupported_author_namespace"
                return None, None
            if parsed.query or parsed.fragment:
                return None, "noncanonical_author_link"
            raw = path.removeprefix("/wiki/User:")
    except (ValueError, UnicodeError):
        return None, "invalid_author_link"
    # Username whitespace normalization must not erase a decoded control character.
    if any(unicodedata.category(c).startswith("C") for c in raw):
        return None, "unsupported_author_username_or_subpage"
    username = normalize_username(raw)
    if username is None:
        return None, "unsupported_author_username_or_subpage"
    return username, None


class _MetadataHTML(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.outside_links: list[str] = []
        self.links: list[dict] = []
        self._anchor: dict | None = None
        self.own_work_marker = False
        self.unsafe_content = False
        self._skip = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs = dict(attrs)
        if tag in {"script", "style", "iframe", "object"}:
            self.unsafe_content = True
            self._skip += 1
        if "int-own-work" in (attrs.get("class") or "").split():
            self.own_work_marker = True
        self.parts.append(" ")
        if tag == "a":
            if self._anchor is not None:
                self.unsafe_content = True
            self._anchor = {"href": attrs.get("href") or "", "parts": []}

    def handle_endtag(self, tag: str) -> None:
        self.parts.append(" ")
        if tag in {"script", "style", "iframe", "object"} and self._skip:
            self._skip -= 1
        if tag == "a" and self._anchor is not None:
            self.links.append({"href": self._anchor["href"], "text": _text("".join(self._anchor["parts"]))})
            self._anchor = None

    def handle_data(self, data: str) -> None:
        if not self._skip:
            self.parts.append(data)
            if self._anchor is not None:
                self._anchor["parts"].append(data)
            else:
                self.outside_links.append(data)

    def result(self) -> str:
        if self._anchor is not None:
            self.links.append({"href": self._anchor["href"], "text": _text("".join(self._anchor["parts"]))})
            self._anchor = None
            self.unsafe_content = True
        return _text("".join(self.parts))


def _parse_html(value: str) -> tuple[str, _MetadataHTML]:
    parser = _MetadataHTML()
    try:
        parser.feed(value)
        parser.close()
    except (ValueError, TypeError):
        parser.unsafe_content = True
    return parser.result(), parser


def parse_artist(artist_html: str, *, users: Mapping[str, int] | None = None) -> dict:
    """Return exclusion aliases and explicit account links, never uploader fallbacks."""
    artist_html = artist_html if isinstance(artist_html, str) else ""
    plain, parsed = _parse_html(artist_html)
    legacy_plain = " ".join(_TAG.sub(" ", artist_html).split())
    names = {name for name in [plain, *(r["text"] for r in parsed.links)] if name}
    usernames, reasons = set(), set()
    for link in parsed.links:
        username, reason = _user_link(link["href"])
        if username:
            usernames.add(username)
        else:
            reasons.add("additional_unresolved_author_link")
        if reason:
            reasons.add(reason)
    outside = _text(" ".join(parsed.outside_links))
    connector = re.sub(r"[\W_]+", " ", outside, flags=re.UNICODE).strip().lower()
    if usernames and connector not in {"", "and", "by", "photo by", "photograph by", "photography by"}:
        reasons.add("additional_unresolved_author_text")
    if not plain:
        reasons.add("missing_artist")
    if not usernames:
        reasons.add("missing_explicit_commons_author")
    if parsed.unsafe_content:
        reasons.add("malformed_or_active_artist_html")
    lookup: dict[str, int] = {}
    for name, numeric_id in (users or {}).items():
        normalized = normalize_username(name)
        if normalized is None or type(numeric_id) is not int or numeric_id <= 0:
            reasons.add("invalid_user_resolution")
            continue
        if normalized in lookup and lookup[normalized] != numeric_id:
            reasons.add("conflicting_user_resolution")
        lookup[normalized] = numeric_id
    ids = sorted({lookup[name] for name in usernames if name in lookup})
    all_resolved = bool(usernames) and all(name in lookup for name in usernames)
    if usernames and not all_resolved:
        reasons.add("author_resolution_needed")
    if len(usernames) > 1 and (not all_resolved or len(ids) != 1):
        reasons.add("multiple_distinct_authors")
    aliases = {legacy_plain, plain, *names, *usernames}
    keys = sorted({key for value in aliases if (key := legacy_group_key(value)) is not None})
    return {"plain_text": plain, "legacy_plain_text": legacy_plain, "names": sorted(names),
            "usernames": sorted(usernames), "user_ids": ids, "all_usernames_resolved": all_resolved,
            "canonical_author_key": f"commons:userid:{ids[0]}" if all_resolved and len(ids) == 1 and not reasons else None,
            "legacy_group_keys": keys, "legacy_artist_group_key": legacy_group_key(legacy_plain),
            "unlinked_artist_text": outside,
            "reasons": sorted(reasons), "authenticated": False}


def normalize_sha1(value: str) -> dict | None:
    """Accept API hex or fixed-width 31-character base36; reject overflow/malformed hashes."""
    if not isinstance(value, str):
        return None
    if _HEX_SHA1.fullmatch(value):
        return {"value": value, "encoding": "hex", "hex": value.lower()}
    if _BASE36_SHA1.fullmatch(value):
        integer = int(value, 36)
        if integer < 1 << 160:
            return {"value": value, "encoding": "base36", "hex": f"{integer:040x}"}
    return None


def _ext_value(ext: dict, key: str) -> str:
    field = ext.get(key)
    return field["value"] if isinstance(field, dict) and isinstance(field.get("value"), str) else ""


def parse_licence(short_name: str, url: str) -> dict:
    """Exact recognised unported CC URLs paired with their corresponding short names."""
    name, _ = _parse_html(short_name if isinstance(short_name, str) else "")
    raw_url = html.unescape(url.strip()) if isinstance(url, str) else ""
    reasons, spdx, canonical = [], None, None
    try:
        parsed = urlparse(raw_url)
        path = parsed.path.rstrip("/")
        if (parsed.scheme not in {"http", "https"} or parsed.hostname != "creativecommons.org"
                or parsed.username or parsed.password or parsed.port is not None or parsed.query or parsed.fragment):
            raise ValueError("unrecognised licence URL")
        if path == "/publicdomain/zero/1.0":
            spdx = "CC0-1.0"
            allowed = {"CC0", "CC0 1.0", "CC0 1.0 Universal", "CC0-1.0"}
        else:
            match = re.fullmatch(r"/licenses/(by|by-sa)/(1\.0|2\.0|2\.5|3\.0|4\.0)", path)
            if match is None:
                raise ValueError("unrecognised licence URL")
            family, version = match.groups()
            spdx = f"CC-{family.upper()}-{version}"
            allowed = {f"CC {family.upper()} {version}", spdx}
        canonical = "https://creativecommons.org" + path + "/"
        if name not in allowed:
            reasons.append("licence_name_url_mismatch")
    except ValueError:
        reasons.append("unsupported_licence_url")
    return {"short_name": name, "url": raw_url, "canonical_url": canonical, "spdx": spdx,
            "per_file_assertion": True, "reasons": reasons}


def _positive_int(value: object) -> bool:
    return type(value) is int and value > 0


def _timestamp(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return parsed.astimezone(UTC).isoformat() if parsed.utcoffset() is not None else None
    except ValueError:
        return None


def _capture_date(value: str | None) -> bool:
    if not value:
        return False
    try:
        datetime.strptime(value, "%Y:%m:%d %H:%M:%S")
        return True
    except ValueError:
        return _timestamp(value) is not None


def camera_metadata(values: object) -> dict:
    entries = values if isinstance(values, list) else []
    result, reasons = {}, set()
    for entry in entries:
        if (not isinstance(entry, dict) or not isinstance(entry.get("name"), str)
                or entry["name"] not in _CAMERA_KEYS):
            continue
        name, value = entry["name"], entry.get("value")
        if not isinstance(value, str) or not value.strip():
            reasons.add("invalid_camera_metadata")
            continue
        value = _text(value)
        if name in result and result[name] != value:
            reasons.add("conflicting_camera_metadata")
        result[name] = value
    if not result.get("Make") or not result.get("Model"):
        reasons.add("missing_camera_make_model")
    if not _capture_date(result.get("DateTimeOriginal")):
        reasons.add("missing_or_invalid_capture_date")
    software = result.get("Software", "")
    if _AI_SOFTWARE.search(software):
        editing = "explicit_ai_processing_software"
        reasons.add("editing_provenance_ai_processing")
    elif _EDITING_SOFTWARE.search(software):
        editing = "editing_software_recorded_authenticity_unknown"
    elif software:
        editing = "software_recorded_authenticity_unknown"
    else:
        editing = "software_not_recorded_authenticity_unknown"
    return {"fields": result, "editing_provenance": editing, "reasons": sorted(reasons), "camera_original_authenticated": False}


def own_work(ext: dict) -> dict:
    fields = {}
    declared, marker, conflict = False, False, False
    for key in ("Credit", "Source"):
        value = _ext_value(ext, key)
        if not value:
            continue
        plain, parsed = _parse_html(value)
        fields[key] = {"plain_text": plain, "int_own_work_marker": parsed.own_work_marker}
        exact = plain.lower() == "own work" and not parsed.unsafe_content
        declared |= exact
        marker |= exact and parsed.own_work_marker
        conflict |= not exact
    reasons = [] if declared and not conflict else ["conflicting_own_work_source" if conflict and declared else "missing_explicit_own_work"]
    return {"declared": declared and not conflict, "int_own_work_marker": marker,
            "fields": fields, "reasons": reasons, "authenticated": False}


def evaluate_candidate(page: dict, *, users: Mapping[str, int] | None = None,
                       max_file_bytes: int = MAX_FILE_BYTES, min_side: int = MIN_SIDE,
                       max_pixels: int = MAX_PIXELS) -> dict:
    """Screen one Action API page; a successful result still requires independent review."""
    if not isinstance(page, dict):
        raise TypeError("page must be an Action API page object")
    if not all(_positive_int(v) for v in (max_file_bytes, min_side, max_pixels)):
        raise ValueError("candidate limits must be positive integers")
    reasons = set()
    infos, revisions = page.get("imageinfo"), page.get("revisions")
    info = infos[0] if isinstance(infos, list) and len(infos) == 1 and isinstance(infos[0], dict) else {}
    revision = revisions[0] if isinstance(revisions, list) and len(revisions) == 1 and isinstance(revisions[0], dict) else {}
    if not info:
        reasons.add("missing_or_ambiguous_image_revision")
    if not _positive_int(page.get("pageid")) or not _positive_int(revision.get("revid")):
        reasons.add("missing_page_or_description_revision_id")
    title = page.get("title")
    if not isinstance(title, str) or not title.startswith("File:") or not title[5:].strip() or page.get("ns", 6) != 6:
        reasons.add("invalid_file_title_or_namespace")
    uploaded = _timestamp(info.get("timestamp"))
    if uploaded is None:
        reasons.add("missing_or_invalid_upload_timestamp")
    content_hash = normalize_sha1(info.get("sha1"))
    if content_hash is None:
        reasons.add("missing_or_invalid_sha1")
    if info.get("mime") != "image/jpeg":
        reasons.add("not_jpeg")
    size, width, height = info.get("size"), info.get("width"), info.get("height")
    if not _positive_int(size) or size > max_file_bytes:
        reasons.add("invalid_or_oversize_file")
    if not _positive_int(width) or not _positive_int(height):
        reasons.add("invalid_dimensions")
    elif min(width, height) < min_side or width * height > max_pixels:
        reasons.add("dimensions_outside_limits")
    url = info.get("url")
    canonical_url = None
    try:
        parsed = urlparse(url) if isinstance(url, str) else None
        if (parsed is None or parsed.scheme != "https" or parsed.hostname != "upload.wikimedia.org"
                or parsed.username or parsed.password or parsed.port is not None or parsed.fragment
                or not parsed.path.startswith("/wikipedia/commons/") or "/thumb/" in parsed.path
                or not parsed.path.lower().endswith((".jpg", ".jpeg"))):
            reasons.add("invalid_original_file_url")
        else:
            tracking = parse_qsl(parsed.query, keep_blank_values=True, strict_parsing=True)
            expected_tracking = {("utm_source", "commons.wikimedia.org"), ("utm_campaign", "imageinfo"),
                                 ("utm_content", "original")}
            if tracking and (len(tracking) != 3 or set(tracking) != expected_tracking):
                reasons.add("invalid_original_file_url")
            else:
                canonical_url = parsed._replace(query="").geturl()
    except ValueError:
        reasons.add("invalid_original_file_url")
    ext = info.get("extmetadata") if isinstance(info.get("extmetadata"), dict) else {}
    creator = parse_artist(_ext_value(ext, "Artist"), users=users)
    licence = parse_licence(_ext_value(ext, "LicenseShortName"), _ext_value(ext, "LicenseUrl"))
    origin = own_work(ext)
    camera = camera_metadata(info.get("metadata"))
    for section in (creator, licence, origin, camera):
        reasons.update(section["reasons"])
    return {"policy": POLICY, "eligible_for_acquisition": not reasons, "reasons": sorted(reasons),
            "quarantine": bool(reasons), "creator": creator,
            "uploader": {"username": info.get("user"), "user_id": info.get("userid"), "creator_inferred": False},
            "identity": {"page_id": page.get("pageid"), "title": title, "description_revision_id": revision.get("revid"),
                         "upload_timestamp": info.get("timestamp"), "upload_timestamp_utc": uploaded, "sha1": content_hash},
            "original_url": canonical_url, "api_reported_original_url": url,
            "mime": info.get("mime"), "size_bytes": size,
            "width": width, "height": height, "camera": camera, "licence": licence, "own_work": origin,
            "limits": {"max_file_bytes": max_file_bytes, "min_side": min_side, "max_pixels": max_pixels},
            "requires_visual_review": True, "research_only": True, "training_admitted": False,
            "limitations": list(LIMITATIONS)}
