"""Offline C2PA validation (I-2; IF §4.4; D-15; DC-06).

- **Offline.** Every read uses `verify.remote_manifest_fetch=false`, `verify.ocsp_fetch=false` and `verify.verify_trust=true`.
  A manifest stored only at a remote URL is reported as `remote` (not fetched); revocation is never checked.
- **Trust lists.** The C2PA conformance Trust List decides Trusted. The legacy CAI interim list (frozen 2026-01-01) is read
  only for manifests that are valid but untrusted, and reported separately as `legacy_trusted`. Both are bundled under
  `trust/` and verified against `trust/snapshot.json` (SHA-256) before use. The C2PA TSA list is bundled and recorded but
  not merged into the claim-signer anchors: c2pa-rs has no separate TSA anchor setting, and an untrusted timestamp is
  informational (`timeStamp.untrusted`) without changing the state.
- **States.** `trusted` (valid and the signer chains to the Trust List), `valid` (intact, signer not on the list), `invalid`
  (a validation failure: modified after signing or a malformed claim), `absent`, `remote`, `unreadable`.
- **Content.** Validation codes (success / informational / failure, active manifest and ingredient deltas), the ingredient
  chain (active → ingredients, recursively), IPTC DigitalSourceType tiers (retired terms parsed), signer and thumbnails.
"""
from __future__ import annotations

import contextlib
import functools
import hashlib
import io
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.storage.files import untrusted_text

TRUST_DIR = Path(__file__).resolve().parent / "trust"
OFFICIAL, LEGACY = "c2pa", "cai-legacy"
OFFLINE_VERIFY = {"remote_manifest_fetch": False, "ocsp_fetch": False, "verify_trust": True}
REVOCATION_NOTE = "Revocation status was not checked (offline)."
IPTC_PREFIXES = ("http://cv.iptc.org/newscodes/digitalsourcetype/", "https://cv.iptc.org/newscodes/digitalsourcetype/")
HARD_BINDING_MATCH = frozenset({"assertion.dataHash.match", "assertion.bmffHash.match", "assertion.boxesHash.match",
                                "assertion.collectionHash.match"})
STATE_LABELS = {"trusted": "Trusted", "valid": "Valid (signer not on the trust list)", "invalid": "Invalid",
                "absent": "No manifest", "remote": "Remote manifest (not fetched)", "unreadable": "Unreadable"}
LEVELS = ("failure", "informational", "success")
MAX_CHAIN_NODES = 64
MAX_CODES = 200
MAX_THUMBNAILS = 4
MAX_THUMBNAIL_BYTES = 1 << 20
MAX_THUMBNAIL_PIXELS = 4096 * 4096
# N-10 (SEC-10): a thumbnail is one of these by its first bytes, whatever the manifest of the uploaded file declares
RASTER_FORMATS = {"image/jpeg": "JPEG", "image/png": "PNG", "image/webp": "WEBP"}


class TrustBundleError(RuntimeError):
    """A bundled trust-list file is missing or does not match its recorded SHA-256."""


@dataclass(frozen=True)
class TrustBundle:
    name: str
    label: str
    revision: str
    retrieved: str
    anchors_pem: str
    anchors_sha256: str
    list_issue_date: str | None = None
    allowed_list: str | None = None
    trust_config: str | None = None

    def summary(self) -> str:
        issued = f", list issued {self.list_issue_date[:10]}" if self.list_issue_date else ""
        revision = self.revision[:12] if len(self.revision) == 40 else self.revision
        return f"{self.label} (revision {revision}{issued}; retrieved {self.retrieved}; anchors sha256 {self.anchors_sha256[:12]})"


@dataclass(frozen=True)
class SourceType:
    uri: str
    term: str
    label: str
    tier: str  # ai | capture | edited | screen_capture | other_creation | unknown
    retired: bool


@dataclass(frozen=True)
class ValidationCode:
    level: str
    code: str
    scope: str
    explanation: str
    url: str


@dataclass(frozen=True)
class ChainNode:
    depth: int
    title: str
    relationship: str
    format: str
    manifest: str | None  # None: an ingredient without its own manifest
    signer: str
    source_types: tuple[str, ...]
    failures: int


@dataclass
class C2PAReport:
    state: str
    error: str | None = None
    remote_url: str | None = None
    legacy_trusted: bool | None = None
    claim_generator: str = ""
    title: str = ""
    format: str = ""
    signer: dict[str, str] = field(default_factory=dict)
    actions: list[dict[str, str]] = field(default_factory=list)
    source_types: list[SourceType] = field(default_factory=list)
    codes: list[ValidationCode] = field(default_factory=list)
    chain: list[ChainNode] = field(default_factory=list)
    thumbnails: list[tuple[str, str, bytes]] = field(default_factory=list)  # (title, media type, bytes)

    @property
    def ai_declared(self) -> bool:
        return any(s.tier == "ai" for s in self.source_types)

    @property
    def capture_declared(self) -> bool:
        return bool(self.source_types) and all(s.tier == "capture" for s in self.source_types)

    @property
    def covers_bytes(self) -> bool:
        """A hard binding (data, BMFF, boxes or collection hash) matched the current bytes."""
        return self.state in ("trusted", "valid") and any(c.level == "success" and c.code in HARD_BINDING_MATCH for c in self.codes)

    @property
    def failure_codes(self) -> list[str]:
        return [c.code for c in self.codes if c.level == "failure"]


# --- bundled data ---------------------------------------------------------------------------------------------

@functools.cache
def _snapshot() -> dict[str, Any]:
    try:
        return json.loads((TRUST_DIR / "snapshot.json").read_text())
    except (OSError, ValueError) as e:
        raise TrustBundleError(f"trust snapshot unreadable: {e}") from e


def _verified_text(relative: str, expected_sha256: str) -> str:
    try:
        data = (TRUST_DIR / relative).read_bytes()
    except OSError as e:
        raise TrustBundleError(f"{relative} is missing") from e
    if hashlib.sha256(data).hexdigest() != expected_sha256:
        raise TrustBundleError(f"{relative} does not match its recorded SHA-256")
    return data.decode("utf-8")


@functools.cache
def load_bundle(name: str) -> TrustBundle:
    record = _snapshot()["bundles"].get(name)
    if record is None:
        raise TrustBundleError(f"unknown trust bundle {name!r}")
    files = {Path(f["file"]).name: _verified_text(f["file"], f["sha256"]) for f in record["files"]}
    sha = {Path(f["file"]).name: f["sha256"] for f in record["files"]}
    return TrustBundle(name=name, label=record["label"], revision=record["revision"], retrieved=record["retrieved"],
                       anchors_pem=files[record["anchors"]], anchors_sha256=sha[record["anchors"]],
                       list_issue_date=record.get("list_issue_date"),
                       allowed_list=files.get(record.get("allowed_list") or ""),
                       trust_config=files.get(record.get("trust_config") or ""))


@functools.cache
def _vocabulary() -> dict[str, dict[str, Any]]:
    record = _snapshot()["iptc_digital_source_type"]
    return json.loads(_verified_text(record["file"], record["sha256"]))["terms"]


def source_type(uri: str) -> SourceType:
    term = next((uri[len(p):] for p in IPTC_PREFIXES if uri.startswith(p)), uri.rsplit("/", 1)[-1])
    entry = _vocabulary().get(term)
    if entry is None or not any(uri.startswith(p) for p in IPTC_PREFIXES):
        return SourceType(uri, term, term, "unknown", False)
    return SourceType(uri, term, entry["label"], entry["tier"], entry["retired"])


def settings(bundle: TrustBundle | None) -> dict[str, Any]:
    out: dict[str, Any] = {"verify": dict(OFFLINE_VERIFY)}
    if bundle is not None:
        trust = {"trust_anchors": bundle.anchors_pem}
        if bundle.allowed_list:
            trust["allowed_list"] = bundle.allowed_list
        if bundle.trust_config:
            trust["trust_config"] = bundle.trust_config
        out["trust"] = trust
    return out


def available() -> bool:
    try:
        import c2pa  # type: ignore  # noqa: F401
    except ImportError:
        return False
    return True


@functools.cache
def sdk_version() -> str:
    try:
        import c2pa  # type: ignore

        return str(c2pa.sdk_version())
    except Exception:
        return "unknown"


# --- container sniffing ---------------------------------------------------------------------------------------

_BMFF_BRANDS = {b"qt  ": "video/quicktime", b"heic": "image/heic", b"heix": "image/heic", b"mif1": "image/heif",
                b"heim": "image/heic", b"heis": "image/heic", b"avif": "image/avif", b"avis": "image/avif",
                b"M4A ": "audio/mp4"}


def sniff_mime(data: bytes | memoryview) -> str | None:
    head = bytes(data[:16])
    if head.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if head.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if head.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if head.startswith((b"II*\x00", b"MM\x00*")):
        return "image/tiff"
    if head.startswith(b"fLaC"):
        return "audio/flac"
    if (len(head) >= 10 and head[:3] == b"ID3" and head[3] in (2, 3, 4)
            and head[4] != 255 and not any(byte & 128 for byte in head[6:10])):
        size = sum(byte << shift for byte, shift in zip(head[6:10], (21, 14, 7, 0), strict=True))
        end = 10 + size + (10 if head[3] == 4 and head[5] & 16 else 0)
        # C2PA's FLAC writer prepends an ID3 manifest tag. Dispatch from the
        # signature immediately after that bounded tag, without scanning audio.
        if bytes(data[end:end + 4]) == b"fLaC":
            return "audio/flac"
        return "audio/mpeg"
    # MPEG layer III without an ID3 header. This is container dispatch, not
    # validation: the SDK still has to validate the file and its hard binding.
    if (len(head) >= 4 and head[0] == 255 and head[1] & 224 == 224
            and (head[1] >> 3) & 3 != 1 and (head[1] >> 1) & 3 == 1
            and (head[2] >> 4) not in (0, 15) and (head[2] >> 2) & 3 != 3):
        return "audio/mpeg"
    if head.startswith(b"RIFF") and head[8:12] in (b"WEBP", b"AVI ", b"WAVE"):
        return {b"WEBP": "image/webp", b"AVI ": "video/avi", b"WAVE": "audio/wav"}[head[8:12]]
    if head[4:8] == b"ftyp":
        return _BMFF_BRANDS.get(head[8:12], "video/mp4")
    return None


_MAX_EBML_HEADER_BYTES = 4096
_MAX_EBML_HEADER_ELEMENTS = 64
_EBML_UINT_DEFAULTS = {0x4286: 1, 0x42F7: 1, 0x42F2: 4, 0x42F3: 8, 0x4287: 1, 0x4285: 1}


def _ebml_vint(data: bytes, offset: int, end: int, *, identifier: bool = False) -> tuple[int, int] | None:
    """Read one bounded EBML ID/size; unknown sizes and reserved IDs cannot describe this finite header."""
    if offset >= end or data[offset] == 0:
        return None
    width = 9 - data[offset].bit_length()
    if width > (4 if identifier else 8) or offset + width > end:
        return None
    raw = int.from_bytes(data[offset:offset + width], "big")
    mask = (1 << (7 * width)) - 1
    value = raw & mask
    if value == mask or (identifier and value == 0):
        return None
    return (raw if identifier else value), offset + width


def unsupported_container(data: bytes | memoryview) -> str | None:
    """Recognize a bounded WebM/Matroska EBML header for C2PA applicability only.

    A finite header, one DocType and consistent standard header fields are required;
    EBML defaults apply when integer fields are absent or empty. Void and a first,
    structurally sized CRC element are permitted. Unknown fields, duplicate non-Void elements, unsupported EBML
    read versions and truncated/unknown sizes fail closed. This neither verifies a CRC
    nor validates the Segment/media body: the downstream media probe owns file validity.
    """
    if bytes(data[:4]) != b"\x1a\x45\xdf\xa3":
        return None
    head = bytes(data[:_MAX_EBML_HEADER_BYTES])
    length = _ebml_vint(head, 4, len(head))
    if length is None:
        return None
    payload_size, offset = length
    end = offset + payload_size
    if end > len(head):
        return None
    values, seen, doctype = dict(_EBML_UINT_DEFAULTS), set(), None
    count = 0
    while offset < end:
        count += 1
        if count > _MAX_EBML_HEADER_ELEMENTS:
            return None
        element_id = _ebml_vint(head, offset, end, identifier=True)
        if element_id is None:
            return None
        ident, offset = element_id
        length = _ebml_vint(head, offset, end)
        if length is None:
            return None
        size, offset = length
        if offset + size > end or (ident != 0xEC and ident in seen):
            return None
        seen.add(ident)
        payload = head[offset:offset + size]
        offset += size
        if ident == 0x4282:
            # RFC 8794 section 13: the first NUL terminates the string; remaining bytes are padding.
            doctype = {b"webm": "webm", b"matroska": "matroska"}.get(payload.split(b"\x00", 1)[0])
            if doctype is None:
                return None
        elif ident in values:
            if size > 8:
                return None
            values[ident] = int.from_bytes(payload, "big") if size else _EBML_UINT_DEFAULTS[ident]
        elif ident == 0xBF:
            if size != 4 or count != 1:
                return None
        elif ident != 0xEC:
            return None
    if not (doctype and values[0x4286] >= values[0x42F7] == 1
            and values[0x42F2] == 4 and 1 <= values[0x42F3] <= 8
            and values[0x4287] >= values[0x4285] >= 1):
        return None
    return doctype


# --- reading --------------------------------------------------------------------------------------------------

def _read_store(data: bytes, mime: str, bundle: TrustBundle | None) -> tuple[dict[str, Any], list[tuple[str, str, bytes]]]:
    import c2pa  # type: ignore

    context = c2pa.Context.from_dict(settings(bundle))
    reader = None
    try:
        reader = c2pa.Reader(mime, io.BytesIO(data), context=context)
        store = json.loads(reader.json())
        return store, _thumbnails(reader, store)
    finally:
        with contextlib.suppress(Exception):
            if reader is not None:
                reader.close()
        with contextlib.suppress(Exception):
            context.close()


def _raster_type(data: bytes) -> str | None:
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


def _reencoded(data: bytes) -> tuple[str, bytes] | None:
    """(media type, bytes) of a JPEG, PNG or WebP thumbnail, encoded again here: JPEG stays JPEG, the others become PNG, and
    no metadata is copied. None for any other content, an image that does not decode or one too large to decode safely.
    The bytes of a manifest are the uploader's: what is stored and later served is this server's encoding of the pixels."""
    from PIL import Image, ImageOps

    kind = _raster_type(data)
    if kind is None:
        return None
    out = io.BytesIO()
    try:
        with Image.open(io.BytesIO(data), formats=[RASTER_FORMATS[kind]]) as source:
            if source.width * source.height > MAX_THUMBNAIL_PIXELS:
                return None
            image = ImageOps.exif_transpose(source)
            image = image.convert("RGB" if kind == "image/jpeg" or image.mode not in ("RGBA", "L", "LA") else image.mode)
            pixels = Image.frombytes(image.mode, image.size, image.tobytes())  # the pixels alone: an encoder also writes the
            if kind == "image/jpeg":                                         # comment and the colour profile it finds in `info`
                pixels.save(out, "JPEG", quality=90)
                return "image/jpeg", out.getvalue()
            pixels.save(out, "PNG")
            return "image/png", out.getvalue()
    except (OSError, ValueError, SyntaxError, Image.DecompressionBombError):
        return None


def _thumbnails(reader: Any, store: dict[str, Any]) -> list[tuple[str, str, bytes]]:
    """The claim and ingredient thumbnails of the active manifest that are JPEG, PNG or WebP, by the declared format and by
    their first bytes, each re-encoded (`_reencoded`). An SVG or HTML "thumbnail" is left out (SEC-10)."""
    out: list[tuple[str, str, bytes]] = []
    active = (store.get("manifests") or {}).get(store.get("active_manifest") or "") or {}
    # N-19p (SEC6-01): a title is the signer's text, filtered and cut to 200 characters as the claim generator is
    def named(title: Any, default: str) -> str:
        return untrusted_text(str(title or default))[:200]

    candidates = [(f"claim thumbnail ({named(active.get('title'), 'active manifest')})", active.get("thumbnail"))]
    candidates += [(f"ingredient thumbnail ({named(i.get('title'), 'untitled')})", i.get("thumbnail")) for i in active.get("ingredients") or []]
    for title, thumb in candidates:
        if len(out) >= MAX_THUMBNAILS or not isinstance(thumb, dict):
            continue
        media_type, identifier = str(thumb.get("format") or ""), thumb.get("identifier")
        if media_type.split(";")[0].strip().lower() not in RASTER_FORMATS or not identifier:
            continue
        buf = io.BytesIO()
        with contextlib.suppress(Exception):
            reader.resource_to_stream(identifier, buf)
            if 0 < buf.tell() <= MAX_THUMBNAIL_BYTES and (stored := _reencoded(buf.getvalue())) is not None:
                out.append((title, *stored))
    return out


def _state(store: dict[str, Any], codes: list[ValidationCode]) -> str:
    state = str(store.get("validation_state") or "").lower()
    if state in ("trusted", "valid", "invalid"):
        return state
    return "invalid" if any(c.level == "failure" for c in codes) else "valid"


def _codes(store: dict[str, Any]) -> list[ValidationCode]:
    rows: list[ValidationCode] = []
    results = store.get("validation_results")
    if isinstance(results, dict):
        active = results.get("activeManifest") or {}
        for level in LEVELS:
            rows += [ValidationCode(level, str(c.get("code", "")), "active manifest", str(c.get("explanation", "")),
                                    str(c.get("url", ""))) for c in active.get(level) or []]
        for delta in results.get("ingredientDeltas") or []:
            scope = "ingredient " + str(delta.get("ingredientAssertionURI", "")).rsplit("/", 1)[-1]
            deltas = delta.get("validationDeltas") or {}
            for level in LEVELS:
                rows += [ValidationCode(level, str(c.get("code", "")), scope, str(c.get("explanation", "")),
                                        str(c.get("url", ""))) for c in deltas.get(level) or []]
    else:  # older stores: a flat validation_status list of failures
        rows += [ValidationCode("failure", str(c.get("code", "")), "active manifest", str(c.get("explanation", "")),
                                str(c.get("url", ""))) for c in store.get("validation_status") or []]
    return rows[:MAX_CODES]


def _software(agent: Any) -> str:
    return str(agent.get("name", "")) if isinstance(agent, dict) else str(agent or "")


def _actions(manifest: dict[str, Any]) -> list[dict[str, str]]:
    out = []
    for assertion in manifest.get("assertions") or []:
        if str(assertion.get("label", "")).startswith("c2pa.actions"):
            for act in (assertion.get("data") or {}).get("actions") or []:
                out.append({"action": str(act.get("action", "")), "digital_source_type": str(act.get("digitalSourceType") or ""),
                            "software_agent": _software(act.get("softwareAgent"))})
    return out


def _source_type_uris(manifest: dict[str, Any]) -> list[str]:
    uris = [a["digital_source_type"] for a in _actions(manifest) if a["digital_source_type"]]
    for assertion in manifest.get("assertions") or []:
        if str(assertion.get("label", "")).startswith("stds.iptc"):
            value = (assertion.get("data") or {}).get("Iptc4xmpExt:DigitalSourceType")
            if isinstance(value, str) and value:
                uris.append(value)
    return list(dict.fromkeys(uris))


def _generator(manifest: dict[str, Any]) -> str:
    info = manifest.get("claim_generator_info") or []
    if info and isinstance(info[0], dict):
        name = " ".join(str(info[0].get(k, "")) for k in ("name", "version") if info[0].get(k))
        if name:
            return name
    return str(manifest.get("claim_generator") or "")


def _signer(manifest: dict[str, Any]) -> dict[str, str]:
    sig = manifest.get("signature_info") or {}
    return {k: str(sig[k]) for k in ("common_name", "issuer", "alg", "time", "cert_serial_number") if sig.get(k)}


def _failures(results: Any) -> int:
    if isinstance(results, dict):
        return len((results.get("activeManifest") or {}).get("failure") or [])
    return len(results or [])


def _chain(store: dict[str, Any]) -> list[ChainNode]:
    manifests = store.get("manifests") or {}
    nodes: list[ChainNode] = []
    seen: set[str] = set()

    def visit(label: str | None, title: str, relationship: str, fmt: str, depth: int, results: Any) -> None:
        if len(nodes) >= MAX_CHAIN_NODES:
            return
        manifest = manifests.get(label) if label else None
        m = manifest or {}
        nodes.append(ChainNode(depth=depth, title=title or str(m.get("title") or ""), relationship=relationship,
                               format=fmt or str(m.get("format") or ""), manifest=label if manifest is not None else None,
                               signer=_signer(m).get("common_name", ""),
                               source_types=tuple(source_type(u).term for u in _source_type_uris(m)), failures=_failures(results)))
        if manifest is None or label in seen:
            return
        seen.add(label)  # type: ignore[arg-type]
        for ingredient in m.get("ingredients") or []:
            visit(ingredient.get("active_manifest"), str(ingredient.get("title") or ""), str(ingredient.get("relationship") or ""),
                  str(ingredient.get("format") or ""), depth + 1,
                  ingredient.get("validation_results") or ingredient.get("validation_status"))

    active = store.get("active_manifest")
    visit(active, "", "active", "", 0, store.get("validation_results") or store.get("validation_status"))
    return nodes


_REMOTE_URL = re.compile(r"from url (\S+)")


def read(data: bytes | memoryview, mime: str, *, bundle: TrustBundle | None = None, legacy: TrustBundle | None = None,
         thumbnails: bool = True) -> C2PAReport:
    """Validate the manifest store embedded in `data` offline. Never raises for content errors."""
    payload = bytes(data)
    try:
        store, thumbs = _read_store(payload, mime, bundle)
    except Exception as e:  # c2pa.C2paError subclasses carry the kind as a message prefix
        message = str(e)
        lowered = message.lower()
        if "manifestnotfound" in lowered or "no jumbf" in lowered:
            return C2PAReport(state="absent")
        if lowered.startswith("remote") or "must fetch remote manifests" in lowered:
            match = _REMOTE_URL.search(message)
            return C2PAReport(state="remote", remote_url=match.group(1) if match else None)
        return C2PAReport(state="unreadable", error=message[:200])
    manifest = (store.get("manifests") or {}).get(store.get("active_manifest") or "") or {}
    codes = _codes(store)
    report = C2PAReport(state=_state(store, codes), claim_generator=_generator(manifest), title=str(manifest.get("title") or ""),
                        format=str(manifest.get("format") or mime), signer=_signer(manifest), actions=_actions(manifest),
                        source_types=[source_type(u) for u in _source_type_uris(manifest)], codes=codes, chain=_chain(store),
                        thumbnails=thumbs if thumbnails else [])
    if report.state == "valid" and legacy is not None:
        with contextlib.suppress(Exception):
            legacy_store, _ = _read_store(payload, mime, legacy)
            report.legacy_trusted = str(legacy_store.get("validation_state") or "").lower() == "trusted"
    return report
