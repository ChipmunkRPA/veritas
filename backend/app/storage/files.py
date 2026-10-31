"""Safe file storage: sanitised names, per-analysis directories, sha256, magic-byte validation."""
from __future__ import annotations

import hashlib
import os
import re
import tempfile
import unicodedata
from pathlib import Path
from typing import Any, BinaryIO

import filetype

from app.core.config import settings

IMAGE_MIMES = {"image/jpeg", "image/png", "image/webp", "image/heic", "image/heif", "image/tiff", "image/bmp", "image/gif", "image/avif"}
VIDEO_MIMES = {"video/mp4", "video/quicktime", "video/x-matroska", "video/webm", "video/x-msvideo", "video/mpeg", "video/3gpp", "video/x-m4v"}
DOC_EXTS = {".pdf", ".docx", ".txt", ".md", ".markdown", ".html", ".htm"}
M4A_BRANDS = {b"M4A ", b"M4B "}
ISO_BRANDS = {b"isom", b"iso2", b"iso4", b"iso5", b"iso6", b"mp41", b"mp42", b"dash"}
M4A_EXTS = {".m4a", ".m4b"}
MAX_DOCX_UNCOMPRESSED = 200 * 1024 * 1024  # zip-bomb guard

_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


SAFE_MAX = 120  # characters of a name on disk and in a download
RECEIVED_MAX = 255  # the columns that hold a name as received
# What is taken out of a name as received: control characters, line and paragraph separators, lone surrogates and the
# characters that reorder the text around them (a name could otherwise be shown with its extension elsewhere)
_BIDI = frozenset("\u200e\u200f\u202a\u202b\u202c\u202d\u202e\u2066\u2067\u2068\u2069")


def _keeping_extension(name: str, limit: int) -> str:
    """`name` cut to `limit` characters at the end of its stem: the extension is never cut."""
    if len(name) <= limit:
        return name
    suffix = Path(name).suffix
    if not 0 < len(suffix) < limit:
        return name[:limit]
    return name[:limit - len(suffix)] + suffix


_SPACING = frozenset("\t\n\r")


def untrusted_text(text: str) -> str:
    """N-19o (security re-check SEC5-01): text that comes from a file, as a received file name is kept: the categories Cc,
    Cs, Zl and Zp and the direction characters dropped (`received_filename`). For every cell of an exhibit table and every
    `observed` text of a check (`ExhibitWriter.table`, `DetectorResult`): an ESC byte stood in the forensic preview, and a
    signer's name could turn the direction of a sentence of a signed report.

    N-19p (code review R10-02): tab, newline, carriage return, Zl and Zp are turned into a space, so that words stay apart
    ("Shot on the pier.\nEdited by Ana" was stored as "Shot on the pier.Edited by Ana"); the rest of the rule is dropped."""
    out = []
    for ch in text:
        category = unicodedata.category(ch)
        if ch in _SPACING or category in ("Zl", "Zp"):
            out.append(" ")
        elif category not in ("Cc", "Cs") and ch not in _BIDI:
            out.append(ch)
    return "".join(out)


def untrusted_strings(value: Any) -> Any:
    """`untrusted_text` over every string of a JSON-like value, keys of a mapping too (N-19p, SEC6-01)."""
    if isinstance(value, str):
        return untrusted_text(value)
    if isinstance(value, dict):
        return {untrusted_strings(k): untrusted_strings(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [untrusted_strings(v) for v in value]
    return value


def received_filename(name: str | None) -> str:
    """The file name as it was received, for the record and for every place that shows it (N-19; browser QA P1-9; design
    review DS-15): any Unicode, with control characters removed. It is the last component of what the client sent (a
    browser can send a path) and at most `RECEIVED_MAX` characters, cut at the end of the stem. It is never a path on disk
    and never a download name: those are `safe_filename` of it. Markup in a name is kept; whatever prints a name escapes it."""
    base = re.split(r"[\\/]", name or "")[-1]
    kept = "".join(ch for ch in base if unicodedata.category(ch) not in ("Cc", "Cs", "Zl", "Zp") and ch not in _BIDI).strip()
    return _keeping_extension(kept, RECEIVED_MAX) if kept.strip(".") else "upload"


def safe_filename(name: str) -> str:
    """A name for a path on disk or a download: ASCII letters, digits, `.`, `_` and `-`, every other run of characters as
    one `_`, at most `SAFE_MAX` characters. For those two uses only (N-19): the record and the screen hold
    `received_filename`. The extension is never cut (a 121-character name ended ".jp"), and a name whose stem has no
    character of the set keeps its extension ("upload.jpg"; it was "jpg"). A name this function made is returned
    unchanged, so the path of an upload stored before N-19 is found from its stored name."""
    base = Path(name or "upload").name
    safe = _SAFE.sub("_", base).strip("._") or "upload"
    suffix = _SAFE.sub("", Path(base).suffix).rstrip(".")
    if len(suffix) > 1 and not safe.endswith(suffix):
        safe = f"upload{suffix}"
    return _keeping_extension(safe, SAFE_MAX)


def write_new_file(path: Path, data: bytes, mode: int = 0o600) -> bool:
    """Create `path` holding `data`, complete or not at all: a temporary file in the same directory is written and synced, then
    linked into place, which fails when the path exists. True when this call created the file, False when another writer did first.
    A concurrent reader never sees a partial file (CR P2-11)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        os.fchmod(fd, mode)
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        try:
            os.link(tmp, path)
        except FileExistsError:
            return False
        return True
    finally:
        os.unlink(tmp)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sniff(data: bytes, filename: str) -> tuple[str | None, str]:
    """Return (mime, ext) from magic bytes, falling back to extension for text-like docs."""
    kind = filetype.guess(data[:8192])
    ext = Path(filename).suffix.lower()
    if kind:
        return kind.mime, f".{kind.extension}"
    return None, ext


def _mpeg_layer3(head: bytes) -> bool:
    """An MPEG audio layer III frame header: frame sync, a valid version, bitrate and sample-rate index."""
    if len(head) < 4 or head[0] != 0xFF or (head[1] & 0xE0) != 0xE0:
        return False
    version, layer = (head[1] >> 3) & 0x03, (head[1] >> 1) & 0x03
    bitrate, rate = head[2] >> 4, (head[2] >> 2) & 0x03
    return version != 1 and layer == 1 and bitrate not in (0, 15) and rate != 3


def sniff_audio(head: bytes, filename: str) -> tuple[str, str] | None:
    """(mime, ext) of a WAV, MP3, M4A, Ogg Vorbis / FLAC, Opus or FLAC upload from its first bytes, else None (A-1).

    M4A needs an `M4A ` / `M4B ` brand, or a generic ISO brand with an .m4a / .m4b name (Android recorders), so an MP4 video
    is not accepted as audio. After an ID3 tag the bytes must be an MPEG layer III frame (or FLAC): a tag that runs past the
    sniffed head is refused, since anything (an HLS playlist, for one) can follow it. ffprobe checks the stream at `probe`."""
    ext = Path(filename).suffix.lower()
    if len(head) >= 12 and head[:4] == b"RIFF" and head[8:12] == b"WAVE":
        return "audio/wav", ".wav"
    if head[:4] == b"fLaC":
        return "audio/flac", ".flac"
    if head[:4] == b"OggS":
        if len(head) < 27:
            return None
        packet = head[27 + head[26]:27 + head[26] + 8]
        if packet.startswith(b"OpusHead"):
            return "audio/opus", ".opus"
        return ("audio/ogg", ".ogg") if packet.startswith((b"\x01vorbis", b"\x7fFLAC")) else None
    if len(head) >= 12 and head[4:8] == b"ftyp":
        end = min(int.from_bytes(head[:4], "big"), len(head))
        brands = {head[8:12], *(head[i:i + 4] for i in range(16, end - 3, 4))}
        if brands & M4A_BRANDS or (brands & ISO_BRANDS and ext in M4A_EXTS):
            return "audio/mp4", ".m4a"
        return None
    body = head
    if head[:3] == b"ID3" and len(head) >= 10:
        tag = 10 + ((head[6] & 0x7F) << 21 | (head[7] & 0x7F) << 14 | (head[8] & 0x7F) << 7 | (head[9] & 0x7F))
        tag += 10 if head[5] & 0x10 else 0  # footer
        if tag + 4 > len(head):  # the frame after the tag is not in the head: nothing verifies it is audio
            return None
        body = head[tag:]
        if body[:4] == b"fLaC":
            return "audio/flac", ".flac"
    return ("audio/mpeg", ".mp3") if _mpeg_layer3(body) else None


def validate_kind(kind: str, data: bytes, filename: str, size: int | None = None) -> tuple[str, str]:
    """Validate upload against declared kind from its first bytes. Returns (mime, ext). Raises ValueError."""
    mime, ext = sniff(data, filename)
    size = len(data) if size is None else size
    if kind == "image":
        if size > settings.max_image_bytes:
            raise ValueError(f"Image exceeds {settings.max_image_bytes // 1048576} MB limit")
        if mime not in IMAGE_MIMES:
            raise ValueError(f"Unsupported or unrecognised image type ({mime or ext})")
    elif kind == "video":
        if size > settings.max_video_bytes:
            raise ValueError(f"Video exceeds {settings.max_video_bytes // 1048576} MB limit")
        if mime not in VIDEO_MIMES:
            raise ValueError(f"Unsupported or unrecognised video type ({mime or ext})")
    elif kind == "audio":
        if size > settings.max_audio_bytes:
            raise ValueError(f"Audio exceeds {settings.max_audio_bytes // 1048576} MB limit")
        sniffed = sniff_audio(data, filename)
        if sniffed is None:
            raise ValueError(f"Unsupported or unrecognised audio type ({mime or ext}); accepted: WAV, MP3, M4A, OGG, OPUS, FLAC")
        mime, ext = sniffed
    elif kind == "document":
        if size > settings.max_document_bytes:
            raise ValueError(f"Document exceeds {settings.max_document_bytes // 1048576} MB limit")
        if mime == "application/pdf":
            ext = ".pdf"
        elif (mime in ("application/zip", "application/vnd.openxmlformats-officedocument.wordprocessingml.document") and Path(filename).suffix.lower() == ".docx") or (mime or "").endswith("wordprocessingml.document"):
            ext = ".docx"
            mime = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        elif mime is None and ext in DOC_EXTS:
            pass
        elif mime and mime.startswith("text/"):
            ext = ext if ext in DOC_EXTS else ".txt"
        else:
            raise ValueError(f"Unsupported document type ({mime or ext}); accepted: PDF, DOCX, TXT, MD, HTML")
        mime = mime or "text/plain"
    else:
        raise ValueError("Unknown analysis kind")
    return mime, ext


def store_upload(analysis_id: str, filename: str, data: bytes) -> Path:
    settings.ensure_dirs()
    d = settings.uploads_dir / analysis_id
    d.mkdir(parents=True, exist_ok=True)
    p = d / safe_filename(filename)
    p.write_bytes(data)
    return p


def stream_upload(analysis_id: str, filename: str, src, limit: int) -> tuple[Path, bytes, str, int]:
    """Stream a file-like object to disk in chunks while hashing. Returns (path, first_8kb, sha256, size).

    Never holds the upload in memory; raises ValueError when `limit` is exceeded (partial file removed).
    """
    settings.ensure_dirs()
    d = settings.uploads_dir / analysis_id
    d.mkdir(parents=True, exist_ok=True)
    p = d / safe_filename(filename)
    tmp = p.with_suffix(p.suffix + ".part")
    h = hashlib.sha256()
    head = b""
    size = 0
    try:
        with open(tmp, "wb") as f:
            while chunk := src.read(1 << 20):
                size += len(chunk)
                if size > limit:
                    raise ValueError(f"File exceeds the {limit // 1048576} MB limit")
                h.update(chunk)
                if len(head) < 8192:
                    head += chunk[: 8192 - len(head)]
                f.write(chunk)
        tmp.rename(p)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    return p, head, h.hexdigest(), size


def check_docx_safe(path: Path | BinaryIO) -> None:
    """Reject DOCX archives whose uncompressed size is implausible (zip bomb). Accepts a path or a seekable binary file."""
    import zipfile

    try:
        with zipfile.ZipFile(path) as z:
            total = sum(i.file_size for i in z.infolist())
            compressed = max(1, sum(i.compress_size for i in z.infolist()))
    except zipfile.BadZipFile as e:
        raise ValueError("The DOCX file is not a valid archive") from e
    if total > MAX_DOCX_UNCOMPRESSED or total / compressed > 200:
        raise ValueError("The DOCX file expands to an implausible size and was rejected")


def artifact_dir(analysis_id: str) -> Path:
    d = settings.artifacts_dir / analysis_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def report_path(analysis_id: str) -> Path:
    d = settings.reports_dir / analysis_id
    d.mkdir(parents=True, exist_ok=True)
    return d / "report.pdf"
