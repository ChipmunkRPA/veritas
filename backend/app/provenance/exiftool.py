"""ExifTool subprocess reader with a Pillow fallback (I-3; IF §4.3; D-07).

- **Fixed arguments.** `-config "" -G1 -a -s -n -j -api RequestAll=3 -` over the original bytes on stdin (IF §4.3), and
  `-config "" -b -ThumbnailImage -` for the embedded thumbnail. `-config ""` keeps a user configuration file from changing
  the output; reading stdin keeps host file-system tags out of the result.
- **Version floor** 12.24 (CVE-2021-22204: code execution through DjVu annotations). An older binary is not used.
- **Fallback.** Without a usable binary, or when it fails or times out, Pillow reads IFD0, the Exif IFD, GPS, IFD1, the
  thumbnail and PNG text under ExifTool's `Group:Tag` names, plus the raw XMP packet; the reason is recorded.
- **Maker-note format** comes from the raw maker-note block, never from Make: ExifTool chooses its maker-note parser from
  Make, so after a make swap it decodes one maker's notes as another's. Headers identify Nikon, Olympus, FujiFilm,
  Panasonic, Sony, Apple, Leica, Pentax, Ricoh, Sigma, Casio and Kodak; a headerless IFD starting with Canon's
  CameraSettings entry and holding a "Canon" image-type string is Canon. Anything else present is `unidentified`.
"""
from __future__ import annotations

import functools
import io
import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from typing import Any

import PIL
from PIL import ExifTags, Image, TiffImagePlugin

from app.core.config import settings
from app.detectors.image.metadata import read_png_text, read_xmp_bytes

READ_ARGS = ("-config", "", "-G1", "-a", "-s", "-n", "-j", "-api", "RequestAll=3", "-")
THUMBNAIL_ARGS = ("-config", "", "-b", "-ThumbnailImage", "-")
MIN_VERSION = (12, 24)
TIMEOUT_S = 20.0
UNIDENTIFIED = "unidentified"
NONDETERMINISTIC = frozenset({"SourceFile", "ExifTool:Now", "ExifTool:NewGUID", "ExifTool:ProcessingTime", "ExifTool:FileSequence"})
HOST_GROUPS = frozenset({"System", "MacOS"})
HOST_WARNING = 'Error running "stat"'  # RequestAll=3 asks for FileCreateDate, which stdin does not have
MAKERNOTE_GROUP_PREFIXES = ("Canon", "Nikon", "Sony", "Panasonic", "Leica", "Olympus", "FujiFilm", "Pentax", "Ricoh", "Apple",
                            "Samsung", "Casio", "Kodak", "Sigma", "DJI", "GoPro", "Minolta", "Sanyo", "Motorola", "Google", "JVC")
MAKERNOTE_HEADERS = ((b"Nikon\x00", "Nikon"), (b"OLYMPUS\x00", "Olympus"), (b"OLYMP\x00", "Olympus"), (b"OM SYSTEM", "Olympus"),
                     (b"FUJIFILM", "FujiFilm"), (b"Panasonic\x00", "Panasonic"), (b"SONY", "Sony"), (b"Apple iOS", "Apple"),
                     (b"LEICA", "Leica"), (b"AOC\x00", "Pentax"), (b"PENTAX ", "Pentax"), (b"RICOH", "Ricoh"), (b"Ricoh", "Ricoh"),
                     (b"SIGMA", "Sigma"), (b"QVC\x00", "Casio"), (b"KDK", "Kodak"), (b"EASTMAN KODAK", "Kodak"))
IFD_GROUPS = ((ExifTags.IFD.Exif, "ExifIFD"), (ExifTags.IFD.GPSInfo, "GPS"), (ExifTags.IFD.IFD1, "IFD1"),
              (ExifTags.IFD.Interop, "InteropIFD"))
POINTER_TAGS = frozenset({0x8769, 0x8825, 0xA005})
MAKER_NOTE_TAG = 0x927C
PILLOW_NAMES = {("IFD0", "DateTime"): "ModifyDate", ("ExifIFD", "DateTimeDigitized"): "CreateDate",
                ("ExifIFD", "SubsecTime"): "SubSecTime", ("ExifIFD", "SubsecTimeOriginal"): "SubSecTimeOriginal",
                ("ExifIFD", "SubsecTimeDigitized"): "SubSecTimeDigitized", ("ExifIFD", "ISOSpeedRatings"): "ISO",
                ("IFD1", "JpegIFOffset"): "ThumbnailOffset", ("IFD1", "JpegIFByteCount"): "ThumbnailLength"}
GPS_COORDINATES = frozenset({"GPSLatitude", "GPSLongitude", "GPSDestLatitude", "GPSDestLongitude"})
MAX_TEXT = 2000


@dataclass
class MetadataRead:
    tags: dict[str, Any]
    source: str  # exiftool | pillow
    version: str
    thumbnail: bytes | None = None
    xmp: str = ""
    maker_notes: str | None = None
    fallback_reason: str | None = None
    error: str | None = None

    @property
    def groups(self) -> set[str]:
        return {k.split(":", 1)[0] for k in self.tags if ":" in k}

    def describe(self) -> str:
        if self.source == "exiftool":
            return f"ExifTool {self.version}"
        return f"Pillow {self.version} (fallback: {self.fallback_reason})"


def _env() -> dict[str, str]:
    return {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "LC_ALL": "C"}


@functools.cache
def binary() -> tuple[str, str] | None:
    """(path, version) of a usable ExifTool binary, or None."""
    path = shutil.which(settings.exiftool_path)
    if path is None:
        return None
    try:
        out = subprocess.run([path, "-ver"], capture_output=True, timeout=10, check=False, env=_env())
    except (OSError, subprocess.SubprocessError):
        return None
    match = re.match(rb"\s*(\d+)\.(\d+)", out.stdout)
    if out.returncode != 0 or match is None or (int(match.group(1)), int(match.group(2))) < MIN_VERSION:
        return None
    return path, f"{int(match.group(1))}.{match.group(2).decode()}"


PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
PNG_TEXT_TYPES = (b"tEXt", b"zTXt", b"iTXt")
PNG_TEXT_KEPT = 64  # N-19o (SEC5-03): as `metadata.read_png_text` bounds the text the checks read
PNG_XMP_KEYWORD = b"XML:com.adobe.xmp\x00"


def bounded_png(payload: bytes) -> bytes:
    """N-19o (security re-check SEC5-03): a PNG with at most `PNG_TEXT_KEPT` text chunks, for ExifTool to read. It read every
    chunk: 32,000 compressed text chunks of a 1.43 MB file gave 32,013 tags in 0.85 s, and the consistency check worked
    through them. The chunks after the bound are left out of the copy (the XMP packet is read from the bytes themselves);
    any other file, and a PNG within the bound, is returned as it is.

    N-19p (SEC6-02, R10-04): the copy runs through IEND and drops what follows; at a chunk that runs past the end it stops
    and keeps what it copied (either sent the whole file to ExifTool: twelve bytes after IEND gave 600,014 tags in 19.8 s);
    it keeps the first iTXt chunk whose keyword is exactly `XML:com.adobe.xmp`, and any other such chunk counts against the
    bound (every chunk whose data began with the keyword was kept: 840,000 ran ExifTool to its timeout)."""
    if not payload.startswith(PNG_SIGNATURE):
        return payload
    out, at, kept, packet = [payload[:8]], 8, 0, False
    while at + 12 <= len(payload):
        length = int.from_bytes(payload[at:at + 4], "big")
        end = at + 12 + length
        if end > len(payload):
            break
        kind = payload[at + 4:at + 8]
        xmp = not packet and kind == b"iTXt" and payload[at + 8:end - 4].startswith(PNG_XMP_KEYWORD)
        packet = packet or xmp
        if kind in PNG_TEXT_TYPES and not xmp:
            kept += 1
            if kept > PNG_TEXT_KEPT:
                at = end
                continue
        out.append(payload[at:end])
        at = end
        if kind == b"IEND":
            break
    copy = b"".join(out)
    return payload if len(copy) == len(payload) else copy


def read(data: bytes | memoryview) -> MetadataRead:
    """All metadata of the original bytes. Never raises for content errors: an unreadable file sets `error`."""
    payload = bytes(data)
    found = binary()
    if found is None:
        return pillow_read(payload, reason=f"ExifTool {MIN_VERSION[0]}.{MIN_VERSION[1]} or later not found")
    path, version = found
    try:
        bounded = bounded_png(payload)  # N-19p (R10-04): both calls read the bounded copy
        out = subprocess.run([path, *READ_ARGS], input=bounded, capture_output=True, timeout=TIMEOUT_S, check=False, env=_env())
        records = json.loads(out.stdout or b"[]")
    except (OSError, subprocess.SubprocessError, ValueError) as e:
        return pillow_read(payload, reason=f"ExifTool failed ({type(e).__name__})")
    if not records or not isinstance(records[0], dict):
        return pillow_read(payload, reason="ExifTool returned no record")
    tags = {k: v for k, v in records[0].items() if k not in NONDETERMINISTIC and k.split(":", 1)[0] not in HOST_GROUPS
            and not (k == "ExifTool:Warning" and str(v).startswith(HOST_WARNING))}
    error = next((str(v) for k, v in tags.items() if k.endswith(":Error")), None)
    thumbnail = None
    if any(k.endswith((":ThumbnailImage", ":ThumbnailLength")) for k in tags):
        try:
            thumb = subprocess.run([path, *THUMBNAIL_ARGS], input=bounded, capture_output=True, timeout=TIMEOUT_S, check=False,
                                   env=_env())
            thumbnail = thumb.stdout if thumb.stdout.startswith(b"\xff\xd8") else None
        except (OSError, subprocess.SubprocessError):
            thumbnail = None
    blob = _maker_note_blob(payload)
    maker_notes = maker_note_format(blob) if blob else _family_from_groups(tags)
    return MetadataRead(tags=tags, source="exiftool", version=version, thumbnail=thumbnail, xmp=read_xmp_bytes(payload),
                        maker_notes=maker_notes, error=error)


# --- maker notes ------------------------------------------------------------------------------------------------

def _family_from_groups(tags: dict[str, Any]) -> str | None:
    """Presence only: ExifTool names the maker-note groups after the parser it chose from Make."""
    groups = {k.split(":", 1)[0] for k in tags if ":" in k}
    if any(g.startswith(MAKERNOTE_GROUP_PREFIXES) for g in groups) or any(k.split(":", 1)[-1].startswith("MakerNote") for k in tags):
        return UNIDENTIFIED
    return None


def _canon_ifd(blob: bytes) -> bool:
    for order in ("little", "big"):
        count = int.from_bytes(blob[0:2], order)
        if not 1 <= count <= 500 or len(blob) < 2 + 12 * count:
            continue
        first_tag, first_type = int.from_bytes(blob[2:4], order), int.from_bytes(blob[4:6], order)
        if (first_tag, first_type) == (0x0001, 3) and b"Canon" in blob:
            return True
    return False


def maker_note_format(blob: bytes) -> str:
    """The maker-note format by its own structure (header, or Canon's headerless IFD)."""
    family = next((family for prefix, family in MAKERNOTE_HEADERS if blob.startswith(prefix)), None)
    if family is not None:
        return family
    return "Canon" if _canon_ifd(blob) else UNIDENTIFIED


def _maker_note_blob(data: bytes) -> bytes | None:
    try:
        with Image.open(io.BytesIO(data)) as im:
            note = im.getexif().get_ifd(ExifTags.IFD.Exif).get(MAKER_NOTE_TAG)
    except Exception:  # not a still Pillow can parse (HEIC, video): presence comes from ExifTool's groups
        return None
    return note if isinstance(note, bytes) and note else None


# --- Pillow fallback --------------------------------------------------------------------------------------------

def _scalar(value: Any) -> Any:
    if isinstance(value, TiffImagePlugin.IFDRational):
        return None if value.denominator == 0 else round(float(value), 7)
    return value


def _value(value: Any, *, coordinate: bool = False, clock: bool = False) -> Any:
    if (coordinate or clock) and isinstance(value, tuple) and len(value) == 3:
        parts = [_scalar(v) for v in value]
        if any(p is None for p in parts):
            return None
        d, m, s = (float(p) for p in parts)
        return round(d + m / 60 + s / 3600, 7) if coordinate else f"{int(d):02d}:{int(m):02d}:{s:02.0f}"
    if isinstance(value, bytes):
        if len(value) <= 8 and all(b < 32 for b in value):
            return " ".join(str(b) for b in value)
        text = value.rstrip(b"\x00").decode("ascii", "ignore")
        return text[:MAX_TEXT] if text and text.isprintable() else f"(binary data {len(value)} bytes)"
    if isinstance(value, tuple):
        return " ".join(str(_scalar(v)) for v in value)
    if isinstance(value, str):
        return value.rstrip("\x00").strip()[:MAX_TEXT]
    return _scalar(value)


def _put(tags: dict[str, Any], group: str, values: dict[int, Any]) -> None:
    names = ExifTags.GPSTAGS if group == "GPS" else ExifTags.TAGS
    for tag_id, raw in values.items():
        if tag_id in POINTER_TAGS or (group == "ExifIFD" and tag_id == MAKER_NOTE_TAG):
            continue
        name = names.get(tag_id, f"0x{tag_id:04x}")
        name = PILLOW_NAMES.get((group, name), name)
        value = _value(raw, coordinate=group == "GPS" and name in GPS_COORDINATES, clock=group == "GPS" and name == "GPSTimeStamp")
        if value is not None and value != "":
            tags[f"{group}:{name}"] = value


def _pillow_thumbnail(im: Image.Image, exif: Image.Exif) -> bytes | None:
    raw = im.info.get("exif")
    if not raw:
        return None
    ifd1 = exif.get_ifd(ExifTags.IFD.IFD1)
    offset, length = ifd1.get(0x0201), ifd1.get(0x0202)
    if not offset or not length:
        return None
    tiff = raw[6:] if raw.startswith(b"Exif\x00\x00") else raw
    blob = tiff[int(offset):int(offset) + int(length)]
    return blob if blob.startswith(b"\xff\xd8") else None


def pillow_read(data: bytes, *, reason: str) -> MetadataRead:
    tags: dict[str, Any] = {}
    thumbnail = maker_notes = error = None
    try:
        with Image.open(io.BytesIO(data)) as im:
            tags["File:FileType"] = im.format or "unknown"
            if im.format in Image.MIME:
                tags["File:MIMEType"] = Image.MIME[im.format]
            tags["File:ImageWidth"], tags["File:ImageHeight"] = im.size
            exif = im.getexif()
            _put(tags, "IFD0", dict(exif))
            for ifd, group in IFD_GROUPS:
                try:
                    values = dict(exif.get_ifd(ifd))
                except KeyError:  # N-19p: Pillow asks the Exif IFD for the Interop pointer, which a file without EXIF lacks
                    continue
                if group == "ExifIFD" and isinstance(values.get(MAKER_NOTE_TAG), bytes) and values[MAKER_NOTE_TAG]:
                    maker_notes = maker_note_format(values[MAKER_NOTE_TAG])
                _put(tags, group, values)
            thumbnail = _pillow_thumbnail(im, exif)
            if "photoshop" in im.info:
                tags["Photoshop:ImageResources"] = "present"
            if im.format == "PNG":  # N-19p (SEC6-02): the text through the bound of the checks; it copied every string
                for key, value in read_png_text(im).items():
                    if isinstance(im.info.get(key), str):
                        tags[f"PNG:{key}"] = value[:MAX_TEXT]
    except Exception as e:  # untrusted input: any decoder error means "unreadable", never a crash
        error = f"the file could not be decoded ({type(e).__name__})"
    return MetadataRead(tags=tags, source="pillow", version=PIL.__version__, thumbnail=thumbnail, xmp=read_xmp_bytes(data),
                        maker_notes=maker_notes, fallback_reason=reason, error=error)
