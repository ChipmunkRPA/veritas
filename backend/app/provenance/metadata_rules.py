"""Metadata deep analysis: rules M1–M7 over an ExifTool or Pillow read (I-3; IF §4.3, §6.3; D-07, D-15).

- **Severities** (UX §3 `metadata_tree`): `info`; `notable`, processing after capture (editing is not deception);
  `anomaly`, fields that contradict each other; `documentary`, a generator declared in the file (M6).
- **Scored rules.** Only anomalies from M1–M5 count as evidence (`img.meta.consistency`). M6 is scored by
  `img.meta.generator` (self-declared, D-15); M7 informs conditions, never authenticity.
- **Thresholds are set a priori, not fitted:** date tolerance 2 s; GPS clock within 1 h of the capture time with its UTC
  offset, 15 h without one; thumbnail SSIM below 0.5; aspect ratios within 5 %.
- **M4** here covers maker notes and recorded dimensions; the quantisation-table half (make/model vs encoder) is I-4.
"""
from __future__ import annotations

import html
import io
import re
import zlib
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import numpy as np
from PIL import Image

from app.detectors.image.metadata import EDITOR_RE, INVOKEAI_PNG_KEYS, read_png_text
from app.provenance import exiftool
from app.provenance.exiftool import MetadataRead

SEVERITIES = ("info", "notable", "anomaly", "documentary")
SEVERITY_RANK = {s: i for i, s in enumerate(SEVERITIES)}
RULE_NAMES = {"M1": "date chain", "M2": "software trail", "M3": "thumbnail vs main image", "M4": "make and model vs maker notes",
              "M5": "GPS plausibility", "M6": "generator fingerprints", "M7": "absence patterns"}
SCORED_RULES = frozenset({"M1", "M2", "M3", "M4", "M5"})
DATE_TOLERANCE = timedelta(seconds=2)
MODIFIED_AFTER = timedelta(minutes=1)
GPS_CLOCK_TOLERANCE = timedelta(hours=1)
GPS_CLOCK_NO_OFFSET = timedelta(hours=15)  # UTC offsets span −12 h … +14 h
ALTITUDE_RANGE_M = (-500.0, 15000.0)
SSIM_ANOMALY = 0.5
ASPECT_TOLERANCE = 0.05
MIN_THUMBNAIL_PX = 32
MAX_TREE_ROWS = 400
MAX_VALUE_CHARS = 200
MAX_HISTORY = 100
PLATFORM_LONG_SIDES = frozenset({320, 480, 640, 720, 800, 960, 1024, 1080, 1200, 1280, 1440, 1600, 2048, 2560, 4096})
PHONE_MAKES = ("apple", "google", "samsung", "xiaomi", "huawei", "oneplus", "oppo", "vivo", "motorola", "honor", "realme")
MAKERNOTES_EXPECTED = ("canon", "nikon", "sony", "panasonic", "olympus", "om digital", "fujifilm", "pentax", "ricoh", "apple",
                       "leica", "casio", "sigma")
MAKER_ALIASES = {"Canon": ("canon",), "Nikon": ("nikon",), "Sony": ("sony", "hasselblad"), "Panasonic": ("panasonic", "leica"),
                 "Leica": ("leica", "panasonic"), "Olympus": ("olympus", "om digital", "om system"),
                 "FujiFilm": ("fujifilm", "fuji photo"), "Pentax": ("pentax", "asahi", "ricoh"), "Ricoh": ("ricoh", "pentax"),
                 "Apple": ("apple",), "Samsung": ("samsung",), "Casio": ("casio",), "Kodak": ("kodak",), "Sigma": ("sigma",),
                 "DJI": ("dji", "hasselblad"), "GoPro": ("gopro",), "Minolta": ("minolta", "konica", "sony"), "Sanyo": ("sanyo",),
                 "Motorola": ("motorola",), "Google": ("google",), "JVC": ("jvc", "victor")}
TREE_COLUMNS = ["group", "tag", "value", "severity", "rule", "note"]
TIMESTAMP_COLUMNS = ["time as written", "tag", "UTC offset", "relative to DateTimeOriginal"]
HISTORY_COLUMNS = ["#", "action", "when", "software agent", "changed"]
TIMESTAMP_SKIP_GROUPS = frozenset({"File", "ExifTool", "Composite"})
DATE_RE = re.compile(r"^(\d{4}):(\d{2}):(\d{2})[ T](\d{2}):(\d{2})(?::(\d{2}))?(\.\d+)?(Z|[+-]\d{2}:?\d{2})?$")
OFFSET_RE = re.compile(r"^([+-])(\d{2}):?(\d{2})$")
STEALTH_MAGIC = {b"stealth_pnginfo": False, b"stealth_pngcomp": True}
STEALTH_HEADER_BITS = 152  # 15-byte magic + 32-bit payload length in bits
MAX_STEALTH_BYTES = 1 << 20
STD_LUMA = np.array([16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57, 69, 56,
                     14, 17, 22, 29, 51, 87, 80, 62, 18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92,
                     49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99], dtype=np.int64)
STD_CHROMA = np.array([17, 18, 24, 47, 99, 99, 99, 99, 18, 21, 26, 66, 99, 99, 99, 99, 24, 26, 56, 99, 99, 99, 99, 99,
                       47, 66, 99, 99, 99, 99, 99, 99] + [99] * 32, dtype=np.int64)
ZIGZAG = np.array([0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6, 7, 14,
                   21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59, 52, 45, 38, 31, 39, 46, 53,
                   60, 61, 54, 47, 55, 62, 63])


@dataclass(frozen=True)
class Hit:
    rule: str
    code: str
    severity: str
    observed: str
    expected: str
    note: str
    tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class Stamp:
    local: datetime  # the wall-clock time as written
    offset: timedelta | None

    def utc(self) -> datetime | None:
        return None if self.offset is None else self.local - self.offset


@dataclass(frozen=True)
class ThumbnailComparison:
    width: int
    height: int
    ssim: float | None  # None: the shapes differ, so no comparison was made
    max_block_diff: float | None
    rotation: int | None
    composite_png: bytes | None


@dataclass
class MetadataAnalysis:
    read: MetadataRead
    hits: list[Hit]
    history: list[dict[str, str]]
    timestamps: list[tuple[str, str, Stamp]]
    thumbnail: ThumbnailComparison | None
    format: str | None
    size: tuple[int, int] | None

    @property
    def readable(self) -> bool:
        return self.size is not None or bool(self.read.groups - {"ExifTool", "File"})

    @property
    def anomalies(self) -> list[Hit]:
        return [h for h in self.hits if h.severity == "anomaly" and h.rule in SCORED_RULES]

    @property
    def notable(self) -> list[Hit]:
        return [h for h in self.hits if h.severity == "notable"]


# --- dates ------------------------------------------------------------------------------------------------------

def parse_offset(value: Any) -> timedelta | None:
    match = OFFSET_RE.match(str(value or "").strip())
    if match is None:
        return None
    sign = -1 if match.group(1) == "-" else 1
    return sign * timedelta(hours=int(match.group(2)), minutes=int(match.group(3)))


def parse_date(value: Any, *, subsec: Any = None, offset: Any = None) -> Stamp | None:
    """An EXIF / XMP date as written. Unset clocks (`0000:00:00 …`) and impossible dates give None."""
    match = DATE_RE.match(str(value or "").strip())
    if match is None:
        return None
    y, mo, d, h, mi = (int(match.group(i)) for i in range(1, 6))
    try:
        local = datetime(y, mo, d, h, mi, int(match.group(6) or 0))
    except ValueError:
        return None
    fraction = match.group(7)
    digits = re.sub(r"\D", "", str(subsec)) if subsec not in (None, "") else ""
    if fraction or digits:
        local += timedelta(microseconds=round(float(fraction or f"0.{digits}") * 1_000_000))
    zone = match.group(8)
    tz = timedelta(0) if zone == "Z" else parse_offset(zone) if zone else parse_offset(offset)
    return Stamp(local, tz)


def _diff(a: Stamp, b: Stamp) -> timedelta:
    """a − b: in UTC when both carry an offset, otherwise as wall-clock times."""
    if a.offset is not None and b.offset is not None:
        return a.local - a.offset - (b.local - b.offset)
    return a.local - b.local


def human_delta(delta: timedelta) -> str:
    seconds = abs(delta.total_seconds())
    if seconds < 90:
        return f"{seconds:.0f} s"
    if seconds < 5400:
        return f"{seconds / 60:.0f} min"
    if seconds < 172800:
        return f"{seconds / 3600:.0f} h"
    return f"{seconds / 86400:.0f} days"


def _gps_datetime(tags: Mapping[str, Any]) -> datetime | None:
    date, clock = str(tags.get("GPS:GPSDateStamp") or ""), str(tags.get("GPS:GPSTimeStamp") or "")
    stamp = parse_date(f"{date} {clock.split('.')[0]}") if date and clock else None
    return stamp.local if stamp else None


def _capture(tags: Mapping[str, Any]) -> Stamp | None:
    return parse_date(tags.get("ExifIFD:DateTimeOriginal"), subsec=tags.get("ExifIFD:SubSecTimeOriginal"),
                      offset=tags.get("ExifIFD:OffsetTimeOriginal"))


# --- M1 date chain ----------------------------------------------------------------------------------------------

def rule_m1(read: MetadataRead) -> list[Hit]:
    t = read.tags
    original = _capture(t)
    digitised = parse_date(t.get("ExifIFD:CreateDate"), subsec=t.get("ExifIFD:SubSecTimeDigitized"),
                           offset=t.get("ExifIFD:OffsetTimeDigitized"))
    modified = parse_date(t.get("IFD0:ModifyDate"), subsec=t.get("ExifIFD:SubSecTime"), offset=t.get("ExifIFD:OffsetTime"))
    hits: list[Hit] = []
    if original and digitised and _diff(original, digitised) > DATE_TOLERANCE:
        hits.append(Hit("M1", "M1.original_after_digitised", "anomaly",
                        f"DateTimeOriginal {t['ExifIFD:DateTimeOriginal']} is {human_delta(_diff(original, digitised))} later "
                        f"than CreateDate {t['ExifIFD:CreateDate']}",
                        "DateTimeOriginal ≤ CreateDate (a camera writes both at capture)", "Date fields are mutually inconsistent",
                        ("ExifIFD:DateTimeOriginal", "ExifIFD:CreateDate")))
    capture = original or digitised
    if capture and modified:
        if _diff(capture, modified) > DATE_TOLERANCE:
            hits.append(Hit("M1", "M1.modified_before_capture", "anomaly",
                            f"ModifyDate {t['IFD0:ModifyDate']} is {human_delta(_diff(capture, modified))} earlier than the capture time",
                            "ModifyDate ≥ the capture time", "Date fields are mutually inconsistent",
                            ("IFD0:ModifyDate", "ExifIFD:DateTimeOriginal")))
        elif _diff(modified, capture) > MODIFIED_AFTER:
            hits.append(Hit("M1", "M1.modified_after_capture", "notable",
                            f"ModifyDate {t['IFD0:ModifyDate']}, {human_delta(_diff(modified, capture))} after capture",
                            "ModifyDate equal to the capture time on an unedited camera file",
                            f"Modified {human_delta(_diff(modified, capture))} after capture", ("IFD0:ModifyDate",)))
    for xmp_key, exif_stamp, exif_key in (("XMP-exif:DateTimeOriginal", original, "ExifIFD:DateTimeOriginal"),
                                          ("XMP-photoshop:DateCreated", original, "ExifIFD:DateTimeOriginal"),
                                          ("XMP-xmp:CreateDate", digitised, "ExifIFD:CreateDate")):
        copy = parse_date(t.get(xmp_key))
        if copy and exif_stamp and abs(copy.local - exif_stamp.local) > DATE_TOLERANCE:
            hits.append(Hit("M1", "M1.xmp_differs", "anomaly",
                            f"{xmp_key.split(':')[1]} {t[xmp_key]} differs from {exif_key.split(':')[1]} {t[exif_key]} by "
                            f"{human_delta(copy.local - exif_stamp.local)}",
                            "the XMP copy of a date equals the EXIF date", "Date fields are mutually inconsistent", (xmp_key, exif_key)))
    make = str(t.get("IFD0:Make") or "").lower()
    subsec_keys = ("ExifIFD:SubSecTimeOriginal", "ExifIFD:SubSecTimeDigitized", "ExifIFD:SubSecTime")
    if original and any(p in make for p in PHONE_MAKES) and all(t.get(k) in (None, "") for k in subsec_keys):
        hits.append(Hit("M1", "M1.subseconds_missing", "notable", f"Make {t['IFD0:Make']}; no SubSecTime fields",
                        "phones record sub-second capture times", "Sub-second times removed (common after export)", ("IFD0:Make",)))
    return hits


# --- M2 software trail ------------------------------------------------------------------------------------------

HISTORY_OPENING_CHARS, HISTORY_ENTRY_CHARS = 2000, 4096  # N-19o (SEC5-02): the bounds of one entry
_WORD = re.compile(r"\w")


def _history_entries(block: str):
    """The text of each `<rdf:li …/>` and `<rdf:li …>…</rdf:li>` of a History block, in one forward pass (N-19o, security
    re-check SEC5-02): the regular expression it replaces scanned to the end from every opening that had no `>` (about 2 s
    for a packet at the 64 KB cut). An opening's attributes are bounded to `HISTORY_OPENING_CHARS`; where no `</rdf:li>`
    follows an opening, none follows a later one, and the reading ends."""
    at, opening = 0, "<rdf:li"
    while (p := block.find(opening, at)) >= 0:
        after = p + len(opening)
        at = after
        if _WORD.match(block, after):  # `<rdf:listing>`: not an entry (the `\b` of the expression it replaces)
            continue
        gt = block.find(">", after, after + HISTORY_OPENING_CHARS + 1)
        if gt < 0:
            continue
        attributes = block[after:gt]
        if attributes.endswith("/"):
            yield attributes[:-1]
            at = gt + 1
            continue
        close = block.find("</rdf:li>", gt + 1)
        if close < 0:
            return
        yield attributes + block[gt + 1:close]
        at = close + len("</rdf:li>")


def xmp_history(xmp: str) -> list[dict[str, str]]:
    """`xmpMM:History` entries in document order, read from the XMP packet (ExifTool flattens missing fields). N-19o
    (SEC5-02): the block found with `str.find` and read in one pass; each entry cut to `HISTORY_ENTRY_CHARS` before its
    fields are searched."""
    start = xmp.find("<xmpMM:History")
    if start < 0 or _WORD.match(xmp, start + len("<xmpMM:History")):
        return []
    head_end = xmp.find(">", start)
    end = xmp.find("</xmpMM:History>", head_end + 1) if head_end >= 0 else -1
    if end < 0:
        return []
    out: list[dict[str, str]] = []
    for whole in _history_entries(xmp[head_end + 1:end]):
        text = whole[:HISTORY_ENTRY_CHARS]
        entry = {}
        for name in ("action", "when", "softwareAgent", "changed"):
            match = re.search(rf'stEvt:{name}\s*=\s*"([^"]*)"', text) or re.search(rf"<stEvt:{name}>(.*?)</stEvt:{name}>", text, re.S)
            if match:
                entry[name] = html.unescape(match.group(1)).strip()[:MAX_VALUE_CHARS]
        if entry:
            out.append(entry)
        if len(out) >= MAX_HISTORY:
            break
    return out


def rule_m2(read: MetadataRead, history: list[dict[str, str]]) -> list[Hit]:
    t = read.tags
    software, creator = str(t.get("IFD0:Software") or "").strip(), str(t.get("XMP-xmp:CreatorTool") or "").strip()
    hits: list[Hit] = []
    if history:
        agents = list(dict.fromkeys(e["softwareAgent"] for e in history if e.get("softwareAgent")))
        actions = ", ".join(e.get("action", "not stated") for e in history[:8])
        hits.append(Hit("M2", "M2.edit_history", "notable",
                        f"{len(history)} history entries ({actions})" + (f" by {', '.join(agents[:3])}" if agents else ""),
                        "a camera original has no edit history",
                        f"Edited in {agents[0]} after capture" if agents else "Edit history recorded",
                        ("XMP-xmpMM:HistoryAction", "XMP-xmpMM:HistorySoftwareAgent")))
    else:
        named = next(((k, v) for k, v in (("IFD0:Software", software), ("XMP-xmp:CreatorTool", creator)) if v and EDITOR_RE.search(v)), None)
        if named:
            hits.append(Hit("M2", "M2.editor_named", "notable", f"{named[0].split(':')[1]} = {named[1]}",
                            "a camera original names the camera firmware", f"Edited in {named[1]} after capture", (named[0],)))
    traces = []
    if "Photoshop" in read.groups:
        traces.append("Photoshop image resources")
    document, original = str(t.get("XMP-xmpMM:DocumentID") or ""), str(t.get("XMP-xmpMM:OriginalDocumentID") or "")
    if document and original and document != original:
        traces.append("a document ID that differs from the original document ID")
    if t.get("XMP-xmpMM:DerivedFromDocumentID") or t.get("XMP-xmpMM:DerivedFromInstanceID"):
        traces.append("a derived-from reference")
    if traces and not history and not software and not creator:
        hits.append(Hit("M2", "M2.trail_removed", "notable",
                        "edit traces remain (" + "; ".join(traces) + ") but no Software, CreatorTool or edit history is recorded",
                        "software that leaves these traces also records its name or an edit history",
                        "Software trail removed or never written", ("XMP-xmpMM:DocumentID", "XMP-xmpMM:OriginalDocumentID")))
    return hits


# --- M3 thumbnail vs main image ---------------------------------------------------------------------------------

def _trim_borders(a: np.ndarray, level: float = 16.0) -> np.ndarray:
    """Remove uniform dark letterbox bands (cameras pad 4:3 thumbnails of 3:2 sensors)."""
    h, w = a.shape
    top, bottom, left, right = 0, h, 0, w
    while top < h // 4 and a[top].mean() < level and a[top].std() < 4:
        top += 1
    while bottom > h - h // 4 and a[bottom - 1].mean() < level and a[bottom - 1].std() < 4:
        bottom -= 1
    while left < w // 4 and a[top:bottom, left].mean() < level and a[top:bottom, left].std() < 4:
        left += 1
    while right > w - w // 4 and a[top:bottom, right - 1].mean() < level and a[top:bottom, right - 1].std() < 4:
        right -= 1
    return a[top:bottom, left:right]


def _ssim(a: np.ndarray, b: np.ndarray) -> float:
    from skimage.metrics import structural_similarity

    win = min(7, *(s if s % 2 else s - 1 for s in a.shape))
    return float(structural_similarity(a, b, data_range=255.0, win_size=win))


def _max_block_mean(diff: np.ndarray, block: int = 8) -> float:
    h, w = (diff.shape[0] // block) * block, (diff.shape[1] // block) * block
    if h == 0 or w == 0:
        return float(diff.mean())
    return float(diff[:h, :w].reshape(h // block, block, w // block, block).mean(axis=(1, 3)).max())


def _composite(a: np.ndarray, b: np.ndarray, diff: np.ndarray) -> bytes:
    gap = np.full((a.shape[0], 4), 255.0)
    strip = Image.fromarray(np.hstack([a, gap, b, gap, np.clip(diff * 4, 0, 255)]).astype(np.uint8), "L")
    scale = max(1, 240 // a.shape[0])
    if scale > 1:
        strip = strip.resize((strip.width * scale, strip.height * scale), Image.Resampling.NEAREST)
    buf = io.BytesIO()
    strip.save(buf, "PNG")
    return buf.getvalue()


def compare_thumbnail(thumbnail: bytes, data: bytes) -> ThumbnailComparison | None:
    try:
        with Image.open(io.BytesIO(thumbnail)) as im:
            thumb = _trim_borders(np.asarray(im.convert("L"), dtype=np.float64))
        th, tw = thumb.shape
        if min(th, tw) < MIN_THUMBNAIL_PX:
            return None
        with Image.open(io.BytesIO(data)) as im:
            if im.format == "JPEG":
                im.draft("L", (tw * 2, th * 2))
            main = im.convert("L")
    except Exception:  # undecodable thumbnail or main image
        return None
    candidates = [r for r in (0, 90, 270)
                  if abs((tw / th) / ((main.width / main.height) if r == 0 else (main.height / main.width)) - 1) <= ASPECT_TOLERANCE]
    if not candidates:
        return ThumbnailComparison(tw, th, None, None, None, None)
    best: tuple[float, int, np.ndarray] | None = None
    for rotation in candidates:
        rotated = main.rotate(rotation, expand=True) if rotation else main
        resized = np.asarray(rotated.resize((tw, th), Image.Resampling.LANCZOS), dtype=np.float64)
        score = _ssim(thumb, resized)
        if best is None or score > best[0]:
            best = (score, rotation, resized)
    score, rotation, resized = best  # type: ignore[misc]
    diff = np.abs(thumb - resized)
    return ThumbnailComparison(tw, th, round(score, 4), round(_max_block_mean(diff), 2), rotation, _composite(thumb, resized, diff))


def rule_m3(read: MetadataRead, comparison: ThumbnailComparison | None, size: tuple[int, int] | None) -> list[Hit]:
    if read.thumbnail is None:
        return []
    tags = ("IFD1:ThumbnailImage", "IFD1:ThumbnailLength")
    if comparison is None:
        return [Hit("M3", "M3.not_assessed", "info", "the embedded thumbnail could not be decoded or is below 32 px",
                    "a decodable thumbnail", "Thumbnail not compared", tags)]
    if comparison.ssim is None:
        main = f"{size[0]}×{size[1]}" if size else "unknown size"
        return [Hit("M3", "M3.aspect_differs", "notable",
                    f"thumbnail {comparison.width}×{comparison.height} (bands removed) vs main image {main}",
                    "the same aspect ratio within 5 %, or rotated",
                    "Thumbnail and main image differ in shape: cropped after the thumbnail was written, or the thumbnail was regenerated",
                    tags)]
    if comparison.ssim < SSIM_ANOMALY:
        return [Hit("M3", "M3.thumbnail_differs", "anomaly",
                    f"SSIM {comparison.ssim:.2f} at {comparison.width}×{comparison.height}; largest 8×8 block difference "
                    f"{comparison.max_block_diff:.0f} of 255",
                    f"SSIM ≥ {SSIM_ANOMALY:.2f} when the thumbnail was written from these pixels",
                    "Content differs between the embedded thumbnail and the main image", tags)]
    return []


# --- M4 make and model vs maker notes ---------------------------------------------------------------------------

def rule_m4(read: MetadataRead, size: tuple[int, int] | None) -> list[Hit]:
    t = read.tags
    make, model = str(t.get("IFD0:Make") or "").strip(), str(t.get("IFD0:Model") or "").strip()
    hits: list[Hit] = []
    if make:
        low, family = make.lower(), read.maker_notes
        if family is not None and family != exiftool.UNIDENTIFIED:
            if not any(alias in low for alias in MAKER_ALIASES.get(family, (family.lower(),))):
                hits.append(Hit("M4", "M4.maker_notes_mismatch", "anomaly",
                                f"Make = {make}, Model = {model or 'not stated'}; the maker notes are in {family}'s format",
                                "maker notes in the format of the claimed make",
                                f"Claimed camera {make} {model}".rstrip() + f"; the maker notes are {family}'s", ("IFD0:Make", "IFD0:Model")))
        elif family is None and any(m in low for m in MAKERNOTES_EXPECTED):
            hits.append(Hit("M4", "M4.maker_notes_absent", "notable", f"Make = {make}; no maker notes",
                            f"{make} cameras write maker notes",
                            "Maker notes removed (common after editing or export), or the make was written by other software",
                            ("IFD0:Make",)))
    if size is not None:
        for wk, hk in (("ExifIFD:ExifImageWidth", "ExifIFD:ExifImageHeight"), ("Canon:CanonImageWidth", "Canon:CanonImageHeight")):
            try:
                w, h = int(t[wk]), int(t[hk])
            except (KeyError, TypeError, ValueError):
                continue
            if w > 0 and h > 0 and (w, h) not in (size, size[::-1]):
                hits.append(Hit("M4", "M4.dimensions_differ", "notable",
                                f"pixel dimensions {size[0]}×{size[1]}; {wk.split(':')[1]} and {hk.split(':')[1]} record {w}×{h}",
                                "the recorded output size", "Resized or cropped after capture", (wk, hk)))
                break
    return hits


# --- M5 GPS plausibility ----------------------------------------------------------------------------------------

def _float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def rule_m5(read: MetadataRead) -> list[Hit]:
    t = read.tags
    lat, lon = _float(t.get("GPS:GPSLatitude")), _float(t.get("GPS:GPSLongitude"))
    if lat is None or lon is None:
        return []
    position = ("GPS:GPSLatitude", "GPS:GPSLongitude")
    lat_ref, lon_ref = str(t.get("GPS:GPSLatitudeRef") or "").strip().upper()[:1], str(t.get("GPS:GPSLongitudeRef") or "").strip().upper()[:1]
    hits: list[Hit] = []
    if not (0 <= lat <= 90 and 0 <= lon <= 180) or (lat_ref and lat_ref not in "NS") or (lon_ref and lon_ref not in "EW"):
        hits.append(Hit("M5", "M5.position_out_of_range", "anomaly", f"latitude {lat:g} {lat_ref}, longitude {lon:g} {lon_ref}",
                        "latitude 0–90 N/S, longitude 0–180 E/W", "GPS block internally inconsistent: position out of range", position))
    elif lat == 0 and lon == 0:
        hits.append(Hit("M5", "M5.null_island", "anomaly", "latitude 0, longitude 0",
                        "a position fix is almost never exactly 0° N, 0° E",
                        "GPS block internally inconsistent: zero position (some devices write zeros without a fix)", position))
    altitude = _float(t.get("GPS:GPSAltitude"))
    if altitude is not None:
        signed = -altitude if str(t.get("GPS:GPSAltitudeRef")).strip() in ("1", "1.0") else altitude
        if not ALTITUDE_RANGE_M[0] <= signed <= ALTITUDE_RANGE_M[1]:
            hits.append(Hit("M5", "M5.altitude", "anomaly", f"altitude {signed:g} m",
                            f"between {ALTITUDE_RANGE_M[0]:g} m and {ALTITUDE_RANGE_M[1]:g} m",
                            "GPS block internally inconsistent: implausible altitude", ("GPS:GPSAltitude",)))
    datum = str(t.get("GPS:GPSMapDatum") or "").strip()
    if datum and re.sub(r"[\s_-]", "", datum).upper() not in ("WGS84", "WGS1984"):
        hits.append(Hit("M5", "M5.datum", "notable", f"GPSMapDatum = {datum}", "WGS-84, which receivers report",
                        "Unusual map datum: the position may have been entered by software", ("GPS:GPSMapDatum",)))
    satellite, capture = _gps_datetime(t), _capture(t)
    if satellite and capture:
        with_offset = capture.offset is not None
        gap = abs(satellite - capture.local + capture.offset) if with_offset else abs(satellite - capture.local)  # type: ignore[operator]
        tolerance = GPS_CLOCK_TOLERANCE if with_offset else GPS_CLOCK_NO_OFFSET
        if gap > tolerance:
            hits.append(Hit("M5", "M5.gps_clock", "anomaly",
                            f"GPS time {t['GPS:GPSDateStamp']} {t['GPS:GPSTimeStamp']} UTC vs DateTimeOriginal "
                            f"{t['ExifIFD:DateTimeOriginal']}" + (f" {t['ExifIFD:OffsetTimeOriginal']}" if with_offset else "")
                            + f": {human_delta(gap)} apart",
                            f"within {human_delta(tolerance)} of the capture time" + ("" if with_offset else " (no UTC offset recorded)"),
                            "GPS time and capture time disagree", ("GPS:GPSDateStamp", "GPS:GPSTimeStamp", "ExifIFD:DateTimeOriginal")))
    return hits


# --- M6 generator fingerprints ----------------------------------------------------------------------------------

def stealth_pnginfo(image: Image.Image) -> str | None:
    """Payload hidden in the alpha channel's least significant bits, read column by column (x outer, y inner): a 15-byte
    magic (`stealth_pnginfo`, or `stealth_pngcomp` for gzip), a 32-bit payload length in bits, then the payload."""
    if "A" not in image.getbands():
        return None
    alpha = np.asarray(image.getchannel("A"), dtype=np.uint8)
    h, w = alpha.shape
    if h * w < STEALTH_HEADER_BITS:
        return None
    header = (alpha[:, :min(w, -(-STEALTH_HEADER_BITS // h))].T & 1).reshape(-1)[:STEALTH_HEADER_BITS]
    magic = np.packbits(header[:120]).tobytes()
    if magic not in STEALTH_MAGIC:
        return None
    length = int.from_bytes(np.packbits(header[120:152]).tobytes(), "big")
    if length <= 0 or length > min(8 * MAX_STEALTH_BYTES, h * w - STEALTH_HEADER_BITS):
        return None
    total = STEALTH_HEADER_BITS + length
    bits = (alpha[:, :min(w, -(-total // h))].T & 1).reshape(-1)[STEALTH_HEADER_BITS:total]
    payload = np.packbits(bits).tobytes()
    if STEALTH_MAGIC[magic]:
        try:
            payload = zlib.decompressobj(16 + zlib.MAX_WBITS).decompress(payload, MAX_STEALTH_BYTES)
        except zlib.error:
            return None
    return payload.decode("utf-8", "replace")[:MAX_VALUE_CHARS * 20]


def generator_fingerprints(png_text: Mapping[str, str], make: Any, software: Any, image: Image.Image | None = None) -> list[tuple[str, str]]:
    """(generator, field) pairs for the IF §4.3 M6 fingerprints. Every one can be written into any file (D-15)."""
    found: list[tuple[str, str]] = []

    def add(name: str, where: str) -> None:
        if (name, where) not in found:
            found.append((name, where))

    keys = {k.lower(): (k, str(v)) for k, v in png_text.items()}
    parameters = keys.get("parameters", ("", ""))[1].lower()
    if "steps:" in parameters and ("sampler:" in parameters or "seed:" in parameters):
        add("Stable Diffusion web UI (A1111 / Forge)", f"PNG text '{keys['parameters'][0]}'")
    for key in ("prompt", "workflow"):
        if key in keys and ('"class_type"' in keys[key][1] or re.search(r'"nodes"\s*:', keys[key][1])):
            add("ComfyUI", f"PNG text '{keys[key][0]}'")
    for key in INVOKEAI_PNG_KEYS:
        if key in keys:
            add("InvokeAI", f"PNG text '{keys[key][0]}'")
    if "fooocus_scheme" in keys:
        add("Fooocus", f"PNG text '{keys['fooocus_scheme'][0]}'")
    for where, value in (("EXIF Software", software), ("PNG text 'Software'", keys.get("software", ("", ""))[1])):
        text = str(value or "").strip()
        if text.lower().startswith("fooocus v"):
            add("Fooocus", where)
        if text == "NovelAI":
            add("NovelAI", where)
            comment = keys.get("comment", ("", ""))[1]
            if comment.lstrip().startswith("{") and '"steps"' in comment:
                add("NovelAI", f"PNG text '{keys['comment'][0]}'")
        if re.fullmatch(r"AI generated;\s*(?:txt2img|img2img)\s*;\s*flux", text, re.IGNORECASE):
            add("FLUX reference CLI", where)
    if str(make or "").strip().lower() == "black forest labs":
        add("FLUX reference CLI", "EXIF Make")
    if image is not None and image.format == "PNG" and stealth_pnginfo(image) is not None:
        add("Stealth pnginfo payload (NovelAI / web UI extension)", "alpha-channel least significant bits")
    return found


# --- M7 absence patterns ----------------------------------------------------------------------------------------

def ijg_table(base: np.ndarray, quality: int) -> np.ndarray:
    scale = 5000 // quality if quality < 50 else 200 - 2 * quality
    return np.clip((base * scale + 50) // 100, 1, 255)


def standard_quality(quantization: Mapping[int, Any] | None) -> int | None:
    """The libjpeg quality whose standard luma and chroma tables equal the file's, in natural or zigzag order."""
    if not quantization or 0 not in quantization or 1 not in quantization:
        return None
    luma, chroma = np.array(list(quantization[0])[:64]), np.array(list(quantization[1])[:64])
    for quality in range(1, 101):
        lt, ct = ijg_table(STD_LUMA, quality), ijg_table(STD_CHROMA, quality)
        if (np.array_equal(luma, lt) and np.array_equal(chroma, ct)) or (np.array_equal(luma, lt[ZIGZAG]) and np.array_equal(chroma, ct[ZIGZAG])):
            return quality
    return None


def rule_m7(read: MetadataRead, fmt: str | None, size: tuple[int, int] | None, quantization: Mapping[int, Any] | None) -> list[Hit]:
    if read.groups & {"IFD0", "ExifIFD", "GPS", "IFD1"} or read.xmp or size is None:
        return []
    quality = standard_quality(quantization) if fmt == "JPEG" else None
    if quality is not None and max(size) in PLATFORM_LONG_SIDES:
        return [Hit("M7", "M7.platform_pattern", "info",
                    f"no EXIF or XMP, standard libjpeg tables (quality {quality}), long side {max(size)} px",
                    "camera files carry EXIF and device tables",
                    "Consistent with a platform re-encode that removed the metadata: informs conditions, not authenticity")]
    return [Hit("M7", "M7.no_metadata", "info", "no EXIF or XMP", "camera files carry EXIF",
                "No capture metadata: uninformative on its own")]


# --- analysis and exhibits --------------------------------------------------------------------------------------

def timestamps(read: MetadataRead) -> list[tuple[str, str, Stamp]]:
    out: list[tuple[str, str, Stamp]] = []
    for key, value in read.tags.items():
        if ":" not in key or key.split(":", 1)[0] in TIMESTAMP_SKIP_GROUPS:
            continue
        values = value if isinstance(value, list) else [value]
        for i, item in enumerate(values[:50]):
            stamp = parse_date(item) if isinstance(item, str) else None
            if stamp is not None:
                out.append((key if len(values) == 1 else f"{key}[{i + 1}]", item, stamp))
    satellite = _gps_datetime(read.tags)
    if satellite is not None:
        out.append(("GPS:GPSDateStamp + GPSTimeStamp", f"{satellite:%Y:%m:%d %H:%M:%S}Z", Stamp(satellite, timedelta(0))))
    return sorted(out, key=lambda row: (row[2].local, row[0]))


def analyse(data: bytes | memoryview, *, read: MetadataRead | None = None) -> MetadataAnalysis:
    payload = bytes(data)
    read = read or exiftool.read(payload)
    fmt, size, quantization, fingerprints = None, None, None, []
    try:
        with Image.open(io.BytesIO(payload)) as im:
            fmt, size, quantization = im.format, im.size, getattr(im, "quantization", None)
            png_text = read_png_text(im)  # N-19o (SEC5-03): bounded as the generator check reads it
            fingerprints = generator_fingerprints(png_text, read.tags.get("IFD0:Make"), read.tags.get("IFD0:Software"), im)
    except Exception:  # not an image Pillow can open: metadata rules still run on what ExifTool read
        pass
    history = xmp_history(read.xmp)
    comparison = compare_thumbnail(read.thumbnail, payload) if read.thumbnail and size else None
    hits = [*rule_m1(read), *rule_m2(read, history), *rule_m3(read, comparison, size), *rule_m4(read, size), *rule_m5(read)]
    for name, where in fingerprints:
        tag = {"EXIF Make": "IFD0:Make", "EXIF Software": "IFD0:Software"}.get(where)
        hits.append(Hit("M6", "M6.generator_declared", "documentary", f"{name} in {where}",
                        "camera and editor files name a camera or an editor in these fields",
                        f"The file declares generator {name} in {where}", (tag,) if tag else ()))
    hits += rule_m7(read, fmt, size, quantization)
    return MetadataAnalysis(read=read, hits=hits, history=history, timestamps=timestamps(read), thumbnail=comparison, format=fmt, size=size)


def _text(value: Any) -> str:
    text = "; ".join(str(v) for v in value) if isinstance(value, list) else f"{value:g}" if isinstance(value, float) else str(value)
    return text if len(text) <= MAX_VALUE_CHARS else text[:MAX_VALUE_CHARS - 1] + "…"


def tree_rows(analysis: MetadataAnalysis) -> tuple[list[list[str]], int]:
    """Metadata tree rows (group, tag, value, severity, rule codes, note) and the total before truncation. Flagged rows
    are kept first when the tree exceeds MAX_TREE_ROWS."""
    tags = analysis.read.tags
    marks: dict[str, list[Hit]] = {}
    rows: list[list[str]] = []
    unattached = []
    for hit in analysis.hits:
        present = [k for k in hit.tags if k in tags]
        for key in present:
            marks.setdefault(key, []).append(hit)
        if not present:
            unattached.append(hit)
    for key in sorted(k for k in tags if ":" in k):
        group, tag = key.split(":", 1)
        found = marks.get(key, [])
        severity = max((h.severity for h in found), key=SEVERITY_RANK.__getitem__, default="info")
        rows.append([group, tag, _text(tags[key]), severity, ", ".join(dict.fromkeys(h.code for h in found)),
                     "; ".join(dict.fromkeys(h.note for h in found))])
    rows += [["(rule)", RULE_NAMES[h.rule], _text(h.observed), h.severity, h.code, h.note] for h in unattached]
    total = len(rows)
    if total > MAX_TREE_ROWS:
        flagged = [r for r in rows if r[3] != "info"]
        rows = flagged + [r for r in rows if r[3] == "info"][:max(MAX_TREE_ROWS - len(flagged), 0)]
    return rows, total


def timestamp_rows(analysis: MetadataAnalysis) -> list[list[str]]:
    capture = _capture(analysis.read.tags)
    rows = []
    for key, value, stamp in analysis.timestamps:
        offset = "not stated" if stamp.offset is None else ("+" if stamp.offset >= timedelta(0) else "−") + \
            f"{int(abs(stamp.offset).total_seconds() // 3600):02d}:{int(abs(stamp.offset).total_seconds() % 3600 // 60):02d}"
        if capture is None:
            relative = "no DateTimeOriginal"
        else:
            delta = _diff(stamp, capture)
            relative = "same time" if abs(delta) <= DATE_TOLERANCE else f"{human_delta(delta)} {'after' if delta > timedelta(0) else 'before'}"
        rows.append([_text(value), key, offset, relative])
    return rows


def history_rows(analysis: MetadataAnalysis) -> list[list[str]]:
    return [[str(i), e.get("action", "not stated"), e.get("when", "not stated"), e.get("softwareAgent", "not stated"),
             e.get("changed", "")] for i, e in enumerate(analysis.history, 1)]
