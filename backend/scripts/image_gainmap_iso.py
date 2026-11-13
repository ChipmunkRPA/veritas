# SPDX-License-Identifier: Apache-2.0
"""Bounded JPEG gain-map intake for the cross-camera image study."""

from __future__ import annotations

import io
import math
import struct
from dataclasses import dataclass

from PIL import Image, ImageFile, JpegImagePlugin

from scripts import image_gainmap_localization_transfer as xmp_reader

ISO = b"urn:iso:std:iso:ts:21496:-1\0"
MPF = b"MPF\0"
EXIF = b"Exif\0\0"
XMP = b"http://ns.adobe.com/xap/1.0/\0"
EXT_XMP = b"http://ns.adobe.com/xmp/extension/\0"
TIFF_XMP = "{http://ns.adobe.com/tiff/1.0/}Orientation"
HDR = "{" + xmp_reader.HDR + "}"
MAX_FILE_BYTES = 32 * 1024 * 1024
MAX_GAIN_BYTES = 2_000_000
MAX_FOOTER_BYTES = 4096
SEF_RECORDS = {
    0x0A01: b"Image_UTC_Data", 0x0AA1: b"MCC_Data", 0x0CC1: b"Color_Display_P3",
    0x0CD2: b"Photo_HDR_Info", 0x0C61: b"Camera_Capture_Mode_Info",
}


class InvalidGainMap(ValueError):
    """The JPEG cannot be assigned one unambiguous supported gain map."""


@dataclass(frozen=True)
class Jpeg:
    end: int
    size: tuple[int, int]
    components: int
    apps: tuple[tuple[int, int, int], ...]


@dataclass(frozen=True)
class IsoPair:
    primary_end: int
    gain_start: int
    gain_end: int
    footer: str | None
    primary_size: tuple[int, int]
    gain_size: tuple[int, int]
    primary_metadata: bytes
    gain_metadata: bytes


def _require(condition, message):
    if not condition:
        raise InvalidGainMap(message)


def _scan(data, start, limit):
    _require(data[start:start + 2] == b"\xff\xd8", "Missing JPEG SOI")
    at = start + 2
    frame = None
    scans = markers = 0
    entropy = False
    apps = []
    while at < limit:
        if entropy:
            at = data.find(b"\xff", at, limit)
            _require(at >= 0, "Unterminated JPEG scan")
        _require(data[at] == 0xFF, "Expected JPEG marker")
        mark_start = at
        while at < limit and data[at] == 0xFF:
            at += 1
        _require(at < limit, "Truncated JPEG marker")
        marker = data[at]
        at += 1
        if entropy and (marker == 0 or 0xD0 <= marker <= 0xD7):
            _require(marker != 0 or at - mark_start == 2, "Ambiguous stuffed scan marker")
            continue
        entropy = False
        markers += 1
        _require(markers <= 512, "Too many JPEG markers")
        if marker == 0xD9:
            _require(frame is not None and scans > 0, "JPEG EOI without image scan")
            return Jpeg(at, frame[:2], frame[2], tuple(apps))
        _require(marker in (0xC0, 0xC1, 0xC2, 0xC4, 0xDA, 0xDB, 0xDD, 0xFE) or 0xE0 <= marker <= 0xEF,
                 "Unsupported JPEG marker")
        _require(at + 2 <= limit, "Truncated JPEG segment length")
        length = int.from_bytes(data[at:at + 2], "big")
        end = at + length
        _require(length >= 2 and end <= limit, "Invalid JPEG segment length")
        body = at + 2
        if marker in (0xC0, 0xC1, 0xC2):
            _require(frame is None and length >= 11 and data[body] == 8, "Invalid JPEG frame")
            height, width, count = struct.unpack_from(">HHB", data, body + 1)
            _require(count in (1, 3) and length == 8 + 3 * count and width > 0 and height > 0,
                     "Invalid JPEG frame dimensions or components")
            frame = width, height, count
        elif marker == 0xDA:
            _require(frame is not None and length >= 8, "JPEG scan without frame")
            count = data[body]
            _require(1 <= count <= frame[2] and length == 6 + 2 * count and scans < 32, "Invalid JPEG scan header")
            entropy = True
            scans += 1
        elif 0xE0 <= marker <= 0xEF:
            _require(scans == 0, "JPEG metadata after first scan")
            apps.append((marker, body, end))
        at = end
    raise InvalidGainMap("Missing JPEG EOI")


def _markers(data, jpeg, marker, prefix):
    return [(start + len(prefix), end) for kind, start, end in jpeg.apps
            if kind == marker and data.startswith(prefix, start, end)]


def _metadata(data, jpeg, marker, prefix):
    locations = _markers(data, jpeg, marker, prefix)
    _require(len(locations) <= 1, "Duplicate metadata marker")
    return data[slice(*locations[0])] if locations else None


def _iso_marker(data, jpeg):
    count = len(_markers(data, jpeg, 0xE2, ISO[:-1]))
    payload = _metadata(data, jpeg, 0xE2, ISO)
    _require(count == int(payload is not None), "Malformed ISO namespace declaration")
    return payload


def _exif_orientation(data, jpeg):
    exif = _metadata(data, jpeg, 0xE1, EXIF)
    if exif is None:
        return
    _require(len(exif) >= 14 and exif[:2] in (b"II", b"MM"), "Malformed Exif TIFF header")
    endian = "<" if exif[:2] == b"II" else ">"
    magic, offset = struct.unpack_from(endian + "HI", exif, 2)
    _require(magic == 42 and 8 <= offset <= len(exif) - 6, "Exif IFD0 outside APP1")
    count, = struct.unpack_from(endian + "H", exif, offset)
    _require(count <= 1024 and offset + 6 + 12 * count <= len(exif), "Exif IFD0 entries outside APP1")
    seen = False
    for at in range(offset + 2, offset + 2 + 12 * count, 12):
        tag, kind, size, value = struct.unpack_from(endian + "HHIH", exif, at)
        if tag == 274:
            _require(not seen and (kind, size, value) == (3, 1, 1), "Unsupported or ambiguous Exif orientation")
            seen = True


def _mpf(data, primary):
    indices = _markers(data, primary, 0xE2, MPF)
    _require(len(indices) == 1, "Expected exactly one MPF index")
    tiff_start, end = indices[0]
    tiff = memoryview(data)[tiff_start:end]
    _require(len(tiff) >= 82 and tiff[:2] in (b"II", b"MM"), "Invalid MPF TIFF header")
    endian = "<" if tiff[:2] == b"II" else ">"
    magic, index = struct.unpack_from(endian + "HI", tiff, 2)
    _require(magic == 42 and 8 <= index <= len(tiff) - 42, "MPF IFD outside APP2")
    count, = struct.unpack_from(endian + "H", tiff, index)
    _require(count == 3, "Expected three unique MPF index tags")
    directory_end = index + 2 + count * 12 + 4
    _require(struct.unpack_from(endian + "I", tiff, directory_end - 4)[0] == 0, "Unexpected linked MPF IFD")
    tags = {}
    for at in range(index + 2, directory_end - 4, 12):
        tag, kind, count = struct.unpack_from(endian + "HHI", tiff, at)
        _require(tag not in tags, "Duplicate MPF index tag")
        tags[tag] = (kind, count, bytes(tiff[at + 8:at + 12]))
    _require(tags.get(0xB000) == (7, 4, b"0100") and tags.get(0xB001) == (4, 1, struct.pack(endian + "I", 2))
             and tags.get(0xB002, ())[:2] == (7, 32), "Invalid MPF version or image count")
    table, = struct.unpack(endian + "I", tags[0xB002][2])
    _require(directory_end <= table and table + 32 == len(tiff), "Invalid MPF entry table bounds")
    kind, size, offset, depend1, depend2 = struct.unpack_from(endian + "IIIHH", tiff, table)
    _require(kind in (0x00030000, 0x20030000) and size == primary.end and offset == depend1 == depend2 == 0,
             "Invalid primary MP image")
    kind, size, offset, depend1, depend2 = struct.unpack_from(endian + "IIIHH", tiff, table + 16)
    start = tiff_start + offset
    _require(kind == depend1 == depend2 == 0 and start == primary.end and 4 <= size <= MAX_GAIN_BYTES
             and start + size <= len(data), "Invalid secondary MP image")
    return start, start + size


def _gain_mpf(data, gain):
    # Some cameras write a version-only attribute IFD in the secondary JPEG.
    marker = _metadata(data, gain, 0xE2, MPF)
    if marker is not None:
        _require(marker in (bytes.fromhex("49492a0008000000010000b00700040000003031303000000000"),
                            bytes.fromhex("4d4d002a000000080001b0000007000000043031303000000000")),
                 "Unexpected secondary MPF index or attributes")


def _iso_gain(payload):
    _require(payload is not None and len(payload) == 61, "Missing or unsupported ISO gain metadata length")
    version, writer, flags, *fields = struct.unpack(">HHB4IiIiIIIiIiI", payload)
    _require((version, writer, flags) == (0, 0, 0x40) and all(fields[1::2]),
             "Unsupported ISO version, gain mode, or rational denominator")
    base, alt, low, high, gamma, base_offset, alt_offset = (
        fields[i] / fields[i + 1] for i in range(0, len(fields), 2)
    )
    _require(0 <= base < alt <= 16 and -16 <= low < high <= 16 and 0 < gamma <= 16
             and 0 <= base_offset <= 1 and 0 <= alt_offset <= 1, "Gain metadata outside supported forward range")
    return base, alt, low, high, gamma, base_offset, alt_offset


def _xml(data, jpeg):
    _require(not _markers(data, jpeg, 0xE1, EXT_XMP), "Extended XMP is outside the supported pair")
    payload = _metadata(data, jpeg, 0xE1, XMP)
    if payload is None:
        return None
    xml = xmp_reader._xml(payload)
    _require(xml is not None, "Malformed XMP")
    _require(all(node.get(TIFF_XMP, "1").strip() == "1"
                 and (node.tag != TIFF_XMP or (node.text or "").strip() == "1") for node in xml.iter()),
             "Unsupported or ambiguous XMP orientation")
    return xml


def _hdr_fields(xml):
    fields = {}
    if xml is None:
        return fields
    for node in xml.iter():
        attributes = [(key, value) for key, value in node.attrib.items() if key.startswith(HDR)]
        if node.tag.startswith(HDR):
            attributes.append((node.tag, (node.text or "").strip()))
        for key, value in attributes:
            field = key.removeprefix(HDR)
            _require(field not in fields, "Conflicting HDR declarations")
            fields[field] = value
    return fields


def _xmp_agrees(data, primary, gain, iso_values):
    base_xml, gain_xml = (_xml(data, image) for image in (primary, gain))
    base_fields, gain_fields = (_hdr_fields(xml) for xml in (base_xml, gain_xml))
    directories = [] if base_xml is None else base_xml.findall(f".//{{{xmp_reader.CONTAINER}}}Directory")
    if not (directories or base_fields or gain_fields):
        return
    _require(len(directories) == 1 and base_fields == {"Version": "1.0"}, "Ambiguous XMP gain directory")
    item = "{" + xmp_reader.ITEM + "}"
    nodes = directories[0].findall(f".//{{{xmp_reader.CONTAINER}}}Item")
    _require([node.get(item + "Semantic") for node in nodes] == ["Primary", "GainMap"]
             and all(node.get(item + "Mime") == "image/jpeg" and node.get(item + "Padding", "0") == "0" for node in nodes)
             and nodes[1].get(item + "Length") == str(gain.end - primary.end), "XMP and MPF gain ranges disagree")
    names = ("HDRCapacityMin", "HDRCapacityMax", "GainMapMin", "GainMapMax", "Gamma", "OffsetSDR", "OffsetHDR")
    _require(gain_fields.get("Version") == "1.0" and gain_fields.get("BaseRenditionIsHDR") == "False"
             and set(gain_fields) == {"Version", "BaseRenditionIsHDR", *names}, "Unsupported dual gain metadata")
    _require(all(math.isclose(float(gain_fields[name]), val, rel_tol=0, abs_tol=2e-6)
                 for name, val in zip(names, iso_values, strict=True)), "XMP and ISO gain values disagree")


def _footer(data, end):
    if end == len(data):
        return None
    _require(20 <= len(data) - end <= MAX_FOOTER_BYTES and data[-4:] == b"SEFT", "Unrecognized bytes after gain map")
    suffix = data[end:]
    length, = struct.unpack_from("<I", suffix, len(suffix) - 8)
    directory = len(suffix) - length - 8
    _require(0 <= directory <= len(suffix) - 20 and suffix[directory:directory + 4] == b"SEFH", "Malformed SEF directory")
    version, count = struct.unpack_from("<II", suffix, directory + 4)
    _require(version == 107 and 1 <= count <= len(SEF_RECORDS) and length == 12 + 12 * count, "Unsupported SEF layout")
    records = []
    kinds = set()
    for at in range(directory + 12, len(suffix) - 8, 12):
        flags, kind, backward, size = struct.unpack_from("<HHII", suffix, at)
        _require(flags == 0 and kind in SEF_RECORDS and kind not in kinds and 8 <= size <= backward <= directory,
                 "Invalid SEF record directory")
        kinds.add(kind)
        start = directory - backward
        header, tag, name_size = struct.unpack_from("<HHI", suffix, start)
        name = SEF_RECORDS[kind]
        _require(header == 0 and tag == kind and name_size == len(name) <= size - 8
                 and suffix[start + 8:start + 8 + name_size] == name
                 and b"\xff\xd8" not in suffix[start + 8 + name_size:start + size], "Ambiguous SEF record payload")
        records.append((start, start + size))
    at = 0
    for start, end in sorted(records):
        _require(start == at, "Gap or overlap in SEF data")
        at = end
    _require(at == directory, "Unaccounted SEF bytes")
    return "samsung-sef"


def _inspect(data, primary):
    base_metadata = _iso_marker(data, primary)
    _require(base_metadata == b"\0" * 4 and primary.components == 3, "Invalid primary ISO declaration")
    gain_start, gain_end = _mpf(data, primary)
    gain = _scan(data, gain_start, gain_end)
    _require(gain.end == gain_end and gain.components == 1, "Ambiguous secondary JPEG or gain map color")
    _gain_mpf(data, gain)
    width, height = primary.size
    gain_width, gain_height = gain.size
    _require(384 <= min(primary.size) <= max(primary.size) <= 8192 and width * height <= 51_000_000
             and 96 <= min(gain.size) <= max(gain.size) <= 4096 and gain_width * gain_height <= 8_000_000
             and gain_width <= width and gain_height <= height
             and abs(width * gain_height / (height * gain_width) - 1) <= .01, "Unsupported gain-map geometry")
    _exif_orientation(data, primary)
    _exif_orientation(data, gain)
    gain_metadata = _iso_marker(data, gain)
    iso_values = _iso_gain(gain_metadata)
    _xmp_agrees(data, primary, gain, iso_values)
    return IsoPair(primary.end, gain_start, gain_end, _footer(data, gain_end), primary.size, gain.size,
                   base_metadata, gain_metadata)


def inspect_iso(data: bytes) -> IsoPair | None:
    """Bind the exact ISO-declared pair and optional bounded SEF suffix before decode."""
    if not 100 <= len(data) <= MAX_FILE_BYTES:
        return None
    try:
        return _inspect(data, _scan(data, 0, len(data)))
    except (ValueError, struct.error):
        return None


def _decode(payload, size, mode):
    # The MPF has already been resolved; prevent Pillow's MPO plugin from reinterpreting the slice.
    with JpegImagePlugin.JpegImageFile(io.BytesIO(payload)) as image:
        _require(image.size == size and image.mode == mode and image.getexif().get(274, 1) == 1,
                 "Decoded JPEG geometry, color or orientation differs")
        image.load()
        _require(image.getexif().get(274, 1) == 1, "Effective orientation differs after decode")
        return image.copy()


def split(data: bytes) -> tuple[Image.Image, Image.Image, bytes] | None:
    """Decode the new ISO pair, or use the established strict XMP-only path."""
    if not 100 <= len(data) <= MAX_FILE_BYTES or ImageFile.LOAD_TRUNCATED_IMAGES:
        return None
    try:
        primary = _scan(data, 0, len(data))
        if _iso_marker(data, primary) is None:
            old = xmp_reader.split(data)
            if old is None:
                return None
            aux = _scan(old[2], 0, len(old[2]))
            _require(aux.end == len(old[2]) and primary.end == len(data) - len(old[2])
                     and _iso_marker(old[2], aux) is None, "Ambiguous legacy gain pair")
            return old
        pair = _inspect(data, primary)
        gain = data[pair.gain_start:pair.gain_end]
        return _decode(data[:pair.primary_end], pair.primary_size, "RGB"), _decode(gain, pair.gain_size, "L"), gain
    except (OSError, ValueError, SyntaxError, struct.error, Image.DecompressionBombError):
        return None
