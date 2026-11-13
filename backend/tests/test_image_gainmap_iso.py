from __future__ import annotations

import io
import struct

import pytest
from PIL import Image, ImageFile

from scripts import image_gainmap_iso as reader
from scripts import image_gainmap_localization_transfer as legacy

GAIN = struct.pack(">HHB4IiIiIIIiIiI", 0, 0, 0x40, 0, 1, 2, 1, 0, 1, 2, 1, 1, 1, 0, 1, 0, 1)
HDR = f'<r xmlns:h="{legacy.HDR}" h:Version="1.0" h:BaseRenditionIsHDR="False" h:HDRCapacityMin="0" '
HDR += 'h:HDRCapacityMax="2" h:GainMapMin="0" h:GainMapMax="2" h:Gamma="1" h:OffsetSDR="0" h:OffsetHDR="0"/>'


def app(number, payload):
    return bytes((0xFF, 0xE0 + number)) + struct.pack(">H", len(payload) + 2) + payload


def jpeg(size, mode, *, progressive=False):
    buffer = io.BytesIO()
    image = Image.effect_noise(size, 30).convert(mode)
    image.save(buffer, "JPEG", progressive=progressive)
    return buffer.getvalue()


def mpf(byte_order, base_size, gain_size, gain_offset):
    little = byte_order == "<"
    directory = b"II" if little else b"MM"
    directory += struct.pack(byte_order + "HIH", 42, 8, 3)
    directory += struct.pack(byte_order + "HHI4s", 0xB000, 7, 4, b"0100")
    directory += struct.pack(byte_order + "HHII", 0xB001, 4, 1, 2)
    directory += struct.pack(byte_order + "HHIII", 0xB002, 7, 32, 50, 0)
    directory += struct.pack(byte_order + "IIIHH", 0x00030000, base_size, 0, 0, 0)
    directory += struct.pack(byte_order + "IIIHH", 0, gain_size, gain_offset, 0, 0)
    return app(2, reader.MPF + directory)


def footer():
    name = b"Photo_HDR_Info"
    record = struct.pack("<HHI", 0, 0x0CD2, len(name)) + name + b"\0"
    directory = b"SEFH" + struct.pack("<IIHHII", 107, 1, 0, 0x0CD2, len(record), len(record))
    return record + directory + struct.pack("<I4s", len(directory), b"SEFT")


def fixture(*, endian=">", dual=False, suffix=b"", gain_size=(256, 192), gain_mode="L", base_extra=b"", gain_extra=b"",
            progressive=False, gain_metadata=GAIN):
    base = jpeg((512, 384), "RGB", progressive=progressive)
    gain = jpeg(gain_size, gain_mode, progressive=progressive)
    gain_xmp = app(1, reader.XMP + HDR.encode()) if dual else b""
    gain = gain[:2] + gain_xmp + gain_extra + app(2, reader.ISO + gain_metadata) + gain[2:]
    base_xmp = app(1, reader.XMP + legacy.directory(len(gain))) if dual else b""
    header = base[:2] + base_xmp + base_extra + app(2, reader.ISO + b"\0" * 4)
    tiff_start = len(header) + 8
    primary_size = len(header) + len(mpf(endian, 0, 0, 0)) + len(base) - 2
    index = mpf(endian, primary_size, len(gain), primary_size - tiff_start)
    return header + index + base[2:] + gain + suffix


def patch(data, at, value):
    return data[:at] + value + data[at + len(value):]


@pytest.mark.parametrize(("endian", "dual", "progressive"), [("<", True, False), (">", False, True)])
def test_exact_mpf_pair_decodes(endian, dual, progressive):
    suffix = footer() if dual else b""
    data = fixture(endian=endian, dual=dual, suffix=suffix, progressive=progressive)
    pair = reader.inspect_iso(data)
    assert pair is not None
    assert pair.primary_end == pair.gain_start < pair.gain_end == len(data) - len(suffix)
    assert pair.footer == ("samsung-sef" if suffix else None)
    assert pair.primary_size == (512, 384) and pair.gain_size == (256, 192)
    assert pair.primary_metadata == b"\0" * 4 and pair.gain_metadata == GAIN
    base, gain, encoded_gain = reader.split(data)
    assert base.mode == "RGB" and base.size == pair.primary_size
    assert gain.mode == "L" and gain.size == pair.gain_size
    assert encoded_gain == data[pair.gain_start:pair.gain_end]


@pytest.mark.parametrize("test", ["marker_length", "tiff_magic", "ifd_offset", "ifd_tags", "mpf_count", "entry_pointer",
                                  "primary_role", "primary_length", "secondary_role", "secondary_offset", "secondary_size",
                                  "gain_eoi", "fake_eoi", "gain_version", "gain_denominator", "namespace", "tail", "sef_pointer",
                                  "sef_record", "sef_name", "xmp_conflict", "xmp_length"])
def test_bad_structure_is_unassessed(test):
    data = fixture(endian="<", dual=True, suffix=footer())
    pair = reader.inspect_iso(data)
    tiff = data.index(reader.MPF) + 4
    iso = data.index(reader.ISO, pair.gain_start) + len(reader.ISO)
    sef = data.index(b"SEFH", pair.gain_end)
    cases = {
        "marker_length": (tiff - 6, b"\x00\x01"), "tiff_magic": (tiff + 2, b"\0\0"),
        "ifd_offset": (tiff + 4, struct.pack("<I", 0xFFFF)), "ifd_tags": (tiff + 8, b"\x04\0"),
        "mpf_count": (tiff + 30, b"\x03\0\0\0"), "entry_pointer": (tiff + 42, struct.pack("<I", 8)),
        "primary_role": (tiff + 50, b"\0" * 4), "primary_length": (tiff + 54, struct.pack("<I", pair.gain_start - 2)),
        "secondary_role": (tiff + 66, b"\x01\0\0\0"),
        "secondary_offset": (tiff + 74, struct.pack("<I", pair.gain_start - tiff + 1)),
        "secondary_size": (tiff + 70, struct.pack("<I", pair.gain_end - pair.gain_start + 2)),
        "gain_eoi": (pair.gain_end - 2, b"\0\0"), "fake_eoi": (pair.gain_end - 12, b"\xff\xd9"),
        "gain_version": (iso, b"\0\1"), "gain_denominator": (iso + 9, b"\0" * 4),
        "namespace": (iso - 1, b"\x01"), "tail": (len(data) - 4, b"NOPE"),
        "sef_pointer": (len(data) - 8, b"\0" * 4), "sef_record": (sef + 16, b"\xff" * 4),
        "sef_name": (pair.gain_end + 8, b"N"), "xmp_conflict": (data.index(b'GainMapMax="2"') + 12, b"3"),
        "xmp_length": (data.index(b'Item:Length="') + 13, b"0"),
    }
    altered = patch(data, *cases[test])
    assert altered != data
    assert reader.inspect_iso(altered) is None
    assert reader.split(altered) is None


@pytest.mark.parametrize("options", [
    {"gain_size": (240, 200)}, {"gain_size": (48, 36)}, {"gain_size": (960, 720)}, {"gain_mode": "RGB"},
    {"base_extra": app(2, reader.ISO + b"\0" * 4)}, {"gain_extra": app(2, reader.ISO + GAIN)},
    {"gain_extra": app(2, reader.MPF + b"Nested")},
    {"base_extra": app(1, reader.XMP + b'<r xmlns:t="http://ns.adobe.com/tiff/1.0/" t:Orientation="6"/>')},
    {"gain_extra": app(1, reader.EXIF + b"II" + struct.pack("<HIHHHIHI", 42, 8, 1, 274, 3, 1, 6, 0))},
    {"gain_metadata": GAIN[:4] + b"\xc0" + GAIN[5:]},
])
def test_unsupported_variant_is_unassessed(options):
    assert reader.split(fixture(**options)) is None


def test_truncated_decode_and_unrecognized_trailer_are_unassessed(monkeypatch):
    data = fixture()
    pair = reader.inspect_iso(data)
    for end in (2, 50, pair.primary_end - 3, pair.gain_start + 12, pair.gain_end - 1):
        assert reader.split(data[:end]) is None
    assert reader.split(data + b"unexpected") is None
    assert reader.split(data + data[pair.gain_start:]) is None
    monkeypatch.setattr(ImageFile, "LOAD_TRUNCATED_IMAGES", True)
    assert reader.split(data) is None


@pytest.mark.parametrize("target", ["primary", "gain"])
def test_structurally_bound_pair_requires_valid_jpeg_decode(target):
    data = fixture()
    pair = reader.inspect_iso(data)
    start = 0 if target == "primary" else pair.gain_start
    table = data.index(b"\xff\xc4", start)
    altered = patch(data, table + 4, b"\xff")
    assert reader.inspect_iso(altered) == pair
    assert reader.split(altered) is None


def test_preserves_established_xmp_path_and_rejects_legacy_mpf_only():
    base = Image.effect_noise((512, 384), 30).convert("RGB")
    gain = jpeg((256, 192), "L")
    gain = gain[:2] + app(1, reader.XMP + HDR.encode()) + gain[2:]
    data = legacy.encode(base, gain, 90)
    prior = legacy.split(data)
    assert prior is not None
    new = reader.split(data)
    assert reader.inspect_iso(data) is None and new[2] == prior[2]
    assert [image.tobytes() for image in new[:2]] == [image.tobytes() for image in prior[:2]]
    assert reader.split(jpeg((512, 384), "RGB") + gain) is None
