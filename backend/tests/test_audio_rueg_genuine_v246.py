"""Exercise ZIP range, group and PCM invariants with authored local fixtures."""

from __future__ import annotations

import hashlib
import io
import shutil
import wave
import zipfile
from dataclasses import replace
from email.message import Message
from types import SimpleNamespace

import pytest

from scripts import audio_rueg_genuine_v246 as study


@pytest.fixture
def zip_fixture():
    buffer = io.BytesIO()
    payload = b"an authored fixture representing member contents" * 64
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("fixture/monolingual/TUmo9001MT_fsT.wav", payload)
        archive.writestr("fixture/monolingual/TUmo9001MT_isT.wav", payload)
        archive.writestr("fixture/monolingual/TUmo9002FT_fsT.wav", payload)
        archive.writestr("fixture/heritage/DEbi9001MT_fsT.wav", payload)
    data = buffer.getvalue()
    sig, _, _, entries, _, size, start, comment = study.END.unpack(data[-study.END.size :])
    assert sig == b"PK\x05\x06" and comment == 0
    return data, entries, size, start, payload


def test_directory_selects_formal_one_per_monolingual_group(zip_fixture):
    data, count, size, start, _ = zip_fixture
    body = data[start : start + size]
    selected = study.parse_directory(body, count, start)
    assert [member.participant for member in selected] == ["TUmo9001MT", "TUmo9002FT"]
    assert all(member.name.endswith("_fsT.wav") for member in selected)
    with pytest.raises(ValueError, match="Truncated|differs"):
        study.parse_directory(body[:-1], count, start)


def test_exact_publisher_identity_blocks_directory_substitution(zip_fixture, monkeypatch):
    data, count, size, start, _ = zip_fixture
    body = data[start : start + size]
    monkeypatch.setattr(study, "DIRECTORY_START", start)
    monkeypatch.setattr(study, "DIRECTORY_SIZE", size)
    monkeypatch.setattr(study, "DIRECTORY_SHA256", hashlib.sha256(body).hexdigest())
    monkeypatch.setattr(study, "ENTRIES", count)
    monkeypatch.setattr(study, "FORMAL_GROUPS", 2)
    monkeypatch.setattr(study, "SLOTS", 1)
    members = study.parse_directory(body, count, start)
    planned = min(
        members, key=lambda m: hashlib.sha256(f"{study.RANK_PREFIX}{m.participant}".encode()).digest()
    )
    monkeypatch.setattr(study, "EXPECTED_COMPRESSED", planned.compressed)
    monkeypatch.setattr(study, "EXPECTED_EXPANDED", planned.expanded)
    footer = data[-study.END.size :]
    assert study.select(footer, body) == [planned]
    with pytest.raises(ValueError, match="Pinned publisher directory differs"):
        study.select(footer, body[:-1] + bytes([body[-1] ^ 1]))


def test_exact_member_byte_ranges_and_crc(zip_fixture, monkeypatch):
    data, count, size, start, payload = zip_fixture
    monkeypatch.setattr(study, "DIRECTORY_START", start)
    member = study.parse_directory(data[start : start + size], count, start)[0]
    requests = []

    def get(offset, length):
        requests.append((offset, length))
        return data[offset : offset + length]

    transport = SimpleNamespace(get=get)
    assert study.extract_member(member, transport) == payload
    assert requests == [
        (member.local_offset, study.LOCAL.size),
        (member.local_offset + study.LOCAL.size, len(member.name) + member.compressed),
    ]
    with pytest.raises(study.Rejected, match="local_header_mismatch"):
        study.extract_member(replace(member, crc32=member.crc32 ^ 1), transport)
    actual = bytearray(data)
    actual[member.local_offset + study.LOCAL.size + len(member.name)] ^= 1
    damaged = SimpleNamespace(get=lambda offset, length: bytes(actual[offset : offset + length]))
    with pytest.raises(study.Rejected, match="bounded_deflate_failed|member_length_or_crc_mismatch"):
        study.extract_member(member, damaged)


@pytest.mark.parametrize("status,total,read_body", [(206, 100, True), (200, 100, False), (206, 99, False)])
def test_range_identity_is_checked_before_any_body_read(monkeypatch, status, total, read_body):
    monkeypatch.setattr(study, "ARCHIVE_BYTES", 100)

    class Response(io.BytesIO):
        headers = Message()

        def geturl(self):
            return study.URL

        def read1(self, length):
            assert read_body
            return super().read1(length)

    response = Response(b"1234")
    response.status = status
    response.headers["Content-Range"] = f"bytes 5-8/{total}"
    response.headers["Content-Length"] = "4"

    def open_range(request, timeout):
        assert request.full_url == study.URL and request.get_header("Range") == "bytes=5-8"
        assert timeout == study.NETWORK_TIMEOUT
        return response

    transport = study.RangeReader(1, 4, opener=SimpleNamespace(open=open_range))
    if read_body:
        assert transport.get(5, 4) == b"1234"
    else:
        with pytest.raises(study.Rejected, match="unexpected_partial_response"):
            transport.get(5, 4)
    with pytest.raises(ValueError, match="bound exceeded"):
        transport.get(5, 4)


@pytest.mark.skipif(
    not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="local decoder tools required"
)
def test_production_clip_replays_native_samples_and_keeps_clock(tmp_path):
    source = tmp_path / "test.wav"
    with wave.open(str(source), "wb") as wav:
        wav.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        wav.writeframes(b"\x07\x02\xfa\xfd" * 24000)
    source_sha = study.digest(source)
    data, result, fingerprints = study.prepare_clip(source, source_sha, set())
    assert result["selection"] == "natural_short_full"
    assert result["samples"] == 48000
    assert result["source_range"] == study.crop.sample_range(0, 48000)
    assert study.crop.pcm_sha(study.read_wav(data)) == result["pcm_sha256"]
    with pytest.raises(ValueError, match="Known byte or PCM overlap"):
        study.prepare_clip(source, source_sha, fingerprints)
