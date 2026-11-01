"""Readability checks run with actual decoders, before a payment can be created."""
from __future__ import annotations

import io
import shutil
import stat
import struct
import subprocess
import time
import wave
import zipfile
import zlib

import pytest
from PIL import Image

from app.billing import preflight


def png() -> bytes:
    data = io.BytesIO()
    Image.new("RGB", (16, 16), (20, 40, 60)).save(data, "PNG")
    return data.getvalue()


def wav(*, samples=800) -> bytes:
    data = io.BytesIO()
    with wave.open(data, "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(8000)
        output.writeframes(b"\x01\x00" * samples)
    return data.getvalue()


def verify(data: bytes, filename: str, kind: str) -> None:
    source = io.BytesIO(data)
    source.seek(min(7, len(data)))
    position = source.tell()
    try:
        preflight.validate_upload(source, filename, kind)
    finally:
        assert source.tell() == position


def test_valid_image_and_truncated_magic_only_image():
    image = png()
    verify(image, "private-file.png", "image")
    with pytest.raises(preflight.PreflightError, match="could not be fully decoded"):
        verify(image[:40], "private-file.png", "image")


def test_image_pixel_limit_is_checked_before_decoding():
    image = bytearray(png())
    image[16:24] = struct.pack(">II", 9000, 9000)
    image[29:33] = struct.pack(">I", zlib.crc32(image[12:29]))
    with pytest.raises(preflight.PreflightError, match="80 MP"):
        verify(bytes(image), "huge.png", "image")


@pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="Local media binaries required")
def test_audio_needs_decodable_samples_and_positive_duration():
    verify(wav(), "voice.wav", "audio")
    with pytest.raises(preflight.PreflightError):
        verify(wav(samples=0), "empty-voice.wav", "audio")
    with pytest.raises(preflight.PreflightError):
        verify(b"RIFF\x24\x00\x00\x00WAVE", "broken.wav", "audio")


@pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="Local media binaries required")
def test_video_needs_an_actual_decodable_frame(tmp_path):
    path = tmp_path / "valid.mp4"
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i", "color=c=blue:s=32x32:r=2",
                    "-t", "1", "-c:v", "mpeg4", "-threads", "1", str(path)], check=True, timeout=10,
                   stdin=subprocess.DEVNULL, capture_output=True)
    verify(path.read_bytes(), "clip.mp4", "video")
    with pytest.raises(preflight.PreflightError):
        verify(path.read_bytes()[:32], "broken.mp4", "video")


def test_readable_pdf_and_password_protected_or_broken_pdf():
    import pymupdf

    with pymupdf.open() as document:
        page = document.new_page()
        page.insert_text((50, 50), "A sample document for file preparation.")
        good = document.tobytes()
        locked = document.tobytes(encryption=pymupdf.PDF_ENCRYPT_AES_256, owner_pw="fixture-owner", user_pw="fixture-reader")
    verify(good, "article.pdf", "document")
    with pytest.raises(preflight.PreflightError, match="password-protected"):
        verify(locked, "locked.pdf", "document")
    with pytest.raises(preflight.PreflightError):
        verify(b"%PDF-1.7\ntruncated document", "broken.pdf", "document")


def test_docx_requires_a_readable_document_package():
    from docx import Document

    document = Document()
    document.add_paragraph("A readable sample document.")
    good = io.BytesIO()
    document.save(good)
    verify(good.getvalue(), "article.docx", "document")
    unrelated = io.BytesIO()
    with zipfile.ZipFile(unrelated, "w") as archive:
        archive.writestr("unrelated.txt", "This zip is not a Word document.")
    with pytest.raises(preflight.PreflightError):
        verify(unrelated.getvalue(), "fake.docx", "document")
    with pytest.raises(preflight.PreflightError):
        verify(good.getvalue()[:100], "truncated.docx", "document")


def test_text_checks_encoding_and_readability_without_running_models():
    verify("A UTF-8 document — with punctuation.".encode(), "article.txt", "document")
    verify("A UTF-16 document with punctuation.".encode("utf-16"), "article.txt", "document")
    for data in (b" \r\n\t", b"\x00\xff\xfe\x80", b""):
        with pytest.raises(preflight.PreflightError):
            verify(data, "article.txt", "document")


def test_email_uses_mime_charset_and_requires_a_readable_body():
    email = (b"From: author@example.com\r\nTo: reader@example.com\r\nSubject: Example\r\n"
             b"Content-Type: text/plain; charset=iso-8859-1\r\nContent-Transfer-Encoding: 8bit\r\n\r\n"
             b"An ordinary caf\xe9 update with a clearly readable email body.\r\n")
    verify(email, "message.eml", "document")
    for empty in (b"From: author@example.com\r\nSubject: Empty message\r\n\r\n",
                  b"Content-Type: multipart/mixed; boundary=fixture\r\n\r\n--fixture\r\n"
                  b"Content-Type: application/octet-stream\r\nContent-Disposition: attachment; filename=sample.bin\r\n"
                  b"Content-Transfer-Encoding: base64\r\n\r\nYWJj\r\n--fixture--\r\n"):
        with pytest.raises(preflight.PreflightError, match="no readable"):
            verify(empty, "empty.eml", "document")


def test_temporary_source_is_private_and_removed(tmp_path, monkeypatch):
    monkeypatch.setattr(preflight.tempfile, "tempdir", str(tmp_path))
    worker = preflight._worker

    def inspect(path, kind, extension):
        assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
        worker(path, kind, extension)

    monkeypatch.setattr(preflight, "_worker", inspect)
    verify(png(), "private.png", "image")
    assert list(tmp_path.iterdir()) == []


def test_timeout_kills_child_and_removes_temporary_upload(tmp_path, monkeypatch):
    monkeypatch.setattr(preflight, "WALL_SECONDS", 0.00001)
    monkeypatch.setattr(preflight.tempfile, "tempdir", str(tmp_path))
    started = time.monotonic()
    with pytest.raises(preflight.PreflightError, match="time limit"):
        verify(png(), "source.png", "image")
    assert time.monotonic() - started < 3
    assert list(tmp_path.iterdir()) == []
