"""Bounded, local readability checks before a customer is sent to checkout.

This is a file preflight, not an authenticity assessment. It checks image decoding,
the first video frame / short audio sample, and document structure without OCR or
model inference. A later damaged portion or an inconclusive result is still possible.
Parsers run in an isolated child with a wall-clock deadline; temporary uploads and
all child processes are removed on failure. Source bytes never leave this machine.
"""
from __future__ import annotations

import json
import math
import os
import signal
import subprocess
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path
from typing import BinaryIO

WALL_SECONDS = 15
PARSER_CPU_SECONDS = 12
MAX_IMAGE_PIXELS = 80_000_000  # Matches engine.media.StoredMedia.
MAX_DOCX_EXPANDED = 200 * 1024 * 1024  # Matches storage.files.check_docx_safe.
CHUNK = 1024 * 1024


class PreflightError(ValueError):
    """A safe, customer-facing refusal; no payment has been created."""


def _worker(path: Path, kind: str, extension: str) -> None:
    environment = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    process = subprocess.Popen([sys.executable, "-I", str(Path(__file__).resolve()), "--check", str(path), kind, extension],
                               stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                               start_new_session=True, env=environment)
    try:
        output, _ = process.communicate(timeout=WALL_SECONDS)
    except subprocess.TimeoutExpired:
        _kill(process)
        raise PreflightError("File preparation exceeded its time limit. Try re-exporting the file. No payment was started.") from None
    finally:
        if process.poll() is None:
            _kill(process)
    try:
        response = json.loads(output)
    except (ValueError, TypeError):
        response = {}
    if process.returncode != 0 or response.get("ok") is not True:
        message = response.get("error")
        raise PreflightError(message if isinstance(message, str) else "This file could not be safely prepared. No payment was started.")


def _kill(process: subprocess.Popen) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except (AttributeError, ProcessLookupError, PermissionError):
        process.kill()
    process.communicate()


def validate_upload(src: BinaryIO, filename: str, kind: str) -> None:
    """Check a seekable upload and restore its position on both success and failure."""
    from app.storage import files

    position = src.tell()
    try:
        src.seek(0, os.SEEK_END)
        size = src.tell()
        src.seek(0)
        if size == 0:
            raise PreflightError("The file is empty. Choose a readable file before paying.")
        head = src.read(8192)
        try:
            _, extension = files.validate_kind(kind, head, filename, size=size)
        except ValueError as error:
            raise PreflightError(str(error)) from None
        src.seek(0)
        with tempfile.TemporaryDirectory(prefix="chipmunk-preflight-") as directory:
            path = Path(directory) / f"source{extension}"
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(descriptor, "wb") as target:
                remaining = size
                while remaining:
                    chunk = src.read(min(CHUNK, remaining))
                    if not chunk:
                        raise PreflightError("The upload ended unexpectedly. Upload it again before paying.")
                    target.write(chunk)
                    remaining -= len(chunk)
            _worker(path, kind, extension)
    finally:
        src.seek(position)


def _limits() -> None:
    # CPU limits are per-process; the parent also kills the entire process group
    # after WALL_SECONDS, including an ffmpeg subprocess that has stopped responding.
    try:
        import resource
        resource.setrlimit(resource.RLIMIT_CPU, (PARSER_CPU_SECONDS, PARSER_CPU_SECONDS + 1))
        if sys.platform != "darwin":
            resource.setrlimit(resource.RLIMIT_AS, (1536 * 1024 * 1024, 1536 * 1024 * 1024))
    except (ImportError, OSError, ValueError):
        pass  # The parent's wall-clock kill still applies.


def _image(path: Path) -> None:
    from PIL import Image, ImageFile

    try:
        from pillow_heif import register_heif_opener
        register_heif_opener()
    except ImportError:
        pass
    Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS
    ImageFile.LOAD_TRUNCATED_IMAGES = False
    with Image.open(path) as image:
        if not (0 < image.width * image.height <= MAX_IMAGE_PIXELS):
            raise PreflightError("The image exceeds the 80 MP limit. Resize it before paying.")
        image.verify()
    with Image.open(path) as image:
        image.load()


def _positive(value: object) -> bool:
    try:
        number = float(value)
        return math.isfinite(number) and number > 0
    except (TypeError, ValueError):
        return False


def _av(path: Path, kind: str) -> None:
    selector = "a:0" if kind == "audio" else "v:0"
    probe = subprocess.run(["ffprobe", "-v", "error", "-protocol_whitelist", "file", "-threads", "1",
                            "-select_streams", selector, "-show_entries", "stream=codec_type,width,height,duration:format=duration",
                            "-of", "json", str(path)], stdin=subprocess.DEVNULL, capture_output=True, timeout=5, check=True)
    info = json.loads(probe.stdout)
    streams = info.get("streams") or []
    if not streams or streams[0].get("codec_type") != kind:
        raise PreflightError(f"This file has no readable {kind} stream. No payment was started.")
    if not any(_positive(duration) for duration in (streams[0].get("duration"), (info.get("format") or {}).get("duration"))):
        raise PreflightError(f"The {kind} has no readable duration. Re-export it before paying.")
    common = ["ffmpeg", "-nostdin", "-v", "error", "-xerror", "-protocol_whitelist", "file", "-threads", "1",
              "-i", str(path), "-map", f"0:{selector}", "-threads", "1"]
    if kind == "video":
        if not (_positive(streams[0].get("width")) and _positive(streams[0].get("height"))):
            raise PreflightError("The video has no readable frame dimensions. No payment was started.")
        if int(streams[0]["width"]) * int(streams[0]["height"]) > MAX_IMAGE_PIXELS:
            raise PreflightError("Video frames exceed the 80 MP limit. Resize the video before paying.")
        command = [*common, "-frames:v", "1", "-vf", "scale=16:16", "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1"]
    else:
        command = [*common, "-t", "0.25", "-ac", "1", "-ar", "8000", "-f", "s16le", "pipe:1"]
    sample = subprocess.run(command, stdin=subprocess.DEVNULL, capture_output=True, timeout=5, check=True)
    if not sample.stdout:
        raise PreflightError(f"The {kind} stream contains no decodable content. No payment was started.")


def _document(path: Path, extension: str) -> None:
    if extension == ".pdf":
        import pymupdf
        with pymupdf.open(path) as document:
            if not document.is_pdf or document.needs_pass or document.page_count < 1:
                raise PreflightError("The PDF is empty, unreadable or password-protected. Export an unlocked PDF before paying.")
            # Loading the first page exercises the page tree without OCR or a raster allocation.
            document.load_page(0)
        return
    if extension == ".docx":
        import zipfile

        from docx import Document
        with zipfile.ZipFile(path) as archive:
            entries = archive.infolist()
            expanded = sum(entry.file_size for entry in entries)
            compressed = max(1, sum(entry.compress_size for entry in entries))
            if expanded > MAX_DOCX_EXPANDED or expanded / compressed > 200 or len(entries) > 10000:
                raise PreflightError("This document exceeds the safe archive expansion limit. Re-export it before paying.")
            if archive.testzip() is not None:
                raise PreflightError("The document archive is damaged. Re-export it before paying.")
        Document(path)
        return
    raw = path.read_bytes()
    if extension == ".eml":
        from email import policy
        from email.parser import BytesParser

        message = BytesParser(policy=policy.default).parsebytes(raw)
        body = message.get_body(preferencelist=("plain", "html"))
        if body is None:
            raise PreflightError("This email has no readable text or HTML body. Attachments are not analysed as its body.")
        text = body.get_content()
        if not isinstance(text, str):
            raise PreflightError("This email body could not be decoded as text. Re-export it before paying.")
    else:
        text = raw.decode("utf-16") if raw.startswith((b"\xff\xfe", b"\xfe\xff")) else raw.decode("utf-8-sig")
    from app.evidence.document_invisible_characters import ENCODED_FORMATTING

    readable = sum(char.isprintable() and not char.isspace() for char in text)
    unusual = sum(not (char.isprintable() or char.isspace() or char in ENCODED_FORMATTING) for char in text)
    if not readable or "\x00" in text or unusual > max(1, len(text) // 100):
        raise PreflightError("This document has no readable text. Export it as UTF-8 text, PDF or DOCX before paying.")


def _main() -> int:
    _limits()
    path, kind, extension = Path(sys.argv[2]), sys.argv[3], sys.argv[4]
    try:
        with redirect_stdout(sys.stderr):
            if kind == "image":
                _image(path)
            elif kind in {"audio", "video"}:
                _av(path, kind)
            elif kind == "document":
                _document(path, extension)
            else:
                raise PreflightError("Choose an image, video, audio or document before paying.")
        print(json.dumps({"ok": True}))
        return 0
    except Exception as error:
        message = str(error) if isinstance(error, PreflightError) else {
            "image": "This image could not be fully decoded. Re-export it before paying.",
            "audio": "This audio could not be decoded. Re-export it before paying.",
            "video": "This video could not be decoded. Re-export it before paying.",
            "document": "This document is unreadable or damaged. Export a readable PDF, DOCX or UTF-8 text file before paying.",
        }.get(kind, "This file could not be prepared. No payment was started.")
        print(json.dumps({"ok": False, "error": message}))
        return 1


if __name__ == "__main__" and len(sys.argv) == 5 and sys.argv[1] == "--check":
    raise SystemExit(_main())
