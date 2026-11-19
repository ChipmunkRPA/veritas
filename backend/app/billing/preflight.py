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
MAX_BLANK_PDF_CHECK_PAGES = 32  # Bound prepayment checks without opening or rendering page content.
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


def _word_checkout_view(document):
    """Use the visible paragraphs for checkout without changing the evidence reader's text."""
    from app.detectors.text.extract import Document

    paragraphs, spans = [], []
    offset = 0
    for start, end in document.paragraphs:
        text = document.word_checkout_paragraphs.get(start, document.text[start:end])
        if text:
            paragraphs.append(text)
            spans.append((offset, offset + len(text)))
            offset += len(text) + 2
    return Document("\n\n".join(paragraphs), spans)


def _document(path: Path, extension: str) -> None:
    if extension == ".pdf":
        import pymupdf

        from app.detectors.text.extract import pdf_is_proven_blank

        with pymupdf.open(path) as document:
            if not document.is_pdf or document.needs_pass or document.page_count < 1:
                raise PreflightError("The PDF is empty, unreadable or password-protected. Export an unlocked PDF before paying.")
            # Loading the first page exercises the page tree without OCR or a raster allocation.
            document.load_page(0)
            if pdf_is_proven_blank(document, max_pages=MAX_BLANK_PDF_CHECK_PAGES):
                raise PreflightError("Every page in this PDF is blank. Export a copy that shows the document. No payment was started.")
        return

    from app.core.config import settings
    from app.detectors.text import html_content
    from app.detectors.text.extract import check_raw_size, extract
    from app.detectors.text.segment import MIN_WORDS, sentences, spans
    from app.evidence import document_bibliography, document_tracked_changes
    from app.evidence.document_invisible_characters import ENCODED_FORMATTING
    from app.evidence.document_native_locations import SAVED_FIELDS_KEY
    from app.storage.files import DOCUMENT_ARCHIVES, HTML_EXTS, check_document_archive_safe

    try:
        check_raw_size(path, extension)
    except ValueError as error:
        raise PreflightError(f"{error} No payment was started.") from None

    if extension in {".txt", ".md", ".markdown", *HTML_EXTS}:
        # Keep checkout's strict encoding validation. The extractor also supports
        # forensic review of damaged text with replacement characters.
        html_content.decode_saved(path.read_bytes(), allow_legacy=extension in {".html", ".htm"})

    try:
        if extension in {".docx", ".odt"}:
            check_document_archive_safe(path, extension, check_crc=True)
        document = extract(path, extension, allow_ocr=False, check_word_checkout_visibility=extension == ".docx")
    except ValueError as error:
        raise PreflightError(str(error)) from None

    if document.meta.get("truncated") or len(document.text) > settings.max_document_chars:
        raise PreflightError(f"Writing analysis allows up to {settings.max_document_chars:,} extracted characters. "
                             "Split the document into smaller files. No payment was started.")

    if settings.environment == "dev" and settings.text_bibliography_scope_review:
        document = document_bibliography.body_view(document)
    saved_fields = document.meta.get(SAVED_FIELDS_KEY, {}).get("included")
    if not saved_fields and (document_tracked_changes.projected(document.word_review)
                             or document_tracked_changes.unsupported(document.word_review)):
        # The writing pipeline can still retain native Word revision evidence.
        return
    excludes_hidden = bool(document.word_checkout_paragraphs) and any(
        start in document.word_checkout_paragraphs for start, _ in document.paragraphs)
    if excludes_hidden:
        document = _word_checkout_view(document)
    text = document.text
    readable = any(char.isprintable() and not char.isspace() for char in text)
    if not readable and excludes_hidden:
        raise PreflightError("The Word document contains only text marked hidden in its supported body. "
                             "Remove the hidden formatting or export a PDF that shows the writing. No payment was started.")
    scope = {".docx": "The Word document", ".odt": "The OpenDocument body", ".eml": "The selected email body"}
    too_short = f"{scope.get(extension, 'This document')} needs at least {MIN_WORDS} readable words for writing analysis. No payment was started."
    if extension in {".pptx", ".ppsx", ".odp", ".epub"}:
        too_short = (f"The {extension[1:].upper()} needs at least {MIN_WORDS} readable words for writing analysis. "
                     "For image-only pages or slides, upload a PDF for text recognition. No payment was started.")
    elif extension in {".xlsx", ".ods", ".csv", ".tsv"}:
        too_short = (f"The {extension[1:].upper()} needs at least {MIN_WORDS} readable words in supported text cells. "
                     "No payment was started.")
    elif extension == ".docx" and not readable:
        raise PreflightError("The Word document has no readable body text. Save scanned pages or unsupported layouts as a PDF before paying.")
    elif extension == ".odt" and not readable:
        raise PreflightError("The OpenDocument file has no readable body text. Save scanned pages as a PDF before paying.")
    elif extension not in DOCUMENT_ARCHIVES:
        unusual = sum(not (char.isprintable() or char.isspace() or char in ENCODED_FORMATTING) for char in text)
        if not readable or "\x00" in text or unusual > max(1, len(text) // 100):
            if extension == ".eml":
                raise PreflightError("The selected email body has no readable text. Export the message as EML or PDF before paying.")
            if extension == ".rtf":
                raise PreflightError("The Rich Text file has no readable text in its supported body. "
                                     "Save the displayed text as DOCX or PDF before paying.")
            if extension in {".md", ".markdown"} and not readable:
                raise PreflightError("The Markdown document has no readable text outside fenced code blocks. "
                                     "Add prose before paying.")
            if extension in HTML_EXTS and not readable:
                raise PreflightError("The saved page has no readable text outside navigation and hidden markup. "
                                     "Save the displayed text as PDF or DOCX before paying.")
            raise PreflightError("This document has no readable text. Export it as UTF-8 text, PDF or DOCX before paying.")
    if excludes_hidden:
        too_short = (f"The Word document needs at least {MIN_WORDS} readable words outside text marked hidden. "
                     "No payment was started.")
    elif saved_fields:
        too_short = (f"The Word document includes saved field text but fewer than {MIN_WORDS} readable words. "
                     "No payment was started.")

    # The pipeline segments each paragraph independently. The same count can
    # stop once the minimum is known without segmenting the remainder here.
    words = 0
    for paragraph in document.paragraphs:
        words += sum(sentence.n_words for sentence in sentences(document, spans(text, (paragraph,))))
        if words >= MIN_WORDS:
            return
    raise PreflightError(too_short)


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
