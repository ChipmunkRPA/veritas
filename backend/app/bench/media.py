"""Media checks and the `norm` variant (ML P3).

- `sniff`: magic bytes decide the modality and extension; a body that does not match the source modality is rejected
  (the v0.3 field-test lesson: HTML error pages served as `.jpg`).
- `normalise_image`: decode → RGB 8-bit → strip metadata → JPEG q 92 4:2:0 (PIL), no resizing, no EXIF transpose (the
  pixel grid as distributed).
- `probe`: width, height, JPEG quality estimate, EXIF / C2PA presence, duration.
"""
from __future__ import annotations

import io
import json
import shutil
import subprocess
import warnings
from pathlib import Path

import filetype
import numpy as np
from PIL import Image, UnidentifiedImageError

NORM_QUALITY = 92
NORM_SUBSAMPLING = 2  # PIL: 2 = 4:2:0
MAX_PIXELS = 20_000 * 20_000

_AUDIO_EXTRA = {b"fLaC": ("audio", "flac"), b"OggS": ("audio", "ogg")}


class MediaError(ValueError):
    """The body is not usable media for the source (wrong type, undecodable, too large)."""


def sniff(data: bytes) -> tuple[str, str] | None:
    """(modality, extension) from magic bytes, or None."""
    kind = filetype.guess(data[:8192])
    if kind is not None:
        major = kind.mime.split("/", 1)[0]
        if major in ("image", "video", "audio"):
            return major, kind.extension
    return _AUDIO_EXTRA.get(data[:4])


def check_modality(data: bytes, modality: str) -> str:
    kind = sniff(data)
    if kind is None:
        raise MediaError("unrecognised bytes (magic-byte check failed)")
    if kind[0] != modality:
        raise MediaError(f"magic bytes say {kind[0]}/{kind[1]}, the source declares {modality}")
    return kind[1]


def open_image(data: bytes) -> Image.Image:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            im = Image.open(io.BytesIO(data))
            if im.width * im.height > MAX_PIXELS:
                raise MediaError(f"image too large ({im.width}×{im.height})")
            im.load()
        return im
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError, Image.DecompressionBombError,
            Image.DecompressionBombWarning) as e:
        if isinstance(e, MediaError):
            raise
        raise MediaError(f"undecodable image ({type(e).__name__})") from None


def to_rgb8(im: Image.Image) -> Image.Image:
    if im.mode.startswith("I") or im.mode == "F":  # 16-bit / 32-bit greyscale: rescale to 8-bit, do not clip
        arr = np.asarray(im, dtype=np.float64)
        top = 65535.0 if im.mode.startswith("I;16") else max(float(arr.max()), 255.0)
        return Image.fromarray(np.clip(np.rint(arr * 255.0 / top), 0, 255).astype(np.uint8), mode="L").convert("RGB")
    if im.mode == "P" and "transparency" in im.info:
        im = im.convert("RGBA")
    return im.convert("RGB")


def normalise_image(data: bytes) -> bytes:
    im = open_image(data)
    out = io.BytesIO()
    to_rgb8(im).save(out, format="JPEG", quality=NORM_QUALITY, subsampling=NORM_SUBSAMPLING, optimize=False)
    return out.getvalue()


def jpeg_quality(im: Image.Image) -> int | None:
    from app.detectors.image.forensics import (
        jpeg_info,  # IJG luminance-table estimate shared with the detector
    )

    info = jpeg_info(im)
    return info.get("estimated_quality") if info.get("jpeg") else None


def has_c2pa(data: bytes) -> bool:
    """A JUMBF box labelled c2pa (JPEG APP11 / BMFF uuid / generic) or a PNG caBX chunk."""
    return (b"jumb" in data and b"c2pa" in data) or (data[:8] == b"\x89PNG\r\n\x1a\n" and b"caBX" in data)


def probe_image(data: bytes) -> dict:
    im = open_image(data)
    exif = im.getexif()
    return {"width": im.width, "height": im.height, "jpeg_q_est": jpeg_quality(im) if im.format == "JPEG" else None,
            "has_exif": bool(len(exif)) or "exif" in im.info, "has_c2pa": has_c2pa(data)}


def av_input(path: Path) -> str:
    """An ffmpeg / ffprobe input with an explicit `file:` protocol, for use with `-protocol_whitelist file`: a playlist or a
    name that looks like a URL or an option is never followed."""
    return "file:" + str(Path(path).resolve())


def probe_av(path: Path) -> dict:
    """Duration and (video) frame size via ffprobe when it is installed; empty values otherwise."""
    ffprobe = shutil.which("ffprobe")
    out: dict = {"width": None, "height": None, "duration_s": None, "has_c2pa": None, "jpeg_q_est": None,
                 "has_exif": None}
    if ffprobe is None:
        return out
    try:
        proc = subprocess.run([ffprobe, "-v", "error", "-protocol_whitelist", "file", "-print_format", "json", "-show_format",
                               "-show_streams", av_input(path)], capture_output=True, timeout=60, check=False, stdin=subprocess.DEVNULL)
        document = json.loads(proc.stdout or b"{}")
    except (OSError, subprocess.TimeoutExpired, ValueError):
        return out
    duration = (document.get("format") or {}).get("duration")
    out["duration_s"] = round(float(duration), 3) if duration else None
    video = next((s for s in document.get("streams", []) if s.get("codec_type") == "video"), None)
    if video is not None:
        out["width"], out["height"] = video.get("width"), video.get("height")
    return out
