"""User-facing error text: no file-system paths, no internals, bounded length. Shared by the v1 job runner and the
v2 detector runner boundary."""
from __future__ import annotations

import re

from app.core import deadline
from app.core.config import settings

FRIENDLY = [
    ("cannot identify image file", "The file is not a decodable image (corrupt or unsupported encoding)."),
    ("image file is truncated", "The image file is truncated or corrupt."),
    ("Decompression", "The image is too large to process safely (decompression-bomb guard)."),
    ("could not be decoded", "The video could not be decoded (corrupt, unsupported codec, or no video stream)."),
    ("No frames could be decoded", "No video frames could be decoded from the file."),
    ("too little text", None),  # keep the original message (already user-facing)
    ("time budget", None),
    ("Weight file", None),
]
USER_FACING = (ValueError, deadline.JobTimeout)
_PATH = re.compile(r"(/[\w.\-]+){2,}")


def sanitise_message(msg: str) -> str:
    """Strip the data directory and any absolute path; cap the length."""
    msg = msg.replace(str(settings.data_dir), "<data>")
    return _PATH.sub("<path>", msg)[:400]


def friendly_error(e: BaseException) -> str:
    """User-facing failure reason: no file-system paths, no class internals, bounded length."""
    msg = str(e) or e.__class__.__name__
    if not isinstance(e, USER_FACING) and not any(n.lower() in msg.lower() for n, _ in FRIENDLY):
        return f"Internal processing error ({e.__class__.__name__}); details are in the server log under this analysis id."
    for needle, text in FRIENDLY:
        if needle.lower() in msg.lower():
            msg = text or msg
            break
    return sanitise_message(msg)
