"""Configure the qualified CPU envelope at owned worker startup, never in a request."""
from __future__ import annotations

import threading

from app.core.config import settings

_lock = threading.Lock()


def configure_worker() -> None:
    jpeg_review = settings.environment == "dev" and settings.image_region_jpeg_review
    if settings.image_region_mode != "candidate" and not jpeg_review:
        return
    import torch

    with _lock:
        if torch.get_num_threads() != 2:
            torch.set_num_threads(2)
