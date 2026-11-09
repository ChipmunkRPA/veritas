"""Configure the qualified CPU envelope at owned worker startup, never in a request."""
from __future__ import annotations

import threading

from app.core.config import settings

_lock = threading.Lock()


def configure_worker() -> None:
    if settings.image_region_mode != "candidate":
        return
    import torch

    with _lock:
        if torch.get_num_threads() != 2:
            torch.set_num_threads(2)
