"""Process-local operational state: readiness and the start time. The counters of `/api/metrics` are read from the database
(`app.api.metrics`; N-13)."""
from __future__ import annotations

import threading
import time

ready = threading.Event()  # set when the API's start is over (migrations, key check, registry, worker); `veritas_ready`
started_at = time.time()
