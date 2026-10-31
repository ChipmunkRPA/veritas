"""The process has been told to stop (BUILD_SPEC N-14; ops review OPS-15).

uvicorn stops accepting connections on SIGTERM and then waits for the open ones to end. An event stream
(`GET /api/v2/analyses/{id}/events`) never ends by itself while its analysis runs, so one open stream kept the API alive, and
refusing connections, until the client went away: 50 seconds and more in the review. `STOPPING` is set by the signal, before
uvicorn's own handler runs, and each stream ends at its next poll; clients reconnect with `Last-Event-ID`. The image also
gives uvicorn `--timeout-graceful-shutdown`, which ends whatever is still open after that many seconds.
"""
from __future__ import annotations

import signal
import threading
from collections.abc import Callable

STOPPING = threading.Event()
SIGNALS = (signal.SIGTERM, signal.SIGINT)


def watch_signals() -> Callable[[], None]:
    """Set `STOPPING` on SIGTERM or SIGINT and then call the handler that was installed before (uvicorn's, which begins the
    shutdown). Returns the function that puts those handlers back. Outside the main thread (the test client) no handler
    can be installed: nothing is changed."""
    STOPPING.clear()
    previous: dict[int, object] = {}

    def on_signal(signum: int, frame: object) -> None:
        STOPPING.set()
        before = previous.get(signum)
        if callable(before):
            before(signum, frame)
        elif before == signal.SIG_DFL:
            signal.signal(signum, signal.SIG_DFL)
            signal.raise_signal(signum)

    try:
        for signum in SIGNALS:
            previous[signum] = signal.signal(signum, on_signal)
    except ValueError:  # not the main thread of the main interpreter
        previous.clear()

    def restore() -> None:
        for signum, before in previous.items():
            if signal.getsignal(signum) is on_signal:
                signal.signal(signum, before)  # type: ignore[arg-type]

    return restore
