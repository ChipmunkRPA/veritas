"""Network access for the benchmark fetcher: HF Hub tree/resolve over HTTP, range-read files, byte meters.

Everything goes through one `httpx.Client`, so a loopback fixture server can stand in for the Hub in tests
(`endpoint=`). Every request calls `require_online()` (VERITAS_OFFLINE) first and every body byte is charged to a
`ByteMeter`, which is how the per-source and per-object caps are enforced exactly.
"""
from __future__ import annotations

import io
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC
from email.utils import parsedate_to_datetime
from urllib.parse import quote

import httpx

from app.engine.fetchers import CHUNK, require_online

HF_ENDPOINT = "https://huggingface.co"
USER_AGENT = "veritas-bench/1.0 (vb-1.0 sampling fetcher; research)"
_TIMEOUT = httpx.Timeout(120.0, connect=20.0)
_RETRY_STATUS = {429, 500, 502, 503, 504}


class RemoteError(RuntimeError):
    """A remote listing or read failed."""


class RetryableStatusError(RemoteError):
    """A retryable HTTP status ended the request; callers must not start another retry path."""


class RetryAfterExceeded(RetryableStatusError):
    """The server's minimum delay exceeds the configured per-retry wait budget."""


class RangeUnsupported(RemoteError):
    """The server ignored a Range request; the caller must not fall back to a bulk download."""


class ByteCapReached(RuntimeError):
    """A byte budget would be exceeded."""

    def __init__(self, scope: str, cap: int):
        super().__init__(f"{scope} byte cap reached ({cap} bytes)")
        self.scope, self.cap = scope, cap


class ByteMeter:
    """Charges transferred bytes against a cap; `charge` raises before the cap is exceeded."""

    def __init__(self, scope: str, cap: int | None, used: int = 0):
        self.scope, self.cap, self.used = scope, cap, used

    def remaining(self) -> int | None:
        return None if self.cap is None else self.cap - self.used

    def charge(self, n: int) -> None:
        if self.cap is not None and self.used + n > self.cap:
            raise ByteCapReached(self.scope, self.cap)
        self.used += n

    def would_exceed(self, n: int) -> bool:
        return self.cap is not None and self.used + n > self.cap


class Meters:
    """The meters a transfer is charged to (source cap and, optionally, the dry-run/metadata meter)."""

    def __init__(self, *meters: ByteMeter):
        self.meters = meters

    def charge(self, n: int) -> None:
        for meter in self.meters:
            if meter.would_exceed(n):
                raise ByteCapReached(meter.scope, meter.cap or 0)
        for meter in self.meters:
            meter.used += n


@dataclass(frozen=True)
class RemoteFile:
    path: str
    size: int


class Http:
    """A paced, retrying GET client."""

    def __init__(self, client: httpx.Client | None = None, *, pace_s: float = 0.0, retries: int = 3, retries_429: int = 6,
                 max_backoff_s: float = 120.0, sleep: Callable[[float], None] = time.sleep):
        self.client = client or httpx.Client(follow_redirects=True, timeout=_TIMEOUT, headers={"User-Agent": USER_AGENT})
        self.pace_s, self.retries, self.retries_429, self.max_backoff_s = pace_s, retries, retries_429, max_backoff_s
        self.sleep = sleep
        self._last = 0.0

    def _pace(self) -> None:
        if self.pace_s > 0:
            wait = self._last + self.pace_s - time.monotonic()
            if wait > 0:
                self.sleep(wait)
        self._last = time.monotonic()

    def _wait_before_retry(self, attempt: int, response: httpx.Response | None) -> float | None:
        """Seconds to wait before another attempt, or None once the budget is spent. 429 has its own, larger
        budget. A valid Retry-After is a server minimum, never shortened to fit the local wait budget."""
        throttled = response is not None and response.status_code == 429
        if attempt >= (self.retries_429 if throttled else self.retries):
            return None
        header = response.headers.get("retry-after", "").strip() if response is not None else ""
        wait = None
        if header.isascii() and header.isdigit():
            wait = float(header)
        elif header:
            try:
                date = parsedate_to_datetime(header)
                # The obsolete asctime HTTP-date format has no explicit timezone; HTTP dates use UTC.
                if date.tzinfo is None:
                    date = date.replace(tzinfo=UTC)
                seconds = date.timestamp() - time.time()
                if seconds > 0:
                    wait = seconds
            except (TypeError, ValueError, OverflowError):
                pass
        if wait is not None:
            if wait > self.max_backoff_s:
                status = response.status_code if response is not None else "unknown"
                raise RetryAfterExceeded(
                    f"HTTP {status}: Retry-After requires {wait:g}s, exceeding the configured "
                    f"maximum wait of {self.max_backoff_s:g}s; retry refused"
                )
            return wait
        # Missing, malformed, negative, or past values retain the bounded exponential fallback.
        return min((5.0 if throttled else 1.0) * 2.0 ** attempt, self.max_backoff_s)

    def get_json(self, url: str, *, params: dict | None = None) -> tuple[object, httpx.Headers]:
        response = self._request(url, params=params)
        try:
            return response.json(), response.headers
        except ValueError as e:
            raise RemoteError(f"{url}: not JSON") from e

    def _request(self, url: str, *, params: dict | None = None) -> httpx.Response:
        require_online()
        attempt = 0
        while True:
            self._pace()
            try:
                response = self.client.get(url, params=params)
            except httpx.HTTPError as e:
                wait = self._wait_before_retry(attempt, None)
                if wait is None:
                    raise RemoteError(f"{url}: request failed ({type(e).__name__})") from e
                self.sleep(wait)
                attempt += 1
                continue
            if response.status_code in _RETRY_STATUS:
                wait = self._wait_before_retry(attempt, response)
                if wait is not None:
                    self.sleep(wait)
                    attempt += 1
                    continue
                raise RetryableStatusError(f"{url}: HTTP {response.status_code}; retry budget exhausted")
            if response.status_code >= 400:
                raise RemoteError(f"{url}: HTTP {response.status_code}")
            return response

    def download(self, url: str, meters: Meters, *, max_bytes: int | None = None) -> bytes:
        """A whole body in memory, charging every chunk; refuses bodies above `max_bytes` (the per-object cap).
        Retries only before the first body byte, so a torn transfer is never stitched together."""
        require_online()
        attempt = 0
        while True:
            self._pace()
            try:
                with self.client.stream("GET", url) as response:
                    if response.status_code in _RETRY_STATUS:
                        wait = self._wait_before_retry(attempt, response)
                        if wait is not None:
                            self.sleep(wait)
                            attempt += 1
                            continue
                        raise RetryableStatusError(f"{url}: HTTP {response.status_code}; retry budget exhausted")
                    if response.status_code >= 400:
                        raise RemoteError(f"{url}: HTTP {response.status_code}")
                    declared = response.headers.get("content-length")
                    if max_bytes is not None and declared and declared.isdigit() and int(declared) > max_bytes:
                        raise RemoteError(f"{url}: {declared} bytes exceeds the per-object cap")
                    body = bytearray()
                    for chunk in response.iter_bytes(CHUNK):
                        if max_bytes is not None and len(body) + len(chunk) > max_bytes:
                            raise RemoteError(f"{url}: body exceeds the per-object cap")
                        meters.charge(len(chunk))
                        body += chunk
                    return bytes(body)
            except httpx.HTTPError as e:
                raise RemoteError(f"{url}: download failed ({type(e).__name__})") from e

    def get_range(self, url: str, start: int, end: int) -> tuple[bytes, str]:
        """Bytes [start, end] inclusive; returns (body, final url after redirects). A server that answers a Range
        request with anything but 206 raises `RangeUnsupported` before its body is read."""
        require_online()
        want = end - start + 1
        attempt = 0
        while True:
            self._pace()
            try:
                with self.client.stream("GET", url, headers={"Range": f"bytes={start}-{end}"}) as response:
                    if response.status_code in _RETRY_STATUS:
                        wait = self._wait_before_retry(attempt, response)
                        if wait is not None:
                            self.sleep(wait)
                            attempt += 1
                            continue
                        raise RetryableStatusError(f"{url}: HTTP {response.status_code}; retry budget exhausted")
                    if response.status_code >= 400:
                        raise RemoteError(f"{url}: HTTP {response.status_code}")
                    if response.status_code != 206:
                        raise RangeUnsupported(f"{url}: the server ignored a Range request (HTTP {response.status_code})")
                    body = bytearray()
                    for chunk in response.iter_bytes(CHUNK):
                        body += chunk
                        if len(body) > want:
                            raise RemoteError(f"{url}: range response longer than requested")
                    return bytes(body), str(response.url)
            except httpx.HTTPError as e:
                wait = self._wait_before_retry(attempt, None)
                if wait is None:
                    raise RemoteError(f"{url}: range read failed ({type(e).__name__})") from e
                self.sleep(wait)
                attempt += 1

    def close(self) -> None:
        self.client.close()


class NoNetwork(Http):
    """An `Http` that refuses every request: the cached-only rebuild of the manifest and report (D-38)."""

    def _refuse(self, url: str, *_args, **_kwargs):
        raise RemoteError(f"{url}: network access refused (cached-only run)")

    _request = download = get_range = _refuse


class RangeFile(io.RawIOBase):
    """A read-only, seekable remote file backed by HTTP Range requests with a single read-ahead block."""

    def __init__(self, http: Http, url: str, size: int, meters: Meters, *, block: int = 1 << 20):
        super().__init__()
        self.http, self.url, self.size, self.meters, self.block = http, url, size, meters, block
        self._resolved = url
        self._pos = 0
        self._buf_start, self._buf = 0, b""

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self._pos, io.SEEK_END: self.size}[whence]
        self._pos = max(0, base + offset)
        return self._pos

    def _fetch(self, start: int, length: int) -> None:
        end = min(self.size, start + max(length, self.block)) - 1
        self.meters.charge(end - start + 1)
        try:
            body, self._resolved = self.http.get_range(self._resolved, start, end)
        except RetryableStatusError:
            # Re-resolving the origin must not bypass retry budgets or a server's minimum delay.
            raise
        except RemoteError:
            if self._resolved == self.url:
                raise
            body, self._resolved = self.http.get_range(self.url, start, end)  # signed CDN URL expired: re-resolve
        if len(body) != end - start + 1:
            raise RemoteError(f"{self.url}: short range read ({len(body)} of {end - start + 1} bytes)")
        self._buf_start, self._buf = start, body

    def readinto(self, buffer) -> int:
        view = memoryview(buffer).cast("B")
        want = min(len(view), self.size - self._pos)
        if want <= 0:
            return 0
        offset = self._pos - self._buf_start
        if not (offset >= 0 and offset + want <= len(self._buf)):
            self._fetch(self._pos, want)
            offset = 0
        view[:want] = self._buf[offset:offset + want]
        self._pos += want
        return want


class HubClient:
    """The subset of the HF Hub HTTP API the fetcher needs (datasets only)."""

    def __init__(self, http: Http, endpoint: str = HF_ENDPOINT):
        self.http, self.endpoint = http, endpoint.rstrip("/")

    def dataset_info(self, repo: str, revision: str) -> dict:
        info, _ = self.http.get_json(f"{self.endpoint}/api/datasets/{repo}/revision/{quote(revision, safe='')}")
        if not isinstance(info, dict):
            raise RemoteError(f"{repo}: unexpected dataset info")
        return info

    def tree(self, repo: str, revision: str) -> list[RemoteFile]:
        url: str | None = f"{self.endpoint}/api/datasets/{repo}/tree/{quote(revision, safe='')}"
        params: dict | None = {"recursive": "true", "expand": "false"}
        files = []
        while url:
            entries, headers = self.http.get_json(url, params=params)
            if not isinstance(entries, list):
                raise RemoteError(f"{repo}: unexpected tree listing")
            files += [RemoteFile(e["path"], int(e.get("size") or 0)) for e in entries if e.get("type") == "file"]
            url, params = _next_link(headers.get("link")), None
        return sorted(files, key=lambda f: f.path)

    def resolve_url(self, repo: str, revision: str, path: str) -> str:
        return f"{self.endpoint}/datasets/{repo}/resolve/{quote(revision, safe='')}/{quote(path)}"


def _next_link(header: str | None) -> str | None:
    for part in (header or "").split(","):
        if 'rel="next"' in part and "<" in part and ">" in part:
            return part[part.index("<") + 1:part.index(">")]
    return None
