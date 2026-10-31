"""Outbound-network monitor for the PL-01 egress test (PRD PL-01, M18; ARCHITECTURE §8.5).

One process, two views:
- **Patched socket.** `socket.socket.connect`, `connect_ex`, `sendto` and `socket.getaddrinfo` record every non-loopback
  destination and refuse it with `EgressBlocked` (an `OSError`, so callers see what an air-gapped host would give them). An
  attempt is counted even when the host has no route.
- **psutil sampler.** A thread polls `psutil.Process().net_connections("inet")` and records any remote address that is not
  loopback, which also covers sockets opened by native code the patch cannot see.

The monitor restores whatever it replaced on exit, so it nests inside the default test suite's socket guard.
"""
from __future__ import annotations

import ipaddress
import socket
import threading
from dataclasses import asdict, dataclass

import psutil


class EgressBlocked(OSError):
    """A non-loopback connection or DNS lookup was attempted under the egress monitor."""


def is_loopback(host) -> bool:
    if host is None:
        return True
    if isinstance(host, bytes):
        host = host.decode("ascii", "replace")
    host = str(host).strip("[]").split("%", 1)[0]
    if host.lower() in ("localhost", "localhost."):
        return True
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return False
    if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped is not None:
        address = address.ipv4_mapped
    return address.is_loopback


@dataclass(frozen=True)
class Attempt:
    via: str  # connect | connect_ex | sendto | getaddrinfo | psutil
    address: str


def _format(address) -> str:
    if isinstance(address, tuple) and len(address) >= 2:
        return f"{address[0]}:{address[1]}"
    return str(address)


class EgressMonitor:
    """`with EgressMonitor() as monitor: ...` then `monitor.attempts` lists every non-loopback destination seen."""

    def __init__(self, *, block: bool = True, sample_s: float = 0.05, process: psutil.Process | None = None):
        self.block = block
        self.sample_s = sample_s
        self._process = process
        self._lock = threading.Lock()
        self._attempts: list[Attempt] = []
        self._seen: set[Attempt] = set()
        self._stop = threading.Event()
        self._sampler: threading.Thread | None = None
        self._saved: tuple | None = None
        self.samples = 0

    @property
    def attempts(self) -> list[Attempt]:
        with self._lock:
            return list(self._attempts)

    def summary(self) -> dict:
        return {"non_loopback_attempts": len(self.attempts), "attempts": [asdict(a) for a in self.attempts],
                "psutil_samples": self.samples}

    def _record(self, via: str, address) -> None:
        attempt = Attempt(via, _format(address))
        with self._lock:
            if attempt not in self._seen:  # the sampler sees one connection many times
                self._seen.add(attempt)
                self._attempts.append(attempt)

    def _check(self, sock: socket.socket, via: str, address) -> None:
        if sock.family not in (socket.AF_INET, socket.AF_INET6):
            return  # AF_UNIX and friends are local by construction
        host = address[0] if isinstance(address, tuple) else address
        if not is_loopback(host):
            self._record(via, address)
            if self.block:
                raise EgressBlocked(f"egress monitor: {via} to {_format(address)} refused")

    def sample(self) -> None:
        """One psutil pass over this process's inet connections."""
        process = self._process or psutil.Process()
        try:
            connections = process.net_connections(kind="inet")
        except (psutil.Error, OSError):
            return
        self.samples += 1
        for c in connections:
            if c.raddr and not is_loopback(c.raddr[0]):
                self._record("psutil", (c.raddr[0], c.raddr[1]))

    def _sample_loop(self) -> None:
        while not self._stop.wait(self.sample_s):
            self.sample()

    def __enter__(self) -> EgressMonitor:
        saved = (socket.socket.connect, socket.socket.connect_ex, socket.socket.sendto, socket.getaddrinfo)
        self._saved = saved
        monitor = self
        connect, connect_ex, sendto, getaddrinfo = saved

        def _connect(sock, address):
            monitor._check(sock, "connect", address)
            return connect(sock, address)

        def _connect_ex(sock, address):
            monitor._check(sock, "connect_ex", address)
            return connect_ex(sock, address)

        def _sendto(sock, data, *args):
            monitor._check(sock, "sendto", args[-1])
            return sendto(sock, data, *args)

        def _getaddrinfo(host, *args, **kwargs):
            if not is_loopback(host):
                monitor._record("getaddrinfo", host)
                if monitor.block:
                    raise EgressBlocked(f"egress monitor: DNS lookup of {host!r} refused")
            return getaddrinfo(host, *args, **kwargs)

        socket.socket.connect = _connect
        socket.socket.connect_ex = _connect_ex
        socket.socket.sendto = _sendto
        socket.getaddrinfo = _getaddrinfo
        self._stop.clear()
        self.sample()
        self._sampler = threading.Thread(target=self._sample_loop, name="veritas-egress-sampler", daemon=True)
        self._sampler.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._sampler is not None:
            self._sampler.join(5.0)
        self.sample()
        assert self._saved is not None
        socket.socket.connect, socket.socket.connect_ex, socket.socket.sendto, socket.getaddrinfo = self._saved
        self._saved = None
