"""Bounded, object-bound HTTP ranges and explicit private gzip checkpoints."""

from __future__ import annotations

import hashlib
import importlib.metadata
import io
import json
import sys
import tarfile
import time
from pathlib import Path
from urllib.parse import urlsplit

import requests

ROOT = Path(__file__).resolve().parents[2]
RUNTIME = ROOT / "backend/data/independent-evaluation/image-positive-v21-runtime/packages"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def indexed_module():
    sys.path.insert(0, str(RUNTIME))
    import indexed_gzip

    require(Path(indexed_gzip.__file__).is_relative_to(RUNTIME) and importlib.metadata.version("indexed_gzip") == "1.10.3", "Pinned isolated indexed-gzip required")
    return indexed_gzip


class RangeFile(io.RawIOBase):
    """Read exact ranges of one pinned remote object; no implicit retry/auth."""

    def __init__(self, url, size, etag, journal, *, block_size=8 * 1024**2, session=None):
        super().__init__()
        parsed = urlsplit(url)
        require(parsed.scheme == "https" and not parsed.username and not parsed.password and not parsed.fragment, "Explicit HTTPS object URL required")
        require(type(size) is int and size > 0 and isinstance(etag, str) and etag, "Fixed object size and ETag required")
        require(0 < block_size <= 8 * 1024**2, "Bounded range block required")
        self.url, self.size, self.etag, self.journal = url, size, etag, journal
        self.block_size, self.position, self.transferred = block_size, 0, 0
        self.cached_start, self.cached, self.requests_made = -1, b"", 0
        self.session = session or requests.Session()
        self.own_session = session is None
        if self.own_session:
            # Keep configured TLS/proxy behavior while bypassing netrc authentication.
            self.session.auth = lambda request: request

    def readable(self):
        return True

    def seekable(self):
        return True

    def writable(self):
        return False

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        require(type(offset) is int and whence in (0, 1, 2), "Integer bounded seek required")
        destination = offset + (0 if whence == 0 else self.position if whence == 1 else self.size)
        require(destination >= 0 and destination <= self.size, "Seek outside declared compressed object")
        self.position = destination
        return destination

    def _block(self, start):
        stop = min(self.size, start + self.block_size) - 1
        record = {"start": start, "end": stop}
        try:
            headers = {"Range": f"bytes={start}-{stop}", "If-Match": self.etag, "X-Requested-With": "XMLHttpRequest", "Accept-Encoding": "identity"}
            with self.session.get(self.url, headers=headers, timeout=(15, 45), allow_redirects=False, stream=True) as response:
                record.update({"http_status": response.status_code, "etag": response.headers.get("ETag"), "content_range": response.headers.get("Content-Range")})
                require(response.status_code == 206, "Exact partial-content response required")
                require(response.headers.get("ETag") == self.etag, "Remote archive ETag changed")
                require(response.headers.get("Content-Range") == f"bytes {start}-{stop}/{self.size}", "Remote range or declared object size differs")
                require(response.headers.get("Content-Encoding", "identity") in {"identity", ""}, "HTTP must not transform compressed archive bytes")
                content = bytearray()
                for chunk in response.iter_content(1024**2):
                    content.extend(chunk)
                    require(len(content) <= stop - start + 1, "Remote range response exceeds requested bound")
            require(len(content) == stop - start + 1, "Truncated compressed range")
            self.cached_start, self.cached = start, bytes(content)
            self.transferred += len(content)
            self.requests_made += 1
            record.update({"bytes": len(content), "sha256": hashlib.sha256(content).hexdigest(), "complete": True})
        except BaseException as error:
            record.update({"complete": False, "error_type": type(error).__name__, "error": str(error)})
            raise
        finally:
            self.journal.write(json.dumps(record, allow_nan=False) + "\n")
            self.journal.flush()

    def read(self, size=-1):
        require(type(size) is int and 0 <= size <= 32 * 1024**2, "Explicit read size at most32MiB required")
        size = min(size, self.size - self.position)
        chunks = []
        while size:
            if not self.cached_start <= self.position < self.cached_start + len(self.cached):
                self._block(self.position // self.block_size * self.block_size)
            relative = self.position - self.cached_start
            count = min(size, len(self.cached) - relative)
            chunks.append(self.cached[relative:relative + count])
            self.position += count
            size -= count
        return b"".join(chunks)

    def readinto(self, buffer):
        data = self.read(len(buffer))
        buffer[:len(data)] = data
        return len(data)

    def close(self):
        if self.own_session:
            self.session.close()
        super().close()


def extract_part(remote, wanted, objects, part, *, previous=None, previous_digest=None, max_seconds=240, max_transfer=2 * 1024**3, spacing=8 * 1024**2):
    """Complete tar members, export the partial index, then exit for a new phase."""
    part.mkdir(parents=True, exist_ok=False)
    objects.mkdir(parents=True, exist_ok=True)
    require(isinstance(wanted, set) and bool(wanted), "Explicit nonempty member set required")
    require(0 < max_seconds <= 240 and 0 < max_transfer <= 2 * 1024**3, "Prospective bounded phase required")
    gzip = indexed_module().IndexedGzipFile(fileobj=remote, mode="rb", spacing=spacing, readbuf_size=1024**2, drop_handles=False)
    prior, previous_sha = {}, None
    if previous is not None:
        require(previous_digest is not None and sha(previous) == previous_digest, "Previously committed checkpoint required")
        prior = json.loads(previous.read_text())
        previous_sha = sha(previous)
        require(prior["completed_phase"] is True and not prior["all_wanted_found"], "Only a completed, incomplete-work checkpoint may continue")
        require(prior["object"] == {"url": remote.url, "bytes": remote.size, "etag": remote.etag}, "Checkpoint refers to a different remote object")
        index = previous.parent / "index.gzi"
        require(sha(index) == prior["index_sha256"], "Private gzip checkpoint changed")
        for row in prior["selected"]:
            require(row["member"] in wanted and sha(objects / row["file"]) == row["sha256"], "Retained source selection/bytes changed")
        gzip.import_index(str(index))
        gzip.seek(prior["next_tar_offset"])
    selected = {r["member"]: r for r in prior.get("selected", [])}
    require(set(selected) <= wanted, "Checkpoint contains unexpected members")
    started, headers, next_offset, reached_end = time.monotonic(), 0, prior.get("next_tar_offset", 0), False
    try:
        with tarfile.open(fileobj=gzip, mode="r:", pax_headers=prior.get("pax_headers", {})) as archive:
            for member in archive:
                require(0 <= member.size <= 30_000_000, "Unexpected oversized publisher archive member")
                headers += 1
                if member.name in wanted:
                    require(member.isfile() and member.name not in selected, "Unique regular source PNG required")
                    with archive.extractfile(member) as stream:
                        data = stream.read(30_000_001)
                    require(len(data) == member.size, "Incomplete selected archive PNG")
                    digest = hashlib.sha256(data).hexdigest()
                    target = objects / (digest + ".png")
                    if not target.exists():
                        with target.open("xb") as output:
                            output.write(data)
                    require(sha(target) == digest, "Content-addressed source object differs")
                    selected[member.name] = {"member": member.name, "file": target.name, "sha256": digest, "bytes": len(data)}
                next_offset = archive.offset
                if len(selected) == len(wanted) or time.monotonic() - started >= max_seconds or remote.transferred >= max_transfer:
                    gzip.seek(next_offset)
                    break
            else:
                reached_end = True
            pax = dict(archive.pax_headers)
        encoded_index = io.BytesIO()
        gzip.export_index(fileobj=encoded_index)
        with (part / "index.gzi").open("xb") as stream:
            stream.write(encoded_index.getvalue())
        receipt = {
            "schema": "chipmunk.ranged-gzip-checkpoint/21", "completed_phase": True,
            "object": {"url": remote.url, "bytes": remote.size, "etag": remote.etag},
            "previous_checkpoint_sha256": previous_sha, "next_tar_offset": next_offset, "pax_headers": pax,
            "headers_this_phase": headers, "selected": list(selected.values()),
            "all_wanted_found": set(selected) == wanted, "archive_end_reached": reached_end,
            "requested_members": len(wanted), "compressed_bytes_transferred": remote.transferred,
            "range_requests": remote.requests_made, "duration_s": time.monotonic() - started,
            "index_sha256": sha(part / "index.gzi"), "index_points": gzip.raw.npoints,
            "gzip_crc_validation_disabled_after_index_import": bool(gzip.raw.skip_crc_check),
            "full_archive_sha256_verified": False,
            "scope": "Private index contains decompression history; no full-archive integrity or license grant is inferred from object ETag. Retained selected member hashes bind subsequent pixel review.",
        }
        with (part / "checkpoint.json").open("x") as stream:
            json.dump(receipt, stream, indent=2, allow_nan=False)
            stream.write("\n")
        return receipt
    finally:
        gzip.close()
