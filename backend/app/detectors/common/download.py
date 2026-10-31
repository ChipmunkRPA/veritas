"""v1 download helper for non-HF weights (UnivFD linear probe, YuNet ONNX, DFDC checkpoints).

Streams with an incremental SHA-256 (`app.engine.fetchers`), refuses unpinned files and VERITAS_OFFLINE=true, and
skips the full re-hash on load when the signed verify record is fresh.
"""
from __future__ import annotations

from pathlib import Path

from app.core.config import settings
from app.core.logging import get_logger
from app.engine.fetchers import FetchError, VerifyRecord, WeightIntegrityError, sha256_file, stream_to_file

log = get_logger(__name__)


def fetch(url: str, filename: str) -> Path:
    """Download (once) and verify a weight file against its pinned SHA-256. Verified again on load."""
    expected = settings.weights_sha256.get(filename)
    if not expected:
        raise FetchError(f"Weight file {filename} has no pinned sha256 (config.weights_sha256)")
    settings.ensure_dirs()
    dest = settings.weights_dir / filename
    record = VerifyRecord()
    if dest.is_file() and dest.stat().st_size > 0:
        if not record.is_fresh(dest, expected):
            actual = sha256_file(dest)
            if actual != expected:
                dest.unlink(missing_ok=True)
                raise WeightIntegrityError(f"Weight file {filename} failed integrity check (sha256 {actual[:12]}… ≠ "
                                           f"pinned {expected[:12]}…); file removed")
            record.record(dest, expected)
        return dest
    log.info("weights.download", url=url, dest=str(dest))
    stream_to_file(url, dest, expected)
    record.record(dest, expected)
    log.info("weights.downloaded", file=filename, sha256=expected)
    return dest
