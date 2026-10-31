"""Read-only benchmark input verification; no decoding or cache regeneration."""
from __future__ import annotations

import hashlib
import os
import re
import stat
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

_SHA256 = re.compile(r"[0-9a-f]{64}")
_TRANSFORM = re.compile(r"[a-z0-9][a-z0-9._-]*")


def _relative_path(value: Any) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise ValueError("benchmark input path must be a nonempty relative POSIX path")
    path = PurePosixPath(value)
    if not path.parts or path.is_absolute() or any(p in {".", ".."} for p in path.parts) or path.as_posix() != value:
        raise ValueError("benchmark input path is unsafe or noncanonical")
    return path


def _regular_path(root: Path, relative: PurePosixPath) -> tuple[Path, os.stat_result]:
    base = Path(root).absolute()
    if ".." in base.parts:
        raise ValueError("benchmark root contains parent traversal")
    path = base.joinpath(*relative.parts)
    for parent in reversed(path.parents):
        info = parent.lstat()
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            raise ValueError("benchmark input has a symlink or non-directory ancestor")
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode):
        raise ValueError("benchmark input is not a regular file (symlinks are refused)")
    return path, info


def _identity(info: os.stat_result) -> tuple[int, int, int, int, int]:
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


def verify_local_input(root: Path, path: str, *, expected_bytes: int | None = None,
                       expected_sha256: str | None = None) -> Path:
    """Verify a clean local input using only pins explicitly supplied by its manifest.

    A missing optional digest makes no content-integrity claim. No media is decoded.
    The returned path is absolute. Missing files and integrity failures raise ValueError.
    """
    relative = _relative_path(path)
    if expected_bytes is not None and (type(expected_bytes) is not int or expected_bytes <= 0):
        raise ValueError("benchmark input byte count must be a positive integer")
    if expected_sha256 is not None and (not isinstance(expected_sha256, str) or not _SHA256.fullmatch(expected_sha256)):
        raise ValueError("benchmark input SHA256 must be 64 lowercase hex characters")
    try:
        target, before = _regular_path(root, relative)
        if expected_bytes is not None and before.st_size != expected_bytes:
            raise ValueError(f"benchmark input size differs: {relative}")
        # O_NONBLOCK prevents an unexpected FIFO replacement from blocking preflight.
        fd = os.open(target, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(fd, "rb") as stream:
            opened = os.fstat(stream.fileno())
            if not stat.S_ISREG(opened.st_mode) or _identity(opened) != _identity(before):
                raise ValueError(f"benchmark input changed before verification: {relative}")
            digest = hashlib.file_digest(stream, "sha256").hexdigest() if expected_sha256 is not None else None
            after = os.fstat(stream.fileno())
        _, current = _regular_path(root, relative)
        if _identity(before) != _identity(after) or _identity(before) != _identity(current):
            raise ValueError(f"benchmark input changed during verification: {relative}")
        if expected_sha256 is not None and digest != expected_sha256:
            raise ValueError(f"benchmark input SHA256 differs: {relative}")
        return target
    except OSError as exc:
        raise ValueError(f"benchmark input missing or unreadable: {relative} ({type(exc).__name__})") from exc


def verify_recorded_artifact(root: Path, record: Mapping[str, Any]) -> Path:
    """Verify a recorded derivative's required path, positive byte count and SHA256.

    Extra manifest/state fields are allowed. This never writes, decodes or repairs files.
    """
    if not isinstance(record, Mapping) or any(record.get(key) is None for key in ("path", "bytes", "sha256")):
        raise ValueError("recorded benchmark artifact requires path, bytes and SHA256")
    return verify_local_input(root, record["path"], expected_bytes=record["bytes"], expected_sha256=record["sha256"])


def recorded_transform_paths(root: Path, selected_ids: list[str], transform: str, *,
                             limit: int | None = None) -> dict[str, Path]:
    """Verify the first ordered, planned derivatives; absent keys remain unplanned.

    The recorded manifest is authoritative. Only chosen rows' payloads and pins are
    verified, so off-limit artifacts cannot enlarge or block a bounded extraction.
    """
    if limit is not None and (type(limit) is not int or limit < 0):
        raise ValueError("benchmark task limit must be a nonnegative integer or None")
    if limit == 0:
        return {}

    import pyarrow.parquet as pq

    if not isinstance(transform, str) or not _TRANSFORM.fullmatch(transform):
        raise ValueError("invalid benchmark transform")
    manifest = verify_local_input(root, "perturbed/manifest.parquet")
    columns = ("item_id", "transform", "path", "bytes", "sha256")
    try:
        table = pq.read_table(manifest, columns=list(columns))
    except Exception as exc:  # pyarrow's malformed-file/schema exceptions vary by release.
        raise ValueError("recorded perturbation manifest is unreadable or missing required columns") from exc
    selected_set = set(selected_ids)
    planned: dict[str, list[dict]] = {}
    for record in table.to_pylist():
        item_id = record["item_id"]
        if not isinstance(item_id, str) or item_id not in selected_set or record["transform"] != transform:
            continue
        planned.setdefault(item_id, []).append(record)
    chosen = [item_id for item_id in selected_ids if item_id in planned][:limit]
    verified = {}
    for item_id in chosen:
        records = planned[item_id]
        if len(records) != 1:
            raise ValueError(f"duplicate recorded benchmark artifact: {item_id}/{transform}")
        record = records[0]
        path = _relative_path(record["path"])
        if (path.parent != PurePosixPath("perturbed") / transform or not path.name.startswith(item_id + ".")
                or path.suffix == ".part"):
            raise ValueError("recorded benchmark artifact path disagrees with item/transform")
        verified[item_id] = verify_recorded_artifact(root, record)
    return verified
