"""Verify private evaluation files against the public receipt contract."""

from __future__ import annotations

import hashlib
import stat
from pathlib import Path, PurePosixPath

from app.bench import independent

INDEX_SCHEMA = "chipmunk.independent-file-index/1"
RECEIPT_SCHEMA = "chipmunk.independent-input-verification/1"


def _file(root, name):
    if not isinstance(name, str) or not name or "\\" in name or "\x00" in name:
        raise ValueError("File index needs relative POSIX paths")
    relative = PurePosixPath(name)
    if relative.is_absolute() or relative.as_posix() != name or ".." in relative.parts or name == ".":
        raise ValueError("File index needs normalized paths inside the supplied root")
    try:
        path = root.joinpath(*relative.parts).resolve()
    except (OSError, RuntimeError):
        raise ValueError("Could not resolve a required indexed file") from None
    if not path.is_relative_to(root):
        raise ValueError("Indexed file resolves outside the supplied root")
    return path


def _signature(state):
    return state.st_dev, state.st_ino, state.st_size, state.st_mtime_ns, state.st_ctime_ns


def _match(path, expected, cache):
    try:
        before = path.stat()
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("Indexed inputs and sources must be regular files")
        identity = _signature(before)
        if path not in cache:
            with path.open("rb") as handle:
                checksum = hashlib.file_digest(handle, "sha256").hexdigest()
            cache[path] = identity, checksum
        captured, checksum = cache[path]
        if captured != identity or identity != _signature(path.stat()):
            raise ValueError("Indexed file changed during verification")
    except OSError:
        raise ValueError("Could not read a required indexed file") from None
    if checksum != expected:
        raise ValueError("Indexed file bytes differ from the manifest SHA-256")


def verify_inputs(manifest, index, root: Path):
    """Check every submitted file and declared parent source, then return only counts."""
    audit = independent.audit_manifest(manifest)
    if (not isinstance(index, dict) or index.keys() != {"schema", "manifest_sha256", "cases"}
            or index["schema"] != INDEX_SCHEMA or index["manifest_sha256"] != audit["manifest_sha256"]
            or not isinstance(index["cases"], list)):
        raise ValueError("File index must bind to the exact evaluation manifest")
    try:
        directory = root.resolve()
    except (OSError, RuntimeError):
        raise ValueError("Could not resolve the supplied file root") from None
    if not directory.is_dir():
        raise ValueError("File root must be a local directory")
    expected = {case["id"]: case for case in manifest["cases"]}
    cache = {}
    for row in index["cases"]:
        if (not isinstance(row, dict) or row.keys() != {"id", "input", "source"}
                or not isinstance(row["id"], str) or row["id"] not in expected):
            raise ValueError("Unexpected or duplicate indexed evaluation case")
        case = expected.pop(row["id"])
        for role in ("input", "source"):
            _match(_file(directory, row[role]), case[f"{role}_sha256"], cache)
    if expected:
        raise ValueError("Every planned input and parent source needs a file index row")
    # Recheck metadata after walking the complete panel to catch changed shared sources.
    for path, (identity, _) in cache.items():
        try:
            if identity != _signature(path.stat()):
                raise ValueError("Indexed file changed during verification")
        except OSError:
            raise ValueError("A required indexed file disappeared during verification") from None
    return {"schema": RECEIPT_SCHEMA, "manifest_sha256": audit["manifest_sha256"],
            "verified_cases": audit["cases"], "verified_file_bindings": 2 * audit["cases"]}
