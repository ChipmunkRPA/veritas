"""Seal one explicitly owned v202 case without copying model links or caches.

The caller creates ``.v202-owned.json`` before starting a case. It must be a
regular file containing schema ``chipmunk.vcd2-case-work/202``, a nonempty
``case_id`` matching the work-directory name, and the exact absolute
``work_path``. Each case gets its own work directory and distinct authored stage.

``allowed_data_links`` is a map from ``data/fusion/...`` to an inventory record
containing ``target`` (the exact absolute link destination) and ``sha256``.
Only explicitly listed fuser links are excluded from the archive; their target
bytes are checked and recorded. Model/HF links and ephemeral cache/tmp contents
are inventoried but never copied. Links are never traversed by directory walks.
The returned receipt is also written beside the tar as an exclusive private
``.tar.receipt.json`` sidecar, which is required for verification and release.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import tarfile
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

MARKER = ".v202-owned.json"
SCHEMA = "chipmunk.vcd2-case-archive/202"
WORK_SCHEMA = "chipmunk.vcd2-case-work/202"
EXCLUDED = frozenset({"models", "hf", "cache", "tmp"})
BLOCK = 1024 * 1024


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def direct_dir(path: Path) -> Path:
    path = Path(path)
    require(path.is_absolute() and path.is_dir() and not path.is_symlink()
            and path.resolve(strict=True) == path, "An exact absolute non-symlink directory is required")
    return path


def safe_rel(value: str) -> bool:
    path = PurePosixPath(value)
    return bool(value and not path.is_absolute() and all(part not in {"", ".", ".."} for part in path.parts)
                and path.as_posix() == value)


def file_hash(path: Path) -> str:
    require(not path.is_symlink(), "File symlinks are not followed")
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    with os.fdopen(fd, "rb") as stream:
        require(stat.S_ISREG(os.fstat(stream.fileno()).st_mode), "Only regular files can be read")
        return hashlib.file_digest(stream, "sha256").hexdigest()


def marker(work: Path) -> dict[str, Any]:
    path = work / MARKER
    require(path.is_file() and not path.is_symlink(), "Study-owned case marker is required")
    item = json.loads(path.read_text())
    case_id = item.get("case_id")
    require(item.get("schema") == WORK_SCHEMA and isinstance(case_id, str)
            and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", case_id) is not None
            and case_id == work.name and item.get("work_path") == str(work),
            "Ownership marker does not name this exact case directory")
    return item


def _expected_links(value: Mapping[str, Mapping[str, str]] | None) -> dict[str, dict[str, str]]:
    result = {}
    for path, row in (value or {}).items():
        require(isinstance(path, str) and safe_rel(path) and path.startswith("data/fusion/")
                and isinstance(row, Mapping), "Invalid explicit fuser link inventory")
        target, digest = row.get("target"), row.get("sha256")
        require(isinstance(target, str) and Path(target).is_absolute() and isinstance(digest, str)
                and re.fullmatch(r"[0-9a-f]{64}", digest) is not None,
                "Each approved fuser link needs an absolute target and SHA-256")
        result[path] = {"target": target, "sha256": digest}
    return result


def _inventory(work: Path, stage: Path, allowed: Mapping[str, Mapping[str, str]] | None):
    approved = _expected_links(allowed)
    used: set[str] = set()
    entries: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []

    def walk(path: Path, root: Path, label: str, exclude: bool = False) -> None:
        relative = path.relative_to(root).as_posix() if path != root else ""
        name = label + ("/" + relative if relative else "")
        require(safe_rel(name), "Unsafe source path")
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode):
            target = os.readlink(path)
            if label == "work" and relative.startswith("data/"):
                expected = approved.get(relative)
                require(expected is not None and target == expected["target"], "Unapproved data link")
                destination = Path(target)
                require(destination.is_absolute() and destination.is_file() and not destination.is_symlink()
                        and destination.resolve(strict=True) == destination
                        and file_hash(destination) == expected["sha256"], "Approved fuser bytes differ")
                used.add(relative)
                excluded.append({"path": name, "kind": "fuser_link", **expected})
                return
            require(label == "work" and exclude, "Unexpected symlink in owned case data or guard stage")
            excluded.append({"path": name, "kind": "link", "target": target})
            return
        if stat.S_ISDIR(info.st_mode):
            (excluded if exclude else entries).append({"path": name, "kind": "directory"})
            with os.scandir(path) as listing:
                children = sorted((Path(item.path) for item in listing), key=lambda item: item.name)
            for child in children:
                child_excluded = exclude or (label == "work" and path == root and child.name in EXCLUDED)
                walk(child, root, label, child_excluded)
            return
        require(stat.S_ISREG(info.st_mode), "Only regular files, directories and approved links are supported")
        record = {"path": name, "kind": "file", "size": info.st_size, "sha256": file_hash(path)}
        (excluded if exclude else entries).append(record)

    walk(work, work, "work")
    walk(stage, stage, "stage")
    require(used == set(approved), "Approved fuser link inventory does not match the case")
    return entries, excluded


def _tar_entry(archive: tarfile.TarFile, record: dict[str, Any], work: Path, stage: Path) -> None:
    name = record["path"]
    label, _, relative = name.partition("/")
    base = work if label == "work" else stage
    path = base / relative if relative else base
    header = tarfile.TarInfo(name)
    header.uid = header.gid = header.mtime = 0
    header.uname = header.gname = ""
    if record["kind"] == "directory":
        header.type, header.mode, header.size = tarfile.DIRTYPE, 0o700, 0
        archive.addfile(header)
    else:
        require(not path.is_symlink(), "Source file changed to a symlink")
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        with os.fdopen(fd, "rb") as stream:
            info = os.fstat(stream.fileno())
            require(stat.S_ISREG(info.st_mode) and info.st_size == record["size"], "Source file changed during sealing")
            header.mode, header.size = 0o600, record["size"]
            archive.addfile(header, stream)


def verify_archive(archive: Path, receipt: Mapping[str, Any]) -> None:
    """Verify the complete tar against its receipt without extracting it."""
    archive = Path(archive)
    require(receipt.get("schema") == SCHEMA and archive.is_absolute() and archive.is_file()
            and not archive.is_symlink() and archive.resolve(strict=True) == archive,
            "Exact direct archive and receipt required")
    require(str(archive) == receipt.get("archive_path") and file_hash(archive) == receipt.get("archive_sha256"),
            "Archive bytes or path differ from receipt")
    receipt_path = Path(str(archive) + ".receipt.json")
    require(str(receipt_path) == receipt.get("receipt_path") and receipt_path.is_file() and not receipt_path.is_symlink()
            and receipt_path.resolve(strict=True) == receipt_path and json.loads(receipt_path.read_text()) == receipt,
            "Adjacent private archive receipt is missing or differs")
    expected = receipt.get("entries")
    require(isinstance(expected, list) and all(isinstance(row, dict) and safe_rel(row.get("path", "")) for row in expected)
            and len({row["path"] for row in expected}) == len(expected), "Invalid archive manifest")
    with tarfile.open(archive, "r:") as stream:
        members = stream.getmembers()
        require([entry.name for entry in members] == [row["path"] for row in expected],
                "Archive member order or names differ")
        for entry, row in zip(members, expected, strict=True):
            if row.get("kind") == "directory":
                require(entry.isdir() and entry.size == 0, "Archive directory differs")
            else:
                require(row.get("kind") == "file" and entry.isfile() and entry.size == row.get("size"),
                        "Archive file type or size differs")
                with stream.extractfile(entry) as data:
                    require(data is not None and hashlib.file_digest(data, "sha256").hexdigest() == row.get("sha256"),
                            "Archive file bytes differ")


def seal_case(work: Path, stage: Path, archive: Path, *,
              allowed_data_links: Mapping[str, Mapping[str, str]] | None = None) -> dict[str, Any]:
    """Seal an owned case, including incomplete cases, without overwriting attempts."""
    work, stage, archive = direct_dir(work), direct_dir(stage), Path(archive)
    receipt_path = Path(str(archive) + ".receipt.json")
    require(not work.is_relative_to(stage) and not stage.is_relative_to(work), "Case and guard stage must be separate")
    require(archive.is_absolute() and archive.suffix == ".tar" and direct_dir(archive.parent) == archive.parent
            and not archive.exists() and not archive.is_symlink() and not receipt_path.exists() and not receipt_path.is_symlink()
            and not archive.is_relative_to(work)
            and not archive.is_relative_to(stage), "Fresh archive path outside source trees required")
    ownership = marker(work)
    allowed = _expected_links(allowed_data_links)
    entries, excluded = _inventory(work, stage, allowed)
    # Exclusive creation deliberately keeps a partial archive on any failure.
    with archive.open("xb") as raw:
        os.chmod(archive, 0o600)
        with tarfile.open(fileobj=raw, mode="w", format=tarfile.PAX_FORMAT) as stream:
            for row in entries:
                _tar_entry(stream, row, work, stage)
        raw.flush()
        os.fsync(raw.fileno())
    receipt = {"schema": SCHEMA, "case_id": ownership["case_id"], "work_path": str(work), "stage_path": str(stage),
               "archive_path": str(archive), "receipt_path": str(receipt_path), "archive_sha256": file_hash(archive), "entries": entries,
               "excluded": excluded, "allowed_data_links": allowed}
    with receipt_path.open("x") as stream:
        os.chmod(receipt_path, 0o600)
        json.dump(receipt, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    verify_archive(archive, receipt)
    require(_inventory(work, stage, allowed) == (entries, excluded), "Source tree changed while sealing")
    return receipt


def release_local_work(work: Path, archive: Path, receipt: Mapping[str, Any]) -> None:
    """Remove only the bound work directory after archive and source re-verification."""
    work = direct_dir(work)
    stage = direct_dir(Path(receipt["stage_path"]))
    require(str(work) == receipt.get("work_path") and marker(work)["case_id"] == receipt.get("case_id")
            and not work.is_relative_to(stage) and not stage.is_relative_to(work), "Receipt does not bind this exact case")
    archive = Path(archive)
    require(not archive.is_relative_to(work) and not archive.is_relative_to(stage), "Archive must remain outside source trees")
    verify_archive(archive, receipt)
    require(_inventory(work, stage, receipt.get("allowed_data_links")) == (receipt.get("entries"), receipt.get("excluded")),
            "Source bytes or inventory changed after sealing")
    require(shutil.rmtree.avoids_symlink_attacks, "Safe directory removal is unavailable on this platform")
    shutil.rmtree(work)
