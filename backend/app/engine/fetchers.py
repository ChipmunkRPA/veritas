"""Weight fetch, pin and verify (ARCHITECTURE §8.5).

- `fetch_model`:
  - HF snapshots are downloaded at the pinned revision and then re-hashed.
  - URL and Google Drive files stream to `<file>.part` with an incremental SHA-256 and are published with an atomic
    rename only after the hash matches. A mismatch deletes the part file and fails.
  - Archives are verified before extraction. Only the members listed in the manifest are extracted, each re-verified,
    behind a path-traversal guard.
  - torch_hub sources download the GitHub archive of the pinned commit. `torch_hub_dir` extracts it for
    `torch.hub.load(<dir>, source="local")` and rebuilds the tree whenever it differs from the verified archive.
- `ensure_verified` is the load-time check:
  - a missing file raises `ModelNotInstalled` (`skipped: model not installed`);
  - the (size, mtime_ns) of each verified file is kept in an HMAC-signed record, and a missing or tampered record, or a
    changed file, forces a full re-hash.
- `VERITAS_OFFLINE=true` makes every fetch raise `OfflineError` before any network I/O.
- `pin_manifest` rewrites a model's revision and per-file hashes in `models.yaml` in place, keeping comments, and
  returns a diff for human review.
"""
from __future__ import annotations

import hashlib
import hmac
import html
import json
import os
import re
import secrets
import shutil
import tarfile
import threading
import zipfile
from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import IO, Any
from urllib.parse import urljoin, urlparse

import httpx
import yaml

from app.core.config import settings
from app.core.logging import get_logger
from app.engine.detector import DetectorResult
from app.engine.manifest import ModelEntry, WeightFile, load_manifest, parse_manifest
from app.engine.models import ModelUnavailable

log = get_logger(__name__)

CHUNK = 1 << 20
MAX_OBJECT_BYTES = 5 * 1024**3  # D-05: no single object above 5 GB
GDRIVE_ENDPOINT = "https://drive.google.com/uc"
GITHUB_ARCHIVE = "https://github.com/{repo}/archive/{ref}.zip"  # torch_hub sources: the repository at the pinned commit
MAX_HUB_MEMBERS, MAX_HUB_BYTES = 10_000, 64 * 1024**2
_TIMEOUT = httpx.Timeout(120.0, connect=20.0)


class FetchError(RuntimeError):
    """A weight file could not be fetched."""


class OfflineError(FetchError):
    """VERITAS_OFFLINE=true forbids network fetches."""


class WeightIntegrityError(RuntimeError):
    """A weight file does not match its pin. Messages start with "Weight file" so they stay user-facing."""


class ModelNotInstalled(ModelUnavailable):
    """Pinned weight files are missing locally."""

    def result(self) -> DetectorResult:
        return DetectorResult.skipped("model not installed")


def require_online() -> None:
    if settings.offline:
        raise OfflineError("VERITAS_OFFLINE=true: network fetches are disabled")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


# --- locations ---------------------------------------------------------------------------------------------

def hf_cache_dir() -> Path:
    if settings.hf_home is not None:
        return Path(settings.hf_home) / "hub"
    from huggingface_hub import constants

    return Path(constants.HF_HUB_CACHE)


def _safe_rel(rel: str, model_id: str) -> PurePosixPath:
    path = PurePosixPath(rel)
    if not path.parts or path.is_absolute() or ".." in path.parts:
        raise FetchError(f"{model_id}: unsafe path {rel!r} in the manifest")
    return path


def _is_archive(entry: ModelEntry) -> bool:
    return entry.source.type == "url" and entry.source.archive is not None


def local_install_dir(entry: ModelEntry) -> Path:
    """Where the files of a `local` source are copied to: under the models directory (`VERITAS_MODELS_DIR`, else the data
    directory). For `path: weights/<name>` this is `settings.weights_dir / <name>`."""
    return (settings.models_dir or settings.data_dir) / _safe_rel(entry.source.path or entry.id, entry.id)


def local_dir(entry: ModelEntry) -> Path:
    """The directory a `local` source is read from (QA P1-10): the models directory, or the data directory when the complete
    set of pinned files is there and not in the models directory (installs made before the two were apart)."""
    primary = local_install_dir(entry)
    if settings.models_dir is None:
        return primary
    legacy = settings.data_dir / _safe_rel(entry.source.path or entry.id, entry.id)
    rels = [_safe_rel(f.path, entry.id) for f in entry.source.files]

    def complete(root: Path) -> bool:
        return bool(rels) and all((root / rel).is_file() for rel in rels)

    return legacy if not complete(primary) and complete(legacy) else primary


def file_path(entry: ModelEntry, weight: WeightFile) -> Path:
    """Where a pinned file lives locally."""
    rel = _safe_rel(weight.path, entry.id)
    source = entry.source
    if source.type == "hf":
        return hf_cache_dir() / f"models--{(source.repo or '').replace('/', '--')}" / "snapshots" / (source.revision or "") / rel
    if source.type == "local":
        return local_dir(entry) / rel
    if source.type in ("url", "gdrive"):
        if _is_archive(entry) and weight is not source.files[0]:
            return settings.weights_dir / entry.id / rel
        return settings.weights_dir / rel
    if source.type == "torch_hub":
        return settings.weights_dir / entry.id / rel
    raise FetchError(f"{entry.id}: {source.type} sources have no local layout")


def installed(entry: ModelEntry) -> bool:
    try:
        return bool(entry.source.files) and all(file_path(entry, f).is_file() for f in entry.source.files)
    except FetchError:
        return False


# --- signed verify record ----------------------------------------------------------------------------------

class VerifyRecord:
    """HMAC-signed record of verified files: resolved path → {sha256, size, mtime_ns, verified_at}."""

    def __init__(self, path: Path | None = None, key_path: Path | None = None):
        self.path = path or settings.weights_dir / ".veritas-verified.json"
        self.key_path = key_path or settings.data_dir / "keys" / "verify_hmac.key"
        self._lock = threading.Lock()

    def _key(self) -> bytes:
        self.key_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            fd = os.open(self.key_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            return self.key_path.read_bytes()
        with os.fdopen(fd, "wb") as f:
            f.write(secrets.token_bytes(32))
        return self.key_path.read_bytes()

    def _sign(self, entries: dict) -> str:
        payload = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()
        return hmac.new(self._key(), payload, hashlib.sha256).hexdigest()

    def _load(self) -> dict[str, dict]:
        try:
            document = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        entries, mac = document.get("entries"), document.get("hmac")
        if not isinstance(entries, dict) or not isinstance(mac, str) or not hmac.compare_digest(mac, self._sign(entries)):
            log.warning("weights.verify_record_untrusted", path=str(self.path))
            return {}
        return entries

    def is_fresh(self, file: Path, sha256: str) -> bool:
        entry = self._load().get(str(file.resolve()))
        if entry is None or not file.is_file():
            return False
        st = file.stat()
        return entry.get("sha256") == sha256 and entry.get("size") == st.st_size and entry.get("mtime_ns") == st.st_mtime_ns

    def record(self, file: Path, sha256: str) -> None:
        with self._lock:
            entries = self._load()
            st = file.stat()
            entries[str(file.resolve())] = {"sha256": sha256, "size": st.st_size, "mtime_ns": st.st_mtime_ns,
                                            "verified_at": datetime.now(UTC).isoformat(timespec="seconds")}
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_name(self.path.name + ".tmp")
            tmp.write_text(json.dumps({"entries": entries, "hmac": self._sign(entries)}, indent=2, sort_keys=True),
                           encoding="utf-8")
            os.replace(tmp, self.path)


class NoRecord(VerifyRecord):
    """Every file is hashed and nothing is written down (`models fetch --no-record`; BUILD_SPEC N-14, SEC-17). The record is
    signed with a key in the data directory. A fetch container that does not mount the data volume has no such key: what it
    signed, the processes that load the weights could not trust, and it would replace the record they keep."""

    def is_fresh(self, file: Path, sha256: str) -> bool:
        return False

    def record(self, file: Path, sha256: str) -> None:
        return None


def ensure_verified(entry: ModelEntry, *, record: VerifyRecord | None = None, force: bool = False) -> list[Path]:
    """Every pinned file is present and matches its hash (re-hashed unless the signed record is fresh)."""
    record = record or VerifyRecord()
    paths = []
    for weight in entry.source.files:
        if weight.sha256 is None:
            reason = "pickle: true without a sha256 is a hard error" if entry.load.pickle else "no pinned sha256"
            raise WeightIntegrityError(f"Weight file {entry.id}/{weight.path}: {reason}")
        path = file_path(entry, weight)
        if not path.is_file():
            hint = (f"copy its files to {local_install_dir(entry)}" if entry.source.type == "local"
                    else f"run `python -m app.cli models fetch {entry.id}`")
            raise ModelNotInstalled(f"model not installed: {entry.id} ({weight.path} is missing; {hint})")
        if force or not record.is_fresh(path, weight.sha256):
            actual = sha256_file(path)
            if actual != weight.sha256:
                raise WeightIntegrityError(f"Weight file {entry.id}/{weight.path} failed integrity check "
                                           f"(sha256 {actual[:12]}… ≠ pinned {weight.sha256[:12]}…)")
            record.record(path, weight.sha256)
        paths.append(path)
    return paths


# --- streaming downloads -----------------------------------------------------------------------------------

def _looks_like_html(chunk: bytes) -> bool:
    return chunk.lstrip()[:15].lower().startswith((b"<!doctype html", b"<html"))


def _write_verified(chunks: Any, dest: Path, expected_sha256: str, *, expected_bytes: int | None,
                    reject_html: bool = False) -> Path:
    """Stream chunks to `<dest>.part`, hash incrementally, publish atomically only if the hash (and size) match."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    digest, n, first = hashlib.sha256(), 0, True
    try:
        with open(part, "wb") as f:
            for chunk in chunks:
                if first and reject_html and _looks_like_html(chunk):
                    raise FetchError(f"{dest.name}: the server returned an HTML page instead of the file")
                first = False
                n += len(chunk)
                if n > MAX_OBJECT_BYTES:
                    raise FetchError(f"{dest.name}: larger than the {MAX_OBJECT_BYTES // 1024**3} GB object cap")
                digest.update(chunk)
                f.write(chunk)
    except BaseException:
        part.unlink(missing_ok=True)
        raise
    actual = digest.hexdigest()
    if actual != expected_sha256 or (expected_bytes is not None and n != expected_bytes):
        part.unlink(missing_ok=True)
        size_note = "" if expected_bytes is None else f", {n} bytes vs pinned {expected_bytes}"
        raise WeightIntegrityError(f"Weight file {dest.name} failed integrity check (sha256 {actual[:12]}… ≠ pinned "
                                   f"{expected_sha256[:12]}…{size_note}); file removed")
    os.replace(part, dest)
    return dest


def _response_to_file(response: httpx.Response, dest: Path, expected_sha256: str, *, expected_bytes: int | None,
                      reject_html: bool) -> Path:
    if response.status_code >= 400:
        raise FetchError(f"{dest.name}: HTTP {response.status_code} from {urlparse(str(response.url)).netloc}")
    if reject_html and "text/html" in response.headers.get("content-type", "").lower():
        raise FetchError(f"{dest.name}: the server returned an HTML page instead of the file")
    return _write_verified(response.iter_bytes(CHUNK), dest, expected_sha256, expected_bytes=expected_bytes,
                           reject_html=reject_html)


def stream_to_file(url: str, dest: Path, expected_sha256: str, *, expected_bytes: int | None = None,
                   client: httpx.Client | None = None, params: dict | None = None, reject_html: bool = False) -> Path:
    require_online()
    if not expected_sha256:
        raise FetchError(f"refusing to download {dest.name} without a pinned sha256")
    own = client is None
    client = client or httpx.Client(follow_redirects=True, timeout=_TIMEOUT)
    try:
        with client.stream("GET", url, params=params) as response:
            return _response_to_file(response, dest, expected_sha256, expected_bytes=expected_bytes, reject_html=reject_html)
    except httpx.HTTPError as e:
        raise FetchError(f"{dest.name}: download failed ({type(e).__name__})") from e
    finally:
        if own:
            client.close()


_CONFIRM_RE = re.compile(r"confirm=([0-9A-Za-z_\-]+)")
_FORM_RE = re.compile(r'<form[^>]*action="([^"]+)"', re.IGNORECASE)
_HIDDEN_RE = re.compile(r'<input[^>]*type="hidden"[^>]*name="([^"]+)"[^>]*value="([^"]*)"', re.IGNORECASE)


def _drive_confirmation(page: str, cookies: httpx.Cookies, endpoint: str) -> tuple[str, dict[str, str]] | None:
    form = _FORM_RE.search(page)
    hidden = {k: html.unescape(v) for k, v in _HIDDEN_RE.findall(page)}
    if form and hidden.get("confirm"):
        return urljoin(endpoint, html.unescape(form.group(1))), hidden
    for name, value in cookies.items():
        if name.startswith("download_warning"):
            return endpoint, {"confirm": value}
    match = _CONFIRM_RE.search(html.unescape(page))
    return (endpoint, {"confirm": match.group(1)}) if match else None


def fetch_gdrive(file_id: str, dest: Path, expected_sha256: str, *, expected_bytes: int | None = None,
                 endpoint: str | None = None, client: httpx.Client | None = None) -> Path:
    """gdown-style fetch: follow the "can't scan for viruses" confirmation; never accept an HTML body as weights."""
    require_online()
    if not expected_sha256:
        raise FetchError(f"refusing to download {dest.name} from Google Drive without a pinned sha256")
    endpoint = endpoint or GDRIVE_ENDPOINT
    params = {"export": "download", "id": file_id}
    own = client is None
    client = client or httpx.Client(follow_redirects=True, timeout=_TIMEOUT)
    try:
        with client.stream("GET", endpoint, params=params) as response:
            if response.status_code >= 400 or "text/html" not in response.headers.get("content-type", "").lower():
                return _response_to_file(response, dest, expected_sha256, expected_bytes=expected_bytes, reject_html=True)
            page = response.read()[:500_000].decode("utf-8", "replace")
        confirmation = _drive_confirmation(page, client.cookies, endpoint)
        if confirmation is None:
            raise FetchError(f"{dest.name}: Google Drive returned an HTML page without a download confirmation token "
                             "(quota exceeded, or the file is not shared publicly)")
        action, form = confirmation
        with client.stream("GET", action, params={**params, **form}) as response:
            return _response_to_file(response, dest, expected_sha256, expected_bytes=expected_bytes, reject_html=True)
    except httpx.HTTPError as e:
        raise FetchError(f"{dest.name}: Google Drive download failed ({type(e).__name__})") from e
    finally:
        if own:
            client.close()


# --- fetch -------------------------------------------------------------------------------------------------

def _extract_archive(entry: ModelEntry, archive: Path, record: VerifyRecord) -> list[Path]:
    """Extract only the manifest-listed members; each is size-capped and re-verified."""
    members = entry.source.files[1:]
    paths = []

    def publish(src: IO[bytes], weight: WeightFile) -> None:
        dest = file_path(entry, weight)
        assert weight.sha256 is not None
        _write_verified(iter(lambda: src.read(CHUNK), b""), dest, weight.sha256, expected_bytes=weight.bytes)
        record.record(dest, weight.sha256)
        paths.append(dest)

    if entry.source.archive == "zip":
        with zipfile.ZipFile(archive) as zf:
            infos = {info.filename: info for info in zf.infolist()}
            for weight in members:
                info = infos.get(weight.path)
                if info is None or info.is_dir():
                    raise FetchError(f"{entry.id}: archive has no member {weight.path!r}")
                if info.file_size > MAX_OBJECT_BYTES:
                    raise FetchError(f"{entry.id}: member {weight.path!r} exceeds the object cap")
                with zf.open(info) as src:
                    publish(src, weight)
    else:
        with tarfile.open(archive) as tf:
            for weight in members:
                try:
                    member = tf.getmember(weight.path)
                except KeyError:
                    raise FetchError(f"{entry.id}: archive has no member {weight.path!r}") from None
                if not member.isfile() or member.size > MAX_OBJECT_BYTES:
                    raise FetchError(f"{entry.id}: member {weight.path!r} is not a regular file within the object cap")
                src = tf.extractfile(member)
                assert src is not None
                with src:
                    publish(src, weight)
    return paths


class Fetched(list[Path]):
    """The verified files of a model after `fetch_model`, and whether any of them was downloaded by that call. A model
    whose files were all there was only hashed: `models fetch` then says `INSTALLED`, not `FETCHED` (BUILD_SPEC N-14c,
    E-12: the log of a start could not tell a download from a check)."""

    def __init__(self, paths: Iterable[Path], *, downloaded: bool):
        super().__init__(paths)
        self.downloaded = downloaded


def _fetched(entry: ModelEntry, paths: Iterable[Path], *, downloaded: bool, **fields: Any) -> Fetched:
    log.info("weights.fetched" if downloaded else "weights.verified", model=entry.id, **fields)
    return Fetched(paths, downloaded=downloaded)


def fetch_model(entry: ModelEntry, *, client: httpx.Client | None = None, record: VerifyRecord | None = None) -> Fetched:
    """Fetch and verify every pinned file of a model. A file that is there with its pinned hash is not asked for again."""
    require_online()
    for weight in entry.source.files:
        _safe_rel(weight.path, entry.id)
        if weight.sha256 is None:
            reason = "pickle: true without a sha256 is a hard error" if entry.load.pickle else \
                "no pinned sha256 (run `models pin` after a manual fetch)"
            raise FetchError(f"{entry.id}/{weight.path}: {reason}")
    record = record or VerifyRecord()
    source = entry.source
    if source.type == "hf":
        from huggingface_hub import snapshot_download

        there = installed(entry)  # every pinned file is in the cache already: the hub client downloads nothing
        snapshot_download(repo_id=source.repo, revision=source.revision, allow_patterns=[f.path for f in source.files],
                          cache_dir=str(hf_cache_dir()))
        paths = ensure_verified(entry, record=record, force=True)
        return _fetched(entry, paths, downloaded=not there, files=len(paths))
    if source.type in ("url", "gdrive"):
        if not source.files or (not _is_archive(entry) and len(source.files) != 1):
            raise FetchError(f"{entry.id}: a {source.type} source must list exactly one file (plus members for archives)")
        first = source.files[0]
        assert first.sha256 is not None
        dest = file_path(entry, first)
        there = dest.is_file() and sha256_file(dest) == first.sha256
        if not there:
            if source.type == "url" or source.mirror_url:
                stream_to_file(source.mirror_url or source.url or "", dest, first.sha256, expected_bytes=first.bytes,
                               client=client)
            else:
                fetch_gdrive(source.file_id or "", dest, first.sha256, expected_bytes=first.bytes, client=client)
        record.record(dest, first.sha256)
        return _fetched(entry, [dest, *(_extract_archive(entry, dest, record) if _is_archive(entry) else [])],
                        downloaded=not there, file=first.path)
    if source.type == "torch_hub":
        if len(source.files) != 1:
            raise FetchError(f"{entry.id}: a torch_hub source lists exactly one file, the repository archive at `ref`")
        first = source.files[0]
        assert first.sha256 is not None
        dest = file_path(entry, first)
        there = dest.is_file() and sha256_file(dest) == first.sha256
        if not there:
            url = source.mirror_url or GITHUB_ARCHIVE.format(repo=source.repo, ref=source.ref)
            stream_to_file(url, dest, first.sha256, expected_bytes=first.bytes, client=client)
        record.record(dest, first.sha256)
        return _fetched(entry, [dest], downloaded=not there, file=first.path)
    if source.type == "local":
        raise FetchError(f"{entry.id} is produced locally by {source.produced_by}; it cannot be fetched "
                         f"(copy its files to {local_install_dir(entry)})")
    raise FetchError(f"{entry.id}: fetching {source.type} sources is not supported")


def _hub_members(entry: ModelEntry, archive: Path) -> dict[str, bytes]:
    """Regular files of a GitHub repository archive, keyed by their path below its single top-level directory."""
    members: dict[str, bytes] = {}
    prefix, total = None, 0
    with zipfile.ZipFile(archive) as zf:
        infos = [info for info in zf.infolist() if not info.is_dir()]
        if len(infos) > MAX_HUB_MEMBERS:
            raise FetchError(f"{entry.id}: the repository archive has more than {MAX_HUB_MEMBERS} files")
        for info in infos:
            rel = _safe_rel(info.filename, entry.id)
            if len(rel.parts) < 2 or rel.parts[0] != (prefix := prefix or rel.parts[0]):
                raise FetchError(f"{entry.id}: archive member {info.filename!r} is outside the repository directory")
            total += info.file_size
            if total > MAX_HUB_BYTES:
                raise FetchError(f"{entry.id}: the repository archive expands beyond {MAX_HUB_BYTES // 1024**2} MB")
            members[PurePosixPath(*rel.parts[1:]).as_posix()] = zf.read(info)
    return members


def _tree_matches(root: Path, members: dict[str, bytes]) -> bool:
    if not root.is_dir():
        return False
    files = {p.relative_to(root).as_posix(): p for p in root.rglob("*") if p.is_file() or p.is_symlink()}
    return files.keys() == members.keys() and all(not p.is_symlink() and p.read_bytes() == members[rel]
                                                   for rel, p in files.items())


def torch_hub_dir(entry: ModelEntry, *, record: VerifyRecord | None = None) -> Path:
    """The local repository of a torch_hub model, for `torch.hub.load(<dir>, entrypoint, source="local")`.

    The pinned archive is verified first (`ModelNotInstalled` when absent). Its files are extracted to
    `weights/<model id>/hub/<owner>_<repo>_<ref>`, and before every load the tree is compared with the archive (names and
    bytes) and rebuilt on any difference, so an edited or added file never reaches the import.
    """
    source = entry.source
    if source.type != "torch_hub":
        raise FetchError(f"{entry.id}: not a torch_hub source")
    members = _hub_members(entry, ensure_verified(entry, record=record)[0])
    owner, _, name = (source.repo or "").partition("/")
    root = settings.weights_dir / entry.id / "hub" / f"{owner}_{name}_{source.ref}"
    if _tree_matches(root, members):
        return root
    staging = root.with_name(f"{root.name}.staging-{secrets.token_hex(4)}")
    try:
        for rel, data in members.items():
            path = staging / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        shutil.rmtree(root, ignore_errors=True)
        os.replace(staging, root)
    except OSError:
        shutil.rmtree(staging, ignore_errors=True)
        if not _tree_matches(root, members):  # another process may have published the same tree first
            raise
    log.info("weights.hub_extracted", model=entry.id, files=len(members))
    return root


# --- pin ---------------------------------------------------------------------------------------------------

PinResolver = Callable[[ModelEntry], tuple[str | None, dict[str, tuple[str, int]]]]


def hf_pin_resolver(entry: ModelEntry) -> tuple[str | None, dict[str, tuple[str, int]]]:
    """Resolve the current commit and per-file sha256 (LFS metadata, or download and hash small files)."""
    require_online()
    from huggingface_hub import HfApi, hf_hub_download

    source = entry.source
    info = HfApi().model_info(source.repo or "", revision=source.revision or "main", files_metadata=True)
    siblings = {s.rfilename: s for s in info.siblings or []}
    hashes: dict[str, tuple[str, int]] = {}
    for weight in source.files:
        sibling = siblings.get(weight.path)
        if sibling is None:
            raise FetchError(f"{entry.id}: {weight.path} is not in {source.repo}@{info.sha}")
        if sibling.lfs is not None and sibling.lfs.sha256:
            hashes[weight.path] = (sibling.lfs.sha256, int(sibling.lfs.size or sibling.size or 0))
        else:
            local = Path(hf_hub_download(source.repo or "", weight.path, revision=info.sha, cache_dir=str(hf_cache_dir())))
            hashes[weight.path] = (sha256_file(local), local.stat().st_size)
    return info.sha, hashes


def local_pin_resolver(entry: ModelEntry) -> tuple[str | None, dict[str, tuple[str, int]]]:
    """Hash files that are already installed (url, gdrive and local sources after a manual fetch)."""
    hashes = {}
    for weight in entry.source.files:
        path = file_path(entry, weight)
        if not path.is_file():
            raise ModelNotInstalled(f"model not installed: {entry.id} ({weight.path} is missing)")
        hashes[weight.path] = (sha256_file(path), path.stat().st_size)
    return None, hashes


_ID_LINE = re.compile(r"^  - id: (\S+)\s*$")


def pin_text(text: str, model_id: str, revision: str | None, hashes: dict[str, tuple[str, int]]) -> str:
    """Rewrite `revision` and each listed file's sha256/bytes inside one model block, leaving everything else intact."""
    lines = text.splitlines(keepends=True)
    start = next((i for i, line in enumerate(lines) if (m := _ID_LINE.match(line)) and m.group(1) == model_id), None)
    if start is None:
        raise KeyError(f"model {model_id!r} is not in the manifest")
    end = next((i for i in range(start + 1, len(lines))
                if _ID_LINE.match(lines[i]) or (lines[i].strip() and not lines[i].startswith((" ", "#")))), len(lines))
    for i in range(start, end):
        line = lines[i]
        if revision is not None and re.match(r"^\s+revision:\s", line):
            # quote values YAML would not read back as strings (e.g. an all-digit commit sha)
            quoted = revision if isinstance(yaml.safe_load(revision), str) else f'"{revision}"'
            line = re.sub(r"(revision:\s*)\S+", lambda m, q=quoted: m.group(1) + q, line, count=1)
        match = re.search(r"path:\s*([^,}\s]+)", line)
        if match and match.group(1) in hashes:
            sha, size = hashes[match.group(1)]
            sha_text = sha if isinstance(yaml.safe_load(sha), str) else f'"{sha}"'
            for key, value in (("sha256", sha_text), ("bytes", str(size))):
                if re.search(rf"{key}:\s", line):
                    line = re.sub(rf"({key}:\s*)[^,}}\s]+", lambda m, v=value: m.group(1) + v, line, count=1)
                else:
                    line = line.replace(match.group(0), f"{match.group(0)}, {key}: {value}", 1)
        lines[i] = line
    return "".join(lines)


def pin_manifest(path: Path, model_id: str, *, resolver: PinResolver | None = None, write: bool = False) -> str:
    """Return the unified diff of pinning `model_id`; write it only when `write` is true."""
    entry = load_manifest(path).model(model_id)
    if entry is None:
        raise KeyError(f"model {model_id!r} is not in the manifest")
    resolver = resolver or (hf_pin_resolver if entry.source.type == "hf" else local_pin_resolver)
    revision, hashes = resolver(entry)
    old = path.read_text(encoding="utf-8")
    new = pin_text(old, model_id, revision, hashes)
    parse_manifest(yaml.safe_load(new), source=f"{path} (pinned)")  # the rewritten manifest must still validate
    from app.core.textdiff import unified_diff  # minimal diff: repeated manifest lines stay anchors

    diff = unified_diff(old.splitlines(keepends=True), new.splitlines(keepends=True), fromfile=str(path),
                        tofile=f"{path} (pinned)")
    if write and new != old:
        path.write_text(new, encoding="utf-8")
    return diff
