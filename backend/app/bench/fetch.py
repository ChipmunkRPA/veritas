"""Sampling fetcher for Veritas-Bench (ML §2.9–2.10; D-05, D-13).

The benchmark is built by sampling, never by bulk download:
- `parquet_stream`: a seeded random subset of shard files is opened with HTTP range reads. The metadata columns of
  every row group are read first; media columns are read only for row groups holding kept rows (the metadata filter
  before decode, F2). Row groups are visited greedily by how many rows the open quotas would admit, with a seeded
  shuffle breaking ties, so rare generators do not spend the byte cap on groups that yield nothing.
- `per_file`: the repo tree is listed, paths are parsed with `path_regex`, candidates are shuffled (seeded) and
  downloaded one by one within quotas. With `extract: tar`, small archives are downloaded whole (≤ 5 GB) and selected
  members extracted.
- `zip_range`: the zip central directory is read over range requests and only selected members are fetched. When a
  server ignores Range the archive is skipped, never bulk-downloaded.
- `api`: Wikimedia Commons, paced, with the magic-byte check.
- `url_list`: a plain-HTTP dataset published as an index of file URLs (VISION). The index is pinned by sha256; candidates
  alternate over the values of one path field (device-diverse samples) and within a value follow their group key, so
  two sources sharing a group-key namespace (natives and their platform copies) visit groups in the same order.
- `excluded_bulk`: listed in the plan and report, never contacted.

Caps (F3): per source `max_items`, `max_bytes` (bytes transferred, including range reads of footers and central
directories) and optionally `max_per_group` items per group key; per object 5 GB for whole downloads; total cache
(native + norm bytes on disk) 55 GB (D-40); a free-disk floor; and an optional time box per run (D-40 Commons top-up).

Resumable (F4): every kept item, durable rejection and finished archive is appended to `state/<source_id>.jsonl`
together with the bytes transferred since the previous event. A re-run replays the log, refills quotas and skips known
row keys. `manifest.parquet` is rebuilt from all state logs after each run.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import random
import re
import shutil
import tarfile
import time
import zipfile
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from fnmatch import fnmatch
from pathlib import Path
from typing import Any

from app.bench import media
from app.bench.remote import (
    ByteCapReached,
    ByteMeter,
    Http,
    HubClient,
    Meters,
    RangeFile,
    RangeUnsupported,
    RemoteError,
    RemoteFile,
)
from app.bench.spec import HEX40, HF_MODES, Bucket, GroupKeyRule, Source, Spec
from app.core.logging import get_logger

log = get_logger(__name__)

MAX_CONSECUTIVE_FAILURES = 10
# pyarrow and zipfile issue exact reads (footer, column chunks, whole members); the block only coalesces tiny reads
# such as zip local headers, so the byte caps charge close to what is actually needed.
PARQUET_BLOCK = 64 << 10
ZIP_BLOCK = 64 << 10
_TAG = re.compile(r"<[^>]+>")


class SourceError(RuntimeError):
    """A source does not match its spec (missing columns, unknown revision)."""


class SourceSkipped(RuntimeError):
    """A source is not fetched (gated, private, unpinned, excluded)."""


class DuplicateItem(RuntimeError):
    """The bytes are already in the cache under another source or row."""


# --- manifest ----------------------------------------------------------------------------------------------

def manifest_schema():
    import pyarrow as pa

    s, i, f, b = pa.string(), pa.int64(), pa.float64(), pa.bool_()
    return pa.schema([
        ("item_id", s), ("modality", s), ("label", s), ("threat", s), ("generator", s), ("family", s), ("lgo_fold", s),
        ("release_date", s), ("source_dataset", s), ("source_id", s), ("source_revision", s), ("source_row_key", s),
        ("group_key", s), ("licence", s), ("commercial_ok", b), ("role", s), ("split", s), ("real_stratum", s),
        ("width", i), ("height", i), ("jpeg_q_est", i), ("has_exif", b), ("has_c2pa", b), ("face_px_max", i),
        ("duration_s", f), ("native_path", s), ("norm_path", s), ("audit_ok", b), ("native_bytes", i), ("norm_bytes", i),
        ("fetched_at", s),
    ])


def manifest_rows(vbench_dir: Path) -> list[dict]:
    rows = []
    for path in sorted((vbench_dir / "state").glob("*.jsonl")):
        rows += [event["row"] for event in StateLog(path).items.values()]
    return sorted(rows, key=lambda r: (r["source_id"], r["item_id"]))


def write_manifest(vbench_dir: Path) -> Path:
    import pyarrow as pa
    import pyarrow.parquet as pq

    schema = manifest_schema()
    rows = manifest_rows(vbench_dir)
    table = pa.Table.from_pylist([{name: row.get(name) for name in schema.names} for row in rows], schema=schema)
    out = vbench_dir / "manifest.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".part")
    pq.write_table(table, tmp)
    os.replace(tmp, out)
    return out


# --- state -------------------------------------------------------------------------------------------------

class StateLog:
    """Append-only JSONL per source. Events: `item` (manifest row + bucket signature), `reject` (durable),
    `archive` (a whole-archive source finished), `outcome` (a fetching run's final status and reason) and `bytes`.
    Each event carries the bytes transferred since the previous event."""

    def __init__(self, path: Path):
        self.path = path
        self.items: dict[str, dict] = {}
        self.rejected: dict[str, str] = {}
        self.archives: set[str] = set()
        self.outcome: dict | None = None
        self.bytes_used = 0
        self._torn = False
        if path.is_file():
            text = path.read_text(encoding="utf-8")
            self._torn = bool(text) and not text.endswith("\n")
            for line in text.splitlines():
                try:
                    event = json.loads(line)
                except ValueError:
                    continue  # a torn last line after a crash
                self.bytes_used += int(event.get("bytes", 0))
                self._apply(event)
        self._logged = self.bytes_used

    def _apply(self, event: dict) -> None:
        kind = event.get("t")
        if kind == "item":
            key = event["row"]["source_row_key"]
            held = self.items.get(key)
            if held is not None and held["row"]["item_id"] != event["row"]["item_id"]:
                key = f"{key}#{event['row']['item_id'][:16]}"  # a non-unique dataset id (two images, one id): keep both
            self.items[key] = event
        elif kind == "reject":
            self.rejected[event["row_key"]] = event["reason"]
        elif kind == "archive":
            self.archives.add(event["path"])
        elif kind == "outcome":
            self.outcome = event

    def seen(self, row_key: str) -> bool:
        return row_key in self.items or row_key in self.rejected

    def append(self, event: dict, bytes_used: int) -> None:
        event = {**event, "bytes": bytes_used - self._logged}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            if self._torn:
                f.write("\n")
                self._torn = False
            f.write(json.dumps(event, sort_keys=True, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        self._logged = bytes_used
        self._apply(event)

    def flush_bytes(self, bytes_used: int) -> None:
        if bytes_used != self._logged:
            self.append({"t": "bytes"}, bytes_used)


# --- quotas and group keys ---------------------------------------------------------------------------------

def bucket_signature(bucket: Bucket) -> str:
    return json.dumps(bucket.model_dump(exclude={"n"}), sort_keys=True)


def describe_bucket(bucket: Bucket) -> str:
    parts = [bucket.label, bucket.threat]
    parts += [p for p in (bucket.lgo_fold, bucket.family, bucket.real_stratum, bucket.generator) if p]
    match = ", ".join(f"{k}={'|'.join(v) if isinstance(v, list) else v}" for k, v in sorted(bucket.match.items()))
    return " · ".join(parts) + (f" [{match}]" if match else "")


class Quotas:
    """Bucket fill counts; a candidate takes the first open bucket it matches, subject to `max_items`."""

    def __init__(self, buckets: list[Bucket], max_items: int, existing: Iterable[str] = ()):
        self.buckets, self.max_items = buckets, max_items
        self.filled = [0] * len(buckets)
        self.total = 0
        index = {bucket_signature(b): i for i, b in enumerate(buckets)}
        for signature in existing:
            self.total += 1
            if (i := index.get(signature)) is not None:
                self.filled[i] += 1

    def reserve(self, meta: dict[str, str]) -> int | None:
        if self.total >= self.max_items:
            return None
        for i, bucket in enumerate(self.buckets):
            if self.filled[i] < bucket.n and bucket.matches(meta):
                self.filled[i] += 1
                self.total += 1
                return i
        return None

    def release(self, i: int) -> None:
        self.filled[i] -= 1
        self.total -= 1

    def open_slots(self) -> list[int]:
        return [max(0, b.n - f) for b, f in zip(self.buckets, self.filled)]

    @property
    def done(self) -> bool:
        return self.total >= self.max_items or all(f >= b.n for f, b in zip(self.filled, self.buckets))


def normalise_text(value: str) -> str:
    return " ".join(value.split()).lower()


def group_key(rule: GroupKeyRule, meta: dict[str, str], row_key: str, dataset: str) -> str:
    value = meta.get(rule.field)
    if value and rule.regex:
        match = re.search(rule.regex, value)
        if match is None:
            value = None
        elif "key" in match.re.groupindex:
            value = match.group("key")
        else:  # all captured groups joined, e.g. (subset)/(index) for matched triplets
            value = "/".join(g for g in match.groups() if g is not None) or match.group(0)
    if not value and rule.fallback:
        value = meta.get(rule.fallback)
    value = f"row:{row_key}" if not value else (normalise_text(value) if rule.normalise == "text" else value)
    return f"{rule.namespace or dataset}:{hashlib.sha256(value.encode('utf-8')).hexdigest()[:16]}"


def _s(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _meta(*sources: dict[str, Any]) -> dict[str, str]:
    merged: dict[str, str] = {}
    for source in sources:
        merged.update({k: s for k, v in source.items() if (s := _s(v)) is not None})
    return merged


# --- candidates and batches --------------------------------------------------------------------------------

@dataclass
class Candidate:
    row_key: str
    meta: dict[str, str]
    ref: Any = None
    size: int | None = None  # known transfer size (per_file, zip member, Commons), for the cap pre-check


@dataclass
class Batch:
    candidates: list[Candidate]
    fetch: Callable[[list[int]], Iterator[bytes | Exception]]  # bodies (or per-item errors) for kept indices, in order


def select_files(files: list[RemoteFile], patterns: list[str], sample: int | None,
                 rng: random.Random) -> tuple[list[RemoteFile], int]:
    matched = [f for f in files if any(fnmatch(f.path, p) for p in patterns)]
    if sample is not None and len(matched) > sample:
        return sorted(rng.sample(matched, sample), key=lambda f: f.path), len(matched)
    return matched, len(matched)


def _single(fn: Callable[[], bytes]) -> Iterator[bytes]:
    yield fn()


def _media_bytes(value: Any) -> bytes | Exception:
    if isinstance(value, dict):
        value = value.get("bytes")
    if isinstance(value, bytes | bytearray) and value:
        return bytes(value)
    return media.MediaError("the media cell holds no bytes (a path reference or empty)")


def _strip_html(value: str) -> str:
    return " ".join(_TAG.sub(" ", value).split())


def commons_licence_ok(licence: str) -> bool:
    return bool(licence) and "nc" not in re.split(r"[\s\-_/]+", licence.lower())


# --- context and report ------------------------------------------------------------------------------------

class TimeBoxReached(RuntimeError):
    """The run's time box expired (D-40: the Commons top-up is time-boxed, then frozen)."""


class DiskFloorReached(RuntimeError):
    """The cache volume has less free space than `limits.min_free_disk_bytes`."""


@dataclass
class FetchContext:
    spec: Spec
    vbench_dir: Path
    http: Http
    hub: HubClient
    cache: ByteMeter | None = None
    known_items: dict[str, str] = field(default_factory=dict)
    cached_only: bool = False  # rebuild the report and manifest from the state logs; no source is contacted (D-38)
    deadline: float | None = None  # a `clock()` value after which no further candidate is fetched (the run's time box)
    exclude_groups: frozenset[str] = frozenset()  # group keys of another bench (field sets hold them out, H-9)
    clock: Callable[[], float] = time.monotonic
    free_disk: Callable[[Path], int] = lambda path: shutil.disk_usage(path).free
    now: Callable[[], str] = lambda: datetime.now(UTC).isoformat(timespec="seconds")

    def __post_init__(self) -> None:
        if self.cache is None:
            used = 0
            for row in manifest_rows(self.vbench_dir):
                used += int(row.get("native_bytes") or 0) + int(row.get("norm_bytes") or 0)
                self.known_items[row["item_id"]] = f"{row['source_id']}/{row['source_row_key']}"
            self.cache = ByteMeter("total cache", self.spec.limits.max_total_bytes, used=used)

    def http_for(self, source: Source) -> Http:
        if source.pace_s <= 0:
            return self.http
        return Http(self.http.client, pace_s=source.pace_s, retries=self.http.retries, retries_429=self.http.retries_429,
                    max_backoff_s=self.http.max_backoff_s, sleep=self.http.sleep)


@dataclass
class SourceReport:
    source: Source
    status: str = "complete"  # complete | shortfall | skipped | error
    reason: str | None = None
    kept_before: int = 0
    kept: int = 0
    bytes_used: int = 0
    byte_cap: int = 0
    filled: list[int] = field(default_factory=list)
    rejects: Counter = field(default_factory=Counter)
    notes: list[str] = field(default_factory=list)
    total_cap_hit: bool = False
    stop_run: str | None = None  # the remaining sources of the run are skipped with this reason (cache cap, time box, disk)


# --- one source --------------------------------------------------------------------------------------------

class SourceRun:
    def __init__(self, ctx: FetchContext, source: Source, report: SourceReport):
        self.ctx, self.source, self.report = ctx, source, report
        self.state = StateLog(ctx.vbench_dir / "state" / f"{source.id}.jsonl")
        self.quotas = Quotas(source.quotas, source.max_items, (e["bucket"] for e in self.state.items.values()))
        self.meter = ByteMeter(f"source {source.id}", report.byte_cap, used=self.state.bytes_used)
        self.http = ctx.http_for(source)
        self.rng = random.Random(f"{ctx.spec.seed}:{source.id}")
        self.group_counts = Counter(e["row"]["group_key"] for e in self.state.items.values())
        report.kept_before = self.quotas.total

    # modes -----------------------------------------------------------------------------------------------

    def batches(self) -> Iterator[Batch]:
        return {"parquet_stream": self._parquet, "per_file": self._per_file, "zip_range": self._zip,
                "api": self._api, "url_list": self._url_list}[self.source.mode]()

    def _url_list(self) -> Iterator[Batch]:
        cfg = self.source.url_list
        assert cfg is not None and self.source.group_key is not None
        index = self.http.download(cfg.index, Meters(self.meter), max_bytes=cfg.max_index_bytes)
        digest = hashlib.sha256(index).hexdigest()
        if digest != cfg.index_sha256:
            raise SourceError(f"{cfg.index}: index sha256 {digest} does not match the pinned {cfg.index_sha256}")
        pattern = re.compile(self.source.path_regex or "")
        lanes: dict[str, list[tuple[str, str, Candidate]]] = defaultdict(list)
        for line in index.decode("utf-8", errors="replace").splitlines():
            url = line.strip()
            if not url.startswith(cfg.base):
                continue
            path = url[len(cfg.base):]
            match = pattern.search(path)
            if match is None:
                continue
            meta = _meta(self.source.constants, match.groupdict(), {"path": path})
            key = group_key(self.source.group_key, meta, path, self.source.dataset)
            lanes[meta.get(cfg.interleave, "") if cfg.interleave else ""].append((key, path, Candidate(path, meta, ref=url)))
        ordered = [[c for _, _, c in sorted(members, key=lambda m: (m[0], m[1]))] for _, members in sorted(lanes.items())]
        self.report.notes.append(f"{sum(len(lane) for lane in ordered)} indexed files matched in {len(ordered)} lanes")
        limit = self.ctx.spec.limits.max_object_bytes
        for c in _round_robin(ordered):
            yield Batch([c], lambda _i, c=c, limit=limit: _single(
                lambda: self.http.download(c.ref, Meters(self.meter), max_bytes=limit)))

    def _tree(self) -> list[RemoteFile]:
        assert self.source.files is not None and self.source.revision is not None
        files, matched = select_files(self.ctx.hub.tree(self.source.dataset, self.source.revision),
                                      self.source.files.pattern, self.source.files.sample, self.rng)
        self.report.notes.append(f"{len(files)} of {matched} matching files selected")
        return files

    def _url(self, path: str) -> str:
        assert self.source.revision is not None
        return self.ctx.hub.resolve_url(self.source.dataset, self.source.revision, path)

    def _parquet(self) -> Iterator[Batch]:
        import pyarrow.parquet as pq

        cfg = self.source.parquet
        assert cfg is not None
        meta_columns = sorted(set(cfg.columns.values()) | {c for m in cfg.media for c in m.meta.values()}
                              | ({cfg.row_key} if cfg.row_key else set()))
        needed = set(meta_columns) | {m.column for m in cfg.media}
        groups: list[tuple[Any, int, list[Candidate]]] = []
        for remote in self._tree():
            handle = RangeFile(self.http, self._url(remote.path), remote.size, Meters(self.meter), block=PARQUET_BLOCK)
            try:
                parquet = pq.ParquetFile(handle)
            except ByteCapReached:
                raise
            except Exception as e:  # noqa: BLE001 - pyarrow raises many types for a bad footer
                raise _unwrap(e) or SourceError(f"{remote.path}: not a readable parquet file ({type(e).__name__})") from None
            missing = needed - set(parquet.schema_arrow.names)
            if missing:
                raise SourceError(f"{remote.path}: columns {sorted(missing)} are not in the shard schema")
            for group in range(parquet.num_row_groups):  # metadata columns only, before any media is read
                groups.append((parquet, group, self._parquet_candidates(remote, parquet, group, meta_columns)))
        self.rng.shuffle(groups)
        counts = [self._bucket_counts(candidates) for _, _, candidates in groups]
        pending = list(range(len(groups)))
        self.report.notes.append(f"{len(groups)} row groups indexed")
        while pending and not self.quotas.done:
            slots = self.quotas.open_slots()
            yields = [sum(min(c, s) for c, s in zip(counts[i], slots)) for i in pending]
            best = max(range(len(pending)), key=yields.__getitem__)  # first maximum: the seeded shuffle breaks ties
            if yields[best] == 0:
                return
            parquet, group, candidates = groups[pending.pop(best)]
            yield Batch(candidates, lambda indices, p=parquet, g=group, c=candidates: _parquet_media(p, g, c, indices))

    def _parquet_candidates(self, remote: RemoteFile, parquet: Any, group: int, meta_columns: list[str]) -> list[Candidate]:
        cfg = self.source.parquet
        assert cfg is not None
        num_rows = parquet.metadata.row_group(group).num_rows
        rows = _read_group(parquet, group, meta_columns) if meta_columns else [{}] * num_rows
        order = list(range(num_rows))
        self.rng.shuffle(order)
        candidates = []
        for i in order:
            row = rows[i]
            base = _meta(self.source.constants, {k: row.get(c) for k, c in cfg.columns.items()})
            key = (_s(row.get(cfg.row_key)) if cfg.row_key else None) or f"{remote.path}#{group}:{i}"
            for column in cfg.media:
                meta = {**base, **_meta({k: row.get(c) for k, c in column.meta.items()})}
                row_key = key if len(cfg.media) == 1 else f"{key}/{column.column}"
                candidates.append(Candidate(row_key, meta, ref=(i, column.column)))
        return candidates

    def _bucket_counts(self, candidates: list[Candidate]) -> list[int]:
        """Unseen candidates per first matching bucket (the greedy row-group order's yield estimate)."""
        counts = [0] * len(self.quotas.buckets)
        for candidate in candidates:
            if self.state.seen(candidate.row_key):
                continue
            for i, bucket in enumerate(self.quotas.buckets):
                if bucket.matches(candidate.meta):
                    counts[i] += 1
                    break
        return counts

    def _per_file(self) -> Iterator[Batch]:
        assert self.source.files is not None
        if self.source.files.extract == "tar":
            yield from self._tar_members()
            return
        pattern = re.compile(self.source.path_regex or "")
        candidates = []
        for remote in self._tree():
            match = pattern.search(remote.path)
            if match is None:
                continue
            meta = _meta(self.source.constants, match.groupdict(), {"path": remote.path})
            candidates.append(Candidate(remote.path, meta, ref=remote.path, size=remote.size))
        self.rng.shuffle(candidates)
        limit = self.ctx.spec.limits.max_object_bytes
        for c in candidates:
            yield Batch([c], lambda _i, c=c, limit=limit: _single(
                lambda: self.http.download(self._url(c.ref), Meters(self.meter), max_bytes=limit)))

    def _tar_members(self) -> Iterator[Batch]:
        files = self.source.files
        assert files is not None
        pattern = re.compile(self.source.path_regex or "")
        limit = self.ctx.spec.limits.max_object_bytes
        archives = self._tree()
        self.rng.shuffle(archives)
        for remote in archives:
            if remote.path in self.state.archives:
                continue
            if remote.size > limit:
                self.report.rejects["archive larger than the per-object cap"] += 1
                continue
            if self.meter.would_exceed(remote.size):
                raise ByteCapReached(self.meter.scope, self.meter.cap or 0)
            try:
                body = self.http.download(self._url(remote.path), Meters(self.meter), max_bytes=limit)
            except RemoteError as e:
                self.report.notes.append(f"{remote.path}: skipped ({e})")
                self.report.rejects["archive skipped: download or read failed"] += 1
                continue
            try:
                with tarfile.open(fileobj=io.BytesIO(body), mode="r:*") as archive:
                    candidates = []
                    for member in archive.getmembers():
                        name = f"{remote.path}/{member.name}"
                        match = pattern.search(name) if member.isfile() else None
                        if match is not None:
                            meta = _meta(self.source.constants, match.groupdict(), {"path": name})
                            candidates.append(Candidate(name, meta, ref=member))
                    self.rng.shuffle(candidates)
                    if files.members_per_archive:
                        candidates = candidates[:files.members_per_archive]
                    for c in candidates:
                        yield Batch([c], lambda _i, c=c, a=archive, limit=limit: _single(
                            lambda: _read_tar_member(a, c.ref, limit)))
                    self.state.append({"t": "archive", "path": remote.path}, self.meter.used)
            except (tarfile.TarError, OSError, EOFError) as e:
                self.report.notes.append(f"{remote.path}: skipped ({e})")
                self.report.rejects["archive skipped: download or read failed"] += 1

    def _zip(self) -> Iterator[Batch]:
        files = self.source.files
        assert files is not None
        pattern = re.compile(self.source.path_regex or "")
        limit = self.ctx.spec.limits.max_object_bytes
        for remote in self._tree():
            handle = RangeFile(self.http, self._url(remote.path), remote.size, Meters(self.meter), block=ZIP_BLOCK)
            try:
                archive = zipfile.ZipFile(handle)
            except ByteCapReached:
                raise
            except (RangeUnsupported, RemoteError, zipfile.BadZipFile, OSError, ValueError) as e:
                self.report.notes.append(f"{remote.path}: skipped, range read failed ({e})")
                self.report.rejects["archive skipped: range read failed"] += 1
                continue
            candidates = []
            for info in archive.infolist():
                if info.is_dir():
                    continue
                name = f"{remote.path}/{info.filename}"
                match = pattern.search(name)
                if match is None:
                    continue
                meta = _meta(self.source.constants, match.groupdict(), {"path": name})
                candidates.append(Candidate(name, meta, ref=info, size=info.compress_size))
            self.rng.shuffle(candidates)
            if files.members_per_archive:
                candidates = candidates[:files.members_per_archive]
            for c in candidates:
                yield Batch([c], lambda _i, c=c, a=archive, limit=limit: _single(lambda: _read_member(a, c.ref, limit)))

    def _api(self) -> Iterator[Batch]:
        api = self.source.api
        assert api is not None
        base = {"action": "query", "format": "json", "formatversion": "2", "prop": "imageinfo",
                "iiprop": "url|size|mime|extmetadata|metadata", "iimetadataversion": "latest"}
        limit = min(api.max_file_bytes, self.ctx.spec.limits.max_object_bytes)
        for query in api.queries:
            if query.random:
                params = {**base, "generator": "random", "grnnamespace": "6", "grnlimit": "50"}
            elif query.search is not None:
                params = {**base, "generator": "search", "gsrsearch": query.search, "gsrnamespace": "6", "gsrlimit": "50"}
            else:
                params = {**base, "generator": "categorymembers", "gcmtitle": query.category, "gcmtype": "file",
                          "gcmlimit": "50"}
            candidates: dict[str, Candidate] = {}
            cont: dict = {}
            for _ in range(query.max_requests):
                document, _headers = self.http.get_json(api.endpoint, params={**params, **cont})
                document = document if isinstance(document, dict) else {}
                for page in (document.get("query") or {}).get("pages") or []:
                    candidate = self._commons_candidate(page, query.meta)
                    if candidate is not None:
                        candidates.setdefault(candidate.row_key, candidate)
                if len(candidates) >= query.max_candidates:
                    break
                cont = document.get("continue") or {}
                if not cont and not query.random:
                    break
            ordered = sorted(candidates.values(), key=lambda c: c.row_key)
            self.rng.shuffle(ordered)
            self.report.notes.append(f"{query.label}: {len(ordered)} candidates after filters")
            for c in ordered:
                yield Batch([c], lambda _i, c=c, limit=limit: _single(
                    lambda: self.http.download(c.ref, Meters(self.meter), max_bytes=limit)))

    def _commons_candidate(self, page: dict, constants: dict[str, str]) -> Candidate | None:
        api = self.source.api
        assert api is not None
        infos = page.get("imageinfo") or []
        if not infos or "pageid" not in page:
            return None
        info = infos[0]
        size = int(info.get("size") or 0)
        if info.get("mime") not in api.mime or not info.get("url") or size > api.max_file_bytes:
            return None
        exif = {m.get("name"): m.get("value") for m in info.get("metadata") or [] if isinstance(m, dict)}
        make = str(exif.get("Make") or "").strip()
        if api.exif_make is not None and not any(make.lower().startswith(p.lower()) for p in api.exif_make):
            return None
        ext = info.get("extmetadata") or {}
        licence = _strip_html(str((ext.get("LicenseShortName") or {}).get("value") or ""))
        meta = _meta(self.source.constants, constants, {
            "pageid": page["pageid"], "title": page.get("title"), "make": make, "model": str(exif.get("Model") or "").strip(),
            "artist": _strip_html(str((ext.get("Artist") or {}).get("value") or "")), "licence": licence or "unknown",
            "commercial_ok": commons_licence_ok(licence)})
        return Candidate(f"commons:{page['pageid']}", meta, ref=info["url"], size=size)

    # engine ----------------------------------------------------------------------------------------------

    def consume(self, batches: Iterator[Batch]) -> None:
        limits = self.ctx.spec.limits
        failures = 0
        for batch in batches:
            if self.ctx.deadline is not None and self.ctx.clock() >= self.ctx.deadline:
                raise TimeBoxReached("time box reached; the run stopped fetching new items")
            keep: list[tuple[int, Candidate, int]] = []
            for j, candidate in enumerate(batch.candidates):
                if self.state.seen(candidate.row_key):
                    continue
                if candidate.size is not None and candidate.size > limits.max_object_bytes:
                    self._reject(candidate, "object larger than the per-object cap", durable=True)
                    continue
                if self.source.max_per_group is not None and self.group_counts[self._group_key(candidate)] >= \
                        self.source.max_per_group:
                    self.report.rejects["group at max_per_group"] += 1  # not durable: the cap may be raised later
                    continue
                if self.ctx.exclude_groups and self._group_key(candidate) in self.ctx.exclude_groups:
                    self.report.rejects["group key held by the excluded bench"] += 1
                    continue
                bucket = self.quotas.reserve(candidate.meta)
                if bucket is None:
                    continue
                if candidate.size is not None and self.meter.would_exceed(candidate.size):
                    self.quotas.release(bucket)
                    self._release(keep)
                    raise ByteCapReached(self.meter.scope, self.meter.cap or 0)
                keep.append((j, candidate, bucket))
            if keep:
                failures = self._fetch_batch(batch, keep, failures)
            if self.quotas.done:
                return

    def _fetch_batch(self, batch: Batch, keep: list[tuple[int, Candidate, int]], failures: int) -> int:
        bodies = batch.fetch([j for j, _, _ in keep])
        for position, (_j, candidate, bucket) in enumerate(keep):
            try:
                body = next(bodies)
            except StopIteration:
                self._release(keep[position:])
                break
            except (RemoteError, SourceError) as e:
                self._release(keep[position:])
                self.report.rejects["download failed"] += len(keep) - position
                log.warning("vbench.download_failed", source=self.source.id, row=candidate.row_key, error=str(e))
                failures += 1
                if failures >= MAX_CONSECUTIVE_FAILURES:
                    raise RemoteError(f"{MAX_CONSECUTIVE_FAILURES} consecutive download failures; last: {e}") from None
                break
            except ByteCapReached:
                self._release(keep[position:])
                raise
            except Exception as e:  # noqa: BLE001 - pyarrow wraps Python exceptions raised by the range file
                self._release(keep[position:])
                raise _unwrap(e) or e from None
            if isinstance(body, Exception):
                self.quotas.release(bucket)
                self._reject(candidate, str(body), durable=True)
                continue
            try:
                self._store(candidate, bucket, body)
                failures = 0
            except (media.MediaError, DuplicateItem) as e:
                self.quotas.release(bucket)
                self._reject(candidate, str(e), durable=True)
            except (ByteCapReached, DiskFloorReached):
                self._release(keep[position:])  # the run ends: reserved but unstored items are not counted as kept
                raise
        return failures

    def _release(self, rest: list[tuple[int, Candidate, int]]) -> None:
        for _, _, bucket in rest:
            self.quotas.release(bucket)

    def _reject(self, candidate: Candidate, reason: str, *, durable: bool) -> None:
        self.report.rejects[_reason_class(reason)] += 1
        if durable:
            self.state.append({"t": "reject", "row_key": candidate.row_key, "reason": reason}, self.meter.used)

    def _group_key(self, candidate: Candidate) -> str:
        assert self.source.group_key is not None
        return group_key(self.source.group_key, candidate.meta, candidate.row_key, self.source.dataset)

    def _store(self, candidate: Candidate, bucket_index: int, body: bytes) -> None:
        source, bucket = self.source, self.source.quotas[bucket_index]
        ext = media.check_modality(body, source.modality)
        item_id = hashlib.sha256(body).hexdigest()
        other = self.ctx.known_items.get(item_id)
        if other is not None:
            raise DuplicateItem(f"duplicate of {other}")
        norm = None
        info: dict = {}
        if source.modality == "image":
            info = media.probe_image(body)
            norm = media.normalise_image(body)
        floor = self.ctx.spec.limits.min_free_disk_bytes
        self.ctx.vbench_dir.mkdir(parents=True, exist_ok=True)  # a fresh data dir (a field set) has nothing to stat yet
        if floor and self.ctx.free_disk(self.ctx.vbench_dir) < floor + len(body) + len(norm or b""):
            raise DiskFloorReached(f"less than {fmt_bytes(floor)} free on the cache volume")
        assert self.ctx.cache is not None
        self.ctx.cache.charge(len(body) + len(norm or b""))
        native_rel = f"native/{item_id}.{ext}"
        _write_atomic(self.ctx.vbench_dir / native_rel, body)
        norm_rel = None
        if norm is not None:
            norm_rel = f"norm/{item_id}.jpg"
            _write_atomic(self.ctx.vbench_dir / norm_rel, norm)
        else:
            info = media.probe_av(self.ctx.vbench_dir / native_rel)
        meta = candidate.meta
        assert source.group_key is not None
        row = {
            "item_id": item_id, "modality": source.modality, "label": bucket.label, "threat": bucket.threat,
            "generator": bucket.generator or (meta.get("generator") if bucket.label == "fake" else None),
            "family": bucket.family, "lgo_fold": bucket.lgo_fold, "release_date": bucket.release_date,
            "source_dataset": source.dataset, "source_id": source.id, "source_revision": source.revision,
            "source_row_key": candidate.row_key,
            "group_key": group_key(source.group_key, meta, candidate.row_key, source.dataset),
            "licence": meta.get("licence") if source.mode == "api" else source.licence,
            "commercial_ok": meta.get("commercial_ok") == "true" if source.mode == "api" else source.commercial_ok,
            "role": source.role, "split": None, "real_stratum": bucket.real_stratum,
            "width": info.get("width"), "height": info.get("height"), "jpeg_q_est": info.get("jpeg_q_est"),
            "has_exif": info.get("has_exif"), "has_c2pa": info.get("has_c2pa"), "face_px_max": None,
            "duration_s": info.get("duration_s"), "native_path": native_rel, "norm_path": norm_rel, "audit_ok": None,
            "native_bytes": len(body), "norm_bytes": len(norm) if norm is not None else None, "fetched_at": self.ctx.now(),
        }
        self.state.append({"t": "item", "bucket": bucket_signature(bucket), "row": row}, self.meter.used)
        self.ctx.known_items[item_id] = f"{source.id}/{candidate.row_key}"
        self.group_counts[row["group_key"]] += 1


def _reason_class(reason: str) -> str:
    return "duplicate" if reason.startswith("duplicate of") else reason


def _round_robin(lanes: list[list[Candidate]]) -> Iterator[Candidate]:
    """One candidate from each lane in turn until every lane is exhausted."""
    for rank in range(max((len(lane) for lane in lanes), default=0)):
        for lane in lanes:
            if rank < len(lane):
                yield lane[rank]


def _unwrap(e: BaseException) -> BaseException | None:
    """The fetcher's own exception if pyarrow wrapped one raised inside the range file."""
    seen: BaseException | None = e
    while seen is not None:
        if isinstance(seen, ByteCapReached | RemoteError):
            return seen
        seen = seen.__cause__ or seen.__context__
    return None


def _read_group(parquet: Any, group: int, columns: list[str]) -> list[dict]:
    try:
        return parquet.read_row_group(group, columns=columns, use_threads=False).to_pylist()
    except Exception as e:  # noqa: BLE001
        raise _unwrap(e) or SourceError(f"row group {group}: read failed ({type(e).__name__}: {e})") from None


def _parquet_media(parquet: Any, group: int, candidates: list[Candidate], indices: list[int]) -> Iterator[bytes | Exception]:
    columns = sorted({candidates[j].ref[1] for j in indices})
    try:
        table = parquet.read_row_group(group, columns=columns, use_threads=False)
    except Exception as e:  # noqa: BLE001
        raise _unwrap(e) or SourceError(f"row group {group}: media read failed ({type(e).__name__})") from None
    for j in indices:
        row, column = candidates[j].ref
        yield _media_bytes(table.column(column)[row].as_py())


def _read_member(archive: zipfile.ZipFile, info: zipfile.ZipInfo, limit: int) -> bytes:
    if info.file_size > limit:
        raise media.MediaError("member larger than the per-object cap")
    try:
        with archive.open(info) as f:
            return f.read()
    except (zipfile.BadZipFile, NotImplementedError, OSError) as e:
        raise _unwrap(e) or RemoteError(f"{info.filename}: member read failed ({type(e).__name__})") from None


def _read_tar_member(archive: tarfile.TarFile, member: tarfile.TarInfo, limit: int) -> bytes:
    if member.size > limit:
        raise media.MediaError("member larger than the per-object cap")
    handle = archive.extractfile(member)
    if handle is None:
        raise media.MediaError("the tar member is not a regular file")
    with handle:
        return handle.read()


def _write_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def check_dataset(ctx: FetchContext, source: Source) -> None:
    """Refuse gated, private or disabled repos (D-03, D-05) and revisions the Hub does not know."""
    assert source.revision is not None
    info = ctx.hub.dataset_info(source.dataset, source.revision)
    if info.get("gated"):
        raise SourceSkipped(f"gated dataset ({info['gated']}): the loop does not accept dataset terms (D-03, D-05)")
    if info.get("private") or info.get("disabled"):
        raise SourceSkipped("private or disabled dataset")
    if info.get("sha") and info["sha"] != source.revision:
        raise SourceError(f"revision {source.revision} resolved to {info['sha']}")


def run_source(ctx: FetchContext, source: Source) -> SourceReport:
    report = SourceReport(source, byte_cap=source.byte_cap(ctx.spec.limits))
    if source.mode == "excluded_bulk":
        report.status, report.reason = "skipped", f"excluded_bulk: {source.notes or 'bulk-only source (ML F1)'}"
        return report
    if not source.pinned:
        report.status, report.reason = "skipped", "revision not pinned (run vbench_fetch.py --pin --write)"
        return report
    run = SourceRun(ctx, source, report)
    fetching = source.frozen is None and not ctx.cached_only
    try:
        if source.frozen is not None:
            frozen = source.frozen
            report.status, report.reason = "frozen", (f"frozen {frozen.date} ({frozen.decision}): {run.quotas.total} of "
                                                     f"{frozen.planned_items} planned items kept; not contacted")
        elif ctx.cached_only:
            if not run.quotas.done:
                last = run.state.outcome
                report.status, report.reason = "shortfall", "cached only; " + (
                    f"the last fetch ended {last['status']}: {last['reason']}" if last and last.get("reason")
                    else "the last fetch recorded no reason")
        else:
            if source.mode in HF_MODES:
                check_dataset(ctx, source)
            if not run.quotas.done:
                run.consume(run.batches())
    except SourceSkipped as e:
        report.status, report.reason = "skipped", str(e)
    except ByteCapReached as e:
        report.status, report.reason = "shortfall", str(e)
        report.total_cap_hit = e.scope == "total cache"
        if report.total_cap_hit:
            report.stop_run = "total cache cap reached"
    except (TimeBoxReached, DiskFloorReached) as e:
        report.status, report.reason = "shortfall", str(e)
        report.stop_run = "time box reached" if isinstance(e, TimeBoxReached) else str(e)
    except (RemoteError, SourceError) as e:
        report.status, report.reason = "error", str(e)
        log.warning("vbench.source_error", source=source.id, error=str(e))
    finally:
        run.state.flush_bytes(run.meter.used)
    report.kept, report.bytes_used, report.filled = run.quotas.total, run.meter.used, list(run.quotas.filled)
    if report.status == "complete" and not run.quotas.done:
        report.status, report.reason = "shortfall", "candidates exhausted before the quotas were met"
    if fetching:  # a later cached-only rebuild reports this reason
        run.state.append({"t": "outcome", "status": report.status, "reason": report.reason, "at": ctx.now()}, run.meter.used)
    log.info("vbench.source_done", source=source.id, status=report.status, kept=report.kept, bytes=report.bytes_used)
    return report


def run_fetch(ctx: FetchContext, sources: list[Source]) -> list[SourceReport]:
    reports = []
    for index, source in enumerate(sources):
        report = run_source(ctx, source)
        reports.append(report)
        if report.stop_run is not None:
            for rest in sources[index + 1:]:
                reports.append(SourceReport(rest, status="skipped", reason=report.stop_run,
                                            byte_cap=rest.byte_cap(ctx.spec.limits)))
            break
    return reports


# --- dry run -----------------------------------------------------------------------------------------------

@dataclass
class SourcePlan:
    source: Source
    status: str = "ok"  # ok | skipped | error
    reason: str | None = None
    matched: int | None = None
    selected: list[RemoteFile] = field(default_factory=list)
    bucket_candidates: list[int] | None = None
    projected_bytes: int | None = None
    projection: str = ""


def plan_source(ctx: FetchContext, source: Source) -> SourcePlan:
    """List trees (metadata only) and project bytes; never reads a data object."""
    plan = SourcePlan(source)
    cap = source.byte_cap(ctx.spec.limits)
    if source.mode == "excluded_bulk":
        plan.status, plan.reason = "skipped", f"excluded_bulk: {source.notes or 'bulk-only source'}"
        return plan
    if not source.pinned:
        plan.status, plan.reason = "skipped", "revision not pinned"
    elif source.mode in HF_MODES:
        try:
            check_dataset(ctx, source)
            assert source.files is not None and source.revision is not None
            rng = random.Random(f"{ctx.spec.seed}:{source.id}")
            plan.selected, plan.matched = select_files(ctx.hub.tree(source.dataset, source.revision),
                                                       source.files.pattern, source.files.sample, rng)
            if source.mode == "per_file" and source.files.extract is None:
                pattern = re.compile(source.path_regex or "")
                metas = [(_meta(source.constants, m.groupdict()), f.size) for f in plan.selected
                         if (m := pattern.search(f.path))]
                plan.bucket_candidates = [sum(1 for meta, _ in metas if b.matches(meta)) for b in source.quotas]
                if metas:
                    average = sum(size for _, size in metas) // len(metas)
                    plan.projected_bytes = min(cap, average * source.max_items)
                    plan.projection = f"{len(metas)} parsed files, mean {fmt_bytes(average)} × {source.max_items} items"
        except SourceSkipped as e:
            plan.status, plan.reason = "skipped", str(e)
        except (RemoteError, SourceError) as e:
            plan.status, plan.reason = "error", str(e)
    if plan.projected_bytes is None:
        if source.avg_item_bytes:
            plan.projected_bytes = min(cap, source.avg_item_bytes * source.max_items)
            plan.projection = f"declared mean {fmt_bytes(source.avg_item_bytes)} × {source.max_items} items"
        else:
            plan.projected_bytes, plan.projection = cap, "byte cap (no per-item estimate)"
    return plan


# --- pin ---------------------------------------------------------------------------------------------------

_SOURCE_LINE = re.compile(r"^  - id: (\S+)\s*$")


def pin_text(text: str, revisions: dict[str, str]) -> str:
    """Rewrite the `revision:` line of each listed source block, leaving everything else intact."""
    lines = text.splitlines(keepends=True)
    current, pinned = None, set()
    for i, line in enumerate(lines):
        if match := _SOURCE_LINE.match(line):
            current = match.group(1)
        elif current in revisions and re.match(r"^    revision:\s", line):
            lines[i] = re.sub(r"(revision:\s*)[^\s#]+", lambda m, r=revisions[current]: m.group(1) + r, line, count=1)
            pinned.add(current)
    missing = sorted(set(revisions) - pinned)
    if missing:
        raise SourceError(f"no `revision:` line for sources: {', '.join(missing)}")
    return "".join(lines)


def resolve_revisions(ctx: FetchContext, sources: list[Source], *, repin: bool = False) -> dict[str, str]:
    revisions = {}
    for source in sources:
        if source.mode not in HF_MODES or (source.pinned and not repin):
            continue
        sha = ctx.hub.dataset_info(source.dataset, "main").get("sha")
        if not isinstance(sha, str) or not HEX40.fullmatch(sha):
            raise SourceError(f"{source.id}: the Hub returned no commit for {source.dataset}")
        revisions[source.id] = sha
    return revisions


# --- markdown ----------------------------------------------------------------------------------------------

def fmt_bytes(n: int | None) -> str:
    if n is None:
        return "n/a"
    for unit, size in (("GB", 1024**3), ("MB", 1024**2), ("KB", 1024)):
        if n >= size:
            return f"{n / size:.2f} {unit}"
    return f"{n} B"


def _cell(text: object) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


def render_plan(spec: Spec, plans: list[SourcePlan], *, spec_label: str, generated_at: str) -> str:
    fetchable = [p for p in plans if p.status == "ok"]
    items = sum(p.source.max_items for p in fetchable)
    projected = sum(p.projected_bytes or 0 for p in fetchable)
    caps = sum(p.source.byte_cap(spec.limits) for p in fetchable)
    lines = [
        f"# {spec.bench} fetch plan (dry run)", "",
        f"Generated {generated_at} by `make vbench-fetch DRY_RUN=1` from `{spec_label}`. No media was downloaded: the dry "
        "run reads HF dataset info and tree listings only. `api` sources are not contacted.", "",
        f"- **Sources:** {len(plans)} ({len(fetchable)} fetchable, "
        f"{sum(1 for p in plans if p.status == 'skipped')} skipped, {sum(1 for p in plans if p.status == 'error')} errors)",
        f"- **Planned items:** {items}",
        f"- **Projected bytes transferred:** {fmt_bytes(projected)}; sum of source byte caps {fmt_bytes(caps)}. The fetcher "
        f"stops at {fmt_bytes(spec.limits.max_total_bytes)} of cache in total and never downloads an object above "
        f"{fmt_bytes(spec.limits.max_object_bytes)} whole.", "",
        "| Source | Dataset | Mode | Modality | Licence | Commercial | Items | Files (selected / matched) | Projected | Cap | Status |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for p in plans:
        s = p.source
        files = f"{len(p.selected)} / {p.matched}" if p.matched is not None else "n/a"
        status = p.status if p.reason is None else f"{p.status}: {p.reason}"
        lines.append(f"| `{s.id}` | {_cell(s.dataset)} | {s.mode} | {s.modality} | {_cell(s.licence)} | "
                     f"{'yes' if s.commercial_ok else 'no'} | {s.max_items} | {files} | {fmt_bytes(p.projected_bytes)} | "
                     f"{fmt_bytes(s.byte_cap(spec.limits))} | {_cell(status)} |")
    lines += ["", "## Exclusions (D-05)", ""]
    lines += [f"- `{e.dataset}`: {e.reason}" for e in spec.exclusions] or ["- none"]
    lines += ["", "## Sources", ""]
    for p in plans:
        s = p.source
        lines += [f"### `{s.id}`", "",
                  f"- **Dataset:** `{s.dataset}` @ `{s.revision or 'n/a'}`; mode `{s.mode}`; role `{s.role}`",
                  f"- **Licence:** {s.licence}; commercial_ok {str(s.commercial_ok).lower()}",
                  f"- **Caps:** {s.max_items} items, {fmt_bytes(s.byte_cap(spec.limits))}; projected "
                  f"{fmt_bytes(p.projected_bytes)} ({p.projection})"]
        if s.group_key is not None:
            rule = s.group_key
            lines.append(f"- **group_key:** `{rule.field}`" + (f" (regex `{rule.regex}`)" if rule.regex else "")
                         + (f", fallback `{rule.fallback}`" if rule.fallback else "")
                         + f", namespace `{rule.namespace or s.dataset}`")
        if p.reason:
            lines.append(f"- **Status:** {p.status}: {p.reason}")
        if p.selected:
            shown = ", ".join(f"`{f.path}` ({fmt_bytes(f.size)})" for f in p.selected[:6])
            more = f" and {len(p.selected) - 6} more" if len(p.selected) > 6 else ""
            lines.append(f"- **Selected files:** {shown}{more}")
        if s.notes:
            lines.append(f"- **Notes:** {s.notes}")
        if s.quotas:
            lines += ["", "| Bucket | Quota | Candidates |", "|---|---|---|"]
            for i, bucket in enumerate(s.quotas):
                candidates = p.bucket_candidates[i] if p.bucket_candidates is not None else "n/a"
                lines.append(f"| {_cell(describe_bucket(bucket))} | {bucket.n} | {candidates} |")
        lines.append("")
    return "\n".join(lines)


def render_report(spec: Spec, reports: list[SourceReport], ctx: FetchContext, *, spec_label: str,
                  generated_at: str) -> str:
    assert ctx.cache is not None
    kept = sum(r.kept for r in reports)
    added = sum(r.kept - r.kept_before for r in reports)
    lines = [
        f"# {spec.bench} fetch report", "",
        f"Generated {generated_at} by `backend/scripts/vbench_fetch.py` from `{spec_label}`.", "",
        f"- **Items:** {kept} in the selected sources ({added} added by this run); manifest rows {len(manifest_rows(ctx.vbench_dir))}",
        f"- **Cache:** {fmt_bytes(ctx.cache.used)} of {fmt_bytes(spec.limits.max_total_bytes)} (native + norm bytes)", "",
        "| Source | Mode | Status | Items (kept / max) | Bytes transferred (used / cap) | Reason |",
        "|---|---|---|---|---|---|",
    ]
    for r in reports:
        lines.append(f"| `{r.source.id}` | {r.source.mode} | {r.status} | {r.kept} / {r.source.max_items} | "
                     f"{fmt_bytes(r.bytes_used)} / {fmt_bytes(r.byte_cap)} | {_cell(r.reason or '')} |")
    shortfalls = [(r, b, f) for r in reports if r.filled for b, f in zip(r.source.quotas, r.filled) if f < b.n]
    lines += ["", "## Shortfalls", ""]
    if shortfalls:
        lines += ["| Source | Bucket | Kept | Quota | Reason |", "|---|---|---|---|---|"]
        lines += [f"| `{r.source.id}` | {_cell(describe_bucket(b))} | {f} | {b.n} | {_cell(r.reason or 'quota not met')} |"
                  for r, b, f in shortfalls]
    else:
        lines.append("None in the selected sources.")
    rejects = [(r, reason, n) for r in reports for reason, n in sorted(r.rejects.items())]
    lines += ["", "## Rejections", ""]
    if rejects:
        lines += ["| Source | Reason | Count |", "|---|---|---|"]
        lines += [f"| `{r.source.id}` | {_cell(reason)} | {n} |" for r, reason, n in rejects]
    else:
        lines.append("None.")
    notes = [(r, note) for r in reports for note in r.notes]
    if notes:
        lines += ["", "## Notes", ""] + [f"- `{r.source.id}`: {_cell(note)}" for r, note in notes]
    return "\n".join(lines) + "\n"
