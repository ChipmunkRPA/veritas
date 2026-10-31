"""`docs/v2/bench/sources.yaml`: schema, loading and invariants (ML §2.2–2.5, §2.10; D-05).

A source declares exactly one access mode, a pinned revision, per-source caps, a licence, a `group_key` rule and
quota buckets. Each bucket matches candidate metadata and carries the manifest attributes (label, threat, family,
LGO fold, real stratum) of the items it admits.

Invariants enforced at load:
- ids are unique; every fetchable source has quotas, `max_items` and a `group_key` rule;
- a source whose dataset is on the exclusion list (D-05) is refused unless its mode is `excluded_bulk`;
- the hard limits can be lowered but never raised above D-05 as amended by D-40 (55 GB total, 5 GB per object);
- `api` sources are paced at ≥ 1 s between requests (ML F1);
- HF-backed sources carry a 40-hex commit (checked when fetching, so `--pin` can fill them in); `url_list` sources pin
  their index file by sha256.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

GB = 1024**3
MB = 1024**2
HARD_MAX_TOTAL_BYTES = 55 * GB  # D-05 set 40 GB; D-40 raised the cache cap to 55 GB for vb-1.1
HARD_MAX_OBJECT_BYTES = 5 * GB
HEX64 = re.compile(r"[0-9a-f]{64}")

Mode = Literal["parquet_stream", "per_file", "zip_range", "api", "url_list", "excluded_bulk"]
HF_MODES = ("parquet_stream", "per_file", "zip_range")
Modality = Literal["image", "video", "audio"]
Label = Literal["real", "fake"]
Threat = Literal["real", "t2i", "edit", "faceswap", "gan_face", "personalised", "t2v", "talking_head", "tts"]
Role = Literal["core", "diagnostic"]

HEX40 = re.compile(r"[0-9a-f]{40}")
_ID = re.compile(r"[a-z0-9][a-z0-9._-]*")
_SIZE = re.compile(r"\s*(\d+(?:\.\d+)?)\s*(B|KB|MB|GB)?\s*", re.IGNORECASE)
_UNITS = {"b": 1, "kb": 1024, "mb": MB, "gb": GB}


class SpecError(ValueError):
    """The benchmark source spec cannot be read or violates an invariant."""


def parse_size(value: int | str | None) -> int | None:
    if value is None or isinstance(value, int):
        return value
    match = _SIZE.fullmatch(str(value))
    if match is None:
        raise ValueError(f"not a size: {value!r} (use e.g. 512MB, 6GB)")
    return int(float(match.group(1)) * _UNITS[(match.group(2) or "b").lower()])


def _size_field(*names: str):
    return field_validator(*names, mode="before")(lambda value: parse_size(value))


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Limits(_Strict):
    max_total_bytes: int = HARD_MAX_TOTAL_BYTES
    max_object_bytes: int = HARD_MAX_OBJECT_BYTES
    default_max_bytes: int = GB
    min_free_disk_bytes: int = 0  # an item is not stored while the cache volume has less free space than this

    sizes = _size_field("max_total_bytes", "max_object_bytes", "default_max_bytes", "min_free_disk_bytes")

    @model_validator(mode="after")
    def _within_d05(self) -> Limits:
        if self.max_total_bytes > HARD_MAX_TOTAL_BYTES:
            raise ValueError("limits.max_total_bytes may not exceed 55 GB (D-05, D-40)")
        if self.max_object_bytes > HARD_MAX_OBJECT_BYTES:
            raise ValueError("limits.max_object_bytes may not exceed 5 GB (D-05)")
        return self


class Subset(_Strict):
    fraction: float = Field(gt=0, le=1)
    min_bytes: int = 512 * MB  # a scaled byte cap never drops below this (one parquet row group can be large)
    max_bytes: dict[str, int] = Field(default_factory=dict)  # per-source scaled cap; never above the source's own cap

    sizes = _size_field("min_bytes")

    @field_validator("max_bytes", mode="before")
    @classmethod
    def _source_sizes(cls, value: dict | None) -> dict:
        return {key: parse_size(size) for key, size in (value or {}).items()}


class Exclusion(_Strict):
    dataset: str
    reason: str


class GroupKeyRule(_Strict):
    """group_key = `<namespace>:<sha256(value)[:16]>` where value = meta[field] (optionally narrowed by `regex`),
    falling back to meta[fallback] and then the row key."""

    field: str
    regex: str | None = None  # applied to the field value; the named group `key` (or group 1) is the value
    fallback: str | None = None
    normalise: Literal["text", "none"] = "text"  # text: strip, collapse whitespace, lower-case
    namespace: str | None = None  # default: the dataset id

    @field_validator("regex")
    @classmethod
    def _compiles(cls, value: str | None) -> str | None:
        if value is not None:
            re.compile(value)
        return value


class Bucket(_Strict):
    """A quota: the first `n` candidates whose metadata match `match` are admitted with these attributes."""

    match: dict[str, str | list[str]] = Field(default_factory=dict)
    n: int = Field(ge=0)
    label: Label
    threat: Threat
    generator: str | None = None  # manifest generator; default: the candidate's `generator` metadata (fakes)
    family: str | None = None
    lgo_fold: str | None = None
    real_stratum: str | None = None
    release_date: str | None = None

    @model_validator(mode="after")
    def _consistent(self) -> Bucket:
        if (self.label == "real") != (self.threat == "real"):
            raise ValueError("label real ⇔ threat real")
        if self.label == "real" and not self.real_stratum:
            raise ValueError("a real bucket needs real_stratum")
        if self.label == "fake" and not self.family:
            raise ValueError("a fake bucket needs family")
        return self

    def matches(self, meta: dict[str, str]) -> bool:
        for key, wanted in self.match.items():
            value = meta.get(key)
            if value is None or value not in (wanted if isinstance(wanted, list) else [wanted]):
                return False
        return True


class MediaColumn(_Strict):
    column: str
    meta: dict[str, str] = Field(default_factory=dict)  # meta key → column read for this media item (e.g. model1)


class ParquetConfig(_Strict):
    columns: dict[str, str] = Field(default_factory=dict)  # meta key → column name (read before decode)
    media: list[MediaColumn] = Field(min_length=1)
    row_key: str | None = None  # column holding a stable row id; default `<shard>#<row_group>:<row>`


class FileSelect(_Strict):
    pattern: list[str]  # fnmatch globs over repo paths
    sample: int | None = Field(default=None, ge=1)  # choose k matching files/shards/archives at random (seeded)
    members_per_archive: int | None = Field(default=None, ge=1)  # zip_range / extract: candidates kept per archive
    extract: Literal["tar"] | None = None  # per_file: download small tar(.gz/.bz2) archives whole, extract members

    @field_validator("pattern", mode="before")
    @classmethod
    def _listify(cls, value: str | list[str]) -> list[str]:
        return [value] if isinstance(value, str) else value


class CommonsQuery(_Strict):
    category: str | None = None  # `Category:…` (files directly in the category)
    search: str | None = None  # CirrusSearch in the File namespace, e.g. `filemime:video/webm interview`
    random: bool = False  # the random-file generator
    meta: dict[str, str] = Field(default_factory=dict)  # constants merged into candidate metadata
    max_candidates: int = Field(default=1000, ge=1)
    max_requests: int = Field(default=100, ge=1)

    @model_validator(mode="after")
    def _one_generator(self) -> CommonsQuery:
        if sum((self.category is not None, self.search is not None, self.random)) != 1:
            raise ValueError("a Commons query needs exactly one of `category`, `search` or `random: true`")
        return self

    @property
    def label(self) -> str:
        return self.category or (f"search {self.search!r}" if self.search else "random files")


class ApiConfig(_Strict):
    provider: Literal["commons"]
    endpoint: str = "https://commons.wikimedia.org/w/api.php"
    queries: list[CommonsQuery] = Field(min_length=1)
    mime: list[str] = Field(default_factory=lambda: ["image/jpeg"])
    exif_make: list[str] | None = None  # keep only files whose EXIF Make starts with one of these (case-insensitive)
    max_file_bytes: int = 40 * MB

    sizes = _size_field("max_file_bytes")


class UrlListConfig(_Strict):
    """A plain-HTTP dataset published as a newline-separated index of file URLs (VISION: `wget -i VISION_files.txt`)."""

    index: str  # URL of the index file
    index_sha256: str  # the pinned index; a changed index is refused
    base: str  # URL prefix removed before `path_regex` is applied; lines outside it are ignored
    interleave: str | None = None  # a `path_regex` group: candidates alternate over its values (one device at a time)
    max_index_bytes: int = 64 * MB

    sizes = _size_field("max_index_bytes")

    @field_validator("index_sha256")
    @classmethod
    def _hex64(cls, value: str) -> str:
        if not HEX64.fullmatch(value):
            raise ValueError("index_sha256 must be 64 lower-case hex digits")
        return value


class Frozen(_Strict):
    """A source closed before its quotas were met (D-38). `max_items` is the count obtained; the fetcher never
    contacts the source again, and the report keeps `planned_items` and the unmet buckets as shortfalls."""

    date: str
    decision: str
    planned_items: int = Field(ge=1)


class Source(_Strict):
    id: str
    dataset: str
    mode: Mode
    revision: str | None = None
    licence: str
    commercial_ok: bool
    modality: Modality
    role: Role = "core"
    files: FileSelect | None = None
    parquet: ParquetConfig | None = None
    path_regex: str | None = None  # per_file / zip_range: named groups → metadata
    constants: dict[str, str] = Field(default_factory=dict)
    api: ApiConfig | None = None
    url_list: UrlListConfig | None = None
    max_items: int = Field(default=0, ge=0)
    max_bytes: int | None = None
    max_per_group: int | None = Field(default=None, ge=1)  # items kept per group key (held items count)
    avg_item_bytes: int | None = None  # declared estimate for the dry-run projection (from probe facts)
    pace_s: float = Field(default=0.0, ge=0)
    group_key: GroupKeyRule | None = None
    quotas: list[Bucket] = Field(default_factory=list)
    frozen: Frozen | None = None
    notes: str | None = None

    sizes = _size_field("max_bytes", "avg_item_bytes")

    @field_validator("id")
    @classmethod
    def _slug(cls, value: str) -> str:
        if not _ID.fullmatch(value):
            raise ValueError(f"source id {value!r} must be a lower-case slug")
        return value

    @field_validator("path_regex")
    @classmethod
    def _compiles(cls, value: str | None) -> str | None:
        if value is not None:
            re.compile(value)
        return value

    @model_validator(mode="after")
    def _mode_fields(self) -> Source:
        if self.mode == "excluded_bulk":
            return self
        if not self.quotas or self.group_key is None or (self.max_items <= 0 and self.frozen is None):
            raise ValueError(f"{self.id}: needs quotas, max_items > 0 and a group_key rule")
        if self.frozen is not None and self.max_items > self.frozen.planned_items:
            raise ValueError(f"{self.id}: a frozen source keeps at most its planned items")
        if self.mode in HF_MODES and self.files is None:
            raise ValueError(f"{self.id}: {self.mode} needs `files`")
        if self.files is not None and self.files.extract is not None and self.mode != "per_file":
            raise ValueError(f"{self.id}: files.extract applies to per_file sources only")
        if self.mode == "parquet_stream" and self.parquet is None:
            raise ValueError(f"{self.id}: parquet_stream needs `parquet`")
        if self.mode in ("per_file", "zip_range", "url_list") and self.path_regex is None:
            raise ValueError(f"{self.id}: {self.mode} needs `path_regex`")
        if (self.mode == "url_list") != (self.url_list is not None):
            raise ValueError(f"{self.id}: `url_list` is required by, and only allowed with, mode url_list")
        if self.mode == "api":
            if self.api is None:
                raise ValueError(f"{self.id}: api needs `api`")
            if self.pace_s < 1.0:
                raise ValueError(f"{self.id}: api sources are paced at ≥ 1 s between requests (ML F1)")
        return self

    @property
    def pinned(self) -> bool:
        return self.mode not in HF_MODES or bool(self.revision and HEX40.fullmatch(self.revision))

    def byte_cap(self, limits: Limits) -> int:
        return self.max_bytes if self.max_bytes is not None else limits.default_max_bytes


class Spec(_Strict):
    schema_version: Literal[1]
    bench: str
    seed: int = 7
    limits: Limits = Field(default_factory=Limits)
    subsets: dict[str, Subset] = Field(default_factory=dict)
    exclusions: list[Exclusion] = Field(default_factory=list)
    sources: list[Source]

    @model_validator(mode="after")
    def _invariants(self) -> Spec:
        ids = [s.id for s in self.sources]
        duplicates = sorted({i for i in ids if ids.count(i) > 1})
        if duplicates:
            raise ValueError(f"duplicate source ids: {', '.join(duplicates)}")
        excluded = {e.dataset.lower(): e.reason for e in self.exclusions}
        for source in self.sources:
            reason = excluded.get(source.dataset.lower())
            if reason is not None and source.mode != "excluded_bulk":
                raise ValueError(f"{source.id}: dataset {source.dataset} is excluded (D-05): {reason}")
        for name, subset in self.subsets.items():
            unknown = sorted(set(subset.max_bytes) - set(ids))
            if unknown:
                raise ValueError(f"subset {name}: max_bytes names unknown source(s): {', '.join(unknown)}")
        return self

    def source(self, source_id: str) -> Source:
        for source in self.sources:
            if source.id == source_id:
                return source
        raise KeyError(f"no source {source_id!r} in the spec")

    def scaled(self, subset: str | None) -> Spec:
        """The spec for a subset: quotas and item caps scaled by `fraction` (ceil), byte caps with a floor.

        File selection does not depend on quotas, so a later full fetch resumes on top of a subset's items. For
        per_file, zip_range and api sources the candidate order is quota-independent too, so a subset's items are the
        full bench's first picks; parquet row groups are visited by quota yield, so there it holds only approximately."""
        if subset is None:
            return self
        if subset not in self.subsets:
            raise SpecError(f"unknown subset {subset!r}; the spec defines: {', '.join(self.subsets) or 'none'}")
        rule = self.subsets[subset]
        sources = []
        for source in self.sources:
            if source.mode == "excluded_bulk":
                sources.append(source)
                continue
            cap = source.byte_cap(self.limits)
            quotas = [b.model_copy(update={"n": scale_count(b.n, rule.fraction)}) for b in source.quotas]
            sources.append(source.model_copy(update={
                "quotas": quotas,
                "max_items": scale_count(source.max_items, rule.fraction),
                "max_bytes": min(cap, rule.max_bytes.get(source.id, max(int(cap * rule.fraction), rule.min_bytes))),
            }))
        return self.model_copy(update={"sources": sources, "bench": f"{self.bench}:{subset}"})


def scale_count(n: int, fraction: float) -> int:
    """ceil(n·fraction), at least 1 for a non-empty quota (rounded to 1e-9 so 0.1·300 stays 30)."""
    if n == 0:
        return 0
    return max(1, int(-(-round(n * fraction * 1e9) // 1e9)))


def parse_spec(raw: object, *, source: str = "<spec>") -> Spec:
    try:
        return Spec.model_validate(raw)
    except ValidationError as e:
        problems = "; ".join(f"{'.'.join(str(p) for p in err['loc']) or '<root>'}: {err['msg']}" for err in e.errors())
        raise SpecError(f"{source}: {problems}") from None


def load_spec(path: Path) -> Spec:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as e:
        raise SpecError(f"{path}: {e}") from None
    return parse_spec(raw, source=str(path))
