"""ModelManager: memory budget, leases, LRU eviction, circuit breaker, CPU fallback, footprints (ARCHITECTURE §2.5).

- **Budget.** `VERITAS_MODEL_MEMORY_BUDGET_MB=auto`: MPS `min(0.5·RAM, RAM−20 GB)` (D-21: 24 GB on the 48 GB host), CUDA
  `0.85·VRAM`, CPU `0.5·RAM`.
- **Leases.** `lease()` pins unique models and reserves one explicit activation allowance per live lease.
  Loading evicts unpinned, non-`keep_warm` models in LRU order until weights and live allowances fit. If everything is pinned, the lease waits up to `wait_s`, then raises
  `MemoryBudgetExceeded`, which the runner reports as `skipped: memory budget`. The contention deadline is shared
  by reservation, all requested models and fallback; an already-started loader is not interrupted.
- **Circuit breaker.** A model that fails to load is unavailable for 10 minutes (`failed: model unavailable`).
- **Device fallback.** An accelerator `NotImplementedError` / `RuntimeError` at load, or a golden-parity delta above 0.01
  in `warm()`, pins the model to CPU in `data/state/model_footprints.json`. Measured load sizes persist there too: on an
  accelerator a larger one raises the model's reservation, on CPU it is an observation of the process (D-75).
- **Batching.** `AdaptiveBatcher.map` halves the batch size on out-of-memory errors, down to 1.

Torch-free at import: the memory probe touches torch only when torch is already imported.
"""
from __future__ import annotations

import gc
import importlib
import json
import os
import platform
import sys
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from numbers import Integral
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from app.core.config import settings
from app.core.logging import get_logger
from app.engine.detector import DetectorResult

if TYPE_CHECKING:
    from app.engine.manifest import Manifest, ModelEntry

log = get_logger(__name__)

MiB = 1024**2
GiB = 1024**3
BREAKER_S = 600.0
PARITY_TOLERANCE = 0.01

LoaderFn = Callable[[str, str, str], Any]  # (model_id, device, dtype) -> loaded model object


class MemoryBudgetExceeded(RuntimeError):
    """The models cannot fit in the budget (or all loaded models stay pinned past the wait)."""

    def result(self) -> DetectorResult:
        return DetectorResult.skipped("memory budget")


class ModelUnavailable(RuntimeError):
    """The model failed to load recently (circuit breaker open) or has no loader."""

    def result(self) -> DetectorResult:
        return DetectorResult.failed("model unavailable")


class _DeviceFallback(Exception):
    """Internal: a model was just pinned to CPU; the lease restarts on CPU."""


def _nonnegative_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return int(value)


def is_oom(exc: BaseException) -> bool:
    return isinstance(exc, MemoryError) or (isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower())


# --- budget ------------------------------------------------------------------------------------------------

def auto_budget_bytes(device_kind: str, total_ram: int, vram: int | None = None) -> int:
    if device_kind == "cuda" and vram:
        return int(0.85 * vram)
    if device_kind == "mps":
        # D-21 formula; hosts with ≤ 40 GB RAM would go below 25 % of RAM, so that is the floor (assumption)
        return int(max(min(0.5 * total_ram, total_ram - 20 * GiB), 0.25 * total_ram))
    return int(0.5 * total_ram)


def memory_for_auto_budget(device_kind: str, budget: int) -> int:
    """The smallest memory at which the automatic rule gives at least `budget`: the machine's memory (the container's
    limit, when one is set) on CPU and MPS, the accelerator's memory on CUDA. The inverse of `auto_budget_bytes`
    (BUILD_SPEC N-14a: "the machine memory per profile and kind under the default rule")."""
    budget = _nonnegative_integer(budget, "budget")
    if device_kind == "cuda":
        memory = int(budget / 0.85)
    elif device_kind == "mps":
        memory = min(4 * budget, max(budget + 20 * GiB, 2 * budget))  # the three branches of the rule, each solved for RAM
    else:
        memory = 2 * budget
    while auto_budget_bytes(device_kind, memory, memory) < budget:  # the rule truncates; never answer one byte short
        memory += 1
    return memory


def resolve_budget_bytes(setting: str | int | None, device_kind: str, total_ram: int | None = None,
                         vram: int | None = None) -> int:
    if setting is None or str(setting).strip().lower() in ("", "auto"):
        return auto_budget_bytes(device_kind, total_ram if total_ram is not None else total_ram_bytes(), vram)
    try:
        mb = float(setting)
    except ValueError as e:
        raise ValueError(f"VERITAS_MODEL_MEMORY_BUDGET_MB must be 'auto' or a number of MB, got {setting!r}") from e
    if mb <= 0:
        raise ValueError("VERITAS_MODEL_MEMORY_BUDGET_MB must be positive")
    return int(mb * MiB)


# cgroup v2, then v1: the memory limit of the container this process runs in, as its own cgroup namespace shows it
CGROUP_LIMIT_FILES = (Path("/sys/fs/cgroup/memory.max"), Path("/sys/fs/cgroup/memory/memory.limit_in_bytes"))
_CGROUP_NO_LIMIT = 1 << 60  # v1 writes a page-rounded number near 2**63 for "no limit"; v2 writes "max"


def cgroup_memory_limit_bytes(files: tuple[Path, ...] | None = None) -> int | None:
    """The container's memory limit in bytes; None without a limit, and outside a container (the first file that can be
    read decides)."""
    for path in CGROUP_LIMIT_FILES if files is None else files:
        try:
            raw = path.read_text(encoding="ascii").strip()
        except (OSError, ValueError):
            continue
        if not raw.isdigit():
            return None  # "max"
        value = int(raw)
        return value if 0 < value < _CGROUP_NO_LIMIT else None
    return None


def machine_ram_bytes() -> int:
    try:
        import psutil

        return int(psutil.virtual_memory().total)
    except ImportError:
        return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))


def total_ram_bytes() -> int:
    """The memory this process may use: the machine's, or the container's limit when that is smaller (BUILD_SPEC N-14,
    N-14a). A Compose `mem_limit` below half of the machine's memory otherwise left the automatic budget above the limit,
    and the kernel killed the worker where the model manager should have evicted."""
    machine, limit = machine_ram_bytes(), cgroup_memory_limit_bytes()
    return min(machine, limit) if limit is not None else machine


def warn_budget_above_limit(budget: int, device_kind: str, limit: int | None = None) -> bool:
    """One warning when the model memory budget cannot hold under the container's memory limit (BUILD_SPEC N-14a).

    On CPU the models are in the process's own memory. `auto` is half of what the process may use, so it is always below
    the limit; a budget set by hand (`VERITAS_MODEL_MEMORY_BUDGET_MB`) at or above the limit means that the kernel kills
    the process before the manager evicts a model. On an accelerator the budget counts its memory, which the limit of
    the container does not bound: nothing is compared. Returns whether it warned."""
    limit = cgroup_memory_limit_bytes() if limit is None else limit
    if device_kind != "cpu" or limit is None or budget < limit:
        return False
    log.warning("models.budget_above_memory_limit", budget_mb=budget // MiB, memory_limit_mb=limit // MiB,
                setting=str(settings.model_memory_budget_mb),
                consequence="the kernel kills this process before a model is evicted: lower the budget or raise the limit")
    return True


def accelerator_kind() -> str:
    """Device class without importing torch: explicit setting, else torch if already loaded, else the platform."""
    if settings.device != "auto":
        return settings.device
    torch = sys.modules.get("torch")
    if torch is not None:
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return "mps" if platform.system() == "Darwin" and platform.machine() == "arm64" else "cpu"


def _kind(device: str) -> str:
    return str(device).split(":", 1)[0]


def measured_reserves(device: str) -> bool:
    """Whether a measured load raises a model's reservation on this device (D-75; BUILD_SPEC N-14c, E-9). On `mps` and
    `cuda` the probe reads the allocator's own count, and a load that measured above its declaration is reserved at the
    measurement from then on. On CPU the probe reads the resident memory of the process, which is not a measure of the
    model (one load of a 1,173 MB model was recorded as 1,160 MB and the next as 2,302 MB; a model read through a memory
    map records a few MB): there the reservation is the declaration, before and after a load, and the measurement is an
    observation that is recorded and reserves nothing."""
    return _kind(device) != "cpu"


def accounted_bytes(device: str, declared: int, measured: int | None) -> int:
    """What a loaded model counts against the budget: its declaration, or a larger measurement where one reserves."""
    return max(declared, measured or 0) if measured_reserves(device) else declared


# --- memory probes -----------------------------------------------------------------------------------------

class MemoryProbe(Protocol):
    def allocated(self, device: str) -> int | None: ...

    def release(self, device: str) -> None: ...


class NullProbe:
    def allocated(self, device: str) -> int | None:
        return None

    def release(self, device: str) -> None:
        return None


class TorchMemoryProbe:
    """MPS driver / CUDA allocator deltas (only if torch is already imported); CPU uses process RSS."""

    def allocated(self, device: str) -> int | None:
        kind, torch = _kind(device), sys.modules.get("torch")
        try:
            if kind == "mps" and torch is not None:
                return int(torch.mps.driver_allocated_memory())
            if kind == "cuda" and torch is not None:
                return int(torch.cuda.memory_allocated())
            if kind == "cpu":
                import psutil

                return int(psutil.Process().memory_info().rss)
        except Exception:  # a probe must never fail a load
            return None
        return None

    def release(self, device: str) -> None:
        kind, torch = _kind(device), sys.modules.get("torch")
        if torch is None:
            return
        try:
            if kind == "mps":
                torch.mps.empty_cache()
            elif kind == "cuda":
                torch.cuda.empty_cache()
        except Exception:
            return


# --- footprints store --------------------------------------------------------------------------------------

class FootprintStore:
    """Raw load deltas, conservative historical upper guards, and device pins.

    Cache keys identify a model and device class, not a weight revision or dtype.
    A historical guard may over-reserve a different dtype; it never lowers the
    current declaration. Reading legacy records does not rewrite the cache.
    A `cpu` row is an observation of the process and reserves nothing (D-75;
    `measured_reserves`): it is kept and shown, and the manager does not ask it.
    """

    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()
        self._warned: set[str] = set()
        self._data: dict[str, dict] = {"measured": {}, "device_pins": {}}
        if path.is_file():
            try:
                loaded = self._mapping(json.loads(path.read_text(encoding="utf-8")), "root")
                self._data = {key: dict(self._mapping(loaded.get(key, {}), key))
                              for key in self._data}
            except (OSError, ValueError) as e:
                log.warning("models.footprints_unreadable", path=str(path), error=str(e))

    def _invalid(self, field_name: str) -> None:
        # Bound both repeated diagnostics and memory used by malformed rows.
        if field_name not in self._warned and len(self._warned) < 64:
            self._warned.add(field_name)
            log.warning("models.footprint_invalid", path=str(self.path), field=field_name)

    def _mapping(self, value: Any, field_name: str) -> Mapping:
        if isinstance(value, Mapping):
            return value
        self._invalid(field_name)
        return {}

    def _row(self, model_id: str, device_kind: str) -> Mapping:
        model = self._mapping(self._data["measured"].get(model_id, {}), f"measured.{model_id}")
        return self._mapping(model.get(device_kind, {}), f"measured.{model_id}.{device_kind}")

    def _positive(self, row: Mapping, key: str, prefix: str) -> int | None:
        if key not in row:
            return None
        value = row[key]
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
            self._invalid(f"{prefix}.{key}")
            return None
        return int(value) if value > 0 else None

    def _maximum(self, row: Mapping, prefix: str) -> int:
        return max(self._positive(row, "bytes", prefix) or 0,
                   self._positive(row, "max_positive_load_delta_bytes", prefix) or 0)

    def measured(self, model_id: str, device_kind: str) -> int | None:
        """Latest valid raw positive delta, never the accounting floor or maximum."""
        with self._lock:
            return self._positive(self._row(model_id, device_kind), "bytes", f"measured.{model_id}.{device_kind}")

    def maximum_positive_delta(self, model_id: str, device_kind: str) -> int:
        with self._lock:
            return self._maximum(self._row(model_id, device_kind), f"measured.{model_id}.{device_kind}")

    def record_measured(self, model_id: str, device_kind: str, nbytes: int, load_s: float, *,
                        dtype: str | None = None, device: str | None = None) -> None:
        with self._lock:
            prefix = f"measured.{model_id}.{device_kind}"
            measured = self._positive({"bytes": nbytes}, "bytes", prefix)
            if measured is None:
                return
            row = dict(self._row(model_id, device_kind))
            maximum = max(self._maximum(row, prefix), measured)
            row.update(bytes=measured, max_positive_load_delta_bytes=maximum,
                       load_s=round(load_s, 3), measured_at=_now())
            # These describe only the latest raw observation, never the historical maximum.
            for key, value in (("observation_dtype", dtype), ("observation_device", device)):
                row.pop(key, None)
                if value is not None:
                    row[key] = value
            model = dict(self._mapping(self._data["measured"].get(model_id, {}), f"measured.{model_id}"))
            model[device_kind] = row
            self._data["measured"][model_id] = model
            self._save()

    def _pin_field(self, model_id: str, key: str) -> str | None:
        row = self._mapping(self._data["device_pins"].get(model_id, {}), f"device_pins.{model_id}")
        value = row.get(key)
        return value if isinstance(value, str) else None

    def pin(self, model_id: str) -> str | None:
        with self._lock:
            return self._pin_field(model_id, "device")

    def pin_reason(self, model_id: str) -> str | None:
        with self._lock:
            return self._pin_field(model_id, "reason")

    def set_pin(self, model_id: str, device: str, reason: str) -> None:
        with self._lock:
            self._data["device_pins"][model_id] = {"device": device, "reason": reason, "pinned_at": _now()}
            self._save()

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self._data, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(self.path)


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


# --- models, entries, leases -------------------------------------------------------------------------------

@dataclass(frozen=True)
class ManagedModel:
    id: str
    fp32_bytes: int = 0
    fp16_bytes: int | None = None
    keep_warm: bool = False
    fp16_safe: bool = False
    unsupported_devices: frozenset[str] = frozenset()
    device_fallback: bool = True  # v1 loaders choose their own device, so they cannot be retried on CPU
    activation_bytes_per_item: int | None = None  # None is unknown; zero is an explicit estimate

    @classmethod
    def from_manifest(cls, entry: ModelEntry) -> ManagedModel:
        fp = entry.footprint
        return cls(id=entry.id, fp32_bytes=int((fp.fp32_mb or 0) * MiB),
                   fp16_bytes=int(fp.fp16_mb * MiB) if fp.fp16_mb else None, keep_warm=entry.keep_warm,
                   fp16_safe=entry.fp16_safe,
                   unsupported_devices=frozenset(d for d, state in entry.devices.items() if state != "ok"),
                   activation_bytes_per_item=(int(fp.activation_mb_per_item * MiB)
                                              if fp.activation_mb_per_item is not None else None))


@dataclass
class _Loaded:
    model_id: str
    device: str
    dtype: str
    obj: Any
    bytes_declared: int
    bytes_measured: int | None
    load_s: float
    keep_warm: bool
    refcount: int = 0
    last_used: float = 0.0

    @property
    def nbytes(self) -> int:
        return accounted_bytes(self.device, self.bytes_declared, self.bytes_measured)


@dataclass(frozen=True)
class LoadedModelInfo:
    id: str
    device: str
    dtype: str
    bytes_declared: int
    bytes_measured: int | None
    last_used: float
    pins: int
    keep_warm: bool
    load_s: float
    bytes_accounted: int = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "bytes_accounted", accounted_bytes(self.device, self.bytes_declared, self.bytes_measured))


@dataclass(frozen=True)
class WarmReport:
    model_id: str
    device: str
    parity_delta: float | None
    pinned_cpu: bool
    reason: str | None = None


@dataclass(frozen=True)
class LeaseTerms:
    """Where a lease of these models loads them, and what it reserves for each before the load (`ModelManager.lease_terms`)."""

    device: str
    dtype: str
    weights: tuple[tuple[str, int], ...]  # (model id, reserved bytes), in lease order

    @property
    def weight_bytes(self) -> int:
        return sum(nbytes for _, nbytes in self.weights)


class _Silent:
    """A logger that says nothing: for a manager that loads no weights, whose "loaded" lines would mislead (`capacity.py`)."""

    def __getattr__(self, name: str) -> Callable[..., None]:
        return lambda *args, **kwargs: None


SILENT = _Silent()


class ModelLease:
    def __init__(self, device: str, dtype: str, entries: Mapping[str, _Loaded]):
        self.device = device
        self.dtype = dtype
        self._entries = dict(entries)

    @property
    def model_ids(self) -> tuple[str, ...]:
        return tuple(self._entries)

    def get(self, model_id: str) -> Any:
        try:
            return self._entries[model_id].obj
        except KeyError:
            raise KeyError(f"model {model_id!r} is not part of this lease") from None


class AdaptiveBatcher[T]:
    """Micro-batches over `items`. `map(fn)` halves the batch size on out-of-memory errors, down to 1."""

    def __init__(self, items: Iterable[T], batch_size: int, *, on_oom: Callable[[], None] | None = None):
        self.items = list(items)
        self.batch_size = max(1, int(batch_size))
        self.on_oom = on_oom
        self.sizes_tried: list[int] = []

    def __iter__(self) -> Iterator[list[T]]:
        for i in range(0, len(self.items), self.batch_size):
            yield self.items[i:i + self.batch_size]

    def map[R](self, fn: Callable[[list[T]], Sequence[R]]) -> list[R]:
        out: list[R] = []
        i, size = 0, self.batch_size
        while i < len(self.items):
            batch = self.items[i:i + size]
            self.sizes_tried.append(len(batch))
            try:
                results = fn(batch)
            except Exception as e:
                if not is_oom(e) or size == 1:
                    raise
                if self.on_oom is not None:
                    self.on_oom()
                size = max(1, size // 2)
                log.warning("models.batch_halved", batch_size=size)
                continue
            if len(results) != len(batch):
                raise ValueError(f"batch function returned {len(results)} results for {len(batch)} items")
            out.extend(results)
            i += len(batch)
        self.batch_size = size
        return out


class ModelManager:
    def __init__(self, *, budget_bytes: int, models: Mapping[str, ManagedModel] | Iterable[ManagedModel],
                 loader_for: Callable[[str], LoaderFn], store: FootprintStore, probe: MemoryProbe | None = None,
                 clock: Callable[[], float] = time.monotonic, breaker_s: float = BREAKER_S, default_device: str = "cpu",
                 v1_resolver: Callable[[str], ManagedModel] | None = None, logger: Any = None):
        self.budget_bytes = int(budget_bytes)
        self._log = log if logger is None else logger
        self.default_device = default_device
        self._models: dict[str, ManagedModel] = dict(models) if isinstance(models, Mapping) else {m.id: m for m in models}
        self._loader_for = loader_for
        self._store = store
        self._probe = probe or NullProbe()
        self._clock = clock
        self._breaker_s = breaker_s
        self._v1_resolver = v1_resolver
        self._v1_loaders: dict[str, Callable[[], Any]] = {}
        self._loaded: OrderedDict[tuple[str, str, str], _Loaded] = OrderedDict()  # exact model/device/precision; LRU oldest first
        self._loading: dict[tuple[str, str, str], int] = {}  # reserved bytes while a load is in progress
        self._activation_bytes = 0  # once per live lease, including acquisition and CPU fallback
        self._breaker: dict[str, float] = {}
        self._cond = threading.Condition(threading.RLock())

    # ---- accounting ----
    def register(self, model: ManagedModel) -> None:
        with self._cond:
            self._models[model.id] = model

    def _model(self, model_id: str) -> ManagedModel:
        try:
            return self._models[model_id]
        except KeyError:
            raise KeyError(f"unknown model {model_id!r}") from None

    def _weight_floor(self, model_id: str, dtype: str) -> int:
        model = self._model(model_id)
        return model.fp16_bytes if dtype == "float16" and model.fp16_bytes else model.fp32_bytes

    def declared_bytes(self, model_id: str, device: str, dtype: str = "float32") -> int:
        """Pre-load reservation: the selected declaration, and on an accelerator any larger historical guard. On CPU a
        recorded load reserves nothing (`measured_reserves`), so a record an earlier version made is ignored."""
        floor = self._weight_floor(model_id, dtype)
        if not measured_reserves(device):
            return floor
        return max(floor, self._store.maximum_positive_delta(model_id, _kind(device)))

    def _used_bytes(self) -> int:
        return sum(e.nbytes for e in self._loaded.values()) + sum(self._loading.values()) + self._activation_bytes

    def activation_bytes_for_items(self, items: Mapping[str, int]) -> int:
        """Sum explicit per-model execution counts; never infer batches or treat unknown estimates as zero."""
        counts = {model_id: _nonnegative_integer(count, "activation item count")
                  for model_id, count in items.items()}
        total = 0
        with self._cond:
            for model_id, count in counts.items():
                model = self._models.get(model_id)
                if model is None or model.activation_bytes_per_item is None:
                    raise ModelUnavailable(f"model unavailable: activation footprint not established for {model_id}")
                per_item = _nonnegative_integer(model.activation_bytes_per_item, "activation_bytes_per_item")
                total += count * per_item
        return total

    def free_bytes(self) -> int:
        with self._cond:
            return max(0, self.budget_bytes - self._used_bytes())

    def batch_size(self, max_batch: int, activation_bytes_per_item: int) -> int:
        if activation_bytes_per_item <= 0:
            return max(1, max_batch)
        return max(1, min(max_batch, self.free_bytes() // activation_bytes_per_item))

    def batches[T](self, items: Iterable[T], *, max_batch: int, activation_bytes_per_item: int,
                device: str) -> AdaptiveBatcher[T]:
        return AdaptiveBatcher(items, self.batch_size(max_batch, activation_bytes_per_item),
                               on_oom=lambda: self._probe.release(device))

    def snapshot(self) -> list[LoadedModelInfo]:
        with self._cond:
            return [LoadedModelInfo(e.model_id, e.device, e.dtype, e.bytes_declared, e.bytes_measured, e.last_used,
                                    e.refcount, e.keep_warm, e.load_s) for e in self._loaded.values()]

    # ---- leases ----
    def _resolve_device(self, model_ids: Sequence[str], prefer: str) -> str:
        if _kind(prefer) == "cpu":
            return "cpu"
        for model_id in model_ids:
            model = self._model(model_id)
            if self._store.pin(model_id) == "cpu" or _kind(prefer) in model.unsupported_devices:
                return "cpu"
        return prefer

    def _dtype(self, model_ids: Sequence[str], device: str) -> str:
        accel = _kind(device) in ("cuda", "mps")
        return "float16" if accel and model_ids and all(self._model(m).fp16_safe for m in model_ids) else "float32"

    def lease_terms(self, model_ids: Sequence[str], *, prefer: str | None = None) -> LeaseTerms:
        """The device and precision `lease` chooses for these models, and the bytes it reserves for each before it loads
        it. Nothing is loaded or reserved (BUILD_SPEC N-14a: the capacity plan reads its numbers here)."""
        model_ids = tuple(dict.fromkeys(model_ids))
        with self._cond:
            device = self._resolve_device(model_ids, prefer or self.default_device)
            dtype = self._dtype(model_ids, device)
            return LeaseTerms(device, dtype, tuple((m, self.declared_bytes(m, device, dtype)) for m in model_ids))

    @contextmanager
    def lease(self, model_ids: Sequence[str], *, prefer: str | None = None, activation_bytes: int = 0,
              wait_s: float = 0.0) -> Iterator[ModelLease]:
        """Pin unique models and reserve one explicit activation allowance for the entire live lease.

        The allowance is an accounting estimate, not a measured process/GPU peak. It remains reserved
        during loading, CPU fallback and the body, even when an external watchdog has returned.
        `wait_s` is one contention budget across reservation, models and fallback, not a loader timeout.
        Immediately available acquisitions remain allowed after it expires; only waiting is refused.
        """
        activation_bytes = _nonnegative_integer(activation_bytes, "activation_bytes")
        model_ids = tuple(dict.fromkeys(model_ids))
        deadline = time.monotonic() + max(0.0, wait_s)
        device = self._resolve_device(model_ids, prefer or self.default_device)
        dtype = self._dtype(model_ids, device)
        acquired: dict[str, _Loaded] = {}
        reserved = False
        try:
            if activation_bytes:
                with self._cond:
                    self._pin_resident(model_ids, device, dtype, acquired)
                    self._reserve_activation(activation_bytes, deadline)
                    reserved = True
            while True:
                try:
                    for model_id in model_ids:
                        if model_id not in acquired:
                            acquired[model_id] = self._acquire(model_id, device, dtype, activation_bytes, deadline)
                    break
                except _DeviceFallback:
                    self._release(acquired.values())
                    acquired.clear()
                    device = "cpu"
                    dtype = self._dtype(model_ids, device)
                    if activation_bytes:
                        with self._cond:
                            self._pin_resident(model_ids, device, dtype, acquired)
            yield ModelLease(device, dtype, {model_id: acquired[model_id] for model_id in model_ids})
        finally:
            with self._cond:
                self._release(acquired.values())
                if reserved:
                    self._activation_bytes -= activation_bytes
                self._cond.notify_all()

    def _pin_resident(self, model_ids: Sequence[str], device: str, dtype: str,
                      acquired: dict[str, _Loaded]) -> None:
        """Protect this lease's already-resident models before making room; caller holds the lock."""
        for model_id in model_ids:
            self._model(model_id)
            self._check_breaker(model_id)
            key = (model_id, device, dtype)
            entry = self._loaded.get(key)
            if entry is not None:
                entry.refcount += 1
                entry.last_used = self._clock()
                self._loaded.move_to_end(key)
                acquired[model_id] = entry

    def _reserve_activation(self, nbytes: int, deadline: float) -> None:
        """Atomically reserve a lease allowance, respecting the existing eviction rules and bounded wait."""
        if nbytes > self.budget_bytes:
            raise MemoryBudgetExceeded("memory budget: activation allowance exceeds the budget")
        while not self._make_room(nbytes):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise MemoryBudgetExceeded("memory budget: no room for activation allowance; models or leases are pinned")
            self._cond.wait(timeout=min(remaining, 0.25))
        self._activation_bytes += nbytes

    def _release(self, entries: Iterable[_Loaded]) -> None:
        with self._cond:
            for entry in entries:
                entry.refcount = max(0, entry.refcount - 1)
                entry.last_used = self._clock()
            self._cond.notify_all()

    def _acquire(self, model_id: str, device: str, dtype: str, activation_bytes: int, deadline: float) -> _Loaded:
        model = self._model(model_id)
        key = (model_id, device, dtype)
        with self._cond:
            while True:
                self._check_breaker(model_id)
                entry = self._loaded.get(key)
                if entry is not None:
                    entry.refcount += 1
                    entry.last_used = self._clock()
                    self._loaded.move_to_end(key)
                    return entry
                if key in self._loading:  # another thread is loading this exact variant
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise MemoryBudgetExceeded(f"memory budget: {model_id} is still loading; lease wait expired")
                    self._cond.wait(timeout=min(remaining, 0.25))
                    continue
                need = self.declared_bytes(model_id, device, dtype)
                if need + activation_bytes > self.budget_bytes:
                    raise MemoryBudgetExceeded(
                        f"memory budget: {model_id} needs {(need + activation_bytes) // MiB} MB, "
                        f"budget is {self.budget_bytes // MiB} MB")
                if self._make_room(need):  # this lease already owns its activation allowance
                    self._loading[key] = need
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise MemoryBudgetExceeded(f"memory budget: no room for {model_id}; loaded models are pinned")
                self._cond.wait(timeout=min(remaining, 0.25))
        try:
            obj, measured, load_s = self._load(model, device, dtype)
        except BaseException:
            with self._cond:
                self._loading.pop(key, None)
                self._cond.notify_all()
            raise
        with self._cond:
            self._loading.pop(key, None)
            actual = accounted_bytes(device, need, measured)
            if not self._make_room(actual):
                # The load may transiently exceed its estimate. Never grant execution over budget;
                # remove the final local object reference before allocator cleanup. This is not a breaker failure.
                obj = None
                gc.collect()
                self._probe.release(device)
                self._cond.notify_all()
                raise MemoryBudgetExceeded(f"memory budget: measured footprint of {model_id} does not fit")
            entry = _Loaded(model_id, device, dtype, obj, need, measured, load_s, model.keep_warm, refcount=1,
                            last_used=self._clock())
            self._loaded[key] = entry
            self._log.info("models.accounted", model=model_id, device=device, dtype=dtype,
                     weight_floor_bytes=self._weight_floor(model_id, dtype), reservation_bytes=need,
                     measured_bytes=measured, accounted_bytes=entry.nbytes)
            self._cond.notify_all()
            return entry

    def _make_room(self, need: int) -> bool:
        """Evict unpinned, non-keep_warm models (LRU first) only if that frees enough; caller holds the lock."""
        free = self.budget_bytes - self._used_bytes()
        if free >= need:
            return True
        evictable = [k for k, e in self._loaded.items() if e.refcount == 0 and not e.keep_warm]
        if free + sum(self._loaded[k].nbytes for k in evictable) < need:
            return False
        for key in evictable:
            self._evict_key(key)
            if self.budget_bytes - self._used_bytes() >= need:
                break
        return True

    def _evict_key(self, key: tuple[str, str, str]) -> None:
        entry = self._loaded.pop(key)
        entry.obj = None
        gc.collect()
        self._probe.release(entry.device)
        self._log.info("models.evicted", model=entry.model_id, device=entry.device, dtype=entry.dtype,
                 mb=entry.nbytes // MiB)

    def evict(self, model_id: str) -> bool:
        """Evict every unpinned copy of `model_id`; returns False when a copy is pinned or nothing was loaded."""
        with self._cond:
            keys = [k for k in self._loaded if k[0] == model_id]
            if not keys or any(self._loaded[k].refcount for k in keys):
                return False
            for key in keys:
                self._evict_key(key)
            return True

    # ---- loading, breaker, fallback ----
    def _loader(self, model_id: str) -> LoaderFn:
        v1_loader = self._v1_loaders.get(model_id)
        if v1_loader is not None:
            return lambda _mid, _device, _dtype: v1_loader()
        return self._loader_for(model_id)

    def _load(self, model: ManagedModel, device: str, dtype: str) -> tuple[Any, int | None, float]:
        loader = self._loader(model.id)
        before = self._probe.allocated(device)
        t0 = time.perf_counter()
        try:
            obj = loader(model.id, device, dtype)
        except ModelUnavailable:
            raise  # e.g. a dependency the loader verifies itself is not installed: `skipped`, no breaker, no device pin
        except Exception as e:
            if is_oom(e):
                self._probe.release(device)
                raise MemoryBudgetExceeded(f"memory budget: out of memory while loading {model.id}") from e
            if _kind(device) != "cpu" and model.device_fallback and isinstance(e, NotImplementedError | RuntimeError):
                reason = f"{type(e).__name__} on {device}: {e}"[:300]
                self._store.set_pin(model.id, "cpu", reason)
                self._log.warning("models.device_fallback", model=model.id, device=device, reason=reason)
                raise _DeviceFallback(model.id) from e
            self._trip_breaker(model.id, e)
            raise ModelUnavailable(f"model unavailable: {model.id} failed to load") from e
        load_s = time.perf_counter() - t0
        after = self._probe.allocated(device)
        valid_samples = all(isinstance(value, Integral) and not isinstance(value, bool) and value >= 0
                            for value in (before, after))
        measured = int(after - before) if valid_samples and after > before else None
        if measured is not None:
            self._store.record_measured(model.id, _kind(device), measured, load_s, dtype=dtype, device=device)
        self._log.info("models.loaded", model=model.id, device=device, dtype=dtype, seconds=round(load_s, 2),
                 measured_mb=None if measured is None else measured // MiB)
        return obj, measured, load_s

    def _check_breaker(self, model_id: str) -> None:
        until = self._breaker.get(model_id)
        if until is None:
            return
        if self._clock() < until:
            raise ModelUnavailable(f"model unavailable: {model_id} failed to load; retry in {until - self._clock():.0f}s")
        del self._breaker[model_id]

    def _trip_breaker(self, model_id: str, error: BaseException) -> None:
        with self._cond:
            self._breaker[model_id] = self._clock() + self._breaker_s
        self._log.error("models.load_failed", model=model_id, error=f"{type(error).__name__}: {error}"[:300],
                  retry_after_s=self._breaker_s)

    def warm(self, model_id: str, *, prefer: str | None = None, golden: Callable[[Any], float]) -> WarmReport:
        """Load and run a golden input on the preferred device and on CPU; pin to CPU if they disagree by > 0.01."""
        device = self._resolve_device([model_id], prefer or self.default_device)
        if device == "cpu":
            with self.lease([model_id], prefer="cpu") as lease:
                golden(lease.get(model_id))
            return WarmReport(model_id, "cpu", None, self._store.pin(model_id) == "cpu", self._pin_reason(model_id))
        try:
            with self.lease([model_id], prefer=device) as lease:
                if lease.device == "cpu":  # the load itself fell back
                    golden(lease.get(model_id))
                    return WarmReport(model_id, "cpu", None, True, self._pin_reason(model_id))
                accelerated = float(golden(lease.get(model_id)))
        except (NotImplementedError, RuntimeError) as e:
            if is_oom(e):
                raise
            self._store.set_pin(model_id, "cpu", f"golden forward failed on {device}: {type(e).__name__}: {e}"[:300])
            self.evict(model_id)
            with self.lease([model_id], prefer="cpu") as lease:
                golden(lease.get(model_id))
            return WarmReport(model_id, "cpu", None, True, self._pin_reason(model_id))
        with self.lease([model_id], prefer="cpu") as lease:
            reference = float(golden(lease.get(model_id)))
        delta = abs(accelerated - reference)
        if delta > PARITY_TOLERANCE:
            self._store.set_pin(model_id, "cpu", f"parity |Δ raw| {delta:.4f} > {PARITY_TOLERANCE} on {device}")
            self.evict(model_id)
            return WarmReport(model_id, "cpu", delta, True, self._pin_reason(model_id))
        return WarmReport(model_id, device, delta, False)

    def _pin_reason(self, model_id: str) -> str | None:
        return self._store.pin_reason(model_id)

    # ---- v1 registry shim ----
    def get_v1(self, name: str, loader: Callable[[], Any], *, wait_s: float = 30.0) -> Any:
        """Load-only compatibility getter. Inference must use `borrow_v1` to retain its reservation."""
        with self.borrow_v1(name, loader, wait_s=wait_s) as model:
            return model

    @contextmanager
    def borrow_v1(self, name: str, loader: Callable[[], Any], *, wait_s: float = 30.0) -> Iterator[Any]:
        """Pin legacy weights for the entire borrow, including all prediction batches and exceptions.

        Callers must finish using the model within this context and drop their reference before acquiring
        another model. This preserves declared weight accounting; it does not estimate legacy activations.
        """
        model_id = f"v1:{name}"
        with self._cond:
            if model_id not in self._models:
                resolved = self._v1_resolver(name) if self._v1_resolver else ManagedModel(id=model_id)
                self._models[model_id] = ManagedModel(id=model_id, fp32_bytes=resolved.fp32_bytes,
                                                      keep_warm=resolved.keep_warm, device_fallback=False)
            self._v1_loaders[model_id] = loader
        with self.lease([model_id], prefer=self.default_device, wait_s=wait_s) as lease:
            yield lease.get(model_id)

    def v1_loaded(self) -> dict[str, dict]:
        with self._cond:
            return {e.model_id[3:]: {"loaded": True, "load_seconds": round(e.load_s, 2)}
                    for e in self._loaded.values() if e.model_id.startswith("v1:")}


# --- process-wide manager ----------------------------------------------------------------------------------

V1_FIXED_MODEL_IDS: dict[str, tuple[str, ...]] = {
    "yunet": ("yunet.2023mar",),
    "univfd": ("clip.vit_l14_openai", "univfd.fc_probe"),
    "dfdc_effnet_b7": ("dfdc.effnet_b7_555", "dfdc.effnet_b7_999"),
    "text:desklib": ("hf.desklib.ai_text_detector_v1_01",),
    "text:binoculars": ("hf.qwen.qwen2_5_1_5b", "hf.qwen.qwen2_5_1_5b_instruct"),
}


def v1_model_ids(name: str, manifest: Manifest) -> tuple[str, ...]:
    """Manifest models behind a v1 registry name (used for declared footprints)."""
    if name in V1_FIXED_MODEL_IDS:
        return V1_FIXED_MODEL_IDS[name]
    prefix, _, repo = name.partition(":")
    if name == "text:secondary":
        repo = settings.text_secondary_classifier_id or ""
    elif prefix not in ("imgclf", "audio"):
        return ()
    return tuple(m.id for m in manifest.models if m.source.type == "hf" and m.source.repo == repo)


def _import_callable(path: str) -> LoaderFn:
    module_name, _, attr = path.partition(":")
    return getattr(importlib.import_module(module_name), attr)


def manifest_models(manifest: Manifest) -> dict[str, ManagedModel]:
    return {m.id: ManagedModel.from_manifest(m) for m in manifest.models}


def process_budget_bytes(kind: str) -> int:
    """The model memory budget of this process on a device class: the setting, or the automatic rule on what this process
    may use (the accelerator's memory on CUDA is read only when torch is already imported)."""
    vram = None
    torch = sys.modules.get("torch")
    if kind == "cuda" and torch is not None and torch.cuda.is_available():
        vram = int(torch.cuda.get_device_properties(0).total_memory)
    return resolve_budget_bytes(settings.model_memory_budget_mb, kind, total_ram_bytes(), vram)


def footprint_store() -> FootprintStore:
    return FootprintStore(settings.data_dir / "state" / "model_footprints.json")


def build_model_manager() -> ModelManager:
    from app.engine.registry import get_registry

    manifest = get_registry().manifest
    kind = accelerator_kind()
    budget = process_budget_bytes(kind)
    warn_budget_above_limit(budget, kind)
    models = manifest_models(manifest)

    def loader_for(model_id: str) -> LoaderFn:
        from app.engine.fetchers import ensure_verified

        entry = manifest.model(model_id)
        if entry is None or not entry.load.adapter_loader:
            raise ModelUnavailable(f"model unavailable: {model_id} has no adapter_loader")
        ensure_verified(entry)  # ModelNotInstalled → skipped: model not installed; a hash mismatch → failed
        return _import_callable(entry.load.adapter_loader)

    def v1_resolver(name: str) -> ManagedModel:
        ids = v1_model_ids(name, manifest)
        accel = kind in ("cuda", "mps")
        nbytes, keep_warm = 0, False
        for model_id in ids:
            m = models[model_id]
            # v1 runs the Binoculars LLM pair in fp16 on accelerators (core/device.py); everything else fp32
            nbytes += m.fp16_bytes if (name == "text:binoculars" and accel and m.fp16_bytes) else m.fp32_bytes
            keep_warm = keep_warm or m.keep_warm
        return ManagedModel(id=f"v1:{name}", fp32_bytes=nbytes, keep_warm=keep_warm, device_fallback=False)

    manager = ModelManager(budget_bytes=budget, models=models, loader_for=loader_for,
                           store=footprint_store(), probe=TorchMemoryProbe(), default_device=kind,
                           v1_resolver=v1_resolver)
    log.info("models.manager", device=kind, budget_mb=budget // MiB)
    return manager


_manager: ModelManager | None = None
_manager_lock = threading.Lock()


def get_model_manager() -> ModelManager:
    global _manager
    with _manager_lock:
        if _manager is None:
            _manager = build_model_manager()
        return _manager


def reset_model_manager() -> None:
    global _manager
    with _manager_lock:
        _manager = None
