"""What a profile needs of the model memory budget, and which of its detectors can run under it (BUILD_SPEC N-14a).

A server whose budget is too small for a detector skips it on every analysis (`skipped: memory budget`), and until this
plan nothing said so before a result did. The plan answers, for a profile, a device and a budget:

- **each enabled detector's need:** the weights that are resident together while it runs, plus the activation allowance
  of its adapter's plan (`Detector.activation_plan`, the bound the adapter declares; no file is read);
- **whether it can run:** asked of the model manager itself. The plan builds a `ModelManager` with the models, the
  budget and the footprint store of the question and a loader that loads nothing, and takes each detector's lease on it
  exactly as the runner boundary does (`boundary.run_device`, `boundary.planned_allowance`, `ModelManager.lease`). A
  refusal is the manager's own `MemoryBudgetExceeded`, with its message; there is no second copy of the arithmetic that
  could disagree with a run;
- **the smallest budget that runs the whole profile.**

Models that are never evicted once loaded (`keep_warm`) stay beside every later lease, so the leases are taken twice and
the second answer counts: it is the state of a server after its first analyses.

No weights are loaded and nothing is written: the placeholder loads are not measured, so the footprint store stays as it
is. What the plan cannot know: on an accelerator, a load that measures larger than its declaration (the store then holds
the larger number and the next plan reads it) and other processes; on every device, memory outside the allowance
(decoding, CPU buffers). On CPU a recorded load reserves nothing (D-75; BUILD_SPEC N-14c): the plan there is the
declarations, and what a server says at its start that it can run stays true while it runs.
"""
from __future__ import annotations

import threading
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from app.core.config import settings
from app.core.logging import get_logger
from app.engine.boundary import planned_allowance, run_device
from app.engine.detector import Detector, Modality, Profile, RunContext, StageOutputs
from app.engine.models import (
    SILENT,
    FootprintStore,
    LeaseTerms,
    ManagedModel,
    MemoryBudgetExceeded,
    MiB,
    ModelManager,
    ModelUnavailable,
    footprint_store,
    manifest_models,
    memory_for_auto_budget,
    process_budget_bytes,
)
from app.engine.profiles import resolve_profile
from app.engine.registry import DetectorRegistry, get_registry

log = get_logger(__name__)

_ROOMY = 1 << 62  # a budget no profile reaches: what stays loaded when nothing is refused


@dataclass(frozen=True)
class DetectorNeed:
    detector_id: str
    name: str
    modality: str
    device: str  # where its models load; "" for a detector without models
    dtype: str
    models: tuple[tuple[str, int], ...]  # (model id, bytes reserved for its weights), resident together
    activation_bytes: int
    beside_bytes: int  # models of other detectors that are never evicted once loaded and stay beside this lease
    refusal: str | None  # why the detector cannot run: the model manager's message; None when it can
    plannable: bool = True  # False: the refusal is not one of the budget (no budget cures it)

    @property
    def weight_bytes(self) -> int:
        return sum(nbytes for _, nbytes in self.models)

    @property
    def need_bytes(self) -> int:
        return self.weight_bytes + self.activation_bytes

    @property
    def can_run(self) -> bool:
        return self.refusal is None


@dataclass(frozen=True)
class ProfilePlan:
    profile: str
    device: str
    budget_bytes: int
    detectors: tuple[DetectorNeed, ...]  # every enabled detector of the profile, in the registry's order

    @property
    def with_models(self) -> tuple[DetectorNeed, ...]:
        return tuple(d for d in self.detectors if d.models or not d.plannable)

    @property
    def cannot_run(self) -> tuple[DetectorNeed, ...]:
        return tuple(d for d in self.detectors if not d.can_run)

    @property
    def largest(self) -> DetectorNeed | None:
        """The detector that sets the smallest budget."""
        needs = [d for d in self.detectors if d.plannable]
        return max(needs, key=lambda d: d.need_bytes + d.beside_bytes, default=None)

    @property
    def smallest_budget_bytes(self) -> int:
        """The smallest budget under which every detector that a budget can run does run."""
        largest = self.largest
        return 0 if largest is None else largest.need_bytes + largest.beside_bytes

    @property
    def memory_for_default_rule_bytes(self) -> int:
        """The memory (the machine's, or the accelerator's on CUDA) at which `auto` gives the smallest budget."""
        return memory_for_auto_budget(self.device, self.smallest_budget_bytes)

    def to_dict(self) -> dict:
        return {"profile": self.profile, "device": self.device, "budget_mb": self.budget_bytes // MiB,
                "smallest_budget_mb": mb_up(self.smallest_budget_bytes),
                "memory_for_default_rule_mb": mb_up(self.memory_for_default_rule_bytes),
                "cannot_run": [d.detector_id for d in self.cannot_run],
                "detectors": [{"detector_id": d.detector_id, "name": d.name, "modality": d.modality, "device": d.device,
                               "dtype": d.dtype, "models": [{"id": m, "weights_mb": mb_up(n)} for m, n in d.models],
                               "weights_mb": mb_up(d.weight_bytes), "activation_mb": mb_up(d.activation_bytes),
                               "need_mb": mb_up(d.need_bytes), "beside_mb": mb_up(d.beside_bytes), "can_run": d.can_run,
                               "refusal": d.refusal} for d in self.detectors]}


def mb_up(nbytes: int) -> int:
    """Whole MB (2**20 bytes, as the budget setting counts them), rounded up: a budget of that many MB always holds it."""
    return -(-nbytes // MiB)


@dataclass(frozen=True)
class _Ask:
    """One detector's lease as the runner boundary would take it."""

    detector: Detector
    prefer: str
    terms: LeaseTerms | None = None
    allowance: int = 0
    unavailable: str | None = None  # a refusal that is not the budget's

    @property
    def keys(self) -> set[tuple[str, str, str]]:
        return set() if self.terms is None else {(m, self.terms.device, self.terms.dtype) for m, _ in self.terms.weights}


def _context(profile: Profile) -> RunContext:
    """What an activation plan may read before a run: the profile. A plan is the adapter's bound, not a file's."""
    return RunContext(analysis_id="", media=None, profile=profile, device=None, dtype=None, lease=None,  # type: ignore[arg-type]
                      outputs=StageOutputs(), deadline=None, exhibits=None, log=None)  # type: ignore[arg-type]


def _manager(budget_bytes: int, models: Mapping[str, ManagedModel], store: FootprintStore, device: str) -> ModelManager:
    return ModelManager(budget_bytes=budget_bytes, models=models, loader_for=lambda model_id: _placeholder, store=store,
                        default_device=device, logger=SILENT)


def _placeholder(model_id: str, device: str, dtype: str) -> object:
    return object()  # stands for the weights: the manager accounts their declared bytes, nothing is read from disk


def _ask(detector: Detector, ctx: RunContext, manager: ModelManager) -> _Ask:
    prefer = run_device(detector) or manager.default_device
    model_ids = detector.spec.requirements.model_ids
    if not model_ids:
        return _Ask(detector, prefer)
    try:
        allowance, _ = planned_allowance(detector, ctx, manager)
        return _Ask(detector, prefer, manager.lease_terms(model_ids, prefer=prefer), allowance)
    except ModelUnavailable as e:
        return _Ask(detector, prefer, unavailable=str(e))
    except Exception as e:  # a plan that reads more than the profile, or a model the manifest does not hold
        return _Ask(detector, prefer, unavailable=f"model unavailable: no activation plan without a file ({type(e).__name__})")


def _refusal(manager: ModelManager, ask: _Ask) -> str | None:
    """Take the lease as a run does (no wait) and give it back: None, or the manager's refusal."""
    if ask.terms is None:
        return ask.unavailable
    try:
        with manager.lease(ask.detector.spec.requirements.model_ids, prefer=ask.prefer, activation_bytes=ask.allowance,
                           wait_s=0.0):
            return None
    except MemoryBudgetExceeded as e:
        return str(e)


def _settled(manager: ModelManager, asks: list[_Ask]) -> list[str | None]:
    """Every lease once, so that what is never evicted is loaded, then again: the second answers are a running server's."""
    for ask in asks:
        _refusal(manager, ask)
    return [_refusal(manager, ask) for ask in asks]


def plan(profile: Profile | str | None = None, *, device: str | None = None, budget_bytes: int | None = None,
         registry: DetectorRegistry | None = None, models: Mapping[str, ManagedModel] | None = None,
         store: FootprintStore | None = None) -> ProfilePlan:
    """The plan of a profile on a device class under a budget.

    Defaults are this process's: its registry, the device its analyses run on (`app.core.device.DEVICE`), the default
    profile for that device (`VERITAS_DEFAULT_PROFILE`), the budget of `VERITAS_MODEL_MEMORY_BUDGET_MB` and the footprint
    store of the data directory, whose recorded loads raise a reservation on an accelerator as they do in a run."""
    registry = get_registry() if registry is None else registry
    if device is None:
        # as a worker's heartbeat reads it. This imports torch, and it has to: before that import `auto` cannot tell a
        # CUDA host from a CPU host, and a worker has imported it (its first heartbeat) before it builds its manager
        from app.core.device import DEVICE

        kind = str(DEVICE.type)
    else:
        kind = device
    chosen = resolve_profile(settings.default_profile if profile is None else str(Profile(profile).value), kind)
    detectors = [registry.instance(i) for modality in Modality for i in registry.enabled(modality, chosen)]
    # after torch is imported (the device above, or the adapters): the accelerator's memory is read only then
    budget = process_budget_bytes(kind) if budget_bytes is None else int(budget_bytes)
    models = manifest_models(registry.manifest) if models is None else models
    store = footprint_store() if store is None else store

    served = _manager(budget, models, store, kind)
    ctx = _context(chosen)
    asks = [_ask(detector, ctx, served) for detector in detectors]
    refusals = _settled(served, asks)

    roomy = _manager(_ROOMY, models, store, kind)
    _settled(roomy, asks)
    kept = {(e.id, e.device, e.dtype): e.bytes_accounted for e in roomy.snapshot() if e.keep_warm}

    needs = []
    for ask, refusal in zip(asks, refusals, strict=True):
        spec, terms = ask.detector.spec, ask.terms
        needs.append(DetectorNeed(
            detector_id=spec.id, name=spec.name, modality=spec.modality.value,
            device=terms.device if terms else "", dtype=terms.dtype if terms else "",
            models=terms.weights if terms else (), activation_bytes=ask.allowance,
            beside_bytes=sum(nbytes for key, nbytes in kept.items() if key not in ask.keys) if terms else 0,
            refusal=refusal, plannable=ask.unavailable is None))
    return ProfilePlan(chosen.value, kind, budget, tuple(needs))


# --- this server, at a worker's start ----------------------------------------------------------------------------------

def summary(plan: ProfilePlan) -> dict[str, Any]:
    """What a worker's heartbeat carries and the readiness detail prints: the profile, the device, the budget and each
    detector that cannot run under it, with its need and the model manager's reason."""
    refused = plan.cannot_run
    return {"profile": plan.profile, "device": plan.device, "budget_mb": plan.budget_bytes // MiB,
            "detectors": len(plan.detectors),
            "cannot_run": [{"detector_id": d.detector_id, "need_mb": mb_up(d.need_bytes) if d.plannable else None,
                            "reason": d.refusal} for d in refused],
            "largest_need_mb": max((mb_up(d.need_bytes) for d in refused if d.plannable), default=None),
            "smallest_budget_mb": mb_up(plan.smallest_budget_bytes)}


def warn(plan: ProfilePlan) -> int:
    """One warning for each detector of the profile that cannot run under the budget, with both numbers. A profile that
    fits is not logged (the readiness detail says so). Returns the number of warnings."""
    budget_mb = plan.budget_bytes // MiB
    for d in plan.cannot_run:
        log.warning("capacity.detector_cannot_run", detector_id=d.detector_id, profile=plan.profile, device=plan.device,
                    need_mb=mb_up(d.need_bytes) if d.plannable else None, budget_mb=budget_mb, reason=d.refusal,
                    consequence="skipped in every analysis it applies to", table="python -m app.cli models plan")
    return len(plan.cannot_run)


_summary: dict[str, Any] | None = None
_planning: threading.Thread | None = None
_state_lock = threading.Lock()


def process_summary() -> dict[str, Any] | None:
    """The summary of this process's plan; None before a worker started here, while the plan is made, and when it failed."""
    with _state_lock:
        return _summary


def _plan_this_server() -> None:
    global _summary
    try:
        made = plan()
        warn(made)
        found = summary(made)
    except Exception as e:  # noqa: BLE001  a plan that cannot be made never stops a worker; the log says why
        log.error("capacity.plan_failed", error=f"{type(e).__name__}: {e}"[:300])
        found = None
    with _state_lock:
        if threading.current_thread() is _planning:  # a later start has its own plan
            _summary = found


def at_start() -> threading.Thread:
    """Make the plan of this server's default profile, in the background: the first plan of a process imports every adapter
    of the profile (seconds), and a start does not wait for that. The warnings are logged and the summary is kept for the
    heartbeat when it is done. Returns the thread (a test joins it)."""
    global _summary, _planning
    with _state_lock:
        _summary = None
        _planning = thread = threading.Thread(target=_plan_this_server, name="veritas-capacity-plan", daemon=True)
    thread.start()
    return thread


def settle(timeout: float = 60.0) -> dict[str, Any] | None:
    """Wait for the plan a start began (tests, and a caller that wants the answer now)."""
    with _state_lock:
        thread = _planning
    if thread is not None:
        thread.join(timeout)
    return process_summary()
