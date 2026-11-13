"""Detector plugin interface (ARCHITECTURE §2.2).

Adapters implement `Detector`. The runner (§2.6) calls `applicable` → lease → `load` → `run` and always persists the
`DetectorResult`. This module imports no torch: the runner passes the resolved device and dtype in `RunContext`.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, ClassVar, Literal, Protocol


class Modality(StrEnum):
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"
    TEXT = "text"
    CONTAINER = "container"


class Hypothesis(StrEnum):  # EA-03
    H1_WHOLE_CONTENT_SYNTHESIS = "H1_whole_content_synthesis"  # t2i, t2v, gan_face, AI-written text
    H2_LOCALISED_MANIPULATION = "H2_localised_manipulation"  # decided per lane
    H3_PROVENANCE_INTEGRITY = "H3_provenance_integrity"  # documentary; not a fused probability


class Lane(StrEnum):  # H2 lanes (VF §8.1, ML §4.1 sub-fusers)
    FACE_SWAP = "face_swap"
    LIP_SYNC = "lip_sync"
    PARTIAL_EDIT = "partial_edit"
    VOICE = "voice"


class Profile(StrEnum):  # D-08
    QUICK = "quick"
    FULL = "full"
    DEEP = "deep"


class InputUnit(StrEnum):
    FILE = "file"  # path + bytes: metadata, C2PA, codec forensics
    IMAGE = "image"  # decoded analysis copy (+ original-size scale)
    FACE_CROP = "face_crop"  # aligned crop per face (image) or per sampled track frame (video)
    FRAME_BATCH = "frame_batch"  # sampled frames with pts
    TRACK_CLIP = "track_clip"  # T×H×W crops over a track window
    CLIP = "clip"  # whole-frame clip windows (synthetic-video detectors)
    AUDIO_WINDOW = "audio_window"  # 16 kHz mono windows with start/end pts
    AV_WINDOW = "av_window"  # mouth crops + aligned audio (lip-sync)
    DOCUMENT = "document"
    SENTENCES = "sentences"


DetectorStatus = Literal["ok", "not_applicable", "skipped", "failed", "timeout"]
RawKind = Literal["probability", "logit", "distance", "llr", "score"]
ProvenanceOverride = Literal["decisive_ai", "self_declared_ai", "trusted_capture"]  # D-15


@dataclass(frozen=True)
class ActivationUse:
    """Declared processing allowance for an adapter's unchanged execution envelope.

    ``max_items`` counts simultaneous model inputs in ``unit`` (or one explicitly
    documented composite work item). Manifest bytes remain estimates; this is not
    a measured peak, a process-memory ceiling, or permission to change batching.
    """

    model_id: str
    max_items: int
    unit: str
    input_bound: str
    limitations: tuple[str, ...] = ()


@dataclass(frozen=True)
class Requirements:
    model_ids: tuple[str, ...]  # keys into models.yaml `models:`
    needs: tuple[str, ...] = ()  # stage outputs: "probe", "faces", "tracks", "frames", "audio", "document"
    devices: tuple[str, ...] = ("cuda", "mps", "cpu")
    max_batch: int = 8
    timeout_s: float = 60.0  # at batch size 1 on the reference host; scaled by profile and device (profiles.py)
    isolation: Literal["thread", "process"] = "thread"
    lane: Literal["accelerator", "cpu"] = "accelerator"


@dataclass(frozen=True)
class DetectorSpec:
    id: str  # stable, dotted: "img.dda", "vid.gend", "aud.w2v2_xlsr", "prov.c2pa"
    version: str  # adapter semver; bump when pre/post-processing changes
    name: str
    modality: Modality
    input_unit: InputUnit
    family: str
    hypotheses: tuple[Hypothesis, ...]
    produces: tuple[str, ...]  # exhibit kinds
    requirements: Requirements
    profiles: frozenset[Profile]


@dataclass(frozen=True)
class Applicability:
    ok: bool
    reason: str | None = None  # "no face ≥ 48 px", "no audio track", "image < 224 px"


class MediaHandle(Protocol):
    """Access the bitstream, native pixels, decoded metadata, and ≤ 4096 px analysis copy."""

    sha256: str
    kind: str

    def original_bytes(self) -> bytes | memoryview: ...

    def native_pixels(self, tile: tuple[int, int, int, int] | None = None) -> Any: ...

    def info(self) -> dict[str, Any]: ...

    def analysis_copy(self) -> tuple[Any, float]: ...  # (array, scale_to_original)


class ModelLease(Protocol):
    device: str

    def get(self, model_id: str) -> Any: ...


class Deadline(Protocol):
    def check(self, label: str = "") -> None: ...

    def remaining(self) -> float | None: ...


class ExhibitWriter(Protocol):
    """Implemented by `app.engine.exhibits.ExhibitWriter` (store-backed) and `fakes.InMemoryExhibitWriter`."""

    def heatmap(self, values: Any, *, semantics: str, title: str, mask: Any = None, **meta: Any) -> str: ...

    def image(self, data: bytes, *, role: str, media_type: str, title: str, **meta: Any) -> str: ...

    def regions(self, regions: list[dict[str, Any]], *, title: str, **meta: Any) -> str: ...

    def series(self, series: list[dict[str, Any]], *, x_unit: str, title: str, **meta: Any) -> str: ...

    def segments(self, segments: list[dict[str, Any]], *, lane: str, title: str, **meta: Any) -> str: ...

    def table(self, columns: list[str], rows: list[list[Any]], *, title: str, **meta: Any) -> str: ...

    def add(self, kind: str, *, title: str, media_ref: str, **fields: Any) -> str: ...


@dataclass
class StageOutputs:
    probe: Any = None
    faces: Any = None
    tracks: Any = None
    frames: Any = None
    audio: Any = None
    document: Any = None
    clips: Any = None  # video: per-shot clip inputs for the synthesis lane (V-4)
    av: Any = None  # video: tracks, sound track and frames for the lip-sync lane (V-5)
    codec: Any = None  # video: original file, frame table and shot cuts for the codec lane (V-6)
    face_review: Any = None  # source-bound consecutive crops; separate from authenticity fusion


@dataclass
class RunContext:
    analysis_id: str
    media: MediaHandle
    profile: Profile
    device: Any  # torch.device resolved per detector by the ModelManager; the only device source
    dtype: Any
    lease: ModelLease
    outputs: StageOutputs
    deadline: Deadline
    exhibits: ExhibitWriter
    log: Any  # structlog BoundLogger bound to analysis_id and detector_id
    seed: int = 0
    batcher: Callable[[InputUnit, int], Iterator[Any]] | None = None

    def batches(self, unit: InputUnit, max_batch: int) -> Iterator[Any]:
        """Memory-aware micro-batches that halve on OOM; bound by the runner from the ModelManager."""
        if self.batcher is None:
            raise RuntimeError("no batcher is bound to this RunContext")
        return self.batcher(unit, max_batch)


@dataclass
class UnitScore:
    unit_ref: str  # "global" | "face:0" | "track:2@pts=90090" | "window:3" | "sentence:17"
    raw: float
    raw_kind: RawKind
    quality: dict[str, float] = field(default_factory=dict)  # face_px, jpeg_q, upscale, snr ...


# The reasons of a skipped run that are the server's and not the file's (BUILD_SPEC N-14a): the model memory budget is too
# small for the detector (`models.MemoryBudgetExceeded`), or its weights are not installed (`fetchers.ModelNotInstalled`).
# Results and reports name the server for these, and their next step is the administrator's.
SERVER_REASONS: tuple[str, ...] = ("memory budget", "model not installed")
ON_THIS_SERVER = "could not run on this server"
SERVER_FEWER = "This result rests on fewer checks than the profile holds."
# N-19k: what a caveat says of a check for which neither a run nor the registry holds a name; never the check's id
UNNAMED_CHECK = "a check without a recorded name"


def of_the_server(status: Any, reason: Any) -> bool:
    return status == "skipped" and reason in SERVER_REASONS


@dataclass
class DetectorResult:
    status: DetectorStatus
    reason: str | None = None  # user-facing, sanitised
    unit_scores: list[UnitScore] = field(default_factory=list)
    aggregate: UnitScore | None = None
    observed: str = ""
    reference: str = ""
    reasoning: str = ""  # templated text; no probability words (calibration is downstream)
    metrics: dict[str, float] = field(default_factory=dict)
    # N-19n (D3-07, 5): what a check found and where, in its own words, for the finding's sentences (the metrics hold
    # numbers only); free-form, JSON, never read by the fusion. The entry and the signal copy it (`SignalV2.details`)
    details: dict[str, Any] = field(default_factory=dict)
    exhibit_ids: list[str] = field(default_factory=list)
    model_revisions: dict[str, str] = field(default_factory=dict)
    device: str = ""
    dtype: str = ""
    timings: dict[str, float] = field(default_factory=dict)
    peak_memory_mb: float | None = None
    provenance_override: ProvenanceOverride | None = None

    def __post_init__(self) -> None:
        # N-19o (SEC5-01): what a check observed can hold text from the file; it reaches findings and signed reports
        from app.storage.files import untrusted_text

        self.observed = untrusted_text(self.observed)

    @classmethod
    def not_applicable(cls, reason: str) -> DetectorResult:
        return cls(status="not_applicable", reason=reason)

    @classmethod
    def skipped(cls, reason: str) -> DetectorResult:
        return cls(status="skipped", reason=reason)

    @classmethod
    def failed(cls, reason: str) -> DetectorResult:
        return cls(status="failed", reason=reason)

    @classmethod
    def timeout(cls, reason: str) -> DetectorResult:
        return cls(status="timeout", reason=reason)


class Detector(ABC):
    spec: ClassVar[DetectorSpec]

    def applicable(self, ctx: RunContext) -> Applicability:
        return Applicability(True)

    def activation_plan(self, ctx: RunContext) -> tuple[ActivationUse, ...]:
        """Cover each leased model explicitly; model-free detectors need no allowance.

        The boundary refuses model-backed adapters without a complete declaration.
        Counts must describe actual adapter limits, never the generic max_batch.
        """
        return ()

    def load(self, lease: ModelLease) -> None:  # noqa: B027 (optional hook)
        """Bind loaded weights from the lease; no I/O except through the lease."""

    @abstractmethod
    def run(self, ctx: RunContext) -> DetectorResult: ...
