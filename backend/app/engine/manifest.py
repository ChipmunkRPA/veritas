"""`models.yaml` manifest: schema, loading and invariant lint (ARCHITECTURE §2.3, VG-01, DC-07)."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.core.config import BACKEND_DIR, settings
from app.engine.detector import Hypothesis, Lane, Modality, Profile

DEFAULT_MANIFEST = BACKEND_DIR / "app" / "models.yaml"

LicenceClass = Literal["commercial", "non_commercial", "unknown"]
# ML §3.1 plus VF; `ai_text` added for the document detectors (not in the ARCHITECTURE list)
Threat = Literal["full_synthesis", "face_swap", "partial_edit", "gan_face", "t2v", "voice", "temporal", "lip_sync", "ai_text"]
SourceType = Literal["hf", "url", "gdrive", "torch_hub", "local"]

_HEX40 = re.compile(r"[0-9a-f]{40}")
_HEX64 = re.compile(r"[0-9a-f]{64}")
_ADAPTER = re.compile(r"[A-Za-z_][\w.]*:[A-Za-z_]\w*")


class ManifestError(ValueError):
    """The manifest cannot be read, parsed or validated."""


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class WeightFile(_Strict):
    path: str
    sha256: str | None = None
    bytes: int | None = None


class Source(_Strict):
    type: SourceType
    repo: str | None = None  # hf, torch_hub
    revision: str | None = None  # hf: 40-hex commit
    url: str | None = None  # url
    archive: Literal["zip", "tar"] | None = None
    file_id: str | None = None  # gdrive
    ref: str | None = None  # torch_hub: 40-hex commit
    entrypoint: str | None = None  # torch_hub
    mirror_url: str | None = None
    path: str | None = None  # local: directory relative to VERITAS_DATA_DIR
    produced_by: str | None = None  # local: script that produced the weights
    files: list[WeightFile] = Field(default_factory=list)


class Licence(_Strict):
    spdx: str
    class_: LicenceClass = Field(alias="class")
    notice: str | None = None


class Footprint(_Strict):
    params_m: float | None = None
    fp32_mb: float | None = None
    fp16_mb: float | None = None
    activation_mb_per_item: float | None = None
    estimated: bool = True


class LoadSpec(_Strict):
    adapter_loader: str | None = None
    pickle: bool = False  # pickle=true ⇒ every file needs a sha256


class ModelEntry(_Strict):
    id: str
    source: Source
    depends_on: list[str] = Field(default_factory=list)
    licence: Licence
    footprint: Footprint = Field(default_factory=Footprint)
    devices: dict[str, Literal["ok", "fallback_cpu", "unsupported"]] = Field(default_factory=dict)
    fp16_safe: bool = False
    keep_warm: bool = False
    load: LoadSpec = Field(default_factory=LoadSpec)
    notes: list[str] = Field(default_factory=list)


class Calibration(_Strict):
    method: Literal["platt", "none"] = "none"  # D-14
    artefact: str | None = None  # relative to VERITAS_DATA_DIR


class EvalRow(_Strict):
    bench: str
    slice: str
    auc: float | None = None
    tpr_at_fpr01: float | None = None
    ece: float | None = None
    n: int | None = None
    date: str | None = None


class Admission(_Strict):
    decision: Literal["pending", "admitted", "rejected", "retired"] = "pending"
    date: str | None = None
    reviewed_by: str | None = None
    evidence: str | None = None
    criteria: dict[str, Any] = Field(default_factory=dict)
    # D-51: stem of the served fusion artefact that still holds a rejected / retired detector until its refit
    served_until: str | None = None


class DetectorEntry(_Strict):
    id: str
    adapter: str | None = None  # "module:Class"; null = no v2 adapter yet (the detector runs in the v1 pipeline)
    version: str
    name: str
    modality: Modality
    family: str
    hypotheses: list[Hypothesis] = Field(default_factory=list)
    models: list[str] = Field(default_factory=list)
    profiles: list[str] = Field(default_factory=list)
    default_enabled: bool = False
    # disabled while the named detector is enabled and its models are installed (D-34: base desklib vs the fine-tune)
    superseded_by: str | None = None
    legacy_weight: float | None = None  # v0.3 hand-set fusion weight (config.py), used only on the fallback path
    calibration: Calibration = Field(default_factory=Calibration)
    eval: list[EvalRow] = Field(default_factory=list)  # written by vbench_report.py, never by hand
    known_limitations: list[str] = Field(default_factory=list)
    # --- model card (VG-01, ARCHITECTURE §2.3) ---
    source_url: str | None = None
    threat: list[Threat] = Field(default_factory=list)
    lane: Lane | None = None
    training_data: str | None = None
    coverage_date: str | None = None
    input_constraints: list[str] = Field(default_factory=list)
    contamination_sets: list[str] = Field(default_factory=list)
    validated_slices: list[dict[str, Any]] = Field(default_factory=list)  # written by vbench_report.py
    fp_modes: list[str] = Field(default_factory=list)
    citation: str | None = None
    admission: Admission | None = None


# Card fields that must be present and non-empty before a detector may be enabled. "not established" is an explicit,
# valid value (D-23); an absent or empty field is not.
REQUIRED_CARD_FIELDS = ("source_url", "threat", "training_data", "coverage_date", "input_constraints", "fp_modes",
                        "citation", "admission")


def card_gaps(det: DetectorEntry) -> list[str]:
    gaps = []
    for name in REQUIRED_CARD_FIELDS:
        value = getattr(det, name)
        if value is None or (isinstance(value, str) and not value.strip()) or (isinstance(value, list) and not value):
            gaps.append(name)
    return gaps


class Manifest(_Strict):
    schema_version: Literal[1]
    models: list[ModelEntry] = Field(default_factory=list)
    detectors: list[DetectorEntry] = Field(default_factory=list)

    def model(self, model_id: str) -> ModelEntry | None:
        return next((m for m in self.models if m.id == model_id), None)

    def detector(self, detector_id: str) -> DetectorEntry | None:
        return next((d for d in self.detectors if d.id == detector_id), None)


def parse_manifest(raw: Any, source: str = "<memory>") -> Manifest:
    if not isinstance(raw, dict):
        raise ManifestError(f"manifest {source} must be a YAML mapping")
    try:
        return Manifest.model_validate(raw)
    except ValidationError as e:
        raise ManifestError(f"manifest {source} failed schema validation:\n{e}") from e


def load_manifest(path: Path | str | None = None) -> Manifest:
    p = Path(path) if path else (settings.models_manifest or DEFAULT_MANIFEST)
    try:
        raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    except OSError as e:
        raise ManifestError(f"cannot read manifest {p}: {e}") from e
    except yaml.YAMLError as e:
        raise ManifestError(f"manifest {p} is not valid YAML: {e}") from e
    return parse_manifest(raw, source=str(p))


@dataclass(frozen=True)
class Issue:
    level: Literal["error", "warning"]
    code: str
    subject: str
    message: str

    def __str__(self) -> str:
        return f"{self.level.upper():7} {self.code} {self.subject}: {self.message}"


def lint(manifest: Manifest, *, data_dir: Path | None = None) -> list[Issue]:
    """Manifest invariants 1–4 (ARCHITECTURE §2.3) plus the DC-07 licence and VG-01 card rules."""
    data_dir = data_dir or settings.data_dir
    issues: list[Issue] = []

    def error(code: str, subject: str, message: str) -> None:
        issues.append(Issue("error", code, subject, message))

    model_ids: set[str] = set()
    for m in manifest.models:
        if m.id in model_ids:
            error("M001", m.id, "duplicate model id")
        model_ids.add(m.id)

    for m in manifest.models:
        s = m.source
        if s.type == "hf":
            if not s.repo:
                error("S001", m.id, "hf source needs `repo`")
            if not (s.revision and _HEX40.fullmatch(s.revision)):
                error("S001", m.id, "hf source needs a 40-hex commit `revision` (branch names are not pins)")
        elif s.type == "url":
            if not s.url:
                error("S002", m.id, "url source needs `url`")
        elif s.type == "gdrive":
            if not s.file_id:
                error("S003", m.id, "gdrive source needs `file_id`")
        elif s.type == "torch_hub":
            if not (s.repo and s.entrypoint and s.ref and _HEX40.fullmatch(s.ref)):
                error("S004", m.id, "torch_hub source needs `repo`, `entrypoint` and a 40-hex commit `ref`")
        elif s.type == "local" and not (s.path and s.produced_by):
            error("S005", m.id, "local source needs `path` and `produced_by`")
        if not s.files:
            error("S006", m.id, "source lists no files")
        for f in s.files:
            subject = f"{m.id}/{f.path}"
            if f.sha256 is None:
                if m.load.pickle:
                    error("F002", subject, "pickle: true without a sha256 is a hard error")
                else:
                    error("F001", subject, "file has no sha256")
            elif not _HEX64.fullmatch(f.sha256):
                error("F003", subject, "sha256 must be 64 lowercase hex characters")
        for dep in m.depends_on:
            if dep not in model_ids:
                error("M002", m.id, f"depends_on references unknown model {dep}")

    valid_profiles = {p.value for p in Profile}
    detector_ids: set[str] = set()
    for d in manifest.detectors:
        if d.id in detector_ids:
            error("D001", d.id, "duplicate detector id")
        detector_ids.add(d.id)
        if d.adapter is not None and not _ADAPTER.fullmatch(d.adapter):
            error("D007", d.id, "adapter must be 'module.path:ClassName'")
        for mid in d.models:
            if mid not in model_ids:
                error("D002", d.id, f"references unknown model {mid}")
        if not d.profiles:
            error("D003", d.id, "no profiles")
        if bad := sorted(set(d.profiles) - valid_profiles):
            error("D003", d.id, f"invalid profile name(s): {', '.join(bad)} (valid: quick, full, deep)")
        if d.calibration.method != "none" and not (
            d.calibration.artefact and (data_dir / d.calibration.artefact).is_file()
        ):
            error("D004", d.id, f"calibration artefact {d.calibration.artefact!r} does not exist (or set method: none)")
        if d.default_enabled:
            restricted = [f"{m.id} ({m.licence.class_})" for m in map(manifest.model, d.models)
                          if m is not None and m.licence.class_ != "commercial"]
            if restricted:
                error("D005", d.id, f"default_enabled: true with non-commercial-class model(s) {', '.join(restricted)} "
                                    "(D-04, DC-07)")
            if gaps := card_gaps(d):
                error("D006", d.id, f"default_enabled: true with an incomplete model card (missing: {', '.join(gaps)})")
            if d.admission is not None and d.admission.decision in ("rejected", "retired") and not d.admission.served_until:
                error("D009", d.id, f"default_enabled: true with admission {d.admission.decision} and no served_until (D-51)")
    for d in manifest.detectors:
        if d.superseded_by is None:
            continue
        other = manifest.detector(d.superseded_by)
        if other is None or other.id == d.id:
            error("D008", d.id, f"superseded_by references unknown detector {d.superseded_by}")
        elif other.superseded_by is not None:
            error("D008", d.id, f"superseded_by chains through {other.id}; one level only")
        elif other.modality != d.modality:
            error("D008", d.id, f"superseded_by references {other.id}, a detector of another modality")
    return issues
