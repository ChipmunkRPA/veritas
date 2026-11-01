"""Detector registry: licence policy gate, card completeness, lazy adapter import (ARCHITECTURE §2.3–2.4, D-04).

A detector is enabled only when:
- it is not in `deny_detectors` (wins over everything);
- every model it needs has a licence class the policy allows, or the detector is in `allow_detectors` (D-04);
- it is `default_enabled` or in `allow_detectors`;
- its model card is complete (VG-01);
- it has a v2 adapter.
Adapters are imported only by `instance()`, so disabled detectors never import torch-heavy modules.
"""
from __future__ import annotations

import importlib
import threading
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.core.config import settings
from app.engine.detector import Detector, DetectorSpec, Modality, Profile
from app.engine.manifest import DetectorEntry, Manifest, ManifestError, card_gaps, lint, load_manifest

Decision = Literal["allow", "deny"]


class LicencePolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    preset: str | None = None
    commercial: Decision = "allow"
    non_commercial: Decision = "deny"
    unknown: Decision = "deny"
    allow_detectors: list[str] = Field(default_factory=list)
    deny_detectors: list[str] = Field(default_factory=list)

    def allows_class(self, licence_class: str) -> bool:
        return getattr(self, licence_class) == "allow"

    @property
    def label(self) -> str:
        """The licence posture: the preset (or `custom`), marked when per-detector lists modify it (D-34)."""
        return ((self.preset or "custom") + ("+allowlist" if self.allow_detectors else "")
                + ("+denylist" if self.deny_detectors else ""))


PRESETS: dict[str, dict[str, Decision]] = {
    "commercial": {"commercial": "allow", "non_commercial": "deny", "unknown": "deny"},
    "research": {"commercial": "allow", "non_commercial": "allow", "unknown": "allow"},
}


def parse_policy(value: str | Mapping[str, Any] | None) -> LicencePolicy:
    """`VERITAS_LICENCE_POLICY`: a preset name or a JSON/YAML mapping, optionally based on `preset:`."""
    if value is None or (isinstance(value, str) and not value.strip()):
        value = "commercial"
    if isinstance(value, str):
        text = value.strip()
        if text in PRESETS:
            return LicencePolicy(preset=text, **PRESETS[text])
        try:
            loaded = yaml.safe_load(text)
        except yaml.YAMLError as e:
            raise ValueError(f"licence policy is neither a preset ({', '.join(PRESETS)}) nor valid JSON/YAML: {e}") from e
        if not isinstance(loaded, Mapping):
            raise ValueError(f"unknown licence policy preset {text!r} (presets: {', '.join(PRESETS)})")
        value = loaded
    data = dict(value)
    preset = data.get("preset")
    if preset is not None:
        if preset not in PRESETS:
            raise ValueError(f"unknown licence policy preset {preset!r} (presets: {', '.join(PRESETS)})")
        data = {**PRESETS[preset], **data}
    try:
        return LicencePolicy.model_validate(data)
    except ValidationError as e:
        raise ValueError(f"invalid licence policy: {e}") from e


@dataclass(frozen=True)
class Enablement:
    detector_id: str
    enabled: bool
    reasons: tuple[str, ...]  # why the detector is disabled; empty when enabled


class DetectorDisabled(RuntimeError):
    """`instance()` was asked for a detector the policy or card gate does not enable."""


class AdapterError(RuntimeError):
    """An adapter cannot be imported or does not match its manifest entry."""


class DetectorRegistry:
    def __init__(self, manifest: Manifest, policy: LicencePolicy):
        self.manifest = manifest
        self.policy = policy
        self._instances: dict[str, Detector] = {}
        self._override: dict[str, type[Detector]] | None = None
        self._lock = threading.RLock()

    def decision(self, detector_id: str) -> Enablement:
        det = self.manifest.detector(detector_id)
        if det is None:
            raise KeyError(f"unknown detector {detector_id!r}")
        policy = self.policy
        reasons: list[str] = []
        if settings.audio_detector_mode == "candidate" and detector_id in {"aud.gustking_xlsr", "aud.mo_thecreator_w2v2"}:
            reasons.append("retained for audio rollback; local candidate is selected")
        if detector_id in policy.deny_detectors:
            reasons.append("listed in deny_detectors")
        allow_listed = detector_id in policy.allow_detectors
        for model_id in self.required_models(det):
            model = self.manifest.model(model_id)
            if model is None:
                reasons.append(f"unknown model {model_id}")
            elif not policy.allows_class(model.licence.class_) and not allow_listed:
                reasons.append(f"model {model_id} has licence class {model.licence.class_}, denied by the "
                               f"{policy.label} licence policy")
        if not det.default_enabled and not allow_listed:
            reasons.append("not default-enabled and not in allow_detectors")
        if det.superseded_by and self._supersedes(det.superseded_by):
            reasons.append(f"superseded by {det.superseded_by}, which is enabled and installed")
        if gaps := card_gaps(det):
            reasons.append(f"model card incomplete (missing: {', '.join(gaps)})")
        if det.adapter is None:
            reasons.append("no v2 adapter yet (runs in the v1 pipeline)")
        return Enablement(detector_id, not reasons, tuple(reasons))

    def _supersedes(self, detector_id: str) -> bool:
        """The superseding detector is enabled and every model it needs is installed (lint D008 forbids chains)."""
        from app.engine.fetchers import installed

        det = self.manifest.detector(detector_id)
        if det is None or not self.decision(detector_id).enabled:
            return False
        return all((m := self.manifest.model(mid)) is not None and installed(m) for mid in det.models)

    def decisions(self) -> list[Enablement]:
        return [self.decision(d.id) for d in self.manifest.detectors]

    def required_models(self, det: Any) -> list[str]:
        """The detector's models and, transitively, every model they depend on: D-04 applies to each component."""
        seen: list[str] = []
        pending = list(det.models)
        while pending:
            model_id = pending.pop(0)
            if model_id in seen:
                continue
            seen.append(model_id)
            model = self.manifest.model(model_id)
            if model is not None:
                pending.extend(model.depends_on)
        return seen

    def disclosures(self, detector_ids: Sequence[str] | None = None) -> list[str]:
        """D-34 lines for enabled detectors that only `allow_detectors` enables (a model class the policy denies)."""
        ids = [d.id for d in self.manifest.detectors] if detector_ids is None else list(detector_ids)
        out: list[str] = []
        for detector_id in ids:
            det = self.manifest.detector(detector_id)
            if det is None or detector_id not in self.policy.allow_detectors or not self.decision(detector_id).enabled:
                continue
            denied = {m.licence.class_ for m in map(self.manifest.model, self.required_models(det))
                      if m is not None and not self.policy.allows_class(m.licence.class_)}
            if "unknown" in denied:
                out.append(f"allowlisted component with unreviewed training-data terms: {detector_id}")
            if "non_commercial" in denied:
                out.append(f"allowlisted component with a non-commercial licence: {detector_id}")
        return out

    def enabled(self, modality: Modality | str, profile: Profile | str) -> list[str]:
        """Ids of enabled detectors for a modality and profile, in manifest order. No adapter is imported."""
        modality, profile = Modality(modality), Profile(profile)
        with self._lock:
            if self._override is not None:
                return [i for i, cls in self._override.items()
                        if cls.spec.modality == modality and profile in cls.spec.profiles]
        return [d.id for d in self.manifest.detectors
                if d.modality == modality and profile.value in d.profiles and self.decision(d.id).enabled]

    def instance(self, detector_id: str) -> Detector:
        with self._lock:
            if detector_id in self._instances:
                return self._instances[detector_id]
            if self._override is not None:
                if detector_id not in self._override:
                    raise KeyError(f"detector {detector_id!r} is not in the active override set")
                detector = self._override[detector_id]()
            else:
                decision = self.decision(detector_id)
                if not decision.enabled:
                    raise DetectorDisabled(f"detector {detector_id} is disabled: {'; '.join(decision.reasons)}")
                entry = self.manifest.detector(detector_id)
                assert entry is not None
                detector = self._import(entry)
            self._instances[detector_id] = detector
            return detector

    def detector_name(self, detector_id: str) -> str | None:
        """The display name of a detector without instantiating it: the override class's spec, else the manifest entry."""
        with self._lock:
            if self._override is not None and detector_id in self._override:
                return self._override[detector_id].spec.name
        entry = self.manifest.detector(detector_id)
        return entry.name if entry is not None else None

    def detector_family(self, detector_id: str) -> str | None:
        """The family of a detector without instantiating it: the override class's spec, else the manifest entry."""
        with self._lock:
            if self._override is not None and detector_id in self._override:
                return self._override[detector_id].spec.family
        entry = self.manifest.detector(detector_id)
        return entry.family if entry is not None else None

    def adapter_spec(self, detector_id: str) -> DetectorSpec | None:
        """The adapter class's spec without the enablement check (D-49 face rule, J-3e): the override class under `override`,
        else the manifest adapter resolved as `_import` resolves it. None when the manifest does not list the detector or its
        adapter does not import."""
        with self._lock:
            if self._override is not None and detector_id in self._override:
                return self._override[detector_id].spec
        entry = self.manifest.detector(detector_id)
        if entry is None or entry.adapter is None:
            return None
        try:
            return self._adapter_class(entry).spec
        except AdapterError:
            return None

    @staticmethod
    def _import(entry: DetectorEntry) -> Detector:
        return DetectorRegistry._adapter_class(entry)()

    @staticmethod
    def _adapter_class(entry: DetectorEntry) -> type[Detector]:
        assert entry.adapter is not None
        module_name, _, class_name = entry.adapter.partition(":")
        try:
            cls = getattr(importlib.import_module(module_name), class_name)
        except (ImportError, AttributeError) as e:
            raise AdapterError(f"cannot import adapter {entry.adapter} for {entry.id}: {e}") from e
        if not (isinstance(cls, type) and issubclass(cls, Detector)):
            raise AdapterError(f"adapter {entry.adapter} is not a Detector subclass")
        spec = getattr(cls, "spec", None)
        if spec is None or spec.id != entry.id or spec.version != entry.version:
            found = f"{spec.id}@{spec.version}" if spec is not None else "no spec"
            raise AdapterError(f"adapter {entry.adapter} declares {found}, manifest expects {entry.id}@{entry.version}")
        return cls

    @contextmanager
    def override(self, detectors: Sequence[type[Detector]]) -> Iterator[DetectorRegistry]:
        """Temporarily replace the manifest set with the given detector classes (tests, VERITAS_DETECTOR_SET=fake)."""
        with self._lock:
            saved = (self._override, self._instances)
            self._override = {cls.spec.id: cls for cls in detectors}
            self._instances = {}
        try:
            yield self
        finally:
            with self._lock:
                self._override, self._instances = saved


def build_registry(manifest_path: Path | str | None = None, policy: str | Mapping[str, Any] | None = None) -> DetectorRegistry:
    """Load and lint the manifest (startup check) and apply the licence policy."""
    manifest = load_manifest(manifest_path)
    errors = [issue for issue in lint(manifest) if issue.level == "error"]
    if errors:
        raise ManifestError("models manifest failed lint:\n" + "\n".join(map(str, errors)))
    return DetectorRegistry(manifest, parse_policy(settings.licence_policy if policy is None else policy))


TEXT_PRIMARY_DETECTORS = ("txt.desklib_ft_v2", "txt.desklib")  # the fine-tune first: it serves when enabled and installed (D-34)


def text_classifier_status(registry: DetectorRegistry, profile: Profile | str) -> dict[str, Any]:
    """Which primary text classifier serves documents under this profile (D-62): `detector` and `name` of the first enabled
    primary whose files are installed (None when none is), and `fine_tune`: `installed`, `not_installed` (enabled, files
    absent: the base classifier serves) or `not_enabled`."""
    from app.engine.fetchers import installed

    enabled = registry.enabled(Modality.TEXT, profile)

    def has_files(detector_id: str) -> bool:
        det = registry.manifest.detector(detector_id)
        return det is not None and all((m := registry.manifest.model(mid)) is not None and installed(m)
                                       for mid in registry.required_models(det))

    fine_tune_id = TEXT_PRIMARY_DETECTORS[0]
    serving = next((d for d in TEXT_PRIMARY_DETECTORS if d in enabled and has_files(d)), None)
    known = registry.manifest.detector(fine_tune_id) is not None
    fine_tune = ("not_enabled" if not known or not registry.decision(fine_tune_id).enabled
                 else "installed" if has_files(fine_tune_id) else "not_installed")
    return {"detector": serving, "name": registry.detector_name(serving) if serving else None, "fine_tune": fine_tune}


_registry: DetectorRegistry | None = None
_registry_lock = threading.Lock()


def get_registry() -> DetectorRegistry:
    """The process registry. `VERITAS_DETECTOR_SET=fake` replaces the manifest set with the model-free fakes (§9.2)."""
    global _registry
    with _registry_lock:
        if _registry is None:
            _registry = build_registry()
            if settings.detector_set == "fake":
                from app.detectors.fakes import STANDARD_FAKES

                _registry._override = {cls.spec.id: cls for cls in STANDARD_FAKES}
        return _registry


def reset_registry() -> None:
    global _registry
    with _registry_lock:
        _registry = None
