"""Re-run with the pinned manifest, and revision deltas (ARCHITECTURE §5.2 `rerun`, §6.5, §6.6; PRD CC-03; BUILD_SPEC X-2).

- **Pins.** A published result's manifest names each detector's version and, for detectors backed by manifest models, the pinned
  weights: the file's SHA-256 when the detector needs one file, else SHA-256 over the sorted `model/path=sha256` lines, with the
  model revisions and licences (`detector_refs`). The decision names its fuser artefact (or the fallback); the manifest names the
  policy thresholds version.
- **Check** (`check`). A re-run is refused, naming each problem, when a pinned detector has left the model manifest, changed
  version or weights, is disabled, or its weights are missing or fail their hash; when the pinned fuser artefact no longer loads;
  or when the policy version is gone. Detectors without pinned weights (model-free ones, test fakes) are checked by version.
- **Revision** (`create_revision`). Revision n+1 of the same media superseding the source; a re-run carries `rerun_pins` for the
  job (`app.api.jobs`): fusion mode and artefact, policy version, profile and devices.
- **Deltas** (`deltas`). A revision against the revision it supersedes: each signal's status, raw score and calibrated
  probability, the decision code and fused probability (`app.reports.reproduce.diff`), the sampled and tracked frame indices of a
  video, and the manifest fields that changed. Tolerance 0.001 when both ran on the same devices, 0.02 across devices (§6.5).
"""
from __future__ import annotations

import hashlib
import shutil
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import settings
from app.engine import fetchers
from app.engine.fusion.registry import select_for_registry
from app.reports.reproduce import diff
from app.storage.db import Analysis, AuditEvent, MediaObject, PolicyThresholdsRow, get_session
from app.storage.files import safe_filename
from app.storage.objects import default_store

SAME_DEVICE_TOLERANCE = 1e-3
CROSS_DEVICE_TOLERANCE = 0.02
MANIFEST_FIELDS = ("veritas_version", "git_commit", "config_sha256", "models_manifest_sha256", "pipeline_version", "profile", "licence_posture", "device", "precision",
                   "library_versions", "calibration_id", "policy_thresholds_id", "template_set", "sampling")
DETECTOR_FIELDS = ("version", "weights_sha256", "revision", "device", "precision")


class RerunRefused(ValueError):
    """The analysis cannot be re-run with its pinned manifest."""

    def __init__(self, message: str, problems: Sequence[str] = ()):
        super().__init__(message)
        self.problems = list(problems)


# --- pins --------------------------------------------------------------------------------------------------------------------

def weights_pin(registry: Any, detector_id: str) -> tuple[str | None, str | None, str | None]:
    """(weights_sha256, revision, licence) of a manifest detector; all None for model-free and override (fake) detectors."""
    manifest = registry.manifest
    det = manifest.detector(detector_id)
    if det is None:
        return None, None, None
    files: list[tuple[str, str, str | None]] = []
    revisions: list[str] = []
    licences: list[str] = []
    for model_id in registry.required_models(det):
        model = manifest.model(model_id)
        if model is None:
            continue
        licences.append(model.licence.spdx)
        if model.source.revision:
            revisions.append(f"{model_id}@{model.source.revision}")
        files += [(model_id, f.path, f.sha256) for f in model.source.files]
    if not files or any(sha is None for _, _, sha in files):
        weights = None
    elif len(files) == 1:
        weights = files[0][2]
    else:
        weights = hashlib.sha256("\n".join(f"{m}/{p}={s}" for m, p, s in sorted(files)).encode("utf-8")).hexdigest()  # type: ignore[type-var]
    return weights, ";".join(revisions) or None, ";".join(dict.fromkeys(licences)) or None


def detector_refs(registry: Any, entries: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """The manifest's detector list (§6.6): version, the card's admission {decision, date} at analysis time (RP-04), pinned
    weights, revision, licence, coverage date, device and precision."""
    refs = []
    for e in entries:
        weights, revision, licence = weights_pin(registry, e["detector_id"])
        det = registry.manifest.detector(e["detector_id"])
        admission = det.admission if det is not None else None
        refs.append({"id": e["detector_id"], "version": e["version"],
                     "admission": {"decision": admission.decision, "date": admission.date} if admission is not None else None,
                     "weights_sha256": weights, "revision": revision,
                     "licence": licence, "coverage_date": det.coverage_date if det is not None else None,
                     "device": e.get("device"), "precision": e.get("dtype")})
    return refs


def devices(manifest: Mapping[str, Any]) -> list[str]:
    return sorted({str(d["device"]) for d in manifest.get("detectors") or [] if d.get("device")})


def pinned_problems(registry: Any, manifest: Mapping[str, Any]) -> list[str]:
    """Why the detectors a manifest pins cannot run again as they ran; empty when they can."""
    problems: list[str] = []
    for ref in manifest.get("detectors") or []:
        detector_id, pinned = ref["id"], ref.get("weights_sha256")
        det = registry.manifest.detector(detector_id)
        if det is None:
            if pinned is not None:
                problems.append(f"{detector_id}: no longer in the model manifest")
            continue
        if det.version != ref.get("version"):
            problems.append(f"{detector_id}: version {det.version} is installed, the analysis ran version {ref.get('version')}")
            continue
        current, _, _ = weights_pin(registry, detector_id)
        if pinned is not None and current != pinned:
            problems.append(f"{detector_id}: the model manifest pins weights {(current or 'none')[:12]}…, the analysis ran {pinned[:12]}…")
            continue
        decision = registry.decision(detector_id)
        if not decision.enabled:
            problems.append(f"{detector_id}: disabled ({'; '.join(decision.reasons)})")
            continue
        for model_id in registry.required_models(det):
            model = registry.manifest.model(model_id)
            if model is None:
                continue
            try:
                fetchers.ensure_verified(model)
            except (fetchers.ModelNotInstalled, fetchers.WeightIntegrityError) as e:
                problems.append(f"{detector_id}: {e}")
    return problems


def check(db: Session, registry: Any, source: Analysis, result: Mapping[str, Any] | None,
          stored: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """The pins of a re-run of `source`, or RerunRefused naming every problem. `stored` are the pins of a revision of
    `source` that was cancelled or failed before it published (K-2): the re-run takes them in place of the source's, and
    they are checked as the source's are (N-10a; code re-verification R2-04: a revision queued with a pin that no longer
    resolves fails before its run)."""
    if source.status != "done" or source.result_schema_version != 2 or not source.result_hash or result is None:
        raise RerunRefused("Only a published v2 analysis has a pinned manifest to re-run (status done, result schema v2).")
    manifest, decision = result.get("manifest") or {}, result.get("decision") or {}
    problems = pinned_problems(registry, manifest)
    mode = decision.get("fusion_mode")
    fuser = decision.get("fuser_sha256") if mode == "artefact" else None
    profile = manifest.get("profile") or source.profile
    if fuser is not None:
        selection = select_for_registry(registry, source.kind, profile, pin=fuser)
        if selection.mode != "artefact":
            problems.append(f"fuser artefact {fuser[:12]}…: {selection.reason}")
    policy_id = manifest.get("policy_thresholds_id")
    if policy_id is not None and db.get(PolicyThresholdsRow, policy_id) is None:
        problems.append(f"policy thresholds {policy_id}: not found")
    pins = {"fusion_mode": mode, "fuser_sha256": fuser, "policy_thresholds_id": policy_id, "profile": profile,
            "devices": devices(manifest)}
    if stored:
        pins = dict(stored)
        kept_fuser = pins.get("fuser_sha256") if pins.get("fusion_mode") == "artefact" else None
        if kept_fuser is not None and kept_fuser != fuser:
            selection = select_for_registry(registry, source.kind, pins.get("profile") or profile, pin=kept_fuser)
            if selection.mode != "artefact":
                problems.append(f"fuser artefact {str(kept_fuser)[:12]}…: {selection.reason}")
        kept_policy = pins.get("policy_thresholds_id")
        if kept_policy is not None and kept_policy != policy_id and db.get(PolicyThresholdsRow, kept_policy) is None:
            problems.append(f"policy thresholds {kept_policy}: not found")
    if problems:
        raise RerunRefused("The analysis cannot be re-run with its pinned manifest: " + "; ".join(problems) + ".", problems)
    return pins


# --- revisions -----------------------------------------------------------------------------------------------------------------

def original_path(db: Session, source: Analysis, sha256: str) -> Path:
    """The verified original of an analysis: the store object once intake or ingest put it there, else the v0.3 upload file."""
    media = db.get(MediaObject, source.media_id) if source.media_id else None
    if media is not None and media.storage_artefact_id is not None:
        return default_store().path_for(sha256)
    return settings.uploads_dir / source.id / safe_filename(source.filename)


def prepare_revision(source_id: str, sha256: str) -> str:
    """The id of a revision that a unit is about to make of `source_id` (N-10a). When uploads and objects are on different
    file systems the revision's upload is a copy, and it is made here, before the unit takes its locks
    (`intake.prepare_upload`); the caller passes the id to `create_revision` and calls `discard_upload` when its unit stores
    nothing. A source that is gone is left for the unit to refuse."""
    from app.cases.intake import prepare_upload

    revision_id = str(uuid.uuid4())
    with get_session() as db:
        source = db.get(Analysis, source_id)
        if source is None:
            return revision_id
        original, filename = original_path(db, source, sha256), safe_filename(source.filename)
    prepare_upload(revision_id, filename, original)
    return revision_id


def create_revision(db: Session, source: Analysis, original: Path, *, principal_id: str, profile: str | None,
                    pins: Mapping[str, Any] | None = None, revision_id: str | None = None) -> Analysis:
    """Revision n+1 of the source's media, superseding it, with the original linked into its upload directory. The row is
    flushed before the link is made, so a concurrent successor of the same revision (unique `analyses.supersedes_id`) raises
    `IntegrityError` with nothing on disk (N-8), or with the copy `prepare_revision` made for `revision_id`, which the
    caller discards. The caller commits, and calls `discard_upload` if its commit fails."""
    from app.cases import review
    from app.cases.intake import _link_upload

    # N-9 (Postgres QA PG-16): the base row's lock before the insert. `analyses.supersedes_id` has no foreign key, so a
    # revision of a base that a purge or a delete is removing would be committed with a `supersedes_id` that names no row
    if not review.hold_revision(db, source.id):
        raise SourceGone(source.id)
    revision = Analysis(id=revision_id or str(uuid.uuid4()), kind=source.kind, filename=source.filename, content_type=source.content_type,
                        size_bytes=source.size_bytes, sha256=source.sha256, status="queued", case_id=source.case_id,
                        media_id=source.media_id, revision=review.next_revision(db, source), supersedes_id=source.id,
                        profile=profile or source.profile, submitted_by=principal_id, rerun_pins=dict(pins) if pins else None)
    db.add(revision)
    db.flush()
    _link_upload(revision.id, safe_filename(source.filename), original)
    return revision


class SourceGone(LookupError):
    """The revision a new revision would supersede no longer exists (purged or deleted while the request waited)."""


def discard_upload(analysis_id: str) -> None:
    """Remove the upload directory linked for a revision whose row was not committed: it holds a link to the original, never
    the only copy."""
    shutil.rmtree(settings.uploads_dir / analysis_id, ignore_errors=True)


def mode_of(db: Session, analysis_id: str) -> str | None:
    """How a revision was made, from its `rerun` event: rerun, reanalyse or refusion."""
    details = db.scalars(select(AuditEvent.details).where(AuditEvent.action == "rerun", AuditEvent.analysis_id == analysis_id)
                         .order_by(AuditEvent.seq.desc()).limit(1)).first()
    return (details or {}).get("mode")


# --- deltas ----------------------------------------------------------------------------------------------------------------------

def sampled_frames(probe: Mapping[str, Any] | None, shots: Mapping[str, Any] | None) -> dict[str, list[int]] | None:
    """The frame indices a video analysis sampled (legacy detectors) and tracked (sparse plan), from its stage outputs."""
    if probe is None:
        return None
    plan = probe.get("plan") or {}
    tracking = (shots or {}).get("plan") or plan
    return {"sampled": list((plan.get("legacy") or {}).get("frames") or []),
            "tracked": list((tracking.get("sparse") or {}).get("frames") or [])}


def _manifest_changes(before: Mapping[str, Any], after: Mapping[str, Any]) -> list[dict[str, Any]]:
    changes = [{"field": k, "before": before.get(k), "after": after.get(k)} for k in MANIFEST_FIELDS if before.get(k) != after.get(k)]
    refs_before = {d["id"]: d for d in before.get("detectors") or []}
    refs_after = {d["id"]: d for d in after.get("detectors") or []}
    for detector_id in sorted(set(refs_before) | set(refs_after)):
        a, b = refs_before.get(detector_id), refs_after.get(detector_id)
        if a is None or b is None:
            changes.append({"field": f"detectors.{detector_id}", "before": "present" if a else "absent",
                            "after": "present" if b else "absent"})
            continue
        changes += [{"field": f"detectors.{detector_id}.{k}", "before": a.get(k), "after": b.get(k)}
                    for k in DETECTOR_FIELDS if a.get(k) != b.get(k)]
    return changes


def deltas(revision: Analysis, source: Analysis, *, before: Mapping[str, Any] | None, after: Mapping[str, Any] | None,
           mode: str | None, before_frames: Mapping[str, Any] | None = None, after_frames: Mapping[str, Any] | None = None,
           tolerance: float | None = None) -> dict[str, Any]:
    """The deltas document of `revision` against `source` (the revision it supersedes)."""
    before_manifest, after_manifest = (before or {}).get("manifest") or {}, (after or {}).get("manifest") or {}
    cross = before is not None and after is not None and devices(before_manifest) != devices(after_manifest)
    tol = tolerance if tolerance is not None else (CROSS_DEVICE_TOLERANCE if cross else SAME_DEVICE_TOLERANCE)
    body: dict[str, Any] = {"analysis_id": revision.id, "compared_with": source.id, "revision": revision.revision, "mode": mode,
                            "status": "ready", "detail": None, "tolerance": tol, "cross_device": cross, "identical": None,
                            "max_delta": None, "deltas": [], "manifest_changes": []}
    if revision.status in ("queued", "running"):
        return {**body, "status": "pending",
                "detail": f"Revision {revision.revision} is {revision.status}; the deltas are computed once it has finished."}
    if revision.status != "done" or after is None:
        return {**body, "status": "failed",
                "detail": revision.error or f"Revision {revision.revision} did not publish a result ({revision.status})."}
    if before is None:
        return {**body, "status": "unavailable",
                "detail": f"Revision {source.revision} has no published v2 result to compare with."}
    rows = [{"item": d.item, "field": d.field, "before": d.packaged, "after": d.rerun, "delta": d.delta, "ok": d.ok}
            for d in diff(dict(before), dict(after), tol)]
    if before_frames is not None or after_frames is not None:
        for name in ("sampled", "tracked"):
            a, b = (before_frames or {}).get(name), (after_frames or {}).get(name)
            rows.append({"item": "frames", "field": name, "before": a, "after": b, "delta": None, "ok": a == b})
    measured = [r["delta"] for r in rows if r["delta"] is not None]
    return {**body, "identical": all(r["ok"] for r in rows), "max_delta": max(measured) if measured else None, "deltas": rows,
            "manifest_changes": _manifest_changes(before_manifest, after_manifest)}
