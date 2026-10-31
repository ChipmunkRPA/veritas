"""Re-fusion (ARCHITECTURE §4.4 "Re-fusion", §3.4 job type `refusion`; RP-06; BUILD_SPEC M-2).

A fuser artefact is applied to the stored raw scores of a published analysis, and the result is published as a new
analysis revision without running detectors.

- **`create_revision`** validates the source (a published v2 analysis with no newer revision), creates the revision row
  (`revision + 1`, `supersedes_id`, the pinned `fuser_sha256`) and copies into it:
  - the detector runs, with their raw-output artefacts indexed for the revision;
  - the persisted detector-side stage outputs (`pipelines.REFUSION_REUSED`).
  Exhibit, artefact and run ids are derived from the analysis id (`exhibits.py`), so the copies are re-keyed
  deterministically for the revision. Binary objects are content-addressed and shared, never rewritten.
- **The `refusion` job** runs the normal pipeline on the revision with the artefact pinned. The copied stages are reused
  (their outputs re-hash), `condition`, `fuse`, `synthesise`, `render` and `publish` recompute. Audio `temporal` also
  recomputes from stored detector window scores and the current content profile, so legacy coverage tables are replaced
  without model inference; video `temporal` remains reused. The copied stages are
  critical and `run_detectors` refuses in refusion mode, so an unusable stored output fails the run.
- **Compatibility** is checked against the source run (D-47; code review P1-6), not against the detectors enabled now: the
  stored scores must come from the same weights the artefact was fitted on. `source_detector_hashes` builds `{detector:
  weights_sha12}` from the source's `detector_runs` and the manifest pins, and a run whose recorded `model_revisions` differ
  from the pins gets a marker no artefact matches. A detector disabled since the source ran still counts; one the source did
  not run does not. An artefact holding a face-dependent detector (or a face gate literal) needs a face-detection detector
  among the source's runs (D-49; `select_for_source`).
- **Provenance** (J-3e): `revision_caveat` names a pipeline or build change since the source ran.
"""
from __future__ import annotations

import hashlib
import json
import uuid
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.canonical import canonical_json
from app.engine import runs
from app.engine.fusion.registry import NOWEIGHTS, Selection, needs_faces, select_fuser
from app.engine.pipelines import REFUSION_OPTIONAL, REFUSION_REUSED
from app.storage.artefacts import index_object, register_artefact
from app.storage.db import Analysis, AnalysisStage, Artefact, DetectorRun, utcnow
from app.storage.objects import ObjectIntegrityError, ObjectNotFound, ObjectStore


class RefusionError(ValueError):
    """The analysis cannot be re-fused (its state, its revisions or the available artefacts)."""


def _derived(new_id: str, old_id: str, prefix: str) -> bytes:
    return hashlib.sha256(f"{new_id}\x1frefusion\x1f{prefix}\x1f{old_id}".encode()).digest()


def _uuid(new_id: str, old_id: str, prefix: str) -> str:
    return str(uuid.UUID(bytes=_derived(new_id, old_id, prefix)[:16], version=4))


def _remap(value: Any, mapping: dict[str, str]) -> Any:
    """Replace string values that are old ids (keys are never ids)."""
    if isinstance(value, dict):
        return {k: _remap(v, mapping) for k, v in value.items()}
    if isinstance(value, list):
        return [_remap(v, mapping) for v in value]
    if isinstance(value, str):
        return mapping.get(value, value)
    return value


def check_source(db: Session, analysis_id: str) -> Analysis:
    source = db.get(Analysis, analysis_id)
    if source is None:
        raise LookupError(f"analysis {analysis_id} not found")
    if source.kind not in REFUSION_REUSED:
        raise RefusionError(f"analyses of kind {source.kind!r} have no v2 pipeline to re-fuse")
    if source.status != "done" or source.result_schema_version != 2 or not source.result_hash:
        raise RefusionError("only a published v2 analysis can be re-fused (status done, schema v2)")
    from app.cases import review

    newer = review.successor(db, source.id)
    if newer is not None:
        raise RefusionError(f"a newer revision ({newer}) exists; re-fuse the latest revision")
    return source


def _pinned_revision(entry: Any) -> str:
    """The revision a detector records for a model (`app.detectors.image.adapters.loaders`): the pinned revision, else the first
    file's sha256."""
    files = list(getattr(entry.source, "files", None) or [])
    return str(entry.source.revision or (files[0].sha256 if files else None) or "")


def source_detector_hashes(db: Session, registry: Any, analysis_id: str) -> dict[str, str]:
    """{detector id: weights_sha12} of the detectors that produced the stored scores of an analysis, with derived features. A
    refusion revision holds copies of its source's runs, so the fuse stage of the revision gets the same answer as the request."""
    from app.bench.score import weights_sha12
    from app.engine.fusion import registry as fusion_registry

    manifest = registry.manifest
    out: dict[str, str] = {}
    for run in runs.current(db, analysis_id):  # the runs the stored scores came from (N-8)
        entry = manifest.detector(run.detector_id)
        if entry is None or not entry.models:
            out[run.detector_id] = NOWEIGHTS
            continue
        try:
            sha12 = weights_sha12(manifest, run.detector_id)
        except KeyError:  # the model left the manifest: no artefact can match it
            continue
        recorded = {str(m): str(rev) for m, rev in (run.model_revisions or {}).items()}
        changed = sorted(m for m, rev in recorded.items() if manifest.model(m) is None or _pinned_revision(manifest.model(m)) != rev)
        out[run.detector_id] = f"revisions changed since the source ran ({', '.join(changed)})" if changed else sha12
    for feature in fusion_registry.derived.FEATURES:  # a derived feature is available with its source detector's weights (H-6)
        if feature.source in out:
            out.setdefault(feature.id, out[feature.source])
    return out


def source_face_detector(db: Session, registry: Any, analysis_id: str) -> bool:
    """Whether the stored runs of an analysis hold a face-detection detector (D-49)."""
    detector_ids = db.scalars(select(DetectorRun.detector_id).where(DetectorRun.analysis_id == analysis_id))
    return any(registry.detector_family(d) == "face-detection" for d in detector_ids)


def select_for_source(db: Session, registry: Any, kind: str, analysis_id: str, pin: str | None = None) -> Selection:
    """The selection of a re-fusion: weights from the source's runs (D-47), face detectors from the source's runs (D-49)."""
    return select_fuser(kind, source_detector_hashes(db, registry, analysis_id), pin=pin,
                        face_detector=source_face_detector(db, registry, analysis_id), needs_faces=needs_faces(registry))


def compatible_selection(db: Session, registry: Any, source: Analysis, pin: str | None = None) -> Selection:
    selection = select_for_source(db, registry, source.kind, source.id, pin)
    if selection.mode != "artefact":
        raise RefusionError(selection.reason)
    return selection


def revision_caveat(db: Session, store: ObjectStore, revision_id: str, pipeline_version: str, git_commit: str | None) -> str | None:
    """The caveat of a re-fused revision whose source ran another pipeline version, or another build (the `git_commit` of the
    source's published manifest; `build_info.git_commit` falls back to the checkout HEAD). None when both match."""
    revision = db.get(Analysis, revision_id)
    source = db.get(Analysis, revision.supersedes_id) if revision is not None and revision.supersedes_id else None
    if source is None:
        return None
    n = source.revision or 1
    if source.pipeline_version != pipeline_version:
        return (f"Re-fused from revision {n} (pipeline {source.pipeline_version}, now {pipeline_version}); the fusion code has changed "
                "since that run.")
    row = db.scalars(select(Artefact).where(Artefact.analysis_id == source.id, Artefact.role == "stage_output",
                                            Artefact.sha256 == source.result_hash)).first() if source.result_hash else None
    try:
        commit = (json.loads(store.read_bytes(row.sha256)).get("manifest") or {}).get("git_commit") if row is not None else None
    except (ObjectNotFound, ObjectIntegrityError, ValueError):
        commit = None
    if commit != git_commit:
        return f"Re-fused from revision {n}; the software build has changed since that run."
    return None


def create_revision(db: Session, store: ObjectStore, analysis_id: str, *, fuser_sha256: str | None) -> Analysis:
    """The revision row with copied detector runs and detector-side stage outputs. The caller commits."""
    from app.cases import review

    source = check_source(db, analysis_id)
    if not review.hold_revision(db, source.id):  # N-9 (PG-16): the base row's lock before the insert, as `rerun.create_revision`
        raise LookupError(f"analysis {analysis_id} not found")
    outputs: dict[str, Any] = {}
    for stage in REFUSION_REUSED[source.kind]:
        row = db.get(AnalysisStage, (source.id, stage))
        artefact = db.get(Artefact, row.output_artefact_id) if row is not None and row.output_artefact_id else None
        if row is None or row.status not in ("done", "reused") or artefact is None:
            raise RefusionError(f"the stored output of stage {stage!r} is not available, so the scores cannot be re-fused")
        try:
            if store.verify(artefact.sha256) != artefact.bytes:
                raise ObjectIntegrityError("size mismatch")
            outputs[stage] = json.loads(store.read_bytes(artefact.sha256))
        except (ObjectNotFound, ObjectIntegrityError):
            raise RefusionError(f"the stored output of stage {stage!r} failed its integrity check") from None
    for stage in REFUSION_OPTIONAL.get(source.kind, ()):  # copied when the source ran them (e.g. the deep second pass)
        row = db.get(AnalysisStage, (source.id, stage))
        artefact = db.get(Artefact, row.output_artefact_id) if row is not None and row.output_artefact_id else None
        if row is None or row.status not in ("done", "reused") or artefact is None:
            continue
        try:
            if store.verify(artefact.sha256) != artefact.bytes:
                raise ObjectIntegrityError("size mismatch")
            outputs[stage] = json.loads(store.read_bytes(artefact.sha256))
        except (ObjectNotFound, ObjectIntegrityError):
            raise RefusionError(f"the stored output of stage {stage!r} failed its integrity check") from None

    new_id = str(uuid.uuid4())
    revision = Analysis(id=new_id, kind=source.kind, filename=source.filename, content_type=source.content_type,
                        size_bytes=source.size_bytes, sha256=source.sha256, status="queued", case_id=source.case_id,
                        media_id=source.media_id, revision=review.next_revision(db, source), supersedes_id=source.id,
                        profile=source.profile, fuser_sha256=fuser_sha256)
    db.add(revision)
    db.flush()

    mapping: dict[str, str] = {}
    for run in runs.current(db, source.id, store):  # not the rows of an interrupted attempt of the source (N-8)
        mapping[run.id] = _uuid(new_id, run.id, "run")
        raw_id = None
        raw = db.get(Artefact, run.raw_output_artefact_id) if run.raw_output_artefact_id else None
        if raw is not None:
            raw_id = index_object(db, raw.sha256, raw.bytes, media_type=raw.media_type, role=raw.role, created_by=raw.created_by,
                                  analysis_id=new_id, artefact_id=_uuid(new_id, raw.id, "raw")).id
        db.add(DetectorRun(id=mapping[run.id], analysis_id=new_id, detector_id=run.detector_id,
                           detector_version=run.detector_version, model_revisions=dict(run.model_revisions or {}),
                           device=run.device, dtype=run.dtype, status=run.status, reason=run.reason, started_at=run.started_at,
                           duration_ms=run.duration_ms, peak_memory_mb=run.peak_memory_mb, raw_output_artefact_id=raw_id))
    for data in outputs.values():
        for exhibit in data.get("exhibits") or []:
            mapping.setdefault(exhibit["id"], "ex_" + _derived(new_id, exhibit["id"], "exhibit").hex()[:24])
        for record in data.get("artefacts") or []:
            mapping.setdefault(record["id"], _uuid(new_id, record["id"], "artefact"))
    for stage, data in outputs.items():
        stored = register_artefact(db, store, canonical_json(_remap(data, mapping)), media_type="application/json",
                                   role="stage_output", created_by=stage, analysis_id=new_id)
        db.add(AnalysisStage(analysis_id=new_id, stage=stage, status="done", attempt=0, started_at=utcnow(),
                             finished_at=utcnow(), output_artefact_id=stored.id))
    db.flush()
    return revision
