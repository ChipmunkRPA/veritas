"""Pipeline definitions (ARCHITECTURE §3.2) and the entry point used by the job handlers.

Image (P-4/P-5), document and video (P-6, on the v0.3 video sampling) stage sets run today, each with the `condition`
stage (M-2) before `fuse`. The video set adds `shots`, `tracks` (V-2), `temporal` and the deep `second_pass` (V-7).

Refusion (M-2, §4.4) runs the same stages on a revision whose detector-side stage outputs were copied from the source
analysis (`app.engine.refusion`). Those stages are made critical, and `run_detectors` refuses to run in refusion mode, so
a stored output that cannot be reused fails the run instead of re-running detectors.
"""
from __future__ import annotations

import dataclasses
from collections.abc import Callable
from functools import partial
from typing import Any

from sqlalchemy.orm import Session

from app.core.config import settings
from app.engine import synthesis
from app.engine.boundary import Deadline
from app.engine.detector import Profile
from app.engine.profiles import resolve_profile
from app.engine.registry import DetectorRegistry, get_registry
from app.engine.runner import Runner, RunOutcome, StageDef
from app.engine.stages import audio as aud
from app.engine.stages import common
from app.engine.stages import condition as cond
from app.engine.stages import document as doc
from app.engine.stages import image as img
from app.engine.stages import video as vid
from app.policy import thresholds as policy_thresholds
from app.storage.db import Analysis, get_session
from app.storage.objects import ObjectStore, default_store

IMAGE_BUDGET_BYTES = 20 * 1024 * 1024  # ARCHITECTURE §4.3 storage budget per image analysis
DOCUMENT_BUDGET_BYTES = 20 * 1024 * 1024  # §4.3 lists no document budget; JSON exhibits and one plot (assumption)
VIDEO_BUDGET_BYTES = 150 * 1024 * 1024  # ARCHITECTURE §4.3 storage budget per video analysis
AUDIO_BUDGET_BYTES = 20 * 1024 * 1024  # §4.3 lists no audio budget; JSON series, segments and tables only (assumption)
BUDGET_BYTES = {"image": IMAGE_BUDGET_BYTES, "document": DOCUMENT_BUDGET_BYTES, "video": VIDEO_BUDGET_BYTES,
                "audio": AUDIO_BUDGET_BYTES}

IMAGE_STAGES: tuple[StageDef, ...] = (
    StageDef("ingest", img.ingest, critical=True, weight=0.5),
    StageDef("probe", img.probe, after=("ingest",), weight=0.5),
    StageDef("provenance", img.provenance, after=("probe",), reuse_if=img.current_provenance_output),
    StageDef("decode", img.decode, after=("ingest",), persist=False, critical=True),
    StageDef("faces", img.faces, after=("decode",), lane="accelerator"),
    StageDef("condition", cond.image, after=("probe", "decode", "faces"), weight=0.5),
    StageDef("detect", img.detect, after=("decode", "faces", "probe"), lane="mixed", weight=4.0),
    StageDef("localise", img.localise, after=("detect",)),
    StageDef("fuse", img.fuse, after=("provenance", "detect", "localise", "condition")),
    StageDef("synthesise", img.synthesise, after=("fuse",), weight=0.5, reuse_if=common.current_synthesis_output),
    StageDef("render", img.render, after=("localise", "synthesise"), weight=0.5),
    StageDef("publish", img.publish, after=("render",), critical=True, weight=0.5),
)
DOCUMENT_STAGES: tuple[StageDef, ...] = (
    StageDef("ingest", common.ingest, critical=True, weight=0.5),
    StageDef("decode", doc.decode, after=("ingest",), critical=True, weight=0.5, reuse_if=doc.current_decode_output),
    # Revalidate stored text on every resume/refusion before any cached result can be reused.
    StageDef("condition", doc.condition, after=("decode",), persist=False, critical=True, weight=0.5),
    StageDef("detect", doc.detect, after=("decode", "condition"), lane="mixed", weight=4.0),
    StageDef("fuse", doc.fuse, after=("decode", "detect", "condition")),
    StageDef("synthesise", doc.synthesise, after=("fuse",), weight=0.5, reuse_if=common.current_synthesis_output),
    StageDef("render", doc.render, after=("fuse", "synthesise"), weight=0.5),
    StageDef("publish", doc.publish, after=("render",), critical=True, weight=0.5),
)
FULL_DEEP = frozenset({Profile.FULL, Profile.DEEP})
DEEP_ONLY = frozenset({Profile.DEEP})
VIDEO_STAGES: tuple[StageDef, ...] = (
    StageDef("ingest", common.ingest, critical=True, weight=0.5),
    StageDef("probe", vid.probe, after=("ingest",), critical=True, weight=0.5),
    StageDef("provenance", vid.provenance, after=("probe",), weight=0.5, reuse_if=vid.current_provenance_output),
    StageDef("decode", vid.decode, after=("probe",), persist=False, critical=True),
    StageDef("shots", vid.shots, after=("probe",), profiles=FULL_DEEP, weight=0.5),
    StageDef("tracks", vid.tracks, after=("probe", "shots"), lane="accelerator", profiles=FULL_DEEP, weight=2.0),
    StageDef("faces", vid.faces, after=("decode",)),
    StageDef("condition", cond.video, after=("probe", "decode", "faces"), weight=0.5),
    StageDef("detect", vid.detect, after=("decode", "faces", "probe", "shots", "tracks"), lane="mixed", weight=6.0),
    StageDef("temporal", vid.temporal, after=("probe", "tracks", "detect"), weight=0.5),
    StageDef("second_pass", vid.second_pass, after=("probe", "decode", "tracks", "temporal"), lane="mixed", profiles=DEEP_ONLY,
             weight=2.0),
    StageDef("fuse", vid.fuse, after=("probe", "decode", "faces", "tracks", "provenance", "detect", "condition", "temporal",
                                      "second_pass")),
    StageDef("synthesise", vid.synthesise, after=("fuse",), weight=0.5, reuse_if=common.current_synthesis_output),
    StageDef("render", vid.render, after=("decode", "fuse", "synthesise")),
    StageDef("publish", vid.publish, after=("render",), critical=True, weight=0.5),
)
AUDIO_STAGES: tuple[StageDef, ...] = (
    StageDef("ingest", common.ingest, critical=True, weight=0.5),
    StageDef("probe", aud.probe, after=("ingest",), critical=True, weight=0.5),
    StageDef("provenance", aud.provenance, after=("probe",), weight=0.5),
    StageDef("decode", aud.decode, after=("probe",), persist=False, critical=True),
    StageDef("condition", cond.audio, after=("probe", "decode"), critical=True, weight=0.5),  # detect reads its speech gate
    StageDef("detect", aud.detect, after=("probe", "decode", "condition"), lane="mixed", weight=4.0),
    StageDef("temporal", aud.temporal, after=("condition", "detect"), weight=0.5),
    StageDef("fuse", aud.fuse, after=("probe", "decode", "provenance", "detect", "condition", "temporal")),
    StageDef("synthesise", aud.synthesise, after=("fuse",), weight=0.5, reuse_if=common.current_synthesis_output),
    StageDef("render", aud.render, after=("fuse", "synthesise"), weight=0.5),
    StageDef("publish", aud.publish, after=("render",), critical=True, weight=0.5),
)
PIPELINES: dict[str, tuple[StageDef, ...]] = {"image": IMAGE_STAGES, "document": DOCUMENT_STAGES, "video": VIDEO_STAGES,
                                              "audio": AUDIO_STAGES}
PIPELINE_VERSIONS = {"image": img.PIPELINE_VERSION, "document": doc.PIPELINE_VERSION, "video": vid.PIPELINE_VERSION,
                     "audio": aud.PIPELINE_VERSION}
# Persisted detector-side stages a refusion revision copies from its source analysis.
# Audio temporal evidence is regenerated from stored window scores and the recomputed content profile.
REFUSION_REUSED: dict[str, tuple[str, ...]] = {
    "image": ("ingest", "probe", "provenance", "faces", "detect", "localise"),
    "document": ("ingest", "decode", "detect"),
    "video": ("ingest", "probe", "provenance", "shots", "tracks", "faces", "detect", "temporal"),
    "audio": ("ingest", "probe", "provenance", "detect"),
}
# Detector-side stages copied only when the source ran them (the deep-only second pass); otherwise the revision skips them
REFUSION_OPTIONAL: dict[str, tuple[str, ...]] = {"video": ("second_pass",)}


def default_profile() -> Profile:
    from app.core.device import DEVICE

    return resolve_profile(settings.default_profile, str(DEVICE))


_default_profile = default_profile


def run_analysis(analysis_id: str, *, deadline: Deadline | None = None, profile: Profile | str | None = None,
                 registry: DetectorRegistry | None = None, models: Any = None,
                 sessions: Callable[[], Session] | None = None, store: ObjectStore | None = None,
                 refusion: bool = False, fuser_pin: str | None = None, force_fallback: bool = False,
                 policy_thresholds_id: str | None = None) -> RunOutcome:
    """Run the v2 pipeline of an analysis. `fuser_pin` / `force_fallback` fix the fusion mode and `policy_thresholds_id` the
    policy version (a re-run, CC-03); by default the newest compatible artefact and the policy version in force apply."""
    sessions = sessions or get_session
    store = store or default_store()
    with sessions() as db:
        analysis = db.get(Analysis, analysis_id)
        if analysis is None:
            raise LookupError(f"analysis {analysis_id} not found")
        kind = analysis.kind
        stages = PIPELINES.get(kind)
        if stages is None:
            raise ValueError(f"no v2 pipeline for kind {kind!r}")
        # A publish may commit before the runner marks its stage/analysis done. A
        # result hash is therefore authoritative even after an interrupted attempt.
        # Changed code/templates require a new revision, never an in-place rewrite.
        if analysis.result_hash and (analysis.pipeline_version != PIPELINE_VERSIONS[kind]
                                     or analysis.template_set != synthesis.TEMPLATE_SET):
            raise ValueError("Published analyses require a new revision when the pipeline or findings template changes")
        policy = policy_thresholds.resolve(policy_thresholds_id, sessions)
        resolved = Profile(profile) if profile else default_profile()
        analysis.profile = resolved.value
        analysis.pipeline_version = PIPELINE_VERSIONS[kind]
        analysis.attempts = (analysis.attempts or 0) + 1
        db.commit()
    active_registry = registry or get_registry()
    if kind in ("image", "video"):
        detector_id = "prov.c2pa" if kind == "image" else "prov.c2pa.video"
        # Refusion copies historical detector evidence intentionally. Fresh/resumed execution
        # expects a C2PA row only when this registry/profile actually schedules provenance.
        reuse = None
        if not refusion:
            expected = (detector_id in active_registry.enabled(kind, resolved)
                        and active_registry.detector_family(detector_id) == "provenance")
            predicate = img.current_provenance_output if kind == "image" else vid.current_provenance_output
            reuse = partial(predicate, c2pa_expected=expected)
        stages = tuple(dataclasses.replace(stage, reuse_if=reuse) if stage.id == "provenance" else stage
                       for stage in stages)
    if refusion:
        if kind == "audio":
            # Fresh detection needs the speech gate; copied scores do not. Recomputing condition must not invalidate
            # the integrity-checked stored detect output and trigger forbidden inference. Temporal still waits for both.
            stages = tuple(dataclasses.replace(s, after=tuple(d for d in s.after if d != "condition"))
                           if s.id == "detect" else s for s in stages)
        stages = tuple(dataclasses.replace(s, critical=True) if s.id in REFUSION_REUSED[kind] else s for s in stages)
    state: dict[str, Any] = {"registry": active_registry, "run_store": common.DbRunStore(sessions, store),
                             "models": models, "kind": kind, "refusion": refusion, "fuser_pin": fuser_pin,
                             "force_fallback": force_fallback, "policy": policy.as_json()}
    runner = Runner(stages, sessions=sessions, store=store, exhibit_budget_bytes=BUDGET_BYTES[kind])
    try:
        return runner.run(analysis_id, profile=resolved, deadline=deadline, state=state)
    finally:
        common.cleanup(state, keep_copy=bool(getattr(deadline, "lease_lost", False)))
