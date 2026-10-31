"""Benchmark scoring through the v2 production pipeline (BUILD_SPEC B-3: "the v2 runner after P-5"; ML §2.9; B-4c video and audio).

`V2Scorer.score(task)` submits the file as an analysis of the scorer's kind (`V2ImageScorer`, `V2VideoScorer`,
`V2AudioScorer`) and runs `pipelines.run_analysis` with the process registry (the licence policy's enabled detectors of that
kind) and ModelManager, as the API worker does. Each detector entry of the kind's entry stages (image `provenance`, `faces`,
`detect`; video `provenance`, `tracks`, `faces`, `detect`; audio `provenance`, `detect`) becomes a score-cache row:
- **key** `<detector_id>@<weights_sha12>` (`noweights` for model-free detectors): the keys the fuser loader matches (M-2);
- **status** `ok` with the aggregate probability. A logit / LLR aggregate is stored as σ(raw); the cache's logit column
  then gives the same z after the ±8 clip. An `ok` run without a probability-scale aggregate → `not_applicable`; a
  probability, logit or LLR aggregate that is present but unusable (non-finite, or a probability outside [0, 1]) → `failed`,
  as the runtime caps it (D-49);
  `not_applicable` and `skipped` as reported; `failed` and `timeout` → `failed`, retried on resume;
- **runtime_ms** from the detector's timings; `aux` holds version, reported status, reason and unit count.

The marker `v2.<kind>_fused@<sha12 of the detector keys and adapter versions>` holds the published probability, the whole
analysis time, the band, fusion mode, profile, condition context and slice id; the video marker also carries the temporal
lanes' features (the V-8 lane sub-fuser inputs). The scorer points `fuser_dir` at an empty directory, so the published
probability is the fallback fuser's and a loaded artefact never feeds back into its own training scores.

An item fails as a whole (and is retried) when no model-backed detector returned `ok`, so a broken model load never
enters the cache as a benchmark score. A recording the audio content gate refuses (music- or silence-dominated) runs no
voice detector by design: its rows are recorded, not failed.

**Isolation.** `settings.data_dir` points at a scratch directory whose `weights` entry links to the real weights
directory. Database, uploads, the artefact store and v1 artefacts live there; each item's files are deleted after its rows
are read, and `close()` restores the settings, releases the models and removes the scratch directory.
"""
from __future__ import annotations

import gc
import hashlib
import math
import mimetypes
import shutil
import time
import uuid
from pathlib import Path
from typing import Any

from app.bench.score import ScoreTask
from app.engine.fusion import derived

V2_FUSED = "v2.image_fused"
V2_FUSED_BY_KIND = {"image": V2_FUSED, "video": "v2.video_fused", "audio": "v2.audio_fused"}
DETECTOR_STAGES = ("provenance", "faces", "detect")
DETECTOR_STAGES_BY_KIND = {"image": DETECTOR_STAGES, "video": ("provenance", "tracks", "faces", "detect"), "audio": ("provenance", "detect")}
FAILED_STATUSES = frozenset({"failed", "timeout"})
PLACEHOLDER_REASON = "the v2 pipeline did not run this detector for this input"  # a row without `aux.reported_status`


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x)) if x >= 0 else math.exp(x) / (1.0 + math.exp(x))


def probability_of(aggregate: dict[str, Any] | None) -> float | None:
    if not aggregate:
        return None
    raw, kind = aggregate.get("raw"), aggregate.get("raw_kind")
    if raw is None or not math.isfinite(float(raw)):
        return None
    if kind == "probability":
        return float(raw) if 0.0 <= float(raw) <= 1.0 else None
    if kind in ("logit", "llr"):
        return _sigmoid(float(raw))
    return None


def detector_row(entry: dict[str, Any], runtime_ms: float | None = None) -> dict[str, Any]:
    status, reason = entry.get("status"), entry.get("reason")
    aggregate = entry.get("aggregate")
    aux = {"version": entry.get("version"), "reported_status": status, "reason": reason,
           "units": len(entry.get("unit_scores") or []),
           # H-6: localiser and forensic statistics (integrity, flagged share, regions …) for the derived partial-edit features
           "metrics": derived.finite_metrics(entry.get("metrics")), "region_area_max": derived.region_area_max(entry)}
    row = {"prob": None, "reason": reason, "aux": aux, "runtime_ms": runtime_ms}
    if status == "ok":
        prob = probability_of(aggregate)
        if prob is not None:
            return {**row, "status": "ok", "prob": prob, "reason": None}
        kind = (aggregate or {}).get("raw_kind")
        if aggregate and kind in ("probability", "logit", "llr"):
            return {**row, "status": "failed", "reason": "the detector returned an unusable score"}
        return {**row, "status": "not_applicable",
                "reason": f"aggregate kind {kind!r} has no probability scale" if aggregate else "the detector reported no aggregate score"}
    if status in ("not_applicable", "skipped"):
        return {**row, "status": status}
    return {**row, "status": "failed", "reason": f"{status}: {reason}" if status in FAILED_STATUSES else f"unknown status {status!r}"}


class V2Scorer:
    """The v2 pipeline of one analysis kind (`run_analysis`, default profile) behind the score-cache `Scorer` protocol."""

    kind = "image"

    def __init__(self, work_dir: Path, *, registry: Any = None, models: Any = None, profile: str | None = None):
        from app.core.config import settings
        from app.engine.fusion.registry import loaded_detector_hashes
        from app.engine.pipelines import default_profile
        from app.engine.registry import get_registry
        from app.storage import db

        self._settings, self._db = settings, db
        self._saved = {"data_dir": settings.data_dir, "fuser_dir": settings.fuser_dir}
        self._saved_db = (db._engine, db._SessionLocal)
        self.work_dir = Path(work_dir).resolve()
        real_weights = settings.weights_dir.resolve()
        real_keys = settings.data_dir / "keys"  # the HMAC key of the weights' verify record (fetchers.VerifyRecord)
        real_keys.mkdir(parents=True, exist_ok=True)
        shutil.rmtree(self.work_dir, ignore_errors=True)
        self.work_dir.mkdir(parents=True)
        (self.work_dir / "weights").symlink_to(real_weights, target_is_directory=True)
        (self.work_dir / "keys").symlink_to(real_keys.resolve(), target_is_directory=True)
        (self.work_dir / "fusion-empty").mkdir()
        settings.data_dir = self.work_dir
        settings.fuser_dir = self.work_dir / "fusion-empty"
        db._engine, db._SessionLocal = None, None
        db.init_db()
        self.registry = registry or get_registry()
        if models is None:
            from app.engine.models import get_model_manager

            models = get_model_manager()
        self.models = models
        self.profile = profile or default_profile().value
        # derived partial-edit features (H-6) are cache rows `derive_features.py` computes from their sources, not runner
        # detectors: they stay out of the keys, so the marker keeps hashing the runner's detectors only
        hashes = {det: sha for det, sha in loaded_detector_hashes(self.registry, self.kind, self.profile).items()
                  if not derived.is_derived(det)}
        self.keys = {det: f"{det}@{sha}" for det, sha in sorted(hashes.items())}
        self.model_backed = {det for det in self.keys if (self.registry.manifest.detector(det) or None) is not None
                             and self.registry.manifest.detector(det).models}
        # keys carry weights only (D-35); the marker also hashes adapter versions, so a new adapter on the same weights
        # (img.univfd 0.4.0, B-4a) rescores every item instead of resuming over the old version's rows
        self.versions = {det: self._version(det) for det in self.keys}
        combined = hashlib.sha256("\n".join(f"{key}:{self.versions[det]}" for det, key in sorted(self.keys.items()))
                                  .encode()).hexdigest()[:12]
        self._marker = f"{V2_FUSED_BY_KIND[self.kind]}@{combined}"

    def _version(self, detector_id: str) -> str:
        entry = self.registry.manifest.detector(detector_id)
        return entry.version if entry is not None else self.registry.instance(detector_id).spec.version

    def marker(self) -> str:
        return self._marker

    def model_keys(self) -> list[str]:
        return sorted(self.keys.values())

    def score(self, task: ScoreTask) -> dict[str, dict]:
        from sqlalchemy import select

        from app.engine.pipelines import run_analysis
        from app.storage.db import Analysis, DetectorRun
        from app.storage.files import safe_filename, sha256_bytes

        data = Path(task.path).read_bytes()
        aid = str(uuid.uuid4())
        name = safe_filename(Path(task.path).name)
        upload = self._settings.uploads_dir / aid
        upload.mkdir(parents=True)
        (upload / name).write_bytes(data)
        with self._db.get_session() as session:
            session.add(Analysis(id=aid, kind=self.kind, filename=name,
                                 content_type=mimetypes.guess_type(name)[0] or "application/octet-stream",
                                 size_bytes=len(data), sha256=sha256_bytes(data), status="queued"))
            session.commit()
        started = time.perf_counter()
        try:
            outcome = run_analysis(aid, profile=self.profile, registry=self.registry, models=self.models)
        finally:
            runtime_ms = (time.perf_counter() - started) * 1000
            self._cleanup(aid)
        if outcome.status != "done":
            raise RuntimeError(f"analysis {outcome.status}: {outcome.error}")
        with self._db.get_session() as session:
            durations = {detector_id: float(ms) for detector_id, ms in session.execute(
                select(DetectorRun.detector_id, DetectorRun.duration_ms).where(DetectorRun.analysis_id == aid)) if ms is not None}
        return self.rows_from_outputs(outcome.outputs, runtime_ms, durations)

    def rows_from_outputs(self, outputs: dict[str, Any], runtime_ms: float,
                          durations: dict[str, float] | None = None) -> dict[str, dict]:
        rows: dict[str, dict] = {}
        ok_models = 0
        for stage in DETECTOR_STAGES_BY_KIND[self.kind]:
            for entry in (outputs.get(stage) or {}).get("detectors") or []:
                key = self.keys.get(entry["detector_id"])
                if key is None:
                    continue  # not an enabled detector of this kind under this registry
                rows[key] = detector_row(entry, (durations or {}).get(entry["detector_id"]))
                ok_models += int(rows[key]["status"] == "ok" and entry["detector_id"] in self.model_backed)
        condition = outputs.get("condition") or {}
        refused = self.kind == "audio" and (condition.get("content") or {}).get("assessable") is False
        if self.model_backed and ok_models == 0 and not refused:
            raise RuntimeError("no model-backed detector returned a score (models missing or failing; see the job log)")
        for key in self.keys.values():
            rows.setdefault(key, {"status": "not_applicable", "prob": None, "runtime_ms": None, "aux": {},
                                  "reason": PLACEHOLDER_REASON})
        fused = outputs.get("fuse") or {}
        decision = fused.get("decision") or {}
        # an artefact publishes a calibrated probability; the fallback fuser only an uncalibrated one (schemas/evidence.py)
        field = next((f for f in ("probability_ai_calibrated", "probability_ai_uncalibrated") if decision.get(f) is not None), None)
        aux = {"band": decision.get("band"), "code": decision.get("code"), "fusion_mode": decision.get("fusion_mode"),
               "probability_field": field, "profile": self.profile, "context": condition.get("context"),
               "slice_id": (condition.get("profile") or {}).get("slice_id")}
        if self.kind == "video":
            aux["lanes"] = [{"lane": lane.get("lane"), "track_id": lane.get("track_id"), "channel": lane.get("channel"),
                             "detectors": lane.get("detectors"), "features": lane.get("features")}
                            for lane in (outputs.get("temporal") or {}).get("lanes") or []]
        if refused:
            aux["content_reason"] = (condition.get("content") or {}).get("reason")
        rows[self._marker] = {
            "status": "ok" if field else "not_applicable",
            "reason": None if field else "the analysis published no probability",
            "prob": decision.get(field) if field else None, "runtime_ms": round(runtime_ms, 1), "aux": aux}
        return rows

    def _cleanup(self, aid: str) -> None:
        settings = self._settings
        for path in (settings.uploads_dir / aid, settings.artifacts_dir / aid, settings.reports_dir / aid):
            shutil.rmtree(path, ignore_errors=True)
        shutil.rmtree(settings.objects_dir, ignore_errors=True)
        settings.objects_dir.mkdir(parents=True, exist_ok=True)

    def close(self) -> None:
        """Release the models, restore the settings and database globals, remove the scratch directory (D-21)."""
        try:
            from app.engine.models import reset_model_manager

            reset_model_manager()
        except Exception:  # noqa: BLE001 - best effort at shutdown
            pass
        engine = self._db._engine
        if engine is not None:
            engine.dispose()
        self._db._engine, self._db._SessionLocal = self._saved_db
        self._settings.data_dir = self._saved["data_dir"]
        self._settings.fuser_dir = self._saved["fuser_dir"]
        shutil.rmtree(self.work_dir, ignore_errors=True)
        gc.collect()
        try:
            import torch

            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except Exception:  # noqa: BLE001
            pass


class V2ImageScorer(V2Scorer):
    kind = "image"


class V2VideoScorer(V2Scorer):
    kind = "video"


class V2AudioScorer(V2Scorer):
    kind = "audio"


V2_SCORERS: dict[str, type[V2Scorer]] = {"image": V2ImageScorer, "video": V2VideoScorer, "audio": V2AudioScorer}
