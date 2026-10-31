"""Score cache (ML §2.9) and the benchmark scoring path.

- One parquet per detector and weights: `scores/<detector_id>@<weights_sha12>.parquet`, keyed by (item_id, variant,
  transform). Columns: ML's `status`, `prob`, `logit`, `aux` (JSON) and `runtime_ms`, plus `reason` and `scored_at`.
- `run_scoring` skips keys the scorer's marker file already holds with a final status (ok / not_applicable /
  skipped); `failed` rows are retried. The cache is checkpointed every `checkpoint_every` scored items and at the end,
  and the marker file is written last, so a crash between files never marks an item done.
- `V1ImageScorer` scores through the production path, `load_image` + `analyze_image_array(fast=True)` (v0.3), until P-5
  moves images to the v2 runner. A v1 signal backed by a registry detector becomes a row of that detector;
  model-free signals get `v1.<signal_id>@noweights` files; the fused verdict is the marker `v1.image_fused@<sha12>`.
  v1 does not time detectors separately, so only the marker row carries `runtime_ms` (the whole call).
- `V1VideoScorer` (`score_video` + `assemble_video`, no artefacts) and `V1AudioScorer` (`audio_signals`, the v0.3
  sound-track path, on an audio file) give the B-5 `baseline-v0.3` video and audio columns. A frame detector's row is
  its mean over the sampled frames it scored, a voice model's row its chunk mean (the v2 adapters' aggregates); the
  fused verdicts are `v1.video_fused@<sha12>` and `v1.audio_fused@<sha12>` (for audio, the fused synthetic-voice signal).
"""
from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import re
import time
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from app.engine.manifest import Manifest, load_manifest

FINAL_STATUSES = frozenset({"ok", "not_applicable", "skipped"})
V1_FUSED = "v1.image_fused"
V1_VIDEO_FUSED = "v1.video_fused"
V1_AUDIO_FUSED = "v1.audio_fused"
_KEY = re.compile(r"[a-z0-9][a-z0-9._-]*@[0-9a-z]+")


def score_schema():
    import pyarrow as pa

    s, f = pa.string(), pa.float64()
    return pa.schema([("item_id", s), ("variant", s), ("transform", s), ("status", s), ("reason", s), ("prob", f),
                      ("logit", f), ("aux", s), ("runtime_ms", f), ("scored_at", s)])


def logit(p: float | None) -> float | None:
    if p is None:
        return None
    q = min(max(float(p), 1e-6), 1 - 1e-6)
    return math.log(q / (1 - q))


class ScoreCache:
    """Per-detector score tables held in memory and rewritten atomically on flush."""

    def __init__(self, scores_dir: Path):
        self.dir = scores_dir
        self._rows: dict[str, dict[tuple[str, str, str], dict]] = {}
        self._dirty: set[str] = set()

    def path(self, detector_key: str) -> Path:
        if not _KEY.fullmatch(detector_key):
            raise ValueError(f"invalid detector key {detector_key!r} (expected <detector_id>@<weights_sha12>)")
        return self.dir / f"{detector_key}.parquet"

    def _table(self, detector_key: str) -> dict[tuple[str, str, str], dict]:
        if detector_key not in self._rows:
            import pyarrow.parquet as pq

            path = self.path(detector_key)
            rows = pq.read_table(path).to_pylist() if path.is_file() else []
            self._rows[detector_key] = {(r["item_id"], r["variant"], r["transform"]): r for r in rows}
        return self._rows[detector_key]

    def status(self, detector_key: str, key: tuple[str, str, str]) -> str | None:
        row = self._table(detector_key).get(key)
        return row["status"] if row else None

    def done(self, detector_key: str, key: tuple[str, str, str]) -> bool:
        return self.status(detector_key, key) in FINAL_STATUSES

    def put(self, detector_key: str, key: tuple[str, str, str], fields: dict, *, scored_at: str) -> None:
        prob = fields.get("prob")
        self._table(detector_key)[key] = {
            "item_id": key[0], "variant": key[1], "transform": key[2], "status": fields["status"],
            "reason": fields.get("reason"), "prob": prob, "logit": logit(prob),
            "aux": json.dumps(fields.get("aux") or {}, sort_keys=True, default=str), "runtime_ms": fields.get("runtime_ms"),
            "scored_at": scored_at,
        }
        self._dirty.add(detector_key)

    def rows(self, detector_key: str) -> list[dict]:
        return sorted(self._table(detector_key).values(), key=lambda r: (r["item_id"], r["variant"], r["transform"]))

    def flush(self, marker: str | None = None) -> None:
        order = sorted(k for k in self._dirty if k != marker) + ([marker] if marker in self._dirty else [])
        for detector_key in order:
            self._write(detector_key)
        self._dirty.clear()

    def _write(self, detector_key: str) -> None:
        import pyarrow as pa
        import pyarrow.parquet as pq

        path = self.path(detector_key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".part")
        pq.write_table(pa.Table.from_pylist(self.rows(detector_key), schema=score_schema()), tmp)
        os.replace(tmp, path)


@dataclass(frozen=True)
class ScoreTask:
    item_id: str
    variant: str
    transform: str
    path: Path

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.item_id, self.variant, self.transform)


class Scorer(Protocol):
    def marker(self) -> str: ...
    def model_keys(self) -> list[str]: ...
    def score(self, task: ScoreTask) -> dict[str, dict]: ...
    def close(self) -> None: ...


@dataclass
class ScoreSummary:
    scored: int = 0
    skipped: int = 0
    failed: int = 0


def keys_without_metrics(cache: ScoreCache, detector_keys: Iterable[str], tasks: Iterable[ScoreTask]) -> set[tuple[str, str, str]]:
    """H-6: task keys whose row in any of `detector_keys` holds a final status but no `metrics` in `aux`: rows scored before
    `score_v2.detector_row` captured localiser and forensic statistics. `run_scoring(force=…)` rescores them."""
    tables = [cache._table(key) for key in detector_keys]
    stale: set[tuple[str, str, str]] = set()
    for task in tasks:
        for table in tables:
            row = table.get(task.key)
            if row and row["status"] in FINAL_STATUSES and "metrics" not in json.loads(row["aux"] or "{}"):
                stale.add(task.key)
                break
    return stale


def run_scoring(scorer: Scorer, tasks: Iterable[ScoreTask], cache: ScoreCache, *, checkpoint_every: int = 500,
                force: set[tuple[str, str, str]] | None = None) -> ScoreSummary:
    """Score every task the marker does not hold as done; keys in `force` are rescored and their rows replaced."""
    marker = scorer.marker()
    summary = ScoreSummary()
    since_checkpoint = 0
    force = force or set()
    try:
        for task in tasks:
            if task.key not in force and cache.done(marker, task.key):
                summary.skipped += 1
                continue
            now = datetime.now(UTC).isoformat(timespec="seconds")
            try:
                rows = scorer.score(task)
                summary.scored += 1
            except Exception as e:  # noqa: BLE001 - one benchmark item never stops the run; the row records why
                reason = f"{type(e).__name__}: {str(e)[:200]}"
                rows = {k: {"status": "failed", "reason": reason} for k in [*scorer.model_keys(), marker]}
                summary.failed += 1
            for detector_key, fields in rows.items():
                cache.put(detector_key, task.key, fields, scored_at=now)
            since_checkpoint += 1
            if since_checkpoint >= checkpoint_every:
                cache.flush(marker)
                since_checkpoint = 0
    finally:
        cache.flush(marker)
    return summary


def build_tasks(vbench_dir: Path, rows: Iterable[dict], *, variant: str, transform: str,
                limit: int | None = None) -> list[ScoreTask]:
    """Plan the ordered task limit before verifying files; absent transform keys are unplanned."""
    from app.bench.input_integrity import recorded_transform_paths, verify_local_input

    if variant not in {"native", "norm"}:
        raise ValueError("benchmark variant must be native or norm")
    if limit is not None and (type(limit) is not int or limit < 0):
        raise ValueError("benchmark task limit must be a nonnegative integer or None")
    if limit == 0:
        return []
    selected = list(rows)
    ids = [row.get("item_id") for row in selected]
    if any(not isinstance(item_id, str) or not item_id for item_id in ids) or len(set(ids)) != len(ids):
        raise ValueError("selected benchmark item IDs must be nonempty and unique")
    if transform != "none":
        paths = recorded_transform_paths(vbench_dir, ids, transform, limit=limit)
        return [ScoreTask(item_id, "native", transform, path) for item_id, path in paths.items()]
    # A missing normalized variant is not a task and must not consume the limit.
    planned = [row for row in selected if variant != "norm" or row.get("norm_path") is not None][:limit]
    tasks = []
    for row in planned:
        path = verify_local_input(vbench_dir, row.get(f"{variant}_path"), expected_bytes=row.get(f"{variant}_bytes"),
                                  expected_sha256=row.get(f"{variant}_sha256"))
        tasks.append(ScoreTask(row["item_id"], variant, "none", path))
    return tasks


# --- detector keys -----------------------------------------------------------------------------------------

def weights_sha12(manifest: Manifest, detector_id: str) -> str:
    """sha256 over every pinned file (and revision) of the detector's models and their dependencies, first 12 hex."""
    detector = manifest.detector(detector_id)
    if detector is None:
        raise KeyError(f"detector {detector_id!r} is not in the manifest")
    return models_sha12(manifest, detector.models, owner=detector_id)


def models_sha12(manifest: Manifest, model_ids: Iterable[str], *, owner: str) -> str:
    """sha256 over every pinned file (and revision) of the models and their dependencies, first 12 hex."""
    parts, seen, stack = [], set(), list(model_ids)
    while stack:
        model_id = stack.pop()
        if model_id in seen:
            continue
        seen.add(model_id)
        entry = manifest.model(model_id)
        if entry is None:
            raise KeyError(f"model {model_id!r} (of {owner}) is not in the manifest")
        parts += [f"{model_id}:{entry.source.revision or ''}:{f.path}:{f.sha256 or ''}" for f in entry.source.files]
        stack += list(getattr(entry, "depends_on", None) or [])
    return hashlib.sha256("\n".join(sorted(parts)).encode()).hexdigest()[:12]


def _signal_key(signal_id: str) -> str:
    return f"v1.{re.sub(r'[^a-z0-9._-]+', '_', signal_id.lower())}@noweights"


def _compact(details: Any) -> dict:
    if not isinstance(details, dict):
        return {}
    return {k: v for k, v in details.items()
            if isinstance(v, bool | int | float) or (isinstance(v, str) and len(v) <= 200)}


class V1ImageScorer:
    """The v0.3 still-image pipeline (`analyze_image_array(fast=True)`), mapped onto registry detector keys."""

    def __init__(self, manifest: Manifest | None = None):
        from app.core.config import settings

        manifest = manifest or load_manifest()
        repo_detector = {}
        for detector in manifest.detectors:
            if detector.modality != "image":
                continue
            for model_id in detector.models:
                entry = manifest.model(model_id)
                if entry is not None and entry.source.type == "hf" and entry.source.repo:
                    repo_detector[entry.source.repo] = detector.id
        signal_detector = {"univfd_clip": "img.univfd", "clf_dfdc_effnet_b7": "img.dfdc_b7"}
        for raw in settings.image_classifiers:  # "repo|positive|weight"; v1 names the signal clf_<repo name>
            repo = raw.split("|", 1)[0].strip()
            if repo in repo_detector:
                signal_detector[f"clf_{repo.split('/')[-1]}"] = repo_detector[repo]
        self.signal_keys = {sid: f"{det}@{weights_sha12(manifest, det)}" for sid, det in signal_detector.items()}
        combined = hashlib.sha256("\n".join(sorted(self.signal_keys.values())).encode()).hexdigest()[:12]
        self._marker = f"{V1_FUSED}@{combined}"

    def marker(self) -> str:
        return self._marker

    def model_keys(self) -> list[str]:
        return sorted(set(self.signal_keys.values()))

    def score(self, task: ScoreTask) -> dict[str, dict]:
        from app.detectors.image.pipeline import analyze_image_array, load_image

        pil, rgb, info = load_image(task.path)
        started = time.perf_counter()
        result = analyze_image_array(rgb, pil, None, fast=True, path=task.path)
        return self.rows_from_result(result, info, (time.perf_counter() - started) * 1000)

    def rows_from_result(self, result: dict, info: dict, runtime_ms: float) -> dict[str, dict]:
        signals = result.get("signals") or []
        if not any(s["id"] in self.signal_keys and s.get("ai_probability") is not None for s in signals):
            # v1 swallows model load failures; a forensics-only fused score must not enter the cache as a v0.3 score
            raise RuntimeError("no model-backed signal: every v1 image model failed to load or run (see the job log)")
        rows: dict[str, dict] = {}
        for signal in signals:
            key = self.signal_keys.get(signal["id"]) or _signal_key(signal["id"])
            prob = signal.get("ai_probability")
            rows[key] = {"status": "ok" if prob is not None else "not_applicable",
                         "reason": None if prob is not None else "the signal carries no probability",
                         "prob": prob, "runtime_ms": None,
                         "aux": {"weight": signal.get("weight"), "category": signal.get("category"),
                                 "observed": signal.get("observed"), "details": _compact(signal.get("details"))}}
        for key in self.model_keys():
            rows.setdefault(key, {"status": "not_applicable", "prob": None, "runtime_ms": None, "aux": {},
                                  "reason": "the v1 pipeline produced no signal for this detector (no face in the image, "
                                            "or the model failed to load; see the job log)"})
        verdict = result.get("verdict") or {}
        rows[self._marker] = {"status": "ok", "reason": None, "prob": verdict.get("probability_ai"),
                              "runtime_ms": round(runtime_ms, 1),
                              "aux": {"band": verdict.get("band"), "faces": len(result.get("faces") or []),
                                      "timings": result.get("timings"), "downscaled": info.get("downscaled")}}
        return rows

    def close(self) -> None:
        """Release v1 models (D-21: scripts release models on exit)."""
        try:
            from app.detectors.common.registry import registry

            clear = getattr(registry, "clear", None)
            if callable(clear):
                clear()
        except Exception:  # noqa: BLE001 - best effort at shutdown
            pass
        gc.collect()
        try:
            import torch

            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except Exception:  # noqa: BLE001
            pass


def _repo_detectors(manifest: Manifest, modality: str, repos: Iterable[str]) -> dict[str, str]:
    """v1 model short name (last path part of the HF repo) → the manifest detector of `modality` that loads that repo."""
    wanted = set(repos)
    out = {}
    for detector in manifest.detectors:
        if detector.modality != modality:
            continue
        for model_id in detector.models:
            entry = manifest.model(model_id)
            if entry is not None and entry.source.type == "hf" and entry.source.repo in wanted:
                out[entry.source.repo.split("/")[-1]] = detector.id
    return out


def _voice_keys(manifest: Manifest) -> dict[str, str]:
    from app.core.config import settings

    repos = [raw.split("|", 1)[0].strip() for raw in settings.audio_classifiers]
    return {short: f"{det}@{weights_sha12(manifest, det)}"
            for short, det in _repo_detectors(manifest, "audio", repos).items()}


def _marker(prefix: str, keys: Iterable[str]) -> str:
    return f"{prefix}@{hashlib.sha256(chr(10).join(sorted(keys)).encode()).hexdigest()[:12]}"


def _mean_row(values: list[float], unit: str, reason: str) -> dict:
    """A detector row from per-unit v1 scores: the mean (as the v2 adapters aggregate), or not_applicable without units."""
    if not values:
        return {"status": "not_applicable", "reason": reason, "prob": None, "runtime_ms": None, "aux": {}}
    return {"status": "ok", "reason": None, "prob": sum(values) / len(values), "runtime_ms": None,
            "aux": {unit: len(values), "max": max(values)}}


def _voice_rows(keys: dict[str, str], voice: dict | None, state: str | None) -> dict[str, dict]:
    per_model = ((voice or {}).get("details") or {}).get("per_model_chunks") or {}
    return {key: _mean_row(list(per_model.get(short) or []), "chunks",
                           f"v0.3 produced no score for this voice model (audio: {state or 'unknown'})")
            for short, key in keys.items()}


class V1VideoScorer:
    """The v0.3 video pipeline (`score_video` + `assemble_video`, no artefacts), mapped onto registry detector keys."""

    COMPOSITE = frozenset({"frames_general", "frames_face", "audio_synthetic_voice"})  # averages over several models

    def __init__(self, manifest: Manifest | None = None):
        from app.core.config import settings

        manifest = manifest or load_manifest()
        repos = [raw.split("|", 1)[0].strip() for raw in settings.image_classifiers]
        frame_detector = {"univfd": "vid.frames.univfd", "dfdc_effnet_b7": "vid.frames.dfdc_b7",
                          **_repo_detectors(manifest, "video", repos)}
        self.frame_keys = {short: f"{det}@{weights_sha12(manifest, det)}" for short, det in frame_detector.items()}
        self.voice_keys = _voice_keys(manifest)  # the sound track, as the v2 video path runs the aud.* detectors
        self._marker = _marker(V1_VIDEO_FUSED, [*self.frame_keys.values(), *self.voice_keys.values()])

    def marker(self) -> str:
        return self._marker

    def model_keys(self) -> list[str]:
        return sorted({*self.frame_keys.values(), *self.voice_keys.values()})

    def score(self, task: ScoreTask) -> dict[str, dict]:
        from app.detectors.video.pipeline import assemble_video, score_video

        started = time.perf_counter()
        scores, _frames, timings = score_video(task.path)
        result = assemble_video(scores, timings=timings)
        return self.rows_from_result(result, (time.perf_counter() - started) * 1000)

    def rows_from_result(self, result: dict, runtime_ms: float) -> dict[str, dict]:
        frames = result.get("frames") or []
        rows = {key: _mean_row([f["scores"][short] for f in frames if (f.get("scores") or {}).get(short) is not None],
                               "frames", "no sampled frame produced a score for this detector (no face in the sampled "
                                         "frames, or the model failed to load; see the job log)")
                for short, key in self.frame_keys.items()}
        if not any(rows[key]["status"] == "ok" for key in self.frame_keys.values()):
            raise RuntimeError("no model-backed frame score: every v1 frame model failed to load or run (see the job log)")
        signals = result.get("signals") or []
        audio = ((result.get("video") or {}).get("audio_analysis") or {}).get("audio")
        rows |= _voice_rows(self.voice_keys, next((s for s in signals if s["id"] == "audio_synthetic_voice"), None), audio)
        for signal in signals:
            if signal["id"] in self.COMPOSITE:
                continue
            prob = signal.get("ai_probability")
            rows[_signal_key(signal["id"])] = {
                "status": "ok" if prob is not None else "not_applicable",
                "reason": None if prob is not None else "the signal carries no probability", "prob": prob, "runtime_ms": None,
                "aux": {"weight": signal.get("weight"), "category": signal.get("category"),
                        "observed": signal.get("observed"), "details": _compact(signal.get("details"))}}
        verdict = result.get("verdict") or {}
        rows[self._marker] = {"status": "ok", "reason": None, "prob": verdict.get("probability_ai"),
                              "runtime_ms": round(runtime_ms, 1),
                              "aux": {"band": verdict.get("band"), "frames": len(frames),
                                      "face_frames": sum(1 for f in frames if f.get("faces")), "audio": audio,
                                      "timings": result.get("timings")}}
        return rows

    close = V1ImageScorer.close


class V1AudioScorer:
    """v0.3 synthetic-voice detection on an audio file: `audio_signals`, the sound-track path of the v0.3 video pipeline
    (v0.3 has no audio upload kind). The fused marker is the fused synthetic-voice signal."""

    def __init__(self, manifest: Manifest | None = None):
        self.voice_keys = _voice_keys(load_manifest() if manifest is None else manifest)
        self._marker = _marker(V1_AUDIO_FUSED, self.voice_keys.values())

    def marker(self) -> str:
        return self._marker

    def model_keys(self) -> list[str]:
        return sorted(self.voice_keys.values())

    def score(self, task: ScoreTask) -> dict[str, dict]:
        from app.detectors.video.audio import audio_signals

        started = time.perf_counter()
        signals, info = audio_signals(task.path, True)
        return self.rows_from_result([s.model_dump() for s in signals], info, (time.perf_counter() - started) * 1000)

    def rows_from_result(self, signals: list[dict], info: dict, runtime_ms: float) -> dict[str, dict]:
        state = info.get("audio")
        if state in ("models_failed", "disabled"):
            # an environment problem, not a property of the file: fail the item so a later run retries it
            raise RuntimeError(f"no v0.3 voice model ran (audio: {state}; see the job log)")
        voice = next((s for s in signals if s["id"] == "audio_synthetic_voice"), None)
        rows = _voice_rows(self.voice_keys, voice, state)
        prob = None if voice is None else voice.get("ai_probability")
        aux = {"audio": state, "seconds": info.get("seconds"), "chunks": info.get("chunks")}
        rows[self._marker] = ({"status": "ok", "reason": None, "prob": prob, "runtime_ms": round(runtime_ms, 1), "aux": aux}
                              if prob is not None else
                              {"status": "not_applicable", "reason": f"v0.3 abstains on this file (audio: {state})",
                               "prob": None, "runtime_ms": round(runtime_ms, 1), "aux": aux})
        return rows

    close = V1ImageScorer.close
