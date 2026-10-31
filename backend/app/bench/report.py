"""Veritas-Bench report (ML §2.8, §5.1, §5.3, §8; D-13): fused scores from the score cache → results, error rates, slices.

- **System.** `ArtefactSystem` applies a `veritas.fuser/1` artefact (Stages 1–4, `Fuser.apply`) to the cached detector
  scores with the manifest context, the inputs `calibrate.py` fits on. `ScoreSystem` takes one score-cache key as s, e.g.
  the v0.3 fused marker `v1.image_fused@<sha12>`; it has no conformal sets, so coverage, selective metrics and the G15
  invariants do not exist for it.
- **Items.** Image rows with `audit_ok` not false and no C2PA manifest (Stage 0 decides those; excluded and counted, as in
  calibrate). An item enters a set only when the system has a final cache row for it: for an artefact, any of its
  detectors with status ok / not_applicable / skipped; for a score key, status ok with a probability.
- **Thresholds.** τ_0.05 and τ_0.01 from `cal` reals with equal real-stratum weights (`metrics.operational_threshold`).
- **Eval split.** `test` for goal evaluation (lockbox: the CLI appends every run to `docs/v2/bench/test_access.log`). The
  smoke run evaluates `cal`: its set names say `image/cal`, and its thresholds, ECE and coverage are in-sample.
- **Sets** over the eval split E: `image/E`; `image/E/t2i`, `/edit`, `/personalised` (those fakes + every real);
  `/faceswap_vs_R5`, `/gan_face_vs_R5` (those fakes + R5 reals); `/legacy` (family `legacy` + COCO / Flickr30k reals).
  Each: AUC, TPR and FPR at τ with Wilson and grouped-bootstrap intervals, balanced accuracy, class-balanced ECE / Brier /
  log-loss, the oracle TPR (τ from E's own reals; a ranking diagnostic only) and, for an artefact, class-conditional
  conformal coverage and selective metrics at α = 0.05.
- **Size-matched view (D-45).** `image/E` and its threat subsets are measured on a size-matched view: in each min-side
  bucket (<384, 384–767, 768–1024, 1025–1536, >1536 px of the scored bytes; norm files keep the native dimensions) every
  item of the rarer class relative to the pooled class ratio, plus a salted sha256("vb-size:" + item_id) sample of the other
  class at that ratio, so image size cannot separate the classes. The full view is `image/E/full` (`/full/<subset>`). LGO
  folds use the same rule (fold entries keep `full`); real strata stay whole and add `real_strata_by_size`.
- **Strata and families.** FPR per real stratum, plus `screenshot` (the `screenshot` transform of robustness R1/R2 reals);
  recall per fake family with Wilson intervals.
- **LGO.** Fold F = every F item in fit / cal / test plus E's reals. An artefact system needs a refit without F
  (`bench.excluded_folds == [F]`), whose own `cal` reals set τ; a score system uses the global τ. Macro and minimum over
  F1–F7 (null while a fold is missing), plus min / median / max of the present folds (no pooled interval, §5.3).
- **Robustness.** Robustness-subset items scored under both the reference (`native`/`none` by default: perturbed files
  are made from native bytes) and transform t: fixed-threshold ΔTPR / ΔFPR at τ_0.05, ΔAUC, and the inconclusive rate
  under t for an artefact.
- **Error rates** (§5.1): one row per (context, family) with n, TPR / FPR at τ_0.05 and τ_0.01 with Wilson intervals and the
  rule-of-three bound, the inconclusive rate, the LGO macro TPR and the score dates. Camera contexts (DSLR = R1,
  smartphone = R2, social-recompressed = R3) take reals by stratum and every fake of the family; feature contexts
  (portrait face, small face, low-res) take both classes by the manifest columns; screenshot uses the transform.
- **Slices** (`abstain.image_slice`): n per class and the AUC when both classes have ≥ 100 items (`metrics.test.slices`).
- **Invariants** (G15, artefact): on up to 1,000 seeded E items, max |Σφ′ + b′ − logit s| and the number of random +Δz
  moves on a scored detector that lowered s.
- **Video and audio** (`modality_systems`, B-5): per modality over the native files (they have no norm variant), with τ
  from that modality's own `cal` reals: sets `<m>/E` and `<m>/E/real`; audio adds `/asvspoof5` and `/in_the_wild` (by
  source) and the EER on every audio set with both classes; video adds `lgo_by_modality["video/lgo"]` over V-Veo …
  V-WanHunyuan (a score system uses the modality's τ; an artefact needs refits, not built yet). `modalities.<m>` holds
  the system, τ and counts. Robustness and error-rate rows stay image-only.

- **Decision rules (D-64, D-65; N-18).** The selective and inconclusive metrics apply the rules of `abstain` that the
  runtime applies. An image system given the content measurements of the scored bytes (`bench.content.ScoredContent`)
  treats an item as inconclusive when it is content the artefact's bench does not hold (flat blocks above the bench's
  limit, the screenshot rule, greyscale), and every set and robustness transform states how many scored items each
  condition holds for and how many decisions it takes away (`decision_rules`). A system without the measurements, or an
  artefact whose bench has no limit, applies no content rule and says so. The same block counts the items in which
  fewer than two checks scored and those whose strong code is not issued for it (D-64): the report's decisions are at
  α = 0.05, the level of the moderate codes, so that rule changes none of them. The report evaluates no uncalibrated
  fallback decision, so the second rule of D-64 has nothing to apply to here.
- **A decision needs a check that supports it (D-76; N-19i).** An item whose set at α = 0.05 holds one answer while no
  check that counts has a row of at least ln 2 toward it (`abstain.supported`, the function the runtime calls) is
  marked, counted both ways in `decision_rules` (the items that carry the mark, and those that lose a decision by it)
  and not decided; neither is an item in which no check entered (N-19f: the runtime's result is not assessable).
  `selective_raw_conformal` takes neither rule.
- **The estimate in its part of the scale, and the checks together (D-77, D-76 as amended; N-19m).** The report asks
  the two conditions of a decisive code by the runtime's function and in its order (`abstain.rule_for`, on the item's
  estimate and the same published view), so that a mark is the result's reason: an item whose estimate prints 35 to 64
  under a set of one answer carries D-77's mark and no mark of D-76; else the mark of D-76's first form (no single
  check), else that of its second (the checks together give the answer less than ln 2). Each is counted as D-76's first
  form is (the mark, both ways, the item not decided). The report applies no configured prior and meets no record.

Context for an artefact follows its `bench.context_source` (ADR 0004): `manifest` columns describe the native bytes
(face_px_max is null), not the norm file; `scored` takes the condition context recorded for the scored bytes of each
(item, variant, transform), so robustness copies carry their own context. A scored item without a recorded context is not
scored. Slices prefer the recorded production condition of the scored bytes; native metadata is only a fallback.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import statistics
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from app.bench import metrics as mt
from app.bench import perturb as pt
from app.bench import slice_map
from app.bench.content import ScoredContent
from app.bench.context import CONTEXT_SOURCES, ScoredContexts, manifest_context
from app.engine.fusion import abstain, content_kinds
from app.engine.fusion.artefact import SCHEMA as FUSER_SCHEMA
from app.engine.fusion.artefact import Fuser, artefact_bytes
from app.engine.fusion.calibration import clamp_probability, platt_params
from app.engine.fusion.features import Context

SCHEMA = "veritas.bench-results/1"
ERROR_RATES_SCHEMA = "veritas.error-rates/1"
ALPHAS = (0.05, 0.01)
FINAL_STATUSES = frozenset({"ok", "not_applicable", "skipped"})
NOT_RUN_ROW_STATUSES = frozenset({"failed", "timeout", "skipped"})  # D-49: the score cache stores a timeout as `failed`
LGO_FOLDS = tuple(f"F{i}" for i in range(1, 8))
MIN_SLICE_CLASS_N = slice_map.MIN_CLASS_N  # H-7 (EA-02): n ≥ 100 per class, else the nearest measured parent slice
INVARIANT_ITEMS = 1000
LEGACY_REAL_SOURCES = frozenset({"coco-val", "flickr30k"})
SCREENSHOT_REAL_STRATA = frozenset({"R1", "R2"})
CAMERA_CONTEXTS = {"DSLR": "R1", "smartphone": "R2", "social-recompressed": "R3"}
NOT_MEASURED_CONTEXTS = ("video T2V", "video talking head", "audio lossy")
AV_MODALITIES = ("video", "audio")
VIDEO_LGO_FOLDS = ("V-Veo", "V-Sora", "V-Kling", "V-Seedance", "V-WanHunyuan")  # the video lgo_fold values in sources.yaml
AUDIO_SOURCE_SETS = {"asvspoof5": frozenset({"asvspoof5-test"}), "in_the_wild": frozenset({"in-the-wild-audio"})}  # G10
_KEY = "{:g}"
H3 = "H3_provenance_integrity"
STRONG_CODE = {"very_likely_ai": "ai_supported_strong", "very_likely_authentic": "no_ai_indicators_strong"}
# per-item marks of the decision rules (D-64, D-65), beside the names of `abstain.CONTENT_CONDITIONS`
CONTENT_MEASURED, CONTENT_NOT_MEASURED = "content_measured", "content_not_measured"
FEWER_THAN_TWO_CHECKS, STRONG_NOT_ISSUED = "fewer_than_two_checks", "strong_not_issued"
# D-76 (N-19i): the set of the item at α = 0.05 holds one answer and no check supports it (`abstain.supported`), so the
# runtime publishes it as inconclusive; and N-19f: no check entered, so the runtime publishes it as not assessable. Neither
# item is decided
NO_SUPPORTING_CHECK, NO_CHECK_ENTERED = "no_supporting_check", "no_check_entered"
# N-19m (D-77; D-76 as amended): the estimate of the item prints outside the part of the scale of the answer of its set;
# a check supports that answer and the checks together do not. One mark of the three per item, the runtime's reason
ESTIMATE_OFF_SCALE, CHECKS_NOT_TOGETHER = "estimate_off_scale", "checks_not_together"
RULE_MARKS = {abstain.RULE_SCALE: ESTIMATE_OFF_SCALE, abstain.RULE_NO_CHECK: NO_SUPPORTING_CHECK,
              abstain.RULE_TOGETHER_LESS: CHECKS_NOT_TOGETHER, abstain.RULE_TOGETHER_AGAINST: CHECKS_NOT_TOGETHER}
NOT_DECIDED_MARKS = (NO_SUPPORTING_CHECK, NO_CHECK_ENTERED, ESTIMATE_OFF_SCALE, CHECKS_NOT_TOGETHER)
ANSWER_CODE = {("ai",): "ai_supported", ("real",): "no_ai_indicators"}  # the code of a set of one answer at the moderate level


class ReportError(ValueError):
    """The manifest, score cache or artefacts cannot be reported."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()




def envelope_fingerprint(slices: Mapping[str, Any] | None) -> dict[str, Any]:
    """Digest canonical JSON for the exact envelope map used by a statistical policy."""
    if slices is None:
        return {"status": "missing", "sha256": None, "slice_count": 0}
    data = json.dumps(_plain(slices), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    return {"status": "present" if slices else "empty", "sha256": hashlib.sha256(data).hexdigest(), "slice_count": len(slices)}


def content_rule(doc: Mapping[str, Any], *, measured: bool) -> dict[str, Any]:
    """D-65 as this report applies it to one artefact: the limit of its bench and whether the scored bytes were measured."""
    bench = doc.get("bench") or {}
    entry = content_kinds.limits().get(str(bench.get("manifest_sha256"))) if doc.get("modality") == "image" else None
    if doc.get("modality") != "image":
        state = "no rule for this modality"
    elif entry is None:
        state = "not applied: the artefact's bench has no limit in content_kinds.json"
    elif not measured:
        state = "not applied: the scored bytes were not measured for this report"
    else:
        state = "applied"
    return {"decision": "D-65", "applied": state == "applied", "state": state,
            "limit_of_manifest_sha256": bench.get("manifest_sha256") if entry else None,
            "limit_of_bench": entry["bench"] if entry else None,
            "flat_block_limit": float(entry["flat_blocks"]["limit"]) if entry else None,
            "conditions": list(abstain.CONTENT_CONDITIONS),
            "measured_on": "the scored bytes of each (item, variant, transform)" if state == "applied" else None}


def decision_policy(doc: Mapping[str, Any], sha256: str, *, content_measured: bool = False) -> dict[str, Any]:
    """Provenance of selective metrics; this benchmark applies the loaded default statistical policy only."""
    slices = ((doc.get("metrics") or {}).get("test") or {}).get("slices")
    content = content_rule(doc, measured=content_measured)
    return {"schema": "veritas.bench-decision-policy/4", "artefact_sha256": sha256,
            "envelope_source": "loaded_artefact.metrics.test.slices", "envelope": envelope_fingerprint(slices),
            "conformal_alpha": 0.05, "min_slice_auc": abstain.MIN_SLICE_AUC,
            "min_support_class_n": abstain.MIN_SUPPORT_CLASS_N, "prior_ai": 0.5,
            "policy": "default statistical conformal decision plus loaded envelope cap"
                      + (" plus the cap of content the bench does not hold" if content["applied"] else ""),
            # D-76 (N-19i): a decision needs a check that supports it; N-19f: and a check that entered
            "supporting_check": {"decision": "D-76", "min_row": "ln 2 toward the answer, one check that counts for D-64",
                                 "effect": "an item whose set at α = 0.05 holds one answer that no check supports is not "
                                           "decided; counted in decision_rules; selective_raw_conformal does not take the rule"},
            # N-19m: the two conditions in the runtime's order, each item under the one that is its result's reason
            "estimate_in_scale": {"decision": "D-77", "bound": f"the estimate in whole points: {abstain.SCALE_CLEAN_MAX} or "
                                  f"less for the clean answer, {abstain.SCALE_AI_MIN} or more for the AI answer; no prior",
                                  "effect": "an item whose set at α = 0.05 holds one answer while its estimate prints outside "
                                            "that answer's part of the scale is not decided; asked before D-76; counted in "
                                            "decision_rules"},
            "checks_together": {"decision": "D-76 as amended", "min_sum": "ln 2 toward the answer, the rows of the checks "
                                "that count for D-64", "effect": "an item that one check supports and the checks together "
                                "do not is not decided; counted in decision_rules"},
            "no_check_entered": {"decision": "D-64 as clarified (N-19f)",
                                 "effect": "an item in which no check entered is not assessable and not decided; counted "
                                           "in decision_rules"},
            "content_rule": content,
            "strong_codes": {"decision": "D-64", "min_scored_checks": abstain.MIN_CHECKS_FOR_STRONG,
                             "effect": "none on these metrics: the decisions are at α = 0.05, the level of the moderate "
                                       "codes; the items it weakens at the strong level are counted in decision_rules"},
            "fallback_content_check": {"decision": "D-64", "effect": "none: no uncalibrated fallback decision is evaluated"},
            "checks_not_run": "inconclusive without an estimate",
            "excludes": ["documentary provenance overrides", "workspace policy overrides", "configured prior changes",
                         "provenance contribution bound", "confidence adjustments"]}


def artefact_write_plan(results: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Declare envelope writes without treating newly measured slices as independently evaluated policy."""
    parts = {"image": {"system": results["system"], "slices": results["slices"]}, **(results.get("modalities") or {})}
    updates = {}
    for modality, part in parts.items():
        system = part["system"]
        if system.get("kind") != "artefact":
            continue
        loaded = system["decision_policy"]["envelope"]
        proposed = envelope_fingerprint(part["slices"])
        changed = loaded != proposed
        notice = (f"{modality}: selective and inconclusive metrics describe the PRE-WRITE policy of artefact "
                  f"{system['sha256']}. --write-artefact changes its envelope using this report's measured slices. "
                  "The new envelope has not been independently re-evaluated; these metrics do not establish post-write serving parity."
                  if changed else None)
        updates[modality] = {"state": "requested", "evaluated_artefact_sha256": system["sha256"],
                             "loaded_envelope": loaded, "proposed_envelope": proposed, "changes_envelope": changed,
                             "metrics_policy": "pre_write_loaded_artefact", "post_write_policy_re_evaluated": False,
                             "written_artefact_sha256": None, "notice": notice}
    return updates


# --- items and scores --------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Item:
    item_id: str
    y: int
    split: str
    group: str
    real_stratum: str | None
    family: str | None
    threat: str
    lgo_fold: str | None
    source_id: str | None
    width: int | None
    height: int | None
    jpeg_q: int | None
    face_px: int | None
    context: Context
    duration_s: float | None = None  # video and audio slices (H-9)

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> Item:
        return cls(item_id=row["item_id"], y=1 if row["label"] == "fake" else 0, split=str(row.get("split")),
                   group=row.get("group_key") or row["item_id"], real_stratum=row.get("real_stratum"),
                   family=row.get("family"), threat=row.get("threat") or "unknown", lgo_fold=row.get("lgo_fold"),
                   source_id=row.get("source_id"), width=row.get("width"), height=row.get("height"),
                   jpeg_q=row.get("jpeg_q_est"), face_px=row.get("face_px_max"),
                   context=manifest_context(row, row.get("modality") or "image"), duration_s=row.get("duration_s"))

    @property
    def stratum(self) -> str:
        return (self.real_stratum if self.y == 0 else self.family) or self.threat


class ScoreStore:
    """Read-only score cache tables `<key>.parquet`, loaded on first use."""

    def __init__(self, scores_dir: Path):
        self.dir = Path(scores_dir)
        self._tables: dict[str, dict[tuple[str, str, str], dict]] = {}

    def has(self, key: str) -> bool:
        return (self.dir / f"{key}.parquet").is_file()

    def table(self, key: str) -> dict[tuple[str, str, str], dict]:
        if key not in self._tables:
            import pyarrow.parquet as pq

            path = self.dir / f"{key}.parquet"
            if not path.is_file():
                raise ReportError(f"score cache has no file {path.name}")
            from app.bench.score_v2 import PLACEHOLDER_REASON

            names = pq.read_schema(path).names
            rows = pq.read_table(path, columns=["item_id", "variant", "transform", "status", "prob", "logit", "scored_at",
                                                *(c for c in ("reason", "aux") if c in names)]).to_pylist()
            table: dict[tuple[str, str, str], dict] = {}
            for r in rows:  # the scorer's placeholder row (no reported status) stands for no entry (D-49)
                aux = r.pop("aux", None)
                r["placeholder"] = r.get("reason") == PLACEHOLDER_REASON and '"reported_status"' not in (aux or "")
                table[(r["item_id"], r["variant"], r["transform"])] = r
            self._tables[key] = table
        return self._tables[key]


@dataclass(frozen=True)
class Scored:
    s: float
    sets05: tuple[str, ...] | None  # conformal set at α = 0.05 (artefact systems)
    scored_at: str | None
    not_run: tuple[str, ...] = ()  # D-49: capped, no estimate (s is NaN and never read)
    slice_id: str | None = None
    within_envelope: bool | None = None
    rules: tuple[str, ...] = ()  # D-64, D-65: the marks of the decision rules that hold for this item


class ScoreSystem:
    """One score-cache key used as s."""

    kind = "score"
    conformal = False

    def __init__(self, store: ScoreStore, key: str):
        if not store.has(key):
            raise ReportError(f"score cache has no file {key}.parquet")
        self.store, self.key = store, key

    def describe(self) -> dict[str, Any]:
        return {"kind": self.kind, "score_key": self.key}

    def score(self, item: Item, variant: str, transform: str) -> Scored | None:
        row = self.store.table(self.key).get((item.item_id, variant, transform))
        if row is None or row["status"] != "ok" or row["prob"] is None or not math.isfinite(row["prob"]):
            return None
        return Scored(s=float(row["prob"]), sets05=None, scored_at=row["scored_at"])


class ArtefactSystem:
    """A fuser artefact applied to cached detector scores (no bootstrap interval: the report needs s and the sets)."""

    kind = "artefact"
    conformal = True

    def __init__(self, store: ScoreStore, path: Path, *, content: ScoredContent | None = None):
        """`content`: the measurements of the scored bytes (D-65). Without them, for another modality than image, or
        for an artefact whose bench has no limit, no content rule is applied."""
        data = Path(path).read_bytes()
        doc = json.loads(data.decode("utf-8"))
        if doc.get("schema") != FUSER_SCHEMA:
            raise ReportError(f"{Path(path).name} is not a {FUSER_SCHEMA} artefact")
        self.store, self.path, self.sha256, self.doc = store, Path(path), hashlib.sha256(data).hexdigest(), doc
        self.fuser = Fuser(doc)
        self.context_source = (doc.get("bench") or {}).get("context_source") or "manifest"
        if self.context_source not in CONTEXT_SOURCES:
            raise ReportError(f"{Path(path).name}: unknown context source {self.context_source!r}")
        self._scored: ScoredContexts | None = None
        self.content = content if content_rule(doc, measured=content is not None)["applied"] else None
        self.h3 = declared_h3(doc)
        self.keys = {d["id"]: d.get("score_key") or f"{d['id']}@{d.get('weights_sha12')}" for d in doc["detectors"]}
        missing = sorted(k for k in self.keys.values() if not store.has(k))
        if missing:
            raise ReportError(f"score cache lacks the artefact's detector files {missing}")

    @property
    def excluded_folds(self) -> list[str]:
        return list((self.doc.get("bench") or {}).get("excluded_folds") or [])

    def describe(self) -> dict[str, Any]:
        bench = self.doc.get("bench") or {}
        return {"kind": self.kind, "artefact": self.path.name, "sha256": self.sha256, "modality": self.doc["modality"],
                "version": self.doc["version"], "bench_version": bench.get("version"),
                "fit_manifest_sha256": bench.get("manifest_sha256"), "excluded_folds": self.excluded_folds,
                "detectors": sorted(self.keys.values()), "reviewed_by": self.doc.get("reviewed_by"),
                "context_source": self.context_source,
                "decision_policy": decision_policy(self.doc, self.sha256, content_measured=self.content is not None)}

    def context(self, item: Item, variant: str, transform: str) -> Context | None:
        if self.context_source == "manifest":
            return item.context
        if self._scored is None:
            self._scored = ScoredContexts(self.store.dir, self.doc["modality"])
        return self._scored.get(item.item_id, variant, transform)

    def slice_id(self, item: Item, variant: str, transform: str, context: Context) -> str:
        """Prefer the production slice for the scored bytes; never borrow native geometry for an unknown transform."""
        if self.context_source == "scored" and self._scored is not None:
            recorded = self._scored.slice_id(item.item_id, variant, transform)
            if recorded:
                return recorded
            if self.doc["modality"] == "image":
                return abstain.image_slice(context.min_side, context.min_side, context.jpeg_q, context.face_px)
        if transform != "none":
            return f"{self.doc['modality']}/unknown"
        if self.doc["modality"] == "video":
            return abstain.video_slice(item.height, item.duration_s, context.face_px)
        if self.doc["modality"] == "audio":
            return abstain.audio_slice(item.duration_s)
        return abstain.image_slice(item.width, item.height, context.jpeg_q, context.face_px)

    def rows(self, item: Item, variant: str, transform: str) -> dict[str, dict | None]:
        return {det: self.store.table(key).get((item.item_id, variant, transform)) for det, key in self.keys.items()}

    def scores(self, rows: Mapping[str, dict | None]) -> dict[str, tuple[float | None, float | None] | None]:
        return {det: (r["prob"], r["logit"]) if r is not None and r["status"] == "ok" else None for det, r in rows.items()}

    def not_run(self, rows: Mapping[str, dict | None], context: Context) -> tuple[str, ...]:
        """D-46 / D-49 as the runtime decides: artefact detectors whose row failed or was skipped, or whose row is absent or the
        scorer's placeholder while their gate is open. The runtime caps such an item and publishes no estimate."""
        terms = context.terms()
        return tuple(d for j, d in enumerate(self.fuser.detectors)
                     if (rows.get(d) is not None and rows[d]["status"] in NOT_RUN_ROW_STATUSES)
                     or ((rows.get(d) is None or rows[d].get("placeholder")) and self.fuser.space.gate_open(j, terms)))

    def score(self, item: Item, variant: str, transform: str) -> Scored | None:
        rows = self.rows(item, variant, transform)
        present = [r for r in rows.values() if r is not None]
        if not present:
            return None
        context = self.context(item, variant, transform)
        if context is None:
            return None
        scored_at = max(r["scored_at"] or "" for r in present)
        not_run = self.not_run(rows, context)
        if not_run:
            return Scored(s=math.nan, sets05=None, scored_at=scored_at, not_run=not_run)
        scores = self.scores(rows)
        out = self.fuser.apply(scores, context)
        slice_id = self.slice_id(item, variant, transform, context)
        _, env = abstain.cap_for_slice("ai_supported", slice_id,
                                      ((self.doc.get("metrics") or {}).get("test") or {}).get("slices"))
        return Scored(s=out.probability, sets05=tuple(out.sets["0.05"]), scored_at=scored_at,
                      slice_id=slice_id, within_envelope=env.within,
                      rules=(*self.content_marks(item, variant, transform),
                             *self.check_marks(scores, out.band, context, tuple(out.sets["0.05"]), out.probability)))

    def content_marks(self, item: Item, variant: str, transform: str) -> tuple[str, ...]:
        """D-65: the conditions of `abstain.content_conditions` that hold for the scored bytes, after the mark that says
        whether they were measured. Empty when this system applies no content rule."""
        if self.content is None:
            return ()
        measured = self.content.get(item.item_id, variant, transform)
        if measured is None:
            return (CONTENT_NOT_MEASURED,)
        profile = {"screenshot_likelihood": measured["screenshot_likelihood"]}
        return (CONTENT_MEASURED, *abstain.content_conditions("image", profile, measured, self.doc.get("bench")))

    def check_marks(self, scores: Mapping[str, tuple[float | None, float | None] | None], band: str,
                    context: Context, sets05: Sequence[str] | None = None, estimate: float | None = None) -> tuple[str, ...]:
        """D-64: fewer than two checks outside H3 count, and whether the runtime then issues a weaker code than the
        strong one of the item's band (`abstain.strong_needs_two`). N-19f: the count is the runtime's, the rows of the
        published view that are not zero (`abstain.checks_entered`), not the usable scores. N-19i: `sets05`, the item's
        set at α = 0.05. No check entered: the runtime's result is not assessable. A set of one answer that a condition
        of a decisive code takes (N-19m: `abstain.rule_for`, the function the runtime calls, on the same view and on
        `estimate`, the item's estimate, which the runtime clamps and reads as published; D-77 first, then D-76 in its two
        forms): inconclusive, under the one mark that is the result's reason. Without `estimate` D-77 is not asked."""
        rows = self.fuser.neutral_view(scores, context).rows
        n_scored = abstain.checks_entered(rows, self.h3)
        marks: list[str] = []
        if n_scored < abstain.MIN_CHECKS_FOR_STRONG:
            code = STRONG_CODE.get(band)
            weakened = code is not None and abstain.strong_needs_two(code, n_scored) != code
            marks += [FEWER_THAN_TWO_CHECKS, *((STRONG_NOT_ISSUED,) if weakened else ())]
        answer = ANSWER_CODE.get(tuple(sets05)) if sets05 is not None else None
        if n_scored == 0:
            marks.append(NO_CHECK_ENTERED)
        elif answer is not None:
            # without the estimate the scale's condition holds (an estimate of 0 or 1 stands in either answer's own part)
            at = clamp_probability(estimate) if estimate is not None else (1.0 if abstain.SIDE_OF_CODE[answer] > 0 else 0.0)
            rule = abstain.rule_for(answer, at, rows, self.h3)
            if rule is not None:
                marks.append(RULE_MARKS[rule])
        return tuple(marks)


def declared_h3(doc: Mapping[str, Any]) -> frozenset[str]:
    """N-19i (R5-01): the detectors of an artefact that are outside every count of checks, as the runtime takes them: from
    the checks' own declarations (the model manifest, whose hypotheses the product's entries carry), else from the
    artefact's document. The document of the packaged video fuser declares none for the container check, which declares
    H3 itself: the report counted its row, and the runtime did not."""
    from app.engine.manifest import load_manifest

    own = {d.id: [h.value for h in d.hypotheses] for d in load_manifest().detectors}
    return frozenset(d["id"] for d in doc["detectors"] if H3 in (own.get(d["id"]) or d.get("hypotheses") or []))


System = ScoreSystem | ArtefactSystem


@dataclass
class Evaluated:
    items: list[Item]
    s: np.ndarray
    sets05: list[tuple[str, ...] | None]
    dates: list[str | None]
    not_run: list[Item] = field(default_factory=list)  # D-49: capped items without an estimate; only inconclusive rates count them
    within_envelope: list[bool | None] = field(default_factory=list)
    slice_ids: list[str | None] = field(default_factory=list)
    rules: list[tuple[str, ...]] = field(default_factory=list)  # D-64, D-65: per scored item, as `Scored.rules`

    @property
    def y(self) -> np.ndarray:
        return np.asarray([i.y for i in self.items], dtype=np.int64)

    @property
    def groups(self) -> list[str]:
        return [i.group for i in self.items]

    def take(self, keep: Callable[[Item], bool]) -> Evaluated:
        idx = [k for k, item in enumerate(self.items) if keep(item)]
        return Evaluated([self.items[k] for k in idx], self.s[idx] if idx else np.zeros(0), [self.sets05[k] for k in idx],
                         [self.dates[k] for k in idx], [i for i in self.not_run if keep(i)],
                         [self.within_envelope[k] for k in idx] if self.within_envelope else [],
                         [self.slice_ids[k] for k in idx] if self.slice_ids else [],
                         [self.rules[k] for k in idx] if self.rules else [])

    def counts(self) -> dict[str, int]:
        y = self.y
        return {"n_real": int((y == 0).sum()), "n_fake": int((y == 1).sum())}


def evaluate(system: System, items: Iterable[Item], *, variant: str = "norm", transform: str = "none") -> tuple[Evaluated, int]:
    """(scored items, with the not-run items (D-49) held apart in `Evaluated.not_run`, count without a final score)."""
    kept, s, sets, dates, not_run, missing = [], [], [], [], [], 0
    within, slice_ids, rules = [], [], []
    for item in items:
        scored = system.score(item, variant, transform)
        if scored is None:
            missing += 1
            continue
        if scored.not_run:
            not_run.append(item)
            continue
        kept.append(item)
        s.append(scored.s)
        sets.append(scored.sets05)
        dates.append(scored.scored_at)
        within.append(scored.within_envelope)
        slice_ids.append(scored.slice_id)
        rules.append(scored.rules)
    return Evaluated(kept, np.asarray(s, dtype=np.float64), sets, dates, not_run, within, slice_ids, rules), missing


# --- size-matched view (D-45) ------------------------------------------------------------------------------

SIZE_BUCKETS = ((0, 384, "<384"), (384, 768, "384-767"), (768, 1025, "768-1024"), (1025, 1537, "1025-1536"), (1537, None, ">1536"))
SIZE_SALT = "vb-size:"


def size_bucket(item: Item) -> str:
    """The min-side bucket of an item's native (= norm) dimensions, `unknown` without them."""
    if not item.width or not item.height:
        return "unknown"
    side = min(item.width, item.height)
    return next(name for lo, hi, name in SIZE_BUCKETS if side >= lo and (hi is None or side < hi))


def size_matched(ev: Evaluated) -> tuple[Evaluated, dict[str, Any]]:
    """In each size bucket every item of the rarer class (relative to the pooled fake-per-real ratio) plus the first items of
    the other class by salted sha256("vb-size:" + item_id), at the pooled ratio. Items keep their order."""
    n0, n1 = ev.counts()["n_real"], ev.counts()["n_fake"]
    if not n0 or not n1:
        return ev, {"matched": False, "reason": "one class only", "buckets": {}}
    ratio = n1 / n0
    members: dict[str, dict[int, list[int]]] = defaultdict(lambda: {0: [], 1: []})
    for k, item in enumerate(ev.items):
        members[size_bucket(item)][item.y].append(k)

    def salted(indices: list[int], n: int) -> list[int]:
        return sorted(indices, key=lambda k: hashlib.sha256(f"{SIZE_SALT}{ev.items[k].item_id}".encode()).hexdigest())[:n]

    keep: list[int] = []
    buckets: dict[str, dict[str, int]] = {}
    for name in [*(b for *_, b in SIZE_BUCKETS), "unknown"]:
        if name not in members:
            continue
        reals, fakes = members[name][0], members[name][1]
        if len(fakes) >= len(reals) * ratio:
            kept_real, kept_fake = reals, salted(fakes, min(len(fakes), round(len(reals) * ratio)))
        else:
            kept_real, kept_fake = salted(reals, min(len(reals), round(len(fakes) / ratio))), fakes
        keep += kept_real + kept_fake
        buckets[name] = {"n_real": len(reals), "n_fake": len(fakes), "kept_real": len(kept_real), "kept_fake": len(kept_fake)}
    keep.sort()
    matched = Evaluated([ev.items[k] for k in keep], ev.s[keep] if keep else np.zeros(0), [ev.sets05[k] for k in keep],
                        [ev.dates[k] for k in keep], list(ev.not_run),
                        [ev.within_envelope[k] for k in keep] if ev.within_envelope else [],
                        [ev.slice_ids[k] for k in keep] if ev.slice_ids else [],  # not-run items have no s to match: all kept
                        [ev.rules[k] for k in keep] if ev.rules else [])
    return matched, {"matched": True, "pooled_fake_per_real": ratio, "buckets": buckets}


# --- metrics on one set ------------------------------------------------------------------------------------

def thresholds(cal_reals: Evaluated, alphas: Sequence[float] = ALPHAS) -> dict[str, dict[str, Any]]:
    if not cal_reals.items:
        raise ReportError("no scored `cal` reals: τ_α cannot be set")
    strata = [i.real_stratum or "unknown" for i in cal_reals.items]
    return {_KEY.format(a): {"tau": mt.operational_threshold(cal_reals.s, a, strata), "n_real": len(strata),
                             "strata": dict(sorted(Counter(strata).items())),
                             "rule": "⌈(n₀+1)(1−α)⌉-th smallest cal real s, equal stratum weights"} for a in alphas}


def _ci(boot: mt.GroupedBootstrap | None, stat: Callable[[np.ndarray], float | None]) -> dict | None:
    return None if boot is None else boot.ci(stat)


def set_metrics(ev: Evaluated, taus: Mapping[str, Mapping[str, Any]], *, conformal: bool, n_boot: int, seed: int) -> dict:
    s, y = ev.s, ev.y
    out: dict[str, Any] = {**ev.counts()}
    if not ev.items:
        return out
    boot = mt.GroupedBootstrap(ev.groups, n_boot=n_boot, seed=seed) if n_boot > 0 else None
    both = out["n_real"] > 0 and out["n_fake"] > 0
    if both:
        out["auc"] = {"value": mt.auc(s, y), "ci95": _ci(boot, lambda i: mt.auc(s[i], y[i]))}
    out["tpr_at_fpr"], out["fpr_at_tau"] = {}, {}
    for key, entry in taus.items():
        tau = entry["tau"]
        if out["n_fake"]:
            rate = mt.tpr_at(s, y, tau).as_json()
            rate["ci95"] = _ci(boot, lambda i, t=tau: mt.tpr_at(s[i], y[i], t).value)
            out["tpr_at_fpr"][key] = rate
        if out["n_real"]:
            out["fpr_at_tau"][key] = mt.fpr_at(s, y, tau).as_json()
    if both:
        out["balanced_accuracy"] = {"0.05": mt.balanced_accuracy(s, y, taus["0.05"]["tau"])}
        ece = mt.ece(s, y)
        assert ece is not None
        out["ece"] = {"value": ece["ece"], "mce": ece["mce"], "bins": ece["bins"], "balanced": True, "table": ece["table"],
                      "ci95": _ci(boot, lambda i: (mt.ece(s[i], y[i]) or {}).get("ece") if len(set(y[i].tolist())) == 2 else None)}
        out["brier"] = {"value": mt.brier(s, y), "balanced": True,
                        "ci95": _ci(boot, lambda i: mt.brier(s[i], y[i]) if len(set(y[i].tolist())) == 2 else None)}
        out["log_loss"] = {"value": mt.log_loss(s, y), "balanced": True}
        reals = ev.take(lambda item: item.y == 0)
        out["oracle_tpr_at_fpr"] = {
            key: mt.tpr_at(s, y, mt.operational_threshold(reals.s, float(key), [i.real_stratum or "unknown" for i in reals.items])).as_json()
            for key in taus}
    if conformal and all(x is not None for x in ev.sets05):
        truth = [("ai" if item.y == 1 else "real") in (sets or ()) for item, sets in zip(ev.items, ev.sets05, strict=True)]
        out["coverage"] = {"0.05": mt.class_coverage(truth, y)}
        out["selective"] = mt.selective(*_decided(ev))
        out["selective_raw_conformal"] = mt.selective(*_decided(ev, apply_envelope=False))
        out["selective_policy"] = ("statistical decision after the loaded artefact envelope cap and, where the content rule is "
                                   "applied (decision_rules), the cap of content the bench does not hold; an item whose "
                                   "estimate prints outside the part of the scale of its answer (D-77), that no check "
                                   "supports or that the checks together do not support (D-76), or in which no check "
                                   "entered is not decided; excludes documentary overrides")
        out["decision_rules"] = rule_counts(ev)
    return out


def _rates(ev: Evaluated, taus: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    s, y = ev.s, ev.y
    out: dict[str, Any] = {**ev.counts(), "tpr_at_fpr": {}, "fpr_at_tau": {}}
    for key, entry in taus.items():
        if out["n_fake"]:
            out["tpr_at_fpr"][key] = mt.tpr_at(s, y, entry["tau"]).as_json()
        if out["n_real"]:
            out["fpr_at_tau"][key] = mt.fpr_at(s, y, entry["tau"]).as_json()
    return out


def _decided(ev: Evaluated, *, apply_envelope: bool = True) -> tuple[list[int | None], list[int]]:
    """Decisions and labels of the scored items plus the not-run items, which publish no decision (D-49)."""
    decided = [1 if sets == ("ai",) else 0 if sets == ("real",) else None for sets in ev.sets05]
    if apply_envelope and ev.within_envelope:
        decided = [decision if inside is not False else None
                   for decision, inside in zip(decided, ev.within_envelope, strict=True)]
    if apply_envelope and ev.rules:  # D-65: content the bench does not hold is capped like an input outside the envelope
        # D-76, N-19f: an item that no check supports, or in which no check entered, is not decided either
        decided = [None if content_capped(marks) or not_decided(marks) else decision
                   for decision, marks in zip(decided, ev.rules, strict=True)]
    return decided + [None] * len(ev.not_run), [*ev.y.tolist(), *(i.y for i in ev.not_run)]


def content_capped(marks: Sequence[str]) -> bool:
    return any(name in marks for name in abstain.CONTENT_CONDITIONS)


def not_decided(marks: Sequence[str]) -> bool:
    """D-76, D-77, N-19f: the runtime publishes the item as inconclusive (a condition takes the answer of its set: the
    estimate's part of the scale, no check that supports it, the checks together) or as not assessable (no check entered)."""
    return any(name in marks for name in NOT_DECIDED_MARKS)


def rule_counts(ev: Evaluated) -> dict[str, Any] | None:
    """D-65 "Bench" and D-64 on the scored items of a set (the items a check did not run on publish no decision and are
    not counted here). For each content condition: `n`, the scored items it holds for, by class; `n_capped`, those of
    them whose decision it takes away (one answer in the set at α = 0.05, inside the slice envelope). An item under two
    conditions is counted under each and once in `any`. `content` is None where no content rule is applied."""
    if not ev.rules:
        return None

    def count(keep: Callable[[Sequence[str]], bool], *, capped: bool = False) -> dict[str, int]:
        y = [item.y for item, marks in zip(ev.items, ev.rules, strict=True) if keep(marks)]
        out = {"n": len(y), "n_real": y.count(0), "n_fake": y.count(1)}
        if capped:
            inside = ev.within_envelope or [True] * len(ev.items)
            out["n_capped"] = sum(1 for marks, sets, within in zip(ev.rules, ev.sets05, inside, strict=True)
                                  if keep(marks) and sets in (("ai",), ("real",)) and within is not False)
        return out

    content = None
    if any(CONTENT_MEASURED in marks or CONTENT_NOT_MEASURED in marks for marks in ev.rules):
        content = {"measured": count(lambda marks: CONTENT_MEASURED in marks)["n"],
                   "not_measured": count(lambda marks: CONTENT_NOT_MEASURED in marks)["n"],
                   **{name: count(lambda marks, name=name: name in marks, capped=True) for name in abstain.CONTENT_CONDITIONS},
                   "any": count(content_capped, capped=True)}
    return {"n_scored": len(ev.items), "content": content,
            "fewer_than_two_checks": count(lambda marks: FEWER_THAN_TWO_CHECKS in marks),
            "strong_not_issued": count(lambda marks: STRONG_NOT_ISSUED in marks),
            # D-76 and N-19f, counted both ways as a content condition is: the items that carry the mark, and those of
            # them that lose a decision by it (one answer in the set, inside the slice envelope)
            NO_SUPPORTING_CHECK: count(lambda marks: NO_SUPPORTING_CHECK in marks, capped=True),
            NO_CHECK_ENTERED: count(lambda marks: NO_CHECK_ENTERED in marks, capped=True),
            # N-19m (D-77; the second condition of D-76), counted as D-76's first form is
            ESTIMATE_OFF_SCALE: count(lambda marks: ESTIMATE_OFF_SCALE in marks, capped=True),
            CHECKS_NOT_TOGETHER: count(lambda marks: CHECKS_NOT_TOGETHER in marks, capped=True)}


def _inconclusive(ev: Evaluated) -> dict | None:
    if not (ev.items or ev.not_run) or any(x is None for x in ev.sets05):
        return None
    return mt.selective(*_decided(ev))["inconclusive"]


def _dates(ev: Evaluated) -> list[str] | None:
    dates = sorted(d for d in ev.dates if d)
    return [dates[0], dates[-1]] if dates else None


# --- the report --------------------------------------------------------------------------------------------

@dataclass
class ReportInputs:
    manifest: Path
    system: System
    eval_split: str = "test"
    bench_version: str = "vb-mini"
    lgo_systems: Mapping[str, ArtefactSystem] | None = None
    robustness_reference: str = "native"
    n_boot: int = mt.BOOTSTRAP_B
    seed: int = 7
    latency: Mapping[str, Any] | None = None
    reproducibility: Mapping[str, Any] | None = None
    command: str | None = None
    git_commit: str | None = None
    generated_at: str | None = None
    modality_systems: Mapping[str, System] | None = None  # "video" / "audio" → a system over the native files (B-5)
    modality_lgo_systems: Mapping[str, Mapping[str, ArtefactSystem]] | None = None  # "video" → {vendor fold: refit} (V-8)
    scope: Path | None = None  # H-6: evaluate only the items of a scoring scope parquet (counted `outside_scope` otherwise);
    # its `robustness` column, when present, names the robustness subset (H-2 scopes flag it; the vb-1.1 manifest has no column)


def read_manifest(path: Path) -> list[dict]:
    import pyarrow.parquet as pq

    if not Path(path).is_file():
        raise ReportError(f"manifest {path} does not exist")
    table = pq.read_table(path)
    required = ("item_id", "modality", "label", "threat", "family", "lgo_fold", "group_key", "split", "real_stratum",
                "source_id", "width", "height", "jpeg_q_est", "has_exif", "has_c2pa", "face_px_max", "audit_ok")
    missing = [c for c in required if c not in table.column_names]
    if missing:
        raise ReportError(f"manifest is missing columns {missing}")
    return table.to_pylist()


def scope_items(path: Path) -> tuple[set[str], set[str] | None]:
    """(item ids, robustness-flagged item ids or None without a `robustness` column) of a scope parquet."""
    import pyarrow.parquet as pq

    if not Path(path).is_file():
        raise ReportError(f"scope {path} does not exist")
    table = pq.read_table(path)
    ids = table.column("item_id").to_pylist()
    if "robustness" not in table.column_names:
        return set(ids), None
    return set(ids), {i for i, flag in zip(ids, table.column("robustness").to_pylist(), strict=True) if flag is True}


def build_report(inputs: ReportInputs) -> tuple[dict, dict]:
    """(results document, error-rates document)."""
    modality_systems = dict(inputs.modality_systems or {})
    unknown = sorted(set(modality_systems) - set(AV_MODALITIES))
    if unknown:
        raise ReportError(f"unknown modality {unknown}: modality systems are {list(AV_MODALITIES)}")
    rows = read_manifest(inputs.manifest)
    system = inputs.system
    excluded: dict[str, Counter] = defaultdict(Counter)
    if inputs.scope is not None:
        in_scope, robust = scope_items(inputs.scope)
        for row in rows:
            if row["item_id"] not in in_scope:
                excluded[str(row["split"])][f"outside_scope:{row['modality']}"] += 1
        rows = [row if robust is None else {**row, "robustness": row["item_id"] in robust} for row in rows if row["item_id"] in in_scope]
    usable: list[Item] = []
    by_id: dict[str, Item] = {}
    for row in rows:
        if row["modality"] != "image":
            continue
        if row["audit_ok"] is False:
            excluded[str(row["split"])]["audit_failed"] += 1
            continue
        if row["has_c2pa"]:
            excluded[str(row["split"])]["c2pa_manifest"] += 1
            continue
        item = Item.from_row(row)
        if item.face_px is None and isinstance(system, ArtefactSystem) and system.context_source == "scored":
            scored = system.context(item, "norm", "none")  # the manifest never fills face_px_max; the face stage did (H-7)
            if scored is not None and scored.face_px:
                item = replace(item, face_px=scored.face_px)
        usable.append(item)
        by_id[item.item_id] = item
    split = inputs.eval_split
    cal_reals, cal_missing = evaluate(system, (i for i in usable if i.split == "cal" and i.y == 0))
    taus = thresholds(cal_reals)
    ev, eval_missing = evaluate(system, (i for i in usable if i.split == split))
    if not ev.items:
        raise ReportError(f"no scored image items in split {split!r}")
    boot = {"n_boot": inputs.n_boot, "seed": inputs.seed}
    name = f"image/{split}"
    matched, view = size_matched(ev)
    sets: dict[str, Any] = {name: set_metrics(matched, taus, conformal=system.conformal, **boot),
                            f"{name}/full": set_metrics(ev, taus, conformal=system.conformal, **boot)}
    views: dict[str, Any] = {"all": view}
    subsets: dict[str, Callable[[Item], bool]] = {
        "t2i": lambda i: i.y == 0 or i.threat == "t2i",
        "edit": lambda i: i.y == 0 or i.threat == "edit",
        "personalised": lambda i: i.y == 0 or i.threat == "personalised",
        "faceswap_vs_R5": lambda i: (i.y == 0 and i.real_stratum == "R5") or i.threat == "faceswap",
        "gan_face_vs_R5": lambda i: (i.y == 0 and i.real_stratum == "R5") or i.threat == "gan_face",
        "legacy": lambda i: (i.y == 0 and i.source_id in LEGACY_REAL_SOURCES) or (i.y == 1 and i.family == "legacy"),
    }
    for sub, keep in subsets.items():
        part = ev.take(keep)
        if part.counts()["n_fake"]:
            part_matched, views[sub] = size_matched(part)
            sets[f"{name}/{sub}"] = set_metrics(part_matched, taus, conformal=system.conformal, **boot)
            sets[f"{name}/full/{sub}"] = set_metrics(part, taus, conformal=system.conformal, **boot)

    real_strata = {k: _rates(ev.take(lambda i, k=k: i.y == 0 and i.real_stratum == k), taus)
                   for k in sorted({i.real_stratum for i in ev.items if i.y == 0 and i.real_stratum})}
    real_strata_by_size = {k: {b: _rates(ev.take(lambda i, k=k, b=b: i.y == 0 and i.real_stratum == k and size_bucket(i) == b), taus)
                               for b in sorted({size_bucket(i) for i in ev.items if i.y == 0 and i.real_stratum == k})}
                           for k in real_strata}
    families = {}
    for fam in sorted({i.family for i in ev.items if i.y == 1 and i.family}):
        part = ev.take(lambda i, f=fam: i.y == 1 and i.family == f)
        families[fam] = {"threat": part.items[0].threat, "lgo_fold": part.items[0].lgo_fold, **_rates(part, taus)}

    lgo = _lgo(system, inputs.lgo_systems, usable, ev, taus, size_match=True)
    robustness = _robustness(inputs, rows, by_id, taus)
    screenshot = (robustness.get("transforms") or {}).get("screenshot")
    if screenshot is not None:
        real_strata["screenshot"] = screenshot["screenshot_reals"]
    modalities = {modality: _modality_report(inputs, rows, modality, modality_system)
                  for modality, modality_system in modality_systems.items()}
    for part in modalities.values():
        sets.update(part.pop("sets"))
    lgo_by_modality = {f"{modality}/lgo": part.pop("lgo") for modality, part in modalities.items() if "lgo" in part}
    results = {
        "schema": SCHEMA,
        "generated_at": inputs.generated_at or datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_commit": inputs.git_commit,
        "command": inputs.command,
        "bench": {"version": inputs.bench_version, "manifest_sha256": sha256_file(inputs.manifest), "eval_split": split,
                  **({"scope_sha256": sha256_file(inputs.scope)} if inputs.scope is not None else {}),
                  "lockbox": split == "test", "variant": "norm",
                  "note": None if split == "test" else f"evaluated on `{split}`, not `test`: τ_α, ECE and coverage are in-sample",
                  "view": {"primary": "size_matched", "secondary": f"{name}/full", "salt": SIZE_SALT,
                           "buckets": [b for *_, b in SIZE_BUCKETS], "sets": views}},
        "system": system.describe(),
        "thresholds": taus,
        "counts": {"cal_reals_scored": len(cal_reals.items), "cal_reals_not_scored": cal_missing,
                   "cal_reals_not_run": len(cal_reals.not_run),
                   "eval_scored": ev.counts(), "eval_not_scored": eval_missing, "eval_not_run": len(ev.not_run),
                   "excluded": {k: dict(sorted(v.items())) for k, v in sorted(excluded.items())}},
        "sets": sets,
        "real_strata": real_strata,
        "real_strata_by_size": real_strata_by_size,
        "families": families,
        "lgo": lgo,
        "lgo_by_modality": lgo_by_modality,
        "modalities": modalities,
        "robustness": robustness,
        "slices": slices(ev),
        "field_conditions": slice_map.field_conditions(ev, robustness.get("_perturbed"), min_n=MIN_SLICE_CLASS_N),
        "invariants": invariants(system, ev, seed=inputs.seed) if isinstance(system, ArtefactSystem) else None,
        "latency": dict(inputs.latency) if inputs.latency else None,
        "checks": {"reproducibility": dict(inputs.reproducibility)} if inputs.reproducibility else {},
        "not_measured": [f"{m} (no {m} system: `--{m}-score-key`)" for m in AV_MODALITIES if m not in modalities],
    }
    return results, error_rates(results, ev, robustness_screenshot=robustness.get("_screenshot_ev"), taus=taus, lgo=lgo)


def _modality_report(inputs: ReportInputs, rows: list[dict], modality: str, system: System) -> dict:
    """Video or audio over the native files: τ from this modality's `cal` reals, sets, and (video) vendor-LGO folds."""
    excluded: dict[str, Counter] = defaultdict(Counter)
    usable: list[Item] = []
    for row in rows:
        if row["modality"] != modality:
            continue
        if row["audit_ok"] is False:
            excluded[str(row["split"])]["audit_failed"] += 1
            continue
        if row["has_c2pa"]:
            excluded[str(row["split"])]["c2pa_manifest"] += 1
            continue
        item = Item.from_row(row)
        if item.face_px is None and isinstance(system, ArtefactSystem) and system.context_source == "scored":
            scored = system.context(item, "native", "none")  # the face stage of the scored clip, as the runtime slice reads it
            if scored is not None and scored.face_px:
                item = replace(item, face_px=scored.face_px)
        usable.append(item)
    split = inputs.eval_split
    try:
        cal_reals, cal_missing = evaluate(system, (i for i in usable if i.split == "cal" and i.y == 0), variant="native")
        taus = thresholds(cal_reals)
    except ReportError as exc:
        raise ReportError(f"{modality}: {exc}") from exc
    ev, eval_missing = evaluate(system, (i for i in usable if i.split == split), variant="native")
    if not ev.items:
        raise ReportError(f"{modality}: no scored items in split {split!r}")
    name = f"{modality}/{split}"
    subsets: dict[str, Callable[[Item], bool]] = {"real": lambda i: i.y == 0}
    if modality == "audio":
        subsets |= {sub: (lambda i, src=sources: i.source_id in src) for sub, sources in AUDIO_SOURCE_SETS.items()}
    sets: dict[str, Any] = {}
    for sub, part in [(None, ev), *((sub, ev.take(keep)) for sub, keep in subsets.items())]:
        if not part.items:
            continue
        entry = set_metrics(part, taus, conformal=system.conformal, n_boot=inputs.n_boot, seed=inputs.seed)
        if modality == "audio" and entry["n_real"] and entry["n_fake"]:
            entry["eer"] = mt.eer(part.s, part.y)
        sets[name if sub is None else f"{name}/{sub}"] = entry
    leaf_slices = slice_map.video_leaf_slices if modality == "video" else slice_map.audio_leaf_slices
    out: dict[str, Any] = {
        "system": system.describe(), "variant": "native", "thresholds": taus, "sets": sets,
        "slices": leaf_slices(ev, min_n=MIN_SLICE_CLASS_N),
        "counts": {"cal_reals_scored": len(cal_reals.items), "cal_reals_not_scored": cal_missing,
                   "cal_reals_not_run": len(cal_reals.not_run),
                   "eval_scored": ev.counts(), "eval_not_scored": eval_missing, "eval_not_run": len(ev.not_run),
                   "excluded": {k: dict(sorted(v.items())) for k, v in sorted(excluded.items())}}}
    if modality == "video":
        out["lgo"] = _lgo(system, (inputs.modality_lgo_systems or {}).get(modality), usable, ev, taus, required=VIDEO_LGO_FOLDS,
                          variant="native")
    return out


def _lgo(system: System, lgo_systems: Mapping[str, ArtefactSystem] | None, usable: list[Item], ev: Evaluated,
         taus: Mapping[str, Mapping[str, Any]], *, required: Sequence[str] = LGO_FOLDS, variant: str = "norm",
         size_match: bool = False) -> dict:
    folds_present = sorted({i.lgo_fold for i in usable if i.y == 1 and i.lgo_fold})
    folds: dict[str, Any] = {}
    missing: list[str] = []
    for fold in sorted(set(folds_present) | set(required)):
        fold_items = [i for i in usable if i.y == 1 and i.lgo_fold == fold and i.split in ("fit", "cal", "test")]
        if not fold_items:
            missing.append(f"{fold}: no items in the manifest")
            continue
        fold_system: System = system
        fold_taus = taus
        if isinstance(system, ArtefactSystem):
            refit = (lgo_systems or {}).get(fold)
            if refit is None:
                missing.append(f"{fold}: no refit artefact without {fold}")
                continue
            if refit.excluded_folds != [fold]:
                raise ReportError(f"LGO artefact for {fold} excludes {refit.excluded_folds}, expected [{fold!r}]")
            fold_system = refit
            cal_reals, _ = evaluate(refit, (i for i in usable if i.split == "cal" and i.y == 0), variant=variant)
            fold_taus = thresholds(cal_reals)
        fakes, not_scored = evaluate(fold_system, fold_items, variant=variant)
        reals = ev.take(lambda i: i.y == 0) if fold_system is system else evaluate(
            fold_system, (i for i in ev.items if i.y == 0), variant=variant)[0]
        full = Evaluated(fakes.items + reals.items, np.concatenate([fakes.s, reals.s]), fakes.sets05 + reals.sets05,
                         fakes.dates + reals.dates, fakes.not_run + reals.not_run,
                         fakes.within_envelope + reals.within_envelope, fakes.slice_ids + reals.slice_ids,
                         fakes.rules + reals.rules)
        joined, fold_view = size_matched(full) if size_match else (full, None)
        entry = {"families": sorted({i.family for i in fakes.items if i.family}), "not_scored": not_scored,
                 "artefact_sha256": fold_system.sha256 if isinstance(fold_system, ArtefactSystem) else None,
                 "thresholds": {k: v["tau"] for k, v in fold_taus.items()}, **_rates(joined, fold_taus)}
        if entry["n_fake"] and entry["n_real"]:
            entry["auc"] = mt.auc(joined.s, joined.y)
            entry["ece"] = (mt.ece(joined.s, joined.y) or {}).get("ece")
        if size_match:  # D-45: the fold's rates on the size-matched view; the full view beside them
            entry["view"], entry["full"] = fold_view, _rates(full, fold_taus)
            if entry["full"]["n_fake"] and entry["full"]["n_real"]:
                entry["full"]["auc"] = mt.auc(full.s, full.y)
                entry["full"]["ece"] = (mt.ece(full.s, full.y) or {}).get("ece")
        folds[fold] = entry
    tprs = {f: e["tpr_at_fpr"]["0.05"]["value"] for f, e in folds.items() if e.get("tpr_at_fpr", {}).get("0.05")}
    eces = {f: e["ece"] for f, e in folds.items() if e.get("ece") is not None}
    complete = all(f in tprs for f in required)
    spread = ({"min": min(tprs.values()), "median": statistics.median(tprs.values()), "max": max(tprs.values())}
              if tprs else None)
    return {"folds": folds, "missing": missing, "required": list(required), "complete": complete,
            "macro": {"tpr_at_fpr": {"0.05": statistics.fmean(tprs[f] for f in required) if complete else None},
                      "ece": statistics.fmean(eces[f] for f in required) if all(f in eces for f in required) else None},
            "min_family": {"tpr_at_fpr": {"0.05": min(tprs[f] for f in required) if complete else None}},
            "spread_present_folds": {"tpr_at_fpr_0.05": spread},
            "note": "fold F = all F items (fit, cal, test) + eval-split reals; artefact systems use a refit without F"}


def _absent_detectors(system: System, subset: Sequence[Item], transform: str) -> list[str]:
    """Artefact detectors without a cache row for `transform` on an item another detector has a row for."""
    if not isinstance(system, ArtefactSystem):
        return []
    absent: set[str] = set()
    for item in subset:
        rows = system.rows(item, "native", transform)
        if any(r is not None for r in rows.values()):
            absent.update(det for det, r in rows.items() if r is None)
    return sorted(absent)


METADATA_DETECTOR_PREFIXES = ("img.meta.", "prov.")
METADATA_CONTEXT = ("has_camera_exif",)


def metadata_inputs(system: System) -> list[str]:
    """The fused inputs that read file metadata: metadata detectors, EXIF context terms and EXIF gates. `exif_forge` changes
    the EXIF block only, so a system without such inputs meets G5c / G11 on it by construction (J-4, ML audit P2)."""
    if isinstance(system, ArtefactSystem):
        detectors, doc = list(system.keys), system.doc
        terms = [f"ctx:{t}" for t in doc.get("context_terms") or [] if t in METADATA_CONTEXT]
        gates = [f"gate:{d}|{g}" for d, names in (doc.get("gates") or {}).items() for g in names if g in METADATA_CONTEXT]
    else:
        detectors, terms, gates = [system.key.split("@", 1)[0]], [], []
    return sorted(d for d in detectors if d.startswith(METADATA_DETECTOR_PREFIXES)) + sorted(terms) + sorted(gates)


def _robustness(inputs: ReportInputs, rows: list[dict], by_id: Mapping[str, Item], taus: Mapping[str, Mapping[str, Any]]) -> dict:
    flagged = [by_id[r["item_id"]] for r in pt.robustness_subset(rows) if r["modality"] == "image" and r["item_id"] in by_id]
    subset = [i for i in flagged if i.split == inputs.eval_split]  # J-4 (ML audit P1-3): robustness goals read the eval split only
    other_splits = Counter(str(i.split) for i in flagged if i.split != inputs.eval_split)
    reference = inputs.robustness_reference
    clean, _ = evaluate(inputs.system, subset, variant=reference, transform="none")
    clean_by_id = {item.item_id: k for k, item in enumerate(clean.items)}
    tau = taus["0.05"]["tau"]
    transforms: dict[str, Any] = {}
    reasons: dict[str, str] = {}
    screenshot_ev = None
    perturbed: dict[str, Evaluated] = {}  # H-7 field conditions pool the scored copies of the social / screenshot transforms
    for transform in pt.IMAGE_TRANSFORMS:
        absent = _absent_detectors(inputs.system, subset, transform)
        if absent:  # fusing without cached rows would stand in for detectors that were never run on t
            reasons[transform] = f"no cached rows for {', '.join(absent)}"
            continue
        pert, _ = evaluate(inputs.system, subset, variant="native", transform=transform)
        pert = pert.take(lambda i: i.item_id in clean_by_id)
        if not pert.items:
            continue
        idx = [clean_by_id[i.item_id] for i in pert.items]
        ref_s, y = clean.s[idx], pert.y
        entry = {"n_real": int((y == 0).sum()), "n_fake": int((y == 1).sum()), "tau": tau,
                 **mt.robustness_deltas(ref_s, y, pert.s, y, tau), "inconclusive_t": _inconclusive(pert),
                 "decision_rules": rule_counts(pert)}
        if transform == "exif_forge" and not metadata_inputs(inputs.system):
            entry["trivially_satisfied"] = "no metadata features"
        if transform == "screenshot":
            screenshot_ev = pert.take(lambda i: i.y == 1 or i.real_stratum in SCREENSHOT_REAL_STRATA)
            entry["screenshot_reals"] = _rates(pert.take(lambda i: i.y == 0 and i.real_stratum in SCREENSHOT_REAL_STRATA), taus)
        transforms[transform], perturbed[transform] = entry, pert
    return {"reference": {"variant": reference, "transform": "none"}, "subset_items": len(subset), "subset_split": inputs.eval_split,
            "subset_other_splits": dict(sorted(other_splits.items())), "metadata_inputs": metadata_inputs(inputs.system),
            "reference_scored": len(clean.items), "transforms": transforms,
            "not_scored": [t for t in pt.IMAGE_TRANSFORMS if t not in transforms], "not_scored_reasons": reasons,
            "_screenshot_ev": screenshot_ev, "_perturbed": perturbed}


def slices(ev: Evaluated) -> dict[str, dict[str, Any]]:
    """Runtime slice ids with their AUC, measured at ≥ `MIN_SLICE_CLASS_N` per class or merged with the nearest parent (H-7)."""
    return slice_map.image_leaf_slices(ev, min_n=MIN_SLICE_CLASS_N)


def invariants(system: ArtefactSystem, ev: Evaluated, *, seed: int = 7, n_items: int = INVARIANT_ITEMS) -> dict[str, Any]:
    fuser = system.fuser
    rng = np.random.default_rng(seed)
    order = sorted(ev.items, key=lambda i: i.item_id)
    chosen = [order[k] for k in sorted(rng.choice(len(order), size=min(n_items, len(order)), replace=False))]
    worst, violations, trials = 0.0, 0, 0
    for item in chosen:
        scores = system.scores(system.rows(item, "norm", "none"))
        context = system.context(item, "norm", "none") or item.context
        out = fuser.apply(scores, context)
        worst = max(worst, abs(math.fsum([out.decomposition.context_prior, *(r.logit for r in out.decomposition.rows)])
                               - out.logit))
        z = fuser.z_vector(scores)
        present = [j for j, v in enumerate(z) if v is not None]
        if not present:
            continue
        a, b = platt_params(fuser.doc["calibration"], context.platt_context())
        j = int(rng.choice(present))
        moved = list(z)
        moved[j] = float(z[j]) + float(rng.uniform(0.01, 2.0))
        before = a * float(np.dot(fuser.space.row(z, context), fuser.coef)) + b
        after = a * float(np.dot(fuser.space.row(moved, context), fuser.coef)) + b
        trials += 1
        violations += int(after < before - 1e-12)
    return {"n_items": len(chosen), "additivity_max_abs": worst, "monotonicity_trials": trials,
            "monotonicity_violations": violations, "seed": seed}


def error_rates(results: Mapping[str, Any], ev: Evaluated, *, robustness_screenshot: Evaluated | None,
                taus: Mapping[str, Mapping[str, Any]], lgo: Mapping[str, Any]) -> dict:
    bench = results["bench"]
    contexts: dict[str, tuple[Evaluated, str]] = {}
    for name, stratum in CAMERA_CONTEXTS.items():
        contexts[name] = (ev.take(lambda i, k=stratum: i.y == 1 or i.real_stratum == k), f"reals: stratum {stratum}; fakes: all")
    contexts["portrait face"] = (ev.take(lambda i: bool(i.face_px) and i.face_px >= 120), "face_px_max ≥ 120, both classes")
    contexts["small face"] = (ev.take(lambda i: bool(i.face_px) and i.face_px < 120), "0 < face_px_max < 120, both classes")
    contexts["low-res"] = (ev.take(lambda i: bool(i.width and i.height) and min(i.width, i.height) < 512),
                           "shorter side < 512 px, both classes")
    if robustness_screenshot is not None:
        contexts["screenshot"] = (robustness_screenshot, "`screenshot` transform: R1/R2 reals and robustness fakes")
    macro = lgo["macro"]["tpr_at_fpr"]["0.05"]
    rows = []
    for name, (part, selector) in contexts.items():
        families = sorted({i.family for i in part.items if i.y == 1 and i.family})
        for family in ["all", *families]:
            sub = part if family == "all" else part.take(lambda i, f=family: i.y == 0 or i.family == f)
            counts = sub.counts()
            if not counts["n_real"] and not counts["n_fake"]:
                continue
            fold = next((i.lgo_fold for i in sub.items if i.y == 1 and i.family == family), None) if family != "all" else None
            fold_tpr = ((lgo["folds"].get(fold) or {}).get("tpr_at_fpr") or {}).get("0.05", {}).get("value") if fold else None
            rows.append({"context": name, "family": family, "selector": selector, "bench_version": bench["version"],
                         **_rates(sub, taus), "inconclusive": _inconclusive(sub), "lgo_macro_tpr_at_fpr_0.05": macro,
                         "lgo_fold": fold, "lgo_fold_tpr_at_fpr_0.05": fold_tpr, "score_dates": _dates(sub)})
    return {"schema": ERROR_RATES_SCHEMA, "generated_at": results["generated_at"], "bench": bench,
            "system": results["system"], "thresholds": {k: v["tau"] for k, v in taus.items()}, "rows": rows,
            "not_measured": list(NOT_MEASURED_CONTEXTS),
            "notes": ["rates come from a benchmark, not from the analysed file",
                      "generators released after the bench date are untested",
                      "probabilities assume a 50:50 prior unless configured"]}


def public(results: Mapping[str, Any]) -> dict:
    """The results document without private working fields."""
    out = dict(results)
    out["robustness"] = {k: v for k, v in results["robustness"].items() if not k.startswith("_")}
    return out


def dumps(doc: Mapping[str, Any]) -> str:
    return json.dumps(_plain(doc), sort_keys=True, indent=1, ensure_ascii=False, allow_nan=False) + "\n"


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_plain(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


# --- artefact metrics and the lockbox log --------------------------------------------------------------------

def modality_artefact_metrics(results: Mapping[str, Any], modality: str) -> dict[str, Any]:
    """`metrics.test` of a video or audio artefact: its set, thresholds and runtime slices (H-9)."""
    if results["bench"]["eval_split"] != "test":
        raise ReportError("only a `test` evaluation may fill an artefact's metrics.test")
    part = (results.get("modalities") or {}).get(modality)
    if part is None:
        raise ReportError(f"the report has no {modality} system")
    main = results["sets"][f"{modality}/test"]
    out: dict[str, Any] = {
        "test": {"bench_version": results["bench"]["version"], "manifest_sha256": results["bench"]["manifest_sha256"],
                 "generated_at": results["generated_at"], "n0": main["n_real"], "n1": main["n_fake"],
                 "auc": (main.get("auc") or {}).get("value"), "ece": (main.get("ece") or {}).get("value"),
                 "brier": (main.get("brier") or {}).get("value"),
                 "tpr_at_fpr": {k: v["value"] for k, v in main.get("tpr_at_fpr", {}).items()},
                 "fpr_at_tau": {k: v["value"] for k, v in main.get("fpr_at_tau", {}).items()},
                 "thresholds": {k: v["tau"] for k, v in part["thresholds"].items()}, "slices": part["slices"]}}
    lgo = (results.get("lgo_by_modality") or {}).get(f"{modality}/lgo")
    if lgo is not None:
        out["lgo"] = {"macro_tpr_at_fpr_0.05": lgo["macro"]["tpr_at_fpr"]["0.05"],
                      "min_family_tpr_at_fpr_0.05": lgo["min_family"]["tpr_at_fpr"]["0.05"],
                      "folds": {f: (e.get("tpr_at_fpr") or {}).get("0.05", {}).get("value") for f, e in lgo["folds"].items()}}
    return out


def artefact_metrics(results: Mapping[str, Any]) -> dict[str, Any]:
    if results["bench"]["eval_split"] != "test":
        raise ReportError("only a `test` evaluation may fill an artefact's metrics.test")
    main = results["sets"][f"image/{results['bench']['eval_split']}"]
    full = results["sets"].get(f"image/{results['bench']['eval_split']}/full") or {}
    view = results["bench"].get("view") or {}
    return {
        "test": {"bench_version": results["bench"]["version"], "manifest_sha256": results["bench"]["manifest_sha256"],
                 "generated_at": results["generated_at"], "n0": main["n_real"], "n1": main["n_fake"],
                 "auc": (main.get("auc") or {}).get("value"), "ece": (main.get("ece") or {}).get("value"),
                 "brier": (main.get("brier") or {}).get("value"),
                 # J-3d: the numbers above are the size-matched view (D-45); the full test set beside them
                 "view": {"primary": view.get("primary"), "all": (view.get("sets") or {}).get("all")},
                 "full": {"n0": full.get("n_real"), "n1": full.get("n_fake"), "auc": (full.get("auc") or {}).get("value"),
                          "ece": (full.get("ece") or {}).get("value"), "brier": (full.get("brier") or {}).get("value")},
                 "tpr_at_fpr": {k: v["value"] for k, v in main.get("tpr_at_fpr", {}).items()},
                 "fpr_at_tau": {k: v["value"] for k, v in main.get("fpr_at_tau", {}).items()},
                 "thresholds": {k: v["tau"] for k, v in results["thresholds"].items()}, "slices": results["slices"]},
        "lgo": {"macro_tpr_at_fpr_0.05": results["lgo"]["macro"]["tpr_at_fpr"]["0.05"],
                "min_family_tpr_at_fpr_0.05": results["lgo"]["min_family"]["tpr_at_fpr"]["0.05"],
                "folds": {f: (e.get("tpr_at_fpr") or {}).get("0.05", {}).get("value") for f, e in results["lgo"]["folds"].items()}},
    }


def write_artefact_metrics(path: Path, results: Mapping[str, Any], modality: str = "image") -> str:
    """Fill `metrics.test` / `metrics.lgo` in place (the bootstrap block is untouched); returns the new sha256. `modality`
    names the report part whose system the artefact is (`image` = the main system)."""
    path = Path(path)
    doc = json.loads(path.read_text(encoding="utf-8"))
    system = results["system"] if modality == "image" else ((results.get("modalities") or {}).get(modality) or {}).get("system") or {}
    if system.get("sha256") != hashlib.sha256(path.read_bytes()).hexdigest():
        raise ReportError(f"{path.name} changed since the report was built")
    metrics = artefact_metrics(results) if modality == "image" else modality_artefact_metrics(results, modality)
    doc["metrics"] = {**doc.get("metrics", {}), **metrics}
    data = artefact_bytes(doc)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)
    return hashlib.sha256(data).hexdigest()


def append_test_access(log_path: Path, results: Mapping[str, Any]) -> None:
    systems = [results["system"], *(part["system"] for part in (results.get("modalities") or {}).values())]
    line = "\t".join([results["generated_at"], results.get("git_commit") or "unknown",
                      "+".join(s.get("sha256") or s.get("score_key") or "unknown" for s in systems),
                      results["bench"]["version"], results["bench"]["manifest_sha256"]])
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    with Path(log_path).open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


# --- markdown ------------------------------------------------------------------------------------------------

def _pct(value: float | None, digits: int = 1) -> str:
    return "–" if value is None else f"{100 * value:.{digits}f} %"


def _num(value: float | None, digits: int = 4) -> str:
    return "–" if value is None else f"{value:.{digits}f}"


def _rate(entry: Mapping[str, Any] | None) -> str:
    if not entry:
        return "–"
    ci = entry.get("wilson95")
    text = f"{_pct(entry['value'])} ({entry['k']}/{entry['n']}"
    text += f"; Wilson {_pct(ci[0])}–{_pct(ci[1])}" if ci else ""
    text += f"; rule of three ≤ {_pct(entry['rule_of_three'])}" if entry.get("rule_of_three") is not None else ""
    return text + ")"


def _interval(entry: Mapping[str, Any] | None) -> str:
    ci = (entry or {}).get("ci95")
    return "–" if not ci or ci.get("lo") is None else f"{_num(ci['lo'])}–{_num(ci['hi'])}"


def _decision_rule_lines(results: Mapping[str, Any]) -> list[str]:
    """The section "Decision rules": per set and robustness transform, how many scored items each rule holds for and how
    many decisions each content condition takes away (D-65 "Bench", D-64). Empty for results made before the rules."""
    named = [(f"`{name}`", entry.get("decision_rules")) for name, entry in results["sets"].items()]
    named += [(f"robustness: {name}", entry.get("decision_rules")) for name, entry in (results["robustness"].get("transforms") or {}).items()]
    named = [(name, rules) for name, rules in named if rules]
    if not named:
        return []
    policy = (results["system"].get("decision_policy") or {}).get("content_rule") or {}
    limit = policy.get("flat_block_limit")
    state = (f"applied: flat-block limit {limit:g} of {policy.get('limit_of_bench')}, measured on the scored bytes" if policy.get("applied")
             else policy.get("state") or "not recorded")

    def cell(entry: Mapping[str, Any] | None) -> str:
        return "–" if entry is None else f"{entry['n']} ({entry['n_real']} real, {entry['n_fake']} AI); {entry['n_capped']} decisions capped"

    def plain(entry: Mapping[str, Any]) -> str:
        return f"{entry['n']} ({entry['n_real']} real, {entry['n_fake']} AI)"

    lines = ["", "## Decision rules (D-64, D-65, D-76, D-77)", "",
             f"Content the bench does not hold (D-65), image system: {state}. A capped decision is one answer in the set at "
             "α = 0.05, inside the slice envelope, that the condition makes inconclusive; an item under two conditions is counted "
             "under each. A strong code needs two checks that scored (D-64): the decisions of this report are at α = 0.05, so that "
             "rule changes none of them; the columns “Fewer than two checks” and “Strong code not issued” count the items it "
             "concerns. A decision needs a check that supports it (D-76): an item whose set holds one answer while no check "
             "gives at least weak support for that answer is not decided, and neither is an item in which no check entered. A "
             "decision also needs the estimate itself in its answer’s part of the scale (D-77: 34 or less for the clean answer, "
             "65 or more for the AI answer, in whole points), and checks that together give the answer at least weak support "
             "(D-76 as amended). The conditions are asked in that order, D-77 first, and an item is counted under the one that "
             "is its result’s reason; the last four columns count them as a content condition is counted.", "",
             "| Set | Scored | Flat blocks | Screenshot rule | Greyscale | Any content condition | Not measured | Fewer than two checks | Strong code not issued | No check supports the answer | No check entered | Estimate outside its answer’s part of the scale | Checks together do not support the answer |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, rules in named:
        content = rules.get("content") or {}
        lines.append(f"| {name} | {rules['n_scored']} | {cell(content.get(abstain.FLAT_BLOCKS))} | {cell(content.get(abstain.SCREENSHOT_RULE))} | "
                     f"{cell(content.get(abstain.GREYSCALE))} | {cell(content.get('any'))} | {content.get('not_measured', '–')} | "
                     f"{plain(rules['fewer_than_two_checks'])} | {plain(rules['strong_not_issued'])} | "
                     f"{cell(rules.get(NO_SUPPORTING_CHECK))} | {cell(rules.get(NO_CHECK_ENTERED))} | "
                     f"{cell(rules.get(ESTIMATE_OFF_SCALE))} | {cell(rules.get(CHECKS_NOT_TOGETHER))} |")
    return lines


def render_markdown(results: Mapping[str, Any]) -> str:
    bench, system = results["bench"], results["system"]
    lines = [f"# Veritas-Bench results: {bench['version']} · `{bench['eval_split']}`", "",
             f"Generated {results['generated_at']} by `backend/scripts/vbench_report.py` (git {results.get('git_commit') or 'unknown'}).",
             ""]
    if bench.get("note"):
        lines += [f"> **Not a goal evaluation.** {bench['note']}.", ""]
    ident = system.get("artefact") and f"artefact `{system['artefact']}` (sha256 `{system['sha256'][:16]}…`, fitted on "
    ident = (f"{ident}{system.get('bench_version')}, reviewed_by: {system.get('reviewed_by')})" if ident
             else f"score key `{system.get('score_key')}`")
    lines += [f"- **System:** {ident}.", f"- **Manifest:** sha256 `{bench['manifest_sha256']}`; variant `{bench['variant']}`"
              + (f"; items of the scope sha256 `{bench['scope_sha256'][:16]}…` only." if bench.get("scope_sha256") else "."),
              "- **Thresholds from `cal` reals** (equal stratum weights): "
              + "; ".join(f"τ_{k} = {_num(v['tau'])} (n = {v['n_real']})" for k, v in results["thresholds"].items()) + ".",
              f"- **Counts:** {results['counts']['eval_scored']['n_real']} reals and {results['counts']['eval_scored']['n_fake']} "
              f"fakes scored; {results['counts']['eval_not_scored']} items without a score; "
              f"{results['counts'].get('eval_not_run', 0)} capped without an estimate because a check did not run "
              f"(cal reals {results['counts'].get('cal_reals_not_run', 0)}); excluded "
              f"{json.dumps(results['counts']['excluded'], ensure_ascii=False)}.", ""]
    if bench.get("view"):
        lines.insert(len(lines) - 1, f"- **View (D-45):** `image/{bench['eval_split']}`, its threat subsets and the LGO folds are "
                     "size-matched in min-side buckets (every item of the rarer class plus a salted sha256 sample of the other at "
                     f"the pooled class ratio); `image/{bench['eval_split']}/full` and `/full/<subset>` hold the full view.")
    for modality, part in {"image": {"system": system}, **(results.get("modalities") or {})}.items():
        policy = part["system"].get("decision_policy")
        if policy:
            env = policy["envelope"]
            lines += [f"- **{modality} selective policy:** loaded artefact `{policy['artefact_sha256']}`; "
                      f"envelope {env['status']} (sha256 `{env['sha256'] or 'not present'}`). "
                      f"Default α = {policy['conformal_alpha']:g}, slice AUC ≥ {policy['min_slice_auc']:.2f}; "
                      f"excludes {', '.join(policy['excludes'])}.", ""]
    for update in (results.get("artefact_writes") or {}).values():
        if update.get("notice"):
            lines += [f"> **Envelope update:** {update['notice']}", ""]
    lines += ["## Sets", "", "| Set | Reals | Fakes | AUC (95 % CI) | TPR@5 %FPR | TPR@1 %FPR | FPR at τ_0.05 | ECE (CI) | Brier | Inconclusive |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for name, entry in results["sets"].items():
        inconclusive = ((entry.get("selective") or {}).get("inconclusive") or {}).get("value")
        lines.append(f"| `{name}` | {entry['n_real']} | {entry['n_fake']} | {_num((entry.get('auc') or {}).get('value'))} "
                     f"({_interval(entry.get('auc'))}) | {_rate(entry.get('tpr_at_fpr', {}).get('0.05'))} | "
                     f"{_rate(entry.get('tpr_at_fpr', {}).get('0.01'))} | {_rate(entry.get('fpr_at_tau', {}).get('0.05'))} | "
                     f"{_num((entry.get('ece') or {}).get('value'))} ({_interval(entry.get('ece'))}) | "
                     f"{_num((entry.get('brier') or {}).get('value'))} | {_pct(inconclusive)} |")
    main = results["sets"][f"image/{bench['eval_split']}"]
    if main.get("coverage"):
        cov = main["coverage"]["0.05"]
        lines += ["", f"Conformal coverage at α = 0.05: reals {_rate(cov['real'])}, AI {_rate(cov['ai'])}."]
    lines += ["", "## Real strata (FPR at τ)", "", "| Stratum | n | FPR at τ_0.05 | FPR at τ_0.01 |", "|---|---|---|---|"]
    for name, entry in results["real_strata"].items():
        lines.append(f"| {name} | {entry['n_real']} | {_rate(entry['fpr_at_tau'].get('0.05'))} | {_rate(entry['fpr_at_tau'].get('0.01'))} |")
    by_size = results.get("real_strata_by_size") or {}
    if by_size:
        lines += ["", "## Real strata by min-side bucket (FPR at τ_0.05)", "", "| Stratum | Min side (px) | n | FPR at τ_0.05 |",
                  "|---|---|---|---|"]
        lines += [f"| {k} | {b} | {e['n_real']} | {_rate(e['fpr_at_tau'].get('0.05'))} |" for k, part in by_size.items()
                  for b, e in part.items()]
    lines += ["", "## Fake families (recall at τ)", "", "| Family | Threat | LGO fold | n | TPR at τ_0.05 | TPR at τ_0.01 |",
              "|---|---|---|---|---|---|"]
    for name, entry in results["families"].items():
        lines.append(f"| {name} | {entry['threat']} | {entry['lgo_fold'] or '–'} | {entry['n_fake']} | "
                     f"{_rate(entry['tpr_at_fpr'].get('0.05'))} | {_rate(entry['tpr_at_fpr'].get('0.01'))} |")
    lgo = results["lgo"]
    lines += ["", "## Leave-generator-out folds", "", "| Fold | Families | Fakes | Reals | TPR at τ_0.05 | AUC | ECE |", "|---|---|---|---|---|---|---|"]
    for fold, entry in lgo["folds"].items():
        lines.append(f"| {fold} | {', '.join(entry['families'])} | {entry['n_fake']} | {entry['n_real']} | "
                     f"{_rate(entry['tpr_at_fpr'].get('0.05'))} | {_num(entry.get('auc'))} | {_num(entry.get('ece'))} |")
    lines += ["", f"Macro TPR at τ_0.05 over F1–F7: {_num(lgo['macro']['tpr_at_fpr']['0.05'])}; minimum "
              f"{_num(lgo['min_family']['tpr_at_fpr']['0.05'])}; macro ECE {_num(lgo['macro']['ece'])}."]
    if lgo["missing"]:
        lines.append("Missing folds: " + "; ".join(lgo["missing"]) + ".")
    rob = results["robustness"]
    lines += ["", "## Robustness (fixed τ_0.05)", "",
              f"Reference `{rob['reference']['variant']}`/`none` on {rob['reference_scored']} of {rob['subset_items']} "
              "robustness items.", ""]
    if rob["transforms"]:
        lines += ["| Transform | Reals | Fakes | ΔTPR | ΔFPR | ΔAUC | Inconclusive under t |", "|---|---|---|---|---|---|---|"]
        for name, entry in rob["transforms"].items():
            lines.append(f"| {name} | {entry['n_real']} | {entry['n_fake']} | {_num(entry['delta_tpr'])} | {_num(entry['delta_fpr'])} | "
                         f"{_num(entry['delta_auc'])} | {_pct((entry.get('inconclusive_t') or {}).get('value'))} |")
    if rob["not_scored"]:
        reasons = rob.get("not_scored_reasons") or {}
        lines.append(f"\nNot scored: {', '.join(rob['not_scored'])}.")
        lines += [f"- {t}: {reasons[t]}" for t in rob["not_scored"] if t in reasons]
    lines += _decision_rule_lines(results)
    inv = results.get("invariants")
    if inv:
        lines += ["", "## Invariants", "", f"On {inv['n_items']} items: max |Σφ′ + b′ − logit s| = {inv['additivity_max_abs']:.3e}; "
                  f"{inv['monotonicity_violations']} of {inv['monotonicity_trials']} random +Δz moves lowered s."]
    sl = results["slices"]
    own = [k for k, v in sl.items() if v.get("measured_at") == k]
    merged = [k for k, v in sl.items() if v["auc"] is not None and v.get("measured_at") != k]
    lines += ["", "## Slices", "", f"{len(sl)} slices; {len(own)} measured (≥ {MIN_SLICE_CLASS_N} items per class), "
              f"{len(merged)} merged with a measured parent slice, {len(sl) - len(own) - len(merged)} not measured."]
    if own or merged:
        lines += ["", "| Slice | Reals | Fakes | AUC | Measured at |", "|---|---|---|---|---|"]
        lines += [f"| `{k}` | {sl[k]['n_real']} | {sl[k]['n_fake']} | {_num(sl[k]['auc'])} | `{sl[k]['measured_at']}` |"
                  for k in own + merged]
    conditions = results.get("field_conditions") or {}
    if conditions:
        lines += ["", "## Field conditions (EA-02 slice validation map)", "",
                  "| Condition | Reals | Fakes | AUC | Measured at |", "|---|---|---|---|---|"]
        lines += [f"| {c['label']} | {c['n_real']} | {c['n_fake']} | {_num(c['auc'])} | "
                  f"{'`' + c['measured_at'] + '`' if c['measured_at'] else 'not measured'} |" for c in conditions.values()]
    if results.get("checks", {}).get("reproducibility"):
        rep = results["checks"]["reproducibility"]
        lines += ["", "## Reproducibility", "", f"Two `calibrate` runs identical except `created_at`: "
                  f"**{rep.get('identical_except_created_at')}** ({rep.get('detail') or 'no detail'})."]
    if results.get("modalities"):
        lines += ["", "## Video and audio", ""]
        for modality, part in results["modalities"].items():
            counts, ident = part["counts"], part["system"].get("score_key") or part["system"].get("artefact")
            lines.append(f"- **{modality}:** system `{ident}`; variant `{part['variant']}`; "
                         + "; ".join(f"τ_{k} = {_num(v['tau'])} (n = {v['n_real']})" for k, v in part["thresholds"].items())
                         + f"; {counts['eval_scored']['n_real']} reals and {counts['eval_scored']['n_fake']} fakes scored, "
                         f"{counts['eval_not_scored']} without a score, {counts.get('eval_not_run', 0)} capped without an estimate "
                         f"because a check did not run (cal reals {counts.get('cal_reals_not_run', 0)}); "
                         f"excluded {json.dumps(counts['excluded'], ensure_ascii=False)}.")
            if part.get("slices"):
                sl = part["slices"]
                inside = sum(abstain.envelope(k, sl).within for k in sl)
                lines.append(f"  Runtime slices: {len(sl)} leaves, {inside} inside the envelope (AUC ≥ {abstain.MIN_SLICE_AUC:.2f} "
                             f"at ≥ {MIN_SLICE_CLASS_N} items per class directly or pooled; both classes present in the leaf): "
                             + "; ".join(f"`{k}` {e['n_real']}/{e['n_fake']} → {_num(e['auc'])} at `{e['measured_at'] or '–'}`"
                                         for k, e in sl.items()) + ".")
        eers = {name: entry["eer"] for name, entry in results["sets"].items() if entry.get("eer") is not None}
        if eers:
            lines += ["", "| Audio set | EER |", "|---|---|"] + [f"| `{name}` | {_pct(value)} |" for name, value in eers.items()]
        for name, entry_lgo in (results.get("lgo_by_modality") or {}).items():
            lines += ["", f"`{name}`:", "", "| Fold | Families | Fakes | Reals | TPR at τ_0.05 | AUC |", "|---|---|---|---|---|---|"]
            for fold, entry in entry_lgo["folds"].items():
                lines.append(f"| {fold} | {', '.join(entry['families'])} | {entry['n_fake']} | {entry['n_real']} | "
                             f"{_rate(entry['tpr_at_fpr'].get('0.05'))} | {_num(entry.get('auc'))} |")
            lines += ["", f"Macro TPR at τ_0.05 over {', '.join(entry_lgo['required'])}: "
                      f"{_num(entry_lgo['macro']['tpr_at_fpr']['0.05'])}; minimum {_num(entry_lgo['min_family']['tpr_at_fpr']['0.05'])}."]
            if entry_lgo["missing"]:
                lines.append("Missing folds: " + "; ".join(entry_lgo["missing"]) + ".")
    lines += ["", "## Not measured", ""] + [f"- {x}" for x in results["not_measured"]]
    return "\n".join(lines) + "\n"


def compare_artefacts(first: Path, second: Path) -> dict[str, Any]:
    """G12: two artefacts (and their bootstrap matrices) identical except `created_at`."""
    a, b = json.loads(Path(first).read_text(encoding="utf-8")), json.loads(Path(second).read_text(encoding="utf-8"))
    a.pop("created_at", None)
    b.pop("created_at", None)
    differing = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
    return {"identical_except_created_at": not differing, "differing_keys": differing,
            "detail": f"{Path(first).name} vs {Path(second).name}" + (f"; differing {differing}" if differing else "")}
