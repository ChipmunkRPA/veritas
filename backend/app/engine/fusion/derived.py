"""Derived features: partial-edit statistics (BUILD_SPEC H-6; ML §4.1 sub-fusers; D-41 context gating in the stacker) and
the calibrated D3 σ (H-8).

Localisers and forensic checks publish uncalibrated aggregates (raw kind `score`) or none, so `runtime.score_of` gives the
fuser nothing from them. A derived feature turns one of their statistics into a logit-scale input `z` that the partial-edit
sub-fuser and the union fuser can weigh like a detector score:

| Feature | Source | Statistic → z |
|---|---|---|
| `pe.mesorch.integrity` | `loc.mesorch` | integrity (mean of the top 1 % of reliable map values) → logit, clipped to ±8 |
| `pe.mesorch.region_area` | `loc.mesorch` | largest `region:` unit area share, 0 without a region → logit of max(area, 1e-4), ±8 |
| `pe.jpeg_dq.support` | `img.forensic.jpeg_dq` | (supporting + 0.5) / (tested + 1) DCT frequencies → logit, ±8 |
| `pe.copymove.pairs` | `img.forensic.copymove` | distinct clone pairs → log2(1 + pairs), ≤ 8 |
| `vd.d3.sigma` | `vid.synth.d3_xclip` | median clip σ → (σ_0.05 − σ) / s_σ, ±8, with σ_0.05 and s_σ from compressed real video (`d3_sigma.json`, `app/bench/d3_calibration.py`) |

- **Same numbers in both places.** The score cache holds the statistics in `aux` (`score_v2.detector_row`: `metrics`,
  `region_area_max`); `cache_row` derives the fitting rows from them and `with_derived` derives the runtime entries from
  the detector entries, through the same functions.
- **Missing.** A source that reported `ok` without the statistic (e.g. not a JPEG, or scored before the H-6 capture), or
  `not_applicable`, gives a missing feature (m_j = 1). So does `vd.d3.sigma` without its threshold file. A source that
  failed, timed out or was skipped passes its status on, so the runtime treats the feature as a check that did not run
  (D-46).
- **Compatibility.** A derived feature carries its source detector's weights hash, so an artefact fitted on it loads only
  while the source detector is enabled with the same weights (`registry.loaded_detector_hashes`).
"""
from __future__ import annotations

import functools
import json
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.engine.fusion.features import Z_CLIP, logit_clip

PREFIXES = ("pe.", "vd.")
VERSION = "0.1.0"
AREA_FLOOR = 1e-4
D3_SIGMA_FILE = Path(__file__).with_name("d3_sigma.json")


@dataclass(frozen=True)
class DerivedFeature:
    id: str
    source: str
    name: str
    value: Callable[[Mapping[str, float], float | None], float | None]  # (finite metrics, largest region area) → z


def _integrity(metrics: Mapping[str, float], area: float | None) -> float | None:
    v = metrics.get("integrity")
    return None if v is None else logit_clip(min(max(v, 0.0), 1.0))


def _region_area(metrics: Mapping[str, float], area: float | None) -> float | None:
    if "regions" not in metrics:
        return None
    return logit_clip(max(area or 0.0, AREA_FLOOR))


def _dq_support(metrics: Mapping[str, float], area: float | None) -> float | None:
    tested, supporting = metrics.get("dq_frequencies_tested"), metrics.get("dq_frequencies_supporting")
    if tested is None or supporting is None:
        return None
    return logit_clip((supporting + 0.5) / (tested + 1.0))


def _clone_pairs(metrics: Mapping[str, float], area: float | None) -> float | None:
    v = metrics.get("clone_pairs_distinct")
    return None if v is None else min(Z_CLIP, math.log2(1.0 + max(v, 0.0)))


@functools.lru_cache(maxsize=1)
def d3_sigma_constants() -> tuple[float, float] | None:
    """(σ_0.05, s_σ) from `d3_sigma.json` (written by `scripts/d3_recalibrate.py --constants`), None without the file."""
    if not D3_SIGMA_FILE.is_file():
        return None
    doc = json.loads(D3_SIGMA_FILE.read_text(encoding="utf-8"))
    threshold, scale = float(doc["sigma_threshold_0.05"]), float(doc["scale"])
    if not (math.isfinite(threshold) and math.isfinite(scale) and scale > 0):
        raise ValueError(f"{D3_SIGMA_FILE} has no finite threshold and positive scale")
    return threshold, scale


def _d3_sigma(metrics: Mapping[str, float], area: float | None) -> float | None:
    v, constants = metrics.get("sigma_median"), d3_sigma_constants()
    if v is None or constants is None:
        return None
    threshold, scale = constants
    return max(-Z_CLIP, min(Z_CLIP, (threshold - v) / scale))


FEATURES = (
    DerivedFeature("pe.mesorch.integrity", "loc.mesorch", "Mesorch integrity score (derived feature)", _integrity),
    DerivedFeature("pe.mesorch.region_area", "loc.mesorch", "Mesorch largest region area (derived feature)", _region_area),
    DerivedFeature("pe.jpeg_dq.support", "img.forensic.jpeg_dq", "Double-JPEG supporting frequencies (derived feature)", _dq_support),
    DerivedFeature("pe.copymove.pairs", "img.forensic.copymove", "Copy-move distinct clone pairs (derived feature)", _clone_pairs),
    DerivedFeature("vd.d3.sigma", "vid.synth.d3_xclip", "D3 σ against the compressed-footage threshold (derived feature)", _d3_sigma),
)
BY_ID = {f.id: f for f in FEATURES}


def is_derived(detector_id: str) -> bool:
    return detector_id.startswith(PREFIXES)


def input_transform(detector_id: str, weights_sha12: str) -> str:
    """The stacker input transform of a score-cache key: derived features are logits; heuristic forensics without pinned
    weights keep the v0.3 cap (features.py)."""
    return "logit_clip8" if is_derived(detector_id) or weights_sha12 != "noweights" else "heuristic_cap"


def finite_metrics(metrics: Mapping[str, Any] | None) -> dict[str, float]:
    out = {}
    for k, v in (metrics or {}).items():
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if math.isfinite(f):
            out[k] = f
    return out


def region_area_max(entry: Mapping[str, Any]) -> float | None:
    """Largest area share among a detector entry's `region:<k>` units, None without a region."""
    areas = [u["quality"]["area_share"] for u in entry.get("unit_scores") or []
             if str(u.get("unit_ref", "")).startswith("region:") and "area_share" in (u.get("quality") or {})]
    return max(areas) if areas else None


def with_derived(by_id: Mapping[str, Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    """The detector entries plus one entry per derived feature whose source ran: `ok` with a logit aggregate;
    `not_applicable` when the source was not applicable or lacks the statistic; the source's status and reason when it
    failed, timed out or was skipped."""
    out: dict[str, Mapping[str, Any]] = dict(by_id)
    for feature in FEATURES:
        source = by_id.get(feature.source)
        if source is None or feature.id in out:
            continue
        status = source.get("status")
        z = feature.value(finite_metrics(source.get("metrics")), region_area_max(source)) if status == "ok" else None
        if z is not None:
            status, reason = "ok", None
        elif status in ("ok", "not_applicable"):
            status, reason = "not_applicable", f"source {feature.source} has no statistic for this feature"
        else:
            reason = f"source {feature.source}: {source.get('reason') or status}"
        out[feature.id] = {"detector_id": feature.id, "name": feature.name, "version": VERSION, "status": status, "reason": reason,
                           "aggregate": None if z is None else {"unit_ref": "global", "raw": z, "raw_kind": "logit"},
                           "hypotheses": list(source.get("hypotheses") or []), "derived_from": feature.source}
    return out


def cache_row(feature: DerivedFeature, row: Mapping[str, Any], source_key: str) -> dict[str, Any]:
    """A score-cache row of `feature` from one row of its source detector's table (`score_schema` columns)."""
    aux = row.get("aux")
    aux = json.loads(aux) if isinstance(aux, str) and aux else dict(aux or {})
    base = {"item_id": row["item_id"], "variant": row["variant"], "transform": row["transform"], "prob": None, "logit": None,
            "aux": json.dumps({"source": source_key, "version": VERSION}, sort_keys=True), "runtime_ms": 0.0,
            "scored_at": row.get("scored_at")}
    if aux.get("reported_status") != "ok":
        status = row["status"] if row["status"] in ("failed", "skipped", "not_applicable") else "not_applicable"
        return {**base, "status": status, "reason": f"source {source_key} reported {aux.get('reported_status') or row['status']}"}
    z = feature.value(finite_metrics(aux.get("metrics")), aux.get("region_area_max"))
    if z is None:
        return {**base, "status": "not_applicable",
                "reason": f"source {source_key} row has no statistic for {feature.id} (not applicable, or scored before the H-6 capture)"}
    return {**base, "status": "ok", "reason": None, "logit": z}
