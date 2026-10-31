"""Format-parity shortcut test (ML §2.1 P3, §7.4): can container features alone separate fake from real images?

A standardised logistic regression on (JPEG quality estimate, width, height, has_exif) is scored by out-of-fold
predictions under `StratifiedGroupKFold(5, shuffle=True, random_state=seed)` over the image items in fit ∪ cal ∪ test.
`norm` features are read from the norm files (the variant pixel detectors are admitted on) and gate at AUC ≤ 0.65; the
same model on the manifest's native features is a diagnostic. A missing quality estimate (a lossless file) is 100.

**D-45 shortcut gate** (ML audit recipe step 2; the linear test above missed the v2.1 size prior at AUC 0.628):
- **2a** `bucketed_shortcut_test`: one-hot buckets of the scored norm bytes (min side, megapixels, aspect, both sides divisible
  by 64) and the scored face context, `HistGradientBoostingClassifier(max_depth=3, max_iter=200, learning_rate=0.1)` with
  class-balanced weights, out-of-fold under `StratifiedGroupKFold(5)` by group. The AUC weighted with the resolution
  balancing weights must be ≤ 0.60; the unweighted AUC and per-threat AUCs are reported; native jpeg_q / has_exif buckets
  are a diagnostic.
- **2b** `context_only_test`: the artefact's context-only logit a_k·(bias + βᵀc) + b_k on `cal`: max(AUC, 1 − AUC) ≤ 0.55.
- **2c** `context_term_check`: removing any remaining additive context term moves the LGO macro by ≤ 2 pts and test AUC by
  ≤ 0.01 (a removal record per term; no remaining term passes).
- **Structure** `d45_structure`: an image artefact has no additive size or face terms, one effective Platt intercept
  (a_k·bias + b_k equal in every context, D-50), and size gates only on size-limited detectors.
"""
from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from app.bench.media import probe_image
from app.engine.fusion.features import METADATA_TERMS, SIZE_FACE_TERMS, Context, fp_guarded

FEATURES = ("jpeg_q_est", "width", "height", "has_exif")
MAX_AUC = 0.65
CV_SPLITS = 5
LOSSLESS_Q = 100


@dataclass(frozen=True)
class ShortcutResult:
    variant: str
    n_real: int
    n_fake: int
    auc: float
    coefficients: dict[str, float]

    @property
    def passed(self) -> bool:
        return self.auc <= MAX_AUC


def population(rows: list[dict]) -> list[dict]:
    return [r for r in rows if r["modality"] == "image" and r.get("split") in ("fit", "cal", "test")]


def _vector(info: dict) -> list[float]:
    q = info.get("jpeg_q_est")
    return [float(LOSSLESS_Q if q is None else q), float(info["width"]), float(info["height"]), float(bool(info.get("has_exif")))]


def features(rows: list[dict], vbench_dir: Path, variant: str) -> np.ndarray:
    if variant == "native":
        return np.array([_vector(r) for r in rows], dtype=float).reshape(-1, len(FEATURES))
    if variant != "norm":
        raise ValueError(f"unknown variant {variant!r}")
    out = []
    for row in rows:
        if not row.get("norm_path"):
            raise ValueError(f"image item {row['item_id'][:12]} has no norm variant")
        out.append(_vector(probe_image((vbench_dir / row["norm_path"]).read_bytes())))
    return np.array(out, dtype=float).reshape(-1, len(FEATURES))


@fp_guarded  # Accelerate matmul raises spurious FP flags on finite inputs; the outputs are checked below
def shortcut_test(x: np.ndarray, y: np.ndarray | list[int], groups: list[str], *, variant: str, seed: int = 7) -> ShortcutResult:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    y = np.asarray(y, dtype=int)
    if len(set(y.tolist())) < 2:
        raise ValueError("the shortcut test needs real and fake items")
    if len(set(groups)) < CV_SPLITS:
        raise ValueError(f"the shortcut test needs at least {CV_SPLITS} groups")

    def model():
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))

    oof = np.zeros(len(y))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        for train, test in StratifiedGroupKFold(CV_SPLITS, shuffle=True, random_state=seed).split(x, y, groups):
            if len(set(y[train].tolist())) < 2:
                oof[test] = float(y[train].mean())
                continue
            oof[test] = model().fit(x[train], y[train]).predict_proba(x[test])[:, 1]
        full = model().fit(x, y)
    coefficients = {name: round(float(c), 4) for name, c in zip(FEATURES, full[-1].coef_[0])}
    return ShortcutResult(variant, int((y == 0).sum()), int((y == 1).sum()), float(roc_auc_score(y, oof)), coefficients)


# --- D-45 bucketed shortcut gate ---------------------------------------------------------------------------

MAX_BUCKETED_AUC = 0.60
MAX_CONTEXT_ONLY_AUC = 0.55
MAX_LGO_MACRO_DELTA_PTS = 2.0
MAX_TEST_AUC_DELTA = 0.01
EFFECTIVE_INTERCEPT_TOLERANCE = 1e-9  # D-50
MIN_SIDE_BUCKETS = ((0, 336, "<336"), (336, 384, "336-383"), (384, 512, "384-511"), (512, 768, "512-767"),
                    (768, 1024, "768-1023"), (1024, 1025, "=1024"), (1025, 1536, "1025-1535"), (1536, 2048, "1536-2047"),
                    (2048, None, ">=2048"))
MEGAPIXEL_BUCKETS = ((0.0, 0.5, "<0.5"), (0.5, 1.0, "0.5-1"), (1.0, 2.0, "1-2"), (2.0, 4.0, "2-4"), (4.0, 8.0, "4-8"),
                     (8.0, None, ">=8"))
ASPECTS = (("1:1", 1.0), ("4:3", 4 / 3), ("3:2", 1.5), ("16:9", 16 / 9))
ASPECT_TOLERANCE = 0.02
JPEG_Q_BUCKETS = ((0, 70, "<70"), (70, 90, "70-89"), (90, None, ">=90"))
FACE_PX_BUCKETS = ((0, 120, "<120"), (120, 256, "120-256"), (256, None, ">=256"))
GEOMETRY_COLUMNS = (*(f"min_side:{name}" for *_, name in MIN_SIDE_BUCKETS), *(f"megapixels:{name}" for *_, name in MEGAPIXEL_BUCKETS),
                    *(f"aspect:{name}" for name, _ in ASPECTS), "aspect:other", "div64")
CONTEXT_COLUMNS = ("has_face", *(f"face_px:{name}" for *_, name in FACE_PX_BUCKETS))
NATIVE_COLUMNS = ("jpeg_q:none", *(f"jpeg_q:{name}" for *_, name in JPEG_Q_BUCKETS), "has_exif")


def _bucket(value: float, buckets) -> str:
    return next(name for lo, hi, name in buckets if value >= lo and (hi is None or value < hi))


def aspect_name(width: int, height: int) -> str:
    """1:1, 4:3, 3:2 or 16:9 within 2 % in either orientation, else other."""
    ratio = max(width, height) / min(width, height)
    return next((name for name, r in ASPECTS if abs(ratio / r - 1.0) <= ASPECT_TOLERANCE), "other")


def bucket_columns(*, native: bool = False) -> tuple[str, ...]:
    return (*GEOMETRY_COLUMNS, *CONTEXT_COLUMNS, *(NATIVE_COLUMNS if native else ()))


def bucket_row(width: int, height: int, *, face_px: int | None, jpeg_q: int | None = None, has_exif: bool = False,
               native: bool = False) -> list[float]:
    """One-hot bucket features of one image (`bucket_columns` order); `native` adds the jpeg_q and has_exif diagnostics."""
    if not width or not height:
        raise ValueError("bucket features need a width and a height")
    on = {f"min_side:{_bucket(min(width, height), MIN_SIDE_BUCKETS)}", f"megapixels:{_bucket(width * height / 1e6, MEGAPIXEL_BUCKETS)}",
          f"aspect:{aspect_name(width, height)}"}
    if width % 64 == 0 and height % 64 == 0:
        on.add("div64")
    if face_px:
        on |= {"has_face", f"face_px:{_bucket(face_px, FACE_PX_BUCKETS)}"}
    if native:
        on.add("jpeg_q:none" if jpeg_q is None else f"jpeg_q:{_bucket(jpeg_q, JPEG_Q_BUCKETS)}")
        if has_exif:
            on.add("has_exif")
    return [1.0 if column in on else 0.0 for column in bucket_columns(native=native)]


FEATURE_GROUPS = {"min_side": ("min_side:",), "megapixels": ("megapixels:",), "aspect": ("aspect:",), "div64": ("div64",),
                  "face_context": ("has_face", "face_px:")}  # D-52: the 2a decomposition


def separation(auc: float) -> float:
    return max(auc, 1.0 - auc)


@dataclass(frozen=True)
class BucketedResult:
    variant: str
    n_real: int
    n_fake: int
    auc_weighted: float
    auc_unweighted: float
    per_threat: dict[str, dict[str, float | int]]
    groups: dict[str, dict[str, float]] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        """The split-time refusal (D-52): max(AUC, 1 − AUC) of the weighted AUC ≤ 0.60."""
        return separation(self.auc_weighted) <= MAX_BUCKETED_AUC

    @property
    def unweighted_passed(self) -> bool:
        return separation(self.auc_unweighted) <= MAX_BUCKETED_AUC

    def as_json(self) -> dict:
        return {"variant": self.variant, "n_real": self.n_real, "n_fake": self.n_fake, "auc_weighted": self.auc_weighted,
                "auc_unweighted": self.auc_unweighted, "separation_weighted": separation(self.auc_weighted),
                "separation_unweighted": separation(self.auc_unweighted), "limit": MAX_BUCKETED_AUC, "passed": self.passed,
                "unweighted_passed": self.unweighted_passed, "per_threat": self.per_threat, "groups": self.groups}


def _oof(x: np.ndarray, y: np.ndarray, group_arr: np.ndarray, seed: int) -> np.ndarray:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import StratifiedGroupKFold

    oof = np.zeros(len(y))
    for train, held in StratifiedGroupKFold(CV_SPLITS, shuffle=True, random_state=seed).split(x, y, group_arr):
        if len(set(y[train].tolist())) < 2:
            oof[held] = float(y[train].mean())
            continue
        model = HistGradientBoostingClassifier(max_depth=3, max_iter=200, learning_rate=0.1, random_state=7, class_weight="balanced")
        oof[held] = model.fit(x[train], y[train]).predict_proba(x[held])[:, 1]
    return oof


@fp_guarded
def bucketed_shortcut_test(x, y, groups: Sequence[str], *, min_side, threats: Sequence[str] | None = None, variant: str = "norm",
                           seed: int = 7, strata: Sequence[str] | None = None) -> BucketedResult:
    """Gate 2a: out-of-fold AUC of a shallow boosted model on bucketed container features, weighted with the resolution
    balancing weights (`stacker.resolution_weights` on `min_side`), unweighted beside it, and per threat against the
    pooled reals. D-50: with `strata` the weights are the fit's ω, `balancing_weights(y, strata, multipliers=resolution_weights(
    y, min_side, strata=strata))`. D-52: when `x` has the `bucket_columns` layout, the same model on each feature group's
    columns alone gives that group's weighted and unweighted AUC."""
    from sklearn.metrics import roc_auc_score

    from app.engine.fusion.stacker import balancing_weights, resolution_weights

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=int)
    group_arr = np.asarray([str(g) for g in groups], dtype=object)
    side = np.asarray(min_side, dtype=np.float64)
    if x.ndim != 2 or not len(x) == len(y) == len(group_arr) == len(side):
        raise ValueError("features, labels, groups and min sides differ in length")
    if len(set(y.tolist())) < 2:
        raise ValueError("the shortcut test needs real and fake items")
    if len(set(group_arr.tolist())) < CV_SPLITS:
        raise ValueError(f"the shortcut test needs at least {CV_SPLITS} groups")
    strata_arr = None if strata is None else np.asarray([str(s) for s in strata], dtype=object)
    if strata_arr is not None and len(strata_arr) != len(y):
        raise ValueError("strata and labels differ in length")
    oof = _oof(x, y, group_arr, seed)

    def aucs(mask: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
        if strata_arr is None:
            weights = resolution_weights(y[mask], side[mask])
        else:
            weights = balancing_weights(y[mask], strata_arr[mask], multipliers=resolution_weights(y[mask], side[mask], strata=strata_arr[mask]))
        return float(roc_auc_score(y[mask], scores[mask], sample_weight=weights)), float(roc_auc_score(y[mask], scores[mask]))

    everything = np.ones(len(y), dtype=bool)
    weighted, unweighted = aucs(everything, oof)
    per_threat: dict[str, dict[str, float | int]] = {}
    if threats is not None:
        threat_arr = np.asarray([str(t) for t in threats], dtype=object)
        for threat in sorted(set(threat_arr[y == 1].tolist())):
            w, u = aucs((y == 0) | (threat_arr == threat), oof)
            per_threat[threat] = {"n_fake": int(((y == 1) & (threat_arr == threat)).sum()), "auc_weighted": w, "auc_unweighted": u}
    columns = next((c for c in (bucket_columns(), bucket_columns(native=True)) if len(c) == x.shape[1]), None)
    by_group: dict[str, dict[str, float]] = {}
    for name, prefixes in (FEATURE_GROUPS.items() if columns is not None else ()):
        index = [j for j, column in enumerate(columns) if column.startswith(prefixes)]
        w, u = aucs(everything, _oof(x[:, index], y, group_arr, seed))
        by_group[name] = {"auc_weighted": w, "auc_unweighted": u}
    return BucketedResult(variant, int((y == 0).sum()), int((y == 1).sum()), weighted, unweighted, per_threat, by_group)


def context_only_logits(doc: Mapping, contexts: Sequence[Context]) -> np.ndarray:
    """a_k·(bias + βᵀc) + b_k per item: what the artefact says before any detector speaks. D-50: a detector that is not
    applicable only for want of a face (`missing: face`) and carries no `has_face` gate adds its missing weight on faceless
    items, `miss:<det>`·(1 − has_face), because that indicator is a function of the context."""
    from app.engine.fusion.calibration import platt_params

    coef, terms_used, gates = doc["coef"], list(doc.get("context_terms") or []), doc.get("gates") or {}
    face_missing = [d["id"] for d in doc.get("detectors") or [] if d.get("missing") == "face" and "has_face" not in gates.get(d["id"], [])]
    out = []
    for ctx in contexts:
        terms = ctx.terms()
        a, b = platt_params(doc["calibration"], ctx.platt_context())
        eta = coef["bias"] + sum(coef[f"ctx:{t}"] * terms[t] for t in terms_used)
        eta += sum(coef.get(f"miss:{det}", 0.0) * (1.0 - terms["has_face"]) for det in face_missing)
        out.append(a * eta + b)
    return np.asarray(out, dtype=np.float64)


def context_only_test(doc: Mapping, contexts: Sequence[Context], y) -> dict:
    """Gate 2b on `cal` items: max(AUC, 1 − AUC) of the context-only logit ≤ 0.55 (a prior in either direction is a
    shortcut)."""
    from sklearn.metrics import roc_auc_score

    y = np.asarray(y, dtype=int)
    if len(y) != len(contexts) or len(set(y.tolist())) < 2:
        raise ValueError("gate 2b needs one context per item and items of both classes")
    auc = float(roc_auc_score(y, context_only_logits(doc, contexts)))
    separation = max(auc, 1.0 - auc)
    return {"auc": auc, "separation": separation, "n_real": int((y == 0).sum()), "n_fake": int((y == 1).sum()),
            "limit": MAX_CONTEXT_ONLY_AUC, "passed": separation <= MAX_CONTEXT_ONLY_AUC,
            "context_terms": list(doc.get("context_terms") or [])}


def context_term_check(doc: Mapping, record: Mapping[str, Mapping] | None) -> dict:
    """Gate 2c: each remaining additive context term needs removal evidence {term: {lgo_macro_delta_pts, test_auc_delta,
    problems?}} with no problem, |Δ LGO macro| ≤ 2 pts and |Δ test AUC| ≤ 0.01. D-52: `scripts/shortcut_gate.py` computes
    the entries from the refit and results files it re-hashes (`context_evidence`). An artefact without additive context
    terms passes."""
    terms = list(doc.get("context_terms") or [])
    if not terms:
        return {"terms": {}, "missing": [], "passed": True, "note": "no remaining additive context term"}
    rows: dict[str, dict] = {}
    missing = []
    for term in terms:
        entry = (record or {}).get(term)
        if entry is None:
            missing.append(term)
            continue
        problems = list(entry.get("problems") or [])
        lgo, auc = entry.get("lgo_macro_delta_pts"), entry.get("test_auc_delta")
        lgo, auc = (None if lgo is None else float(lgo)), (None if auc is None else float(auc))
        rows[term] = {**entry, "lgo_macro_delta_pts": lgo, "test_auc_delta": auc, "problems": problems,
                      "passed": not problems and lgo is not None and auc is not None
                      and abs(lgo) <= MAX_LGO_MACRO_DELTA_PTS and abs(auc) <= MAX_TEST_AUC_DELTA}
    return {"terms": rows, "missing": missing, "passed": not missing and all(r["passed"] for r in rows.values()),
            "limits": {"lgo_macro_delta_pts": MAX_LGO_MACRO_DELTA_PTS, "test_auc_delta": MAX_TEST_AUC_DELTA}}


def d45_structure(doc: Mapping) -> dict:
    """An image artefact: no additive size or face terms, one effective Platt intercept, size gates only on size-limited
    detectors."""
    problems: list[str] = []
    if doc.get("modality") == "image":
        additive = [t for t in doc.get("context_terms") or [] if t in SIZE_FACE_TERMS]
        if additive:
            problems.append(f"additive size or face context terms {additive}")
        calibration, bias = doc["calibration"], float(doc["coef"]["bias"])
        effective = [float(entry["a"]) * bias + float(entry["b"]) for entry in (calibration["global"], *calibration["per_context"].values())]
        if max(effective) - min(effective) > EFFECTIVE_INTERCEPT_TOLERANCE:  # D-50: a_k·bias + b_k equal in every Platt context
            problems.append("per-context effective Platt intercepts (a_k·bias + b_k differ)")
        metadata = [t for t in doc.get("context_terms") or [] if t in METADATA_TERMS]
        if metadata:
            problems.append(f"metadata context terms {metadata}")
        metadata_gated = sorted(det for det, literals in (doc.get("gates") or {}).items() if any(lit.lstrip("!") in METADATA_TERMS for lit in literals))
        if metadata_gated:  # D-50: has_camera_exif is never a gate literal of an image fit; has_face / !has_face gates pass
            problems.append(f"metadata gate literals on {metadata_gated}")
        fixed = set((doc.get("size_policy") or {}).get("fixed_missing") or [])
        size_gated = sorted(det for det, literals in (doc.get("gates") or {}).items()
                            if any(lit.lstrip("!").startswith("min_side") for lit in literals) and det not in fixed)
        if size_gated:
            problems.append(f"size gates on detectors that are not size-limited {size_gated}")
    return {"problems": problems, "passed": not problems}
