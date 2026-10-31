"""Fuser artefact `veritas.fuser/1` (ML §4.8; ARCHITECTURE §4.4): fitting, serialisation and the runtime `Fuser`.

- `fit_fuser` fits the union fuser (Stages 2–4) on `fit` / `cal`, the per-detector Platt fits, d95, the grouped bootstrap
  and the hypothesis sub-fusers (same method on all reals plus the fakes of the hypothesis' threats; reported, never
  deciding). `test` items are never read (lockbox, ML §2.6).
- `write_artefact` writes the `.boot.npy` matrix, then the JSON with full float precision (sorted keys, indent 2), so two
  runs on the same inputs differ only in `created_at`.
- `Fuser.apply` runs Stages 1–4 on one input: s, band and sets, exact contributions, CI90, disagreement D. Bands use s
  before a deployment prior shift, because the conformal thresholds were fitted on the 50:50 scale.

- `Fuser.apply_sub` runs a fitted sub-fuser (same feature space, its own coefficients, Platt and conformal thresholds).

Stage 0 overrides (`decisive.py`), the loader (`registry.py`), abstention (`abstain.py`) and the published decision
(`runtime.py`) sit on top of this module.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import os
import re
import sys
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from app.engine.fusion import bootstrap as bs
from app.engine.fusion.calibration import clamp_probability, fit_calibration, platt_params, prior_logit
from app.engine.fusion.conformal import band, coverage, fit_conformal
from app.engine.fusion.contributions import (
    Decomposition,
    NeutralView,
    decompose,
    effective_coefficients,
    neutral_view,
)
from app.engine.fusion.disagreement import calibrated_logit, d95, disagreement_index, fit_signal_calibration
from app.engine.fusion.features import (
    CONTEXT_TERMS,
    GROUPS,
    METADATA_TERMS,
    SIZE_FACE_TERMS,
    Context,
    FeatureSpace,
    fp_guarded,
    transform_score,
)
from app.engine.fusion.policy import SizePolicy, default_policy
from app.engine.fusion.stacker import (
    CV_SPLITS,
    L2_GRID,
    balancing_weights,
    fit_stacker,
    resolution_weights,
    select_l2,
)

SCHEMA = "veritas.fuser/1"
LOOP_REVIEW_FLAG = "ralph-loop (automated; pending human review)"  # D-17
DEFAULT_BOOTSTRAP = 200
SUB_FUSERS = {
    "h1_synthesis": ("t2i", "personalised", "gan_face", "t2v"),
    "h2_face_swap": ("faceswap", "talking_head"),
    "h2_partial_edit": ("edit",),
    "voice": ("tts",),
}
MIN_SUB_FUSER_CLASS_N = 10
_SEMVER = re.compile(r"\d+\.\d+\.\d+(-[0-9A-Za-z.-]+)?")
_LABELS = {"real": 0, "fake": 1}


def _column(rows: Sequence[Mapping], key: str, default: str = "") -> np.ndarray:
    return np.asarray([str(r.get(key) if r.get(key) is not None else default) for r in rows], dtype=object)


@dataclass(frozen=True)
class Dataset:
    """Benchmark items sorted by item_id; z is n × J with NaN where a detector gave no usable score."""

    item_id: np.ndarray
    y: np.ndarray
    split: np.ndarray
    group_key: np.ndarray
    stratum: np.ndarray
    threat: np.ndarray
    contexts: tuple[Context, ...]
    z: np.ndarray
    detectors: tuple[str, ...]

    @classmethod
    def from_records(cls, records: Sequence[Mapping], detectors: Sequence[str], transforms: Mapping[str, str]) -> Dataset:
        """records: item_id, label (real | fake), split, group_key, stratum, threat, context, scores {det: (prob, logit)}."""
        if not records:
            raise ValueError("no benchmark items to fit")
        rows = sorted(records, key=lambda r: r["item_id"])
        ids = [r["item_id"] for r in rows]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate item ids")
        bad = sorted({str(r["label"]) for r in rows} - set(_LABELS))
        if bad:
            raise ValueError(f"unknown labels {bad} (expected real or fake)")
        detectors = tuple(detectors)
        z = np.full((len(rows), len(detectors)), np.nan)
        for i, row in enumerate(rows):
            for j, det in enumerate(detectors):
                score = row["scores"].get(det)
                value = None if score is None else transform_score(score[0], score[1], transforms[det])
                if value is not None:
                    z[i, j] = value
        return cls(item_id=np.asarray(ids, dtype=object), y=np.asarray([_LABELS[r["label"]] for r in rows]),
                   split=_column(rows, "split"), group_key=_column(rows, "group_key"),
                   stratum=_column(rows, "stratum", "unknown"), threat=_column(rows, "threat", "unknown"),
                   contexts=tuple(r["context"] for r in rows), z=z, detectors=detectors)

    def __len__(self) -> int:
        return len(self.item_id)

    def take(self, mask: np.ndarray) -> Dataset:
        idx = np.flatnonzero(mask)
        return Dataset(item_id=self.item_id[idx], y=self.y[idx], split=self.split[idx], group_key=self.group_key[idx],
                       stratum=self.stratum[idx], threat=self.threat[idx],
                       contexts=tuple(self.contexts[i] for i in idx), z=self.z[idx], detectors=self.detectors)

    def z_for(self, space: FeatureSpace) -> np.ndarray:
        return self.z[:, [self.detectors.index(d) for d in space.detectors]]

    def design(self, space: FeatureSpace) -> np.ndarray:
        return space.matrix(self.z_for(space), self.contexts)

    def platt_contexts(self) -> np.ndarray:
        return np.asarray([c.platt_context() for c in self.contexts], dtype=object)

    def min_sides(self) -> np.ndarray:
        return np.asarray([np.nan if c.min_side is None else float(c.min_side) for c in self.contexts], dtype=np.float64)

    def split_data(self, space: FeatureSpace) -> bs.SplitData:
        return bs.SplitData(a=self.design(space), y=self.y, groups=self.group_key, strata=self.stratum,
                            contexts=self.platt_contexts(), min_side=self.min_sides())


# --- fitting -----------------------------------------------------------------------------------------------

def _counts(y: np.ndarray) -> tuple[int, int]:
    return int((y == 0).sum()), int((y == 1).sum())


def _auc(s: np.ndarray, y: np.ndarray) -> float:
    ranks = rankdata(s)
    n0, n1 = _counts(y)
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n0 * n1))


def calibration_metrics(s: np.ndarray, y: np.ndarray, conformal: dict, contexts: Sequence[str]) -> dict:
    n0, n1 = _counts(y)
    clipped = np.clip(s, 1e-12, 1 - 1e-12)
    loss = -(y * np.log(clipped) + (1 - y) * np.log(1 - clipped))
    brier = (s - y) ** 2
    bands = [band(float(v), conformal, k)[0] for v, k in zip(s, contexts, strict=True)]
    return {
        "note": "in-sample: Platt parameters and conformal thresholds were fitted on these items",
        "n0": n0, "n1": n1, "auc": _auc(s, y),
        "logloss_balanced": float((loss[y == 0].mean() + loss[y == 1].mean()) / 2),
        "brier_balanced": float((brier[y == 0].mean() + brier[y == 1].mean()) / 2),
        "coverage_0.05": coverage(s, y, conformal, contexts, 0.05),
        "coverage_0.01": coverage(s, y, conformal, contexts, 0.01),
        "inconclusive_rate": float(np.mean([b == "uncertain" for b in bands])),
    }


def _calibrated(eta: np.ndarray, calibration: dict, contexts: np.ndarray) -> np.ndarray:
    ab = np.asarray([platt_params(calibration, k) for k in contexts], dtype=np.float64).reshape(len(eta), 2)
    return ab[:, 0] * eta + ab[:, 1]


def _fit_head(fit: Dataset, cal: Dataset, space: FeatureSpace, *, seed: int, grid: Sequence[float], policy: SizePolicy,
              fixed_missing: Sequence[str] = ()) -> tuple[dict, np.ndarray]:
    """Stacker, Platt and conformal thresholds for one label set (the union fuser or a sub-fuser)."""
    a_fit = fit.design(space)
    # D-50: one ω (stratum-balanced, resolution ratios on the stratum-weighted masses) for the stacker, Platt and the bootstrap
    multipliers = resolution_weights(fit.y, fit.min_sides(), strata=fit.stratum) if policy.resolution_balance else None
    weights = balancing_weights(fit.y, fit.stratum, multipliers=multipliers)
    l2, table = select_l2(a_fit, fit.y, weights, fit.group_key, space, grid=grid, seed=seed, fixed_missing=fixed_missing)
    result = fit_stacker(a_fit, fit.y, weights, space, l2, fixed_missing=fixed_missing)
    contexts = cal.platt_contexts()
    eta = cal.design(space) @ result.coef
    cal_weights = (balancing_weights(cal.y, cal.stratum, multipliers=resolution_weights(cal.y, cal.min_sides(), strata=cal.stratum))
                   if policy.resolution_balance else None)
    calibration = fit_calibration(eta, cal.y, contexts, weights=cal_weights, global_intercept=policy.global_intercept,
                                  centre=float(result.coef[0]))  # D-50: centred on the stacker bias (space.names[0])
    s = 1.0 / (1.0 + np.exp(-_calibrated(eta, calibration, contexts)))
    conformal = fit_conformal(s, cal.y, contexts)
    head = {
        "coef": {name: float(v) for name, v in zip(space.names, result.coef, strict=True)},
        "fit": {"solver": "scipy L-BFGS-B", "l2": l2, "l2_grid": sorted(float(v) for v in grid),
                "cv": f"StratifiedGroupKFold({CV_SPLITS}, shuffle=True, seed={seed})", "cv_table": table,
                "cv_logloss": next(r["cv_logloss"] for r in table if r["l2"] == l2), "seed": seed,
                "converged": result.converged, "n_iter": result.n_iter, "objective": result.objective,
                "weights": "class- and stratum-balanced, Σω = n" + (
                    "; reals × P(min-side bucket | fake) / P(bucket | real), clipped [0.1, 10] (D-45)"
                    if policy.resolution_balance else ""),
                "parameterisation": "w = t + Σ_g r_g, v_g = u_g − r_g; t, r, u ≥ 0"},
        "calibration": calibration,
        "conformal": conformal,
        "metrics": {"cal": calibration_metrics(s, cal.y, conformal, contexts)},
    }
    return head, result.coef


def _disagreement(space: FeatureSpace, coef: np.ndarray, calibration: dict, signal_calibration: dict,
                  z: Sequence[float | None], ctx: Context) -> float | None:
    a, _ = platt_params(calibration, ctx.platt_context())
    z_tilde, weights = [], []
    terms = ctx.terms()
    for j, (det, value, (plus, minus)) in enumerate(zip(space.detectors, z, effective_coefficients(space, coef, ctx), strict=True)):
        if value is None or not math.isfinite(value) or not space.gate_open(j, terms):
            z_tilde.append(None)
            weights.append(None)
            continue
        z_tilde.append(calibrated_logit(signal_calibration.get(det), float(value)))
        weights.append(a * (plus if value >= 0 else minus))
    return disagreement_index(z_tilde, weights)


def env_versions() -> dict:
    import scipy
    import sklearn

    return {"python": f"{sys.version_info.major}.{sys.version_info.minor}", "numpy": np.__version__,
            "scipy": scipy.__version__, "sklearn": sklearn.__version__}


@fp_guarded
def fit_fuser(ds: Dataset, *, modality: str, version: str, detector_meta: Mapping[str, Mapping], seed: int = 7,
              grid: Sequence[float] = L2_GRID, n_boot: int = DEFAULT_BOOTSTRAP, bench: Mapping | None = None,
              command: str | None = None, git_commit: str | None = None, reviewed_by: str | None = LOOP_REVIEW_FLAG,
              created_at: str | None = None, gates: Mapping[str, Sequence[str]] | None = None,
              size_policy: SizePolicy | None = None, drop_context_terms: Sequence[str] = ()) -> tuple[dict, np.ndarray]:
    """`size_policy` defaults by modality (`policy.default_policy`: D-45 for images). A detector whose meta says
    `missing: size` has its missing weight fixed at 0 under a policy with `neutral_size_missing`. `drop_context_terms` are
    left out of the fit (`removed_context_terms` reason `requested`): the removal refit of gate 2c (D-52)."""
    if not _SEMVER.fullmatch(version):
        raise ValueError(f"version {version!r} is not semver")
    unknown_terms = sorted(set(drop_context_terms) - set(CONTEXT_TERMS))
    if unknown_terms:
        raise ValueError(f"drop_context_terms names unknown context terms {unknown_terms}")
    unknown = [d for d in (gates or {}) if d not in ds.detectors]
    if unknown:
        raise ValueError(f"gates name detectors without scores: {unknown}")
    fit, cal = ds.take(ds.split == "fit"), ds.take(ds.split == "cal")
    for name, part in (("fit", fit), ("cal", cal)):
        if 0 in _counts(part.y):
            raise ValueError(f"split {name!r} needs items of both classes (real {_counts(part.y)[0]}, "
                             f"fake {_counts(part.y)[1]})")
    active = [d for j, d in enumerate(ds.detectors) if np.isfinite(fit.z[:, j]).any()]
    skipped = [{"id": d, "reason": "no usable scores on fit items"} for d in ds.detectors if d not in active]
    if not active:
        raise ValueError("no detector has usable scores on fit items")
    policy = size_policy or default_policy(modality)
    if modality == "image":  # D-50: metadata terms are never gate literals of an image fit
        metadata_gated = sorted(d for d, literals in (gates or {}).items() if any(lit.lstrip("!") in METADATA_TERMS for lit in literals))
        if metadata_gated:
            raise ValueError(f"an image fit refuses metadata gate literals {list(METADATA_TERMS)} on {metadata_gated}")
    if modality == "image" and not policy.additive_size_face_terms:  # D-50: face-driven missing needs a has_face gate
        ungated = sorted(d for d in active if (detector_meta.get(d) or {}).get("missing") == "face" and "has_face" not in (gates or {}).get(d, []))
        if ungated:
            raise ValueError(f"detectors not applicable only for want of a face need a has_face gate in an image fit: {ungated}")
    present = [t for t in CONTEXT_TERMS if any(c.terms()[t] for c in fit.contexts)]
    removed: dict[str, str] = {}  # D-50: each dropped term with its reason
    for t in present:
        if not policy.additive_size_face_terms and t in SIZE_FACE_TERMS:
            removed[t] = "size_face"
        elif modality == "image" and t in METADATA_TERMS:
            removed[t] = "metadata"
        elif t in drop_context_terms:
            removed[t] = "requested"
        elif all(c.terms()[t] for c in fit.contexts):
            removed[t] = "constant_on_fit"
    terms = [t for t in present if t not in removed]
    fixed_missing = tuple(d for d in active if policy.neutral_size_missing and (detector_meta.get(d) or {}).get("missing") == "size")
    space = FeatureSpace(tuple(active), tuple(terms), tuple(g for g in GROUPS if g in present),
                         gates=tuple((d, tuple(gates[d])) for d in active if d in (gates or {})))

    union, coef = _fit_head(fit, cal, space, seed=seed, grid=grid, policy=policy, fixed_missing=fixed_missing)
    z_cal = cal.z_for(space)
    signal_calibration = fit_signal_calibration(z_cal, cal.y, space.detectors)
    d_values = [_disagreement(space, coef, union["calibration"], signal_calibration, z_cal[i].tolist(), ctx)
                for i, ctx in enumerate(cal.contexts)]
    boot = bs.refit_matrix(fit.split_data(space), cal.split_data(space), space, union["fit"]["l2"], n_boot=n_boot,
                           seed=seed, policy=policy, fixed_missing=fixed_missing)

    sub_fusers: dict[str, dict] = {}
    for name, threats in SUB_FUSERS.items():
        subset = ds.take((ds.y == 0) | np.isin(ds.threat, threats))
        s_fit, s_cal = subset.take(subset.split == "fit"), subset.take(subset.split == "cal")
        if min(*_counts(s_fit.y), *_counts(s_cal.y)) < MIN_SUB_FUSER_CLASS_N:
            sub_fusers[name] = {"threats": list(threats),
                                "skipped": f"fewer than {MIN_SUB_FUSER_CLASS_N} items of a class in fit or cal"}
            continue
        try:
            head, _ = _fit_head(s_fit, s_cal, space, seed=seed, grid=grid, policy=policy, fixed_missing=fixed_missing)
        except ValueError as exc:
            sub_fusers[name] = {"threats": list(threats), "skipped": str(exc)}
            continue
        sub_fusers[name] = {"threats": list(threats), **head}

    doc = {
        "schema": SCHEMA,
        "modality": modality,
        "version": version,
        "created_at": created_at or datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_commit": git_commit,
        "bench": {**dict(bench or {}), "fit_n": len(fit), "cal_n": len(cal)},
        "detectors": [{**dict(detector_meta[d]), "id": d} for d in space.detectors],
        "skipped_detectors": skipped,
        "context_terms": list(space.context_terms),
        "groups": list(space.groups),
        "gates": {det: list(literals) for det, literals in space.gates},
        "size_policy": {**policy.as_json(), "fixed_missing": list(fixed_missing),
                        "removed_context_terms": dict(removed)},
        "features": list(space.names[1:]),
        "coef": union["coef"],
        "fit": union["fit"],
        "calibration": union["calibration"],
        "conformal": union["conformal"],
        "signal_calibration": signal_calibration,
        "bootstrap": {"B": n_boot, "grouped_by": "group_key", "seed": seed, "columns": bs.bootstrap_columns(space),
                      "file": None, "sha256": None},
        "disagreement": {"d95": d95(d_values), "n_defined": sum(v is not None for v in d_values),
                         "weights": "|effective weight| = a_k × context-effective coefficient"},
        "sub_fusers": sub_fusers,
        "metrics": {"cal": union["metrics"]["cal"], "test": {}, "lgo": {}},
        "command": command,
        "env": env_versions(),
        "reviewed_by": reviewed_by,
    }
    return doc, boot


# --- serialisation -----------------------------------------------------------------------------------------

def _plain(value):
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_plain(v) for v in value]
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def artefact_bytes(doc: Mapping) -> bytes:
    return (json.dumps(_plain(doc), sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()


def _atomic_write(path: Path, data: bytes) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def write_artefact(out_dir: Path, doc: Mapping, boot: np.ndarray, *, force: bool = False) -> tuple[Path, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"fuser-{doc['modality']}-{doc['version']}"
    json_path, boot_path = out_dir / f"{stem}.json", out_dir / f"{stem}.boot.npy"
    if not force and (json_path.exists() or boot_path.exists()):
        raise FileExistsError(f"{json_path.name} already exists in {out_dir}; pass --force to replace it")
    buffer = io.BytesIO()
    np.save(buffer, np.ascontiguousarray(boot, dtype=np.float32), allow_pickle=False)
    data = buffer.getvalue()
    doc = {**doc, "bootstrap": {**doc["bootstrap"], "file": boot_path.name, "sha256": hashlib.sha256(data).hexdigest()}}
    _atomic_write(boot_path, data)
    _atomic_write(json_path, artefact_bytes(doc))
    return json_path, boot_path


# --- runtime -----------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class FusionOutput:
    eta: float  # Stage 2 stacker logit
    logit: float  # logit s = a_k·η + b_k (+ logit π)
    probability: float  # σ(logit), unclamped
    probability_display: float  # clamped to [0.02, 0.98]
    context: str
    band: str
    sets: dict[str, list[str]]
    decomposition: Decomposition
    ci90: tuple[float, float] | None
    disagreement: float | None


class Fuser:
    def __init__(self, doc: Mapping, boot: np.ndarray | None = None):
        if doc.get("schema") != SCHEMA:
            raise ValueError(f"not a {SCHEMA} artefact")
        self.doc = doc
        self.detectors = tuple(d["id"] for d in doc["detectors"])
        self.transforms = {d["id"]: d.get("transform", "logit_clip8") for d in doc["detectors"]}
        self.space = FeatureSpace(self.detectors, tuple(doc["context_terms"]), tuple(doc["groups"]),
                                  gates=tuple((d, tuple(v)) for d, v in sorted((doc.get("gates") or {}).items())))
        if list(self.space.names[1:]) != list(doc["features"]):
            raise ValueError("artefact feature list does not match its detector and context layout")
        missing = [n for n in self.space.names if n not in doc["coef"]]
        if missing:
            raise ValueError(f"artefact is missing coefficients {missing[:3]}")
        self.coef = np.asarray([float(doc["coef"][n]) for n in self.space.names])
        if boot is not None and boot.shape != (doc["bootstrap"]["B"], len(bs.bootstrap_columns(self.space))):
            raise ValueError(f"bootstrap matrix shape {boot.shape} does not match the artefact")
        self.boot = boot

    @classmethod
    def load(cls, path: Path) -> Fuser:
        path = Path(path)
        return cls.from_bytes(path.read_bytes(), path.parent)

    @classmethod
    def from_bytes(cls, data: bytes, directory: Path) -> Fuser:
        """The artefact JSON as read (so the caller can hash exactly these bytes), with its bootstrap matrix beside it."""
        doc = json.loads(data.decode("utf-8"))
        info = doc.get("bootstrap") or {}
        boot = None
        if info.get("file"):
            matrix = (Path(directory) / Path(info["file"]).name).read_bytes()
            if hashlib.sha256(matrix).hexdigest() != info.get("sha256"):
                raise ValueError("bootstrap matrix sha256 does not match the artefact")
            boot = np.load(io.BytesIO(matrix), allow_pickle=False)
        return cls(doc, boot)

    def sub_fusers(self) -> tuple[str, ...]:
        """Names of the fitted sub-fusers (skipped ones carry no coefficients)."""
        return tuple(sorted(name for name, head in (self.doc.get("sub_fusers") or {}).items() if "coef" in head))

    @fp_guarded
    def apply_sub(self, name: str, scores: Mapping[str, tuple[float | None, float | None] | None], ctx: Context, *,
                  prior_ai: float | None = None, inactive: Collection[str] = ()) -> FusionOutput:
        """A sub-fuser on one input: s, band and exact contributions (no bootstrap interval, no disagreement)."""
        head = (self.doc.get("sub_fusers") or {}).get(name)
        if not head or "coef" not in head:
            raise KeyError(f"sub-fuser {name!r} is not fitted")
        missing = [n for n in self.space.names if n not in head["coef"]]
        if missing:
            raise ValueError(f"sub-fuser {name!r} is missing coefficients {missing[:3]}")
        coef = np.asarray([float(head["coef"][n]) for n in self.space.names])
        z = self.z_vector(scores)
        context = ctx.platt_context()
        a, b = platt_params(head["calibration"], context)
        shift = prior_logit(prior_ai)
        eta = float(np.dot(self.space.row(z, ctx, inactive), coef))
        if not math.isfinite(eta):
            raise FloatingPointError("non-finite sub-fuser logit")
        logit = a * eta + b + shift
        probability = 1.0 / (1.0 + math.exp(-logit))
        band_key, sets = band(1.0 / (1.0 + math.exp(-(a * eta + b))), head["conformal"], context)
        return FusionOutput(eta=eta, logit=logit, probability=probability, probability_display=clamp_probability(probability),
                            context=context, band=band_key, sets=sets,
                            decomposition=decompose(self.space, coef, z, ctx, a, b, prior_logit=shift, inactive=inactive), ci90=None,
                            disagreement=None)

    @fp_guarded
    def neutral_view(self, scores: Mapping[str, tuple[float | None, float | None] | None], ctx: Context, *,
                     prior_ai: float | None = None, inactive: Collection[str] = (), sub: str | None = None) -> NeutralView:
        """The contributions of `apply` for the same input, measured from each scoring check's neutral point (D-63; N-17):
        the start is the estimate with every scoring check at the score where its stand-alone likelihood ratio is 1.
        `sub`: the same view of `apply_sub(sub, …)`; a check's neutral point is the artefact's, whichever head reads it."""
        coef, calibration = self.coef, self.doc["calibration"]
        if sub is not None:
            head = (self.doc.get("sub_fusers") or {}).get(sub)
            if not head or "coef" not in head:
                raise KeyError(f"sub-fuser {sub!r} is not fitted")
            coef, calibration = np.asarray([float(head["coef"][n]) for n in self.space.names]), head["calibration"]
        a, b = platt_params(calibration, ctx.platt_context())
        return neutral_view(self.space, coef, self.z_vector(scores), ctx, a, b, self.doc.get("signal_calibration") or {},
                            prior_logit=prior_logit(prior_ai), inactive=inactive)

    @fp_guarded
    def probabilities(self, ds: Dataset) -> np.ndarray:
        """s for every item of a Dataset (Stages 2–3, no prior shift): the vectorised `apply(...).probability`."""
        eta = ds.design(self.space) @ self.coef
        logit = _calibrated(eta, self.doc["calibration"], ds.platt_contexts())
        if not np.isfinite(logit).all():
            raise FloatingPointError("non-finite fused logit")
        return 1.0 / (1.0 + np.exp(-np.clip(logit, -700.0, 700.0)))

    def z_vector(self, scores: Mapping[str, tuple[float | None, float | None] | None]) -> list[float | None]:
        out = []
        for det in self.detectors:
            score = scores.get(det)
            out.append(None if score is None else transform_score(score[0], score[1], self.transforms[det]))
        return out

    @fp_guarded
    def apply(self, scores: Mapping[str, tuple[float | None, float | None] | None], ctx: Context, *,
              prior_ai: float | None = None, inactive: Collection[str] = ()) -> FusionOutput:
        """Stages 1–4 on one input. `inactive`: artefact detectors that did not run (D-46), which add nothing."""
        z = self.z_vector(scores)
        x = self.space.row(z, ctx, inactive)
        context = ctx.platt_context()
        a, b = platt_params(self.doc["calibration"], context)
        shift = prior_logit(prior_ai)
        eta = float(np.dot(x, self.coef))
        if not math.isfinite(eta):
            raise FloatingPointError("non-finite stacker logit")
        logit = a * eta + b + shift
        probability = 1.0 / (1.0 + math.exp(-logit))
        band_key, sets = band(1.0 / (1.0 + math.exp(-(a * eta + b))), self.doc["conformal"], context)
        return FusionOutput(
            eta=eta, logit=logit, probability=probability, probability_display=clamp_probability(probability),
            context=context, band=band_key, sets=sets,
            decomposition=decompose(self.space, self.coef, z, ctx, a, b, prior_logit=shift, inactive=inactive),
            ci90=None if self.boot is None else bs.interval(self.boot, self.space, x, context, prior_logit=shift),
            disagreement=_disagreement(self.space, self.coef, self.doc["calibration"],
                                       self.doc.get("signal_calibration") or {}, z, ctx),
        )
