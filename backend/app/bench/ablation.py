"""Detector ablation for admission (ML §3.2 A3a, A3b, A4; BUILD_SPEC I-6 `calibrate --ablate <id>`).

- **Systems.** The fuser over the baseline detectors, and the fuser over the baseline plus the candidate, fitted on the
  same items with the `calibrate` method (`fit_fuser`, Stages 2–3; bootstrap B = 1 because the matrix is not used here).
  Context gates (`calibrate --gates`, H-3) apply to every fit in both systems, for the detectors each system holds.
- **Items.** `fit` ∪ `cal` only: tuning decisions may use only fit / cal and the LGO folds (lockbox, ML §2.6). Items from
  the candidate card's `contamination_sets` are dropped from fitting and evaluation for both systems (ML P2), using
  explicit collection aliases. Declared exclusions require known source_dataset on every eligible row. Matching policy
  and dropped original source names/counts are recorded; split-qualified and unknown names keep exact semantics.
- **LGO** (A3a; A3b's "LGO macro does not fall"). For each F, both systems are refitted on fit ∪ cal minus F (the items'
  own fit / cal labels keep the 35:20 ratio); τ_0.05 and q come from each refit's own `cal` reals, as in `vbench-report`;
  TPR is the mean flag weight of F's fit / cal fakes. Macro = mean over F1–F7, null while a fold has no fakes.
- **Cross-fit** (A3b, A4). `StratifiedGroupKFold(5, shuffle, seed)` over fit ∪ cal groups: both systems are fitted on
  four folds and score the fifth, giving out-of-fold s for every item. τ_0.05 per system from the out-of-fold reals with
  equal stratum weights (`metrics.operational_threshold`) on the weighting strata `report.thresholds` would use on the
  `cal` split (D-51): `metrics.merge_map` of the `cal` reals' strata relabels every out-of-fold real (a stratum without
  `cal` reals counts as below 19; R6 → R3 on vb-1.1). Recall per family and per threat, FPR per original real stratum.
- **Operating point** (D-53, this harness only). An item above τ is flagged with weight 1, an item scoring exactly τ with
  weight q, chosen so that the weighted flagged share of the threshold reals equals the rank rule's share
  (`metrics.operating_point`, `metrics.flag_weights`); the same q applies to fakes at τ, in the LGO folds and the
  cross-fit, for both systems. Each system's served-rule (s ≥ τ) weighted FPR is recorded beside it
  (`cross_fit.operating_point`); `vbench-report`, the runtime and every G goal keep s ≥ τ.
- **Intervals.** Paired grouped bootstrap (B = 2,000 by default, ML §5.3): the same group resamples for both systems,
  τ and q fixed; percentile interval of the difference (with − without) of the mean flag weights.
- **Criteria.** A3a: Δ macro ≥ 2.0 pts and CI lower bound > 0. A3b: some family or threat has Δ recall ≥ 10 pts with CI
  lower bound > 0, and Δ macro ≥ 0. A4 (D-44 amended by D-53): pooled reals of fit ∪ cal block only with a paired lower
  bound of ΔFPR above +1.0 pt (established pooled harm; the interval is reported). A real stratum blocks only when harm is established: ≥ 150 fit ∪ cal reals **and**
  a paired **lower** bound above +2.0 pts. Every other stratum is reported with its CI as "not established". A stratum
  with ≥ 150 reals whose point ΔFPR exceeds +2.0 pts without established harm asks for its runtime context interaction
  term (`STRATUM_TERMS`); a smaller stratum is "not established; no term" (D-51). A criterion without the data to decide
  it is null (not established). A1, A2, A5 and A6 come from the card, the parity test, `admission_evidence.py` and the
  latency records, not from here. The `screenshot` stratum needs transform rows and is not evaluated.
- **Term support** (D-51; `term_support`, written as `method.term_support`). Per context term: fit and cal counts by
  class on each side of the term, and whether the fuser can represent it (`features.GROUPS`). A record keeps a term only
  when it is a group member carried by ≥ 20 fit and ≥ 20 cal items of each class; otherwise the term is "not learnable",
  the detector runs ungated and the G3 test FPR decides. An ablation without term support leaves the term not established.
- **Decision** (`admission_decision`, D-44 amended by D-51). Rejected when A1, A2, A4 or A6 fails or A3 (A3a or A3b)
  fails; not established while one of them is null; otherwise admitted, and admitted **with context interaction terms**
  when A5 fails (terms of the failing transforms, `A5_TRANSFORM_TERMS`; A5 never rejects on its own), when a stratum asks
  for its term, or when H-3 context-gated the detector, each term subject to its support. A camera-original stand-alone
  FPR above 10 % on R1 ∪ R2 (n ≥ 300) needs an H-3 decision: retire, or context-gate with named features.
- **Gated retry** (D-51). A rejected candidate with at least one blocking stratum, A1, A2 and A6 not failing, and every
  blocking stratum's term carried by ≥ 20 fit and ≥ 20 cal items of each class on both sides, is ablated again gated off
  where the term holds (`<id>=!<term>`). Until that ablation exists the decision is not established. With it: admitted
  context-gated (record `gates`) when the gated ablation has no blocking stratum and A1, A2, A3, A4 and A6 pass;
  otherwise rejected, citing both ablations.
- **Runtime terms** (D-44, D-50, D-51): only terms the runtime `Context` computes at inference and the fuser can learn.
  Strata: R3 (social-platform bytes) → `jpeg_q<70`; R5 (faces) → `has_face`; `screenshot` → `screenshot_like`; R1, R2
  and R6 (camera and smartphone originals, Commons hard negatives) and R4 (web / stock) have no runtime term, so their
  harm is judged by the fused G3 per-stratum FPR on test (D-02). A5 transforms: JPEG and social-platform recompression →
  `jpeg_q<70` (the condition stage computes no platform dimensions); screenshots → `screenshot_like`.
"""
from __future__ import annotations

import statistics
from collections.abc import Mapping, Sequence

import numpy as np

from app.bench import metrics as mt
from app.bench.contamination import contamination_filter
from app.engine.fusion.artefact import Dataset, Fuser, fit_fuser
from app.engine.fusion.features import CONTEXT_TERMS, GROUPS
from app.engine.fusion.policy import SizePolicy, default_policy

LGO_FOLDS = tuple(f"F{i}" for i in range(1, 8))
ALPHA = 0.05
CV_SPLITS = 5
A3A_MIN_GAIN, A3B_MIN_GAIN = 0.02, 0.10
A4_POOLED_MIN_LOWER = 0.01  # D-53: pooled reals block only with a paired lower bound of ΔFPR above +1.0 pt
A4_STRATUM_MIN_N, A4_STRATUM_MIN_LOWER = 150, 0.02  # D-44: a stratum blocks only with ≥ 150 reals and lower bound > 2 pts
A4_INTERACTION_MIN_POINT = 0.02  # D-44: point ΔFPR above 2 pts without established harm → context interaction term
CAMERA_FPR_MAX, CAMERA_MIN_N = 0.10, 300  # D-41: stand-alone FPR at τ_0.05 on R1 ∪ R2 reals
CAMERA_STRATA = ("R1", "R2")
STRATUM_TERMS = {"R1": None, "R2": None, "R3": "jpeg_q<70", "R4": None, "R5": "has_face", "R6": None,
                 "screenshot": "screenshot_like"}  # D-44, D-51: runtime context terms per real stratum
TERM_MIN_N = 20  # D-51: items of each class carrying a term on fit and on cal
GATE_WORDS = {"has_face": ("a face is detected", "no face is detected"),
              "jpeg_q<70": ("the JPEG quality is below 70", "the image is not a JPEG below quality 70"),
              "screenshot_like": ("the image looks like a screenshot", "the image does not look like a screenshot")}
# card text for a gate literal (term holds, term absent): reports lint snake_case terms (D-51)
A5_TRANSFORM_TERMS = {"jpeg75": "jpeg_q<70", "jpeg50": "jpeg_q<70", "social_ig": "jpeg_q<70", "social_x": "jpeg_q<70",
                      "screenshot": "screenshot_like", "screenshot_jpeg": "screenshot_like"}  # D-44 A5 terms
ABLATION_VERSION = "0.0.0-ablation"
SYSTEMS = ("without", "with")


class AblationError(ValueError):
    """The ablation cannot run on these inputs."""


def drop_detector(ds: Dataset, detector: str) -> Dataset:
    keep = [j for j, d in enumerate(ds.detectors) if d != detector]
    return Dataset(item_id=ds.item_id, y=ds.y, split=ds.split, group_key=ds.group_key, stratum=ds.stratum, threat=ds.threat,
                   contexts=ds.contexts, z=ds.z[:, keep], detectors=tuple(ds.detectors[j] for j in keep))


def _fit(train: Dataset, meta: Mapping[str, Mapping], *, modality: str, seed: int, where: str,
         gates: Mapping[str, Sequence[str]] | None = None, size_policy: SizePolicy | None = None) -> Fuser:
    try:
        doc, _ = fit_fuser(train, modality=modality, version=ABLATION_VERSION, detector_meta=meta, seed=seed, n_boot=1,
                           reviewed_by=None, created_at="1970-01-01T00:00:00Z",
                           gates={d: v for d, v in (gates or {}).items() if d in train.detectors}, size_policy=size_policy)
    except (ValueError, FloatingPointError) as exc:
        raise AblationError(f"{where}: {exc}") from exc
    return Fuser(doc)


def select_detectors(ds: Dataset, detectors: Sequence[str]) -> Dataset:
    """The dataset restricted to `detectors` (columns in the dataset's order)."""
    keep = [j for j, d in enumerate(ds.detectors) if d in set(detectors)]
    return Dataset(item_id=ds.item_id, y=ds.y, split=ds.split, group_key=ds.group_key, stratum=ds.stratum, threat=ds.threat,
                   contexts=ds.contexts, z=ds.z[:, keep], detectors=tuple(ds.detectors[j] for j in keep))


Systems = Mapping[str, tuple[Sequence[str], Mapping[str, Sequence[str]]]]  # {"without" | "with": (detectors, gates)}


def _leave_one_out(ds: Dataset, candidate: str, gates: Mapping[str, Sequence[str]] | None) -> Systems:
    return {"without": ([d for d in ds.detectors if d != candidate], gates or {}), "with": (list(ds.detectors), gates or {})}


def _systems(train: Dataset, systems: Systems) -> dict[str, tuple[Dataset, Mapping[str, Sequence[str]]]]:
    return {name: (select_detectors(train, systems[name][0]), systems[name][1]) for name in SYSTEMS}


def _point(s: np.ndarray, strata: np.ndarray) -> mt.OperatingPoint:
    return mt.operating_point(s, ALPHA, [str(k) for k in strata])


def _ci(ci: dict) -> dict:
    return {"lo": ci["lo"], "hi": ci["hi"], "B": ci["B"], "valid": ci["valid"]}


def _mean_hits(hits: np.ndarray) -> float | None:
    return float(hits.mean()) if len(hits) else None


def lgo(ds: Dataset, lgo_fold: np.ndarray, candidate: str | None, meta: Mapping[str, Mapping], *, modality: str, seed: int,
        n_boot: int, required: Sequence[str] = LGO_FOLDS, gates: Mapping[str, Sequence[str]] | None = None,
        size_policy: SizePolicy | None = None, systems: Systems | None = None, label: str | None = None) -> dict:
    """LGO refits of both systems: leave `candidate` out (with `gates` on both), or the named `systems` (`compare`)."""
    systems = systems if systems is not None else _leave_one_out(ds, candidate, gates)
    label = label or candidate
    present = sorted({str(f) for f, y in zip(lgo_fold, ds.y, strict=True) if y == 1 and f})
    folds: dict[str, dict] = {}
    missing: list[str] = []
    fold_index = np.full(len(ds), -1)
    hits = {name: np.zeros(len(ds)) for name in SYSTEMS}  # flag weights (D-53)
    order = sorted(set(present) | set(required))
    for k, fold in enumerate(order):
        in_fold = lgo_fold == fold
        fakes = in_fold & (ds.y == 1)
        if not fakes.any():
            missing.append(f"{fold}: no fit / cal fakes")
            continue
        train = ds.take(~in_fold)
        cal_reals = train.take((train.split == "cal") & (train.y == 0))
        entry: dict = {"n_fake": int(fakes.sum()), "n_train": len(train), "tau": {}, "q": {}, "tpr": {}}
        for name, (system, system_gates) in _systems(train, systems).items():
            fuser = _fit(system, meta, modality=modality, seed=seed, where=f"LGO refit without {fold} ({name} {label})",
                         gates=system_gates, size_policy=size_policy)
            point = _point(fuser.probabilities(cal_reals), cal_reals.stratum)
            flags = mt.flag_weights(fuser.probabilities(ds.take(fakes)), point)
            hits[name][fakes] = flags
            entry["tau"][name], entry["q"][name], entry["tpr"][name] = point.tau, point.q, _mean_hits(flags)
            if name == "with" and candidate is not None:
                entry["candidate_active"] = candidate in fuser.detectors
        entry["delta"] = entry["tpr"]["with"] - entry["tpr"]["without"]
        fold_index[fakes] = k
        folds[fold] = entry
    complete = all(f in folds for f in required)
    used = [f for f in order if f in folds]  # criteria need `complete`; the macro over present folds is informative
    macro = {name: statistics.fmean(folds[f]["tpr"][name] for f in used) if used else None for name in SYSTEMS}
    out = {"required": list(required), "missing": missing, "complete": complete, "folds": folds, "macro_over": used,
           "macro": {**macro, "delta": None if macro["with"] is None else macro["with"] - macro["without"], "ci95": None}}
    if used and n_boot > 0:
        idx = np.flatnonzero(fold_index >= 0)
        codes = np.unique(fold_index[idx], return_inverse=True)[1]
        boot = mt.GroupedBootstrap(ds.group_key[idx], n_boot=n_boot, seed=seed)

        def stat(name: str):
            h = hits[name][idx].astype(np.float64)

            def macro_of(ix: np.ndarray) -> float | None:
                counts = np.bincount(codes[ix], minlength=len(used))
                if (counts == 0).any():
                    return None
                return float((np.bincount(codes[ix], weights=h[ix], minlength=len(used)) / counts).mean())
            return macro_of

        out["macro"]["ci95"] = _ci(boot.paired(stat("with"), stat("without")))
    return out


def cross_fit(ds: Dataset, candidate: str | None, meta: Mapping[str, Mapping], *, modality: str, seed: int,
              n_splits: int = CV_SPLITS, gates: Mapping[str, Sequence[str]] | None = None,
              size_policy: SizePolicy | None = None, systems: Systems | None = None, label: str | None = None) -> dict[str, np.ndarray]:
    from sklearn.model_selection import StratifiedGroupKFold

    systems = systems if systems is not None else _leave_one_out(ds, candidate, gates)
    label = label or candidate
    for y, name in ((0, "real"), (1, "fake")):
        groups = len(set(ds.group_key[ds.y == y]))
        if groups < n_splits:
            raise AblationError(f"cross-fit needs ≥ {n_splits} groups of {name} items (found {groups})")
    oof = {name: np.full(len(ds), np.nan) for name in SYSTEMS}
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for k, (_, held_idx) in enumerate(splitter.split(np.zeros(len(ds)), ds.y, ds.group_key)):
        held = np.zeros(len(ds), dtype=bool)
        held[held_idx] = True
        for name, (system, system_gates) in _systems(ds.take(~held), systems).items():
            fuser = _fit(system, meta, modality=modality, seed=seed, where=f"cross-fit fold {k} ({name} {label})", gates=system_gates,
                         size_policy=size_policy)
            oof[name][held] = fuser.probabilities(ds.take(held))
    return oof


def _paired_rate(mask: np.ndarray, flags: Mapping[str, np.ndarray], groups: np.ndarray, *, n_boot: int, seed: int) -> dict:
    idx = np.flatnonzero(mask)
    rates = {name: _mean_hits(flags[name][idx]) for name in SYSTEMS}
    entry = {"n": int(len(idx)), **rates, "delta": rates["with"] - rates["without"], "ci95": None}
    if n_boot > 0:
        boot = mt.GroupedBootstrap(groups[idx], n_boot=n_boot, seed=seed)
        f_with, f_without = flags["with"][idx], flags["without"][idx]
        entry["ci95"] = _ci(boot.paired(lambda ix: float(f_with[ix].mean()), lambda ix: float(f_without[ix].mean())))
    return entry


def slices(ds: Dataset, oof: Mapping[str, np.ndarray], *, n_boot: int, seed: int) -> dict:
    reals, fakes = ds.y == 0, ds.y == 1
    mapping = mt.merge_map(ds.stratum[reals & (ds.split == "cal")], ds.stratum[reals])  # D-51: the served weighting strata
    threshold_strata = np.asarray([mapping[str(k)] for k in ds.stratum[reals]], dtype=object)
    points = {name: _point(oof[name][reals], threshold_strata) for name in SYSTEMS}
    flags = {name: mt.flag_weights(oof[name], points[name]) for name in SYSTEMS}  # D-53: weighted ties at τ
    strata = {str(k): _paired_rate(reals & (ds.stratum == k), flags, ds.group_key, n_boot=n_boot, seed=seed)
              for k in sorted(set(ds.stratum[reals]))}
    families = {str(k): _paired_rate(fakes & (ds.stratum == k), flags, ds.group_key, n_boot=n_boot, seed=seed)
                for k in sorted(set(ds.stratum[fakes]))}
    threats = {str(k): _paired_rate(fakes & (ds.threat == k), flags, ds.group_key, n_boot=n_boot, seed=seed)
               for k in sorted(set(ds.threat[fakes]))}
    pooled = _paired_rate(reals, flags, ds.group_key, n_boot=n_boot, seed=seed)
    return {"tau": {name: p.tau for name, p in points.items()},
            "operating_point": {name: p.as_json() for name, p in points.items()},
            "threshold_strata": {k: v for k, v in sorted(mapping.items()) if k != v},
            "auc": {name: mt.auc(oof[name], ds.y) for name in SYSTEMS},
            "tpr": {name: _mean_hits(flags[name][fakes]) for name in SYSTEMS},
            "fpr_pooled": pooled, "fpr_per_stratum": strata, "recall_per_family": families, "recall_per_threat": threats}


def _lo(entry: Mapping) -> float | None:
    return (entry.get("ci95") or {}).get("lo")


def _hi(entry: Mapping) -> float | None:
    return (entry.get("ci95") or {}).get("hi")


def criteria(lgo_result: Mapping, cross: Mapping) -> dict:
    macro = lgo_result["macro"]
    delta, lo = macro["delta"], _lo(macro)
    a3a = None if not lgo_result["complete"] or delta is None or lo is None else bool(delta >= A3A_MIN_GAIN and lo > 0)
    gains = [f"{kind} {name}" for kind, table in (("family", cross["recall_per_family"]), ("threat", cross["recall_per_threat"]))
             for name, e in table.items() if e["delta"] is not None and e["delta"] >= A3B_MIN_GAIN and (_lo(e) or 0) > 0]
    a3b = None if not lgo_result["complete"] or delta is None else bool(gains and delta >= 0)
    strata, pooled = cross["fpr_per_stratum"], cross.get("fpr_pooled") or {}
    large = {k: e for k, e in strata.items() if e["n"] >= A4_STRATUM_MIN_N}
    blocking = [k for k, e in large.items() if _lo(e) is not None and _lo(e) > A4_STRATUM_MIN_LOWER]
    undecided = [k for k, e in large.items() if _lo(e) is None]
    not_established = [k for k in strata if k not in blocking and k not in undecided]
    above = [k for k in not_established if strata[k].get("delta") is not None and strata[k]["delta"] > A4_INTERACTION_MIN_POINT]
    interaction = [k for k in above if k in large]  # D-51: only strata with ≥ 150 reals ask for a term
    small_above = [k for k in above if k not in large]
    pooled_lo, pooled_hi = _lo(pooled), _hi(pooled)
    pooled_harm = None if pooled_lo is None else bool(pooled_lo > A4_POOLED_MIN_LOWER)
    a4 = None if pooled_harm is None or undecided else bool(not pooled_harm and not blocking)
    a3 = True if (a3a or a3b) else (None if a3a is None or a3b is None else False)
    return {
        "A3a": {"pass": a3a, "lgo_macro_delta": delta, "ci95_lo": lo,
                "rule": "Δ LGO macro TPR@5%FPR ≥ 2.0 pts and paired-bootstrap 95 % CI lower bound > 0"},
        "A3b": {"pass": a3b, "qualifying": gains, "lgo_macro_delta": delta,
                "rule": "Δ recall ≥ 10 pts with CI lower bound > 0 on ≥ 1 family or threat, and Δ LGO macro ≥ 0"},
        "A3": {"pass": a3, "rule": "A3a or A3b"},
        "A4": {"pass": a4, "pooled_delta": pooled.get("delta"), "pooled_ci95_lo": pooled_lo, "pooled_ci95_hi": pooled_hi,
               "pooled_harm_established": pooled_harm, "blocking_strata": blocking,
               "not_established_strata": not_established, "small_strata": [k for k in strata if k not in large],
               "undecided_strata": undecided, "interaction_strata": interaction, "small_point_strata": small_above,
               "interaction_terms": {k: STRATUM_TERMS.get(k) for k in interaction},
               "blocking_detail": {k: {"n": strata[k]["n"], "delta": strata[k]["delta"], "lo": _lo(strata[k]), "hi": _hi(strata[k])}
                                   for k in blocking},
               "rule": "pooled reals of fit ∪ cal: the paired ΔFPR at τ_0.05 and its interval are reported; pooled reals block "
                       "with a paired lower bound above +1.0 pt (D-53); a real stratum blocks only with ≥ 150 reals and a paired "
                       "lower bound above +2.0 pts (established harm); other strata are not established, and a point ΔFPR above "
                       "+2.0 pts asks for the stratum's runtime context term only with ≥ 150 reals (D-44, D-51, D-53)"},
        "not_decided_here": ["A1 (card + parity test)", "A2 (vbench-report)", "A5 (vbench-report)", "A6 (latency)",
                             "A4 screenshot stratum (transform rows)"],
    }


def term_support(ds: Dataset) -> dict:
    """D-51: per context term, fit and cal counts by class on each side of the term, and `features.GROUPS` membership."""
    on = {term: np.asarray([c.terms()[term] == 1.0 for c in ds.contexts], dtype=bool) for term in CONTEXT_TERMS}
    out = {}
    for term, mask in on.items():
        entry: dict = {"group": term in GROUPS}
        for split in ("fit", "cal"):
            entry[split] = {name: {"on": int((mask & (ds.split == split) & (ds.y == y)).sum()),
                                   "off": int((~mask & (ds.split == split) & (ds.y == y)).sum())}
                            for y, name in ((0, "real"), (1, "fake"))}
        out[term] = entry
    return out


def term_learnable(support: Mapping | None, term: str, *, both_sides: bool = False) -> tuple[bool | None, str]:
    """(verdict, counts text): a `features.GROUPS` member carried by ≥ 20 fit and ≥ 20 cal items of each class (and, with
    `both_sides`, as many without it). None without term support (not established)."""
    if support is None:
        return None, "no term support recorded"
    entry = support.get(term)
    if entry is None:
        return False, "not a context term"
    sides = ("on", "off") if both_sides else ("on",)
    counts = "; ".join(f"{side}: " + ", ".join(f"{split} real {entry[split]['real'][side]} / fake {entry[split]['fake'][side]}"
                                              for split in ("fit", "cal")) for side in sides)
    enough = all(entry[split][name][side] >= TERM_MIN_N for split in ("fit", "cal") for name in ("real", "fake") for side in sides)
    return bool(entry["group"] and enough), ("" if entry["group"] else "not a fuser group; ") + counts


def gate_words(literals: Sequence[str]) -> str:
    """A gate in words for card text: `!has_face` → "no face is detected"."""
    parts = []
    for lit in literals:
        term, absent = (lit[1:], True) if lit.startswith("!") else (lit, False)
        parts.append(GATE_WORDS[term][1 if absent else 0] if term in GATE_WORDS else "a named image condition holds")
    return " and ".join(parts)


def ablate(ds: Dataset, meta: Mapping[str, Mapping], candidate: str, *, lgo_fold: Sequence[str | None],
           source_dataset: Sequence[str | None], contamination_sets: Sequence[str], modality: str = "image", seed: int = 7,
           n_boot: int = mt.BOOTSTRAP_B, required_folds: Sequence[str] = LGO_FOLDS, cv_splits: int = CV_SPLITS,
           gates: Mapping[str, Sequence[str]] | None = None, size_policy: SizePolicy | None = None) -> dict:
    """`ds` holds every detector including the candidate; lgo_fold and source_dataset follow ds's row order. `size_policy`
    defaults by modality (D-45 for images), so admissions are measured in the shipped feature space. "Without" is the base
    minus the candidate, "with" the base, both with `gates` (`compare`)."""
    if candidate not in ds.detectors:
        raise AblationError(f"no score file for the candidate {candidate}")
    if len(ds.detectors) < 2:
        raise AblationError(f"the ablation needs at least one baseline detector besides {candidate}")
    unknown = [d for d in (gates or {}) if d not in ds.detectors]
    if unknown:
        raise AblationError(f"gates name detectors without scores: {unknown}")
    result = compare(ds, meta, with_detectors=list(ds.detectors), without_detectors=[d for d in ds.detectors if d != candidate],
                     gates_with=gates or {}, gates_without={d: v for d, v in (gates or {}).items() if d != candidate},
                     lgo_fold=lgo_fold, source_dataset=source_dataset,
                     contamination_sets=contamination_sets, modality=modality, seed=seed, n_boot=n_boot,
                     required_folds=required_folds, cv_splits=cv_splits, size_policy=size_policy, candidate=candidate)
    method = {k: v for k, v in result["method"].items() if k != "reference_gates"}
    return {"candidate": candidate, "baseline_detectors": result["systems"]["without"]["detectors"], "items": result["items"],
            "method": method, "lgo": result["lgo"], "cross_fit": result["cross_fit"], "criteria": result["criteria"]}


def compare(ds: Dataset, meta: Mapping[str, Mapping], *, with_detectors: Sequence[str], without_detectors: Sequence[str],
            gates_with: Mapping[str, Sequence[str]], gates_without: Mapping[str, Sequence[str]], lgo_fold: Sequence[str | None],
            source_dataset: Sequence[str | None], contamination_sets: Sequence[str], modality: str = "image", seed: int = 7,
            n_boot: int = mt.BOOTSTRAP_B, required_folds: Sequence[str] = LGO_FOLDS, cv_splits: int = CV_SPLITS,
            size_policy: SizePolicy | None = None, candidate: str | None = None, label: str | None = None) -> dict:
    """Two detector sets fitted on the same items (`ds` over their union), each with its own gates: the LGO, cross-fit and
    criteria tables of `ablate`, with "with" = `with_detectors` and "without" = `without_detectors` (D-53 joint check)."""
    for name, dets, system_gates in (("with", with_detectors, gates_with), ("without", without_detectors, gates_without)):
        if not dets:
            raise AblationError(f"the {name} system has no detector")
        absent = sorted(set(dets) - set(ds.detectors))
        if absent:
            raise AblationError(f"the {name} system names detectors without scores: {absent}")
        stray = sorted(set(system_gates) - set(dets))
        if stray:
            raise AblationError(f"the {name} gates name detectors outside that system: {stray}")
    if candidate is None and set(with_detectors) == set(without_detectors) and _canonical(gates_with) == _canonical(gates_without):
        raise AblationError("the reference equals the configuration: same detectors and gates")
    if set(ds.split) - {"fit", "cal"}:
        raise AblationError(f"the ablation reads fit / cal items only (found splits {sorted(set(ds.split) - {'fit', 'cal'})})")
    folds = np.asarray([f or "" for f in lgo_fold], dtype=object)
    sources = np.asarray(source_dataset, dtype=object)
    if len(folds) != len(ds) or len(sources) != len(ds):
        raise AblationError("lgo_fold and source_dataset must follow the dataset rows")
    try:
        dropped, contamination = contamination_filter(sources, contamination_sets)
    except ValueError as exc:
        raise AblationError(str(exc)) from exc
    contaminated = np.asarray(dropped, dtype=bool)
    kept = select_detectors(ds.take(~contaminated), sorted(set(with_detectors) | set(without_detectors)))
    systems: Systems = {"without": (list(without_detectors), gates_without), "with": (list(with_detectors), gates_with)}
    label = label or candidate or "comparison"
    lgo_result = lgo(kept, folds[~contaminated], candidate, meta, modality=modality, seed=seed, n_boot=n_boot,
                     required=required_folds, size_policy=size_policy, systems=systems, label=label)
    oof = cross_fit(kept, candidate, meta, modality=modality, seed=seed, n_splits=cv_splits, size_policy=size_policy,
                    systems=systems, label=label)
    cross = slices(kept, oof, n_boot=n_boot, seed=seed)
    return {
        "label": label,
        "systems": {name: {"detectors": [d for d in kept.detectors if d in set(systems[name][0])],
                           "gates": _canonical(systems[name][1])} for name in SYSTEMS},
        "items": {"n": len(kept), "n_real": int((kept.y == 0).sum()), "n_fake": int((kept.y == 1).sum()),
                  **contamination},
        "method": {"alpha": ALPHA, "cv_splits": cv_splits, "bootstrap_B": n_boot, "seed": seed, "fuser_bootstrap_B": 1,
                   "splits": ["fit", "cal"], "gates": _canonical(gates_with), "reference_gates": _canonical(gates_without),
                   "size_policy": (size_policy or default_policy(modality)).as_json(), "term_support": term_support(kept)},
        "lgo": lgo_result,
        "cross_fit": cross,
        "criteria": criteria(lgo_result, cross),
    }


def _canonical(gates: Mapping[str, Sequence[str]] | None) -> dict[str, list[str]]:
    return {d: list(v) for d, v in sorted((gates or {}).items())}


def camera_original_check(fpr: float | None, n: int) -> dict:
    """D-41: a stand-alone FPR at τ_0.05 above 10 % on R1 ∪ R2 reals (n ≥ 300) needs a retire / context-gate decision."""
    triggered = fpr is not None and n >= CAMERA_MIN_N and fpr > CAMERA_FPR_MAX
    return {"fpr": fpr, "n": n, "max_fpr": CAMERA_FPR_MAX, "min_n": CAMERA_MIN_N, "triggered": triggered,
            "established": fpr is not None and n >= CAMERA_MIN_N}


def gated_retry(crit: Mapping, *, a1: bool | None, a2: bool | None, a6: bool | None,
                support: Mapping | None) -> dict:
    """D-51 trigger: at least one blocking stratum, A1 / A2 / A6 not failing, and every blocking stratum's runtime term
    learnable on both sides. `gate` is the candidate's gate (`!<term>` per distinct term); `qualifies` is None while a
    blocking stratum's term lacks recorded support."""
    blocking = list(crit["A4"].get("blocking_strata") or [])
    terms = {k: STRATUM_TERMS.get(k) for k in blocking}
    checks = {k: term_learnable(support, t, both_sides=True) if t else (False, "no runtime term") for k, t in terms.items()}
    if not blocking or False in (a1, a2, a6) or any(v is False for v, _ in checks.values()):
        qualifies: bool | None = False
    else:
        qualifies = None if any(v is None for v, _ in checks.values()) else True
    gate = sorted({f"!{t}" for t in terms.values() if t}) if blocking and all(terms.values()) else None
    return {"qualifies": qualifies, "blocking_strata": blocking, "terms": terms, "gate": gate,
            "support": {k: text for k, (_, text) in checks.items()}}


def _harm_text(crit: Mapping) -> str:
    detail = crit["A4"].get("blocking_detail") or {}
    return "; ".join(f"established {k} harm ungated ({_pts(e['delta'])} pts [{_pts(e['lo'])}, {_pts(e['hi'])}], n {e['n']})"
                     for k, e in detail.items()) or ("established harm ungated in " + ", ".join(crit["A4"]["blocking_strata"]))


def admission_decision(crit: Mapping, *, a1: bool | None, a2: bool | None, a5: bool | None, a6: bool | None,
                       a5_failing: Sequence[str] = (), camera: Mapping | None = None, camera_decision: str | None = None,
                       camera_features: Sequence[str] = (), term_support: Mapping | None = None, bench: str | None = None,
                       candidate: str | None = None, gated: Mapping | None = None, carried: Sequence[str] | None = None) -> dict:
    """The admission decision from the criteria verdicts (D-44 amended by D-51 and D-53).

    - with `carried` (D-53: the member's own negated gate, carried in the base and measured in the main ablation): admitted
      context-gated (`gates` {candidate: carried}) when A1, A2, A3, A4 and A6 pass and no stratum blocks; otherwise
      rejected. No gated retry runs;

    - rejected: A1, A2, A3 (A3a or A3b), A4 or A6 fails, or H-3 retired it, unless the gated retry qualifies;
    - not established: one of those is null, the camera-original check triggered without an H-3 decision, a term has no
      recorded support, or the gated retry qualifies and `gated` ({criteria, gates, term_support, command}) is absent;
    - with `gated`: admitted context-gated (`gates` {candidate: literals}) when the gated ablation has no blocking stratum
      and A1, A2, A3, A4 and A6 pass; otherwise rejected, citing both ablations;
    - admitted with context interaction terms: the runtime terms of the transforms that fail A5 (`a5_failing`; A5 never
      rejects on its own), of the strata with ≥ 150 reals whose point ΔFPR exceeds +2.0 pts without established harm, and
      the H-3 gate features, each kept only when learnable (`term_learnable`); otherwise admitted ungated."""
    if camera_decision not in (None, "retire", "context_gate"):
        raise AblationError(f"camera decision {camera_decision!r} is neither retire nor context_gate")
    if camera_decision == "context_gate" and not camera_features:
        raise AblationError("an H-3 context gate needs named features")
    verdicts = {"A1": a1, "A2": a2, "A3": crit["A3"]["pass"], "A4": crit["A4"]["pass"], "A6": a6}
    failing = [k for k, v in verdicts.items() if v is False]
    open_ = [k for k, v in verdicts.items() if v is None]
    reasons = [f"{k} fails" for k in failing] + [f"{k} not established" for k in open_]
    triggered = bool(camera and camera.get("triggered"))
    if triggered:
        reasons.append(f"camera-original FPR {100 * camera['fpr']:.1f} % on R1 ∪ R2 (n {camera['n']}) above 10 %: "
                       + (camera_decision or "no H-3 decision"))
    elif camera_decision:
        reasons.append(f"H-3 {camera_decision}" + (f" on {', '.join(camera_features)}" if camera_features else "")
                       + " without the D-41 trigger")
    if carried is not None and gated is not None:
        raise AblationError(f"{candidate}: a member with a carried gate (D-53) takes no gated retry")
    retry = gated_retry(crit, a1=a1, a2=a2, a6=a6, support=term_support) if camera_decision != "retire" and carried is None else {
        "qualifies": False, "blocking_strata": list(crit["A4"].get("blocking_strata") or []), "terms": {}, "gate": None, "support": {}}
    if gated is not None:
        if retry["qualifies"] is not True:
            raise AblationError(f"{candidate}: a gated ablation was given, but the candidate does not qualify for a gated retry")
        if list(gated.get("gates") or []) != retry["gate"]:
            raise AblationError(f"{candidate}: the gated ablation gates it with {gated.get('gates')}, not the negated stratum term "
                                f"{retry['gate']}")
    terms: list[str] = []
    gates: dict[str, list[str]] = {}
    unsupported: list[str] = []

    def add(term: str, support: Mapping | None) -> None:
        verdict, counts = term_learnable(support, term)
        if verdict is None:
            unsupported.append(term)
            reasons.append(f"term {term} not established: the ablation records no term support")
        elif not verdict:
            reasons.append(f"term {term} not learnable on {bench or 'this bench'} ({counts}): the detector runs ungated; "
                           "the G3 test FPR decides")
        elif term not in terms:
            terms.append(term)

    def admitted_terms(c: Mapping, support: Mapping | None) -> None:
        if a5 is False:
            mapped = {t: A5_TRANSFORM_TERMS.get(t) for t in a5_failing}
            reasons.append("A5 fails on " + (", ".join(f"{t} → {v or 'no runtime term'}" for t, v in mapped.items())
                                            or "no named transform: no runtime term"))
            for term in dict.fromkeys(v for v in mapped.values() if v):
                add(term, support)
        elif a5 is None:
            reasons.append("A5 not established: no A5 interaction term")
        for stratum, term in (c["A4"].get("interaction_terms") or {}).items():
            reasons.append(f"{stratum} point ΔFPR above +2.0 pts without established harm: "
                           + (f"interaction term {term}" if term else "no runtime context term; the G3 test FPR decides"))
            if term:
                add(term, support)
        for stratum in c["A4"].get("small_point_strata") or []:
            reasons.append(f"{stratum} point ΔFPR above +2.0 pts on fewer than {A4_STRATUM_MIN_N} reals: not established; no term")
        if camera_decision == "context_gate":
            for f in camera_features:
                add(f.removeprefix("!"), support)

    if carried is not None:
        literal = f"{candidate}={','.join(carried)}"
        blocking = list(crit["A4"].get("blocking_strata") or [])
        if all(v is True for v in verdicts.values()) and not blocking and camera_decision != "retire":
            reasons.append(f"carried gate {literal} (D-53): measured as it stands in the base")
            admitted_terms(crit, term_support)
            decision = "not_established" if unsupported else "admitted_context_gated"
            gates = {candidate: list(carried)}
        else:
            decision = "rejected"
            reasons.append(f"carried gate {literal} (D-53)" + (f": {_harm_text(crit)}" if blocking else "")
                           + ("; H-3 retire" if camera_decision == "retire" else ""))
    elif (failing or camera_decision == "retire") and retry["qualifies"] is None:
        decision = "not_established"
        reasons.append(f"{_harm_text(crit)}; gated retry not established: " + "; ".join(retry["support"].values()))
    elif (failing or camera_decision == "retire") and retry["qualifies"]:
        literal = ",".join(retry["gate"])
        if gated is None:
            decision = "not_established"
            reasons.append(f"{_harm_text(crit)}; gated retry {candidate}={literal} qualifies and has not run")
        else:
            g = gated["criteria"]
            g_verdicts = {"A1": a1, "A2": a2, "A3": g["A3"]["pass"], "A4": g["A4"]["pass"], "A6": a6}
            g_blocking = list(g["A4"].get("blocking_strata") or [])
            if all(v is True for v in g_verdicts.values()) and not g_blocking:
                reasons = [_harm_text(crit), f"admitted gated {literal} ({gated.get('command') or 'gated ablation'})"]
                admitted_terms(g, gated.get("term_support"))
                decision = "not_established" if unsupported else "admitted_context_gated"
                gates = {candidate: list(retry["gate"])}
            else:
                decision = "rejected"
                reasons.append(_harm_text(crit))
                reasons.append(f"gated {literal} ({gated.get('command') or 'gated ablation'}): "
                               + "; ".join([f"{k} {_verdict(v)}" for k, v in g_verdicts.items() if v is not True]
                                           + ([f"blocking {', '.join(g_blocking)}"] if g_blocking else [])))
    elif failing or camera_decision == "retire":
        decision = "rejected"
    elif open_ or (triggered and camera_decision is None):
        decision = "not_established"
    else:
        admitted_terms(crit, term_support)
        decision = "not_established" if unsupported else ("admitted_context_gated" if terms else "admitted")
    return {"decision": decision, "context_terms": terms, "gates": gates, "reasons": reasons, "verdicts": {**verdicts, "A5": a5},
            "gated_retry": {k: retry[k] for k in ("qualifies", "blocking_strata", "gate")},
            "camera_original": dict(camera) if camera else None, "rule": "D-51"}


def _pts(value: float | None) -> str:
    return "–" if value is None else f"{100 * value:+.1f}"


def _pct(value: float | None) -> str:
    return "–" if value is None else f"{100 * value:.1f} %"


def _q(value: float | None) -> str:
    return "–" if value is None else f"{value:.3f}"


def _ci_pts(entry: Mapping) -> str:
    ci = entry.get("ci95") or {}
    return "–" if ci.get("lo") is None else f"[{100 * ci['lo']:+.1f}, {100 * ci['hi']:+.1f}]"


def _verdict(value: bool | None) -> str:
    return "not established" if value is None else ("pass" if value else "fail")


def render_markdown(result: Mapping, *, command: str, manifest_sha256: str) -> str:
    c, items, lg, cross = result["criteria"], result["items"], result["lgo"], result["cross_fit"]
    if result.get("candidate"):
        head = [f"# Ablation: {result['candidate']}", "", f"- Command: `{command}`", f"- Manifest sha256: `{manifest_sha256}`",
                f"- Baseline detectors: {', '.join(result['baseline_detectors'])}"]
    else:
        systems = result["systems"]
        head = [f"# Comparison: {result['label']}", "", f"- Command: `{command}`", f"- Manifest sha256: `{manifest_sha256}`",
                *(f"- {name.capitalize()}: " + ", ".join(d + (f" ({'|'.join(systems[name]['gates'][d])})" if d in systems[name]["gates"] else "")
                                                         for d in systems[name]["detectors"]) for name in ("with", "without"))]
    lines = [*head,
             f"- Items (fit ∪ cal): {items['n']} ({items['n_real']} real, {items['n_fake']} fake); contaminated dropped "
             f"{items['contaminated_dropped']} ({', '.join(items['contamination_sets']) or 'none listed'})",
             f"- Paired grouped bootstrap B = {result['method']['bootstrap_B']}, seed {result['method']['seed']}; "
             f"cross-fit {result['method']['cv_splits']} folds; τ at α = {result['method']['alpha']}; weighting strata merged "
             f"on the cal reals: {', '.join(f'{k} → {v}' for k, v in (cross.get('threshold_strata') or {}).items()) or 'none'}",
             "- Operating point (D-53): reals scoring exactly τ count as flagged with weight q (the rank rule's share); served "
             "rule (s ≥ τ) weighted FPR: " + ", ".join(
                 f"{name} {_pct(((cross.get('operating_point') or {}).get(name) or {}).get('served_rule_weighted_fpr'))} "
                 f"(q {_q(((cross.get('operating_point') or {}).get(name) or {}).get('q'))})" for name in SYSTEMS), "",
             "| Criterion | Result | Evidence |", "|---|---|---|",
             f"| A3a | {_verdict(c['A3a']['pass'])} | Δ LGO macro {_pts(c['A3a']['lgo_macro_delta'])} pts, CI {_ci_pts(lg['macro'])} |",
             f"| A3b | {_verdict(c['A3b']['pass'])} | {', '.join(c['A3b']['qualifying']) or 'no family or threat gains ≥ 10 pts'} |",
             f"| A4 | {_verdict(c['A4']['pass'])} | pooled reals: Δ FPR {_pts(c['A4']['pooled_delta'])} pts, CI "
             f"{_ci_pts(cross['fpr_pooled'])} (blocks with a lower bound above +1.0, D-53); blocking strata (n ≥ 150, lower bound > +2.0): "
             f"{', '.join(c['A4']['blocking_strata']) or 'none'}; not established: "
             f"{', '.join(c['A4']['not_established_strata']) or 'none'}; point above +2.0 without established harm: "
             f"{', '.join(c['A4'].get('interaction_strata') or []) or 'none'}; above +2.0 on fewer than 150 reals (no term): "
             f"{', '.join(c['A4'].get('small_point_strata') or []) or 'none'} |", "",
             "## LGO folds (TPR@5%FPR on fit / cal fakes of F; refits without F)", "",
             "| Fold | n fake | without | with | Δ pts |", "|---|---|---|---|---|"]
    for fold, e in lg["folds"].items():
        lines.append(f"| {fold} | {e['n_fake']} | {_pct(e['tpr']['without'])} | {_pct(e['tpr']['with'])} | {_pts(e['delta'])} |")
    lines += ["", f"Macro over {', '.join(lg['macro_over']) or '–'}: without {_pct(lg['macro']['without'])}, with "
              f"{_pct(lg['macro']['with'])}. Missing: {'; '.join(lg['missing']) or 'none'}.", "",
              "## Real strata (out-of-fold FPR at τ_0.05)", "", "| Stratum | n | without | with | Δ pts | 95 % CI |",
              "|---|---|---|---|---|---|"]
    for name, e in cross["fpr_per_stratum"].items():
        lines.append(f"| {name} | {e['n']} | {_pct(e['without'])} | {_pct(e['with'])} | {_pts(e['delta'])} | {_ci_pts(e)} |")
    lines += ["", "## Recall per threat and family (out-of-fold, τ_0.05)", "", "| Slice | n | without | with | Δ pts | 95 % CI |",
              "|---|---|---|---|---|---|"]
    for kind, table in (("threat", cross["recall_per_threat"]), ("family", cross["recall_per_family"])):
        for name, e in table.items():
            lines.append(f"| {kind} {name} | {e['n']} | {_pct(e['without'])} | {_pct(e['with'])} | {_pts(e['delta'])} | {_ci_pts(e)} |")
    lines += ["", f"Not decided here: {'; '.join(c['not_decided_here'])}.", ""]
    return "\n".join(lines)
