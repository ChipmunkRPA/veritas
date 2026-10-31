"""Goal gate (ML §8; D-13): `goals.yaml` × `results.json` → one PASS / FAIL row per goal check.

- **Refusals** (`GateRefused`, exit 2): a goals file without goals, results from another bench version than
  `goals.yaml` `bench` (unless the caller names the bench explicitly, which marks the run as not a goal evaluation),
  results not evaluated on `test`, a results manifest sha256 different from the manifest on disk, or different from
  `goals.yaml` `manifest_sha256` once that is pinned.
- **Rows.** A goal with several parts (each stratum, class or transform) gives one row per part. A metric that the results
  do not contain is `MISSING`, which fails the gate like `FAIL` (ML §8: "exits non-zero on any FAIL or on a missing metric").
- **Lookups** follow `app/bench/report.py`: `sets[<set>]`, `real_strata`, `lgo.macro` / `lgo.min_family`,
  `robustness.transforms`, `invariants`, `checks.reproducibility` and `latency` (`scripts/bench.py` labels).
"""
from __future__ import annotations

import operator
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml

from app.bench import perturb as pt
from app.bench.performance_validity import latency_rejection

OPS: dict[str, Callable[[Any, Any], bool]] = {">=": operator.ge, "<=": operator.le, "==": operator.eq}
REQUIRED_KEYS = ("id", "metric", "set", "op", "target")


class GateRefused(ValueError):
    """The gate cannot judge these results against these goals."""


@dataclass(frozen=True)
class Row:
    goal: str
    part: str
    value: Any
    target: Any
    op: str
    status: str  # PASS | FAIL | MISSING
    detail: str = ""


def load_goals(path: Path) -> dict[str, Any]:
    doc = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    goals = doc.get("goals")
    if not isinstance(goals, list) or not goals:
        raise GateRefused(f"{Path(path).name} lists no goals")
    for goal in goals:
        missing = [k for k in REQUIRED_KEYS if k not in goal]
        if missing:
            raise GateRefused(f"goal {goal.get('id', '?')} lacks {missing}")
        if goal["op"] not in OPS:
            raise GateRefused(f"goal {goal['id']} has unknown op {goal['op']!r}")
    ids = [g["id"] for g in goals]
    if len(set(ids)) != len(ids):
        raise GateRefused("goal ids are not unique")
    if not doc.get("bench"):
        raise GateRefused(f"{Path(path).name} names no bench")
    return doc


def check_identity(goals: Mapping[str, Any], results: Mapping[str, Any], *, manifest_sha256: str,
                   bench_override: str | None = None) -> list[str]:
    """Raise GateRefused on a mismatch; return notes for the printed table."""
    bench = results.get("bench") or {}
    notes = []
    expected = bench_override or goals["bench"]
    if bench.get("version") != expected:
        raise GateRefused(f"results are from bench {bench.get('version')!r}, the goals need {expected!r}")
    if bench_override and bench_override != goals["bench"]:
        notes.append(f"bench override {bench_override!r}: goals apply to {goals['bench']}, so this is not a goal evaluation")
    if bench.get("eval_split") != "test":
        raise GateRefused(f"results were evaluated on {bench.get('eval_split')!r}; goals are judged on `test` only")
    if bench.get("manifest_sha256") != manifest_sha256:
        raise GateRefused(f"results were produced on manifest {str(bench.get('manifest_sha256'))[:12]}…, the manifest on disk "
                          f"is {manifest_sha256[:12]}…")
    pinned = goals.get("manifest_sha256")
    if pinned and not bench_override and pinned != manifest_sha256:
        raise GateRefused(f"goals pin manifest {pinned[:12]}…, the manifest on disk is {manifest_sha256[:12]}…")
    if not pinned:
        notes.append("goals.yaml does not pin a manifest sha256 yet")
    return notes


def _dig(doc: Any, *path: Any) -> Any:
    for key in path:
        if not isinstance(doc, Mapping) or key not in doc:
            return None
        doc = doc[key]
    return doc


def _key(value: Any) -> str:
    return f"{float(value):g}"


def _judge(goal: Mapping[str, Any], part: str, value: Any, *, detail: str = "", extra_ok: bool = True,
           extra_detail: str = "") -> Row:
    if value is None:
        return Row(goal["id"], part, None, goal["target"], goal["op"], "MISSING", detail or "metric not in results")
    ok = OPS[goal["op"]](value, goal["target"]) and extra_ok
    return Row(goal["id"], part, value, goal["target"], goal["op"], "PASS" if ok else "FAIL",
               "; ".join(x for x in (detail, extra_detail) if x))


def _rate_row(goal: Mapping[str, Any], part: str, rate: Mapping[str, Any] | None) -> Row:
    if not rate:
        return _judge(goal, part, None)
    ci = rate.get("wilson95")
    detail = f"{rate['k']}/{rate['n']}" + (f", Wilson {ci[0]:.3f}–{ci[1]:.3f}" if ci else "")
    upper = goal.get("wilson_upper")
    if upper is None:
        return _judge(goal, part, rate.get("value"), detail=detail)
    ok = ci is not None and ci[1] <= upper
    return _judge(goal, part, rate.get("value"), detail=detail, extra_ok=ok, extra_detail=f"Wilson upper ≤ {upper}")


EXTRA_LATENCY_LABELS = {"image": ("image_24mp_camera",)}  # J-4 (ML audit P2): G13a also times a 24 MP camera original


def _trivial(reason: str | None) -> str:
    return f"trivially satisfied ({reason})" if reason else ""


def _with_tpr(row: Row, fpr: Mapping[str, Any] | None, tpr: Mapping[str, Any] | None) -> Row:
    """J-4 (ML audit P2, G10b): the TPR of the set's fakes at the same τ beside a false-positive rate, so a pass that flags
    nothing is visible; `degenerate` when the TPR does not exceed the FPR."""
    if not tpr or tpr.get("value") is None:
        return row
    note = f"TPR at τ {tpr['k']}/{tpr['n']} = {tpr['value']:.3f}"
    if fpr and fpr.get("value") is not None and tpr["value"] <= fpr["value"]:
        note += " (degenerate: TPR ≤ FPR)"
    return replace(row, detail="; ".join(x for x in (row.detail, note) if x))


def _transforms(goal: Mapping[str, Any]) -> Sequence[str]:
    transforms = goal.get("transforms") or []
    return list(pt.IMAGE_TRANSFORMS) if transforms == "all" else list(transforms)


def evaluate(goals: Mapping[str, Any], results: Mapping[str, Any]) -> list[Row]:
    rows: list[Row] = []
    for goal in goals["goals"]:
        rows += _goal_rows(goal, results)
    return rows


def _goal_rows(goal: Mapping[str, Any], results: Mapping[str, Any]) -> list[Row]:
    metric, target_set, agg = goal["metric"], goal["set"], goal.get("agg")
    sets = results.get("sets") or {}
    if metric == "tpr_at_fpr":
        fpr = _key(goal["fpr"])
        if target_set == "image/lgo":
            return [_judge(goal, agg or "macro", _dig(results, "lgo", agg or "macro", "tpr_at_fpr", fpr))]
        if target_set.endswith("/lgo"):
            return [_judge(goal, agg or "macro", _dig(results, "lgo_by_modality", target_set, agg or "macro", "tpr_at_fpr", fpr))]
        return [_rate_row(goal, target_set, _dig(sets, target_set, "tpr_at_fpr", fpr))]
    if metric == "fpr_at_tau":
        tau = _key(goal["tau"])
        if agg == "each_stratum":
            return [_rate_row(goal, stratum, _dig(results, "real_strata", stratum, "fpr_at_tau", tau))
                    for stratum in goal.get("strata") or []]
        return [_with_tpr(_rate_row(goal, target_set, _dig(sets, target_set, "fpr_at_tau", tau)),
                          _dig(sets, target_set, "fpr_at_tau", tau), _dig(sets, target_set, "tpr_at_fpr", tau))]
    if metric in ("ece", "brier"):
        if target_set == "image/lgo":
            return [_judge(goal, agg or "macro", _dig(results, "lgo", agg or "macro", metric))]
        return [_judge(goal, target_set, _dig(sets, target_set, metric, "value"))]
    if metric in ("delta_tpr_fixed", "delta_fpr_fixed"):
        field = "delta_tpr" if metric == "delta_tpr_fixed" else "delta_fpr"
        return [_judge(goal, t, _dig(results, "robustness", "transforms", t, field),
                       detail=_trivial(_dig(results, "robustness", "transforms", t, "trivially_satisfied")))
                for t in _transforms(goal)]
    if metric == "conformal_coverage":
        alpha = _key(goal["alpha"])
        return [_rate_row(goal, cls, _dig(sets, target_set, "coverage", alpha, cls)) for cls in ("real", "ai")]
    if metric == "inconclusive_rate":
        if target_set.startswith("image/robustness/"):
            transform = target_set.rsplit("/", 1)[1]
            return [_rate_row(goal, target_set, _dig(results, "robustness", "transforms", transform, "inconclusive_t"))]
        return [_rate_row(goal, target_set, _dig(sets, target_set, "selective", "inconclusive"))]
    if metric == "auc":
        return [_judge(goal, target_set, _dig(sets, target_set, "auc", "value"))]
    if metric == "eer":
        return [_judge(goal, target_set, _dig(sets, target_set, "eer"))]
    if metric == "reproducibility":
        return [_judge(goal, target_set, _dig(results, "checks", "reproducibility", "identical_except_created_at"),
                       detail=str(_dig(results, "checks", "reproducibility", "detail") or ""))]
    if metric == "latency_p95_s":
        labels = list(goal.get("labels") or [])
        latency = results.get("latency")
        latency = latency if isinstance(latency, Mapping) else {}
        if target_set in EXTRA_LATENCY_LABELS and labels:  # J-4: a supplied same-kind workload joins the max, even if invalid
            labels += [label for label in EXTRA_LATENCY_LABELS[target_set] if label in latency and label not in labels]
        values, errors = [], []
        for label in labels:
            record = latency.get(label)
            reason = latency_rejection(record, expected_profile="full")
            if reason is None and record.get("workload") != label:
                reason = "latency evidence names another workload"
            if reason:
                errors.append(f"{label}: {reason}")
            else:
                values.append(record["p95"])
        value = max(values) if values and not errors else None
        return [_judge(goal, ", ".join(labels), value, detail="; ".join(errors))]
    if metric == "additivity_max_abs":
        return [_judge(goal, target_set, _dig(results, "invariants", "additivity_max_abs"))]
    if metric == "monotonicity_violations":
        trials = _dig(results, "invariants", "monotonicity_trials")
        value = _dig(results, "invariants", "monotonicity_violations") if trials else None
        return [_judge(goal, target_set, value, detail=f"{trials} trials" if trials else "")]
    return [Row(goal["id"], target_set, None, goal["target"], goal["op"], "MISSING", f"unknown metric {metric!r}")]


def exit_code(rows: Sequence[Row]) -> int:
    return 0 if rows and all(r.status == "PASS" for r in rows) else 1


def _fmt(value: Any) -> str:
    if isinstance(value, bool) or value is None:
        return "–" if value is None else str(value).lower()
    if isinstance(value, float):
        return f"{value:.3e}" if value and abs(value) < 1e-3 else f"{value:.4f}"
    return str(value)


def render(rows: Sequence[Row], notes: Sequence[str] = ()) -> str:
    header = f"{'goal':6s} {'part':34s} {'value':>12s} {'op':>3s} {'target':>10s}  {'status':8s} detail"
    lines = [*(f"note: {n}" for n in notes), header, "-" * len(header)]
    for r in rows:
        lines.append(f"{r.goal:6s} {r.part[:34]:34s} {_fmt(r.value):>12s} {r.op:>3s} {_fmt(r.target):>10s}  {r.status:8s} {r.detail}")
    passed = sum(r.status == "PASS" for r in rows)
    failed = sum(r.status == "FAIL" for r in rows)
    missing = sum(r.status == "MISSING" for r in rows)
    lines.append(f"{passed} PASS, {failed} FAIL, {missing} MISSING of {len(rows)} checks → "
                 f"{'PASS' if exit_code(rows) == 0 else 'FAIL'}")
    return "\n".join(lines)
