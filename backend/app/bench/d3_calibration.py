"""D3 σ threshold on compressed real video (BUILD_SPEC H-8 "recalibrate D3 on compressed reals"; VF §5, §10).

`vid.synth.d3_xclip` publishes the median σ of its clips without a probability scale (`raw_kind` score); the method reads a
low σ as generated. This module sets the σ threshold on real footage in its compressed forms and measures what it does.

- **Real rows.** Video reals of the given splits (`audit_ok` not false): the native file, plus the V-4 re-encodes of the same
  file (libx264 veryfast at CRF 23, 28, 35 and CRF 28 at ≤ 720 px; `data/vbench/eval/d3_lane/crf.parquet`). Native σ comes
  from the score cache (`aux.metrics.sigma_median`, captured since H-6) and, for items the cache has no σ for, from the V-4
  native run (`d3_lane/native.parquet`, the same staged pipeline with the plan seeded from sha256(item_id) instead of the media
  sha256, i.e. other clip positions in the same shots); where both exist their difference and medians are reported.
- **Threshold.** s = −σ; τ_α = `metrics.operational_threshold` over every real row with the encodes as equally weighted
  strata (native, crf23 …), so a flagged row has σ ≤ σ_α = −τ_α. Nothing about fakes is fitted.
- **Scale.** s_σ = IQR(σ over the real rows) / 1.349, a robust standard deviation.
- **Feature.** The derived feature `vd.d3.sigma` (`app/engine/fusion/derived.py`) is z = clip((σ_0.05 − σ) / s_σ, ±8): 0 at
  the 5 % real threshold, positive below it.
- **Rates.** FPR per encode and per real stratum (native rows), TPR per threat on fakes of the same splits (native), with
  Wilson 95 % intervals; stand-alone AUC of −σ, native reals vs fakes.
"""
from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np

from app.bench import metrics as mt

SCHEMA = "veritas.d3-sigma/1"
ALPHAS = (0.05, 0.01)
IQR_TO_SD = 1.349
NATIVE = "native"


def cache_sigma(aux: str | Mapping[str, Any] | None) -> float | None:
    """σ of one score-cache row of `vid.synth.d3_xclip`, None when the detector did not report `ok` with the statistic."""
    doc = json.loads(aux) if isinstance(aux, str) and aux else dict(aux or {})
    value = (doc.get("metrics") or {}).get("sigma_median")
    if doc.get("reported_status") != "ok" or value is None or not math.isfinite(float(value)):
        return None
    return float(value)


def _eval_sigma(row: Mapping[str, Any]) -> float | None:
    value = row.get("score")
    if row.get("status") != "ok" or value is None or not math.isfinite(float(value)):
        return None
    return float(value)


def collect(items: Iterable[Mapping[str, Any]], cache_rows: Iterable[Mapping[str, Any]], eval_native: Iterable[Mapping[str, Any]],
            eval_crf: Iterable[Mapping[str, Any]], *, splits: Sequence[str]) -> dict[str, Any]:
    """{reals: [(item_id, encode, σ, stratum)], fakes: [(item_id, σ, threat)], agreement, missing} for video items of `splits`."""
    if "test" in splits:
        raise ValueError("test is the lockbox: read it through vbench_report.py")
    cached: dict[str, float | None] = {}
    for row in cache_rows:
        if row["variant"] == NATIVE and row["transform"] == "none":
            cached[row["item_id"]] = cache_sigma(row.get("aux"))  # the cache keeps the newest row last
    native = {r["item_id"]: _eval_sigma(r) for r in eval_native if r.get("variant", NATIVE) == NATIVE}
    crf: dict[str, list[tuple[str, float]]] = {}
    for r in eval_crf:
        sigma = _eval_sigma(r)
        if sigma is not None:
            crf.setdefault(r["item_id"], []).append((str(r["variant"]), sigma))
    reals, fakes, diffs, pairs = [], [], [], []
    sources = {"cache": 0, "v4_native": 0}
    missing = {"real": 0, "fake": 0}
    for item in items:
        if item["modality"] != "video" or item["split"] not in splits or item.get("audit_ok") is False:
            continue
        iid, a, b = item["item_id"], cached.get(item["item_id"]), native.get(item["item_id"])
        if a is not None and b is not None:
            diffs.append(abs(a - b))
            pairs.append((a, b))
        sigma = a if a is not None else b
        if sigma is not None:
            sources["cache" if a is not None else "v4_native"] += 1
        if item["label"] == "real":
            if sigma is None:
                missing["real"] += 1
            else:
                reals.append((iid, NATIVE, sigma, item.get("real_stratum") or "unknown"))
            reals.extend((iid, encode, s, item.get("real_stratum") or "unknown") for encode, s in sorted(crf.get(iid, [])))
        elif sigma is None:
            missing["fake"] += 1
        else:
            fakes.append((iid, sigma, item.get("threat") or "unknown"))
    agreement = {"n": len(diffs), "max_abs_diff": max(diffs) if diffs else None,
                 "median_abs_diff": float(np.median(diffs)) if diffs else None,
                 "median_sigma_cache": float(np.median([a for a, _ in pairs])) if pairs else None,
                 "median_sigma_v4": float(np.median([b for _, b in pairs])) if pairs else None}
    return {"reals": reals, "fakes": fakes, "agreement": agreement, "native_sources": sources, "missing": missing}


def _rate(k: int, n: int) -> dict[str, Any]:
    ci = mt.wilson(k, n)
    return {"k": k, "n": n, "rate": k / n if n else None, "wilson95": list(ci) if ci else None}


def calibrate(collected: Mapping[str, Any]) -> dict[str, Any]:
    """Thresholds, scale and rates (see the module docstring)."""
    reals, fakes = collected["reals"], collected["fakes"]
    if not reals:
        raise ValueError("no real rows with a σ")
    sigma = np.array([s for _, _, s, _ in reals], dtype=np.float64)
    encodes = [e for _, e, _, _ in reals]
    thresholds: dict[str, float | None] = {}
    for alpha in ALPHAS:
        tau = mt.operational_threshold(-sigma, alpha, strata=encodes)
        thresholds[f"{alpha:g}"] = None if tau is None else -tau
    q1, q3 = np.percentile(sigma, [25, 75])
    scale = float((q3 - q1) / IQR_TO_SD)
    if not scale > 0:
        raise ValueError("the real σ rows have no spread")
    sigma_05 = thresholds["0.05"]

    def below(values: Sequence[float]) -> int:
        return 0 if sigma_05 is None else int(sum(1 for v in values if v <= sigma_05))

    by_encode: dict[str, list[float]] = {}
    by_stratum: dict[str, list[float]] = {}
    for _, encode, s, stratum in reals:
        by_encode.setdefault(encode, []).append(s)
        if encode == NATIVE:
            by_stratum.setdefault(stratum, []).append(s)
    by_threat: dict[str, list[float]] = {}
    for _, s, threat in fakes:
        by_threat.setdefault(threat, []).append(s)
    fake_sigma = [s for _, s, _ in fakes]
    native_real = by_encode.get(NATIVE, [])
    auc = mt.auc([-v for v in native_real] + [-v for v in fake_sigma], [0] * len(native_real) + [1] * len(fake_sigma)) \
        if native_real and fake_sigma else None

    def table(groups: Mapping[str, list[float]]) -> dict[str, Any]:
        return {k: {**_rate(below(v), len(v)), "median_sigma": float(np.median(v))} for k, v in sorted(groups.items())}

    return {"sigma_threshold": thresholds, "scale": scale, "iqr": [float(q1), float(q3)], "n_real_rows": len(reals),
            "n_real_items": len({i for i, *_ in reals}),
            "fpr": {"all": {**_rate(below(sigma.tolist()), len(reals)), "median_sigma": float(np.median(sigma))},
                    "encode": table(by_encode), "stratum": table(by_stratum)},
            "tpr": {"all": {**_rate(below(fake_sigma), len(fake_sigma)),
                            "median_sigma": float(np.median(fake_sigma)) if fake_sigma else None},
                    "threat": table(by_threat)},
            "auc_native_reals_vs_fakes": auc}


def _cell(entry: Mapping[str, Any]) -> str:
    if entry["rate"] is None:
        return "–"
    lo, hi = entry["wilson95"]
    return f"{entry['rate'] * 100:.1f} % ({entry['k']}/{entry['n']}; {lo * 100:.1f}–{hi * 100:.1f} %)"


def render_markdown(doc: Mapping[str, Any]) -> str:
    res = doc["result"]
    col = doc["collected"]
    thr = res["sigma_threshold"]
    fmt = lambda v: "not established" if v is None else f"{v:.6f}"  # noqa: E731
    agree = col["agreement"]
    lines = [f"# D3 σ threshold on compressed real video ({doc['bench']}, {', '.join(doc['splits'])})", "",
             f"- Command: `{doc['command']}`", f"- Manifest sha256: `{doc['manifest_sha256']}`",
             f"- Score key: `{doc['score_key']}`; V-4 runs: `{doc['eval_dir']}` (native, crf)",
             f"- Real rows: {res['n_real_rows']} from {res['n_real_items']} items (native file plus its re-encodes); "
             f"reals without a σ {col['missing']['real']}, fakes without a σ {col['missing']['fake']} (not counted)",
             f"- Native σ source: score cache {col['native_sources']['cache']}, V-4 native run {col['native_sources']['v4_native']}; "
             f"cache vs V-4 on {agree['n']} items: max |Δσ| {fmt(agree['max_abs_diff'])}, median {fmt(agree['median_abs_diff'])}; "
             f"median σ {fmt(agree['median_sigma_cache'])} vs {fmt(agree['median_sigma_v4'])}. Both runs score the same shots and clip "
             "counts, but the V-4 script seeded the sampling plan from sha256(item_id) and the production path seeds it from the media "
             "sha256, so the clips sit at other positions and σ differs per item; the threshold reads only the distribution of σ.",
             "- Threshold: rank rule on s = −σ over every real row, encodes weighted equally; a row is flagged when σ ≤ σ_α. "
             "Fitted on these reals and reported on them (in-sample); no fake is used for fitting.",
             "",
             "| Quantity | Value |", "|---|---|",
             f"| σ_0.05 | {fmt(thr['0.05'])} |", f"| σ_0.01 | {fmt(thr['0.01'])} |",
             f"| s_σ = IQR / 1.349 | {res['scale']:.6f} (IQR {res['iqr'][0]:.4f}–{res['iqr'][1]:.4f}) |",
             f"| Stand-alone AUC of −σ, native reals vs fakes | {'–' if res['auc_native_reals_vs_fakes'] is None else f'{res['auc_native_reals_vs_fakes']:.4f}'} |",
             "", "Derived feature `vd.d3.sigma`: z = clip((σ_0.05 − σ) / s_σ, ±8).", ""]
    for title, key, groups in (("False-positive rate at σ_0.05 per encode", "Encode", res["fpr"]["encode"]),
                               ("False-positive rate at σ_0.05 per real stratum (native files)", "Stratum", res["fpr"]["stratum"]),
                               ("Recall at σ_0.05 per threat (native files)", "Threat", res["tpr"]["threat"])):
        lines += [f"## {title}", "", f"| {key} | n | median σ | at or below σ_0.05 |", "|---|---|---|---|"]
        for name, entry in groups.items():
            lines.append(f"| {name} | {entry['n']} | {entry['median_sigma']:.4f} | {_cell(entry)} |")
        lines.append("")
    lines += [f"All real rows: {_cell(res['fpr']['all'])}. All fakes: {_cell(res['tpr']['all'])}.", ""]
    return "\n".join(lines)
