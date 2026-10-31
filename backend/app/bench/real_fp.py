"""Real-photo false-positive analysis (BUILD_SPEC H-3; D-41 camera-original rule).

- **Items.** Image reals of fit ∪ cal as `calibrate.load_records` reads them (audit not failed, no C2PA manifest); megapixels,
  JPEG quality, source and native path come from the manifest row.
- **Stand-alone FPR per detector.** τ_0.05 from the detector's scored fit ∪ cal reals with equal stratum weights
  (`metrics.operational_threshold`): the stratum-weighted FPR is about 5 % by construction, so the per-stratum rates show where
  the false positives concentrate. Wilson 95 % intervals on R1 ∪ R2, on R1 ∪ R2 below and at or above 8 MP, and per stratum.
  Detectors that score only when they find a pattern (copy-move, JPEG double quantisation) abstain on most reals and give one
  capped value, so τ is that value and every scored real is flagged: `camera_all_reals` counts abstentions as not flagged, and
  `tau_is_max` marks the tie. `camera_by_source` counts the flagged R1 ∪ R2 reals per manifest source.
  D-41: FPR above 10 % on R1 ∪ R2 with n ≥ 300 triggers the retire / context-gate decision (`ablation.camera_original_check`).
- **Fused view.** The image fuser's s for every real (Stages 2–3, no deployment prior); τ_0.05 from its `cal` reals with equal
  stratum weights, as `vbench-report` sets it. For the flagged R1 ∪ R2 reals: the mean exact contribution (probability points)
  per detector, how often each detector gives the largest positive contribution, and per source of R1 ∪ R2 reals the flagged
  share, the median s and the median context prior (Stage 2 before detector terms), beside the median s of that source's fakes.
- **Worst camera-original false positives.** R1 ∪ R2 reals by descending s, 60 items for the contact sheet: every flagged real
  first; when fewer than 60 are flagged, the highest-scoring unflagged reals fill the sheet (`flagged` false).
"""
from __future__ import annotations

import statistics
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence

import numpy as np

from app.bench import metrics as mt
from app.bench.ablation import CAMERA_STRATA, camera_original_check
from app.engine.fusion.artefact import Fuser

ALPHA = 0.05
MP_SPLIT = 8.0
SHEET_ITEMS = 60


def megapixels(row: Mapping | None) -> float | None:
    if not row or not row.get("width") or not row.get("height"):
        return None
    return row["width"] * row["height"] / 1e6


def rate(flags: np.ndarray) -> dict:
    k, n = int(np.asarray(flags, dtype=bool).sum()), int(len(flags))
    ci = mt.wilson(k, n) if n else None
    return {"k": k, "n": n, "value": k / n if n else None, "wilson95": None if ci is None else {"lo": ci[0], "hi": ci[1]}}


def _per_stratum(flags: np.ndarray, strata: np.ndarray) -> dict[str, dict]:
    return {str(k): rate(flags[strata == k]) for k in sorted(set(strata.tolist()))}


def _by_source(flags: np.ndarray, sources: Sequence[str | None]) -> dict[str, dict]:
    counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for flag, src in zip(flags.tolist(), sources, strict=True):
        counts[str(src or "unknown")][0] += int(flag)
        counts[str(src or "unknown")][1] += 1
    return {src: {"k": k, "n": n} for src, (k, n) in sorted(counts.items())}


def standalone(records: Sequence[Mapping], rows_by_id: Mapping[str, Mapping], detectors: Sequence[str]) -> dict[str, dict]:
    reals = [r for r in records if r["label"] == "real"]
    n_camera = sum(str(r["stratum"]) in CAMERA_STRATA for r in reals)
    out: dict[str, dict] = {}
    for det in detectors:
        scored = [r for r in reals if (r["scores"].get(det) or (None,))[0] is not None]
        if not scored:
            out[det] = {"n": 0, "tau": None, "tau_is_max": False, "camera": rate(np.zeros(0, dtype=bool)),
                        "camera_below_8mp": rate(np.zeros(0, dtype=bool)), "camera_8mp_or_more": rate(np.zeros(0, dtype=bool)),
                        "camera_all_reals": rate(np.zeros(n_camera, dtype=bool)), "per_stratum": {}, "camera_by_source": {},
                        "check": camera_original_check(None, 0)}
            continue
        s = np.asarray([r["scores"][det][0] for r in scored], dtype=np.float64)
        strata = np.asarray([str(r["stratum"]) for r in scored], dtype=object)
        tau = mt.operational_threshold(s, ALPHA, strata.tolist())
        flags = mt.flagged(s, tau)
        camera = np.isin(strata, CAMERA_STRATA)
        mp = np.asarray([megapixels(rows_by_id.get(r["item_id"])) or np.nan for r in scored], dtype=np.float64)
        cam = rate(flags[camera])
        out[det] = {"n": len(scored), "tau": tau, "tau_is_max": tau is not None and tau >= float(s.max()), "camera": cam,
                    "camera_below_8mp": rate(flags[camera & (mp < MP_SPLIT)]), "camera_8mp_or_more": rate(flags[camera & (mp >= MP_SPLIT)]),
                    "camera_all_reals": rate(np.arange(n_camera) < cam["k"]), "per_stratum": _per_stratum(flags, strata),
                    "camera_by_source": _by_source(flags[camera], [rows_by_id.get(r["item_id"], {}).get("source_id")
                                                                   for r, c in zip(scored, camera, strict=True) if c]),
                    "check": camera_original_check(cam["value"], cam["n"])}
    return out


def fused(fuser: Fuser, records: Sequence[Mapping], rows_by_id: Mapping[str, Mapping] | None = None, *,
          sheet_items: int = SHEET_ITEMS) -> dict:
    reals = [r for r in records if r["label"] == "real"]
    if not reals:
        raise ValueError("no real items in fit ∪ cal")
    outputs = [fuser.apply(r["scores"], r["context"]) for r in reals]
    s = np.asarray([o.probability for o in outputs], dtype=np.float64)
    strata = np.asarray([str(r["stratum"]) for r in reals], dtype=object)
    cal = np.asarray([r["split"] == "cal" for r in reals])
    tau = mt.operational_threshold(s[cal], ALPHA, strata[cal].tolist()) if cal.any() else None
    flags = mt.flagged(s, tau)
    camera = np.isin(strata, CAMERA_STRATA)
    ranked = sorted(np.flatnonzero(camera), key=lambda i: (-s[i], reals[i]["item_id"]))
    flagged = [i for i in ranked if flags[i]]
    def source(record: Mapping) -> str:
        return str((rows_by_id or {}).get(record["item_id"], {}).get("source_id") or "unknown")

    sources: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for i in ranked:
        entry = sources[source(reals[i])]
        entry["flags"].append(bool(flags[i]))
        entry["s"].append(float(s[i]))
        entry["prior"].append(1.0 / (1.0 + np.exp(-outputs[i].decomposition.context_prior)))
    fake_s: dict[str, list[float]] = defaultdict(list)
    for r in records:
        if r["label"] == "fake" and source(r) in sources:
            fake_s[source(r)].append(fuser.apply(r["scores"], r["context"]).probability)
    by_source = {src: {"flagged": sum(e["flags"]), "n": len(e["flags"]), "median_s": statistics.median(e["s"]),
                       "median_context_prior": float(statistics.median(e["prior"])), "fake_n": len(fake_s[src]),
                       "fake_median_s": statistics.median(fake_s[src]) if fake_s[src] else None}
                 for src, e in sorted(sources.items(), key=lambda kv: (-sum(kv[1]["flags"]), kv[0]))}
    pp: dict[str, list[float]] = defaultdict(list)
    largest: Counter = Counter()
    for i in flagged:
        rows = [row for row in outputs[i].decomposition.rows if row.kind == "detector"]
        for row in rows:
            pp[row.detector].append(row.pp)
        positive = [row for row in rows if row.pp > 0]
        if positive:
            largest[max(positive, key=lambda row: (row.pp, row.detector)).detector] += 1
    contributions = {det: {"mean_pp": statistics.fmean(pp[det]) if pp[det] else None, "largest_positive": int(largest[det])}
                     for det in fuser.detectors}
    return {"tau": tau, "n_real": len(reals), "n_cal_real": int(cal.sum()), "camera": rate(flags[camera]),
            "per_stratum": _per_stratum(flags, strata), "contributions": contributions, "camera_by_source": by_source,
            "worst": [{"item_id": reals[i]["item_id"], "s": float(s[i]), "stratum": str(strata[i]), "flagged": bool(flags[i])}
                      for i in ranked[:sheet_items]],
            "n_flagged_camera": len(flagged)}


def _pct(entry: Mapping) -> str:
    if not entry or entry.get("value") is None:
        return "–"
    ci = entry["wilson95"]
    return f"{100 * entry['value']:.1f} % ({entry['k']}/{entry['n']}; {100 * ci['lo']:.1f}–{100 * ci['hi']:.1f})"


def render_tables(result: Mapping) -> str:
    strata = sorted({k for d in result["standalone"].values() for k in d["per_stratum"]})
    lines = [f"# H-3 tables: real-photo false positives on {result['bench']} fit ∪ cal", "",
             f"Generated {result['generated']} by `backend/scripts/real_fp_analysis.py`. Manifest sha256 `{result['manifest_sha256']}`; "
             f"fuser `{result['fuser']['id']}` (sha256 `{result['fuser']['sha256']}`). τ_0.05 with equal stratum weights; "
             "rates with Wilson 95 % intervals. Numbers only: causes and decisions are in `h3_real_fp_analysis.md`.", "",
             "## Stand-alone detectors (τ from the detector's own fit ∪ cal reals)", "",
             "| Detector | n reals | τ | R1 ∪ R2 FPR | R1 ∪ R2 < 8 MP | R1 ∪ R2 ≥ 8 MP | R1 ∪ R2, abstentions not flagged | D-41 check | "
             + " | ".join(strata) + " |", "|---|---|---|---|---|---|---|---|" + "---|" * len(strata)]
    for det, d in result["standalone"].items():
        check = d["check"]
        verdict = "triggered" if check["triggered"] else ("clear" if check["established"] else "n < 300")
        tau = "–" if d["tau"] is None else f"{d['tau']:.4f}" + (" (max, ties)" if d.get("tau_is_max") else "")
        cells = [_pct(d["per_stratum"].get(k, {})) for k in strata]
        lines.append(f"| `{det}` | {d['n']} | {tau} | {_pct(d['camera'])} | {_pct(d['camera_below_8mp'])} | "
                     f"{_pct(d['camera_8mp_or_more'])} | {_pct(d.get('camera_all_reals', {}))} | {verdict} | " + " | ".join(cells) + " |")
    sources = sorted({src for d in result["standalone"].values() for src in d.get("camera_by_source", {})})
    lines += ["", "## Stand-alone R1 ∪ R2 flags per source (k / n at the detector's τ)", "",
              "| Detector | " + " | ".join(sources) + " |", "|---|" + "---|" * len(sources)]
    for det, d in result["standalone"].items():
        by = d.get("camera_by_source", {})
        if by:
            lines.append(f"| `{det}` | " + " | ".join(f"{by[src]['k']} / {by[src]['n']}" if src in by else "–" for src in sources) + " |")
    f = result["fused"]
    lines += ["", f"## Fused ({result['fuser']['id']}; τ {f['tau'] if f['tau'] is None else round(f['tau'], 4)} from "
              f"{f['n_cal_real']} cal reals)", "", "| Slice | FPR |", "|---|---|", f"| R1 ∪ R2 | {_pct(f['camera'])} |"]
    lines += [f"| {k} | {_pct(v)} |" for k, v in f["per_stratum"].items()]
    lines += ["", f"## Contributions on the {f['n_flagged_camera']} flagged R1 ∪ R2 reals", "",
              "| Detector | Mean contribution (pp) | Largest positive contribution (items) |", "|---|---|---|"]
    for det, c in sorted(f["contributions"].items(), key=lambda kv: -(kv[1]["mean_pp"] or 0.0)):
        mean = "–" if c["mean_pp"] is None else f"{100 * c['mean_pp']:+.1f}"
        lines.append(f"| `{det}` | {mean} | {c['largest_positive']} |")
    lines += ["", f"## Flagged R1 ∪ R2 reals per source ({f['n_flagged_camera']} flagged)", "",
              "| Source | Flagged | FPR | Median s | Median context prior | Fakes of the source: n, median s |", "|---|---|---|---|---|---|"]
    for src, c in f.get("camera_by_source", {}).items():
        fakes = "–" if c["fake_median_s"] is None else f"{c['fake_n']}, {c['fake_median_s']:.3f}"
        lines.append(f"| {src} | {c['flagged']} / {c['n']} | {100 * c['flagged'] / c['n']:.1f} % | {c['median_s']:.3f} | "
                     f"{c['median_context_prior']:.3f} | {fakes} |")
    n_above = sum(bool(w.get("flagged", True)) for w in result["worst"])
    lines += ["", f"## Worst camera-original false positives (contact sheet order, {len(result['worst'])} items; {n_above} at or above τ)", "",
              "| # | Item | s | Above τ | Stratum | Source | MP | JPEG q |", "|---|---|---|---|---|---|---|---|"]
    for index, w in enumerate(result["worst"], 1):
        mp = "–" if w.get("megapixels") is None else f"{w['megapixels']:.1f}"
        lines.append(f"| {index} | `{w['item_id'][:12]}` | {w['s']:.3f} | {'yes' if w.get('flagged', True) else 'no'} | {w['stratum']} | "
                     f"{w.get('source_id') or '–'} | {mp} | {w.get('jpeg_q_est') if w.get('jpeg_q_est') is not None else '–'} |")
    if result.get("sheet"):
        lines += ["", f"Contact sheet `{result['sheet']['path']}` (sha256 `{result['sheet']['sha256']}`, gitignored)."]
    return "\n".join(lines) + "\n"
