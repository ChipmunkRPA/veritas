"""Localiser behaviour on benchmark reals and edits from the score cache (BUILD_SPEC H-6; I-10 T-6 reading).

- **Rows.** A localiser row is kept when its source reported `ok` (`aux.reported_status`): localisers publish an uncalibrated
  `score` aggregate, so the cache row status is `not_applicable` while the map and metrics are there (`score_v2.detector_row`).
- **Readings** (as `scripts/v2/localiser_fpr.py`, I-10): a file with ≥ 1 region (`metrics.regions`); the integrity score read
  as a probability, ≥ 2/3 (weak or stronger) and ≥ 10/11 (≥ moderate).
- **Population.** Image items of the given splits with `audit_ok` not false and no C2PA manifest; reals per `real_stratum`,
  fakes of threat `edit` per `source_id`. Rates carry 95 % Wilson intervals; the stand-alone AUC of the integrity score is
  edits vs reals.
"""
from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from app.bench import metrics as mt

WEAK, MODERATE = 2 / 3, 10 / 11
READINGS = ("region", "integrity_weak", "integrity_moderate")


def reading(aux: str | Mapping[str, Any] | None) -> dict[str, Any] | None:
    """{integrity, regions, region, integrity_weak, integrity_moderate} of one cached row, None without a reported map."""
    doc = json.loads(aux) if isinstance(aux, str) and aux else dict(aux or {})
    metrics = doc.get("metrics") or {}
    if doc.get("reported_status") != "ok" or "integrity" not in metrics or "regions" not in metrics:
        return None
    integrity, regions = float(metrics["integrity"]), float(metrics["regions"])
    return {"integrity": integrity, "regions": regions, "region": regions >= 1, "integrity_weak": integrity >= WEAK,
            "integrity_moderate": integrity >= MODERATE}


def _rates(found: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {"n": len(found)}
    for name in READINGS:
        k = sum(1 for r in found if r[name])
        lo_hi = mt.wilson(k, len(found)) if found else None
        out[name] = {"k": k, "rate": k / len(found) if found else None, "wilson95": list(lo_hi) if lo_hi else None}
    return out


def localiser_rates(items: Iterable[Mapping[str, Any]], rows: Iterable[Mapping[str, Any]], *, splits: Sequence[str],
                    variant: str = "norm") -> dict[str, Any]:
    """Rates on reals per stratum and on edits per source, from manifest rows and one localiser's cache rows."""
    latest: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if row["variant"] == variant and row["transform"] == "none":
            latest[row["item_id"]] = row  # the cache keeps the newest row last
    reals: dict[str, list] = {}
    edits: dict[str, list] = {}
    missing = {"real": 0, "edit": 0}
    for item in items:
        if item["modality"] != "image" or item["split"] not in splits or item["audit_ok"] is False or item["has_c2pa"]:
            continue
        if item["label"] == "real":
            kind, key, table = "real", item["real_stratum"] or "unknown", reals
        elif item["threat"] == "edit":
            kind, key, table = "edit", item["source_id"] or "unknown", edits
        else:
            continue
        row = latest.get(item["item_id"])
        found = reading(row.get("aux")) if row else None
        if found is None:
            missing[kind] += 1
            continue
        table.setdefault(key, []).append(found)
    all_reals = [r for v in reals.values() for r in v]
    all_edits = [r for v in edits.values() for r in v]
    scores = [r["integrity"] for r in all_reals] + [r["integrity"] for r in all_edits]
    labels = [0] * len(all_reals) + [1] * len(all_edits)
    auc = mt.auc(scores, labels) if all_reals and all_edits else None
    return {"splits": list(splits), "variant": variant, "missing": missing, "thresholds": {"weak": WEAK, "moderate": MODERATE},
            "reals": {"all": _rates(all_reals), **{k: _rates(v) for k, v in sorted(reals.items())}},
            "edits": {"all": _rates(all_edits), **{k: _rates(v) for k, v in sorted(edits.items())}},
            "integrity_auc_edits_vs_reals": auc}


def _cell(entry: Mapping[str, Any]) -> str:
    if entry["rate"] is None:
        return "–"
    lo, hi = entry["wilson95"]
    return f"{entry['k']} ({100 * entry['rate']:.1f} %; {100 * lo:.1f}–{100 * hi:.1f} %)"


def render_markdown(result: Mapping[str, Any], *, detector: str, command: str, manifest_sha256: str) -> str:
    lines = [f"# Localiser rates: {detector} on {', '.join(result['splits'])} ({result['variant']} bytes)", "",
             f"- Command: `{command}`", f"- Manifest sha256: `{manifest_sha256}`",
             f"- Rows without a reported map: {result['missing']['real']} reals, {result['missing']['edit']} edits (not counted)",
             "- Readings: a file with ≥ 1 region; integrity read as a probability ≥ 2/3 (weak or stronger) and ≥ 10/11 (≥ moderate).",
             "  Counts with 95 % Wilson intervals.",
             f"- Integrity AUC, edits vs reals: {'–' if result['integrity_auc_edits_vs_reals'] is None else f'{result['integrity_auc_edits_vs_reals']:.4f}'}",
             ""]
    for title, table, key in (("Reals (false-positive rates)", result["reals"], "Stratum"), ("Edits (recall)", result["edits"], "Source")):
        lines += [f"## {title}", "", f"| {key} | n | ≥ 1 region | integrity ≥ 2/3 | integrity ≥ 10/11 |", "|---|---|---|---|---|"]
        for name, entry in table.items():
            lines.append(f"| {name} | {entry['n']} | {_cell(entry['region'])} | {_cell(entry['integrity_weak'])} | "
                         f"{_cell(entry['integrity_moderate'])} |")
        lines.append("")
    return "\n".join(lines)
