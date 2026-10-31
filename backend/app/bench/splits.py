"""Veritas-Bench splits (ML §2.1 P1–P2, §2.6; D-06): audit exclusions, group-stratified fit / cal / test, LGO labels.

- `excluded`: every item of a source that failed the label audit (D-06).
- `diagnostic`: items of `role: diagnostic` sources (CommunityForensics-Eval, a contamination check); never fitted.
- `wild`: in-the-wild fakes (family `wild`), reported separately and never fitted (§2.6). R3 social-media reals stay in
  fit / cal / test so that their error rate exists (P4); the wild report pairs the wild fakes with the R3 `test` reals.
- everything else: `StratifiedGroupKFold(20, shuffle=True, random_state=seed)` over the items in item-id order,
  stratified by (modality, threat, family, real stratum) with `group_key` as the groups, whole folds assigned: folds 0–6
  → `fit` (35 %), 7–10 → `cal` (20 %), 11–19 → `test` (45 %).
- LGO fold labels come from the source quotas; `check_lgo_folds` refuses a fake family carrying two labels, a real item
  carrying one, or a fold spanning modalities.
"""
from __future__ import annotations

import warnings
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable

import numpy as np

from app.bench.identity import audit as identity_audit
from app.bench.identity import grouped_rows

N_FOLDS = 20
FIT_FOLDS = 7
CAL_FOLDS = 4
FITTED = ("fit", "cal", "test")
SPLITS = (*FITTED, "wild", "diagnostic", "excluded")
MIN_TEST_REALS = 300  # D-40 vb-1.1 target (D-38 flagged vb-1.0 strata below 360): fewer `test` reals are flagged


class SplitError(ValueError):
    """The manifest cannot be split (unaudited sources, inconsistent LGO labels, too few groups)."""


def stratum_key(row: dict) -> str:
    return " · ".join(str(row.get(k) or "–") for k in ("modality", "threat", "family", "real_stratum"))


def fold_split(fold: int) -> str:
    return "fit" if fold < FIT_FOLDS else "cal" if fold < FIT_FOLDS + CAL_FOLDS else "test"


def assign_splits(rows: list[dict], passed: dict[str, bool], *, seed: int = 7) -> list[dict]:
    """New rows (item-id order) with `audit_ok` and `split` set; `passed` maps every source id to its audit result."""
    from sklearn.model_selection import StratifiedGroupKFold

    missing = sorted({r["source_id"] for r in rows} - set(passed))
    if missing:
        raise SplitError(f"sources without an audit record (D-06): {', '.join(missing)}")
    rows = grouped_rows(rows)
    out = [{**r, "audit_ok": passed[r["source_id"]], "split": None} for r in sorted(rows, key=lambda r: r["item_id"])]
    pool = []
    for row in out:
        if not row["audit_ok"]:
            row["split"] = "excluded"
        elif row.get("role") == "diagnostic":
            row["split"] = "diagnostic"
        elif row["label"] == "fake" and row.get("family") == "wild":
            row["split"] = "wild"
        else:
            pool.append(row)
    if not pool:
        _check_identity(out)
        return out
    groups = [r["group_key"] for r in pool]
    if len(set(groups)) < N_FOLDS:
        raise SplitError(f"{len(set(groups))} groups cannot fill {N_FOLDS} folds")
    y = np.unique([stratum_key(r) for r in pool], return_inverse=True)[1]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # strata smaller than the fold count
        folds = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed).split(np.zeros(len(pool)), y, groups)
        for fold, (_, members) in enumerate(folds):
            for i in members:
                pool[i]["split"] = fold_split(fold)
    _check_identity(out)
    return out


def _check_identity(rows: list[dict]) -> None:
    result = identity_audit(rows)
    if not result["passed"]:
        raise SplitError("source/content identity audit failed: "
                         f"{len(result['cross_split_components'])} cross-split components, "
                         f"{len(result['exact_content_label_conflicts'])} content label conflicts, "
                         f"{len(result['unrecognised_openvid_items'])} unrecognised OpenVid sources, "
                         f"{len(result['unrecognised_talkinghead_items'])} unrecognised TalkingHeadBench sources")


def check_lgo_folds(rows: Iterable[dict]) -> Counter:
    """Items per LGO fold; raises when the labels are inconsistent."""
    problems: list[str] = []
    families: dict[tuple[str, str], set] = defaultdict(set)
    modalities: dict[str, set] = defaultdict(set)
    counts: Counter = Counter()
    for row in rows:
        fold = row.get("lgo_fold")
        if row["label"] == "real":
            if fold:
                problems.append(f"real item {row['item_id'][:12]} carries LGO fold {fold}")
            continue
        families[(row["modality"], row.get("family") or "")].add(fold)
        if fold:
            modalities[fold].add(row["modality"])
            counts[fold] += 1
    problems += [f"family {m}/{f} carries folds {sorted(str(v) for v in folds)}"
                 for (m, f), folds in sorted(families.items()) if len(folds) > 1]
    problems += [f"fold {fold} spans modalities {sorted(ms)}" for fold, ms in sorted(modalities.items()) if len(ms) > 1]
    if problems:
        raise SplitError("; ".join(problems[:10]))
    return counts


def mixed_groups(rows: Iterable[dict]) -> list[str]:
    """Group keys whose items sit in more than one of fit / cal / test (P1 requires none)."""
    seen: dict[str, set] = defaultdict(set)
    for row in rows:
        if row.get("split") in FITTED:
            seen[row["group_key"]].add(row["split"])
    return sorted(g for g, s in seen.items() if len(s) > 1)


# --- summary -----------------------------------------------------------------------------------------------

def _pivot(rows: list[dict], key: Callable[[dict], tuple]) -> list[tuple[tuple, Counter]]:
    table: dict[tuple, Counter] = defaultdict(Counter)
    for row in rows:
        table[key(row)][row["split"]] += 1
    return sorted(table.items(), key=lambda kv: tuple(str(k) for k in kv[0]))


def _table(header: list[str], rows: list[dict], key: Callable[[dict], tuple], splits: list[str]) -> list[str]:
    lines = ["| " + " | ".join(header + splits + ["total"]) + " |", "|" + "---|" * (len(header) + len(splits) + 1)]
    for values, counts in _pivot(rows, key):
        cells = [f"`{v}`" if i == 0 and header[0] == "Source" else str(v if v is not None else "–") for i, v in enumerate(values)]
        lines.append("| " + " | ".join(cells + [str(counts.get(s, 0)) for s in splits] + [str(sum(counts.values()))]) + " |")
    return lines


def render_summary(rows: list[dict], *, bench: str, seed: int, manifest_sha256: str, audit: dict[str, str],
                   shortcut: list, lgo: Counter, generated_at: str, manifest_label: str) -> str:
    splits = [s for s in SPLITS if any(r["split"] == s for r in rows)]
    totals = Counter(r["split"] for r in rows)
    groups = {s: len({r["group_key"] for r in rows if r["split"] == s}) for s in splits}
    passed = sum(1 for v in audit.values() if v == "pass")
    lines = [
        f"# {bench} summary", "",
        f"Generated {generated_at} by `backend/scripts/vbench_split.py` from `{manifest_label}`.", "",
        f"- **Manifest:** {len(rows)} items, sha256 `{manifest_sha256}` (after the split and audit columns were written).",
        f"- **Splits:** `StratifiedGroupKFold({N_FOLDS}, shuffle=True, random_state={seed})` by (modality, threat, family, "
        f"real stratum) over `group_key`; folds 0–{FIT_FOLDS - 1} fit, {FIT_FOLDS}–{FIT_FOLDS + CAL_FOLDS - 1} cal, "
        f"{FIT_FOLDS + CAL_FOLDS}–{N_FOLDS - 1} test (ML §2.6). In-the-wild fakes → `wild`, diagnostic sources → "
        "`diagnostic`, sources failing the label audit → `excluded`.",
        f"- **Group disjointness:** {len(mixed_groups(rows))} group keys span more than one of fit / cal / test.",
        f"- **Label audit (D-06, agent):** {len(audit)} sources, {passed} passed, {len(audit) - passed} failed "
        "(`docs/v2/bench/audit.md`).", "",
        "## Splits", "",
        "| Split | Items | Real | Fake | Groups | Share of fit ∪ cal ∪ test |", "|---|---|---|---|---|---|",
    ]
    fitted = sum(totals[s] for s in FITTED) or 1
    for s in splits:
        real = sum(1 for r in rows if r["split"] == s and r["label"] == "real")
        share = f"{100 * totals[s] / fitted:.1f} %" if s in FITTED else "–"
        lines.append(f"| `{s}` | {totals[s]} | {real} | {totals[s] - real} | {groups[s]} | {share} |")
    real_rows = [r for r in rows if r["label"] == "real"]
    fake_rows = [r for r in rows if r["label"] == "fake"]
    lines += ["", "## Real strata", ""]
    lines += _table(["Modality", "Stratum"], real_rows, lambda r: (r["modality"], r.get("real_stratum")), splits)
    lines += ["", "## Test reals per stratum", "",
              f"Strata with fewer than {MIN_TEST_REALS} `test` reals are flagged: the gate reports their false-positive rates "
              "with Wilson intervals, and `shortfalls.md` names the sample size as the cause (D-38).", "",
              "| Modality | Stratum | Test reals | Flag |", "|---|---|---|---|"]
    for (modality, stratum), counts in _pivot(real_rows, lambda r: (r["modality"], r.get("real_stratum"))):
        if not any(counts.get(s) for s in FITTED):  # diagnostic or excluded strata never reach `test`
            continue
        test = counts.get("test", 0)
        flag = f"below {MIN_TEST_REALS}" if test < MIN_TEST_REALS else "–"
        lines.append(f"| {modality} | {stratum or '–'} | {test} | {flag} |")
    lines += ["", "## Fake families", ""]
    lines += _table(["Modality", "Threat", "Family", "LGO fold"], fake_rows,
                    lambda r: (r["modality"], r["threat"], r.get("family"), r.get("lgo_fold")), splits)
    lines += ["", "## LGO folds", "", "| Fold | Items |", "|---|---|"]
    lines += [f"| {fold} | {n} |" for fold, n in sorted(lgo.items())]
    lines += ["", "## Sources", ""]
    lines += _table(["Source", "Modality", "Audit", "Commercial use"], rows,
                    lambda r: (r["source_id"], r["modality"], audit.get(r["source_id"], "–"),
                               "yes" if r.get("commercial_ok") else "no"), splits)
    lines += ["", "## Shortcut test (ML §2.1 P3)", "",
              "Standardised logistic regression on (JPEG quality estimate, width, height, has_exif), out-of-fold predictions "
              "under `StratifiedGroupKFold(5, shuffle=True, random_state=7)` over the image items in fit ∪ cal ∪ test. A "
              "lossless file's quality is encoded as 100. The gate applies to `norm`; `native` is a diagnostic, because "
              "metadata and container signals are evaluated on native bytes and are expected to differ by source.", "",
              "| Variant | Real | Fake | AUC | Gate (≤ 0.65) | Coefficients (standardised: q, width, height, has_exif) |",
              "|---|---|---|---|---|---|"]
    for result in shortcut:
        gate = ("pass" if result.passed else "FAIL") if result.variant == "norm" else "diagnostic"
        coefs = ", ".join(f"{v:+.3f}" for v in result.coefficients.values())
        lines.append(f"| `{result.variant}` | {result.n_real} | {result.n_fake} | {result.auc:.4f} | {gate} | {coefs} |")
    return "\n".join(lines) + "\n"
