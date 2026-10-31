"""`make vbench-core` (B-4c, D-39): the fixed, seeded stratified subset `vb-1.0-core` of vb-1.0.

- **Pool.** Rows with role `core` in `fit`, `cal` or `test` and `audit_ok` not false. In-the-wild, diagnostic and excluded
  rows stay out: no goal set reads them.
- **Images ≤ 2,000.** First the robustness images: 150 real and 150 fake of the vb-1.0 robustness subset
  (`perturb.robustness_subset`, whose perturbed copies and native scores are cached), spread evenly over their strata.
  Then the remaining budget over (modality · threat · family · real stratum) by largest remainders with at least one item
  per stratum.
- **Video ≤ 150:** 75 fake spread evenly over the families (vendors) and 75 real over the real strata. **Audio ≤ 300:**
  150 real and 150 fake, spread the same way.
- **Splits.** Each stratum's quota goes to test 50 %, fit 33 % and cal 17 %, capped by availability, so the core keeps the
  vb-1.0 split labels (calibration fits on `fit`, thresholds on `cal`, goals on `test`).
- **Order inside a (stratum, split) cell.** sha256("core:<item_id>"); the robustness images sha256("core-robustness:<item_id>").
  The same vb-1.0 manifest gives the same core.
- **Robustness on core** uses the transforms jpeg75, social_ig, screenshot_jpeg and resize0.5 only. The core manifest adds a
  `robustness` column, which `perturb.robustness_subset` honours.
"""
from __future__ import annotations

import hashlib
import math
import os
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from app.bench import perturb as pt
from app.bench.smoke import allocate_proportional
from app.bench.splits import stratum_key

CORE_BENCH = "vb-1.0-core"
SOURCE_BENCH = "vb-1.0"
SPLITS = ("test", "fit", "cal")
SPLIT_SHARES: dict[str, float] = {"test": 0.50, "fit": 1 / 3, "cal": 1 / 6}
IMAGE_CAP = 2000
LABEL_TARGETS: dict[str, dict[str, int]] = {"video": {"real": 75, "fake": 75}, "audio": {"real": 150, "fake": 150}}
ROBUSTNESS_TARGETS: dict[str, int] = {"real": 150, "fake": 150}
CORE_TRANSFORMS = ("jpeg75", "social_ig", "screenshot_jpeg", "resize0.5")
MIN_TEST_SHARE = 0.45
FLAG_TEST_REALS = 360


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pool(rows: Iterable[Mapping]) -> list[dict]:
    return [dict(r) for r in rows if (r.get("role") or "core") == "core" and r.get("split") in SPLITS and r.get("audit_ok") is not False]


def label_stratum(row: Mapping) -> str:
    return str((row.get("real_stratum") if row.get("label") == "real" else row.get("family")) or "unknown")


def _ordered(rows: Iterable[dict], salt: str) -> list[dict]:
    return sorted(rows, key=lambda r: hashlib.sha256(f"{salt}:{r['item_id']}".encode()).hexdigest())


def split_quota(available: Mapping[str, int], n: int) -> dict[str, int]:
    """`n` items over the splits by `SPLIT_SHARES` (largest remainders, test first on ties), capped by availability; a split
    without room passes its share to the others."""
    n = min(n, sum(available.get(s, 0) for s in SPLITS))
    quota = {s: n * SPLIT_SHARES[s] for s in SPLITS}
    alloc = {s: min(available.get(s, 0), math.floor(quota[s])) for s in SPLITS}
    order = sorted(SPLITS, key=lambda s: (-(quota[s] - math.floor(quota[s])), SPLITS.index(s)))
    remaining = n - sum(alloc.values())
    while remaining > 0:
        progressed = False
        for s in order:
            if remaining and alloc[s] < available.get(s, 0):
                alloc[s] += 1
                remaining -= 1
                progressed = True
        if not progressed:
            break
    return alloc


def _spread(groups: Mapping[str, list[dict]], quotas: Mapping[str, int], salt: str = "core") -> list[dict]:
    chosen: list[dict] = []
    for key in sorted(groups):
        by_split: dict[str, list[dict]] = defaultdict(list)
        for row in groups[key]:
            by_split[row["split"]].append(row)
        take = split_quota({s: len(v) for s, v in by_split.items()}, quotas.get(key, 0))
        for s in SPLITS:
            chosen += _ordered(by_split.get(s, []), salt)[:take[s]]
    return chosen


def robustness_items(rows: Sequence[Mapping], targets: Mapping[str, int] = ROBUSTNESS_TARGETS) -> list[dict]:
    """Images of the vb-1.0 robustness subset that are in the pool: `targets` per label, spread evenly over strata."""
    eligible = {r["item_id"] for r in pool(rows)}
    groups: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for row in pt.robustness_subset([dict(r) for r in rows]):
        if row["modality"] == "image" and row["item_id"] in eligible:
            groups[row["label"]][label_stratum(row)].append(row)
    chosen: list[dict] = []
    for label, target in targets.items():
        strata = groups.get(label, {})
        alloc = pt.allocate({k: len(v) for k, v in strata.items()}, target)
        for key in sorted(strata):
            chosen += _ordered(strata[key], "core-robustness")[:alloc[key]]
    return chosen


def select(rows: Iterable[Mapping], *, image_cap: int = IMAGE_CAP, label_targets: Mapping[str, Mapping[str, int]] = LABEL_TARGETS,
           robustness_targets: Mapping[str, int] = ROBUSTNESS_TARGETS) -> list[dict]:
    """The core rows (item-id order), each with `robustness` set."""
    rows = [{k: v for k, v in r.items() if k != "robustness"} for r in rows]
    candidates = pool(rows)
    robust = robustness_items(rows, robustness_targets)
    robust_ids = {r["item_id"] for r in robust}
    chosen = [{**r, "robustness": True} for r in robust]

    images: dict[str, list[dict]] = defaultdict(list)
    for row in candidates:
        if row["modality"] == "image" and row["item_id"] not in robust_ids:
            images[stratum_key(row)].append(row)
    quotas = allocate_proportional({k: len(v) for k, v in images.items()}, max(0, image_cap - len(robust)))
    chosen += [{**r, "robustness": False} for r in _spread(images, quotas)]

    for modality, targets in label_targets.items():
        for label, target in targets.items():
            groups: dict[str, list[dict]] = defaultdict(list)
            for row in candidates:
                if row["modality"] == modality and row["label"] == label:
                    groups[label_stratum(row)].append(row)
            quotas = pt.allocate({k: len(v) for k, v in groups.items()}, target)
            chosen += [{**r, "robustness": False} for r in _spread(groups, quotas)]
    return sorted(chosen, key=lambda r: r["item_id"])


@dataclass(frozen=True)
class Built:
    rows: list[dict]
    source_rows: list[dict]
    manifest_sha256: str
    source_manifest_sha256: str


def write_core(source_manifest: Path, out_manifest: Path, **caps) -> Built:
    """Write the core manifest (the source schema plus `robustness`) atomically."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pq.read_table(source_manifest)
    if "split" not in table.column_names or not any(table.column("split").to_pylist()):
        raise ValueError(f"{Path(source_manifest).name} has no splits: run `make vbench-split` first")
    source_rows = table.to_pylist()
    rows = select(source_rows, **caps)
    if not rows:
        raise ValueError(f"{Path(source_manifest).name} has no usable items in fit, cal or test")
    fields = [f for f in table.schema if f.name != "robustness"]
    schema = pa.schema([*fields, pa.field("robustness", pa.bool_())], metadata=table.schema.metadata)
    out_manifest = Path(out_manifest)
    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_manifest.with_name(out_manifest.name + ".part")
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), tmp)
    os.replace(tmp, out_manifest)
    return Built(rows=rows, source_rows=source_rows, manifest_sha256=sha256_file(out_manifest),
                 source_manifest_sha256=sha256_file(source_manifest))


# --- checks and summary ------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Check:
    name: str
    ok: bool
    detail: str


def checks(rows: Sequence[Mapping], source_rows: Sequence[Mapping], *, image_cap: int = IMAGE_CAP,
           label_targets: Mapping[str, Mapping[str, int]] = LABEL_TARGETS, robustness_targets: Mapping[str, int] = ROBUSTNESS_TARGETS) -> list[Check]:
    by_modality = Counter(r["modality"] for r in rows)
    out = [Check(f"images ≤ {image_cap}", by_modality["image"] <= image_cap, f"{by_modality['image']} images")]
    for modality, targets in label_targets.items():
        labels = Counter(r["label"] for r in rows if r["modality"] == modality)
        out.append(Check(f"{modality} ≤ {sum(targets.values())} ({' / '.join(f'{t} {k}' for k, t in targets.items())})",
                         all(labels[k] <= t for k, t in targets.items()) and by_modality[modality] <= sum(targets.values()),
                         ", ".join(f"{labels[k]} {k}" for k in targets)))
    images = [r for r in rows if r["modality"] == "image"]
    share = sum(r["split"] == "test" for r in images) / len(images) if images else 0.0
    out.append(Check(f"image test share ≥ {MIN_TEST_SHARE * 100:.0f} %", share >= MIN_TEST_SHARE, f"{share * 100:.1f} %"))
    expected = {(r["modality"], stratum_key(r)) for r in pool(source_rows)}
    missing = sorted(expected - {(r["modality"], stratum_key(r)) for r in rows})
    out.append(Check("every real stratum and fake family represented", not missing,
                     f"{len(expected)} strata" if not missing else "missing: " + "; ".join(k for _, k in missing)))
    robust = [r for r in rows if r.get("robustness")]
    labels = Counter(r["label"] for r in robust)
    out.append(Check(f"robustness ≤ {sum(robustness_targets.values())} images", all(r["modality"] == "image" for r in robust)
                     and all(labels[k] <= t for k, t in robustness_targets.items()), f"{labels['real']} real, {labels['fake']} fake"))
    ids = [r["item_id"] for r in rows]
    out.append(Check("no duplicate items", len(ids) == len(set(ids)), f"{len(ids)} items"))
    return out


def _table(header: Sequence[str], body: Iterable[Sequence[object]]) -> list[str]:
    return ["| " + " | ".join(header) + " |", "|" + "---|" * len(header), *("| " + " | ".join(str(c) for c in row) + " |" for row in body)]


def summary_markdown(built: Built, results: Sequence[Check], *, generated: str, source_label: str, core_label: str) -> str:
    rows, source_rows = built.rows, built.source_rows
    lines = [
        f"# {CORE_BENCH} summary", "",
        f"Generated {generated} by `backend/scripts/vbench_core.py` from `{source_label}` (B-4c, D-39).", "",
        f"- **Source:** {SOURCE_BENCH}, {len(source_rows)} items, sha256 `{built.source_manifest_sha256}`.",
        f"- **Core manifest:** `{core_label}`, {len(rows)} items, sha256 `{built.manifest_sha256}`, pinned in "
        f"`docs/v2/bench/goals.yaml` as `bench: {CORE_BENCH}`.",
        "- **Pool:** rows with role `core` in `fit`, `cal` or `test` and `audit_ok` not false; in-the-wild, diagnostic and "
        "excluded rows are left out (no goal set reads them).",
        f"- **Images ≤ {IMAGE_CAP}:** {sum(ROBUSTNESS_TARGETS.values())} robustness images first (150 real, 150 fake of the vb-1.0 "
        "robustness subset, even over strata), then the rest by largest remainders over (modality · threat · family · real "
        "stratum) with at least one item per stratum.",
        "- **Video ≤ 150:** 75 fake even over families (vendors), 75 real over the real strata. **Audio ≤ 300:** 150 real, "
        "150 fake, even over strata.",
        "- **Splits:** each stratum's quota goes to test 50 %, fit 33 %, cal 17 %, capped by availability.",
        "- **Order:** sha256(`core:<item_id>`) inside a (stratum, split) cell, sha256(`core-robustness:<item_id>`) for the "
        "robustness images. The same vb-1.0 manifest gives the same core.",
        f"- **Robustness transforms on core:** {', '.join(CORE_TRANSFORMS)} (D-39). The other vb-1.0 transforms are not measured "
        "on core.",
        "- **Records:** every gate record measured on core states it and reports confidence intervals (D-39). The counts are "
        "small, so the intervals are wide.", "",
        "## Checks", "", *_table(("Check", "Result", "Detail"), ((c.name, "PASS" if c.ok else "FAIL", c.detail) for c in results)), "",
        "## Modalities", "",
    ]
    modality_rows = []
    for modality in ("image", "video", "audio"):
        items = [r for r in rows if r["modality"] == modality]
        if not items:
            continue
        splits, labels = Counter(r["split"] for r in items), Counter(r["label"] for r in items)
        modality_rows.append((modality, len(items), labels["real"], labels["fake"], splits["fit"], splits["cal"], splits["test"],
                              f"{splits['test'] / len(items) * 100:.1f} %"))
    lines += [*_table(("Modality", "Items", "Real", "Fake", "fit", "cal", "test", "Test share"), modality_rows), ""]

    pooled = Counter((r["modality"], r["label"], r.get("threat") or "–", label_stratum(r)) for r in pool(source_rows))
    cells = Counter((r["modality"], r["label"], r.get("threat") or "–", label_stratum(r), r["split"]) for r in rows)
    strata_rows = []
    for key in sorted(pooled):
        counts = [cells[(*key, s)] for s in ("fit", "cal", "test")]
        strata_rows.append((*key, pooled[key], *counts, sum(counts)))
    lines += ["## Strata", "", "Every stratum of the pool with its pool size and the core counts per split.", "",
              *_table(("Modality", "Label", "Threat", "Stratum", "Pool", "fit", "cal", "test", "Core"), strata_rows), ""]

    robust = [r for r in rows if r.get("robustness")]
    rsplits = Counter((r["label"], r["split"]) for r in robust)
    lines += ["## Robustness subset", "",
              f"Drawn from the vb-1.0 robustness subset ({sum(1 for r in pt.robustness_subset(source_rows) if r['modality'] == 'image')} "
              f"images), whose perturbed copies and native scores are cached. Transforms on core: {', '.join(CORE_TRANSFORMS)}.", "",
              *_table(("Label", "fit", "cal", "test", "Total"),
                      ((label, rsplits[(label, "fit")], rsplits[(label, "cal")], rsplits[(label, "test")],
                        sum(rsplits[(label, s)] for s in ("fit", "cal", "test"))) for label in ("real", "fake"))), ""]

    reals = Counter((r["modality"], label_stratum(r)) for r in rows if r["label"] == "real" and r["split"] == "test")
    lines += ["## Test reals per stratum", "",
              f"Strata with fewer than {FLAG_TEST_REALS} `test` reals are flagged: the gate reports their false-positive rates with "
              "Wilson intervals, and `shortfalls.md` names the sample size as the cause (D-38).", "",
              *_table(("Modality", "Stratum", "Test reals", "Flag"),
                      ((m, s, n, "below 360" if n < FLAG_TEST_REALS else "–") for (m, s), n in sorted(reals.items()))), ""]
    return "\n".join(lines)


def pin_goals(path: Path, *, manifest_sha256: str, source_manifest_sha256: str, date: str) -> None:
    """Point `goals.yaml` at the core: `bench`, `manifest_sha256` and `source_manifest_sha256` (the vb-1.0 manifest)."""
    text = Path(path).read_text(encoding="utf-8")
    pin = f"manifest_sha256: {manifest_sha256}  # {CORE_BENCH} built {date} (B-4c, D-39)"
    source = f"source_manifest_sha256: {source_manifest_sha256}  # {SOURCE_BENCH}, the manifest the core is drawn from"
    text, n_bench = re.subn(r"(?m)^bench: .*$", f"bench: {CORE_BENCH}", text, count=1)
    text, n_pin = re.subn(r"(?m)^manifest_sha256: .*$", pin, text, count=1)
    if not (n_bench and n_pin):
        raise ValueError(f"{Path(path).name} has no top-level `bench` and `manifest_sha256` lines to pin")
    if re.search(r"(?m)^source_manifest_sha256: ", text):
        text = re.sub(r"(?m)^source_manifest_sha256: .*$", source, text, count=1)
    else:
        text = text.replace(pin, f"{pin}\n{source}", 1)
    Path(path).write_text(text, encoding="utf-8")
