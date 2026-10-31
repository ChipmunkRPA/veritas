"""`make vbench-smoke` subset (D-13): a fixed 300-item stratified subset of `fit ∪ cal` image items, never `test`.

- **Scope.** Served scope membership is applied before stratification; cached failures never cause replacement.
- **Pool.** Image rows in `fit` or `cal` with `audit_ok` not false and no C2PA manifest (calibrate drops those anyway).
- **Strata.** (split, modality · threat · family · real stratum). `fit` and `cal` keep their shares and their split labels,
  so calibrate fits on the subset's `fit` and calibrates on its `cal`.
- **Allocation.** Proportional with largest remainders (ties by stratum name), at least one item per stratum while the
  total allows it.
- **Order inside a stratum.** sha256("smoke:<item_id>"), so the subset is fixed for a given manifest. Re-splitting the
  bench (B-5, vb-1.0) changes the subset; the run-specific `subset.json` records the manifest sha256 it came from.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Collection, Iterable, Mapping
from pathlib import Path

from app.bench.splits import stratum_key

SMOKE_N = 300


def pool(rows: Iterable[Mapping]) -> list[dict]:
    return [dict(r) for r in rows if r.get("modality") == "image" and r.get("split") in ("fit", "cal")
            and r.get("audit_ok") is not False and not r.get("has_c2pa")]


def allocate_proportional(counts: Mapping[str, int], total: int) -> dict[str, int]:
    """Largest-remainder proportional allocation, capped by availability, ≥ 1 per stratum when total ≥ #strata."""
    available = {k: int(v) for k, v in counts.items() if v > 0}
    n = sum(available.values())
    if total >= n:
        return dict(available)
    keys = sorted(available)
    quota = {k: total * available[k] / n for k in keys}
    alloc = {k: min(available[k], math.floor(quota[k])) for k in keys}
    if total >= len(keys):
        alloc = {k: max(1, v) for k, v in alloc.items()}
    remaining = total - sum(alloc.values())
    by_remainder = sorted(keys, key=lambda k: (-(quota[k] - math.floor(quota[k])), k))
    while remaining > 0:
        progressed = False
        for k in by_remainder:
            if remaining == 0:
                break
            if alloc[k] < available[k]:
                alloc[k] += 1
                remaining -= 1
                progressed = True
        if not progressed:
            break
    while remaining < 0:  # the one-per-stratum floor overshot: take back from the most over-served strata
        k = max((k for k in keys if alloc[k] > 1), key=lambda k: (alloc[k] - quota[k], k))
        alloc[k] -= 1
        remaining += 1
    return alloc


def select(rows: Iterable[Mapping], n: int = SMOKE_N, *, scope_ids: Collection[str] | None = None) -> list[dict]:
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError("smoke size must be a positive integer")
    rows = list(rows)
    ids = [r.get("item_id") for r in rows]
    if any(not isinstance(item, str) or not item for item in ids):
        raise ValueError("manifest needs nonempty string item ids")
    if len(ids) != len(set(ids)):
        raise ValueError("manifest has duplicate item ids")
    candidates = pool(rows)
    if scope_ids is not None:
        allowed = set(scope_ids)
        candidates = [r for r in candidates if r["item_id"] in allowed]
    strata: dict[str, list[dict]] = {}
    for row in candidates:
        strata.setdefault(f"{row['split']} · {stratum_key(row)}", []).append(row)
    alloc = allocate_proportional({k: len(v) for k, v in strata.items()}, n)
    chosen: list[dict] = []
    for key in sorted(strata):
        ordered = sorted(strata[key], key=lambda r: hashlib.sha256(f"smoke:{r['item_id']}".encode()).hexdigest())
        chosen += ordered[:alloc.get(key, 0)]
    return sorted(chosen, key=lambda r: r["item_id"])


def write_subset(source_manifest: Path, out_manifest: Path, record_path: Path, n: int = SMOKE_N, *,
                 scope: Path | None = None, contract: Mapping | None = None) -> dict:
    """Publish a new, pinned subset; existing records and manifests are never replaced.

    Scope filtering precedes allocation. Score availability/outcomes never affect selection.
    Source bytes are read once so hashes describe exactly the parsed metadata.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    source_bytes = source_manifest.read_bytes()
    source_sha = hashlib.sha256(source_bytes).hexdigest()
    table = pq.read_table(pa.BufferReader(source_bytes))
    scope_sha, scope_ids = None, None
    if scope is not None:
        scope_bytes = scope.read_bytes()
        scope_sha = hashlib.sha256(scope_bytes).hexdigest()
        scope_values = pq.read_table(pa.BufferReader(scope_bytes), columns=["item_id"]).column("item_id").to_pylist()
        if any(not isinstance(item, str) or not item for item in scope_values):
            raise ValueError("scope needs nonempty string item ids")
        if len(scope_values) != len(set(scope_values)):
            raise ValueError("scope has duplicate item ids")
        scope_ids = set(scope_values)
        if scope_ids - set(table.column("item_id").to_pylist()):
            raise ValueError("scope contains item ids absent from the source manifest")
    binding = dict(contract or {})
    for key, actual in (("source_manifest_sha256", source_sha), ("scope_sha256", scope_sha)):
        if key in binding and binding[key] != actual:
            raise ValueError(f"{key} mismatch: input changed after contract validation")
    subset = select(table.to_pylist(), n, scope_ids=scope_ids)
    if not subset:
        raise ValueError(f"{source_manifest} has no usable image items in fit or cal within the requested scope")
    output = pa.BufferOutputStream()
    pq.write_table(pa.Table.from_pylist(subset, schema=table.schema), output)
    subset_bytes = output.getvalue().to_pybytes()
    record = {
        "schema": "veritas.vbench-smoke-subset/2",
        "source_manifest_sha256": source_sha,
        "scope_sha256": scope_sha,
        "subset_manifest_sha256": hashlib.sha256(subset_bytes).hexdigest(),
        "contract": binding,
        "requested_n": n,
        "n": len(subset),
        "splits": dict(sorted(Counter(r["split"] for r in subset).items())),
        "labels": dict(sorted(Counter(r["label"] for r in subset).items())),
        "split_labels": {split: dict(sorted(Counter(r["label"] for r in subset if r["split"] == split).items()))
                         for split in ("fit", "cal")},
        "strata": dict(sorted(Counter(f"{r['split']} · {stratum_key(r)}" for r in subset).items())),
        "rule": "scope membership before proportional largest-remainder allocation over "
                "(split, modality · threat · family · real stratum) of fit ∪ cal image items with "
                "audit_ok not false and no C2PA; sha256('smoke:<item_id>') order inside a stratum; "
                "no selection or replacement based on cached score status or value",
        "item_ids": [r["item_id"] for r in subset],
    }
    # Validate serialization before creating either output.
    record_bytes = (json.dumps(record, indent=1, ensure_ascii=False, allow_nan=False) + "\n").encode()
    if out_manifest.resolve() == record_path.resolve():
        raise ValueError("subset manifest and record need distinct output paths")
    for target in (out_manifest, record_path):
        if target.exists() or target.is_symlink():
            raise ValueError(f"smoke output already exists: {target}; use a new run directory")
    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    record_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with out_manifest.open("xb") as handle:
            handle.write(subset_bytes)
        with record_path.open("xb") as handle:
            handle.write(record_bytes)
    except FileExistsError as exc:
        raise ValueError("smoke output already exists; use a new run directory") from exc
    return record
