"""Pin an isolated norm/none score-cache subset for a serving-contract smoke.

No rows are repaired, imputed or selected by score/status. Missing rows remain
missing; failed and not-applicable rows remain available to calibration policy.
A failed snapshot may leave its exclusive output directory for inspection, but
never returns a successful pin record and must not be consumed.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

_KEY = re.compile(r"[a-z0-9][a-z0-9._-]*@[0-9a-z]+")
_MARKER = "v2.image_fused@"
_REQUIRED = {"item_id", "variant", "transform", "status", "reason", "prob", "logit", "aux", "runtime_ms", "scored_at"}


def _sha256(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"score cache input must be a regular non-symlink file: {path.name}")
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _markers(scores_dir: Path) -> list[str]:
    names = sorted(path.name for path in scores_dir.glob(f"{_MARKER}*.parquet"))
    for name in names:
        if not _KEY.fullmatch(name.removesuffix(".parquet")):
            raise ValueError(f"invalid image marker filename: {name}")
    return names


def _read_selected(path: Path, item_ids: list[str]):
    import pyarrow as pa
    import pyarrow.parquet as pq

    schema = pq.read_schema(path)
    if not _REQUIRED.issubset(schema.names) or len(set(schema.names)) != len(schema.names):
        raise ValueError(f"malformed score cache schema: {path.name}")
    for name in ("item_id", "variant", "transform", "status", "aux", "scored_at"):
        if schema.field(name).type != pa.string():
            raise ValueError(f"malformed score cache {name} type: {path.name}")
    total = pq.read_metadata(path).num_rows
    table = pq.read_table(path, filters=[("item_id", "in", item_ids), ("variant", "=", "norm"), ("transform", "=", "none")])
    return table.sort_by([("item_id", "ascending")]), total


def _context_ties(marker_rows: Mapping[str, list[dict]]) -> dict[str, Any]:
    # Deliberately match ScoredContexts' effective context and lexical timestamp
    # semantics, including its separate slice-id lookup. Do not invent a newer
    # timestamp parser or include fields its consumer does not use.
    from app.bench.context import _context

    best: dict[str, tuple[str, dict[str, list[str]]]] = {}
    ignored = 0
    for name, rows in marker_rows.items():
        for row in rows:
            try:
                aux = json.loads(row["aux"]) if row["aux"] else {}
            except json.JSONDecodeError:
                ignored += 1  # ScoredContexts ignores malformed JSON too.
                continue
            if not isinstance(aux, Mapping):
                raise ValueError(f"malformed marker auxiliary object: {name}")
            raw = aux.get("context")
            if not isinstance(raw, Mapping):
                ignored += 1
                continue
            try:
                fingerprint = json.dumps({"context": asdict(_context("image", raw)), "slice_id": aux.get("slice_id")},
                                         sort_keys=True, allow_nan=False)
            except (ValueError, TypeError, OverflowError) as exc:
                raise ValueError(f"malformed scored context: {name}") from exc
            item_id, stamp = row["item_id"], row["scored_at"] or ""
            if item_id not in best or stamp > best[item_id][0]:
                best[item_id] = (stamp, {fingerprint: [name]})
            elif stamp == best[item_id][0]:
                best[item_id][1].setdefault(fingerprint, []).append(name)
    conflicts = sorted(item_id for item_id, (_, contexts) in best.items() if len(contexts) > 1)
    if conflicts:
        raise ValueError(f"conflicting equal latest scored contexts for selected items: {', '.join(conflicts)}")
    return {"items_with_context": len(best), "ignored_context_rows": ignored,
            "identical_latest_tie_items": sum(len(next(iter(contexts.values()))) > 1 for _, contexts in best.values()),
            "timestamp_policy": "lexical scored_at; null is empty string (ScoredContexts)",
            "tie_policy": "equal latest effective context and slice_id required"}


def snapshot_cache(scores_dir: Path, out_dir: Path, keys: tuple[str, ...], item_ids: list[str]) -> dict[str, Any]:
    """Snapshot supplied exact member keys plus every image fused marker.

    Return JSON-safe ``source_sha256``/``cache_sha256`` filename maps and a
    ``files`` map with source/selected row, status and missing-item counts. The
    caller validates serving admission; this helper never changes membership.
    Only selected norm/none rows are inspected for duplicate keys/context ties.
    All marker files are emitted, including schema-only empty selected tables.
    """
    import pyarrow.parquet as pq

    scores_dir, out_dir = Path(scores_dir), Path(out_dir)
    if out_dir.exists() or out_dir.is_symlink():
        raise ValueError("snapshot output directory already exists")
    if not scores_dir.is_dir() or scores_dir.is_symlink():
        raise ValueError("score cache directory must be an existing non-symlink directory")
    if (not isinstance(keys, tuple) or not keys or any(not isinstance(k, str) or not _KEY.fullmatch(k)
            or k.startswith(_MARKER) for k in keys) or len(set(keys)) != len(keys)):
        raise ValueError("snapshot needs unique exact member keys")
    if (not isinstance(item_ids, list) or not item_ids or any(not isinstance(i, str) or not i for i in item_ids)
            or len(set(item_ids)) != len(item_ids)):
        raise ValueError("snapshot needs unique nonempty item IDs")
    marker_names = _markers(scores_dir)
    member_names = [f"{key}.parquet" for key in keys]
    names = sorted([*member_names, *marker_names])
    source_sha = {name: _sha256(scores_dir / name) for name in names}
    tables, files, markers = {}, {}, {}
    for name in names:
        path = scores_dir / name
        try:
            table, total = _read_selected(path, item_ids)
        except (OSError, ValueError) as exc:
            raise ValueError(f"cannot read selected score cache rows: {name}: {exc}") from exc
        if _sha256(path) != source_sha[name]:
            raise ValueError(f"source score cache changed while reading: {name}")
        rows = table.to_pylist()
        counts = Counter(row["item_id"] for row in rows)
        if any(count > 1 for count in counts.values()):
            raise ValueError(f"duplicate selected norm/none rows: {name}")
        if any(not isinstance(row["status"], str) or not row["status"] for row in rows):
            raise ValueError(f"missing selected score status: {name}")
        tables[name] = table
        files[name] = {"kind": "marker" if name in marker_names else "member", "source_rows": total,
                       "selected_rows": len(rows), "status_counts": dict(sorted(Counter(row["status"] for row in rows).items())),
                       "missing_item_ids": [i for i in item_ids if i not in counts]}
        if name in marker_names:
            markers[name] = rows
    context = _context_ties(markers)
    if _markers(scores_dir) != marker_names:
        raise ValueError("source image marker inventory changed while reading")
    # Exclusive publication: never overwrite an old smoke, including an empty
    # or failed run directory. Subsequent failures retain partial bytes only.
    try:
        out_dir.mkdir(parents=True, exist_ok=False, mode=0o700)
    except FileExistsError as exc:
        raise ValueError("snapshot output directory already exists") from exc
    cache_sha = {}
    for name, table in tables.items():
        with (out_dir / name).open("xb") as handle:
            pq.write_table(table, handle)
        cache_sha[name] = _sha256(out_dir / name)
    after = {name: _sha256(scores_dir / name) for name in names}
    if after != source_sha or _markers(scores_dir) != marker_names:
        raise ValueError("source score cache changed during snapshot publication")
    return {"schema": "veritas.smoke-cache-snapshot/1", "source_dir": str(scores_dir.resolve()),
            "cache_dir": str(out_dir.resolve()), "keys": list(keys), "item_ids": list(item_ids),
            "filter": {"variant": "norm", "transform": "none"}, "source_sha256": source_sha,
            "source_sha256_after": after, "cache_sha256": cache_sha, "files": files, "contexts": context}
