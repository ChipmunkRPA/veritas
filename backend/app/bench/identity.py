"""Source/content identity checks independent of catalogue group keys.

No detector scores or labels are used to group data. Existing source groups,
identical native hashes, identical normalised bytes, dataset rows, OpenVid source
videos and documented TalkingHeadBench face/driver identities form connected components. Apply before splitting NEW data;
use audit() to diagnose a frozen manifest without modifying it.
"""
from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from pathlib import Path

EVALUATED_SPLITS = frozenset({"fit", "cal", "test", "wild", "diagnostic"})
_OPENVID = re.compile(r"([A-Za-z0-9_-]{11})_\d+_\d+to\d+\.mp4")
_SHA = re.compile(r"[0-9a-f]{64}")
_TALKINGHEAD = re.compile(
    r"fake/(?P<generator>[^/]+)/(?:train|val|test)/(?P<face>[0-9]+)--"
    r"(?P<video>[A-Za-z0-9_-]{11})_(?P<segment>[0-9]+)--(?P=generator)\.mp4"
)


def talkinghead_lineage(row: dict) -> dict[str, str] | None:
    """Parse only the catalogue's documented FFHQ--driver--generator layout.

    The driver carries a case-sensitive YouTube source ID plus a segment index.
    The additional MAGI-1 and Hallo3 filename layouts have not been established
    by the catalogue; do not infer their missing or differently encoded ancestry.
    """
    if row.get("source_id") != "talkingheadbench":
        return None
    match = _TALKINGHEAD.fullmatch(str(row.get("source_row_key", "")))
    if match is None or (row.get("generator") is not None and row["generator"] != match["generator"]):
        return None
    face, segment = (match[k].lstrip("0") or "0" for k in ("face", "segment"))
    return {"ffhq_source_image": face, "talkinghead_driver_clip": f"{match['video']}_{segment}",
            "youtube_source_video": match["video"]}


def source_video_id(row: dict) -> str | None:
    if row.get("source_id") != "openvid-1m":
        return None
    match = _OPENVID.fullmatch(str(row.get("source_row_key", "")).rsplit("/", 1)[-1])
    return match[1] if match else None  # YouTube IDs are case sensitive.


def identities(row: dict) -> list[tuple[str, str]]:
    keys = [("group_key", str(row["group_key"]))]
    item_id = str(row["item_id"])
    if _SHA.fullmatch(item_id):
        keys.append(("native_sha256", item_id))  # fetcher defines item_id as SHA256(native bytes)
    if digest := row.get("norm_sha256"):
        if not _SHA.fullmatch(digest):
            raise ValueError(f"invalid normalised SHA256 for {item_id}")
        keys.append(("norm_sha256", digest))
    if row.get("source_dataset") and row.get("source_row_key"):
        keys.append(("dataset_row", f"{row['source_dataset']}\0{row['source_row_key']}"))
    if video_id := source_video_id(row):
        keys.append(("openvid_source_video", video_id))  # retained for existing audit consumers
        keys.append(("youtube_source_video", video_id))
    if lineage := talkinghead_lineage(row):
        keys.extend(lineage.items())
    return keys


def components(rows: list[dict]) -> list[list[int]]:
    """Deterministic connected groups, including transitive source/content links."""
    parent = list(range(len(rows)))

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    first: dict[tuple[str, str], int] = {}
    for i, row in enumerate(rows):
        for key in identities(row):
            if key in first:
                a, b = root(i), root(first[key])
                parent[max(a, b)] = min(a, b)
            else:
                first[key] = i
    grouped: dict[int, list[int]] = defaultdict(list)
    for i in range(len(rows)):
        grouped[root(i)].append(i)
    return sorted(grouped.values(), key=lambda inds: min(rows[i]["item_id"] for i in inds))


def grouped_rows(rows: list[dict]) -> list[dict]:
    """Copy rows and coalesce connected source groups before a new split assignment."""
    out = [dict(row) for row in rows]
    for indices in components(rows):
        groups = sorted({rows[i]["group_key"] for i in indices})
        if len(groups) > 1:
            digest = hashlib.sha256("\0".join(groups).encode()).hexdigest()
            for i in indices:
                out[i]["group_key"] = f"identity:{digest}"
    return out


def hash_normalised(rows: list[dict], data_dir: Path) -> list[dict]:
    """Attach hashes of available normalised paths; fail closed for missing/escaped files.

    Streams local bytes only. No file or manifest is modified. Native content is
    identified by the fetcher's existing item_id; this is not a native-file integrity audit.
    """
    base = data_dir.resolve()
    out = []
    cache: dict[Path, str] = {}
    for row in rows:
        result = dict(row)
        if relative := row.get("norm_path"):
            path = (base / relative).resolve()
            if not path.is_relative_to(base):
                raise ValueError(f"normalised path escapes data directory: {relative}")
            if path not in cache:
                with path.open("rb") as stream:
                    cache[path] = hashlib.file_digest(stream, "sha256").hexdigest()
            result["norm_sha256"] = cache[path]
        out.append(result)
    return out


def audit(rows: list[dict]) -> dict:
    """Identity leakage and exact-content label conflicts; never touches media/scores."""
    crosses = []
    conflicts = []
    for indices in components(rows):
        members = [rows[i] for i in indices]
        splits = sorted({r.get("split") for r in members if r.get("split") in EVALUATED_SPLITS})
        by_key: dict[tuple[str, str], list[dict]] = defaultdict(list)
        for row in members:
            for key in identities(row):
                by_key[key].append(row)
        shared = [{"kind": k[0], "value": k[1]} for k, v in sorted(by_key.items()) if len(v) > 1]
        if len(splits) > 1:
            crosses.append({"splits": splits, "links": shared,
                            "items": [{k: r.get(k) for k in ("item_id", "source_id", "split", "group_key")}
                                      for r in sorted(members, key=lambda r: r["item_id"])]})
        for key, items in sorted(by_key.items()):
            if key[0] in {"native_sha256", "norm_sha256"} and len({r.get("label") for r in items}) > 1:
                conflicts.append({"kind": key[0], "sha256": key[1],
                                  "item_ids": sorted(r["item_id"] for r in items)})
    unknown = sorted(r["item_id"] for r in rows if r.get("source_id") == "openvid-1m" and source_video_id(r) is None)
    unknown_talkinghead = sorted(r["item_id"] for r in rows
                                if r.get("source_id") == "talkingheadbench" and talkinghead_lineage(r) is None)
    return {"schema_version": 1, "items": len(rows), "components": len(components(rows)),
            "normalised_paths": sum(bool(r.get("norm_path")) for r in rows),
            "normalised_hashes": sum(bool(r.get("norm_sha256")) for r in rows),
            "cross_split_components": crosses, "exact_content_label_conflicts": conflicts,
            "unrecognised_openvid_items": unknown,
            "unrecognised_talkinghead_items": unknown_talkinghead,
            # Match the existing OpenVid rule: unknown lineage remains blocking,
            # including excluded rows. Never silently promote a partial audit.
            "passed": not (crosses or conflicts or unknown or unknown_talkinghead),
            "limitations": ["No perceptual duplicate or general speaker/person identity matching.",
                            "Native hashes rely on manifest item_id; native bytes are not rehashed.",
                            "Prompt text is not retained in this manifest; cross-source prompt matches are not inferred.",
                            "TalkingHeadBench lineage is limited to documented FFHQ--driver--generator filenames; "
                            "other formats are reported and fail closed."]}
