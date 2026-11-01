"""Prepare a disjoint, fixed-order OpenSDI holdout without running inference.

`plan` writes the evaluation policy before reading remote metadata/key columns.
`prepare` then reads original embedded image/mask bytes from those pinned shards.
Existing manifests and source files are never overwritten.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.localization_benchmark import REPO, REVISION, ROOT, SHARDS, policy, sha, source  # noqa: E402

SCHEMA = "chipmunk.localization-holdout-plan/1"


def write_new(path: Path, value: dict) -> None:
    with path.open("x") as output:
        output.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def exclusions(paths: list[Path]) -> tuple[list[dict], set[str]]:
    records, digests = [], set()
    for path in paths:
        data = path.read_bytes()
        manifest = json.loads(data)
        if (manifest.get("dataset"), manifest.get("revision")) != (REPO, REVISION):
            raise ValueError("Excluded manifests must use the same pinned dataset revision")
        hashes = [item["image_sha256"] for item in manifest["items"]]
        if not hashes or any(not re.fullmatch(r"[a-f0-9]{64}", h) for h in hashes):
            raise ValueError("Invalid exclusion image hashes")
        records.append({"path": str(path.resolve()), "sha256": sha(data), "items": len(hashes),
                        "groups": dict(sorted(Counter(f"{i['family']}/{i['label']}" for i in manifest["items"]).items()))})
        digests.update(hashes)
    return records, digests


def declared_policy() -> dict:
    return {"schema": "chipmunk.localization-holdout-policy/1", "declared_at": datetime.now(UTC).isoformat(),
            "evaluation_has_started": False, "image_gate_threshold": 0.5, "image_gate_comparison": "greater_or_equal",
            "mask_threshold": 0.5, "mask_comparison": "greater_or_equal", "threshold_tuning": False,
            "report_both": ["raw_mask", "component_filtered", "component_filtered_image_gated"],
            "component_audit": {"connectivity": 8, "minimum_component_area_fraction": 0.001,
                                "minimum_pixels": "ceil(0.001 * original_stored_width * original_stored_height)",
                                "coordinate_space": "original stored pixels", "fill_bounding_boxes": False},
            "production_component_policy": policy(),
            "production_policy_scope": "The existing FUSED audit only reports uncapped native-pixel area-filtered masks and their image-gated version. The complete production grid quantization, reliability mask, component ranking/caps and cross-map packet eligibility are not reproduced by that audit.",
            "input_protocol": "Original image/mask bytes; RGB PIL bilinear 512-square inference; float32 bilinear mask projection to original stored dimensions; no independent EXIF transform.",
            "interpretation": "Disjoint convenience holdout relative to the excluded files, not independent proof against training overlap, a representative accuracy estimate, or product admission."}


def preflight_shard(family: str, label: int, shard: str) -> dict:
    import fsspec
    import pyarrow.parquet as pq

    url = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/data/{shard}.parquet"
    with fsspec.open(url, "rb", block_size=2**20).open() as stream:
        parquet = pq.ParquetFile(stream)
        metadata = parquet.metadata
        group = metadata.row_group(0)
        # Count eligible rows before reading the image/mask payload columns.
        small = parquet.read_row_group(0, columns=["key", "label"]).to_pylist()
        counts = Counter(int(row["label"]) for row in small if str(row["key"]).startswith("partial/"))
        return {"family": family, "label": label, "shard": shard, "url": url, "row_group": 0,
                "file_rows": metadata.num_rows, "file_row_groups": metadata.num_row_groups,
                "row_group_rows": group.num_rows, "partial_rows_by_label": dict(sorted(counts.items())),
                "expected_label_partial_rows": counts[label],
                "row_group_compressed_column_bytes": sum(group.column(i).total_compressed_size for i in range(group.num_columns)),
                "row_group_uncompressed_bytes": group.total_byte_size,
                "count_scope": "Physical eligible rows; unique-byte/exclusion counts are checked when image bytes are read."}


def plan(root: Path, *, offset: int, count: int, exclude_paths: list[Path]) -> dict:
    excluded, digests = exclusions(exclude_paths)
    policy_path = root / "evaluation-policy.json"
    if root.exists():
        # A failed metadata-only request can retry without changing its policy.
        if not root.is_dir() or list(root.iterdir()) != [policy_path]:
            raise FileExistsError("Use a fresh --root; neither an existing plan nor sample data may be overwritten")
        saved, expected = json.loads(policy_path.read_bytes()), declared_policy()
        saved.pop("declared_at", None)
        expected.pop("declared_at")
        if saved != expected:
            raise ValueError("The policy-only retry directory has a different evaluation policy")
    else:
        root.mkdir(parents=True)
        write_new(policy_path, declared_policy())
    print(json.dumps({"excluded_unique_images": len(digests), "existing_manifests": excluded,
                      "policy_written_before_download": str(policy_path)}, indent=2), flush=True)
    shards = []
    for family, names in SHARDS.items():
        for label, shard in zip((1, 0), names, strict=True):
            facts = preflight_shard(family, label, shard)
            print(json.dumps(facts), flush=True)
            if facts["expected_label_partial_rows"] < offset + count:
                raise ValueError(f"Insufficient physical rows for {family}/{label}; no image payloads were requested")
            shards.append(facts)
    document = {"schema": SCHEMA, "dataset": REPO, "revision": REVISION, "offset": offset, "count_per_group": count,
                "expected_samples": len(shards) * count, "excluded_manifests": excluded,
                "excluded_image_sha256": sorted(digests), "evaluation_policy_sha256": sha(policy_path.read_bytes()),
                "selection": "Scan row group zero in physical order; count unique partial image bytes within each family/label; skip the first offset unique rows, then exclude every baseline/previously-selected hash and take the next count. No score-based selection.",
                "shards": shards, "image_payload_columns_read": False,
                "planned_row_group_compressed_column_bytes": sum(s["row_group_compressed_column_bytes"] for s in shards)}
    write_new(root / "download-plan.json", document)
    return document


def select_rows(rows, *, label: int, offset: int, count: int, excluded: set[str], selected: set[str]):
    """Deterministic offset is before exclusion; duplicates never consume its rank."""
    seen, out = set(), []
    unique_index = -1
    for row_index, row in enumerate(rows):
        if not str(row["key"]).startswith("partial/") or int(row["label"]) != label:
            continue
        image = row["image"]["bytes"]
        digest = sha(image)
        if digest in seen:
            continue
        seen.add(digest)
        unique_index += 1
        if unique_index < offset or digest in excluded or digest in selected:
            continue
        mask = (row.get("mask") or {}).get("bytes")
        if label and not mask:
            raise ValueError("Edited holdout item is missing its original reference mask")
        out.append((row_index, unique_index, row, digest))
        selected.add(digest)
        if len(out) == count:
            return out
    raise ValueError(f"Only {len(out)} disjoint samples remain; required {count}. No complete manifest will be written.")


def original_bytes(path: Path, data: bytes) -> None:
    if path.exists():
        if path.read_bytes() != data:
            raise FileExistsError("A prior partial download differs; choose a fresh root")
        return  # An interrupted preparation may reuse identical bytes, never overwrite them.
    with path.open("xb") as output:
        output.write(data)


def prepare(root: Path) -> dict:
    import fsspec
    import pyarrow.parquet as pq

    if (root / "manifest.json").exists():
        raise FileExistsError("The completed holdout already exists; it must not be overwritten")
    plan_bytes = (root / "download-plan.json").read_bytes()
    planned = json.loads(plan_bytes)
    if (planned.get("schema"), planned.get("dataset"), planned.get("revision")) != (SCHEMA, REPO, REVISION):
        raise ValueError("The download plan is not for this pinned holdout")
    if sha((root / "evaluation-policy.json").read_bytes()) != planned["evaluation_policy_sha256"]:
        raise ValueError("The predeclared evaluation policy changed")
    paths = [Path(e["path"]) for e in planned["excluded_manifests"]]
    current, excluded = exclusions(paths)
    if current != planned["excluded_manifests"] or sorted(excluded) != planned["excluded_image_sha256"]:
        raise ValueError("The excluded baseline manifests changed after planning")
    manifest = {"dataset": REPO, "revision": REVISION, "source_url": f"https://huggingface.co/datasets/{REPO}",
                "license": "CC-BY-SA-4.0; authors describe academic use; evaluation only, not redistributed",
                "attribution": "OpenSDI dataset, nebula/OpenSDI_test; pinned source revision and original shard/key retained for each item.",
                "selection": planned["selection"], "offset": planned["offset"], "count_per_group": planned["count_per_group"],
                "download_plan_sha256": sha(plan_bytes), "evaluation_policy_sha256": planned["evaluation_policy_sha256"],
                "excluded_manifests": planned["excluded_manifests"], "items": []}
    selected = set()
    for group in planned["shards"]:
        family, label = group["family"], group["label"]
        expected_url = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/data/{group['shard']}.parquet"
        if group["shard"] != SHARDS[family][0 if label == 1 else 1] or group["url"] != expected_url or group["row_group"] != 0:
            raise ValueError("The planned shard identity changed")
        with fsspec.open(group["url"], "rb", block_size=2**20).open() as stream:
            table = pq.ParquetFile(stream).read_row_group(0, columns=["key", "label", "image", "mask"])
        rows = (batch.to_pylist()[0] for batch in table.to_batches(max_chunksize=1))
        chosen = select_rows(rows, label=label, offset=planned["offset"], count=planned["count_per_group"],
                             excluded=excluded, selected=selected)
        for row_index, rank, row, digest in chosen:
            item_id = f"{family}-{label}-holdout-{rank:04d}-{digest[:10]}"
            mask = (row.get("mask") or {}).get("bytes")
            item = {"id": item_id, "family": family, "key": row["key"], "label": label, "shard": group["shard"],
                    "row_group": 0, "source_row_index": row_index, "partial_unique_index": rank,
                    "image": item_id + ".image", "image_sha256": digest,
                    "mask": item_id + ".mask" if mask else None, "mask_sha256": sha(mask) if mask else None}
            original_bytes(root / item["image"], row["image"]["bytes"])
            if mask:
                original_bytes(root / item["mask"], mask)
            source(root, item)  # Check original-byte hashes, dimensions, binary labels and nonempty edited truth.
            manifest["items"].append(item)
        print(f"Prepared {family}/{label}: {len(chosen)} original-byte disjoint samples", flush=True)
    counts = Counter((i["family"], i["label"]) for i in manifest["items"])
    expected = {(family, label): planned["count_per_group"] for family in SHARDS for label in (0, 1)}
    if counts != expected or selected.intersection(excluded) or len(selected) != len(manifest["items"]):
        raise ValueError("The final holdout is not complete and disjoint")
    write_new(root / "manifest.json", manifest)
    write_new(root / "preparation-verification.json", {"manifest_sha256": sha((root / "manifest.json").read_bytes()),
              "samples": len(selected), "edited": sum(i["label"] == 1 for i in manifest["items"]),
              "authentic": sum(i["label"] == 0 for i in manifest["items"]), "overlap_with_exclusions": 0,
              "baseline_manifests_unchanged": exclusions(paths)[0] == current,
              "all_source_and_mask_hashes_verified": True, "inference_run": False,
              "preparation_script_sha256": sha(Path(__file__).read_bytes())})
    print(f"Completed {len(selected)} unique samples: {root / 'manifest.json'}", flush=True)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "prepare"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--offset", type=int, default=20)
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--exclude-manifest", type=Path, action="append", help="Additional baseline manifest; the original 200-image manifest is always excluded")
    args = parser.parse_args()
    if not 0 <= args.offset <= 10000 or not 1 <= args.count <= 100:
        parser.error("offset must be 0..10000 and count must be 1..100 per family/label")
    if args.action == "plan":
        plan(args.root, offset=args.offset, count=args.count, exclude_paths=[ROOT / "manifest.json", *(args.exclude_manifest or [])])
    else:
        prepare(args.root)
