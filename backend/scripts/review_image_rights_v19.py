"""Separate, model-free photo-license/count reconstruction; no upstream code."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import zipfile
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse

ROLES = ("training", "validation", "testing")
STEMS = {"original": "metadata/orig", "firefly": "metadata/ps", "sd2": "metadata/sd2", "flux-fill": "metadata-flux/flux1filldev", "flux-original": "metadata-flux/orig-flux"}


def must(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def status(identifier, table):
    if type(identifier) is not int or identifier not in table:
        return "unknown"
    if identifier not in range(1, 7):
        return "additional_rights_review"
    license_slug = {1: "by-nc-sa", 2: "by-nc", 3: "by-nc-nd", 4: "by", 5: "by-sa", 6: "by-nd"}[identifier]
    raw = table[identifier].get("url", "")
    parsed = urlparse(raw)
    if (parsed.scheme not in {"http", "https"} or parsed.netloc != "creativecommons.org"
            or parsed.path != "/licenses/" + license_slug + "/2.0/" or parsed.params or parsed.query or parsed.fragment
            or raw != parsed.scheme + "://creativecommons.org" + parsed.path):
        return "unknown"
    if identifier <= 3:
        return "excluded_noncommercial"
    return {4: "metadata_by", 5: "metadata_by_sa", 6: "excluded_no_derivatives"}[identifier]


def counts(ids, images, table):
    tally = Counter(status(images[i].get("license"), table) if i in images else "missing_coco_record" for i in ids)
    return {
        "sources": len(ids), "license_status": dict(sorted(tally.items())),
        "preliminary_metadata_compatible": tally["metadata_by"] + tally["metadata_by_sa"],
        "training_admitted": 0, "attribution_unverified": len(ids), "source_bytes_lineage_unverified": len(ids),
    }


def review(source, summary_path, output, private=None, root=None):
    summary = json.loads(summary_path.read_text())
    must(summary["schema"] == "chipmunk.image-source-rights-summary/19", "Expected public summary schema")
    for name, expected in summary["input_sha256"].items():
        must(digest(source / name) == expected, "Public input hash mismatch")
    archive_path = source / "annotations_trainval2017.zip"
    must(archive_path.stat().st_size == 252907541, "Archive byte count mismatch")
    with archive_path.open("rb") as stream:
        must(hashlib.file_digest(stream, "md5").hexdigest() == "f4bbac642086de4f52a3fdda2de5fa2c", "Archive MD5 mismatch")
    groups, csv_counts = {}, {}
    for family, stem in STEMS.items():
        groups[family] = {}
        for role in ROLES:
            name = stem + "_" + role + ".csv"
            with (source / name).open(newline="") as stream:
                reader = csv.reader(stream)
                header = next(reader)
                index = header.index("image_id")
                values = [int(row[index]) for row in reader]
            must(values and min(values) > 0, "Invalid image identifier")
            groups[family][role] = set(values)
            csv_counts[name] = {"rows": len(values), "unique_source_ids": len(set(values))}
    must(csv_counts == summary["csv_counts"], "CSV denominator differs")
    wanted = set().union(*(ids for arms in groups.values() for ids in arms.values()))
    private_doc, consumed, local_hashes = None, set(), {}
    if private is not None:
        must(root is not None, "Repository root required for private source review")
        private_doc = json.loads((private / "private-sources.json").read_text())
        bindings = json.loads((private / "bindings.json").read_text())
        must(bindings["summary_sha256"] == digest(summary_path), "Private summary binding differs")
        must(bindings["private_sources_sha256"] == digest(private / "private-sources.json"), "Private source binding differs")
        actual_manifests = {str(p.relative_to(root)): digest(p) for p in sorted((root / "backend/data/localization-eval").glob("tgif-*/manifest.json"))}
        must(actual_manifests == bindings["consumed_manifest_sha256"] == private_doc["consumed_manifest_sha256"], "Complete consumed-source manifest set differs")
        for name in actual_manifests:
            path = root / name
            manifest = json.loads(path.read_text())
            must(manifest["dataset"] == "TGIF and TGIF2", "Unexpected source dataset")
            for item in manifest["items"]:
                identifier = int(item["source_id"])
                consumed.add(identifier)
                if item["label"] == 0:
                    must(digest(path.parent / item["image"]) == item["image_sha256"], "Local original digest differs")
                    local_hashes.setdefault(identifier, set()).add(item["image_sha256"])
        wanted |= consumed
    images, table, member_proofs = {}, None, {}
    with zipfile.ZipFile(archive_path) as archive:
        for role in ("train", "val"):
            name = "annotations/captions_" + role + "2017.json"
            must(archive.namelist().count(name) == 1, "Ambiguous caption member")
            info = archive.getinfo(name)
            must(0 < info.file_size < 300_000_000 and not info.flag_bits & 1, "Caption safety bounds differ")
            raw = archive.read(name)
            member_proofs[name] = {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(), "zip_crc32": info.CRC}
            doc = json.loads(raw)
            licenses = {r["id"]: r for r in doc["licenses"]}
            must(len(licenses) == len(doc["licenses"]), "Duplicate license entry")
            must(table is None or table == licenses, "Split-specific license conflict")
            table = licenses
            source_ids = [r["id"] for r in doc["images"]]
            must(len(source_ids) == len(set(source_ids)), "Duplicate COCO source identifier")
            member_proofs[name]["image_count"] = len(source_ids)
            for image in doc["images"]:
                if image["id"] in wanted:
                    must(image["id"] not in images, "Repeated source in COCO train/val")
                    images[image["id"]] = dict(image, coco_split=name)
            del raw, doc
    must(member_proofs == summary["coco_members"], "Caption member proofs differ")
    must([table[k] for k in sorted(table)] == summary["license_table"], "Public license table differs")
    must(counts(wanted, images, table) == summary["all_sources"], "All-source license counts differ")
    for role in ROLES:
        must(counts(groups["original"][role], images, table) == summary["publisher_base_roles"][role], "Role license counts differ")
    overlap = {}
    for family, arms in groups.items():
        overlap[family] = {}
        for left, right in (("training", "validation"), ("training", "testing"), ("validation", "testing")):
            overlap[family][left + "/" + right] = sum(i in arms[right] for i in arms[left])
    must(overlap == summary["publisher_role_intersections"], "Split-overlap counts differ")
    mismatches = {}
    for family, arms in groups.items():
        if family != "original":
            mismatches[family] = {role: {
                "missing_from_base_original": sum(i not in arms[role] for i in groups["original"][role]),
                "outside_base_original": sum(i not in groups["original"][role] for i in arms[role]),
            } for role in ROLES}
    must(mismatches == summary["family_vs_base_role_counts"], "Family/base source differences differ")
    must(summary["missing_coco_records"] == sum(i not in images for i in wanted), "Missing sources were omitted")
    must(summary["training_eligible_after_full_review"] == 0 and summary["new_neural_forwards"] == 0 and summary["fit_or_confidence_change"] is False, "Metadata is not training admission or model evidence")
    if private_doc is not None:
        must(counts(consumed, images, table) == summary["consumed_local_sources"], "Consumed-source denominators differ")
        variants = dict(sorted(Counter(len(v) for v in local_hashes.values()).items()))
        must({str(k): v for k, v in variants.items()} == summary["source_original_digest_variants"], "Original variant counts differ")
        expected = []
        for identifier in sorted(wanted):
            image = images.get(identifier)
            expected.append({
                "source_id": identifier, "coco_image": image,
                "recorded_license": table.get(image["license"]) if image else None,
                "license_status": status(image.get("license"), table) if image else "missing_coco_record",
                "publisher_roles": {family: [role for role in ROLES if identifier in arms[role]] for family, arms in groups.items()},
                "consumed_local_evaluation": identifier in consumed,
                "local_original_sha256": sorted(local_hashes.get(identifier, ())),
                "attribution_verified": False, "source_bytes_lineage_verified": False, "training_admitted": False,
            })
        must(expected == private_doc["sources"], "Full private source reconstruction differs")
    result = {
        "schema": "chipmunk.image-source-rights-review/19", "public_metadata_counts_match": True,
        "source_records_reconstructed": len(wanted), "private_consumed_inventory_assessed": private_doc is not None,
        "private_consumed_source_count": len(consumed) if private_doc is not None else None,
        "private_source_rows_match": True if private_doc is not None else None,
        "freshness_or_training_permission_asserted": False, "outside_human_review_claimed": False,
        "summary_sha256": digest(summary_path), "review_program_sha256": digest(__file__),
        "limitations": "Separate local arithmetic/source reconstruction, not an outside human review or per-photo attribution/lineage clearance.",
    }
    with output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("summary", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--private", type=Path)
    parser.add_argument("--root", type=Path)
    args = parser.parse_args()
    review(args.source, args.summary, args.output, args.private, args.root)
