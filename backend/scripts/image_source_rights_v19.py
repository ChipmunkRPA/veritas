"""Private photo-license/lineage inventory; never a training permission grant."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import socket
import zipfile
from collections import Counter
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[2]
TGIF = "fc5c557bd1a2d49a5dabc3676a54cfc44798f4ea"
COCO_SITE = "5e1c4da72464b1c6f068df0c02c91e3000ea62c4"
ARCHIVE_URL = "https://s3.amazonaws.com/images.cocodataset.org/annotations/annotations_trainval2017.zip"
ARCHIVE_BYTES = 252907541
ARCHIVE_MD5 = "f4bbac642086de4f52a3fdda2de5fa2c"
CSV_PINS = {
    "metadata-flux/flux1filldev_testing.csv": ("47e2b81b194b2ff0695e69c9a7a521916b093f9e", 211849),
    "metadata-flux/flux1filldev_training.csv": ("b5a740965dcc4555d8861be7931eb71d28294adc", 1509758),
    "metadata-flux/flux1filldev_validation.csv": ("988835404c72ec8c5c2eaa69b1e0def601ffd11c", 210850),
    "metadata-flux/orig-flux_testing.csv": ("dea8d4c191ac7cac25f9903edf3d56af18516898", 63069),
    "metadata-flux/orig-flux_training.csv": ("dea8d4c191ac7cac25f9903edf3d56af18516898", 63069),
    "metadata-flux/orig-flux_validation.csv": ("dea8d4c191ac7cac25f9903edf3d56af18516898", 63069),
    "metadata/orig_testing.csv": ("6c88261c1412f11f21a4aee1f68bb13e639b5bb3", 78572),
    "metadata/orig_training.csv": ("8f8b605d9a3bac074407d51436b99ad3ce132e24", 558034),
    "metadata/orig_validation.csv": ("33368c9b4b3b33e53aee02d106b5a08b1acd21d0", 78357),
    "metadata/ps_testing.csv": ("29f26491531db856552709b0e8c7fdcbc87c5161", 217536),
    "metadata/ps_training.csv": ("9b02d535f98347e9992e10a8030d1810b0ff807d", 1532964),
    "metadata/ps_validation.csv": ("7163841ad989a6792efdf3db05d1fd948b4cafd3", 215824),
    "metadata/sd2_testing.csv": ("303b5facf007aa465154e4a6d96a47bafc066a45", 218434),
    "metadata/sd2_training.csv": ("919ba024145d9b798a2abe98036c4b3e65bbd1ff", 1546024),
    "metadata/sd2_validation.csv": ("62bc9970dff19f7d6f62707471ddf6b719a4339b", 216774),
}
SOURCE_PINS = {
    "tgif-README.md": ("IDLabMedia/tgif-dataset", TGIF, "README.md", "c76f48e21f031fc08ef162d4873eb2a7671a03cc"),
    "tgif-LICENSE.txt": ("IDLabMedia/tgif-dataset", TGIF, "LICENSE.txt", "2c66ec6f7ed3cd792822993d91d1f5fdda9c9e69"),
    "tgif-inpaint-loop.py.txt": ("IDLabMedia/tgif-dataset", TGIF, "code/inpaint-loop.py", "73af877dd5394dacf3be20180d2043a4b8726863"),
    "coco-terms.htm": ("cocodataset/cocodataset.github.io", COCO_SITE, "dataset/termsofuse.htm", "70b68cd7e22da5fcc90b0c32455ddf789050e53d"),
    "coco-download.htm": ("cocodataset/cocodataset.github.io", COCO_SITE, "dataset/download.htm", "b75be340244caff18f8f50fdb885d7c5e9e49ddf"),
}
ROLES = ("training", "validation", "testing")
EXPECTED_LICENSES = {
    1: "by-nc-sa", 2: "by-nc", 3: "by-nc-nd", 4: "by", 5: "by-sa", 6: "by-nd",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def blob(data):
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data, usedforsecurity=False).hexdigest()


def write(path, value):
    with Path(path).open("x") as out:
        json.dump(value, out, indent=2, allow_nan=False)
        out.write("\n")


def acquire(folder):
    folder.mkdir(parents=True, exist_ok=False)
    receipt = {"schema": "chipmunk.image-source-rights-acquisition/19", "files": {}, "completed": False}
    try:
        receipt["official_host_dns"] = socket.gethostbyname_ex("images.cocodataset.org")
        for name, (repo, ref, path, expected) in SOURCE_PINS.items():
            url = f"https://raw.githubusercontent.com/{repo}/{ref}/{path}"
            response = requests.get(url, timeout=(15, 45))
            response.raise_for_status()
            require(blob(response.content) == expected, "Pinned primary-source text changed")
            (folder / name).write_bytes(response.content)
            receipt["files"][name] = {"url": url, "git_blob": expected, "bytes": len(response.content), "sha256": sha(folder / name)}
        for path, (expected, size) in CSV_PINS.items():
            url = f"https://raw.githubusercontent.com/IDLabMedia/tgif-dataset/{TGIF}/{path}"
            response = requests.get(url, timeout=(15, 45))
            response.raise_for_status()
            require(len(response.content) == size and blob(response.content) == expected, "Pinned CSV changed")
            target = folder / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(response.content)
            receipt["files"][path] = {"url": url, "git_blob": expected, "bytes": size, "sha256": sha(target)}
        target = folder / "annotations_trainval2017.zip"
        with requests.get(ARCHIVE_URL, timeout=(15, 90), stream=True) as response:
            response.raise_for_status()
            require(int(response.headers["Content-Length"]) == ARCHIVE_BYTES, "Official annotation archive size changed")
            require(response.headers.get("ETag", "").strip('"') == ARCHIVE_MD5, "Official annotation object ETag changed")
            size, md5 = 0, hashlib.md5(usedforsecurity=False)
            with target.open("xb") as stream:
                for chunk in response.iter_content(1024 * 1024):
                    size += len(chunk)
                    require(size <= ARCHIVE_BYTES, "Annotation download exceeds expected size")
                    stream.write(chunk)
                    md5.update(chunk)
        require(size == ARCHIVE_BYTES and md5.hexdigest() == ARCHIVE_MD5, "Annotation archive integrity mismatch")
        receipt["files"][target.name] = {
            "url": ARCHIVE_URL, "bytes": size, "md5": md5.hexdigest(), "sha256": sha(target),
            "etag": response.headers.get("ETag"), "last_modified": response.headers.get("Last-Modified"),
            "identity_limit": "Size and object ETag observed over HTTPS; not a publisher-signed SHA256 assertion.",
        }
        receipt["completed"] = True
    except BaseException as error:
        receipt["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        receipt["program_sha256"] = sha(__file__)
        write(folder / "acquisition.json", receipt)
    print(json.dumps({"completed": True, "files": len(receipt["files"]), "archive_sha256": sha(target)}), flush=True)


def license_status(license_id, licenses):
    """Only exact recorded BY/BY-SA 2.0 schemas pass this preliminary filter."""
    entry = licenses.get(license_id)
    if not entry or type(license_id) is not int:
        return "unknown"
    expected = EXPECTED_LICENSES.get(license_id)
    if expected:
        url = entry.get("url", "")
        if url not in {f"http://creativecommons.org/licenses/{expected}/2.0/", f"https://creativecommons.org/licenses/{expected}/2.0/"}:
            return "unknown"
        if "nc" in expected.split("-"):
            return "excluded_noncommercial"
        if "nd" in expected.split("-"):
            return "excluded_no_derivatives"
        return "metadata_by_sa" if license_id == 5 else "metadata_by"
    return "additional_rights_review"


def source_ids(path):
    data = Path(path).read_text()
    rows = list(csv.DictReader(io.StringIO(data)))
    require(rows and "image_id" in rows[0], "Nonempty source-ID CSV required")
    ids = {int(row["image_id"]) for row in rows}
    require(all(value > 0 for value in ids), "Positive COCO identifiers required")
    return ids, len(rows)


def coco_images(archive, wanted):
    images, licenses, proofs = {}, None, {}
    with zipfile.ZipFile(archive) as zipped:
        names = [f"annotations/captions_{role}2017.json" for role in ("train", "val")]
        require(all(zipped.namelist().count(n) == 1 for n in names), "Unique exact caption members required")
        for name in names:
            info = zipped.getinfo(name)
            require(0 < info.file_size < 300_000_000 and info.flag_bits & 1 == 0, "Unexpected caption member")
            raw = zipped.read(name)  # ZipFile validates CRC; no extraction or upstream code execution.
            proofs[name] = {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(), "zip_crc32": info.CRC}
            doc = json.loads(raw)
            current = {r["id"]: r for r in doc["licenses"]}
            require(len(current) == len(doc["licenses"]), "Duplicate license identifier")
            require(licenses is None or current == licenses, "Train/val license tables differ")
            licenses = current
            all_ids = [r["id"] for r in doc["images"]]
            require(len(all_ids) == len(set(all_ids)), "Duplicate image ID in caption member")
            proofs[name]["image_count"] = len(all_ids)
            for image in doc["images"]:
                if image["id"] in wanted:
                    require(image["id"] not in images, "A source occurs in both COCO splits")
                    images[image["id"]] = {**image, "coco_split": name}
            del doc, raw
    return images, licenses, proofs


def overlap_counts(groups):
    return {f"{a}/{b}": len(groups[a] & groups[b]) for i, a in enumerate(ROLES) for b in ROLES[i + 1:]}


def aggregate(rows):
    counts = dict(sorted(Counter(r["license_status"] for r in rows).items()))
    return {
        "sources": len(rows), "license_status": counts,
        "preliminary_metadata_compatible": sum(r["license_status"] in {"metadata_by", "metadata_by_sa"} for r in rows),
        "training_admitted": 0,
        "attribution_unverified": len(rows), "source_bytes_lineage_unverified": len(rows),
    }


def verify_acquisition(folder):
    receipt = json.loads((folder / "acquisition.json").read_text())
    require(receipt["completed"] is True, "Complete acquisition required")
    require(receipt["program_sha256"] == sha(__file__), "Acquisition program changed; retain original and use a new version")
    require(set(receipt["files"]) == set(SOURCE_PINS) | set(CSV_PINS) | {"annotations_trainval2017.zip"}, "Complete pinned sources required")
    for name, record in receipt["files"].items():
        require(sha(folder / name) == record["sha256"], "Acquired source changed")
    for path, (expected, size) in CSV_PINS.items():
        data = (folder / path).read_bytes()
        require(len(data) == size and blob(data) == expected, "CSV pin differs from acquisition")
    for name, (_, _, _, expected) in SOURCE_PINS.items():
        require(blob((folder / name).read_bytes()) == expected, "Primary-source pin differs from acquisition")
    archive = folder / "annotations_trainval2017.zip"
    with archive.open("rb") as stream:
        md5 = hashlib.file_digest(stream, "md5").hexdigest()
    require(archive.stat().st_size == ARCHIVE_BYTES and md5 == ARCHIVE_MD5, "Annotation archive identity differs")
    return receipt


def audit(folder, output):
    output.mkdir(parents=True, exist_ok=False)
    receipt = verify_acquisition(folder)
    groups, csv_counts = {}, {}
    for family, stem in {"original": "metadata/orig", "firefly": "metadata/ps", "sd2": "metadata/sd2", "flux-fill": "metadata-flux/flux1filldev", "flux-original": "metadata-flux/orig-flux"}.items():
        groups[family] = {}
        for role in ROLES:
            path = f"{stem}_{role}.csv"
            ids, row_count = source_ids(folder / path)
            groups[family][role] = ids
            csv_counts[path] = {"rows": row_count, "unique_source_ids": len(ids)}
    consumed, manifests, local_originals = set(), {}, {}
    for path in sorted((ROOT / "backend/data/localization-eval").glob("tgif-*/manifest.json")):
        doc = json.loads(path.read_text())
        require(doc.get("dataset") == "TGIF and TGIF2", "Unexpected localization source manifest")
        manifests[str(path.relative_to(ROOT))] = sha(path)
        for row in doc["items"]:
            source_id = int(row["source_id"])
            consumed.add(source_id)
            if row["label"] == 0:
                image = path.parent / row["image"]
                require(sha(image) == row["image_sha256"], "Consumed original bytes changed")
                local_originals.setdefault(source_id, set()).add(row["image_sha256"])
    require(manifests and consumed, "Full existing TGIF source inventory required")
    wanted = consumed | set().union(*(ids for family in groups.values() for ids in family.values()))
    images, licenses, members = coco_images(folder / "annotations_trainval2017.zip", wanted)
    rows = []
    for source_id in sorted(wanted):
        image = images.get(source_id)
        status = license_status(image.get("license"), licenses) if image else "missing_coco_record"
        rows.append({
            "source_id": source_id, "coco_image": image,
            "recorded_license": licenses.get(image["license"]) if image else None,
            "license_status": status,
            "publisher_roles": {family: [role for role, ids in arms.items() if source_id in ids] for family, arms in groups.items()},
            "consumed_local_evaluation": source_id in consumed,
            "local_original_sha256": sorted(local_originals.get(source_id, ())),
            "attribution_verified": False, "source_bytes_lineage_verified": False,
            "training_admitted": False,
        })
    role_rows = {role: [r for r in rows if role in r["publisher_roles"]["original"]] for role in ROLES}
    overlaps = {family: overlap_counts(arms) for family, arms in groups.items()}
    mismatches = {family: {role: {"missing_from_base_original": len(groups["original"][role] - arms[role]), "outside_base_original": len(arms[role] - groups["original"][role])} for role in ROLES} for family, arms in groups.items() if family != "original"}
    public = {
        "schema": "chipmunk.image-source-rights-summary/19",
        "scope": "Metadata compatibility and consumed-source inventory, not legal clearance or camera-authenticity proof.",
        "input_sha256": {name: r["sha256"] for name, r in receipt["files"].items()},
        "coco_members": members, "license_table": [licenses[k] for k in sorted(licenses)],
        "csv_counts": csv_counts, "all_sources": aggregate(rows),
        "publisher_base_roles": {role: aggregate(items) for role, items in role_rows.items()},
        "consumed_local_sources": aggregate([r for r in rows if r["consumed_local_evaluation"]]),
        "publisher_role_intersections": overlaps, "family_vs_base_role_counts": mismatches,
        "training_eligible_after_full_review": 0,
        "missing_coco_records": sum(r["coco_image"] is None for r in rows),
        "source_original_digest_variants": dict(sorted(Counter(len(local_originals[i]) for i in local_originals).items())),
        "new_neural_forwards": 0, "fit_or_confidence_change": False,
        "limitations": [
            "BY/BY-SA metadata is a preliminary commercial/adaptation filter, not a final grant.",
            "NC, ND, unknown and other restricted/unreviewed records are not admitted.",
            "Photographer attribution, underlying high-resolution pixels and transformations remain unverified.",
            "The publisher fetches higher-resolution Flickr bytes, may rotate them, and crops model-specific originals.",
            "TGIF ShareAlike obligations, source terms, privacy and generator-output terms need separate review.",
            "No raw photo, new source pairing or private vector is in this public aggregate.",
            "Consumed files are development data; no role name makes them fresh confirmation.",
        ],
    }
    write(output / "private-sources.json", {"sources": rows, "consumed_manifest_sha256": manifests})
    write(output / "summary.json", public)
    write(output / "bindings.json", {"program_sha256": sha(__file__), "acquisition_sha256": sha(folder / "acquisition.json"), "private_sources_sha256": sha(output / "private-sources.json"), "summary_sha256": sha(output / "summary.json"), "consumed_manifest_sha256": manifests})
    print(json.dumps({"all_sources": public["all_sources"], "consumed": public["consumed_local_sources"], "role_intersections": overlaps}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("acquire", "audit"))
    parser.add_argument("folder", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.command == "acquire":
        acquire(args.folder)
    else:
        require(args.output is not None, "Audit output folder required")
        audit(args.folder, args.output)
