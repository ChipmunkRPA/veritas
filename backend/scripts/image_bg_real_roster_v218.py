"""Plan a bounded BG-REAL LaMa PNG roster from verified metadata only."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urlsplit

import yaml

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "backend/data/independent-evaluation"
PUBLIC = ROOT / "docs/evaluation-panels"
REPO = "anush-123/bg-real"
REVISION = "aaba6cd1ced4ce3c846be9641fed092adb4919d1"
PAPER_REPO = "NoirZangetsu/bg-real"
PAPER_REVISION = "87f209e3196df622972bf7e66de50a1d2847a733"
PRIMARY, BACKUP = 24, 12
TEST_SPLITS = ("test_id", "test_background_ood", "test_subject_ood")
SPLITS = (*TEST_SPLITS, "val", "train")
SALT = "chipmunk-bg-real-v218/20261111/"
PHOTO_LICENCE = "https://creativecommons.org/licenses/by/2.0/"
MODEL = {
    "key": "lama",
    "repo": "https://github.com/enesmsahin/simple-lama-inpainting/releases/download/v0.1.0/big-lama.pt",
    "revision": "v0.1.0",
    "license": "apache-2.0",
}
METADATA = {
    "README.md": ("README.md", 8264, "b33ed5d0e41c33876fedf528940e12d0a26f99d7525195ef54f1e0786e8758d0"),
    "manifest_v2.csv": (
        "v2/manifest_v2.csv",
        12922417,
        "773548c9aa03011f308059b87ac47fe72110783b8d70804dc6d4f837372cf5f4",
    ),
    "splits_v2.yaml": (
        "v2/splits_v2.yaml",
        693311,
        "e9fe9ddcbf04c5b36905c5006d797d5b9fa0e304a8de7b28ecb82c96faecb0da",
    ),
    "attribution.csv": (
        "v2/meta/attribution.csv",
        1078451,
        "9d7b9019055f689d656312c457cab464066167d02b9fe56b8dcf89921832423f",
    ),
    "model_revisions.csv": (
        "v2/meta/model_revisions.csv",
        967,
        "e7d970bfeeb8126c6b07059b9423f71055c5ebc9834bd89a41e665dd305e7bf6",
    ),
    "recipients.csv": (
        "v2/meta/recipients.csv",
        405123,
        "a833fafa06243a6138c46a7858112f30e42640072c028c6d5736909860dedeef",
    ),
}
EXPOSURE_FILES = {
    "v19_inventory": "image-source-rights-v19/audit/private-sources.json",
    "v26_plan": "image-regional-supervised-v26/plan.json",
    "v39_graph": "image-fresh-source-cache-v39/private-groups.json",
    "v189_inputs": "image-jpeg-diagnostic-v189/inputs.json",
}
EXPOSURE_REFERENCES = (
    "image-jpeg-diagnostic-freeze-v189.json",
    "image-fresh-source-cache-v39-results.json",
    "image-legacy-authors-v25-freeze.json",
)
REVIEW_FILES = (
    "backend/scripts/image_bg_real_roster_v218.py",
    "backend/tests/test_image_bg_real_roster_v218.py",
    "docs/image-bg-real-roster-v218-method.md",
)
SHA = re.compile(r"[0-9a-f]{64}")
SOURCE = re.compile(r"oi_[0-9a-f]{16}")
FLICKR_OWNER = re.compile(r"[\w@.-]+", re.ASCII)
STATIC_PHOTO = re.compile(r"([1-9][0-9]*)_[a-zA-Z0-9]+(?:_[a-zA-Z0-9]+)?\.[a-zA-Z0-9]+")


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def table(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def exposure_keys(root=DATA, public=PUBLIC):
    freeze, graph_receipt, legacy = (read(public / name) for name in EXPOSURE_REFERENCES)
    expected = {
        "v19_inventory": legacy["prior_inventory_sha256"],
        "v26_plan": freeze["source_plan_sha256"],
        "v39_graph": graph_receipt["private_groups_sha256"],
        "v189_inputs": freeze["private_input_bindings_sha256"],
    }
    paths = {name: root / path for name, path in EXPOSURE_FILES.items()}
    require(
        all(sha(path) == expected[name] for name, path in paths.items()), "Exposure inventory bytes changed"
    )
    legacy_rows, plan, graph, jpeg = (read(paths[name]) for name in expected)
    source_ids = {r["source_id"] for r in graph["sources"]}
    v26 = {r["source_id"] for r in plan["rows"]}
    diagnostic = {r["source_id"] for r in plan["rows"] if r["role"] == "diagnostic"}
    require(
        len(source_ids) == 866
        and len(v26) == 397
        and v26 <= source_ids
        and len(diagnostic) == 59
        and diagnostic == {r["source_id"] for r in jpeg["sources"]},
        "Exposure graph must cover every v26 and v189 source",
    )
    owners = {row["author"].casefold() for row in graph["sources"] if row["author"]}
    photos = {str(row["flickr_photo_id"]) for row in plan["rows"]}
    for row in graph["sources"]:
        if row.get("photo_page"):
            _, photo = landing_identity(row["photo_page"])
            photos.add(photo)
    for row in legacy_rows["sources"]:
        if row["source_id"] in source_ids:
            photos.add(static_photo_id(row["coco_image"]["flickr_url"], historical=True))
    require(
        len(photos) == len(source_ids), "Known exposure records need complete cross-provider photo identity"
    )
    hashes = {row.get("original_sha256") or row.get("source_jpeg_sha256") for row in graph["sources"]}
    require(
        all(isinstance(value, str) and SHA.fullmatch(value) for value in hashes),
        "Invalid exposure byte digest",
    )
    keys = {("owner", owner) for owner in owners} | {("photo", photo) for photo in photos}
    return keys | {("original_sha256", value) for value in hashes}, expected


def verified_url(value, hosts, *, schemes=("https",)):
    url = urlsplit(value)
    require(
        url.scheme in schemes and url.netloc in hosts and not url.query and not url.fragment,
        "Unrecognized attribution URL",
    )
    return [unquote(part) for part in url.path.strip("/").split("/")]


def canonical_owner(value):
    require(
        FLICKR_OWNER.fullmatch(value) is not None and value not in {".", ".."}, "Unrecognized Flickr owner"
    )
    return value.casefold()


def landing_identity(value):
    parts = verified_url(value, {"flickr.com", "www.flickr.com"})
    if parts[0] == "photos":
        parts = parts[1:]
    if len(parts) == 4 and parts[2] == "in":
        parts = parts[:2]
    require(len(parts) == 2 and re.fullmatch(r"[1-9][0-9]*", parts[1]), "Unrecognized Flickr photo page")
    return canonical_owner(parts[0]), parts[1]


def static_photo_id(value, *, historical=False):
    url = urlsplit(value)
    host = url.hostname or ""
    require(
        host == "staticflickr.com" or host.endswith(".staticflickr.com"), "Unrecognized Flickr image host"
    )
    path = verified_url(value, {host}, schemes=("http", "https") if historical else ("https",))
    match = STATIC_PHOTO.fullmatch(path[-1])
    require(match is not None, "Unrecognized Flickr image name")
    return match[1]


def attribution(record, source):
    require(
        record["bgreal_id"] == source
        and record["ImageID"] == source.removeprefix("oi_")
        and record["role"] == "recipient"
        and record["License"] == PHOTO_LICENCE
        and record["Author"].strip(),
        "Missing per-photo source or licence attribution",
    )
    parts = verified_url(record["AuthorProfileURL"], {"flickr.com", "www.flickr.com"})
    require(len(parts) == 2 and parts[0] in {"photos", "people"}, "Missing photographer profile")
    owner = canonical_owner(parts[1])
    linked_owner, photo = landing_identity(record["OriginalLandingURL"])
    require(
        linked_owner == owner and static_photo_id(record["OriginalURL"]) == photo,
        "Unresolved photo or owner alias",
    )
    return owner, photo


def base_png(row):
    return (
        row["robustness"] == "none"
        and row["encoding"] == "png"
        and row["jpeg_quality"] == "none"
        and row["base_sample_id"] == row["sample_id"]
    )


def named_png(path, folder, filename):
    expected = folder / filename
    require(path == str(expected), "Publisher image or mask path is outside the expected asset location")
    return path


def dimensions(row):
    values = row["width"], row["height"]
    require(
        all(re.fullmatch(r"[0-9]{1,3}", value) for value in values), "Unmatched or unsupported dimensions"
    )
    return tuple(int(value) for value in values)


def pair(edited, original, license_row, recipient, memberships):
    source, split = edited["source_image_id"], edited["split"]
    require(
        SOURCE.fullmatch(source) is not None
        and split in SPLITS
        and edited["sample_id"] == f"{source}_F5"
        and original["sample_id"] == f"{source}_A0"
        and original["source_image_id"] == recipient["source_image_id"] == source
        and recipient["image_id"] == source.removeprefix("oi_")
        and original["split"] == recipient["split"] == split
        and {edited["sample_id"], original["sample_id"]} <= memberships[split]
        and base_png(original)
        and original["label"] == "0"
        and original["generator"] == "none"
        and not original["donor_id"]
        and not edited["donor_id"]
        and edited["label"] == "1",
        "Base PNG pair or declared split cannot be verified",
    )
    size = dimensions(edited)
    require(
        size == dimensions(original) == dimensions(recipient) and 64 <= min(size) <= max(size) <= 768,
        "Unmatched or unsupported dimensions",
    )
    require(
        SHA.fullmatch(original["sha256"])
        and SHA.fullmatch(edited["sha256"])
        and original["sha256"] != edited["sha256"],
        "Missing or identical image digest",
    )
    try:
        edit_area = float(edited["edit_area"])
    except ValueError:
        raise ValueError("Invalid semantic edit area") from None
    require(0 < edit_area < 1, "Nonlocal or empty semantic edit")
    owner, photo = attribution(license_row, source)
    group = source[3]
    images = PurePosixPath("v2/images") / group
    mask = named_png(
        edited["mask_edit_path"], PurePosixPath("v2/masks/edit") / group, f"{source}_F5_edit.png"
    )
    return {
        "recipient": source,
        "publisher_split": split,
        "size": list(size),
        "semantic_edit_area": edit_area,
        "flickr_photo_id": photo,
        "flickr_owner": owner,
        "photo_attribution": license_row,
        "original": {
            "path": named_png(original["image_path"], images, f"{source}_A0.png"),
            "sha256": original["sha256"],
        },
        "edited": {
            "path": named_png(edited["image_path"], images, f"{source}_F5.png"),
            "sha256": edited["sha256"],
        },
        "semantic_mask": {"path": mask, "sha256": None},
    }


def indexed(rows, key):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row[key]].append(row)
    return grouped


def one(mapping, key, description):
    matches = mapping.get(key, [])
    require(len(matches) == 1, description)
    return matches[0]


def source_keys(record):
    return {
        ("owner", record["flickr_owner"]),
        ("photo", record["flickr_photo_id"]),
        ("original_sha256", record["original"]["sha256"]),
    }


def rank(record):
    split = record["publisher_split"]
    tier = 0 if split in TEST_SPLITS else 1 if split == "val" else 2
    rank_key = hashlib.sha256((SALT + record["flickr_photo_id"]).encode()).hexdigest()
    return tier, rank_key, record["recipient"]


def select(records, exposed):
    components = []
    for record in sorted(records, key=rank):
        keys = source_keys(record)
        linked = [i for i, (known, _) in enumerate(components) if keys & known]
        linked_records = [record]
        for i in reversed(linked):
            known, group = components.pop(i)
            keys |= known
            linked_records.extend(group)
        components.append((keys, linked_records))
    allowed, exclusions = [], []
    for keys, group in components:
        group.sort(key=rank)
        if keys & exposed:
            exclusions.extend(
                {"recipient": r["recipient"], "reason": "prior_exposure_component"} for r in group
            )
        else:
            allowed.append(group[0])
            exclusions.extend(
                {"recipient": r["recipient"], "reason": "shared_recipient_component"} for r in group[1:]
            )
    return sorted(allowed, key=rank), exclusions


def plan(manifest, licenses, recipients, memberships, exposed):
    target = [
        row
        for row in manifest
        if row["family"] == "F5"
        and row["generator"] == "big-lama"
        and row["edit_kind"] == "object_removal"
        and base_png(row)
    ]
    require(
        len({r["sample_id"] for r in target}) == len(target), "Duplicate F5 identity in publisher manifest"
    )
    base_originals = indexed(
        (r for r in manifest if r["family"] == "A0" and r["robustness"] == "none"),
        "source_image_id",
    )
    by_recipient = indexed(recipients, "source_image_id")
    by_licence = indexed((r for r in licenses if r["role"] == "recipient"), "bgreal_id")
    candidates, rejected = [], []
    for edit in target:
        source = edit["source_image_id"]
        try:
            original = one(base_originals, source, "Exactly one base original required")
            licence = one(by_licence, source, "Exactly one source attribution required")
            recipient = one(by_recipient, source, "Exactly one recipient metadata row required")
            candidates.append(pair(edit, original, licence, recipient, memberships))
        except ValueError as error:
            rejected.append({"recipient": source, "reason": str(error)})
    ordered, excluded = select(candidates, exposed)
    selected = ordered[: PRIMARY + BACKUP]
    for position, row in enumerate(selected):
        row.update(
            ordinal=position,
            tier="primary" if position < PRIMARY else "backup",
            binary_qualification="pending",
        )
    return (
        {
            "publisher_candidates_by_split": dict(sorted(Counter(r["split"] for r in target).items())),
            "metadata_rejections": dict(sorted(Counter(r["reason"] for r in rejected).items())),
            "component_exclusions": dict(sorted(Counter(r["reason"] for r in excluded).items())),
            "metadata_eligible_components": len(ordered),
            "primary_planned_by_split": dict(
                sorted(Counter(r["publisher_split"] for r in selected[:PRIMARY]).items())
            ),
            "backup_planned_by_split": dict(
                sorted(Counter(r["publisher_split"] for r in selected[PRIMARY:]).items())
            ),
            "primary_planned": min(len(selected), PRIMARY),
            "backup_planned": max(len(selected) - PRIMARY, 0),
            "eligible_but_outside_bounded_pool": len(ordered) - len(selected),
        },
        {
            "ordered_candidates": selected,
            "excluded": sorted(rejected + excluded, key=lambda row: row["recipient"]),
        },
    )


def inspect(metadata):
    commitments = {}
    for name, (repo_path, length, digest) in METADATA.items():
        path = metadata / name
        require(
            path.is_file()
            and not path.is_symlink()
            and path.stat().st_size == length
            and sha(path) == digest,
            f"Unexpected pinned metadata bytes for {name}",
        )
        commitments[name] = {"repo_path": repo_path, "bytes": length, "sha256": digest}
    models = indexed(table(metadata / "model_revisions.csv"), "key")
    require(
        one(models, "lama", "One LaMa provenance row required") == MODEL, "LaMa revision or licence changed"
    )
    with (metadata / "splits_v2.yaml").open(encoding="utf-8") as handle:
        splits = yaml.safe_load(handle)
    memberships = {key: set(splits[key]) for key in SPLITS}
    require(
        all(len(memberships[key]) == len(splits[key]) for key in SPLITS),
        "Duplicate publisher split membership",
    )
    exposed, exposure_commitments = exposure_keys()
    summary, private = plan(
        table(metadata / "manifest_v2.csv"),
        table(metadata / "attribution.csv"),
        table(metadata / "recipients.csv"),
        memberships,
        exposed,
    )
    require(
        sum(summary["publisher_candidates_by_split"].values()) == 73,
        "Exact advertised base PNG LaMa roster required",
    )
    scope = {
        "schema": "chipmunk.image-bg-real-metadata-roster/218",
        "publisher_repo": REPO,
        "publisher_revision": REVISION,
        "paper_repo": PAPER_REPO,
        "paper_repo_revision": PAPER_REVISION,
        "metadata_commitments": commitments,
        "known_exposure_commitments": exposure_commitments,
        "exposure_reference_commitments": {name: sha(PUBLIC / name) for name in EXPOSURE_REFERENCES},
        "code_commitments": {name: sha(ROOT / name) for name in REVIEW_FILES},
        "publisher_generator_record": MODEL,
        "source_photo_licence": PHOTO_LICENCE,
        "maximum_primary": PRIMARY,
        "maximum_backups": BACKUP,
        "rank": "Publisher test splits first, then validation, then training. Fixed salted SHA-256 within each tier.",
        "media_opened": False,
        "exact_rgb_and_perceptual_duplicate_checks_pending": True,
        "neural_calls": 0,
        "serving_admitted": False,
    }
    return {**scope, **summary}, {**scope, **private}


def write(path, value):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    preflight = commands.add_parser("preflight")
    roster = commands.add_parser("roster")
    for action in (preflight, roster):
        action.add_argument("metadata", type=Path)
    roster.add_argument("output", type=Path)
    args = parser.parse_args()
    summary, private = inspect(args.metadata.resolve())
    if args.command == "roster":
        args.output.mkdir(mode=0o700)
        target = args.output / "private-roster.json"
        write(target, private)
        summary["private_roster_sha256"] = sha(target)
        write(args.output / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
