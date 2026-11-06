"""Acquire a fixed, source-paired subset of the TGIF and TGIF2 test archives."""
from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import tarfile
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import requests
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.localization_benchmark import sha, source  # noqa: E402

REVISION = "fc5c557bd1a2d49a5dabc3676a54cfc44798f4ea"
METADATA = f"https://raw.githubusercontent.com/IDLabMedia/tgif-dataset/{REVISION}"
CLOUD = "https://cloud.ilabt.imec.be"
SHARES = {"tgif": "xEeAzrY7ES9KA8o", "flux": "KG48tLZZzifC5WE"}
# All three archives contain spliced edits with their source background restored.
FAMILIES = {
    "firefly": ("tgif", "ps-sp", "metadata/ps_testing.csv", "ps", "orig", "masks"),
    "sd2": ("tgif", "sd2-sp", "metadata/sd2_testing.csv", "sd2", "orig", "masks"),
    "flux-fill": ("flux", "flux1filldev-sp", "metadata-flux/flux1filldev_testing.csv", "flux1filldev", "orig-flux", "masks-flux"),
}
SELECTION = "tgif-chipmunk-localization-test-v1"


def json_new(path: Path, doc: dict) -> None:
    with path.open("x") as out:
        json.dump(doc, out, indent=2, allow_nan=False)
        out.write("\n")


def archive_url(share: str, name: str) -> str:
    return f"{CLOUD}/public.php/dav/files/{SHARES[share]}/{name}/{name}_testing.tar.gz"


def plan(root: Path, count: int, exclude: set[str]) -> dict:
    groups: dict[str, list[dict]] = {}
    metadata = {}
    selected = set(exclude)
    for family, (share, archive, meta, marker, orig, mask) in FAMILIES.items():
        response = requests.get(f"{METADATA}/{meta}", timeout=20)
        response.raise_for_status()
        metadata[meta] = sha(response.content)
        rows = csv.DictReader(io.StringIO(response.text))
        unique = {(r["image_id"], r["category"], r["mask_variation"]) for r in rows if r["gen_variation_id"] == "0"}
        groups[family] = []
        for variation in ("bbox", "segm"):
            order = sorted((r for r in unique if r[2] == variation), key=lambda r: sha(f"{SELECTION}/{family}/{r}".encode()))
            needed = count // 2
            for image_id, category, _ in order:
                if image_id in selected:
                    continue
                selected.add(image_id)
                stem = f"{image_id}_mask_{variation}.png"
                intermediate = "_ps_mask.png" if family != "firefly" else ""
                path = f"testing/{category}"
                groups[family].append({
                    "source_id": image_id, "family": family, "category": category, "mask_variant": variation,
                    "image": [archive_url(share, archive), f"{path}/{stem}{intermediate}_{marker}_0.png"],
                    "original": [archive_url(share, orig), f"{path}/{image_id}_orig.png"],
                    "mask": [archive_url(share, mask), f"{path}/{stem}"],
                })
                needed -= 1
                if needed == 0:
                    break
            if needed:
                raise ValueError(f"Not enough distinct sources for {family}/{variation}")
    policy = {
        "schema": "chipmunk.tgif-localization/1", "declared_at": datetime.now(UTC).isoformat(),
        "metadata_revision": REVISION, "metadata_sha256": metadata,
        "source_license": "CC BY-SA 4.0; https://github.com/IDLabMedia/tgif-dataset",
        "excluded_coco_ids": sorted(exclude), "sources_per_family": count,
        "selection": f"{SELECTION}: SHA-256 order within family and mask variant, gen_variation_id 0; "
                     "one use of each COCO source across families; half bounding-box, half segmentation masks",
        "reference": "Exact RGB changes between the published full-image splice and its uncropped source. "
                     "Author masks retained separately for coordinate and overlap diagnostics.",
        "evaluation": "Fixed image gate 0.5 and map threshold 0.5 with current production region policy. "
                      "Both checkpoints score the identical bytes. Neither selection nor thresholds use model scores.",
        "groups": groups,
    }
    json_new(root / "tgif-plan.json", policy)
    return policy


def fetch_members(url: str, members: set[str], root: Path, *, allow_missing: bool = False) -> dict[str, dict]:
    index = root / f"archive-{sha(url.encode())[:16]}.jsonl"
    result = {}
    if index.exists():
        for line in index.read_text().splitlines():
            asset = json.loads(line)
            if (asset["archive"] != url or asset["file"] != f"{asset['sha256']}.png"
                    or sha((root / asset["file"]).read_bytes()) != asset["sha256"]):
                raise ValueError("Cached archive member differs from its recorded identity")
            result[asset["member"]] = asset
    if members <= result.keys():
        return result
    token = url.split("/files/")[1].split("/")[0]
    with requests.Session() as session, index.open("a") as log:
        session.get(f"{CLOUD}/index.php/s/{token}", timeout=20).raise_for_status()
        with session.get(url, headers={"X-Requested-With": "XMLHttpRequest"}, timeout=(20, 90), stream=True) as response:
            response.raise_for_status()
            with tarfile.open(fileobj=response.raw, mode="r|gz", bufsize=2**20) as archive:
                for entry in archive:
                    if entry.name not in members or entry.name in result:
                        continue
                    if not entry.isfile() or not 0 < entry.size <= 30_000_000:
                        raise ValueError(f"Unsafe expected archive entry {entry.name}")
                    with archive.extractfile(entry) as stream:
                        data = stream.read()
                    with Image.open(io.BytesIO(data)) as image:
                        image.verify()
                    digest = sha(data)
                    name = f"{digest}.png"
                    output = root / name
                    try:
                        with output.open("xb") as stream:
                            stream.write(data)
                    except FileExistsError as exc:
                        if output.read_bytes() != data:
                            raise ValueError("Local object differs from its committed digest") from exc
                    result[entry.name] = {"file": name, "sha256": digest, "bytes": len(data),
                                          "archive": url, "member": entry.name}
                    log.write(json.dumps(result[entry.name], allow_nan=False) + "\n")
                    log.flush()
                    if members <= result.keys():
                        break
    missing = members - result.keys()
    if missing and not allow_missing:
        raise ValueError(f"Missing {len(missing)} members in {url}: {sorted(missing)[:4]}")
    if missing:
        print(f"Unavailable secondary author masks in {url}: {sorted(missing)}", flush=True)
    print(f"Verified {len(result)} files from {url.rsplit('/', 1)[1]}", flush=True)
    return result


def acquire(root: Path, *, allow_missing_author_masks: bool = False) -> dict:
    if (root / "manifest.json").exists():
        raise FileExistsError("Use a fresh output directory")
    plan_bytes = (root / "tgif-plan.json").read_bytes()
    protocol = json.loads(plan_bytes)
    if protocol["metadata_revision"] != REVISION:
        raise ValueError("Unrecognized plan metadata revision")
    pairs = [pair for family in protocol["groups"].values() for pair in family]
    archives: dict[str, set[str]] = {}
    mask_archives = {pair["mask"][0] for pair in pairs}
    for pair in pairs:
        for role in ("image", "original", "mask"):
            url, member = pair[role]
            archives.setdefault(url, set()).add(member)
    with ThreadPoolExecutor(max_workers=5) as pool:
        downloads = {url: pool.submit(fetch_members, url, members, root,
                                     allow_missing=allow_missing_author_masks and url in mask_archives)
                     for url, members in archives.items()}
        assets = {url: task.result() for url, task in downloads.items()}

    rows = []
    for pair in pairs:
        linked = {role: assets[pair[role][0]].get(pair[role][1]) for role in ("image", "original", "mask")}
        edit, original, author = linked["image"], linked["original"], linked["mask"]
        if edit is None or original is None or (author is None and not allow_missing_author_masks):
            raise ValueError("A required image or reference source is unavailable; do not resample the case")
        with Image.open(root / edit["file"]) as edited, Image.open(root / original["file"]) as authentic:
            if (edited.size != authentic.size or min(edited.size) < 64
                    or any(i.getexif().get(274, 1) != 1 for i in (edited, authentic))):
                raise ValueError(f"Source and edit lack matching native coordinates: {pair['source_id']}")
            changed = np.any(np.asarray(edited.convert("RGB")) != np.asarray(authentic.convert("RGB")), axis=2)
        intended = None
        if author is not None:
            with Image.open(root / author["file"]) as mask_image:
                if mask_image.size != (changed.shape[1], changed.shape[0]):
                    raise ValueError(f"Published mask lacks matching native coordinates: {pair['source_id']}")
                intended = np.asarray(mask_image.convert("L")) > 127
        changed_count = int(changed.sum())
        overlap = int((changed & intended).sum()) if intended is not None else None
        if not 0 < changed_count < changed.size or overlap == 0:
            raise ValueError(f"Source comparison does not support a local edit: {pair['source_id']}")
        # A published intended mask may have feathering and may extend beyond the
        # actual changed pixels. Preserve both reference commitments for audit.
        mask_buffer = io.BytesIO()
        Image.fromarray(changed.astype(np.uint8) * 255).save(mask_buffer, format="PNG")
        data = mask_buffer.getvalue()
        digest = sha(data)
        mask_name = f"{digest}.mask.png"
        (root / mask_name).write_bytes(data)
        source_id = pair["source_id"]
        for label, asset in ((0, original), (1, edit)):
            row = {"id": f"{pair['family']}-{source_id}-{label}", "family": pair["family"], "label": label,
                   "source_id": source_id, "image": asset["file"], "image_sha256": asset["sha256"],
                   "archive_source": asset, "mask": mask_name if label else None,
                   "mask_sha256": digest if label else None}
            if label:
                row["author_mask"] = author
                row["author_mask_status"] = "available" if author else "unavailable"
                if author is None:
                    row["author_mask_reference"] = pair["mask"]
                row["changed_fraction"] = float(changed.mean())
                row["changed_within_author_mask"] = overlap / changed_count if overlap is not None else None
            source(root, row)
            rows.append(row)
    if len({r["image_sha256"] for r in rows}) != len(rows):
        raise ValueError("Duplicate source or edit bytes across sample pairs")
    manifest = {"dataset": "TGIF and TGIF2", "revision": REVISION, "plan_sha256": sha(plan_bytes), "items": rows}
    if allow_missing_author_masks:
        manifest["secondary_author_mask_policy"] = (
            "Missing published author masks are recorded, not substituted or silently resampled. "
            "The primary reference remains the exact RGB difference between the delivered splice and its original. "
            "Every original and edited image is still required and verified. Available author masks retain geometry and overlap checks.")
    json_new(root / "manifest.json", manifest)
    print(f"Verified {len(rows) // 2} matched pairs and froze {root / 'manifest.json'}", flush=True)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "acquire"))
    parser.add_argument("root", type=Path)
    parser.add_argument("--count", type=int, default=24)
    parser.add_argument("--exclude-coco-id", action="append", default=[])
    parser.add_argument("--exclude-manifest", type=Path, action="append", default=[],
                        help="Exclude every COCO source ID in earlier paired evaluation manifests")
    parser.add_argument("--allow-missing-author-masks", action="store_true",
                        help="Record unavailable secondary masks while retaining the original exact-pixel reference and source selection")
    args = parser.parse_args()
    if args.count < 2 or args.count % 2:
        parser.error("--count must be a positive even number")
    if args.action == "plan":
        excluded = set(args.exclude_coco_id)
        for path in args.exclude_manifest:
            earlier = json.loads(path.read_bytes())
            excluded.update(str(row["source_id"]) for row in earlier["items"])
        args.root.mkdir(parents=True, exist_ok=False)
        plan(args.root, args.count, excluded)
    else:
        acquire(args.root, allow_missing_author_masks=args.allow_missing_author_masks)
