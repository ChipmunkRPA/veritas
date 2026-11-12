# SPDX-License-Identifier: Apache-2.0
"""Freeze a small, commercially reusable source roster for whole-image transfer."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
from urllib import error, parse, request

from PIL import Image, ImageOps

from scripts import image_reference_orientation_transfer as retained
from scripts import image_thumbnail_authentic_intake as commons

REPOSITORY = "Qwen/Qwen-Image-Bench"
REVISION = "d2493deb153b020cf169c7e3f57d15e4dd697038"
HUB = "https://huggingface.co"
TERMS = "https://www.apache.org/licenses/LICENSE-2.0"
GENERATORS = ("Qwen-Image-2512", "GLM-Image", "FLUX.2_max")
GENERATED_PER_FAMILY, ORDINARY_COUNT = 6, 16
SALT = "chipmunk-whole-image-transfer-2026-11-12"
LIMITS = {"synthetic": 2_000_000, "ordinary": 4_000_000}
MAX_PIXELS = 24_000_000
OLDER = Path("docs/evaluation-panels/image-v7-smaug-development-panel.json")
ARCHIVE = Path("backend/data/independent-evaluation/image-affine-native-scale-2026-11-11")
POINT = Path("docs/evaluation-panels/image-v7-smaug-genuine-operating-point.json")
WEIGHTS = Path("docs/evaluation-panels/image-v7-smaug-acquisition.json")
JPEG75 = {"quality": 75, "subsampling": 2, "optimize": False, "progressive": False}


def digest(body):
    return hashlib.sha256(body).hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def rank(value):
    return digest(f"{SALT}/{value}".encode())


def get(url, limit, allowed):
    with request.urlopen(request.Request(url, headers={"User-Agent": commons.AGENT}), timeout=25) as response:
        host = parse.urlsplit(response.geturl()).hostname or ""
        require(any(host == name or host.endswith(f".{name}") for name in allowed), "Unexpected source redirect")
        body = response.read(limit + 1)
    require(0 < len(body) <= limit, "Source exceeds its pinned byte budget")
    return body


def hub(path):
    return f"{HUB}/datasets/{REPOSITORY}/resolve/{REVISION}/{parse.quote(path, safe='/')}"


def generated_plan():
    api = get(f"{HUB}/api/datasets/{REPOSITORY}/revision/{REVISION}", 4_000_000, ("huggingface.co",))
    meta = json.loads(api)
    card = get(hub("README.md"), 180_000, ("huggingface.co", "hf.co"))
    require(meta["sha"] == REVISION and meta["cardData"]["license"] == "apache-2.0"
            and b"license: apache-2.0" in card and b"text-to-image" in card.lower(), "Dataset rights or source revision changed")
    files = {PurePosixPath(item["rfilename"]) for item in meta["siblings"]}
    used = set()
    entries = []
    for family in GENERATORS:
        available = {}
        for item in files:
            if item.parts[:2] == ("images", family) and item.suffix.lower() in {".jpg", ".jpeg"}:
                prompt = int(item.stem.split("_", 1)[0])
                if 1 <= prompt <= 120:
                    require(prompt not in available, "The pinned prompt has more than one delivery")
                    available[prompt] = item
        selected = sorted(set(available) - used, key=rank)[:GENERATED_PER_FAMILY]
        require(len(selected) == GENERATED_PER_FAMILY, "Incomplete generator roster")
        used.update(selected)
        entries.extend({"label": "synthetic", "family": family, "group": f"prompt-{prompt:04d}", "prompt_id": prompt,
                        "path": str(available[prompt]), "url": hub(str(available[prompt])),
                        "license": TERMS} for prompt in selected)
    return {"repository": REPOSITORY, "revision": REVISION, "license": TERMS, "license_url": hub("README.md"),
            "pinned_card_sha256": digest(card), "pinned_file_listing_sha256": digest(api), "sources": entries}


def ordinary_plan(archive):
    exact = retained.binding(archive)
    public = read(retained.SOURCE)
    rows = []
    for entry, credited, intake in zip(exact["sources"], public["source_roster"]["sources"],
                                       public["source_intake"]["attempts"], strict=True):
        require(entry["slot"] == credited["slot"] == intake["slot"] and credited["licence_url"] in commons.LICENSES,
                "Ordinary photograph attribution or commercial license changed")
        rows.append({"label": "ordinary", "family": credited["category"], "group": credited["author"].casefold(),
                     "title": credited["title"], "page": credited["page"], "author": credited["author"],
                     "license": credited["licence_url"], "file": entry["file"], "publisher_delivery": intake["delivery"],
                     "archive_sha256": entry["sha256"]})
    require(len(rows) == len({row["group"] for row in rows}) == 38, "Ordinary photo contributors are repeated")
    return {"source_receipt": exact["public_source_receipt"], "sources": sorted(rows, key=lambda row: rank(row["title"]))}


def rgb_identity(image):
    rgb = ImageOps.exif_transpose(image).convert("RGB")
    gray = list(rgb.convert("L").resize((9, 8), Image.Resampling.LANCZOS).getdata())
    bits = [gray[9*y + x] > gray[9*y + x + 1] for y in range(8) for x in range(8)]
    score = sum(int(bit) << i for i, bit in enumerate(bits))
    return digest(str(rgb.size).encode() + rgb.tobytes()), score


def earlier_commitments():
    panel = read(OLDER)["record"]
    encoded, rgb, hashes = set(), set(), []
    for row in panel["inventory"] + panel["items"]:
        encoded.add(row["sha256"])
        if "input_sha256" in row:
            encoded.add(row["input_sha256"])
        rgb.add(row["rgb_sha256"])
        hashes.append(int(row["dhash64"], 16))
    return encoded, rgb, hashes


def acquire(directory, plan, archive, cache):
    objects = directory / "objects"
    objects.mkdir()
    encoded, rgbs, hashes = earlier_commitments()
    groups, out, attempts = set(), [], []
    lanes = [("synthetic", plan["generated"]["sources"], len(plan["generated"]["sources"]))]
    lanes.append(("ordinary", plan["ordinary"]["sources"], ORDINARY_COUNT))
    previous = {} if cache is None else {row["url"]: row for row in read(cache / "intake.json")["sources"]
                                         if row["label"] == "synthetic"}
    for label, rows, required in lanes:
        accepted = 0
        for row in rows:
            if row["group"] in groups:
                continue
            status = "qualified"
            try:
                if label == "ordinary":
                    path = archive / row["file"]
                    require(0 < path.stat().st_size <= LIMITS[label], "Retained ordinary photograph is out of scope")
                    body = path.read_bytes()
                    require(digest(body) == row["archive_sha256"], "Retained photograph bytes changed")
                elif saved := previous.get(row["url"]):
                    path = cache / "objects" / f"{saved['sha256']}.jpg"
                    body = path.read_bytes()
                    require(len(body) == saved["bytes"] and digest(body) == saved["sha256"], "Earlier source intake changed")
                else:
                    body = get(row["url"], LIMITS[label], ("huggingface.co", "hf.co"))
                sha = digest(body)
                with Image.open(io.BytesIO(body)) as photo:
                    require(photo.format == "JPEG" and photo.width * photo.height <= MAX_PIXELS, "Unexpected source pixels or format")
                    rgb, similarity = rgb_identity(photo)
                require(sha not in encoded and rgb not in rgbs and all((similarity ^ old).bit_count() > 4 for old in hashes),
                        "Encoded bytes, oriented pixels or perceptual fingerprint overlap")
                (objects / f"{sha}.jpg").write_bytes(body)
                out.append({**row, "sha256": sha, "rgb_sha256": rgb, "difference_hash": f"{similarity:016x}",
                            "bytes": len(body)})
                encoded.add(sha)
                rgbs.add(rgb)
                hashes.append(similarity)
                groups.add(row["group"])
                accepted += 1
            except (ValueError, OSError, KeyError, SyntaxError) as exc:
                status = f"http_{exc.code}" if isinstance(exc, error.HTTPError) else str(exc)
            attempts.append({"label": label, "source": row.get("url") or row["page"], "status": status})
            if accepted == required or status == "http_429":
                break
    return {"schema": "chipmunk.image-whole-fresh-intake/1", "sources": out, "attempts": attempts,
            "scope": "Pure text-to-image and attributed ordinary photograph transfer. Photo capture provenance and detector training overlap remain unverified."}


def freeze(directory):
    rows = read(directory / "intake.json")["sources"]
    synthetics = [row for row in rows if row["label"] == "synthetic"]
    ordinary = [row for row in rows if row["label"] == "ordinary"]
    require(len(ordinary) == ORDINARY_COUNT and len(synthetics) == len(GENERATORS) * GENERATED_PER_FAMILY,
            "Whole-image comparison requires the entire qualified roster")
    require(len({row["group"] for row in rows}) == len(rows) and len({row["sha256"] for row in rows}) == len(rows),
            "Source groups or exact bytes overlap")
    delivered = directory / "deliveries"
    delivered.mkdir()
    items = []
    for index, row in enumerate(rows):
        source = directory / "objects" / f"{row['sha256']}.jpg"
        require(digest(source.read_bytes()) == row["sha256"], "Qualified source bytes changed")
        with Image.open(source) as image, ImageOps.exif_transpose(image) as upright:
            converted = upright.convert("RGB")
            target = delivered / f"{index:03d}-jpeg75.jpg"
            converted.save(target, "JPEG", **JPEG75)
        item = {"source_index": index, "label": row["label"], "family": row["family"]}
        items.extend(({**item, "format": "native", "path": str(source.relative_to(directory)), "sha256": row["sha256"]},
                      {**item, "format": "jpeg75", "path": str(target.relative_to(directory)), "sha256": digest(target.read_bytes())}))
    weight = read(WEIGHTS)
    return {"schema": "chipmunk.image-whole-fresh-source-freeze/1",
            "plan_sha256": digest((directory / "plan.json").read_bytes()),
            "intake_sha256": digest((directory / "intake.json").read_bytes()),
            "prior_operating_point_sha256": digest(POINT.read_bytes()), "operating_point": read(POINT)["gate"],
            "candidate_weight_record_sha256": digest(WEIGHTS.read_bytes()),
            "candidate": {"repository": weight["repository"], "revision": weight["revision"],
                          "files": {entry["name"]: {"sha256": entry["sha256"], "bytes": entry["bytes"]}
                                    for entry in weight["files"] if entry["name"].endswith(".safetensors")}},
            "jpeg75": JPEG75, "items": items,
            "criteria": {"format_minimum_added_synthetic": 2, "format_minimum_represented_generator_families": 2,
                         "format_maximum_additional_ordinary_groups": 1, "minimum_complete_deliveries": len(items)},
            "scope": "Compare the actual current Full statistical AI direction against a positive-only supplement. "
                     "Report the baseline and the supplement separately and do not adjust the inherited point. "
                     "Run no serving or overall confidence change based on this small cohort."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("plan", "acquire", "freeze"))
    parser.add_argument("directory", type=Path)
    parser.add_argument("--ordinary-archive", type=Path, default=ARCHIVE)
    parser.add_argument("--prior-generated", type=Path)
    args = parser.parse_args()
    if args.stage == "plan":
        args.directory.mkdir(mode=0o700)
        generated = generated_plan() if args.prior_generated is None else read(args.prior_generated / "plan.json")["generated"]
        plan = {"schema": "chipmunk.image-whole-fresh-plan/2", "generated": generated,
                "ordinary": ordinary_plan(args.ordinary_archive),
                "earlier_panel_sha256": digest(OLDER.read_bytes()), "ranking_salt": SALT,
                "generated_per_family": GENERATED_PER_FAMILY, "ordinary_count": ORDINARY_COUNT}
        write(args.directory / "plan.json", plan)
        print("Frozen source plan", digest((args.directory / "plan.json").read_bytes()), flush=True)
    elif args.stage == "acquire":
        plan = read(args.directory / "plan.json")
        require(digest(OLDER.read_bytes()) == plan["earlier_panel_sha256"], "Earlier project exposure bindings changed")
        require(ordinary_plan(args.ordinary_archive) == plan["ordinary"], "Ordinary source roster changed")
        intake = acquire(args.directory, plan, args.ordinary_archive, args.prior_generated)
        write(args.directory / "intake.json", intake)
        print({label: sum(row["label"] == label for row in intake["sources"]) for label in LIMITS}, flush=True)
    else:
        lock = args.directory / "freeze.json"
        write(lock, freeze(args.directory))
        print("Frozen evaluation inputs", digest(lock.read_bytes()), flush=True)


if __name__ == "__main__":
    main()
