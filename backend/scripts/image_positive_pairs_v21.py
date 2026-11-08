"""Qualify all397 source-matched SD2 pairs using bounded private checkpoints."""

from __future__ import annotations

import argparse
import importlib.metadata
import io
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from scripts import image_pair_truth_v21 as truth
from scripts.ranged_gzip_v21 import ROOT, RUNTIME, RangeFile, extract_part, require, sha

V20 = ROOT / "backend/data/independent-evaluation/image-pixel-lineage-v20-v2"
DISCOVERY = ROOT / "backend/data/independent-evaluation/image-positive-v21-discovery"
CODE = (
    "backend/scripts/ranged_gzip_v21.py", "backend/scripts/image_pair_truth_v21.py",
    "backend/scripts/image_positive_pairs_v21.py", "backend/scripts/image_pairs_runtime_v21.py",
    "backend/tests/test_ranged_gzip_v21.py", "backend/tests/test_image_pair_truth_v21.py",
)


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def plan(folder):
    folder.mkdir(parents=True, exist_ok=False)
    public = read(ROOT / "docs/evaluation-panels/image-source-lineage-v20.json")
    index_path = V20 / "private-source-index.json"
    require(sha(index_path) == public["private_source_index_sha256"], "Published v20 source-index commitment required")
    prior = read(V20 / "plan.json")
    sources = [r for r in read(index_path)["sources"] if r["pixel_lineage_status"] == "exact_source_rgb"]
    cases = []
    for source in sources:
        case = next(c for c in prior["cases"] if c["source_id"] == source["source_id"])
        member = case["archive_member"].removesuffix("_orig.png")
        original = ROOT / source["original"]["root"] / source["original"]["file"]
        require(source["license_id"] in (4, 5) and sha(original) == source["original"]["sha256"], "Exact licensed source bytes required")
        cases.append({
            "source_id": source["source_id"], "publisher_role": source["publisher_role"],
            "author_profile": source["author_url"], "flickr_photo_id": source["flickr_photo_id"],
            "original": str(original.relative_to(ROOT)), "original_sha256": source["original"]["sha256"],
            "source_rgb_sha256": source["rgb_sha256"], "license_id": source["license_id"],
            "edit_member": member + "_mask_bbox.png_ps_mask.png_sd2_0.png",
            "mask_member": member + "_mask_bbox.png_ps_mask.png",
        })
    require(len(cases) == len({r["source_id"] for r in cases}) == 397, "Complete397-source denominator required")
    archives = {}
    for kind, name in (("edits", "sd2"), ("masks", "masks")):
        rows = read(DISCOVERY / f"{name}-directory.json")["entries"]
        for role in ("training", "validation", "testing"):
            suffix = ("sd2-sp_" if kind == "edits" else "masks_") + role + ".tar.gz"
            entry = next(r for r in rows if r["href"].endswith("/" + suffix))
            archives[f"{role}-{kind}"] = {"url": "https://cloud.ilabt.imec.be" + entry["href"], "bytes": int(entry["bytes"]), "etag": entry["etag"]}
    library = read(RUNTIME.parent / "install.json")
    require(library["version"] == "1.10.3" and all(sha(RUNTIME / name) == value for name, value in library["installed_files"].items()), "Pinned isolated reader binary/source files required")
    permissions = ROOT / "backend/data/independent-evaluation/image-source-rights-v19/source/tgif-LICENSE.txt"
    doc = {
        "schema": "chipmunk.image-positive-pair-plan/21", "code_sha256": {p: sha(ROOT / p) for p in CODE},
        "runtime_versions": {n: importlib.metadata.version(n) for n in ("numpy", "Pillow", "requests")},
        "reader_install_sha256": sha(RUNTIME.parent / "install.json"), "source_index_sha256": sha(index_path),
        "dataset_license_sha256": sha(permissions), "archives": archives, "cases": cases,
        "scope": "Publisher-released SD2-splice data under declared CC BY-SA4.0, with current BY/BY-SA2.0 photo attribution and exact original RGB lineage. Local paired-data qualification only; no generator weights, new Firefly/FLUX data, neural inference, fitting, production admission or blanket rights/camera-authenticity claim.",
        "fixed_rules": "One preselected shared category, bbox mask and generator variation0 per photo inherited from v20. All397 retained in denominator. Exact native source/edit shape and orientation; nonempty observed RGB change contained in published positive compositing alpha. Whole-mask/truth and two non-generative controls committed. No thresholds selected from model outcomes. Photographer/source-connected fitting roles and fresh confirmation remain separate required work.",
        "bounded_acquisition": "Each step is a new explicit phase, continuing only a hash-committed successful predecessor. Stop after a full tar member at240seconds or2GiB network transfer;900s/6GiB outer guard,1536MiB reserves. No retry on phase failure. Keep all indexes/checkpoints/range journals private. Imported indexes disable whole-gzip CRC validation; selected PNGs get exact hashes and independent pixel review.",
    }
    write(folder / "plan.json", doc)
    print(json.dumps({"planned_pairs": len(cases), "publisher_roles": dict(Counter(c["publisher_role"] for c in cases)), "plan_sha256": sha(folder / "plan.json")}), flush=True)


def preflight(folder):
    protocol = read(folder / "plan.json")
    require(all(sha(ROOT / name) == value for name, value in protocol["code_sha256"].items()), "Frozen pair/reader code changed")
    require(all(importlib.metadata.version(name) == value for name, value in protocol["runtime_versions"].items()), "Frozen source/decoder environment changed")
    require(sha(RUNTIME.parent / "install.json") == protocol["reader_install_sha256"], "Reader installation record changed")
    library = read(RUNTIME.parent / "install.json")
    require(all(sha(RUNTIME / name) == value for name, value in library["installed_files"].items()), "Installed reader file changed")
    return protocol


def step(folder, key, number):
    protocol = preflight(folder)
    role, kind = key.split("-")
    record = protocol["archives"][key]
    wanted = {c["edit_member" if kind == "edits" else "mask_member"] for c in protocol["cases"] if c["publisher_role"] == role}
    root = folder / key
    root.mkdir(exist_ok=True)
    require(number >= 0 and number < 40, "Explicit bounded next-step number required")
    previous, digest = None, None
    require(len(list(root.glob("part-*"))) == number, "Exact sequential completed-phase count required")
    if number:
        previous = root / f"part-{number - 1:03d}/checkpoint.json"
        seal = read(root / f"completed-{number - 1:03d}.json")
        require(seal["checkpoint_sha256"] == sha(previous) and seal["plan_sha256"] == sha(folder / "plan.json"), "Previous completed checkpoint seal required")
        previous_profile = read(folder / f"{key}-{number - 1:03d}-profile/profile.json")
        require(previous_profile["child_exit_code"] == 0 and previous_profile["guard"] is None and previous_profile["sample_errors"] == 0, "Failed phase cannot be implicitly restarted or skipped")
        require(not read(previous)["archive_end_reached"], "Archive ended with missing source members; keep them unresolved")
        digest = seal["checkpoint_sha256"]
    part = root / f"part-{number:03d}"
    part.parent.mkdir(exist_ok=True)
    with (root / f"ranges-{number:03d}.jsonl").open("x") as journal:
        remote = RangeFile(record["url"], record["bytes"], record["etag"], journal)
        try:
            result = extract_part(remote, wanted, root / "objects", part, previous=previous, previous_digest=digest)
        finally:
            remote.close()
    write(root / f"completed-{number:03d}.json", {"plan_sha256": sha(folder / "plan.json"), "checkpoint_sha256": sha(part / "checkpoint.json"), "range_journal_sha256": sha(root / f"ranges-{number:03d}.jsonl")})
    print(json.dumps({"archive": key, "step": number, "selected": len(result["selected"]), "planned": len(wanted), "all_found": result["all_wanted_found"], "archive_end_reached": result["archive_end_reached"], "compressed_bytes_transferred": result["compressed_bytes_transferred"], "duration_s": result["duration_s"], "next_tar_offset": result["next_tar_offset"]}), flush=True)


def assets(folder, key):
    root = folder / key
    parts = sorted(root.glob("part-*"))
    require(parts, "Archive acquisition required")
    result = read(parts[-1] / "checkpoint.json")
    seal = read(root / f"completed-{len(parts) - 1:03d}.json")
    require(sha(parts[-1] / "checkpoint.json") == seal["checkpoint_sha256"] and (result["all_wanted_found"] or result["archive_end_reached"]), "Complete or explicitly exhausted archive required")
    return {r["member"]: {**r, "root": str((root / "objects").relative_to(ROOT))} for r in result["selected"]}


def qualify(folder):
    protocol = preflight(folder)
    output = folder / "qualification"
    output.mkdir(exist_ok=False)
    objects = {}
    for key in protocol["archives"]:
        objects[key] = assets(folder, key)
    counts = Counter()
    with (output / "pairs.jsonl").open("x") as journal:
        for case in protocol["cases"]:
            row = {"source_id": case["source_id"], "publisher_role": case["publisher_role"], "status": "unresolved", "training_admitted": False}
            try:
                role = case["publisher_role"]
                edit, mask = objects[role + "-edits"].get(case["edit_member"]), objects[role + "-masks"].get(case["mask_member"])
                require(edit is not None and mask is not None, "Required splice or compositing mask unavailable")
                row.update({"edited_image": edit, "compositing_mask": mask})
                original_path = ROOT / case["original"]
                require(sha(original_path) == case["original_sha256"], "Original source bytes changed")
                arrays = []
                for path, mode, expected in ((original_path, "RGB", case["original_sha256"]), (ROOT / edit["root"] / edit["file"], "RGB", edit["sha256"]), (ROOT / mask["root"] / mask["file"], "L", mask["sha256"])):
                    require(sha(path) == expected, "Selected source asset changed")
                    with Image.open(path) as image:
                        require(image.format == "PNG" and max(image.size) <= 2048 and image.getexif().get(274, 1) == 1, "Bounded upright native PNG required")
                        arrays.append(np.asarray(image.convert(mode)).copy())
                changed, proof = truth.qualify(*arrays)
                buffer = io.BytesIO()
                Image.fromarray(changed.astype(np.uint8) * 255).save(buffer, format="PNG")
                data = buffer.getvalue()
                name = str(case["source_id"]) + ".truth.png"
                with (output / name).open("xb") as file:
                    file.write(data)
                row.update({"status": "qualified_source_relative_pair", "proof": proof, "truth_mask": name, "truth_png_sha256": sha(output / name)})
            except Exception as error:
                row["error"] = {"type": type(error).__name__, "message": str(error)}
            counts[row["status"]] += 1
            journal.write(json.dumps(row, allow_nan=False) + "\n")
            journal.flush()
    write(output / "summary.json", {"schema": "chipmunk.image-pair-qualification-summary/21", "planned": 397, "completed": sum(counts.values()), "statuses": dict(counts), "plan_sha256": sha(folder / "plan.json"), "pairs_sha256": sha(output / "pairs.jsonl"), "neural_forwards": 0, "training_admitted": 0, "outside_human_review": False})
    print(json.dumps(read(output / "summary.json")), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "step", "qualify"))
    parser.add_argument("folder", type=Path)
    parser.add_argument("--archive", choices=tuple(r + "-" + k for r in ("training", "validation", "testing") for k in ("edits", "masks")))
    parser.add_argument("--number", type=int)
    args = parser.parse_args()
    if args.command == "plan":
        plan(args.folder.resolve())
    elif args.command == "step":
        require(args.archive is not None and args.number is not None, "Explicit archive and next-phase number required")
        step(args.folder.resolve(), args.archive, args.number)
    else:
        qualify(args.folder.resolve())
