"""Measure crop-and-resize evidence for independently delivered earlier and submitted JPEGs."""
from __future__ import annotations

import argparse
import io
import json
import time
from collections import Counter
from hashlib import sha256
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageOps
from PIL import __version__ as pillow_version

from app.detectors.image import (
    reference_alignment,
    reference_compare,
    reference_crop_resize,
    reference_resize,
)
from app.evidence import image_reference

SCHEMA = "chipmunk.image-reference-jpeg90-transfer/1"
GROUPS = 24
ROLES = ("unchanged", "feathered", "sharp", "independent_earlier_delivery", "prior_recompression",
         "global_contrast", "fractional", "rotation", "anisotropic", "unrelated", "q87_unchanged", "q75_regression")
NUISANCE = {"unchanged", "independent_earlier_delivery", "prior_recompression", "global_contrast"}
INCOMPATIBLE = {"fractional", "rotation", "anisotropic", "unrelated"}
TARGETS = {"sharp", "feathered", "q75_regression"}
EXPOSURE = (
    "docs/evaluation-panels/image-small-boundary-screenshot-admission-2026-11-12.json",
    "docs/evaluation-panels/image-small-boundary-interior-transfer-2026-11-12.json",
    "docs/evaluation-panels/image-small-boundary-control-stress-2026-11-12.json",
    "docs/evaluation-panels/image-reference-oriented-crop-2026-11-12.json",
    "docs/evaluation-panels/image-reference-oriented-resize-transfer-2026-11-12.json",
    "docs/evaluation-panels/image-copy-tone-transfer-2026-11-11.json",
    "backend/data/independent-evaluation/image-flux-photo-preparation-v276/private-inputs.json",
    "backend/data/independent-evaluation/image-flux-photo-expansion-v285/private-inputs.json",
)
DESIGN = {
    "earlier": "One new standard RGB 4:2:0 JPEG at quality 82 or 95 per source. For masters exceeding 768 pixels use a native centered subcrop without resizing. Native decoded earlier coordinates define all bounds.",
    "submitted": "Fixed asymmetric native crop followed by proportional resize at .74, .86 or 1.13 and a separate standard RGB 4:2:0 JPEG at quality 90. Quality 87 and 75 are separate branch controls.",
    "candidate": "Only extend the existing crop-and-resize branch to quality 90 for earlier RGB 4:2:0 JPEG with standard 82 or 95 tables. Reuse all frozen search, registration, delivery and mask checks.",
    "role": "24 hash-ordered, distinct credited contributors and previously unlinked source components. Separate by earlier quality and report each 12-source stratum.",
    "useful": "Exact earlier native crop and at least .30 IoU and .50 recall on the submitted pre-encode authored edit.",
    "advance": "Per stratum: at least six new useful feathered locations, eight exact unchanged crops with no highlighted pixels, zero wrong or incompatible accepted candidate crop, and no more than four nuisance-highlighted files across at most three sources. No change in historical quality-75 results or quality-87 branch admission.",
}


def digest(value):
    return sha256(value).hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def write(path, data):
    with path.open("x") as output:
        json.dump(data, output, indent=2, allow_nan=False)
        output.write("\n")


def sources(cache, root):
    inventory = read(cache / "private-groups.json")
    metadata = {row["source_id"]: row for row in read(cache / "plan.json")["items"]}
    records = {row["key"]: row for row in inventory["sources"]}
    index = root / "backend/data/independent-evaluation/image-native-tile-agreement-2026-11-12"
    evidence = [root / path for path in EXPOSURE] + sorted(index.glob("*.json"))
    archive = [path.read_bytes() for path in evidence]
    selected, authors = [], set()
    components = sorted(inventory["candidate_only_unlinked_components"],
                        key=lambda group: digest(f"jpeg90-transfer/1/{min(group)}".encode()))
    for group in components:
        members = [records[key] for key in group]
        if any(row.get(field, "").encode() in ledger for row in members for ledger in archive
               for field in ("source_jpeg_sha256", "rgb_sha256") if row.get(field)):
            continue
        for row in sorted(members, key=lambda record: record["key"]):
            if not row["key"].startswith("fresh-jpeg/") or not 350 <= min(row["size"]) <= max(row["size"]) <= 1024:
                continue
            info = metadata[row["source_id"]]
            credit = info["attribution"]
            if (credit["license_id"] not in {4, 5} or row["license_id"] != credit["license_id"]
                    or row["license_url"] != credit["license_url"] or not credit["provider_attribution_and_license_match"]
                    or row["metadata_body_sha256"] != info["metadata_body_sha256"] or row["author"] in authors):
                continue
            path = Path(row["source_jpeg_file"])
            if not path.is_file():
                continue
            body = path.read_bytes()
            assert digest(body) == row["source_jpeg_sha256"]
            with Image.open(io.BytesIO(body)) as original:
                rgb = ImageOps.exif_transpose(original).convert("RGB")
                assert list(rgb.size) == row["size"] and digest(rgb.tobytes()) == row["rgb_sha256"]
            selected.append({"source": row["key"], "sha256": row["source_jpeg_sha256"], "rgb_sha256": row["rgb_sha256"],
                             "credit": {key: credit[key] for key in ("title", "author_name", "photo_page", "license_url")},
                             "author_key": row["author"], "size": row["size"], "file": str(path)})
            authors.add(row["author"])
            break
        if len(selected) == GROUPS:
            break
    assert len(selected) == GROUPS, len(selected)
    return selected, {str(path.relative_to(root)): digest(blob) for path, blob in zip(evidence, archive, strict=True)}


def jpeg(pixels, quality):
    output = io.BytesIO()
    Image.fromarray(pixels).save(output, format="JPEG", quality=quality, subsampling=2)
    return output.getvalue()


def pixels(body):
    with Image.open(io.BytesIO(body)) as picture:
        return np.asarray(ImageOps.exif_transpose(picture).convert("RGB"))


def master(row):
    raw = Path(row["file"]).read_bytes()
    assert digest(raw) == row["sha256"]
    rgb = pixels(raw)
    h, w = rgb.shape[:2]
    left, top = max(0, (w - 768) // 2), max(0, (h - 768) // 2)
    return rgb[top:top + min(h, 768), left:left + min(w, 768)]


def deliveries(slot, latent, other):
    quality = (82, 95)[slot % 2]
    earlier = jpeg(latent, quality)
    before = pixels(earlier)
    h, w = before.shape[:2]
    rect = (w * (8 + slot % 3) // 100, h * 7 // 100, w - w * 8 // 100, h - h * (6 + slot % 3) // 100)
    left, top, right, bottom = rect
    rw, rh = right-left, bottom-top
    crop = {"x": left, "y": top, "w": rw, "h": rh}
    scale = (.74, .86, 1.13)[(slot // 2) % 3]
    size = (round(rw * scale), round(rh * scale))
    kernel = (Image.Resampling.BICUBIC, Image.Resampling.LANCZOS, Image.Resampling.BILINEAR)[(slot // 2) % 3]

    def resize(image):
        return np.asarray(image.resize(size, kernel))

    image = Image.fromarray(before).crop(rect)
    native = np.asarray(image)
    original = resize(image)
    yy, xx = np.ogrid[:rh, :rw]
    center_x = rw * (.41 + .10 * (slot % 2))
    radius = np.maximum(np.abs(xx-center_x)/(rw*.17), np.abs(yy-rh*.52)/(rh*.15))
    hard = (radius <= 1).astype(np.float32)
    feather = np.clip((1-radius)*4, 0, 1).astype(np.float32)

    def colored(alpha):
        change = np.array([75, -57, 52] if slot % 2 else [-62, 65, 55], dtype=np.float32)
        edited = np.clip(np.rint(native + alpha[..., None] * change), 0, 255).astype(np.uint8)
        return resize(Image.fromarray(edited))

    feathered, sharp = colored(feather), colored(hard)
    sw, sh = size
    fraction = cv2.warpAffine(original, np.array(((1, 0, .5 * sw / rw), (0, 1, .5 * sh / rh))),
                              size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    views = {
        "unchanged": original, "feathered": feathered, "sharp": sharp,
        "independent_earlier_delivery": resize(Image.fromarray(latent).crop(rect)),
        "prior_recompression": resize(Image.fromarray(pixels(jpeg(before, 80))).crop(rect)),
        "global_contrast": resize(ImageEnhance.Contrast(image).enhance(1.12)),
        "fractional": fraction, "rotation": resize(image.rotate(1.5, resample=Image.Resampling.BICUBIC)),
        "anisotropic": np.asarray(image.resize((round(sw * 1.06), sh), kernel)),
        "unrelated": resize(Image.fromarray(other)), "q87_unchanged": original, "q75_regression": feathered,
    }
    for role in ROLES:
        image = views[role]
        delivery_quality = 87 if role == "q87_unchanged" else 75 if role == "q75_regression" else 90
        mask = np.any(image != original, axis=2) if role in TARGETS else np.zeros(image.shape[:2], bool)
        yield role, quality, earlier, jpeg(image, delivery_quality), mask, None if role in INCOMPATIBLE else crop


def collect(rows, run):
    latents = [master(row) for row in rows]
    case_dir = run / "cases"
    case_dir.mkdir(mode=0o700)
    manifest = []
    for slot, latent in enumerate(latents):
        for role, quality, earlier, submitted, truth, crop in deliveries(slot, latent, latents[(slot+1) % len(rows)]):
            reference = f"{slot:02d}-earlier.jpg"
            filename = f"{slot:02d}-{role}.jpg"
            if role == ROLES[0]:
                (case_dir / reference).write_bytes(earlier)
            (case_dir / filename).write_bytes(submitted)
            np.save(case_dir / f"{filename}.npy", truth, allow_pickle=False)
            manifest.append({"slot": slot, "quality": quality, "role": role, "earlier": reference, "submitted": filename,
                             "earlier_sha256": digest(earlier), "submitted_sha256": digest(submitted),
                             "truth_sha256": digest(truth.tobytes()), "crop": crop, "size": list(truth.shape[1::-1])})
    return manifest


def metrics(mask, truth):
    found = int(np.count_nonzero(mask)) if mask is not None else 0
    overlap = int(np.count_nonzero(mask & truth)) if mask is not None else 0
    count = int(np.count_nonzero(truth))
    useful = bool(count and overlap * 2 >= count and 10 * overlap >= 3 * (found + count - overlap))
    return {"pixels": found, "truth": count, "overlap": overlap, "useful": useful}


def score(case, run):
    path = run / "cases"
    earlier, submitted = (path / case[key] for key in ("earlier", "submitted"))
    before, after = earlier.read_bytes(), submitted.read_bytes()
    truth = np.load(path / f"{submitted.name}.npy", allow_pickle=False)
    assert (digest(before), digest(after), digest(truth.tobytes())) == tuple(
        case[f"{key}_sha256"] for key in ("earlier", "submitted", "truth"))
    baseline = image_reference.inspect(earlier, submitted, allow_integer_alignment=True, allow_proportional_resize=True,
                                       allow_oriented_crop=True, allow_oriented_resize=True)
    row = {"baseline_status": baseline["status"], "baseline_schema": baseline["schema"],
           "candidate": None, "regions": [], "mask_sha256": None}
    if (baseline["status"] == "comparable" or case["role"] in {"q75_regression", "q87_unchanged"}
            or (baseline["schema"] != image_reference.RESIZE_SCHEMA and baseline["status"] != "integer_alignment_unresolved")):
        return row
    with Image.open(io.BytesIO(after)) as carrier:
        candidate, mask = reference_crop_resize.compare(pixels(before), pixels(after), carrier)
        if not image_reference.accepted_crop_resize(candidate, carrier.size):
            return row
    crop = {key: candidate["reference_crop_bounds"][key] for key in ("x", "y", "w", "h")}
    return {**row, "candidate": crop, "regions": candidate["regions"],
            "mask_sha256": digest(mask.tobytes()), "metrics": metrics(mask, truth)}


def summarize(cases, outcomes):
    results = {}
    for quality in (82, 95):
        counts, nuisance = {}, set()
        for case, row in zip(cases, outcomes, strict=True):
            if case["quality"] != quality:
                continue
            role = case["role"]
            cell = counts.setdefault(role, Counter())
            present = row["candidate"] is not None
            exact = present and row["candidate"] == case["crop"]
            marked = bool(row["regions"])
            if role in NUISANCE and marked:
                nuisance.add(case["slot"])
            cell.update(cases=1, old_comparable=row["baseline_status"] == "comparable",
                        new_comparable=present, exact=exact, wrong=present and not exact,
                        highlighted=marked, useful=exact and row.get("metrics", {}).get("useful", False))
        passed = (counts["unchanged"]["exact"] >= 8 and not counts["unchanged"]["highlighted"]
                  and counts["feathered"]["useful"] >= 6 and not any(cell["wrong"] for cell in counts.values())
                  and sum(counts[role]["highlighted"] for role in NUISANCE) <= 4 and len(nuisance) <= 3
                  and not counts["q75_regression"]["new_comparable"] and not counts["q87_unchanged"]["new_comparable"])
        results[str(quality)] = {"by_role": counts, "nuisance_groups": len(nuisance), "passed": passed}
    return results


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("stage", choices=("freeze", "measure", "replay"))
    cli.add_argument("run", type=Path)
    cli.add_argument("--cache", type=Path, default=Path("backend/data/independent-evaluation/image-fresh-source-cache-v39"))
    args = cli.parse_args()
    cv2.setNumThreads(1)
    root = Path(__file__).resolve().parents[2]
    run = args.run
    method_paths = (Path(__file__), Path(image_reference.__file__), Path(reference_crop_resize.__file__),
                    Path(reference_resize.__file__), Path(reference_alignment.__file__), Path(reference_compare.__file__))
    method = {str(path.relative_to(root)): digest(path.read_bytes()) for path in method_paths}
    if args.stage == "freeze":
        selected, exposure = sources(args.cache, root)
        run.mkdir(mode=0o700)
        cases = collect(selected, run)
        write(run / "sources.json", selected)
        write(run / "cases.json", cases)
        write(run / "freeze.json", {"schema": SCHEMA, "design": DESIGN, "exposure": exposure, "method": method,
              "runtime": {"pillow": pillow_version, "numpy": np.__version__, "opencv": cv2.__version__},
              "files": {name: digest((run / name).read_bytes()) for name in ("sources.json", "cases.json")}})
        print(digest((run / "freeze.json").read_bytes()), len(cases), flush=True)
        return
    frozen = read(run / "freeze.json")
    assert frozen["method"] == method
    assert all(digest((run / name).read_bytes()) == value for name, value in frozen["files"].items())
    cases = read(run / "cases.json")
    outcomes = []
    started = time.monotonic()
    for case in cases:
        outcomes.append(score(case, run))
        if len(outcomes) % 24 == 0:
            print(len(outcomes), len(cases), round(time.monotonic() - started, 1), flush=True)
    result = {"outcomes": outcomes, "summary": summarize(cases, outcomes)}
    if args.stage == "measure":
        write(run / "result.json", result)
    else:
        assert read(run / "result.json") == result
    print(json.dumps(result["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
