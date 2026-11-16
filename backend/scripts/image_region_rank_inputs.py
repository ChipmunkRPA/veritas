"""Prepare paired small generative JPEG edits and their known native regions."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
from pathlib import Path
from types import SimpleNamespace

from scripts.video_face_temporal_fit import read, save, sha

SCHEMA = "chipmunk.image-region-ranking-deliveries/1"
POSITIVE = ("sharp", "feather")
QUALITY = (75, 90)
CONTROLS = ("ordinary", "local_blur", "local_tone", "local_copy")


def tgif_sources(root):
    from scripts.image_jpeg_signed_transitions import PARTS, archive_pairs, sources

    groups = sources(SimpleNamespace(sources=root))
    manifests = {name: sha(root / name / "manifest.json") for panels in PARTS.values() for name, _ in panels}
    for role in ("fit", "calibration", "held"):
        yield from ((role, source, family, original, edited, exact, manifests)
                    for source, family, original, edited, exact in archive_pairs(groups, role))


def bg_sources(qualification, media, index):
    import numpy as np
    from PIL import Image

    data = read(qualification)
    for row in data["qualified"]:
        credit = row["source"]["photo_attribution"]
        if credit["License"] != "https://creativecommons.org/licenses/by/2.0/":
            raise ValueError("Photo permission differs from the evaluation scope")
        item = row["measurement"]
        arrays = []
        for key in ("original", "edited", "semantic_mask"):
            file = item["files"][key]
            path = media / file["file"]
            if sha(path) != file["sha256"]:
                raise ValueError("A transfer asset differs from the qualified download")
            with Image.open(path) as image:
                arrays.append(np.array(image.convert("L" if key == "semantic_mask" else "RGB")))
        original, edited, intent = arrays
        exact = np.any(original != edited, axis=2)
        if np.any(exact & (intent == 0)) or not .001 <= exact.mean() <= .08:
            raise ValueError("Transfer mask does not bind the small local edit")
        release = index[item["files"]["edited"]["repo_path"]]
        if release["sha256"] != item["files"]["edited"]["sha256"] or release["family"] != "F5":
            raise ValueError("Transfer generator record differs from the verified asset")
        family = release["generator"]
        yield "transfer", item["files"]["original"]["sha256"], family, original, edited, exact, {
            qualification.parent.name: sha(qualification),
        }


def views(original, edited, exact):
    import cv2
    import numpy as np
    from PIL import Image, ImageEnhance, ImageFilter

    raw = Image.fromarray(original)
    blur = np.asarray(raw.filter(ImageFilter.GaussianBlur(1.4)))
    tone = np.clip(np.asarray(ImageEnhance.Color(raw).enhance(.72), dtype=np.int16) + [12, -6, 7], 0, 255)
    # Choose a translated same-photo donor with precisely the same blend support.
    copy = np.roll(original, shift=(original.shape[0] // 4, original.shape[1] // 4), axis=(0, 1))
    distance = cv2.distanceTransform(exact.astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    alpha = (.5 * (1 - np.cos(np.clip(distance / 8, 0, 1) * np.pi)))[..., None]
    changes = {key: np.rint(original * (1 - alpha) + value * alpha).astype(np.uint8)
               for key, value in (("local_blur", blur), ("local_tone", tone), ("local_copy", copy), ("feather", edited))}
    return {"ordinary": original, "sharp": edited, **changes}


def prepare(tgif, transfer, study, *, publisher_index=None):
    from itertools import chain

    import cv2
    import numpy as np
    from PIL import Image

    cv2.setNumThreads(1)
    study.mkdir(mode=0o700)
    groups, manifests, images, ids, rgb_hashes = [], {}, [], set(), set()
    index = {}
    if transfer:
        if publisher_index is None:
            raise ValueError("Transfer requires the pinned publisher metadata")
        with publisher_index.open(newline="") as stream:
            index = {row["image_path"]: row for row in csv.DictReader(stream)}
        manifests["bg-real-index"] = sha(publisher_index)
    cases = chain(tgif_sources(tgif), *(bg_sources(receipt, media, index) for receipt, media in transfer))
    for split, source, family, original, edited, exact, pins in cases:
        if (not np.array_equal(np.any(original != edited, axis=2), exact) or not .001 <= exact.mean() <= .08
                or min(exact.shape) < 280 or max(exact.shape) > 1280):
            raise ValueError(f"Out-of-scope source in reserved {split} group")
        rgb = hashlib.sha256(original.tobytes()).hexdigest()
        if source in ids or rgb in rgb_hashes:
            raise ValueError("An identical original appears in more than one role")
        ids.add(source)
        rgb_hashes.add(rgb)
        manifests.update(pins)
        tag = f"{split}-{len(groups):03d}"
        folder = study / tag
        folder.mkdir(mode=0o700)
        groups.append({"tag": tag, "source": source, "rgb_sha256": rgb, "family": family, "split": split,
                       "shape": list(exact.shape), "first_changed_pixels": int(exact.sum())})
        for role, pixels in views(original, edited, exact).items():
            changed = np.any(original != pixels, axis=2) if role in POSITIVE else np.zeros_like(exact)
            mask = folder / f"{role}-truth.png"
            Image.fromarray(changed.astype(np.uint8) * 255).save(mask, "PNG")
            for quality in QUALITY:
                relative = f"{tag}/{role}-{quality}.jpg"
                with io.BytesIO() as output:
                    Image.fromarray(pixels).save(output, "JPEG", quality=quality, subsampling=2)
                    body = output.getvalue()
                (study / relative).write_bytes(body)
                images.append({"group": tag, "role": role, "quality": quality, "file": relative,
                               "sha256": hashlib.sha256(body).hexdigest(), "mask": f"{tag}/{mask.name}",
                               "mask_sha256": sha(mask), "changed_pixels": int(changed.sum())})
        if len(groups) % 12 == 0:
            print(f"Verified and rendered {len(groups)} independent source groups.", flush=True)
    save(study / "manifest.json", {"schema": SCHEMA, "groups": groups, "images": images, "source_manifests": manifests})
    print(f"Frozen {len(images)} JPEG deliveries from {len(groups)} source groups.", flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("tgif", type=Path)
    cli.add_argument("study", type=Path)
    cli.add_argument("--transfer", nargs=2, type=Path, action="append", default=[])
    cli.add_argument("--publisher-index", type=Path)
    args = cli.parse_args()
    prepare(args.tgif, args.transfer, args.study, publisher_index=args.publisher_index)
