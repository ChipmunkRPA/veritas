# SPDX-License-Identifier: Apache-2.0
"""Build disjoint, reproducible JPEG removals from licensed public photographs."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath

from scripts import image_bg_real_roster_v218 as roster
from scripts import image_cross_quality_sources as sources
from scripts import image_region_rank_inputs as files

SCHEMA = "chipmunk.image-jpeg-band-sources/1"
TARGET = 24
EDITOR_WEIGHTS = {
    "lama": {"file": "big-lama.pt", "sha256": "344c77bbcb158f17dd143070d1e789f38a66c04202311ae3a258ef66667a9ea9",
             "url": "https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt", "license": "Apache-2.0"},
    "migan": {"file": "migan.onnx", "sha256": "6f1f3530a1a2324b19752018ce756088b07973cda8d7d890034ace5c8a48c40b",
              "url": "https://huggingface.co/andraniksargsyan/migan/resolve/1538c13/migan_pipeline_v2.onnx", "license": "MIT"},
}


def exposures(retained):
    directory = sources.PUBLIC
    keys, fingerprints, pins = sources.known_sources(retained, directory / "image-removal-peer-sources-2026-11-16.json")
    further = directory / "image-cross-quality-sources-2026-11-16.json"
    for row in files.read(further)["accepted"]:
        keys.update(roster.source_keys(row["source"]))
        for record in row["fingerprints"].values():
            fingerprints[0].append(record)
            fingerprints[1].add(record["encoded_sha256"])
            fingerprints[2].add(record["rgb_sha256"])
    return keys, fingerprints, pins | {further.name: files.sha(further)}


def reserve(metadata, retained, output):
    import yaml

    excluded, _, pins = exposures(retained)
    hashes = {name: files.sha(metadata / name) for name in roster.METADATA}
    if any(digest != roster.METADATA[name][2] for name, digest in hashes.items()):
        raise ValueError("BG-REAL metadata differs from the reviewed release")
    photos = roster.indexed((row for row in roster.table(metadata / "attribution.csv") if row["role"] == "recipient"), "bgreal_id")
    people = roster.indexed(roster.table(metadata / "recipients.csv"), "source_image_id")
    splits = {name: set(values) for name, values in yaml.safe_load((metadata / "splits_v2.yaml").read_bytes()).items()}
    candidates = []
    for row in roster.table(metadata / "manifest_v2.csv"):
        if row["family"] != "A0" or row["label"] != "0" or not roster.base_png(row) or row["generator"] != "none":
            continue
        source = row["source_image_id"]
        split = row["split"]
        try:
            details = roster.one(people, source, "Need an original source")
            rights = roster.one(photos, source, "Need published attribution")
            owner, photo = roster.attribution(rights, source)
            size = roster.dimensions(row)
            area = float(details["mask_area"])
            if (not roster.SOURCE.fullmatch(source) or not roster.SHA.fullmatch(row["sha256"])
                    or row["sample_id"] != f"{source}_A0" or row["sample_id"] not in splits[split]
                    or size != roster.dimensions(details) or not 384 <= min(size) <= max(size) <= 768
                    or not .045 <= area <= .16):
                continue
            original = roster.named_png(row["image_path"], PurePosixPath("v2/images") / source[3], f"{source}_A0.png")
            mask = roster.named_png(row["mask_subject_path"], PurePosixPath("v2/masks/subject"), f"{source}_subject.png")
            candidates.append({"recipient": source, "publisher_split": split, "size": list(size), "source_mask_area": area,
                               "flickr_photo_id": photo, "flickr_owner": owner, "photo_attribution": rights,
                               "original": {"path": original, "sha256": row["sha256"]}, "subject": mask})
        except (ValueError, KeyError):
            continue
    qualified, _ = roster.select(candidates, excluded)
    qualified.sort(key=lambda row: (row["publisher_split"] not in roster.TEST_SPLITS,
                                    hashlib.sha256(f"{SCHEMA}/{row['recipient']}".encode()).hexdigest()))
    plan = {"schema": SCHEMA, "revision": roster.REVISION, "metadata": hashes, "exposure": pins,
            "editors": EDITOR_WEIGHTS, "target": TARGET, "eligible": len(qualified), "cases": qualified[:TARGET * 2]}
    files.save(output, plan)
    print(f"Reserved {len(plan['cases'])} photos from {len(qualified)} eligible groups", flush=True)


def acquire(plan_path, retained, output):
    import requests

    from scripts.image_bg_real_acquisition_v221 import fingerprint, near, png_pixels, prior_matches
    from scripts.image_bg_real_transport_v225 import download

    plan = files.read(plan_path)
    keys, known, pins = exposures(retained)
    if plan["schema"] != SCHEMA or plan["exposure"] != pins or plan["revision"] != roster.REVISION:
        raise ValueError("The source plan or earlier exposure changed")
    output.mkdir(mode=0o700)
    deadline = time.monotonic() + 240

    def fetch(row):
        try:
            with requests.Session() as session:
                session.trust_env = False
                image, _ = download(session, row["original"]["path"], row["original"]["sha256"], deadline)
                mask, _ = download(session, row["subject"], None, deadline)
            pixels = png_pixels(image, row["size"])
            signature = fingerprint(image, pixels)
            subject = png_pixels(mask, row["size"])[..., 0] > 127
            if abs(float(subject.mean()) - row["source_mask_area"]) > .003:
                raise ValueError("Decoded subject mask differs from its published area")
            if prior_matches(row, {"original": signature}, keys, known):
                raise ValueError("Source matches an earlier photo or file")
            return row, signature, image, mask, None
        except (OSError, ValueError, requests.RequestException) as error:
            return row, None, None, None, f"{type(error).__name__}: {error}"

    accepted, failed = [], Counter()
    with ThreadPoolExecutor(max_workers=4) as pool:
        for row, signature, image, mask, error in pool.map(fetch, plan["cases"]):
            if len(accepted) == plan["target"]:
                break
            if signature and any(near(signature, former["fingerprints"]["original"]) for former in accepted):
                error = "Within-panel near duplicate"
            if error:
                failed[error] += 1
                continue
            tag = f"native-{len(accepted):02d}"
            (output / f"{tag}.png").write_bytes(image)
            (output / f"{tag}-subject.png").write_bytes(mask)
            accepted.append({"source": row, "tag": tag, "fingerprints": {"original": signature},
                             "subject_sha256": hashlib.sha256(mask).hexdigest()})
    files.save(output / "intake.json", {"schema": SCHEMA, "plan_sha256": files.sha(plan_path), "cases": accepted, "exclusions": dict(failed)})
    print(json.dumps({"retained": len(accepted), "exclusions": failed}, sort_keys=True), flush=True)


def editors(root):
    import numpy as np
    import onnxruntime as ort
    import torch

    torch.set_num_threads(2)
    paths = {name: root / item["file"] for name, item in EDITOR_WEIGHTS.items()}
    if any(files.sha(path) != EDITOR_WEIGHTS[name]["sha256"] for name, path in paths.items()):
        raise ValueError("Object-removal fixture weights differ from the reviewed releases")
    config = ort.SessionOptions()
    config.intra_op_num_threads = 2
    graph = ort.InferenceSession(str(paths["migan"]), config, providers=["CPUExecutionProvider"])
    lama = torch.jit.load(str(paths["lama"]), map_location="cpu").eval()

    def fourier(image, remove):
        height, width = remove.shape
        pad = ((0, -height % 8), (0, -width % 8))
        frame = np.pad(image, (*pad, (0, 0)), mode="reflect")
        mask = np.pad(remove, pad).astype(np.float32)
        with torch.inference_mode():
            result = lama(torch.from_numpy(frame.transpose(2, 0, 1)[None].copy()).float() / 255,
                          torch.from_numpy(mask[None, None].copy()))[0].numpy().transpose(1, 2, 0)
        return np.clip(np.rint(result[:height, :width] * 255), 0, 255).astype(np.uint8)

    def mobile(image, remove):
        result, = graph.run(None, {"image": np.ascontiguousarray(image.transpose(2, 0, 1)[None]),
                                   "mask": np.ascontiguousarray((~remove)[None, None].astype(np.uint8) * 255)})
        return result[0].transpose(1, 2, 0)

    return {"lama": fourier, "migan": mobile}


def small_subject_area(subject, digest):
    """Choose a compact portion on one side of the annotated subject."""
    import numpy as np

    points = np.argwhere(subject)
    axis = int(np.ptp(points, axis=0).argmax())
    percentile = .20 if int(digest[0], 16) % 2 else .80
    position = np.quantile(points[:, axis], percentile)
    nearest = np.argsort(np.abs(points[:, axis] - position), kind="stable")[:max(4, len(points) // 8)]
    center = np.median(points[nearest], axis=0)
    distance = ((points - center) ** 2).sum(axis=1)
    selected = points[np.argsort(distance, kind="stable")[:int(subject.size * .027)]]
    compact = np.zeros_like(subject)
    compact[tuple(selected.T)] = True
    return compact


def prepare(intake, models, output, *, partial=False):
    import cv2
    import numpy as np

    from scripts.image_bg_real_acquisition_v221 import fingerprint, png_pixels

    assembler = editors(models)
    dossier = files.read(intake / "intake.json")
    output.mkdir(mode=0o700)
    cv2.setNumThreads(1)
    groups, rows, receipts = [], [], []
    for index, record in enumerate(dossier["cases"]):
        key, row = record["tag"], record["source"]
        raw = (intake / f"{key}.png").read_bytes()
        mask = (intake / f"{key}-subject.png").read_bytes()
        original = png_pixels(raw, row["size"])
        if fingerprint(raw, original) != record["fingerprints"]["original"] or hashlib.sha256(mask).hexdigest() != record["subject_sha256"]:
            raise ValueError("Held image or subject changed after acquisition")
        subject = png_pixels(mask, row["size"])[..., 0] > 127
        if partial:
            subject = small_subject_area(subject, row["original"]["sha256"])
        support = cv2.dilate(subject.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))) > 0
        yy, xx = np.where(support)
        height, width = support.shape
        y0, y1 = max(0, yy.min() - 96), min(height, yy.max() + 97)
        x0, x1 = max(0, xx.min() - 96), min(width, xx.max() + 97)
        for name, remove in assembler.items():
            crop = (slice(y0, y1), slice(x0, x1))
            generated = remove(original[crop], support[crop])
            if generated.shape != original[crop].shape or generated.dtype != np.uint8:
                raise ValueError("Editor changed the intended source geometry or sample type")
            edited = original.copy()
            edited[crop] = np.where(support[crop][..., None], generated, original[crop])
            truth = (original != edited).any(axis=-1)
            if np.any(truth & ~support) or not .01 < truth.mean() < (.08 if partial else .23):
                raise ValueError("Rendered object replacement differs from the bounded subject mask")
            prepared = record | {"group": f"confirmation-{name}-{index:02d}", "source": row | {"generator": name}}
            group, encoded = sources.render(output, prepared, {"original": original, "edited": edited}, truth)
            groups.append(group | {"split": "confirmation"})
            rows.extend(encoded)
            receipts.append({"group": group["tag"], "editor": name, "source": record["fingerprints"]["original"]["rgb_sha256"],
                             "region": [int(x0), int(y0), int(x1), int(y1)], "pixels": int(truth.sum())})
        print(f"Rendered both held removals from {index + 1}/{len(dossier['cases'])} photos", flush=True)
    files.save(output / "manifest.json", {"schema": files.SCHEMA, "groups": groups, "images": rows,
               "source_manifests": {"intake": files.sha(intake / "intake.json")}})
    record = {"schema": SCHEMA, "editors": EDITOR_WEIGHTS, "rows": receipts}
    if partial:
        record["scope"] = "partial_subject"
    files.save(output / "render-receipt.json", record)


if __name__ == "__main__":
    command = argparse.ArgumentParser(description=__doc__)
    modes = command.add_subparsers(dest="step", required=True)
    for mode, fields in (("reserve", ("metadata", "retained", "output")), ("acquire", ("plan_path", "retained", "output")),
                         ("prepare", ("intake", "models", "output"))):
        stage = modes.add_parser(mode)
        for name in fields:
            stage.add_argument(name, type=Path)
        if mode == "prepare":
            stage.add_argument("--partial", action="store_true", help="Render an independently fixed smaller part of the same subject")
    values = vars(command.parse_args())
    {"reserve": reserve, "acquire": acquire, "prepare": prepare}[values.pop("step")](**values)
