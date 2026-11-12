# SPDX-License-Identifier: Apache-2.0
"""Freeze and replay compact embedded-preview extent against exact source masks."""

from __future__ import annotations

import argparse
import inspect
import io
import json
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from app.evidence import image_thumbnail_evidence as exhibits
from app.provenance import image_thumbnail_localization as detector
from scripts import image_background_patch_score as paired
from scripts import image_edge_spread_transfer as regional
from scripts import image_thumbnail_localization_transfer as earlier
from scripts import image_thumbnail_orientation_transfer as orientation

ROOT = earlier.ROOT
SCHEMA = "chipmunk.thumbnail-extent-transfer/1"
sha, read, write, require = earlier.sha, earlier.read, earlier.write, earlier.REQUIRE
DELIVERIES = (("upright", 1, 0), ("mirror-stored", 2, 2), ("rotate-display", 6, 0))
RULE = {"added_groups": 3, "added_generated_groups": 1, "added_conventional_groups": 2, "maximum_lost": 1,
        "maximum_new_ordinary_groups": 0, "maximum_changed_ordinary_groups": 1, "maximum_incompatible": 0}
NORMAL = {"unchanged", "jpeg75", "global_tone", "resize", "camera", "camera_jpeg75", "camera_global_tone"}
INCOMPATIBLE = {"foreign_preview", "no_preview"}
SALT = "embedded-preview-compact-extent-2026-11-12/1/"


def wavelet_sources(archive):
    source = {r["slot"]: r for r in read(archive / "plan.json")["cases"]}
    previous = set()
    for suffix in ("", "-confirmation"):
        data = read(ROOT / f"image-thumbnail-localization{suffix}-2026-11-12" / "freeze.json")
        previous.update(row["source_sha256"] for row in data["records"])
    count = Counter()
    for result, images, known in paired.inputs(archive):
        row = source[result["slot"]]
        if sha(str(row["flickr_photo_id"]).encode()) in previous:
            continue
        family = result["generator"]
        variants = regional.native_variants(images, result["box"])
        before, after = variants["original"], variants["feathered_edit"]
        changed = np.any(before != after, axis=2)
        credit = row["photo_attribution"]
        require(changed.any() and not np.any(changed & ~known) and credit["License"].rstrip("/") == "https://creativecommons.org/licenses/by/2.0",
                "Alternate generator source, rights or exact patch commitment changed")
        group = f"{family}-{result['slot']}"
        count[family] += 1
        yield group, {"family": family, "author": credit["Author"], "page": credit["OriginalLandingURL"],
                      "license": credit["License"], "source_rgb_sha256": sha(before.tobytes())}, before, after
    require(count == {"sdxl": 3, "kandinsky": 3}, "Expected three unused sources from each alternate generator")


def compact_sources(archive):
    previous = read(ROOT / "image-flux-photo-expansion-v285" / "summary.json")
    require(previous["status"] == "complete" and previous["source_groups"] == 6, "Earlier compact roster unavailable")
    prior_inputs = read(ROOT / "image-flux-photo-expansion-v285" / "private-inputs.json")["rows"]
    prior_hashes = {row["source_jpeg_sha256"] for row in prior_inputs}
    prior_pages = {row["attribution"]["photo_page"] for row in prior_inputs}
    output = archive / "authored" / "output"
    private = {row["slot"]: row for row in read(archive / "private-inputs.json")["rows"]}
    rows = read(output / "result.json")["cases"]
    require(len(rows) == 2, "Expected two disjoint compact generated sources")
    for row in rows:
        slot = row["slot"]
        spec, credit = private[slot], private[slot]["attribution"]
        require(spec["source_jpeg_sha256"] not in prior_hashes and credit["photo_page"] not in prior_pages
                and credit["license_url"].rstrip("/") == "https://creativecommons.org/licenses/by/2.0", "Compact source identity or rights changed")
        source, composed = (archive / slot / "source.png").read_bytes(), (output / slot / "composite.png").read_bytes()
        require(sha(source) == spec["png_sha256"]["source"] and sha(composed) == row["composite_png_sha256"], "Compact source file changed")
        before, after = [rgb(data) for data in (source, composed)]
        require(sha(before.tobytes()) == row["source_rgb_sha256"] and sha(after.tobytes()) == row["composite_rgb_sha256"]
                and np.any(before != after, axis=2).sum() == row["changed_pixels"], "Compact RGB changed")
        yield f"flux-{slot}", {"family": "flux", "author": credit["author_name"], "page": credit["photo_page"],
                               "license": credit["license_url"], "source_rgb_sha256": row["source_rgb_sha256"]}, before, after


def rgb(data):
    with Image.open(io.BytesIO(data)) as image:
        return np.asarray(image.convert("RGB"))


def commons_sources(archive):
    plan = read(archive / "plan.json")["sources"]
    acquired = read(archive / "acquisition.json")["attempts"]
    rows = sorted(zip(plan, acquired, strict=True), key=lambda pair: sha((SALT + pair[0]["page"]).encode()))
    selected, used = [], set()
    for credit, item in rows:
        author = credit["author"].casefold()
        if (item["delivery"] != "thumb" or item["status"] != "qualified" or author in used
                or min(item["size"]) < 256 or len(selected) >= 10):
            continue
        require(credit["licence_url"].startswith(("https://creativecommons.org/licenses/by/", "https://creativecommons.org/licenses/by-sa/")),
                "Commons reference lacks commercial-use terms")
        data = (archive / item["name"]).read_bytes()
        raw = rgb(data)
        require(sha(data) == item["sha256"] and sha(raw.tobytes()) == item["rgb_sha256"], "Commons source file or RGB changed")
        y, x = [(length - 256) // 2 for length in raw.shape[:2]]
        crop = raw[y:y + 256, x:x + 256].copy()
        used.add(author)
        selected.append((f"commons-{item['slot']}", {"family": "conventional", "author": credit["author"], "page": credit["page"],
                         "license": credit["licence_url"], "source_rgb_sha256": item["rgb_sha256"], "source_sha256": item["sha256"],
                         "native_crop_sha256": sha(crop.tobytes())}, crop))
    require(len(selected) == 10, "Expected ten separately credited ordinary source photos")
    return selected


def authored(photo, slot):
    # One exact compact copy from a separate area in the same photo and one feathered color adjustment.
    y, x = (84 + slot % 3 * 14), (0 if slot % 2 else 90)
    copy = photo.copy()
    copy[y:y + 64, x:x + 64] = photo[15:79, 181:245, :][:, ::-1]
    native = photo.astype(np.float32)
    axis = np.arange(64)
    distance = np.minimum(axis, axis[::-1])
    alpha = np.minimum(distance[:, None], distance)[:, :, None].clip(0, 9) / 9
    retouch = native.copy()
    retouch[y:y + 64, x:x + 64] += alpha * np.array((36., -8., -27.)) * (-1 if slot % 2 else 1)
    return copy, np.rint(retouch.clip(0, 255)).astype(np.uint8)


def group_cases(group, photo, changes, index, foreign):
    earlier_image = Image.fromarray(photo)
    edge = (144, 168, 192)[index % 3]
    preview = earlier.thumbnail(earlier_image, edge)
    for role, edit in changes:
        changed = np.any(photo != edit, axis=2)
        for delivery, tag, operation in DELIVERIES:
            image = Image.fromarray(edit)
            thumb = preview if not operation else earlier.thumbnail(earlier_image.transpose(detector.TRANSFORMS[operation]), edge)
            if tag == 1:
                data = earlier.jpeg(image, 90, thumb)
            else:
                inverse = {2: Image.Transpose.FLIP_LEFT_RIGHT, 6: Image.Transpose.ROTATE_90}[tag]
                data = orientation.encode(image.transpose(inverse), tag, thumb)
            require(earlier.preview_from(data) == thumb, "Embedded preview bytes changed")
            yield (group, delivery, role), data, changed, operation
    tone = Image.fromarray(np.rint((photo.astype(np.float32) * .92 + 10).clip(0, 255)).astype(np.uint8))
    reduced = earlier_image.resize(tuple(round(n * .72) for n in earlier_image.size), Image.Resampling.LANCZOS)
    resized = reduced.resize(earlier_image.size, Image.Resampling.BICUBIC)
    false_thumb = earlier.thumbnail(Image.fromarray(foreign).resize(earlier_image.size), edge)
    for role, image, quality, thumb in (("unchanged", earlier_image, 90, preview), ("jpeg75", earlier_image, 75, preview),
                                       ("global_tone", tone, 90, preview), ("resize", resized, 90, preview),
                                       ("foreign_preview", earlier_image, 90, false_thumb), ("no_preview", earlier_image, 90, None)):
        yield (group, "upright", role), earlier.jpeg(image, quality, thumb), None, 0 if role in NORMAL else None


def cameras(archive):
    acquisition = read(archive / "acquisition.json")
    require(acquisition["license"] == "https://creativecommons.org/licenses/by/4.0/" and len(acquisition["files"]) == 10,
            "Missing camera originals or publisher terms")
    credits, cases = {}, []
    for slot, row in enumerate(acquisition["files"]):
        body = (archive / row["name"]).read_bytes()
        require(len(body) == row["bytes"] and sha(body) == row["sha256"], "Original camera bytes changed")
        group = f"pixel-{slot:02d}"
        with Image.open(io.BytesIO(body)) as image:
            preview = earlier.preview_from(body)
            require(image.getexif().get(272) == "Pixel 6 Pro" and preview is not None, "Camera retained preview unavailable")
            image.draft("RGB", (1920, 1440))
            earlier_image = image.convert("RGB")
        tone = Image.fromarray(np.rint((np.asarray(earlier_image).astype(np.float32) * .95 + 7).clip(0, 255)).astype(np.uint8))
        credits[group] = {"family": "camera", "author": acquisition["authors"][0], "page": row["url"],
                          "license": acquisition["license"], "source_sha256": row["sha256"], "same_contributor_group": "pixel-camera"}
        for role, source in (("camera", body), ("camera_jpeg75", earlier.jpeg(earlier_image, 75, preview)),
                             ("camera_global_tone", earlier.jpeg(tone, 90, preview))):
            cases.append(((group, "upright", role), source, None, 0))
    return credits, cases


def implementation():
    paths = [Path(__file__), *(Path(module.__file__) for module in (detector, earlier, orientation, paired, regional))]
    geometry = inspect.getsource(exhibits._stored_preview_bounds) + inspect.getsource(exhibits._preview_bounds)
    return {p.name: sha(p.read_bytes()) for p in paths} | {"evidence_crop_geometry": sha((geometry + str(exhibits.MAX_REGIONS)).encode())}


def freeze(args):
    args.run.mkdir(mode=0o700)
    objects = args.run / "objects"
    objects.mkdir()
    sources = list(wavelet_sources(args.wavelet)) + list(compact_sources(args.compact))
    ordinary = commons_sources(args.commons)
    credits = {group: credit for group, credit, *_ in sources + ordinary}
    require(len({row["author"].casefold() for row in credits.values()}) == len(credits) == 18
            and len({row["source_rgb_sha256"] for row in credits.values()}) == 18, "Source roles overlap")
    captures, camera_cases = cameras(args.camera)
    groups = [(group, photo, (("generated", edit),)) for group, _, photo, edit in sources]
    groups.extend((group, photo, tuple(zip(("copy", "local_color"), authored(photo, index), strict=True)))
                  for index, (group, _, photo) in enumerate(ordinary))
    cases = list(camera_cases)
    for slot, (group, photo, edits) in enumerate(groups):
        cases.extend(group_cases(group, photo, edits, slot, groups[(slot + 1) % len(groups)][1]))
    rows = []
    for slot, (key, data, truth, operation) in enumerate(cases):
        (objects / f"{slot:03d}.jpg").write_bytes(data)
        if truth is not None:
            np.save(objects / f"{slot:03d}.npy", truth, allow_pickle=False)
        rows.append({"slot": slot, "key": key, "input_sha256": sha(data), "expected_preview_orientation": operation,
                     "truth_sha256": None if truth is None else sha(truth.tobytes()), "target": 0 if truth is None else int(truth.sum())})
    write(args.run / "freeze.json", {"schema": SCHEMA, "criteria": RULE, "scope": {"confidence_weight": 0,
          "extent": [detector.EXTENT_FRACTION, detector.EXTENT_MAX_AXIS_SHARE, detector.EXTENT_MAX_GROWTH]},
          "code": implementation(), "sources": {**credits, **captures}, "cases": rows})
    print(f"Sealed {len(rows)} inputs. SHA-256 {sha((args.run / 'freeze.json').read_bytes())}", flush=True)


def measure(data, truth, extend):
    response, masks = detector.measure(data, minimum_component_fraction=detector.CONFIRMATION_COMPONENT_FRACTION, extend=extend)
    if masks is None:
        return {"status": response["status"], "count": 0, "mask_sha256": None, "boxes": [], "orientation": None,
                "mask": [0, 0], "boxes_overlap": [0, 0]}
    mask = np.zeros_like(masks[0])
    trims = response["trim"]
    records = []
    for region in response["regions"][:exhibits.MAX_REGIONS]:
        x, y, w, h = (region["embedded_thumbnail"][k] for k in exhibits.BOX_KEYS)
        x, y = x - trims[0], y - trims[1]
        bounds = np.s_[y:y + h, x:x + w]
        mask[bounds] = masks[0][bounds]
        records.append({"main": region["submitted"], "preview": exhibits._stored_preview_bounds(region, response)})
    values = earlier.area(mask, truth)
    # Measure the exact source rectangles that the app will publish as evidence.
    extent = np.zeros((response["native_size"][1], response["native_size"][0]), dtype=bool) if truth is not None else None
    for box in records:
        x, y, w, h = (box["main"][key] for key in exhibits.BOX_KEYS)
        if extent is not None:
            extent[y:y + h, x:x + w] = True
    return {"status": response["status"], "count": int(mask.sum()), "mask_sha256": sha(mask.tobytes()), "boxes": records,
            "orientation": response.get("preview_orientation", 0),
            "mask": [values["selected_native"] or 0, values["native_overlap"]],
            "boxes_overlap": [int(extent.sum()), int((extent & truth).sum())] if extent is not None else [0, 0]}


def useful(values, target):
    selected, overlap = values
    return target > 0 and overlap * 2 >= target and 10 * overlap >= 3 * (selected + target - overlap)


def summary(rows, sources):
    counts, groups = {}, {name: set() for name in ("added", "generated", "conventional", "new_ordinary", "changed_ordinary")}
    losses = incompatible = wrong = 0
    for row in rows:
        group, _, role = row["key"]
        source_group = sources[group].get("same_contributor_group", group)
        old, new, target = row["baseline"], row["candidate"], row["target"]
        cell = counts.setdefault(role, Counter(total=0, baseline_boxes=0, candidate_boxes=0, baseline_mask=0, candidate_mask=0,
                                                baseline_flags=0, candidate_flags=0, extent_changes=0))
        cell["total"] += 1
        cell["baseline_flags"] += bool(old["count"])
        cell["candidate_flags"] += bool(new["count"])
        cell["extent_changes"] += old["mask_sha256"] != new["mask_sha256"]
        if role in INCOMPATIBLE:
            incompatible += new["status"] == "comparable"
        elif new["status"] == "comparable":
            wrong += new["orientation"] != row["expected_preview_orientation"]
        if role in NORMAL:
            if new["count"] and not old["count"]:
                groups["new_ordinary"].add(source_group)
            if old["mask_sha256"] != new["mask_sha256"]:
                groups["changed_ordinary"].add(source_group)
        elif target:
            base, candidate = [useful(values["boxes_overlap"], target) for values in (old, new)]
            cell["baseline_boxes"] += base
            cell["candidate_boxes"] += candidate
            cell["baseline_mask"] += useful(old["mask"], target)
            cell["candidate_mask"] += useful(new["mask"], target)
            losses += base and not candidate
            if candidate and not base:
                groups["added"].add(group)
                groups["generated" if role == "generated" else "conventional"].add(group)
    passed = (len(groups["added"]) >= RULE["added_groups"] and len(groups["generated"]) >= RULE["added_generated_groups"]
              and len(groups["conventional"]) >= RULE["added_conventional_groups"] and losses <= RULE["maximum_lost"]
              and len(groups["new_ordinary"]) <= RULE["maximum_new_ordinary_groups"]
              and len(groups["changed_ordinary"]) <= RULE["maximum_changed_ordinary_groups"] and incompatible <= RULE["maximum_incompatible"] and not wrong)
    return {"by_role": counts, "groups": {key: sorted(values) for key, values in groups.items()}, "lost_useful_views": losses,
            "incompatible_comparisons": incompatible, "wrong_preview_orientation": wrong, "gate_passed": passed}


def score(args):
    frozen = args.run / "freeze.json"
    sealed = read(frozen)
    require(sealed["code"] == implementation(), "Frozen implementation changed")
    rows = []
    start = time.monotonic()
    for row in sealed["cases"]:
        slot = row["slot"]
        data = (args.run / "objects" / f"{slot:03d}.jpg").read_bytes()
        require(sha(data) == row["input_sha256"], "Frozen JPEG differs")
        truth = None if row["truth_sha256"] is None else np.load(args.run / "objects" / f"{slot:03d}.npy", allow_pickle=False)
        require(truth is None or (truth.dtype == bool and sha(truth.tobytes()) == row["truth_sha256"]
                                 and truth.sum() == row["target"]), "Native pixel truth differs")
        rows.append({**row, "baseline": measure(data, truth, False), "candidate": measure(data, truth, True)})
    outcome = summary(rows, sealed["sources"])
    write(args.run / "result.json", {"freeze_sha256": sha(frozen.read_bytes()), "frozen": sealed, "summary": outcome,
                                     "cases": rows, "elapsed_seconds": round(time.monotonic() - start, 3)})
    print(json.dumps(outcome, sort_keys=True), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "score"))
    parser.add_argument("run", type=Path)
    parser.add_argument("--wavelet", type=Path)
    parser.add_argument("--compact", type=Path, default=ROOT / "image-flux-photo-preparation-v276")
    parser.add_argument("--commons", type=Path, default=ROOT / "image-affine-native-scale-2026-11-11")
    parser.add_argument("--camera", type=Path, default=ROOT / "image-gainmap-localization-2026-11-12")
    config = parser.parse_args()
    cv2.setNumThreads(1)
    if config.stage == "freeze":
        require(config.wavelet is not None, "Supply the retained alternate-generation holdout")
        freeze(config)
    else:
        score(config)
