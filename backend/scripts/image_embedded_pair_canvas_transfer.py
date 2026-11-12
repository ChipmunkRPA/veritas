# SPDX-License-Identifier: Apache-2.0
"""Freeze and measure geometry-only expansion of the exact single-image comparison."""
from __future__ import annotations

import argparse
import hashlib
import json
import resource
import sys
import time
from collections import Counter
from pathlib import Path

SCHEMA = "chipmunk.image-embedded-pair-canvas-transfer/1"
SIZES = ((768, 768), (1024, 768), (1280, 720), (1366, 768), (1920, 1080))
ROLES = ("sharp", "feather", "ordinary_local", "duplicate", "unrelated", "third_full", "third_partial",
         "third_064", "third_080", "third_118")
VALID = ("sharp", "feather", "ordinary_local")
GUARDS = ROLES[5:]
CONTROLS = ("duplicate", "unrelated", "screenshot", "standalone")
MIN_RSS_LIMIT = 512 * 1024**2
RULE = {"minimum_exact_sharp": 6, "minimum_exact_feather": 6, "minimum_exact_ordinary_local": 10,
        "retained_exact_group_fraction": .75, "incorrect_valid_masks": 0, "reports_on_ordinary_controls": 0,
        "reports_with_additional_third": 0, "minimum_unguarded_third_challenges": 3,
        "maximum_per_case_seconds": 5.0, "maximum_peak_rss_bytes": MIN_RSS_LIMIT}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def fs(path):
    return sha(path.read_bytes())


def read(path):
    return json.loads(path.read_bytes())


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def need(ok, detail):
    if not ok:
        raise ValueError(detail)


def load_png(path, checksum, mode="RGB"):
    import numpy as np
    from PIL import Image

    need(fs(path) == checksum, "Encoded source commitment differs")
    with Image.open(path) as image:
        need(image.format == "PNG" and image.mode == mode, "Unexpected source encoding")
        return np.asarray(image).copy()


def bounds(mask):
    import numpy as np

    y, x = np.nonzero(mask)
    return int(x.min()), int(y.min()), int(x.max() + 1), int(y.max() + 1)


def feather(before, after, mask):
    import cv2
    import numpy as np

    distance = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    alpha = (1 - np.cos(np.minimum(distance / 12, 1) * np.pi))[:, :, None] / 2
    return np.rint(before * (1 - alpha) + after * alpha).astype(np.uint8)


def author_id(value):
    return str(value).rstrip("/").split("/")[-1].casefold()


def photo_sources(data):
    import numpy as np

    flux = data / "image-flux-photo-expansion-v285"
    backup = data / "image-bg-real-backup-intake-v262"
    old = read(backup / "qualified-private.json")
    need(fs(backup / "qualified-private.json") == read(backup / "summary.json")["qualified_private_sha256"],
         "LaMa source roster changed")
    prepared = read(flux / "private-inputs.json")
    summary = read(flux / "summary.json")
    flux_rows = {}
    for batch in range(1, 4):
        output = flux / f"batch-{batch:02d}/authored/output"
        need(fs(output / "result.json") == summary["batch_receipts"][str(batch)]["result_sha256"],
             "Retained reference-edit receipt changed")
        result = read(output / "result.json")
        for row in result["cases"]:
            need(row["status"] == "qualified_geometry", "Retained edit failed geometry qualification")
            flux_rows[row["slot"]] = row, output / row["slot"]
    excluded_ids, excluded_authors = set(), set()
    for name in ("image-bg-real-acquisition-v221", "image-bg-real-acquisition-v225", "image-bg-real-backup-intake-v262"):
        for row in read(data / name / "qualified-private.json")["qualified"]:
            excluded_ids.add(str(row["source"]["flickr_photo_id"]))
            excluded_authors.add(author_id(row["source"]["flickr_owner"]))
    for root in ("image-flux-photo-preparation-v276", "image-flux-photo-expansion-v285"):
        for row in read(data / root / "private-inputs.json")["rows"]:
            excluded_ids.add(author_id(row["attribution"]["photo_page"]))
            excluded_authors.add(author_id(row["attribution"]["author_url"]))
    for row in prepared["rows"]:
        slot = row["slot"]
        attr = row["attribution"]
        need(attr["license_url"] == "https://creativecommons.org/licenses/by/2.0/", "Photo rights differ")
        authored, output = flux_rows[slot]
        before = load_png(flux / slot / "source.png", row["png_sha256"]["source"])
        after = load_png(output / "composite.png", authored["composite_png_sha256"])
        mask = load_png(output / "changed.png", authored["changed_png_sha256"], "L") > 0
        need(np.array_equal(mask, (before != after).any(axis=2)), "Reference-edit pixels differ")
        yield "flux", author_id(attr["photo_page"]), author_id(attr["author_url"]), before, after, mask
    ranked = sorted(old["qualified"], key=lambda row: sha(f"embedded-pair/1/{row['source']['flickr_photo_id']}".encode()))
    for row in ranked[8:]:
        cache = Path(old["cache"])
        inputs = row["measurement"]["files"]
        before, after = (load_png(cache / inputs[name]["file"], inputs[name]["sha256"]) for name in ("original", "edited"))
        mask = load_png(cache / row["truth_file"], row["truth_png_sha256"], "L") > 0
        need(np.array_equal(mask, (before != after).any(axis=2)), "LaMa native mask differs")
        x, y, right, bottom = bounds(mask)
        height, width = mask.shape
        left, top = min(max(0, (x + right - 272) // 2), width - 272), min(max(0, (y + bottom - 272) // 2), height - 272)
        window = np.s_[top:top + 272, left:left + 272]
        need(mask[window].sum() == mask.sum(), "Prepared photo pane loses known edited pixels")
        source = row["source"]
        yield "lama", str(source["flickr_photo_id"]), author_id(source["flickr_owner"]), before[window], after[window], mask[window]
    for slot, (image, identifier, author) in enumerate(extra_sources(data, excluded_ids, excluded_authors)):
        if slot >= 6:
            yield "standalone", identifier, author, image, None, None
            continue
        image = center_pane(image, 280, 280)
        after = image.copy()
        y, x = 75 + 9 * (slot % 4), 81 + 11 * (slot % 3)
        after[y:y + 44, x:x + 52] = image[16:60, 14:66]
        yield "authored_paste", identifier, author, image, after, (image != after).any(axis=2)


def extra_sources(data, excluded_ids, excluded_authors):
    import numpy as np
    from PIL import Image, ImageOps

    folder = data / "image-fresh-source-cache-v39"
    path = folder / "private-groups.json"
    need(fs(path) == read(folder / "summary.json")["private_groups_sha256"], "Retained photo inventory changed")
    roster = read(path)
    permitted = {member for group in roster["candidate_only_unlinked_components"] for member in group}
    accepted = []
    for row in sorted(roster["sources"], key=lambda row: sha(f"canvas-photo/1/{row['key']}".encode())):
        if row["key"] not in permitted or not row["key"].startswith("fresh-jpeg/") or row.get("license_id") != 4:
            continue
        identifier = author_id(row["photo_page"])
        author = author_id(row["author_profile"])
        if identifier in excluded_ids or author in excluded_authors or min(row["size"]) < 384:
            continue
        need(row["license_url"] == "https://creativecommons.org/licenses/by/2.0/", "Ordinary photo rights differ")
        source = Path(row["source_jpeg_file"])
        need(fs(source) == row["source_jpeg_sha256"], "An ordinary photo changed")
        with Image.open(source) as image:
            native = np.asarray(ImageOps.exif_transpose(image).convert("RGB")).copy()
        need(sha(native.tobytes()) == row["rgb_sha256"], "An ordinary native RGB photo changed")
        accepted.append((native, identifier, author))
        excluded_ids.add(identifier)
        excluded_authors.add(author)
        if len(accepted) == 12:
            return accepted
    raise ValueError("Twelve distinct attributed photo sources are needed")


def center_pane(photo, width, height):
    import numpy as np
    from PIL import Image, ImageOps

    return np.asarray(ImageOps.fit(Image.fromarray(photo), (width, height), Image.Resampling.BICUBIC)).copy()


def panel(size, slot, role, source, variant, third=None):
    import numpy as np

    width, height = size
    ph, pw = source.shape[:2]
    x = 20 + ((slot * 11 + width) % 19)
    y = 36 + ((slot * 7 + height) % 19)
    if width == 768:
        dx, dy = pw + 30 + ((slot * 7 + width) % 19), (slot % 11) - 5
    else:
        dx = pw + 39 + ((slot * 13 + height) % 19) + min(58, (width - 1024) // 12)
        dy = (slot * 7 + width) % 19 - 9 + (slot % 3) * 17
    a, b = (x, y, x + pw, y + ph), (x + dx, y + dy, x + dx + pw, y + dy + ph)
    bg = ((32, 42, 58), (245, 239, 231), (218, 232, 240))[slot % 3]
    rgb = np.full((height, width, 3), bg, np.uint8)
    for index, (box, photo) in enumerate(((a, source), (b, variant))):
        left, top, right, bottom = box
        rgb[top - 4:bottom + 4, left - 4:right + 4] = (121 + 12 * index, 133 + 6 * index, 149 - 7 * index)
        rgb[top:bottom, left:right] = photo
    if third is not None:
        h3, w3 = third.shape[:2]
        right_x = min(b[2] + 33 + slot * 11 % 17, width - w3 - 7)
        if width == 768 or slot % 2 == 0 or right_x < b[2] + 8:
            tx = 74 + slot * 23 % 81
            ty = min(max(a[3], b[3]) + 32 + (slot * 7) % 23, height - h3 - 4)
        else:
            tx = right_x
            ty = max(y - 18, 10) + (slot % 3) * 41
        third_box = (tx, ty, tx + w3, ty + h3)
        need(all(0 <= z <= stop for z, stop in zip(third_box, (width, height, width, height), strict=True))
             and (third_box[1] >= max(a[3], b[3]) + 8 or third_box[0] >= b[2] + 8),
             "A constructed third pane overlaps or is clipped")
        rgb[ty:ty + h3, tx:tx + w3] = third
    else:
        third_box = None
    return rgb, a, b, third_box


def cases(data, run, size=None):
    import numpy as np
    from PIL import Image, ImageFilter

    photos = list(photo_sources(data))
    need(len(photos) == 22 and len({row[1] for row in photos}) == len({row[2] for row in photos}) == 22,
         "A held photo or attribution group repeats")
    targets = [row for row in photos if row[0] != "standalone"]
    standalones = [row for row in photos if row[0] == "standalone"]
    viewports = SIZES if size is None else (size,)
    screenshots = read(run / "screenshots.json")["rows"]
    for dimensions in viewports:
        for slot, (family, _, _, before, after, mask) in enumerate(targets):
            ph, pw = before.shape[:2]
            pillow = Image.fromarray(before)
            adjusted = before.copy()
            window = np.s_[ph//3:ph//3 + 37, pw//3:pw//3 + 43]
            adjusted[window] = np.asarray(pillow.filter(ImageFilter.GaussianBlur(1.7)))[window]
            different = center_pane(standalones[slot % len(standalones)][3], pw, ph)
            blend = feather(before, after, mask)
            thirds = (before, before[:, (pw-152)//2:(pw+152)//2], *(np.asarray(pillow.resize(
                (round(pw*scale), round(ph*scale)), Image.Resampling.LANCZOS)) for scale in (.64, .80, 1.18)))
            alternatives = (after, blend, adjusted, before, different, *(adjusted for _ in GUARDS))
            for role, alternate, third in zip(ROLES, alternatives, (None,) * 5 + thirds, strict=True):
                rgb, a, b, third_box = panel(dimensions, slot, role, before, alternate, third)
                truth = (before != alternate).any(axis=2) if role in VALID else None
                yield dimensions, slot, role, family, rgb, a, b, third_box, truth
        for slot, (_, _, _, before, _, _) in enumerate(standalones):
            rgb = center_pane(before, *dimensions)
            yield dimensions, slot, "standalone", "credited_photo", rgb, None, None, None, None
        for slot, row in enumerate(entry for entry in screenshots if tuple(entry["size"]) == dimensions):
            rgb = load_png(run / row["filename"], row["png_sha256"])
            yield dimensions, slot, "screenshot", row["role"], rgb, None, None, None, None


def identity(case):
    size, slot, role, family, rgb, a, b, third, truth = case
    need(rgb.shape == (size[1], size[0], 3), "Native canvas size differs")
    return {"size": list(size), "slot": slot, "role": role, "family": family, "rgb_sha256": sha(rgb.tobytes()),
            "pane_a": list(a) if a else None, "pane_b": list(b) if b else None, "third": list(third) if third else None,
            "truth": int(truth.sum()) if truth is not None else None,
            "truth_sha256": sha(truth.tobytes()) if truth is not None else None}


def source_code():
    root = Path(__file__).resolve().parent
    return {"worker": fs(Path(__file__)), "capture": fs(root / "capture_image_embedded_pair_canvas_controls.cjs"),
            "serving": fs(root.parent / "app/detectors/image/embedded_pair.py"),
            "serving_scope": fs(root.parent / "app/evidence/image_embedded_pair_binding.py")}


def freeze(data, run):
    source = {name: fs(data / name) for name in (
        "image-flux-photo-expansion-v285/summary.json", "image-flux-photo-expansion-v285/private-inputs.json",
        "image-bg-real-backup-intake-v262/qualified-private.json", "image-fresh-source-cache-v39/private-groups.json")}
    identities = [identity(case) for case in cases(data, run)]
    need(len(identities) == len(SIZES) * (16 * len(ROLES) + 14), "Holdout roles differ")
    return {"schema": SCHEMA, "code": source_code(), "rule": RULE, "source_receipts": source,
            "screenshots_sha256": fs(run / "screenshots.json"), "sizes": [list(size) for size in SIZES],
            "cases": identities}


def inspect(rgb):
    from app.detectors.image import embedded_pair as method

    atlas = method._atlas(rgb)
    status, pair = method._pair(rgb, atlas)
    if pair is None:
        return status, False, None
    if method._exact_third(atlas, pair) or method._geometric_third(rgb, pair):
        return "unassessed_ambiguous_match", True, None
    return status, True, pair


def observe(case):
    size, slot, role, _, rgb, _, _, _, _ = case
    started = time.monotonic()
    status, unguarded, pair = inspect(rgb)
    result = {"size": list(size), "slot": slot, "role": role, "status": status, "unguarded": unguarded,
              "seconds": round(time.monotonic() - started, 5), "pair": None}
    if pair is not None:
        result["pair"] = {"a": list(pair.area_a), "b": list(pair.area_b), "difference": list(pair.difference_bounds),
                          "mask_sha256": sha(pair.difference.tobytes()), "pixels": int(pair.difference.sum()),
                          "witnesses": [list(point) for point in pair.witnesses]}
    return result


def run_detector(data, run, size):
    import cv2

    cv2.setNumThreads(1)
    lock = read(run / "freeze.json")
    need(lock["code"] == source_code() and lock["screenshots_sha256"] == fs(run / "screenshots.json"), "Frozen code changed")
    start = time.monotonic()
    expected = [row for row in lock["cases"] if tuple(row["size"]) == size]
    observed = []
    for case, entry in zip(cases(data, run, size), expected, strict=True):
        need(identity(case) == entry, "Frozen case bytes differ")
        observed.append(observe(case))
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
    key = f"{size[0]}x{size[1]}"
    write(run / f"score-{key}.json", {"freeze_sha256": fs(run / "freeze.json"), "rows": observed,
                                    "seconds": round(time.monotonic() - start, 3), "peak_rss_bytes": rss})
    print(key, len(observed), "views", rss, "bytes peak RSS", flush=True)


def audit_pair(case, pair):
    import numpy as np

    _, _, role, _, rgb, a, b, _, truth = case
    x, y, r, d = pair["a"]
    u, v, s, t = pair["b"]
    first, second = rgb[y:d, x:r], rgb[v:t, u:s]
    observed = (first != second).any(axis=2)
    need(sha(observed.tobytes()) == pair["mask_sha256"] and int(observed.sum()) == pair["pixels"]
         and list(bounds(observed)) == pair["difference"], "Replayed native differences changed")
    for wx, wy in pair["witnesses"]:
        left, top = wx - x, wy - y
        f, g = first[top:top+16, left:left+16], second[top:top+16, left:left+16]
        need(np.array_equal(f, g) and np.unique(f.reshape(-1, 3), axis=0).shape[0] >= 48, "Native texture proof changed")
    if role not in VALID:
        return False
    yy, xx = np.nonzero(observed)
    ty, tx = np.nonzero(truth)
    def order(col, row, ax, ay):
        return (row + ay) * rgb.shape[1] + col + ax
    direct = np.array_equal(order(xx, yy, x, y), order(tx, ty, a[0], a[1]))
    mapped = np.array_equal(order(xx, yy, u, v), order(tx, ty, b[0], b[1]))
    return direct and mapped


def aggregate(rows, measurements):
    results = {}
    previous = None
    for size in SIZES:
        key = f"{size[0]}x{size[1]}"
        local = [row for row in rows if tuple(row["size"]) == size]
        groups = {role: Counter() for role in (*ROLES, *CONTROLS[2:])}
        by_family = {family: Counter() for family in ("flux", "lama", "authored_paste")}
        exact = {(row["role"], row["slot"]) for row in local if row["exact"]}
        if previous is None:
            previous = exact
        for row in local:
            counter = groups[row["role"]]
            counter.update(total=1, current_reports=row["current_report"], unguarded=row["unguarded"],
                           reports=row["reported"], exact=row["exact"], ambiguous=row["status"] == "unassessed_ambiguous_match")
            if row["family"] in by_family and row["role"] in VALID:
                by_family[row["family"]].update({row["role"]+"_total":1, row["role"]+"_exact":row["exact"]})
        checks = {
            "minimum_valid": all(groups[role]["exact"] >= RULE[f"minimum_exact_{role}"] for role in VALID),
            "retained_exact_groups": len(previous & exact) >= RULE["retained_exact_group_fraction"] * len(previous),
            "no_inexact_valid": not any(groups[role]["reports"] != groups[role]["exact"] for role in VALID),
            "no_ordinary_reports": not any(groups[role]["reports"] for role in CONTROLS),
            "no_third_reports": not any(groups[role]["reports"] for role in GUARDS),
            "adequate_third_challenge": all(groups[role]["unguarded"] >= RULE["minimum_unguarded_third_challenges"] for role in GUARDS),
            "runtime": measurements[key]["max_case_seconds"] <= RULE["maximum_per_case_seconds"]
            and measurements[key]["peak_rss_bytes"] <= RULE["maximum_peak_rss_bytes"],
        }
        results[key] = {"roles": {role: dict(counts) for role, counts in groups.items()}, "families": {
            family: dict(counts) for family, counts in by_family.items()}, "paired_exact_retained": len(previous & exact),
            "paired_exact_reference": len(previous), "paired_exact_gains": len(exact - previous), "checks": checks,
            "passed": all(checks.values()), "resources": measurements[key]}
    return results


def audit(data, run):
    lock = read(run / "freeze.json")
    need(lock == freeze(data, run), "Frozen dataset or method changed")
    public, measurements = [], {}
    all_cases = cases(data, run)
    witnesses = 0
    for size in SIZES:
        key = f"{size[0]}x{size[1]}"
        score = read(run / f"score-{key}.json")
        need(score["freeze_sha256"] == fs(run / "freeze.json"), "Measured cases are outside the frozen roster")
        measurements[key] = {"peak_rss_bytes": score["peak_rss_bytes"], "seconds": score["seconds"],
                             "max_case_seconds": max(row["seconds"] for row in score["rows"])}
        for row in score["rows"]:
            case = next(all_cases)
            need((tuple(row["size"]), row["slot"], row["role"]) == case[:3], "Measured case order changed")
            pair = row["pair"]
            exact = audit_pair(case, pair) if pair is not None else False
            if pair is not None:
                witnesses += len(pair["witnesses"])
            public.append({"size": row["size"], "slot": row["slot"], "role": row["role"], "family": case[3],
                           "status": row["status"], "reported": pair is not None, "exact": bool(exact),
                           "current_report": size == SIZES[0] and pair is not None, "unguarded": row["unguarded"],
                           "pixels": pair["pixels"] if pair else None})
    need(next(all_cases, None) is None and len(public) == len(lock["cases"]), "Native replay incomplete")
    result = {"schema": SCHEMA, "freeze_sha256": fs(run / "freeze.json"), "code": lock["code"],
              "rules": RULE, "source_receipts": lock["source_receipts"], "rows": public,
              "measurements": measurements, "witnesses_replayed": witnesses, "results": aggregate(public, measurements)}
    write(run / "public.json", result)
    print(json.dumps({key: value["checks"] for key, value in result["results"].items()}, indent=2), flush=True)


def review(path):
    report = read(path)
    need(report["schema"] == SCHEMA and report["rules"] == RULE and len(report["rows"]) == 870, "Public contract changed")
    need(report["results"] == aggregate(report["rows"], report["measurements"]), "Public decision cannot be reproduced")
    print(json.dumps({key: {"passed": row["passed"], "roles": row["roles"]} for key, row in report["results"].items()}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "score", "audit", "review"))
    parser.add_argument("output", type=Path)
    parser.add_argument("--data", type=Path, default=Path("backend/data/independent-evaluation"))
    parser.add_argument("--size", choices=[f"{w}x{h}" for w, h in SIZES])
    args = parser.parse_args()
    if args.stage == "freeze":
        write(args.output / "freeze.json", freeze(args.data, args.output))
        print(fs(args.output / "freeze.json"), flush=True)
    elif args.stage == "score":
        if args.size is None:
            parser.error("score requires --size")
        run_detector(args.data, args.output, tuple(map(int, args.size.split("x"))))
    elif args.stage == "audit":
        audit(args.data, args.output)
    else:
        review(args.output)


if __name__ == "__main__":
    main()
