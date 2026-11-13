# SPDX-License-Identifier: Apache-2.0
"""Test exact overlap of regional and earlier-copy evidence on held JPEG90 pairs."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from collections import Counter
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[2]
PANEL = ROOT / "docs/evaluation-panels"
PRIVATE = ROOT / "backend/data/independent-evaluation"
PRIOR = PANEL / "image-regional-jpeg90-quality-confirmation-2026-11-13.json"
SCHEMA = "chipmunk.image-jpeg90-paired-corroboration/1"
POSITIVE = ("sharp", "feathered")
LOCAL = ("local_blur", "local_tone", "local_copy")
NUISANCE = ("original", "prior_jpeg", "resize", "tone_roundtrip")
INCOMPATIBLE = ("different_photo", "fractional_shift")
ROLES = POSITIVE + LOCAL + NUISANCE + INCOMPATIBLE
ARMS = ("regional", "earlier_copy", "shared")
SOURCE_COUNTS = {"lama": 8, "flux": 6}
DESIGN = {
    "source_selection": "Eight ranked retained LaMa sources, six retained licensed FLUX sources. Exclude prior JPEG90 banks by recorded photo ID, author, and original digest.",
    "delivery": "Upright same-frame JPEG90; ordinary alternatives include local edit, re-encode, resize, tone, unrelated source, and fractional shift.",
    "corroboration": "Require a production integer alignment with zero offset. Intersect retained regional pixels with the measured earlier-copy mask, without filling bounding boxes.",
    "useful": {"iou_min": .30, "recall_min": .50},
    "review": {"minimum_useful": 4, "minimum_families": 2, "retain_regional_useful": True,
               "max_global_processing_flags": 0, "max_incompatible_flags": 0},
    "ai_weight": 0,
}


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def sha(value):
    return hashlib.sha256(value).hexdigest()


def digest(path):
    return sha(path.read_bytes())


def read(path):
    return json.loads(path.read_bytes())


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def identity(photo, owner, name, original_sha):
    return {("photo", str(photo).casefold()), ("author", str(owner).casefold()),
            ("author", str(name).casefold()), ("bytes", original_sha)}


def terminal(url):
    return urlsplit(url).path.rstrip("/").rsplit("/", 1)[-1]


def row_identity(row):
    credit = row["photo_attribution"]
    return (identity(row["flickr_photo_id"], row["flickr_owner"], credit["Author"], row["original"]["sha256"])
            | {("author", terminal(credit["AuthorProfileURL"]).casefold())})


def sources(args):
    from scripts import image_reference_delivery_envelope_lama as lama
    from scripts import image_reference_png_jpeg_stability as flux

    plans = {"main": args.main, "external": args.external, "confirmation_plan": args.confirmation}
    require(all(digest(path) == read(PRIOR)["freeze"]["source_sha256"][name] for name, path in plans.items()),
            "Prior JPEG90 source roster differs")
    excluded = set()
    for name in ("main", "external"):
        for row in read(plans[name])["cases"]:
            excluded.update(row_identity(row))
    for row in read(args.confirmation)["rows"]:
        credit = row["attribution"]
        url = urlsplit(credit["OriginalLandingURL"]).path.strip("/").split("/")
        excluded.update(identity(url[-1], terminal(credit["AuthorProfileURL"]), credit["Author"], row["source"]["sha256"]))
        excluded.add(("author", url[-2].casefold()))
    kept = excluded.copy()

    roster = args.lama / "qualified-private.json"
    summary = args.lama / "summary-v225.json"
    require(digest(roster) == read(summary)["qualified_private_sha256"], "LaMa source commitment differs")
    records = read(roster)
    la = []
    for row in sorted(records["qualified"], key=lambda item: sha(f"{SCHEMA}/{item['source']['flickr_photo_id']}".encode())):
        source = row["source"]
        keys = row_identity(source)
        if keys & kept or not 512 <= min(source["size"]) <= max(source["size"]) == 768:
            continue
        require(source["photo_attribution"]["License"] == "https://creativecommons.org/licenses/by/2.0/",
                "Source photo licence differs")
        a, b, _ = lama.source_pixels(Path(records["cache"]), row)
        la.append((source["original"]["sha256"], a, b))
        kept.update(keys)
        if len(la) == SOURCE_COUNTS["lama"]:
            break
    fr, _ = flux._licensed_flux({"cases": []})
    flux_plan = read(PRIVATE / "image-flux-photo-expansion-v285/private-inputs.json")["rows"]
    require(len(la) == SOURCE_COUNTS["lama"] and len(fr) == SOURCE_COUNTS["flux"], "Panel source counts differ")
    for (source_sha, _, _, _), row in zip(fr, flux_plan, strict=True):
        attribution = row["attribution"]
        keys = identity(terminal(attribution["photo_page"]), terminal(attribution["author_url"]),
                        attribution["author_name"], source_sha)
        require(row["source_jpeg_sha256"] == source_sha and not (keys & kept), "FLUX source overlaps")
        kept.update(keys)
    output = {"lama": la, "flux": [(pin, a, b) for pin, a, b, _ in fr]}
    rgb = [sha(a.tobytes()) for group in output.values() for _, a, _ in group]
    require(len(set(rgb)) == len(rgb), "Retained native RGB repeats")
    return output


def cases(args):
    import cv2
    import numpy as np
    from PIL import Image

    from scripts import image_residual_encoder_transfer as edits

    for family, bank in sources(args).items():
        for slot, (source, original, edited) in enumerate(bank):
            exact = np.any(original != edited, axis=2)
            require(exact.any(), "Missing exact edited pixels")
            ordinary, generated = edits.authored({"original": original, "edited": edited}, exact, source)
            h, w = exact.shape
            donor = cv2.resize(bank[(slot + 1) % len(bank)][1], (w, h), interpolation=cv2.INTER_AREA)
            tone = np.clip(np.rint(original.astype(np.float32) * [1.10, .91, 1.06] + [-5, 8, 4]), 0, 255).astype(np.uint8)
            fractional = cv2.warpAffine(original, np.float32([[1, 0, .55], [0, 1, -.55]]), (w, h),
                                       flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            all_views = ordinary | generated | {"different_photo": donor, "fractional_shift": fractional,
                                                 "tone_roundtrip": tone, "prior_jpeg": ordinary["second_jpeg"],
                                                 "resize": ordinary["global_resize"]}
            reference = edits.encode(original, 90)
            for role in ROLES:
                image = all_views[role]
                truth = np.any(image != original, axis=2) if role in (*POSITIVE, *LOCAL) else np.zeros_like(exact)
                after = edits.encode(image, 90)
                key = {"family": family, "slot": slot, "source_sha256": source, "shape": [h, w], "role": role,
                       "before_sha256": sha(reference), "after_sha256": sha(after), "truth_sha256": sha(truth.tobytes()),
                       "truth_pixels": int(truth.sum())}
                with Image.open(io.BytesIO(after), formats=["JPEG"]) as jpeg:
                    require(jpeg.size == (w, h), "JPEG native geometry differs")
                yield key, reference, after, truth


def useful(area, overlap, target):
    return bool(target and 2 * overlap >= target and 10 * overlap >= 3 * (area + target - overlap))


def measure(key, before, after, truth, model):
    import numpy as np
    from PIL import Image

    from app.detectors.image import reference_alignment
    from scripts.image_regional_jpeg_transfer import native_readout

    with Image.open(io.BytesIO(before)) as earlier, Image.open(io.BytesIO(after)) as submitted:
        a, b = np.asarray(earlier), np.asarray(submitted)
        raw = model.predict(submitted)
        regional = native_readout(raw, after, b).source_mask(submitted.width, submitted.height)
        if np.array_equal(a, b):
            result, paired = {"status": "comparable"}, np.zeros_like(truth)
            alignment = "identical"
        else:
            result, paired = reference_alignment.compare(a, b, submitted)
            offset = result.get("alignment", {}).get("reference_minus_submitted")
            if result["status"] == "comparable" and offset == {"x": 0, "y": 0}:
                alignment = "zero"
            else:
                alignment, paired = "unassessed", None
        shared = None if paired is None else regional & paired
    measures = {}
    for name, mask in zip(ARMS, (regional, paired, shared), strict=True):
        if mask is None:
            measures[name] = None
            continue
        ys, xs = np.where(mask)
        extent = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1] if xs.size else None
        area, overlap = int(mask.sum()), int((mask & truth).sum())
        measures[name] = {"mask_sha256": sha(mask.tobytes()), "bounds": extent, "area": area, "overlap": overlap}
    return {"pair_status": result["status"], "alignment": alignment, "regional_input_sha256": raw.geometry["input_float32_sha256"],
            "arms": measures}


def summarize(lock, rows):
    require(lock["schema"] == SCHEMA and lock["design"] == DESIGN and lock["source_counts"] == SOURCE_COUNTS,
            "Frozen comparison design differs")
    expected = [(family, slot, role) for family in ("lama", "flux") for slot in range(SOURCE_COUNTS[family]) for role in ROLES]
    require([(case["family"], case["slot"], case["role"]) for case in lock["cases"]] == expected,
            "Unexpected case schedule")
    groups = {}
    totals = {name: Counter() for name in ARMS}
    roles = {name: {role: Counter() for role in ROLES} for name in ARMS}
    useful_families = {name: Counter() for name in ARMS}
    losses = gains = 0
    for key, row in zip(lock["cases"], rows, strict=True):
        role, family, slot = key["role"], key["family"], key["slot"]
        group = groups.setdefault((family, slot), (key["source_sha256"], key["before_sha256"], key["shape"]))
        require(group == (key["source_sha256"], key["before_sha256"], key["shape"]), "Source commitments differ within a group")
        h, w = key["shape"]
        target = key["truth_pixels"]
        require(type(target) is int and 0 <= target <= w*h and bool(target) == (role in (*POSITIVE, *LOCAL)),
                "Unexpected exact truth area")
        for pin in (key["source_sha256"], key["before_sha256"], key["after_sha256"], key["truth_sha256"],
                    row["regional_input_sha256"]):
            require(len(pin) == 64 and bytes.fromhex(pin).hex() == pin, "Unexpected source commitment")
        arms = row["arms"]
        require(set(arms) == set(ARMS) and arms["regional"] is not None and row["alignment"] in {"identical", "zero", "unassessed"}
                and (arms["earlier_copy"] is None) == (arms["shared"] is None) == (row["alignment"] == "unassessed"),
                "Measurement availability differs")
        hits = {}
        for name, item in arms.items():
            roles[name][role]["planned"] += 1
            if item is None:
                continue
            area, overlap = item["area"], item["overlap"]
            require(type(area) is int and type(overlap) is int and 0 <= overlap <= min(area, target)
                    and area + target - overlap <= w * h and bool(area) == bool(item["bounds"]), "Impossible overlap or bounds")
            if area:
                x, y, right, bottom = item["bounds"]
                require(0 <= x < right <= w and 0 <= y < bottom <= h and area <= (right - x) * (bottom - y),
                        "Selected pixels fall outside their native extent")
            require(len(item["mask_sha256"]) == 64 and bytes.fromhex(item["mask_sha256"]).hex() == item["mask_sha256"],
                    "Invalid mask commitment")
            hits[name] = useful(area, overlap, target)
            roles[name][role].update(assessed=1, regions=int(area > 0), useful=int(hits[name]))
            if role in POSITIVE:
                totals[name].update(generated_useful=int(hits[name]), misplaced=int(area > 0 and not hits[name]))
                useful_families[name][family] += int(hits[name])
            elif role in LOCAL:
                totals[name].update(ordinary_local_useful=int(hits[name]))
            else:
                totals[name].update({("global_or_untouched" if role in NUISANCE else "incompatible"): int(area > 0)})
        if (shared := arms["shared"]) is not None:
            require(all(shared[metric] <= min(arms[arm][metric] for arm in ARMS[:-1]) for metric in ("area", "overlap")),
                    "Intersection exceeds one of its source masks")
        if role in POSITIVE:
            gains += int(hits.get("shared", False) and not hits["regional"])
            losses += int(hits["regional"] and not hits.get("shared", False))
    require(len({group[0] for group in groups.values()}) == sum(SOURCE_COUNTS.values()), "Source group repeated")
    advance = (losses == 0 and totals["shared"]["generated_useful"] >= DESIGN["review"]["minimum_useful"]
               and sum(n > 0 for n in useful_families["shared"].values()) >= DESIGN["review"]["minimum_families"]
               and totals["shared"]["global_or_untouched"] == totals["shared"]["incompatible"] == 0)
    return {"by_role": roles, "totals": totals, "useful_families": useful_families,
            "shared_vs_regional": {"gains": gains, "losses": losses}, "advances": advance}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "score", "replay", "review"))
    parser.add_argument("path", type=Path)
    parser.add_argument("--main", type=Path)
    parser.add_argument("--external", type=Path)
    parser.add_argument("--confirmation", type=Path)
    parser.add_argument("--lama", type=Path, default=PRIVATE / "image-bg-real-acquisition-v225")
    args = parser.parse_args()
    try:
        if args.phase == "review":
            receipt = read(args.path)
            require(receipt["freeze_sha256"] == sha(json.dumps(receipt["freeze"], sort_keys=True).encode()),
                    "Frozen input digest differs")
            result = summarize(receipt["freeze"], receipt["rows"])
            require(result == receipt["summary"], "Recorded outcomes differ")
        else:
            import cv2
            import torch

            from app.detectors.image import reference_alignment, reference_compare, regional, regional_support
            from scripts import image_regional_jpeg_transfer as existing

            require(all((args.main, args.external, args.confirmation)), "Prior source plans are required")
            cv2.setNumThreads(1)
            torch.set_num_threads(2)
            torch.set_num_interop_threads(1)
            modules = (reference_alignment, reference_compare, regional, regional_support, existing)
            lock = {"schema": SCHEMA, "design": DESIGN, "source_counts": SOURCE_COUNTS, "prior_sha256": digest(PRIOR),
                    "checkpoint_sha256": regional.CHECKPOINT_SHA256,
                    "code_sha256": {Path(path).name: digest(Path(path)) for path in
                                    (__file__, *(module.__file__ for module in modules))},
                    "cases": [key for key, *_ in cases(args)]}
            if args.phase == "freeze":
                args.path.mkdir(mode=0o700)
                write(args.path / "freeze.json", lock)
                return
            require(lock == read(args.path / "freeze.json"), "Frozen method or native bytes differ")
            model = regional.load_model(existing.CHECKPOINT)
            rows = []
            for planned, case in zip(lock["cases"], cases(args), strict=True):
                require(planned == case[0], "Frozen delivery differs")
                rows.append(measure(*case, model))
                if len(rows) % len(ROLES) == 0:
                    print(f"Measured {len(rows)} JPEG90 pairs", flush=True)
            result = summarize(lock, rows)
            receipt = {"freeze": lock, "freeze_sha256": sha(json.dumps(lock, sort_keys=True).encode()), "rows": rows, "summary": result}
            path = args.path / "receipt.json"
            if args.phase == "score":
                write(path, receipt)
            else:
                require(receipt == read(path), "Native replay differs")
        print(json.dumps(result, sort_keys=True))
    except (OSError, KeyError, TypeError, ValueError) as error:
        parser.exit(1, f"JPEG90 paired corroboration failed: {error}\n")


if __name__ == "__main__":
    main()
