"""Confirm the fixed JPEG90 fallback on independent recorded source groups."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from itertools import combinations
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts import image_reference_same_jpeg_transfer as previous

PANEL = previous.PANEL
SCHEMA = "chipmunk.image-reference-same-jpeg-confirmation/1"
EARLIER = {
    "diagnostic": "a97704735e878d1e04adbb9d45e197d8c5920412adfdf90171615bb6116cf2c6",
    "held": "7b1b9253d0bbe6fa34cae1895ab3e80c2cf370a53943ae568e2756095b2bf9f0",
}
QUALIFICATION = {
    "primary": "image-bg-real-pixel-qualification-complete-v225",
    "backup": "image-bg-real-backup-intake-v262-complete",
    "wavelet": "image-wavelet-region-transfer-2026-11-11",
    "flux": "image-flux-photo-expansion-v285-freeze",
    "patch": "image-three-class-patch-fresh-2026-11-12",
    "texture": "image-jpeg-texture-rank-transfer-2026-11-12",
    "editors": "image-masked-context-new-editors-2026-11-13",
}
ASSETS = ("original", "edited", "semantic_mask")
read, write, require, sha, digest = previous.read, previous.write, previous.require, previous.sha, previous.digest


def published(name):
    return read(PANEL / f"{QUALIFICATION[name]}.json")


def earlier_sources(args):
    prior = previous.prior
    seen, records = set(), []
    for name, pin in EARLIER.items():
        path = PANEL / f"image-reference-same-jpeg-{name}-2026-11-13.json"
        require(digest(path) == pin, "Earlier exact fallback receipt differs")
        seen.update(row["source_sha256"] for row in read(path)["freeze"]["cases"])
    for name in ("primary", "backup"):
        path = getattr(args, name)
        require(digest(path) == published(name)["qualified_private_sha256"], "Earlier LaMa roster differs")
        records.extend(row["source"] for row in read(path)["qualified"])
    require(digest(args.wavelet) == published("wavelet")["receipts"]["fresh_intake"]["plan_sha256"],
            "Earlier wavelet roster differs")
    records.extend(read(args.wavelet)["cases"])
    require(digest(args.flux) == published("flux")["private_inputs_sha256"], "Earlier FLUX roster differs")
    excluded = set()
    found = set()
    for row in records:
        pin = row["original"]["sha256"]
        if pin in seen:
            excluded.update(prior.row_identity(row))
            found.add(pin)
    for row in read(args.flux)["rows"]:
        credit, pin = row["attribution"], row["source_jpeg_sha256"]
        if pin in seen:
            excluded.update(prior.identity(prior.terminal(credit["photo_page"]), prior.terminal(credit["author_url"]),
                                           credit["author_name"], pin))
            found.add(pin)
    require(found == seen and len(found) == 37, "Missing earlier source identity")
    return excluded


def qualified(args):
    from scripts import image_jpeg_texture_rank_transfer as texture
    from scripts import image_masked_context_new_editors as editors
    from scripts import image_three_class_patch_fresh as patch

    patch_plan = read(args.patch / "plan.json")
    require(read(args.patch / "intake.json") == published("patch")["intake"], "Patch intake differs")
    yield from (("patch", *case) for case in patch.groups(args.patch, patch_plan, ([], set(), set())))
    pins = published("texture")["commitments"]
    require(all(digest(args.texture / name) == pins[name] for name in ("plan.json", "intake.json")),
            "Texture intake differs")
    yield from (("texture", *case) for case in texture.held_inputs(args.texture, read(args.texture / "plan.json")))
    pins = published("editors")
    require(all(digest(args.editors / f"{name}.json") == pins[f"{name}_sha256"] for name in ("plan", "intake")),
            "Editor intake differs")
    plan = read(args.editors / "plan.json")
    for item, row in zip(plan["cases"], read(args.editors / "intake.json")["rows"], strict=True):
        require((item["slot"], item["generator"]) == (row["slot"], row["family"]), "Editor source binding differs")
        if row["status"] != "qualified" or item["generator"] == "kandinsky":
            continue  # The SDXL case already represents the same original.
        bodies = {}
        for role, proof in row["proofs"].items():
            data = (args.editors / "assets" / f"{row['slot']:02d}-{role}.png").read_bytes()
            require(len(data) == proof["bytes"] and sha(data) == proof["sha256"], "Editor source bytes differ")
            bodies[role] = data
        native, truth, _, _ = editors.bound(item, bodies)
        require(sha(truth.tobytes()) == row["mask_sha256"], "Editor truth differs")
        yield "editors", item, native, truth


def sources(args):
    from scripts import image_bg_real_acquisition_v221 as originals
    from scripts.image_source_groups_v22 import near

    excluded = earlier_sources(args)
    retained = []
    for bank, item, native, truth in qualified(args):
        if not 512 <= min(item["size"]) <= max(item["size"]) <= 768:
            continue
        identity = previous.prior.row_identity(item)
        signature = originals.fingerprint(b"", native["original"])
        require(not (identity & excluded), "A confirmation photo, photographer or source hash overlaps")
        require(item["photo_attribution"]["License"] == "https://creativecommons.org/licenses/by/2.0/",
                "Source photo licence differs")
        excluded.update(identity)
        retained.append((bank, item, native, truth, signature))
    require(Counter(row[1]["generator"] for row in retained) == {"lama": 13, "sd15": 14, "kandinsky": 6, "sdxl": 1},
            "Confirmation source counts differ")
    for a, b in combinations((row[-1] for row in retained), 2):
        require(a["rgb_sha256"] != b["rgb_sha256"] and not near(a, b), "Repeated or visually near confirmation photo")
    return retained


def cases(roster):
    import cv2
    import numpy as np

    from scripts import image_residual_encoder_transfer as edits

    for slot, (_, item, views, truth, _) in enumerate(roster):
        original = views["original"]
        pin = item["original"]["sha256"]
        h, w = truth.shape
        ordinary, generated = edits.authored(views, truth, pin)
        donor = cv2.resize(roster[(slot + 1) % len(roster)][2]["original"], (w, h), interpolation=cv2.INTER_AREA)
        fractional, integer = [cv2.warpAffine(original, np.float32([[1, 0, x], [0, 1, y]]), (w, h),
                                              flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
                               for x, y in ((.55, -.55), (16, 0))]
        tone = np.clip(np.rint(original.astype(np.float32) * [1.10, .91, 1.06] + [-5, 8, 4]), 0, 255).astype(np.uint8)
        delivered = ordinary | generated | {"prior_jpeg": ordinary["second_jpeg"], "resize": ordinary["global_resize"],
                                             "tone_roundtrip": tone, "global_blur": cv2.GaussianBlur(original, (0, 0), 1.25),
                                             "different_photo": donor, "fractional_shift": fractional,
                                             "integer_shift": integer, "reflection": original[:, ::-1]}
        before = edits.encode(original, 90)
        for role in previous.ROLES:
            rgb = delivered[role]
            exact = np.any(rgb != original, axis=2) if role in (*previous.POSITIVE, *previous.LOCAL) else np.zeros_like(truth)
            after = edits.encode(rgb, 90)
            yield {"family": item["generator"], "slot": slot, "source_sha256": pin, "shape": [h, w], "role": role,
                   "before_sha256": sha(before), "after_sha256": sha(after), "truth_sha256": sha(exact.tobytes()),
                   "truth_pixels": int(exact.sum())}, before, after, exact


def recount(receipt):
    lock = receipt["freeze"]
    require(lock["confirmation_schema"] == SCHEMA and lock["earlier_receipts"] == EARLIER
            and sha(json.dumps(lock, sort_keys=True).encode()) == receipt["freeze_sha256"], "Confirmation freeze differs")
    groups = lock["sources"]
    require(len(groups) == len({r["source_sha256"] for r in groups}) == len({r["rgb_sha256"] for r in groups}) == 34,
            "Confirmation source roster incomplete")
    require([(row["slot"], row["role"], row["source_sha256"], row["family"]) for row in lock["cases"]]
            == [(i, role, r["source_sha256"], r["family"]) for i, r in enumerate(groups) for role in previous.ROLES],
            "Case roster differs from source roster")
    result = previous.summary(lock, receipt["rows"])
    require(result == receipt["summary"], "Confirmation result differs")
    return result


def freeze_roster(roster):
    from app.detectors.image import reference_same_jpeg
    from app.evidence import image_reference

    return {"schema": previous.SCHEMA, "stage": "held", "design": previous.DESIGN,
            "confirmation_schema": SCHEMA, "earlier_receipts": EARLIER,
            "sources": [{"bank": bank, "family": item["generator"], "source_sha256": item["original"]["sha256"],
                         "rgb_sha256": signature["rgb_sha256"]} for bank, item, _, _, signature in roster],
            "code": {Path(path).name: digest(Path(path)) for path in
                     (__file__, previous.__file__, image_reference.__file__, reference_same_jpeg.__file__)},
            "cases": [key for key, *_ in cases(roster)]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "score", "replay", "review"))
    parser.add_argument("run", type=Path)
    for name in QUALIFICATION:
        parser.add_argument(f"--{name}", type=Path)
    args = parser.parse_args()
    if args.phase == "review":
        print(json.dumps(recount(read(args.run)), sort_keys=True))
        return

    import cv2

    require(all(getattr(args, name) for name in QUALIFICATION), "Supply the seven retained source rosters")
    cv2.setNumThreads(1)
    roster = sources(args)
    lock = freeze_roster(roster)
    if args.phase == "freeze":
        args.run.mkdir(mode=0o700)
        write(args.run / "freeze.json", lock)
        print(f"Froze {len(lock['cases'])} pairs on {len(roster)} source photos", flush=True)
        return
    require(read(args.run / "freeze.json") == lock, "Frozen code or prepared images changed")
    rows = []
    for case in cases(roster):
        with TemporaryDirectory(dir=args.run) as scratch:
            rows.append(previous.measure(case, Path(scratch)))
        if len(rows) % (2 * len(previous.ROLES)) == 0:
            print(f"Measured {len(rows)} of {len(lock['cases'])} pairs", flush=True)
    receipt = {"freeze": lock, "freeze_sha256": sha(json.dumps(lock, sort_keys=True).encode()),
               "rows": rows, "summary": previous.summary(lock, rows)}
    if args.phase == "replay":
        require(read(args.run / "receipt.json") == receipt, "Native replay differs")
    else:
        write(args.run / "receipt.json", receipt)
    print(json.dumps(recount(receipt), sort_keys=True))


if __name__ == "__main__":
    main()
