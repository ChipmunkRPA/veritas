# SPDX-License-Identifier: Apache-2.0
"""Test exact native JPEG90 anchors as a fallback for earlier-copy comparison."""

from __future__ import annotations

import argparse
import io
import json
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts import image_jpeg90_paired_corroboration as prior

ROOT = prior.ROOT
PANEL = prior.PANEL
SCHEMA = "chipmunk.image-reference-same-jpeg-transfer/1"
PRIOR_PAIRED = PANEL / "image-jpeg90-paired-corroboration-2026-11-13.json"
BACKUP = PANEL / "image-bg-real-backup-intake-v262-complete.json"
WAVELET = PANEL / "image-wavelet-region-transfer-2026-11-11.json"
POSITIVE = prior.POSITIVE
LOCAL = prior.LOCAL
ARMS = ("current", "combined", "also_empty")
NUISANCE = ("original", "prior_jpeg", "resize", "tone_roundtrip", "global_blur")
INCOMPATIBLE = ("different_photo", "fractional_shift", "integer_shift", "reflection")
ROLES = POSITIVE + LOCAL + NUISANCE + INCOMPATIBLE
OPTIONS = dict(allow_proportional_resize=True, allow_oriented_crop=True, allow_oriented_resize=True,
               allow_embedded_photo=True, allow_exact_jpeg_identity=True, allow_shifted_jpeg=True,
               allow_structure_jpeg=True)
DESIGN = {
    "scope": "Only unassessed pairs of upright same-framing native 8-bit RGB JPEG90 files with matching 4:2:0 encoding.",
    "proposal": "Existing unique-anchor proof on decoded pixels with at least 85% same-position exact blocks.",
    "region": "Existing native RGB comparison on the delivered arrays without another encode or threshold fit.",
    "qualification": "At least two useful additions in two held source groups, no additional global-control highlights, "
                     "incompatible pairs assessed or misplaced edited masks. All incumbent results kept.",
    "empty_followup": "Also test nonidentical pairs whose current result is comparable with no selected region. "
                      "Replace only with a nonempty exact-anchor mask. The older source bank supplied the hypothesis.",
    "useful": {"iou_min": .30, "recall_min": .50},
    "ai_weight": 0,
}
read, write, require, sha, digest = prior.read, prior.write, prior.require, prior.sha, prior.digest


def held_sources(args):
    import numpy as np

    from scripts import image_background_patch_score as wavelet
    from scripts import image_reference_delivery_envelope_lama as lama
    from scripts import image_reference_png_jpeg_stability as flux

    paired = read(PRIOR_PAIRED)["freeze"]["cases"]
    require({pin for pin, *_ in prior.sources(args)["lama"]} == {row["source_sha256"] for row in paired if row["family"] == "lama"},
            "Previously measured LaMa source roster differs")
    earlier_plan = read(args.lama / "qualified-private.json")
    exposed = {row["source_sha256"] for row in paired}
    blocked = set()
    for path in (args.main, args.external):
        for row in read(path)["cases"]:
            blocked.update(prior.row_identity(row))
    for row in read(args.confirmation)["rows"]:
        credit = row["attribution"]
        photo_url = prior.urlsplit(credit["OriginalLandingURL"]).path.strip("/").split("/")
        blocked.update(prior.identity(photo_url[-1], photo_url[-2], credit["Author"], row["source"]["sha256"]))
        blocked.add(("author", prior.terminal(credit["AuthorProfileURL"]).casefold()))
    for row in earlier_plan["qualified"]:
        if row["source"]["original"]["sha256"] in exposed:
            blocked.update(prior.row_identity(row["source"]))
    for row in read(flux.PRIVATE / "image-flux-photo-expansion-v285/private-inputs.json")["rows"]:
        credit = row["attribution"]
        blocked.update(prior.identity(prior.terminal(credit["photo_page"]), prior.terminal(credit["author_url"]),
                                      credit["author_name"], row["source_jpeg_sha256"]))

    qualified = args.backup / "qualified-private.json"
    require(digest(qualified) == read(BACKUP)["qualified_private_sha256"], "Backup rights and qualification ledger differs")
    backup = read(qualified)
    require(read(args.wavelet / "result.json") == read(WAVELET)["receipts"]["fresh_intake"]
            and digest(args.wavelet / "plan.json") == read(args.wavelet / "result.json")["plan_sha256"],
            "Earlier diffusion source commitments differ")
    wavelet_rows = read(args.wavelet / "plan.json")["cases"]
    diffusion = [(wavelet_rows[row["slot"]], views["original"], views["edited"]) for row, views, _ in wavelet.inputs(args.wavelet)]

    def backups():
        for row in backup["qualified"]:
            a, b, _ = lama.source_pixels(Path(backup["cache"]), row)
            yield row["source"], a, b

    banks = {"lama": list(backups()), "diffusion": diffusion}
    output = []
    for bank, items in banks.items():
        for row, original, edited in items:
            identities = prior.row_identity(row)
            if identities & blocked or not 512 <= min(original.shape[:2]) <= max(original.shape[:2]) <= 768:
                continue
            require(row["photo_attribution"]["License"] == "https://creativecommons.org/licenses/by/2.0/"
                    and np.any(original != edited), "Missing exact edits or commercial photo rights")
            blocked.update(identities)
            output.append((row.get("generator", bank), row["original"]["sha256"], original, edited))
    require(Counter(family for family, *_ in output) == {"lama": 11, "sdxl": 6, "kandinsky": 6},
            "Held source denominators differ")
    require(len({sha(a.tobytes()) for _, _, a, _ in output}) == len(output), "Held source pixels repeat")
    return output


def cases(args, stage):
    import cv2
    import numpy as np

    from scripts import image_residual_encoder_transfer as edits

    if stage == "diagnostic":
        # Preserve the previous positive and ordinary rows and their exact byte hashes.
        yield from prior.cases(args)
        return
    bank = held_sources(args)
    for slot, (family, pin, original, edited) in enumerate(bank):
        h, w = original.shape[:2]
        ordinary, generated = edits.authored({"original": original, "edited": edited},
                                             np.any(original != edited, axis=2), pin)
        donor = cv2.resize(bank[(slot + 1) % len(bank)][2], (w, h), interpolation=cv2.INTER_AREA)

        shifts = [cv2.warpAffine(original, np.float32([[1, 0, x], [0, 1, y]]), (w, h),
                                 flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
                  for x, y in ((.55, -.55), (16, 0))]

        tone = np.clip(np.rint(original.astype(np.float32) * [1.10, .91, 1.06] + [-5, 8, 4]), 0, 255).astype(np.uint8)
        all_views = ordinary | generated | {"prior_jpeg": ordinary["second_jpeg"], "resize": ordinary["global_resize"],
                                            "tone_roundtrip": tone, "global_blur": cv2.GaussianBlur(original, (0, 0), 1.25),
                                            "different_photo": donor, "fractional_shift": shifts[0],
                                            "integer_shift": shifts[1], "reflection": original[:, ::-1]}
        before = edits.encode(original, 90)
        for role in ROLES:
            rgb = all_views[role]
            truth = np.any(rgb != original, axis=2) if role in (*POSITIVE, *LOCAL) else np.zeros((h, w), bool)
            after = edits.encode(rgb, 90)
            key = {"family": family, "slot": slot, "source_sha256": pin, "shape": [h, w], "role": role,
                   "before_sha256": sha(before), "after_sha256": sha(after), "truth_sha256": sha(truth.tobytes()),
                   "truth_pixels": int(truth.sum())}
            yield key, before, after, truth


def measure(case, path):
    import numpy as np
    from PIL import Image

    from app.detectors.image import reference_same_jpeg
    from app.evidence import image_reference

    key, before, after, truth = case
    earlier, submitted = path / "before.jpg", path / "after.jpg"
    earlier.write_bytes(before)
    submitted.write_bytes(after)
    result = image_reference.inspect(earlier, submitted, evidence_dir=path / "base", **OPTIONS)
    if asset := result.get("mask"):
        with Image.open(path / "base" / asset["file"]) as image:
            mask = np.asarray(image) > 0
    else:
        mask = np.zeros_like(truth) if result["status"] == "comparable" else None
    arms = {"current": (result, mask)}
    if mask is None or (before != after and not mask.any()):
        with Image.open(io.BytesIO(before)) as a, Image.open(io.BytesIO(after)) as b:
            candidate, selected = reference_same_jpeg.compare(a, b, before, after)
        if candidate is not None:
            if mask is None:
                integrated = image_reference.inspect(earlier, submitted, evidence_dir=path / "fallback",
                                                     allow_same_jpeg=True, **OPTIONS)
                require(integrated["schema"] == reference_same_jpeg.SCHEMA and integrated["affects_overall_confidence"] is False,
                        "Native fallback differs from the exact-block candidate")
                if integrated.get("mask"):
                    with Image.open(path / "fallback" / integrated["mask"]["file"]) as image:
                        require(np.array_equal(np.asarray(image) > 0, selected), "Integrated mask differs from the candidate")
                else:
                    require(not selected.any(), "Integrated mask is missing")
                arms["combined"] = (integrated, selected)
            elif selected.any():
                arms["also_empty"] = (candidate, selected)
    arms.setdefault("combined", arms["current"])
    arms.setdefault("also_empty", arms["combined"])
    measured = {}
    for name, (record, mask) in arms.items():
        selected = None
        if mask is not None:
            yy, xx = np.where(mask)
            area, overlap = int(mask.sum()), int((mask & truth).sum())
            selected = {"area": area, "overlap": overlap, "mask_sha256": sha(mask.tobytes()),
                        "bounds": [int(xx.min()), int(yy.min()), int(xx.max())+1, int(yy.max())+1] if area else None}
        proof = record.get("alignment", {})
        measured[name] = {"status": record["status"], "measure": selected,
                          "proof": {field: proof[field] for field in ("full_tiles", "exact_tiles", "anchor_count",
                                                                    "anchor_sha256", "supported_zones",
                                                                    "reference_decoded_rgb_sha256")}
                          if proof.get("method") == reference_same_jpeg.METHOD else None}
    return key | measured


def summary(freeze, rows):
    require(freeze["schema"] == SCHEMA and freeze["design"] == DESIGN, "Unexpected evaluation scope")
    counts = {name: {role: Counter() for role in ROLES} for name in ARMS}
    additions = {name: Counter() for name in ARMS[1:]}
    distinct = {name: set() for name in ARMS[1:]}
    groups = {}
    for frozen, row in zip(freeze["cases"], rows, strict=True):
        require(all(row[key] == value for key, value in frozen.items()), "Result input commitment differs")
        role, group = row["role"], row["source_sha256"]
        groups.setdefault(group, set())
        require(role not in groups[group], "Repeated delivery role")
        groups[group].add(role)
        h, w = row["shape"]
        flags = {}
        for name, counts_by_role in counts.items():
            item, totals = row[name], counts_by_role[role]
            totals["files"] += 1
            selected, proof = item["measure"], item["proof"]
            require((selected is not None) == (item["status"] == "comparable"), "Availability disagrees with mask")
            if proof:
                incumbent = row["current"]["measure"]
                require(name != "current" and (incumbent is None or
                        (name == "also_empty" and incumbent["area"] == 0 and selected["area"] > 0)),
                        "Fallback overwrote an incumbent highlight")
                full = (h // 16) * (w // 16)
                exact, anchors, zones = proof["exact_tiles"], proof["anchor_count"], proof["supported_zones"]
                require(proof["full_tiles"] == full and full >= exact and 20 * exact >= 17 * full
                        and exact >= anchors >= 24 and len({tuple(zone) for zone in zones}) >= 6
                        and all({zone[axis] for zone in zones} == {0, 1, 2} for axis in (0, 1)),
                        "Impossible exact-anchor proof")
            if selected is None:
                flags[name] = (False, False)
                continue
            area, overlap, target = selected["area"], selected["overlap"], row["truth_pixels"]
            require(0 <= overlap <= min(area, target) and area + target - overlap <= w*h, "Impossible pixel overlap")
            box = selected["bounds"]
            require(bool(area) == bool(box), "Unexpected mask bounds")
            if box:
                x, y, right, bottom = box
                require(0 <= x < right <= w and 0 <= y < bottom <= h and area <= (right-x)*(bottom-y),
                        "Out-of-source mask bounds")
            useful = prior.useful(area, overlap, target)
            flags[name] = (bool(area), useful)
            totals.update(assessed=1, highlighted=int(area > 0), useful=int(useful))
        for name, totals in additions.items():
            if row[name]["proof"] is None:
                require(row[name] == row["current"], "An incumbent result changed without exact anchors")
                continue
            flagged, useful = flags[name]
            totals.update(assessed=1, useful=int(useful and role in POSITIVE),
                          misplaced=int(flagged and not useful and role in POSITIVE),
                          nuisance=int(flagged and role in NUISANCE), incompatible=int(role in INCOMPATIBLE))
            if useful and role in POSITIVE:
                distinct[name].add(group)
    planned = set(prior.ROLES if freeze["stage"] == "diagnostic" else ROLES)
    require(all(roles == planned for roles in groups.values()), "Incomplete source groups")
    qualifies = {name: (freeze["stage"] == "held" and totals["useful"] >= 2 and len(distinct[name]) >= 2
                        and all(totals[key] == 0 for key in ("misplaced", "nuisance", "incompatible")))
                 for name, totals in additions.items()}
    return {"groups": len(groups), "counts": counts, "additions": additions,
            "new_useful_groups": {name: len(groups) for name, groups in distinct.items()}, "qualifies": qualifies}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "score", "replay", "review"))
    parser.add_argument("run", type=Path)
    parser.add_argument("--stage", choices=("diagnostic", "held"), default="held")
    for name in ("main", "external", "confirmation", "wavelet"):
        parser.add_argument(f"--{name}", type=Path)
    parser.add_argument("--lama", type=Path, default=prior.PRIVATE / "image-bg-real-acquisition-v225")
    parser.add_argument("--backup", type=Path, default=prior.PRIVATE / "image-bg-real-backup-intake-v262")
    args = parser.parse_args()
    if args.phase == "review":
        receipt = read(args.run)
        require(receipt["freeze_sha256"] == sha(json.dumps(receipt["freeze"], sort_keys=True).encode()), "Freeze differs")
        result = summary(receipt["freeze"], receipt["rows"])
        require(result == receipt["summary"], "Recorded result differs")
    else:
        import cv2

        from app.detectors.image import (
            reference_alignment,
            reference_compare,
            reference_exact_jpeg,
            reference_same_jpeg,
        )
        from app.evidence import image_reference

        require(all((args.main, args.external, args.confirmation, args.wavelet)), "Native source archives required")
        cv2.setNumThreads(1)
        modules = (prior, reference_alignment, reference_compare, reference_exact_jpeg,
                   reference_same_jpeg, image_reference)
        lock = {"schema": SCHEMA, "stage": args.stage, "design": DESIGN, "prior_paired_sha256": digest(PRIOR_PAIRED),
                "code": {Path(path).name: digest(Path(path)) for path in (__file__, *(m.__file__ for m in modules))},
                "cases": [case for case, *_ in cases(args, args.stage)]}
        if args.stage == "diagnostic":
            require(lock["cases"] == read(PRIOR_PAIRED)["freeze"]["cases"], "Exact prior panel was altered")
        if args.phase == "freeze":
            args.run.mkdir(mode=0o700)
            write(args.run / "freeze.json", lock)
            print(f"Froze {len(lock['cases'])} {args.stage} inputs", flush=True)
            return
        require(lock == read(args.run / "freeze.json"), "Frozen code or input bytes differ")
        rows = []
        for case in cases(args, args.stage):
            with TemporaryDirectory(dir=args.run) as scratch:
                rows.append(measure(case, Path(scratch)))
            if len(rows) % 25 == 0:
                print(f"Measured {len(rows)} {args.stage} pairs", flush=True)
        result = summary(lock, rows)
        record = {"freeze": lock, "freeze_sha256": sha(json.dumps(lock, sort_keys=True).encode()),
                  "rows": rows, "summary": result}
        if args.phase == "score":
            write(args.run / "receipt.json", record)
        else:
            require(read(args.run / "receipt.json") == record, "Native replay differs")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
