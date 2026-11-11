"""Freeze and run a bounded model-free copy corroboration transfer on v39 photos."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import platform
import re
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageFilter, features

from scripts import image_copy_corroboration_v295 as cue
from scripts import image_flux_photo_preparation_v276 as pilot
from scripts import image_source_groups_v22 as grouping

TRIAL = pilot.DATA / "image-copy-transfer-v298"
SIZE, PATCH, GROUPS, MAX_BYTES = 256, 64, 8, 5_000_000
SALT = "chipmunk-image-copy-transfer-source-order-v298"
VARIANTS = ("original", "tone", "blur", "translated_copy", "reflected_copy")
FORMATS = ("native_png", "jpeg90")
COPY_TRANSFORMS = {"translated_copy": "translation", "reflected_copy": "horizontal_reflection"}
TARGETS = {"interior": (96, 96, PATCH, PATCH), "edge": (0, 96, PATCH, PATCH)}
DONORS = {"aligned": (176, 176, PATCH, PATCH), "shifted": (171, 179, PATCH, PATCH)}
PLAN = "private-freeze.json"
RESULT = "private-result.json"
RECEIPTS = {
    "image-fresh-source-cache-v39-results.json": "f56c9290f190f3a8a29654b5781d1a233474b963429541cca93f706b6e18c537",
    "image-flux-photo-preparation-freeze-v276.json": "56fca6203ac581d4a5b6a290fc588d97483147fb47523d0b1dc41200939e112d",
    "image-flux-photo-expansion-v285-freeze.json": "119e72d3d2672689560a3833f43237710fefbf616e3475352ddca4764c7f9827",
    "image-flux-product-comparison-v288-freeze.json": "74737eb4624484009b76577e9328289544f9714c1703323cffd60247882cf821",
    "image-flux-product-comparison-complete-v288.json": "26feef3b938bca091a3e84d48c8f5d52fcd0921d425c7dcfc8c0a742ff2225ba",
    "image-copy-corroboration-v295-authored.json": "bc1634ee3ef41384e94430966cd8978044d9eaf45beea29be656e03abeadfdaa",
}
CODE = (
    Path(__file__),
    pilot.ROOT / "backend/tests/test_image_copy_transfer_v298.py",
    pilot.ROOT / "docs/image-copy-transfer-v298-method.md",
    Path(pilot.__file__),
    Path(grouping.__file__),
)
DETECTOR_STATUS = "unassessed_no_detector_run"
require, read_exact, digest = pilot.require, pilot.read_exact, pilot.flux.digest


def center_crop(size):
    width, height = size
    left, top = (width - SIZE) // 2, (height - SIZE) // 2
    return [left, top, left + SIZE, top + SIZE]


def source_order(row):
    return hashlib.sha256(f"{SALT}/{row['source_jpeg_sha256']}".encode()).hexdigest()


def identity(row):
    return (
        ("key", row["key"]),
        ("author", row["author"]),
        ("encoded", row["source_jpeg_sha256"]),
        ("rgb", row["rgb_sha256"]),
    )


def eligible_sources(graph, consumed):
    by_key = {row["key"]: row for row in graph["sources"]}
    groups = graph["candidate_only_unlinked_components"]
    exposed = {key for row in consumed for key in row["group"]}
    prior_hashes = {row["source_jpeg_sha256"] for row in consumed}
    prior_identity = {value for key in exposed for value in identity(by_key[key])}
    require(len(prior_hashes) == len(consumed), "Distinct consumed sources required")
    consumed_components = 0
    available = []
    for group in groups:
        members = [by_key[key] for key in group]
        if any(value in prior_identity for row in members for value in identity(row)):
            consumed_components += 1
            continue
        candidates = [
            row
            for row in members
            if row["license_id"] == 4 and row["license_url"] == pilot.CC_BY and min(row["size"]) >= 384
        ]
        if candidates:
            available.append((sorted(candidates, key=source_order), group))
    require(consumed_components == len(consumed), "All and only consumed v39 components must be excluded")
    available.sort(key=lambda pair: source_order(pair[0][0]))
    return available


def assign(available, attribution):
    planned = []
    strata = tuple((placement, phase) for placement in TARGETS for phase in DONORS) * 2
    require(len(strata) == GROUPS <= len(available), "Eight source groups required")
    seen = set()
    for index, ((candidates, group), (placement, phase)) in enumerate(
        zip(available[:GROUPS], strata, strict=True)
    ):
        row = candidates[0]
        source_identity = identity(row)
        require(
            row["author"] and not seen.intersection(source_identity), "Selected sources or authors overlap"
        )
        seen.update(source_identity)
        attr = attribution[row["source_id"]]
        require(
            attr["author_url"] == row["author_profile"]
            and grouping.author_key(attr["author_url"]) == row["author"]
            and attr["photo_page"] == row["photo_page"]
            and attr["license_url"] == row["license_url"] == pilot.CC_BY,
            "Exact recorded source attribution required",
        )
        planned.append(
            {
                "slot": index,
                "component": group,
                "source": row,
                "attribution": attr,
                "crop_ltrb_exclusive": center_crop(row["size"]),
                "placement": placement,
                "jpeg_phase": phase,
                "target_box_xywh": list(TARGETS[placement]),
                "donor_box_xywh": list(DONORS[phase]),
            }
        )
    return planned


def source_path(row):
    path = pilot.CACHE / f"{row['source_jpeg_sha256']}.bin"
    require(
        path.is_file()
        and path.resolve(strict=True) == path == Path(row["source_jpeg_file"])
        and 0 < path.stat().st_size == row["source_jpeg_bytes"] <= MAX_BYTES,
        "Exact direct cached source path and recorded size required",
    )
    return path


def metadata_plan():
    documents = {name: read_exact(pilot.PANELS / name, expected) for name, expected in RECEIPTS.items()}
    v39, v276, v285, v288, completed, authored = documents.values()
    v39_root = pilot.DATA / "image-fresh-source-cache-v39"
    graph = read_exact(v39_root / "private-groups.json", v39["private_groups_sha256"])
    catalog = read_exact(v39_root / "plan.json", v39["private_plan_sha256"])
    require(
        len(graph["sources"]) == 866
        and len(graph["candidate_only_unlinked_components"]) == 155
        and len(catalog["items"]) == 163
        and v276["audit"]["eligible_cc_by_2_0_groups"] == 89
        and v285["audit"]["eligible_cc_by_2_0_groups"] == 87
        and v276["audit"]["v39_later_exposure_linked"] == v285["audit"]["v39_later_exposure_linked"] == 0
        and v288["parent_receipts_sha256"]["image-flux-photo-expansion-v285-freeze.json"]
        == RECEIPTS["image-flux-photo-expansion-v285-freeze.json"]
        and completed["status"] == authored["status"] == "complete"
        and completed["freeze_sha256"] == RECEIPTS["image-flux-product-comparison-v288-freeze.json"],
        "Exact recorded source and exposure denominators required",
    )
    consumed = []
    for folder, prior in (
        ("image-flux-photo-preparation-v276", v276),
        ("image-flux-photo-expansion-v285", v285),
    ):
        data = read_exact(pilot.DATA / folder / "private-inputs.json", prior["private_inputs_sha256"])
        require(
            data["audit"] == prior["audit"] and len(data["rows"]) == prior["planned_calls"],
            "Exact source exposure required",
        )
        consumed.extend(data["rows"])
    available = eligible_sources(graph, consumed)
    require(
        len(available) == 81 and sum(len(rows) for rows, _ in available) == 83,
        "Exact unused v39 roster required",
    )
    planned = assign(available, {row["source_id"]: row["attribution"] for row in catalog["items"]})
    for row in planned:
        source_path(row["source"])
    method = cue.protocol()
    require(method == authored["protocol"], "Exact authored v295 method and runtime required")
    return {
        "schema": "chipmunk.image-copy-transfer-freeze/298",
        "receipts_sha256": RECEIPTS,
        "code_sha256": {str(path.relative_to(pilot.ROOT)): pilot.sha(path) for path in CODE},
        "v295": method,
        "python": platform.python_version(),
        "jpeg_turbo": features.version_feature("libjpeg_turbo"),
        "roster_salt": SALT,
        "eligible_unused_groups": len(available),
        "eligible_unused_files": sum(len(rows) for rows, _ in available),
        "formats": list(FORMATS),
        "variants": list(VARIANTS),
        "planned_groups": GROUPS,
        "planned_views": GROUPS * len(VARIANTS) * len(FORMATS),
        "detector_status": DETECTOR_STATUS,
        "rows": planned,
    }


def load_crop(row):
    source = row["source"]
    path = source_path(source)
    with path.open("rb") as stream:
        encoded = stream.read(MAX_BYTES + 1)
    require(
        len(encoded) == source["source_jpeg_bytes"]
        and hashlib.sha256(encoded).hexdigest() == source["source_jpeg_sha256"],
        "Encoded source binding failed",
    )
    with Image.open(io.BytesIO(encoded)) as image:
        require(
            image.format == "JPEG"
            and list(image.size) == source["size"]
            and image.getexif().get(274, 1) == 1,
            "Native upright source geometry differs",
        )
        rgb = image.convert("RGB")
        require(digest(np.asarray(rgb)) == source["rgb_sha256"], "Native source RGB differs")
        require(center_crop(rgb.size) == row["crop_ltrb_exclusive"], "Frozen crop geometry differs")
        return np.asarray(rgb.crop(row["crop_ltrb_exclusive"]))


def controls(source, target, donor):
    require(source.shape == (SIZE, SIZE, 3) and source.dtype == np.uint8, "Expected native RGB crop")
    proposal = cue.mask_for(source.shape[:2], target)
    donor_mask = cue.mask_for(source.shape[:2], donor)
    require(
        proposal.sum() == donor_mask.sum() == PATCH * PATCH and not (proposal & donor_mask).any(),
        "Disjoint, fully in-frame target and donor rectangles required",
    )
    # Tone and blur match the v276 recipes; a local explicit donor replaces its whole-frame mirror.
    tone = np.clip(
        np.rint((source.astype(np.float64) - 127.5) * 1.12 + 127.5 + np.asarray([3, -2, 1])), 0, 255
    ).astype(np.uint8)
    blur = np.asarray(Image.fromarray(source).filter(ImageFilter.GaussianBlur(1.25)))
    x, y, width, height = donor
    patch = source[y : y + height, x : x + width]
    rendered = {"original": source}
    for kind, pixels in (
        ("tone", tone[proposal]),
        ("blur", blur[proposal]),
        ("translated_copy", patch.reshape(-1, 3)),
        ("reflected_copy", patch[:, ::-1].reshape(-1, 3)),
    ):
        rendered[kind] = source.copy()
        rendered[kind][proposal] = pixels
    return proposal, rendered


def encode_view(rgb, encoding):
    buffer = io.BytesIO()
    image = Image.fromarray(rgb)
    if encoding == "jpeg90":
        image.save(buffer, format="JPEG", quality=cue.JPEG_QUALITY)
    else:
        image.save(buffer, format="PNG")
    payload = buffer.getvalue()
    with Image.open(io.BytesIO(payload)) as decoded:
        pixels = np.asarray(decoded.convert("RGB"))
    require(pixels.shape == rgb.shape, "Encoding changed source geometry")
    if encoding == "native_png":
        require(np.array_equal(pixels, rgb), "Lossless view changed source RGB")
    return pixels, hashlib.sha256(payload).hexdigest()


def score_group(row, source):
    proposal, variants = controls(source, row["target_box_xywh"], row["donor_box_xywh"])
    original = {encoding: encode_view(source, encoding) for encoding in FORMATS}
    expected_mask_digest = digest(proposal)
    for kind, image in variants.items():
        truth = np.any(image != source, axis=2)
        require(not (truth & ~proposal).any(), "Ordinary edit escaped the frozen target")
        for encoding in FORMATS:
            source_view, source_encoded_hash = original[encoding]
            submitted, encoded_hash = (
                (source_view, source_encoded_hash) if kind == "original" else encode_view(image, encoding)
            )
            observed = np.any(submitted != source_view, axis=2)
            result = cue.corroborate(submitted, proposal)
            require(
                result["proposal_mask_sha256"] == digest(proposal) == expected_mask_digest,
                "Corroboration changed the source highlight",
            )
            donor = result["donor"]
            recovered = bool(
                truth.any()
                and donor
                and donor["transform"] == COPY_TRANSFORMS.get(kind)
                and donor["box_xywh"] == row["donor_box_xywh"]
            )
            yield {
                **result,
                "slot": row["slot"],
                "placement": row["placement"],
                "jpeg_phase": row["jpeg_phase"],
                "variant": kind,
                "format": encoding,
                "submitted_rgb_sha256": digest(submitted),
                "submitted_encoded_sha256": encoded_hash,
                "native_change_mask_sha256": digest(truth),
                "native_changed_pixels": int(truth.sum()),
                "encoded_changes_outside_native_truth": int((observed & ~truth).sum()),
                "exact_ordinary_donor_recovered": recovered,
                "detector_status": DETECTOR_STATUS,
            }


def readout(plan):
    cv2.setNumThreads(1)
    results, input_errors = [], []
    for row in plan["rows"]:
        try:
            source = load_crop(row)
        except (OSError, ValueError) as error:
            input_errors.append({"slot": row["slot"], "reason": str(error)})
            continue
        results.extend(score_group(row, source))
    return results, input_errors


def aggregate(plan, results, input_errors):
    keys = {(row["slot"], row["variant"], row["format"]) for row in results}
    error_slots = {row["slot"] for row in input_errors}
    require(
        len(plan["rows"]) == GROUPS
        and len(error_slots) == len(input_errors)
        and error_slots <= {row["slot"] for row in plan["rows"]},
        "Expected a complete fixed source denominator",
    )
    expected = {
        (row["slot"], kind, encoding)
        for row in plan["rows"]
        if row["slot"] not in error_slots
        for kind in VARIANTS
        for encoding in FORMATS
    }
    require(len(keys) == len(results) and keys == expected, "Missing or duplicate scored view")
    by_format = {}
    for encoding in FORMATS:
        by_format[encoding] = {}
        for kind in VARIANTS:
            rows = [row for row in results if (row["format"], row["variant"]) == (encoding, kind)]
            statuses = Counter(row["status"] for row in rows)
            changes = [row["encoded_changes_outside_native_truth"] for row in rows]
            by_format[encoding][kind] = {
                "planned": GROUPS,
                "scored": len(rows),
                "source_unavailable": len(error_slots),
                "outcomes": dict(sorted(statuses.items())),
                "by_placement_and_jpeg_phase": {
                    f"{position}_{phase}": dict(
                        sorted(
                            Counter(
                                row["status"]
                                for row in rows
                                if (row["placement"], row["jpeg_phase"]) == (position, phase)
                            ).items()
                        )
                    )
                    for position in TARGETS
                    for phase in DONORS
                },
                "exact_ordinary_donors_recovered": sum(row["exact_ordinary_donor_recovered"] for row in rows),
                "copy_supported_without_exact_ordinary_donor": sum(
                    row["status"] == "copy_consistent" and not row["exact_ordinary_donor_recovered"]
                    for row in rows
                ),
                "unchanged_native_controls": sum(row["native_changed_pixels"] == 0 for row in rows),
                "encoded_changes_outside_native_truth_range": [min(changes), max(changes)]
                if changes
                else None,
            }
    return {
        "schema": "chipmunk.image-copy-transfer-aggregate/298",
        "status": "complete" if not input_errors else "incomplete_source_inputs",
        "planned_groups": GROUPS,
        "planned_views": plan["planned_views"],
        "scored_views": len(results),
        "source_unavailable_groups": len(error_slots),
        "cue_evaluated_views": sum(
            row["status"] not in {"unsupported_geometry", "insufficient_texture"} for row in results
        ),
        "detector_unassessed_views": plan["planned_views"],
        "generated_edits": 0,
        "generator_calls": 0,
        "detector_calls": 0,
        "by_format": by_format,
    }


def run(freeze_sha):
    require(re.fullmatch(r"[a-f0-9]{64}", freeze_sha), "Explicit frozen metadata digest required")
    frozen = read_exact(TRIAL / PLAN, freeze_sha)
    require(frozen == metadata_plan(), "Frozen roster, code, or runtime changed")
    require(not (TRIAL / RESULT).exists(), "This frozen run already has a result")
    rows, errors = readout(frozen)
    result = aggregate(frozen, rows, errors)
    pilot.write(
        TRIAL / RESULT,
        {"aggregate": result, "freeze_sha256": freeze_sha, "rows": rows, "input_errors": errors},
    )
    return {**result, "freeze_sha256": freeze_sha, "private_result_sha256": pilot.sha(TRIAL / RESULT)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("preflight")
    commands.add_parser("freeze")
    commands.add_parser("run").add_argument("--freeze-sha256", required=True)
    args = parser.parse_args()
    if args.command == "run":
        report = run(args.freeze_sha256)
    else:
        plan = metadata_plan()
        report = {
            key: plan[key]
            for key in ("eligible_unused_groups", "eligible_unused_files", "planned_groups", "planned_views")
        }
        report.update(media_opened=False, detector_calls=0, generator_calls=0)
        if args.command == "freeze":
            TRIAL.mkdir(mode=0o700)
            pilot.write(TRIAL / PLAN, plan)
            report["freeze_sha256"] = pilot.sha(TRIAL / PLAN)
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0 if report.get("status") != "incomplete_source_inputs" else 1


if __name__ == "__main__":
    raise SystemExit(main())
