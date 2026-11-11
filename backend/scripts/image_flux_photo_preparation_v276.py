"""Prepare an exact, private, model-free two-photo reference-edit pilot."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import subprocess
import sys
import tempfile
import types
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from scripts import image_flux_reference_edit_v271 as flux
from scripts.image_source_groups_v22 import components, sketch

ROOT, DATA, ARCHIVE = flux.ROOT, flux.DATA, flux.ARCHIVE
PANELS = ROOT / "docs/evaluation-panels"
TRIAL = DATA / "image-flux-photo-preparation-v276"
STAGE = TRIAL / "authored"
OUTPUT = STAGE / "output"
MODULE = "scripts.image_flux_photo_preparation_v276"
PUBLIC = PANELS / "image-flux-photo-preparation-freeze-v276.json"
CACHE = ARCHIVE / "image-fresh-source-cache-v39/objects"
RAW_TRIAL = DATA / "image-rawnind-phase-v239-attempt02"
CC_BY = "https://creativecommons.org/licenses/by/2.0/"
SIZE = flux.SIZE
SEED = 276_004
SALT = "chipmunk-licensed-flux-pilot-source-order-v276"
CASES = (
    (
        "interior",
        (96, 96, 160, 160),
        "Add one small bright red square sticker in the exact center of the photo. Keep the framing and rest of the scene unchanged.",
    ),
    (
        "edge",
        (0, 96, 64, 160),
        "Add one small bright yellow square sticker centered on the left edge of the photo, partially outside the frame. Keep the framing and rest of the scene unchanged.",
    ),
)
RECEIPTS = {
    "image-fresh-source-cache-v39-results.json": "f56c9290f190f3a8a29654b5781d1a233474b963429541cca93f706b6e18c537",
    "image-bg-real-pixel-qualification-complete-v225.json": "3a46ad593be6a095f25e5998f7c51d9d592de6d413915348b57271111ec50306",
    "image-bg-real-backup-intake-v262-complete.json": "74a07ec74a63e0e252904a37f915d1cb1d70d1b0cba01aae4be6a94f7412e51e",
    "image-rawnind-phase-complete-v239.json": "7e75fab7c5e07ba479e2afe3f51f26cbfb3402b196b8208a37587db38e5e55fa",
    "image-flux-reference-edit-smoke-freeze-v271.json": "ad88fb0eda60b9427536ea8a2a8ca049a603d48996fa44ce27f96e2512f0ceff",
    "image-flux-reference-edit-complete-v271.json": "3f9f2b8b60326d423e4bf899c47204c3473bed8b4c6926c850ba9c201ab3efeb",
}
EXPOSURE_SCOPE = {
    "v19_v22_v26_v39": "Original 705-record v39 graph, v19 cross-provider metadata and every v26 source",
    "v41_v42_v45_v46_v47_v49_v50_v51_v189": "Regional and diagnostic panels reuse v26 sources; product replay also used an authored 16x16 image",
    "v218_v221_v225_v229_v262_v264_v265": "All 36 qualified BG-REAL native originals; downstream probes derive from these",
    "v239_v244": "Four RawNIND original captures decoded with the exact retained wheel",
    "v271": "Two authored images without project-photo exposure",
}

require, sha, write = flux.require, flux.sha, flux.write


def read_exact(path, expected):
    require(path.resolve(strict=True) == path and sha(path) == expected, "Committed local record differs")
    return json.loads(path.read_bytes())


def bg_sources(receipt, private_path, count):
    sealed = read_exact(private_path, receipt["qualified_private_sha256"])
    folder = Path(sealed["cache"])
    require(
        folder.parent == ARCHIVE and len(sealed["qualified"]) == count,
        "Complete sealed BG-REAL group required",
    )
    rows = []
    for index, item in enumerate(sealed["qualified"]):
        source, measurement = item["source"], item["measurement"]
        original = measurement["fingerprints"]["original"]
        spec = measurement["files"]["original"]
        path = folder / f"{index:02d}/original.png"
        require(
            measurement["ordinal"] == index
            and path.is_file()
            and sha(path) == original["encoded_sha256"] == spec["sha256"],
            "Retained exact BG-REAL original required",
        )
        with Image.open(path) as decoded:
            rgb = decoded.convert("RGB")
            require(
                list(rgb.size) == original["size"]
                and flux.digest(np.asarray(rgb)) == original["rgb_sha256"]
                and f"{sketch(rgb):064x}" == original["sketch"],
                "BG-REAL source readback differs",
            )
        rows.append(
            {
                "key": f"bg-real/{source['recipient']}",
                "source_id": f"flickr/{source['flickr_photo_id']}",
                "author": source["flickr_owner"].casefold(),
                "encoded_sha256": original["encoded_sha256"],
                **original,
            }
        )
    return rows


def raw_sources(receipt):
    from scripts import image_rawnind_phase_v239 as raw

    roster = raw.frozen_roster(RAW_TRIAL)
    require(
        receipt["captures"] == len(roster) == 4 and receipt["roster_sha256"] == raw.ROSTER_SHA,
        "Exact RawNIND replay required",
    )
    replay = read_exact(
        RAW_TRIAL / "replay-private.json", "2337ac7cfc417ea506348e79b2847714dfce5d8f50846485d75aa143294d1b8d"
    )
    native = [row for row in replay["rows"] if row["view"] == "native"]
    require(
        len(native) == len(roster) and "rawpy" not in sys.modules,
        "Fresh, complete native exposure audit required",
    )
    wheel = raw.verified_wheel(RAW_TRIAL / raw.WHEEL)
    rows = []
    with tempfile.TemporaryDirectory(prefix="chipmunk-rawnind-v276-") as directory:
        raw.unpack_decoder(wheel, directory)
        sys.path.insert(0, directory)
        try:
            decoder = importlib.import_module("rawpy")
            require(
                Path(decoder.__file__).resolve().is_relative_to(Path(directory).resolve()),
                "Exact isolated RawNIND decoder required",
            )
            for index, (spec, expected) in enumerate(zip(roster, native, strict=True)):
                path = raw.raw_path(RAW_TRIAL, index, spec)
                raw.verify_raw(raw.read_private(path, spec["bytes"]), spec)
                pixels = raw.decode(decoder, path)
                rgb = Image.fromarray(pixels)
                require(
                    flux.digest(pixels) == expected["rgb_sha256"]
                    and list(rgb.size) == expected["source_size"],
                    "RawNIND RGB replay differs",
                )
                rows.append(
                    {
                        "key": f"rawnind/{index}",
                        "source_id": f"rawnind/{index}",
                        "author": None,
                        "size": list(rgb.size),
                        "encoded_sha256": sha(path),
                        "rgb_sha256": expected["rgb_sha256"],
                        "sketch": f"{sketch(rgb):064x}",
                    }
                )
                del rgb, pixels
        finally:
            sys.path.pop(0)
    return rows


def cross_exposure(known, extra):
    graphs, _ = components(known + extra)
    prior = {row["key"] for row in known if not row["key"].startswith("fresh-jpeg/")} | {
        row["key"] for row in extra
    }
    rejected = {key for group in graphs if prior.intersection(group) for key in group}
    return [group for group in graphs if not rejected.intersection(group)]


def source_order(row):
    return hashlib.sha256(f"{SALT}/{row['source_jpeg_sha256']}".encode()).hexdigest()


def select(candidates, groups):
    by_key = {row["key"]: row for row in candidates}
    eligible = []
    for group in groups:
        eligible_group = [
            by_key[key]
            for key in group
            if key in by_key and by_key[key]["license_id"] == 4 and min(by_key[key]["size"]) >= 384
        ]
        if eligible_group:
            eligible.append((min(eligible_group, key=source_order), group))
    eligible.sort(key=lambda pair: source_order(pair[0]))
    require(len(eligible) >= len(CASES), "Insufficient distinct CC BY source groups")
    return eligible[: len(CASES)], len(eligible)


def audit():
    from scripts import image_bg_real_roster_v218 as bg_roster
    from scripts import image_fresh_source_cache_v39 as cache

    receipts = {name: read_exact(PANELS / name, value) for name, value in RECEIPTS.items()}
    v39 = receipts["image-fresh-source-cache-v39-results.json"]
    qualified = receipts["image-flux-reference-edit-complete-v271.json"]
    require(
        qualified["status"] == "complete"
        and qualified["new_neural_calls"] == 2
        and qualified["project_photos_used"] == 0
        and qualified["freeze_sha256"] == RECEIPTS["image-flux-reference-edit-smoke-freeze-v271.json"],
        "Verified authored v271 prerequisites required",
    )
    # Also rebind the exact v19/v26/v39/v189 source universe recorded in v218.
    bg_roster.exposure_keys()
    root = DATA / "image-fresh-source-cache-v39"
    graph = read_exact(root / "private-groups.json", v39["private_groups_sha256"])
    plan = read_exact(root / "plan.json", v39["private_plan_sha256"])
    by_source = {row["source_id"]: row for row in plan["items"]}
    known = graph["sources"]
    candidates = [row for row in known if row["key"].startswith("fresh-jpeg/")]
    require(
        len(known) == 866 and len(candidates) == 161 and CACHE.is_dir(),
        "Complete connected v39 reference cache required",
    )
    indexed = []
    for row in candidates:
        item = by_source[row["source_id"]]
        path = CACHE / f"{row['source_jpeg_sha256']}.bin"
        require(
            path.resolve(strict=True) == path
            and cache.pixel_record(item, path, row["source_jpeg_sha256"]) == row,
            "v39 source or attribution changed",
        )
        attr = item["attribution"]
        indexed.append(
            {
                **row,
                "source_id": f"flickr/{attr['photo_id']}",
                "attribution": attr,
                "encoded_sha256": row["source_jpeg_sha256"],
            }
        )
    bg = bg_sources(
        receipts["image-bg-real-pixel-qualification-complete-v225.json"],
        DATA / "image-bg-real-acquisition-v225/qualified-private.json",
        24,
    )
    bg += bg_sources(
        receipts["image-bg-real-backup-intake-v262-complete.json"],
        DATA / "image-bg-real-backup-intake-v262/qualified-private.json",
        12,
    )
    raw_rows = raw_sources(receipts["image-rawnind-phase-complete-v239.json"])
    prior_rows = [row for row in known if not row["key"].startswith("fresh-jpeg/")]
    groups = cross_exposure(prior_rows + indexed, bg + raw_rows)
    covered = {key for group in groups for key in group}
    exact_exposed = {row["encoded_sha256"] for row in bg + raw_rows} | {
        row["original_sha256"] for row in prior_rows if "original_sha256" in row
    }
    collision_groups = {
        key
        for group in groups
        if any(row["key"] in group and row["encoded_sha256"] in exact_exposed for row in indexed)
        for key in group
    }
    eligible = [row for row in indexed if row["key"] not in collision_groups]
    selected, group_count = select(eligible, groups)
    return selected, {
        "v39_candidates_verified": len(candidates),
        "v39_prior_records": len(prior_rows),
        "bg_real_native_sources_verified": len(bg),
        "rawnind_native_sources_verified": len(raw_rows),
        "v39_later_exposure_linked": len(indexed) - len(covered - collision_groups),
        "licenses": dict(Counter(str(row["license_id"]) for row in candidates)),
        "eligible_cc_by_2_0_groups": group_count,
        "scope": EXPOSURE_SCOPE,
    }


def load_crop(path):
    with Image.open(path) as image:
        require(
            image.format == "JPEG" and min(image.size) >= 384 and image.getexif().get(274, 1) == 1,
            "Expected upright native JPEG crop",
        )
        left, top = (image.width - SIZE) // 2, (image.height - SIZE) // 2
        bounds = (left, top, left + SIZE, top + SIZE)
        return np.array(image.convert("RGB").crop(bounds)), bounds


def make_controls(source, bounds):
    from scripts import image_jpeg_diagnostic_v189 as jpeg

    left, top, right, bottom = bounds
    mask = np.zeros((SIZE, SIZE), dtype=np.uint8)
    mask[top:bottom, left:right] = 255
    image, alpha = Image.fromarray(source), Image.fromarray(mask)
    tone = np.clip(
        np.rint((source.astype(np.float64) - 127.5) * 1.12 + 127.5 + np.asarray([3, -2, 1])), 0, 255
    ).astype(np.uint8)
    arrays = {
        "source": source,
        "mask": mask,
        "tone": np.asarray(Image.composite(Image.fromarray(tone), image, alpha)),
        "blur": np.asarray(Image.composite(image.filter(ImageFilter.GaussianBlur(1.25)), image, alpha)),
        "copy_move": np.asarray(jpeg.ordinary_copy(image, alpha)),
    }
    changes = {}
    for key in ("tone", "blur", "copy_move"):
        exact = np.any(source != arrays[key], axis=2)
        require(exact.any() and not exact[mask == 0].any(), "Ordinary control violates exact mask geometry")
        arrays[f"{key}_changed"] = exact.astype(np.uint8) * 255
        changes[key] = int(exact.sum())
    return arrays, changes


def inputs(selected):
    for (case, bounds, prompt), (row, group) in zip(CASES, selected, strict=True):
        path = CACHE / f"{row['source_jpeg_sha256']}.bin"
        require(
            row["license_id"] == 4 and row["license_url"] == CC_BY, "Exact CC BY 2.0 attribution required"
        )
        source, crop = load_crop(path)
        arrays, changes = make_controls(source, bounds)
        attr = row["attribution"]
        yield (
            {
                "slot": case,
                "group": group,
                "source_jpeg_sha256": row["source_jpeg_sha256"],
                "source_rgb_sha256": row["rgb_sha256"],
                "crop_ltrb_exclusive": list(crop),
                "mask_ltrb_exclusive": list(bounds),
                "seed": SEED,
                "prompt": prompt,
                "attribution": {
                    key: attr[key]
                    for key in ("author_name", "author_url", "title", "photo_page", "license_url")
                },
                "changed_pixels": changes,
                "rgb_or_mask_sha256": {key: flux.digest(value) for key, value in arrays.items()},
            },
            arrays,
        )


def prepare(output):
    require(
        output.parent.resolve(strict=True) == output.parent and not output.exists(),
        "Fresh private preparation folder required",
    )
    selected, audit_report = audit()
    ready = list(inputs(selected))
    output.mkdir(mode=0o700, exist_ok=False)
    for row, arrays in ready:
        folder = output / row["slot"]
        folder.mkdir(mode=0o700, exist_ok=False)
        row["png_sha256"] = {
            key: flux.save_png(folder / f"{key}.png", value) for key, value in arrays.items()
        }
    private = {
        "schema": "chipmunk.image-flux-photo-prepared-private/276",
        "audit": audit_report,
        "rows": [row for row, _ in ready],
    }
    write(output / "private-inputs.json", private)
    return audit_report


def verify_prepared(folder):
    manifest = folder / "private-inputs.json"
    private = read_exact(manifest, sha(manifest))
    require(
        private["schema"] == "chipmunk.image-flux-photo-prepared-private/276"
        and len(private["rows"]) == len(CASES),
        "Exact two pilot cases required",
    )
    keys = {
        "source",
        "mask",
        "tone",
        "blur",
        "copy_move",
        "tone_changed",
        "blur_changed",
        "copy_move_changed",
    }
    for row, (slot, bounds, prompt) in zip(private["rows"], CASES, strict=True):
        require(
            row["slot"] == slot
            and row["mask_ltrb_exclusive"] == list(bounds)
            and row["prompt"] == prompt
            and row["seed"] == SEED
            and set(row["png_sha256"]) == set(row["rgb_or_mask_sha256"]) == keys,
            "Exact case, control and geometry keys required",
        )
        for key, expected in row["png_sha256"].items():
            path = folder / row["slot"] / f"{key}.png"
            require(path.resolve(strict=True) == path and sha(path) == expected, "Prepared PNG changed")
            with Image.open(path) as img:
                require(
                    img.size == (SIZE, SIZE)
                    and flux.digest(np.asarray(img)) == row["rgb_or_mask_sha256"][key],
                    "Prepared pixels changed",
                )
    return private


def freeze_plan(folder):
    manifest = folder / "private-inputs.json"
    private = verify_prepared(folder)
    code = (
        Path(__file__),
        ROOT / "backend/tests/test_image_flux_photo_preparation_v276.py",
        ROOT / "backend/scripts/image_flux_reference_edit_v271.py",
        ROOT / "backend/scripts/image_fresh_source_cache_v39.py",
        ROOT / "backend/scripts/image_source_groups_v22.py",
        ROOT / "backend/scripts/image_rawnind_phase_v239.py",
        ROOT / "backend/scripts/image_jpeg_diagnostic_v189.py",
        ROOT / "backend/scripts/image_bg_real_roster_v218.py",
        ROOT / "backend/scripts/image_pair_truth_v21.py",
        ROOT / "backend/scripts/local_probe_limits_v24.py",
        ROOT / "backend/scripts/audio_tree_sampler_v64.py",
        ROOT / "backend/scripts/video_appearance_runtime_v108.py",
        ROOT / "backend/scripts/video_appearance_native_v108.py",
        ROOT / "backend/scripts/video_appearance_corrected_v112.py",
        ROOT / "docs/image-flux-photo-preparation-v276-method.md",
    )
    return {
        "schema": "chipmunk.image-flux-photo-preparation-freeze/276",
        "private_inputs_sha256": sha(manifest),
        "exposure_receipts_sha256": RECEIPTS,
        "code_sha256": {str(path.relative_to(ROOT)): sha(path) for path in code},
        "audit": private["audit"],
        "cases": [
            {"slot": slot, "mask_ltrb_exclusive": list(bounds), "prompt": prompt}
            for slot, bounds, prompt in CASES
        ],
        "planned_calls": 2,
        "seed": SEED,
        "model_revision": flux.REVISION,
        "acquisition_sha256": flux.ACQUISITION_SHA,
        "steps": flux.STEPS,
        "formats": ["lossless PNG"],
        "detector_calls": 0,
        "guidance": 1.0,
        "scheduler": flux.SCHEDULER,
        "use_kv_cache": False,
        "device": "mlx.gpu",
        "initial_ram_bytes": 6 * 1024**3,
        "max_sampled_tree_rss_bytes": 6 * 1024**3,
        "seconds_limit": 900,
        "scope": "Two licensed center crops for construction feasibility. Exact source-coordinate insertion; native reference-edit output is composed after generation. No detector outcome or training admission.",
    }


def freeze():
    private = verify_prepared(TRIAL)
    selected, audit_report = audit()
    prepared = [{key: value for key, value in row.items() if key != "png_sha256"} for row in private["rows"]]
    require(
        audit_report == private["audit"] and prepared == [row for row, _ in inputs(selected)],
        "Prepared roster and controls must replay before freeze",
    )
    write(PUBLIC, freeze_plan(TRIAL))
    return {"freeze_sha256": sha(PUBLIC)}


def generation_inputs(folder=TRIAL):
    expected = read_exact(PUBLIC, sha(PUBLIC))
    require(expected == freeze_plan(folder), "Exact prospective v276 source freeze required")
    return json.loads((folder / "private-inputs.json").read_bytes())["rows"]


def worker():
    require(
        (STAGE / "resources.jsonl").is_file()
        and (STAGE / "worker.log").is_file()
        and os.environ.get("HF_HUB_OFFLINE") == os.environ.get("TRANSFORMERS_OFFLINE") == "1",
        "Guarded offline worker required",
    )
    source_rows = generation_inputs()
    network = []

    def offline(event, _args):
        if event in {
            "socket.connect",
            "socket.sendto",
            "socket.getaddrinfo",
            "socket.gethostbyname",
            "socket.gethostbyaddr",
        }:
            network.append(event)
            raise RuntimeError("Offline worker refused network activity")

    sys.addaudithook(offline)
    runtime = flux.runtime_identity()
    flux.verify_bundle(flux.BUNDLE, flux.inventory())
    import mlx.core as mx
    from mflux.models.flux2.variants.edit.flux2_klein_edit import Flux2KleinEdit

    for name in ("mflux", "numpy", "PIL", "mlx.core"):
        require(
            Path(sys.modules[name].__file__).resolve().is_relative_to(flux.ENV),
            "Dependency escaped isolated v40 runtime",
        )
    mx.set_default_device(mx.gpu)
    mx.set_cache_limit(128 * 1024**2)
    model = Flux2KleinEdit(model_path=str(flux.BUNDLE), small_text_encoder=False, small_decoder=False)
    require(model.bits == 4, "Loaded model did not report 4-bit weights")
    OUTPUT.mkdir(mode=0o700)
    journal = OUTPUT / "attempts.jsonl"
    rows = []
    with journal.open("x") as stream:

        def record(value):
            stream.write(json.dumps(value, allow_nan=False) + "\n")
            stream.flush()

        for item in source_rows:
            slot = item["slot"]
            source_path = TRIAL / slot / "source.png"
            with Image.open(source_path) as image, Image.open(TRIAL / slot / "mask.png") as selection:
                source, mask = np.asarray(image), np.asarray(selection)
            require(
                sha(source_path) == item["png_sha256"]["source"]
                and flux.digest(source) == item["rgb_or_mask_sha256"]["source"]
                and flux.digest(mask) == item["rgb_or_mask_sha256"]["mask"],
                "Frozen source or selection changed before generation",
            )
            folder = OUTPUT / slot
            folder.mkdir(mode=0o700)
            record({"slot": slot, "status": "generator_started"})
            output = model.generate_image(
                seed=item["seed"],
                prompt=item["prompt"],
                num_inference_steps=flux.STEPS,
                width=SIZE,
                height=SIZE,
                guidance=1.0,
                scheduler=flux.SCHEDULER,
                use_kv_cache=False,
                image_paths=[source_path],
            )
            row = {"slot": slot, "status": "unsupported_output_geometry"}
            if output.image.mode == "RGB" and output.image.size == (SIZE, SIZE):
                raw_rgb = np.asarray(output.image)
                changes = np.any(source != raw_rgb, axis=2)
                row = {
                    "slot": slot,
                    "status": "no_in_mask_change",
                    "raw_changes_inside": int((changes & (mask > 0)).sum()),
                    "raw_changes_outside": int((changes & (mask == 0)).sum()),
                    "raw_png_sha256": flux.save_png(folder / "raw.png", raw_rgb),
                }
                if row["raw_changes_inside"]:
                    composite, truth, proof = flux.compose(source, raw_rgb, mask)
                    row |= {
                        **proof,
                        "status": "qualified_geometry",
                        "composite_png_sha256": flux.save_png(folder / "composite.png", composite),
                        "changed_png_sha256": flux.save_png(folder / "changed.png", truth),
                    }
            rows.append(row)
            record(row)
            mx.clear_cache()
    require(not network and len(rows) == len(CASES), "All frozen source attempts and offline state required")
    write(
        OUTPUT / "result.json",
        {
            "schema": "chipmunk.image-flux-photo-generation-result/276",
            "status": "complete",
            "freeze_sha256": sha(PUBLIC),
            "runtime": runtime,
            "attempt_journal_sha256": sha(journal),
            "new_neural_calls": len(rows),
            "detector_calls": 0,
            "network_attempts": 0,
            "case_outcomes": dict(Counter(row["status"] for row in rows)),
            "mlx_peak_active_memory_bytes": int(mx.get_peak_memory()),
            "cases": rows,
        },
    )


def run():
    from scripts import video_appearance_runtime_v108 as guard
    from scripts.video_appearance_corrected_v112 import scoped

    def launch(argv, **kwargs):
        require(
            argv[1:] == ["-B", "-m", "scripts.video_appearance_native_v108", "authored", str(OUTPUT)],
            "Exact guarded photo worker required",
        )
        kwargs["env"].update(
            PYTHONNOUSERSITE="1",
            HF_HUB_DISABLE_TELEMETRY="1",
            HF_HUB_DISABLE_IMPLICIT_TOKEN="1",
            HF_HOME=str(STAGE / "hub-offline"),
        )
        return subprocess.Popen([str(flux.ENV / "bin/python"), "-B", "-m", MODULE, "worker"], **kwargs)

    proxy = types.SimpleNamespace(
        Popen=launch, STDOUT=subprocess.STDOUT, TimeoutExpired=subprocess.TimeoutExpired
    )
    scoped(
        guard.run,
        {
            "TRIAL": TRIAL,
            "verify": generation_inputs,
            "prerequisites": lambda *_: True,
            "subprocess": proxy,
            "__file__": __file__,
        },
    )("authored")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("audit", "prepare", "freeze", "run", "worker"))
    args = parser.parse_args()
    if args.command == "audit":
        chosen, report = audit()
        report["pilot_cases"] = [
            {"slot": row["slot"], "controls": row["changed_pixels"]} for row, _ in inputs(chosen)
        ]
    elif args.command == "prepare":
        report = prepare(TRIAL)
    elif args.command == "freeze":
        report = freeze()
    else:
        {"run": run, "worker": worker}[args.command]()
        report = None
    if report is not None:
        print(json.dumps(report, indent=2), flush=True)
