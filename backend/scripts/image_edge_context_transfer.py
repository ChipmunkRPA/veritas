# SPDX-License-Identifier: Apache-2.0
"""Compare edge-extended padding on known locations and matched ordinary uploads."""
from __future__ import annotations

import argparse
import io
import json
from collections import Counter, defaultdict
from pathlib import Path

from scripts.image_small_region_repeat import CHECKPOINT, read, require, save, sha, useful

ROOT = Path(__file__).resolve().parents[2]
PUBLIC = ROOT / "docs/evaluation-panels"
DATA = ROOT / "backend/data/independent-evaluation"
SCHEMA = "chipmunk.image-edge-context-transfer/1"
ARMS = ("current", "edge", "supported")
BACKGROUND_GROUPS = 8
RECODE_GROUPS = 20
NATIVE_ROLES = ("ai", "original", "tone", "blur")
RECODE_ROLES = ("ai", "original")
BACKGROUND_ROLES = ("ai", "original", "tone", "blur")
METHOD = {
    "native": "Keep all 59 previously measured native images and their four roles (48 source groups). "
              "Verify the exact archived current responses, delivered PNG bytes, and changed-pixel truth.",
    "recoded": "Take the first 20 distinct source groups in frozen confirmation order with a short side "
               "at least two thirds of the long side. Encode the AI edit and its original as upright "
               "768-pixel-long-side JPEG75. Use the exact post-resize, pre-encode changed pixels.",
    "background": "Take eight distinct qualified LaMa groups in byte-hash order, disjoint from the two prior "
                  "padding/small-region and progressive-earlier-copy panels. Keep the exact AI edit and "
                  "matched original, local tone and local blur as native 768-side JPEG75 deliveries.",
    "arms": "Current black fill. Edge uses the nearest actual border pixel outside the source. Supported keeps "
            "each current component with at least half its grid pixels also retained in the edge pass. "
            "All views use the same source crop, thresholds, selection caps and reliability grid.",
    "useful": "At least 0.30 intersection over union and 0.50 recall on exact pre-encode changed pixels. "
              "Count useful losses, gains, misplaced flags and ordinary flags separately by cohort and geometry.",
}


def native_deliveries(args):
    import cv2
    import numpy as np
    from PIL import Image

    from scripts.image_native_confirmation_v46 import comparison_tasks, encoded_view
    from scripts.image_regional_training_v26 import source_views

    parent = read(PUBLIC / "native-regional-serving-v47-freeze.json")
    plan, references = (args.native / name for name in ("plan.json", "references.json"))
    for path, key in ((plan, "plan_sha256"), (references, "references_sha256")):
        require(sha(path.read_bytes()) == parent[key], "Archived native reference changed")
    plan, references = read(plan), read(references)
    cases = {row["source_id"]: row for row in plan["rows"]}
    qualifications = {row["source_id"]: row for row in (
        json.loads(line) for line in (args.pairs / "qualification/pairs.jsonl").read_text().splitlines())}
    recoded = set()
    for pair, task in enumerate(comparison_tasks(plan)[::4]):
        case = cases[task["source_id"]]
        edited, controls, truth = source_views(case, qualifications[case["source_id"]])
        source = case["original_sha256"]
        views = {"ai": edited, **controls}
        for index, role in enumerate(NATIVE_ROLES):
            reference = references[4 * pair + index]["candidate"]
            body, rgb = encoded_view(views[role], "native")
            require(reference["source_id"] == case["source_id"] and reference["kind"] == role
                    and sha(body) == reference["input_sha256"], "Archived native view changed")
            yield ("native_png", task["component"], source, role, body, np.asarray(rgb),
                   (truth > 0) if role == "ai" else np.zeros_like(truth, bool), reference)
        w, h = edited.size
        if min(w, h) * 3 < max(w, h) * 2 or task["component"] in recoded or len(recoded) == RECODE_GROUPS:
            continue
        recoded.add(task["component"])
        size = (w * 768 // max(w, h), h * 768 // max(w, h))
        a, b = (cv2.resize(np.asarray(views[role]), size, interpolation=cv2.INTER_AREA) for role in RECODE_ROLES)
        affected = np.any(a != b, axis=-1)
        for role, view in zip(RECODE_ROLES, (a, b), strict=True):
            body, rgb = encoded_view(Image.fromarray(view), "jpeg75")
            yield ("recoded_jpeg75", task["component"], source, role, body, np.asarray(rgb),
                   affected if role == "ai" else np.zeros_like(affected), None)
    require(len(recoded) == RECODE_GROUPS, "Recoded source groups are incomplete")


def background_deliveries(args):
    import numpy as np
    from PIL import Image

    from scripts.image_reference_delivery_envelope_lama import source_pixels
    from scripts.image_residual_encoder_transfer import authored, encode

    roster_path = args.background / "qualified-private.json"
    official = read(PUBLIC / "image-bg-real-pixel-qualification-complete-v225.json")
    require(sha(roster_path.read_bytes()) == official["qualified_private_sha256"], "Background roster changed")
    roster = read(roster_path)
    prior = read(PUBLIC / "image-small-region-repeat-2026-11-14.json")
    excluded = {row["source_sha256"] for row in prior["freeze"]["cases"]}
    excluded.update(read(PUBLIC / "image-reference-progressive-2026-11-14.json")["sources"])
    authors = set()
    selected = 0
    for row in sorted(roster["qualified"], key=lambda row: row["source"]["original"]["sha256"]):
        source = row["source"]
        digest = source["original"]["sha256"]
        identities = {source["flickr_owner"].casefold(), source["photo_attribution"]["Author"].casefold()}
        if digest in excluded or min(source["size"]) < 512 or authors & identities:
            continue
        require(source["photo_attribution"]["License"] == "https://creativecommons.org/licenses/by/2.0/"
                and max(source["size"]) == 768, "Unexpected source license or geometry")
        original, edited, _ = source_pixels(Path(roster["cache"]), row)
        changed = np.any(original != edited, axis=-1)
        ordinary, _ = authored({"original": original, "edited": edited}, changed, digest)
        views = {"ai": edited, "original": original, "tone": ordinary["local_tone"], "blur": ordinary["local_blur"]}
        for role in BACKGROUND_ROLES:
            body = encode(views[role], 75)
            with Image.open(io.BytesIO(body), formats=["JPEG"]) as delivered:
                yield ("lama_jpeg75", digest, digest, role, body, np.asarray(delivered.convert("RGB")),
                       changed if role == "ai" else np.zeros_like(changed), None)
        authors.update(identities)
        excluded.add(digest)
        selected += 1
        if selected == BACKGROUND_GROUPS:
            return
    raise ValueError("Background source groups are incomplete")


def deliveries(args):
    yield from native_deliveries(args)
    yield from background_deliveries(args)


def identity(item):
    cohort, group, source, role, body, rgb, truth, archived = item
    return {"cohort": cohort, "group": group, "source_sha256": source, "role": role, "body_sha256": sha(body),
            "rgb_sha256": sha(rgb.tobytes()), "truth_sha256": sha(truth.tobytes()),
            "size": list(truth.shape[::-1]), "truth_pixels": int(truth.sum()),
            "archived_map_sha256": archived["raw_map_sha256"] if archived else None}


def bindings():
    detector = ROOT / "backend/app/detectors/image"
    paths = [Path(__file__).resolve(), detector / "regional.py", detector / "regional_support.py",
             detector / "localiser.py", detector / "jpeg_bitstream.py",
             ROOT / "backend/app/evidence/image_eligibility.py", *sorted((detector / "regional_graph").glob("*.py")),
             *[ROOT / "backend/scripts" / name for name in (
                 "image_native_confirmation_v46.py", "image_regional_inference_v28.py",
                 "image_regional_training_v26.py", "image_reference_delivery_envelope_lama.py",
                 "image_residual_encoder_transfer.py", "replay_regional_calibration_v43.py",
             )]]
    return {str(path.relative_to(ROOT)): sha(path.read_bytes()) for path in paths}


def freeze(args):
    import cv2

    cv2.setNumThreads(1)
    cases = [identity(item) for item in deliveries(args)]
    return {"schema": SCHEMA, "method": METHOD, "checkpoint_sha256": CHECKPOINT, "bindings": bindings(), "cases": cases}


def extend_edges(model, tensor, geometry):
    """Clamp only the padding's indices to the closest fitted source pixel."""
    import torch

    from app.detectors.image import regional

    fw, fh = geometry["fit_size"]
    row = torch.arange(1024).clamp_(max=fh - 1)
    col = torch.arange(1024).clamp_(max=fw - 1)
    extended = tensor[:, :, row[:, None], col[None, :]].contiguous()
    require(torch.equal(extended[:, :, :fh, :fw], tensor[:, :, :fh, :fw]), "Padded fit changed the original input")
    expected = torch.stack([
        extended[0, :, y * 496:y * 496 + 528, x * 496:x * 496 + 528]
        for y in range((2 * fh + 1023) // 1024) for x in range((2 * fw + 1023) // 1024)
    ])
    used = []

    def bind(_module, args):
        require(torch.equal(args[0], expected), "Actual edge-extended patches differ")
        used.append(regional.tensor_sha(args[0]))

    hook = model.model.vit.patch_embed.register_forward_pre_hook(bind)
    try:
        with torch.inference_mode():
            output = model.model(image=extended, origin_shape=torch.tensor([[fh, fw]]), clip_len=torch.tensor([1]))
    finally:
        hook.remove()
    require(len(used) == 1, "Padded patch input binding incomplete")
    return regional.source_map(output, geometry), used[0]


def measure(model, item, args):
    import numpy as np
    from PIL import Image

    from app.detectors.image import regional, regional_support
    from scripts.replay_regional_calibration_v43 import load_map

    body, rgb, truth, archived = item[4:]
    image = Image.fromarray(rgb)
    tensor, geometry = regional.prepare(image)
    if archived:
        require(geometry == archived["geometry"], "Archived current model input changed")
        native = load_map(archived, args.raw_maps, truth.shape)
    else:
        native = model.predict(image).raw_mask
    edge, patch = extend_edges(model, tensor, geometry)

    def selected(raw):
        gate = np.count_nonzero(raw > regional.PIXEL_THRESHOLD) >= np.ceil(regional.MINIMUM_AREA_FRACTION * raw.size)
        return regional_support.readout(raw, gate, regional.PIXEL_THRESHOLD, body, rgb)

    def pixels(mask):
        shape = image.size
        restored = np.asarray(Image.fromarray(mask.astype(np.uint8)).resize(shape, Image.Resampling.NEAREST)) > 0
        return [int(restored.sum()), int((restored & truth).sum())]

    current, candidate = map(selected, (native, edge))
    support = []
    for mask in current.component_masks:
        count = int(mask.sum())
        shared = int((mask & candidate.selected_grid_mask).sum())
        support.append([count, shared, *pixels(mask)])
    require(np.array_equal(current.reliable, candidate.reliable), "Candidate changed the reliable grid")
    return {"body_sha256": sha(body), "edge_patch_sha256": patch, "fit_size": geometry["fit_size"],
            "current": pixels(current.selected_grid_mask), "edge": pixels(candidate.selected_grid_mask),
            "support": support, "max_delta": round(float(np.max(np.abs(native - edge))), 7)}


def summarize(lock, rows):
    require(lock["schema"] == SCHEMA and lock["method"] == METHOD and lock["checkpoint_sha256"] == CHECKPOINT,
            "Unexpected study contract")
    stats = defaultdict(lambda: defaultdict(Counter))
    roles = defaultdict(set)
    groups = defaultdict(set)
    source_groups = defaultdict(set)
    for case, row in zip(lock["cases"], rows, strict=True):
        require(row["body_sha256"] == case["body_sha256"], "Measured bytes differ from the freeze")
        width, height = case["size"]
        target, cohort, role = case["truth_pixels"], case["cohort"], case["role"]
        require(0 <= target <= width * height and bool(target) == (role == "ai"), "Invalid exact-truth count")
        require(len(bytes.fromhex(row["edge_patch_sha256"])) == 32 and 0 <= row["max_delta"] <= 1,
                "Invalid patch witness")
        require(row["fit_size"] == case["size"], "Study requires native decoded pixels")
        if width == height == 1024:
            require(row["max_delta"] == 0 and row["current"] == row["edge"], "Square no-padding control changed")
        original = row["current"]
        parts = row["support"]
        require(len(parts) <= 2 and all(0 <= shared <= count and count > 0 and 0 <= hit <= area
                                        for count, shared, area, hit in parts)
                and [sum(part[k] for part in parts) for k in (2, 3)] == original,
                "Current component memberships disagree")
        retained = [part for part in parts if 2 * part[1] >= part[0]]
        supported = [sum(part[k] for part in retained) for k in (2, 3)]
        key = cohort, case["source_sha256"]
        require(role not in roles[key], "Duplicate delivery role")
        roles[key].add(role)
        groups[cohort].add(case["group"])
        source_groups[key].add(case["group"])
        for arm, value in zip(ARMS, (original, row["edge"], supported), strict=True):
            area, hit = value
            require(0 <= area <= width * height and max(0, area + target - width * height) <= hit <= min(area, target),
                    "Selected and shared source pixels are out of bounds")
            is_useful = useful(value, target)
            was_useful = useful(original, target)
            stats[cohort][arm].update({
                f"{role}_total": 1, f"{role}_flags": bool(area),
                "useful": is_useful, "misplaced": bool(target and area and not is_useful),
                "useful_added": is_useful and not was_useful, "useful_lost": was_useful and not is_useful,
                "ordinary_added": not target and bool(area) and not original[0],
                "ordinary_removed": not target and not area and bool(original[0]),
            })
    expected = {"native_png": (59, 48, NATIVE_ROLES), "recoded_jpeg75": (RECODE_GROUPS, RECODE_GROUPS, RECODE_ROLES),
                "lama_jpeg75": (BACKGROUND_GROUPS, BACKGROUND_GROUPS, BACKGROUND_ROLES)}
    require(set(groups) == set(expected) and all(len(value) == 1 for value in source_groups.values()),
            "Missing cohort or inconsistent source group")
    for cohort, (photos, count, required_roles) in expected.items():
        selected = [value for key, value in roles.items() if key[0] == cohort]
        require(len(groups[cohort]) == count and len(selected) == photos
                and all(value == set(required_roles) for value in selected), "Source denominators or roles changed")
    require(groups["recoded_jpeg75"] <= groups["native_png"] and not groups["lama_jpeg75"] & groups["native_png"],
            "Correlated and distinct source groups are incorrectly labeled")
    return {cohort: dict(arms) for cohort, arms in stats.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "measure", "review"))
    parser.add_argument("path", type=Path)
    parser.add_argument("--native", type=Path, default=DATA / "native-regional-serving-v47")
    parser.add_argument("--pairs", type=Path, default=DATA / "image-positive-pairs-v21-cache")
    parser.add_argument("--background", type=Path, default=DATA / "image-bg-real-acquisition-v225")
    parser.add_argument("--raw-maps", type=Path)
    parser.add_argument("--checkpoint", type=Path,
                        default=ROOT / "models/weights/native-regional-v1/regional-model.safetensors")
    args = parser.parse_args()
    if args.phase == "review":
        receipt = read(args.path)
        result = summarize(receipt["freeze"], receipt["rows"])
        require(result == receipt["summary"], "Saved summary differs")
    elif args.phase == "freeze":
        args.path.mkdir(mode=0o700)
        save(args.path / "freeze.json", freeze(args))
        return
    else:
        import cv2
        import torch

        from app.detectors.image import regional

        lock = read(args.path / "freeze.json")
        require(lock["bindings"] == bindings() and args.raw_maps is not None, "Bound code and archived maps are required")
        cv2.setNumThreads(1)
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        # Load transitive source readers before any inference so import failures
        # cannot first surface after the archived cohort has been forwarded.
        require(freeze(args) == lock, "Native source preflight differs from freeze")
        model = regional.load_model(args.checkpoint)
        rows = []
        with (args.path / "measurements.jsonl").open("x") as progress:
            for index, (item, case) in enumerate(zip(deliveries(args), lock["cases"], strict=True), 1):
                require(identity(item) == case, "Source role or input changed after freeze")
                row = measure(model, item, args)
                print(json.dumps(row, sort_keys=True), file=progress, flush=True)
                rows.append(row)
                if index % 12 == 0:
                    print(f"Measured {index}/{len(lock['cases'])} uploads", flush=True)
        result = summarize(lock, rows)
        save(args.path / "receipt.json", {"freeze": lock, "rows": rows, "summary": result})
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
