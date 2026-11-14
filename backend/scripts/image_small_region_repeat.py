"""Audit and test small regional proposals from one JPEG upload."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from collections import Counter, defaultdict
from pathlib import Path

SCHEMA = "chipmunk.image-small-region-repeat/1"
ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT = "587d553353ddab1c81f45e86fbc9ad2326dee9086c6b566d9abcacc221380590"
PLAN = {
    "sources": "Use 18 fixed retained TGIF photos across three editors. Select seven primary and five backup BG-REAL "
               "LaMa photos in original byte hash order, excluding photos in the previous progressive comparison.",
    "carrier": "Upright RGB JPEG quality 75 with 4:2:0 sampling, 768-pixel long side and at least 512-pixel short side.",
    "serving": "Use the existing area rule and exact two-component selection on the original file.",
    "ungated": "Diagnostic. Run the existing 0.90 component readout even if the ordinary 1% source-area rule closes.",
    "repeat": "Preserve all open-gate outputs. For a closed gate, keep the strongest original component only when "
              "at least two-thirds of it and two-thirds of one of two aligned horizontal-repeat components overlap. "
              "Apply the original upload's reliable grid in both views. Keep the exact original component.",
    "useful": "At least 0.30 mask intersection over union and 0.50 changed-pixel recall.",
    "advance": "At least three added useful locations from different photos across at least two editor families, "
               "at most two additional ordinary source groups, and no lost useful location.",
}
TGIF_ROLES = ("ordinary", "local_blur", "local_tone", "sharp", "feather")
LAMA_ROLES = ("original", "local_blur", "local_tone", "local_copy", "global_resize", "second_jpeg", "sharp", "feathered")
POSITIVE = {"sharp", "feather", "feathered"}
ARMS = ("serving", "ungated", "repeat")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_bytes())


def save(path, data):
    with path.open("x") as stream:
        print(json.dumps(data, sort_keys=True, allow_nan=False, separators=(",", ":")), file=stream)


def deliveries(args):
    import numpy as np

    from scripts.image_reference_delivery_envelope_lama import source_pixels
    from scripts.image_regional_progressive_transfer import deliveries as prior_tgif
    from scripts.image_residual_encoder_transfer import authored, encode

    for group, row, body, truth in prior_tgif(args.sources):
        if row["role"] not in POSITIVE:
            truth = np.zeros_like(truth)
        yield group, row["source"], row["family"], row["role"], body, truth
    previous = read(ROOT / "docs/evaluation-panels/image-reference-progressive-2026-11-14.json")
    excluded = set(previous["sources"])
    group = 18
    ledgers = ("image-bg-real-pixel-qualification-complete-v225", "image-bg-real-backup-intake-v262-complete")
    seen = set()
    for directory, ledger, target in zip(args.bg, ledgers, (7, 5), strict=True):
        path = directory / "qualified-private.json"
        public = read(ROOT / "docs/evaluation-panels" / (ledger + ".json"))
        require(sha(path.read_bytes()) == public["qualified_private_sha256"], "BG-REAL source roster differs")
        roster = read(path)
        chosen = 0
        for row in sorted(roster["qualified"], key=lambda item: item["source"]["original"]["sha256"]):
            source = row["source"]
            pin = source["original"]["sha256"]
            if pin in excluded or max(source["size"]) != 768 or min(source["size"]) < 512:
                continue
            identity = {str(source["flickr_photo_id"]), source["flickr_owner"].casefold(),
                        source["photo_attribution"]["Author"].casefold(), pin}
            if seen & identity:
                continue
            require(source["photo_attribution"]["License"] == "https://creativecommons.org/licenses/by/2.0/",
                    "Unexpected original photo license")
            original, edited, _ = source_pixels(Path(roster["cache"]), row)
            scene = np.any(original != edited, axis=2)
            controls, known = authored({"original": original, "edited": edited}, scene, pin)
            group += 1
            for role, rgb in (controls | known).items():
                truth = np.any(original != rgb, axis=2) if role in POSITIVE else np.zeros_like(scene)
                yield group, pin, "lama", role, encode(rgb, 75), truth
            seen.update(identity)
            chosen += 1
            if chosen == target:
                break
        require(chosen == target, "Not enough other qualified photos in an archived roster")


def identity(group, pin, family, role, body, truth):
    return {"group": group, "source_sha256": pin, "family": family, "role": role,
            "jpeg_sha256": sha(body), "truth_sha256": sha(truth.tobytes()),
            "pixels": truth.size, "changed": int(truth.sum())}


def freeze(args):
    cases = [identity(*entry) for entry in deliveries(args)]
    require(len(cases) == 186, "Unexpected frozen source cohort")
    return {"schema": SCHEMA, "plan": PLAN, "checkpoint_sha256": CHECKPOINT, "cases": cases}


def useful(measurement, target):
    size, overlap = measurement
    return bool(target and 2 * overlap >= target and 10 * overlap >= 3 * (size + target - overlap))


def replay(receipt):
    lock, rows = receipt["freeze"], receipt["rows"]
    require(lock["schema"] == SCHEMA and lock["plan"] == PLAN and lock["checkpoint_sha256"] == CHECKPOINT
            and len(lock["cases"]) == len(rows) == 186, "Incorrect freeze or measurements")
    counts = {arm: defaultdict(Counter) for arm in ARMS}
    groups = defaultdict(set)
    gained, added_ordinary = defaultdict(set), set()
    for case, row in zip(lock["cases"], rows, strict=True):
        group, family, role, target = (case[key] for key in ("group", "family", "role", "changed"))
        groups[(group, family, case["source_sha256"])].add(role)
        require(row["jpeg_sha256"] == case["jpeg_sha256"] and bool(target) == (role in POSITIVE),
                "A measured file or edit truth does not match its freeze")
        gate, forward, overlap = (row[key] for key in ("area_gate_open", "repeat_forward", "component_support"))
        native, diagnostic, candidate = (row[arm] for arm in ARMS)
        require(type(gate) is bool and type(forward) is bool, "Invalid gate disposition")
        require(forward == (not gate and bool(diagnostic[0])), "Repeat must follow an ungated proposal on a closed gate")
        require(native == (diagnostic if gate else [0, 0]) and (not gate or native == candidate),
                "The old open-gate result was modified")
        require(all(len(item) == 4 and item[0] > 0 and 0 <= item[1] <= min(item[0], item[2])
                    and item[2] > 0 and item[3] in (0, 1) for item in overlap), "Invalid support counts")
        require(len(bytes.fromhex(row["native_input_sha256"])) == 32
                and (row["reflected_input_sha256"] is not None) == forward, "Missing input commitment")
        if forward:
            require(len(bytes.fromhex(row["reflected_input_sha256"])) == 32, "Invalid repeat input commitment")
        supported = any(3 * shared >= 2 * max(size, other) for size, shared, other, _ in overlap)
        if not gate:
            require(bool(candidate[0]) == supported, "Repeat selection does not follow the fixed overlap rule")
        for arm in ARMS:
            size, hit = measurement = row[arm]
            require(0 <= size <= case["pixels"] and 0 <= hit <= min(size, target), "Invalid source pixel count")
            count = counts[arm][family]
            count.update(edited=bool(target), ordinary=not target, useful=useful(measurement, target),
                         wrong=bool(target and size and not useful(measurement, target)), ordinary_flags=bool(not target and size))
        if useful(candidate, target) and not useful(native, target):
            gained[family].add(group)
        if not target and candidate[0] and not native[0]:
            added_ordinary.add(group)
    require(len(groups) == len({group for group, _, _ in groups}) == len({pin for _, _, pin in groups}) == 30
            and Counter(family for _, family, _ in groups) == {"firefly": 6, "sd2": 6, "flux-fill": 6, "lama": 12}
            and all(roles == set(LAMA_ROLES if family == "lama" else TGIF_ROLES)
                    for (_, family, _), roles in groups.items()), "Unexpected or repeated source groups")
    result = {"counts": {arm: dict(by) for arm, by in counts.items()}, "gained_groups": {k: sorted(v) for k, v in gained.items()},
              "additional_ordinary_groups": sorted(added_ordinary),
              "open_gates": sum(row["area_gate_open"] for row in rows),
              "repeat_forwards": sum(row["repeat_forward"] for row in rows),
              "passes": len(gained) >= 2 and sum(map(len, gained.values())) >= 3 and len(added_ordinary) <= 2}
    if "summary" in receipt:
        require(receipt["summary"] == result, "Recorded outcome does not replay")
    return result


def measure(args):
    import cv2
    import numpy as np
    import torch
    from PIL import Image

    from app.detectors.image import regional, regional_support

    lock = read(args.run / "freeze.json")
    require(freeze(args) == lock, "Frozen source bytes differ")
    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    model = regional.load_model(args.checkpoint)
    rows = []
    for entry, case in zip(deliveries(args), lock["cases"], strict=True):
        require(identity(*entry) == case, "Input differs from the frozen image")
        body, truth = entry[-2:]
        with Image.open(io.BytesIO(body), formats=["JPEG"]) as source:
            rgb = np.array(source.convert("RGB"))
        original = model.predict(Image.fromarray(rgb))
        gate = original.candidate_eligibility["eligible"]
        direct = regional_support.readout(original.raw_mask, True, regional.PIXEL_THRESHOLD, body, rgb)
        empty = np.zeros_like(direct.selected_grid_mask)
        confirmed, trace, repeat_hash = (direct.selected_grid_mask if gate else empty), [], None
        if not gate and direct.component_masks:
            flip = np.ascontiguousarray(rgb[:, ::-1])
            reflected = model.predict(Image.fromarray(flip))
            # Map back before the normal readout so both passes use the exact
            # same reliability grid and JPEG block alignment.
            mirrored = regional_support.readout(reflected.raw_mask[:, ::-1], True, regional.PIXEL_THRESHOLD, body, rgb)
            repeat_hash = reflected.geometry["input_float32_sha256"]
            for a in direct.component_masks:
                size = int(a.sum())
                matches = []
                for i, b in enumerate(mirrored.component_masks):
                    shared = int((a & b).sum())
                    other = int(b.sum())
                    trace.append([size, shared, other, i])
                    matches.append(3 * shared >= 2 * max(size, other))
                if any(matches):
                    confirmed = a
                    break

        def pixels(mask, truth=truth):
            h, w = truth.shape
            native = np.asarray(Image.fromarray(mask.astype(np.uint8)).resize((w, h), Image.Resampling.NEAREST)) > 0
            return [int(native.sum()), int((native & truth).sum())]

        rows.append({"jpeg_sha256": case["jpeg_sha256"], "area_gate_open": gate, "repeat_forward": repeat_hash is not None,
                     "native_input_sha256": original.geometry["input_float32_sha256"], "reflected_input_sha256": repeat_hash,
                     "component_support": trace, "serving": pixels(direct.selected_grid_mask if gate else empty),
                     "ungated": pixels(direct.selected_grid_mask), "repeat": pixels(confirmed)})
        if len(rows) % 15 == 0:
            print(f"Measured {len(rows)} of {len(lock['cases'])} JPEG deliveries", flush=True)
    receipt = {"freeze": lock, "rows": rows}
    save(args.run / "receipt.json", receipt | {"summary": replay(receipt)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "measure", "review"))
    parser.add_argument("run", type=Path)
    parser.add_argument("--sources", type=Path, default=ROOT / "backend/data/localization-eval")
    parser.add_argument("--bg", type=Path, nargs=2)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "models/weights/native-regional-v1/regional-model.safetensors")
    args = parser.parse_args()
    if args.phase == "review":
        print(json.dumps(replay(read(args.run)), sort_keys=True))
    elif args.phase == "freeze":
        lock = freeze(args)
        args.run.mkdir(mode=0o700)
        save(args.run / "freeze.json", lock)
    else:
        measure(args)


if __name__ == "__main__":
    main()
