"""Test local reconstruction agreement for small edits without an earlier copy."""

from __future__ import annotations

import argparse
import json
import time
from importlib.metadata import version
from pathlib import Path

import cv2
import numpy as np
from PIL import __version__ as pillow_version

from app.detectors.image import jpeg_dq, sharp_boundary
from scripts import image_edge_spread_transfer as panel
from scripts import image_residual_color_transfer as reference

SCHEMA = "chipmunk.image-reconstruction-phase-transfer/1"
ARMS = ("fixed_phase", "best_phase", "phase_agreement")
BASELINE = "format_reference"
PHASES = ((0, 0), (0, 4), (4, 0), (4, 4))
CELL = panel.CELL
FLOOR = .003
MIN_VALID_SHARE = .5
PADDING = 32
ADVANCE = {"minimum_extra_feathered_per_panel": 2, "maximum_extra_ordinary_per_panel": 4,
           "maximum_extra_ordinary_groups_per_panel": 3}
MODEL_REVISION = "9565f9356505d7c3537d0e15c558520ed95142ee"
WEIGHTS = "diffusion_pytorch_model.safetensors"
WEIGHTS_SHA256 = "db169d69145ec4ff064e49d99c95fa05d3eb04ee453de35824a6d0f325513549"
LICENSE_SHA256 = "261497cf8e9ed613ed1fa5ddf48af32cdf50545b2ccd9edd7eacc91fb9255d6e"
WHEELS = {
    "diffusers-0.42.1-py3-none-any.whl": "9154faa704ce00e0b77f568e0d64786064368528aab5fc14eeea92f87d788014",
    "huggingface_hub-1.32.0-py3-none-any.whl": "b0c7c80561969d9cdacdd55fce67ba9584cca0b9d4ea80957a3a5c1445fac5c8",
    "importlib_metadata-8.7.0-py3-none-any.whl": "e5dd1551894c77868a30651cef00984d50e1002d06942a7101d34870c5f02afd",
    "zipp-3.23.0-py3-none-any.whl": "071652d6115ed432f5ce1d34c336c0adfd6a884660d1e9712a256d3d3bd4b14e",
}
sha, write = panel.sha, panel.write


def phase_padding(shape, phase):
    height, width = shape
    top, left = (PADDING + value for value in phase)
    bottom = PADDING + (-height - top - PADDING) % 8
    right = PADDING + (-width - left - PADDING) % 8
    return left, right, top, bottom


class Reconstruction:
    def __init__(self, weights, device):
        import torch
        from diffusers import AutoencoderTiny
        from torch.nn import functional

        if device == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("Frozen comparison requires an available Metal device")
        torch.set_num_threads(1)
        self.torch = torch
        self.pad = functional.pad
        self.device = device
        self.model = AutoencoderTiny.from_pretrained(weights, local_files_only=True, use_safetensors=True).eval().to(device)
        self.forwards = 0

    def errors(self, rgb):
        torch = self.torch
        height, width = rgb.shape[:2]
        pixels = np.ascontiguousarray(rgb.transpose(2, 0, 1), dtype=np.float32) / 127.5 - 1
        image = torch.from_numpy(pixels[None]).to(self.device)
        responses = []
        with torch.inference_mode():
            for phase in PHASES:
                pads = phase_padding((height, width), phase)
                left, _, top, _ = pads
                padded = self.pad(image, pads, mode="reflect")
                reconstructed = self.model.decode(self.model.encode(padded).latents).sample
                if reconstructed.shape != padded.shape:
                    raise ValueError("The autoencoder changed the padded image grid")
                aligned = reconstructed[:, :, top:top + height, left:left + width].clamp(-1, 1)
                error = ((aligned - image).abs().mean(dim=1)[0] * .5).cpu().numpy()
                if not np.isfinite(error).all():
                    raise ValueError("The autoencoder returned a non-finite reconstruction")
                responses.append(error)
                self.forwards += 1
        return np.stack(responses, axis=-1)


def tiles(values):
    rows, cols = (side // CELL for side in values.shape[:2])
    return values[:rows * CELL, :cols * CELL].reshape(rows, CELL, cols, CELL, *values.shape[2:]).mean(axis=(1, 3))


def local_proposals(rgb, error):
    pixels = rgb.astype(np.float32) / 255
    texture = tiles(np.abs(pixels - cv2.GaussianBlur(pixels, (0, 0), 1)).mean(axis=-1))
    clip = tiles(np.any((rgb < 3) | (rgb > 252), axis=-1))
    valid = (texture >= .002) & (clip <= .25)
    loss = np.log(tiles(error) + FLOOR) - .5 * np.log(texture[..., None] + FLOOR)
    responses = np.concatenate((loss, loss.min(axis=-1, keepdims=True)), axis=-1)
    maps = np.concatenate((valid[..., None], np.where(valid[..., None], responses, 0)), axis=-1)
    integral = np.pad(maps.astype(np.float64).cumsum(0).cumsum(1), ((1, 0), (1, 0), (0, 0)))
    boxes = panel.candidate_boxes(*valid.shape)
    if not len(boxes):
        return dict.fromkeys(ARMS)
    inner = boxes + np.array([1, 1, -1, -1])
    gap = boxes + np.array([-1, -1, 1, 1])
    outer = boxes + np.array([-3, -3, 3, 3])
    core = panel.sums(integral, inner)
    ring = panel.sums(integral, outer) - panel.sums(integral, gap)
    core_area = np.prod(inner[:, 2:] - inner[:, :2], axis=1)
    ring_area = np.prod(outer[:, 2:] - outer[:, :2], axis=1) - np.prod(gap[:, 2:] - gap[:, :2], axis=1)
    eligible = (core[:, 0] >= MIN_VALID_SHARE * core_area) & (ring[:, 0] >= MIN_VALID_SHARE * ring_area)
    differences = ring[:, 1:] / np.maximum(ring[:, :1], 1) - core[:, 1:] / np.maximum(core[:, :1], 1)
    three_of_four = np.sort(differences[:, :4], axis=1)[:, 1]
    results = {}
    for arm, score in zip(ARMS, (differences[:, 0], differences[:, 4], three_of_four), strict=True):
        rank = np.where(eligible, score, -np.inf)
        best = int(rank.argmax())
        results[arm] = {"box": (boxes[best] * CELL).tolist(), "score": float(rank[best]),
                        "core_tiles": int(core[best, 0]), "ring_tiles": int(ring[best, 0])} if eligible[best] else None
    return results


def freeze(args):
    dirs = (args.development, args.earlier, args.later)
    plans = [panel.read(folder / "plan.json")["cases"] for folder in dirs]
    for field in ("flickr_photo_id", "flickr_owner"):
        rosters = [{str(row[field]) for row in plan} for plan in plans]
        if [len(roster) for roster in rosters] != [8, 12, 12] or len(set.union(*rosters)) != 32:
            raise ValueError("Development and confirmation must have disjoint source photos and photographers")
    if sha(args.weights / WEIGHTS) != WEIGHTS_SHA256 or sha(args.weights / "LICENSE") != LICENSE_SHA256:
        raise ValueError("The checkpoint or publisher notice differs from its pinned bytes")
    if {name: sha(args.runtime / name) for name in WHEELS} != WHEELS:
        raise ValueError("The isolated wheel files differ from their pinned bytes")
    versions = {name: version(name) for name in ("torch", "diffusers", "huggingface-hub", "safetensors", "importlib-metadata", "zipp")}
    if versions["diffusers"] != "0.42.1" or versions["huggingface-hub"] != "1.32.0":
        raise ValueError("Use the pinned isolated Python path for inference")
    code = (Path(__file__), Path(panel.__file__), Path(reference.__file__), Path(jpeg_dq.__file__), Path(sharp_boundary.__file__),
            Path(panel.paired.__file__), Path(panel.scoring.__file__))
    return {"schema": SCHEMA, "code_sha256": {path.name: sha(path) for path in code},
            "input_intakes": [{name: sha(folder / name) for name in ("plan.json", "attempts.jsonl", "result.json")}
                              for folder in dirs],
            "model": {"revision": MODEL_REVISION, "weights_sha256": WEIGHTS_SHA256, "license": "MIT",
                      "notice_sha256": LICENSE_SHA256, "config_sha256": sha(args.weights / "config.json")},
            "runtime": {**versions, "wheel_sha256": WHEELS, "numpy": np.__version__, "opencv": cv2.__version__,
                        "pillow": pillow_version, "device": args.device, "precision": "float32", "torch_threads": 1},
            "method": {"phases_yx": [list(value) for value in PHASES], "padding": PADDING, "cell": CELL,
                       "floor": FLOOR, "minimum_valid_tile_share": MIN_VALID_SHARE, "arms": list(ARMS),
                       "primary": "phase_agreement", "baseline": "local PNG sharp-boundary or native JPEG quantization",
                       "formats": list(panel.FORMATS), "held_roles": [*panel.POSITIVE, *panel.CONTROLS],
                       "point_rule": "Strictly exceed the second-highest of eight source-level control maxima or zero, whichever is greater.",
                       "geometry": "Useful requires native changed-pixel IoU at least .30 and recall at least .50.",
                       "advance": ADVANCE, "confidence_contribution": 0}}


def fit(folder, model):
    _, plan, receipts = panel.scoring.checked_inputs(folder)
    maxima = {fmt: {arm: [] for arm in ARMS} for fmt in panel.FORMATS}
    for slot, _, images, _ in panel.scoring.panels(folder, plan, receipts):
        controls = panel.development_controls(images["original"], images["processed"])
        for fmt in panel.FORMATS:
            values = {arm: [] for arm in ARMS}
            for image in controls.values():
                rgb = np.asarray(panel.paired.encoded(image, fmt)[1])
                result = local_proposals(rgb, model.errors(rgb))
                for arm in ARMS:
                    if result[arm] is not None:
                        values[arm].append(result[arm]["score"])
            for arm, scores in values.items():
                maxima[fmt][arm].append(max(scores, default=0))
        print(f"development source {slot + 1}/8 measured", flush=True)
    if any(len(values) != 8 for arms in maxima.values() for values in arms.values()):
        raise ValueError("Incomplete development control panel")
    return {fmt: {arm: {"control_maxima": values, "point": max(0, sorted(values)[-2])} for arm, values in arms.items()}
            for fmt, arms in maxima.items()}


def observe(folder, model, points):
    for case, native, known in panel.paired.inputs(folder):
        for role, pixels in panel.native_variants(native, case["box"]).items():
            truth = np.any(pixels != native["original"], axis=2) if role in panel.POSITIVE else np.zeros_like(known)
            for fmt in panel.FORMATS:
                body, image = panel.paired.encoded(pixels, fmt)
                rgb = np.asarray(image)
                responses = local_proposals(rgb, model.errors(rgb))
                masks = {arm: panel.selection(proposal, points[fmt][arm]["point"], truth.shape) for arm, proposal in responses.items()}
                measured = reference.format_reference(fmt, body, rgb)
                masks[BASELINE] = panel.paired.boxed(measured["box"], truth.shape, True) if measured and measured["selected"] else np.zeros_like(truth)
                yield {"slot": case["slot"], "role": role, "format": fmt, "input_sha256": panel.paired.shared.digest(body),
                       "truth_sha256": panel.paired.shared.digest(truth.tobytes()), "proposals": {**responses, BASELINE: measured},
                       "measurements": {arm: panel.scoring.measure(mask, truth) for arm, mask in masks.items()}}


def summarize(rows):
    if len(rows) != 192 or len({(r["slot"], r["role"], r["format"]) for r in rows}) != 192:
        raise ValueError("Incomplete confirmation panel")
    summary = {}
    for fmt in panel.FORMATS:
        cohort = [row for row in rows if row["format"] == fmt]
        summary[fmt] = {}
        for arm in (*ARMS, BASELINE):
            positives = [r for r in cohort if r["role"] in panel.POSITIVE]
            ordinary = [r for r in cohort if r["role"] in panel.CONTROLS]
            useful = [r for r in positives if panel.useful(r["measurements"][arm])]
            flags = [r for r in ordinary if r["measurements"][arm]["reported"] > 0]
            new_flags = [r for r in flags if not r["measurements"][BASELINE]["reported"]]
            summary[fmt][arm] = {"sharp": sum(r["role"] == "edited" for r in useful),
                                 "feathered": sum(r["role"] == "feathered_edit" for r in useful), "ordinary_flags": len(flags),
                                 "ordinary_groups": len({r["slot"] for r in flags}), "extra_ordinary_flags": len(new_flags),
                                 "extra_ordinary_groups": len({r["slot"] for r in new_flags}),
                                 "extra_feathered": sum(r["role"] == "feathered_edit" and not panel.useful(r["measurements"][BASELINE])
                                                         for r in useful),
                                 "unassessed_positive": sum(r["proposals"][arm] is None for r in positives),
                                 "unassessed_ordinary": sum(r["proposals"][arm] is None for r in ordinary)}
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "run"))
    for key in ("weights", "runtime", "development", "earlier", "later", "output"):
        parser.add_argument(key, type=Path)
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    args = parser.parse_args()
    frozen = freeze(args)
    freeze_path = args.output / "freeze.json"
    if args.stage == "freeze":
        args.output.mkdir(mode=0o700)
        write(freeze_path, frozen)
        print(sha(freeze_path), flush=True)
        return
    if frozen != json.loads(freeze_path.read_bytes()):
        raise ValueError("Runtime, code or inputs differ from the frozen experiment")
    cv2.setNumThreads(1)
    started = time.monotonic()
    model = Reconstruction(args.weights, args.device)
    points = fit(args.development, model)
    points_path = args.output / "points.json"
    write(points_path, {"freeze_sha256": sha(freeze_path), "points": points, "development_forwards": model.forwards})
    result, journals = {}, {}
    for name, folder in (("earlier", args.earlier), ("later", args.later)):
        rows = []
        journal = args.output / f"{name}-rows.jsonl"
        with journal.open("x") as stream:
            for row in observe(folder, model, points):
                stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
                rows.append(row)
                if len(rows) % 16 == 0:
                    stream.flush()
                    print(f"{name}: {len(rows)}/192 views measured", flush=True)
        journals[name], result[name] = sha(journal), summarize(rows)
    advances = {fmt: all((arm := cohort[fmt]["phase_agreement"])["extra_feathered"] >= ADVANCE["minimum_extra_feathered_per_panel"]
                         and arm["extra_ordinary_flags"] <= ADVANCE["maximum_extra_ordinary_per_panel"]
                         and arm["extra_ordinary_groups"] <= ADVANCE["maximum_extra_ordinary_groups_per_panel"] for cohort in result.values())
                for fmt in panel.FORMATS}
    write(args.output / "aggregate.json", {"schema": SCHEMA, "freeze_sha256": sha(freeze_path), "points_sha256": sha(points_path),
                                            "journals_sha256": journals, "confirmed_views": 384,
                                            "model_round_trips": model.forwards, "elapsed_s": round(time.monotonic() - started, 3),
                                            "results": result, "advances": advances})
    print(sha(args.output / "aggregate.json"), flush=True)


if __name__ == "__main__":
    main()
