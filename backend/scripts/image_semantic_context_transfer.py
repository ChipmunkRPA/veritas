"""Test whether matching distant scene content improves regional forensic contrast."""

from __future__ import annotations

import argparse
import json
import resource
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance

from scripts import image_background_patch_score as paired
from scripts import image_bg_real_insertion_score as scoring
from scripts import image_boundary_blend_transfer as blend
from scripts import image_relative_patch_transfer as patch

SCHEMA = "chipmunk.image-semantic-context-transfer/1"
SEMANTIC_SHA256 = "ae1e99fcefd534ed978cdeb8326f08030c96e28b7a81ffcbc98a857c84d14be1"
CONFIG_SHA256 = "1809f83e3bdb1609a501a610ad4a742f4fd8ae44d72ca4aa0df52d1f2ac8628d"
NEIGHBORS = 12
EXCLUDE_RADIUS = 5
ARMS = ("global_patch_response", "semantic_matched_residual", "joint_regional_response")
ROLES = ("edited", "feathered_edit")
GRID = patch.GRID
sha, write = patch.file_digest, patch.write


def semantic_residual(forensic: np.ndarray, features: np.ndarray) -> np.ndarray:
    """Compare each patch with semantically similar patches beyond its immediate area."""
    coords = np.indices((GRID, GRID)).reshape(2, -1).T
    local = np.max(np.abs(coords[:, None, :] - coords[None, :, :]), axis=2) <= EXCLUDE_RADIUS
    lengths = np.linalg.norm(features, axis=1, keepdims=True)
    normalized = features / np.maximum(lengths, 1e-8)
    affinity = normalized @ normalized.T
    affinity[local] = -np.inf
    neighbors = np.argsort(-affinity, axis=1, kind="stable")[:, :NEIGHBORS]
    return (forensic - np.median(forensic[neighbors], axis=1)).reshape(GRID, GRID)


def standardized(values: np.ndarray) -> np.ndarray:
    center = np.median(values)
    return (values - center) / max(float(np.median(np.abs(values - center))) * 1.4826, .02)


def reader(forensics: Path, semantic: Path):
    import torch
    from transformers import Dinov2Model

    from app.detectors.image import commfor

    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    forgery = commfor.load_model(forensics, "cpu")
    embedding = Dinov2Model.from_pretrained(semantic, local_files_only=True, use_safetensors=True).eval()
    mean = np.array(commfor.MEAN, dtype=np.float32)
    std = np.array(commfor.STD, dtype=np.float32)

    def tensor(image: Image.Image, size: int):
        rgb = np.asarray(image.resize((size, size), Image.Resampling.BICUBIC), dtype=np.float32) / 255
        return torch.from_numpy((rgb - mean) / std).permute(2, 0, 1).unsqueeze(0)

    @torch.inference_mode()
    def read(image: Image.Image):
        tokens = forgery.forward_features(tensor(image, 16 * GRID))[:, forgery.num_prefix_tokens:]
        forensic = forgery.head(tokens).reshape(-1).numpy()
        hidden = embedding(pixel_values=tensor(image, 14 * GRID)).last_hidden_state[0, 1:].numpy()
        assert forensic.shape == (GRID ** 2,) and hidden.shape == (GRID ** 2, 384)
        assert np.isfinite(forensic).all() and np.isfinite(hidden).all()
        global_map = forensic.reshape(GRID, GRID)
        semantic_map = semantic_residual(forensic, hidden)
        maps = (global_map, semantic_map, np.minimum(standardized(global_map), standardized(semantic_map)))
        return {arm: {"score": score, "region": list(region), "map_sha256": patch.digest(values.tobytes())}
                for arm, values in zip(ARMS, maps, strict=True) for score, region in (patch.strongest_region(values),)}

    return read


def operating_points(folder, read):
    _, plan, journal = scoring.checked_inputs(folder)
    maxima = {fmt: {arm: [] for arm in ARMS} for fmt in paired.FORMATS}
    for _, _, native, _ in scoring.panels(folder, plan, journal):
        controls = blend.control_views(native["original"], native["processed"])
        for fmt in paired.FORMATS:
            scores = [read(paired.encoded(pixels, fmt)[1]) for pixels in controls]
            for arm in ARMS:
                maxima[fmt][arm].append(max(result[arm]["score"] for result in scores))
    assert all(len(values) == 8 for arms in maxima.values() for values in arms.values())
    return {fmt: {arm: {"point": sorted(values)[-2], "group_maxima": values} for arm, values in arms.items()}
            for fmt, arms in maxima.items()}


def compare(intake, receipt, read, points):
    bound = {(row["slot"], row["generator"], row["format"], row["role"]): row for row in receipt}
    assert len(bound) == 192
    for source, native, known in paired.inputs(intake):
        photo, box = native["original"], source["box"]
        tone = np.asarray(ImageEnhance.Contrast(Image.fromarray(photo)).enhance(1.2))
        views = native | {"feathered_edit": blend.local(photo, native["edited"], box, feather=True),
                          "feathered_processing": blend.local(photo, native["local_processing"], box, feather=True),
                          "local_contrast": blend.local(photo, tone, box)}
        for role, pixels in views.items():
            truth = np.any(pixels != photo, axis=2) if role in ROLES else np.zeros_like(known)
            assert not (truth & ~known).any()
            for fmt in paired.FORMATS:
                body, image = paired.encoded(pixels, fmt)
                row = bound.pop((source["slot"], source["generator"], fmt, role))
                assert patch.digest(body) == row["input_sha256"]
                assert int(truth.sum()) == next(iter(row["measurements"].values()))["target"]
                if "truth_sha256" in row:
                    assert patch.digest(truth.tobytes()) == row["truth_sha256"]
                proposals = read(image)
                grid = patch.source_grid(truth.shape)
                measurements = {}
                for arm, proposal in proposals.items():
                    y, x, side = proposal["region"]
                    yy, xx = grid // GRID, grid % GRID
                    selected = (y <= yy) & (yy < y + side) & (x <= xx) & (xx < x + side)
                    measurements[arm] = scoring.measure(selected & (proposal["score"] > points[fmt][arm]["point"]), truth)
                yield {**{key: row[key] for key in ("slot", "generator", "role", "format", "input_sha256")},
                       "proposals": proposals, "measurements": measurements}
    assert not bound


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "run"))
    for name in ("forensics", "semantic", "development", "earlier_intake", "earlier_panel", "later_intake", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    weights = {"forensics": sha(args.forensics), "semantic": sha(args.semantic / "model.safetensors"),
               "semantic_config": sha(args.semantic / "config.json")}
    assert weights == {"forensics": patch.WEIGHT_SHA, "semantic": SEMANTIC_SHA256, "semantic_config": CONFIG_SHA256}
    sources = [{case["flickr_photo_id"] for case in json.loads((folder / "plan.json").read_bytes())["cases"]}
               for folder in (args.development, args.earlier_intake, args.later_intake)]
    assert [len(items) for items in sources] == [8, 12, 12] and len(set.union(*sources)) == 32
    receipts = {"earlier": args.earlier_panel / "rows.json", "later": args.later_intake / "outcomes.json"}
    assert sha(receipts["earlier"]) == json.loads((args.earlier_panel / "result.json").read_bytes())["journal_sha256"]
    assert sha(receipts["later"]) == json.loads((args.later_intake / "score-result.json").read_bytes())["outcomes_sha256"]
    frozen = {"schema": SCHEMA, "implementation_sha256": sha(Path(__file__)), "grid": GRID, "neighbors": NEIGHBORS,
              "exclude_radius": EXCLUDE_RADIUS, "arms": list(ARMS), "weights_sha256": weights,
              "intake_sha256": [sha(folder / "result.json") for folder in (args.development, args.earlier_intake, args.later_intake)],
              "receipt_sha256": {name: sha(path) for name, path in receipts.items()},
              "point": "strictly above the second highest development control-group maximum per arm and format",
              "versions": {name: version(name) for name in ("torch", "transformers", "numpy", "pillow", "timm", "safetensors")}}
    if args.stage == "freeze":
        args.output.mkdir(mode=0o700)
        write(args.output / "freeze.json", frozen)
        return
    assert frozen == json.loads((args.output / "freeze.json").read_bytes())
    started = time.monotonic()
    read = reader(args.forensics, args.semantic)
    points = operating_points(args.development, read)
    write(args.output / "points.json", points)
    print("Sealed comparison points after 64 control views", flush=True)
    summaries = {}
    for name, folder in (("earlier", args.earlier_intake), ("later", args.later_intake)):
        rows = list(compare(folder, json.loads(receipts[name].read_bytes()), read, points))
        assert len(rows) == 192
        journal = args.output / f"{name}-rows.json"
        write(journal, rows)
        feathered = [{**row, "delivery": row["format"], "role": "edited"} for row in rows if row["role"] == "feathered_edit"]
        summaries[name] = {"views": len(rows), "journal_sha256": sha(journal), "feathered": scoring.summarize(feathered),
                           "sharp_and_controls": paired.summarise([r for r in rows if r["role"] != "feathered_edit"])}
        print(f"Scored {name} paired views", flush=True)
    write(args.output / "result.json", {"schema": SCHEMA, "freeze_sha256": sha(args.output / "freeze.json"),
          "points_sha256": sha(args.output / "points.json"), "elapsed_s": round(time.monotonic() - started, 3),
          "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, "control_views": 64,
          "forwards_per_encoder": 448, "panels": summaries})


if __name__ == "__main__":
    main()
