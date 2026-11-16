"""Measure native JPEG cells with a frozen local encoder and residual filters."""

from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path

from scripts.video_face_temporal_fit import read, save, sha

CELL = 14
SIDE = 16
OVERLAP_STEP = 12
LAYERS = (3, 6)
PROJECTION_SIZE = 16
WEIGHTS = "dinov2.small"
STAT_SCALE = (.6, 1.2, 2.4, 4.8)


def residual_grid(pixels):
    import cv2
    import numpy as np

    height, width = (side // CELL for side in pixels.shape[:2])
    colors = cv2.cvtColor(pixels, cv2.COLOR_RGB2YCrCb).astype(np.float32)
    bands = [cv2.GaussianBlur(colors, (0, 0), value) for value in STAT_SCALE]
    power = []
    for original, smooth in zip((colors, *bands), bands):
        delta = (original - smooth)[:height*CELL, :width*CELL].reshape(height, CELL, width, CELL, 3)
        variance = (delta * delta).mean(axis=(1, 3))
        power.append(.5 * np.log(variance + .25))
    signal = bands[-1][:height*CELL, :width*CELL].reshape(height, CELL, width, CELL, 3).mean(axis=(1, 3))
    horizontal = np.abs(np.diff(bands[1][..., 0], append=bands[1][:, -1:, 0], axis=1))
    vertical = np.abs(np.diff(bands[1][..., 0], append=bands[1][-1:, :, 0], axis=0))
    texture = np.stack((horizontal, vertical), axis=-1)[:height*CELL, :width*CELL]
    texture = np.log1p(texture.reshape(height, CELL, width, CELL, 2).mean(axis=(1, 3)))
    return np.concatenate((*power, *(a - b for a, b in zip(power, power[1:])), signal / 64., texture), axis=2)


def tile_starts(size):
    return sorted({*range(0, size - SIDE + 1, OVERLAP_STEP), size - SIDE})


class FrozenEncoder:
    def __init__(self, device):
        import numpy as np
        from app.detectors.image.adapters.loaders import _entry
        from app.detectors.video.adapters.loaders import load_dinov2_small
        from app.engine.fetchers import ensure_verified

        entry = _entry(WEIGHTS)
        if entry.licence.class_ != "commercial" or entry.licence.spdx != "Apache-2.0":
            raise ValueError("The pinned encoder needs a commercial licence")
        ensure_verified(entry, force=True)
        self.hashes = [file.sha256 for file in entry.source.files]
        self.device = device
        self.model = load_dinov2_small(WEIGHTS, device, "float32").obj.model.eval()
        random = np.random.default_rng(111607)
        self.projections = [random.choice((-1., 1.), size=(384, PROJECTION_SIZE)).astype(np.float32) for _ in LAYERS]

    def __call__(self, rgb):
        import numpy as np
        import torch
        from app.detectors.video.tracking import DinoEmbedder

        height, width = (size // CELL for size in rgb.shape[:2])
        limit = SIDE * CELL
        positions = list(product(tile_starts(height), tile_starts(width)))
        weight = np.minimum(3, np.minimum(np.arange(SIDE) + 1, SIDE - np.arange(SIDE))).astype(np.float32)
        window = weight[:, None] * weight[None, :]
        summed = np.zeros((height, width, len(LAYERS) * PROJECTION_SIZE), np.float32)
        coverage = np.zeros((height, width), np.float32)
        with torch.inference_mode():
            for start in range(0, len(positions), 8):
                sites = positions[start:start + 8]
                chips = np.stack([rgb[y * CELL:y * CELL + limit, x * CELL:x * CELL + limit] for y, x in sites])
                normalized = (chips.astype(np.float32) / 255. - DinoEmbedder.MEAN) / DinoEmbedder.STD
                batch = torch.from_numpy(np.ascontiguousarray(normalized.transpose(0, 3, 1, 2))).to(self.device)
                readings = self.model(pixel_values=batch, output_hidden_states=True, return_dict=True).hidden_states
                features = []
                for layer, projection in zip(LAYERS, self.projections, strict=True):
                    tokens = readings[layer][:, 1:].cpu().numpy()
                    if tokens.shape[1:] != (SIDE * SIDE, 384):
                        raise ValueError("Patch tokens no longer align with the native image cells")
                    tokens = tokens - tokens.mean(axis=2, keepdims=True)
                    tokens /= np.maximum(np.linalg.norm(tokens, axis=2, keepdims=True), 1e-6)
                    features.append((tokens @ projection).reshape(-1, SIDE, SIDE, PROJECTION_SIZE))
                for (y, x), grid in zip(sites, np.concatenate(features, axis=3), strict=True):
                    summed[y:y+SIDE, x:x+SIDE] += grid * window[..., None]
                    coverage[y:y+SIDE, x:x+SIDE] += window
        if not np.all(coverage > 0):
            raise ValueError("A complete native image cell lacks a model reading")
        return summed / coverage[..., None]


def measure(study, output, roles, device):
    from concurrent.futures import ThreadPoolExecutor

    import cv2
    import numpy as np
    import torch
    from PIL import Image

    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    manifest = study / "manifest.json"
    plan = read(manifest)
    groups = {row["tag"]: row for row in plan["groups"] if row["split"] in roles}
    work = [row for row in plan["images"] if row["group"] in groups]
    if not work:
        raise ValueError("No images admitted to these feature roles")
    encoder = FrozenEncoder(device)
    output.mkdir(mode=0o700, exist_ok=True)
    identity = {"source_sha256": sha(manifest), "code_sha256": sha(Path(__file__)), "weight_sha256": encoder.hashes,
                "device": device, "roles": list(roles), "feature_cell_px": CELL, "tile_size_cells": SIDE,
                "tile_step_cells": OVERLAP_STEP, "layers": list(LAYERS), "residual_scales": list(STAT_SCALE)}
    header = output / "features.json"
    if header.exists():
        if read(header) != identity:
            raise ValueError("The retained feature cache uses different inputs or code")
    else:
        save(header, identity)

    def decode(row):
        path = study / row["file"]
        if sha(path) != row["sha256"]:
            raise ValueError("Encoded image differs from the frozen plan")
        with Image.open(path, formats=["JPEG"]) as image:
            rgb = np.array(image.convert("RGB"))
        if list(rgb.shape[:2]) != groups[row["group"]]["shape"]:
            raise ValueError("JPEG source dimensions changed")
        return rgb, residual_grid(rgb)

    with ThreadPoolExecutor(max_workers=1) as pool:
        upcoming = tuple((row, output / (row["file"].replace("/", "-") + ".npz")) for row in work)
        remaining = [(row, path) for row, path in upcoming if not path.exists()]
        if not remaining:
            return
        future = pool.submit(decode, remaining[0][0])
        for index, (row, path) in enumerate(remaining, 1):
            rgb, physical = future.result()
            if index < len(remaining):
                future = pool.submit(decode, remaining[index][0])
            native = encoder(rgb)
            features = {"source": row["sha256"], "visual": native.astype(np.float16), "residual": physical.astype(np.float16)}
            temporary = path.with_suffix(".partial.npz")
            np.savez(temporary, **features)
            temporary.replace(path)
            if index % 24 == 0 or index == len(remaining):
                print(f"Measured {index}/{len(remaining)} native feature grids", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("deliveries", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("roles", nargs="+", choices=("fit", "calibration", "held", "transfer"))
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    args = parser.parse_args()
    measure(args.deliveries, args.output, args.roles, args.device)
