"""Measure multiscale patch-feature changes on a tracked face and native clock."""

from __future__ import annotations

import argparse
import hashlib
import json
from fractions import Fraction
from pathlib import Path

from scripts import video_common_sources_v89 as io
from scripts.video_face_temporal_readout import table_for

SIZE = 168
GRID = SIZE // 14
LAYERS = (3, 6)
CHANNELS = 24
SCHEDULE = tuple(range(5, 100, 6))
MINIMUM_TRACK = 8
LOCATOR, ENCODER = "yunet.2023mar", "dinov2.small"
SCHEMA = "chipmunk.video-face-residual-readings/1"
METHOD = {
    "native_profile": "100 verified 25 fps pictures at 640 by 360, with fixed scheduled positions.",
    "schedule": SCHEDULE, "alignment": "Two eye positions with verified native pixel support and uninterrupted face overlap.",
    "patch_side": SIZE, "encoder_layers": LAYERS, "projected_channels_per_layer": CHANNELS,
    "spatial_scales": (.55, 1.1, 2.2), "temporal_scales": (.60, 1.2, 2.4),
    "regions": "Fixed inner facial ellipse, surrounding ring and wider side context on the aligned canvas.",
    "features": "Normalized patch appearance, spatial and temporal Gaussian differences and consecutive changes.",
}


def align(pixels, face):
    import cv2
    import numpy as np

    eye, other = sorted(face.landmarks[:2])
    dx, dy = other[0] - eye[0], other[1] - eye[1]
    norm = dx * dx + dy * dy
    if norm < 18 ** 2:
        return None
    r, s = (.32 * SIZE * value / norm for value in (dx, dy))
    transform = np.array([[r, s, .34 * SIZE - r * eye[0] - s * eye[1]],
                          [-s, r, .38 * SIZE + s * eye[0] - r * eye[1]]])
    backward = cv2.invertAffineTransform(transform)
    region = np.array([[.10, .14], [.90, .14], [.10, .88], [.90, .88]]) * SIZE
    source = region @ backward[:, :2].T + backward[:, 2]
    if (source.min(axis=0) < 1).any() or (source.max(axis=0) >= np.asarray(pixels.shape[1::-1]) - 1).any():
        return None
    return cv2.warpAffine(pixels, transform, (SIZE, SIZE), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)


def overlap(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    area = max(0, min(ax + aw, bx + bw) - max(ax, bx)) * max(0, min(ay + ah, by + bh) - max(ay, by))
    return area / (aw * ah + bw * bh - area)


def tracked_faces(path, table, detector):
    import cv2

    from app.detectors.image.faces import detect_faces
    from app.detectors.video.frame_store import FrameStore

    store = FrameStore(path, table, max_bytes=12 * 1024**2)
    groups, span = [], []
    for index, pixels in store.iter_many(SCHEDULE):
        if not store.native_pts_verified(index):
            raise ValueError("A sampled face frame lacks a verified decoder position")
        faces = detect_faces(cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR), detector=detector, min_size=56)
        if not faces or (crop := align(pixels, faces[0])) is None:
            if span:
                groups.append(span)
            span = []
            continue
        box = faces[0].box()
        if span and (index - span[-1][0] != 6 or overlap(box, span[-1][2]) < .25):
            groups.append(span)
            span = []
        span.append((index, table.pts[index], box, crop, hashlib.sha256(pixels.tobytes()).hexdigest()))
    return max((*groups, span), key=len)


class PatchEncoder:
    def __init__(self, model, device):
        import numpy as np

        self.model, self.device = model, device
        rng = np.random.default_rng(621112)
        self.projection = [np.linalg.qr(rng.choice((-1., 1.), (384, CHANNELS)))[0].astype(np.float32) for _ in LAYERS]

    def __call__(self, crops):
        import numpy as np
        import torch

        from app.detectors.video.tracking import DinoEmbedder

        images = np.stack(crops).astype(np.float32) / 255.
        images = (images - DinoEmbedder.MEAN) / DinoEmbedder.STD
        values = []
        with torch.inference_mode():
            for group in np.array_split(images, (len(images) + 7) // 8):
                batch = torch.from_numpy(np.ascontiguousarray(group.transpose(0, 3, 1, 2))).to(self.device)
                response = self.model(pixel_values=batch, output_hidden_states=True, return_dict=True)
                channels = []
                for depth, projection in zip(LAYERS, self.projection, strict=True):
                    patches = response.hidden_states[depth][:, 1:].cpu().numpy()
                    if patches.shape[1:] != (GRID * GRID, 384):
                        raise ValueError("The pinned encoder has a different patch geometry")
                    patches = patches - patches.mean(axis=-1, keepdims=True)
                    patches /= np.maximum(np.linalg.norm(patches, axis=-1, keepdims=True), 1e-6)
                    channels.append((patches @ projection).reshape(-1, GRID, GRID, CHANNELS))
                values.append(np.concatenate(channels, axis=-1))
        return np.concatenate(values)


def regions():
    import numpy as np

    u = (np.arange(GRID) + .5) / GRID
    x, y = np.meshgrid(u, u)
    distance = ((x - .5) / .35) ** 2 + ((y - .53) / .37) ** 2
    eligible = (.10 <= x) & (x <= .90) & (.14 <= y) & (y <= .88)
    return (distance <= .61) & eligible, (.61 < distance) & (distance <= 1.16) & eligible, (distance > 1.16) & eligible


def energy_features(bands, zones):
    import numpy as np

    measurements = []
    for band in bands:
        power = np.linalg.norm(band, axis=-1)
        # Keep spatial support fixed. An energetic ordinary edge must compete
        # with measures on the surrounding picture and interior.
        values = [np.log(np.maximum(power[:, zone].mean(axis=1), 1e-8)) for zone in zones]
        for series in (*values, values[0] - values[1], values[1] - values[2]):
            measurements.extend(np.quantile(series, (.50, .90)).tolist())
    return measurements


def representation(values):
    import numpy as np
    from scipy.ndimage import gaussian_filter, gaussian_filter1d

    zones = regions()
    spatial = [gaussian_filter(values, sigma=(scale, scale), axes=(1, 2), mode="nearest")
               for scale in METHOD["spatial_scales"]]
    temporal = [gaussian_filter1d(values, scale, axis=0, mode="nearest") for scale in METHOD["temporal_scales"]]
    return {
        "appearance": np.concatenate([np.median(values[:, z].mean(axis=1), axis=0) for z in zones]).tolist(),
        "spatial": energy_features([a - b for a, b in zip(spatial, spatial[1:])], zones),
        "temporal": energy_features([a - b for a, b in zip(temporal, temporal[1:])] + [np.diff(values, axis=0)], zones),
    }


def measure(folder, output, device):
    import cv2
    import torch

    from app.detectors.image.adapters import loaders
    from app.detectors.video.adapters.loaders import load_dinov2_small
    from app.engine.fetchers import ensure_verified

    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    manifest = folder / "manifest.json"
    cases = {row["file"]: row for row in io.read(manifest)["rows"] if "file" in row and row["delivery"] != "middle"}
    assets = {}
    for name in (LOCATOR, ENCODER):
        entry = loaders._entry(name)
        if entry.licence.class_ != "commercial":
            raise ValueError("The residual experiment requires commercially permitted source weights")
        ensure_verified(entry, force=True)
        assets[name] = [file.sha256 for file in entry.source.files]
    detector = loaders.load_yunet(LOCATOR, "cpu", "float32").obj
    encoder = PatchEncoder(load_dinov2_small(ENCODER, device, "float32").obj.model, device)
    identity = json.loads(json.dumps({"schema": SCHEMA, "manifest_sha256": io.sha(manifest), "weights": assets,
                                     "method": METHOD, "code_sha256": io.sha(Path(__file__)), "device": device}))
    existing = set()
    if output.exists():
        with output.open() as stream:
            if json.loads(next(stream)) != identity:
                raise ValueError("Cannot reuse a different source or inference journal")
            for line in stream:
                row = json.loads(line)
                if row["file"] in existing or row["sha256"] != cases[row["file"]]["sha256"]:
                    raise ValueError("A recorded observation differs from the frozen delivery")
                existing.add(row["file"])
    with output.open("a" if existing else "w") as stream:
        if not existing:
            stream.write(json.dumps(identity, sort_keys=True) + "\n")
        for index, (name, case) in enumerate(cases.items(), 1):
            if name in existing:
                continue
            path = folder / name
            if io.sha(path) != case["sha256"]:
                raise ValueError("Delivered video differs from its source commitment")
            table = table_for(path)
            if (len(table) != 100 or table.width != 640 or table.height != 360 or table.rotation or table.interpolated
                    or any(tick * table.tb != Fraction(i, 25) for i, tick in enumerate(table.pts))):
                raise ValueError("Invalid picture geometry or source clock")
            track = tracked_faces(path, table, detector)
            row = {"file": name, "sha256": case["sha256"], "time_base": table.time_base,
                   "frames": [{"index": number, "pts": pts, "rgb_sha256": rgb,
                               "crop_sha256": hashlib.sha256(crop.tobytes()).hexdigest()}
                              for number, pts, _, crop, rgb in track],
                   "features": representation(encoder([item[3] for item in track])) if len(track) >= MINIMUM_TRACK else None}
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
            stream.flush()
            if index % 20 == 0 or index == len(cases):
                print(f"Read {index}/{len(cases)} videos using verified native frames", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("deliveries", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    args = parser.parse_args()
    measure(args.deliveries, args.output, args.device)
