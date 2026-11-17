"""Measure multiscale patch-feature changes on a tracked face and native clock."""

from __future__ import annotations

import argparse
import hashlib
import json
from fractions import Fraction
from pathlib import Path

from app.detectors.video.face_detail_inputs import CHANNELS, LAYERS, SIZE, PatchEncoder, appearance, regions
from app.detectors.video.face_detail_inputs import tracked_faces as track_store
from app.evidence.face_detail_binding import MINIMUM_TRACK, SCHEDULE
from scripts import video_common_sources_v89 as io
from scripts.video_face_temporal_readout import table_for

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


def tracked_faces(path, table, detector):
    from app.detectors.video.frame_store import FrameStore

    return [row[:5] for row in track_store(FrameStore(path, table, max_bytes=12 * 1024**2), detector)]


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
        "appearance": appearance(values),
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
