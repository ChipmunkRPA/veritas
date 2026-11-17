"""Read facial band energies, patch appearance and coded quantization."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from app.evidence import h264_quantization
from scripts import video_face_residual_readout as track
from scripts.video_face_residual_fit import journal
from scripts.video_face_temporal_fit import read, save, sha

SCHEMA = "chipmunk.face-frequency-observations/1"
RETAINED_TRACK_CODE = "4e68ec968c44ae0de309222a79296596b0283f470ca07d5c455699bf280288ba"


def camera_panel(roster, output):
    import cv2

    from scripts.video_face_temporal_delivery import encode, sample, tandem
    from scripts.video_face_temporal_panel import RECIPE

    cv2.setNumThreads(1)
    sources = read(roster)
    if len({row["sha256"] for row in sources}) != len(sources):
        raise ValueError("Recordings in the camera panel must be distinct")
    output.mkdir(mode=0o700)
    rows = []
    for source in sources:
        path = Path(source["path"])
        if sha(path) != source["sha256"]:
            raise ValueError("Camera source differs from its published acquisition digest")
        pictures, mapping = sample(path)
        first = output / f"{source['id']}-ordinary-direct.mp4"
        second = output / f"{source['id']}-ordinary-tandem.mp4"
        encode(pictures, first)
        tandem(first, second)
        for label, delivery in (("direct", first), ("tandem", second)):
            rows.append({"group": source["sha256"], "source_sha256": source["sha256"], "role": "held",
                         "family": "ordinary", "file": delivery.name, "delivery": label,
                         "sha256": sha(delivery), "native_map": mapping})
    save(output / "manifest.json", {"roster_sha256": sha(roster), "rows": rows, "recipe": RECIPE})


def measure(folder, output, *, saved=None, device="mps"):
    import cv2
    import numpy as np
    import torch

    from app.detectors.image.adapters import loaders
    from app.detectors.video import face_frequency
    from app.detectors.video.adapters.loaders import load_dinov2_small
    from app.engine.fetchers import ensure_verified

    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    manifest = folder / "manifest.json"
    inputs = {row["file"]: row for row in read(manifest)["rows"] if "file" in row and row["delivery"] in ("direct", "tandem")}
    weights = {}
    for identifier in (track.LOCATOR, track.ENCODER):
        entry = loaders._entry(identifier)
        if entry.licence.class_ != "commercial":
            raise ValueError("This comparison requires commercially permitted weights")
        ensure_verified(entry, force=True)
        weights[identifier] = [item.sha256 for item in entry.source.files]
    cached, encoder = {}, None
    if saved is not None:
        original, prior = journal(saved)
        if (set(inputs) != {row["file"] for row in prior} or original["manifest_sha256"] != sha(manifest)
                or original["weights"] != weights or original["code_sha256"] not in {RETAINED_TRACK_CODE, sha(Path(track.__file__))}):
            raise ValueError("The retained appearance responses use different inputs or assets")
        cached = {row["file"]: row for row in prior}
    else:
        encoder = track.PatchEncoder(load_dinov2_small(track.ENCODER, device, "float32").obj.model, device)
    locator = loaders.load_yunet(track.LOCATOR, "cpu", "float32").obj
    header = {"schema": SCHEMA, "manifest_sha256": sha(manifest), "weights": weights,
              "retained_sha256": sha(saved) if saved else None, "frequency_sha256": sha(Path(face_frequency.__file__)),
              "track_sha256": sha(Path(track.__file__)), "readout_sha256": sha(Path(__file__)), "device": device}
    current = set()
    if output.exists():
        previous, rows = journal(output)
        if previous != header or len({row["file"] for row in rows}) != len(rows):
            raise ValueError("Cannot resume a frequency journal with different inputs")
        for row in rows:
            if row["sha256"] != inputs[row["file"]]["sha256"]:
                raise ValueError("Journal response differs from its source commitment")
            current.add(row["file"])
    with output.open("a" if current else "w") as out:
        if not current:
            out.write(json.dumps(header, sort_keys=True) + "\n")
        for index, (name, case) in enumerate(inputs.items(), 1):
            if name in current:
                continue
            video = folder / name
            if sha(video) != case["sha256"]:
                raise ValueError("A delivered face video changed")
            table = track.table_for(video)
            evidence = track.tracked_faces(video, table, locator)
            anchors = [{"index": frame, "pts": pts, "rgb_sha256": digest,
                        "crop_sha256": hashlib.sha256(crop.tobytes()).hexdigest()} for frame, pts, _, crop, digest in evidence]
            try:
                quantization = h264_quantization.measure(video, table.pts)
            except h264_quantization.QuantizationUnavailable:
                quantization = None
            response = None
            if len(anchors) >= track.MINIMUM_TRACK:
                crops = np.stack([row[3] for row in evidence])
                if saved is not None:
                    row = cached[name]
                    if row["sha256"] != case["sha256"] or row["frames"] != anchors:
                        raise ValueError("The retained response does not reproduce exact aligned crops")
                    appearance = row["features"]["appearance"]
                else:
                    appearance = track.appearance(encoder(crops))
                response = face_frequency.frequency_detail(crops) | {"appearance": appearance}
            elif saved is not None and cached[name]["features"] is not None:
                raise ValueError("Face coverage differs from the retained response")
            item = {key: case[key] for key in ("file", "group", "family", "delivery", "sha256", "source_sha256")}
            item.update(time_base=table.time_base, frames=anchors, quantization=quantization, features=response)
            out.write(json.dumps(item, allow_nan=False, sort_keys=True) + "\n")
            out.flush()
            if index % 24 == 0 or index == len(inputs):
                print(f"Measured facial detail on {index}/{len(inputs)} deliveries.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("cameras", "measure"))
    parser.add_argument("inputs", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--saved", type=Path)
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    args = parser.parse_args()
    if args.phase == "cameras":
        camera_panel(args.inputs, args.output)
    else:
        measure(args.inputs, args.output, saved=args.saved, device=args.device)
