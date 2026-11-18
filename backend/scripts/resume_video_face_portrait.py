# SPDX-License-Identifier: Apache-2.0
"""Resume an interrupted portrait panel without re-scoring its committed rows."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from scripts import video_face_portrait_panel as panel

_models = None


def initialize():
    import cv2

    from app.detectors.image.adapters.loaders import load_yunet
    from app.detectors.video.adapters.loaders import load_face_appearance

    global _models
    cv2.setNumThreads(1)
    _models = (load_yunet(panel.weights.LOCATOR, "cpu", "float32").obj,
               load_face_appearance(panel.weights.ENCODER, "cpu", "float32").obj)


def process(item):
    root, dimensions, source = item
    path = root / source["file"]
    if panel.io.sha(path) != source["input_sha256"]:
        raise ValueError("A staged file differs from the bound manifest")
    return source | panel.measure_clip(path, *_models, dimensions)


def resume(inputs, journal, *, jobs):
    from app.detectors.video import face_local_detail as check

    manifest = inputs / "manifest.json"
    data = panel.io.read(manifest)
    pins = panel.weights.assets()
    header = {"schema": panel.SCHEMA, "manifest_sha256": panel.io.sha(manifest), "role": data["role"],
              "size": data["size"], "weights": pins, "script_sha256": panel.io.sha(Path(panel.__file__)),
              "proposal_sha256": panel.io.sha(Path(check.__file__)), "score_sha256": panel.io.sha(Path(panel.local.__file__))}
    rows = data["rows"]
    prefix = []
    if journal.exists():
        with journal.open(encoding="utf-8") as stream:
            if json.loads(next(stream)) != header:
                raise ValueError("Interrupted measurements used a different input or method")
            for line in stream:
                observed = json.loads(line)
                expected = rows[len(prefix)]
                if any(observed.get(key) != value for key, value in expected.items()):
                    raise ValueError("Interrupted measurements do not match the staged input order")
                prefix.append(observed["file"])
        if len(prefix) == len(rows):
            print(f"All {len(rows)} measurements are already present.", flush=True)
            return
    work = [(inputs, tuple(data["size"]), row) for row in rows[len(prefix):]]
    print(f"Keeping {len(prefix)} measured files. Measuring {len(work)} remaining files.", flush=True)
    with journal.open("a" if journal.exists() else "x", encoding="utf-8") as stream:
        if not prefix and stream.tell() == 0:
            stream.write(json.dumps(header, sort_keys=True) + "\n")
            stream.flush()
        with ProcessPoolExecutor(max_workers=jobs, initializer=initialize) as pool:
            for index, observed in enumerate(pool.map(process, work), len(prefix) + 1):
                stream.write(json.dumps(observed, sort_keys=True, allow_nan=False) + "\n")
                stream.flush()
                if index % 14 == 0 or index == len(rows):
                    print(f"Measured {index}/{len(rows)} {data['role']} files at {data['size'][1]} pixels", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path)
    parser.add_argument("journal", type=Path)
    parser.add_argument("--jobs", type=int, default=1)
    resume(**vars(parser.parse_args()))
