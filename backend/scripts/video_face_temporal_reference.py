# SPDX-License-Identifier: MIT
"""Measure the app's paired whole-clip face check on the frozen deliveries."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from scripts import video_common_sources_v89 as files
from scripts.video_face_temporal_readout import table_for


def run(folder, journal, split, device):
    import cv2
    import torch

    from app.detectors.image.adapters import loaders
    from app.detectors.video.adapters.face_sequence import SequenceInputs
    from app.detectors.video.adapters.whole_face import WholeFaceReviewDetector
    from app.detectors.video.frame_store import FrameStore
    from app.engine.boundary import Deadline
    from app.engine.fetchers import ensure_verified

    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    check = WholeFaceReviewDetector()
    manifest = folder / "manifest.json"
    targets = {row["file"]: row for row in files.read(manifest)["rows"] if row["role"] == split and "file" in row}
    model_loaders = (loaders.load_yunet, loaders.load_dfdc_b7, loaders.load_dfdc_b7, loaders.load_gend_pe)
    models = {}
    identity = {"manifest_sha256": files.sha(manifest), "detector": check.spec.version, "split": split, "device": device,
                "weights": {}}
    for key, loader in zip(check.MODELS, model_loaders, strict=True):
        entry = loaders._entry(key)
        ensure_verified(entry, force=True)
        identity["weights"][key] = [weight.sha256 for weight in entry.source.files]
        models[key] = loader(key, "cpu" if key == check.YUNET else device, "float32")
    done = set()
    if journal.exists():
        rows = [json.loads(line) for line in journal.read_text().splitlines()]
        assert rows and rows[0] == identity
        for row in rows[1:]:
            assert row["file"] in targets and row["file"] not in done and row["sha256"] == targets[row["file"]]["sha256"]
            done.add(row["file"])
    with journal.open("a" if done else "w") as output:
        if not done:
            output.write(json.dumps(identity, sort_keys=True) + "\n")
        for name, source in targets.items():
            if name in done:
                continue
            path = folder / name
            assert files.sha(path) == source["sha256"], name
            table = table_for(path)
            store = FrameStore(path, table, max_bytes=8 * 1024**2)
            inputs = SequenceInputs(table, store, centers=(), exhibits=())
            context = SimpleNamespace(outputs=SimpleNamespace(face_review=inputs), lease=SimpleNamespace(get=models.__getitem__),
                                      device=device, deadline=Deadline(120))
            capability = check.applicable(context)
            assert capability.ok, name
            result = check.run(context)
            proof = (result.details or {}).get("whole_face_review") if result.status == "ok" else None
            row = {"file": name, "sha256": source["sha256"], "status": result.status, "review": proof}
            output.write(json.dumps(row, allow_nan=False, sort_keys=True) + "\n")
            output.flush()
            print(name, result.status, proof["flagged"] if proof else None, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("deliveries", type=Path)
    parser.add_argument("journal", type=Path)
    parser.add_argument("split", choices=("development", "held"))
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    args = parser.parse_args()
    run(args.deliveries, args.journal, args.split, args.device)
