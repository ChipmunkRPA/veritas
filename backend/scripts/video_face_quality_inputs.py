# SPDX-License-Identifier: MIT
"""Add a fresh intermediate H.264 delivery and bind measured slice quantization."""
from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from scripts.video_face_temporal_delivery import _encoder, encode, sample, tandem
from scripts.video_face_temporal_fit import read, save, sha

INTERMEDIATE_CRF = 30


def add_intermediate(pair, output):
    prior = read(pair / "manifest.json")
    output.mkdir(mode=0o700)
    rows = []
    for row in prior["rows"]:
        rows.append(row)
        if "file" in row:
            os.link(pair / row["file"], output / row["file"])

    def build(row):
        before = output / row["file"]
        assert sha(before) == row["sha256"]
        target = before.with_name(before.name.replace("-direct.", "-middle."))
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-xerror", "-threads", "1", "-i", str(before),
                        *_encoder(INTERMEDIATE_CRF), str(target)], capture_output=True, check=True, timeout=120)
        return row | {"file": target.name, "delivery": "middle", "sha256": sha(target)}

    with ThreadPoolExecutor(max_workers=3) as pool:
        rows.extend(pool.map(build, [row for row in prior["rows"] if row.get("delivery") == "direct"]))
    save(output / "manifest.json", prior | {"rows": rows, "middle": {"crf": INTERMEDIATE_CRF, "source": "direct"}})
    print("Prepared", len(rows), "deliveries with the intermediate encode.")


def measure(folder, output):
    from app.evidence import h264_quantization
    from scripts.video_face_temporal_readout import table_for

    manifest = folder / "manifest.json"
    targets = [row for row in read(manifest)["rows"] if "file" in row]

    def measure_one(row):
        path = folder / row["file"]
        assert sha(path) == row["sha256"]
        table = table_for(path)
        try:
            trace = h264_quantization.measure(path, table.pts)
        except h264_quantization.QuantizationUnavailable:
            trace = None
        return {"file": row["file"], "sha256": row["sha256"], "time_base": table.time_base, "quantization": trace}

    with ThreadPoolExecutor(max_workers=4) as pool:
        measured = list(pool.map(measure_one, targets))
    save(output, {"manifest_sha256": sha(manifest), "rows": measured})
    print("Measured coded quantization on", sum(row["quantization"] is not None for row in measured), "/", len(measured))


def camera_controls(source, output):
    """Build ordinary controls from the eight published, previously qualified camera slots."""
    import cv2

    from scripts.video_face_temporal_panel import RECIPE

    cv2.setNumThreads(1)
    output.mkdir(mode=0o700)
    qualification = source / "summary.json"
    summary = read(qualification)
    assert summary["source_groups"] == 8 and summary["status"] == "complete"

    def build(index):
        folder = source / f"slot-{index:02d}-attempt-01"
        receipt = read(folder / "qualification.json")
        original = folder / "unchanged.mp4"
        checksum = sha(original)
        assert checksum == receipt["cases"]["unchanged"]["sha256"]
        pictures, frame_map = sample(original)
        group = hashlib.sha256(f"vcd2-v290/{checksum}".encode()).hexdigest()
        common = {"group": group, "role": "held", "family": "ordinary", "source_sha256": checksum, "native_map": frame_map}
        direct = output / f"camera-{index:02d}-ordinary-direct.mp4"
        strong = output / f"camera-{index:02d}-ordinary-tandem.mp4"
        encode(pictures, direct)
        tandem(direct, strong)
        return [common | {"file": path.name, "delivery": path.stem.rsplit("-", 1)[1], "sha256": sha(path)}
                for path in (direct, strong)]

    with ThreadPoolExecutor(max_workers=2) as pool:
        rows = [row for clips in pool.map(build, range(1, 9)) for row in clips]
    save(output / "manifest.json", {"camera_qualification_sha256": sha(qualification), "recipe": RECIPE, "rows": rows})
    print("Prepared", len(rows), "camera control deliveries.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("middle", "measure", "cameras"))
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.action == "middle":
        add_intermediate(args.source, args.output)
    elif args.action == "cameras":
        camera_controls(args.source, args.output)
    else:
        measure(args.source, args.output)
