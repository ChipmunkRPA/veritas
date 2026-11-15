# SPDX-License-Identifier: MIT
"""Stress the existing whole-image check with paired low-quality JPEGs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[2]
BASELINE = ROOT / "docs/evaluation-panels/image-whole-fresh-comparison-2026-11-12.json"
ROSTER = ROOT / "docs/evaluation-panels/image-whole-fresh-sources-2026-11-12.json"
POINT = ROOT / "docs/evaluation-panels/image-v7-smaug-genuine-operating-point.json"
FREEZE_SHA = "729042123b1f39efff8b4224f89455d7d4a18db938af31d052bf5a7cb69d97c4"
SCHEMA = "chipmunk.image-whole-jpeg-transfer/1"
QUALITIES = (35, 15)
METHOD = {
    "sources": "Use all 34 native original files of the earlier held JPEG panel, not chosen by their scores. "
               "Retain the native and JPEG75 readings as earlier matched measurements.",
    "transform": "Encode the native upright RGB pixels at Pillow JPEG qualities 35 and 15. No resize, crop, "
                 "orientation change, metadata, optimization or progressive coding. Use standard 4:2:0 sampling.",
    "decision": "Use the current production worker's three exact views and reviewed CPU model, with its existing "
                "native mean and point. Compare matched gains and losses without fitting. These are whole-image "
                "responses for fully generated and ordinary scenes, without localized edit or real-world error claims.",
}


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def public_binding():
    return {"baseline_sha256": sha(BASELINE), "operating_point_sha256": sha(POINT),
            "source_roster_sha256": sha(ROSTER), "native_freeze_sha256": FREEZE_SHA}


def sources(directory):
    from scripts.run_image_whole_fresh import prepared

    directory = directory.resolve()
    require(sha(directory / "freeze.json") == FREEZE_SHA, "The earlier native delivery plan differs")
    _, panel = prepared(directory)
    native = [row for row in panel if row["format"] == "native"]
    require(len(native) == len({row["source_index"] for row in native}) == 34, "The full earlier roster is required")
    return native


def write_delivery(source, destination, quality):
    source.save(destination, format="JPEG", quality=quality, subsampling=2, progressive=False, optimize=False)


def measure(directory, model_dir):
    import PIL
    from PIL import Image

    from scripts.image_whole_evidence_worker import load_reviewed_model, score_views

    native = sources(directory)
    model = load_reviewed_model(model_dir.resolve())
    samples = []
    with TemporaryDirectory(prefix="chipmunk-jpeg-delivery-") as staging:
        destination = Path(staging) / "delivery.jpg"
        for row in native:
            with Image.open(row["input_path"], formats=["JPEG"]) as source:
                require(source.mode == "RGB" and source.getexif().get(274, 1) == 1
                        and 1024 <= min(source.size) <= max(source.size) <= 4920, "Unsupported source image scope")
                shape = list(source.size)
                deliveries = {}
                for quality in QUALITIES:
                    write_delivery(source, destination, quality)
                    _, traces, values = score_views(model, destination)
                    deliveries[str(quality)] = {"input_sha256": sha(destination), "mean": float(values.mean()),
                                                "readings": values.tolist(),
                                                "crop_edges": [trace["source_edge_bounds_rational"] for trace in traces]}
            samples.append({"index": row["source_index"], "family": row["family"], "label": row["label"],
                            "size": shape, "native_sha256": row["sha256"], "deliveries": deliveries})
            print(f"Measured {len(samples)}/{len(native)} paired JPEG sources.", flush=True)
    point = read(POINT)["gate"]["raw_mean_logit"]
    record = {"schema": SCHEMA, "method": METHOD, "inputs": public_binding(), "point": point,
              "codec": {"pillow": PIL.__version__, "libjpeg": Image.core.jpeglib_version}, "samples": samples}
    return record | {"outcomes": evaluate(record)}


def evaluate(record):
    point = read(POINT)["gate"]["raw_mean_logit"]
    require(record["schema"] == SCHEMA and record["method"] == METHOD and record["inputs"] == public_binding()
            and record["point"] == point, "The pinned inputs or point changed")
    baseline = {row["id"]: row for row in read(BASELINE)["cases"]}
    labels = {row["sha256"]: (row["label"], row["family"]) for row in read(ROSTER)["sources"]}
    samples = record["samples"]
    require(len(samples) == 34 and {item["index"] for item in samples} == set(range(34)),
            "The exact paired delivery roster is required")
    counts = defaultdict(Counter)
    for item in samples:
        prefix, label, family = f"{item['index']:03d}", item["label"], item["family"]
        width, height = item["size"]
        reference = {fmt: baseline[f"{prefix}-{fmt}"] for fmt in ("native", "jpeg75")}
        require(reference["native"]["input_sha256"] == item["native_sha256"]
                and labels[item["native_sha256"]] == (label, family)
                and 1024 <= min(width, height) <= max(width, height) <= 4920
                and set(item["deliveries"]) == {str(q) for q in QUALITIES}, "A paired source or delivery changed")
        flags = {fmt: row["candidate_flag"] for fmt, row in reference.items()}
        for delivery, data in item["deliveries"].items():
            values = data["readings"]
            require(len(values) == len(data["crop_edges"]) == 3 and all(math.isfinite(n) for n in values)
                    and math.isfinite(data["mean"]) and abs(data["mean"] - math.fsum(values) / 3) < 1e-5,
                    "The recorded mean differs from the three crop readings")
            for bounds in data["crop_edges"]:
                require(len(bounds) == 4 and all(type(n) is int and type(d) is int and d > 0 and 0 <= n <= d * limit
                                                for (n, d), limit in zip(bounds, (width, height) * 2, strict=True)),
                        "A measured view leaves the source")
            flags[f"jpeg{delivery}"] = data["mean"] > point
        for fmt, flag in flags.items():
            for group in (label, f"{label}/{family}"):
                counts[f"{group}/{fmt}"].update(files=1, flagged=flag,
                                                gained=flag and not flags["jpeg75"], lost=flags["jpeg75"] and not flag)
    require(counts["ordinary/native"]["files"] == 16 and counts["synthetic/native"]["files"] == 18
            and sorted(row["files"] for key, row in counts.items() if key.startswith("synthetic/") and key.count("/") == 2
                       and key.endswith("/native")) == [6, 6, 6], "The annotated source groups differ")
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    capture = modes.add_parser("capture")
    for name in ("sources", "model", "output"):
        capture.add_argument(name, type=Path)
    replay = modes.add_parser("review")
    replay.add_argument("receipt", type=Path)
    args = parser.parse_args()
    if args.mode == "capture":
        record = measure(args.sources, args.model)
        with args.output.open("x") as out:
            json.dump(record, out, indent=2, sort_keys=True, allow_nan=False)
            out.write("\n")
    else:
        record = read(args.receipt)
        summary = evaluate(record)
        require(summary == record["outcomes"], "The published paired results do not reproduce")
        print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
