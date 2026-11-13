# SPDX-License-Identifier: Apache-2.0
"""Replay a frozen same-file thumbnail panel using JPEG image resources."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import struct
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, features

from app.provenance import image_thumbnail_localization as localization
from app.provenance import jpeg_editor_thumbnail
from scripts import image_thumbnail_extent_transfer as earlier

SCHEMA = "chipmunk.image-editor-preview-transfer/1"
SOURCE_FREEZE = "3a4cbc6ae8f8ee74da958df8ca0698da9917fdd8fd7817ccefaeabefe8da9b5a"
PUBLIC_ROSTER = Path("docs/evaluation-panels/image-affine-native-scale-2026-11-11.json")
GATE = {"minimum_useful_generated": 4, "maximum_ordinary_flags": 0, "maximum_incompatible_flags": 0,
        "maximum_changed_comparisons": 0, "maximum_malformed_acceptances": 0, "minimum_native_controls": 2}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def segment(blob, *, name=b"", geometry=None, resource_id=1036):
    """Encode a test resource using the public editor thumbnail header."""
    with Image.open(io.BytesIO(blob), formats=["JPEG"]) as image:
        width, height = geometry or image.size
    stride = (3 * width + 3) & ~3
    payload = struct.pack(">6I2H", 1, width, height, stride, stride * height, len(blob), 24, 1) + blob
    pascal = bytes([len(name)]) + name
    if len(pascal) % 2:
        pascal += b"\0"
    resources = (jpeg_editor_thumbnail.SIGNATURE + b"8BIM" + struct.pack(">H", resource_id) + pascal
                 + struct.pack(">I", len(payload)) + payload + b"\0" * (len(payload) % 2))
    return b"\xff\xed" + struct.pack(">H", len(resources) + 2) + resources


def without_exif(data):
    require = earlier.require
    require(data[:2] == b"\xff\xd8", "Expected a JPEG source")
    header = [data[:2]]
    pos = 2
    while pos < len(data):
        start = pos
        require(data[pos] == 255, "Invalid marker")
        while data[pos] == 255:
            pos += 1
        if data[pos] == 0xDA:
            return b"".join(header + [data[start:]])
        length = int.from_bytes(data[pos + 1:pos + 3], "big")
        require(length >= 2 and pos + 1 + length <= len(data), "Invalid segment length")
        if not (data[pos] == 0xE1 and data[pos + 3:pos + 9] == b"Exif\0\0"):
            header.append(data[start:pos + 1 + length])
        pos += 1 + length
    raise ValueError("Missing scan")


def attach(data, resource):
    return data[:2] + resource + data[2:]


def cases(archive, plan):
    for item in plan["cases"]:
        if item["key"][1] != "upright":
            continue
        path = archive / "objects" / f"{item['slot']:03d}"
        data = path.with_suffix(".jpg").read_bytes()
        earlier.require(digest(data) == item["input_sha256"], "Existing image source changed")
        with Image.open(io.BytesIO(data), formats=["JPEG"]) as image:
            if image.getexif().get(274, 1) != 1:
                continue
            earlier.require(jpeg_editor_thumbnail.extract(image) is None, "Source already contains an editor preview")
        preview = earlier.earlier.preview_from(data)
        if preview is None:
            yield item, data, data, None, None
            continue
        truth = np.load(path.with_suffix(".npy"), allow_pickle=False) if item["truth_sha256"] else None
        if truth is not None:
            earlier.require(digest(truth.tobytes()) == item["truth_sha256"] and truth.sum() == item["target"],
                            "Source changed-pixel mask differs")
        fixture = attach(without_exif(data), segment(preview, name=b"preview"))
        with Image.open(io.BytesIO(fixture), formats=["JPEG"]) as decoded:
            earlier.require(localization.retained_preview(decoded, decoded.getexif()) == ("jpeg_editor_resource", preview),
                            "Fixture did not bind the expected image resource")
        yield item, data, fixture, truth, preview


def adversarial(base, preview):
    resource = segment(preview)
    return {
        "duplicated_segment": attach(base, resource + resource),
        "duplicated_resource": attach(base, _resource_payload_twice(resource)),
        "forged_dimensions": attach(base, segment(preview, geometry=(144, 140))),
        "truncated_resource": attach(base, resource[:2] + struct.pack(">H", len(resource) - 6) + resource[4:-4]),
        "truncated_thumbnail": attach(base, segment(preview[:-2] + b"\x00\x00")),
        "competing_legacy_resource": attach(base, resource + segment(preview, resource_id=1033)),
    }


def _resource_payload_twice(resource):
    signature = jpeg_editor_thumbnail.SIGNATURE
    block = resource[4 + len(signature):]
    return b"\xff\xed" + struct.pack(">H", 2 + len(signature) + len(block) * 2) + signature + block * 2


def actual_resources(root):
    roster = json.loads(PUBLIC_ROSTER.read_bytes())
    for slot in (0, 1):
        source, record = (roster["source_roster"]["sources"][slot], roster["source_intake"]["attempts"][slot])
        original = (root / record["name"]).read_bytes()
        earlier.require(source["slot"] == slot == record["slot"] and record["delivery"] == "original"
                        and digest(original) == record["sha256"]
                        and hashlib.sha1(original).hexdigest() == source["original_sha1"]
                        and source["licence_url"].startswith(("https://creativecommons.org/licenses/by/",
                                                             "https://creativecommons.org/licenses/by-sa/")),
                        "Native editor source differs from its published source or license")
        isolated = without_exif(original)
        with Image.open(io.BytesIO(isolated), formats=["JPEG"]) as image:
            blob = jpeg_editor_thumbnail.extract(image)
        earlier.require(blob is not None, "Native file lacks a usable retained editor thumbnail")
        yield {"slot": slot, "author": source["author"], "page": source["page"], "license": source["licence_url"],
               "original_sha256": digest(original), "exif_removed_sha256": digest(isolated),
               "preview_sha256": digest(blob)}, isolated


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "run", "review"))
    parser.add_argument("archive", type=Path)
    parser.add_argument("public", type=Path)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    locked = (args.archive / "freeze.json").read_bytes()
    earlier.require(digest(locked) == SOURCE_FREEZE, "Prior panel commitment changed")
    plan = json.loads(locked)
    views = list(cases(args.archive, plan))
    native = list(actual_resources(args.public))
    fixture = next((raw, blob) for row, raw, _, truth, blob in views if truth is not None)
    negatives = adversarial(without_exif(fixture[0]), fixture[1])
    bindings = [{"slot": row["slot"], "key": row["key"], "original_sha256": row["input_sha256"],
                 "fixture_sha256": digest(candidate), "preview_sha256": digest(blob) if blob is not None else None,
                 "truth_sha256": row["truth_sha256"]}
                for row, _, candidate, _, blob in views]
    code = (Path(__file__), Path(localization.__file__), Path(jpeg_editor_thumbnail.__file__), Path(earlier.__file__))
    freeze = {"schema": SCHEMA, "source_freeze_sha256": SOURCE_FREEZE, "criteria": GATE, "confidence_weight": 0,
              "inputs": bindings, "native_controls": [record for record, _ in native],
              "public_roster_sha256": digest(PUBLIC_ROSTER.read_bytes()),
              "rejected": {key: digest(body) for key, body in negatives.items()},
              "code": {path.name: digest(path.read_bytes()) for path in code},
              "runtime": {"cv2": cv2.__version__, "numpy": np.__version__, "jpeg": features.version_codec("jpg")}}
    target = args.run / "freeze.json"
    if args.action == "freeze":
        args.run.mkdir(mode=0o700)
        earlier.write(target, freeze)
        print(f"Frozen {len(views)} images and {len(negatives)} malformed cases: {digest(target.read_bytes())}", flush=True)
        return
    earlier.require(earlier.read(target) == freeze, "Frozen scope differs")
    rows, counters = [], Counter()
    for (original, raw, candidate, truth, blob), pin in zip(views, bindings, strict=True):
        baseline = earlier.measure(raw, truth, False)
        tested = earlier.measure(candidate, truth, False)
        role = original["key"][-1]
        useful = earlier.useful(tested["boxes_overlap"], original["target"])
        counters["changed_comparisons"] += baseline != tested
        counters["ordinary_flags"] += role in earlier.NORMAL and bool(tested["boxes"])
        counters["incompatible_flags"] += role in earlier.INCOMPATIBLE and bool(tested["boxes"])
        counters["useful_generated"] += role == "generated" and useful
        rows.append({**pin, "measurement": tested, "same_as_exif": baseline == tested, "useful": useful})
    failures = {}
    for key, body in negatives.items():
        with Image.open(io.BytesIO(body), formats=["JPEG"]) as image:
            failures[key] = jpeg_editor_thumbnail.extract(image) is None
    counters["malformed_acceptances"] = sum(not value for value in failures.values())
    native_readback = []
    for record, body in native:
        status = earlier.measure(body, None, False)
        counters["native_controls"] += status["status"] == "comparable" and not status["boxes"]
        native_readback.append({**record, "status": status["status"], "regions": len(status["boxes"])})
    passed = all(counters[name] >= value if label.startswith("minimum") else counters[name] <= value
                 for label, value in GATE.items() for name in (label.split("_", 1)[1],))
    result = {"schema": SCHEMA, "freeze_sha256": digest(target.read_bytes()), "gate_passed": passed,
              "counters": dict(counters), "malformed_rejected": failures, "rows": rows, "native_readback": native_readback}
    output = args.run / "result.json"
    if args.action == "review":
        earlier.require(earlier.read(output) == result, "Replay differs")
    else:
        earlier.write(output, result)
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
