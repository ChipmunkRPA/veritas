# SPDX-License-Identifier: Apache-2.0
"""Compare the JPEG copy cascade on a frozen quality-85 public-photo panel."""

from __future__ import annotations

import argparse
import inspect
import io
import json
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image, features

from scripts import image_jpeg_copy_public_replay as replay

SCHEMA = "chipmunk.image-jpeg-offgrid-quality85/1"
EDIT_SOURCES = (16, 17, 21, 22, 26, 27, 32, 37)
REPETITION_CONTROLS = (1, 3, 9, 12, 13, 14, 15, 18, 19, 23, 25, 28)
QUALITY_85 = "25bb3739664e47c1cd8aadf2ff486d477738b962bc84a097ae7bab5d88084a12"
GATE = {"minimum_useful_small_source_groups": 3, "minimum_useful_medium_source_groups": 3,
        "maximum_ordinary_source_groups": 1, "maximum_repetition_source_groups": 0,
        "maximum_wrong_edit_views": 1}
source_review = replay.image_jpeg_offgrid_source


def roster():
    with patch.object(replay, "SLOTS", EDIT_SOURCES + REPETITION_CONTROLS):
        rows = replay.roster()
    return [{**row, "file": f"source-{row['slot']:02d}.jpg",
             "role": "edits_and_controls" if row["slot"] in EDIT_SOURCES else "repetition_control"} for row in rows]


def prepared(folder, sources):
    for row in sources:
        _, image = replay._source(folder / row["file"], row)
        with patch.object(replay, "QUALITIES", (85,)):
            for view, body, changed in replay.views(image, row["slot"], row["slot"]):
                if row["role"] == "edits_and_controls" or view["treatment"] == "ordinary":
                    yield view, body, changed


def freeze(folder):
    modules = (replay, replay.copy_move, replay.copy_similarity, replay.jpeg_copy_blocks,
               replay.jpeg_dq, replay.jpeg_offgrid_copy, replay.verify_image_source, replay.original, replay.offgrid)
    functions = (source_review.supported, source_review.review, source_review.verify,
                 replay.image_packets_v7.selected_translation_pairs, replay.image_packets_v7.translation_pairs)
    pins = {Path(module.__file__).name: replay.digest(Path(module.__file__).read_bytes()) for module in modules}
    pins.update({fn.__name__: replay.digest(inspect.getsource(fn).encode()) for fn in functions})
    sources = roster()
    return {"schema": SCHEMA, "gate": GATE, "new_profile": QUALITY_85, "confidence_weight": 0,
            "source_receipt_sha256": replay.digest(replay.ROSTER.read_bytes()), "sources": sources,
            "views": [row for row, _, _ in prepared(folder, sources)],
            "code": pins | {"runner": replay.digest(Path(__file__).read_bytes())},
            "runtime": {"numpy": np.__version__, "opencv": cv2.__version__,
                        "pillow": replay.version("pillow"), "jpeglib": replay.version("jpeglib"),
                        "jpeg": features.version_codec("jpg")}}


def score(folder, plan, scratch):
    rows, altered = [], set()
    profiles = source_review._QUANTIZATION_SHA256
    with patch.object(source_review, "_QUANTIZATION_SHA256", profiles | {QUALITY_85}):
        for (view, body, changed), committed in zip(prepared(folder, plan["sources"]), plan["views"], strict=True):
            replay.require(view == committed, "Frozen quality-85 delivery differs")
            replay.require(source_review.supported(body), "Fixture outside the candidate's JPEG profile")
            with Image.open(io.BytesIO(body), formats=["JPEG"]) as decoded:
                row = replay.score(decoded.convert("RGB"), view, body, changed, scratch / "submitted.jpg", altered)
            is_addition = row["basis"] == replay.OFFGRID_BASIS
            row["added"] = is_addition
            row["incumbent_useful"] = row["serving_useful"] and not is_addition
            rows.append(row)
    return rows, sorted(altered)


def summarize(rows):
    results = {}
    for treatment in ("ordinary", "contrast", "blur", *replay.EDITED):
        panel = [row for row in rows if row["treatment"] == treatment]
        results[treatment] = {
            "views": len(panel), "incumbent_useful": sum(row["incumbent_useful"] for row in panel),
            "candidate_useful": sum(row["serving_useful"] for row in panel),
            "incumbent_flags": sum(bool(row["verified_pairs"]) and not row["added"] for row in panel),
            "added_flags": sum(row["added"] for row in panel),
            "assessed_by_candidate": sum((row["offgrid"] or {}).get("status") in {"no_pair", "supported_pair"}
                                         for row in panel),
        }
    edited = [row for row in rows if row["treatment"] in replay.EDITED and row["added"]]
    ordinary = {row["slot"] for row in rows if row["treatment"] not in replay.EDITED and row["added"]}
    actual = {
        "minimum_useful_small_source_groups": results["offgrid_40x48"]["candidate_useful"]
        - results["offgrid_40x48"]["incumbent_useful"],
        "minimum_useful_medium_source_groups": results["offgrid_64"]["candidate_useful"]
        - results["offgrid_64"]["incumbent_useful"],
        "maximum_ordinary_source_groups": len(ordinary & set(EDIT_SOURCES)),
        "maximum_repetition_source_groups": len(ordinary & set(REPETITION_CONTROLS)),
        "maximum_wrong_edit_views": sum(not row["serving_useful"] for row in edited),
    }
    passed = all(actual[key] >= bound if key.startswith("minimum") else actual[key] <= bound
                 for key, bound in GATE.items())
    return {"per_treatment": results, "gate_actual": actual, "gate_passed": passed,
            "verified_crop_assets": sum(2 * len(row["verified_pairs"]) for row in rows)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("acquire", "freeze", "run", "review"))
    parser.add_argument("sources", type=Path)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    if args.action == "acquire":
        args.sources.mkdir(mode=0o700, exist_ok=True)
        for row in roster():
            path = args.sources / row["file"]
            if not path.exists():
                body = replay.publisher._fetch(row["url"])
                replay.require(replay.digest(body) == row["source_sha256"], "Downloaded source digest differs")
                with path.open("xb") as output:
                    output.write(body)
            replay._source(path, row)
        return
    path = args.run / "freeze.json"
    actual = freeze(args.sources)
    if args.action == "freeze":
        args.run.mkdir(mode=0o700)
        replay._save(path, actual)
        print(f"Frozen {len(actual['views'])} inputs: {replay.digest(path.read_bytes())}", flush=True)
        return
    replay.require(replay._json(actual) == path.read_bytes(), "Input, method, or runtime differs from frozen scope")
    rows, altered = score(args.sources, actual, args.run)
    result = {"schema": SCHEMA, "freeze_sha256": replay.digest(path.read_bytes()),
              "rows": rows, "alterations_rejected": altered, **summarize(rows)}
    output = args.run / "result.json"
    if args.action == "review":
        replay.require(replay._json(result) == output.read_bytes(), "Exact native replay differs from the original")
    else:
        replay._save(output, result)
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
