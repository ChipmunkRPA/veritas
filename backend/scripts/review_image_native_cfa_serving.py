# SPDX-License-Identifier: Apache-2.0
"""Replay the held camera PNG ledger through the reusable image detector."""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import cv2

from app.detectors.image import native_cfa
from scripts import image_native_cfa_png_transfer as transfer

prior = transfer.prior
PUBLISHED = prior.ROOT / "docs/evaluation-panels/image-native-cfa-png-transfer-2026-11-13.json"
require = prior.raw_source.require


def verify(cached, archive, run):
    measured = run / "held.json"
    published = prior.read(PUBLISHED)
    require(prior.sha(measured) == published["held_sha256"], "Frozen held ledger differs")
    require(prior.sha(Path(transfer.__file__)) == published["runner_sha256"], "Frozen renderer differs")
    rows = iter(prior.read(measured)["rows"])
    counts = Counter()

    def serving_readout(case, role, rgb, truth):
        expected = next(rows)
        body = transfer.png_body(rgb)
        require((case, role) == (expected["camera_id"], expected["role"]), "Held source or role differs")
        require(prior.raw_source.sha256(body) == expected["png_sha256"] and prior.digest(truth) == expected["truth_sha256"],
                "Held native PNG or exact truth differs")
        reviewed = native_cfa.inspect_png(memoryview(body))
        require(reviewed.status == expected["status"], "Serving assessment status differs")
        counts[reviewed.status] += 1
        if reviewed.status == "assessed":
            expected_box = tuple(expected["box"]) if expected["box"] else None
            required_region = expected_box if expected["score"] > transfer.POINT else None
            require((reviewed.phase, reviewed.score, reviewed.proposal, reviewed.region)
                    == (expected["phase"], expected["score"], expected_box, required_region),
                    "Serving phase, scalar or native coordinates differ")
            if reviewed.region:
                require(reviewed.decoded_rgb_sha256 == prior.digest(rgb) and reviewed.supported_cells > 0
                        and reviewed.reference_closure >= .58, "Serving native pixel digest or fold support differs")
                counts["selected_native_hashes"] += 1
            else:
                require(reviewed.decoded_rgb_sha256 is None, "Unselected analysis exposed a pixel digest")
        return expected

    with patch.object(transfer, "readout", serving_readout):
        comparisons = sum(1 for _ in transfer.observe(cached.resolve(), archive))
    require(next(rows, None) is None and comparisons == published["exact_held_replays"] == 310,
            "Serving replay did not consume all held cases")
    return {"schema": "chipmunk.image-native-cfa-serving-parity/1", "exact_comparisons": comparisons,
            "counts": dict(counts), "held_sha256": prior.sha(measured), "transfer_receipt_sha256": prior.sha(PUBLISHED),
            "serving_detector_sha256": prior.sha(Path(native_cfa.__file__)), "reader_sha256": prior.sha(Path(__file__))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cached", type=Path)
    parser.add_argument("archive", type=Path)
    parser.add_argument("run", type=Path)
    parser.add_argument("receipt", type=Path)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    prior.write(args.receipt, verify(args.cached, args.archive, args.run))


if __name__ == "__main__":
    main()
