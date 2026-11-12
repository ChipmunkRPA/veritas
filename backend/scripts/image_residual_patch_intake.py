# SPDX-License-Identifier: Apache-2.0
"""Qualify additional source-separated SDXL and Kandinsky image edits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts import image_background_patch_transfer as base
from scripts.video_appearance_corrected_v112 import scoped

SCHEMA = "chipmunk.image-residual-patch-intake/1"
FILES = ("plan.json", "attempts.jsonl", "result.json")
MAX_PER_FAMILY = 4


def intake(args):
    keys, known, exposure = base.match_exposure(args.fit, args.calibration)
    for label, folder in (("earlier", args.earlier), ("later", args.later)):
        receipt = base.roster.read(folder / "result.json")
        digests = {name: base.roster.sha(folder / name) for name in FILES}
        base.roster.require(digests["plan.json"] == receipt["plan_sha256"]
                            and digests["attempts.jsonl"] == receipt["attempts_sha256"],
                            "Historical image intake commitments differ")
        exposure[label] = digests
        cases = base.roster.read(folder / "plan.json")["cases"]
        attempts = [json.loads(line) for line in (folder / "attempts.jsonl").read_text().splitlines()]
        for case, result in zip(cases, attempts, strict=True):
            base.roster.require(case["slot"] == result["slot"], "Historical source slot differs")
            keys.update(base.roster.source_keys(case))
            if result["status"] == "qualified":
                fingerprint = result["fingerprint"]
                known[0].append(fingerprint)
                known[1].add(fingerprint["encoded_sha256"])
                known[2].add(fingerprint["rgb_sha256"])
    selection = scoped(base.bindings, {"GROUPS_PER_GENERATOR": MAX_PER_FAMILY})
    plan = selection(args.metadata, keys.copy()) | {
        "intake_schema": SCHEMA,
        "exposure_sha256": exposure,
        "implementations": {Path(module.__file__).name: base.roster.sha(Path(module.__file__))
                            for module in (base, base.roster, base.source, base.transport)}
                           | {Path(__file__).name: base.roster.sha(Path(__file__))},
    }
    return plan, keys, known


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "acquire"))
    for key in ("metadata", "fit", "calibration", "earlier", "later", "output"):
        parser.add_argument(key, type=Path)
    args = parser.parse_args()
    plan, keys, known = intake(args)
    saved = args.output / "plan.json"
    if args.stage == "freeze":
        args.output.mkdir(mode=0o700)
        base.shared.write(saved, plan)
        print(json.dumps(plan["source_counts"]), flush=True)
        return
    base.roster.require(base.roster.read(saved) == plan, "Source-only freeze changed")
    base.acquire(plan, args.output, keys, known)


if __name__ == "__main__":
    main()
