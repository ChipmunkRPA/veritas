"""Replay frozen article calibrations on a second publisher's held-out source."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from calibrate_text_upgrade import features, paired_delta, quality, read_predictions
from prepare_text_upgrade import digest, file_hash
from scipy.special import expit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("frozen_comparison", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("predictions", type=Path, nargs="+")
    args = parser.parse_args()
    frozen = json.loads(args.frozen_comparison.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest["previous_manifest_sha256"] != frozen["manifest_sha256"]:
        raise ValueError("These calibrations were fitted against a different first dataset")
    rows = manifest["cases"]
    cases = {row["id"]: row for row in rows}
    if len(cases) != len(rows) or any(digest(row["text"]) != row["text_sha256"] for row in rows):
        raise ValueError("Second source manifest contains a duplicate or changed item")
    manifest_sha = file_hash(args.manifest)
    raw = dict(read_predictions(path, manifest_sha, cases) for path in args.predictions)
    if "desklib" not in raw or len(raw) != len(args.predictions):
        raise ValueError("Supply the incumbent and at most one receipt per candidate")
    y = np.array([int(row["label"] == "ai") for row in rows])
    groups = np.array([row["group"] for row in rows])
    estimates = {}
    out = {"manifest_sha256": manifest_sha, "frozen_fit_sha256": file_hash(args.frozen_comparison), "models": {}}
    for name, values in raw.items():
        params = frozen["models"][name]["calibration"]
        scores = np.asarray([values[row["id"]] for row in rows]).reshape(-1, 1)
        estimate = expit(params["intercept"] + features(scores, (name,)) @ params["slopes"])
        estimates[name] = estimate
        out["models"][name] = quality(y, estimate)
        print(name, out["models"][name])
    for name, estimate in estimates.items():
        if name != "desklib":
            out["models"][name]["matched_length_pair_delta"] = paired_delta(groups, y, estimates["desklib"], estimate)
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(out, output, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
