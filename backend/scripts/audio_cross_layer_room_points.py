"""Freeze competing-room operating points before a separate room/speaker replay."""
from __future__ import annotations

import argparse
from pathlib import Path

from scripts import audio_cross_layer_transfer as heads

NOISY = ("mostly_single_interferer", "overlapping_interferers")
METHOD = {
    "schema": "chipmunk.audio-room-points/1",
    "calibration": "One maximum per genuine source group over assessed, incumbent-inconclusive noisy views. Use the second highest maximum. A strict excess can flag at most one fitting group.",
    "source_separation": "Exclude exact foreground commitments, declared groups, and original natural meeting IDs used for point selection. Reuse the frozen model heads without refitting.",
    "policy": "Supplement only assessed incumbent-inconclusive results under known noisy conditions. Clean controls retain the incumbent direction.",
    "advance": "Recover at least two synthetic groups overall with a gain in both noisy scenes, retain at least six assessed genuine views per scene, add at most one genuine flag, and exceed both direct raw and concatenated arm recovery.",
}


def room_point(rows, values):
    maxima = {}
    for row, response in zip(rows, values, strict=True):
        if (row["condition"] in NOISY and row["label"] == 0
                and row["decision"] == "inconclusive" and response is not None):
            group = row["group"]
            maxima[group] = max(response, maxima.get(group, response))
    if len(maxima) < 8:
        raise ValueError("At least eight eligible genuine calibration groups required")
    return {"value": sorted(maxima.values())[-2], "genuine_groups": len(maxima), "rank": len(maxima)-1}


def load_bound(folder):
    source = folder / "native-manifest.json"
    scores = folder / "native-scores.json"
    freeze = folder / "native-freeze.json"
    frozen = heads.read(freeze)
    manifest = heads.read(source)
    measured = heads.read(scores)
    if (frozen["manifest_sha256"] != heads.digest(source)
            or measured["native_freeze_sha256"] != heads.digest(freeze)):
        raise ValueError("The saved native measurement is unbound")
    for left, right in zip(manifest["rows"], measured["rows"], strict=True):
        if any(right[key] != value for key, value in left.items()):
            raise ValueError("Native rows differ from their frozen inputs")
    return measured["rows"], manifest, scores


def freeze(calibration, output):
    rows, manifest, scores = load_bound(calibration)
    fitted_path = calibration / "fitted.json"
    fitted = heads.read(fitted_path)
    calibrated = {arm: room_point(rows, heads.score(rows, arm, fitted["heads"].get(arm))) for arm in heads.ARMS}
    heads.write(output, {**METHOD, "points": calibrated, "source_scores_sha256": heads.digest(scores),
                         "fitted_sha256": heads.digest(fitted_path), "method_sha256": heads.digest(Path(__file__)),
                         "source_groups": len({r["group"] for r in rows}), "scenes": len(manifest["scenes"])})


def readout(calibration, transfer, output, points_path):
    points = heads.read(points_path)
    fit_rows, fit_manifest, fit_scores = load_bound(calibration)
    test_rows, test_manifest, _ = load_bound(transfer)
    fitted_path = transfer / "fitted.json"
    fitted_sha = heads.digest(fitted_path)
    if (fitted_sha != points["fitted_sha256"] or heads.digest(fit_scores) != points["source_scores_sha256"]
            or heads.digest(Path(__file__)) != points["method_sha256"]
            or test_manifest.get("room_points_sha256") != heads.digest(points_path)):
        raise ValueError("The transfer does not bind the fixed heads and points")
    for key in ("group", "source_sha256", "pcm_sha256"):
        if {r[key] for r in fit_rows}.intersection(r[key] for r in test_rows):
            raise ValueError("Room calibration and confirmation overlap")
    if {r["meeting"] for r in fit_manifest["scenes"]}.intersection(r["meeting"] for r in test_manifest["scenes"]):
        raise ValueError("Room calibration and confirmation meetings overlap")
    fitted = heads.read(fitted_path)
    noisy = [r for r in test_rows if r["condition"] in NOISY]
    if set(NOISY) != {r["condition"] for r in noisy}:
        raise ValueError("Both predeclared noisy scenes are required")
    results = {arm: heads.summarize(noisy, heads.score(noisy, arm, fitted["heads"].get(arm)),
                                    points["points"][arm]["value"]) for arm in heads.ARMS}
    heads.write(output, {"schema": METHOD["schema"], "points_sha256": heads.digest(points_path),
                         "clean_controls": sum(r["condition"] == "clean" for r in test_rows), "arms": results})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    first = commands.add_parser("freeze")
    first.add_argument("calibration", type=Path)
    first.add_argument("output", type=Path)
    later = commands.add_parser("readout")
    for name in ("calibration", "transfer", "output", "points"):
        later.add_argument(name, type=Path)
    args = parser.parse_args()
    if args.action == "freeze":
        freeze(args.calibration, args.output)
    else:
        readout(args.calibration, args.transfer, args.output, args.points)


if __name__ == "__main__":
    main()
