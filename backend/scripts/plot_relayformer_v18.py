"""Private source-bound diagnostic figure; never publish benchmark photographs."""

import argparse
import hashlib
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from scripts import image_relayformer_v18 as method
from scripts import run_relayformer_v18 as run
from scripts import score_splice_localizers as display
from scripts.audit_fused_production_grid import component_masks, to_source


def reference_mask(case, data, rgb, truth):
    if case["kind"] == "ordinary":
        row = next(
            r
            for r in run.prior.lines(run.OLD / ("reference-" + case["format"] + ".jsonl"))
            if r["id"] == case["id"]
        )
        path = run.OLD / "private" / row["raw_file"]
        method.require(run.prior.sha(path) == row["raw_file_sha256"], "Reference map changed")
        with np.load(path, allow_pickle=False) as packet:
            raw = packet["raw"].copy()
    else:
        raw, row = run.prior.cached(case, current=True)
    h, w = truth.shape
    grid = display.loc.grid_shape(w, h, display.loc.GRID_MAX_SIDE["screening"])
    values = (
        np.rint(np.clip(cv2.resize(raw, grid, interpolation=cv2.INTER_LINEAR).astype(np.float64), 0, 1) * 255)
        / 255
    )
    structure = display.jpeg_bitstream.read(data)
    excluded, _ = display.loc.conditions_mask(rgb, structure.quality_luma if structure else None)
    reliable = display.loc.reliable_grid(excluded, grid)
    threshold = run.prior.splice.PIXEL_THRESHOLD if case["format"] == "native" else 0.5
    masks, _ = component_masks(values, reliable, w, h, threshold=threshold)
    selected = to_source(masks["per_map_display_cap"], w, h) & bool(row["gate"])
    method.require(
        method.counts(selected, truth) == case["current"],
        "Displayed reference pixels do not reproduce bound counts",
    )
    return selected


def figure(output):
    permitted = (run.ROOT / "output/evaluation").resolve()
    method.require(
        output.resolve().is_relative_to(permitted) and not output.exists(),
        "Private fresh figure path required",
    )
    run.check()
    method.require(
        run.read(run.TRIAL / "review.json")["status"] == "complete", "Complete local review required"
    )
    cases = {r["id"]: r for r in run.read(run.TRIAL / "plan.json")}
    rows = run.journal("diagnostic")
    used = set()
    selected = []
    specifications = (
        (
            "Strongest ungated edit proposal",
            lambda r: r["kind"] == "ai",
            lambda r: r["ungated"]["candidate"]["iou"],
        ),
        (
            "Large edit missed before gating",
            lambda r: r["kind"] == "ai" and r["ungated"]["candidate"]["intersection"] == 0,
            lambda r: r["ungated"]["candidate"]["reference_positive_pixels"],
        ),
        (
            "Benchmark original: largest false region",
            lambda r: r["kind"] == "original",
            lambda r: r["metrics"]["candidate"]["flagged_area_fraction"],
        ),
        (
            "Ordinary human retouch control",
            lambda r: r["kind"] == "ordinary",
            lambda r: r["ungated"]["candidate"]["flagged_area_fraction"],
        ),
    )
    for label, allowed, key in specifications:
        options = [r for r in rows if allowed(r) and r["source_id"] not in used]
        method.require(options, "Complete diagnostic illustration categories required")
        row = min(options, key=lambda r: (-key(r), r["id"]))
        used.add(row["source_id"])
        selected.append((label, row))
    fig, axes = plt.subplots(4, 5, figsize=(18, 13), squeeze=False)
    headings = (
        "Input / known AI-edit annotation",
        "Current selected region",
        "Uncalibrated response (fixed0–1 scale)",
        "Candidate proposals before gate",
        "Candidate after fixed genuine gate",
    )
    for j, label in enumerate(headings):
        axes[0, j].set_title(label, fontsize=9, pad=10)
    for i, (label, row) in enumerate(selected):
        case = cases[row["id"]]
        data, rgb, truth = run.source(case)
        path = run.TRIAL / "private" / row["raw_map_file"]
        method.require(run.prior.sha(path) == row["raw_map_sha256"], "Candidate map changed")
        with np.load(path, allow_pickle=False) as packet:
            raw = packet["raw"].copy()
            proposal = np.unpackbits(packet["mask"], count=raw.size).reshape(raw.shape).astype(bool)
        retained = (
            proposal
            if method.eligible(row["ungated"]["candidate"], run.points()[row["format"]])
            else np.zeros_like(proposal)
        )
        current = reference_mask(case, data, rgb, truth)
        overlays = (truth, current, None, proposal, retained)
        for j, mask in enumerate(overlays):
            axis = axes[i, j]
            axis.imshow(rgb)
            if j == 2:
                axis.imshow(raw, cmap="inferno", vmin=0, vmax=1, alpha=0.6)
            elif mask.any():
                color = np.zeros((*mask.shape, 4), np.float32)
                color[:] = [0, 0.75, 0.9, 0] if j == 0 else [1, 0.45, 0, 0]
                color[..., 3] = mask * 0.5
                axis.imshow(color)
            axis.set_xticks([])
            axis.set_yticks([])
            if j == 0:
                axis.set_ylabel(label + "\n" + case["family"] + " / " + case["format"], fontsize=8)
        if truth.any():
            axes[i, 1].set_xlabel(
                f"Known-edit overlap: IoU {row['metrics']['current']['iou']:.3f}", fontsize=8
            )
            axes[i, 3].set_xlabel(
                f"Known-edit overlap: IoU {row['ungated']['candidate']['iou']:.3f}", fontsize=8
            )
            axes[i, 4].set_xlabel(
                f"Known-edit overlap: IoU {row['metrics']['candidate']['iou']:.3f}", fontsize=8
            )
        else:
            axes[i, 1].set_xlabel(
                f"Highlighted source area: {100 * row['metrics']['current']['flagged_area_fraction']:.2f}%",
                fontsize=8,
            )
            axes[i, 3].set_xlabel(
                f"Proposed source area: {100 * row['ungated']['candidate']['flagged_area_fraction']:.2f}%",
                fontsize=8,
            )
            axes[i, 4].set_xlabel(
                f"Highlighted source area: {100 * row['metrics']['candidate']['flagged_area_fraction']:.2f}%",
                fontsize=8,
            )
    fig.suptitle("Chipmunk Verify · private benchmark diagnosis · candidate rejected", fontsize=15, y=0.985)
    fig.text(
        0.5,
        0.015,
        "Illustrative cases selected after results; aggregate decisions use all288 diagnostics. Heatmaps are uncalibrated inspection signals. Raw photographs stay private.",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    fig.savefig(output, dpi=150)
    plt.close(fig)
    run.write(
        output.with_suffix(".private.json"),
        {
            "selected_cases": [r["id"] for _, r in selected],
            "figure_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            "public": False,
            "selection": "Post-result illustrative failure categories,not model selection",
        },
    )
    print("Private diagnostic figure saved")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    figure(parser.parse_args().output)
