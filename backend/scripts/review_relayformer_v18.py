"""Complete local source/native replay with separate exact component arithmetic."""

import json
import time
import traceback
from fractions import Fraction

import cv2
import numpy as np
from PIL import Image

from scripts import image_relayformer_v18 as method
from scripts import run_relayformer_v18 as run
from scripts.relayformer_native_v18 import Native, prepare


def independent_mask(raw):
    binary = raw >= np.float32(0.5)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary.astype(np.uint8), connectivity=8)
    candidates = []
    for i in range(1, n):
        x, y, w, h, area = map(int, stats[i])
        if 1000 * area < raw.size:
            continue
        bits = np.ascontiguousarray(raw[labels == i]).view(np.uint32)
        # Independent IEEE754 readout: exponent126 on[.5,1), or exact1.
        units = np.where(bits == 0x3F800000, 1 << 24, (bits & 0x7FFFFF) + (1 << 23)).astype(np.int64)
        total = int(units.sum(dtype=np.int64))
        candidates.append((Fraction(total, area * (1 << 24)), area, y, x, i, [x, y, x + w, y + h], total))
    candidates.sort(key=lambda r: (-r[0], -r[1], r[2], r[3]))
    selected = candidates[:2]
    return np.isin(labels, [r[4] for r in selected]), {
        "minimum_source_pixels": max(1, (raw.size + 999) // 1000),
        "qualifying_components": len(candidates),
        "retained_components": [
            {"pixels": r[1], "bbox_xyxy": r[5], "float32_units_sum": r[6]} for r in selected
        ],
        "source_map_quantized": False,
        "ranking": "Exact float32 mean,area,y,x;cap2",
    }


def separate_counts(mask, truth):
    # Sum a four-cell contingency table instead of the producer's three Boolean
    # intersections. Coordinates remain source pixels throughout.
    cells = np.bincount((truth.astype(np.uint8) * 2 + mask.astype(np.uint8)).ravel(), minlength=4)
    tn, fp, fn, tp = map(int, cells)
    pixels = tn + fp + fn + tp
    return {
        "intersection": tp,
        "false_positive_pixels": fp,
        "false_negative_pixels": fn,
        "pixels": pixels,
        "reference_positive_pixels": tp + fn,
        "predicted_positive_pixels": tp + fp,
        "iou": tp / (tp + fp + fn) if tp + fp + fn else 1.0,
        "pixel_f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "flagged_area_fraction": (tp + fp) / pixels,
    }


def review():
    started = time.monotonic()
    path = run.TRIAL / "review.json"
    method.require(
        not path.exists() and not (run.TRIAL / "review-start.json").exists(), "New replay attempt required"
    )
    run.write(run.TRIAL / "review-start.json", {"freeze_sha256": run.prior.sha(run.TRIAL / "freeze.json")})
    result = {
        "schema": "chipmunk.relayformer-local-review/18",
        "status": "failed",
        "outside_human_reviewers": 0,
    }
    native = None
    try:
        plan = run.check()
        points = run.points()
        previous = {r["id"]: r for role in ("point", "diagnostic") for r in run.journal(role)}
        for role in ("point", "diagnostic"):
            receipt = run.read(run.TRIAL / (role + ".json"))
            method.require(
                receipt["status"] == "complete"
                and receipt["journal_sha256"] == run.prior.sha(run.TRIAL / (role + ".jsonl")),
                "Complete original phase required",
            )
        native = Native()
        cv2.setNumThreads(1)
        masks_count = 0
        for i, case in enumerate(plan):
            data, rgb, truth = run.source(case)
            original = previous[case["id"]]
            method.require(original["status"] == "ok", "No missing native case")
            tensor, geometry = prepare(Image.fromarray(rgb))
            raw, proof = native.predict(tensor, geometry)
            method.require(
                geometry == original["geometry"]
                and proof == original["native_proof"]
                and run.prior.digest(raw) == original["raw_float32_sha256"],
                "Complete native input/map repeat differs",
            )
            saved = run.TRIAL / "private" / original["raw_map_file"]
            method.require(
                run.prior.sha(saved) == original["raw_map_sha256"], "Retained lossless map changed"
            )
            with np.load(saved, allow_pickle=False) as packet:
                method.require(
                    set(packet.files) == {"raw", "mask"} and np.array_equal(raw, packet["raw"]),
                    "Retained full float32 values differ",
                )
                mask, facts = independent_mask(raw)
                original_mask = np.unpackbits(packet["mask"], count=raw.size).reshape(raw.shape).astype(bool)
                method.require(
                    np.array_equal(mask, original_mask)
                    and facts == original["regions"]
                    and run.prior.digest(mask) == original["source_mask_sha256"],
                    "Separate IEEE ranking/source component mask differs",
                )
            ungated = separate_counts(mask, truth)
            method.require(
                ungated == original["ungated"]["candidate"], "Separate ungated source counts differ"
            )
            if case["role"] == "diagnostic":
                q = Fraction(ungated["predicted_positive_pixels"], ungated["pixels"])
                chosen = q > Fraction(
                    *points[case["format"]]["strict_source_second_maximum"]
                ) and q >= Fraction(1, 1000)
                selected = mask if chosen else np.zeros(truth.shape, bool)
                method.require(
                    separate_counts(selected, truth) == original["metrics"]["candidate"],
                    "Separate gated source counts differ",
                )
                current = run.prior.current_record(run.OLD, case, data, rgb, truth)
                method.require(
                    current == case["current"] == original["metrics"]["current"],
                    "Canonical retained reference changed",
                )
                masks_count += 1
            if (i + 1) % 16 == 0:
                print(
                    json.dumps(
                        {
                            "replayed": i + 1,
                            "planned": 384,
                            "actual_model_forwards": native.completed_forwards,
                        }
                    ),
                    flush=True,
                )
        for fmt in run.prior.FORMATS:
            repeated = method.point([r for r in run.journal("point") if r["format"] == fmt], "candidate")
            method.require(repeated == points[fmt], "Exact genuine-only point refit failed")
        method.require(
            method.aggregate(run.journal("diagnostic")) == run.read(run.TRIAL / "result.json"),
            "Frozen summary/gates differ",
        )
        result.update(
            status="complete",
            source_and_native_cases=384,
            separately_reconstructed_components=384,
            separately_recounted_diagnostic_masks=masks_count,
            exact_genuine_point_refits=2,
            complete_float32_maps_repeated_exactly=True,
            serving_admitted=False,
            reference_scope="Retained canonical current-selector maps/counts, not a new full SaaS neural run",
            historical_failed_iml_audit_cleared=False,
        )
    except BaseException:
        result["failure"] = traceback.format_exc()
        raise
    finally:
        result["duration_s"] = time.monotonic() - started
        if native is not None:
            result["attempted_model_forwards"] = native.attempted_forwards
            result["completed_model_forwards"] = native.completed_forwards
        run.write(path, result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    review()
