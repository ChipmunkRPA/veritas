"""Float32-preserving source-mask selection and exact genuine-only fraction gates."""

from fractions import Fraction

import numpy as np
from scipy import ndimage

ARMS = ("candidate",)


def require(value, reason):
    if not value:
        raise ValueError(reason)


def native_mask(raw):
    require(
        raw.ndim == 2
        and raw.dtype == np.float32
        and np.isfinite(raw).all()
        and 0 <= float(raw.min()) <= float(raw.max()) <= 1,
        "Complete finite native float32 source map required",
    )
    labels, n = ndimage.label(raw >= np.float32(0.5), structure=np.ones((3, 3), np.uint8))
    minimum = max(1, (raw.size + 999) // 1000)
    candidates = []
    for i in range(1, n + 1):
        where = labels == i
        count = int(where.sum())
        if count < minimum:
            continue
        y, x = np.nonzero(where)
        # On[.5,1], every float32 is an exact multiple of2^-24. Integer
        # sums/comparisons retain all ranking bits, including true ties.
        units = int(np.rint(raw[where].astype(np.float64) * (1 << 24)).astype(np.int64).sum(dtype=np.int64))
        candidates.append(
            (
                Fraction(units, count * (1 << 24)),
                count,
                int(y.min()),
                int(x.min()),
                i,
                [int(x.min()), int(y.min()), int(x.max()) + 1, int(y.max()) + 1],
                units,
            )
        )
    candidates.sort(key=lambda r: (-r[0], -r[1], r[2], r[3]))
    chosen = candidates[:2]
    mask = np.isin(labels, [r[4] for r in chosen])
    return mask, {
        "minimum_source_pixels": minimum,
        "qualifying_components": len(candidates),
        "retained_components": [
            {"pixels": r[1], "bbox_xyxy": r[5], "float32_units_sum": r[6]} for r in chosen
        ],
        "source_map_quantized": False,
        "ranking": "Exact float32 mean,area,y,x;cap2",
    }


def point(rows, arm):
    require(arm in ARMS and len(rows) == 48, "All48 genuine point rows required")
    grouped = {}
    for row in rows:
        require(
            row["role"] == "point"
            and row["label"] == 0
            and row["kind"] in ("original", "ordinary")
            and row["status"] == "ok",
            "Only successful genuine controls may select a point",
        )
        metric = row["ungated"][arm]
        require(metric["pixels"] > 0, "Nonempty source mask required")
        grouped.setdefault(row["source_id"], []).append(
            (row["kind"], Fraction(metric["predicted_positive_pixels"], metric["pixels"]))
        )
    require(
        len(grouped) == 24
        and all(len(r) == 2 and {k for k, _ in r} == {"original", "ordinary"} for r in grouped.values()),
        "Exactly24 source groups with both genuine operations required",
    )
    maxima = sorted([max(q for _, q in r) for r in grouped.values()], reverse=True)
    bound = maxima[1]
    result = {
        "strict_source_second_maximum": [bound.numerator, bound.denominator],
        "minimum_area_fraction": [1, 1000],
        "source_groups": 24,
        "genuine_group_budget": 1,
        "uses_ai_or_diagnostic": False,
    }
    result["groups_flagged"] = sum(q > bound and q >= Fraction(1, 1000) for q in maxima)
    return result


def eligible(metric, policy):
    ratio = Fraction(metric["predicted_positive_pixels"], metric["pixels"])
    return ratio > Fraction(*policy["strict_source_second_maximum"]) and ratio >= Fraction(
        *policy["minimum_area_fraction"]
    )


def counts(mask, truth):
    require(
        mask.dtype == truth.dtype == np.bool_ and mask.shape == truth.shape and mask.ndim == 2,
        "Exact aligned source masks required",
    )
    tp, fp, fn = int((mask & truth).sum()), int((mask & ~truth).sum()), int((~mask & truth).sum())
    return {
        "intersection": tp,
        "false_positive_pixels": fp,
        "false_negative_pixels": fn,
        "pixels": int(truth.size),
        "reference_positive_pixels": tp + fn,
        "predicted_positive_pixels": tp + fp,
        "iou": tp / (tp + fp + fn) if tp + fp + fn else 1.0,
        "pixel_f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "flagged_area_fraction": (tp + fp) / truth.size,
    }


def useful(metric):
    return metric["reference_positive_pixels"] > 0 and 10 * metric["intersection"] >= 3 * (
        metric["intersection"] + metric["false_positive_pixels"] + metric["false_negative_pixels"]
    )


def aggregate(outcomes):
    require(
        len(outcomes) == 288 and len({r["id"] for r in outcomes}) == 288,
        "All288 diagnostic outcomes required",
    )
    formats = {}
    for fmt in ("native", "jpeg75"):
        part = [r for r in outcomes if r["format"] == fmt]
        require(
            len(part) == 144
            and all(sum(r["kind"] == k for r in part) == 48 for k in ("ai", "original", "ordinary")),
            "All format/source conditions required",
        )
        summary = {}
        for arm in ("current", *ARMS):
            edited = [r for r in part if r["kind"] == "ai"]
            summary[arm] = {
                "edited": 48,
                "useful_locations": sum(useful(r["metrics"][arm]) for r in edited),
                "any_overlap": sum(r["metrics"][arm]["intersection"] > 0 for r in edited),
                "mean_iou": float(np.mean([r["metrics"][arm]["iou"] for r in edited])),
                "mean_pixel_f1": float(np.mean([r["metrics"][arm]["pixel_f1"] for r in edited])),
            }
            for kind in ("original", "ordinary"):
                controls = [r["metrics"][arm] for r in part if r["kind"] == kind]
                summary[arm][kind + "_highlights"] = sum(r["predicted_positive_pixels"] > 0 for r in controls)
                summary[arm][kind + "_mean_highlighted_area"] = float(
                    np.mean([r["flagged_area_fraction"] for r in controls])
                )
        gates = {}
        for arm in ARMS:
            a, b = summary[arm], summary["current"]
            lost = sum(
                useful(r["metrics"]["current"]) and not useful(r["metrics"][arm])
                for r in part
                if r["kind"] == "ai"
            )
            checks = {
                "more_useful_locations": a["useful_locations"] > b["useful_locations"],
                "higher_mean_iou": a["mean_iou"] > b["mean_iou"],
                "no_lost_current_useful_location": lost == 0,
                "original_highlight_budget": a["original_highlights"] <= b["original_highlights"] + 1,
                "ordinary_highlight_budget": a["ordinary_highlights"] <= b["ordinary_highlights"] + 1,
            }
            diff = np.asarray(
                [
                    r["metrics"][arm]["iou"] - r["metrics"]["current"]["iou"]
                    for r in sorted(part, key=lambda r: r["source_id"])
                    if r["kind"] == "ai"
                ]
            )
            rng = np.random.default_rng(20261108)
            intervals = np.quantile(diff[rng.integers(0, 48, (2000, 48))].mean(1), [0.025, 0.975]).tolist()
            gates[arm] = {
                "checks": checks,
                "passes": all(checks.values()),
                "lost_current_useful_locations": lost,
                "paired_source_mean_iou_difference_bootstrap95": intervals,
            }
        formats[fmt] = {"arms": summary, "gates": gates}
    selected = next((arm for arm in ARMS if all(formats[f]["gates"][arm]["passes"] for f in formats)), None)
    return {
        "schema": "chipmunk.relayformer-development-result/18",
        "formats": formats,
        "source_groups": 48,
        "diagnostic_inputs": 288,
        "advance_to_separate_fresh_confirmation_arm": selected,
        "serving_admitted": False,
        "used_in_confidence": False,
        "fresh_source_validation": False,
        "outside_human_reviewers": 0,
    }
