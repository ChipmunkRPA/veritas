"""Portable metadata consistency review; no media, models or third-party imports.

This checks published construction claims, not the source bytes or their labels.
It cannot substitute for the native-source audit or independent human review.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import re
from fractions import Fraction
from pathlib import Path

CONDITIONS = (
    "original",
    "local-light",
    "local-resample",
    "muse-resample",
    "local-blur-2",
    "local-blur-4",
    "ai-redraw",
    "ai-retime",
    "muse-ai-aligned",
    "muse-ai-earlier",
)


def require(value, reason):
    if not value:
        raise ValueError(reason)


def digest_string(value):
    return isinstance(value, str) and bool(re.fullmatch("[0-9a-f]{64}", value))


def review(body):
    plan, inputs, audit = body["plan"], body["inputs"], body["source_audit"]
    require(plan["policy"] == "chipmunk.mouth-temporal-preparation/11", "Wrong preparation policy")
    require(
        inputs["plan_sha256"]
        == audit["plan_sha256"]
        == body["lock"]["fit_plan_sha256"]
        == body["original_plan_sha256"],
        "Plan identity mismatch",
    )
    require(audit["manifest_sha256"] == body["original_manifest_sha256"], "Manifest identity mismatch")
    require(
        not plan["deployment_eligible"]
        and not inputs["deployment_eligible"]
        and not body["lock"]["deployment_eligible"],
        "Research preparation cannot declare deployment readiness",
    )
    sources = plan["sources"]
    groups = {s["group"] for s in sources}
    require(
        len(sources) == len(groups) == 16 and all(s["role"] == "fit" for s in sources),
        "Keep all sixteen fitting actors",
    )
    require(
        not groups & set(plan["excluded_groups"]) and groups == set(body["lock"]["fit_groups"]),
        "Fitting and excluded roles overlap or changed",
    )
    require(
        tuple(plan["conditions"]) == CONDITIONS and plan["planned_cases"] == 160,
        "Changed condition population",
    )
    rows = inputs["items"]
    expected = [(s["id"] + ":v11:" + c, s["group"], c) for s in sources for c in CONDITIONS]
    require(
        [(r["id"], r["group"], r["condition"]) for r in rows] == expected,
        "Missing, repeated or reordered source condition",
    )
    require([r["id"] for r in audit["cases"]] == [r["id"] for r in rows], "Missing audit outcomes")
    totals = {
        "native_frames": 0,
        "exact_masks": 0,
        "common_pcm_samples": 0,
        "neural_input_bindings": 0,
        "sampled_neural_output_replays": 0,
    }
    for source in sources:
        local = [r for r in rows if r["group"] == source["group"]]
        span = source["clock_span"]
        n = span["frames"]
        require(51 <= n <= 71 and span["source_frame_range"] == [0, n], "Changed complete picture span")
        require(
            span["excluded_end_picture_frames"] == span["source_video_frames"] - n >= 0,
            "Padded or inconsistent picture count",
        )
        source_pts = span["source_pts"]
        tb = Fraction(span["source_time_base"])
        require(
            len(source_pts) == n
            and all((p - source_pts[0]) * tb == Fraction(i, 25) for i, p in enumerate(source_pts)),
            "Non-native source frame clock",
        )
        require(
            math.isclose(span["source_origin_s"], float(source_pts[0] * tb), abs_tol=1e-12),
            "Incorrect source origin",
        )
        start, end = (n - 20) // 2, (n - 20) // 2 + 20
        ap = {m["frame_index"]: m["model_proof"]["source_aperture"] for m in local[6]["masks"]}
        for row in local:
            c = row["condition"]
            ai = c in CONDITIONS[6:]
            require(
                row["status"] == "ready" and row["role"] == "fit" and row["label"] == int(ai),
                "Changed assessment, role or construction label",
            )
            require(
                row["source_sha256"] == source["sha256"]
                and row["source_url"] == source["download_url"]
                and row["source_license"] == source["source_license"],
                "Source attribution mismatch",
            )
            require(row["source_clock_span"] == span and row["frames"] == n, "Different common source span")
            require(
                row["edit_frame_range"] == [start, end]
                and row["edit_time_range_s"] == [start / 25, end / 25]
                and row["source_ranges"] == {"before": [0, start], "during": [start, end], "after": [end, n]},
                "Changed known edit boundaries",
            )
            require(
                len(row["pts"]) == n
                and all(p * Fraction(row["time_base"]) == Fraction(i, 25) for i, p in enumerate(row["pts"])),
                "Encoded picture clock differs",
            )
            mapping = row["original_audio_mapping"]
            timeline = mapping["timeline"]
            require(
                mapping["sample_rate"] == 16000 and len(timeline) == 1 and timeline[0]["start_sample"] == 0,
                "Non-contiguous original audio mapping",
            )
            offset = source_pts[0] * tb * 16000 - timeline[0]["source_pts"]
            require(
                offset.denominator == 1
                and offset >= 0
                and span["source_sample_range"] == [int(offset), int(offset) + n * 640],
                "Audio/picture offset differs",
            )
            require(
                row["audio_samples"] == n * 640
                and row["pcm_sha256"] == local[0]["pcm_sha256"]
                and mapping == local[0]["original_audio_mapping"],
                "Changed original PCM span",
            )
            require(
                len(row["original_rgb_sha256"]) == n
                and all(digest_string(h) for h in row["original_rgb_sha256"]),
                "Missing original pixel bindings",
            )
            require(
                [m["frame_index"] for m in row["masks"]] == list(range(start, end)),
                "Missing or extra edit mask",
            )
            for mask in row["masks"]:
                i = mask["frame_index"]
                require(
                    mask["pts"] == source_pts[i]
                    and math.isclose(mask["source_time_s"], float(source_pts[i] * tb), abs_tol=1e-12),
                    "Mask bound to another source time",
                )
                require(
                    mask["source_rgb_sha256"] == row["original_rgb_sha256"][i]
                    and digest_string(mask["sha256"]),
                    "Mask/source pixel identity mismatch",
                )
                require(
                    mask["opacity"] == min(1.0, (i - start + 1) / 2, (end - i) / 2)
                    and (mask["changed_pixels"] > 0) == (c != "original"),
                    "Invalid support opacity or changed-pixel count",
                )
                proof = mask["model_proof"]
                if c.startswith("ai-"):
                    target = i if c == "ai-redraw" else i - 8
                    require(
                        proof["target_frame_index"] == target
                        and 0 <= proof["source_aperture"] <= 1
                        and 0 <= proof["target_aperture"] <= 1,
                        "Portrait target binding mismatch",
                    )
                    if target in ap:
                        require(
                            proof["target_aperture"] == ap[target],
                            "Portrait aperture refers to another frame",
                        )
                    require(digest_string(proof["generated_rgb_sha256"]), "Missing portrait output binding")
                elif c.startswith("muse-ai-"):
                    tick = (source_pts[i] * tb * 16000 - timeline[0]["source_pts"]) / 640
                    require(
                        tick.denominator == 1
                        and proof["conditioning_frame"] == int(tick) - (8 if c.endswith("earlier") else 0),
                        "Speech conditioning clock mismatch",
                    )
                    seed = int.from_bytes(
                        hashlib.sha256(f"chipmunk.mouth-temporal/11:{source['id']}:{i}".encode()).digest()[
                            :8
                        ],
                        "big",
                    ) % (2**63)
                    require(proof["seed"] == seed, "Speech latent seed is not source/frame bound")
                    require(
                        all(digest_string(v) for k, v in proof.items() if k.endswith("_sha256")),
                        "Invalid neural input/output digest",
                    )
                else:
                    require(proof is None, "Ordinary control has neural proof")
            record = next(r for r in audit["cases"] if r["id"] == row["id"])
            require(
                record["status"] == "ok"
                and record["native_frames"] == n
                and record["masks"] == 20
                and record["neural_bindings"] == (20 if ai else 0)
                and record["neural_replays"] == (2 if ai else 0),
                "Incomplete source audit population",
            )
            totals["native_frames"] += n
            totals["exact_masks"] += 20
            totals["common_pcm_samples"] += n * 640
            totals["neural_input_bindings"] += 20 if ai else 0
            totals["sampled_neural_output_replays"] += 2 if ai else 0
    require(
        audit["planned"] == audit["passed"] == 160 and all(audit[k] == v for k, v in totals.items()),
        "Aggregate audit counts differ",
    )
    return {
        "passed": True,
        "cases": 160,
        "source_groups": 16,
        **totals,
        "scope": "Published metadata consistency only; not source pixels, actual neural computation, label truth, performance or outside participation.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capsule", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = review(json.loads(gzip.decompress(args.capsule.read_bytes())))
    if args.output:
        with args.output.open("x") as stream:
            json.dump(result, stream, indent=2)
            stream.write("\n")
    print(json.dumps(result))
