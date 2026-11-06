"""Standard-library retained-window, source-pair and fitting-target review for V9.

Model forwards, omitted windows, masks/spatial IoU and generalization are
outside this arithmetic review. Local execution is not outside validation.
"""

from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
import struct
import tempfile
from pathlib import Path

if __package__:
    from .review_mouth_learning_v8 import review as previous_review
else:
    from review_mouth_learning_v8 import review as previous_review


def unpack(path):
    return json.loads(gzip.decompress(path.read_bytes()))


def review(root):
    panels = root / "evaluation-panels"
    summary = root / "mouth-paired-v9-evaluation.json"
    result = json.loads(summary.read_text())
    head_path = panels / "mouth-v9-head.json"
    head = json.loads(head_path.read_text())
    responses_path = panels / "mouth-v9-review-responses.json.gz"
    responses = unpack(responses_path)
    prior = unpack(panels / "mouth-v8-review-responses.json.gz")
    if (
        responses["header"]["parent_scores_sha256"] != prior["original_receipt_sha256"]
        or result["parent_scores_sha256"] != prior["original_receipt_sha256"]
    ):
        raise ValueError("V9 comparison refers to a different prior candidate")
    old = previous_review(
        panels / "mouth-v8-review-responses.json.gz",
        root / "mouth-learning-v8-evaluation.json",
        panels / "mouth-v8-head.json",
    )
    # The unchanged shared reviewer names the production cue 'baseline'. V9
    # names V8 'baseline' and keeps the production cue as a third 'current' arm.
    # Project those names without altering any outcome or model/receipt hash.
    projected = copy.deepcopy(result)
    projected["baseline"] = projected["current"]
    projected["cases"]["baseline"] = projected["cases"]["current"]
    with tempfile.TemporaryDirectory(prefix="mouth-v9-review-") as directory:
        path = Path(directory) / "current-versus-v9.json"
        path.write_text(json.dumps(projected))
        new = previous_review(responses_path, path, head_path)
    training = unpack(panels / "mouth-v8-training-bindings.json.gz")
    prefixes = unpack(panels / "mouth-v8-prefix-bindings.json.gz")
    fits = unpack(panels / "mouth-v9-fit-replay.json.gz")
    rows = {r["id"]: r for r in training["items"]}
    bindings = {r["id"]: r for r in prefixes["bindings"]}
    if (
        fits["head_sha256"] != hashlib.sha256(head_path.read_bytes()).hexdigest()
        or fits["cache_receipt_sha256"] != prefixes["original_receipt_sha256"]
    ):
        raise ValueError("Published fitting replay uses different model/cache identities")
    expected_pairs = []
    lookup = {(r["group"], r["frame_index"], r["condition"]): r for r in rows.values()}
    for positive in rows.values():
        if positive["role"] == "fit" and positive["condition"] in ("ai-redraw", "ai-retime"):
            for condition in ("original", "local-light", "local-resample"):
                negative = lookup[(positive["group"], positive["frame_index"], condition)]
                if (
                    negative["role"] != "fit"
                    or positive["label"] != 1
                    or negative["label"] != 0
                    or any(positive[k] != negative[k] for k in ("source_sha256", "original_rgb_sha256"))
                ):
                    raise ValueError("Source-matched fitting pair changed native identity or role")
                expected_pairs.append({"positive_id": positive["id"], "negative_id": negative["id"]})
    if (
        len(expected_pairs) != 768
        or fits["pairs"] != expected_pairs
        or head["learning"]["pairs"] != expected_pairs
    ):
        raise ValueError("Fitting pairs differ or omit a planned condition")
    target_count = fits["target_count"]
    targets = [0.0] * target_count
    mask = bytearray(target_count)
    expected_offsets = {}
    for row in rows.values():
        if row["role"] == "fit" and row["condition"] not in ("ai-redraw", "ai-retime"):
            for binding in bindings[row["id"]]["views"]:
                expected_offsets[binding["offset"]] = row["id"], binding["view"]
    seen = set()
    for target in fits["teacher_fit_targets"]:
        k = target["offset"]
        if k not in expected_offsets or expected_offsets[k] != (target["id"], target["view"]) or k in seen:
            raise ValueError("Teacher target includes a reserved/unknown/repeated source input")
        targets[k] = target["target"]
        mask[k] = 1
        seen.add(k)
    if len(seen) != 4096 or seen != set(expected_offsets):
        raise ValueError("Missing fitting-only teacher target")
    target_sha = hashlib.sha256(struct.pack(f"<{target_count}f", *targets)).hexdigest()
    mask_sha = hashlib.sha256(mask).hexdigest()
    if (
        target_sha != fits["all_target_f32_sha256"]
        or target_sha != head["learning"]["teacher_target_f32_sha256"]
        or mask_sha != fits["eligible_bool_sha256"]
        or mask_sha != head["learning"]["teacher_eligible_mask_sha256"]
    ):
        raise ValueError("Teacher target or fitting-only eligibility payload changed")
    if old["counts"]["probe"] != {k: result["baseline"][k] for k in old["counts"]["probe"]}:
        raise ValueError("Prior candidate denominators/outcomes changed in V9 summary")
    return {
        "schema": "chipmunk.mouth-paired-public-arithmetic-review/9",
        "passed": True,
        "clips_three_arms": 88,
        "retained_candidate_windows_checked": old["retained_candidate_windows_checked"]
        + new["retained_candidate_windows_checked"],
        "source_matched_fit_pairs_checked": len(expected_pairs),
        "fitting_teacher_targets_checked": len(seen),
        "counts": {
            "current": new["counts"]["base"],
            "v8": old["counts"]["probe"],
            "v9": new["counts"]["probe"],
        },
        "scope": "Retained-window arithmetic/times/flags/denominators, source-role pair correspondence and teacher float32/eligibility payload hashes. Does not verify neural inference, omitted windows, native masks/spatial IoU or generalization; not outside review.",
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("docs", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    result = review(a.docs)
    with a.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result))
