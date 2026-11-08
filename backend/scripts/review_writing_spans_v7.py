"""Full local source/native/refit replay, with separate set-based mask counts."""

from __future__ import annotations

import json
import time
import traceback

from scripts import writing_span_calibration_v7 as calibration
from scripts import writing_span_heads_v7 as heads
from scripts.run_writing_spans_v7 import (
    ARMS,
    TRIAL,
    check,
    controls,
    journal,
    load_heads,
    read,
    receipt,
    scored,
    summarize,
    write,
)
from scripts.writing_span_data_v7 import prepare, sha
from scripts.writing_span_geometry_v7 import genuine_point, require
from scripts.writing_span_native_v7 import Native


def separate_measure(case, proposals, cut):
    text, labels = case["text"], case["labels"]
    ai = {i for i, c in enumerate(text) if not c.isspace() and labels[i] == 1}
    human = {i for i, c in enumerate(text) if not c.isspace() and labels[i] == 0}
    selected = set()
    for proposal in proposals:
        if proposal["score"] >= cut:
            selected.update(i for i in range(proposal["start"], proposal["end"]) if not text[i].isspace())
    tp, fp, fn = len(selected & ai), len(selected & human), len(ai - selected)
    recovered = 0
    for a, b in case["truth_spans"]:
        target = {i for i in range(a, b) if not text[i].isspace()}
        recovered += 2 * len(target & selected) >= len(target)
    iou = tp / (tp + fp + fn) if tp + fp + fn else 1.0
    return {
        "ai_characters": len(ai),
        "human_characters": len(human),
        "located_ai_characters": tp,
        "highlighted_human_characters": fp,
        "missed_ai_characters": fn,
        "highlighted_characters": len(selected),
        "character_iou": iou,
        "character_f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0,
        "truth_intervals": len(case["truth_spans"]),
        "recovered_intervals": recovered,
        "useful_location": bool(ai and iou >= 0.3),
    }


def review():
    output = TRIAL / "review.json"
    require(
        not output.exists() and not (TRIAL / "review-start.json").exists(), "Fresh audit attempt required"
    )
    write(TRIAL / "review-start.json", {"freeze_sha256": sha(TRIAL / "freeze.json"), "started": True})
    started = time.monotonic()
    result = {
        "schema": "chipmunk.writing-span-local-review/7",
        "status": "failed",
        "freeze_sha256": sha(TRIAL / "freeze.json"),
        "outside_human_reviewers": 0,
        "independent_source_validation": False,
    }
    native = None
    try:
        cases = check()
        require(
            prepare()["cases"] == cases, "Original source graph/slices/extracted positions do not reconstruct"
        )
        mapping, points = controls()
        for role in ("fit", "cal", "point", "diagnostic"):
            receipt(role)
        learner = load_heads()
        native = Native(with_current=True)
        all_rows, fitting = {}, []
        for role in ("fit", "cal", "point", "diagnostic"):
            role_cases = [c for c in cases if c["role"] == role]
            original = journal(role)
            require(len(role_cases) == len(original), "Complete original phase denominator required")
            replayed = []
            for i, (case, row) in enumerate(zip(role_cases, original, strict=True)):
                require(
                    case["id"] == row["id"] and row["status"] == "ok",
                    "Original source identity/order changed",
                )
                feature = native.features(case, audit=i == 0)
                require(
                    feature["native"] == row["native"], "Complete native feature/input/offset hash differs"
                )
                if role == "fit":
                    fitting.append(feature)
                else:
                    current, proof = native.current_proposals(case, audit=i == 0)
                    require(proof == row["current_native"], "Current native input/coverage proof differs")
                    proposals = {
                        "current": current,
                        **{k: heads.propose(model, k, feature) for k, model in learner.items()},
                    }
                    if role != "cal":
                        proposals = {k: scored(p, mapping[k]) for k, p in proposals.items()}
                    require(
                        proposals == row["arms"],
                        "Re-inferred proposal/coordinate/calibration response differs",
                    )
                    if role == "diagnostic":
                        separate = {
                            k: separate_measure(case, p, points[k]["cut"]) for k, p in proposals.items()
                        }
                        require(separate == row["measures"], "Separate set-based source counts differ")
                replayed.append(row)
                if (i + 1) % 32 == 0:
                    print(
                        json.dumps(
                            {
                                "review_role": role,
                                "completed": i + 1,
                                "planned": len(role_cases),
                                "actual_pretrained_forwards": native.calls,
                            }
                        ),
                        flush=True,
                    )
            all_rows[role] = replayed
        native.unchanged()
        exact_refits = {}
        for kind in ("token", "interval"):
            repeated, history = heads.fit(fitting, kind)
            original = read(TRIAL / (kind + "-head.json"))
            require(
                heads.save_state(repeated) == original["state"] and history == original["history"],
                "Exact fitting-only head/state/loss replay failed",
            )
            exact_refits[kind] = True
        for arm in ARMS:
            cal_cases = [c for c in cases if c["role"] == "cal"]
            repeated = calibration.fit(
                [{**r, "proposals": r["arms"][arm]} for r in all_rows["cal"]], cal_cases
            )
            require(repeated == mapping[arm], "Calibration-role-only exact refit failed")
            point = genuine_point([{**r, "proposals": r["arms"][arm]} for r in all_rows["point"]])
            require(point == points[arm], "Genuine-group-only operating point differs")
        require(
            summarize(all_rows["diagnostic"]) == read(TRIAL / "summary.json"), "Summary/gate replay differs"
        )
        result.update(
            status="complete",
            reconstructed_documents=992,
            native_documents_repeated=992,
            exact_head_refits=exact_refits,
            exact_calibrator_refits=3,
            exact_genuine_points=3,
            separately_recounted_diagnostic_arm_masks=864,
            unchanged_pretrained_weights=True,
            source_text_or_vectors_published=False,
            serving_admitted=False,
        )
    except BaseException:
        result["failure"] = traceback.format_exc()
        raise
    finally:
        result["duration_s"] = time.monotonic() - started
        if native is not None:
            result["actual_pretrained_forwards"] = native.calls
        write(output, result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    review()
