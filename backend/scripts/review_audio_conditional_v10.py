"""Audit all v10 decisions, independent guards and a fitting-only native refit.

Shares the published JSON predictor and declared metric/bootstrap implementation;
separately reconstructs guards/stability and checks the original service cache.
No new neural forwards or confirmation cases are used here.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from scripts import audio_conditional_v10 as method
from scripts import run_audio_conditional_v10 as execution
from scripts.audio_guard_v7 import initial
from scripts.audio_source_identity_v9 import digest
from scripts.evaluate_audio_multisource_v8 import summarize
from scripts.prepare_audio_guard_v7 import write
from scripts.reproduce_audio_confirmation_v7 import same_tree
from scripts.review_audio_guard_v7 import close, metric
from scripts.review_audio_identity_numerics_v9 import decisions
from scripts.run_audio_guard_v7 import load_rows


def guards(rows, predictions):
    if len(rows) != len(predictions) or any(r["role"] != "cal" for r in rows):
        raise ValueError("Independent calibration rows and predictions required")
    pools = {0: defaultdict(list), 1: defaultdict(list)}
    for row, p in zip(rows, predictions, strict=True):
        pools[row["label"]][row["group"]].append(None if p is None else p if row["label"] == 0 else 1 - p)
    return {
        str(c): [
            {
                "group": g,
                "maximum": None if any(v is None for v in values) else max(values),
                "rows": len(values),
            }
            for g, values in sorted(pools[c].items())
        ]
        for c in (0, 1)
    }


def review(lock_path, head_path, response_path, summary_path, output, refit):
    lock, (_, reference, all_rows, baseline, transform) = execution.check(lock_path)
    head = json.loads(head_path.read_text())
    fit_record = json.loads((head_path.parent / "fit.json").read_text())
    header, responses = load_rows(response_path)
    summary = json.loads(summary_path.read_text())
    if (
        head["lock_sha256"] != digest(lock_path)
        or head["fit_sha256"] != digest(head_path.parent / "fit.json")
        or fit_record["fitted"] != head["fitted"]
        or fit_record["lock_sha256"] != digest(lock_path)
        or header["lock_sha256"] != digest(lock_path)
        or header["head_sha256"] != digest(head_path)
        or summary["responses_sha256"] != digest(response_path)
        or summary["head_sha256"] != digest(head_path)
        or summary["lock_sha256"] != digest(lock_path)
    ):
        raise ValueError("Original fitting, calibration and complete response/summary lineage required")
    rows = [r for r in all_rows if r["role"] == "development"]
    calibration = [r for r in all_rows if r["role"] == "cal"]
    if [r["id"] for r in responses] != [r["id"] for r in rows]:
        raise ValueError("Every development outcome in its original order required")
    mapped = {a: method.predict_rows(rows, head["fitted"], a) for a in method.ARMS}
    for arm in method.ARMS:
        expected = guards(calibration, method.predict_rows(calibration, head["fitted"], arm))
        if not same_tree(expected, head["guards"][arm]):
            raise ValueError("Independent class/group calibration ranks differ")
    computed = []
    for i, (expected, row) in enumerate(zip(rows, responses, strict=True)):
        if any(
            expected[k] != row[k]
            for k in (
                "id",
                "base_id",
                "group",
                "role",
                "label",
                "generator",
                "dataset",
                "condition",
                "sha256",
                "pcm_sha256",
                "background_id",
            )
        ):
            raise ValueError("Source/truth/group/clock binding changed")
        if row["status"] == "failed":
            if (
                row["original_p"] is not None
                or row["original_decision"] != "inconclusive"
                or any(
                    v["p"] is not None or v["decision"] != "inconclusive" for v in row["candidates"].values()
                )
            ):
                raise ValueError("A failure cannot establish a directional verdict")
            continue
        old = baseline[row["id"]]
        close(row["original_p"], old["original_p"])
        if row["original_decision"] != old["original_decision"] or row["status"] != expected["status"]:
            raise ValueError("Actual current-service verdict or assessment status changed")
        needed = any(initial(mapped[a][i], head["guards"][a]) == "original" for a in method.ARMS)
        probe = row["probe"]
        if probe["status"] not in (("computed", "failed") if needed else ("not_required",)):
            raise ValueError("Missing or unexpected original-stability probe")
        if probe["status"] == "computed":
            execution.original.validate_feature(
                {**probe, "status": "ok", "windows": [{"start_sample": 0, "end_sample": 64000}]}, reference
            )
            computed.append(probe)
        for arm in method.ARMS:
            p = mapped[arm][i]
            close(row["candidates"][arm]["p"], p)
            band_p = (
                method.predict_rows([probe], head["fitted"], arm)[0]
                if probe["status"] == "computed"
                else None
            )
            close(probe["p"][arm], band_p)
            detail = decisions(p, band_p, expected["raw"], probe["raw"], head["guards"][arm])
            if (
                not same_tree(detail, row["candidates"][arm]["detail"])
                or detail["decision"] != row["candidates"][arm]["decision"]
            ):
                raise ValueError("Independent dual-stability conclusion/explanation differs")
    parity = None
    if refit:
        fitted, parity = method.fit(
            [r for r in all_rows if r["role"] == "fit"], transform, parity_rows=all_rows + computed
        )
        if not same_tree(fitted, head["fitted"]):
            raise ValueError("Fitting-only native CPU refit differs from the original model")
    reconstructed = {}
    for arm in method.ARMS:
        flat = [
            {
                **r,
                "candidate_p": r["candidates"][arm]["p"],
                "candidate_decision": r["candidates"][arm]["decision"],
            }
            for r in responses
        ]
        value = summarize(flat, 161)
        if not same_tree(value, summary["arms"][arm]):
            raise ValueError("Complete statistical summary or frozen gates differ")
        for condition in ("all", *execution.original.CONDITIONS):
            panel = [r for r in flat if condition == "all" or r["condition"] == condition]
            for key, decision_key, name in (
                ("original_p", "original_decision", "service"),
                ("candidate_p", "candidate_decision", "candidate"),
            ):
                for field, count in metric(panel, key, decision_key).items():
                    actual = value["groups"][condition][name][field]
                    if field in ("brier", "log_loss"):
                        close(count, actual)
                    elif count != actual:
                        raise ValueError("Separate complete-denominator operational counts differ")
        reconstructed[arm] = value
    selected = method.select(reconstructed)
    if selected != summary["selected_arm_for_possible_confirmation"]:
        raise ValueError("Predeclared arm selection differs")
    result = {
        "schema": "chipmunk.audio-conditional-numerical-review/10",
        "passed": True,
        "lock_sha256": digest(lock_path),
        "head_sha256": digest(head_path),
        "responses_sha256": digest(response_path),
        "summary_sha256": digest(summary_path),
        "review_code_sha256": digest(Path(__file__)),
        "refit_performed": refit,
        "native_json_parity": parity,
        "inputs": 4248,
        "outcomes": 1449,
        "computed_probes": len(computed),
        "selected_arm_for_possible_confirmation": selected,
        "scope": "Separate cached-service/guard/dual-stability/count reconstruction. Shared published JSON predictor and statistical gates; optional fitting-only CPU refit/native parity on all originals and computed probes. No new neural forward or outside human review.",
    }
    write(output, result)
    print(
        json.dumps(
            {
                **result,
                "native_json_parity": {
                    a: {k: v[k] for k in ("rows", "max_absolute_error", "trees")} for a, v in parity.items()
                }
                if parity is not None
                else None,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("lock", "head", "responses", "summary", "output"):
        p.add_argument(name, type=Path)
    p.add_argument("--refit", action="store_true")
    a = p.parse_args()
    review(a.lock, a.head, a.responses, a.summary, a.output, a.refit)
