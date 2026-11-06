"""Publish/reproduce v10 using the already public v9 vectors and v10 quality.

No private manifest, source audio or pretrained download is needed for numerical
review. Original fit/head bytes are preserved; outcome/summary identifiers are
sanitized, so their original digests are lineage commitments, not byte replay.
"""

from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
from pathlib import Path

from scripts import audio_conditional_v10 as method
from scripts import export_audio_conditional_v10 as prospective
from scripts import publish_audio_identity_results_v9 as original
from scripts import run_audio_conditional_v10 as execution
from scripts.audio_guard_v7 import initial
from scripts.audit_audio_identity_forwards_v9 import select
from scripts.evaluate_audio_multisource_v8 import summarize
from scripts.prepare_audio_guard_v7 import write
from scripts.reproduce_audio_confirmation_v7 import same_tree
from scripts.review_audio_conditional_v10 import guards
from scripts.review_audio_guard_v7 import close, metric
from scripts.review_audio_identity_numerics_v9 import decisions
from scripts.run_audio_guard_v7 import load_rows

REFERENCES = {
    "audio-v9-identity-plan.json.gz": prospective.SOURCE_SHA,
    "audio-v9-development-review.json.gz": "0546940a656ce0f718318975491ee041f2a429be15ac57d316e8e5e3ea6345c3",
    "audio-v9-native-development-plan.json": "48795283220152d5de07356ff54c018b8da797749957b6480d27582de97eed1e",
    "audio-v10-conditional-freeze.json": "bc5711f4b2f3fdbc68509d7f89c7e7647179fcbe5056e9fa817352b7e2b977a8",
    "audio-v10-quality-inputs.jsonl.gz": "1b9f5cc344a607831a4c610be0a837e04adedbbe3c8ddb5d535db30ac3be2711",
}
LINEAGE = {
    "lock_sha256": "263419ee9743939beaf767b519ed64ea935bf6768312ffd927caa47cebbfb7c9",
    "head_sha256": "b925102ba4931943789d041f807ef80a89402b86f7e98f1aff6f210d37b83ecf",
    "fit_sha256": "3c92116d7abcb4114a341c6fcda61b788f0eb260a9acec43c712931fdf1c32c7",
    "responses_sha256": "a2a83749942f71df0ea923534537fae28a4b2e372a7a101b591e0097f262c06d",
    "summary_sha256": "38c8cf95964f2fe5ca1812b6e1e385774934f9fa8c6c3ea141bbb41c0a2d7608",
}
RESPONSE_FIELDS = {"status", "error_type", "original_p", "original_decision", "candidates", "probe"}
PROBE_FIELDS = {
    "status",
    "raw",
    "embedding",
    "quality",
    "p",
    "embedding_sha256",
    "reused_v9_probe",
    "filtered_pcm_sha256",
    "error_type",
}
NATIVE_FIELDS = {
    "schema",
    "lock_sha256",
    "head_sha256",
    "responses_sha256",
    "plan_sha256",
    "audit_code_sha256",
    "passed",
    "predetermined_cells",
    "new_probe_cases",
    "unique_cases",
    "cases",
    "scope",
}


def exact_artifact(value, expected):
    encoded = (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()
    if hashlib.sha256(encoded).hexdigest() != expected:
        raise ValueError("Exact original fitted artifact bytes required")


def inputs(folder, project=None):
    if any(prospective.digest(folder / name) != sha for name, sha in REFERENCES.items()):
        raise ValueError("Exact original public references required")
    source = original.load(folder / "audio-v9-identity-plan.json.gz")
    prior = original.load(folder / "audio-v9-development-review.json.gz")
    freeze = original.load(folder / "audio-v10-conditional-freeze.json")
    quality_raw = gzip.decompress((folder / "audio-v10-quality-inputs.jsonl.gz").read_bytes())
    prospective.review(freeze, quality_raw, source, project)
    quality = [json.loads(line) for line in quality_raw.splitlines()][1:]
    variants = source["variants"]["development"]
    features = prior["features"]
    if [r["id"] for r in features] != [r["id"] for r in variants]:
        raise ValueError("Original complete representation matrix required")
    rows = []
    for variant, feature, measured in zip(variants, features, quality, strict=True):
        if any(feature[k] != variant[k] for k in ("id", "role", "sha256", "pcm_sha256")):
            raise ValueError("Original source/feature binding differs")
        rows.append(
            {
                **original.metadata(variant),
                **feature,
                "quality": measured["quality"],
                "cached_probe_quality": measured["cached_probe_quality"],
            }
        )
    return source, prior, freeze, rows


def review_native(audit, rows, outcomes, folder, project):
    plan = original.load(folder / "audio-v9-native-development-plan.json")
    new = [r["id"] for r in outcomes if r["probe"].get("reused_v9_probe") is False]
    ids = sorted(set(plan["ids"]) | set(new))
    if (
        set(audit) != NATIVE_FIELDS
        or plan["ids"] != select(rows)
        or [r["id"] for r in audit["cases"]] != ids
        or audit["predetermined_cells"] != 99
        or audit["new_probe_cases"] != 22
        or len(new) != 22
        or audit["unique_cases"] != len(ids)
        or audit["passed"] is not True
        or audit["plan_sha256"] != REFERENCES["audio-v9-native-development-plan.json"]
        or any(audit[k] != LINEAGE[k] for k in ("lock_sha256", "head_sha256", "responses_sha256"))
    ):
        raise ValueError("All predetermined native cells and every new probe required")
    by_id = {r["id"]: r for r in rows}
    response = {r["id"]: r for r in outcomes}
    for case in audit["cases"]:
        expected, probe = by_id[case["id"]], response[case["id"]]["probe"]
        if (
            any(case[k] != expected[k] for k in ("sha256", "pcm_sha256"))
            or case["predetermined_cell"] != (case["id"] in plan["ids"])
            or case["new_probe_case"] != (case["id"] in new)
            or case["probe_status"] != probe["status"]
            or any(case[k] is not True for k in ("passed", "original_passed", "quality_passed"))
            or (probe["status"] == "computed" and case["probe_passed"] is not True)
        ):
            raise ValueError("Native case source/PCM/quality/probe receipt differs")
    if (
        project is not None
        and prospective.digest(project / "backend/scripts/audit_audio_conditional_v10.py")
        != audit["audit_code_sha256"]
    ):
        raise ValueError("Native audit implementation bytes changed")


def comparable(value):
    return {a: original.comparable(s) for a, s in value.items()}


def review(body, folder, *, refit=False, project=None):
    original.privacy_check(body)
    if (
        body["schema"] != "chipmunk.audio-conditional-public-development/10"
        or body["references"] != REFERENCES
        or body["original_lineage"] != LINEAGE
        or body["deployment_eligible"] is not False
    ):
        raise ValueError("Original public schema/reference/lineage commitments required")
    source, prior, freeze, all_rows = inputs(folder, project)
    head, fit_record = body["head"], body["fit_record"]
    exact_artifact(head, LINEAGE["head_sha256"])
    exact_artifact(fit_record, LINEAGE["fit_sha256"])
    if (
        head["fit_sha256"] != LINEAGE["fit_sha256"]
        or head["lock_sha256"] != LINEAGE["lock_sha256"]
        or fit_record["lock_sha256"] != LINEAGE["lock_sha256"]
        or not same_tree(fit_record["fitted"], head["fitted"])
        or head["deployment_eligible"] is not False
        or fit_record["deployment_eligible"] is not False
    ):
        raise ValueError("Original fitting-before-calibration lineage required")
    rows = [r for r in all_rows if r["role"] == "development"]
    expected_variants = [r for r in source["variants"]["development"] if r["role"] == "development"]
    outcomes = body["outcomes"]
    if [r["id"] for r in outcomes] != [r["id"] for r in rows]:
        raise ValueError("Every original development outcome in order required")
    header = body["response_header"]
    if header != {
        "schema": "chipmunk.audio-conditional-development-responses/10",
        "lock_sha256": LINEAGE["lock_sha256"],
        "head_sha256": LINEAGE["head_sha256"],
        "original_responses_sha256": execution.RESPONSES_SHA,
        "planned": 1449,
    }:
        raise ValueError("Original response header required")
    cal = [r for r in all_rows if r["role"] == "cal"]
    for arm in method.ARMS:
        if not same_tree(guards(cal, method.predict_rows(cal, head["fitted"], arm)), head["guards"][arm]):
            raise ValueError("Independent group/class calibration differs")
    mapped = {a: method.predict_rows(rows, head["fitted"], a) for a in method.ARMS}
    baseline = {r["id"]: r for r in prior["outcomes"]}
    computed = []
    for i, (expected, variant, row) in enumerate(zip(rows, expected_variants, outcomes, strict=True)):
        metadata = original.metadata(variant)
        if set(row) - set(metadata) - RESPONSE_FIELDS or any(row[k] != v for k, v in metadata.items()):
            raise ValueError("Sanitized outcome source/role/truth/group differs")
        if row["status"] != expected["status"]:
            raise ValueError("Assessment status differs from complete original inputs")
        old = baseline[row["id"]]
        close(row["original_p"], old["original_p"])
        if row["original_decision"] != old["original_decision"]:
            raise ValueError("Actual current-service verdict changed")
        needed = any(initial(mapped[a][i], head["guards"][a]) == "original" for a in method.ARMS)
        probe = row["probe"]
        if set(probe) - PROBE_FIELDS or probe["status"] != ("computed" if needed else "not_required"):
            raise ValueError("Every required stability probe and exact field allowlist required")
        if probe["status"] == "computed":
            execution.original.validate_feature(
                {**probe, "status": "ok", "windows": [{"start_sample": 0, "end_sample": 64000}]},
                prior["representation_header"],
            )
            if probe["reused_v9_probe"] and (
                old["probe"]["status"] != "computed"
                or any(
                    probe[k] != old["probe"][k]
                    for k in ("raw", "embedding", "embedding_sha256", "filtered_pcm_sha256")
                )
                or probe["quality"] != expected["cached_probe_quality"]
            ):
                raise ValueError("Cached probe/filtered quality differs from original public reference")
            computed.append(probe)
        elif any(probe[k] is not None for k in ("raw", "embedding", "quality")):
            raise ValueError("An uncomputed probe cannot carry measured values")
        if set(row["candidates"]) != set(method.ARMS) or set(probe["p"]) != set(method.ARMS):
            raise ValueError("Every eligible and nuisance arm required")
        for arm in method.ARMS:
            p = mapped[arm][i]
            close(row["candidates"][arm]["p"], p)
            band_p = method.predict_rows([probe], head["fitted"], arm)[0] if needed else None
            close(probe["p"][arm], band_p)
            detail = decisions(p, band_p, expected["raw"], probe["raw"], head["guards"][arm])
            if (
                not same_tree(detail, row["candidates"][arm]["detail"])
                or detail["decision"] != row["candidates"][arm]["decision"]
            ):
                raise ValueError("Independent calibrated/raw stability decision differs")
    review_native(body["native_audit"], all_rows, outcomes, folder, project)
    numerical = body["numerical_audit"]
    if (
        numerical["passed"] is not True
        or numerical["refit_performed"] is not True
        or any(
            numerical[k] != LINEAGE[k]
            for k in ("lock_sha256", "head_sha256", "responses_sha256", "summary_sha256")
        )
        or numerical["inputs"] != 4248
        or numerical["outcomes"] != 1449
        or numerical["computed_probes"] != len(computed)
        or numerical["selected_arm_for_possible_confirmation"] is not None
        or any(
            p["rows"] != 4516 or p["max_absolute_error"] != 0.0
            for p in numerical["native_json_parity"].values()
        )
    ):
        raise ValueError("Complete original numerical/refit/parity receipt required")
    if (
        project is not None
        and prospective.digest(project / "backend/scripts/review_audio_conditional_v10.py")
        != numerical["review_code_sha256"]
    ):
        raise ValueError("Numerical audit implementation bytes changed")
    parity = None
    if refit:
        fitted, parity = method.fit(
            [r for r in all_rows if r["role"] == "fit"],
            prior["head"]["fitted"]["transform"],
            parity_rows=all_rows + computed,
        )
        if not same_tree(fitted, head["fitted"]):
            raise ValueError("Fitting-only native CPU refit differs")
    summaries = {}
    for arm in method.ARMS:
        flat = [
            {
                **r,
                "candidate_p": r["candidates"][arm]["p"],
                "candidate_decision": r["candidates"][arm]["decision"],
            }
            for r in outcomes
        ]
        summary = summarize(flat, 161)
        for condition in ("all", *execution.original.CONDITIONS):
            panel = [r for r in flat if condition == "all" or r["condition"] == condition]
            for key, decision, name in (
                ("original_p", "original_decision", "service"),
                ("candidate_p", "candidate_decision", "candidate"),
            ):
                for field, count in metric(panel, key, decision).items():
                    actual = summary["groups"][condition][name][field]
                    if field in ("brier", "log_loss"):
                        close(count, actual)
                    elif count != actual:
                        raise ValueError("Complete-denominator operational counts differ")
        summaries[arm] = summary
    if not same_tree(comparable(summaries), comparable(body["summaries"])):
        raise ValueError("Complete summary, strata, uncertainty or gates differ")
    selected = method.select(summaries)
    if selected != body["selected_arm_for_possible_confirmation"]:
        raise ValueError("Predeclared arm selection differs")
    return {
        "schema": "chipmunk.audio-conditional-public-result-review/10",
        "passed": True,
        "inputs": len(all_rows),
        "outcomes": len(outcomes),
        "computed_probes": len(computed),
        "original_fit_head_reproduced": True,
        "refit_performed": refit,
        "native_json_parity": parity,
        "gates": {a: s["gates"] for a, s in summaries.items()},
        "selected_arm_for_possible_confirmation": selected,
        "serving_changed": False,
        "outside_reviewer_participation": False,
        "native_reexecution_performed_by_this_review": False,
        "scope": "Public-input numerical reconstruction and optional fitting-only CPU refit; original native receipts checked without re-running audio. No media, private manifest, pretrained download, fresh confirmation or outside human validation.",
    }


def export(root, destination, folder):
    catalogue, _, _, rows = inputs(folder, Path(__file__).resolve().parents[2])
    header, responses = load_rows(root / "responses.jsonl")
    expected = [r for r in rows if r["role"] == "development"]
    variants = [r for r in catalogue["variants"]["development"] if r["role"] == "development"]
    outcomes = []
    for row, source, variant in zip(responses, expected, variants, strict=True):
        if any(
            row[k] != source[k]
            for k in (
                "id",
                "group",
                "role",
                "label",
                "generator",
                "dataset",
                "condition",
                "sha256",
                "pcm_sha256",
            )
        ):
            raise ValueError("Private outcome source binding differs")
        public = original.metadata(variant)
        public.update(original.pick(row, RESPONSE_FIELDS))
        public["probe"] = original.pick(row["probe"], PROBE_FIELDS)
        outcomes.append(public)
    for name, file in (
        ("lock_sha256", "lock.json"),
        ("head_sha256", "trained/head.json"),
        ("fit_sha256", "trained/fit.json"),
        ("responses_sha256", "responses.jsonl"),
        ("summary_sha256", "summary.json"),
    ):
        if prospective.digest(root / file) != LINEAGE[name]:
            raise ValueError("Original completed experiment artifact changed")
    summary = original.load(root / "summary.json")
    _, private_responses = load_rows(root / "responses.jsonl")
    mapping_rows = [
        {**r, "background_sha256": s["background_sha256"]}
        for r, s in zip(private_responses, expected, strict=True)
    ]
    body = {
        "schema": "chipmunk.audio-conditional-public-development/10",
        "references": REFERENCES,
        "original_lineage": LINEAGE,
        "fit_record": original.load(root / "trained/fit.json"),
        "head": original.load(root / "trained/head.json"),
        "response_header": header,
        "outcomes": outcomes,
        "summaries": {a: original.public_summary(s, mapping_rows) for a, s in summary["arms"].items()},
        "native_audit": original.load(root / "native-review.json"),
        "numerical_audit": original.load(root / "numerical-review.json"),
        "selected_arm_for_possible_confirmation": summary["selected_arm_for_possible_confirmation"],
        "deployment_eligible": False,
        "scope": "Complete consumed-source development; vectors/quality referenced rather than duplicated. Original fit/head bytes retained; sanitized outcome and summary bytes differ from original lineage artifacts. No source audio, customer files, transcripts, raw provider identifiers, private paths or full pretrained checkpoints. No fresh result or outside reviewer participation.",
        "data_terms": original.load(folder / "audio-v9-identity-plan.json.gz")["data_terms"],
    }
    result = review(body, folder, project=Path(__file__).resolve().parents[2])
    encoded = gzip.compress(
        (json.dumps(body, separators=(",", ":"), allow_nan=False) + "\n").encode(), mtime=0
    )
    with destination.open("xb") as stream:
        stream.write(encoded)
    print(
        json.dumps(
            {
                "sha256": prospective.digest(destination),
                "bytes": len(encoded),
                "outcomes": len(outcomes),
                "passed": result["passed"],
            }
        ),
        flush=True,
    )


def tamper(body, folder):
    assessed = next(i for i, r in enumerate(body["outcomes"]) if r["status"] == "ok")
    unassessed = next(i for i, r in enumerate(body["outcomes"]) if r["status"] == "not_applicable")
    probe = next(i for i, r in enumerate(body["outcomes"]) if r["probe"].get("reused_v9_probe") is True)
    names = (
        "changed fitted tree",
        "changed source group",
        "changed candidate probability",
        "unassessed called original",
        "changed cached probe quality",
        "changed complete denominator",
        "dropped native case",
        "hidden local path",
    )
    cases = []
    for name in names:
        altered = copy.deepcopy(body)
        if name == names[0]:
            altered["head"]["fitted"]["arms"]["nonlinear"]["trees"][0]["value"][0] += 0.1
        elif name == names[1]:
            altered["outcomes"][0]["group"] = "altered"
        elif name == names[2]:
            altered["outcomes"][assessed]["candidates"]["nonlinear"]["p"] = None
        elif name == names[3]:
            altered["outcomes"][unassessed]["candidates"]["nonlinear"]["decision"] = "original"
        elif name == names[4]:
            altered["outcomes"][probe]["probe"]["quality"]["flatness"] += 0.2
        elif name == names[5]:
            altered["summaries"]["nonlinear"]["complete_inputs"] -= 1
        elif name == names[6]:
            altered["native_audit"]["cases"].pop()
        else:
            altered["notes"] = "/Users/placeholder/private-input.wav"
        try:
            review(altered, folder)
        except ValueError as error:
            cases.append({"alteration": name, "rejected": True, "reason": str(error)})
        else:
            raise ValueError("Altered actual capsule incorrectly passed: " + name)
    return {
        "schema": "chipmunk.audio-conditional-public-result-tamper/10",
        "passed": True,
        "cases": cases,
        "scope": "Local deliberate actual-capsule alterations, not outside review or neural inference.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("export", "review", "tamper", "diagnose"))
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--references", type=Path)
    parser.add_argument("--project", type=Path)
    parser.add_argument("--refit", action="store_true")
    args = parser.parse_args()
    folder = args.references or Path("docs/evaluation-panels")
    if args.stage == "export":
        export(args.input, args.output, folder)
    else:
        body = original.load(args.input)
        if args.stage == "tamper":
            result = tamper(body, folder)
        elif args.stage == "diagnose":
            review(body, folder, project=args.project)
            result = {
                "schema": "chipmunk.audio-conditional-guard-only-bound/10",
                "arms": {
                    a: original.guard_only_ceiling(
                        [
                            {
                                **r,
                                "candidate_p": r["candidates"][a]["p"],
                                "candidate_decision": r["candidates"][a]["decision"],
                            }
                            for r in body["outcomes"]
                        ]
                    )
                    for a in ("nonlinear", "quality")
                },
            }
        else:
            result = review(body, folder, refit=args.refit, project=args.project)
        write(args.output, result)
        print(
            json.dumps({k: v for k, v in result.items() if k not in ("native_json_parity", "arms", "cases")}),
            flush=True,
        )
