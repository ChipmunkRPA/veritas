"""Test a relative paragraph evidence rule on a sealed article transfer cohort."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path
from statistics import mean
from types import SimpleNamespace

from scripts import text_elife_paragraph_rank_v281 as rank
from scripts import text_elife_passage_score_v275 as previous
from scripts import text_elife_transfer_rewrites_v289 as source
from scripts.text_short_insert_intake_v203 import fingerprints

ROOT, PANEL, BASE = source.ROOT, source.PANEL, source.BASE
NAME = "text-elife-relative-paragraph-v292"
MODULE = "scripts.text_elife_relative_paragraph_v292"
TRIAL = BASE / NAME
PUBLIC = PANEL / f"{NAME}-freeze.json"
POINT = TRIAL / "development-point.json"
GENERATED = PANEL / "text-elife-transfer-rewrites-complete-v289.json"
CASES = source.TRIAL / "private-cases.jsonl"
GENERATED_SHA = "d4e13615a3f080ed03d159a2c4219f2cc67e907559c7929db0fb6a6b78da8580"
GENERATION_FREEZE_SHA = "e63b6bee860917815cfde7c9fc28415562088a8c48526928aa11524c3e756cd6"
CURRENT_FREEZE_SHA = "6c11cb1bf13ceccf6ad7c6851eacee58d4992c8475701a25bf855145745eb141"
EXPECTED = {"development": {"qwen": 8, "smol2": 4}, "confirmation": {"qwen": 5, "smol2": 5}}
EXPECTED_POSITIVE_GROUPS = {"development": 10, "confirmation": 8}
BATCH_WIDTHS = (15, 15, 9)
CRITERIA = {
    "minimum_development_originals_fully_assessed": 12,
    "minimum_confirmation_originals_fully_assessed": 12,
    "minimum_confirmation_mixed_fully_assessed": 9,
    "minimum_added_useful_cases": 3,
    "minimum_added_useful_source_groups": 2,
    "minimum_added_useful_cases_per_family": 1,
    "maximum_added_original_flags": 1,
    "maximum_wrong_added_mixed_paragraphs": 2,
    "maximum_added_unchanged_character_share": 0.05,
}
COUNTS = (
    "planned",
    "preparation_unavailable",
    "eligible",
    "fully_assessed",
    "partially_assessed",
    "unassessed",
    "known_human_characters",
    "known_generated_characters",
    "covered_characters",
    "added_highlights",
    "added_unchanged_characters",
    "wrong_added_mixed_paragraphs",
    "added_useful_locations",
)
read, write, sha, require, scoped = (
    previous.read,
    previous.write,
    previous.sha,
    previous.require,
    previous.scoped,
)
layout = previous.prior.layout


def validate(rows, partitions, expected):
    require(len(rows) == 78, "Exact denominator including unavailable attempts required")
    roster = [(role, item) for role, originals in partitions.items() for item in originals]
    by_role, fragments, dispositions = defaultdict(list), defaultdict(set), defaultdict(Counter)
    positive = defaultdict(set)
    for index, (role, original) in enumerate(roster):
        group = original["group"]
        control, *mixed = rows[3 * index : 3 * index + 3]
        require(
            [(row["generator"], row["arm"]) for row in (control, *mixed)] == list(previous.ROLES)
            and all(row["role"] == role and row["article_group"] == group for row in (control, *mixed))
            and control == source.control(original, role),
            "Frozen group, role, and original ordering required",
        )
        previous.prior.labels_for(control)
        fragments[role].update(fingerprints(control["text"]))
        by_role[role].extend((control, *mixed))
        a, b = original["edit_target_codepoints"]
        for row in mixed:
            family = row["generator"]
            require(
                row["case"] == f"{group}-{family}"
                and row["source_target"] == [a, b]
                and row["source_sha256"] == original["text_sha256"],
                "Exact generated target required",
            )
            if row["status"] == "unavailable":
                require(
                    row["text"] is row["text_sha256"] is row["ai_spans"] is None and row["reason"],
                    "Unavailable attempts must supply no source truth or score input",
                )
            else:
                require(
                    row["status"] == "constructed" and row["reason"] is None and len(row["ai_spans"]) == 1,
                    "Exactly one recorded generated paragraph required",
                )
                labels = previous.prior.labels_for(row)
                start, end = row["ai_spans"][0]
                text, raw = row["text"], original["text"]
                require(
                    start == a
                    and text[:start] == raw[:a]
                    and text[end:] == raw[b:]
                    and text[start:end] != raw[a:b]
                    and 0 < sum(labels.values()) < len(labels),
                    "The Unicode splice must reconstruct the original prefix and suffix",
                )
                rank.paragraph_edges(original, row)
                positive[role].add(group)
                fragments[role].update(fingerprints(text))
            dispositions[f"{role}:{family}"][row["reason"] or "constructed"] += 1
    require(
        dispositions == expected
        and {role: len(groups) for role, groups in positive.items()} == EXPECTED_POSITIVE_GROUPS
        and {role: len(items) for role, items in by_role.items()}
        == dict.fromkeys(source.ROLES, sum(BATCH_WIDTHS))
        and not fragments["development"] & fragments["confirmation"],
        "Exact balanced roster and zero cross-role 12-token matches required",
    )
    return dict(by_role)


def inputs():
    receipt = read(GENERATED)
    require(
        sha(GENERATED) == GENERATED_SHA == sha(source.TRIAL / "result.json")
        and sha(source.FREEZE) == GENERATION_FREEZE_SHA == receipt["freeze_sha256"]
        and receipt["schema"] == "chipmunk.text-elife-transfer-rewrites-result/289"
        and receipt["status"] == "complete"
        and receipt["private_cases_sha256"] == sha(CASES)
        and all(
            receipt["generation_dispositions_by_stage"][f"{role}:{family}"]["constructed"] == count
            for role, by_family in EXPECTED.items()
            for family, count in by_family.items()
        ),
        "Exact completed v289 originals and rewrites required",
    )
    return validate(
        source.previous.records(CASES), source.verify(), receipt["generation_dispositions_by_stage"]
    )


def identity(rows):
    require(
        sha(previous.PUBLIC) == CURRENT_FREEZE_SHA
        and previous.prior.identity() == read(previous.prior.PUBLIC),
        "Exact v275 current classifier, calibrator and research point required",
    )
    paths = (
        Path(__file__),
        Path(rank.__file__),
        Path(previous.__file__),
        ROOT / f"docs/{NAME}-method.md",
        ROOT / "backend/tests/test_text_elife_relative_paragraph_v292.py",
    )
    return {
        "schema": "chipmunk.text-elife-relative-paragraph-freeze/292",
        "v289_result_sha256": GENERATED_SHA,
        "v289_freeze_sha256": GENERATION_FREEZE_SHA,
        "private_cases_sha256": sha(CASES),
        "v275_current_freeze_sha256": CURRENT_FREEZE_SHA,
        "code_sha256": {str(path.relative_to(ROOT)): sha(path) for path in paths},
        "source_groups_by_role": dict.fromkeys(source.ROLES, 13),
        "role_batches_planned": list(BATCH_WIDTHS),
        "planned_cases_by_role": {role: len(items) for role, items in rows.items()},
        "prepared_cases_by_role": {
            role: sum(row["status"] == "constructed" for row in items) for role, items in rows.items()
        },
        "cross_role_12_token_matches": 0,
        "point": "Strictly above second-largest raw paragraph lead among fully covered development originals only",
        "candidate": "On full coverage, add one uniquely leading paragraph only when existing findings are empty",
        "criteria": CRITERIA,
        "serving_changed": False,
    }


def freeze():
    require(not TRIAL.exists() and not PUBLIC.exists(), "Preserve every earlier attempt")
    locked = identity(inputs())
    TRIAL.mkdir(mode=0o700)
    (TRIAL / "development").mkdir(mode=0o700)
    write(TRIAL / "freeze.json", locked)
    write(PUBLIC, locked)
    return {"status": "frozen", "development_batches": 3, "confirmation_admitted": False}


def verify():
    rows = inputs()
    require(
        identity(rows) == read(PUBLIC) == read(TRIAL / "freeze.json"), "Exact prospective freeze required"
    )
    return rows


def cases_for(rows, role, batch):
    start = sum(BATCH_WIDTHS[:batch])
    return rows[role][start : start + BATCH_WIDTHS[batch]]


def current_only():
    from scripts import writing_span_native_v7 as native

    # The saved Native method needs only current; do not load its separate encoder.
    native.torch.set_num_threads(2)
    native.torch.use_deterministic_algorithms(True)
    paths = read(native.OLD / "freeze.json")["model_paths"]
    native.old.assets(Path(paths["baseline"]), native.identities.BASELINE_ASSETS)
    current = native.Native.__new__(native.Native)
    current.current = native.old.load_model("baseline", paths)
    current.calls = {"current_classifier": 0}

    def count(*_):
        current.calls["current_classifier"] += 1

    current.current.model.register_forward_pre_hook(count, with_kwargs=True)

    def unchanged():
        require(
            all(not p.requires_grad and p.grad is None for p in current.current.model.parameters()),
            "Current model parameters must stay frozen",
        )
        native.old.assets(Path(paths["baseline"]), native.identities.BASELINE_ASSETS)

    return current, unchanged


def predict(native, text, audit, calibrator):
    from scripts.writing_span_calibration_v7 import predict as calibrated

    try:
        view = previous.prior.views(text)[0]
    except ValueError:
        return {"status": "unassessed", "reason": "extraction_correspondence"}
    try:
        raw, proof = native.current_proposals({"text": view["text"]}, audit=audit)
    except ValueError as error:
        if "foci omit source characters" not in str(error):
            raise
        raw = proof = None
    return {
        "status": "measured",
        "primary_current": [{**item, "score": calibrated(item["raw"], calibrator)} for item in raw]
        if raw is not None
        else None,
        "native_proofs": [proof],
    }


def worker(role, batch, output):
    rows, point_hash = authorized(role, batch)
    folder = TRIAL / role / f"extract-{batch:02d}"
    require(
        output == folder / "output"
        and (folder / "resources.jsonl").is_file()
        and (folder / "worker.log").is_file()
        and not (folder / "profile.json").exists()
        and os.environ.get("TRANSFORMERS_OFFLINE") == "1"
        and os.environ.get("HF_HUB_OFFLINE") == "1",
        "Run the exact owned offline CPU stage",
    )
    native, unchanged = current_only()
    calibrator = read(previous.prior.historical.PRIOR / "calibrators.json")["current"]
    output.mkdir(mode=0o700)
    journal, controls = output / "private-responses.jsonl", output / "private-controls.jsonl"
    audit = True
    with journal.open("x") as stream, controls.open("x") as human_stream:
        for row in rows:
            if row["status"] == "constructed":
                value = predict(native, row["text"], audit, calibrator)
                audit = audit and (value.get("primary_current") is None)
            else:
                value = {"status": "preparation_unavailable"}
            line = json.dumps({"case": row["case"], **value}, allow_nan=False) + "\n"
            stream.write(line)
            stream.flush()
            if row["generator"] is None:
                human_stream.write(line)
                human_stream.flush()
    unchanged()
    write(
        output / "result.json",
        {
            "status": "complete",
            "role": role,
            "batch": batch,
            "planned_cases": len(rows),
            "freeze_sha256": sha(PUBLIC),
            "point_sha256": point_hash,
            "journal_sha256": sha(journal),
            "controls_sha256": sha(controls),
            "new_neural_calls": native.calls["current_classifier"],
        },
    )


def run(role, batch):
    from scripts import video_appearance_runtime_v108 as runtime

    authorized(role, batch)
    directory = TRIAL / role

    def launch(argv, **kwargs):
        expected = directory / f"extract-{batch:02d}/output"
        require(
            argv[3:] == ["scripts.video_appearance_native_v108", "extract", str(expected), str(batch)],
            "Exact guarded classifier command required",
        )
        return subprocess.Popen([*argv[:3], MODULE, "_worker", role, str(batch), str(expected)], **kwargs)

    scoped(
        runtime.run,
        {
            "TRIAL": directory,
            "verify": lambda: authorized(role, batch),
            "prerequisites": lambda *_: True,
            "__file__": __file__,
            "subprocess": SimpleNamespace(
                Popen=launch, STDOUT=subprocess.STDOUT, TimeoutExpired=subprocess.TimeoutExpired
            ),
        },
    )("extract", batch)


def observations(rows, role, point_hash, controls_only=False):
    responses, resources, calls = [], [], 0
    for batch in range(len(BATCH_WIDTHS)):
        stage = TRIAL / role / f"extract-{batch:02d}"
        result_path = stage / "output/result.json"
        result = read(result_path)
        previous.prior.qualification_profile(
            stage, result["new_neural_calls"], {}, Path(__file__).name, "extract", batch
        )
        selected = cases_for(rows, role, batch)
        journal = stage / "output/private-responses.jsonl"
        control_journal = stage / "output/private-controls.jsonl"
        require(
            result["role"] == role
            and result["batch"] == batch
            and result["planned_cases"] == len(selected)
            and result["freeze_sha256"] == sha(PUBLIC)
            and result["point_sha256"] == point_hash
            and result["controls_sha256"] == sha(control_journal),
            "Exact role, stage and control commitments required",
        )
        controls = source.previous.records(control_journal)
        if controls_only:
            saved = controls
            selected = [row for row in selected if row["generator"] is None]
        else:
            require(result["journal_sha256"] == sha(journal), "Recorded inference journal changed")
            saved = source.previous.records(journal)
            require(
                [response for row, response in zip(selected, saved, strict=True) if row["generator"] is None]
                == controls,
                "Separate original journal differs from complete journal",
            )
        require(
            [row["case"] for row in selected] == [row["case"] for row in saved],
            "Exact response roster required",
        )
        responses.extend(saved)
        resources.append(sha(stage / "profile.json"))
        calls += result["new_neural_calls"]
    return responses, resources, calls


def leader(text, scores):
    spans, start = [], 0
    for part in text.split("\n\n"):
        stop = start + len(part)
        positions = {i for i in range(start, stop) if not text[i].isspace()}
        if positions:
            spans.append((start, stop, positions))
        start = stop + 2
    known = set().union(*(span[2] for span in spans))
    require(scores.keys() <= known, "Projected native responses leave the article")
    if not known or scores.keys() != known or len(spans) < 2:
        return None
    exact = {value: Fraction(value) for value in set(scores.values())}
    means = [mean(exact[scores[index]] for index in positions) for _, _, positions in spans]
    indices = sorted(range(len(spans)), key=means.__getitem__, reverse=True)
    first, second = indices[:2]
    a, b, positions = spans[first]
    return means[first] - means[second], [a, b], positions


def control_point(controls, responses):
    require(
        len(controls) == 13
        and all(
            row["role"] == "development"
            and row["generator"] is None
            and row["arm"] == "human"
            and row["status"] == "constructed"
            for row in controls
        ),
        "Only the thirteen development originals may select the point",
    )
    leads = []
    for row, response in zip(controls, responses, strict=True):
        require(response["case"] == row["case"], "Development control order differs")
        candidate = leader(row["text"], rank.current_positions(row, response))
        if candidate is not None:
            leads.append(candidate[0])
    require(
        len(leads) >= CRITERIA["minimum_development_originals_fully_assessed"],
        "Insufficient full development control coverage to select a point",
    )
    return sorted(leads)[-2], len(leads)


def point_record(rows):
    controls = [row for row in rows["development"] if row["generator"] is None]
    responses, resources, _ = observations(rows, "development", None, controls_only=True)
    cutoff, full = control_point(controls, responses)
    return {
        "schema": "chipmunk.text-elife-relative-paragraph-point/292",
        "freeze_sha256": sha(PUBLIC),
        "development_originals": len(controls),
        "fully_assessed": full,
        "resource_profile_sha256": resources,
        "strictly_above": [cutoff.numerator, cutoff.denominator],
    }


def seal():
    from scripts.text_wikinews_archive_v205 import retain

    require(
        not (TRIAL / "confirmation").exists(), "Confirmation has already been admitted; preserve the point"
    )
    point = point_record(verify())
    retain(POINT, (json.dumps(point, indent=2, allow_nan=False) + "\n").encode())
    (TRIAL / "confirmation").mkdir(mode=0o700)
    return {
        "status": "sealed_before_confirmation",
        "point_sha256": sha(POINT),
        "development_originals": 13,
        "fully_assessed": point["fully_assessed"],
    }


def authorized(role, batch):
    rows = verify()
    folder = TRIAL / role
    require(
        all((folder / f"extract-{index:02d}/profile.json").is_file() for index in range(batch)),
        "Run role batches sequentially",
    )
    if role == "confirmation":
        require(
            folder.is_dir() and POINT.is_file() and read(POINT) == point_record(rows),
            "Confirmation requires the exact pre-response development-only point",
        )
        return cases_for(rows, role, batch), sha(POINT)
    require(
        not POINT.exists() and not (TRIAL / "confirmation").exists(),
        "Development inference ends when the point is sealed",
    )
    return cases_for(rows, role, batch), None


def compare(text, response, baseline_cut, margin_cut):
    """Inference reads only submitted text and its own current responses."""
    scores = rank.current_positions({"text": text}, response)
    if not scores:
        return None, None, None
    baseline = layout.project(previous.prior.views(text)[0], response["primary_current"], baseline_cut)
    require(baseline.covered == scores.keys(), "Calibrated and raw scores cover different positions")
    candidate = leader(text, scores)
    if baseline.mask or candidate is None or candidate[0] <= margin_cut:
        return baseline, baseline, None
    _, span, positions = candidate
    return baseline, layout.Evidence(frozenset(positions), baseline.covered), span


def summarize(rows, responses, baseline_cut, margin_cut):
    by_arm = defaultdict(lambda: Counter(dict.fromkeys(COUNTS, 0)))
    useful_groups, audit = set(), []
    for row, response in zip(rows, responses, strict=True):
        require(row["case"] == response["case"], "Exact matched inference result required")
        arm = row["generator"] or "human"
        count = by_arm[arm]
        count["planned"] += 1
        if row["status"] == "unavailable":
            require(
                response == {"case": row["case"], "status": "preparation_unavailable"},
                "Unavailable attempts have no predictions",
            )
            count["preparation_unavailable"] += 1
            continue
        labels = previous.prior.labels_for(row)
        baseline, augmented, added = compare(row["text"], response, baseline_cut, margin_cut)
        count["eligible"] += 1
        count["known_human_characters"] += len(labels) - sum(labels.values())
        count["known_generated_characters"] += sum(labels.values())
        if baseline is None:
            count["unassessed"] += 1
        else:
            count["fully_assessed" if baseline.covered == labels.keys() else "partially_assessed"] += 1
            count["covered_characters"] += len(baseline.covered)
        a, b = (layout.measure(labels, mask) for mask in (baseline, augmented))
        for name, measured in (("baseline", a), ("augmented", b)):
            for metric in (
                "genuine_document_flagged",
                "useful_ai_location",
                "highlighted_human_characters",
                "located_ai_characters",
                "recovered_ai_intervals",
            ):
                count[f"{name}_{metric}"] += measured.get(metric, 0)
            count[f"{name}_positive_overlaps"] += bool(measured.get("located_ai_characters"))
        if added is not None:
            count["added_highlights"] += 1
            count["added_unchanged_characters"] += (
                b["highlighted_human_characters"] - a["highlighted_human_characters"]
            )
            count["wrong_added_mixed_paragraphs"] += arm != "human" and not b["located_ai_characters"]
            count["added_useful_locations"] += b["useful_ai_location"]
            if b["useful_ai_location"]:
                useful_groups.add(row["article_group"])
        audit.append({"case": row["case"], "known_generated_spans": row["ai_spans"], "added_span": added})
    return by_arm, useful_groups, audit


def acceptance(by_arm, useful_groups):
    originals = by_arm["human"]
    mixed = [by_arm[family] for family in EXPECTED["confirmation"]]
    total = sum(mixed, Counter())
    return {
        "coverage": originals["fully_assessed"] >= CRITERIA["minimum_confirmation_originals_fully_assessed"]
        and total["fully_assessed"] >= CRITERIA["minimum_confirmation_mixed_fully_assessed"],
        "useful_locations": total["added_useful_locations"] >= CRITERIA["minimum_added_useful_cases"]
        and len(useful_groups) >= CRITERIA["minimum_added_useful_source_groups"]
        and all(
            row["added_useful_locations"] >= CRITERIA["minimum_added_useful_cases_per_family"]
            for row in mixed
        ),
        "false_highlights": originals["added_highlights"] <= CRITERIA["maximum_added_original_flags"]
        and total["wrong_added_mixed_paragraphs"] <= CRITERIA["maximum_wrong_added_mixed_paragraphs"]
        and total["added_unchanged_characters"]
        <= Fraction(str(CRITERIA["maximum_added_unchanged_character_share"]))
        * total["known_human_characters"],
    }


def readout():
    from scripts.text_wikinews_archive_v205 import retain

    rows = verify()
    point = read(POINT)
    require(point == point_record(rows), "Reproduce the sealed development point before evaluation")
    cutoff = Fraction(*point["strictly_above"])
    baseline = read(previous.prior.historical.PRIOR / "points.json")["current"]["cut"]
    by_role, audit, conf_checks = {}, [], {}
    for role in source.ROLES:
        responses, resources, calls = observations(rows, role, sha(POINT) if role == "confirmation" else None)
        by_arm, useful_groups, local = summarize(rows[role], responses, baseline, cutoff)
        by_role[role] = {
            "by_arm": by_arm,
            "added_useful_source_groups": len(useful_groups),
            "actual_current_forwards": calls,
            "resource_profile_sha256": resources,
            "planned_cases": len(rows[role]),
            "preparation_unavailable": sum(row["status"] == "unavailable" for row in rows[role]),
        }
        audit.extend({"role": role, **row} for row in local)
        if role == "confirmation":
            conf_checks = acceptance(by_arm, useful_groups)
    audit_path = TRIAL / "private-interval-audit.jsonl"
    retain(audit_path, "".join(json.dumps(row) + "\n" for row in audit).encode())
    return {
        "schema": "chipmunk.text-elife-relative-paragraph-result/292",
        "status": "complete",
        "freeze_sha256": sha(PUBLIC),
        "development_point_sha256": sha(POINT),
        "by_role": by_role,
        "confirmation_acceptance": conf_checks,
        "all_research_criteria_met": all(conf_checks.values()),
        "private_interval_audit_sha256": sha(audit_path),
        "serving_changed": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="mode", required=True)
    for action in ("preflight", "freeze", "seal", "readout"):
        commands.add_parser(action)
    for action in ("run", "_worker"):
        command = commands.add_parser(action)
        command.add_argument("role", choices=source.ROLES)
        command.add_argument("batch", type=int, choices=range(len(BATCH_WIDTHS)))
        if action == "_worker":
            command.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.mode == "run":
        run(args.role, args.batch)
    elif args.mode == "_worker":
        worker(args.role, args.batch, args.output)
    else:
        action = {
            "preflight": lambda: identity(inputs()),
            "freeze": freeze,
            "seal": seal,
            "readout": readout,
        }[args.mode]
        print(json.dumps(action(), indent=2, allow_nan=False))
