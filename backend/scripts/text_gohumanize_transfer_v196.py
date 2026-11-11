"""Compare fixed passage evidence on independently published article rewrites."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

from scripts import text_format_comparison_v94 as historical
from scripts import text_layout_methods_v101 as layout
from scripts import video_appearance_runtime_v108 as runtime
from scripts.grid_source_speech_v147 import qualification_profile
from scripts.text_format_views_v93 import correspondence, from_string, text_sha
from scripts.video_appearance_corrected_v112 import scoped

ROOT = historical.ROOT
PANEL = ROOT / "docs/evaluation-panels"
INPUT = ROOT / "backend/data/independent-evaluation/text-gohumanize-intake-v193"
CASES = INPUT / "preparation/private-cases.jsonl"
SOURCE_RECEIPT = PANEL / "text-gohumanize-intake-complete-v193.json"
REFERENCE = PANEL / "text-format-response-freeze-v94.json"
TRIAL = INPUT.with_name("text-gohumanize-transfer-v196")
PUBLIC = PANEL / "text-gohumanize-transfer-freeze-v196.json"
MODULE = "scripts.text_gohumanize_transfer_v196"
ARMS = ("human", "full_ai", "short_ai", "short_format", "long_ai", "long_format")
BATCHES = 6
PAIRS_PER_BATCH = 9
SCOPE = (
    "Federal publication excerpts and matched authored rewrites from two generators. "
    "Constructed replacements test source-aligned evidence, with unchanged internal research points."
)
read, write, sha, require = layout.read, layout.write, layout.sha, layout.require


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def labels_for(row):
    raw, spans = row["text"], row["ai_spans"]
    require(text_sha(raw) == row["text_sha256"], "Prepared source text changed")
    require(
        isinstance(spans, list)
        and all(type(a) is type(b) is int and 0 <= a < b <= len(raw) for a, b in spans)
        and all(left[1] <= right[0] for left, right in zip(spans, spans[1:])),
        "Ordered non-overlapping Unicode source spans required",
    )
    labels = dict.fromkeys((i for i, value in enumerate(raw) if not value.isspace()), 0)
    require(bool(labels), "Whitespace alone cannot supply passage evidence")
    for start, stop in spans:
        for i in range(start, stop):
            if i in labels:
                labels[i] = 1
    require(
        bool(sum(labels.values())) == row["arm"].endswith("_ai")
        and (row["arm"] != "full_ai" or all(labels.values())),
        "Source labels do not match the fixed arm",
    )
    return labels


def inputs():
    source = read(SOURCE_RECEIPT)
    require(
        source["schema"] == "chipmunk.text-gohumanize-preparation-result/193"
        and source["private_cases_sha256"] == sha(CASES),
        "Exact verified preparation journal required",
    )
    cases = records(CASES)
    require(
        len(cases) == len(ARMS) * BATCHES * PAIRS_PER_BATCH
        and len({r["case"] for r in cases}) == len(cases)
        and len({r["pair"] for r in cases}) == BATCHES * PAIRS_PER_BATCH
        and len({r["article_group"] for r in cases}) == 30
        and {r["generator"] for r in cases} == {"gpt-4o-mini", "deepseek-chat"},
        "Complete fixed preparation denominator required",
    )
    for i in range(0, len(cases), len(ARMS)):
        group = cases[i : i + len(ARMS)]
        require(
            tuple(row["arm"] for row in group) == ARMS
            and len({(row["pair"], row["article_group"], row["generator"], row["agency"]) for row in group})
            == 1
            and all(row["status"] == "constructed" for row in group[:2]),
            "Each article pair must preserve its six ordered arms",
        )
        for row in group:
            if row["status"] == "constructed":
                labels_for(row)
            else:
                require(
                    row["status"] == "unavailable" and row["text"] is row["ai_spans"] is None,
                    "Unavailable preparation must not supply a prediction input",
                )
    return cases


def identity():
    prior = read(REFERENCE)
    paths = (
        Path(__file__),
        ROOT / "backend/tests/test_text_gohumanize_transfer_v196.py",
        ROOT / "docs/text-gohumanize-transfer-v196-method.md",
        ROOT / "backend/scripts/text_layout_methods_v101.py",
        ROOT / "backend/scripts/video_appearance_runtime_v108.py",
        ROOT / "backend/scripts/grid_source_speech_v147.py",
    )
    require(
        prior["schema"] == "chipmunk.text-format-response-freeze/94"
        and all(sha(ROOT / path) == digest for path, digest in prior["code_sha256"].items())
        and all(
            sha(historical.PRIOR / path) == digest for path, digest in prior["prior_artifacts_sha256"].items()
        )
        and all(importlib.metadata.version(name) == version for name, version in prior["packages"].items()),
        "Keep the exact pretrained runtime, calibration maps, head and points",
    )
    return {
        "schema": "chipmunk.text-gohumanize-transfer-freeze/196",
        "preparation_receipt_sha256": sha(SOURCE_RECEIPT),
        "private_cases_sha256": sha(CASES),
        "reference_sha256": sha(REFERENCE),
        "code_sha256": {str(path.relative_to(ROOT)): sha(path) for path in paths},
        "batches": BATCHES,
        "pairs_per_batch": PAIRS_PER_BATCH,
        "methods": list(layout.METHODS),
        "scope": SCOPE,
    }


def freeze():
    require(not TRIAL.exists() and not PUBLIC.exists(), "Preserve every prior attempt")
    cases, locked = inputs(), identity()
    TRIAL.mkdir(mode=0o700)
    write(TRIAL / "freeze.json", locked)
    write(PUBLIC, locked)
    return {
        "status": "frozen",
        "planned_cases": len(cases),
        "constructible_cases": sum(row["status"] == "constructed" for row in cases),
    }


def verify():
    require(read(PUBLIC) == read(TRIAL / "freeze.json") == identity(), "Exact frozen transfer required")
    return inputs()


def views(raw):
    primary = from_string(raw).text
    canonical = from_string(" ".join(raw.split())).text
    return tuple(
        {"text": text, "source_positions": correspondence(raw, text)} for text in (primary, canonical)
    )


def infer(native, head, maps, row, audit):
    from scripts import writing_span_calibration_v7 as calibration
    from scripts import writing_span_heads_v7 as heads

    def calibrated(proposals, method):
        return [{**item, "score": calibration.predict(item["raw"], maps[method])} for item in proposals]

    try:
        primary, canonical = views(row["text"])
    except ValueError:
        return {"status": "unassessed", "reason": "extraction_correspondence"}

    def source_input(view):
        return {"id": row["case"], "group": row["article_group"], "role": "transfer", "text": view["text"]}

    def current(view):
        try:
            result, binding = native.current_proposals(source_input(view), audit=audit)
        except ValueError as error:
            if "foci omit source characters" not in str(error):
                raise
            return None, None
        return calibrated(result, "current"), binding

    primary_response, primary_proof = current(primary)
    if primary["text"] == canonical["text"]:
        canonical_response, canonical_proof = primary_response, primary_proof
    else:
        canonical_response, canonical_proof = current(canonical)
    try:
        features = native.features(source_input(canonical), audit=audit)
    except ValueError as error:
        if "coverage omits" not in str(error):
            raise
        interval_response = interval_proof = None
    else:
        interval_response = calibrated(heads.propose(head, "interval", features), "interval")
        interval_proof = features["native"]
    return {
        "status": "measured",
        "primary_current": primary_response,
        "canonical_current": canonical_response,
        "canonical_interval": interval_response,
        "canonical_shared_current": primary["text"] == canonical["text"],
        "native_proofs": [primary_proof, canonical_proof, interval_proof],
    }


def worker(batch, output):
    rows = verify()
    folder = TRIAL / f"extract-{batch:02d}"
    require(
        output == folder / "output"
        and (folder / "resources.jsonl").is_file()
        and (folder / "worker.log").is_file()
        and not (folder / "profile.json").exists(),
        "Run inference through the owned guard",
    )
    from scripts import writing_span_heads_v7 as heads
    from scripts.writing_span_native_v7 import Native

    head = heads.restore("interval", read(historical.PRIOR / "interval-head.json")["state"])
    maps = read(historical.PRIOR / "calibrators.json")
    native = Native(with_current=True)
    width = len(ARMS) * PAIRS_PER_BATCH
    selected = rows[batch * width : (batch + 1) * width]
    output.mkdir(mode=0o700)
    journal = output / "private-responses.jsonl"
    first = True
    with journal.open("x") as stream:
        for row in selected:
            if row["status"] == "constructed":
                response = infer(native, head, maps, row, first)
                first = False
            else:
                response = {"status": "preparation_unavailable"}
            stream.write(json.dumps({"case": row["case"], **response}, allow_nan=False) + "\n")
            stream.flush()
    native.unchanged()
    write(
        output / "result.json",
        {
            "status": "complete",
            "batch": batch,
            "planned_cases": len(selected),
            "freeze_sha256": sha(PUBLIC),
            "journal_sha256": sha(journal),
            "actual_forwards": native.calls,
            "new_neural_calls": sum(native.calls.values()),
        },
    )


def run(batch):
    require(0 <= batch < BATCHES, "One fixed transfer batch required")
    verify()

    def launch(argv, **kwargs):
        expected = TRIAL / f"extract-{batch:02d}/output"
        require(
            argv[3:] == ["scripts.video_appearance_native_v108", "extract", str(expected), str(batch)],
            "Exact owned transfer command required",
        )
        return subprocess.Popen([*argv[:3], MODULE, "_worker", str(batch), str(expected)], **kwargs)

    scoped(
        runtime.run,
        {
            "TRIAL": TRIAL,
            "verify": verify,
            "prerequisites": lambda *_: True,
            "__file__": __file__,
            "subprocess": SimpleNamespace(
                Popen=launch, STDOUT=subprocess.STDOUT, TimeoutExpired=subprocess.TimeoutExpired
            ),
        },
    )("extract", batch)


def measurements(row, response, points):
    if row["status"] != "constructed":
        require(
            response["status"] == "preparation_unavailable", "Unexpected prediction for unavailable source"
        )
        return dict.fromkeys(layout.METHODS, {"status": "preparation_unavailable"})
    labels = labels_for(row)
    if response["status"] != "measured":
        require(response["status"] == "unassessed", "Unknown worker disposition")
        return {method: layout.measure(labels, None) for method in layout.METHODS}
    primary, canonical = views(row["text"])

    def project(key, view, model):
        proposals = response[key]
        return layout.project(view, proposals, points[model]["cut"]) if proposals is not None else None

    methods = layout.compare(
        project("primary_current", primary, "current"),
        project("canonical_current", canonical, "current"),
        project("canonical_interval", canonical, "interval"),
    )
    return {method: layout.measure(labels, evidence) for method, evidence in methods.items()}


def readout():
    rows = verify()
    points = read(historical.PRIOR / "points.json")
    responses, counts, profiles = [], Counter(), []
    for batch in range(BATCHES):
        stage = TRIAL / f"extract-{batch:02d}"
        journal = stage / "output/private-responses.jsonl"
        result = read(stage / "output/result.json")
        qualification_profile(stage, result["new_neural_calls"], {}, Path(__file__).name, "extract", batch)
        require(
            result["batch"] == batch
            and result["planned_cases"] == len(ARMS) * PAIRS_PER_BATCH
            and result["freeze_sha256"] == sha(PUBLIC)
            and result["journal_sha256"] == sha(journal),
            "Complete exact neural batch required",
        )
        responses.extend(records(journal))
        counts.update(result["actual_forwards"])
        profiles.append(sha(stage / "profile.json"))
    require([r["case"] for r in rows] == [r["case"] for r in responses], "Prediction roster mismatch")
    tables, flagged = defaultdict(Counter), defaultdict(set)
    for row, response in zip(rows, responses, strict=True):
        for method, measured in measurements(row, response, points).items():
            for generator in ("all", row["generator"]):
                key = (generator, row["arm"], method)
                status = measured["status"]
                tables[key][status] += 1
                for field, value in measured.items():
                    if isinstance(value, (int, float)):
                        tables[key][field] += value
                if measured.get("genuine_document_flagged"):
                    flagged[key].add(row["article_group"])
    results = {}
    for (generator, arm, method), tally in sorted(tables.items()):
        row = dict(tally)
        row["planned"] = tally["ok"] + tally["unassessed"] + tally["preparation_unavailable"]
        row["mean_character_iou_assessed"] = (
            row.pop("character_iou", 0) / tally["ok"] if tally["ok"] else None
        )
        row["genuine_article_groups_flagged"] = len(flagged[generator, arm, method])
        results.setdefault(generator, {}).setdefault(arm, {})[method] = row
    return {
        "schema": "chipmunk.text-gohumanize-transfer-readout/196",
        "status": "complete",
        "freeze_sha256": sha(PUBLIC),
        "actual_forwards": dict(counts),
        "profile_sha256": profiles,
        "source_pairs": BATCHES * PAIRS_PER_BATCH,
        "article_groups": 30,
        "scope": SCOPE,
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for action in ("freeze", "readout"):
        commands.add_parser(action)
    for action in ("run", "_worker"):
        command = commands.add_parser(action)
        command.add_argument("batch", type=int, choices=range(BATCHES))
        if action == "_worker":
            command.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.command == "run":
        run(args.batch)
    elif args.command == "_worker":
        worker(args.batch, args.output)
    else:
        print(json.dumps({"freeze": freeze, "readout": readout}[args.command](), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
