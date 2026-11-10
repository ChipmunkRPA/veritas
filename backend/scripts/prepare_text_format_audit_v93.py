"""Prepare consumed-source formatting pairs; no detector imports or forwards."""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from collections import Counter
from pathlib import Path

import psutil

from scripts.text_format_views_v93 import ingress_views, project_labels, text_sha

ROOT = Path(__file__).resolve().parents[2]
PRIOR = ROOT / "backend/data/independent-evaluation/writing-spans-v7"
TRIAL = ROOT / "backend/data/independent-evaluation/text-format-audit-v93"
CODE = (
    "backend/scripts/text_format_views_v93.py",
    "backend/scripts/prepare_text_format_audit_v93.py",
    "backend/app/detectors/text/extract.py",
    "backend/tests/test_text_format_views_v93.py",
    "docs/text-format-audit-v93-method.md",
)


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def write(path, body):
    with path.open("x") as stream:
        json.dump(body, stream, indent=2, allow_nan=False)
        stream.write("\n")


def build_case(case):
    require(text_sha(case["text"]) == case["text_sha256"], "Source text identity changed")
    effective, indices, requests = [], {}, []
    for view in ingress_views(case["text"]):
        labels = project_labels(case["text"], case["labels"], view)
        key = text_sha(view.text)
        if key not in indices:
            indices[key] = len(effective)
            effective.append(
                {
                    "text": view.text,
                    "text_sha256": key,
                    "source_positions": list(view.source_positions),
                    "labels": labels,
                    "views": [],
                }
            )
        item = effective[indices[key]]
        require(
            item["text"] == view.text
            and item["source_positions"] == list(view.source_positions)
            and item["labels"] == labels,
            "Equal effective text must have equal original correspondence",
        )
        item["views"].append(view.name)
        requests.append(
            {
                "view": view.name,
                "input_text_sha256": text_sha(view.input_text),
                "effective_index": indices[key],
            }
        )
    require(effective[0]["text"] == case["text"], "Prior extracted source must be idempotent")
    return {
        "id": case["id"],
        "group": case["group"],
        "prior_role": case["role"],
        "recipe": case["recipe"],
        "original_text_sha256": case["text_sha256"],
        "requested_views": requests,
        "effective": effective,
        "offset_unit": "Unicode code points, end exclusive",
    }


def prepare():
    TRIAL.mkdir(exist_ok=False)
    observations, started, status, error_type = [], time.monotonic(), "failed", None

    def observe(initial=False):
        row = {
            "elapsed_s": time.monotonic() - started,
            "rss_bytes": psutil.Process().memory_info().rss,
            "available_ram_bytes": psutil.virtual_memory().available,
            "free_disk_bytes": shutil.disk_usage(TRIAL).free,
        }
        observations.append(row)
        require(
            row["elapsed_s"] <= 900
            and row["rss_bytes"] <= 256 * 1024**2
            and row["available_ram_bytes"] >= (2 * 1024**3 if initial else 1536 * 1024**2)
            and row["free_disk_bytes"] >= (3 * 1024**3 if initial else 1536 * 1024**2),
            "Data-only preparation budget exceeded",
        )

    try:
        observe(initial=True)
        lock = json.loads((PRIOR / "freeze.json").read_text())
        public = json.loads((ROOT / "docs/evaluation-panels/writing-spans-v7-freeze.json").read_text())
        require(
            public["private_lock_sha256"] == sha(PRIOR / "freeze.json"),
            "Published prior experiment identity required",
        )
        require(sha(PRIOR / "inputs.json") == lock["input_sha256"], "Exact consumed source inputs required")
        diagnostic = json.loads((PRIOR / "diagnostic.json").read_text())
        require(
            diagnostic["status"] == "complete"
            and diagnostic["documents"] == 288
            and diagnostic["freeze_sha256"] == sha(PRIOR / "freeze.json")
            and diagnostic["journal_sha256"] == sha(PRIOR / "diagnostic.jsonl"),
            "Complete consumed diagnostic role required",
        )
        seal = json.loads((PRIOR / "control-seal.json").read_text())
        prior_hashes = {}
        for name, expected in seal["files"].items():
            require(
                Path(name).name == name and sha(PRIOR / name) == expected, "Prior sealed controls changed"
            )
            prior_hashes[name] = expected
        for name in ("interval-head.json", "token-head.json"):
            path = PRIOR / name
            fitted = json.loads((PRIOR / "fit.json").read_text())
            require(sha(path) == fitted["head_sha256"][name.split("-")[0]], "Prior fitted head changed")
            prior_hashes[name] = sha(path)
        cases = [
            c for c in json.loads((PRIOR / "inputs.json").read_text())["cases"] if c["role"] == "diagnostic"
        ]
        require(
            len(cases) == 288 and len({c["group"] for c in cases}) == 32,
            "All288 consumed documents and32 groups required",
        )
        packet = TRIAL / "inputs.jsonl"
        changed = Counter()
        unique = Counter()
        requests = 0
        with packet.open("x") as stream:
            for case in cases:
                record = build_case(case)
                stream.write(json.dumps(record, allow_nan=False) + "\n")
                stream.flush()
                requests += len(record["requested_views"])
                unique[case["recipe"]] += len(record["effective"])
                for request in record["requested_views"]:
                    changed[request["view"]] += (
                        record["effective"][request["effective_index"]]["text_sha256"] != case["text_sha256"]
                    )
                observe()
        freeze = {
            "schema": "chipmunk.text-format-input-freeze/93",
            "code_sha256": {p: sha(ROOT / p) for p in CODE},
            "prior_freeze_sha256": sha(PRIOR / "freeze.json"),
            "prior_inputs_sha256": sha(PRIOR / "inputs.json"),
            "prior_diagnostic_sha256": sha(PRIOR / "diagnostic.json"),
            "prior_control_seal_sha256": sha(PRIOR / "control-seal.json"),
            "prior_artifacts_sha256": prior_hashes,
            "input_packet_sha256": sha(packet),
            "new_neural_calls": 0,
            "scope": "Consumed diagnostic-source formatting stress preparation, not fresh authors, new detector accuracy or permission to retune/adopt old heads. Exact source text, per-character labels/mappings and record hashes remain private.",
        }
        write(TRIAL / "freeze.json", freeze)
        summary = {
            "schema": "chipmunk.text-format-input-preparation/93",
            "documents": 288,
            "known_source_groups": 32,
            "requested_views": requests,
            "unique_effective_classifier_texts": sum(unique.values()),
            "equivalent_view_runs_avoided": requests - sum(unique.values()),
            "effective_text_changed_documents": dict(changed),
            "effective_texts_by_recipe": dict(unique),
            "input_packet_sha256": sha(packet),
            "freeze_sha256": sha(TRIAL / "freeze.json"),
            "exact_nonwhitespace_and_token_boundaries_preserved": True,
            "source_character_labels_preserved": True,
            "new_neural_calls": 0,
            "model_response_comparison_complete": False,
            "defaults_changed": False,
            "scope": "Five paired formatting views of each already consumed160-word controlled document. Actual app extraction is measured before deduplicating identical effective text. This is input/provenance verification only, not detector stability or natural-document accuracy.",
        }
        write(TRIAL / "summary.json", summary)
        observe()
        status = "complete"
        print(json.dumps(summary), flush=True)
    except BaseException as error:
        error_type = type(error).__name__
        raise
    finally:
        write(
            TRIAL / "preparation-profile.json",
            {
                "status": status,
                "error_type": error_type,
                "duration_s": time.monotonic() - started,
                "maximum_per_case_observed_process_rss_bytes": max(
                    (o["rss_bytes"] for o in observations), default=0
                ),
                "minimum_observed_available_ram_bytes": min(
                    (o["available_ram_bytes"] for o in observations), default=None
                ),
                "resource_observations": observations,
                "new_neural_calls": 0,
                "scope": "Data-only per-case observations, not continuous sampling or guaranteed peaks. No model, child process, retry or overwrite.",
            },
        )


if __name__ == "__main__":
    prepare()
