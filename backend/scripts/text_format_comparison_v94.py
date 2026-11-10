"""Fixed-head text-format response comparison on consumed controlled sources.

Model imports are confined to score(); preparation and summary are non-neural.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
from collections import Counter, defaultdict

from scripts.prepare_text_format_audit_v93 import CODE as INPUT_CODE
from scripts.prepare_text_format_audit_v93 import PRIOR, ROOT, require, sha, write
from scripts.text_format_views_v93 import text_sha

INPUT = ROOT / "backend/data/independent-evaluation/text-format-audit-v93"
TRIAL = ROOT / "backend/data/independent-evaluation/text-format-comparison-v94"
ARMS = ("current", "interval")
METHODS = (*ARMS, "union", "intersection")
CODE = (
    *INPUT_CODE,
    "backend/scripts/text_format_comparison_v94.py",
    "backend/scripts/text_format_runtime_v94.py",
    "backend/scripts/local_probe_limits_v24.py",
    "backend/scripts/audio_tree_sampler_v64.py",
    "backend/tests/test_text_format_comparison_v94.py",
    "docs/text-format-comparison-v94-protocol.md",
)
RULES = {
    "documents": 288,
    "source_groups": 32,
    "batches": 8,
    "documents_per_batch": 36,
    "effective_texts": 800,
    "arms": list(ARMS),
    "combinations": ["union", "intersection"],
    "models": "Existing frozen current passage classifier and ModernBERT encoder/interval head; not four independent models.",
    "parity": "Every unchanged document must exactly reproduce saved raw/calibrated current/interval proposals, native feature/offset hashes and actual current input IDs before comparing its reformatted views.",
    "mapping": "Project actual covered non-whitespace source positions; uncovered scores remain missing. No bounding-range filling or zero-probability imputation.",
    "calibration": "Keep the v7 separately fitted maps and genuine-only research points unchanged. No new fit, tuning, selection or serving admission.",
    "resource": "Sequential CPU float32,2threads,900s,6GiB sampled tree RSS/initial RAM,3GiB initial disk,1.5GiB running RAM/disk reserve.",
}


def read(path):
    return json.loads(path.read_text())


def identity():
    prior = read(PRIOR / "freeze.json")
    return {
        "schema": "chipmunk.text-format-response-freeze/94",
        "rules": RULES,
        "code_sha256": {p: sha(ROOT / p) for p in set(CODE) | set(prior["code_sha256"])},
        "input_freeze_sha256": sha(INPUT / "freeze.json"),
        "input_packet_sha256": sha(INPUT / "inputs.jsonl"),
        "prior_diagnostic_sha256": sha(PRIOR / "diagnostic.json"),
        "prior_diagnostic_journal_sha256": sha(PRIOR / "diagnostic.jsonl"),
        "prior_artifacts_sha256": read(INPUT / "freeze.json")["prior_artifacts_sha256"],
        "packages": {
            n: importlib.metadata.version(n)
            for n in ("torch", "transformers", "numpy", "scipy", "safetensors", "psutil")
        },
        "new_neural_calls": 0,
        "defaults_changed": False,
        "scope": "Formatting stress on reused controlled documents, not fresh-author accuracy, new calibration or outside human review. Public thresholds, provider-watermark claims and serving changes are excluded.",
    }


def inputs():
    lock = read(INPUT / "freeze.json")
    require(
        lock == read(ROOT / "docs/evaluation-panels/text-format-input-freeze-v93.json"),
        "Published input freeze required",
    )
    require(lock["code_sha256"] == {p: sha(ROOT / p) for p in INPUT_CODE}, "Input implementation changed")
    require(lock["input_packet_sha256"] == sha(INPUT / "inputs.jsonl"), "Private input packet changed")
    old = read(PRIOR / "freeze.json")
    require(
        all(sha(ROOT / p) == v for p, v in old["code_sha256"].items()), "Prior native/scoring code changed"
    )
    require(sha(PRIOR / "freeze.json") == lock["prior_freeze_sha256"], "Prior source experiment changed")
    require(
        all(sha(PRIOR / p) == v for p, v in lock["prior_artifacts_sha256"].items()),
        "Prior maps/heads/points changed",
    )
    rows = [json.loads(line) for line in (INPUT / "inputs.jsonl").read_text().splitlines()]
    require(
        len(rows) == 288
        and len({r["id"] for r in rows}) == 288
        and len({r["group"] for r in rows}) == 32
        and sum(len(r["effective"]) for r in rows) == 800,
        "Complete consumed paired design required",
    )
    return rows


def freeze():
    TRIAL.mkdir(exist_ok=False)
    inputs()
    body = identity()
    write(TRIAL / "freeze.json", body)
    write(ROOT / "docs/evaluation-panels/text-format-response-freeze-v94.json", body)
    print(json.dumps({"status": "frozen", "documents": 288, "new_neural_calls": 0}), flush=True)


def check():
    require(read(TRIAL / "freeze.json") == identity(), "Frozen response comparison changed")
    require(
        read(TRIAL / "freeze.json")
        == read(ROOT / "docs/evaluation-panels/text-format-response-freeze-v94.json"),
        "Published response freeze required",
    )
    return inputs()


def projected(view, proposals, cut):
    require(
        type(cut) in (int, float) and math.isfinite(cut) and cut >= 0, "Finite fixed research point required"
    )
    positions = view["source_positions"]
    text = view["text"]
    require(len(positions) == len(text), "Exact source mapping required")
    mapped = [p for p in positions if p is not None]
    require(
        all(type(p) is int and p >= 0 for p in mapped) and all(a < b for a, b in zip(mapped, mapped[1:])),
        "Strictly increasing original-character positions required",
    )
    mask, scores = set(), {}
    for p in proposals:
        a, b, q = p["start"], p["end"], p["score"]
        require(
            type(a) is type(b) is int
            and 0 <= a <= b <= len(text)
            and type(q) in (int, float)
            and math.isfinite(q)
            and 0 <= q <= 1,
            "Finite source-bounded proposal required",
        )
        for i in range(a, b):
            source = positions[i]
            if source is None:
                require(text[i].isspace(), "A retained character lacks original support")
                continue
            require(
                type(source) is int and source >= 0 and not text[i].isspace(), "Invalid original position"
            )
            scores[source] = max(scores.get(source, q), q)
            if q >= cut:
                mask.add(source)
    return mask, scores


def difference(reference, alternate, labels):
    """None represents unassessed input, not an empty/negative observed result."""
    if reference is None or alternate is None:
        return {"status": "unassessed"}
    first, p = reference
    second, q = alternate
    require(
        all(
            i in labels and type(labels[i]) is int and labels[i] in (0, 1)
            for i in first | second | p.keys() | q.keys()
        ),
        "Known original-character labels required",
    )
    shared = p.keys() & q.keys()
    errors = [abs(p[i] - q[i]) for i in shared]
    added, removed = second - first, first - second
    return {
        "status": "ok",
        "highlight_changed": bool(added or removed),
        "original_highlighted": len(first),
        "view_highlighted": len(second),
        "added_ai_characters": sum(labels[i] == 1 for i in added),
        "lost_ai_characters": sum(labels[i] == 1 for i in removed),
        "added_human_characters": sum(labels[i] == 0 for i in added),
        "removed_human_characters": sum(labels[i] == 0 for i in removed),
        "highlight_jaccard": len(first & second) / len(first | second) if first | second else 1.0,
        "shared_scored_characters": len(shared),
        "only_original_scored_characters": len(p.keys() - q.keys()),
        "only_view_scored_characters": len(q.keys() - p.keys()),
        "absolute_score_change_sum": sum(errors),
        "maximum_score_change": max(errors) if errors else None,
    }


def methods(view, response, points):
    if response["status"] != "ok":
        return dict.fromkeys(METHODS)
    result = {k: projected(view, response["arms"][k], points[k]["cut"]) for k in ARMS}
    a, b = result["current"][0], result["interval"][0]
    # Combined masks have no combined probability. Missing coverage is not zero.
    result.update(union=(a | b, {}), intersection=(a & b, {}))
    return result


def score(index):
    require(type(index) is int and 0 <= index < 8, "One fixed response batch required")
    cases = check()[index * 36 : (index + 1) * 36]
    old_rows = {
        r["id"]: r
        for r in (json.loads(line) for line in (PRIOR / "diagnostic.jsonl").read_text().splitlines())
    }
    # Imports occur only inside the monitored neural child.
    from scripts import writing_span_calibration_v7 as calibration
    from scripts import writing_span_heads_v7 as heads
    from scripts.writing_span_native_v7 import Native

    head = heads.restore("interval", read(PRIOR / "interval-head.json")["state"])
    maps = read(PRIOR / "calibrators.json")
    native = Native(with_current=True)
    path = TRIAL / f"batch-{index:02d}.jsonl"
    counts = Counter()
    with path.open("x") as stream:
        for ordinal, case in enumerate(cases):
            responses = []
            for j, view in enumerate(case["effective"]):
                require(text_sha(view["text"]) == view["text_sha256"], "Effective input changed")
                minimal = {
                    "id": case["id"] + ":" + str(j),
                    "group": case["group"],
                    "role": "format-audit",
                    "text": view["text"],
                }
                try:
                    features = native.features(minimal, audit=ordinal == 0 and j == 0)
                    current, proof = native.current_proposals(minimal, audit=ordinal == 0 and j == 0)
                    raw = {"current": current, "interval": heads.propose(head, "interval", features)}
                    arms = {
                        k: [{**p, "score": calibration.predict(p["raw"], maps[k])} for p in v]
                        for k, v in raw.items()
                    }
                    result = {
                        "status": "ok",
                        "arms": arms,
                        "native": features["native"],
                        "current_native": proof,
                    }
                except ValueError as error:
                    if j == 0 or "coverage omits" not in str(error):
                        raise
                    result = {
                        "status": "unassessed",
                        "reason": "unsupported_native_character_coverage",
                        "arms": None,
                    }
                if j == 0:
                    prior = old_rows[case["id"]]
                    require(
                        all(result["arms"][k] == prior["arms"][k] for k in ARMS),
                        "Original native proposals differ from saved baseline",
                    )
                    require(
                        all(
                            result["native"][k] == prior["native"][k]
                            for k in ("complete_features_sha256", "native_offsets_sha256")
                        )
                        and proof["actual_input_ids_sha256"]
                        == prior["current_native"]["actual_input_ids_sha256"],
                        "Original native feature/offset/input proof differs",
                    )
                counts[result["status"]] += 1
                responses.append(result)
            stream.write(
                json.dumps(
                    {"id": case["id"], "group": case["group"], "effective_responses": responses},
                    allow_nan=False,
                )
                + "\n"
            )
            stream.flush()
            print(
                json.dumps(
                    {
                        "batch": index,
                        "documents_completed": ordinal + 1,
                        "effective_responses": sum(counts.values()),
                    }
                ),
                flush=True,
            )
    native.unchanged()
    write(
        TRIAL / f"batch-{index:02d}-complete.json",
        {
            "batch": index,
            "documents": 36,
            "counts": dict(counts),
            "actual_pretrained_forwards": native.calls,
            "journal_sha256": sha(path),
            "freeze_sha256": sha(TRIAL / "freeze.json"),
        },
    )


def summarize():
    cases = check()
    by_id = {r["id"]: r for r in cases}
    responses = []
    profiles = []
    forwards = Counter()
    statuses = Counter()
    for i in range(8):
        complete = read(TRIAL / f"batch-{i:02d}-complete.json")
        profile = read(TRIAL / f"batch-{i:02d}-profile/profile.json")
        require(
            profile["child_exit_code"] == 0 and profile["guard"] is None and profile["sample_errors"] == 0,
            "Successful bounded response execution required",
        )
        require(
            complete["documents"] == 36
            and complete["freeze_sha256"] == sha(TRIAL / "freeze.json")
            and complete["journal_sha256"] == sha(TRIAL / f"batch-{i:02d}.jsonl"),
            "Complete unchanged response journal required",
        )
        profiles.append(profile)
        forwards.update(complete["actual_pretrained_forwards"])
        statuses.update(complete["counts"])
        responses += [json.loads(line) for line in (TRIAL / f"batch-{i:02d}.jsonl").read_text().splitlines()]
    require(
        [r["id"] for r in responses] == [r["id"] for r in cases] and sum(statuses.values()) == 800,
        "Every original document and effective response required",
    )
    points = read(PRIOR / "points.json")
    groups = defaultdict(set)
    totals = defaultdict(Counter)
    jaccards = defaultdict(list)
    maximums = defaultdict(list)
    for row in responses:
        case = by_id[row["id"]]
        first = case["effective"][0]
        labels = {
            p: v for p, v in zip(first["source_positions"], first["labels"], strict=True) if p is not None
        }
        reference = methods(first, row["effective_responses"][0], points)
        for request in case["requested_views"]:
            j = request["effective_index"]
            view = case["effective"][j]
            values = methods(view, row["effective_responses"][j], points)
            for method in METHODS:
                key = (request["view"], method)
                delta = difference(reference[method], values[method], labels)
                totals[key]["documents"] += 1
                if delta["status"] != "ok":
                    totals[key]["unassessed_pairs"] += 1
                    continue
                totals[key]["assessed_pairs"] += 1
                if delta["highlight_changed"]:
                    groups[key].add(case["group"])
                for k, v in delta.items():
                    if k not in {"status", "highlight_jaccard", "maximum_score_change"}:
                        totals[key][k] += v
                jaccards[key].append(delta["highlight_jaccard"])
                if delta["maximum_score_change"] is not None:
                    maximums[key].append(delta["maximum_score_change"])
    result = {}
    for (view, method), counts in totals.items():
        n = counts["shared_scored_characters"]
        result.setdefault(view, {})[method] = {
            **counts,
            "source_groups_with_changed_highlight": len(groups[(view, method)]),
            "mean_highlight_jaccard": sum(jaccards[(view, method)]) / len(jaccards[(view, method)])
            if jaccards[(view, method)]
            else None,
            "mean_absolute_score_change_on_shared_coverage": counts["absolute_score_change_sum"] / n
            if n
            else None,
            "maximum_score_change": max(maximums[(view, method)]) if maximums[(view, method)] else None,
        }
    summary = {
        "schema": "chipmunk.text-format-response-comparison/94",
        "documents": 288,
        "known_source_groups": 32,
        "requested_views": 1440,
        "effective_texts_attempted": 800,
        "effective_texts_scored": statuses["ok"],
        "effective_texts_unassessed": statuses["unassessed"],
        "actual_pretrained_forwards": dict(forwards),
        "results": result,
        "freeze_sha256": sha(TRIAL / "freeze.json"),
        "source_input_packet_sha256": sha(INPUT / "inputs.jsonl"),
        "successful_stages": 8,
        "duration_s_sum": sum(p["duration_s"] for p in profiles),
        "maximum_sampled_tree_rss_bytes": max(p["sampled_tree_rss_lower_bound_bytes"] for p in profiles),
        "profile_sha256": [sha(TRIAL / f"batch-{i:02d}-profile/profile.json") for i in range(8)],
        "journal_sha256": [sha(TRIAL / f"batch-{i:02d}.jsonl") for i in range(8)],
        "defaults_changed": False,
        "scope": "Same reused controlled documents under formatting interventions; not fresh-author detection, population confidence or a provider watermark. Scores describe existing separately mapped interval responses, not a whole-file probability; unions/intersections have masks only. Empty-highlight agreement is not correct authorship. Missing responses remain unassessed. No fitting, point changes, serving change or outside human review.",
    }
    write(TRIAL / "result.json", summary)
    write(ROOT / "docs/evaluation-panels/text-format-response-result-v94.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "score", "summarize"))
    parser.add_argument("index", nargs="?", type=int)
    args = parser.parse_args()
    if args.command == "score":
        score(args.index)
    else:
        require(args.index is None, "This command takes no batch index")
        freeze() if args.command == "freeze" else summarize()
