"""Measure same-window detector agreement and a second-checkpoint rescue."""

from __future__ import annotations

import argparse
import hashlib
import math
import os
from collections import defaultdict
from pathlib import Path
from time import monotonic

from scripts.writing_email_phrase_persistence import digest, load, require, save
from scripts.writing_paired_email_revision import SECONDARY, encoded, native_contexts, native_scores

SCHEMA = "chipmunk.writing-checkpoint-agreement/1"
HUMAN = frozenset({"original", "human_splice"})
ARMS = ("current", "second_only", "agreement", "current_plus_rescue")
PANEL_NAMES = ("writing-grammar-seams", "writing-option-order-transfer")
PINS = {
    PANEL_NAMES[0]: {
        "baseline.json": "4695353edaf8138c62c13e66ca3fa1c901e62ee86574812834f628d3600842ad",
        "cases.json": "81b6925c6203907253dd9e2b225d1b63e64703d4c31fb332dab88ec263de4239",
    },
    PANEL_NAMES[1]: {
        "private-order-readout.json": "6be4681485d371770eeb1f4dfd6cb525c949c767c0663ba208bf66ed8c154c75",
        "sources.json": "6c782213773cddbf029076f0ea8dce6872f84c80a8511a023e51eb3e63e97ab6",
    },
}
METHOD = {
    "current": "First six context responses at or above .5, in source order. Reuse the previously audited current scores on exact matching inputs.",
    "second_only": "First six second-checkpoint responses at or above its publisher's .5 point, in source order.",
    "agreement": "Keep the subset of current displayed windows also at or above .5 in the second checkpoint. Do not backfill omitted windows.",
    "rescue": "Keep all current windows. Also consider full three-sentence windows with a focus of at least 12 words. In a multi-paragraph document, take the maximum for each focus paragraph. In a single paragraph, compare individual foci. Add the strongest window only if it is at least .5, exceeds the second place by a factor of two in model odds, and is outside the existing displayed windows.",
    "scope": "Same exact text and source positions, independent checkpoint weights. No new fit. Mixed English excerpts and controls already consumed by other project experiments.",
    "useful": "One displayed window must intersect the known edit at .30 non-whitespace character IoU.",
    "advance_agreement": "No lost useful file and at least two cleared human source groups across the full panel.",
    "advance_rescue": "At least two newly useful source groups, at least two publishers and generator families with added locations, and at most one new human source group flagged.",
}


def text_sha(value):
    return hashlib.sha256(value.encode()).hexdigest()


def grammar_cases(run):
    baseline = load(run / "baseline.json")
    for row in load(run / "cases.json")["cases"]:
        case = f"{row['group']}/{row['kind']}"
        old = baseline["contexts"][case]
        readings = [baseline["native"][item["sha256"]] for item in old]
        yield row["text"], row["text_sha256"], row["publisher"], row["group"], row["kind"], row["span"], old, readings


def option_cases(run):
    publishers = load(run / "sources.json")["sources"]
    for row in load(run / "private-order-readout.json"):
        case = row["case"]
        text = (run / "fixtures" / f"{case['id']}.txt").read_text()
        publisher = publishers[case["group"]]["publisher"]
        yield text, case["sha256"], publisher, case["group"], case["kind"], case["replacement"], row["contexts"], [
            context["native"] for context in row["contexts"]
        ]


def frozen_inputs(root):
    from app.detectors.text.extract import from_string
    from app.detectors.text.segment import segment
    from app.evidence.document_matched_context import one_complete_window

    rows, inputs = [], {}
    for panel, reader in zip(PANEL_NAMES, (grammar_cases, option_cases), strict=True):
        directory = root / f"{panel}-2026-11-11"
        require(all(digest(directory / name) == pin for name, pin in PINS[panel].items()), "Retained panel commitments changed")
        for text, pin, publisher, group, kind, truth, previous, native in reader(directory):
            doc = from_string(text)
            require(doc.text == text and text_sha(text) == pin, "Source text or extraction changed")
            sentences = segment(doc)
            windows = []
            for focus, nearby, saved, reading in zip(sentences, native_contexts(sentences), previous, native, strict=True):
                context, value = encoded(nearby)
                require(all(saved[key] == context[key] for key in ("sha256", "spans"))
                        and one_complete_window(reading["coverage"]), "Current source positions or token coverage changed")
                index = len(windows)
                inputs[context["sha256"]] = value
                windows.append({
                    **context, "current": reading["score"],
                    "comparison_group": index if len(doc.paragraphs) == 1 else focus.paragraph,
                    "rescue_eligible": len(nearby) == 3 and focus.n_words >= 12,
                })
            start, end = truth
            require(0 <= start < end <= len(text) and text[start:end].strip(), "Invalid truth range")
            rows.append({"panel": panel, "publisher": publisher, "group": group, "kind": kind, "text": text,
                         "text_sha256": pin, "truth": truth, "windows": windows})
    require(len(rows) == 100 and len({(row["group"], row["kind"]) for row in rows}) == 100
            and sum(row["kind"] in HUMAN for row in rows) == 56, "Panel denominator differs")
    return rows, inputs


def strictly_stronger(first, second):
    # Compare odds by multiplication so probability one has a defined limit.
    return first > second and first * (1 - second) >= 2 * second * (1 - first)


def select(windows, second_scores):
    current = [i for i, row in enumerate(windows) if row["current"] >= .5][:6]
    second = [i for i, value in enumerate(second_scores) if value >= .5][:6]
    agreement = [i for i in current if second_scores[i] >= .5]
    ranked = {}
    for index, (row, score) in enumerate(zip(windows, second_scores, strict=True)):
        if row["rescue_eligible"]:
            key = row["comparison_group"]
            if key not in ranked or score > second_scores[ranked[key]]:
                ranked[key] = index
    leaders = sorted(ranked.values(), key=second_scores.__getitem__, reverse=True)
    rescue = []
    if len(leaders) > 1:
        first, second_place = leaders[:2]
        if first not in current and second_scores[first] >= .5 and strictly_stronger(second_scores[first], second_scores[second_place]):
            rescue.append(first)
    return dict(zip(ARMS, (current, second, agreement, current + rescue), strict=True))


def useful(text, truth, spans):
    start, end = truth
    reference = {i for i in range(start, end) if not text[i].isspace()}
    marked = {i for a, b in spans for i in range(a, b) if not text[i].isspace()}
    return len(reference & marked) * 10 >= 3 * len(reference | marked)


def readout(rows, responses):
    result = []
    for row in rows:
        scores = [responses[window["sha256"]]["score"] for window in row["windows"]]
        require(all(math.isfinite(x) and 0 <= x <= 1 for x in scores), "Invalid second-checkpoint response")
        marks = select(row["windows"], scores)
        result.append({
            **{key: row[key] for key in ("panel", "publisher", "group", "kind", "text_sha256")},
            "marks": marks,
            "useful": {arm: row["kind"] not in HUMAN and any(
                useful(row["text"], row["truth"], row["windows"][i]["spans"]) for i in selected
            ) for arm, selected in marks.items()},
        })
    return result


def aggregate(rows):
    parts = defaultdict(list)
    for row in rows:
        parts["all"].append(row)
        for key in ("publisher", "kind"):
            parts[row[key]].append(row)
    outcomes = {}
    for scope, subset in sorted(parts.items()):
        result = {}
        controls = [row for row in subset if row["kind"] in HUMAN]
        edits = [row for row in subset if row["kind"] not in HUMAN]
        for arm in ARMS:
            hits = [row for row in edits if row["useful"][arm]]
            gained = [row for row in hits if not row["useful"]["current"]]
            flags = [row for row in controls if row["marks"][arm]]
            groups = {row["group"] for row in flags}
            current_groups = {row["group"] for row in controls if row["marks"]["current"]}
            result[arm] = {
                "human_flagged": len(flags), "human_groups_flagged": len(groups), "human_groups_added": len(groups - current_groups),
                "human_groups_cleared": len(current_groups - groups), "useful": len(hits), "gained": len(gained),
                "gained_groups": len({row["group"] for row in gained}),
                "lost": sum(row["useful"]["current"] and not row["useful"][arm] for row in edits),
            }
        outcomes[scope] = {"human": len(controls), "edits": len(edits), "arms": result}
    agreement = outcomes["all"]["arms"]["agreement"]
    rescue = outcomes["all"]["arms"]["current_plus_rescue"]
    scopes_gained = {
        key: sorted({row[key] for row in rows if row["useful"]["current_plus_rescue"] and not row["useful"]["current"]})
        for key in ("publisher", "kind")
    }
    return {
        "outcomes": outcomes,
        "rescue_gained_scopes": scopes_gained,
        "advance_agreement": agreement["lost"] == 0 and agreement["human_groups_cleared"] >= 2,
        "advance_rescue": rescue["gained_groups"] >= 2 and rescue["human_groups_added"] <= 1
        and all(len(values) >= 2 for values in scopes_gained.values()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "score", "readout"))
    parser.add_argument("directory", type=Path)
    parser.add_argument("panels", type=Path)
    parser.add_argument("snapshot", type=Path)
    args = parser.parse_args()
    os.umask(0o077)
    require(os.getenv("HF_HUB_OFFLINE") == os.getenv("TRANSFORMERS_OFFLINE") == "1", "Run with local model access only")
    started = monotonic()
    run = args.directory
    protocol = {
        "schema": SCHEMA, "method": METHOD, "code_sha256": digest(Path(__file__)), "panels": PINS,
        "second_assets": {name: digest(args.snapshot / name) for name in SECONDARY},
    }
    require(protocol["second_assets"] == SECONDARY, "Second checkpoint differs")
    rows, inputs = frozen_inputs(args.panels)
    if args.stage == "freeze":
        protocol["inputs"] = len(inputs)
        protocol["documents"] = len(rows)
        save(run / "freeze.json", protocol)
        return
    frozen = run / "freeze.json"
    require(load(frozen) == {**protocol, "inputs": len(inputs), "documents": len(rows)}, "Study freeze differs")
    if args.stage == "score":
        import torch

        torch.set_num_threads(1)
        result, forwards = native_scores(args.snapshot, inputs, second=True)
        save(run / "scores.json", {"freeze_sha256": digest(frozen), "responses": result, "forwards": forwards,
                                   "score_seconds": round(monotonic() - started, 3)})
        return
    score_path = run / "scores.json"
    scored = load(score_path)
    require(scored["freeze_sha256"] == digest(frozen) and set(scored["responses"]) == set(inputs), "Second score receipt differs")
    result = readout(rows, scored["responses"])
    save(run / "private-readout.json", result)
    save(run / "aggregate.json", {
        **protocol, "freeze_sha256": digest(frozen), "scores_sha256": digest(score_path),
        "readout_sha256": digest(run / "private-readout.json"), "documents": len(rows),
        "window_occurrences": sum(len(row["windows"]) for row in rows), "unique_inputs": len(inputs),
        "forwards": scored["forwards"], "score_seconds": scored["score_seconds"], **aggregate(result),
    })


if __name__ == "__main__":
    main()
