"""Replay exact isolated writing foci against archived same-context agreement."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import Counter
from pathlib import Path
from time import monotonic

from scripts import review_writing_agreement_revision_transfer as prior

SCHEMA = "chipmunk.writing-agreement-focus-replay/1"
METHOD = {
    "scope": "Retrospective replay of all forum, OLMo and eLife files in the prior editorial agreement panel.",
    "inputs": "Use the exact center sentence alone when it contains at least 12 whitespace-delimited words. "
              "Retain the original native window, scope, score and first-six cap for current positives.",
    "native_agreement": "Current positive and second native window at least 0.5.",
    "focus_agreement": "Current positive and second exact focus at least 0.5, if the focus is eligible.",
    "either_agreement": "Current positive and either of the same-window or exact-focus second responses at least 0.5.",
    "localization": "Report full-window non-whitespace character IoU of at least .30 and exact focus separately.",
    "validation": "No fitting or retuning. For a new-source follow-up require retention of every current useful "
                  "location, a reduction of at least two currently flagged human groups and at most one "
                  "newly flagged human group relative to native agreement in either of the two control panels.",
}
ARMS = ("current", "native_agreement", "focus_agreement", "either_agreement")
ROOT = Path(__file__).resolve().parents[2]
PUBLIC = ROOT / "docs/evaluation-panels/writing-agreement-ordinary-edit-2026-11-11.json"


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def write(path, data):
    with path.open("x", encoding="utf-8") as target:
        json.dump(data, target, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        target.write("\n")


def panels(parent, elife_source):
    source_audit = prior.review(parent)
    source_audit["publisher_controls"] = prior.review_elife(parent / "elife", elife_source)
    source_audit["third_generator"] = prior.review_olmo(parent / "olmo")
    public = prior.read(PUBLIC)
    controls = {
        row["case"]: row["text"]
        for row in (json.loads(line) for line in (elife_source / "private-controls.jsonl").read_text().splitlines())
        if row["role"] == "confirmation" and row["status"] == "constructed"
    }
    inputs, rows, commitments = {}, [], {}
    for name in ("forum", "olmo", "elife"):
        folder = parent if name == "forum" else parent / name
        result = prior.read(folder / "aggregate.json")
        public_result = public if name == "forum" else public["third_generator" if name == "olmo" else "publisher_controls"]
        prior.check(result["score_receipts"] == public_result["score_receipts"]
                    and result["readout_sha256"] == public_result["readout_sha256"], "Panel differs from public commitment")
        commitments[name] = {filename: prior.sha(folder / filename)
                             for filename in ("aggregate.json", "private-readout.json", "current.json", "second.json")}
        scores = {arm: prior.read(folder / f"{arm}.json")["responses"] for arm in ("current", "second")}
        archived_cases = {} if name == "elife" else {
            (case["group"], case["role"]): case["text"] for case in prior.read(folder / "cases.json")["cases"]
        }
        for case in prior.read(folder / "private-readout.json"):
            if name == "elife":
                spans = case["source_paragraphs"]
                text = controls[case["case"]][spans[0][0]:spans[-1][1]]
                role, label, truth, target = case["arm"], "elife", None, None
            else:
                text = archived_cases[case["group"], case["role"]]
                role, label, truth, target = case["role"], case["publisher"], case["focus"], case["target"]
            prior.check(digest(text) == case["text_sha256"], "Exact source excerpt differs")
            windows = []
            for index, window in enumerate(case["windows"]):
                spans = window["spans"]
                center = spans[0 if index == 0 else 1]
                value = text[center[0]:center[1]]
                key = digest(value) if len(value.split()) >= 12 else None
                if key is not None:
                    inputs[key] = value
                prior.check(window["sha256"] == digest(" ".join(text[start:end] for start, end in spans)),
                            "Native input differs from its exact source ranges")
                responses = {arm: packet[window["sha256"]]["score"] for arm, packet in scores.items()}
                relevant = False
                if truth:
                    authored, observed = prior.mask(text, [truth]), prior.mask(text, spans)
                    relevant = (authored & observed).bit_count() * 10 >= (authored | observed).bit_count() * 3
                windows.append({"focus_range": center, "focus_sha256": key, "native_sha256": window["sha256"],
                                "native": responses, "useful_iou": relevant})
            rows.append({"panel": name, "publisher": label, "group": case["group"], "role": role,
                         "text_sha256": case["text_sha256"], "target": target, "windows": windows})
    prior.check(len(rows) == 40 and sum(len(row["windows"]) for row in rows) == 276, "Panel denominator differs")
    return rows, inputs, source_audit, commitments


def selections(row, second_scores):
    current = [index for index, item in enumerate(row["windows"]) if item["native"]["current"] >= .5][:6]
    native = [index for index in current if row["windows"][index]["native"]["second"] >= .5]
    focus = []
    for index in current:
        key = row["windows"][index]["focus_sha256"]
        if key is not None and second_scores[key]["score"] >= .5:
            focus.append(index)
    return {
        "current": current,
        "native_agreement": native,
        "focus_agreement": focus,
        "either_agreement": sorted(set(native) | set(focus)),
    }


def evaluate(rows, packets):
    checked = []
    transitions = {arm: Counter() for arm in ("current", "second")}
    target_diagnostic = {}
    for row in rows:
        marks = selections(row, packets["second"])
        useful = {arm: any(row["windows"][i]["useful_iou"] for i in marks[arm]) for arm in ARMS}
        for window in row["windows"]:
            key = window["focus_sha256"]
            if key is None:
                continue
            for arm in ("current", "second"):
                native = "positive" if window["native"][arm] >= .5 else "negative"
                focus = "positive" if packets[arm][key]["score"] >= .5 else "negative"
                transitions[arm][f"{native}_to_{focus}"] += 1
        if row["role"] in {"granite", "olmo"}:
            values = target_diagnostic.setdefault(row["role"], {arm: Counter() for arm in ("current", "second")})
            target = row["windows"][row["target"]]
            key = target["focus_sha256"]
            for arm in ("current", "second"):
                local = packets[arm][key]["score"] if key is not None else None
                values[arm].update(cases=1, native_positive=target["native"][arm] >= .5,
                                   focus_positive=local is not None and local >= .5,
                                   focus_above_native=local is not None and local > target["native"][arm],
                                   focus_unavailable=local is None)
        checked.append({"text_sha256": row["text_sha256"], "publisher": row["publisher"], "group": row["group"],
                        "role": row["role"], "target": row["target"], "selected": marks, "useful": useful})
    return checked, transitions, target_diagnostic


def counts(rows):
    scopes = {
        "forum_before": lambda row: row["role"] == "before",
        "forum_edited": lambda row: row["role"] == "edited",
        "forum_human": lambda row: row["role"] in {"before", "edited"},
        "elife_human": lambda row: row["publisher"] == "elife",
        "all_human": lambda row: row["role"] not in {"granite", "olmo"},
        "granite": lambda row: row["role"] == "granite",
        "olmo": lambda row: row["role"] == "olmo",
        "all_synthetic": lambda row: row["role"] in {"granite", "olmo"},
    }
    output = {}
    for name, scope in scopes.items():
        subset = [row for row in rows if scope(row)]
        eligible = name in {"granite", "olmo", "all_synthetic"}
        cell = {}
        for arm in ARMS:
            marked = [row for row in subset if row["selected"][arm]]
            cell[arm] = {"flagged_files": len(marked),
                         "flagged_groups": len({(row["publisher"], row["group"]) for row in marked})}
            if eligible:
                hits = sum(row["useful"][arm] for row in marked)
                cell[arm].update(useful=hits, wrong_location=len(marked) - hits, unflagged=len(subset) - len(marked),
                                 exact_focus=sum(row["target"] in row["selected"][arm] for row in subset),
                                 current_useful_lost=sum(row["useful"]["current"] and not row["useful"][arm] for row in subset))
        output[name] = {"files": len(subset), "outcomes": cell}
    return output


def verify_focus_receipts(run, inputs, lock):
    result = {}
    for arm in ("current", "second"):
        packet = prior.read(run / f"{arm}-focus.json")
        prior.check(packet["freeze_sha256"] == prior.sha(lock) and set(packet["responses"]) == set(inputs),
                    "The focus score does not bind exactly to the source freeze")
        result[arm] = packet["responses"]
        prior.check(all(math.isfinite(record["score"]) and 0 <= record["score"] <= 1
                        and record["coverage"]["token_windows"] == 1
                        and 0 < record["coverage"]["tokens"] <= 512
                        and record["coverage"].get("omitted_tokens", 0) == 0
                        for record in result[arm].values()), "A focus did not receive a finite, complete score")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "current", "second", "report", "replay"))
    parser.add_argument("run", type=Path)
    parser.add_argument("parent", type=Path)
    parser.add_argument("elife_source", type=Path)
    parser.add_argument("--model", type=Path)
    args = parser.parse_args()
    os.umask(0o077)
    rows, inputs, source_audit, commitments = panels(args.parent, args.elife_source)
    lock = args.run / "freeze.json"
    plan = {
        "schema": SCHEMA, "method": METHOD, "parent_public_sha256": prior.sha(PUBLIC),
        "code_sha256": prior.sha(Path(__file__)), "source_receipts": commitments, "source_audit": source_audit,
        "inputs": sorted(inputs), "cases": len(rows),
        "eligible_occurrences": sum(window["focus_sha256"] is not None for row in rows for window in row["windows"]),
    }
    if args.stage == "freeze":
        write(lock, plan)
        print(json.dumps({"cases": len(rows), "unique_foci": len(inputs), "eligible_occurrences": plan["eligible_occurrences"]}))
        return
    prior.check(prior.read(lock) == plan, "Frozen scope or source receipts differ")
    if args.stage in {"current", "second"}:
        import torch

        from scripts.writing_focus_v3 import BASELINE_ASSETS
        from scripts.writing_paired_email_revision import SECONDARY, native_scores

        pins = BASELINE_ASSETS if args.stage == "current" else SECONDARY
        prior.check(os.getenv("HF_HUB_OFFLINE") == os.getenv("TRANSFORMERS_OFFLINE") == "1",
                    "Inference requires offline checkpoint access")
        prior.check(args.model is not None and all(prior.sha(args.model / file) == expected for file, expected in pins.items()),
                    "A locally installed classifier asset differs")
        torch.set_num_threads(1)
        start = monotonic()
        responses, forwards = native_scores(args.model, dict(sorted(inputs.items())), second=args.stage == "second")
        result = {"freeze_sha256": prior.sha(lock), "forwards": forwards, "seconds": round(monotonic() - start, 3),
                  "responses": responses}
        write(args.run / f"{args.stage}-focus.json", result)
        print(json.dumps({key: result[key] for key in ("forwards", "seconds")}))
        return

    packets = verify_focus_receipts(args.run, inputs, lock)
    evaluated, transitions, diagnostic = evaluate(rows, packets)
    output = {
        "schema": SCHEMA, "method": METHOD, "freeze_sha256": prior.sha(lock),
        "parent_public_sha256": prior.sha(PUBLIC), "source_audit": source_audit,
        "cases": len(rows), "unique_foci": len(inputs), "eligible_occurrences": plan["eligible_occurrences"],
        "native_window_occurrences": 276, "scope_transitions": transitions, "synthetic_target_diagnostic": diagnostic,
        "outcomes": counts(evaluated),
        "score_receipts": {arm: {key: prior.read(args.run / f"{arm}-focus.json")[key] for key in ("forwards", "seconds")}
                           | {"sha256": prior.sha(args.run / f"{arm}-focus.json")} for arm in ("current", "second")},
    }
    if args.stage == "report":
        write(args.run / "private-readout.json", evaluated)
        write(args.run / "aggregate.json", output)
    else:
        prior.check(prior.read(args.run / "private-readout.json") == evaluated
                    and prior.read(args.run / "aggregate.json") == output, "Regenerated output differs")
    print(json.dumps({"cases": len(rows), "unique_foci": len(inputs), "outcomes": output["outcomes"]}, sort_keys=True))


if __name__ == "__main__":
    main()
