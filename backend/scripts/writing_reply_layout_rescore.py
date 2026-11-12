# SPDX-License-Identifier: Apache-2.0
"""Score changed native reply contexts, and verify exact-input score reuse on the rest."""

from __future__ import annotations

import argparse
import os
from collections import Counter, defaultdict
from pathlib import Path
from time import monotonic

from app.detectors.text.extract import extract
from app.detectors.text.segment import segment
from app.evidence.document_matched_context import one_complete_window
from scripts.writing_agreement_document_format_transfer import source_inputs
from scripts.writing_checkpoint_agreement import HUMAN, text_sha, useful
from scripts.writing_email_phrase_persistence import digest, load, require, save
from scripts.writing_focus_v3 import BASELINE_ASSETS
from scripts.writing_paired_email_revision import SECONDARY, encoded, native_contexts, native_scores
from scripts.writing_reply_paragraph_transfer import LAYOUTS, mapped_body

SCHEMA = "chipmunk.writing-reply-layout-rescore/1"
ASSETS = {"primary": BASELINE_ASSETS, "second": SECONDARY}
CONTROL_COUNT = 24
POINT = .5
WINDOWS = 6
SCORE_TOLERANCE = 2e-5
SPEC = {
    "roles": "All 308 retained mapped layouts and 24 native MIME controls. The four cases with changed sentence boundaries belong to one previously tested source group. The remaining cases test exact source transfer, not new language.",
    "method": "Remeasure both commercially qualified classifiers on every reply context that differs from the same source's pinned native contexts, and on a deterministic 24-input subset of reused contexts. Every reported context stays within the original mapped reply before the trailing quote or signature.",
    "gate": "All native contexts have complete recorded responses, the exact-match subset preserves decisions, at least 90 percent of useful same-source locations remain in each layout, and each layout adds at most one human file highlight. No effect on the AI confidence estimate.",
}


def truth_range(original, current, bounds):
    old = [(index, char) for index, char in enumerate(original) if not char.isspace()]
    new = [(index, char) for index, char in enumerate(current) if not char.isspace()]
    require([char for _, char in old] == [char for _, char in new], "Reformatting changed non-whitespace source content")
    selected = [b for (a, _), (b, _) in zip(old, new, strict=True) if bounds[0] <= a < bounds[1]]
    require(selected, "No exact replacement coordinates survived reformatting")
    return [selected[0], selected[-1] + 1]


def reply_contexts(path, original, count):
    doc = extract(path, ".eml")
    body = mapped_body(doc, count)
    require(body is not None, "Email does not match its contiguous native reply")
    text = doc.text[:body[-1]["end"]]
    truth = truth_range(original["text"], text, original["truth"])
    sentences = segment(doc)
    saved = {w["sha256"]: w["current"] for w in original["windows"]}
    values, windows = {}, []
    for focus, group in zip(sentences, native_contexts(sentences), strict=True):
        if not all(any(p["start"] <= s.start < s.end <= p["end"] for p in body) for s in group):
            continue
        record, value = encoded(group)
        key = record["sha256"]
        values[key] = value
        windows.append({**record, "focus": [focus.start, focus.end], "reused_primary": saved.get(key)})
    require(windows, "Reply has no exact in-body contexts")
    return {"body": body, "body_sha256": text_sha(text), "truth": truth, "windows": windows}, text, values


def roster(panels, earlier):
    source, secondary, _ = source_inputs(panels)
    prior = load(earlier / "result.json")
    require(len(prior["cases"]) == 308 and len(prior["controls"]) == 24
            and digest(earlier / "freeze.json") == prior["freeze_sha256"], "Pinned layout replay differs")
    cases, texts, parity, new = [], {}, {}, set()
    for saved in prior["cases"]:
        index, count = saved["index"], saved["paragraph_count"]
        file = earlier / "eml" / f"{index:03d}-{count}.eml"
        original = source[index]
        require(saved["admitted"] and digest(file) == saved["file_sha256"], "Mapped reply bytes changed")
        record, _, values = reply_contexts(file, original, count)
        for w in record["windows"]:
            key = w["sha256"]
            if w["reused_primary"] is None:
                new.add(key)
            else:
                require(key in secondary, "A reused second response is missing")
                parity.setdefault(key, [w["reused_primary"], secondary[key]["score"]])
        cases.append({"case": index, "layout": count, "sha256": saved["file_sha256"],
                      "source_group": original["group"], "generated": original["kind"] not in HUMAN,
                      "segmentation_changed": bool(saved.get("unavailable")),
                      "original_useful": saved["baseline_useful"], "original_human_flag": saved["baseline_human_flag"],
                      **record})
        texts.update(values)
    controls = []
    for i in range(12):
        for delivery in ("inline_quote", "html"):
            path = earlier / "eml" / f"control-{i:02d}-{delivery}.eml"
            require(mapped_body(extract(path, ".eml"), 2) is None, "Unsupported material entered the reply check")
            controls.append(digest(path))
    require(controls == prior["controls"], "Excluded MIME files differ from the pinned controls")
    require(len(parity.keys() - new) >= CONTROL_COUNT and new, "Fresh and matched inputs are both required")
    spots = sorted(parity.keys() - new, key=lambda key: text_sha(SCHEMA + key))[:CONTROL_COUNT]
    inputs = {key: texts[key] for key in sorted(new | set(spots))}
    return cases, inputs, {key: parity[key] for key in spots}, secondary


def protocol(earlier, current, second, cases, inputs, parity):
    assets = {arm: {name: digest(path / name) for name in ASSETS[arm]}
              for arm, path in (("primary", current), ("second", second))}
    require(assets == ASSETS, "A licensed checkpoint no longer matches its reviewed assets")
    return {"schema": SCHEMA, "spec": SPEC, "code_sha256": digest(Path(__file__)),
            "prior_layout_freeze": digest(earlier / "freeze.json"), "prior_layout_outcomes": digest(earlier / "result.json"),
            "checkpoints": assets, "case_commitment": text_sha(repr(cases)),
            "forward_input_hashes": list(inputs), "parity_input_hashes": list(parity), "max_score_delta": SCORE_TOLERANCE}


def score(folder, lock, arm, path, inputs):
    import torch

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    started = monotonic()
    result, forwards = native_scores(path, inputs, second=arm == "second")
    require(set(result) == set(inputs), "Every planned exact model input needs a measured response")
    save(folder / f"{arm}.json", {"freeze_sha256": lock, "responses": result,
                                  "forwards": forwards, "elapsed_s": round(monotonic() - started, 3)})
    print(f"{arm}: {len(result)} actual inputs in {forwards} forwards", flush=True)


def report(folder, earlier, cases, parity, secondary):
    lock = digest(folder / "freeze.json")
    a, b = (load(folder / f"{arm}.json") for arm in ASSETS)
    require(all(s["freeze_sha256"] == lock for s in (a, b)), "Scores detached from their exact plan")
    primary, second = a["responses"], b["responses"]
    deltas = {arm: max(abs(scores[key]["score"] - expected[index]) for key, expected in parity.items())
              for index, (arm, scores) in enumerate(zip(ASSETS, (primary, second), strict=True))}
    require(all((scores[key]["score"] >= POINT) == (expected[index] >= POINT)
                and one_complete_window(scores[key]["coverage"]) for index, scores in enumerate((primary, second))
                for key, expected in parity.items()), "Previously observed decision changed on an exact input")
    require(max(deltas.values()) <= SCORE_TOLERANCE, "Previously observed exact response did not reproduce")
    counts, new_false_groups = defaultdict(Counter), defaultdict(set)
    outcomes = []
    for case in cases:
        path = earlier / "eml" / f"{case['case']:03d}-{case['layout']}.eml"
        doc = extract(path, ".eml")
        body = doc.text[:case["body"][-1]["end"]]
        chosen, fresh_count = [], 0
        for w in case["windows"]:
            key, previous = w["sha256"], w["reused_primary"]
            if previous is None:
                fresh_count += 1
                aa, bb = primary[key], second[key]
                require(all(one_complete_window(row["coverage"]) for row in (aa, bb)), "A changed input did not finish")
                first, follow = aa["score"], bb["score"]
            else:
                first, follow = previous, secondary[key]["score"]
            if first >= POINT:
                chosen.append((w, follow))
        shared = [window for window, follow in chosen[:WINDOWS] if follow >= POINT]
        hit = case["generated"] and any(useful(body, case["truth"], w["spans"]) for w in shared)
        flag = not case["generated"] and bool(shared)
        if flag and not case["original_human_flag"]:
            new_false_groups[case["layout"]].add(case["source_group"])
        counts[case["layout"]].update({"files": 1, "generated": case["generated"], "body_contexts": len(case["windows"]),
                "fresh_occurrences": fresh_count, "segmentation_changed": case["segmentation_changed"],
                "old_useful": case["original_useful"], "new_useful": hit, "old_human_flags": case["original_human_flag"],
                "new_human_flags": flag, "added_human_flags": flag and not case["original_human_flag"]})
        outcomes.append({"case": case["case"], "layout": case["layout"], "fresh_occurrences": fresh_count,
                         "shared_hashes": [w["sha256"] for w in shared], "useful": hit, "human_flag": flag})
    verdicts = {layout: bool(counts[layout]["files"] and counts[layout]["new_useful"] >= .9 * counts[layout]["old_useful"]
                            and counts[layout]["added_human_flags"] <= 1) for layout in LAYOUTS}
    public = {"schema": SCHEMA, "spec": SPEC, "freeze_sha256": lock, "parity_max_deltas": deltas,
              "layout_outcomes": counts, "new_human_groups": {layout: len(groups) for layout, groups in new_false_groups.items()},
              "layouts_pass": verdicts, "all_pass": all(verdicts.values()),
              "unique_fresh_inputs": len(primary) - len(parity), "parity_input_count": len(parity), "excluded_controls": 24,
              "actual_forwards": {arm: row["forwards"] for arm, row in zip(ASSETS, (a, b), strict=True)}}
    save(folder / "decisions.json", outcomes)
    save(folder / "aggregate.json", public)
    print(public, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "primary", "second", "report"))
    for key in ("folder", "panels", "earlier", "current", "second"):
        parser.add_argument(key, type=Path)
    args = parser.parse_args()
    require(os.getenv("HF_HUB_OFFLINE") == os.getenv("TRANSFORMERS_OFFLINE") == "1", "Only pinned local model access is supported")
    cases, inputs, parity, secondary = roster(args.panels, args.earlier)
    pin = protocol(args.earlier, args.current, args.second, cases, inputs, parity)
    path = args.folder / "freeze.json"
    if args.phase == "freeze":
        args.folder.mkdir(mode=0o700)
        save(path, pin)
        print(f"Frozen {len(cases)} replies, {len(inputs)-len(parity)} changed and {len(parity)} parity inputs per classifier")
        return
    require(load(path) == pin, "Frozen reply or weight plan changed")
    if args.phase == "report":
        report(args.folder, args.earlier, cases, parity, secondary)
    else:
        score(args.folder, digest(path), args.phase, args.current if args.phase == "primary" else args.second, inputs)


if __name__ == "__main__":
    main()
