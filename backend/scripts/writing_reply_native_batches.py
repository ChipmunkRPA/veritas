# SPDX-License-Identifier: Apache-2.0
"""Recheck writing layouts using each reply's actual second-check batch."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from time import monotonic

from app.detectors.text.adapters import agreement
from app.evidence.document_matched_context import one_complete_window
from scripts.writing_checkpoint_agreement import HUMAN, frozen_inputs, text_sha, useful
from scripts.writing_email_phrase_persistence import digest, load, require, save
from scripts.writing_paired_email_revision import SECONDARY
from scripts.writing_reply_layout_rescore import POINT, WINDOWS, reply_contexts, roster

SCHEMA = "chipmunk.writing-reply-native-batches/1"
SPEC = {
    "scope": "Two-, three-, five- and six-paragraph plain-text replies followed by a mapped signature or quotation. All 308 retained fixtures and the 24 prior rejected MIME controls. Compare with the same original excerpt's in-body contexts.",
    "primary": "Reuse unchanged exact primary inputs only after the same-window spot check. Use freshly measured complete primary responses for changed hashes. Select the first six positive in-body contexts in native source order.",
    "second": "Run the app's second check on each exact ordered per-email request. Collapse identical complete requests only. Keep its standard native microbatch and token cap.",
    "gate": "Retain at least nine tenths of the useful source-comparison locations per layout and add at most one human highlight per layout. All selected inputs require complete measurements and every excluded control stays outside the second check.",
}


def key(hashes):
    return text_sha(json.dumps(hashes, separators=(",", ":")))


def prepare(panels, earlier, initial):
    cases, planned, parity, _ = roster(panels, earlier)
    initial_lock = load(initial / "freeze.json")
    a, b = (load(initial / f"{arm}.json") for arm in ("primary", "second"))
    require(set(planned) == set(initial_lock["forward_input_hashes"]) == set(a["responses"]) == set(b["responses"])
            and all(s["freeze_sha256"] == digest(initial / "freeze.json") for s in (a, b)),
            "Initial measurements differ from the frozen exact-input roster")
    delta = [max(abs(scores["responses"][h]["score"] - expected[n]) for h, expected in parity.items())
             for n, scores in enumerate((a, b))]
    crossings = [sum((scores["responses"][h]["score"] >= POINT) != (expected[n] >= POINT) for h, expected in parity.items())
                 for n, scores in enumerate((a, b))]
    require(delta[0] <= initial_lock["max_score_delta"] < delta[1] and crossings == [0, 0],
            "This transfer requires the measured response instability and a reproduced primary rule")
    saved_sources, values = frozen_inputs(panels)
    matches, layouts, requests = [], [], {}

    def register(windows):
        hashes = [w["sha256"] for w in windows[:WINDOWS]]
        pin = key(hashes)
        requests[pin] = hashes
        return pin

    for index, source in enumerate(saved_sources):
        windows = [w for w in source["windows"][:-1] if w["current"] >= POINT][:WINDOWS]
        matches.append({"source": index, "request": register(windows), "windows": windows})
    for case in cases:
        index = case["case"]
        path = earlier / "eml" / f"{index:03d}-{case['layout']}.eml"
        replay, text, material = reply_contexts(path, saved_sources[index], case["layout"])
        require(all(replay[name] == case[name] for name in replay), "Native reply context construction changed")
        values.update(material)
        selected = []
        for window in case["windows"]:
            response = window["reused_primary"]
            if response is None:
                saved = a["responses"][window["sha256"]]
                require(one_complete_window(saved["coverage"]), "Changed native input was only partly measured")
                response = saved["score"]
            if response >= POINT:
                selected.append(window)
        selected = selected[:WINDOWS]
        layouts.append({"case": case, "request": register(selected), "windows": selected, "body": text})
    candidate_requests = {h: [values[value] for value in hashes] for h, hashes in sorted(requests.items())}
    public = {"initial_max_raw_deltas": dict(zip(("primary", "second"), delta, strict=True)),
              "initial_point_crossings": dict(zip(("primary", "second"), crossings, strict=True)),
              "initial_parity_inputs": len(parity)}
    return saved_sources, matches, layouts, requests, candidate_requests, public


def pin(earlier, initial, snapshot, requests):
    assets = {name: digest(snapshot / name) for name in SECONDARY}
    require(assets == SECONDARY, "The second checkpoint does not match its licensed bytes")
    return {"schema": SCHEMA, "spec": SPEC, "runner_sha256": digest(Path(__file__)),
            "scoped_inputs": digest(earlier / "result.json"),
            "initial": {name: digest(initial / name) for name in ("freeze.json", "primary.json", "second.json")},
            "assets": assets, "requests": requests}


def measure(folder, snapshot, batches):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    started = monotonic()
    meta = load(snapshot / "detector-config.json")
    model, info = AutoModelForSequenceClassification.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False,
                                                                     use_safetensors=True, output_loading_info=True)
    require(not any(info.get(field) for field in ("missing_keys", "mismatched_keys", "error_msgs", "unexpected_keys")),
            "The second checkpoint did not load exactly")
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False)
    model.eval()
    actual_forwards = 0

    def forward(*_):
        nonlocal actual_forwards
        actual_forwards += 1

    readings = {}
    hook = model.register_forward_hook(forward)
    try:
        for i, (request, texts) in enumerate(batches.items(), 1):
            require(key([text_sha(text) for text in texts]) == request, "A native request has different inputs")
            if texts:
                scores, coverage = agreement.score_inputs(texts, (tokenizer, model, meta["temperature"]), "cpu", check=lambda _: None)
                require(all(one_complete_window(window) for window in coverage), "A native reply was only partly assessed")
                readings[request] = [{"score_hex": value.hex(), "tokens": window["tokens"]}
                                     for value, window in zip(scores, coverage, strict=True)]
            else:
                readings[request] = []
            if i % 15 == 0:
                print(f"Measured {i}/{len(batches)} exact reply requests", flush=True)
    finally:
        hook.remove()
    save(folder / "responses.json", {"freeze_sha256": digest(folder / "freeze.json"), "readings": readings,
                                     "actual_forwards": actual_forwards, "seconds": round(monotonic() - started, 3)})


def hits(item, readings):
    return [window for window, response in zip(item["windows"], readings[item["request"]], strict=True)
            if float.fromhex(response["score_hex"]) >= POINT]


def report(folder, sources, original, layouts, requests, instability):
    receipt = load(folder / "responses.json")
    require(receipt["freeze_sha256"] == digest(folder / "freeze.json") and set(receipt["readings"]) == set(requests),
            "Native responses differ from the frozen reply batches")
    readings = receipt["readings"]
    old = {}
    for item, source in zip(original, sources, strict=True):
        generated = source["kind"] not in HUMAN
        shared = hits(item, readings)
        old[item["source"]] = (generated and any(useful(source["text"], source["truth"], w["spans"]) for w in shared),
                               not generated and bool(shared))
    counts, false_groups, rows = defaultdict(Counter), defaultdict(set), []
    for row in layouts:
        case = row["case"]
        hit_before, flag_before = old[case["case"]]
        shared = hits(row, readings)
        useful_after = case["generated"] and any(useful(row["body"], case["truth"], w["spans"]) for w in shared)
        flag_after = not case["generated"] and bool(shared)
        if flag_after and not flag_before:
            false_groups[case["layout"]].add(case["source_group"])
        counts[case["layout"]].update({
            "files": 1, "generated": case["generated"], "candidate_contexts": len(row["windows"]),
            "corroborated_contexts": len(shared), "segmentation_changed": case["segmentation_changed"],
            "old_useful": hit_before, "new_useful": useful_after, "old_human_flags": flag_before,
            "new_human_flags": flag_after, "added_human_flags": flag_after and not flag_before,
        })
        rows.append({"source": case["case"], "layout": case["layout"], "request": row["request"],
                     "corroborated": [window["sha256"] for window in shared], "useful": useful_after, "human_flag": flag_after})
    decisions = {str(layout): 10 * row["new_useful"] >= 9 * row["old_useful"] and row["added_human_flags"] <= 1
                 for layout, row in counts.items()}
    save(folder / "decisions.json", rows)
    result = {"schema": SCHEMA, "freeze_sha256": receipt["freeze_sha256"], "initial_measurements": instability,
              "actual_forwards": receipt["actual_forwards"], "duration_s": receipt["seconds"],
              "distinct_requests": len(requests), "second_context_inputs": sum(len(row) for row in requests.values()),
              "layout_outcomes": counts, "new_human_groups": {layout: len(value) for layout, value in false_groups.items()},
              "layouts_pass": decisions, "all_pass": all(decisions.values()), "excluded_controls": 24}
    save(folder / "aggregate.json", result)
    print(json.dumps(result, sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "score", "report"))
    for key_name in ("folder", "panels", "earlier", "initial", "second"):
        parser.add_argument(key_name, type=Path)
    args = parser.parse_args()
    require(os.getenv("HF_HUB_OFFLINE") == os.getenv("TRANSFORMERS_OFFLINE") == "1", "Only local inference is supported")
    sources, old, layouts, requests, values, instability = prepare(args.panels, args.earlier, args.initial)
    frozen = pin(args.earlier, args.initial, args.second, requests)
    path = args.folder / "freeze.json"
    if args.phase == "freeze":
        args.folder.mkdir(mode=0o700)
        save(path, frozen)
        print(f"Frozen {len(requests)} exact requests from {len(layouts)} reformatted replies and {len(old)} source excerpts")
        return
    require(load(path) == frozen, "Frozen exact reply batches differ")
    if args.phase == "score":
        measure(args.folder, args.second, values)
    else:
        report(args.folder, sources, old, layouts, requests, instability)


if __name__ == "__main__":
    main()
