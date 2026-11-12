"""Evaluate exact email revision evidence with two paired classifier responses."""

from __future__ import annotations

import argparse
import gc
import math
import time
from collections import Counter, defaultdict
from pathlib import Path

from app.detectors.text.extract import extract, text_hash
from app.detectors.text.segment import segment
from app.evidence.document_matched_context import one_complete_window
from app.evidence.document_revision import differences, tokens
from scripts.writing_email_phrase_persistence import cases, digest, load, require, save
from scripts.writing_focus_v3 import BASELINE_ASSETS

SCHEMA = "chipmunk.writing-paired-email-revision/1"
SECONDARY = {
    "config.json": "dc4511bb21dfa4991688aba8548086c8fa6947d94018ac1a8d09ae9843749703",
    "tokenizer.json": (
        "6c8aaa9a542084f2457eab775d4eeb51f92a70c0fd9de28d5edb0ddec3c08d30"
    ),
    "model.safetensors": "e7f7a617ad4af3abd8bcb939d564d2a2ea36fdb24b893177ae10c12ebe869361",
    "detector-config.json": "e72a616a5cbb0bbbe7b75d3d5e40af3c5cafdb0f0a8299ce04c5921e046619c7",
    "LICENSE": "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4",
}
METHOD = {
    "baseline": "First six serving three-sentence context inputs with raw AI-like response at least .5 on the later body, including retained quotes.",
    "geometry": "Exact Unicode tokens within each mapped role. No case folding. Merge adjacent edit opcodes separated by at most two unchanged tokens in both versions. Report original offsets and changed tokens separately.",
    "scope": "Each file has plain-text body plus at most one separately marked quote, and no body paragraph after the quote. Classifier comparisons require a changed focus in each version and identical outer sentences.",
    "direction": "Flag experimental corroboration only if both classifiers independently at least double their model-specific AI-like odds on the later matched context. This is a paired response direction and adds zero confidence weight.",
    "outcomes": "Useful localization means non-whitespace character IoU at least .30 and recall at least .50. Geometry-only true edits remain separate from authorship false alarms.",
    "advance": "At least two additional correctly located AI edits from distinct groups, no loss of current useful locations and at most one additional false-alarm group over the current check. This is source-exposed development only.",
}


def native_contexts(sentences):
    return [sentences[max(0, i - 1):i + 2] for i in range(len(sentences))]


def encoded(group):
    text = " ".join(sentence.text for sentence in group)
    return {"spans": [[sentence.start, sentence.end] for sentence in group], "sha256": text_hash(text)}, text


def focus_context(sentences, extent):
    if extent is None or extent[0] == extent[1]:
        return None
    for i, sentence in enumerate(sentences):
        if sentence.start <= extent[0] < extent[1] <= sentence.end:
            group = native_contexts(sentences)[i]
            outer = (sentences[i - 1].text if i else None, sentences[i + 1].text if i + 1 < len(sentences) else None)
            return group, outer
    return None


def source(run, fixture):
    path = run / fixture["file"]
    require(digest(path) == fixture["sha256"], "Email fixture bytes differ")
    doc = extract(path, ".eml")
    email = doc.meta["email"]
    require(email["body_content_type"] == "text/plain" and text_hash(doc.text) == fixture["text_sha256"],
            "Decoded text or MIME body changed")
    mapping = [(row["quote_role"], row["start"], row["end"]) for row in doc.source_map]
    require(mapping and [list(item) for item in mapping] == fixture["roles"]
            and all(row["mime_part"] == email["body_part"] and row.get("source_ranges") for row in doc.source_map)
            and all(item[0] in {"body", "quoted"} for item in mapping)
            and sum(item[0] == "quoted" for item in mapping) <= 1
            and all(role == "body" for role, _, _ in mapping[:-1]), "Email body or quote mapping changed")
    sentences = segment(doc)
    body = [s for s in sentences if any(role == "body" and x <= s.start < s.end <= y for role, x, y in mapping)]
    return doc, mapping, sentences, body


def pair(run, row):
    old, old_roles, _, old_body = source(run, row["earlier"])
    new, new_roles, current, new_body = source(run, row["later"])
    texts = {}

    def register(group):
        entry, text = encoded(group)
        texts[entry["sha256"]] = text
        return entry

    changes = {}
    for role in ("body", "quoted"):
        left = tokens(old.text, [(a, b) for label, a, b in old_roles if label == role])
        right = tokens(new.text, [(a, b) for label, a, b in new_roles if label == role])
        changes[role] = differences(left, right)
    for hunk in changes["body"]:
        before, after = focus_context(old_body, hunk["earlier"]), focus_context(new_body, hunk["later"])
        if before is not None and after is not None and before[1] == after[1]:
            hunk["comparison"] = {"earlier": register(before[0]), "later": register(after[0])}
    return {"case": row["case"], "group": row["group"], "body": changes["body"], "quoted": changes["quoted"],
            "earlier_roles": old_roles, "later_roles": new_roles,
            "current": [register(window) for window in native_contexts(current)]}, texts, new.text


def native_scores(path, texts, *, second):
    import torch

    from app.detectors.text.classifier import FourClassTextClassifier
    from app.detectors.text.coverage import classify_complete

    values = list(texts.values())
    forwards = 0
    if second:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        meta = load(path / "detector-config.json")
        require(meta["label_mapping"] == {"human": 0, "ai": 1} and meta["temperature"] > 0, "Unexpected comparison labels")
        model, info = AutoModelForSequenceClassification.from_pretrained(path, local_files_only=True, trust_remote_code=False,
                                                                         use_safetensors=True, output_loading_info=True)
        require(not any(info.get(key) for key in ("missing_keys", "mismatched_keys", "error_msgs", "unexpected_keys")),
                "Independent checkpoint did not load exactly")
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)

        def evaluate(encoded):
            nonlocal forwards
            forwards += 1
            logits = model(encoded["input_ids"], attention_mask=encoded["attention_mask"]).logits.float()
            require(torch.isfinite(logits).all() and logits.shape[-1] == 2, "Invalid independent model response")
            return (logits / meta["temperature"]).softmax(dim=-1)[:, 1]

        with torch.inference_mode():
            scores, coverage = classify_complete(values, tokenizer, "cpu", evaluate, max_len=512, batch=4)
    else:
        classifier = FourClassTextClassifier.from_snapshot(path, "cpu")

        def counted(*_args):
            nonlocal forwards
            forwards += 1

        hook = classifier.model.register_forward_hook(counted)
        try:
            scores = classifier.predict(values, batch=4, check=lambda _message: None)
            coverage = classifier.last_coverage
        finally:
            hook.remove()
    require(all(one_complete_window(row) for row in coverage), "An input did not receive one complete native reading")
    return {digest: {"score": response, "coverage": proof}
            for digest, response, proof in zip(texts, scores, coverage, strict=True)}, forwards


def rise(earlier, later):
    def odds(score):
        value = min(1 - 1e-7, max(1e-7, score))
        return value / (1 - value)

    return math.log(odds(later) / odds(earlier))


def assess(row, record, text, first, second):
    body = []
    for hunk in record["body"]:
        if comparison := hunk.get("comparison"):
            earlier, later = (comparison[part]["sha256"] for part in ("earlier", "later"))
            deltas = [rise(values[earlier]["score"], values[later]["score"]) for values in (first, second)]
            hunk["responses"] = {"earlier": [first[earlier], second[earlier]], "later": [first[later], second[later]],
                                 "log_odds_change": deltas, "both_at_least_double": min(deltas) >= math.log(2)}
            if hunk["responses"]["both_at_least_double"]:
                body.append([hunk["later"]])
    for window in record["current"]:
        window["response"] = first[window["sha256"]]
    current = [item["spans"] for item in record["current"] if item["response"]["score"] >= .5][:6]

    def chars(spans):
        return {i for a, b in spans if a is not None for i in range(a, b) if not text[i].isspace()}

    truth = chars(row["body_truth"])
    quote = chars(row["quote_truth"])
    body_extent = chars([(a, b) for role, a, b in record["later_roles"] if role == "body"])
    diff_body = [[hunk["later"]] for hunk in record["body"] if hunk["later"] is not None]
    diff_quote = [[hunk["later"]] for hunk in record["quoted"] if hunk["later"] is not None]

    def useful(regions, truth):
        return bool(truth) and any(10 * len(pixels & truth) >= 3 * len(pixels | truth)
                                   and 2 * len(pixels & truth) >= len(truth) for pixels in map(chars, regions))

    return {"current_highlight": bool(current), "current_body_highlight": any(chars(item) & body_extent for item in current),
            "current_useful_body": useful(current, truth),
            "body_diff_found": bool(record["body"]), "body_diff_useful": useful(diff_body, truth),
            "quote_diff_found": bool(record["quoted"]), "quote_diff_useful": useful(diff_quote, quote),
            "matched_body_context": any("comparison" in hunk for hunk in record["body"]),
            "paired_highlight": bool(body), "paired_useful_body": useful(body, truth)}


def main():
    from inspect import getsource

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "run"))
    parser.add_argument("directory", type=Path)
    parser.add_argument("current_snapshot", type=Path)
    parser.add_argument("second_snapshot", type=Path)
    args = parser.parse_args()
    run = args.directory
    case_file = run / "private-pairs.jsonl"
    rows = cases(case_file)
    identities = {"current": {name: digest(args.current_snapshot / name) for name in BASELINE_ASSETS},
                  "second": {name: digest(args.second_snapshot / name) for name in SECONDARY}}
    require(identities == {"current": BASELINE_ASSETS, "second": SECONDARY}, "One checkpoint differs from its frozen assets")
    lock = {"schema": SCHEMA, "method": METHOD, "code_sha256": digest(Path(__file__)), "case_sha256": digest(case_file),
            "source_sha256": digest(run / "source.json"), "weights": identities,
            "shared_comparison_sha256": text_hash(getsource(tokens) + getsource(differences))}
    freeze = run / "freeze.json"
    if args.stage == "freeze":
        save(freeze, lock)
        print(digest(freeze), flush=True)
        return
    require(load(freeze) == lock, "The pair experiment changed after freeze")
    import torch

    torch.set_num_threads(1)
    started = time.monotonic()
    records, inputs = [], {}
    for row in rows:
        record, texts, latest = pair(run, row)
        inputs.update(texts)
        records.append((row, record, latest))
    first, first_forwards = native_scores(args.current_snapshot, inputs, second=False)
    print(f"Current checkpoint scored {len(first)} distinct windows with {first_forwards} forwards", flush=True)
    gc.collect()
    paired_ids = {record["comparison"][side]["sha256"] for _, row, _ in records for record in row["body"]
                  if "comparison" in record for side in ("earlier", "later")}
    second, second_forwards = native_scores(args.second_snapshot, {key: value for key, value in inputs.items() if key in paired_ids},
                                           second=True)
    print(f"Independent checkpoint scored {len(second)} distinct windows with {second_forwards} forwards", flush=True)
    counts, groups = defaultdict(Counter), defaultdict(set)
    for row, record, latest in records:
        result = assess(row, record, latest, first, second)
        record["outcome"] = result
        counts[row["kind"]].update({key: int(value) for key, value in result.items()})
        counts[row["kind"]]["cases"] += 1
        for key, value in result.items():
            if value:
                groups[(row["kind"], key)].add(row["group"])
    evidence = run / "private-responses.json"
    save(evidence, [record for _, record, _ in records])
    report = {"schema": SCHEMA, "method": METHOD, "freeze_sha256": digest(freeze), "response_sha256": digest(evidence),
              "counts": {kind: dict(outcome) for kind, outcome in counts.items()},
              "groups": {kind: {key: len(groups[(kind, key)]) for key in outcome if key != "cases"}
                         for kind, outcome in counts.items()}, "cases": len(rows),
              "native_forwards": {"current": first_forwards, "second": second_forwards},
              "distinct_inputs": {"current": len(first), "second": len(second)}, "seconds": round(time.monotonic() - started, 3)}
    save(run / "result.json", report)


if __name__ == "__main__":
    main()
