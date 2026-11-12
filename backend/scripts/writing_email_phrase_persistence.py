"""Test source-mapped phrase evidence in short email with a frozen local comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

from app.detectors.text.extract import extract, text_hash
from app.detectors.text.pipeline import windows
from app.detectors.text.segment import segment
from scripts.writing_focus_v3 import BASELINE_ASSETS

SCHEMA = "chipmunk.writing-email-phrase-persistence/1"
REVISION = "f1795c86806e6838d4afa33d0b1427f8430c9615"
WORDS = re.compile(r"[^\W_]+(?:[’'-][^\W_]+)*")
WIDTH = 8
BASELINE_LIMIT = 6
METHOD = {
    "baseline": "First six positive native three-sentence contexts in source order.",
    "candidate_inputs": "Score complete three-sentence windows truncated at the current MIME body-role boundary. Use body roles only and keep the same sentence coordinates.",
    "token_score": "Combined AI-vs-human logit margin at the contextual token positions. Average available margins by exact whole-word source range.",
    "controls": "Within each eight-word candidate span, the unsmoothed arm scores the center-left word. The persistence arm takes the smaller of the mean margin across the four center words and the mean across all eight.",
    "review": "Each candidate arm keeps at most its single strongest qualifying eight-word span. Separate review points are strictly above the second-largest human calibration source-group maximum. No probability claim.",
    "advance": "At least two newly localized body edit cases from distinct groups, no lost current useful localization, and at most one extra flagged human source group over current. Phrase persistence must find more useful locations or flag fewer human groups than unsmoothed token review.",
    "overlap": "Useful localization needs non-whitespace source-character intersection over union of at least 0.30 with the admitted generated phrase.",
}
ARMS = ("baseline", "raw_token", "phrase_persistence")


def require(condition: bool, reason: str) -> None:
    if not condition:
        raise ValueError(reason)


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load(path: Path):
    return json.loads(path.read_bytes())


def cases(path: Path):
    return [json.loads(line) for line in path.read_bytes().splitlines()]


def save(path: Path, value) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def span_values(tokens, sentence_text: str, start: int) -> list[dict]:
    """Calculate fixed-width source-word ranges and both measurements."""
    readings = []
    for word in WORDS.finditer(sentence_text):
        a, b = word.span()
        by_character = [[score for x, y, score in tokens if x <= pos < y] for pos in range(a, b)]
        require(all(by_character), "A focus word has incomplete native token coverage")
        mean = sum(sum(values) / len(values) for values in by_character) / (b - a)
        readings.append((a + start, b + start, mean))
    output = []
    for i in range(len(readings) - WIDTH + 1):
        group = readings[i:i + WIDTH]
        scores = [entry[2] for entry in group]
        output.append({"span": [group[0][0], group[-1][1]], "raw_token": scores[3],
                       "phrase_persistence": min(sum(scores[2:6]) / 4, sum(scores) / WIDTH)})
    return output


def run_native(clf, contents: list[str]) -> dict:
    import torch

    outputs = {}
    forwards, max_error = 0, 0.
    captured = []
    hook = clf.model.model.register_forward_hook(lambda _model, _args, output: captured.append(output[0]))
    try:
        for start in range(0, len(contents), 6):
            batch = contents[start:start + 6]
            inputs = clf.tok(batch, padding=True, truncation=False, return_offsets_mapping=True,
                             return_special_tokens_mask=True, return_tensors="pt")
            attention = inputs["attention_mask"] != 0
            ordinary = attention & (inputs["special_tokens_mask"] == 0)
            require(inputs["input_ids"].shape[1] <= 512, "Email context exceeds native one-window limit")
            with torch.inference_mode():
                native = clf.model(inputs["input_ids"], attention_mask=inputs["attention_mask"])["logits"].float()
                require(len(captured) == 1, "Expected one native hidden-state tensor")
                token_logits = clf.model.classifier(captured.pop()).float()
                reduced = (token_logits * attention.unsqueeze(-1)).sum(1) / attention.sum(1).unsqueeze(-1)
                error = float((native - reduced).abs().max())
                require(error < 3e-5 and torch.isfinite(token_logits).all(), "Native logits did not reproduce from token head")
                margins = torch.logsumexp(token_logits[:, :, 1:], dim=-1) - token_logits[:, :, 0]
                responses = 1 - torch.softmax(native, dim=-1)[:, 0]
                max_error = max(max_error, error)
            for index, content in enumerate(batch):
                ids = inputs["input_ids"][index][attention[index]].tolist()
                require(ids == clf.tok(content, truncation=False, verbose=False)["input_ids"], "Native token IDs differ")
                offsets = inputs["offset_mapping"][index][ordinary[index]].tolist()
                output = [(a, b, float(score)) for (a, b), score in zip(offsets, margins[index][ordinary[index]], strict=True) if a < b]
                outputs[content] = {"raw": float(responses[index]), "offsets": output,
                                    "input_sha256": text_hash(content), "token_ids_sha256": text_hash(json.dumps(ids))}
            forwards += 1
    finally:
        hook.remove()
    return outputs, forwards, max_error


def measure(clf, doc, *, audit=False):
    sents = segment(doc)
    roles = []
    for sentence in sents:
        match = [row["quote_role"] for row in doc.source_map if row["start"] <= sentence.start < sentence.end <= row["end"]]
        require(len(match) == 1, "Sentence does not map to one exact email role")
        roles.append(match[0])
    baseline = windows(sents, 3)
    groups = [sents[max(0, i - 1):i + 2] for i in range(len(sents))]
    body_inputs = []
    for i, sentence in enumerate(sents):
        if roles[i] == "body" and len(list(WORDS.finditer(sentence.text))) >= WIDTH:
            preceding = sents[i - 1:i] if i and roles[i - 1] == "body" else []
            following = sents[i + 1:i + 2] if i + 1 < len(sents) and roles[i + 1] == "body" else []
            body_inputs.append((sentence, preceding, following))
    measured, forwards, error = run_native(clf, list(dict.fromkeys([
        *baseline, *(" ".join(s.text for s in [*prev, focus, *following]) for focus, prev, following in body_inputs),
    ])))
    if audit:
        exact = clf.predict(baseline, batch=6, check=lambda _label: None)
        require(all(abs(measured[content]["raw"] - response) < 1e-6 for content, response in zip(baseline, exact, strict=True)),
                "Independent direct baseline response differs")
    existing = [{"spans": [[s.start, s.end] for s in group], "raw": measured[content]["raw"],
                 "input_sha256": measured[content]["input_sha256"]} for group, content in zip(groups, baseline, strict=True)]
    candidates = []
    for focus, prev, following in body_inputs:
        group = [*prev, focus, *following]
        content = " ".join(s.text for s in group)
        left = sum(len(s.text) + 1 for s in prev)
        right = left + len(focus.text)
        bound = [(max(a, left) - left, min(b, right) - left, value)
                 for a, b, value in measured[content]["offsets"] if a < right and b > left]
        require(content[left:right] == focus.text and all(0 <= a < b <= len(focus.text) for a, b, _ in bound),
                "Native token offsets do not bind to exact sender focus")
        spans = span_values(bound, focus.text, focus.start)
        candidates.extend({**row, "focus": [focus.start, focus.end], "context_spans": [[s.start, s.end] for s in group],
                           "input_sha256": measured[content]["input_sha256"],
                           "token_ids_sha256": measured[content]["token_ids_sha256"]} for row in spans)
    return {"baseline": existing, "candidates": candidates, "forwards": forwards, "max_error": error}


def score(classifier, run: Path, rows):
    out = []
    for row in rows:
        path = run / row["fixture"]
        require(digest(path) == row["eml_sha256"], "Frozen email fixture changed")
        doc = extract(path, ".eml")
        require(doc.text == row["text"] and text_hash(doc.text) == row["text_sha256"], "Decoded email differs from frozen source")
        roles = [{"span": [part["start"], part["end"]], "role": part["quote_role"]} for part in doc.source_map]
        require(roles == row["roles"] and all(part["mime_part"] == doc.meta["email"]["body_part"] for part in doc.source_map),
                "MIME selection or exact role coordinates changed")
        out.append({"case": row["case"], "group": row["group"], "roles": roles, **measure(classifier, doc, audit=not out)})
        print("scored", len(out), len(rows), flush=True)
    return out


def thresholds(records):
    by_group = defaultdict(lambda: {arm: -math.inf for arm in ARMS[1:]})
    for record in records:
        for arm in ARMS[1:]:
            if record["candidates"]:
                by_group[record["group"]][arm] = max(by_group[record["group"]][arm], *(r[arm] for r in record["candidates"]))
    require(len(by_group) >= 4, "Too few previously used human calibration groups")
    return {arm: sorted((row[arm] for row in by_group.values()), reverse=True)[1] for arm in ARMS[1:]}


def select(record: dict, points: dict) -> dict:
    selected = {"baseline": [item["spans"] for item in record["baseline"] if item["raw"] >= .5][:BASELINE_LIMIT]}
    for arm in ARMS[1:]:
        best = max(record["candidates"], key=lambda row: (row[arm], -row["span"][0]), default=None)
        selected[arm] = [[best["span"]]] if best is not None and best[arm] > points[arm] else []
    return selected


def readout(rows, responses, points):
    totals = {arm: Counter() for arm in ARMS}
    false_groups, gained_groups = defaultdict(set), defaultdict(set)
    for case, record in zip(rows, responses, strict=True):
        require(case["case"] == record["case"], "Response order differs from frozen cases")
        text = case["text"]

        def chars(spans, text=text):
            return {i for a, b in spans for i in range(a, b) if not text[i].isspace()}

        expected = chars(case["ai_spans"])
        body = chars([item["span"] for item in record["roles"] if item["role"] == "body"])
        quote = chars([item["span"] for item in record["roles"] if item["role"] == "quoted"])
        geometry = select(record, points)
        outcomes = {}
        for arm in ARMS:
            regions = [chars(window) for window in geometry[arm]]
            useful = bool(expected and any(10 * len(region & expected) >= 3 * len(region | expected) for region in regions))
            flagged_body = any(region & body for region in regions)
            flagged_quote = any(region & quote for region in regions)
            assessed = bool(record["baseline"] if arm == "baseline" else record["candidates"])
            totals[arm].update({f'{case["kind"]}_cases': 1, f'{case["kind"]}_assessed': int(assessed),
                                f'{case["kind"]}_useful_body': int(useful), f'{case["kind"]}_flagged_body': int(flagged_body),
                                f'{case["kind"]}_flagged_quote': int(flagged_quote)})
            outcomes[arm] = useful
            if not expected and flagged_body:
                false_groups[arm].add(case["group"])
        if expected:
            for arm in ARMS[1:]:
                gain = outcomes[arm] and not outcomes["baseline"]
                totals[arm].update(new_body_location=int(gain), lost_body_location=int(outcomes["baseline"] and not outcomes[arm]))
                if gain:
                    gained_groups[arm].add(case["group"])
    for arm in ARMS:
        totals[arm].update(false_body_source_groups=len(false_groups[arm]), gained_source_groups=len(gained_groups[arm]))
    return {arm: dict(totals[arm]) for arm in ARMS}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze-calibration", "calibrate", "freeze-holdout", "holdout"))
    parser.add_argument("run", type=Path)
    parser.add_argument("snapshot", type=Path)
    args = parser.parse_args()
    run = args.run
    phase = "holdout" if args.stage.endswith("holdout") else "calibration"
    case_file = run / f"private-{phase}.jsonl"
    rows = cases(case_file)
    require(args.snapshot.name == REVISION, "Expected pinned writing checkpoint")
    assets = {name: digest(args.snapshot / name) for name in BASELINE_ASSETS}
    require(assets == BASELINE_ASSETS, "Pinned checkpoint assets differ")
    lock = {"schema": SCHEMA, "method": METHOD, "code": digest(Path(__file__)), "cases_sha256": digest(case_file),
            "checkpoint_revision": REVISION, "assets": assets}
    if phase == "holdout":
        calibration_groups = {row["group"] for row in cases(run / "private-calibration.jsonl")}
        require(not calibration_groups & {row["group"] for row in rows}, "Calibration and holdout groups overlap")
        lock["calibration_sha256"] = digest(run / "calibration.json")
        lock["source_intake_sha256"] = digest(run / "source.json")
    freeze = run / f"freeze-{phase}.json"
    if args.stage.startswith("freeze-"):
        save(freeze, lock)
        print(digest(freeze), flush=True)
        return
    require(load(freeze) == lock, "Frozen scorer, assets or inputs changed")
    import torch

    from app.detectors.text.classifier import FourClassTextClassifier

    torch.set_num_threads(1)
    started = time.monotonic()
    classifier = FourClassTextClassifier.from_snapshot(args.snapshot, "cpu")
    responses = score(classifier, run, rows)
    response_path = run / f"private-{phase}-responses.json"
    save(response_path, responses)
    result = {"schema": SCHEMA, "method": METHOD, "scored_cases": len(rows), "scored_groups": len({r["group"] for r in rows}),
              "native_forwards": sum(r["forwards"] for r in responses) + math.ceil(len(responses[0]["baseline"]) / 6),
              "max_logit_reduction_error": max(r["max_error"] for r in responses),
              "seconds": round(time.monotonic() - started, 3), "freeze_sha256": digest(freeze), "responses_sha256": digest(response_path)}
    if phase == "calibration":
        result["points"] = thresholds(responses)
    else:
        result["counts"] = readout(rows, responses, load(run / "calibration.json")["points"])
        result["calibration_sha256"] = lock["calibration_sha256"]
    save(run / f"{phase}.json", result)
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
