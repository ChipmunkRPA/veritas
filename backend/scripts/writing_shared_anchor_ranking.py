"""Compare local sentence rankings with identical outer sentences as anchors."""
from __future__ import annotations

import argparse
import math
from collections import Counter, defaultdict
from pathlib import Path
from time import monotonic

from app.detectors.text.extract import from_string
from app.detectors.text.segment import segment
from app.evidence import document_matched_context as serving
from scripts import writing_two_sided_context as native
from scripts import writing_two_sided_transfer as prior
from scripts.writing_focus_v3 import digest

SCHEMA = "chipmunk.writing-shared-anchor-ranking/1"
NATIVE_WINDOWS = ((0, 1, 2), (1, 2, 3))
SHARED_WINDOWS = ((0, 1, 3), (0, 2, 3))
RANKS = ("native_rank", "shared_rank", "agreement_rank")
MARKS = ("matched", "matched_top_one", "matched_agreement")
METHODS = (*RANKS, *MARKS)


def unique_top(scores: tuple[float, float]) -> int | None:
    first, second = scores
    if any(not math.isfinite(score) or not 0 <= score <= 1 for score in scores):
        raise ValueError("Ranking requires two finite raw responses between zero and one")
    return None if first == second else int(second > first)


def selections(native_scores: tuple[float, float], shared_scores: tuple[float, float]) -> dict[str, tuple[int, ...]]:
    native_top, shared_top = unique_top(native_scores), unique_top(shared_scores)
    agreed = native_top if native_top == shared_top else None
    marks = tuple(index for index, score in enumerate(native_scores) if serving.selected(score))
    return {
        "native_rank": () if native_top is None else (native_top,),
        "shared_rank": () if shared_top is None else (shared_top,),
        "agreement_rank": () if agreed is None else (agreed,),
        "matched": marks,
        "matched_top_one": (native_top,) if native_top in marks else (),
        "matched_agreement": (agreed,) if agreed in marks else (),
    }


def source_windows(text: str) -> list[tuple[tuple[int, int], ...]]:
    doc = from_string(text)
    sentences = segment(doc)
    if (len(sentences) != 4 or min(sentence.n_words for sentence in sentences[1:3]) < serving.MIN_FOCUS_WORDS
            or not serving._scope(doc.text, doc.source_map, "plain")):
        return []
    spans = [(sentence.start, sentence.end) for sentence in sentences]
    return [tuple(spans[index] for index in indexes) for indexes in (*NATIVE_WINDOWS, *SHARED_WINDOWS)]


def lock(directory: Path) -> dict:
    prior.fit()
    inputs = ("source.json", "private-holdout.jsonl", "protocol.json", "generation-freeze.json", "admission.json")
    return {
        "schema": SCHEMA,
        "inputs": {name: digest(directory / name) for name in inputs if (directory / name).is_file()},
        "implementation": digest(Path(__file__)),
        "classifier_assets": native.BASELINE_ASSETS,
        "matched_fit": prior.FIT_SHA256,
        "matched_selector": digest(Path(serving.__file__)),
        "native_scoring": digest(Path(native.__file__)),
    }


def summary(measured: list[tuple[dict, dict | None]]) -> dict:
    counts = defaultdict(lambda: defaultdict(Counter))
    groups = {method: set() for method in MARKS}
    paired = defaultdict(Counter)
    source_outcomes = defaultdict(dict)
    for case, responses in measured:
        arm, group = case["arm"], case["group"]
        if responses is None:
            reason = "generation_unavailable" if case["status"] != "constructed" else "outside_paired_scope"
            counts[arm]["coverage"][reason] += 1
            source_outcomes[group][arm] = {"status": reason}
            continue
        counts[arm]["coverage"]["assessed"] += 1
        native_scores = tuple(row["response"] for row in responses["windows"][:2])
        shared_scores = tuple(row["response"] for row in responses["windows"][2:])
        chosen = selections(native_scores, shared_scores)
        if chosen["native_rank"] and chosen["shared_rank"]:
            rank = "same_first" if chosen["agreement_rank"] else "different_first"
        else:
            rank = "tie"
        counts[arm]["coverage"][rank] += 1
        outcomes = {}
        targets = case["ai_spans"] or case["ordinary_spans"]
        for method, indexes in chosen.items():
            ranges = [responses["focus_spans"][i] for i in indexes]
            metrics = native.evaluate_regions(case["text"], ranges, case["ai_spans"], case["ordinary_spans"])
            outcome = {"locations": len(ranges), "selected": bool(ranges), "target_selected": any(
                list(span) in targets for span in ranges
            ), "useful": metrics["useful"], "wrong_location": metrics["wrong_location"]}
            counts[arm][method].update({key: int(value) for key, value in outcome.items()})
            outcomes[method] = outcome
            if method in MARKS and ranges and not case["ai_spans"]:
                groups[method].add(group)
        comparison = "useful" if case["ai_spans"] else "selected"
        base = outcomes["matched_top_one"][comparison]
        changed = outcomes["matched_agreement"][comparison]
        paired[arm][f"{int(base)}->{int(changed)}"] += 1
        source_outcomes[group][arm] = {"status": "assessed", "rank": rank, "methods": outcomes}
    return {"counts": counts, "human_source_groups_marked": {key: len(value) for key, value in groups.items()},
            "agreement_vs_matched_top_one": paired, "source_outcomes": source_outcomes}


def score(directory: Path, snapshot: Path) -> None:
    expected = prior.read(directory / "frozen.json")
    if expected != lock(directory) or not {"source.json", "private-holdout.jsonl", "protocol.json"} <= expected["inputs"].keys():
        raise ValueError("Writing ranking inputs differ from the freeze")
    cases = native.read_cases(directory / "private-holdout.jsonl")
    windows = {case["case"]: source_windows(case["text"]) for case in cases if case["status"] == "constructed"}
    inputs = dict.fromkeys(native.native_input(case["text"], spans)
                           for case in cases for spans in windows.get(case["case"], ()))
    start = monotonic()
    classifier = native.open_classifier(snapshot)
    forwards = [0]
    hook = classifier.model.register_forward_pre_hook(lambda *_: forwards.__setitem__(0, forwards[0] + 1))
    try:
        values = native.native_responses(classifier, list(inputs), lambda: None)
    finally:
        hook.remove()
    measured = []
    for case in cases:
        spans = windows.get(case["case"])
        if not spans:
            measured.append((case, None))
            continue
        rows = []
        for window in spans:
            text = native.native_input(case["text"], window)
            response, token_sha = values[text]
            rows.append({"spans": window, "response": response, "source_text_sha256": native.text_sha(text),
                         "input_token_sha256": token_sha})
        measured.append((case, {"focus_spans": [row[1] for row in spans[:2]], "windows": rows}))
    native.write(directory / "private-native.json", {case["case"]: response for case, response in measured})
    native.write(directory / "aggregate.json", {
        "schema": SCHEMA,
        **summary(measured),
        "model_forward_calls": forwards[0],
        "unique_inputs": len(inputs),
        "elapsed_seconds": round(monotonic() - start, 3),
        "commitments": {"freeze": digest(directory / "frozen.json"), "matched_fit": prior.FIT_SHA256,
                        "native": digest(directory / "private-native.json")},
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "run"))
    parser.add_argument("directory", type=Path)
    parser.add_argument("--snapshot", type=Path)
    args = parser.parse_args()
    if args.stage == "freeze":
        native.write(args.directory / "frozen.json", lock(args.directory))
    elif args.snapshot is None:
        parser.error("A local classifier snapshot is required")
    else:
        score(args.directory, args.snapshot)


if __name__ == "__main__":
    main()
