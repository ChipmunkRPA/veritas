"""Replay the frozen speech-transcript comparisons from retained local receipts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

ARMS = ("raw", "paragraph", "isolated")
VARIANTS = {"original", "human_whole", "human_sentence", "qwen", "smol2"}
GENERATED = {"qwen", "smol2"}


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def text_sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def read(path: Path):
    return json.loads(path.read_bytes())


def records(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def require(check: bool, detail: str) -> None:
    if not check:
        raise ValueError(detail)


def masks(case: dict, answer: dict) -> tuple[dict[str, int], int, int]:
    """Use integer bitsets to independently measure the selected character extents."""
    text = case["text"]

    def bits(ranges):
        value = 0
        for a, b in ranges:
            require(type(a) is type(b) is int and 0 <= a < b <= len(text), "Invalid source extent")
            value |= sum(1 << i for i in range(a, b) if not text[i].isspace())
        return value

    current = answer["current"]
    scopes = [(row["start"], row["end"]) for row in current]
    require(all(a[1] <= b[0] for a, b in zip(scopes, scopes[1:])) and bits(scopes) == bits([(0, len(text))]),
            "Current positions are out of order or omit measured source characters")
    require(all(math.isfinite(row["raw"]) and 0 <= row["raw"] <= 1 for row in current), "Invalid current response")
    marked = [i for i, row in enumerate(current) if row["raw"] >= .5][:6]
    full_context = [position for i in marked for position in scopes[max(0, i - 1):i + 2]]
    ranked = []
    for a, b in case["paragraph_codepoints"][1:-1]:
        scores = [row["raw"] for row in current if a <= row["start"] < row["end"] <= b]
        if scores and max(scores) >= .5 and 18 <= len(text[a:b].split()) <= 145 and b - a <= 1200:
            ranked.append((-max(scores), a, b))
    ranked.sort()

    paragraph = {}
    for row in answer["paragraphs"]:
        (span,) = row["spans"]
        a, b = span["start"], span["end"]
        coverage = row["token_coverage"]
        response = float.fromhex(row["score_exact_hex"])
        require((a, b) not in paragraph and (a, b) in {(x, y) for _, x, y in ranked[:2]}
                and text_sha(text[a:b]) == row["input_sha256"] and response == row["score"]
                and math.isfinite(response) and 0 <= response <= 1
                and coverage["tokens"] == coverage["covered_tokens"] > 0
                and coverage["token_windows"] == 1 and coverage["omitted_tokens"] == 0,
                "Isolated response is not tied to the recorded whole paragraph")
        paragraph[a, b] = response
    supported = [r for r in ranked if paragraph.get(r[1:], -1) >= .5]
    return {"raw": bits(full_context), "paragraph": bits([ranked[0][1:]] if ranked else []),
            "isolated": bits([supported[0][1:]] if supported else [])}, bits(case["ai_spans"]), len(paragraph)


def verify(directory: Path, packet_path: Path) -> dict:
    freeze_path = directory / "freeze.json"
    originals_path = directory / "private-originals.jsonl"
    controls_path = directory / "private-controls.jsonl"
    positives_path = directory / "private-positives.jsonl"
    responses_path = directory / "private-responses.jsonl"
    source_path = directory / "source.json"
    source, packet = read(source_path), read(packet_path)
    require(sha(freeze_path) == packet["terminal"]["freeze_sha256"] == packet["preparation"]["freeze_sha256"]
            and sha(source_path) == packet["freeze"]["source_sha256"] and packet["source"] == source
            and read(freeze_path) == packet["freeze"] and sha(originals_path) == source["private_originals_sha256"]
            and sha(controls_path) == source["private_controls_sha256"]
            and sha(positives_path) == packet["preparation"]["case_sha256"]
            and sha(responses_path) == packet["terminal"]["responses_sha256"], "Frozen input or response commitments differ")
    cases, answers = records(controls_path) + records(positives_path), records(responses_path)
    sources = source["sources"]
    genres = {f"replication-{row['key']}": row["genre"] for row in sources}
    originals = {row["group"]: row for row in records(originals_path)}
    require(len({row["speaker"] for row in sources}) == len(genres) == source["source_groups"]
            and originals.keys() == genres.keys()
            and len(cases) == len(answers) == packet["terminal"]["planned_views"] == 5 * len(genres)
            and len({row["case"] for row in cases}) == len(cases)
            and Counter((row["article_group"], row["arm"]) for row in cases)
            == Counter((g, arm) for g in genres for arm in VARIANTS), "Missing or duplicate held case")
    for row in sources:
        ext = "pdf" if row["genre"] == "oral_history" else "html"
        native = directory / "sources" / f"{row['key']}.{ext}"
        require(sha(native) == row["source_sha256"] and native.stat().st_size == row["source_bytes"],
                "Retained publisher source bytes differ")
    counts = defaultdict(Counter)
    transitions = defaultdict(Counter)
    groups = defaultdict(set)
    views, checked = 0, 0
    for case, answer in zip(cases, answers, strict=True):
        group, arm = case["article_group"], case["arm"]
        genre = genres[group]
        require(case["case"] == answer["case"] and case["status"] == answer["status"], "Case and response differ")
        original = originals[group]
        source_text = original["text"]
        start, stop = original["edit_target_codepoints"]
        require(case["source_target"] == [start, stop]
                and case["source_sha256"] == original["text_sha256"] == text_sha(source_text),
                "Case cites a different original or edit target")
        if case["status"] == "unavailable":
            require(arm in GENERATED and not case["text"] and "current" not in answer, "Unavailable case carries evidence")
            for rule in ARMS:
                counts[genre, arm, rule]["preparation_unavailable"] += 1
            continue
        text = case["text"]
        end = len(text) - (len(source_text) - stop)
        require(case["status"] == "constructed" and [start, end] in case["paragraph_codepoints"][1:-1]
                and text[:start] == source_text[:start] and text[end:] == source_text[stop:]
                and case["ai_spans"] == ([[start, end]] if arm in GENERATED else [])
                and (arm != "original" or text == source_text)
                and all(start <= a < b <= end for a, b in case["ordinary_spans"])
                and text_sha(text) == case["text_sha256"] == answer["text_sha256"]
                == sha(directory / "article-files" / f"{case['case']}.txt"), "Scored file or known edit label differs")
        marks, truth, paragraphs = masks(case, answer)
        views += 1
        checked += paragraphs
        useful = {rule: bool(truth and 10 * (mask & truth).bit_count() >= 3 * (mask | truth).bit_count())
                  for rule, mask in marks.items()}
        before = useful["raw"] if truth else bool(marks["raw"])
        for rule, mask in marks.items():
            counts[genre, arm, rule].update(assessed=1, flagged=bool(mask), useful=useful[rule])
            after = useful[rule] if truth else bool(mask)
            transitions[genre, arm, rule][f"{int(before)}->{int(after)}"] += 1
            if mask and not truth:
                groups[genre, rule].add(group)
    for (genre, arm, rule), actual in counts.items():
        stored = packet["outcomes"][genre][arm][rule]
        require(all(stored[key] == value for key, value in actual.items())
                and dict(transitions[genre, arm, rule])
                == packet["paired_transitions_relative_to_raw"].get(genre, {}).get(arm, {}).get(rule, {}),
                "Saved paired outcomes differ from the independent replay")
    flagged_groups = {genre: {rule: len(groups[genre, rule]) for rule in ARMS} for genre in sorted(set(genres.values()))}
    require(views == packet["terminal"]["assessed_views"], "Verified view total differs from the terminal receipt")
    if published := packet.get("independent_replay"):
        require(published["human_flagged_source_groups"] == flagged_groups, "Public source group counts differ")
    return {"verified_views": views, "planned_views": len(cases), "verified_paragraph_inputs": checked,
            "verified_publisher_sources": len(sources),
            "verified_primary_rollups": len(counts), "human_flagged_source_groups": flagged_groups}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("packet", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.directory, args.packet), sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
