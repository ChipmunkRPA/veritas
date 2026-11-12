"""Replay adjacent writing contexts on committed notice and article panels."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

SCHEMA = "chipmunk.writing-shared-sentence-transfer/1"
ROOT = Path(__file__).resolve().parents[2]
PUBLIC = ROOT / "docs/evaluation-panels"
NOTICE = PUBLIC / "writing-practical-notice-transfer-2026-11-11.json"
ARTICLE = PUBLIC / "writing-multiblock-independent-2026-11-12.json"
OUTPUT = PUBLIC / "writing-shared-sentence-transfer-2026-11-12.json"
PLAN = {
    "point": 0.5,
    "baseline": "First six source-ordered positive native contexts. Compare both individual cards and their union.",
    "shared": "Two adjacent complete three-sentence windows with distinct outside sentences and two shared middle sentences. All four sentences must be in one interior paragraph. Rank by the smaller raw response, abstain on a tie, and propose only the shared pair.",
    "paragraph": "For notices also require the retained whole-paragraph response at the same fixed point.",
    "labels": "A shared range records the intersection of two contexts. Neither response directly measures the shared range alone.",
    "comparison": "Six Federal Register groups with whole-paragraph known edits. Separately replay ten Kubernetes groups to measure ordinary flags and single-sentence spill. The two packets are disjoint and were previously measured for other rules. No fitting or inference.",
    "useful": "At least 0.30 non-whitespace character intersection over union and 0.50 recall.",
    "advance": "In the notice panel, add useful locations in at least two distinct works across both generator families, with at most one additional ordinary source group. Keep every useful current location. The article packet measures specificity only.",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(raw):
    return hashlib.sha256(raw).hexdigest()


def digest(path):
    return sha256(path.read_bytes())


def read(path):
    return json.loads(path.read_bytes())


def lines(path):
    return [json.loads(row) for row in path.read_bytes().splitlines()]


def serialize(record):
    return json.dumps(record, indent=2, sort_keys=True) + "\n"


def freeze():
    return {"schema": SCHEMA, "method": PLAN, "runner_sha256": digest(Path(__file__)),
            "source_receipts_sha256": {path.name: digest(path) for path in (NOTICE, ARTICLE)}}


def notice_cases(folder):
    packet = read(NOTICE)
    public_result, source = packet["result"], packet["source"]
    specs = {
        "source.json": packet["freeze"]["source_sha256"],
        "freeze.json": packet["freeze_sha256"],
        "private-controls.jsonl": source["private_controls_sha256"],
        "private-positives.jsonl": packet["preparation"]["case_sha256"],
        "comparison/private-responses.jsonl": public_result["confirmation_receipt"]["responses_sha256"],
        "private-docx-verification.json": public_result["docx_verification_sha256"],
    }
    require(all(digest(folder / name) == pin for name, pin in specs.items()), "Notice packet hash differs")
    controls = lines(folder / "private-controls.jsonl")
    positives = lines(folder / "private-positives.jsonl")
    records = lines(folder / "comparison/private-responses.jsonl")
    files = {row["case"]: row for row in read(folder / "private-docx-verification.json")}
    require(len(controls) == 18 and len(positives) == 12 and len(files) == 26, "Notice denominator differs")
    for case, record in zip(controls + positives, records, strict=True):
        require((case["case"], case["status"]) == (record["case"], record["status"]), "Notice response identity differs")
        if case["status"] != "constructed":
            require(case["arm"] in {"qwen", "smol2"} and not case["text"], "Unavailable notice carries text")
            continue
        text, proposals = case["text"], record["proposals"]
        file = files[case["case"]]
        require(digest(folder / "docx" / f"{case['case']}.docx") == file["docx_sha256"]
                and sha256(text.encode()) == case["text_sha256"] == file["text_sha256"] == record["source_sha256"],
                "Notice text or exact DOCX commitment differs")
        proof = record["current_proof"]
        require(len(proposals) == proof["foci"] == proof["actual_input_windows"]
                and 0 <= proof["direct_audits"] <= proof["foci"],
                "Notice native coverage is incomplete")
        foci = [(row["start"], row["end"]) for row in proposals]
        paragraphs = [tuple(span) for span in case["paragraph_codepoints"]]
        truth = [tuple(span) for span in case["ai_spans"]]
        require(not truth or len(truth) == 1 and truth[0] in paragraphs[1:-1], "Notice target is outside an interior paragraph")
        isolated = {tuple(row["span"]): row["isolated"] for row in record["paragraph_measurements"]}
        yield (case["article_group"], case["arm"], text, paragraphs, foci, [r["raw"] for r in proposals], truth, isolated)


def article_cases(parent, folder):
    packet = read(ARTICLE)
    commitments = packet["private_commitments"]
    require(digest(folder / "freeze.json") == packet["freeze_sha256"], "Article freeze differs")
    lock = read(folder / "freeze.json")
    require(all(digest(parent / name) == pin for name, pin in lock["parent"].items()), "Parent article bytes differ")
    require(all(digest(folder / name) == commitments[name] for name in ("cases.json", "primary.json")),
            "Article case or response bytes differ")
    responses = read(parent / "primary.json")["responses"] | read(folder / "primary.json")["responses"]
    for case in read(folder / "cases.json"):
        if case["partition"] != "held":
            continue
        text = case["text"]
        require(sha256(text.encode()) == case["text_sha256"], "Article text differs")
        paragraphs, cursor = [], 0
        for paragraph in text.split("\n\n"):
            paragraphs.append((cursor, cursor + len(paragraph)))
            cursor += len(paragraph) + 2
        windows = case["windows"]
        foci = [tuple(row["focus"]) for row in windows]
        scores = []
        for index, window in enumerate(windows):
            spans = foci[max(0, index - 1):index + 2]
            reading = responses[window["primary"]]
            require([tuple(span) for span in window["spans"]] == spans
                    and sha256(" ".join(text[a:b] for a, b in spans).encode()) == window["primary"]
                    and reading["coverage"]["token_windows"] == 1 and reading["coverage"]["tokens"] > 0,
                    "Article window or complete response differs")
            scores.append(reading["score"])
        truth = [tuple(case["truth"])] if case["kind"] in {"qwen", "smol2"} else []
        require(not truth or truth[0] in foci, "Article generated target is not a single focus")
        yield (case["group"], case["kind"], text, paragraphs, foci, scores, truth, {})


def bitset(text, spans):
    return sum(1 << i for i in {i for a, b in spans for i in range(a, b) if not text[i].isspace()})


def measure(case, include_paragraph):
    group, kind, text, paragraphs, foci, scores, truth, isolated = case
    point = PLAN["point"]
    require(all(0 <= a < b <= len(text) for a, b in foci + paragraphs + truth)
            and all(b <= c for (_, b), (c, _) in zip(foci, foci[1:]))
            and all(math.isfinite(score) and 0 <= score <= 1 for score in scores)
            and bitset(text, foci) == bitset(text, [(0, len(text))]), "Invalid exact source geometry or score")
    wanted = bitset(text, truth)
    current = [foci[max(0, i - 1):i + 2] for i, value in enumerate(scores) if value >= point][:6]
    pairs = []
    for i in range(1, len(foci) - 2):
        paragraph = next(((a, b) for a, b in paragraphs[1:-1] if a <= foci[i - 1][0] < foci[i + 2][1] <= b), None)
        if paragraph is not None:
            pairs.append((min(scores[i:i + 2]), foci[i:i + 2], paragraph))

    def choose(candidates):
        ordered = sorted(candidates, key=lambda row: -row[0])
        if not ordered or ordered[0][0] < point or len(ordered) > 1 and ordered[0][0] == ordered[1][0]:
            return []
        return [ordered[0][1]]

    def figures(windows):
        marks = [bitset(text, spans) for spans in windows]
        union = bitset(text, [span for spans in windows for span in spans])

        def useful(mask):
            overlap = (mask & wanted).bit_count()
            return bool(wanted and 10 * overlap >= 3 * (mask | wanted).bit_count() and 2 * overlap >= wanted.bit_count())

        return {"flagged": bool(union), "useful": any(map(useful, marks)), "useful_union": useful(union),
                "human_characters": (union & ~wanted).bit_count(), "edit_characters": (union & wanted).bit_count()}

    shared = choose(pairs)
    observations = {"current": figures(current), "shared": figures(shared), "current_plus_shared": figures(current + shared)}
    if include_paragraph:
        confirmed = choose([row for row in pairs if isolated.get(row[2], -1) >= point])
        observations["paragraph_confirmed"] = figures(confirmed)
        observations["current_plus_confirmed"] = figures(current + confirmed)
    return {"group": sha256(group.encode())[:16], "kind": kind, "possible_pairs": len(pairs),
            "pair_can_localize": any(figures([spans])["useful"] for _, spans, _ in pairs), "outcomes": observations}


def summarize(rows):
    outcomes, false_groups = defaultdict(Counter), defaultdict(set)
    gained = defaultdict(set)
    for row in rows:
        positive = row["kind"] in {"qwen", "smol2"}
        for arm, result in row["outcomes"].items():
            role = "generated" if positive else "ordinary"
            totals = outcomes[role, arm]
            totals.update(cases=1, **result,
                          flagged_but_insufficient=positive and result["flagged"] and not result["useful"],
                          flagged_with_no_edit_overlap=positive and result["flagged"] and result["edit_characters"] == 0)
            if not positive and result["flagged"]:
                false_groups[arm].add(row["group"])
            if positive and result["useful"] and not row["outcomes"]["current"]["useful"]:
                gained[arm, row["kind"]].add(row["group"])
    return {"outcomes": {f"{role}/{arm}": dict(values) for (role, arm), values in outcomes.items()},
            "ordinary_groups_flagged": {arm: len(false_groups[arm]) for arm in rows[0]["outcomes"]},
            "gained_groups_by_generator": {f"{arm}/{kind}": sorted(groups) for (arm, kind), groups in gained.items()}}


def run(lock, notice, parent, article):
    require(read(lock) == freeze(), "Frozen comparison differs")
    notices = [measure(case, True) for case in notice_cases(notice)]
    articles = [measure(case, False) for case in article_cases(parent, article)]
    require(len(notices) == 26 and len(articles) == 31 and len({row["group"] for row in notices}) == 6
            and len({row["group"] for row in articles}) == 10, "Held panel denominator differs")
    return {**read(lock), "freeze_sha256": digest(lock), "notices": {"rows": notices, **summarize(notices)},
            "articles": {"rows": articles, **summarize(articles)}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "publish", "verify"))
    parser.add_argument("lock", type=Path)
    parser.add_argument("--notice", type=Path)
    parser.add_argument("--article-parent", type=Path)
    parser.add_argument("--article", type=Path)
    args = parser.parse_args()
    if args.command == "freeze":
        with args.lock.open("x") as stream:
            stream.write(serialize(freeze()))
        return
    result = run(args.lock, args.notice, args.article_parent, args.article)
    if args.command == "publish":
        with OUTPUT.open("x") as stream:
            stream.write(serialize(result))
    else:
        require(read(OUTPUT) == result, "Published outcome differs from native replay")
    print(serialize({name: summarize(result[name]["rows"]) for name in ("notices", "articles")}))


if __name__ == "__main__":
    main()
