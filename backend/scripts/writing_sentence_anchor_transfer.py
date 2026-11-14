"""Evaluate paired wording alignment on new OpAI-Bench works, without publishing prose."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

SCHEMA = "chipmunk.writing-sentence-anchor-transfer/1"
GENERATORS = ("gemini-2.5-flash", "gpt-5.4-nano")
ROOT = Path(__file__).resolve().parents[2]
SOURCES = (
    "writing-journal-body-transfer-2026-11-11",
    "writing-archive-prospective-transfer-2026-11-12",
)
PROTOCOL = {
    "admission": "Exclude earlier report work hashes and normalized twelve-word matches, including the Qwen shard. "
                 "Use exact three-sentence anchors within a source paragraph, with publisher spans matching both "
                 "the complete edited sentence and the tagged text. Prefer paraphrase, compress, then polish and "
                 "the earliest source position. Keep every work with a case from both registered generators.",
    "construction": "Join each selected report edit with a historical journal or technical documentation revision. "
                    "Alternate paragraph order and include separate ordinary-only pairs with distinct sources.",
    "comparison": "Compare the existing complete-draft word diff against a local word diff constrained by unchanged "
                  "neighboring sentences. Keep precisely the same pinned model responses and doubling-of-odds rule. "
                  "Count exact later sentence matches, assessability, and false flags by original ordinary work.",
    "decision": "Require at least four additional exact locations across both generators and at least three report "
                "works on the new source panel, with no more than two additional ordinary works flagged. Preserve "
                "all assessed incumbent outcomes. The optional check contributes zero to authorship confidence.",
}


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def source_spans(raw):
    from app.detectors.text.extract import from_string
    from app.detectors.text.segment import segment
    from app.evidence.document_revision_writing import _context

    doc = from_string(raw)
    units = segment(doc)
    if not units:
        return []
    spans = _context(doc, units, raw, full_range=True)[0]["context_spans"]
    return [(unit, start, end) for unit, (start, end) in zip(units, spans, strict=True)]


def source_windows(raw):
    units = source_spans(raw)
    for left, center, right in zip(units, units[1:], units[2:]):
        prev, a, b = left
        focus, start, end = center
        next_, c, d = right
        if prev.paragraph == focus.paragraph == next_.paragraph and 10 <= focus.n_words <= 90:
            yield (raw[a:b], raw[c:d]), (a, start, end, d)


def excerpt(text, positions):
    a, start, end, d = positions
    return text[a:d], [start - a, end - a]


def fresh_edits(cache, excluded_hashes):
    from scripts import writing_interval_transfer as interval
    from scripts import writing_revision_trajectory_transfer as publisher

    for name, expected in publisher.METADATA.items():
        publisher.require(publisher.digest(cache / name) == expected, "Publisher metadata differs")
    publisher.require("license: apache-2.0" in (cache / "README.md").read_text(), "Publisher terms differ")
    tables = {key: interval.read_shard(cache, key, pin)
              for key, pin in (interval.FAMILIES | interval.EXCLUDED_SHARD).items()}
    gemini, nano = (tables[key] for key in GENERATORS)
    excluded = tables["qwen3-8b"].keys() | {g for g in gemini.keys() | nano.keys() if sha(g) in excluded_hashes}
    historical = set()
    for group in excluded:
        for table in tables.values():
            if group in table:
                historical.update(interval.fingerprints(table[group]["v0"]["text"]))
    names = sorted((gemini.keys() & nano.keys()) - excluded, key=lambda key: sha(f"separate-anchors/{key}"))
    stats = Counter(eligible=len(names))
    for group in names:
        original = gemini[group]["v0"]
        if original["text"] != nano[group]["v0"]["text"]:
            stats["original_differs"] += 1
            continue
        fingerprints = interval.fingerprints(original["text"])
        if fingerprints & historical:
            stats["earlier_overlap"] += 1
            continue
        try:
            raw, labels = publisher.publisher_case(original)
            publisher.require(not labels, "Original has labeled generated text")
            windows = list(source_windows(raw))
        except ValueError:
            stats["rejected_original"] += 1
            continue
        counts = Counter(signature for signature, _ in windows)
        lookup = {signature: positions for signature, positions in windows if counts[signature] == 1}
        selected = {}
        for family in GENERATORS:
            for variant in ("v2", "v4", "v1"):
                try:
                    edited, labels = publisher.publisher_case(tables[family][group][variant])
                    proposed = list(source_windows(edited))
                except ValueError:
                    continue
                label_set = set(map(tuple, labels))
                for signature, (a, start, end, d) in proposed:
                    old = lookup.get(signature)
                    if old is None or (start, end) not in label_set or raw[old[1]:old[2]] == edited[start:end]:
                        continue
                    if any(lo < d and hi > a and (lo, hi) != (start, end) for lo, hi in label_set):
                        continue
                    selected[family] = (variant, (excerpt(raw, old), excerpt(edited, (a, start, end, d))))
                    break
                if family in selected:
                    break
        if selected.keys() == set(GENERATORS):
            stats["paired_sources"] += 1
            historical.update(fingerprints)
            for family, (variant, pair) in selected.items():
                yield (sha(group), family, pair), variant
    print("Source admission", dict(stats), flush=True)


def ordinary_edits(root):
    edits = []
    hashes = {}
    for name in SOURCES:
        raw = (root / name / "originals.json").read_text()
        hashes[name] = sha(raw)
        for row in sorted(json.loads(raw), key=lambda item: item["group"]):
            by_side = []
            for side in ("before", "after"):
                text = row[side]["text"]
                focus = row[side]["focus"]
                spans = source_spans(text)
                matches = [i for i, (_, lo, hi) in enumerate(spans) if [lo, hi] == focus]
                if len(matches) != 1 or not 0 < matches[0] < len(spans) - 1:
                    break
                i = matches[0]
                by_side.append(excerpt(text, (spans[i - 1][1], *focus, spans[i + 1][2])))
            if len(by_side) == 2:
                edits.append((row["group"], "ordinary", tuple(by_side)))
    return edits, hashes


def freeze(root):
    from app.evidence import document_revision_writing as paired
    from scripts import writing_interval_transfer as interval
    from scripts import writing_revision_trajectory_transfer as publisher
    from scripts.writing_separate_revision_transfer import join, load_cases

    previous, old_hashes = load_cases(root)
    ordinary, ordinary_hashes = ordinary_edits(root)
    ledgers = {path.name: path.read_text() for path in sorted((ROOT / "docs/evaluation-panels").glob("writing-*.json"))
               if not path.name.startswith("writing-sentence-anchor-")}
    hashes = set().union(*(set(re.findall(r'(?<=")[0-9a-f]{64}(?=")', raw)) for raw in ledgers.values()))
    fresh = list(fresh_edits(root / "text-opai-bench-intake-v257", hashes))
    assert len(fresh) >= 32 and len(ordinary) >= 40
    cases = [("prior/" + bank, old, new, truth, None) for bank, old, new, truth in previous]
    cases.extend(("ordinary_pair", *join((a, b)), None) for a, b in zip(ordinary[::2], ordinary[1::2]))
    for i, (edit, variant) in enumerate(fresh):
        control = ordinary[(i // 2 + (i % 2) * 13) % len(ordinary)]
        parts = (control, edit) if i % 2 else (edit, control)
        cases.append(("fresh", *join(parts), variant))
    return cases, {
        "schema": SCHEMA, "protocol": PROTOCOL, "runner_sha256": publisher.digest(Path(__file__)),
        "anchor_sha256": publisher.digest(Path(paired.__file__)), "old_rosters": old_hashes,
        "ordinary_rosters": ordinary_hashes, "publisher_revision": publisher.REVISION,
        "publisher_assets": publisher.METADATA | interval.FAMILIES | interval.EXCLUDED_SHARD,
        "earlier_ledgers": {name: sha(raw) for name, raw in ledgers.items()},
    }


def geometry(candidates, truth):
    rows = []
    for candidate, label in zip(candidates, truth, strict=True):
        later = candidate.geometry["later"]
        if [later["start"], later["end"]] != label["later"]:
            raise ValueError("Mapped sentence differs from its publisher label")
        rows.append({side: {k: v for k, v in candidate.geometry[side].items() if k != "text"}
                     for side in ("earlier", "later")})
    return rows


def mapped(cases):
    from app.evidence import document_revision as draft
    from app.evidence import document_revision_writing as paired

    for bank, before, after, truth, operation in cases:
        comparison, old, new = draft.compare_with_text(before.encode(), after.encode(), ".txt")
        existing = paired.separate_revisions(comparison, old, new)
        anchored = paired.anchor_revisions(comparison, old, new) if not existing else None
        recovered = anchored[1] if anchored else []
        candidates = existing or recovered
        yield {
            "bank": bank, "source_sha256": [sha(before), sha(after)], "truth": truth, "variant": operation,
            "first_supported_by": "current" if existing else "anchors" if recovered else None,
            "geometry": geometry(candidates, truth) if candidates else [],
            "changes": ([{side: {key: item[side][key] for key in ("start", "end", "changed_tokens")}
                           for side in ("earlier", "later")} for item in anchored[0]["changes"]] if anchored else []),
        }, candidates


def summary(rows):
    reports = {}
    for bank in sorted({row["bank"] for row in rows} | {"fresh_and_ordinary"}):
        stats = Counter()
        assessed_ordinary, old_ordinary, added_ordinary, works, families = set(), set(), set(), set(), set()
        included = {"fresh", "ordinary_pair"} if bank == "fresh_and_ordinary" else {bank}
        for row in rows:
            if row["bank"] not in included:
                continue
            labels = row["truth"]
            previous = row["first_supported_by"] == "current"
            for label, (status, scores) in zip(labels, row["results"], strict=True):
                if scores is not None:
                    if (len(scores) != 2 or any(set(pair) != {"earlier", "later"} for pair in scores)
                            or any(type(value) not in (int, float) or not 0 <= value <= 1
                                   for pair in scores for value in pair.values())):
                        raise ValueError("Expected two complete paired responses")
                    expected = "joint_increase" if all(
                        clamped_odds(pair["later"]) >= 2 * clamped_odds(pair["earlier"]) for pair in scores
                    ) else "no_joint_increase"
                    if expected != status:
                        raise ValueError("Recorded response arithmetic differs")
                elif status != "unavailable":
                    raise ValueError("A status requires two complete responses")
                supported, flagged = scores is not None, status == "joint_increase"
                work, family = label["group"], label["family"]
                if family == "ordinary":
                    if supported:
                        assessed_ordinary.add(work)
                    if flagged:
                        (old_ordinary if previous else added_ordinary).add(work)
                else:
                    stats["total_generated"] += 1
                    stats["old_assessed"] += previous and supported
                    stats["old_exact_flags"] += previous and flagged
                    stats["added_assessed"] += not previous and supported
                    stats["added_exact_flags"] += not previous and flagged
                    if not previous and flagged:
                        works.add(work)
                        families.add(family)
        reports[bank] = dict(stats) | {
            "assessed_ordinary_works": len(assessed_ordinary), "old_ordinary_flags": len(old_ordinary),
            "additional_ordinary_flags": len(added_ordinary - old_ordinary),
            "added_edited_works": len(works), "added_generator_families": sorted(families),
        }
    return reports


def clamped_odds(value):
    p = min(1 - 1e-7, max(1e-7, value))
    return p / (1 - p)


def review(packet):
    if packet.get("schema") != SCHEMA or packet.get("protocol") != PROTOCOL:
        raise ValueError("The receipt does not use the frozen comparison")
    for row in packet["rows"]:
        truth, geometry, arm = (row[key] for key in ("truth", "geometry", "first_supported_by"))
        if len(truth) != 2 or arm not in (None, "current", "anchors"):
            raise ValueError("Expected a record of the two separate changes and their assessment scope")
        if arm is None:
            if geometry or row["changes"] or row["results"] != [["unavailable", None]] * 2:
                raise ValueError("An unassessed draft pair cannot carry evidence")
            continue
        for label, revision in zip(truth, geometry, strict=True):
            if label["later"] != [revision["later"]["start"], revision["later"]["end"]]:
                raise ValueError("A scored location differs from the known revision")
            for side in ("earlier", "later"):
                reading = revision[side]
                spans = reading["context_spans"]
                if len(spans) != 3 or spans[1] != [reading["start"], reading["end"]]:
                    raise ValueError("The recorded neighbors do not bound the changed sentence")
        if arm == "anchors":
            for edit in row["changes"]:
                matches = [region for region in geometry if all(
                    region[side]["start"] <= edit[side]["start"] <= edit[side]["end"] <= region[side]["end"]
                    for side in ("earlier", "later")
                )]
                if len(matches) != 1:
                    raise ValueError("A recorded wording change crosses its anchored sentence")
    counts = summary(packet["rows"])
    if counts != packet["summary"]:
        raise ValueError("Receipt counts differ from its readings")
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("freeze", "score", "review"))
    parser.add_argument("file", type=Path)
    parser.add_argument("--root", type=Path, default=Path("data/independent-evaluation"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.mode == "review":
        counts = review(json.loads(args.file.read_text()))
        print(json.dumps(counts, indent=2))
        return
    cases, metadata = freeze(args.root)
    items = list(mapped(cases))
    metadata["rows"] = [row for row, _ in items]
    if args.mode == "freeze":
        with args.file.open("x") as stream:
            json.dump(metadata, stream, separators=(",", ":"))
        print(Counter((row["bank"], row["first_supported_by"]) for row, _ in items), flush=True)
        return
    if args.output is None or metadata != json.loads(args.file.read_text()):
        raise ValueError("Holdout geometry differs from the freeze or no score output was specified")
    import torch

    from app.evidence import document_revision_writing as paired

    torch.set_num_threads(2)
    cache = {}
    for row, candidates in items:
        unseen = [item for item in candidates if (item.earlier, item.later) not in cache]
        if unseen:
            scores = paired.score_separate(unseen) or [None] * len(unseen)
            for item, score in zip(unseen, scores, strict=True):
                cache[item.earlier, item.later] = score
        responses = [cache[item.earlier, item.later] for item in candidates]
        measured = paired.multiple_result(candidates, responses)["revisions"] if candidates else ()
        row["results"] = [[item["status"], item.get("responses")] for item in measured] if measured else [
            ["unavailable", None] for _ in row["truth"]]
    metadata["summary"] = summary(metadata["rows"])
    review(metadata)
    with args.output.open("x") as stream:
        json.dump(metadata, stream, indent=2)
        stream.write("\n")
    print(json.dumps(metadata["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
