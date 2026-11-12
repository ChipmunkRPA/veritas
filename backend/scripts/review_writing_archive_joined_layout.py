"""Check the frozen joined-layout claims against source text and known edit positions."""

from __future__ import annotations

import argparse
import re
from collections import Counter, defaultdict
from hashlib import sha256
from pathlib import Path

from scripts.writing_archive_joined_layout_transfer import create_cases, digest, read, save

WORD_OR_PUNCTUATION = re.compile(r"\w+(?:[’'-]\w+)*|[^\w\s]")


def audit(parent, run, catalog):
    pins, cases = create_cases(parent, catalog)
    freeze = read(run / "freeze.json")
    recorded = read(run / "results.json")
    assert pins == freeze["volumes"]
    counts, verified = defaultdict(Counter), Counter()
    for case, binding, row in zip(cases, freeze["cases"], recorded["decisions"], strict=True):
        group, role, text, expected, truth = case
        assert binding == [group, role, sha256(text.encode()).hexdigest(), expected, truth]
        assert (group, role, expected, truth) == (row["group"], row["role"], row["expected"], row["truth"])
        source, spans, changed = row["source"], row["aligned"], row["changes"]
        words = {match.span() for match in WORD_OR_PUNCTUATION.finditer(text)}
        assert all(tuple(span) in words for span in changed)
        assert all(left[1] <= right[0] for left, right in zip(changed, changed[1:]))
        if source is None:
            assert not spans and not changed
        else:
            assert source == expected
            parts = {part["ordinal"] + 1: part["text"] for part in catalog[source]["paragraphs"]}
            assert spans[0][1] == 0 and spans[-1][2] == len(text)
            for before, after in zip(spans, spans[1:]):
                assert after[0] == before[0] + 1 and text[before[2]:after[1]].isspace()
            for ordinal, start, stop in spans:
                assert 0 <= start < stop <= len(text)
                actual, reference = text[start:stop], parts[ordinal]
                if truth is None or not start <= truth[0] < truth[1] <= stop:
                    assert actual == reference
                elif role == "joined_edit":
                    a, b = (offset - start for offset in truth)
                    assert len(actual) == len(reference) and actual[a:b] != reference[a:b]
                    assert (actual[:a], actual[b:]) == (reference[:a], reference[b:])
                else:
                    actual_words = [m[0] for m in WORD_OR_PUNCTUATION.finditer(actual)]
                    reference_words = [m[0] for m in WORD_OR_PUNCTUATION.finditer(reference)]
                    assert actual_words[:8] == reference_words[:8] and actual_words[-8:] == reference_words[-8:]
                verified["source_spans"] += 1
        useful = bool(truth and any(max(a, truth[0]) < min(b, truth[1]) for a, b in changed))
        exact = bool(truth and changed == [truth])
        assert useful == row["useful"] and exact == row["exact"]
        counts[role].update(total=1, baseline=int(row["baseline"] is not None), matched=int(source is not None),
                            false_attribution=int(source is not None and source != expected),
                            useful=int(useful), exact=int(exact), has_changes=int(bool(changed)))
        verified["decisions"] += 1
        verified["changed_tokens"] += len(changed)
    assert counts == recorded["counts"]
    originals, edits = counts["joined_original"], counts["joined_edit"]
    assert recorded["advance"] == (originals["total"] >= 12 and originals["matched"] * 4 >= originals["total"] * 3
                                    and edits["exact"] * 4 >= edits["total"] * 3 and not originals["has_changes"]
                                    and not any(r["false_attribution"] for r in counts.values()))
    return {**verified, "freeze_sha256": digest(run / "freeze.json"), "results_sha256": digest(run / "results.json"),
            "counts": counts, "advance": recorded["advance"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parent", type=Path)
    parser.add_argument("run", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    index = args.parent / "index.json"
    assert digest(index) == read(args.run / "freeze.json")["archive_sha256"]
    result = audit(args.parent, args.run, read(index))
    save(args.output, result)
    print({key: value for key, value in result.items() if key != "counts"})


if __name__ == "__main__":
    main()
