"""Verify the frozen archive edge-change receipt with the Python standard library."""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from hashlib import sha256
from pathlib import Path

TOKEN = re.compile(r"\w+(?:[’'-]\w+)*|[^\w\s]")
SCHEMA = "chipmunk.writing-archive-changed-edge/1"


def load(path):
    return json.loads(path.read_bytes())


def file_sha(path):
    return sha256(path.read_bytes()).hexdigest()


def matches(text):
    return [(m[0], m.start(), m.end()) for m in TOKEN.finditer(text)]


def verify(run, catalog_path):
    lock = load(run / "freeze.json")
    assert lock["schema"] == SCHEMA and lock["archive"] == file_sha(catalog_path)
    catalog = load(catalog_path)
    work_rows = load(run / "private-sources.json")
    works = {row["source"]: row for row in work_rows}
    assert len(works) == len(work_rows) == len({row["sender"] for row in work_rows})
    outcomes = load(run / "result.json")
    counts, checks = {}, Counter()
    for case, bound, row in zip(load(run / "private-cases.json"), lock["cases"], outcomes["decisions"], strict=True):
        group, role, text, truth = (case[key] for key in ("group", "role", "text", "truth"))
        assert bound == {**case, "text": sha256(text.encode()).hexdigest()}
        source, aligned = row["source"], row["aligned"]
        if source is None:
            assert not aligned and not row["submitted"] and not row["earlier"]
        else:
            assert source == group == case["expected"]
            references = {p["ordinal"] + 1: p["text"] for p in catalog[source]["paragraphs"]}
            assert references == dict(zip(works[source]["ordinals"], works[source]["parts"], strict=True))
            assert aligned[0][1] == 0 and aligned[-1][2] == len(text)
            assert all(b[0] == a[0] + 1 and text[a[2]:b[1]] == " " for a, b in zip(aligned, aligned[1:]))
            earlier, submitted = [], []
            for i, (ordinal, start, stop) in enumerate(aligned):
                original, current = matches(references[ordinal]), matches(text[start:stop])
                assert len(original) == len(current)
                for j, ((before, a, b), (after, c, d)) in enumerate(zip(original, current, strict=True)):
                    if before == after:
                        continue
                    if role == "before_break":
                        assert i == 0 and j >= len(original) - 8
                    else:
                        assert role == "after_break" and i == 2 and j < 8
                    assert references[ordinal][:a] + after + references[ordinal][b:] == text[start:stop]
                    earlier.append([a, b])
                    submitted.append([start + c, start + d])
                checks["source_paragraphs"] += 1
            assert submitted == row["submitted"] and earlier == row["earlier"]
            assert submitted == ([truth] if truth is not None else [])
            checks["changed_words"] += len(submitted)
        counts.setdefault(role, Counter()).update(total=1, baseline=bool(row["baseline"]), matched=bool(source),
            incorrect=source is not None and source != case["expected"], exact=truth is not None and row["submitted"] == [truth],
            changed=bool(row["submitted"]))
        checks["decisions"] += 1
    assert counts == outcomes["counts"]
    original = counts["original"]
    passed = (len(works) >= 12 and original["total"] == original["matched"] and not original["changed"]
              and not any(cell["incorrect"] for cell in counts.values())
              and all(counts[role]["exact"] * 10 >= counts[role]["total"] * 9 for role in ("before_break", "after_break")))
    assert passed == outcomes["passed"]
    return {"schema": SCHEMA, "design": lock["design"], "passed": passed, "counts": counts,
            "source_groups": len(works), "publisher_volumes": dict(Counter(group.split("/")[0] for group in works)),
            "independent": checks, "committed_inputs": {"freeze": file_sha(run / "freeze.json"),
                                                        "result": file_sha(run / "result.json"), "archive": lock["archive"]}}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "catalog", "receipt"):
        cli.add_argument(name, type=Path)
    args = cli.parse_args()
    result = verify(args.run, args.catalog)
    with args.receipt.open("x") as output:
        json.dump(result, output, indent=2, sort_keys=True)
        output.write("\n")
    print(json.dumps(result["independent"]))


if __name__ == "__main__":
    main()
