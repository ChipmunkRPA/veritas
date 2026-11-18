# SPDX-License-Identifier: Apache-2.0
"""Recheck paired HTML source positions on the previously held writing excerpts."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
from html import escape, unescape
from pathlib import Path

SCHEMA = "chipmunk.html-draft-transfer/1"
ROLES = {"authentic", "same_article_swap", "smol2", "granite"}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    return json.loads(gzip.decompress(path.read_bytes()))


def markup(text, *, restyled=False):
    blocks = []
    for part in text.split("\n\n"):
        content = escape(part)
        if restyled and (word := re.search(r"\b[a-zA-Z]{8,}\b", part)):
            split = word.start() + 3
            content = escape(part[:split]) + "<em>" + escape(part[split:word.end()]) + "</em>" + escape(part[word.end():])
        blocks.append(f"<p>{content}</p>")
    chrome = "<nav>Updated site menu</nav>" if restyled else "<nav>Earlier site menu</nav>"
    decoration = '<aside hidden>Editorial layout fixture</aside>' if restyled else ""
    return chrome + "<main><article>" + "\n".join(blocks) + decoration + "</article></main>"


def locations(raw, text, excerpt):
    rows = []
    for token in (excerpt or {}).get("html_tokens", []):
        start, end = token["start"], token["end"]
        ranges = token["source_ranges"]
        reading = "".join(unescape(raw[a:b]) for a, b in ranges)
        if reading != text[start:end]:
            raise ValueError("An exported source range decodes to different wording")
        rows.append({"span": [start, end], "html": ranges})
    if len(rows) != len((excerpt or {}).get("changed_tokens", [])):
        raise ValueError("A reported change has no source ranges")
    return rows


def _bound(bounds, limit):
    if len(bounds) != 2 or any(type(value) is not int for value in bounds) or not 0 <= bounds[0] < bounds[1] <= limit:
        raise ValueError("Invalid saved text range")
    return bounds


def summarize(rows):
    totals = {role: {"pairs": 0, "with_changes": 0, "with_target_tokens": 0, "with_outside_tokens": 0}
              for role in sorted(ROLES)}
    seen = set()
    for row in rows:
        key = row["id"], row["role"]
        if key in seen:
            raise ValueError("Repeated held source and role")
        seen.add(key)
        count = totals[row["role"]]
        count["pairs"] += 1
        changed = inside = outside = False
        for side in ("earlier", "later"):
            view = row[side]
            if row["role"] in {"smol2", "granite"}:
                lo, hi = _bound(view["target"], view["characters"])
            elif view["target"] is not None:
                raise ValueError("A comparison control has an edit label")
            previous = 0
            for token in view["tokens"]:
                start, end = _bound(token["span"], view["characters"])
                if not token["html"] or start < previous:
                    raise ValueError("Invalid ordered token locations")
                previous = end
                for source in token["html"]:
                    _bound(source, view["html_characters"])
                changed = True
                if row["role"] in {"smol2", "granite"}:
                    if lo <= start < end <= hi:
                        inside = True
                    else:
                        outside = True
        count["with_changes"] += changed
        count["with_target_tokens"] += inside
        count["with_outside_tokens"] += outside
    return totals


def native_snapshots(roster, pages):
    from app.evidence.document_revision import _article_source, _html_token_sources, compare

    indexed = {digest(path.read_bytes()): path for path in sorted(pages.glob("*.html"))}
    rows = []
    for pin, source in sorted(roster.items()):
        body = indexed[pin].read_bytes()
        control = compare(body, body, ".html")
        if control["changes"]:
            raise ValueError("An identical native HTML snapshot produced text changes")
        before_text, _, _, layout = _article_source(body)
        raw = layout[0]
        for word in re.finditer(r"\b[A-Za-z]{10,}\b", before_text):
            try:
                [candidate] = _html_token_sources(before_text, layout, [list(word.span())])
            except ValueError:
                continue
            ranges = candidate["source_ranges"]
            if len(ranges) == 1 and raw[slice(*ranges[0])] == word[0]:
                break
        else:
            raise ValueError("The native snapshot has no literal word for the scoped position control")
        a, b = ranges[0]
        replacement = "verifiedreplacement"
        edited = (raw[:a] + replacement + raw[b:]).encode()
        result = compare(body, edited, ".html")
        [change] = result["changes"]
        old = locations(raw, before_text, change["earlier"])
        after_text, _, _, _ = _article_source(edited)
        new = locations(edited.decode(), after_text, change["later"])
        expected = [[[a, b]], [[a, a + len(replacement)]]]
        if [[row["html"] for row in side] for side in (old, new)] != [[part] for part in expected]:
            raise ValueError("A single changed word did not map back to the exact editor positions")
        rows.append({"publisher_url": source["url"], "native_sha256": pin, "edited_sha256": digest(edited),
                     "scope": control["earlier"]["html_article_scope"], "identical_control_changes": 0,
                     "earlier_ranges": expected[0], "later_ranges": expected[1], "native_characters": len(raw),
                     "edited_characters": len(edited.decode())})
    return rows


def count_native(rows):
    if len(rows) != len({row["native_sha256"] for row in rows}):
        raise ValueError("Repeated native source snapshot")
    for row in rows:
        if (row["identical_control_changes"] != 0 or row["scope"]["included_blocks"] <= 0
                or len(row["earlier_ranges"]) != 1 or len(row["later_ranges"]) != 1):
            raise ValueError("Invalid native snapshot result")
        _bound(row["earlier_ranges"][0], row["native_characters"])
        _bound(row["later_ranges"][0], row["edited_characters"])
        if row["earlier_ranges"][0][0] != row["later_ranges"][0][0]:
            raise ValueError("The replacement start differs between the two native snapshots")
    return {"pairs": len(rows), "extra_unmapped_text": sum(row["scope"]["unmapped_text"] for row in rows)}


def measure(plan, previous, pages, output):
    from app.evidence.document_revision import ARTICLE_SCHEMA, compare_with_sources

    frozen, replay = read(plan), read(previous)
    cases = [row for row in frozen["cases"] if row["split"] == "held"]
    retained = {(row["id"], row["role"]): row for row in replay["held"]["cases"]}
    if len(cases) != len(retained) or set(retained) != {(row["id"], row["role"]) for row in cases}:
        raise ValueError("The input does not reproduce the published held roster")
    originals = {row["id"]: row for row in cases if row["role"] == "authentic"}
    output_rows = []
    for case in cases:
        previous_row = retained[case["id"], case["role"]]
        if any(case[field] != previous_row[field] for field in ("sha256", "truth", "target", "original_sha256")):
            raise ValueError("The held text commitment or truth range differs")
        if digest(case["text"].encode()) != case["sha256"]:
            raise ValueError("The prepared body differs from its input hash")
        earlier = originals[case["id"]]["text"]
        later = case["text"]
        raw_before, raw_after = markup(earlier), markup(later, restyled=True)
        before, after = raw_before.encode(), raw_after.encode()
        result, old_text, new_text, _ = compare_with_sources(before, after, ".html")
        if result["schema"] != ARTICLE_SCHEMA or (old_text, new_text) != (earlier, later):
            raise ValueError("HTML wrapping changed the prepared body")
        summary = {"id": case["id"], "role": case["role"], "publisher_url": case["url"],
                   "publisher_sha256": case["original_sha256"], "input_sha256": case["sha256"]}
        for side, raw, text, target in (("earlier", raw_before, old_text, case["target"] if case["truth"] else None),
                                        ("later", raw_after, new_text, case["truth"])):
            tokens = [token for change in result["changes"] for token in locations(raw, text, change[side])]
            summary[side] = {"html_sha256": result[side]["sha256"], "characters": len(text),
                             "html_characters": len(raw), "target": target, "tokens": tokens}
        output_rows.append(summary)
    totals = summarize(output_rows)
    roster = {row["original_sha256"]: row for row in frozen["cases"] if row["role"] == "authentic"}
    snapshots = native_snapshots(roster, pages)
    proof = {"schema": SCHEMA, "held_plan_sha256": digest(plan.read_bytes()), "published_replay_sha256": digest(previous.read_bytes()),
             "source_exposure": "previously scored writing excerpts, new paired HTML renditions",
             "cases": output_rows, "totals": totals, "native_snapshots": snapshots, "native_totals": count_native(snapshots)}
    with output.open("xb") as stream:
        stream.write(gzip.compress((json.dumps(proof, sort_keys=True, indent=2) + "\n").encode(), mtime=0))
    print(json.dumps(totals, sort_keys=True))


def review(receipt):
    saved = read(receipt)
    if (saved["schema"] != SCHEMA or summarize(saved["cases"]) != saved["totals"]
            or count_native(saved["native_snapshots"]) != saved["native_totals"]):
        raise ValueError("The saved counts differ from the case measurements")
    print(json.dumps(saved["totals"], sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command, names in (("measure", ("plan", "previous", "pages", "output")), ("review", ("receipt",))):
        child = sub.add_parser(command)
        for name in names:
            child.add_argument(name, type=Path)
    fields = vars(parser.parse_args())
    {"measure": measure, "review": review}[fields.pop("command")](**fields)
