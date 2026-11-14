"""Transfer a frozen two-scale draft rule to held news and archival prose."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

from scripts.writing_sentence_anchor_transfer import GENERATORS, clamped_odds

SCHEMA = "chipmunk.writing-paired-scale-transfer/1"
ROOT = Path(__file__).resolve().parents[2]
SHARDS = {
    "news": {"gemini-2.5-flash": "44161adb78c0dfa4cc06d61d1af6a3a2a63ecc1b4f0c41ba0525d1cd0fab69ab",
             "gpt-5.4-nano": "8e7c2b3dc843c0665be33e45f9708cdf724c59aff8953d58dc108a4cf1c43a6b"},
    "abstracts": {"gemini-2.5-flash": "2c626b0597f6082908a0183b31c48884a45d310fea8bb1a84043ed2432568b30",
                  "gpt-5.4-nano": "9b8873c000ee712f0d1a7bf325cba9ea8245bab5b34b206fbad79968b4af4ea9"},
}
HISTORIC = "writing-paired-center-transfer-2026-11-12.json"
SOURCE_REPLAY = "writing-sentence-anchor-transfer-2026-11-14.json"
POINT = 0.12844311183234092
VARIANTS = {"v1": "polish", "v2": "paraphrase", "v4": "compress"}
LIMIT = 40
PROTOCOL = {
    "source": "Use new English news or scientific abstracts in deterministic digest order with an exact labeled center and unchanged "
              "native neighbors for both generators. Exclude prior public writing work hashes and overlapping "
              "12-word phrases against excluded seeds. Use the first 40 eligible works, or all if fewer qualify. "
              "For abstracts, verify character tags and native sentence geometry directly because the separately "
              "published sentence array does not reproduce the clean text for the shared seeds.",
    "control": "Combine each news rewrite with an archival ordinary revision or same-work human-prose substitution. "
               "Also score ordinary-only pairs. Report flags by distinct ordinary work.",
    "comparison": "Use both classifiers after verifying that their weights match the app pins. Measure the exact "
                  "center alone and its three-sentence context. Retain the existing context decisions. The added "
                  "rule needs every log-odds change to exceed the previously frozen single-boundary point. "
                  "Measure existing app scope and the optional sentence-anchored expansion separately.",
    "decision": "For local opt-in on an arm require five added exact locations across three source works and both generators, "
                "with at most two additional ordinary works flagged. Keep the earlier signal weight at zero.",
}


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def read_shard(path, family, domain):
    from scripts.writing_revision_trajectory_transfer import digest, require

    require(digest(path) == SHARDS[domain][family], "Data shard differs from the pinned release")
    csv.field_size_limit(1024 * 1024)
    works = defaultdict(dict)
    with path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream, strict=True):
            require((row["domain"], row["split"], row["subset"], row["generator"]) == (domain, "test", "main", family),
                    "Unexpected source role")
            variant = row["version"]
            if variant == "v0" or variant in VARIANTS:
                work = works[row["id"]]
                require(variant not in work and row["edit_operation"] == VARIANTS.get(variant, "none"),
                        "Incorrect or repeated source variant")
                work[variant] = row
    require(all(set(rows) == {"v0", *VARIANTS} for rows in works.values()), "Incomplete benchmark work")
    return works


def labeled_text(row, domain):
    from scripts.writing_revision_trajectory_transfer import publisher_case, ranges_from_tags

    if domain == "news":
        return publisher_case(row)
    text, tagged = ranges_from_tags(row["text_tagged"])
    labeled = json.loads(row["ai_spans_char"])
    if text != row["text"] or tagged != labeled:
        raise ValueError("Character labels do not match the clean source and inline tags")
    return text, labeled


def editor_pair(raw, originals, table, domain):
    from scripts.writing_sentence_anchor_transfer import excerpt, source_windows

    for variant in VARIANTS:
        try:
            edited, labels = labeled_text(table[variant], domain)
            windows = list(source_windows(edited))
        except ValueError:
            continue
        label_set = set(map(tuple, labels))
        for signature, (a, start, end, d) in windows:
            old = originals.get(signature)
            if old is None or (start, end) not in label_set or raw[old[1]:old[2]] == edited[start:end]:
                continue
            if any(lo < d and hi > a and (lo, hi) != (start, end) for lo, hi in label_set):
                continue
            return variant, (excerpt(raw, old), excerpt(edited, (a, start, end, d)))
    return None


def cases(root, directory, domain="news"):
    from scripts.writing_interval_transfer import fingerprints
    from scripts.writing_sentence_anchor_transfer import ordinary_edits, source_windows
    from scripts.writing_separate_revision_transfer import join

    ledgers = {p.name: p.read_text() for p in sorted((ROOT / "docs/evaluation-panels").glob("writing-*.json"))
               if not p.name.startswith(("writing-paired-scale-", "writing-independent-anchor-"))}
    seen = set().union(*(set(re.findall(r'(?<=")[0-9a-f]{64}(?=")', payload)) for payload in ledgers.values()))
    ordinary, ordinary_hashes = ordinary_edits(root)
    gemini, nano = (read_shard(directory / f"{domain}_{family}.csv", family, domain) for family in GENERATORS)
    order = sorted(gemini.keys() & nano.keys(), key=lambda work: sha(f"{SCHEMA}/{work}"))
    history = set()
    for work in order:
        if sha(work) in seen:
            history.update(fingerprints(gemini[work]["v0"]["text"]))
    count = Counter(reserved_by_id=len(order))
    for work in order:
        if count["selected"] == LIMIT:
            break
        if sha(work) in seen:
            count["prior_id"] += 1
            continue
        record = gemini[work]["v0"]
        if record["text"] != nano[work]["v0"]["text"]:
            count["different_seeds"] += 1
            continue
        phrases = fingerprints(record["text"])
        if phrases & history:
            count["source_overlap"] += 1
            continue
        try:
            raw, labels = labeled_text(record, domain)
            originals = list(source_windows(raw))
        except ValueError:
            count["source_geometry"] += 1
            continue
        if labels:
            raise ValueError("A source seed contains generated labels")
        totals = Counter(signature for signature, _ in originals)
        unique = {signature: ranges for signature, ranges in originals if totals[signature] == 1}
        matches = [editor_pair(raw, unique, table[work], domain) for table in (gemini, nano)]
        if any(item is None for item in matches):
            count["no_matched_edit"] += 1
            continue
        group = sha(work)
        for lane, (family, (variant, pair)) in enumerate(zip(GENERATORS, matches, strict=True)):
            control = ordinary[(count["selected"] + 17 * lane) % len(ordinary)]
            generated = (group, family, pair)
            parts = (generated, control) if lane else (control, generated)
            yield (domain, *join(parts), variant)
        history.update(phrases)
        count["selected"] += 1
    if count["selected"] < 20:
        raise ValueError(f"Too few admitted source works {count}")
    for left, right in zip(ordinary[::2], ordinary[1::2]):
        yield ("ordinary", *join((left, right)), None)
    print("Source admission", dict(count), flush=True)
    yield {
        "ordinary_rosters": ordinary_hashes, "earlier_ledgers": {name: sha(text) for name, text in ledgers.items()},
        "admission": dict(count), "shards": SHARDS[domain],
    }


def roster(root, directory, domain):
    from app.evidence import document_revision as draft
    from app.evidence import document_revision_writing as paired

    *pairs, source = cases(root, directory, domain)
    rows, items = [], []
    for bank, before, after, truth, variant in pairs:
        comparison, old, new = draft.compare_with_text(before.encode(), after.encode(), ".txt")
        current = paired.separate_revisions(comparison, old, new)
        fallback = paired.anchor_revisions(comparison, old, new) if not current else None
        candidates = current or (fallback[1] if fallback else [])
        geometry = []
        if candidates:
            for candidate, target in zip(candidates, truth, strict=True):
                spans = {side: [candidate.geometry[side]["start"], candidate.geometry[side]["end"]]
                         for side in ("earlier", "later")}
                if spans["later"] != target["later"]:
                    raise ValueError("The candidate did not match the independently labeled sentence")
                geometry.append(spans)
        rows.append({"bank": bank, "source_sha256": [sha(before), sha(after)], "truth": truth,
                     "variant": variant, "scope": "current" if current else "anchored" if fallback else None,
                     "ranges": geometry})
        items.append(candidates)
    calibration = ROOT / "docs/evaluation-panels" / HISTORIC
    earlier = ROOT / "docs/evaluation-panels" / SOURCE_REPLAY
    metadata = {"schema": SCHEMA, "protocol": PROTOCOL, "point": POINT, "sources": source,
                "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "calibration_sha256": hashlib.sha256(calibration.read_bytes()).hexdigest(),
                "prior_report_geometry_sha256": hashlib.sha256(earlier.read_bytes()).hexdigest(), "rows": rows}
    if json.loads(calibration.read_text())["result"]["points"]["joint"] != POINT:
        raise ValueError("The historic operating point changed")
    return metadata, items


def increases(readings):
    if (len(readings) != 2 or any(set(pair) != {"earlier", "later"} for pair in readings)
            or any(type(value) not in (float, int) or not 0 <= value <= 1
                   for pair in readings for value in pair.values())):
        raise ValueError("Expected two complete measured model responses")
    return [math.log(clamped_odds(pair["later"]) / clamped_odds(pair["earlier"])) for pair in readings]


def outcome(rows):
    counts = Counter()
    positives = defaultdict(set)
    sources = defaultdict(set)
    for row in rows:
        scope = row["scope"]
        if scope not in (None, "current", "anchored") or len(row["truth"]) != 2:
            raise ValueError("Unexpected draft scope or missing truth")
        if scope is None and (row["ranges"] or row["readings"]):
            raise ValueError("Ineligible drafts cannot carry model responses")
        ranges = row["ranges"] if scope else [None, None]
        readings = row["readings"] if scope else [None, None]
        for label, extent, measured in zip(row["truth"], ranges, readings, strict=True):
            family, group = label["family"], label["group"]
            if extent is not None and extent["later"] != label["later"]:
                raise ValueError("Saved extent differs from the independently labeled sentence")
            context = increases(measured["context"]) if measured and measured.get("context") else []
            center = increases(measured["center"]) if measured and measured.get("center") else []
            ordinary = family == "ordinary"
            if not ordinary:
                counts["generated"] += 1
            if not context:
                continue
            side = ("ordinary" if ordinary else "generated") + "/" + scope
            counts[f"{side}/context_assessed"] += 1
            incumbent = scope == "current" and min(context) >= math.log(2)
            if ordinary:
                sources[f"{scope}/assessed"].add(group)
            if incumbent:
                if ordinary:
                    sources["incumbent_flags"].add(group)
                else:
                    counts["incumbent_exact"] += 1
            if center:
                counts[f"{side}/joint_assessed"] += 1
                if not incumbent and min(context + center) > POINT:
                    if ordinary:
                        sources[f"{scope}/added_flags"].add(group)
                    else:
                        counts[f"{scope}/added_exact"] += 1
                        positives[f"{scope}/works"].add(group)
                        positives[f"{scope}/families"].add(family)
    result = dict(sorted(counts.items()))
    result["ordinary_incumbent_flags"] = len(sources["incumbent_flags"])
    for scope in ("current", "anchored"):
        result[scope] = {"new_varied_works": len(positives[f"{scope}/works"]),
                         "generator_families": sorted(positives[f"{scope}/families"]),
                         "ordinary_works_assessed": len(sources[f"{scope}/assessed"]),
                         "additional_ordinary_works": len(sources[f"{scope}/added_flags"] - sources["incumbent_flags"])}
        values = result[scope]
        values["passes_local_trial"] = (counts[f"{scope}/added_exact"] >= 5 and values["new_varied_works"] >= 3
                                       and len(values["generator_families"]) == 2 and values["additional_ordinary_works"] <= 2)
    return result


def measure(items):
    from app.detectors.text.extract import _normalise
    from app.evidence import document_revision_writing as paired

    cache = {}
    for candidates in items:
        jobs = [(text, clean) for candidate in candidates for text, clean in zip(
            (candidate.earlier, candidate.later),
            (_normalise(candidate.geometry[side]["text"])[0] for side in ("earlier", "later")), strict=True)]
        inputs = list(dict.fromkeys(s for job in jobs for s in job if s not in cache))
        if inputs:
            measured = paired._score(inputs, allow_partial=True)
            if measured is None:
                raise ValueError("The production paired checkpoints are unavailable")
            cache.update(zip(inputs, zip(*measured, strict=True), strict=True))

        def response(a, b):
            values = [dict(earlier=l, later=r) for l, r in zip(cache[a], cache[b], strict=True)]
            return values if all(None not in row.values() for row in values) else None

        yield [{"context": response(a[0], b[0]), "center": response(a[1], b[1])}
               for a, b in zip(jobs[::2], jobs[1::2], strict=True)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("freeze", "score", "review"))
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--root", type=Path, default=Path("data/independent-evaluation"))
    parser.add_argument("--news", type=Path)
    parser.add_argument("--abstracts", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.mode == "review":
        record = json.loads(args.receipt.read_text())
        counts = outcome(record["rows"])
        if record["schema"] != SCHEMA or record["protocol"] != PROTOCOL or record["point"] != POINT or counts != record["result"]:
            raise ValueError("The receipt does not replay with the frozen comparison")
        print(json.dumps(counts, indent=2))
        return
    if (args.news is None) == (args.abstracts is None):
        parser.error("freeze and score require either --news or --abstracts")
    packet, inputs = roster(args.root, args.news or args.abstracts, "news" if args.news else "abstracts")
    if args.mode == "freeze":
        target = args.receipt
        print(Counter((row["bank"], row["scope"]) for row in packet["rows"]), flush=True)
    else:
        if packet != json.loads(args.receipt.read_text()) or args.output is None:
            raise ValueError("Frozen roster differs or no --output was supplied")
        import torch

        torch.set_num_threads(2)
        for row, readings in zip(packet["rows"], measure(inputs), strict=True):
            row["readings"] = readings
        packet["result"] = outcome(packet["rows"])
        print(json.dumps(packet["result"], indent=2), flush=True)
        target = args.output
    with target.open("x") as stream:
        json.dump(packet, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
