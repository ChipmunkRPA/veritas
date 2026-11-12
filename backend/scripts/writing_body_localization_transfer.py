"""Evaluate exact sentence evidence across archived multi-paragraph article and MIME bodies."""

from __future__ import annotations

import argparse
import os
from collections import Counter, defaultdict
from email.message import EmailMessage
from email.policy import SMTP
from pathlib import Path
from statistics import median
from time import monotonic

from scripts import writing_paired_revision_replication as generation
from scripts.video_appearance_native_v108 import read, require, write
from scripts.writing_archive_author_transfer import POINT, odds
from scripts.writing_blind_rewrite_probe import generator_configuration, text_hash, valid_sentence, words
from scripts.writing_focus_v3 import BASELINE_ASSETS, digest

SCHEMA = "chipmunk.writing-body-localization-transfer/1"
ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "backend/data/independent-evaluation"
PRIOR = BASE / "writing-archive-prospective-transfer-2026-11-12"
ARCHIVE = BASE / "writing-archive-author-transfer-2026-11-11"
PRIOR_PUBLIC = ROOT / "docs/evaluation-panels/writing-archived-passage-transfer-2026-11-12.json"
MAX_EXPOSED = 12
MAX_FRESH = 10
SECONDARY_ASSETS = {
    "config.json": "dc4511bb21dfa4991688aba8548086c8fa6947d94018ac1a8d09ae9843749703",
    "tokenizer.json": "6c8aaa9a542084f2457eab775d4eeb51f92a70c0fd9de28d5edb0ddec3c08d30",
    "model.safetensors": "e7f7a617ad4af3abd8bcb939d564d2a2ea36fdb24b893177ae10c12ebe869361",
    "detector-config.json": "e72a616a5cbb0bbbe7b75d3d5e40af3c5cafdb0f0a8299ce04c5921e046619c7",
    "LICENSE": "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4",
}
ARMS = ("current", "relative", "confirmed", "current_plus_confirmed")
METHOD = {
    "scope": "Same archive site with source-disjoint fresh works and separate source-exposed diagnostic works. Use real article body paragraphs in source order around the unchanged target. Evaluate both extracted plain text and controlled multipart email containers with genuine quoted prose.",
    "selection": "Freeze the original 36 source commitments. Replay metadata and source qualification with room for at most 10 additional byline-disjoint works, no prior outcome inputs. Select the first 12 qualified source-exposed works by a hash of the work identifier. No coverage exclusion after scoring.",
    "fixture": "Select up to six supported real body prose blocks nearest the exact target within the same article, minimum four. Preserve each original paragraph and its order. Non-prose or unsupported markup can create gaps. Require 100–600 original words and at most 48 native sentences. Construct one exact center replacement and a genuine quote tail per message.",
    "current": "First six complete native three-sentence context windows at raw probability at least .50, in source order. This benchmark compares the incumbent window rule.",
    "relative": "Measure exact focus and neighbor sentences in isolation. Consider only native three-sentence body contexts wholly inside a single paragraph, with focus 12–40 words and 6–65-word neighbors. Require positive isolated focus log-odds margin over both neighbors, incumbent native log odds over the fixed PEP point, and native odds at least the median of all eligible contexts in the same paragraph. Select only the uniquely largest isolated margin across body paragraphs.",
    "confirmed": "Require an independent checkpoint to assign at least .50 to that same exact isolated sentence, with complete token coverage. Measure only unique top-ranked candidates. No retry or alternative after rejection.",
    "current_plus_confirmed": "Retain incumbent windows and add the confirmed exact sentence. Report highlights in unrelated body and quoted paragraphs separately.",
    "localization": "Per displayed location nonwhitespace IoU at least .30 and recall at least .50 with the known edit. Also report wrong-only results, any unrelated paragraph touched, quote flags and exact target eligibility.",
    "advance": "Use only fresh work groups for promotion comparisons. Require either two independent added useful work groups with at least one in each generator and at most one new human flag group, or retain all current useful locations and clear at least two genuine false-flag groups. At least six eligible fresh groups and at least six admitted edits per generator are required. Formats are paired within each group.",
}


def assets(snapshot, second):
    from scripts.text_elife_rewrites_v273 import GENERATORS, MLX_VERSIONS

    require(all(digest(snapshot / name) == pin for name, pin in BASELINE_ASSETS.items()), "Primary checkpoint differs")
    require(all(digest(second / name) == pin for name, pin in SECONDARY_ASSETS.items()), "Independent checkpoint differs")
    config = generator_configuration()
    require(all(config[key]["license"] == "Apache-2.0" and digest(value[3] / "model.safetensors") == value[2]
                for key, value in GENERATORS.items()), "Generation license or checkpoint differs")
    return {"primary": BASELINE_ASSETS, "independent": SECONDARY_ASSETS, "generators": config, "runtime": MLX_VERSIONS}


def protocol(run, snapshot, second):
    public = read(PRIOR_PUBLIC)
    return {"schema": SCHEMA, "method": METHOD, "point": POINT, "code_sha256": digest(Path(__file__)),
            "source_roster_sha256": digest(run / "sources.json"), "fresh_originals_sha256": digest(run / "originals.json"),
            "prior_public_sha256": digest(PRIOR_PUBLIC), "prior_generation_sha256": {
                key: digest(PRIOR / f"generated-{key}.json") for key in generation.SEEDS},
            "prior_originals_sha256": public["audit"]["commitments"]["originals.json"],
            "prior_source_freeze_sha256": public["audit"]["commitments"]["source-freeze.json"],
            "support_sha256": {name: digest(ROOT / name) for name in (
                "backend/scripts/writing_archive_prospective_intake.py",
                "backend/scripts/writing_archive_author_intake.py",
                "backend/scripts/writing_paired_email_revision.py",
                "backend/scripts/writing_paired_revision_replication.py",
                "backend/app/detectors/text/extract.py", "backend/app/detectors/text/segment.py",
                "backend/app/detectors/text/coverage.py", "backend/app/detectors/text/classifier.py",
            )},
            "seeds": generation.SEEDS, "prompt": generation.EDIT_PROMPT, "settings": generation.SETTINGS,
            **assets(snapshot, second)}


def article(row, repository):
    from app.detectors.text.extract import from_string
    from app.detectors.text.segment import segment
    from scripts import writing_archive_author_intake as history
    from scripts import writing_archive_prospective_intake as prose

    source = row["source"]
    commit = source["commit"] if "commit" in source else source["before"]
    expected = source["blob_sha256"] if "commit" in source else source["blob_sha256"][0]
    raw = history.git(repository, "show", f"{commit}:{source['path']}")
    require(text_hash(raw) == expected, "Original archive article differs")
    front = prose.FRONT.match(raw)
    writer = prose.byline(raw, front) if front else None
    forbidden = [match.span() for match in prose.FENCE.finditer(raw)] + ([writer["span"]] if writer else [])
    witness = row["before"]["witness"][2]
    source_range = witness["raw_span"]
    require(text_hash(raw[slice(*source_range)]) == witness["raw_sha256"], "Archived target span differs")
    body, center = [], None
    for start, stop in history.region_blocks(raw, True):
        if any(start < hi and lo < stop for lo, hi in forbidden):
            continue
        text, mapping = prose.visible(raw, start, stop)
        doc = from_string(text)
        units = segment(doc) if text.endswith((".", "!", "?")) and doc.text == text else []
        if not units or not all(unit.text.endswith((".", "!", "?")) for unit in units):
            continue
        if start <= source_range[0] < source_range[1] <= stop:
            for unit in units:
                scope = [mapping[unit.start][0], mapping[unit.end - 1][1]]
                if unit.text == witness["text"] and scope == source_range:
                    center = len(body), unit.start, unit.end
        body.append({"text": text, "source_span": [start, stop], "source_sha256": text_hash(raw[start:stop]),
                     "sentences": len(units), "words": len(words(text))})
    if center is None:
        return None
    candidates = []
    for count in range(4, min(6, len(body)) + 1):
        for i in range(max(0, center[0] - count + 1), min(center[0] + 1, len(body) - count + 1)):
            sample = body[i:i + count]
            nwords = sum(item["words"] for item in sample)
            nsents = sum(item["sentences"] for item in sample)
            if 100 <= nwords <= 600 and nsents <= 48:
                # Prefer balanced real context, then more source paragraphs.
                position = center[0] - i
                candidates.append(((min(position, count - 1 - position), count, -i), i, sample))
    if not candidates:
        return None
    _, begin, selected = max(candidates)
    number, lo, hi = center
    focus = [number - begin, lo, hi]
    return {"group": row["group"], "path": source["path"], "commit": commit, "archive_blob_sha256": expected,
            "paragraphs": selected, "paragraphs_in_archive": len(body), "focus": focus,
            "target_raw_span": source_range, "quote": row["after"]["text"], "quote_sha256": text_hash(row["after"]["text"]),
            "known_genuine_alternate_sha256": text_hash(row["after"]["focus_text"]),
            "before": {"focus_text": witness["text"]}}


def prepare(run):
    from scripts import writing_archive_author_intake as history
    from scripts import writing_archive_prospective_intake as prose

    public = read(PRIOR_PUBLIC)
    require(all(digest(PRIOR / name) == public["audit"]["commitments"][name]
                for name in ("originals.json", "source-freeze.json")), "Published previous source commitments differ")
    frozen = read(PRIOR / "source-freeze.json")
    repo = ARCHIVE / history.SOURCES["kubernetes"]["archive"]
    require(prose.freeze(repo, ARCHIVE) == frozen, "Previous source-only freeze differs")
    already = read(PRIOR / "originals.json")
    original_limit = prose.MAX_GROUPS
    try:
        prose.MAX_GROUPS += MAX_FRESH
        extended, _ = prose.qualify(repo, ARCHIVE, frozen)
    finally:
        prose.MAX_GROUPS = original_limit
    require(extended[:len(already)] == already, "Fresh source extension must preserve the admitted prefix")
    exposed = sorted(already, key=lambda row: text_hash(row["group"]))
    fresh = extended[len(already):]
    require(not ({row["group"] for row in exposed} & {row["group"] for row in fresh}), "Work group overlap")
    prepared, counts = [], defaultdict(Counter)
    for scope, source in (("exposed_rewrite_transfer", exposed), ("fresh", fresh)):
        for row in source:
            item = article(row, repo)
            counts[scope]["considered"] += 1
            if item is None:
                counts[scope]["unsupported_article_geometry"] += 1
                continue
            if scope == "exposed_rewrite_transfer" and counts[scope]["qualified"] >= MAX_EXPOSED:
                counts[scope]["qualified_over_cap"] += 1
                continue
            counts[scope]["qualified"] += 1
            prepared.append(item | {"scope": scope})
    require(any(row["scope"] == "fresh" for row in prepared), "No eligible fresh archive articles")
    output = {"schema": SCHEMA, "counts": {scope: dict(row) for scope, row in counts.items()}, "works": prepared}
    run.mkdir(exist_ok=True)
    write(run / "sources.json", output)
    write(run / "originals.json", [row for row in prepared if row["scope"] == "fresh"])
    print(output["counts"], flush=True)


def mime(body, quote):
    message = EmailMessage(policy=SMTP)
    message["From"] = "archive-fixture@example.test"
    message["To"] = "fixture-reader@example.test"
    message["Subject"] = "Controlled writing localization fixture"
    quoted = "> " + quote
    message.set_content(f"{body}\n\n{quoted}", charset="utf-8", cte="quoted-printable")
    message.add_alternative("<p>Excluded alternate fixture body.</p>", subtype="html")
    message.add_attachment(b"excluded fixture attachment", maintype="application", subtype="octet-stream", filename="excluded.bin")
    message.set_boundary("chipmunk-localization-fixture-mixed")
    next(message.iter_parts()).set_boundary("chipmunk-localization-fixture-alternative")
    return message.as_bytes(), quoted


def edits(run):
    frozen = digest(run / "freeze.json")
    prior = {family: read(PRIOR / f"generated-{family}.json")["results"] for family in generation.SEEDS}
    fresh = {}
    for family in generation.SEEDS:
        packet = read(run / f"generated-{family}.json")
        require(packet["freeze_sha256"] == frozen, "Generator freeze differs")
        fresh[family] = packet["results"]
    accepted = read(run / "sources.json")["works"]
    require(all(set(packet) == {row["group"] for row in accepted if row["scope"] == "fresh"}
                for packet in fresh.values()), "Fresh generation work roster differs")
    attempts = defaultdict(Counter)
    yield_rows = []
    for row in accepted:
        cases = {"human": None}
        original = row["before"]["focus_text"]
        batch = fresh if row["scope"] == "fresh" else prior
        for family in generation.SEEDS:
            attempt = batch[family][row["group"]]
            require(attempt["input_sha256"] == text_hash(original), "Unexpected generator input")
            counts = attempts[f"{row['scope']}/{family}"]
            counts["attempted"] += 1
            revised = valid_sentence(attempt, original, generated=True)
            if revised is None or text_hash(revised) == row["known_genuine_alternate_sha256"]:
                counts["excluded_response"] += 1
                continue
            cases[family] = revised
        yield_rows.append((row, cases))
    return yield_rows, attempts


def gather(run, *, create):
    from app.detectors.text.extract import extract, from_string
    from app.detectors.text.segment import segment
    from scripts.writing_paired_email_revision import encoded, native_contexts

    entries, attempts = edits(run)
    result, values = [], {}
    files = run / "fixtures"
    if create:
        files.mkdir()
    for row, cases in entries:
        original_blocks = [part["text"] for part in row["paragraphs"]]
        position, lo, hi = row["focus"]
        old_block = original_blocks[position]
        truth_start = sum(len(part) + 2 for part in original_blocks[:position]) + lo
        for family, replacement in cases.items():
            blocks = original_blocks[:]
            truth = None
            if replacement is not None:
                blocks[position] = old_block[:lo] + replacement + old_block[hi:]
                truth = [truth_start, truth_start + len(replacement)]
                native = segment(from_string(old_block))
                original_units = [unit.text for unit in native]
                revised_units = [unit.text for unit in segment(from_string(blocks[position]))]
                focus = next(unit.idx for unit in native if (unit.start, unit.end) == (lo, hi))
                expected = original_units[:focus] + [replacement] + original_units[focus + 1:]
                if revised_units != expected:
                    attempts[f"{row['scope']}/{family}"]["excluded_article_segmentation"] += 1
                    continue
                attempts[f"{row['scope']}/{family}"]["admitted"] += 1
            text = "\n\n".join(blocks)
            message, quote = mime(text, row["quote"])
            for fmt, data, expected in (("txt", text.encode(), text), ("eml", message, f"{text}\n\n{quote}")):
                path = files / f"{row['group']}-{family}.{fmt}"
                if create:
                    with path.open("xb") as stream:
                        stream.write(data)
                require(path.read_bytes() == data, "Fixture bytes differ")
                doc = extract(path, "." + fmt)
                require(doc.text == expected, "Decoded fixture prose or its role separator changed")
                roles = [item.get("quote_role", "body") for item in doc.source_map]
                require(roles == ["body"] * len(blocks) + (["quoted"] if fmt == "eml" else []),
                        "Exact paragraph source map changed")
                if fmt == "eml":
                    require(doc.meta["email"]["body_content_type"] == "text/plain"
                            and len(doc.meta["email"]["attachments"]) == 1, "Wrong MIME body selected")
                units = segment(doc)
                windows, single = [], []
                for unit, group in zip(units, native_contexts(units), strict=True):
                    window, native_text = encoded(group)
                    values[window["sha256"]] = native_text
                    windows.append(window | {"paragraph": unit.paragraph, "focus": [unit.start, unit.end]})
                    single.append(text_hash(unit.text))
                eligible = []
                for i in range(1, len(units) - 1):
                    left, mid, right = units[i - 1:i + 2]
                    if (left.paragraph == mid.paragraph == right.paragraph and roles[mid.paragraph] == "body"
                            and 12 <= len(words(mid.text)) <= 40 and all(6 <= len(words(item.text)) <= 65 for item in (left, right))):
                        eligible.append(i)
                        for item in (left, mid, right):
                            values[text_hash(item.text)] = item.text
                if truth:
                    require(any([unit.start, unit.end] == truth and roles[unit.paragraph] == "body" for unit in units)
                            and doc.text[slice(*truth)] == replacement, "Known edit no longer maps to one native body sentence")
                result.append({"group": row["group"], "scope": row["scope"], "kind": family, "format": fmt,
                               "file": path.name, "file_sha256": digest(path), "text_sha256": text_hash(doc.text),
                               "paragraphs": [list(bounds) for bounds in doc.paragraphs], "roles": roles, "truth": truth,
                               "windows": windows, "isolated": single, "eligible": eligible})
    return result, values, {key: dict(value) for key, value in attempts.items()}


def shortlist(row, primary):
    by_paragraph = defaultdict(list)
    for index in row["eligible"]:
        entry = row["windows"][index]
        by_paragraph[entry["paragraph"]].append(odds(primary[entry["sha256"]]["score"]))
    margins = []
    for index in row["eligible"]:
        entry = row["windows"][index]
        native = odds(primary[entry["sha256"]]["score"])
        a, b, c = (odds(primary[key]["score"]) for key in row["isolated"][index - 1:index + 2])
        contrast = b - max(a, c)
        if native > POINT and native >= median(by_paragraph[entry["paragraph"]]) and contrast > 0:
            margins.append((contrast, index))
    margins.sort(reverse=True)
    return margins[0][1] if margins and (len(margins) == 1 or margins[0][0] > margins[1][0]) else None


def independent_inputs(rows, primary, values):
    selections, inputs = [], {}
    for row in rows:
        index = shortlist(row, primary)
        selections.append(index)
        if index is not None:
            key = row["isolated"][index]
            inputs[key] = values[key]
    return selections, inputs


def observe(snapshot, values, output, binding, *, secondary):
    import torch

    from scripts.writing_paired_email_revision import native_scores

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    begin = monotonic()
    responses, forwards = native_scores(snapshot, values, second=secondary) if values else ({}, 0)
    write(output, {"binding": binding, "responses": responses, "forwards": forwards, "seconds": round(monotonic() - begin, 3)})
    print({"stage": "independent" if secondary else "primary", "contexts": len(values), "forwards": forwards}, flush=True)


def span_positions(text, spans):
    return {i for a, b in spans for i in range(a, b) if not text[i].isspace()}


def evaluate(run, rows, primary, secondary, selections):
    from app.detectors.text.extract import extract

    output = []
    for row, selected in zip(rows, selections, strict=True):
        path = run / "fixtures" / row["file"]
        doc = extract(path, "." + row["format"])
        truth = span_positions(doc.text, [row["truth"]]) if row["truth"] else set()
        para = next((i for i, (a, b) in enumerate(row["paragraphs"]) if row["truth"] and a <= row["truth"][0] < b), None)
        current = [item["spans"] for item in row["windows"] if primary[item["sha256"]]["score"] >= .5][:6]
        relative = [[row["windows"][selected]["focus"]]] if selected is not None else []
        confirmed = relative if selected is not None and secondary[row["isolated"][selected]]["score"] >= .5 else []
        marks = dict(zip(ARMS, (current, relative, confirmed, current + confirmed), strict=True))
        metrics = {}
        for arm, regions in marks.items():
            guess = [span_positions(doc.text, ranges) for ranges in regions]
            touched = {i for i, (a, b) in enumerate(row["paragraphs"])
                       if any(any(a <= char < b for char in span) for span in guess)}
            useful = bool(truth) and any(len(span & truth) * 10 >= 3 * len(span | truth) and len(span & truth) * 2 >= len(truth)
                                        for span in guess)
            metrics[arm] = {"flag": bool(regions), "useful": useful, "marked": regions,
                            "outside_target_paragraph": bool(truth) and any(i != para for i in touched),
                            "quote_flag": any(row["roles"][i] == "quoted" for i in touched)}
        output.append({name: row[name] for name in ("group", "scope", "kind", "format", "file_sha256", "truth", "roles")} | {
            "body_paragraphs": row["roles"].count("body"),
            "eligible_body_paragraphs": len({row["windows"][i]["paragraph"] for i in row["eligible"]}),
            "target_eligible": any(row["windows"][i]["focus"] == row["truth"] for i in row["eligible"]),
            "arms": metrics,
        })
    return output


def aggregate(rows, admission):
    bins = defaultdict(list)
    for row in rows:
        for key in (row["scope"], f"{row['scope']}/{row['format']}", f"{row['scope']}/{row['format']}/{row['kind']}"):
            bins[key].append(row)
    output = {}
    for key, group in bins.items():
        controls = [item for item in group if item["kind"] == "human"]
        rewrites = [item for item in group if item["kind"] != "human"]
        output[key] = {"human": len(controls), "mixed": len(rewrites), "target_eligible": sum(r["target_eligible"] for r in rewrites),
                       "body_paragraphs": sum(r["body_paragraphs"] for r in group),
                       "eligible_body_paragraphs": sum(r["eligible_body_paragraphs"] for r in group),
                       "arms": {arm: {"human_flagged": sum(r["arms"][arm]["flag"] for r in controls),
                                      "human_flagged_groups": len({r["group"] for r in controls if r["arms"][arm]["flag"]}),
                                      "useful": sum(r["arms"][arm]["useful"] for r in rewrites),
                                      "wrong_only": sum(r["arms"][arm]["flag"] and not r["arms"][arm]["useful"] for r in rewrites),
                                      "no_flag": sum(not r["arms"][arm]["flag"] for r in rewrites),
                                      "outside_target_paragraph": sum(r["arms"][arm]["outside_target_paragraph"] for r in rewrites),
                                      "quote_flags": sum(r["arms"][arm]["quote_flag"] for r in group)} for arm in ARMS}}
    fresh = [row for row in rows if row["scope"] == "fresh"]
    decisions = {}
    for arm in ARMS[1:]:
        gains, losses, extra, cleared = set(), set(), set(), set()
        for row in fresh:
            before, after = row["arms"]["current"], row["arms"][arm]
            if row["kind"] == "human":
                if after["flag"] and not before["flag"]:
                    extra.add(row["group"])
                if before["flag"] and not after["flag"]:
                    cleared.add(row["group"])
            elif after["useful"] and not before["useful"]:
                gains.add((row["group"], row["kind"]))
            elif before["useful"] and not after["useful"]:
                losses.add(row["group"])
        adequate = (len({r["group"] for r in fresh}) >= 6
                    and all(admission.get(f"fresh/{family}", {}).get("admitted", 0) >= 6 for family in generation.SEEDS))
        gain_groups = {group for group, _ in gains}
        decisions[arm] = {"gained_groups": len(gain_groups), "gained_families": sorted({family for _, family in gains}),
                          "lost_groups": len(losses), "new_human_flag_groups": len(extra), "cleared_human_flag_groups": len(cleared),
                          "adequate": adequate, "advances": adequate and not losses and (
                              len(gain_groups) >= 2 and {family for _, family in gains} == set(generation.SEEDS) and len(extra) <= 1
                              or len(cleared) >= 2 and not extra)}
    return {"counts": output, "admission": admission, "decisions": decisions}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("phase", choices=("prepare", "freeze", "generate", "lock-inputs", "primary", "independent", "report"))
    cli.add_argument("run", type=Path)
    cli.add_argument("snapshot", type=Path)
    cli.add_argument("second", type=Path)
    cli.add_argument("--family", choices=generation.SEEDS)
    args = cli.parse_args()
    os.umask(0o077)
    run, phase = args.run, args.phase
    if phase == "prepare":
        prepare(run)
        return
    require(os.getenv("HF_HUB_OFFLINE") == os.getenv("TRANSFORMERS_OFFLINE") == "1", "Require local model snapshots")
    locked = protocol(run, args.snapshot, args.second)
    if phase == "freeze":
        write(run / "freeze.json", locked)
        return
    require(read(run / "freeze.json") == locked, "Frozen implementation differs")
    if phase == "generate":
        require(args.family, "Choose one generator")
        generation.generate(run, run, locked, args.family)
        return
    rows, values, admission = gather(run, create=phase == "lock-inputs")
    binding = {"rows": rows, "inputs": sorted(values), "admission": admission, "freeze_sha256": digest(run / "freeze.json"),
               "generated_sha256": {family: digest(run / f"generated-{family}.json") for family in generation.SEEDS}}
    inputs = run / "inputs.json"
    if phase == "lock-inputs":
        write(inputs, binding)
        print({"cases": len(rows), "primary_inputs": len(values), "admission": admission}, flush=True)
        return
    require(read(inputs) == binding, "Exact frozen fixture inputs differ")
    primary, independent = run / "primary.json", run / "independent.json"
    if phase == "primary":
        observe(args.snapshot, values, primary, digest(inputs), secondary=False)
        return
    packet = read(primary)
    require(packet["binding"] == digest(inputs) and packet["responses"].keys() == values.keys(), "Primary reading inventory differs")
    indexes, shortlist = independent_inputs(rows, packet["responses"], values)
    short = run / "shortlist.json"
    short_lock = {"input_sha256": digest(inputs), "primary_sha256": digest(primary),
                  "selected": indexes, "inputs": sorted(shortlist)}
    if phase == "independent":
        write(short, short_lock)
        observe(args.second, shortlist, independent, digest(short), secondary=True)
        return
    require(read(short) == short_lock, "Independent candidate input inventory differs")
    second = read(independent)
    require(second["binding"] == digest(short) and second["responses"].keys() == shortlist.keys(), "Independent observations differ")
    results = evaluate(run, rows, packet["responses"], second["responses"], indexes)
    write(run / "outcomes.json", results)
    write(run / "aggregate.json", aggregate(results, admission))
    print({"results": len(results), "primary_forwards": packet["forwards"], "independent_forwards": second["forwards"]}, flush=True)


if __name__ == "__main__":
    main()
