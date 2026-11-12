"""Confirm the frozen exact-sentence writing rule on a separate eLife source panel."""
from __future__ import annotations

import argparse
import math
import os
from collections import Counter
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic

from scripts.video_appearance_native_v108 import read, require, write
from scripts.writing_blind_rewrite_probe import EDIT_PROMPT, text_hash, valid_sentence, words
from scripts.writing_focus_v3 import BASELINE_ASSETS, digest

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "backend/data/independent-evaluation"
REFERENCE = ROOT / "docs/evaluation-panels/writing-multiblock-independent-2026-11-12.json"
PROTOCOL = DATA / "writing-grammar-seams-2026-11-11/protocol.json"
SOURCES = DATA / "writing-elife-body-replication-2026-11-11"
PRIMARY = ROOT / "models/hf/hub/models--wasitaigeneratedcom--ai-text-detector-small/snapshots/f1795c86806e6838d4afa33d0b1427f8430c9615"
SECONDARY = ROOT / "models/hf/hub/models--rasbt--ai-text-detector-modernbert/snapshots/a6efd4bf6ecd852f332b2569ae0f98cae13b7985"
SCHEMA = "chipmunk.writing-elife-exact-sentence-confirmation/1"
GROUPS = 15
FAMILIES = {"granite": 395641, "olmo": 729451}
GENERATION = {"attempts": 3, "temperature": .65, "top_p": .90, "max_tokens": 96, "prompt": EDIT_PROMPT}
METHOD = {
    "scope": "Preflight 15 byline-disjoint, article-level CC BY 4.0 eLife articles with pinned final JATS and normalized 12-token disjointness against eleven retained packets. One plain-text original and one exact ordinary same-work substitution per source. Native bibliography references omitted during prose extraction.",
    "generation": "Three attempts with each of two pinned Apache-2.0 instruction model families. Keep the first completed single-sentence rewrite satisfying the retained length, copy and unchanged-segmentation rules. Record all attempted outputs. Selection uses no classifier responses.",
    "reference": "First six source-ordered native primary contexts at raw .50. The overlap read covers each complete measured context.",
    "exact": "Use the previously published independent sentence score and cutoff without any fit. Compare the unique highest eligible isolated sentence to the cutoff by strict excess. Require an interior focus of 12–65 lexical words and native immediate neighbors of 8–80 words each. The classifier sees only the focus. All measured inputs require one complete tokenizer window.",
    "locations": "A useful location covers at least half the known edited nonwhitespace characters and reaches .30 intersection over union. Record exact offsets, wrong-only marks, misses and work-level false alarms. Union preserves the incumbent marks.",
    "advance": "Require six admitted rewrites per family, all admitted known targets measured, at least two added held work groups across both generator families, and at most one additional ordinary work group compared with the incumbent. A pass supports bounded further development, with no change to file-level confidence.",
}


def assets():
    from scripts.writing_body_localization_transfer import SECONDARY_ASSETS

    prior = read(PROTOCOL)
    generators = {family: prior["weights"][family] for family in FAMILIES}
    for family, record in generators.items():
        target = PROTOCOL.parent / "models" / family
        require(record["license"] == "Apache-2.0" and all(digest(target / name) == pin for name, pin in record["assets"].items()),
                f"Pinned generation files differ for {family}")
    for snapshot, pins in ((PRIMARY, BASELINE_ASSETS), (SECONDARY, SECONDARY_ASSETS)):
        require(all(digest(snapshot / name) == pin for name, pin in pins.items()), "Pinned classifier files differ")
    return {"generators": generators, "generation_runtime": prior["generation_runtime"],
            "primary": BASELINE_ASSETS, "independent": SECONDARY_ASSETS}


def load_sources(preflight):
    from lxml import etree

    from scripts.writing_reference_revision_sources import blocks

    handoff = read(preflight / "handoff.json")
    require(digest(REFERENCE) == handoff["exact_only_reference_sha256"]
            and digest(PROTOCOL) == handoff["generation_protocol_sha256"]
            and digest(SOURCES / "source-receipt.json") == handoff["article_source_receipt_sha256"]
            and all(digest(preflight / name) == pin for name, pin in handoff["files"].items()),
            "One source handoff commitment differs")
    previous = read(REFERENCE)
    point = previous["points"]["independent_focus"]
    require(point == handoff["exact_only_log_odds"] and math.isfinite(point), "Previously published review point differs")
    official = read(SOURCES / "source-receipt.json")["source_versions"]
    raw = read(preflight / "candidates.json")
    donors = read(preflight / "ordinary-donors.json")
    require(len(raw["candidates"]) == GROUPS and raw["candidates"].keys() == donors.keys(),
            "Frozen candidate and ordinary donor sets differ")
    used_names = set()
    for group, candidate in raw["candidates"].items():
        path = SOURCES / "publisher" / f"elife-{group}-v2.xml"
        require(digest(path) == official[group]["xml_sha256"][1] == candidate["after_sha256"], "Publisher bytes differ")
        root = etree.parse(path, etree.XMLParser(resolve_entities=False, no_network=True)).getroot()
        meta = root.find("front/article-meta")
        names = [" ".join(" ".join(author.itertext()).split()).casefold()
                 for author in meta.findall('contrib-group/contrib[@contrib-type="author"]/name')]
        licenses = {node.get("{http://www.w3.org/1999/xlink}href", "").rstrip("/")
                    for node in meta.findall("permissions/license")}
        require(meta.findtext('article-id[@pub-id-type="doi"]') == candidate["after_doi"]
                and names == candidate["authors"] and not used_names.intersection(names)
                and licenses == set(candidate["license"])
                and any(uri in {"http://creativecommons.org/licenses/by/4.0", "https://creativecommons.org/licenses/by/4.0"} for uri in licenses),
                "Article DOI, byline or explicit CC BY license differs")
        used_names.update(names)
        source_blocks = blocks(root.find("body"), True)
        body = source_blocks[candidate["source_block"]].split("\n\n")
        start = candidate["source_first_paragraph"]
        require(candidate["text"] == "\n\n".join(body[start:start + candidate["paragraph_count"]]), "Original article text differs")
        donor = donors[group]
        require(donor["donor"] == source_blocks[donor["source_block"]][slice(*donor["source_range"])]
                and donor["donor"] not in candidate["text"], "Ordinary donor differs from the publisher source")
    return raw["candidates"], donors, point, handoff


def freeze(preflight, run):
    candidates, donors, point, handoff = load_sources(preflight)
    weights = assets()
    run.mkdir(mode=0o700)
    write(run / "source.json", {"articles": candidates, "ordinary": donors})
    write(run / "freeze.json", {
        "schema": SCHEMA, "method": METHOD, "generation": GENERATION, "seeds": FAMILIES, "point": point,
        "runner_sha256": digest(Path(__file__)), "reference_sha256": digest(REFERENCE),
        "source_sha256": digest(run / "source.json"), "preflight_sha256": digest(preflight / "handoff.json"),
        "source_rights": handoff["preflight_counts"], "overlap_source_packets": handoff["normalized_overlap_source_packets"],
        **weights,
    })
    print(f"Froze {len(candidates)} eLife work groups and the published independent point.", flush=True)


def read_lock(run):
    lock = read(run / "freeze.json")
    require(lock["schema"] == SCHEMA and lock["method"] == METHOD and lock["generation"] == GENERATION
            and lock["seeds"] == FAMILIES and lock["runner_sha256"] == digest(Path(__file__))
            and lock["source_sha256"] == digest(run / "source.json"), "Frozen method or source changed")
    return lock


def generate(run, family):
    import mlx.core as mx
    from mlx_lm import load, stream_generate
    from mlx_lm.sample_utils import make_sampler

    lock = read_lock(run)
    require(os.environ.get("HF_HUB_OFFLINE") == "1" and os.environ.get("TRANSFORMERS_OFFLINE") == "1"
            and {name: version(name) for name in lock["generation_runtime"]} == lock["generation_runtime"],
            "Generation requires the pinned offline runtime")
    model_assets = lock["generators"][family]
    directory = PROTOCOL.parent / "models" / family
    require(all(digest(directory / name) == pin for name, pin in model_assets["assets"].items()), "Generator asset changed")
    source = read(run / "source.json")["articles"]
    frozen = digest(run / "freeze.json")
    outputs = run / f"generation-{family}"
    outputs.mkdir(mode=0o700, exist_ok=True)
    mx.set_default_device(mx.gpu)
    mx.set_cache_limit(128 * 1024 * 1024)
    sampler = make_sampler(temp=GENERATION["temperature"], top_p=GENERATION["top_p"])
    started = monotonic()
    with TemporaryDirectory(prefix=f"local-{family}-", dir=run) as folder:
        local = Path(folder)
        for name in model_assets["assets"]:
            if name == "config.json" and model_assets["typed_config_overrides"]:
                write(local / name, read(directory / name) | model_assets["typed_config_overrides"])
            else:
                (local / model_assets["loader_aliases"].get(name, name)).symlink_to((directory / name).resolve())
        model, tokenizer = load(str(local), tokenizer_config={"trust_remote_code": False})
        for ordinal, (group, article) in enumerate(source.items()):
            target = article["text"][slice(*article["focus"])]
            path = outputs / f"{group}.json"
            if path.exists():
                previous = read(path)
                require(previous["freeze_sha256"] == frozen and previous["input_sha256"] == text_hash(target)
                        and len(previous["attempts"]) == GENERATION["attempts"], "A resumed generation record differs")
                continue
            count = len(words(target))
            prompt = GENERATION["prompt"].format(sentence=target, minimum=max(12, int(.65 * count)), maximum=min(40, int(1.4 * count)))
            context = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
            attempts = []
            for attempt in range(GENERATION["attempts"]):
                seed = FAMILIES[family] + ordinal * GENERATION["attempts"] + attempt
                mx.random.seed(seed)
                reply = list(stream_generate(model, tokenizer, context, sampler=sampler, max_tokens=GENERATION["max_tokens"]))
                require(reply and reply[-1].finish_reason in {"stop", "length"}, "Generator returned an incomplete record")
                attempts.append({"seed": seed, "eos": reply[-1].finish_reason == "stop",
                                 "text": "".join(token.text for token in reply)})
            write(path, {"freeze_sha256": frozen, "input_sha256": text_hash(target), "attempts": attempts})
            print(f"{family}: {ordinal + 1}/{GROUPS} source groups", flush=True)
    print(f"{family} completed in {monotonic() - started:.1f}s. Peak MLX allocation {mx.get_peak_memory()} bytes.", flush=True)


def plan(run):
    from app.detectors.text.extract import from_string
    from app.detectors.text.segment import segment

    read_lock(run)
    source = read(run / "source.json")
    frozen = digest(run / "freeze.json")
    cases, primary, exact = [], {}, {}
    admission = {family: Counter() for family in FAMILIES}
    for ordinal, (group, record) in enumerate(source["articles"].items()):
        original = record["text"]
        a, b = record["focus"]
        target = original[a:b]
        doc = from_string(original)
        native = segment(doc)
        source_units = [unit.text for unit in native]
        require(doc.text == original and 4 <= len(doc.paragraphs) <= 6 and 100 <= len(words(original)) <= 600
                and len(native) <= 48 and native, "Original article is outside the frozen geometry")
        index = next(i for i, unit in enumerate(native) if (unit.start, unit.end) == (a, b))
        donor = source["ordinary"][group]["donor"]
        variants = [("original", None), ("ordinary", donor)]
        for family, seed in FAMILIES.items():
            entry = read(run / f"generation-{family}/{group}.json")
            attempts = entry["attempts"]
            require(entry["freeze_sha256"] == frozen and entry["input_sha256"] == text_hash(target)
                    and [row["seed"] for row in attempts] == [seed + ordinal * GENERATION["attempts"] + i for i in range(GENERATION["attempts"])],
                    "Generator result is not bound to this article")
            counts = admission[family]
            counts["groups"] += 1
            counts["attempted"] += len(attempts)
            for attempt in attempts:
                replacement = valid_sentence(attempt, target, generated=True)
                if replacement is None or replacement in original or replacement == donor:
                    counts["unsuitable_attempts_before_selection"] += 1
                    continue
                revised = original[:a] + replacement + original[b:]
                expected = source_units[:index] + [replacement] + source_units[index + 1:]
                if [unit.text for unit in segment(from_string(revised))] != expected:
                    counts["unsuitable_attempts_before_selection"] += 1
                    continue
                variants.append((family, replacement))
                counts["admitted"] += 1
                break
        for role, replacement in variants:
            text = original if replacement is None else original[:a] + replacement + original[b:]
            target_span = None if replacement is None else [a, a + len(replacement)]
            decoded = from_string(text)
            units = segment(decoded)
            expected = source_units if replacement is None else source_units[:index] + [replacement] + source_units[index + 1:]
            require(decoded.text == text and [unit.text for unit in units] == expected
                    and (replacement is None or [units[index].start, units[index].end] == target_span),
                    "An edit does not preserve exact source sentence geometry")
            counts = [len(words(unit.text)) for unit in units]
            native_windows, exact_sentences = [], []
            for i, unit in enumerate(units):
                left = max(0, i - 1)
                group_units = units[left:i + 2]
                context = " ".join(item.text for item in group_units)
                pin = text_hash(context)
                primary[pin] = context
                native_windows.append({"spans": [[item.start, item.end] for item in group_units], "pin": pin})
                if 0 < i < len(units) - 1 and 12 <= counts[i] <= 65 and 8 <= counts[i - 1] <= 80 and 8 <= counts[i + 1] <= 80:
                    pin = text_hash(unit.text)
                    exact[pin] = unit.text
                    exact_sentences.append({"span": [unit.start, unit.end], "pin": pin})
            cases.append({"group": group, "role": role, "text": text, "sha256": text_hash(text), "truth": target_span,
                          "contexts": native_windows, "isolated": exact_sentences})
    write(run / "plan.json", {"freeze_sha256": frozen, "cases": cases, "primary": primary, "independent": exact,
                             "admission": {family: dict(value) for family, value in admission.items()}})
    print(f"Planned {len(cases)} files, {len(primary)} native contexts and {len(exact)} exact sentence inputs.", flush=True)


def score(run, side):
    from scripts.writing_body_localization_transfer import observe

    lock = read_lock(run)
    plan_path = run / "plan.json"
    saved = read(plan_path)
    require(saved["freeze_sha256"] == digest(run / "freeze.json"), "Input plan is from another freeze")
    snapshot = PRIMARY if side == "primary" else SECONDARY
    require(all(digest(snapshot / name) == pin for name, pin in lock[side].items()), "Classifier asset changed")
    observe(snapshot, saved[side], run / f"{side}.json", digest(plan_path), secondary=side == "independent")


def counts(cases):
    roles = ("original", "ordinary", *FAMILIES)
    return {arm: {
        "ordinary_groups_flagged": len({row["group"] for row in cases if row["role"] in roles[:2] and row[arm]["flag"]}),
        "by_role": {role: {
            "files": len(selected),
            "flags": sum(row[arm]["flag"] for row in selected),
            "at_known_location": sum(row[arm]["useful"] for row in selected),
            "wrong_only": sum(row[arm]["flag"] and not row[arm]["useful"] for row in selected if role in FAMILIES),
        } for role in roles for selected in [[row for row in cases if row["role"] == role]]},
    } for arm in ("current", "exact", "union")}


def positions(text, truth, spans):
    occupied = {i for a, b in spans for i in range(a, b) if not text[i].isspace()}
    return [len(truth & occupied), len(truth | occupied), len(truth)]


def summarize(case, primary, independent, point):
    from scripts.writing_multiblock_independent_transfer import log_odds, useful

    text = case["text"]
    truth = set() if case["truth"] is None else {i for i in range(*case["truth"]) if not text[i].isspace()}
    original = [{"spans": row["spans"], "overlap": positions(text, truth, row["spans"]),
                 "score": primary[row["pin"]]["score"]} for row in case["contexts"]]
    measured = [{"span": row["span"], "overlap": positions(text, truth, [row["span"]]),
                 "score": independent[row["pin"]]["score"]} for row in case["isolated"]]
    current_marks = [row["overlap"] for row in original if row["score"] >= .5][:6]
    rank = sorted(measured, key=lambda item: log_odds(item["score"]), reverse=True)
    hit = bool(rank and log_odds(rank[0]["score"]) > point
               and (len(rank) == 1 or log_odds(rank[0]["score"]) > log_odds(rank[1]["score"])))
    current = {"flag": bool(current_marks), "useful": any(map(useful, current_marks))}
    exact = {"flag": hit, "useful": hit and useful(rank[0]["overlap"])}
    return {"group": text_hash(case["group"])[:16], "role": case["role"], "sha256": case["sha256"], "truth": case["truth"],
            "native": original, "isolated": measured, "target_measured": any(useful(row["overlap"]) for row in measured),
            "current": current, "exact": exact,
            "union": {name: current[name] or exact[name] for name in ("flag", "useful")}}


def publish(run, path):
    lock = read_lock(run)
    plan_path = run / "plan.json"
    saved = read(plan_path)
    packets = {side: read(run / f"{side}.json") for side in ("primary", "independent")}
    require(saved["freeze_sha256"] == digest(run / "freeze.json"), "Input plan is from another freeze")
    for side, packet in packets.items():
        require(packet["binding"] == digest(plan_path) and packet["responses"].keys() == saved[side].keys()
                and all(value["coverage"]["token_windows"] == 1 and value["coverage"]["tokens"] > 0
                        for value in packet["responses"].values()), "Responses differ from the complete frozen inputs")
    rows = [summarize(case, packets["primary"]["responses"], packets["independent"]["responses"], lock["point"])
            for case in saved["cases"]]
    generated = [row for row in rows if row["role"] in FAMILIES]
    gain = [row for row in generated if row["exact"]["useful"] and not row["current"]["useful"]]
    ordinary = [row for row in rows if row["role"] in {"original", "ordinary"}]
    new_groups = {row["group"] for row in ordinary if row["exact"]["flag"]} - {
        row["group"] for row in ordinary if row["current"]["flag"]}
    checks = {"minimum_per_family": all(saved["admission"][family].get("admitted", 0) >= 6 for family in FAMILIES),
              "known_targets_measured": all(row["target_measured"] for row in generated),
              "added_groups": len({row["group"] for row in gain}),
              "added_families": sorted({row["role"] for row in gain}),
              "added_ordinary_groups": len(new_groups)}
    passed = (checks["minimum_per_family"] and checks["known_targets_measured"] and checks["added_groups"] >= 2
              and checks["added_families"] == sorted(FAMILIES) and checks["added_ordinary_groups"] <= 1)
    write(path, {
        "schema": SCHEMA, "method": METHOD, "point": lock["point"], "freeze_sha256": digest(run / "freeze.json"),
        "private_sha256": {name: digest(run / name) for name in ("source.json", "plan.json", "primary.json", "independent.json")},
        "source_rights": lock["source_rights"], "overlap_source_packets": lock["overlap_source_packets"],
        "generation_admission": saved["admission"], "counts": counts(rows), "checks": checks, "advanced": passed, "cases": rows,
        "execution": {side: {"forwards": packet["forwards"], "inputs": len(packet["responses"]), "seconds": packet["seconds"]}
                      for side, packet in packets.items()},
    })
    print({"counts": counts(rows), "checks": checks, "advanced": passed}, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "generate", "plan", "primary", "independent", "publish"))
    parser.add_argument("run", type=Path)
    parser.add_argument("--preflight", type=Path)
    parser.add_argument("--family", choices=FAMILIES)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.phase == "freeze":
        freeze(args.preflight, args.run)
    elif args.phase == "generate":
        generate(args.run, args.family)
    elif args.phase == "plan":
        plan(args.run)
    elif args.phase in {"primary", "independent"}:
        score(args.run, args.phase)
    else:
        publish(args.run, args.output)


if __name__ == "__main__":
    main()
