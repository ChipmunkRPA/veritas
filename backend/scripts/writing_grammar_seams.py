"""Evaluate local changes in grammatical token sequences at paragraph boundaries."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from collections import Counter, defaultdict
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory

SCHEMA = "chipmunk.writing-grammar-seams/1"
FAMILIES = ("olmo", "tiny", "granite")
PUBLISHERS = ("census", "copyright")
SEED = {"olmo": 389033, "tiny": 877631, "granite": 667279}
GENERATION_RUNTIME = {"mlx": "0.32.4", "mlx-lm": "0.31.3", "transformers": "5.22.1",
                      "huggingface_hub": "2.4.0", "tokenizers": "0.23.3"}
ORDERS = (1, 3)
WORD = re.compile(r"[^\W_]+(?:[’'-][^\W_]+)*")
TOKEN = re.compile(r"[^\W_]+(?:[’'-][^\W_]+)*|[.!?;:,-]")
CLASSES = {
    "article": "a an the this that those these some any each every such either both neither all no other",
    "pronoun": "i you he she it we they me him her us them my your his its our their mine yours herself himself itself themselves",
    "aux": "am are is was were be been being have has had having do does did doing",
    "modal": "can could may might must shall should will would",
    "coord": "and but or nor yet",
    "clause": "if because although though unless while whereas whether which who whom whose what when where why how",
    "preposition": "above across after against along among around as at before behind below beneath beside between by despite down during except for from in into near of off on onto out over past since through to toward under until upon via with within without",
    "adverb": "again also already always ever just never not now only quite rather really still then too very",
}
CLASS = {word: key for key, members in CLASSES.items() for word in members.split()}
PROMPT = ("Rewrite the paragraph below using different wording and sentence structure. Keep its meaning, names, and "
          "numbers. Write only one complete paragraph of approximately {count} words, with no title or explanation.\n\n{body}")
METHOD = {
    "candidate": "Classify closed-class words and punctuation; collapse other lexical words to one symbol. Compare normalized symbol sequence counts using squared Hellinger distance.",
    "seam": "Smaller left/target and target/right distance, minus left/right distance. Select at most one uniquely highest interior paragraph above the operating point.",
    "diagnostic": "Repeat with one-symbol distributions to isolate what the sequence signal adds.",
    "fit": "Use the inclusive nearest-rank 90th percentile of maximum eligible original and human-edited scores per development group. Above means strictly greater.",
    "baseline": "Existing raw classifier score of at least .5, with first six native positive sentence contexts in source order.",
    "truth": "Exact non-whitespace character intersection over union of at least .30 against the full known replacement.",
    "advance": "Require at least two generator families with admitted edits. Adding the sequence cue must recover at least one missed edit group in each publisher and each represented generator family, with at most one newly flagged human source group total.",
}


def require(value, explanation):
    if not value:
        raise ValueError(explanation)


def sha(value):
    if isinstance(value, Path):
        with value.open("rb") as stream:
            return hashlib.file_digest(stream, "sha256").hexdigest()
    return hashlib.sha256(value.encode()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    with path.open("x") as stream:
        stream.write(json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2) + "\n")


def rows(path):
    with path.open() as stream:
        return [json.loads(line) for line in stream]


def symbols(value):
    mapped = []
    for match in TOKEN.finditer(value.casefold()):
        token = match.group()
        if token in ".!?":
            mapped.append("sentence")
        elif token in ";:,-":
            mapped.append(token)
        else:
            mapped.append(CLASS.get(token, "number" if any(char.isdecimal() for char in token) else "content"))
    return mapped


def grams(sequence, order):
    return Counter(tuple(sequence[index:index + order]) for index in range(len(sequence) - order + 1))


def distance(left, right):
    scale = math.sqrt(sum(left.values()) * sum(right.values()))
    return 1 - math.fsum(math.sqrt(left[key] * right[key]) for key in sorted(left.keys() & right.keys())) / scale


def seam_scores(value, bounds, order):
    paragraphs = [symbols(value[a:b]) for a, b in bounds]
    bag = [grams(part, order) if len(part) >= 20 else None for part in paragraphs]
    return {
        index: min(distance(left, focus), distance(focus, right)) - distance(left, right)
        for index, (left, focus, right) in enumerate(zip(bag, bag[1:], bag[2:]), 1)
        if left is not None and focus is not None and right is not None
    }


def select(scores, cutoff):
    if not scores:
        return None
    highest = max(scores.values())
    winners = [index for index, value in scores.items() if abs(value - highest) <= 1e-12]
    return winners[0] if highest > cutoff and len(winners) == 1 else None


def normalize(text):
    text = re.sub(r"(\w)-\s*\n\s*(?=\w)", r"\1", text)
    return " ".join(text.split())


def source_blocks(path, publisher):
    if publisher == "census":
        from lxml import html

        tree = html.fromstring(path.read_bytes())
        title = tree.xpath("string(//meta[@property='og:title']/@content)") or tree.xpath("string(//title)")
        date = tree.xpath("string(//meta[@property='article:published_time']/@content)")[:10]
        reviewed = tree.xpath("string(//meta[@property='article:modified_time']/@content)")[:10]
        units = []
        for index, item in enumerate(tree.xpath("//p")):
            parents = list(item.iterancestors())
            classes = {name for parent in parents for name in parent.get("class", "").split()}
            if not ({"uscb-text-media-text", "uscb-main-responsivegrid"} & classes):
                continue
            if "uscb-caption-text" in classes or any(parent.tag in ("blockquote", "figure", "aside", "li") for parent in parents):
                continue
            units.append(([index], normalize(item.text_content())))
    else:
        import pymupdf

        with pymupdf.open(path) as document:
            title = document.metadata.get("title") or path.stem
            date = (document.metadata.get("creationDate") or "")[2:10]
            reviewed = (document.metadata.get("modDate") or "")[2:10]
            units = [([i, int(block[5])], normalize(block[4])) for i, page in enumerate(document)
                     for block in page.get_text("blocks", sort=True) if block[6] == 0]
    require(date and (date < "20221130" if publisher == "copyright" else date < "2022-11-30"), "Source is newer than the publication cutoff")
    if publisher == "copyright":
        require(not reviewed or reviewed < "20221130", "PDF revision is newer than the publication cutoff")
    return title.strip(), date, reviewed, [
        (anchor, value) for anchor, value in units
        if value and value[0].isupper() and value[-1] in ".!?" and not any(mark in value for mark in ('“', '”', '"', '•', 'http', '@'))
        and 20 <= len(WORD.findall(value)) <= 180
    ]


def options(units, group):
    targets = (1, 2) if int(sha(group)[-1], 16) % 2 else (2, 1)
    sizes = [len(WORD.findall(value)) for _, value in units]
    for i in range(len(units) - 3):
        parts = units[i:i + 4]
        for focus in targets:
            target = sizes[i + focus]
            if not 27 <= target <= 115:
                continue
            for j, size in enumerate(sizes):
                if not i - 1 <= j <= i + 4 and .72 <= size / target <= 1.28:
                    yield parts, focus, units[j]


def prep(run, historical_paths):
    from scripts.prepare_writing_option_order_transfer import benchmark_texts
    from scripts.text_elife_source_intake_v272 import overlap
    from scripts.text_short_insert_intake_v203 import fingerprints

    roster = read(run / "source-roster.json")
    inventories, rejected, candidates = {}, {}, {}
    for publisher, urls in roster.items():
        for group, url in urls.items():
            path = run / "sources" / (group + (".pdf" if publisher == "copyright" else ".html"))
            if not path.is_file():
                rejected[group] = "Source unavailable during intake"
                continue
            try:
                title, date, revised, units = source_blocks(path, publisher)
            except ValueError as exc:
                rejected[group] = str(exc)
                continue
            inventories[group] = (publisher, url, title, date, revised, path)
            candidates[group] = list(options(units, group))

    signatures = {
        (group, n): fingerprints("\n\n".join(text for _, text in parts)) | fingerprints(donor[1])
        for group, choices in candidates.items() for n, (parts, _, donor) in enumerate(choices)
    }
    history, external, _ = overlap([(name, sha(path), None, benchmark_texts(path)) for name, path in historical_paths], signatures)
    admitted, used, provenance = {}, set(), {}
    for group, (publisher, url, title, date, revised, path) in inventories.items():
        for n, (parts, target, donor) in enumerate(candidates[group]):
            key = group, n
            if key in external or signatures[key] & used:
                continue
            used |= signatures[key]
            admitted[group] = {"publisher": publisher, "paragraphs": [text for _, text in parts], "target": target, "donor": donor[1]}
            provenance[group] = {
                "publisher": publisher, "url": url, "title": title, "publication_or_creation": date,
                "publisher_review_or_pdf_revision": revised, "sha256": sha(path), "target": target,
                "source_paragraphs": [{"location": anchor, "sha256": sha(text)} for anchor, text in [*parts, donor]],
            }
            break
        else:
            rejected[group] = "No eligible non-overlapping four-block excerpt and length-matched donor"
    counts = Counter(row["publisher"] for row in admitted.values())
    require(all(counts[publisher] >= 4 for publisher in PUBLISHERS), "Too few eligible source groups")
    write(run / "sources.json", {"historical": history, "admitted": provenance, "excluded": rejected, "counts": counts})
    write(run / "originals.json", admitted)
    print(dict(counts), len(history), sum(row["text_entries"] for row in history), flush=True)


def model_assets(run):
    out = {}
    for family in FAMILIES:
        original = read(run / "model-metadata" / f"{family}-original.json")
        quantized = read(run / "model-metadata" / f"{family}-quantized.json")
        require(all(row["cardData"]["license"] == "apache-2.0" for row in (original, quantized)), "Unexpected weight license")
        files = {path.name: sha(path) for path in (run / "models" / family).glob("*") if path.is_file()}
        expected = {row["rfilename"]: row["lfs"]["sha256"] for row in quantized["siblings"] if row["rfilename"].endswith(".safetensors")}
        require(expected and all(files.get(name) == digest for name, digest in expected.items()), "Generator weight digest differs")
        require({"config.json", "tokenizer_config.json", "tokenizer.json"} <= files.keys(), "Generator configuration incomplete")
        config = read(run / "models" / family / "config.json")
        overrides = {name: bool(config[name]) for name in ("use_cache", "attention_bias", "tie_word_embeddings")
                     if type(config.get(name)) is int and config[name] in (0, 1)}
        out[family] = {"license": "Apache-2.0", "upstream": original["id"], "upstream_revision": original["sha"],
                       "quantized": quantized["id"], "revision": quantized["sha"], "assets": files,
                       "loader_aliases": {name: "model" + name.removeprefix("weights") for name in expected if name.startswith("weights")},
                       "typed_config_overrides": overrides}
    return out


def protocol(run, development):
    return {"schema": SCHEMA, "method": METHOD, "word_classes": CLASSES, "orders": ORDERS, "prompt": PROMPT,
            "temperature": .65, "top_p": .92, "seed": SEED, "maximum_generated_tokens": 220,
            "generation_runtime": GENERATION_RUNTIME,
            "script_sha256": sha(Path(__file__)), "source_sha256": sha(run / "sources.json"),
            "originals_sha256": sha(run / "originals.json"), "development_sha256": sha(development),
            "weights": model_assets(run)}


def generate(run, family):
    import mlx.core as mx
    from mlx_lm import load, stream_generate
    from mlx_lm.sample_utils import make_sampler

    require(os.environ.get("HF_HUB_OFFLINE") == os.environ.get("TRANSFORMERS_OFFLINE") == "1", "Generator must run offline")
    frozen = read(run / "protocol.json")
    require(frozen["script_sha256"] == sha(Path(__file__)) and frozen["originals_sha256"] == sha(run / "originals.json")
            and frozen["weights"] == model_assets(run), "Frozen generation inputs differ")
    require({name: version(name) for name in GENERATION_RUNTIME} == frozen["generation_runtime"], "Generator runtime differs")
    mx.set_default_device(mx.gpu)
    mx.random.seed(frozen["seed"][family])
    source = run / "models" / family
    aliases = frozen["weights"][family]["loader_aliases"]
    overrides = frozen["weights"][family]["typed_config_overrides"]
    with TemporaryDirectory(prefix="writing-grammar-model-", dir=run) as temporary:
        staging = Path(temporary)
        for name in frozen["weights"][family]["assets"]:
            if name == "config.json" and overrides:
                write(staging / name, {**read(source / name), **overrides})
            else:
                (staging / aliases.get(name, name)).symlink_to((source / name).resolve())
        model, tokenizer = load(str(staging), tokenizer_config={"trust_remote_code": False})
    sampler = make_sampler(temp=frozen["temperature"], top_p=frozen["top_p"])
    out = {}
    for group, row in read(run / "originals.json").items():
        text = row["paragraphs"][row["target"]]
        request = PROMPT.format(count=len(WORD.findall(text)), body=text)
        chat = tokenizer.apply_chat_template([{"role": "user", "content": request}], tokenize=False, add_generation_prompt=True)
        response = list(stream_generate(model, tokenizer, chat, sampler=sampler, max_tokens=frozen["maximum_generated_tokens"]))
        out[group] = {"text": "".join(part.text for part in response), "finish": response[-1].finish_reason, "input_sha256": sha(text)}
        print(f"{family} {len(out)}", flush=True)
    write(run / f"generated-{family}.json", {"protocol_sha256": sha(run / "protocol.json"), "groups": out})


def make_case(group, publisher, kind, paragraphs, target):
    text = "\n\n".join(paragraphs)
    start, bounds = 0, []
    for paragraph in paragraphs:
        end = start + len(paragraph)
        bounds.append([start, end])
        start = end + 2
    return {"group": group, "publisher": publisher, "kind": kind, "text": text, "text_sha256": sha(text),
            "bounds": bounds, "target": target, "span": bounds[target]}


def eligible(value, original):
    from scripts.writing_blind_rewrite_probe import distance as edit_distance

    value = value.strip()
    ratio = len(WORD.findall(value)) / len(WORD.findall(original))
    if (not .65 <= ratio <= 1.35 or "\n" in value or value[:1] in ('"', '“', '*', '#')
            or value[-1:] not in ".!?" or edit_distance(original, value) < .25):
        return None
    return normalize(value)


def fixtures(run):
    generated = {family: read(run / f"generated-{family}.json") for family in FAMILIES}
    require(all(value["protocol_sha256"] == sha(run / "protocol.json") for value in generated.values()), "Generation did not match the protocol")
    result, admission = [], {}
    for group, row in read(run / "originals.json").items():
        target, parts = row["target"], row["paragraphs"]
        original = parts[target]
        variants = {"original": original, "human_splice": row["donor"]}
        for family, packet in generated.items():
            candidate = packet["groups"][group]
            require(candidate["input_sha256"] == sha(original), "Generator source differed")
            value = eligible(candidate["text"], original) if candidate["finish"] == "stop" else None
            admission[f"{group}/{family}"] = "admitted" if value is not None else "declined"
            if value is not None:
                variants[family] = value
        for kind, replacement in variants.items():
            result.append(make_case(group, row["publisher"], kind, [*parts[:target], replacement, *parts[target + 1:]], target))
    write(run / "cases.json", {"admission": admission, "cases": result})
    print(Counter(row["kind"] for row in result), flush=True)


def fit(path):
    maxima = {order: defaultdict(float) for order in ORDERS}
    for row in read(path):
        text, bounds, group = row["text"], row["bounds"], row["group"]
        require(row["sha256"] == sha(text), "Calibration text digest differs")
        for order in ORDERS:
            scores = seam_scores(text, bounds, order)
            if scores:
                maxima[order][group] = max(maxima[order].get(group, -1), max(scores.values()))
    points = {}
    for order, values in maxima.items():
        ranked = sorted(values.values())
        require(len(ranked) >= 12, "Too few measurable development groups")
        points[str(order)] = {"group_maxima": dict(values), "cutoff": ranked[math.ceil(.9 * len(ranked)) - 1]}
    return points


def baseline(cases, snapshot):
    import torch

    from app.detectors.text.extract import from_string
    from app.detectors.text.segment import segment
    from scripts.writing_focus_v3 import BASELINE_ASSETS
    from scripts.writing_paired_email_revision import encoded, native_contexts, native_scores

    require(all(sha(snapshot / name) == digest for name, digest in BASELINE_ASSETS.items()), "Baseline model differs")
    torch.set_num_threads(1)
    mapping, texts = {}, {}
    for case in cases:
        doc = from_string(case["text"])
        require(doc.text == case["text"] and [list(pair) for pair in doc.paragraphs] == case["bounds"], "Fixture source mapping changed")
        label = f"{case['group']}/{case['kind']}"
        mapping[label] = []
        for context in native_contexts(segment(doc)):
            entry, text = encoded(context)
            mapping[label].append(entry)
            texts[entry["sha256"]] = text
    scored, forwards = native_scores(snapshot, texts, second=False)
    return {"contexts": mapping, "native": scored, "forwards": forwards}


def overlap(text, truth, region):
    a, b = truth
    truth_chars = {i for i in range(a, b) if not text[i].isspace()}
    selected = {i for start, end in region for i in range(start, end) if not text[i].isspace()}
    return len(truth_chars & selected) / len(truth_chars | selected)


def summarize(run):
    cases = read(run / "cases.json")["cases"]
    responses, points = read(run / "baseline.json"), read(run / "points.json")
    results, useful, humans, details = defaultdict(Counter), defaultdict(set), defaultdict(set), []
    for case in cases:
        text, bounds, target, group, kind, publisher = (case[key] for key in ("text", "bounds", "span", "group", "kind", "publisher"))
        key = f"{group}/{kind}"
        current = [row["spans"] for row in responses["contexts"][key] if responses["native"][row["sha256"]]["score"] >= .5][:6]
        scores = {str(order): seam_scores(text, bounds, order) for order in ORDERS}
        selected = {order: select(values, points[order]["cutoff"]) for order, values in scores.items()}
        arms = {"current": current, "unigram": [], "sequence": []}
        for order, arm in (("1", "unigram"), ("3", "sequence")):
            if selected[order] is not None:
                arms[arm] = [[bounds[selected[order]]]]
        arms["union"] = current + arms["sequence"]
        positive = {arm: any(overlap(text, target, region) >= .30 for region in choices) for arm, choices in arms.items()}
        details.append({"case": key, "scores": scores, "selected": selected, "regions": arms, "useful": positive})
        for scope in ("all", publisher):
            tally = results[scope, kind]
            tally["files"] += 1
            for order, arm in (("1", "unigram"), ("3", "sequence")):
                tally[f"{arm}_unassessed"] += not scores[order]
            for arm, regions in arms.items():
                tally[f"{arm}_flags"] += bool(regions)
                tally[f"{arm}_useful"] += positive[arm] and kind in FAMILIES
                if kind in FAMILIES and positive[arm]:
                    for axis in {scope, kind}:
                        useful[axis, arm].add(group)
                elif kind not in FAMILIES and regions:
                    humans[scope, arm].add(group)
    write(run / "private-readout.json", details)
    gained = {scope: len(useful[scope, "union"] - useful[scope, "current"]) for scope in ("all", *PUBLISHERS, *FAMILIES)}
    represented = {case["kind"] for case in cases} & set(FAMILIES)
    additional_human = len(humans["all", "union"] - humans["all", "current"])
    write(run / "result.json", {
        "schema": SCHEMA, "method": METHOD, "files": len(cases), "native_forwards": responses["forwards"],
        "native_contexts": len(responses["native"]), "counts": {f"{scope}/{kind}": tally for (scope, kind), tally in results.items()},
        "additional_useful_groups": gained, "additional_human_groups": additional_human,
        "assessed_generator_families": sorted(represented),
        "advance": len(represented) >= 2 and all(gained[scope] > 0 for scope in set(PUBLISHERS) | represented)
        and additional_human <= 1,
        "human_groups": {arm: len(humans["all", arm]) for arm in ("current", "unigram", "sequence", "union")},
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "freeze", "generate", "admit", "baseline", "summarize"))
    parser.add_argument("directory", type=Path)
    parser.add_argument("--development", type=Path)
    parser.add_argument("--history", nargs=2, action="append", metavar=("NAME", "JSONL"))
    parser.add_argument("--family", choices=FAMILIES)
    parser.add_argument("--current-snapshot", type=Path)
    args = parser.parse_args()
    run = args.directory
    os.umask(0o077)
    if args.stage == "prepare":
        prep(run, [(name, Path(path)) for name, path in args.history])
    elif args.stage == "freeze":
        write(run / "points.json", fit(args.development))
        write(run / "protocol.json", {**protocol(run, args.development), "points_sha256": sha(run / "points.json")})
    elif args.stage == "generate":
        generate(run, args.family)
    elif args.stage == "admit":
        fixtures(run)
    elif args.stage == "baseline":
        write(run / "baseline.json", baseline(read(run / "cases.json")["cases"], args.current_snapshot))
    else:
        summarize(run)


if __name__ == "__main__":
    main()
