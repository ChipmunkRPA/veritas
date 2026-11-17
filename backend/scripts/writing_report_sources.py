"""Prepare official economic reports and matched, locally generated revisions."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from scripts import text_elife_rewrites_v273 as generation
from scripts import writing_multi_passage_panel as panel

SCHEMA = "chipmunk.writing-report-sources/1"
BOARD = "https://www.federalreserve.gov/monetarypolicy/beigebook/"
DISTRICTS = ("boston", "chicago", "san_francisco")
GENERATION_TOKENS = 160
OPAI_REVISION = "072c03c7051409346aaba05168d183cf6d863ee9"
OPAI_SHARDS = {
    "gemini-2.5-flash": "e4a52b8cd3a56ac9d748d101f9d1ea75e6ae13b6eb6c38dc95e41f64488c45b5",
    "gpt-5.4-nano": "912d90f2f22b2ea97bdf69930f6891584112381f5b811aedcac390bbdf1da064",
}
OPAI_EXTRA = "c0b0189df99b6be49f4917dd3db32e268deddd366a641c337f3a0e07789f2d9e"
GAO_SHA256 = "671287085feaa971c501be2e2db7d50a36ca2a9adba0642c9540f4add35b341e"
GAO_REVIEW_PER_GENERATOR = 16


def exposed_reports(folder):
    reports, digests, inputs = set(), set(), {}
    for path in sorted(folder.iterdir()):
        if (not path.name.startswith(("writing-", "text-", "document-gao-"))
                or path.suffix not in {".gz", ".json"} or path.name.startswith("writing-report-confirmation-")):
            continue
        data = path.read_bytes()
        inputs[path.name] = panel.digest(data)
        content = gzip.decompress(data) if path.suffix == ".gz" else data
        for identifier in re.findall(rb"(?:GAO_|gao\.gov/products/)([A-Za-z]+-[0-9A-Za-z-]+)", content, re.I):
            reports.add(identifier.decode().upper())
        digests.update(value.decode() for value in re.findall(rb'\b[0-9a-f]{64}\b', content))
    return reports, digests, inputs


def test_pairs(path, family, split, digest):
    with path.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != digest:
            raise ValueError("OpAI-Bench file differs from the published shard")
    csv.field_size_limit(1024 * 1024)
    sources = defaultdict(dict)
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            version = row["version"]
            if version not in {"v0", "v1"}:
                continue
            if (row["split"], row["domain"], row["generator"], row["subset"], row["edit_operation"]) != (
                split, "reports", family, "main", {"v0": "none", "v1": "polish"}[version]
            ) or version in sources[row["id"]]:
                raise ValueError("Unexpected OpAI-Bench delivery")
            sources[row["id"]][version] = row
    return sources


def gao(cache, pack, output, *, additional=None):
    from scripts.writing_revision_trajectory_transfer import publisher_case

    if panel.digest(pack.read_bytes()) != GAO_SHA256:
        raise ValueError("Expected the independently pinned GovReport highlights")
    upstream = panel.read(pack)
    previous, digests, files = exposed_reports(Path(__file__).resolve().parents[2] / "docs/evaluation-panels")
    works, additions, rejected = [], Counter(), Counter()
    shards = [(family, "test", cache / f"reports_{family}.csv", digest) for family, digest in OPAI_SHARDS.items()]
    if additional is not None:
        shards.append(("gpt-5.4-nano", "train", additional, OPAI_EXTRA))
    for family, split, path, digest in shards:
        entries = test_pairs(path, family, split, digest)
        for key in sorted(entries, key=lambda item: panel.digest((SCHEMA + item).encode())):
            if additions[family] >= GAO_REVIEW_PER_GENERATOR:
                break
            versions = entries[key]
            if set(versions) != {"v0", "v1"}:
                rejected["incomplete_pair"] += 1
                continue
            original, altered = versions["v0"], versions["v1"]
            reference_id = key.removeprefix("GAO_").upper()
            document_hashes = {panel.digest(value.encode()) for value in
                               (key, reference_id, original["text"], altered["text"], original["document_hash_id"])}
            if reference_id in previous or document_hashes & digests:
                rejected["earlier_source_or_text"] += 1
                continue
            authority = upstream["reports"].get(reference_id)
            if (not authority or authority["published_date"] > "2020-12-31"
                    or panel.compact(authority["text"]) != panel.compact(original["text"])):
                rejected["unbound_original"] += 1
                continue
            try:
                seed, ordinary_truth = publisher_case(original)
                edit, truth = publisher_case(altered)
            except ValueError:
                rejected["invalid_publisher_coordinates"] += 1
                continue
            starts, revised = (json.loads(row["sentences"]) for row in (original, altered))
            labels = json.loads(altered["sentence_labels"])
            if ordinary_truth or not 1 <= len(truth) <= 2 or len(starts) != len(revised) or not all(
                (before != after) == bool(mark) for before, after, mark in zip(starts, revised, labels, strict=True)
            ):
                rejected["unsupported_edit_geometry"] += 1
                continue
            untouched = variants(seed, (), version="ordinary")
            synthetic = variants(edit, truth, version="mixed", generator=family)
            if untouched is None or synthetic is None:
                rejected["unsupported_extraction"] += 1
                continue
            works.append({
                "document": key, "group_sha256": panel.digest(key.encode()), "split": "test", "genre": "gao_report",
                "source_reference": f"https://www.gao.gov/products/{reference_id.lower()}",
                "source_record_sha256": authority["source_record_sha256"], "publisher_split": split,
                "published_date": authority["published_date"],
                "publisher_ids": {version: row["record_id"] for version, row in versions.items()},
                "variants": untouched + synthetic,
            })
            previous.add(reference_id)
            additions[family] += 1
    panel.save(output, {"schema": SCHEMA, "opai_revision": OPAI_REVISION,
                        "opai_shards": {f"{split}/{family}": digest for family, split, _, digest in shards},
                        "govreport_revision": upstream["revision"], "govreport_pack_sha256": GAO_SHA256,
                        "prior_public_receipts": files, "excluded": dict(rejected),
                        "admitted": dict(additions), "works": works})
    print(json.dumps({"admitted": additions, "excluded": rejected}, sort_keys=True), flush=True)


def official(folder, manifest):
    from lxml import html

    paths = sorted(name for name in manifest if re.fullmatch(r"beigebook201[456]\d{2}(?:\d{2})?\.htm", name))
    if len(paths) != 24 or any(sum(name.startswith(f"beigebook{year}") for name in paths) != 8
                               for year in range(2014, 2017)):
        raise ValueError("Use the 24 dated Board releases from 2014–2016")
    originals = []
    for index, name in enumerate(paths):
        asset = manifest[name]
        raw = (folder / name).read_bytes()
        if asset["url"] != BOARD + name or panel.digest(raw) != asset["sha256"]:
            raise ValueError("Board report differs from the recorded publisher file")
        district = DISTRICTS[index % len(DISTRICTS)]
        tree = html.fromstring(raw)
        section, = tree.xpath('//*[@id = $selected]', selected=f"div_{district}")
        paragraphs = [" ".join(node.text_content().split()) for node in section.xpath("./p")]
        text = "\n\n".join(item for item in paragraphs if len(item.split()) >= 12)
        ordinary = panel.native(text, ())
        if ordinary is None or ordinary["text"] != text:
            raise ValueError(f"Unmapped report text in {name} {district}")
        sentences = ordinary["sentences"]
        options = []
        for first in range(2, len(sentences) - 3):
            a, b = sentences[first:first + 2]
            if a[4] == b[4] and 33 <= a[5] + b[5] <= 66:
                options.append([a[0], b[1]])
        if not options:
            raise ValueError(f"No eligible two-sentence target in {name}")
        chosen = int(panel.digest((SCHEMA + name).encode())[:8], 16) % len(options)
        originals.append({"document": f"{name}#{district}", "source_reference": asset["url"],
                          "source_sha256": asset["sha256"], "edition": name, "district": district,
                          "group": panel.digest(name.encode()), "text": text,
                          "edit_target_codepoints": options[chosen]})
    return originals


def record(folder, output):
    assets = panel.read(folder / "assets.json")
    sources = official(folder, assets)
    panel.save(output, {"schema": SCHEMA, "official_assets": assets, "sources": sources})


def prompt(tokenizer, source, start, stop):
    words = len(source[start:stop].split())
    content = (
        "Rewrite the two sentences under Target in two complete English sentences for a formal economic report. "
        "Keep the meaning and any numbers. Use different wording. "
        f"Write approximately {words} words. Output only the two new sentences as one paragraph.\n\n"
        f"Previous context: {source[max(0, start - 240):start].strip()}\n"
        f"Target: {source[start:stop]}\n"
        f"Following context: {source[stop:stop + 240].strip()}"
    )
    return tokenizer.apply_chat_template([{"role": "user", "content": content}], tokenize=False,
                                         add_generation_prompt=True)


def generate(plan, family, output):
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm import batch_generate, load
    from mlx_lm.sample_utils import make_sampler

    _, revision, weights, path = generation.GENERATORS[family]
    with (path / "model.safetensors").open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != weights:
            raise ValueError("Rewrite generator weight does not match the recorded edition")
    model, tokenizer = load(str(path), tokenizer_config={"trust_remote_code": False, "local_files_only": True})
    nn.quantize(model, group_size=64, bits=4, mode="affine")
    mx.eval(model.parameters())
    sources = panel.read(plan)["sources"]
    rows = []
    for offset in range(0, len(sources), 4):
        batch = sources[offset:offset + 4]
        tokens = [tokenizer.encode(prompt(tokenizer, row["text"], *row["edit_target_codepoints"]),
                                   add_special_tokens=False) for row in batch]
        responses = batch_generate(model, tokenizer, tokens, max_tokens=GENERATION_TOKENS, sampler=make_sampler(temp=0.))
        for row, response in zip(batch, responses.texts, strict=True):
            rows.append({"document": row["document"], "response": response})
        print(f"Generated {len(rows)}/{len(sources)} two-sentence revisions", flush=True)
    panel.save(output, {"source_plan_sha256": panel.digest(plan.read_bytes()), "family": family,
                        "revision": revision, "weights_sha256": weights, "responses": rows})


def label(raw, target, response):
    from app.detectors.text.extract import from_string
    from app.detectors.text.segment import segment

    start, stop = target
    text = " ".join(response.split())
    if (not text or generation.WRAPPER.match(text) or generation.CONTACT.search(text)
            or re.search(r"```|\n\s*\n", response.strip())
            or not text.endswith((".", "?", "!")) or len(segment(from_string(text))) != 2):
        return None
    if not .60 <= len(text.split()) / len(raw[start:stop].split()) <= 1.55:
        return None
    if generation.copy_fraction(text, raw) > .35 or panel.compact(text) == panel.compact(raw[start:stop]):
        return None
    mixed = raw[:start] + text + raw[stop:]
    return mixed, [(start, start + len(text))]


def variants(raw, truth, *, version, generator=None):
    outcome = []
    for layout, text in (("published", raw), ("neutral", re.sub(r"\s", " ", raw))):
        native = panel.native(text, truth)
        if native is None:
            return None
        outcome.append(native | {"version": version, "layout": layout, "generator": generator})
    return outcome


def assemble(plan, generations, output):
    source = panel.read(plan)
    wanted = [row["document"] for row in source["sources"]]
    expected = panel.digest(plan.read_bytes())
    transcripts = [panel.read(path) for path in generations]
    if {row["family"] for row in transcripts} != set(generation.GENERATORS) or len(transcripts) != 2:
        raise ValueError("Use exactly one response packet from each local generator")
    for packet in transcripts:
        _, rev, weights, _ = generation.GENERATORS[packet["family"]]
        if (packet["source_plan_sha256"] != expected or packet["revision"] != rev or packet["weights_sha256"] != weights
                or [row["document"] for row in packet["responses"]] != wanted):
            raise ValueError("Generation differs from the selected sources and weights")
    works, excluded, admitted = [], Counter(), Counter()
    for index, row in enumerate(source["sources"]):
        ordinary = variants(row["text"], (), version="ordinary")
        if ordinary is None:
            raise ValueError("Original report became unassessable")
        all_versions = list(ordinary)
        prefix = panel._prefix(row["text"])
        start, stop = row["edit_target_codepoints"]
        for variant in ordinary:
            variant["matched_targets"] = [[prefix[start], prefix[stop]]]
        for packet in transcripts:
            family = packet["family"]
            replacement = label(row["text"], row["edit_target_codepoints"], packet["responses"][index]["response"])
            derived = variants(*replacement, version="mixed", generator=family) if replacement is not None else None
            if derived is not None:
                all_versions.extend(derived)
                admitted[family] += 1
            else:
                excluded[family] += 1
        works.append({key: value for key, value in row.items() if key != "text"} | {
            "group_sha256": row["group"], "split": "test", "genre": "economic_report", "variants": all_versions,
        })
    panel.save(output, {"schema": SCHEMA, "official_assets": source["official_assets"],
                        "source_plan_sha256": expected, "admitted": dict(admitted), "excluded": dict(excluded),
                        "generator_revisions": {p["family"]: p["revision"] for p in transcripts}, "works": works})
    print(json.dumps({"ordinary": len(works), "admitted": admitted, "excluded": excluded}, sort_keys=True), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="stage", required=True)
    build = commands.add_parser("record")
    build.add_argument("folder", type=Path)
    build.add_argument("output", type=Path)
    expand = commands.add_parser("generate")
    expand.add_argument("plan", type=Path)
    expand.add_argument("family", choices=tuple(generation.GENERATORS))
    expand.add_argument("output", type=Path)
    combine = commands.add_parser("assemble")
    combine.add_argument("plan", type=Path)
    combine.add_argument("output", type=Path)
    combine.add_argument("generations", nargs=2, type=Path)
    held = commands.add_parser("gao")
    for arg in ("cache", "pack", "output"):
        held.add_argument(arg, type=Path)
    held.add_argument("--additional", type=Path, help="Pinned OpAI-Bench report shard for an additional generator")
    args = parser.parse_args()
    if args.stage == "record":
        record(args.folder, args.output)
    elif args.stage == "generate":
        generate(args.plan, args.family, args.output)
    elif args.stage == "gao":
        gao(args.cache, args.pack, args.output, additional=args.additional)
    else:
        assemble(args.plan, args.generations, args.output)
