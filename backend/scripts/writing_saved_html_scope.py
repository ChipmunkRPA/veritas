# SPDX-License-Identifier: Apache-2.0
"""Recount standalone HTML article scopes and known paragraph locations."""

from __future__ import annotations

import argparse
import math
import re
from collections import Counter
from html import escape, unescape
from pathlib import Path
from urllib.parse import urlparse

from scripts.writing_transition_deviation import file_hash, read, save, text_hash

SCHEMA = "chipmunk.writing-saved-html-scope/1"
# The first frozen run used the alternate article extractor before report-only alignment was implemented.
HELD_IMPLEMENTATION = "8b9902952cc5f1034d88e5d61fc1dff9216d5c71e37e456eab30a69130a4e2d9"
PUBLISHERS = {"www.ncei.noaa.gov": "noaa", "science.nasa.gov": "nasa", "www.nist.gov": "nist"}
PARAGRAPH = re.compile(r"<p(?:\s[^<>]*)?>([^<>]+)</p\s*>", re.I)
PROMPT = ("Rewrite the following article paragraph in natural English while preserving its facts, quoted values and proper names. "
          "Change its sentence structure and wording. Use the same approximate length and output only the revised paragraph.")
MATCH = {"iou": .30, "recall": .50}
SETTINGS = {"detection": "pinned 3-sentence classifier, raw 0.5 boundary, first 6 eligible foci",
            "originals": "previously unscored official article HTML obtained from pinned URLs and published before 2021",
            "focus": "one literal paragraph without links or other nested markup; keep unchanged raw HTML outside its text",
            "match": MATCH, "prompt_sha256": text_hash(PROMPT)}


def normalized(text):
    return " ".join(text.split())


def _article_projection(raw):
    from app.detectors.text import html_article
    from app.detectors.text.extract import _join_mapped
    from app.evidence.document_scope import valid_html_article_scope

    article = html_article.parts(raw)
    if article is None or not valid_html_article_scope(article[1]):
        return None, None
    return _join_mapped(article[0]), article[1]


def _frozen_implementation(plan, parser):
    return (plan["implementation_sha256"] in {HELD_IMPLEMENTATION, file_hash(Path(__file__))}
            and plan["parser_sha256"] == file_hash(Path(parser.__file__)))


def _body_targets(raw, scope, doc, old, seen):
    from app.detectors.text.extract import from_string
    from app.detectors.text.segment import segment
    from scripts.writing_economic_paraphrase_holdout import fragments

    start, end = scope["scope"]
    for match in PARAGRAPH.finditer(raw, start, end):
        value = normalized(unescape(match[1]))
        if (not 30 <= len(value.split()) <= 90 or not 2 <= len(segment(from_string(value))) <= 4
                or re.search(r"(?i)\b(?:image|credits?|license|copyright|references?|via|according to|contact)\b|[“”©]|https?:", value)
                or seen & fragments(value) or doc.text.count(value) != 1 or old.text.count(value) != 1):
            continue
        extracted = doc.text.index(value)
        if [extracted, extracted + len(value)] not in [list(pair) for pair in doc.paragraphs]:
            continue
        yield match.span(1), value


def freeze(folder, urls, output, prior):
    from lxml import html

    from app.detectors.text import html_article
    from app.detectors.text.extract import extract
    from scripts.writing_economic_paraphrase_holdout import find_texts, fragments

    seen = {key for path in prior for value in find_texts(read(path)) for key in fragments(value)}
    sources, rejected = [], Counter()
    for url in urls.read_text().splitlines():
        publisher = PUBLISHERS[urlparse(url).hostname]
        source = folder / (url.rstrip("/").rsplit("/", 1)[1] + ".html")
        if not source.exists():
            rejected["not_downloaded"] += 1
            continue
        raw = source.read_bytes().decode("utf-8")
        tree = html.fromstring(raw)
        years = tree.xpath('//meta[@property="article:published_time" or @name="dcterms.date"]/@content')
        if not years:
            years = tree.xpath('//meta[contains(@property,"published") or contains(@name,"date")]/@content')
        years = [date for date in years if re.match(r"^20(?:1[0-9]|20)-\d\d-\d\dT", date)]
        if not years:
            rejected["no_historical_date"] += 1
            continue
        candidate, scope = _article_projection(raw)
        if scope is None:
            rejected[f"ambiguous_{publisher}"] += 1
            continue
        before = extract(source, ".html", include_html_article=False)
        candidates = sorted(_body_targets(raw, scope, candidate, before, seen), key=lambda row: text_hash(row[1]))
        if not candidates:
            rejected["no_licensed_exact_paragraph"] += 1
            continue
        target, value = candidates[0]
        sources.append({"id": source.stem, "url": url, "publisher": publisher, "published": years[0][:10],
                        "sha256": file_hash(source), "scope": scope, "focus": target, "focus_text": value})
        seen.update(fragments(value))
    save(output, {"schema": SCHEMA, "settings": SETTINGS, "implementation_sha256": file_hash(Path(__file__)),
                  "parser_sha256": file_hash(Path(html_article.__file__)), "sources": sources,
                  "rejected": dict(rejected), "prior_sha256": [file_hash(path) for path in prior]})
    by_origin = Counter(source["publisher"] for source in sources)
    print("Admitted", dict(by_origin), "Excluded", dict(rejected), flush=True)
    if len(sources) < 6 or len(by_origin) < 2:
        raise ValueError("Need distinct sources from two government publishers for this comparison")


def generate(plan, family, output):
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm import batch_generate, load
    from mlx_lm.sample_utils import make_sampler

    from scripts.text_elife_rewrites_v273 import GENERATORS

    _, revision, sha, folder = GENERATORS[family]
    if file_hash(folder / "model.safetensors") != sha:
        raise ValueError("The generator must match its pinned public checkpoint")
    model, tokenizer = load(str(folder), tokenizer_config={"local_files_only": True, "trust_remote_code": False})
    nn.quantize(model, group_size=64, bits=4, mode="affine")
    mx.set_cache_limit(96 * 1024 * 1024)
    mx.eval(model.parameters())
    sources = read(plan)["sources"]
    prompts = [tokenizer.encode(tokenizer.apply_chat_template([
        {"role": "user", "content": PROMPT + "\n\n" + source["focus_text"]}], tokenize=False, add_generation_prompt=True),
        add_special_tokens=False) for source in sources]
    replies = []
    for index in range(0, len(prompts), 3):
        batch = prompts[index:index + 3]
        replies.extend(batch_generate(model, tokenizer, batch, max_tokens=170, sampler=make_sampler(temp=0.0)).texts)
    save(output, {"schema": SCHEMA, "source_sha256": file_hash(plan), "family": family, "revision": revision,
                  "model_sha256": sha, "replies": {source["id"]: reply for source, reply in zip(sources, replies, strict=True)}})


def accepted(reply, source):
    from scripts.text_elife_rewrites_v273 import CONTACT, WRAPPER
    from scripts.writing_blind_rewrite_probe import distance

    value = normalized(reply)
    if (not value or value[-1] not in ".?!" or WRAPPER.match(value) or CONTACT.search(value) or "```" in value
            or re.search(r"\n\s*\n", reply.strip()) or not .65 <= len(value.split()) / len(source.split()) <= 1.45
            or not .12 <= distance(source, value) <= .80):
        return None
    return value


def qualify(plan, folder, output, transcripts):
    from app.detectors.text import html_article
    from app.detectors.text.extract import extract
    from app.evidence.document_native_locations import clip_source_ranges
    from scripts.text_elife_rewrites_v273 import GENERATORS

    frozen = read(plan)
    if not _frozen_implementation(frozen, html_article):
        raise ValueError("The recorded implementation differs from the prospective source commitment")
    by_family, input_hashes = {}, {}
    for path in transcripts:
        row = read(path)
        family = row["family"]
        _, revision, sha, _ = GENERATORS[family]
        if (row["source_sha256"] != file_hash(plan) or row["schema"] != SCHEMA or row["revision"] != revision
                or row["model_sha256"] != sha or family in by_family or set(row["replies"]) != {original["id"] for original in frozen["sources"]}):
            raise ValueError("A generator receipt differs from the fixed source list")
        by_family[family] = row["replies"]
        input_hashes[family] = file_hash(path)
    if by_family.keys() != GENERATORS.keys():
        raise ValueError("Keep both source-bound generator receipts")
    staging = output.parent / "projected-source.html"
    works, rejected, exact = [], Counter(), Counter()
    for item in frozen["sources"]:
        path = folder / f"{item['id']}.html"
        if file_hash(path) != item["sha256"]:
            raise ValueError("The original publisher file changed")
        observed = []
        raw = path.read_bytes().decode("utf-8")
        start, end = item["focus"]
        ordinary = item["focus_text"]
        for role, proposal in (("ordinary", ordinary), *((family, accepted(values[item["id"]], ordinary))
                                  for family, values in sorted(by_family.items()))):
            if proposal is None:
                rejected[role] += 1
                continue
            rendered = raw[start:end] if role == "ordinary" else escape(proposal, quote=False)
            edited = raw[:start] + rendered + raw[end:]
            staging.write_text(edited)
            legacy = extract(staging, ".html", include_html_article=False)
            current = extract(staging, ".html")
            article, _ = _article_projection(edited)
            if legacy.text != current.text or legacy.paragraphs != current.paragraphs:
                raise ValueError("Exact location attachment changed the writing input")
            variants = {}
            for name, parsed in (("generic", legacy), ("article", article)):
                if parsed.text.count(proposal) != 1:
                    raise ValueError("The rewritten paragraph does not map uniquely to both scopes")
                offset = parsed.text.index(proposal)
                stop = offset + len(proposal)
                ranges = []
                if name == "article":
                    block = next(row for row in parsed.source_map if row["start"] <= offset < stop <= row["end"])
                    ranges = clip_source_ranges(block["source_ranges"], offset, stop)
                    if not ranges or not all(start <= a < b <= start + len(rendered) for a, b in ranges):
                        raise ValueError("Article range does not correspond to the unaltered physical paragraph position")
                variants[name] = {"text": parsed.text, "paragraphs": parsed.paragraphs, "focus": [offset, stop], "source_ranges": ranges}
            low, high = variants["generic"]["focus"]
            current_block = next((row for row in current.source_map if row["start"] <= low < high <= row["end"]), {})
            restored = clip_source_ranges(current_block.get("source_ranges", []), low, high)
            if restored == variants["article"]["source_ranges"]:
                exact[role] += 1
            observed.append({"role": role, "sha256": text_hash(edited), "views": variants})
        works.append({key: value for key, value in item.items() if key != "focus_text"} | {"variants": observed})
    staging.unlink()
    save(output, {key: value for key, value in frozen.items() if key != "sources"} | {"source_sha256": file_hash(plan),
                 "generation_sha256": input_hashes, "cases": works, "generation_exclusions": dict(rejected)})
    print("Admitted", dict(Counter(row["role"] for work in works for row in work["variants"])), "Excluded", dict(rejected), flush=True)
    print("Unchanged detector input with exact attached HTML locations", dict(exact), flush=True)


def _hit(selected, target):
    a, b = target
    return any((count := max(0, min(b, y) - max(a, x))) >= (b - a) * MATCH["recall"]
               and count >= (max(b, y) - min(a, x)) * MATCH["iou"] for x, y in selected)


def totals(cases):
    output = {}
    for name in ("generic", "article"):
        counts = Counter()
        for work in cases:
            for row in work["variants"]:
                view = row["views"][name]
                role = row["role"]
                focus, selected = view["compact_focus"], view["selected_compact_ranges"]
                envelopes = []
                for start, end in selected:
                    if envelopes and envelopes[-1][1] == start:
                        envelopes[-1][1] = end
                    else:
                        envelopes.append([start, end])
                counts.update({f"{role}_files": 1, f"{role}_flags": bool(selected),
                               f"{role}_located": _hit(envelopes, focus),
                               f"{role}_source_offsets": bool(view["source_ranges"])})
        output[name] = dict(counts)
    return output


def evaluate(plan, snapshot, output):
    import torch

    from app.detectors.text import html_article
    from app.detectors.text.classifier import FourClassTextClassifier
    from app.detectors.text.extract import Document
    from app.detectors.text.pipeline import windows
    from app.detectors.text.segment import segment
    from scripts.audio_strong_traffic_view import network_free
    from scripts.writing_focus_v3 import BASELINE_ASSETS, BASELINE_REVISION
    from scripts.writing_multi_passage_panel import _prefix
    from scripts.writing_word_edit_transfer import model_assets

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    network_free()
    model_assets(snapshot)
    frozen = read(plan)
    if not _frozen_implementation(frozen, html_article):
        raise ValueError("The prepared plan or scoring implementation changed")
    inputs = set()
    cases = frozen["cases"]
    for work in cases:
        for row in work["variants"]:
            for view in row["views"].values():
                text = view.pop("text")
                units = segment(Document(text, view.pop("paragraphs")))
                view["sentences"] = [(unit.start, unit.end) for unit in units]
                view["native_sha256"] = text_hash(text)
                view["word_count"] = len(text.split())
                compressed = _prefix(text)
                view["compact_focus"] = [compressed[point] for point in view["focus"]]
                view["compact_sentences"] = [(compressed[unit.start], compressed[unit.end]) for unit in units]
                context = windows(units, 3)
                inputs.update(context)
                view["inputs"] = context
    detector = FourClassTextClassifier.from_snapshot(snapshot, "cpu")
    order = sorted(inputs, key=lambda value: (len(value), text_hash(value)))
    readings = {}
    for index in range(0, len(order), 192):
        batch = order[index:index + 192]
        scored = list(map(float, detector.predict(batch, batch=12, check=lambda _: None)))
        if any(not math.isfinite(value) for value in scored):
            raise ValueError("Non-finite classifier response")
        readings.update(zip(batch, map(float.hex, scored), strict=True))
    for work in cases:
        for row in work["variants"]:
            for view in row["views"].values():
                context = view.pop("inputs")
                view["input_sha256"] = [text_hash(value) for value in context]
                view["scores"] = [readings[value] for value in context]
                view["selected_compact_ranges"] = [region for region, score in zip(view["compact_sentences"], view["scores"], strict=True)
                                                   if float.fromhex(score) >= .5][:6]
    save(output, frozen | {"classifier_revision": BASELINE_REVISION, "classifier_assets": BASELINE_ASSETS,
                           "input_plan_sha256": file_hash(plan), "unique_contexts": len(inputs), "results": totals(cases)})
    review(output)


def review(path):
    value = read(path)
    if value["schema"] != SCHEMA or value["settings"] != SETTINGS or value["results"] != totals(value["cases"]):
        raise ValueError("The public outcome does not replay")
    for work in value["cases"]:
        for row in work["variants"]:
            for view in row["views"].values():
                selected = [region for region, response in zip(view["compact_sentences"], view["scores"], strict=True)
                            if float.fromhex(response) >= .5][:6]
                if selected != view["selected_compact_ranges"]:
                    raise ValueError("A recorded location differs from the frozen classifier comparison")
    print(value["results"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    for action, fields in (("freeze", ("folder", "urls", "output")), ("generate", ("plan", "output")),
                           ("qualify", ("plan", "folder", "output")), ("evaluate", ("plan", "snapshot", "output")),
                           ("review", ("path",))):
        mode = actions.add_parser(action)
        for field in fields:
            mode.add_argument(field, type=Path)
        if action == "freeze":
            mode.add_argument("prior", type=Path, nargs="+")
        if action == "generate":
            mode.add_argument("--family", choices=("qwen", "smol2"), required=True)
        if action == "qualify":
            mode.add_argument("transcripts", type=Path, nargs="+")
    options = vars(parser.parse_args())
    {"freeze": freeze, "generate": generate, "qualify": qualify, "evaluate": evaluate, "review": review}[options.pop("action")](**options)
