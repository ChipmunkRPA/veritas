# SPDX-License-Identifier: Apache-2.0
"""Evaluate a base-versus-instruct likelihood contrast on exact native writing spans."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

from scripts.writing_transition_deviation import file_hash, read, save, text_hash

SCHEMA = "chipmunk.writing-alignment-preference/1"
ROOT = Path(__file__).resolve().parents[2]
HF = ROOT / "models/hf/hub"
BASE = "8faed761d45a263340a0528343f099c05c9a4323"
INSTRUCT = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
CLASSIFIER = "f1795c86806e6838d4afa33d0b1427f8430c9615"
SNAPSHOTS = {
    "base": ("Qwen/Qwen2.5-1.5B", BASE, "a961db72e75d52b18e6b0c9d379e51a26973b233385e0e127fdda7d648aec796"),
    "instruct": ("Qwen/Qwen2.5-1.5B-Instruct", INSTRUCT, "dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee"),
}
TOKENIZER_SHA256 = "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539"
MIN_TOKENS, MAX_TOKENS, MAX_TEXT_TOKENS = 20, 144, 768
METHODS = ("self_information", "unweighted")
PROMPT = ("Paraphrase the one sentence below. Change its grammar and wording without changing any facts. "
          "Keep the original digits and proper names exactly. Use about {words} words in one complete sentence. "
          "Return only the new sentence.\n\n{target}")
RULE = {"primary": "self_information", "floor": "second-highest ordinary source-group maximum on independent calibration sources",
        "candidate": "at most one complete extra one- or two-sentence interval, disjoint from incumbent intervals",
        "match_non_whitespace": {"recall_min": .5, "iou_min": .30},
        "advancement": "at least 2 new held source groups across both generators and no more than 1 extra ordinary source group",
        "prompt_sha256": text_hash(PROMPT), "attempts": 4, "sample_seed": 236694}
OMIT = re.compile(r"(?i)\b(?:references|image[s]? (?:by|credit|acquired)|courtesy|credit|copyright|report contact|https?)\b|[©“”]")


def sentences(text):
    from app.detectors.text.extract import from_string
    from app.detectors.text.segment import segment

    doc = from_string(text)
    if doc.text != text:
        raise ValueError("Native passage normalization changed the study text")
    return segment(doc)


def _selected_source(raw, seen):
    from app.detectors.text import html_article
    from scripts.writing_economic_paraphrase_holdout import fragments

    article = html_article.parts(raw)
    if not article:
        return None
    source = []
    for value, _ in article[0]:
        text = " ".join(value.split())
        if 14 <= len(text.split()) <= 190 and not OMIT.search(text):
            source.append(text)
    candidates = []
    for start in range(len(source)):
        for width in (3, 4, 5, 6):
            if start + width > len(source):
                break
            body = source[start:start + width]
            text = "\n\n".join(body)
            if not 165 <= sum(len(p.split()) for p in body) <= 350 or seen & fragments(text):
                continue
            focus = sentences(text)
            if not 8 <= len(focus) <= 24:
                continue
            roles = [(idx, row) for idx, row in enumerate(focus) if 18 <= row.n_words <= 54
                     and row.text.endswith(".") and not re.search(r"[;“”]", row.text)]
            for index, row in roles:
                if not 2 <= index < len(focus) - 2 or text.count(row.text) != 1:
                    continue
                donors = [(other, control) for other, control in roles if abs(index - other) > 1
                          and .77 <= control.n_words / row.n_words <= 1.30 and text.count(control.text) == 1]
                if donors:
                    # Exchange the two authentic source sentences. Keep both in the file and preserve all other characters.
                    other, donor = min(donors, key=lambda item: (abs(row.n_words - item[1].n_words), text_hash(item[1].text)))
                    candidates.append((len(focus) < 12, text_hash(row.text), start, width, text, [row.start, row.end],
                                       [donor.start, donor.end], index, other))
    if not candidates:
        return None
    _, _, start, width, text, target, donor, index, other = min(candidates)
    return {"text": text, "target": target, "donor": donor, "target_sentence": index,
            "donor_sentence": other, "body_blocks": [start, start + width], "scope": article[1]}


def freeze(folder, urls, output, prior):
    from lxml import html

    from scripts.writing_economic_paraphrase_holdout import find_texts, fragments

    seen = {piece for path in prior for value in find_texts(read(path)) for piece in fragments(value)}
    accepted, excluded = defaultdict(list), Counter()
    for url in urls.read_text().splitlines():
        slug = url.rstrip("/").split("/")[-1]
        path = folder / f"{slug}.html"
        publisher = "noaa" if "ncei.noaa.gov" in url else "nasa"
        if not path.exists():
            excluded[f"unavailable_{publisher}"] += 1
            continue
        raw = path.read_text()
        tree = html.fromstring(raw)
        metadata = tree.xpath('//meta[contains(@property,"published") or contains(@name,"date")]/@content')
        dates = [value[:10] for value in metadata if re.match(r"^20(?:1[0-9]|20)-\d\d-\d\dT", value)]
        if not dates or (selected := _selected_source(raw, seen)) is None:
            excluded[f"no_eligible_excerpt_{publisher}"] += 1
            continue
        accepted[publisher].append(selected | {"id": slug, "date": min(dates), "publisher": publisher,
                                                "url": url, "original_sha256": file_hash(path)})
        seen.update(fragments(selected["text"]))
    selected = []
    for publisher, rows in sorted(accepted.items()):
        for index, row in enumerate(sorted(rows, key=lambda value: text_hash(value["url"]))):
            selected.append(row | {"split": "calibration" if index % 2 == 0 else "held"})
    print("Sources", dict(Counter((row["publisher"], row["split"]) for row in selected)), "Excluded", dict(excluded), flush=True)
    if len(accepted) < 2 or min(Counter(row["split"] for row in selected).values(), default=0) < 8:
        raise ValueError("Need eight independent calibration and eight held source groups from both publishers")
    save(output, {"schema": SCHEMA, "rule": RULE, "runner_sha256": file_hash(Path(__file__)), "sources": selected,
                  "excluded": dict(excluded), "exposure_sha256": [file_hash(path) for path in prior]})


def model_location(repo, revision):
    return HF / f"models--{repo.replace('/', '--')}" / "snapshots" / revision


def rewrite_family(family, checkpoint=None):
    if family == "smol2":
        from scripts.text_elife_rewrites_v273 import GENERATORS

        repo, revision, digest, snapshot = GENERATORS["smol2"]
        return repo, revision, digest, checkpoint or snapshot
    return ("mlx-community/granite-3.3-2b-instruct-4bit", "58246c5498495c14599525c852cfadb66c9f3084",
            "f37ff14b14f2db07b4fc476e57e619b038b264fcc9f0681bc13274772698ac93", checkpoint)


def generate(source, family, output, checkpoint):
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm import batch_generate, load
    from mlx_lm.sample_utils import make_sampler

    repo, revision, digest, snapshot = rewrite_family(family, checkpoint)
    if snapshot is None or file_hash(snapshot / "model.safetensors") != digest:
        raise ValueError("Generator weights differ from the pinned Apache-2.0 checkpoint")
    mx.set_cache_limit(96 * 1024 * 1024)
    model, tokenizer = load(str(snapshot), tokenizer_config={"trust_remote_code": False, "local_files_only": True})
    if "quantization" not in json.loads((snapshot / "config.json").read_text()):
        nn.quantize(model, mode="affine", group_size=64, bits=4)
    mx.eval(model.parameters())
    held = [row for row in read(source)["sources"] if row["split"] == "held"]
    prompts = []
    for row in held:
        original = row["text"][slice(*row["target"])]
        prompt = PROMPT.format(words=len(original.split()), target=original)
        prompts.append(tokenizer.encode(tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                                                    add_generation_prompt=True), add_special_tokens=False))
    attempts = []
    mx.random.seed(RULE["sample_seed"])
    for attempt in range(RULE["attempts"]):
        values = []
        sampler = make_sampler(temp=.62 if attempt else 0., top_p=.94)
        for index in range(0, len(prompts), 4):
            values.extend(batch_generate(model, tokenizer, prompts[index:index + 4], max_tokens=102, sampler=sampler).texts)
        attempts.append(values)
    save(output, {"schema": SCHEMA, "source_sha256": file_hash(source), "model": [repo, revision, digest], "family": family,
                  "replies": {row["id"]: value for row, value in zip(held, zip(*attempts, strict=True), strict=True)}})


def _replacement(value, original):
    from scripts.writing_blind_rewrite_probe import distance

    value = " ".join(value.strip().split())
    if (not value or not value.endswith(".") or OMIT.search(value) or re.search(r"[\"`]|(?i:here is|revised sentence|rewritten sentence)", value)
            or not .72 <= len(value.split()) / len(original.split()) <= 1.35
            or not .18 <= distance(value, original) <= .84 or len(sentences(value)) != 1):
        return None
    numbers = r"\d[\d,.]*\d|\d"
    return value if set(re.findall(numbers, value)) == set(re.findall(numbers, original)) else None


def qualify(source, output, replies):
    plan = read(source)
    if plan["runner_sha256"] != file_hash(Path(__file__)):
        raise ValueError("Frozen source plan uses a different implementation")
    held = {row["id"] for row in plan["sources"] if row["split"] == "held"}
    editors = {}
    for path in replies:
        row = read(path)
        family = row["family"]
        if (row["source_sha256"] != file_hash(source) or set(row["replies"]) != held or family in editors
                or row["model"] != list(rewrite_family(family)[:3])):
            raise ValueError("A generator transcript differs from the pinned sources or weights")
        editors[family] = row["replies"]
    if set(editors) != {"smol2", "granite"}:
        raise ValueError("Run both pinned independent editor families")
    cases, omitted = [], Counter()
    for row in plan["sources"]:
        text, (start, end), (left, right) = (row[key] for key in ("text", "target", "donor"))
        early, late = sorted(((start, end), (left, right)))
        swapped = text[:early[0]] + text[slice(*late)] + text[early[1]:late[0]] + text[slice(*early)] + text[late[1]:]
        versions = {"authentic": (text, None), "same_article_swap": (swapped, None)}
        if row["split"] == "held":
            for family, values in editors.items():
                proposals = values[row["id"]]
                if len(proposals) != RULE["attempts"]:
                    raise ValueError("Generator transcript lacks the fixed attempt count")
                for reply in proposals:
                    change = _replacement(reply, text[start:end])
                    if change is None:
                        continue
                    candidate = text[:start] + change + text[end:]
                    focus = [start, start + len(change)]
                    g = [[s.start, s.end] for s in sentences(candidate)]
                    if len(g) == len(sentences(text)) and g[row["target_sentence"]] == focus:
                        versions[family] = candidate, focus
                        break
                else:
                    omitted[family] += 1
        for role, (content, truth) in versions.items():
            cases.append({key: value for key, value in row.items() if key != "text"} | {
                "text": content, "truth": truth, "role": role, "sha256": text_hash(content),
            })
    print("Cases", dict(Counter((row["split"], row["role"]) for row in cases)), "Omitted", dict(omitted), flush=True)
    save(output, {"schema": SCHEMA, "rule": RULE, "runner_sha256": plan["runner_sha256"], "cases": cases,
                  "sources_sha256": file_hash(source), "transcripts_sha256": [file_hash(path) for path in replies],
                  "exposure_sha256": plan["exposure_sha256"], "source_exclusions": plan["excluded"], "omitted_rewrites": dict(omitted)})


def causal_reader():
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm import load
    from transformers import AutoTokenizer

    from scripts.writing_token_curvature import target_token_indices

    mx.set_cache_limit(96 * 1024 * 1024)
    models, heads = [], []
    for repo, revision, digest in SNAPSHOTS.values():
        snapshot = model_location(repo, revision)
        if file_hash(snapshot / "model.safetensors") != digest or file_hash(snapshot / "tokenizer.json") != TOKENIZER_SHA256:
            raise ValueError("The causal pair must use the pinned shared vocabulary and pinned model weights")
        model, _ = load(str(snapshot), tokenizer_config={"trust_remote_code": False, "local_files_only": True})
        nn.quantize(model, mode="affine", group_size=64, bits=4)
        mx.eval(model.parameters())
        head = model.model.embed_tokens.as_linear if model.args.tie_word_embeddings else model.lm_head
        models.append(model)
        heads.append(head)
    tokenizer = AutoTokenizer.from_pretrained(model_location(*SNAPSHOTS["base"][:2]), local_files_only=True, trust_remote_code=False)

    def reader(text, units):
        encoded = tokenizer(text, add_special_tokens=False, truncation=False, return_offsets_mapping=True)
        ids = [tokenizer.eos_token_id, *encoded["input_ids"]]
        offsets = [(0, 0), *encoded["offset_mapping"]]
        if len(ids) - 1 > MAX_TEXT_TOKENS:
            return [], len(ids) - 1
        candidates = []
        for first, start in enumerate(units):
            for after in (first + 1, first + 2):
                if after > len(units) or units[after - 1].paragraph != start.paragraph:
                    continue
                end = units[after - 1]
                try:
                    positions = target_token_indices(text, offsets, (start.start, end.end))
                except ValueError:
                    continue
                if MIN_TOKENS <= len(positions) <= MAX_TOKENS:
                    candidates.append((first, after, positions))
        if not candidates:
            return [], len(ids) - 1
        selected = sorted({index for _, _, positions in candidates for index in positions})
        hidden = [model.model(mx.array([ids]))[0] for model in models]
        mx.eval(*hidden)
        moments = {}
        for batch in range(0, len(selected), 16):
            indices = selected[batch:batch + 16]
            previous, actual = mx.array([index - 1 for index in indices]), mx.array([ids[index] for index in indices])
            logits = [head(vector[previous]).astype(mx.float32) for head, vector in zip(heads, hidden, strict=True)]
            base, aligned = (value - mx.logsumexp(value, axis=1, keepdims=True) for value in logits)
            delta, probabilities = aligned - base, mx.exp(base)
            measures = []
            for values in (-aligned * delta, delta):
                expected = mx.sum(probabilities * values, axis=1)
                variance = mx.sum(probabilities * mx.square(values - expected[:, None]), axis=1)
                residual = values[mx.arange(len(indices)), actual] - expected
                measures.extend((residual, variance))
            results = mx.stack(measures, axis=1).tolist()
            if any(not all(map(math.isfinite, row)) or row[1] <= 0 or row[3] <= 0 for row in results):
                raise ValueError("Degenerate conditional likelihood contrast")
            moments.update((index, [round(value, 7) for value in row]) for index, row in zip(indices, results, strict=True))
        result = []
        for first, after, positions in candidates:
            values = [math.fsum(channel) for channel in zip(*(moments[index] for index in positions), strict=True)]
            result.append([first, after, len(positions), values])
        return result, len(ids) - 1

    return reader


def capture(plan_path, split, output, lock=None):
    from scripts.writing_multi_passage_panel import _prefix

    plan = read(plan_path)
    if plan["runner_sha256"] != file_hash(Path(__file__)):
        raise ValueError("Prospectively fixed implementation differs")
    if split == "held" and (lock is None or read(lock)["plan_sha256"] != file_hash(plan_path)):
        raise ValueError("Freeze the operating points before held inference")
    measure = causal_reader()
    results = []
    for row in (case for case in plan["cases"] if case["split"] == split):
        text = row["text"]
        if text_hash(text) != row["sha256"]:
            raise ValueError("Prepared study text differs from its hash")
        units = sentences(text)
        compact = _prefix(text)
        readings, tokens = measure(text, units)
        results.append({key: value for key, value in row.items() if key != "text"} | {
            "tokens": tokens, "sentence_bounds": [[unit.start, unit.end] for unit in units],
            "compressed_bounds": [[compact[unit.start], compact[unit.end]] for unit in units],
            "compressed_truth": [compact[i] for i in row["truth"]] if row["truth"] else None, "readings": readings,
        })
        if len(results) % 10 == 0:
            print(split, len(results), flush=True)
    save(output, {"schema": SCHEMA, "plan_sha256": file_hash(plan_path), "runtime": "MLX offline affine four-bit",
                  "split": split, "cases": results, "lock_sha256": file_hash(lock) if lock else None})


def bounds(cases):
    return {arm: _ordinary_point(cases, index) for index, arm in enumerate(METHODS)}


def score(row, index):
    numerator, variance = row[3][index * 2:index * 2 + 2]
    return round(numerator / math.sqrt(variance), 8)


def _ordinary_point(cases, index):
    groups = defaultdict(list)
    for case in cases:
        if case["role"] not in {"authentic", "same_article_swap"}:
            raise ValueError("Calibrate on ordinary writing only")
        groups[case["id"]].extend(score(row, index) for row in case["readings"])
    points = sorted(max(group) for group in groups.values() if group)
    if len(points) < 8:
        raise ValueError("Too few independent ordinary groups to fix review points")
    return points[-2]


def calibrate(plan, earlier, output):
    receipt = read(earlier)
    if receipt["plan_sha256"] != file_hash(plan) or receipt["split"] != "calibration":
        raise ValueError("Calibration source receipt differs")
    points = bounds(receipt["cases"])
    print("Frozen points", points, flush=True)
    save(output, {"rule": RULE, "plan_sha256": file_hash(plan), "calibration_sha256": file_hash(earlier), "points": points})


def baseline(plan_path, output):
    import torch

    from app.detectors.text.classifier import FourClassTextClassifier
    from app.detectors.text.pipeline import windows
    from app.evidence.document_passage_intervals import MIN_SENTENCES, _reviewed_choices
    from scripts.writing_word_edit_transfer import model_assets

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    snapshot = model_location("wasitaigeneratedcom/ai-text-detector-small", CLASSIFIER)
    model_assets(snapshot)
    detector = FourClassTextClassifier.from_snapshot(snapshot, "cpu")
    plan = read(plan_path)
    cases = [case for case in plan["cases"] if case["split"] == "held"]
    inputs, prepared = set(), []
    for row in cases:
        units = sentences(row["text"])
        contextual = windows(units, 3)
        isolated = [sentence.text for sentence in units] if len(units) >= MIN_SENTENCES else []
        inputs.update([*contextual, *isolated])
        prepared.append((row, units, contextual, isolated))
    order = sorted(inputs, key=lambda value: (len(value), text_hash(value)))
    responses = {}
    for start in range(0, len(order), 192):
        chunk = order[start:start + 192]
        responses.update(zip(chunk, map(float, detector.predict(chunk, batch=12, check=lambda _: None)), strict=True))
    results = []
    for row, units, contextual, isolated in prepared:
        geometry = [[unit.start, unit.end, unit.n_words] for unit in units]
        readings = {name: [round(responses[value], 6) for value in view]
                    for name, view in (("context", contextual), ("isolated", isolated))}
        existing = [i for i, value in enumerate(contextual) if responses[value] >= .5][:6]
        intervals = _reviewed_choices(geometry, readings, document_baseline=True, report_review=True)[0] if isolated else []
        results.append({"id": row["id"], "role": row["role"], "source_sha256": row["sha256"], "geometry": geometry,
                        "card_centers": existing, "intervals": [[a, b] for a, b, *_ in intervals], "readings": readings})
    save(output, {"schema": SCHEMA, "plan_sha256": file_hash(plan_path), "cases": results})


def _location(bounds, span):
    first, after = span
    return bounds[first][0], bounds[after - 1][1]


def _useful(box, truth):
    first, stop = box
    start, end = truth
    shared = min(stop, end) - max(first, start)
    return shared >= .5 * (end - start) and shared >= .30 * (max(stop, end) - min(first, start))


def summarize(held, current, points):
    by_case = {(row["id"], row["role"]): row for row in current}
    totals = defaultdict(Counter)
    added_groups, added_families, ordinary_groups = defaultdict(set), defaultdict(set), defaultdict(set)
    for row in held:
        existing = by_case[row["id"], row["role"]]
        if existing["source_sha256"] != row["sha256"] or [p[:2] for p in existing["geometry"]] != row["sentence_bounds"]:
            raise ValueError("Current and candidate readings use different source coordinates")
        incumbent = [*existing["intervals"], *[[idx, idx + 1] for idx in existing["card_centers"]]]
        truth = row["compressed_truth"]
        mapped = row["compressed_bounds"]
        hit = bool(truth and any(_useful(_location(mapped, region), truth) for region in incumbent))
        for index, key in enumerate(METHODS):
            options = [reading for reading in row["readings"] if not any(max(reading[0], a) < min(reading[1], b) for a, b in incumbent)]
            selected = max(options, key=lambda reading: (score(reading, index), -reading[0], -reading[1]), default=None)
            best = _location(mapped, selected[:2]) if selected is not None else None
            above = selected is not None and score(selected, index) > points[key]
            positive = bool(best and truth and _useful(best, truth))
            for group in ("all", row["publisher"], row["role"]):
                counts = totals[key, group]
                counts.update({"files": 1, "assessed": bool(row["readings"]), "existing_alerts": bool(incumbent), "existing_hits": hit,
                               "additional_hits": above and positive and not hit, "additional_ordinary_alerts": above and truth is None and not incumbent,
                               "additional_ordinary_locations": above and truth is None, "misplaced_additions": above and truth is not None and not positive,
                               "rank_only_correct": positive and not hit})
            if above and positive and not hit:
                added_groups[key].add(row["id"])
                added_families[key].add(row["role"])
            if above and truth is None and not incumbent:
                ordinary_groups[key].add(row["id"])
    return {key: {group: dict(value) for (name, group), value in totals.items() if name == key} | {
        "added_groups": len(added_groups[key]), "added_families": sorted(added_families[key]),
        "new_ordinary_groups": len(ordinary_groups[key]),
        "advances": len(added_groups[key]) >= 2 and len(added_families[key]) == 2 and len(ordinary_groups[key]) <= 1,
    } for key in METHODS}


def publish(earlier, held, locked, existing, output):
    calibration, current, settings, testing = (read(path) for path in (earlier, existing, locked, held))
    if (settings["calibration_sha256"] != file_hash(earlier) or testing["lock_sha256"] != file_hash(locked)
            or len({row["plan_sha256"] for row in (calibration, current, settings, testing)}) != 1):
        raise ValueError("Evaluation phases differ from the pinned design")
    if {row["id"] for row in calibration["cases"]} & {row["id"] for row in testing["cases"]}:
        raise ValueError("Calibration and held cases share a source group")
    outcomes = summarize(testing["cases"], current["cases"], settings["points"])
    save(output, {"schema": SCHEMA, "rule": RULE, "calibration": calibration, "held": testing, "baseline": current,
                  "points": settings["points"], "outcomes": outcomes, "models": SNAPSHOTS})
    review(output)


def review(path):
    receipt = read(path)
    current, calibration, held = (receipt[key]["cases"] for key in ("baseline", "calibration", "held"))
    points = bounds(calibration)
    outcomes = summarize(held, current, points)
    if receipt["points"] != points or outcomes != receipt["outcomes"]:
        raise ValueError("Saved writing outcomes do not replay")
    print(outcomes, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    commands = {"freeze": (freeze, ("folder", "urls", "output")), "generate": (generate, ("source", "output")),
                "qualify": (qualify, ("source", "output")), "capture": (capture, ("plan_path", "output")),
                "calibrate": (calibrate, ("plan", "earlier", "output")), "baseline": (baseline, ("plan_path", "output")),
                "publish": (publish, ("earlier", "held", "locked", "existing", "output")), "review": (review, ("path",))}
    for action, (command, fields) in commands.items():
        mode = actions.add_parser(action)
        for field in fields:
            mode.add_argument(field, type=Path)
        if action == "freeze":
            mode.add_argument("prior", type=Path, nargs="+")
        if action == "generate":
            mode.add_argument("--family", choices=("smol2", "granite"), required=True)
            mode.add_argument("--checkpoint", type=Path)
        if action == "qualify":
            mode.add_argument("replies", type=Path, nargs="+")
        if action == "capture":
            mode.add_argument("--split", choices=("calibration", "held"), required=True)
            mode.add_argument("--lock", type=Path)
    options = vars(parser.parse_args())
    commands[options.pop("action")][0](**options)
