"""Compare exact eight-word localization across adjacent contexts on archived prose."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from collections import Counter, defaultdict
from importlib.metadata import version
from pathlib import Path
from statistics import mean

from app.detectors.text.extract import from_string, text_hash
from app.detectors.text.segment import segment
from scripts import writing_archive_author_intake as history
from scripts import writing_archive_prospective_intake as prose
from scripts.text_elife_source_intake_v272 import overlap
from scripts.text_short_insert_intake_v203 import fingerprints
from scripts.video_appearance_native_v108 import read, require, write
from scripts.writing_blind_rewrite_probe import EDIT_PROMPT, generator_configuration, valid_sentence, words
from scripts.writing_email_phrase_persistence import REVISION, WORDS, run_native
from scripts.writing_focus_v3 import BASELINE_ASSETS, digest

SCHEMA = "chipmunk.writing-phrase-context-stability/1"
BASE = Path(__file__).resolve().parents[1] / "data/independent-evaluation"
ROOT = Path(__file__).resolve().parents[2]
ANCHOR = history.SOURCES["kubernetes"]["anchor"]
FILES = re.compile(r"content/en/blog/(?:_posts/)?202[01]-\d{2}-[A-Za-z0-9_./-]+\.md")
WIDTH = 8
FLANK = 4
CAL_GROUPS = 11
HOLD_GROUPS = 12
SEEDS = {"qwen": 202611121, "smol2": 202611122}
PHRASE_PROMPT = ("Replace the bracketed phrase in the sentence with exactly eight words that preserve the meaning. "
                 "Return only the eight replacement words, separated by spaces, without brackets, explanation or punctuation. "
                 "Keep the original grammar. Sentence: {marked}")
GENERATION = {
    "direct_phrase": "One direct attempt to rewrite a marked phrase in exactly eight words.",
    "sentence_fragment": "One completed sentence rewrite. Select the contiguous punctuation-free eight-word interior "
                         "fragment with center closest to the original phrase's relative center. Compare only fixtures "
                         "that preserve the source sentences and exact flanking characters.",
}
ARMS = ("current", "one_context", "stable", "local_contrast", "current_plus_local_contrast")
METHOD = {
    "sources": "Eleven earlier bylined works set review points on untouched and ordinary same-work phrase substitutions. "
               "A fresh dated archive, author exclusion, historical text inventories, and exact 12-token exclusion fix held works.",
    "fixture": "Change exactly one internal eight-word phrase in the central sentence of five source sentences. Match each "
               "untouched excerpt with a same-work phrase substitution and two attempts from pinned Apache-2.0 generators. "
               "Keep all other source characters exact. Accept one response per generator only with original sentence boundaries.",
    "current": "First six complete native three-sentence context windows at raw response at least 0.50.",
    "one_context": "Use the smaller of the mean token margins over the central four words and all eight in the native context.",
    "stable": "Use the smaller of that persistence value in two contexts: previous+focus and focus+next.",
    "local_contrast": "Subtract the larger mean of the four preceding or following words from each paired-context persistence "
                      "value. Keep the smaller contrast.",
    "selection": "One uniquely strongest eligible candidate per experimental arm. Points equal the second-largest earlier "
                 "human source group maximum and must be strictly exceeded.",
    "qualification": "Useful means at least 0.30 nonwhitespace character IoU and 0.50 recall. Report best possible "
                     "overlap for every arm independently of scores, and false source groups including same-work substitutions.",
    "advance": "At least two additional held work groups across both generators with at most one extra human group when "
               "combined with the current arm. At least six admitted edits per generator.",
}


def clean_phrase(text):
    words = list(WORDS.finditer(text))
    return len(words) == WIDTH and " ".join(word[0] for word in words) == text


def phrase_ranges(text):
    words = list(WORDS.finditer(text))
    ranges = []
    for index in range(FLANK, len(words) - WIDTH - FLANK + 1):
        first, last = words[index].start(), words[index + WIDTH - 1].end()
        if clean_phrase(text[first:last]):
            ranges.append([first, last])
    return ranges


def paths():
    original = BASE / "writing-archive-prospective-transfer-2026-11-12"
    later = BASE / "writing-body-localization-transfer-2026-11-12"
    historical = BASE / "writing-archive-author-transfer-2026-11-11"
    return original / "originals.json", later / "originals.json", historical / history.SOURCES["kubernetes"]["archive"]


def prepare():
    used, later, archive = paths()
    old_rows = read(used)
    require(len(old_rows) == 36 and len(read(later)) == 10, "Retain the existing source rosters")
    excluded = {row["source"]["path"] for row in old_rows}
    prior_owners = {identity for row in old_rows for identity in row["byline"] + row["source"].get("editor", [])}
    for row in read(later):
        excluded.add(row["path"])
        text = history.git(archive, "show", f"{row['commit']}:{row['path']}")
        owner = prose.byline(text, prose.FRONT.match(text))
        require(owner is not None, "Earlier byline evidence differs")
        prior_owners.update(owner["names"])
    rights = history.rights_record(archive, ANCHOR, history.SOURCES["kubernetes"])
    require(rights is not None, "Archive terms did not verify")
    names = sorted((path for path in history.git(archive, "ls-tree", "-r", "--name-only", ANCHOR, "content/en/blog")
                    .splitlines() if FILES.fullmatch(path)), key=text_hash)
    pool, disposition = [], Counter()
    for path in names:
        raw = history.git(archive, "show", f"{ANCHOR}:{path}", limit=600_000)
        front = prose.FRONT.match(raw)
        owners = prose.byline(raw, front) if front else None
        if path in excluded or not owners or set(owners["names"]) & prior_owners:
            disposition["previous_work_or_no_fresh_byline"] += 1
            continue
        if history.RESTRICTED.search(raw):
            disposition["restricted_source"] += 1
            continue
        pair = prose.chosen_pair([item for item in prose.excerpts(raw, owners) if phrase_ranges(item["focus_text"])])
        if pair is None:
            disposition["no_matching_phrase_pair"] += 1
            continue
        pool.append({"group": "blog-" + text_hash(path)[:16], "before": pair[0], "after": pair[1],
                     "byline": owners["names"], "source": {"path": path, "commit": ANCHOR,
                     "byline_span": owners["span"], "blob_sha256": text_hash(raw), "license_sha256": rights[0]}})

    signatures = {row["group"]: fingerprints(row["before"]["text"]) | fingerprints(row["after"]["text"]) for row in pool}
    previous = [(name, digest(path), None, history.historical_texts(path)) for name, path in history.HISTORY.items()]
    previous += [("used_archive", digest(used), None, [row[part]["text"] for row in old_rows for part in ("before", "after")]),
                 ("used_archive_bodies", digest(later), None, [entry["text"] for row in read(later) for entry in row["paragraphs"]])]
    inventory, matched, _ = overlap(previous, signatures)
    chosen, seen = [], set()
    for item in pool:
        marks = signatures[item["group"]]
        owners = set(item["byline"])
        if item["group"] in matched or marks & seen or owners & prior_owners:
            disposition["source_or_fragment_overlap"] += 1
            continue
        chosen.append(item)
        seen.update(marks)
        prior_owners.update(owners)
        if len(chosen) == HOLD_GROUPS:
            break
    require(len(chosen) >= 8, "Too few eligible fresh works")
    calibration = [row for row in old_rows if phrase_ranges(row["before"]["focus_text"])
                   and phrase_ranges(row["after"]["focus_text"])][:CAL_GROUPS]
    require(len(calibration) == CAL_GROUPS, "Too few earlier groups with phrase controls")
    return {"calibration": calibration, "holdout": chosen, "source_qualification": dict(disposition),
            "archive_commit": ANCHOR, "archive_license_sha256": rights[0], "intake_paths": len(names), "prior_corpus": inventory}


def replacements(row):
    before, after = row["before"], row["after"]
    focus = before["focus_text"]
    first, last = min(phrase_ranges(focus), key=lambda span: text_hash(row["group"] + focus[slice(*span)]))
    original = focus[first:last]
    donors = [after["focus_text"][slice(*bounds)] for bounds in phrase_ranges(after["focus_text"])]
    donor = min((item for item in donors if item.casefold() != original.casefold()),
                key=lambda value: (abs(len(value) - len(original)), text_hash(value)))
    return first, last, original, donor


def seal(run, snapshot, generation_mode):
    from scripts.text_elife_rewrites_v273 import GENERATORS, MLX_VERSIONS

    require(snapshot.name == REVISION and all(digest(snapshot / name) == pin for name, pin in BASELINE_ASSETS.items()),
            "Pinned primary classifier differs")
    gen = generator_configuration()
    for family, (_, _, weight, folder) in GENERATORS.items():
        require(gen[family]["license"] == "Apache-2.0" and digest(folder / "model.safetensors") == weight,
                "Pinned generator assets or license differ")
    return {"schema": SCHEMA, "method": METHOD, "source_sha256": digest(run / "sources.json"),
            "impl_sha256": digest(Path(__file__)), "generators": gen, "seed": SEEDS, "generation_mode": generation_mode,
            "generation_method": GENERATION[generation_mode],
            "prompt": PHRASE_PROMPT if generation_mode == "direct_phrase" else EDIT_PROMPT,
            "generator_runtime": MLX_VERSIONS, "checkpoint_assets": BASELINE_ASSETS, "revision": REVISION,
            "support": {str(path.relative_to(ROOT)): digest(path) for path in map(Path, (
                prose.__file__, history.__file__, ROOT / "backend/scripts/writing_email_phrase_persistence.py",
                ROOT / "backend/app/detectors/text/extract.py", ROOT / "backend/app/detectors/text/segment.py",
                ROOT / "backend/app/detectors/text/classifier.py"))}}


def generation(run, freeze, family):
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm import load, stream_generate
    from mlx_lm.sample_utils import make_sampler

    from scripts.text_elife_rewrites_v273 import GENERATORS

    require(all(version(key) == value for key, value in freeze["generator_runtime"].items()), "Pinned MLX runtime differs")
    mx.set_default_device(mx.gpu)
    mx.random.seed(SEEDS[family])
    model, tokenizer = load(str(GENERATORS[family][3]), tokenizer_config={"trust_remote_code": False})
    nn.quantize(model, group_size=64, bits=4, mode="affine")
    mx.eval(model.parameters())
    results = {}
    for row in read(run / "sources.json")["holdout"]:
        first, last, _, _ = replacements(row)
        focus = row["before"]["focus_text"]
        prompt = generation_prompt(focus, first, last, freeze["generation_mode"])
        chat = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
        tokens = tokenizer.encode(chat, add_special_tokens=False)
        limit = 32 if freeze["generation_mode"] == "direct_phrase" else 96
        parts = list(stream_generate(model, tokenizer, tokens, max_tokens=limit, sampler=make_sampler(temp=.60, top_p=.90)))
        results[row["group"]] = {"text": "".join(part.text for part in parts), "finish": parts[-1].finish_reason,
                                 "focus_sha256": text_hash(focus), "prompt_sha256": text_hash(prompt)}
        print(f"{family} {len(results)} complete", flush=True)
    write(run / f"generate-{family}.json", {"freeze_sha256": digest(run / "freeze.json"), "outputs": results,
                                           "mlx_peak": mx.get_peak_memory()})


def generation_prompt(focus, first, last, mode):
    if mode == "direct_phrase":
        marked = focus[:first] + "[" + focus[first:last] + "]" + focus[last:]
        return PHRASE_PROMPT.format(marked=marked)
    count = len(words(focus))
    return EDIT_PROMPT.format(sentence=focus, minimum=max(12, int(.65 * count)), maximum=min(40, int(1.4 * count)))


def sentence_fragment(result, focus, start, stop):
    sentence = valid_sentence({"text": result["text"], "eos": result["finish"] == "stop"}, focus, generated=True)
    if sentence is None:
        return None
    target_position = 2 * len(list(WORDS.finditer(focus[:start]))) + WIDTH
    target_length = len(list(WORDS.finditer(focus)))
    locations = list(WORDS.finditer(sentence))
    candidates = []
    for index in range(1, len(locations) - WIDTH):
        text = sentence[locations[index].start():locations[index + WIDTH - 1].end()]
        if clean_phrase(text) and sum(a != b for a, b in zip(text.casefold().split(), focus[start:stop].casefold().split(), strict=True)) >= 2:
            offset = abs((2 * index + WIDTH) * target_length - target_position * len(locations))
            candidates.append((offset, index, text))
    return min(candidates)[2] if candidates else None


def build(run):
    sources = read(run / "sources.json")
    mode = read(run / "freeze.json")["generation_mode"]
    generated = {family: read(run / f"generate-{family}.json") for family in SEEDS}
    expected = {row["group"] for row in sources["holdout"]}
    require(all(set(item["outputs"]) == expected and item["freeze_sha256"] == digest(run / "freeze.json")
                for item in generated.values()), "Generated response roster differs")
    counts, fixtures = defaultdict(Counter), []
    for role in ("calibration", "holdout"):
        for row in sources[role]:
            first, last, old, donor = replacements(row)
            focus = row["before"]["focus_text"]
            versions = {"original": old, "ordinary": donor}
            if role == "holdout":
                for family, packet in generated.items():
                    result = packet["outputs"][row["group"]]
                    prompt = generation_prompt(focus, first, last, mode)
                    require(result["focus_sha256"] == text_hash(focus) and result["prompt_sha256"] == text_hash(prompt),
                            "Generation did not read the exact selected phrase")
                    candidate = (result["text"].strip().strip('"').removesuffix(".") if mode == "direct_phrase"
                                 else sentence_fragment(result, focus, first, last))
                    if result["finish"] != "stop":
                        counts[family]["unfinished"] += 1
                    elif candidate is None or not clean_phrase(candidate):
                        counts[family]["no_valid_eight_word_fragment"] += 1
                    elif sum(a.casefold() != b.casefold() for a, b in zip(candidate.split(), old.split(), strict=True)) < 2:
                        counts[family]["too_similar"] += 1
                    else:
                        versions[family] = candidate
            text, start = row["before"]["text"], row["before"]["focus"][0]
            start += first
            stop = start + (last - first)
            require(text[start:stop] == old, "Archive phrase does not bind to excerpt")
            for kind, value in versions.items():
                submitted = text[:start] + value + text[stop:]
                doc = from_string(submitted)
                units = segment(doc)
                expected_sentences = [entry["text"] for entry in row["before"]["witness"]]
                expected_sentences[2] = focus[:first] + value + focus[last:]
                if doc.text != submitted or [unit.text for unit in units] != expected_sentences:
                    require(kind in SEEDS, "Ordinary fixture changed source sentence geometry")
                    counts[kind]["segmentation_changed"] += 1
                    continue
                fixtures.append({"group": row["group"], "kind": kind, "role": role, "text": submitted,
                                 "sha256": text_hash(submitted), "span": [start, start + len(value)]})
                counts[role][kind] += 1
    return {"fixtures": fixtures, "counts": {key: dict(value) for key, value in counts.items()}}


def bindings(text):
    units = segment(from_string(text))
    require(len(units) == 5 and all(text[unit.start:unit.end] == unit.text for unit in units), "Native context geometry differs")
    by_focus = []
    for index, focus in enumerate(units):
        previous, following = units[max(0, index - 1):index], units[index + 1:index + 2]
        groups = {"full": [*previous, focus, *following]}
        if previous and following and phrase_ranges(focus.text):
            groups |= {"left": [*previous, focus], "right": [focus, *following]}
        by_focus.append({"span": [focus.start, focus.end], "views": {
            view: {"text": " ".join(unit.text for unit in members),
                   "spans": [[unit.start, unit.end] for unit in members],
                   "offset": sum(len(unit.text) + 1 for unit in members if unit.start < focus.start)}
            for view, members in groups.items()}})
    return by_focus


def reading(tokens, start, stop):
    values = [[margin for a, b, margin in tokens if a <= index < b] for index in range(start, stop)]
    require(all(values), "Native contextual token coverage is incomplete")
    return mean(map(mean, values))


def measure(row, measured):
    baseline, candidates = [], []
    text = row["text"]
    for focus in bindings(text):
        views = focus["views"]
        baseline.append((views["full"]["spans"], measured[views["full"]["text"]]["raw"]))
        if len(views) != 3:
            continue
        start, stop = focus["span"]
        units = list(WORDS.finditer(text[start:stop]))
        word_views = {}
        for name, entry in views.items():
            offset = entry["offset"]
            require(entry["text"][offset:offset + stop - start] == text[start:stop], "Model prompt differs from source focus")
            tokens = measured[entry["text"]]["offsets"]
            word_views[name] = [reading(tokens, offset + word.start(), offset + word.end()) for word in units]
        for index in range(FLANK, len(units) - WIDTH - FLANK + 1):
            end = index + WIDTH
            if not clean_phrase(text[start + units[index].start():start + units[end - 1].end()]):
                continue
            persistence = {name: min(mean(values[index + 2:index + 6]), mean(values[index:end]))
                           for name, values in word_views.items()}
            contrast = {name: persistence[name] - max(mean(word_views[name][index - FLANK:index]),
                                                       mean(word_views[name][end:end + FLANK])) for name in ("left", "right")}
            candidates.append({"span": [start + units[index].start(), start + units[end - 1].end()],
                               "one_context": persistence["full"], "stable": min(persistence["left"], persistence["right"]),
                               "local_contrast": min(contrast.values())})
    return baseline, candidates


def score(run, snapshot, phase):
    import torch

    from app.detectors.text.classifier import FourClassTextClassifier

    locked = read(run / "fixtures.json")
    require(locked == build(run), "Fixture bytes or native sentence geometry changed")
    rows = [row for row in locked["fixtures"] if row["role"] == phase]
    values = list(dict.fromkeys(entry["text"] for row in rows for focus in bindings(row["text"])
                                for entry in focus["views"].values()))
    torch.set_num_threads(1)
    classifier = FourClassTextClassifier.from_snapshot(snapshot, "cpu")
    measured, forwards, reduction_error = run_native(classifier, values)
    sample = values[:6]
    exact = classifier.predict(sample, batch=6, check=lambda _label: None)
    require(all(abs(measured[content]["raw"] - probability) < 1e-6 for content, probability in zip(sample, exact, strict=True)),
            "Native logits diverge from classifier predict")
    output = {"responses": measured, "forwards": forwards + 1, "inputs": len(values), "max_reduce_error": reduction_error,
              "fixture_sha256": digest(run / "fixtures.json")}
    write(run / f"{phase}-responses.json", output)
    return [(row, *measure(row, measured)) for row in rows]


def points(rows):
    groups = defaultdict(lambda: {name: -math.inf for name in ARMS[1:-1]})
    for row, _, candidates in rows:
        require(candidates and row["kind"] not in SEEDS, "A calibration source lacks candidate coverage or human scope")
        for name in ARMS[1:-1]:
            groups[row["group"]][name] = max(groups[row["group"]][name], *(item[name] for item in candidates))
    return {name: sorted((row[name] for row in groups.values()), reverse=True)[1] for name in ARMS[1:-1]}


def chosen(baseline, candidates, thresholds):
    groups = {"current": [spans for spans, probability in baseline if probability >= .5][:6]}
    for name in ARMS[1:-1]:
        values = sorted(candidates, key=lambda item: item[name], reverse=True)
        winner = values[0] if values and values[0][name] > thresholds[name] and (
            len(values) == 1 or values[0][name] > values[1][name]) else None
        groups[name] = [[winner["span"]]] if winner else []
    groups[ARMS[-1]] = groups["current"] + groups["local_contrast"]
    return groups


def useful(text, target, ranges):
    expected = {index for index in range(*target) if not text[index].isspace()}
    marked = {index for first, last in ranges for index in range(first, last) if not text[index].isspace()}
    common = len(expected & marked)
    return 10 * common >= 3 * len(expected | marked) and 2 * common >= len(expected)


def summarize(rows, thresholds):
    counts, human, gained = defaultdict(Counter), defaultdict(set), defaultdict(set)
    cases = []
    for row, baseline, candidates in rows:
        selected = chosen(baseline, candidates, thresholds)
        geometry = {"current": [span for span, _ in baseline],
                    **{name: [[item["span"]] for item in candidates] for name in ARMS[1:-1]}}
        geometry[ARMS[-1]] = geometry["current"] + geometry["local_contrast"]
        ledger = {}
        kind = row["kind"]
        for arm in ARMS:
            where = selected[arm]
            hit = kind in SEEDS and any(useful(row["text"], row["span"], region) for region in where)
            possible = kind in SEEDS and any(useful(row["text"], row["span"], region) for region in geometry[arm])
            counts[kind].update({f"{arm}_useful": int(hit), f"{arm}_flag": int(bool(where)), f"{arm}_possible": int(possible)})
            if kind not in SEEDS and where:
                human[arm].add(row["group"])
            if hit and not ledger.get("current", {}).get("useful", False):
                gained[arm].add((row["group"], kind))
            ledger[arm] = {"spans": where, "useful": hit, "geometry_allows_useful": possible}
        counts[kind]["cases"] += 1
        cases.append({"group": row["group"], "kind": kind, "fixture_sha256": row["sha256"], "target": row["span"],
                      "candidate_count": len(candidates), "outcomes": ledger})
    added = gained[ARMS[-1]]
    new_human = human[ARMS[-1]] - human["current"]
    advance = (len({group for group, _ in added}) >= 2 and {kind for _, kind in added} == set(SEEDS)
               and len(new_human) <= 1 and all(counts[family]["cases"] >= 6 for family in SEEDS))
    return {"counts": {key: dict(value) for key, value in counts.items()},
            "false_groups": {name: len(human[name]) for name in ARMS}, "added_useful_groups": sorted(map(list, added)),
            "new_false_groups": len(new_human), "advance": advance, "cases": cases}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("phase", choices=("freeze", "generate", "fixtures", "calibration", "holdout"))
    cli.add_argument("run", type=Path)
    cli.add_argument("snapshot", type=Path)
    cli.add_argument("--family", choices=SEEDS)
    cli.add_argument("--generation-mode", choices=GENERATION, default="sentence_fragment")
    args = cli.parse_args()
    require(os.getenv("HF_HUB_OFFLINE") == os.getenv("TRANSFORMERS_OFFLINE") == "1", "Require offline model runtime")
    os.umask(0o077)
    run = args.run
    if args.phase == "freeze":
        run.mkdir(exist_ok=False)
        sources = prepare()
        write(run / "sources.json", sources)
        write(run / "freeze.json", seal(run, args.snapshot, args.generation_mode))
        print(f"Frozen {len(sources['calibration'])} earlier groups and {len(sources['holdout'])} fresh works", flush=True)
        return
    freeze = read(run / "freeze.json")
    require(freeze == seal(run, args.snapshot, freeze["generation_mode"]), "Methods, source or checkpoint differ from the freeze")
    if args.phase == "generate":
        require(args.family is not None, "A frozen generator family is required")
        generation(run, freeze, args.family)
    elif args.phase == "fixtures":
        fixtures = build(run)
        write(run / "fixtures.json", fixtures)
        print(fixtures["counts"], flush=True)
    elif args.phase == "calibration":
        outputs = score(run, args.snapshot, args.phase)
        point = points(outputs)
        write(run / "points.json", {"points": point, "response_sha256": digest(run / "calibration-responses.json"),
                                    "fixture_sha256": digest(run / "fixtures.json")})
        print(point, flush=True)
    else:
        locked = read(run / "points.json")
        require(locked["response_sha256"] == digest(run / "calibration-responses.json")
                and locked["fixture_sha256"] == digest(run / "fixtures.json"), "Calibration commitments differ")
        result = summarize(score(run, args.snapshot, args.phase), locked["points"])
        write(run / "result.json", result)
        print(json.dumps({key: value for key, value in result.items() if key != "cases"}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
