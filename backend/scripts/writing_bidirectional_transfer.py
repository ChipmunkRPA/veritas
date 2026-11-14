"""Test whether masked words gain less support from an AI-edited sentence's true flanks."""

from __future__ import annotations

import argparse
import importlib.metadata
import os
import re
from collections import Counter, defaultdict
from math import fsum, isfinite, sqrt
from pathlib import Path
from statistics import median

from scripts import writing_revision_trajectory_transfer as source
from scripts import writing_temperature_transfer as short
from scripts.writing_focus_v3 import BASELINE_ASSETS, BASELINE_REVISION

SCHEMA = "chipmunk.writing-bidirectional-transfer/1"
REVISION = "0772d26477d98b5896774fa570b581100fb60dcc"
ASSETS = {
    "README.md": "0b189aba67afc438cb4c5a484e8835d86d269d0f23f1da66189b34a6590f8928",
    "config.json": "6216e741e14a715237a4aa385eb7b95d675d8e0846b03aed1ee7696341ca27ea",
    "tokenizer_config.json": "06e673dd9251e2b2ba8a6e2fd15bd4f57de436ed43f02e757f20380952185080",
    "vocab.txt": "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3",
    "model.safetensors": "d07ef6c8ddf291b1f1444fe62e18b856a320d4f699876363b840713cb488438f",
}
NATIVE = {
    "old_a": "217eb06cc145d060cfdb1242d28219d1a0a9817a3278744bb57bdcc45a739ee9",
    "old_b": "dfb6a52072e32af9acf75b42f84b671850d402a0ba534f414e4d8fa31aa6ad58",
    "additional_revisions": "e2147c608e0924eff09ab6e0128112347075c0de0f7e7a9486c2a05085ef42ea",
}
RETAINED = {
    "writing-word-edit-transfer-2026-11-13.json": "86837d924e3c5256f1bd731ffb56fcf0363090df48f3883bce3ac8a9da4367ec",
    "writing-surprisal-dynamics-2026-11-13.json": "157eea47ab989440608a0d0f91f19a38dab2a1dd795a6024032b4adc33032900",
}
ARMS = ("sentence_only", "exterior")
GENERATORS = {"gemini-2.5-flash", "gpt-5.4-nano"}
STOP = frozenset({"a", "an", "the", "to", "of", "at", "and", "or", "in", "on", "as", "is", "it", "by", "be", "we", "for", "from", "that",
                  "this", "are", "was", "were", "which", "who", "but", "would", "may", "should", "must", "can", "will", "has", "have", "had",
                  "all", "their", "with", "than", "could", "been", "not", "our", "your", "such", "only", "when", "after", "before", "these"})
WORD = re.compile(r"[^\W\d_]+(?:['’\-][^\W\d_]+)*")
METHOD = {
    "claim": "A local rewrite may change how much bidirectional prediction of individual content words benefits from the sentence's true left and right neighbors. Compare with sentence-only token predictability.",
    "input": "For each interior native sentence, compare the exact sentence alone with the same sentence flanked by its two native neighbors. Choose up to eight eligible individual whole words at evenly spaced positions. Each word must map exactly to at most four identical tokenizer pieces in both views. Mask the predicted piece and any following pieces of the same word. Read mean log probability across the pieces. At least five paired words are required. No earlier version enters inference.",
    "scores": "Measure target log probability and its change by word. Fit one work-balanced ridge logistic head using sentence-only mean and median log probability and median word length, and a second head adding mean, median and upper-quartile exterior context gain. Include each feature's difference from its median across the file. At least three measured interior sentences to rank the unique top sentence.",
    "point": "Train on 24 previously measured report works. Freeze each arm's point on the second-highest maximum among 35 independent official historical revision pairs: 21 earlier five-sentence pairs and 14 F1000Research longer works represented by a fixed common-anchor window. Require strictly higher response.",
    "fresh": "Fix 20 previously unused GAO test works, excluding known reports and normalized 12-word overlap with every earlier report roster or retained Qwen test source. Each has original plus exact same-position polished sentence from Gemini Flash and GPT Nano. The previous short GAO, eLife, forum and USWDS samples are scored as retrospective transfer only.",
    "comparison": "First six incumbent native three-sentence context responses at the unchanged .50 research operating point. A useful location covers at least half of the edited nonspace characters and .30 intersection over union. All rejected or unassessed cases remain in the denominator.",
    "advance": "At least four additional fresh locations across both generators and three source works, at least two more gains than sentence-only, at least 36 of 40 edits assessed, and at most two new ordinary work groups. An observational response carries zero authorship confidence.",
    "earlier_feasibility": "The single-piece-only procedure stopped after 114 development files, with 15 of 21 ordinary works assessed. Whole-word geometry raised that to 16, still below the unchanged minimum 17. Both trials stopped before an operating point or held scoring. This version adds all 14 available verified F1000Research calibration works, clips each at its first shared exact center sentence, and preserves prior works and rules.",
}


def verify_private(path, role):
    source.require(source.digest(path) == NATIVE[role], f"Retained {role} roster differs")
    return source.read(path)


def get_control_inputs(prior_a, prior_b):
    return [*(r for r in prior_a if r["cohort"] != "report"),
            *({**r, "cohort": "old_a"} for r in prior_a if r["cohort"] == "report"),
            *({**r, "cohort": "old_b"} for r in prior_b if r["cohort"] == "report")]


def extra_revisions(path):
    by_work = defaultdict(dict)
    for item in verify_private(path, "additional_revisions"):
        if item["role"] == "calibration":
            by_work[item["group"]][item["variant"]] = item["text"]
    source.require(len(by_work) == 14 and all(set(v) == {"before", "after"} for v in by_work.values()), "Wrong revision set")
    cases = []
    for group, versions in sorted(by_work.items()):
        before, after = (short.native(versions[side]) for side in ("before", "after"))
        anchors = {unit.text: i for i, unit in enumerate(after[2:-2], 2)}
        matches = [(i, anchors[unit.text]) for i, unit in enumerate(before[2:-2], 2)
                   if unit.text in anchors and sum(v.text == unit.text for v in before) == 1
                   and sum(v.text == unit.text for v in after) == 1]
        source.require(bool(matches), "Calibration source lacked a shared exact center")
        for side, units, target in zip(("before", "after"), (before, after), matches[0], strict=True):
            a, b = units[target - 2].start, units[target + 2].end
            clipped = short.record(versions[side][a:b], "calibration", f"f1000-{group}", side, source_range=[a, b])
            source.require(len(clipped["spans"]) == 5, "Calibration clipping changed geometry")
            cases.append(clipped)
    return cases


def intake(cache, history):
    from scripts import writing_interval_transfer as interval

    source.require(all(source.digest(cache / name) == h for name, h in source.METADATA.items()), "Publisher metadata differs")
    source.require("released under Apache-2.0" in (cache / "README.md").read_text(), "Dataset terms differ")
    tables = {family: interval.read_shard(cache, family, digest)
              for family, digest in (interval.FAMILIES | interval.EXCLUDED_SHARD).items()}
    gem, nano, qwen = tables.values()
    older = {row["group"] for row in history if row["cohort"] in {"fit", "old_a", "old_b"}}
    for ledger in short.known_sources():
        older.update(r["group"] for r in source.read(ledger)["rows"])
    # The previous report scan has covered these source works with other methods.
    exclusions = set(qwen) | {n for n in gem.keys() | nano.keys() if source.sha(n) in older}
    phrase_history = set().union(*(interval.fingerprints(table[n]["v0"]["text"])
                                   for n in exclusions for table in tables.values() if n in table))
    order = sorted((gem.keys() & nano.keys()) - exclusions, key=lambda name: source.sha(f"{SCHEMA}/{name}"))
    cases = []
    admit = Counter(eligible=len(order))
    for name in order:
        if admit["admitted"] == 20:
            break
        admit["examined"] += 1
        group = source.sha(name)
        if gem[name]["v0"]["text"] != nano[name]["v0"]["text"]:
            admit["different_original"] += 1
            continue
        try:
            base, gold, units = short.annotated(gem[name]["v0"])
            source.require(not gold, "Original unexpectedly labeled")
            rewrites = {f: short.annotated(tables[f][name]["v1"]) for f in sorted(GENERATORS)}
        except ValueError:
            admit["geometry"] += 1
            continue
        fingerprints = interval.fingerprints(base)
        if phrase_history & fingerprints:
            admit["prior_phrase"] += 1
            continue
        matched = {f: short.align(v, base, units, (2, 2)) for f, v in rewrites.items()}
        targets = set.intersection(*(set(v) for v in matched.values()))
        if not targets:
            admit["no_shared_edit"] += 1
            continue
        target = min(targets, key=lambda i: source.sha(f"{SCHEMA}/{name}/{i}"))
        a, b = units[target - 2].start, units[target + 2].end
        try:
            options = [short.record(base[a:b], "fresh", group, "original", source_range=[a, b])]
            options.extend(short.record(text, "fresh", group, f, truth, source_range=bounds)
                           for f, (text, truth, bounds) in ((f, m[target]) for f, m in matched.items()))
            source.require(all(len(row["spans"]) == 5 and (not row["truth"] or row["spans"][2] == row["truth"])
                               for row in options), "Clipped focus changed")
        except ValueError:
            admit["clipping"] += 1
            continue
        cases.extend({**row, "publisher_source": source.normalize_title_source(name)} for row in options)
        phrase_history.update(fingerprints)
        admit["admitted"] += 1
    source.require(admit["admitted"] == 20, f"Too few fresh held report works: {dict(admit)}")
    return cases, dict(admit)


def freeze(run, cache, prior_a, prior_b, additional_revisions):
    histories = get_control_inputs(verify_private(prior_a, "old_a"), verify_private(prior_b, "old_b")) + extra_revisions(additional_revisions)
    held, admit = intake(cache, histories)
    roster = histories + held
    for row in roster:
        expected = short.record(row["text"], row["cohort"], row["group"], row["family"], row["truth"])
        source.require(all(row[key] == expected[key] for key in ("text_sha256", "spans", "context", "spaces", "length")),
                       "Native passage geometry changed")
    counts = Counter(row["cohort"] for row in roster)
    source.require(counts == {"fit": 72, "calibration": 70, "forum": 16, "uswds": 12, "elife": 36,
                              "old_a": 60, "old_b": 60, "fresh": 60}, "Incomplete assessment roster")
    source.require(len({r["id"] for r in roster}) == len(roster), "Repeated case")
    group_sets = [{r["group"] for r in roster if r["cohort"] == k} for k in sorted(counts)]
    source.require(sum(map(len, group_sets)) == len(set().union(*group_sets)), "Overlapping case roles")
    source.seal(run / "roster.json", roster)
    lock = {"schema": SCHEMA, "method": METHOD, "runner_sha256": source.digest(Path(__file__)), "revision": REVISION,
            "assets": ASSETS, "publisher_revision": source.REVISION, "primary_revision": BASELINE_REVISION,
            "primary_assets": BASELINE_ASSETS, "old_roster_hashes": NATIVE, "prior": RETAINED,
            "roster_sha256": source.digest(run / "roster.json"), "admission": admit,
            "versions": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "pysbd")}}
    source.seal(run / "freeze.json", lock)
    print(admit, flush=True)


def locked(run):
    lock = source.read(run / "freeze.json")
    source.require(lock["runner_sha256"] == source.digest(Path(__file__)) and lock["roster_sha256"] == source.digest(run / "roster.json"),
                   "Run source or method differs")
    return lock, source.read(run / "roster.json")


def torch_runtime(lock, snapshot, assets):
    import torch

    source.require(all(os.getenv(k) == "1" for k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"))
                   and all(lock["versions"][n] == importlib.metadata.version(n) for n in lock["versions"]), "Use pinned offline runtime")
    source.require(all(source.digest(snapshot / name) == digest for name, digest in assets.items()), "Pinned model files differ")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    return torch


def contiguous_pieces(tokens, start, end):
    selected = [(i, ident) for i, (a, b), ident in tokens if a < end and start < b]
    covered = [(a, b) for _, (a, b), _ in tokens if a < end and start < b]
    if not 1 <= len(selected) <= 4 or covered[0][0] != start or covered[-1][1] != end:
        return None
    if any(b != c for (_, b), (c, _) in zip(covered, covered[1:])):
        return None
    slots, words = zip(*selected, strict=True)
    return (slots, words) if slots == tuple(range(slots[0], slots[-1] + 1)) else None


def features(tokenizer, row, index):
    spans = row["spans"]
    a, b = spans[index]
    target = row["text"][a:b]
    left, right = (row["text"][p:q] for p, q in (spans[index - 1], spans[index + 1]))
    context = f"{left} {target} {right}"
    if context != row["context"][index] or min(len(left.split()), len(right.split())) < 5:
        return None
    encoded = tokenizer([target, context], return_offsets_mapping=True, return_special_tokens_mask=True, truncation=False, verbose=False)
    single, full = encoded["input_ids"]
    if not 10 <= len(single) < len(full) <= 192:
        return None
    token_positions = []
    for offsets, special, ids in zip(encoded["offset_mapping"], encoded["special_tokens_mask"], encoded["input_ids"], strict=True):
        token_positions.append([(i, (a, b), ident) for i, ((a, b), flag, ident) in enumerate(zip(offsets, special, ids, strict=True))
                                if not flag and ident not in tokenizer.all_special_ids and b > a])
    eligible = []
    shift = len(left) + 1
    for word in WORD.finditer(target):
        if word.end() - word.start() < 3 or word.group().casefold() in STOP:
            continue
        lo, hi = word.span()
        small = contiguous_pieces(token_positions[0], lo, hi)
        wide = contiguous_pieces(token_positions[1], lo + shift, hi + shift)
        if small is None or wide is None or small[1] != wide[1]:
            continue
        eligible.append((lo + a, hi + a, small[0], wide[0], small[1]))
    n = min(8, len(eligible))
    if n < 5:
        return None
    positions = [eligible[(2 * i + 1) * len(eligible) // (2 * n)] for i in range(n)]
    return target, context, single, full, positions


def masked_predict(model, tokenizer, requests, torch):
    results = {}
    ordered = sorted(requests, key=lambda key: (len(key[0]), source.sha(str(key))))
    for start in range(0, len(ordered), 32):
        batch = ordered[start:start + 32]
        size = max(len(ids) for ids, _, _, _ in batch)
        samples = torch.full((len(batch), size), tokenizer.pad_token_id, dtype=torch.long)
        attention = torch.zeros_like(samples)
        position, truth = [], []
        for i, (ids, offset, end, actual) in enumerate(batch):
            samples[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
            samples[i, offset:end] = tokenizer.mask_token_id
            attention[i, :len(ids)] = 1
            position.append(offset)
            truth.append(actual)
        with torch.inference_mode():
            x = model.bert(input_ids=samples, attention_mask=attention).last_hidden_state
            logits = model.cls(x[torch.arange(len(batch)), torch.tensor(position)]).float()
            values = torch.log_softmax(logits, dim=-1).gather(1, torch.tensor(truth).unsqueeze(1)).flatten().tolist()
        source.require(all(isfinite(v) and v <= 0 for v in values), "Non-finite masked word likelihood")
        results.update(zip(batch, (round(v, 7) for v in values), strict=True))
        if not start % 1024:
            print("masked requests", len(results), len(ordered), flush=True)
    return results


def measure(run, snapshot, phase):
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    lock, cases = locked(run)
    torch = torch_runtime(lock, snapshot, ASSETS)
    if phase == "held":
        bound = source.read(run / "points.json")
        source.require(bound["freeze_sha256"] == source.digest(run / "freeze.json"), "Operating point was not frozen")
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False, use_fast=True)
    model, info = AutoModelForMaskedLM.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False,
                                                      use_safetensors=True, output_loading_info=True)
    source.require(not info["missing_keys"] and not info["mismatched_keys"] and model.config.hidden_size == 256,
                   f"Trained masked-language head unavailable: {info}")
    source.require(tokenizer.is_fast and tokenizer.mask_token_id == 103, "Pinned mask geometry differs")
    model.eval()
    subset = [r for r in cases if (r["cohort"] in {"fit", "calibration"}) == (phase == "development")]
    pending = {}
    requests = set()
    for row in subset:
        focuses = []
        for index in range(1, len(row["spans"]) - 1):
            geometry = features(tokenizer, row, index)
            if geometry is None:
                continue
            target, context, single, full, positions = geometry
            singles = tuple(single)
            expanded = tuple(full)
            observed = [([(singles, j, a[-1] + 1, w) for j, w in zip(a, ids, strict=True)],
                         [(expanded, j, b[-1] + 1, w) for j, w in zip(b, ids, strict=True)]) for _, _, a, b, ids in positions]
            requests.update(k for pair in observed for view in pair for k in view)
            focuses.append((index, target, context, positions, observed))
        pending[row["id"]] = focuses
    print(phase, len(subset), len(requests), flush=True)
    values = masked_predict(model, tokenizer, requests, torch)
    observations = {}
    for record, focuses in pending.items():
        observations[record] = [{"index": index, "isolated_sha256": source.sha(target), "context_sha256": source.sha(context),
            "words": [[lo, hi, fsum(values[k] for k in solo) / len(solo), fsum(values[k] for k in full) / len(full)]
                      for (lo, hi, _, _, _), (solo, full) in zip(positions, observed, strict=True)]}
            for index, target, context, positions, observed in focuses]
    source.seal(run / f"{phase}.json", {"freeze_sha256": source.digest(run / "freeze.json"), "requests": len(values), "rows": observations})


def averages(positions, arm):
    values = []
    for row in positions:
        words = row["words"]
        isolated = [s for _, _, s, _ in words]
        gain = sorted(f - s for _, _, s, f in words)
        result = [fsum(isolated) / len(words), median(isolated), median(b - a for a, b, _, _ in words)]
        if arm == "exterior":
            result.extend((fsum(gain) / len(words), median(gain), gain[3 * (len(words) - 1) // 4]))
        values.append((row["index"], result))
    if len(values) < 3:
        return []
    centers = [median(data[i] for _, data in values) for i in range(len(values[0][1]))]
    return [(i, data + [x - y for x, y in zip(data, centers, strict=True)]) for i, data in values]


def fit(cases, observations, arm):
    from scripts.writing_interval_transfer import sigmoid

    groups = defaultdict(lambda: defaultdict(list))
    for row in cases:
        if row["cohort"] != "fit":
            continue
        for index, x in averages(observations[row["id"]], arm):
            groups[row["group"]][row["spans"][index] == row["truth"]].append(x)
    usable = [v for v in groups.values() if len(v) == 2]
    source.require(len(usable) >= 16, "Too few complete native training groups")
    samples = [(x, label, 1 / (len(usable) * 2 * len(items))) for group in usable
               for label, items in sorted(group.items()) for x in items]
    n = len(samples[0][0])
    centers = [fsum(w * x[i] for x, _, w in samples) for i in range(n)]
    scales = [max(.05, sqrt(fsum(w * (x[i] - c) ** 2 for x, _, w in samples))) for i, c in enumerate(centers)]
    features = [([1., *((a - b) / s for a, b, s in zip(x, centers, scales, strict=True))], y, w) for x, y, w in samples]
    params = [0.] * (n + 1)
    for _ in range(160):
        error = [(v, w * (sigmoid(fsum(a * b for a, b in zip(v, params, strict=True))) - y)) for v, y, w in features]
        for i in range(n + 1):
            params[i] -= .25 * (fsum(x[i] * e for x, e in error) + (.10 * params[i] if i else 0.))
    return {"centers": centers, "scales": scales, "parameters": params, "works": len(usable), "cases": len(samples)}


def top(positions, arm, model):
    choices = sorted(((round(model["parameters"][0] + fsum((a - b) / c * p for a, b, c, p in
                zip(x, model["centers"], model["scales"], model["parameters"][1:], strict=True)), 9), i)
                    for i, x in averages(positions, arm)), reverse=True)
    return choices[0] if choices and (len(choices) == 1 or choices[0][0] > choices[1][0]) else None


def calibrated(cases, observations):
    models = {}
    for arm in ARMS:
        model = fit(cases, observations, arm)
        maxima = defaultdict(list)
        for row in cases:
            if row["cohort"] == "calibration" and (result := top(observations[row["id"]], arm, model)) is not None:
                maxima[row["group"]].append(result[0])
        source.require(len(maxima) >= 17, "Not enough assessed genuine point works")
        maxima = {group: max(values) for group, values in maxima.items()}
        models[arm] = model | {"maxima": maxima, "point": sorted(maxima.values())[-2]}
    return models


def point(run):
    _, cases = locked(run)
    development = source.read(run / "development.json")
    digest = source.digest(run / "freeze.json")
    source.require(development["freeze_sha256"] == digest, "Different development source")
    models = calibrated(cases, development["rows"])
    source.seal(run / "points.json", {"freeze_sha256": digest, "models": models})
    print("points", {a: m["point"] for a, m in models.items()}, flush=True)


def baseline(run, snapshot):
    from app.detectors.text.classifier import FourClassTextClassifier

    lock, cases = locked(run)
    torch_runtime(lock, snapshot, BASELINE_ASSETS)
    for name, fingerprint in RETAINED.items():
        source.require(source.digest(short.PANELS / name) == fingerprint, "Retained comparator differs")
    a, b = (source.read(short.PANELS / name) for name in RETAINED)
    retained = {r["id"]: r["primary"] for r in a["rows"]} | b["primary"]
    fresh = [r for r in cases if r["cohort"] == "fresh"]
    texts = sorted({t for row in fresh for t in row["context"]}, key=source.sha)
    clf = FourClassTextClassifier.from_snapshot(snapshot, "cpu")
    values = {}
    for start in range(0, len(texts), 48):
        batch = texts[start:start + 48]
        values.update(zip(batch, clf.predict(batch, batch=6, check=lambda _: None), strict=True))
        print("baseline", len(values), len(texts), flush=True)
    rows = {row["id"]: [values[t] for t in row["context"]] if row["cohort"] == "fresh" else retained[row["id"]]
            for row in cases if row["cohort"] not in {"fit", "calibration"}}
    source.seal(run / "primary.json", {"freeze_sha256": source.digest(run / "freeze.json"), "requests": len(values), "rows": rows})


def evaluate(cases, scores, primary, model):
    metrics = defaultdict(lambda: defaultdict(Counter))
    extra = defaultdict(lambda: defaultdict(list))
    alarms = defaultdict(lambda: defaultdict(set))
    decisions = []
    for row in cases:
        cohort = row["cohort"]
        if cohort in {"fit", "calibration"}:
            continue
        spans = row["spans"]
        incumbent = primary[row["id"]]
        source.require(len(incumbent) == len(spans) and all(isfinite(p) and 0 <= p <= 1 for p in incumbent), "Invalid incumbent evidence")
        existing = [[spans[max(0, i - 1)][0], spans[min(len(spans) - 1, i + 1)][1]] for i, p in enumerate(incumbent) if p >= .5][:6]
        result = {"existing": existing}
        spaces = set(row["spaces"])
        def useful(ranges, truth=row["truth"], ignored=spaces):
            return any(short.useful(v, truth, ignored) for v in ranges)
        old_hit = useful(existing)
        assessment = {"existing": True}
        for arm in ARMS:
            peak = top(scores[row["id"]], arm, model[arm])
            assessment[arm] = peak is not None
            chosen = [spans[peak[1]]] if peak is not None and peak[0] > model[arm]["point"] else []
            result[arm] = chosen
            result[f"plus_{arm}"] = existing + chosen
            assessment[f"plus_{arm}"] = True
            if useful(chosen) and not old_hit:
                extra[arm][cohort].append((row["group"], row["family"]))
            if row["truth"] is None and chosen and not existing:
                alarms[arm][cohort].add(row["group"])
        for arm, ranges in result.items():
            hit = useful(ranges)
            metrics[f"{cohort}/{row['family']}"][arm].update(files=1, assessed=assessment[arm], flagged=bool(ranges),
                         useful=hit, wrong=bool(row["truth"] and ranges and not hit))
        decisions.append({"id": row["id"], "locations": result})
    recovered = extra["exterior"]["fresh"]
    scores_without = extra["sentence_only"]["fresh"]
    passed = (len(recovered) >= max(4, len(scores_without) + 2) and {f for _, f in recovered} == GENERATORS
              and len({g for g, _ in recovered}) >= 3 and len(alarms["exterior"]["fresh"]) <= 2 and
              sum(metrics[f"fresh/{f}"]["exterior"]["assessed"] for f in GENERATORS) >= 36)
    return {"metrics": metrics, "extra": {a: {p: sorted([list(x) for x in rows]) for p, rows in groups.items()} for a, groups in extra.items()},
            "alarms": {a: {p: sorted(g) for p, g in groups.items()} for a, groups in alarms.items()}, "passes": passed, "decisions": decisions}


def replay(data):
    lock, cases = data["freeze"], data["roster"]
    source.require(lock["method"] == METHOD and lock["assets"] == ASSETS and lock["schema"] == SCHEMA, "Wrong frozen measurement")
    observations = data["development"] | data["held"]
    source.require(set(observations) == {row["id"] for row in cases}, "Missing native cases")
    for row in cases:
        checked = observations[row["id"]]
        indices = [entry["index"] for entry in checked]
        source.require(indices == sorted(set(indices)), "Duplicated sentence measurement")
        for entry in checked:
            index = entry["index"]
            source.require(0 < index < len(row["spans"]) - 1 and 5 <= len(entry["words"]) <= 8, "Unsupported masked measurement")
            start, end = row["spans"][index]
            source.require(all(start <= a < b <= end and all(isfinite(v) and v <= 0 for v in (single, full))
                               for a, b, single, full in entry["words"]), "Masked values did not match native bounds")
    fresh = calibrated(cases, data["development"])
    source.require(fresh == data["points"], "Frozen models or points differ")
    actual = evaluate(cases, data["held"], data["primary"], fresh)
    source.require(actual == data["result"], "Recomputed results differ")
    return actual


def publish(run, output):
    lock, rows = locked(run)
    tables = {name: source.read(run / f"{name}.json") for name in ("development", "held", "primary", "points")}
    source.require(all(table["freeze_sha256"] == source.digest(run / "freeze.json") for table in tables.values()), "Inputs differ from lock")
    result = {"freeze": lock, "roster": [{k: v for k, v in row.items() if k not in {"text", "context"}} for row in rows],
              "points": tables["points"]["models"], **{k: tables[k]["rows"] for k in ("development", "held", "primary")},
              "requests": {k: tables[k]["requests"] for k in ("development", "held", "primary")}}
    result["result"] = evaluate(result["roster"], result["held"], result["primary"], result["points"])
    replay(result)
    source.seal(output, result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "development", "point", "held", "baseline", "publish", "review"))
    parser.add_argument("run", type=Path)
    for key in ("cache", "prior-a", "prior-b", "additional-revisions", "snapshot", "output"):
        parser.add_argument(f"--{key}", type=Path)
    args = parser.parse_args()
    if args.phase == "freeze":
        freeze(args.run, args.cache, args.prior_a, args.prior_b, args.additional_revisions)
    elif args.phase in {"development", "held"}:
        measure(args.run, args.snapshot, args.phase)
    elif args.phase == "point":
        point(args.run)
    elif args.phase == "baseline":
        baseline(args.run, args.snapshot)
    elif args.phase == "publish":
        publish(args.run, args.output)
    elif args.phase == "review":
        values = replay(source.read(args.run))
        print(source.serialize({k: v for k, v in values.items() if k != "decisions"}), end="")


if __name__ == "__main__":
    main()
