"""Compare localized writing checks on externally authored revision trajectories."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "chipmunk.writing-revision-trajectory/1"
REVISION = "072c03c7051409346aaba05168d183cf6d863ee9"
BASE_URL = f"https://huggingface.co/datasets/OpAI-Bench1/OpAI-Bench/resolve/{REVISION}"
METADATA = {"README.md": "42905a7a65e6a6aa24ef6f00c07c15f0690ca40cc6b07b1874762e331a65d379",
            "croissant.json": "70fda4e6335688de89b47fef7d365b1922185c58039f9134b58f6e49ce118cce"}
SHARDS = {"gemini-2.5-flash": "e4a52b8cd3a56ac9d748d101f9d1ea75e6ae13b6eb6c38dc95e41f64488c45b5",
          "qwen3-8b": "23e2b29270910b2a54d1ae2859e9e2b1e89cee1dd179f16d2ea2b39c7e60af25"}
STAGES = {"v1": "polish", "v2": "paraphrase", "v4": "compress"}
ARMS = ("isolated", "contiguous")
MAX_CONTEXTS = 6
FIT_GROUPS = 16
HELD_GROUPS = 20
PRIOR = ROOT / "docs/evaluation-panels/writing-multiblock-independent-2026-11-12.json"
PYTHON_BINDINGS = [
    "backend/app/detectors/text/classifier.py", "backend/app/detectors/text/extract.py",
    "backend/app/detectors/text/segment.py", "backend/app/detectors/text/coverage.py",
    "backend/app/detectors/text/pipeline.py", "backend/scripts/score_text_upgrade.py",
    "backend/scripts/writing_focus_v3.py", "backend/scripts/writing_body_localization_transfer.py",
]


def read(path):
    return json.loads(path.read_bytes())


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def sha(value):
    return hashlib.sha256(value.encode()).hexdigest()


def serialize(value):
    return json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"


def seal(path, value):
    with path.open("x", encoding="utf-8") as stream:
        stream.write(serialize(value))


def require(test, message):
    if not test:
        raise ValueError(message)


def ranges_from_tags(tagged):
    parts, spans = [], []
    offset = 0
    opened = None
    for piece in re.split(r"(<AI_Start>|</AI_End>)", tagged):
        if piece == "<AI_Start>":
            require(opened is None, "Nested generator annotation")
            opened = offset
        elif piece == "</AI_End>":
            require(opened is not None and opened < offset, "Empty or unpaired generator annotation")
            spans.append([opened, offset])
            opened = None
        else:
            parts.append(piece)
            offset += len(piece)
    require(opened is None, "Unclosed generator annotation")
    return "".join(parts), spans


def publisher_case(row):
    text = row["text"]
    tagged_text, tagged_spans = ranges_from_tags(row["text_tagged"])
    gold = json.loads(row["ai_spans_char"])
    require(text == tagged_text and gold == tagged_spans, "Publisher character labels differ from tagged text")
    sentences, labels = json.loads(row["sentences"]), json.loads(row["sentence_labels"])
    require(len(sentences) == len(labels) and all(type(v) is int and v in (0, 1) for v in labels),
            "Incomplete publisher sentence labels")
    spans, last = [], 0
    for sentence, label in zip(sentences, labels, strict=True):
        start = text.find(sentence, last)
        require(sentence and start >= last and not text[last:start].strip(), "Publisher sentence offset differs")
        last = start + len(sentence)
        if label:
            spans.append([start, last])
    require(not text[last:].strip() and spans == gold, "Publisher sentence and span labels disagree")
    return text, gold


def open_rows(cache):
    for name, expected in METADATA.items():
        require(digest(cache / name) == expected, f"Changed publisher metadata: {name}")
    card = (cache / "README.md").read_text()
    require("license: apache-2.0" in card and "released under Apache-2.0" in card, "Unexpected dataset terms")
    metadata = {row.get("name"): row.get("sha256") for row in read(cache / "croissant.json")["distribution"]}
    csv.field_size_limit(1024 * 1024)
    all_rows = {}
    for family, expected in SHARDS.items():
        name = f"reports_{family}.csv"
        require(digest(cache / name) == expected == metadata[f"default/test/{name}"], "Publisher shard digest differs")
        by_group = defaultdict(dict)
        with (cache / name).open(encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                require((row["domain"], row["generator"], row["split"], row["subset"]) == ("reports", family, "test", "main"),
                        "Wrong publisher split or generator")
                group, version = row["id"], row["version"]
                require(re.fullmatch(r"GAO_[A-Z]+-[0-9A-Z-]+", group) is not None, "Unknown report provenance")
                require(version not in by_group[group], "Duplicate publisher revision")
                if version == "v0" or version in STAGES:
                    require(row["edit_operation"] == STAGES.get(version, "none"), "Unexpected publisher operation")
                    by_group[group][version] = row
        require(all(set(rows) == {"v0", *STAGES} for rows in by_group.values()), "Incomplete publisher trajectory")
        all_rows[family] = by_group
    return all_rows


def normalize_title_source(report_id):
    match = re.fullmatch(r"GAO_([A-Z]+-[0-9A-Z-]+)", report_id)
    return f"https://www.gao.gov/products/{match[1].lower()}"


def source_geometry(row):
    from app.detectors.text.extract import from_string
    from app.detectors.text.pipeline import windows
    from app.detectors.text.segment import segment

    text, truth = publisher_case(row)
    doc = from_string(text)
    if doc.text != text:
        return None
    sentences = segment(doc)
    if not 6 <= len(sentences) <= 32 or not 100 <= sum(s.n_words for s in sentences) <= 700:
        return None
    nonspace = {i for i, char in enumerate(text) if not char.isspace()}
    gold_pixels = [nonspace.intersection(range(start, end)) for start, end in truth]
    gold_characters = [len(span) for span in gold_pixels]
    ranges = []
    for s in sentences:
        own = nonspace.intersection(range(s.start, s.end))
        overlaps = [len(own.intersection(span)) for span in gold_pixels]
        ranges.append({"start": s.start, "end": s.end, "paragraph": s.paragraph,
                       "characters": len(own), "ai_characters": sum(overlaps), "gold_overlaps": overlaps})
    require(sum(r["characters"] for r in ranges) == len(nonspace)
            and sum(r["ai_characters"] for r in ranges) == sum(gold_characters), "Unmapped characters or edit positions")
    return {"text": text, "text_sha256": sha(text), "ranges": ranges, "gold": truth,
            "gold_characters": gold_characters, "truth_characters": sum(gold_characters),
            "context": windows(sentences, 3), "isolated": [s.text for s in sentences]}


def rank(value):
    return sha(f"chipmunk-opai-2026-11-13/{value}")


def select_roster(cache):
    gen_a, gen_b = SHARDS
    shards = open_rows(cache)
    common = set(shards[gen_a]) & set(shards[gen_b])
    require(len(common) == 30, "Changed shared publisher sources")
    admissions = Counter()
    roster = []
    for role, groups, cap in (("held", common, HELD_GROUPS), ("fit", set(shards[gen_a]) - common, FIT_GROUPS)):
        for group in sorted(groups, key=rank):
            if admissions[f"{role}/admitted"] == cap:
                break
            admissions[f"{role}/attempted"] += 1
            human = shards[gen_a][group]["v0"]
            candidates = [("original", "v0", human)]
            if role == "held":
                require(shards[gen_b][group]["v0"]["text"] == human["text"], "Mismatched matched original")
                candidates += [(family, v, shards[family][group][v]) for family in SHARDS for v in STAGES]
            cases = [(family, version, row, source_geometry(row)) for family, version, row in candidates]
            if any(case is None for _, _, _, case in cases):
                admissions[f"{role}/unsupported_geometry"] += 1
                continue
            require(cases[0][-1]["truth_characters"] == 0
                    and all(case["truth_characters"] > 0 for _, _, _, case in cases[1:]), "Invalid original/edit truth")
            admissions[f"{role}/admitted"] += 1
            for family, version, row, case in cases:
                roster.append({"role": role, "group": sha(group), "family": family, "version": version,
                               "source": normalize_title_source(group), "record_id": row["record_id"], **case})
    require(admissions["fit/admitted"] == FIT_GROUPS and admissions["held/admitted"] >= 14, "Insufficient exact held trajectories")
    original_hashes = [r["text_sha256"] for r in roster if r["family"] == "original"]
    require(len(set(original_hashes)) == len(original_hashes), "Duplicate source text across selected reports")
    return roster, dict(admissions)


def freeze(run, cache, primary, second):
    from scripts.writing_body_localization_transfer import SECONDARY_ASSETS
    from scripts.writing_focus_v3 import BASELINE_ASSETS, BASELINE_REVISION

    snapshots = ((primary, BASELINE_ASSETS), (second, SECONDARY_ASSETS))
    require(all(digest(path / name) == expected for path, assets in snapshots for name, expected in assets.items()),
            "Pinned checkpoint files differ")
    require(primary.name == BASELINE_REVISION and second.name == "a6efd4bf6ecd852f332b2569ae0f98cae13b7985",
            "Unexpected checkpoint revision")
    roster, admissions = select_roster(cache)
    seal(run / "roster.json", roster)
    seal(run / "freeze.json", {"schema": SCHEMA, "revision": REVISION, "base_url": BASE_URL,
         "metadata": METADATA, "shards": SHARDS, "runner_sha256": digest(Path(__file__)),
         "native_support_sha256": {name: digest(ROOT / name) for name in PYTHON_BINDINGS},
         "roster_sha256": digest(run / "roster.json"), "admissions": admissions,
         "prior_sha256": digest(PRIOR), "prior_isolated_odds": read(PRIOR)["points"]["independent_focus"],
         "checkpoint_revisions": {"primary": primary.name, "independent": second.name},
         "checkpoints": {"primary": digest(primary / "model.safetensors"), "independent": digest(second / "model.safetensors")},
         "runtime": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "pysbd")},
         "method": {"baseline": "First six 3-sentence contexts with raw response at least 0.50",
                    "isolated": "Max independent exact-sentence log odds, sentences with 60 or more nonspace characters",
                    "contiguous": "Strongest 1 to 3 consecutive eligible sentences in one paragraph, all positive log odds; mean odds plus weakest odds times 0.25 per added sentence",
                    "fit": "Second-highest untouched source maximum for each arm, separate from held source IDs; also measure unchanged prior isolated point",
                    "hit": "At least 0.30 nonspace IoU with all authored spans and 0.50 edited-character recall within one highlighted range; additionally measure any half-covered gold span and union character recall",
                    "decision": "At least three held works with a newly half-covered gold span across both generators, at most two additional untouched works, no new conclusion or confidence"}})


def local_score(run, primary, second, role, model, device):
    import torch

    from scripts.score_text_upgrade import sequence_predictor
    from scripts.writing_body_localization_transfer import SECONDARY_ASSETS
    from scripts.writing_focus_v3 import BASELINE_ASSETS

    lock = read(run / "freeze.json")
    require(all(os.getenv(k) == "1" for k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")), "Offline model runtime required")
    require(lock["runner_sha256"] == digest(Path(__file__)) and lock["roster_sha256"] == digest(run / "roster.json")
            and all(digest(ROOT / name) == expected for name, expected in lock["native_support_sha256"].items())
            and lock["runtime"] == {name: importlib.metadata.version(name) for name in lock["runtime"]},
            "Frozen runner, input, or runtime changed")
    path, assets, view, name = ((primary, BASELINE_ASSETS, "context", "tropa") if model == "primary"
                               else (second, SECONDARY_ASSETS, "isolated", "modernbert"))
    require(path.name == lock["checkpoint_revisions"][model]
            and all(digest(path / k) == expected for k, expected in assets.items()), "Changed checkpoint assets")
    if role == "held":
        require((run / "points.json").exists(), "Lock genuine-reference points before held scoring")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    if device == "mps":
        require(torch.backends.mps.is_available(), "Apple GPU unavailable")
    rows = [row for row in read(run / "roster.json") if row["role"] == role]
    requests = list(dict.fromkeys(text for row in rows for text in row[view]))
    predictor = sequence_predictor(name, path, device, batch=8, max_len=512)
    scored = {}
    for start in range(0, len(requests), 64):
        batch = requests[start:start + 64]
        values = predictor(batch)
        require(all(math.isfinite(score) and 0 <= score <= 1 for score in values), "Invalid score")
        scored.update(zip(batch, values, strict=True))
        print(role, model, len(scored), "/", len(requests), flush=True)
    seal(run / f"{role}-{model}.json", {"freeze_sha256": digest(run / "freeze.json"), "device": device,
         "inputs": len(requests), "cases": {row["record_id"]: [[sha(s), scored[s]] for s in row[view]] for row in rows}})


def odds(p):
    p = min(1 - 1e-7, max(1e-7, p))
    return math.log(p / (1 - p))


def proposal(row, arm):
    values = [odds(v[1]) for v in row["independent"]]
    spans = row["ranges"]
    ranked = []
    for left, score in enumerate(values):
        if spans[left]["characters"] < 60:
            continue
        ranked.append((score, left, left + 1))
        if arm == "contiguous" and score > 0:
            for right in range(left + 1, min(left + 3, len(values))):
                if (spans[right]["characters"] < 60 or spans[right]["paragraph"] != spans[left]["paragraph"]
                        or values[right] <= 0):
                    break
                block = values[left:right + 1]
                ranked.append((math.fsum(block) / len(block) + 0.25 * (len(block) - 1) * min(block), left, right + 1))
    ranked.sort(reverse=True)
    if ranked and (len(ranked) == 1 or ranked[0][0] > ranked[1][0]):
        return ranked[0]
    return None


def points(rows):
    fit = [row for row in rows if row["role"] == "fit"]
    require(len({r["group"] for r in fit}) == len(fit) == FIT_GROUPS and all(not r["gold"] for r in fit),
            "Genuine-reference groups are incomplete")
    maxima = {arm: {r["group"]: p[0] for r in fit if (p := proposal(r, arm))} for arm in ARMS}
    require(all(len(arm) >= FIT_GROUPS - 1 for arm in maxima.values()), "Insufficient complete maxima")
    return {"maxima": maxima, "cutoffs": {arm: sorted(v.values(), reverse=True)[1] for arm, v in maxima.items()}}


def rows_with_scores(run, role):
    roster = [row for row in read(run / "roster.json") if row["role"] == role]
    data = {name: read(run / f"{role}-{name}.json") for name in ("primary", "independent")}
    require(all(v["freeze_sha256"] == digest(run / "freeze.json") for v in data.values()), "Unbound score output")
    public = []
    for row in roster:
        result = {key: v for key, v in row.items() if key not in ("text", "context", "isolated")}
        for name, view in (("primary", "context"), ("independent", "isolated")):
            scores = data[name]["cases"][row["record_id"]]
            require([s[0] for s in scores] == [sha(s) for s in row[view]], "Scored view differs from frozen inputs")
            result[name] = scores
        public.append(result)
    return public, {name: {key: v[key] for key in ("device", "inputs")} for name, v in data.items()}


def outcome(row, intervals):
    spans = row["ranges"]
    hits = []
    for left, right in intervals:
        selected = spans[left:right]
        common = sum(s["ai_characters"] for s in selected)
        union = row["truth_characters"] + sum(s["characters"] for s in selected) - common
        hits.append(common > 0 and 2 * common >= row["truth_characters"] and 10 * common >= 3 * union)
    assessed = {i for left, right in intervals for i in range(left, right)}
    selected = [spans[i] for i in sorted(assessed)]
    # Each native sentence stores exact overlap for every gold span. This also
    # handles publisher spans split or merged by the app's sentence segmenter.
    target_hits = [sum(s["gold_overlaps"][j] for s in selected) * 2 >= size
                   for j, size in enumerate(row["gold_characters"])]
    return {"flag": bool(intervals), "useful_document_location": any(hits),
            "useful_individual_span": any(target_hits), "gold_spans_half_covered": sum(target_hits),
            "highlighted_ai_characters": sum(s["ai_characters"] for s in selected),
            "highlighted_ordinary_characters": sum(s["characters"] - s["ai_characters"] for s in selected),
            "ranges": [[spans[a]["start"], spans[b - 1]["end"]] for a, b in intervals]}


def all_outcomes(rows, fitted, prior):
    details = {}
    groups = defaultdict(list)
    for row in rows:
        if row["role"] == "fit":
            continue
        n = len(row["ranges"])
        baseline = [(max(0, i - 1), min(n, i + 2)) for i, (_, p) in enumerate(row["primary"]) if p >= 0.5][:MAX_CONTEXTS]
        interval_sets = {"current": baseline}
        cuts = {**fitted["cutoffs"], "isolated_prior": prior}
        for arm, point in cuts.items():
            top = proposal(row, "isolated" if arm == "isolated_prior" else arm)
            interval_sets[arm] = [(top[1], top[2])] if top and top[0] > point else []
            interval_sets[f"current_plus_{arm}"] = baseline + interval_sets[arm]
        results = {arm: outcome(row, ranges) for arm, ranges in interval_sets.items()}
        details[row["record_id"]] = results
        groups[f"{row['family']}/{row['version']}"].append((row, results))
    summary = {}
    for stratum, cases in sorted(groups.items()):
        metrics = ["flag", "useful_document_location", "useful_individual_span", "gold_spans_half_covered",
                   "highlighted_ai_characters", "highlighted_ordinary_characters"]
        summary[stratum] = {"files": len(cases), "gold_spans": sum(len(r["gold"]) for r, _ in cases),
             "gold_characters": sum(r["truth_characters"] for r, _ in cases),
             "methods": {arm: {k: sum(arms[arm][k] for _, arms in cases) for k in metrics}
                         for arm in cases[0][1]}}
    held = {row["record_id"]: row for row in rows if row["role"] == "held"}
    advances = {}
    for arm in (*ARMS, "isolated_prior"):
        new = {k: held[k] for k, arms in details.items() if arms[arm]["useful_individual_span"]
               and not arms["current"]["useful_individual_span"]}
        ordinary = sum(arms[arm]["flag"] and not arms["current"]["flag"] for _, arms in groups["original/v0"])
        works, families = sorted({r["group"] for r in new.values()}), sorted({r["family"] for r in new.values()})
        advances[arm] = {"additional_files": sorted(new), "additional_works": works, "families": families,
                         "additional_ordinary_works": ordinary,
                         "passed": len(works) >= 3 and len(families) == len(SHARDS) and ordinary <= 2}
    return {"summary": summary, "advance": advances, "cases": details}


def verify_receipt(data):
    require(data["schema"] == SCHEMA and data["freeze_sha256"] == sha(serialize(data["freeze"])), "Invalid frozen metadata")
    lock = data["freeze"]
    require(lock["runner_sha256"] in {digest(Path(__file__)), "6c5d588c518e6a8fb8f5d89c6b2579c21579157861957150e54025059216dfcc"}
            and lock["prior_sha256"] == digest(PRIOR), "Code or comparison changed")
    rows = data["rows"]
    require(len({r["record_id"] for r in rows}) == len(rows), "Duplicate scored record")
    by_role = {role: {r["group"] for r in rows if r["role"] == role} for role in ("fit", "held")}
    require(not (by_role["fit"] & by_role["held"]) and len(by_role["held"]) == lock["admissions"]["held/admitted"], "Group roles overlap")
    for r in rows:
        require(len(r["ranges"]) == len(r["primary"]) == len(r["independent"])
                and sum(r["gold_characters"]) == r["truth_characters"] and all(r["gold_characters"]), "Incomplete source map")
        end = 0
        for s in r["ranges"]:
            require(end <= s["start"] < s["end"] and 0 <= s["ai_characters"] <= s["characters"] <= s["end"] - s["start"]
                    and len(s["gold_overlaps"]) == len(r["gold_characters"])
                    and sum(s["gold_overlaps"]) == s["ai_characters"] and all(v >= 0 for v in s["gold_overlaps"]), "Invalid source geometry")
            end = s["end"]
        require(all(sum(s["gold_overlaps"][j] for s in r["ranges"]) == n for j, n in enumerate(r["gold_characters"])), "Edit coverage changed")
        require(all(re.fullmatch(r"[a-f0-9]{64}", d) and math.isfinite(p) and 0 <= p <= 1
                    for model in ("primary", "independent") for d, p in r[model]), "Invalid saved response")
    fitted = points(rows)
    require(fitted == data["points"]["values"] and data["points"]["freeze_sha256"] == data["freeze_sha256"], "Genuine fit differs")
    expected = all_outcomes(rows, fitted, lock["prior_isolated_odds"])
    require(expected == data["results"], "Published outcomes differ")
    return {"rows": len(rows), "admission": lock["admissions"], "cutoffs": fitted["cutoffs"], **{k: expected[k] for k in ("summary", "advance")}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "score", "fit", "publish", "verify"))
    parser.add_argument("path", type=Path)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--primary", type=Path)
    parser.add_argument("--second", type=Path)
    parser.add_argument("--role", choices=("fit", "held"))
    parser.add_argument("--model", choices=("primary", "independent"))
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    parser.add_argument("--publication", type=Path)
    args = parser.parse_args()
    run = args.path
    if args.phase == "freeze":
        freeze(run, args.cache, args.primary, args.second)
    elif args.phase == "score":
        local_score(run, args.primary, args.second, args.role, args.model, args.device)
    elif args.phase == "fit":
        rows, _ = rows_with_scores(run, "fit")
        seal(run / "points.json", {"freeze_sha256": digest(run / "freeze.json"), "values": points(rows)})
    elif args.phase == "publish":
        fit, dev = rows_with_scores(run, "fit")
        held, test = rows_with_scores(run, "held")
        locked, fitted = read(run / "freeze.json"), read(run / "points.json")
        data = {"schema": SCHEMA, "freeze": locked, "freeze_sha256": digest(run / "freeze.json"), "points": fitted,
                "scoring": {"fit": dev, "held": test}, "rows": fit + held,
                "results": all_outcomes(held, fitted["values"], locked["prior_isolated_odds"])}
        verify_receipt(data)
        seal(args.publication, data)
    else:
        print(serialize(verify_receipt(read(run))), end="")


if __name__ == "__main__":
    main()
