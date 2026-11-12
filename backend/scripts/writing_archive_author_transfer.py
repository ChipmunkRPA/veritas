"""Compare the frozen single-file writing ranker on two archived prose sources."""

from __future__ import annotations

import argparse
import os
from collections import Counter, defaultdict
from pathlib import Path

from scripts import writing_paired_revision_replication as generation
from scripts.video_appearance_native_v108 import read, require, write
from scripts.writing_blind_rewrite_probe import generator_configuration, text_hash, valid_sentence
from scripts.writing_center_deletion_transfer import odds, score, useful
from scripts.writing_focus_v3 import BASELINE_ASSETS, digest

SCHEMA = "chipmunk.writing-archive-author-transfer/1"
SOURCE_SCHEMA = "chipmunk.writing-archive-prospective-intake/1"
ROOT = Path(__file__).resolve().parents[2]
PEP_REPORT = ROOT / "docs/evaluation-panels/writing-pep-editorial-intake-2026-11-11.json"
PEP_SCORES = ROOT / "backend/data/independent-evaluation/writing-pep-prose-diff-extension-2026-11-11/scored"
PEP_POINTS_SHA256 = "a7d4ab9eb612b8d80cca92965dec671b47ace4ce742cc0aa9ac4f54db78360ee"
PEP_DEVELOPMENT_SHA256 = "64abf6a6ed98718e3c0bd478a7be2a5596679dd98af39874d637bc01251dc2ad"
POINT = -0.8544108248390134
ARMS = ("current", "matched_context", "supplement")
MEASUREMENT_CODE = (
    "backend/scripts/writing_center_deletion_transfer.py",
    "backend/scripts/writing_paired_email_revision.py",
    "backend/app/detectors/text/classifier.py",
    "backend/app/detectors/text/coverage.py",
    "backend/app/detectors/text/extract.py",
    "backend/app/detectors/text/segment.py",
)
METHOD = {
    "scope": "Exact five-sentence plain snippets. Measure only the single supplied version. Both genuine controls and each admitted generated edit are separate runs.",
    "current": "First six native windows with raw score at least 0.5. This is the incumbent localized window comparison.",
    "matched_context": "Among the eligible 12–40-word interior sentences, use the uniquely largest native log odds if strictly greater than the frozen PEP transfer operating point.",
    "supplement": "Union of incumbent windows and the exact matched-context sentence, with per-arm reporting.",
    "generation": "One offline attempt per source and pinned generator. Keep a complete genuine rewrite with the other four sentences identical. No retries or replacement of failed attempts.",
    "localization": "At least 0.30 nonwhitespace character intersection over union and 0.50 recall of the known changed sentence.",
    "advance": "At least six admitted outputs and one additional useful group per generator, three distinct new useful groups overall, and at most one additional genuine later false-flag group. Treat each underlying source as one group, regardless of generator.",
    "change": "Study only, zero contribution to authorship confidence. The prior PEP operating point is fixed and the archive sources are never used for fitting.",
}


def protocol(source, snapshot):
    from scripts.text_elife_rewrites_v273 import GENERATORS, MLX_VERSIONS

    receipt = read(source / "source-receipt.json")
    require(receipt["schema"] == SOURCE_SCHEMA and receipt["originals_sha256"] == digest(source / "originals.json")
            and receipt["qualified_groups"] >= 6, "No adequate frozen source roster")
    points = PEP_SCORES / "operating-points.json"
    development = PEP_SCORES / "calibration.json"
    published = read(PEP_REPORT)["trial_commitments"]
    require(published["operating-points.json"] == digest(points) == PEP_POINTS_SHA256
            and read(points)["cutoff"]["matched_context"] == POINT
            and digest(development) == PEP_DEVELOPMENT_SHA256, "Prior cutoff or local input inventory differs")
    config = generator_configuration()
    for key, entry in GENERATORS.items():
        require(config[key]["license"] == "Apache-2.0" and digest(entry[3] / "model.safetensors") == entry[2],
                "Pinned generation asset differs")
    require(all(digest(snapshot / key) == value for key, value in BASELINE_ASSETS.items()), "Pinned checkpoint differs")
    return {"schema": SCHEMA, "method": METHOD, "code_sha256": digest(Path(__file__)),
            "prior_points_sha256": PEP_POINTS_SHA256, "matched_log_odds_point": POINT,
            "prior_development_sha256": digest(development),
            "source_sha256": receipt["originals_sha256"], "source_receipt_sha256": digest(source / "source-receipt.json"),
            "current_assets": BASELINE_ASSETS, "generator_assets": config, "seeds": generation.SEEDS,
            "prompt": generation.EDIT_PROMPT, "settings": generation.SETTINGS, "runtime": MLX_VERSIONS,
            "measurement_code_sha256": {name: digest(ROOT / name) for name in MEASUREMENT_CODE},
            "generation_impl_sha256": digest(Path(generation.__file__))}


def inputs(source):
    from app.detectors.text.extract import from_string
    from app.detectors.text.segment import segment
    from scripts.writing_blind_rewrite_probe import words
    from scripts.writing_paired_email_revision import encoded, native_contexts

    originals = read(source / "originals.json")
    attempts = {family: read(source / f"generated-{family}.json") for family in generation.SEEDS}
    freeze = digest(source / "freeze.json")
    groups = {item["group"] for item in originals}
    require(len(groups) == len(originals) and all(value["freeze_sha256"] == freeze and value["results"].keys() == groups
                for value in attempts.values()), "Frozen source or exact generation roster differs")
    rows, values, dispositions = [], {}, defaultdict(Counter)
    for item in originals:
        group = item["group"]
        before, after = item["before"], item["after"]
        old, known = before["focus_text"], after["focus_text"]
        start, end = before["focus"]
        before_units = segment(from_string(before["text"]))
        flanks = [unit.text for i, unit in enumerate(before_units) if i != 2]
        variations = {"before": (before["text"], before["focus"]), "after": (after["text"], after["focus"])}
        for family, packet in attempts.items():
            attempt = packet["results"][group]
            require(attempt["input_sha256"] == text_hash(old), "Generator saw an unexpected source input")
            dispositions[family]["attempted"] += 1
            revised = valid_sentence(attempt, old, generated=True)
            if revised is None or revised == known:
                dispositions[family]["failed_admission"] += 1
                continue
            text = before["text"][:start] + revised + before["text"][end:]
            doc = from_string(text)
            units = segment(doc)
            if (len(units) != 5 or doc.text != text or units[2].text != revised
                    or [unit.text for i, unit in enumerate(units) if i != 2] != flanks):
                dispositions[family]["changed_native_geometry"] += 1
                continue
            variations[family] = (text, [start, start + len(revised)])
            dispositions[family]["admitted"] += 1
        for role, (text, focus) in variations.items():
            doc = from_string(text)
            units = segment(doc)
            require(len(units) == 5 and doc.text == text and [units[2].start, units[2].end] == focus,
                    "Changed sentence offsets differ from native extraction")
            windows = []
            for unit, context in zip(units, native_contexts(units), strict=True):
                entry, value = encoded(context)
                values[entry["sha256"]] = value
                windows.append(entry | {"focus": [unit.start, unit.end]})
            rows.append({"group": group, "archive": item["archive"], "genre": item["source"]["genre"],
                         "recorded_authorship": item["recorded_authorship"], "role": role, "text": text,
                         "text_sha256": text_hash(text), "focus": focus, "windows": windows,
                         "centers": [i for i in (1, 2, 3) if 12 <= len(words(units[i].text)) <= 40]})
    return rows, values, {family: dict(counts) for family, counts in dispositions.items()}


def outcomes(rows, responses):
    result = []
    for row in rows:
        windows, text = row["windows"], row["text"]
        scores = [responses[entry["sha256"]]["score"] for entry in windows]
        baseline = [entry["spans"] for entry, value in zip(windows, scores, strict=True) if value >= .5][:6]
        ranked = sorted(((odds(scores[i]), i) for i in row["centers"]), reverse=True)
        winner = ranked[0][1] if ranked and (len(ranked) == 1 or ranked[0][0] > ranked[1][0]) else None
        candidate = [[windows[winner]["focus"]]] if winner is not None and ranked[0][0] > POINT else []
        selected = {"current": baseline, "matched_context": candidate, "supplement": baseline + candidate}
        truth = {i for i in range(*row["focus"]) if not text[i].isspace()}
        result.append({key: row[key] for key in ("group", "archive", "genre", "recorded_authorship", "role",
                                                "text_sha256", "focus")} | {
            "ranked_exact_center_first": winner == 2,
            "arms": {arm: {"flag": bool(items), "useful": any(useful(spans, truth, text) for spans in items),
                           "spans": items} for arm, items in selected.items()},
        })
    return result


def summary(rows, admission):
    def aggregate(files):
        return {"files": len(files), "ranked_exact_center_first": sum(item["ranked_exact_center_first"] for item in files),
                **{arm: dict(Counter(
                    "useful" if item["arms"][arm]["useful"] else "wrong_only" if item["arms"][arm]["flag"] else "no_flag"
                    for item in files)) | {"flag": sum(item["arms"][arm]["flag"] for item in files)} for arm in ARMS}}

    def breakdown(items):
        return {role: aggregate([item for item in items if item["role"] == role])
                for role in ("before", "after", *generation.SEEDS)}

    generated = [item for item in rows if item["role"] in generation.SEEDS]
    added = {arm: {family: {row["group"] for row in generated
                           if row["role"] == family and row["arms"][arm]["useful"] and not row["arms"]["current"]["useful"]}
                   for family in generation.SEEDS} for arm in ARMS[1:]}
    decision = {}
    for arm in ARMS[1:]:
        gains = set.union(*added[arm].values())
        lost = {row["group"] for row in generated if row["arms"]["current"]["useful"] and not row["arms"][arm]["useful"]}
        false_additions = {row["group"] for row in rows if row["role"] == "after" and row["arms"][arm]["flag"]
                           and not row["arms"]["current"]["flag"]}
        decision[arm] = {"new_useful_groups": len(gains), "lost_useful_groups": len(lost),
                         "new_later_false_flag_groups": len(false_additions),
                         "new_useful_by_generator": {family: len(group) for family, group in added[arm].items()},
                         "advance": (all(admission[family].get("admitted", 0) >= 6 and added[arm][family]
                                         for family in generation.SEEDS)
                                     and len(gains) >= 3 and not lost and len(false_additions) <= 1)}
    strata = sorted({f"{row['archive']}/{row['genre']}/{row['recorded_authorship']}" for row in rows})
    return {"overall": breakdown(rows), "admission": admission, "decision": decision,
            "strata": {stratum: breakdown([row for row in rows if
                       f"{row['archive']}/{row['genre']}/{row['recorded_authorship']}" == stratum]) for stratum in strata}}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("phase", choices=("freeze", "generate", "lock-inputs", "score", "report"))
    cli.add_argument("source", type=Path)
    cli.add_argument("snapshot", type=Path)
    cli.add_argument("--family", choices=generation.SEEDS)
    args = cli.parse_args()
    os.umask(0o077)
    require(os.getenv("HF_HUB_OFFLINE") == os.getenv("TRANSFORMERS_OFFLINE") == "1", "Models must be offline")
    source = args.source
    lock = source / "freeze.json"
    frozen = protocol(source, args.snapshot)
    if args.phase == "freeze":
        write(lock, frozen)
        return
    require(read(lock) == frozen, "Study freeze changed")
    if args.phase == "generate":
        require(args.family, "Select one generator")
        generation.generate(source, source, frozen, args.family)
        return
    rows, values, admission = inputs(source)
    require(not (values.keys() & read(PEP_SCORES / "calibration.json")["responses"].keys()),
            "A held native context already belongs to fitted development input")
    input_lock = source / "input-lock.json"
    binding = {"freeze_sha256": digest(lock), "generation_sha256": {family: digest(source / f"generated-{family}.json")
                                                                  for family in generation.SEEDS},
               "admission": admission, "input_hashes": sorted(values),
               "rows": [{key: value for key, value in row.items() if key != "text"} for row in rows]}
    if args.phase == "lock-inputs":
        write(input_lock, binding)
        print({"files": len(rows), "unique_contexts": len(values), "admission": admission}, flush=True)
        return
    require(read(input_lock) == binding, "Frozen exact input inventory differs")
    output = source / "responses.json"
    if args.phase == "score":
        score(args.snapshot, values, output, digest(input_lock))
        return
    packet = read(output)
    require(packet["binding"] == digest(input_lock) and packet["responses"].keys() == values.keys()
            and packet["forwards"] == (len(values) + 3) // 4, "Frozen inference receipts differ")
    result = outcomes(rows, packet["responses"])
    aggregate = summary(result, admission)
    write(source / "outcomes.json", result)
    write(source / "aggregate.json", aggregate)
    print(aggregate, flush=True)


if __name__ == "__main__":
    main()
