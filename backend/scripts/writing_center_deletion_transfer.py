"""Compare a single-file, equal-sentence-count local writing contrast on PEP edits."""

from __future__ import annotations

import argparse
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from time import monotonic

from scripts import writing_paired_revision_replication as generation
from scripts.video_appearance_native_v108 import ROOT, read, require, write
from scripts.writing_blind_rewrite_probe import text_hash, valid_sentence, words
from scripts.writing_focus_v3 import BASELINE_ASSETS, digest

SCHEMA = "chipmunk.writing-center-deletion-transfer/1"
DATA = ROOT / "backend/data/independent-evaluation"
CALIBRATION = {
    "f1000": DATA / "writing-reference-revision-transfer-2026-11-11/expanded/originals.json",
    "publisher-pilot": DATA / "writing-publisher-revision-replication-2026-11-11/holdout/originals.json",
    "wellcome-gates": DATA / "writing-journal-body-transfer-2026-11-11/originals.json",
}
EXACT_ARMS = ("matched_context", "center_deletion")
ARMS = ("current", *EXACT_ARMS, "combined")
PROTOCOL = {
    "scope": "Single English paragraph with five completely measured native sentences. Compare all three eligible "
             "interior sentences of 12–40 words. Left and right probes each have two sentences and preserve order.",
    "current": "First six native context windows whose current raw score is at least 0.5.",
    "matched_context": "Rank eligible centers using full native context log odds and output the uniquely highest exact sentence.",
    "center_deletion": "Use min(log_odds(L+C), log_odds(C+R)) - log_odds(L+R); output the uniquely highest exact center.",
    "combined": "Union of incumbent windows and center-deletion selected sentence. Keep source localization by arm.",
    "calibration": "Use the second largest per-source maximum across original and human-edited files from 32 prior groups. "
                   "Each arm has its own cutoff. Only strictly greater passes.",
    "scores": "Raw current classifier measurements from exact input strings. Clamp to [1e-7, 1-1e-7] for odds.",
    "advance": "Require at least six admitted edits from each generator, three useful new groups overall and at least one "
               "from each generator, at most one new ordinary later false-flag group, and at least two more useful "
               "groups than matched-context-only combined with current.",
}


def freeze(run, source, snapshot):
    from scripts.text_elife_rewrites_v273 import GENERATORS, MLX_VERSIONS
    from scripts.writing_blind_rewrite_probe import generator_configuration

    receipt, originals = source / "source-receipt.json", source / "originals.json"
    accepted = read(receipt)
    roster = read(originals)
    require(accepted["schema"] == "chipmunk.writing-pep-revision-intake/1" and accepted["originals_sha256"] == digest(originals)
            and len(roster) >= 4, "Insufficient qualified and replayed fresh source groups")
    config = generator_configuration()
    identities = {}
    for name in generation.SEEDS:
        _, _, expected, folder = GENERATORS[name]
        require(config[name]["license"] == "Apache-2.0" and digest(folder / "model.safetensors") == expected,
                "Generator weight or commercial-use terms differ")
        identities[name] = config[name]
    require(all(digest(snapshot / key) == value for key, value in BASELINE_ASSETS.items()), "Current checkpoint differs")
    return {
        "schema": SCHEMA, "scope": "holdout", "source_groups": len(roster), "study_only": True,
        "code_sha256": digest(Path(__file__)),
        "method": PROTOCOL, "source_sha256": digest(originals), "source_receipt_sha256": digest(receipt),
        "calibration": {name: digest(path) for name, path in CALIBRATION.items()},
        "generator_assets": identities, "runtime": MLX_VERSIONS, "seeds": generation.SEEDS,
        "prompt": generation.EDIT_PROMPT, "settings": generation.SETTINGS,
        "generation_impl_sha256": digest(Path(generation.__file__)), "current_assets": BASELINE_ASSETS,
    }


def contextualize(rows):
    from app.detectors.text.extract import from_string
    from app.detectors.text.segment import segment
    from scripts.writing_paired_email_revision import encoded, native_contexts

    values = {}
    mapped = []
    for row in rows:
        doc = from_string(row["text"])
        units = segment(doc)
        require(len(units) == 5 and doc.text == row["text"], "Source fixture changed its native five-sentence geometry")

        def register(group, *, inventory=values):
            entry, value = encoded(group)
            inventory[entry["sha256"]] = value
            return entry

        native = [register(group) for group in native_contexts(units)]
        centers = []
        for left, center, right in zip(units[:3], units[1:4], units[2:], strict=True):
            if not 12 <= len(words(center.text)) <= 40:
                continue
            probes = [register(group) for group in ((left, center), (center, right), (left, right))]
            centers.append({"position": center.idx, "span": [center.start, center.end], "probes": probes})
        mapped.append(row | {"text_sha256": text_hash(row["text"]), "native": native, "centers": centers})
    return mapped, values


def development():
    rows = []
    for prefix, path in CALIBRATION.items():
        for item in read(path):
            for role in ("before", "after"):
                row = item[role]
                rows.append({"group": f"{prefix}/{item['group']}", "role": role,
                             "focus": row["focus"], "text": row["text"]})
    require(len(rows) == 64 and len({row["group"] for row in rows}) == 32, "Calibration cohort changed")
    return contextualize(rows)


def confirmation(run, source):
    from app.detectors.text.extract import from_string
    from app.detectors.text.segment import segment

    originals = read(source / "originals.json")
    attempts = {name: read(run / f"generated-{name}.json") for name in generation.SEEDS}
    require(all(packet["freeze_sha256"] == digest(run / "freeze.json")
                and set(packet["results"]) == {row["group"] for row in originals} for packet in attempts.values()),
            "Frozen source and generator roster differ")
    rows, counts = [], defaultdict(Counter)
    for item in originals:
        a, b = item["before"], item["after"]
        before = a["focus_text"]
        lo, hi = a["focus"]
        unchanged = [unit.text for i, unit in enumerate(segment(from_string(a["text"]))) if i != 2]
        versions = {"before": a, "after": b}
        for name, packet in attempts.items():
            generated = packet["results"][item["group"]]
            require(generated["input_sha256"] == text_hash(before), "Generator saw another input")
            counts[name]["attempted"] += 1
            response = valid_sentence(generated, before, generated=True)
            if response is None or response == b["focus_text"]:
                continue
            text = a["text"][:lo] + response + a["text"][hi:]
            doc = from_string(text)
            units = segment(doc)
            if (doc.text != text or len(units) != 5 or units[2].text != response
                    or [unit.text for i, unit in enumerate(units) if i != 2] != unchanged):
                continue
            versions[name] = {"text": text, "focus": [lo, lo + len(response)]}
            counts[name]["admitted"] += 1
        for role, revision in versions.items():
            rows.append({"group": item["group"], "role": role, "focus": revision["focus"], "text": revision["text"]})
    for row in rows:
        units = segment(from_string(row["text"]))
        require(len(units) == 5 and [units[2].start, units[2].end] == row["focus"], "Exact center location differs")
    return (*contextualize(rows), {key: dict(value) for key, value in counts.items()})


def odds(p):
    require(type(p) is float and math.isfinite(p) and 0 <= p <= 1, "Invalid model response")
    value = min(1 - 1e-7, max(1e-7, p))
    return math.log(value / (1 - value))


def measurements(row, responses):
    native = [responses[entry["sha256"]]["score"] for entry in row["native"]]
    points = {}
    for focus in row["centers"]:
        left, right, flanks = (odds(responses[entry["sha256"]]["score"]) for entry in focus["probes"])
        position = focus["position"]
        points[position] = {"matched_context": odds(native[position]), "center_deletion": min(left, right) - flanks}
    return native, points


def calibration_points(rows, responses):
    groups = {arm: {} for arm in EXACT_ARMS}
    for row in rows:
        _, candidates = measurements(row, responses)
        if candidates:
            for arm in EXACT_ARMS:
                peak = max(item[arm] for item in candidates.values())
                group = row["group"]
                groups[arm][group] = max(peak, groups[arm].get(group, -math.inf))
    require(all(len(group) >= 24 for group in groups.values()), "Too little eligible grouped calibration coverage")
    return {"group_maxima": groups, "cutoff": {arm: sorted(group.values())[-2] for arm, group in groups.items()}}


def useful(spans, truth, text):
    guess = {i for a, b in spans for i in range(a, b) if not text[i].isspace()}
    both = len(guess & truth)
    return 10 * both >= 3 * len(guess | truth) and 2 * both >= len(truth)


def evaluate(rows, responses, points):
    results = []
    for row in rows:
        native, contenders = measurements(row, responses)
        current = [entry["spans"] for entry, value in zip(row["native"], native, strict=True) if value >= .5][:6]
        exact = {}
        for arm in EXACT_ARMS:
            ranked = sorted(((values[arm], position) for position, values in contenders.items()), reverse=True)
            if ranked and ranked[0][0] > points[arm] and (len(ranked) == 1 or ranked[0][0] > ranked[1][0]):
                pos = ranked[0][1]
                exact[arm] = [[next(focus["span"] for focus in row["centers"] if focus["position"] == pos)]]
            else:
                exact[arm] = []
        chosen = {"current": current, **exact, "combined": current + exact["center_deletion"]}
        truth = {i for i in range(*row["focus"]) if not row["text"][i].isspace()}
        matched = current + exact["matched_context"]
        report = {arm: {"flag": bool(windows),
                        "useful": any(useful(spans, truth, row["text"]) for spans in windows)}
                  for arm, windows in chosen.items()}
        results.append({"group": row["group"], "role": row["role"], "arms": report,
                        "matched_plus_current": any(useful(spans, truth, row["text"]) for spans in matched)})
    return results


def summarize(outcomes, admission):
    details = {role: [row for row in outcomes if row["role"] == role]
               for role in ("before", "after", *generation.SEEDS)}
    breakdown = {role: {arm: {"flags": sum(row["arms"][arm]["flag"] for row in rows),
                              "useful": sum(row["arms"][arm]["useful"] for row in rows)}
                        for arm in ARMS} | {"files": len(rows)} for role, rows in details.items()}
    gains = {name: {row["group"] for row in details[name]
                    if row["arms"]["combined"]["useful"] and not row["arms"]["current"]["useful"]}
             for name in generation.SEEDS}
    ordinary = {row["group"] for row in details["after"]
                if row["arms"]["combined"]["flag"] and not row["arms"]["current"]["flag"]}
    generated = [row for name in generation.SEEDS for row in details[name]]
    added_over_matched = {row["group"] for row in generated
                          if row["arms"]["combined"]["useful"] and not row["matched_plus_current"]}
    decisions = {
        "unique_gain_groups": len(set.union(*gains.values())),
        "gain_groups_by_generator": {name: len(group) for name, group in gains.items()},
        "added_later_false_groups": len(ordinary), "added_over_matched_groups": len(added_over_matched),
        "advances": (all(admission.get(name, {}).get("admitted", 0) >= 6 and gains[name] for name in generation.SEEDS)
                     and len(set.union(*gains.values())) >= 3 and len(ordinary) <= 1 and len(added_over_matched) >= 2),
    }
    return {"counts": breakdown, "decision": decisions}


def score(path, values, outfile, binding):
    import torch

    from scripts.writing_paired_email_revision import native_scores

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    begin = monotonic()
    results, forwards = native_scores(path, values, second=False)
    write(outfile, {"binding": binding, "responses": results, "forwards": forwards, "seconds": round(monotonic() - begin, 3)})
    print(f"Scored {len(values)} exact contexts in {forwards} CPU forwards.", flush=True)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("phase", choices=("freeze", "generate", "calibrate", "lock-holdout", "holdout", "report"))
    cli.add_argument("run", type=Path)
    cli.add_argument("source", type=Path)
    cli.add_argument("snapshot", type=Path)
    cli.add_argument("--family", choices=generation.SEEDS)
    args = cli.parse_args()
    os.umask(0o077)
    require(os.getenv("HF_HUB_OFFLINE") == os.getenv("TRANSFORMERS_OFFLINE") == "1", "Expected offline snapshots")
    run, phase, source = args.run, args.phase, args.source
    protocol = freeze(run, source, args.snapshot)
    lock = run / "freeze.json"
    if phase == "freeze":
        run.mkdir(exist_ok=True)
        write(lock, protocol)
        return
    require(read(lock) == protocol, "Frozen protocol differs")
    if phase == "generate":
        require(args.family, "Select one generator")
        generation.generate(run, source, protocol, args.family)
        return
    dev_rows, dev_inputs = development()
    calibration, point_file = run / "calibration.json", run / "operating-points.json"
    if phase == "calibrate":
        score(args.snapshot, dev_inputs, calibration, digest(lock))
        points = calibration_points(dev_rows, read(calibration)["responses"])
        write(point_file, points | {"calibration_sha256": digest(calibration)})
        return
    fitted = read(point_file)
    require(fitted["calibration_sha256"] == digest(calibration), "Calibration receipt differs")
    rows, inputs, admission = confirmation(run, source)
    cases, scored = run / "holdout-cases.json", run / "holdout.json"
    holdout_lock = run / "holdout-freeze.json"
    pins = {"freeze_sha256": digest(lock), "point_sha256": digest(point_file),
            "attempts": {name: digest(run / f"generated-{name}.json") for name in generation.SEEDS},
            "input_hashes": sorted(inputs)}
    if phase == "lock-holdout":
        write(cases, rows)
        write(holdout_lock, pins | {"case_sha256": digest(cases)})
        return
    require(read(cases) == rows and read(holdout_lock) == pins | {"case_sha256": digest(cases)}, "Holdout inputs differ")
    if phase == "holdout":
        score(args.snapshot, inputs, scored, digest(holdout_lock))
        return
    receipt = read(scored)
    require(receipt["binding"] == digest(holdout_lock) and receipt["responses"].keys() == inputs.keys(),
            "Holdout scores do not match frozen inputs")
    outcome_file = run / "outcomes.json"
    outcomes = evaluate(rows, receipt["responses"], fitted["cutoff"])
    write(outcome_file, outcomes)
    result = summarize(outcomes, admission)
    write(run / "aggregate.json", result | {"schema": SCHEMA, "generator_admission": admission,
          "holdout_sha256": digest(scored), "outcome_sha256": digest(outcome_file), "point_sha256": digest(point_file),
          "distinct_contexts": {"calibration": len(dev_inputs), "holdout": len(inputs)},
          "cpu_forwards": {"calibration": read(calibration)["forwards"], "holdout": receipt["forwards"]}})
    print(result, flush=True)


if __name__ == "__main__":
    main()
