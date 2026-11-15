# SPDX-License-Identifier: MIT
"""Measure word responses when neighboring sentences are hidden from the encoder."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from scripts import writing_edit_occurrence_transfer as occurrence
from scripts import writing_local_word_spans as spans

SCHEMA = "chipmunk.writing-context-response-words/1"
TRANSFER_SCHEMA = "chipmunk.writing-context-response-transfer/1"
RULE = occurrence.RULE | {
    "context": "Use the frozen word encoder and retain each word state and its difference from its own sentence. "
               "Replace the neighboring-sentence mean contrast with each word's full-context state minus its state "
               "when tokens outside the exact target sentence are hidden by the attention mask. Preserve all token "
               "positions and original special tokens in both views.",
}
PRIOR = spans.PRIMARY.with_name("writing-edit-occurrence-2026-11-15.json.gz")
ORDINARY = spans.PRIMARY.with_name("writing-edit-occurrence-elife-2026-11-15.json")
KEYS = ("role", "group", "family", "sha256", "spans", "position", "words", "changed", "assessed")


def review(path):
    candidate, prior = map(spans.read, (path, PRIOR))
    if (candidate["schema"] != SCHEMA or candidate["rule"] != RULE
            or candidate["encoder_sha256"] != prior["encoder_sha256"]
            or candidate["fit"] != prior["fit"] or candidate["fit_words"] != prior["fit_words"]):
        raise ValueError("The matched writing comparison changed its inputs")
    for phase in ("calibration", "held"):
        for before, after in zip(prior[phase], candidate[phase], strict=True):
            if any(before[key] != after[key] for key in KEYS):
                raise ValueError("A contextual reading changed native word geometry")
    points = occurrence.points(candidate["calibration"])
    result = occurrence.compare(candidate["held"], points)
    if points != candidate["points"] or result != candidate["results"]:
        raise ValueError("The contextual response operating points or results differ")
    changes = paired_outcomes(zip(prior["held"], candidate["held"], strict=True), prior["points"], points)
    return {"counts": result, "paired": changes,
            "matched_sentences": matched_sentence_alerts(candidate["held"], points)}


def selected(row, arm, point):
    best = row["best"][arm]
    return (best is not None and best["score"] > point and
            (row["family"] in {"original", "before", "edited"}
             or occurrence.prior.useful_words(set(best["words"]), set(row["changed"]))))


def paired_outcomes(pairs, before_points, after_points):
    changes = defaultdict(Counter)
    for before, after in pairs:
        for arm, cuts in after_points.items():
            for level, point in cuts.items():
                a = selected(before, arm, before_points[arm][level])
                b = selected(after, arm, point)
                changes[" / ".join((after["family"], arm, level))].update(
                    before=int(a), after=int(b), gained=int(b and not a), lost=int(a and not b), either=int(a or b))
    return {key: dict(values) for key, values in sorted(changes.items())}


def matched_sentence_alerts(rows, points):
    ordinary = {row["group"]: row for row in rows if row["family"] == "original"}
    counts = defaultdict(Counter)
    for row in rows:
        if not row["changed"] or row["group"] not in ordinary:
            continue
        mate = ordinary[row["group"]]
        for arm, cuts in points.items():
            local, control = row["best"][arm], mate["best"][arm]
            for level, cut in cuts.items():
                if (local is not None and local["score"] > cut
                        and occurrence.prior.useful_words(set(local["words"]), set(row["changed"]))):
                    counts[" / ".join((row["family"], arm, level))].update(
                        locations=1, matched_ordinary_alert=int(control is not None and control["score"] > cut
                                                                and control["sentence"] == local["sentence"]))
    return {key: dict(value) for key, value in sorted(counts.items())}


def transfer_points(calibration, report):
    if any(row["changed"] or row["family"] != "original" for row in calibration):
        raise ValueError("Genre points need ordinary excerpts alone")
    points = {"report": report}
    for domain in ("news", "abstracts"):
        works = [row for row in calibration if row["domain"] == domain]
        if len({row["group"] for row in works}) != len(works):
            raise ValueError("A genre calibration work was repeated")
        points[domain] = {arm: occurrence.prior.points([row | {"best": row["best"][arm], "truth": None} for row in works], minimum=24)
                          for arm in report}
    return points


def transfer_results(rows, points):
    return {domain: {label: occurrence.compare(cases, points[label])
                     for label in ("report", domain) if label in points}
            for domain in ("news", "abstracts", "journals") if (cases := [row for row in rows if row["domain"] == domain])}


def transfer(args):
    import torch
    from safetensors.torch import load_file

    torch.set_num_threads(4)
    candidate_path = args.fitted / "receipt.json"
    review(candidate_path)
    candidate = spans.read(candidate_path)
    previous = spans.read(PRIOR)
    prior = occurrence.prior
    digest = prior.parent.pool.file_hash
    paths = {"prior": args.before / "head.safetensors", "context": args.fitted / "head.safetensors"}
    if (digest(args.encoder_weights) != candidate["encoder_sha256"] or digest(paths["context"]) != candidate["head_sha256"]
            or digest(paths["prior"]) != previous["head_sha256"]):
        raise ValueError("The frozen fitted assets differ")
    rows, truth, excluded = prior.transfer_plan(args.roster, args.matched, args.additional)
    journals, changed, journal_source = occurrence.ordinary_inputs(args.editorial)
    rows += journals
    truth.update(changed)
    tokenizer, backbone, _, device = prior.parent.init_encoder(args.encoder)
    raw = load_file(args.encoder_weights)
    backbone.load_state_dict({key.removeprefix("encoder."): value for key, value in raw.items() if key.startswith("encoder.")}, strict=True)
    backbone.eval()
    heads = {view: occurrence.load_fitted_heads(path) for view, path in paths.items()}

    def measure(inputs):
        full = occurrence.measure(inputs, truth, tokenizer, backbone, device, heads["prior"])
        masked = occurrence.measure(inputs, truth, tokenizer, backbone, device, heads["context"], context_response=True)
        for source, a, b in zip(inputs, full, masked, strict=True):
            if any(a[key] != b[key] for key in KEYS) or source["sha256"] != a["sha256"]:
                raise ValueError("The two readings do not refer to the same native words")
            yield a | {"domain": source["domain"], "best": {
                f"{view}/{arm}": best for view, reading in (("prior", a), ("context", b)) for arm, best in reading["best"].items()
            }}

    calibration = list(measure([row for row in rows if row["role"] == "calibration"]))
    report_points = {f"{view}/{arm}": point for view, reading in (("prior", previous), ("context", candidate))
                     for arm, point in reading["points"].items()}
    points = transfer_points(calibration, report_points)
    args.output.mkdir(mode=0o700)
    prior.parent.previous.write(args.output / "points.json", points)
    held = list(measure([row for row in rows if row["role"] != "calibration"]))
    outcomes = transfer_results(held, points)
    record = {"schema": TRANSFER_SCHEMA, "held": held, "calibration": calibration, "points": points, "results": outcomes,
              "excluded_groups": excluded, "editorial": journal_source, "fit_sha256": prior.receipt_digest(candidate)}
    prior.parent.previous.write(args.output / "receipt.json", record)


def review_transfer(path, fitted):
    record, candidate, previous = map(spans.read, (path, fitted, PRIOR))
    if (record["schema"] != TRANSFER_SCHEMA or record["fit_sha256"] != occurrence.prior.receipt_digest(candidate)
            or any(row["role"] != "calibration" for row in record["calibration"])
            or any(row["role"] == "calibration" for row in record["held"])):
        raise ValueError("Transfer scores differ from the sealed fit")
    report_points = {f"{view}/{arm}": point for view, reading in (("prior", previous), ("context", candidate))
                     for arm, point in occurrence.points(reading["calibration"]).items()}
    points = transfer_points(record["calibration"], report_points)
    native = spans.read(spans.TRANSFER)
    expected = {(r["role"], r["group"], r["family"]): r for r in native["calibration"] + native["held"]}
    ordinary = spans.read(ORDINARY)
    if record["editorial"] != ordinary["source_receipt"]:
        raise ValueError("The verified ordinary revision roster differs")
    journals = {row["sha256"]: row for row in ordinary["rows"]}
    for row in record["calibration"] + record["held"]:
        if row["domain"] == "journals":
            prior = journals.pop(row["sha256"])
            if any(prior[key] != row[key] for key in KEYS):
                raise ValueError("A journal source or its native word geometry differs")
            continue
        ref = expected.pop((row["role"], row["group"], row["family"]))
        if (any(row[key] != ref[key] for key in ("sha256", "domain", "spans", "position", "assessed"))
                or row["words"] != ref["word_ranges"] or row["changed"] != ref["changed_words"]):
            raise ValueError("The genre test changed its original source geometry")
    outcomes = transfer_results(record["held"], points)
    if expected or journals or points != record["points"] or outcomes != record["results"]:
        raise ValueError("A transfer source is missing or its result differs")
    matches, paired = {}, {}
    for domain in ("news", "abstracts"):
        rows = [row for row in record["held"] if row["domain"] == domain]
        matches[domain] = {label: matched_sentence_alerts(rows, points[label]) for label in ("report", domain)}
        paired[domain] = {}
        for label in ("report", domain):
            prior, context = ({arm: points[label][f"{view}/{arm}"] for arm in occurrence.ARMS} for view in ("prior", "context"))
            pairs = ((row | {"best": {arm: row["best"][f"prior/{arm}"] for arm in occurrence.ARMS}},
                      row | {"best": {arm: row["best"][f"context/{arm}"] for arm in occurrence.ARMS}}) for row in rows)
            paired[domain][label] = paired_outcomes(pairs, prior, context)
    return {"counts": outcomes, "matched_sentences": matches, "paired": paired}


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    actions = cli.add_subparsers(dest="phase", required=True)
    fit = actions.add_parser("run")
    for name in ("roster", "encoder", "encoder_weights", "output"):
        fit.add_argument(name, type=Path)
    actions.add_parser("review").add_argument("receipt", type=Path)
    test = actions.add_parser("transfer")
    for name in ("roster", "matched", "additional", "editorial", "encoder", "encoder_weights", "before", "fitted", "output"):
        test.add_argument(name, type=Path)
    audit = actions.add_parser("review-transfer")
    audit.add_argument("receipt", type=Path)
    audit.add_argument("fitted", type=Path)
    args = cli.parse_args()
    if args.phase == "run":
        occurrence.run(args, schema=SCHEMA, rule=RULE, context_response=True)
    elif args.phase == "review":
        print(json.dumps(review(args.receipt), sort_keys=True, indent=2))
    elif args.phase == "transfer":
        transfer(args)
    else:
        print(json.dumps(review_transfer(args.receipt, args.fitted), sort_keys=True, indent=2))
