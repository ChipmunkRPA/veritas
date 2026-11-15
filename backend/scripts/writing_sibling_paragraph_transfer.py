"""Test whether nearby paragraphs contextualize formal-writing passage flags."""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

from scripts import writing_context_pool_transfer as pool
from scripts import writing_focus_inside_context as intake
from scripts import writing_matched_encoder as matched
from scripts import writing_missed_passage_transfer as causal
from scripts import writing_sentence_probe as prior
from scripts import writing_topic_response_transfer as topic
from scripts.writing_interval_transfer import fingerprints

SCHEMA = "chipmunk.writing-sibling-paragraph-transfer/1"
SOURCES = (
    ("calibration", "dev", "gpt-5.4-nano", matched.SOURCES[2][4], 32),
    ("held", "test", topic.GENERATORS[1], topic.PINS[1], 40),
    ("held", "test", topic.GENERATORS[0], topic.PINS[0], 40),
)
METHOD = {
    "delivery": "Three ordered, non-overlapping excerpts of five native sentences from one publisher-flattened source work. "
                "Each excerpt is separated by at least 200 original characters and delivered as a separate paragraph. "
                "Exactly one of the three central interior sentences has a publisher-verified rewrite.",
    "comparator": "Unmodified three-sentence classifier windows and exact native writing-pattern reductions or their recorded expansions.",
    "matched_controls": "For each eligible ratio-only interior sentence, find the three closest interior lengths from each other paragraph. "
                        "Control lengths must be within a factor of two. Compare the candidate's ratio with the median in each paragraph.",
    "points": "With a strict exceedance, allow at most ten or twenty-five percent of eligible genuine calibration files. "
              "Use the smaller of both contrasts. Keep classifier flags and ratio-only exact locations beyond the point. "
              "Count removed ordinary flags, lost rewrite locations, and remaining wrong-position flags.",
}


def group_blocks(raw, windows):
    from app.detectors.text.extract import _normalise

    by_text = defaultdict(list)
    for spans in windows:
        start, end = spans[0][0], spans[-1][1]
        normalized = _normalise(raw[start:end])[0]
        by_text[normalized].append((start, end, spans))
    return by_text


def compose(before, middle, after, *, flank_size=5, min_words=180):
    from app.detectors.text.extract import from_string
    from app.detectors.text.segment import segment

    doc = from_string("\n\n".join((before, middle, after)))
    units = segment(doc)
    sizes = (flank_size, 5, flank_size)
    if (len(doc.paragraphs) != 3 or [s.paragraph for s in units] != [p for p, size in enumerate(sizes) for _ in range(size)]
            or not min_words <= doc.meta["words"] <= 540):
        raise ValueError("Control paragraphs differ from the native source positions")
    return doc.text, [[s.start, s.end] for s in units]


def shorter_windows(raw, size):
    from scripts.writing_sentence_anchor_transfer import source_spans

    units = source_spans(raw)
    for i in range(len(units) - size + 1):
        window = units[i:i + size]
        if (window[0][0].paragraph == window[-1][0].paragraph
                and all(8 <= sentence.n_words <= 80 for sentence, _, _ in window)):
            yield [(a, b) for _, a, b in window]


def pair(name, versions, position_order, *, flank_size=5, separation=200, min_words=180):
    raw, _ = pool.truth_text(versions["v0"])
    by_text = group_blocks(raw, pool.source_windows(raw))
    options = intake.excerpt_options(name, versions)
    if not options:
        return None
    controls = by_text if flank_size == 5 else group_blocks(raw, shorter_windows(raw, flank_size))
    for position in position_order:
        if position not in options:
            continue
        (original, _, _), (rewrite, _, variant) = options[position]
        anchor = by_text[original]
        if len(anchor) != 1:
            continue
        start, end, _ = anchor[0]
        flanks = [[], []]
        for excerpt, locations in controls.items():
            if len(locations) != 1:
                continue
            a, b, spans = locations[0]
            left = start - b >= separation
            right = a - end >= separation
            if not (left or right):
                continue
            clean = excerpt
            if flank_size == 5:
                try:
                    clean, _ = pool.normalize(raw, spans)
                except ValueError:
                    continue
            flanks[int(right)].append((start - b if left else a - end, a, b, clean))
        if not all(flanks):
            continue
        _, a, b, before = min(flanks[0])
        _, c, d, after = min(flanks[1])
        try:
            clean, spans = compose(before, original, after, flank_size=flank_size, min_words=min_words)
            edited, edit_spans = compose(before, rewrite, after, flank_size=flank_size, min_words=min_words)
        except ValueError:
            continue
        k = flank_size + position
        unchanged = [(clean[x:y], edited[a:b]) for i, ((x, y), (a, b)) in enumerate(zip(spans, edit_spans, strict=True))
                     if i != k]
        if any(before != after for before, after in unchanged):
            continue
        return ({"position": k, "source_anchors": [[a, b], [start, end], [c, d]]},
                [(clean, spans, None), (edited, edit_spans, variant)])
    return None


def earlier_panels():
    return {path.name: pool.file_hash(path) for path in sorted(pool.PANEL_DIR.glob("writing-*-2026-11-14.json"))
            if not path.name.startswith("writing-other-paragraphs-")}


def source_exclusions(pins):
    values = set()
    for name, digest in pins.items():
        if Path(name).name != name or not name.startswith("writing-") or not name.endswith("-2026-11-14.json"):
            raise ValueError("Unexpected prior panel filename")
        path = pool.PANEL_DIR / name
        if pool.file_hash(path) != digest:
            raise ValueError("A prior source panel changed")
        values.update(prior.SHA256.findall(path.read_text()))
    if not values:
        raise ValueError("The evaluation must use the prior source exclusions")
    return values


def freeze(directory, development, output, *, previous):
    excluded = source_exclusions(previous)
    readouts = {}
    rows, used = [], set()
    for role, split, generator, pin, wanted in SOURCES:
        folder = development if role == "calibration" else directory
        sources = matched.load_shard(folder / f"{split}-reports_{generator}.csv", split, generator, pin)
        # Exclude overlap with source works already exposed to prior experiments.
        exposed = {name for name in sources if pool.sha(name) in excluded}
        for name in exposed:
            if "v0" in sources[name]:
                used.update(fingerprints(sources[name]["v0"]["text"]))
        positions = Counter()
        for name in sorted(sources, key=lambda key: pool.sha(f"{SCHEMA}/{role}/{generator}/{key}")):
            group = pool.sha(name)
            if group in excluded or any(v not in sources[name] for v in ("v0", *pool.VARIANTS)):
                continue
            raw = sources[name]["v0"]["text"]
            if not 1700 <= len(raw) <= 65000:
                continue
            tokens = fingerprints(raw)
            if used.intersection(tokens):
                continue
            order = sorted((1, 2, 3), key=lambda pos: (positions[5 + pos], pos))
            try:
                deliveries = pair(name, sources[name], order)
            except ValueError:
                continue
            if deliveries is None:
                continue
            offsets, cases = deliveries
            positions[offsets["position"]] += 1
            excluded.add(group)
            used.update(tokens)
            for text, spans, variant in cases if role == "held" else cases[:1]:
                rows.append(offsets | {"group": group, "role": role, "generator": generator,
                                       "origin": variant, "sha256": pool.sha(text), "text": text, "spans": spans,
                                       "truth": spans[offsets["position"]] if variant else None,
                                       "spaces": [i for i, char in enumerate(text) if char.isspace()]})
            if positions.total() >= wanted:
                break
        readouts[f"{role}/{generator}"] = dict(positions)
        print(role, generator, positions.total(), flush=True)
        if not positions:
            raise ValueError("A named study arm contains no eligible source works")
    prior.write(output, {"schema": SCHEMA, "method": METHOD, "publishers": SOURCES,
                         "earlier_panels": previous,
                         "admissions": readouts, "rows": rows})


def readings(path, role, phase, output):
    from app.detectors.text.extract import from_string
    from app.detectors.text.pipeline import windows
    from app.detectors.text.segment import segment

    rows = [row for row in intake.read(path)["rows"] if row["role"] == role]
    scores = {}
    if phase == "classifier":
        import torch

        from app.detectors.text.classifier import FourClassTextClassifier

        if any(pool.file_hash(pool.SNAPSHOT / name) != digest for name, digest in pool.BASELINE_ASSETS.items()):
            raise ValueError("Classifier assets do not match the pinned revision")
        torch.set_num_threads(2)
        classifier = FourClassTextClassifier.from_snapshot(pool.SNAPSHOT, "mps" if torch.backends.mps.is_available() else "cpu")
    else:
        from app.detectors.text.binoculars import sentence_score_records

        ratio, mirror = causal.load_pair()
    for index, row in enumerate(rows, 1):
        text = row["text"]
        doc = from_string(text)
        units = segment(doc)
        if doc.text != text or pool.sha(text) != row["sha256"] or [[x.start, x.end] for x in units] != row["spans"]:
            raise ValueError("The committed writing inputs differ")
        if phase == "classifier":
            values = classifier.predict(windows(units, 3), batch=8)
        else:
            observed = ratio.token_scores(text)
            peer = mirror(text, return_offsets_mapping=True, add_special_tokens=False)
            if peer["input_ids"] != observed.source_token_ids or list(map(tuple, peer["offset_mapping"])) != observed.offsets:
                raise ValueError("The token identities or offsets differ")
            values = [[r["score"], r["tokens_scored"], r["neighbor_expansion"]]
                      for r in sentence_score_records(observed, units)]
        scores[row["sha256"]] = values
        if index % 16 == 0:
            print(phase, role, index, len(rows), flush=True)
    prior.write(output, {"role": role, "phase": phase, "roster_sha256": pool.file_hash(path), "scores": scores})


def compare(row):
    ratios = row["ratio"]
    for index, (reading, tokens, expanded) in enumerate(ratios):
        if reading is None or expanded or reading >= causal.POINT or row["classifier"][index] >= .5 or index % 5 in {0, 4}:
            continue
        centers = []
        for p in range(3):
            if p == index // 5:
                continue
            # Match sentence length on the extreme-ratio check without using the outcome.
            alternatives = [(abs(math.log(n / tokens)), j, score) for j in range(p * 5 + 1, p * 5 + 4)
                            for score, n, broad in (ratios[j],) if score is not None and not broad and tokens / 2 <= n <= tokens * 2]
            if len(alternatives) < 2:
                break
            centers.append(median(value for _, _, value in sorted(alternatives)[:3]))
        if len(centers) == 2:
            yield index, round(min(centers) - reading, 8)


def evaluate(rows, points):
    from scripts.writing_editorial_worker_audit import covers

    def location(row, i, is_ratio):
        spans = row["spans"]
        if is_ratio and not row["ratio"][i][2]:
            return spans[i]
        return [spans[max(0, i - 1)][0], spans[min(len(spans) - 1, i + 1)][1]]

    summary = defaultdict(Counter)
    for row in rows:
        if row["role"] != "held":
            continue
        truth = row["truth"]
        spaces = set(row["spaces"])
        outcome = summary[f"{row['generator']}/{'rewritten' if truth else 'ordinary'}"]
        classifier = [location(row, i, False) for i, s in enumerate(row["classifier"]) if s >= .5]
        raw = [location(row, i, True) for i, (r, _, _) in enumerate(row["ratio"]) if r is not None and r < causal.POINT]
        scored = list(compare(row))
        arms = {"classifier": classifier, "current": classifier + raw,
                "eligibility_only": classifier + [location(row, i, True) for i, _ in scored]}
        for name, cutoff in points.items():
            used = [location(row, i, True) for i, score in scored if score > cutoff]
            arms[name] = classifier + used
        outcome["total"] += 1
        outcome["contrasts"] += int(bool(scored))
        for name, spans in arms.items():
            success = covers(spans, truth, spaces) if truth else bool(spans)
            outcome[name] += int(success)
            if truth:
                outcome[name + "_wrong_only"] += int(bool(spans) and not success)
    return dict(summary)


def points(rows):
    maxima = sorted((max(score for _, score in uses) for r in rows if r["role"] == "calibration" and (uses := list(compare(r)))),
                    reverse=True)
    if len(maxima) < 12:
        raise ValueError("Insufficient eligible, genuine calibration rows with two measured comparison paragraphs")
    return {f"allow_{p}_percent": maxima[len(maxima) * p // 100] for p in (10, 25)}


def measurements(roster, inputs):
    records = [intake.read(path) for path in inputs]
    pin = pool.file_hash(roster)
    if any(r["roster_sha256"] != pin for r in records):
        raise ValueError("The inference stages did not read the same roster")
    by_phase = {(r["role"], r["phase"]): r["scores"] for r in records}
    roles = {role for role, _ in by_phase}
    source = intake.read(roster)
    rows = [{k: v for k, v in row.items() if k != "text"} | {
            phase: by_phase[row["role"], phase][row["sha256"]] for phase in ("classifier", "ratio")}
            for row in source.pop("rows") if row["role"] in roles]
    return source | {"rows": rows, "roster_sha256": pin}


def calibrate(roster, inputs, output):
    source = measurements(roster, inputs)
    if {r["role"] for r in source["rows"]} != {"calibration"}:
        raise ValueError("Point selection requires only calibration measurements")
    prior.write(output, {"schema": SCHEMA, "roster_sha256": source["roster_sha256"], "cutoff": points(source["rows"])})


def analyze(roster, inputs, frozen, output):
    source = measurements(roster, inputs)
    rows = source["rows"]
    cutoff = points(rows)
    if (intake.read(frozen) != {"schema": SCHEMA, "roster_sha256": source["roster_sha256"], "cutoff": cutoff}
            or {r["role"] for r in rows} != {"calibration", "held"}):
        raise ValueError("The held evaluation requires both arms and frozen points")
    receipt = source | {"cutoff": cutoff, "outcomes": evaluate(rows, cutoff)}
    review(receipt)
    prior.write(output, receipt)


def review(receipt):
    rows = receipt["rows"]
    if (receipt["schema"] != SCHEMA or receipt["method"] != METHOD
            or receipt["publishers"] != [list(row) for row in SOURCES]):
        raise ValueError("Incompatible evaluation")
    groups = defaultdict(list)
    for row in rows:
        spans, anchors, position = row["spans"], row["source_anchors"], row["position"]
        if (position not in (6, 7, 8) or row["truth"] not in (None, spans[position])
                or len(spans) != 15 or len(row["classifier"]) != 15 or len(row["ratio"]) != 15
                or not prior.SHA256.fullmatch(row["group"]) or not prior.SHA256.fullmatch(row["sha256"])
                or not all(0 <= a < b for a, b in spans + anchors) or len(anchors) != 3
                or any(a[1] + 200 > b[0] for a, b in zip(anchors, anchors[1:]))
                or any(a[1] >= b[0] for a, b in zip(spans, spans[1:]))
                or not all(math.isfinite(s) and 0 <= s <= 1 for s in row["classifier"])
                or any(type(tokens) is not int or tokens < 0 or type(expanded) is not bool
                       or reading is not None and (not math.isfinite(reading) or reading <= 0 or not tokens)
                       for reading, tokens, expanded in row["ratio"])):
            raise ValueError("Missing or invalid native passage responses")
        groups[row["group"]].append(row)
    observed = defaultdict(Counter)
    if groups.keys() & source_exclusions(receipt["earlier_panels"]):
        raise ValueError("An assessed work also appears in a pinned prior study")
    for deliveries in groups.values():
        even, = [row for row in deliveries if row["truth"] is None]
        role = even["role"]
        keys = ("role", "source_anchors", "generator", "position")
        if (len(deliveries) != (1 if role == "calibration" else 2)
                or any(row[key] != even[key] for row in deliveries for key in keys)):
            raise ValueError("Source pairs do not match or overlap partitions")
        observed[f"{role}/{even['generator']}"][str(even["position"])] += 1
    if dict(observed) != receipt["admissions"]:
        raise ValueError("Source coverage differs from the admitted roster")
    cutoff = points(rows)
    outcomes = evaluate(rows, cutoff)
    if receipt["cutoff"] != cutoff or receipt["outcomes"] != outcomes:
        raise ValueError("The replay changed the recorded decision")
    return outcomes


def worker_sample(rows):
    held = defaultdict(dict)
    for row in rows:
        if row["role"] == "held":
            held[row["generator"]].setdefault(row["group"], []).append(row)
    for works in held.values():
        for deliveries in list(works.values())[:2]:
            yield from deliveries


def worker_readout(bench, observed):
    from scripts.writing_editorial_worker_audit import covers

    planned = list(worker_sample(bench["rows"]))
    if (observed["schema"] != SCHEMA + "/worker" or observed["roster_sha256"] != bench["roster_sha256"]
            or observed["billing"] != {"simulated": True, "amount_minor": 999, "currency": "usd", "unpaid_jobs": 0}
            or len(planned) != len(observed["rows"])):
        raise ValueError("Worker run differs from the sampled held roster or simulated purchase")
    totals = defaultdict(Counter)
    verified = Counter()
    for reference, row in zip(planned, observed["rows"], strict=True):
        if (row["sha256"] != reference["sha256"] or row["group"] != reference["group"]
                or row["checks"] != {"passage_classifier": "completed", "writing_pattern_ratio": "completed"}
                or any(not 0 <= a < b <= reference["spans"][-1][1] for card in row["cards"] for a, b in card["spans"])):
            raise ValueError("A stored report differs from its held source or lacks both checks")
        truth = reference["truth"]
        tally = totals[f"{reference['generator']}/{'rewritten' if truth else 'ordinary'}"]
        spans = [span for card in row["cards"] for span in card["spans"]]
        tally["total"] += 1
        tally["actual"] += int(covers(spans, truth, set(reference["spaces"])) if truth else bool(spans))
        tally["inconclusive"] += int(row["decision"] == "inconclusive")
        if row["signed_source_replay"]:
            if truth is None:
                raise ValueError("The replay was planned on a rewritten test case")
            verified[reference["generator"]] += 1
    if verified != {row["generator"]: 1 for row in planned}:
        raise ValueError("Each held generator requires one signed source replay")
    return dict(totals)


def run_worker(roster, benchmark, output):
    from scripts import video_short_return_encoded_app as paid
    from scripts.writing_context_pool_worker import check_report
    from scripts.writing_revision_worker_transfer import assess, upload

    bench = intake.read(benchmark)
    review(bench)
    if pool.file_hash(roster) != bench["roster_sha256"]:
        raise ValueError("The supplied source roster differs from the benchmark")
    cases = list(worker_sample(intake.read(roster)["rows"]))
    paid.configure(output, pool.ROOT)
    os.environ.update(
        VERITAS_DEVICE="mps", VERITAS_RUN_WORKER="false", VERITAS_ENVIRONMENT="dev", VERITAS_LOG_LEVEL="WARNING",
        VERITAS_TEXT_DETECTOR_MODE="candidate", VERITAS_TEXT_WINDOW_SENTENCES="3", VERITAS_WEBHOOKS_ENABLED="false",
        VERITAS_TEXT_ISOLATED_PARAGRAPH_REVIEW="true", VERITAS_TEXT_PARAGRAPH_EVIDENCE_PRIORITY="true",
        VERITAS_TEXT_MATCHED_CONTEXT_REVIEW="true", VERITAS_TEXT_CHECKPOINT_AGREEMENT_REVIEW="true",
        VERITAS_TEXT_PASSAGE_POSITION_REVIEW="true", VERITAS_TEXT_LONG_PASSAGE_POSITION_REVIEW="true",
        VERITAS_RATE_LIMIT_PER_MINUTE="1800", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
    )
    import torch
    from fastapi.testclient import TestClient

    from app.api import auth, jobs
    from app.billing import service
    from app.core.config import settings
    from app.detectors.text.extract import from_string
    from app.detectors.text.pipeline import windows
    from app.detectors.text.segment import segment
    from app.main import app
    from app.storage.db import get_session
    from scripts.verify_multimedia_exports import scan

    if (not torch.backends.mps.is_available() or settings.data_dir != output.resolve() / "app"
            or settings.run_worker or not settings.offline):
        raise RuntimeError("Use the isolated offline Apple GPU worker")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    service._stripe = stub = paid.PaymentStub()
    verified, rows = set(), []
    with TestClient(app, client=("127.0.0.1", 50000)) as client:
        with get_session() as db:
            principal = auth.new_principal("local-paragraph-control-audit", "examiner")
            db.add(principal)
            db.flush()
            client.headers["X-API-Key"], _ = auth.issue_api_key(db, principal)
            db.commit()
        worker = jobs.build_worker()
        for index, row in enumerate(cases):
            parts = segment(from_string(row["text"]))
            case = row | {"record_id": str(index), "text_sha256": row["sha256"], "context": windows(parts, 3),
                          "ranges": [{"start": a, "end": b} for a, b in row["spans"]]}
            aid = upload(client, stub, case, worker)
            actual = assess(client, case, aid, export=True)
            evidence = paid.body(client.get(f"/api/v2/analyses/{aid}/evidence-packets"))
            scan(json.dumps(evidence))
            generator = row["generator"]
            signature = row["truth"] is not None and generator not in verified
            if signature:
                source = output / "submitted.txt"
                source.write_text(row["text"], encoding="utf-8")
                check_report(client, worker, aid, source)
                verified.add(generator)
            rows.append({"sha256": row["sha256"], "group": row["group"], "cards": actual["packets"],
                         "decision": actual["document_decision"], "signed_source_replay": signature,
                         "checks": evidence["coverage"]["writing_check_coverage"]["checks"]})
            print("Verified simulated prepaid report", index + 1, "of", len(cases), flush=True)
    receipt = {"schema": SCHEMA + "/worker", "roster_sha256": bench["roster_sha256"],
               "billing": {"simulated": True, "amount_minor": 999, "currency": "usd", "unpaid_jobs": 0}, "rows": rows}
    receipt["outcomes"] = worker_readout(bench, receipt)
    prior.write(output / "receipt.json", receipt)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "classifier", "ratio", "calibrate", "analyze", "worker", "review"))
    parser.add_argument("path", type=Path)
    parser.add_argument("--shards", type=Path)
    parser.add_argument("--calibration-shards", type=Path)
    parser.add_argument("--earlier-panels", type=Path, help="Reuse the pinned exclusions from an existing numeric receipt.")
    parser.add_argument("--role", choices=("calibration", "held"))
    parser.add_argument("--parts", nargs="+", type=Path)
    parser.add_argument("--frozen", type=Path)
    parser.add_argument("--benchmark", type=Path)
    parser.add_argument("--worker-receipt", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.phase == "freeze":
        previous = intake.read(args.earlier_panels)["earlier_panels"] if args.earlier_panels else earlier_panels()
        freeze(args.shards, args.calibration_shards, args.path, previous=previous)
    elif args.phase in {"classifier", "ratio"}:
        readings(args.path, args.role, args.phase, args.output)
    elif args.phase == "calibrate":
        calibrate(args.path, args.parts, args.output)
    elif args.phase == "analyze":
        analyze(args.path, args.parts, args.frozen, args.output)
    elif args.phase == "worker":
        run_worker(args.path, args.benchmark, args.output)
    else:
        bench = intake.read(args.path)
        result = {"benchmark": review(bench)}
        if args.worker_receipt:
            observed = intake.read(args.worker_receipt)
            result["worker"] = worker_readout(bench, observed)
            if result["worker"] != observed["outcomes"]:
                raise ValueError("Replayed worker counts differ")
        print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
