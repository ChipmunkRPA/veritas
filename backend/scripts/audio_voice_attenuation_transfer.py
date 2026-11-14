"""Evaluate whether localized attenuation identifies short synthetic speech edits."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

from scripts import review_audio_edge_window_grouped_transfer as grouped
from scripts import review_audio_stock_voice_transfer as stock

ROOT = Path(__file__).resolve().parents[2]
PANELS = ROOT / "docs/evaluation-panels"
STAMP = "2026-11-13"
SCHEMA = "chipmunk.audio-voice-attenuation/1"
PLAN = f"audio-voice-attenuation-plan-{STAMP}.json"
ROLES = ("development", "random_placement", "three_generators")
BACKGROUND = {"background", "background_only", "isolated"}
ARMS = ("drop", "contrast")
RATE = stock.RATE
FADE_SAMPLES = 640
SOURCE_CHECKS = (0, 7, 21, 39)


def load(path):
    return json.loads(path.read_bytes())


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def top_slots(base):
    return sorted((i for i, value in enumerate(base) if value is not None), key=lambda i: (-base[i], i))[:2]


def incumbent(row):
    slots = top_slots(row["base"])
    if slots and row["base"][slots[0]] > stock.SHORT_POINT:
        return stock.WINDOWS[slots[0]]
    return None


def evaluate_halves(row):
    drop, contrast = [], []
    for view in row["views"]:
        slot = view["slot"]
        for half, value in enumerate(view["scores"]):
            loss = row["base"][slot] - value
            difference = view["scores"][1 - half] - value
            start = (slot + half) * RATE
            span = start, start + RATE
            drop.append((loss, -start, span))
            if difference > 0:
                contrast.append(((loss + difference) / 2, -start, span))
    return {name: max(readings, default=None) for name, readings in zip(ARMS, (drop, contrast))}


def points(rows):
    per_arm = {arm: {} for arm in ARMS}
    for row in rows:
        if row["truth"] or incumbent(row):
            continue
        for arm, result in evaluate_halves(row).items():
            if result is None:
                continue
            key = "background" if row["kind"] in BACKGROUND else row["group"]
            per_arm[arm][key] = max(per_arm[arm].get(key, -math.inf), result[0])
    fitted = {}
    for arm, controls in per_arm.items():
        background = controls.pop("background")
        fitted[arm] = max(sorted(controls.values(), reverse=True)[1], background)
    return fitted


def tally(rows, arm=None, point=None, *, additive=False):
    counts, locations, false_groups = Counter(), [], set()
    for row in rows:
        truth = row["truth"]
        kind = "edited" if truth else "background" if row["kind"] in BACKGROUND else "ordinary"
        counts[kind] += 1
        current = incumbent(row)
        span = current
        if arm and not (additive and current):
            reading = evaluate_halves(row)[arm]
            span = reading[2] if reading and reading[0] > point else None
        if span is None:
            counts["unassessed"] += not row["views"]
            counts["missed"] += kind == "edited"
            continue
        if truth:
            outcome = "useful" if stock.useful(span, truth) else "misplaced"
        else:
            outcome = "flagged_" + kind
            if kind == "ordinary":
                false_groups.add(row["group"])
        counts[outcome] += 1
        if span != current:
            locations.append({"group": row["group"], "kind": row["kind"], "outcome": outcome, "range": list(span)})
    return dict(counts, false_ordinary_groups=len(false_groups), changed_locations=locations)


def outcomes(panels, fitted):
    results = {}
    for role, rows in panels.items():
        results[role] = {"existing": tally(rows)}
        for arm, point in fitted.items():
            results[role][arm] = {"standalone": tally(rows, arm, point), "added_to_existing": tally(rows, arm, point, additive=True)}
    return results


def advancement(results):
    outcome = {}
    for arm in ARMS:
        selected = [results[role][arm]["added_to_existing"] for role in ROLES[1:]]
        families = {hit["kind"] for panel in selected for hit in panel["changed_locations"] if hit["outcome"] == "useful"}
        adds_enough = sum(hit["outcome"] == "useful" for panel in selected for hit in panel["changed_locations"]) >= 2
        controlled = all(
            candidate["false_ordinary_groups"] <= results[role]["existing"]["false_ordinary_groups"] + 1
            and candidate.get("flagged_background", 0) == results[role]["existing"].get("flagged_background", 0)
            for role, candidate in zip(ROLES[1:], selected)
        )
        outcome[arm] = adds_enough and len(families) >= 2 and controlled
    return outcome


def prior_path(role):
    names = ("audio-stock-voice-transfer", "audio-random-clock-stock", "audio-edge-window-grouped")
    date = STAMP if role != ROLES[2] else "2026-11-12"
    return PANELS / f"{names[ROLES.index(role)]}-{date}.json"


def prior_rows(role):
    source = load(prior_path(role))
    if role == ROLES[0]:
        return [(row["group"], row["kind"], row["truth"], row["pcm_sha256"], row["scores"])
                for row in source["rows"] if row["condition"] in {"-5dB", "background"}]
    if role == ROLES[1]:
        return [(row["group"], row["kind"], row["edit_clock"], row["pcm_sha256"], row["original_scores"])
                for row in source["rows"] if row["snr"] == -5 or row["kind"] == "isolated"]
    grouped.verify_held(source["held"])
    return [(row["group"] - 1, row["family"] if row["truth"] else row["arm"], row["truth"], None, row["scores"][2:])
            for row in source["held"]]


def validate(panels):
    commitments = set()
    for role, count in zip(ROLES, (44, 43, 50)):
        rows = panels[role]
        expected = prior_rows(role)
        stock.require(len(rows) == len(expected) == count, "Incomplete evaluation panel")
        for row, (group, kind, truth, pcm, base) in zip(rows, expected):
            pin = row["pcm_sha256"]
            stock.require(set(row) == {"group", "kind", "truth", "pcm_sha256", "base", "views"}
                          and (row["group"], row["kind"], row["truth"], row["base"]) == (group, kind, truth, base)
                          and (pcm is None or pin == pcm) and stock.sha256(pin) and pin not in commitments,
                          "Source roles, waveform hashes or original response changed")
            commitments.add(pin)
            stock.require([view["slot"] for view in row["views"]] == top_slots(base), "Incorrect source-window selection")
            for view in row["views"]:
                stock.require(set(view) == {"slot", "scores", "input_hashes"} and len(view["scores"]) == len(view["input_hashes"]) == 2
                              and all(type(score) in (float, int) and math.isfinite(score) for score in view["scores"])
                              and all(stock.sha256(digest) for digest in view["input_hashes"]), "Invalid attenuation response")


def signals(role, folder):
    import numpy as np

    source = load(prior_path(role))
    if role == ROLES[2]:
        bank, scores = (folder / name for name in ("audio-edge-window-grouped-held-2026-11-12",
                                                  "audio-edge-window-grouped-2026-11-12"))
        files = {"held_native_scores": scores / "held.json", "freeze": scores / "freeze.json",
                 "held_sources": bank / "sources.json"}
        pins = source["commitments"]
        stock.require(all(sha(path.read_bytes()) == pins[key] for key, path in files.items()), "Grouped reference changed")
        stock.require(sha((bank / "roster.json").read_bytes()) == load(bank / "sources.json")["roster_sha256"],
                      "Grouped source roster changed")
        raw = [(bank / "pcm" / f"{r['id']}.pcm").read_bytes() for r in load(bank / "roster.json")]
        records = load(scores / "held.json")["records"]
        expected = source["held"]
        for signal, record, published in zip(raw, records, expected, strict=True):
            stock.require(len(signal) == 12 * RATE and sha(signal) == record["pcm_sha256"]
                          and [w["score"] for w in record["windows"]] == published["scores"]
                          and all(sha(signal[2*w["start"]:2*w["end"]]) == w["pcm_sha256"] for w in record["windows"]),
                          "Grouped native input or scores differ")
        return dict(zip((sha(data) for data in raw), raw, strict=True))
    dev = role == ROLES[0]
    manifest = (folder / ("freeze.json" if dev else "placements-freeze.json")).read_bytes()
    stock.require(sha(manifest) == source["native_freeze_sha256"], "Stock source roster changed")
    roster = json.loads(manifest)
    if dev:
        signal = np.load(folder / "signals.npy", mmap_mode="r", allow_pickle=False)
        rows = roster["cases"]
    else:
        with np.load(folder / "placements-native.npz", allow_pickle=False) as archive:
            signal = archive["native"]
        rows = roster["native_manifest_rows"]
    result = {}
    for row, clip in zip(rows, signal, strict=True):
        selected = row["condition"] in {"-5dB", "background"} if dev else row["snr"] == -5 or row["kind"] == "isolated"
        if selected:
            raw = clip.astype("<i2", copy=False).tobytes()
            stock.require(clip.shape == (6 * RATE,) and sha(raw) == row["pcm_sha256"], "Stock native input differs")
            result[row["pcm_sha256"]] = raw
    return result


def attenuated_views(signal, slot):
    import numpy as np

    envelope = np.full(RATE, .25, np.float32)
    envelope[:FADE_SAMPLES] = .25 + .75 * (1 + np.cos(np.linspace(0, np.pi, FADE_SAMPLES))) / 2
    envelope[-FADE_SAMPLES:] = envelope[:FADE_SAMPLES][::-1]
    start, end = stock.WINDOWS[slot]
    original = signal[start:end]
    variants = []
    for half in (0, 1):
        amended = original.copy()
        amended[half * RATE:(half + 1) * RATE] *= envelope
        variants.append(amended)
    return original, variants


def input_hashes(variants):
    return [sha(signal.astype("<f4", copy=False).tobytes()) for signal in variants]


def measure(role, folder, output):
    import numpy as np

    from app.detectors.audio import calibration
    from scripts.prepare_audio_specialists_v15 import ASSETS
    from scripts.run_audio_specialists_v15 import load_voice

    waveforms = signals(role, folder)
    source = prior_rows(role)
    if role == ROLES[2]:
        source = [(group, kind, truth, key, base) for (group, kind, truth, _, base), key in zip(source, waveforms, strict=True)]
    model = load_voice({"weights": str(ROOT / "models/weights/audio-upgrade"), "assets": ASSETS})
    with output.open("x") as stream:
        for index, (group, kind, truth, commitment, base) in enumerate(source):
            signal = np.frombuffer(waveforms[commitment], "<i2").astype(np.float32) / 32768
            views = []
            for slot in top_slots(base):
                original, variants = attenuated_views(signal, slot)
                if index in SOURCE_CHECKS:
                    probability, = model.predict([original])
                    stock.require(abs(calibration.logit(probability) - base[slot]) < .003, "Original model response changed")
                views.append({"slot": slot, "scores": [calibration.logit(p) for p in model.predict(variants)],
                              "input_hashes": input_hashes(variants)})
            stream.write(json.dumps({"group": group, "kind": kind, "truth": truth, "pcm_sha256": commitment,
                                     "base": base, "views": views}, separators=(",", ":")) + "\n")
            stream.flush()


def audit_inputs(receipt, role, folder):
    import numpy as np

    waveforms = signals(role, folder)
    rows = receipt["rows"][role]
    stock.require(set(waveforms) == {row["pcm_sha256"] for row in rows}, "Waveform audit roster differs")
    if role == ROLES[2]:
        stock.require(list(waveforms) == [row["pcm_sha256"] for row in rows], "Grouped native row order differs")
    count = 0
    for row in rows:
        signal = np.frombuffer(waveforms[row["pcm_sha256"]], "<i2").astype(np.float32) / 32768
        for view in row["views"]:
            original, variants = attenuated_views(signal, view["slot"])
            stock.require(all(np.array_equal(chunk[(1-half)*RATE:(2-half)*RATE], original[(1-half)*RATE:(2-half)*RATE])
                              for half, chunk in enumerate(variants))
                          and input_hashes(variants) == view["input_hashes"], "An attenuated input differs")
            count += len(variants)
    return {"files": len(rows), "attenuated_windows": count}


def review(receipt):
    stock.require(receipt["schema"] == SCHEMA and receipt["plan_sha256"] == sha((PANELS / PLAN).read_bytes()),
                  "Evaluation plan changed")
    stock.require(receipt["source_receipts"] == {role: sha(prior_path(role).read_bytes()) for role in ROLES},
                  "Source receipts changed")
    panels = receipt["rows"]
    validate(panels)
    fitted = points(panels[ROLES[0]])
    stock.require(fitted == receipt["points"], "Operating points do not replay")
    results = outcomes(panels, fitted)
    stock.require(results == receipt["outcomes"] and advancement(results) == receipt["advances"],
                  "Reported outcome does not replay")
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("measure", "seal", "assemble", "review", "audit-inputs"))
    parser.add_argument("output", type=Path)
    parser.add_argument("--role", choices=ROLES)
    parser.add_argument("--folder", type=Path)
    parser.add_argument("--points", type=Path)
    parser.add_argument("--rows", nargs="+", type=Path)
    args = parser.parse_args()
    if args.phase in {"measure", "audit-inputs"} and (args.role is None or args.folder is None):
        parser.error("Supply --role and --folder for native source access.")
    if args.phase == "measure":
        measure(args.role, args.folder, args.output)
        return
    if args.phase == "audit-inputs":
        receipt = load(args.output)
        review(receipt)
        print(json.dumps(audit_inputs(receipt, args.role, args.folder), sort_keys=True))
        return
    if args.phase == "review":
        print(json.dumps(review(load(args.output)), indent=2, sort_keys=True))
        return
    roles = ROLES[:1] if args.phase == "seal" else ROLES
    if len(args.rows or ()) != len(roles) or args.phase == "assemble" and args.points is None:
        parser.error("Supply the row files in phase order and the sealed points for assembly.")
    plan = sha((PANELS / PLAN).read_bytes())
    rows = {key: [json.loads(line) for line in path.read_text().splitlines()] for key, path in zip(roles, args.rows, strict=True)}
    fitted = points(rows[ROLES[0]])
    if args.phase == "seal":
        output = {"plan_sha256": plan, "points": fitted}
    else:
        stock.require(load(args.points) == {"plan_sha256": plan, "points": fitted}, "Operating point freeze changed")
        results = outcomes(rows, fitted)
        output = {"schema": SCHEMA, "plan_sha256": plan, "points": fitted, "rows": rows, "outcomes": results,
                  "source_receipts": {role: sha(prior_path(role).read_bytes()) for role in ROLES},
                  "advances": advancement(results)}
        review(output)
    with args.output.open("x") as stream:
        json.dump(output, stream, allow_nan=False, separators=(",", ":"), sort_keys=True)
        stream.write("\n")


if __name__ == "__main__":
    main()
