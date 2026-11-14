"""Check whether existing flagged speech ranges can be narrowed on the source clock."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from scripts import audio_shared_mel_context as shared

sine, source, stock = shared.sine, shared.source, shared.stock
PANELS = source.PANELS
PARENT = PANELS / "audio-shared-mel-context-2026-11-14.json"
SCHEMA = "chipmunk.audio-narrow-flag-transfer/1"
SHARDS = (0, 9, 31)
VIEWS = ("local", "shared_scale", "agreement")
METHOD = {
    "cases": "Three previously scored banks for retrospective comparison. For additional confirmation take one new "
             "publisher-labeled 0.20–1.00 s cut-and-paste edit per previously unused SINE reader from three pinned archive prefixes. "
             "Retain one six-second original/edit pair per reader. Match each pair clean and at −5 dB with previously used AMI rooms.",
    "choose": "Keep the existing five-window review point. Consider only one-second windows wholly contained in the "
              "already flagged two-second range. Each normalization route must exceed its own earlier development "
              "point. The agreement route changes the displayed range only if both routes choose the same interval. "
              "Otherwise retain the broad range. No route creates an additional file flag.",
    "assessment": "Require 0.30 temporal intersection over union and 50% recall of the labeled edit. Count gains, "
                  "losses, incorrect refinements, ordinary refinements, and unchanged dispositions. The latest "
                  "speaker-disjoint confirmation shares editor lineage and background sources with earlier work.",
}


def freeze(output, cache):
    """Freeze selected publisher data without loading a detector."""
    parent = source.read(shared.SINE_PARENT)
    lookup = sine.source_listing(cache)
    used = {row["reader"] for row in parent["freeze"]["pairs"]} | {"100"}
    eligible, shard_pins = [], []
    for shard in SHARDS:
        members, labels, pin = sine.shard_members(cache, shard, lookup[f"data/split-{shard}.tar.gz"])
        shard_pins.append(pin)
        for key in members["real"].keys() & members["cut_paste"].keys():
            reader = key.split("-", 1)[0]
            if reader in used:
                continue
            body = {"original": members["real"][key], "variant": members["cut_paste"][key]}
            extent = {name: sine.wav(raw) for name, raw in body.items()}
            limit = min(value[0] for value in extent.values()) - 6 * sine.sine.RATE
            if limit < 0:
                continue
            clock, label = labels["medium_cut_paste.txt"][f"dev_cut_paste_medium-{key}"]
            source.check(label == "0", "Publisher edit label differs")
            intervals, _ = sine.sine.label_clock(clock, extent["variant"][0], "cut_paste")
            a, b, _ = intervals[1]
            if not 4800 <= b - a <= 24000:
                continue
            starts = range(0, limit + 1, 2400)
            valid = [t for t in starts if t <= a < b <= t + 144000]
            if not valid:
                continue
            tick = min(valid, key=lambda t: source.digest(f"speech-refinement-clock/1/{key}/{t}".encode()))
            row = {"reader": reader, "stem": key, "shard": shard, "role": "cut_paste", "native_start": tick,
                   "truth": [(a - tick) * 2 // 3, (b - tick) * 2 // 3], "publisher_cut_paste_clock": clock,
                   "files": {name: {"wav_sha256": source.digest(raw), "samples": extent[name][0],
                                    "pcm_sha256": extent[name][1]} for name, raw in body.items()}}
            eligible.append((source.digest(f"speech-refinement-source/1/{key}".encode()), row, body))
    backgrounds = parent["freeze"]["backgrounds"]
    pairs = []
    output.mkdir(mode=0o700)
    (output / "sine").mkdir()
    for _, row, bodies in sorted(eligible, key=lambda item: (item[0], item[1]["shard"])):
        if row["reader"] in used:
            continue
        used.add(row["reader"])
        index = len(pairs)
        for role, payload in bodies.items():
            (output / "sine" / f"{index:03}-{role}.wav").write_bytes(payload)
        pairs.append(row | {"id": index, "room": backgrounds[index % len(backgrounds)]["room"]})
    source.check(len(pairs) >= 12 and len({row["shard"] for row in pairs}) == len(SHARDS), "Incomplete unused-reader panel")
    locked = {"schema": SCHEMA, "method": METHOD, "parent_sha256": source.path_sha(PARENT),
              "source_parent_sha256": source.path_sha(shared.SINE_PARENT), "runner_sha256": source.path_sha(Path(__file__)),
              "shared_reader_sha256": source.path_sha(Path(shared.__file__)), "publisher_revision": sine.sine.REVISION,
              "prefix_bytes": sine.PREFIX, "shards": shard_pins, "pairs": pairs, "backgrounds": backgrounds}
    source.write(output / "freeze.json", locked)
    print(f"Froze {len(pairs)} previously unused SINE readers", flush=True)


def refine(broad, views, points):
    if broad is None:
        return dict.fromkeys(VIEWS)
    result = {}
    for arm in VIEWS[:2]:
        eligible = [(score, -start) for start, score in zip(shared.STARTS, views[arm], strict=True)
                    if score is not None and score > points[arm] and broad[0] <= start <= broad[1] - shared.WIDTH]
        best = max(eligible, default=None)
        result[arm] = [-best[1], -best[1] + shared.WIDTH] if best else None
    result["agreement"] = result["local"] if result["local"] == result["shared_scale"] else None
    return result


def tally(rows, points):
    counts, changed = {arm: Counter() for arm in VIEWS}, {arm: [] for arm in VIEWS}
    for row, broad in rows:
        narrow = refine(broad, row["views"], points)
        truth = row["truth"]
        kind = ("edited" if truth else "isolated" if row["kind"] in shared.interval.BACKGROUNDS else
                "same_words" if row["kind"] == "resyn" else "ordinary")
        for arm in VIEWS:
            shown = narrow[arm] or broad
            reading = counts[arm]
            before = bool(truth and broad and stock.useful(broad, truth))
            after = bool(truth and shown and stock.useful(shown, truth))
            reading.update({kind: 1, f"{kind}_flagged": bool(broad), f"{kind}_narrowed": narrow[arm] is not None,
                            "useful_broad": before, "useful_displayed": after, "timing_gains": after and not before,
                            "timing_losses": before and not after,
                            "wrong_narrowing": bool(truth and narrow[arm] and not after)})
            if narrow[arm]:
                changed[arm].append({"kind": kind, "group": row["group"], "pcm_sha256": row["pcm_sha256"],
                                     "truth": truth, "broad": list(broad), "narrow": shown,
                                     "useful_broad": before, "useful_displayed": after})
    return {arm: {"counts": dict(counts[arm]), "changed": changed[arm]} for arm in VIEWS}


def validate_native(lock, rows):
    parent = source.read(shared.SINE_PARENT)
    source.check(lock["schema"] == SCHEMA and lock["method"] == METHOD and lock["parent_sha256"] == source.path_sha(PARENT)
                 and lock["source_parent_sha256"] == source.path_sha(shared.SINE_PARENT)
                 and lock["backgrounds"] == parent["freeze"]["backgrounds"] and lock["publisher_revision"] == sine.sine.REVISION
                 and lock["prefix_bytes"] == sine.PREFIX and [row["shard"] for row in lock["shards"]] == list(SHARDS),
                 "Native input freeze differs")
    pairs, backgrounds = lock["pairs"], lock["backgrounds"]
    seen = {row["reader"] for row in parent["freeze"]["pairs"]} | {"100"}
    source.check(len(pairs) >= 12 and len(rows) == 4 * len(pairs) + len(backgrounds), "Incomplete confirmation panel")
    for index, row in enumerate(pairs):
        a, b = row["truth"]
        source.check(row["id"] == index and row["reader"] not in seen and row["stem"].split("-", 1)[0] == row["reader"]
                     and row["room"] == backgrounds[index % len(backgrounds)]["room"] and row["shard"] in SHARDS
                     and 0 <= a < b <= sine.SAMPLES and 3200 <= b - a <= 16000, "Invalid source geometry")
        start = row["native_start"]
        clock, _ = sine.sine.label_clock(row["publisher_cut_paste_clock"], row["files"]["variant"]["samples"], "cut_paste")
        low, high, _ = clock[1]
        source.check([(low - start) * 2 // 3, (high - start) * 2 // 3] == [a, b]
                     and start % 2400 == 0 and all(start + 144000 <= f["samples"] for f in row["files"].values()),
                     "Edit label does not map to the published source clock")
        seen.add(row["reader"])
    expected = [(p["id"], c, s, p["truth"] if s == "variant" else None)
                for p in pairs for c in sine.TREATMENTS for s in ("original", "variant")]
    expected += [(i, "competing_speech", bg["room"], None) for i, bg in enumerate(backgrounds)]
    for row, identity in zip(rows, expected, strict=True):
        source.check((row["group"], row["condition"], row["side"], row["truth"]) == identity
                     and row["kind"] == ("background_only" if row["side"] not in ("original", "variant")
                                         else "cut_paste" if row["truth"] else "untouched")
                     and stock.sha256(row["pcm_sha256"]) and len(row["existing_scores"]) == len(stock.WINDOWS),
                     "Invalid measured clip identity")
        base = shared.existing(row)
        source.check(set(row["views"]) == set(VIEWS[:2])
                     and all(len(values) == len(shared.STARTS) for values in row["views"].values())
                     and all((a is None) == (b is None) and (base is not None or a is None)
                             for a, b in zip(*row["views"].values(), strict=True)), "Unexpected window coverage")
    source.check(len({row["pcm_sha256"] for row in rows}) == len(rows), "Native deliveries repeat")


def summary(receipt):
    parent = source.read(PARENT)
    points = parent["points"]
    source.check(receipt["points"] == points, "Existing operating points changed")
    rows = receipt["records"]
    validate_native(receipt["freeze"], rows)
    results = {name: tally([(row, shared.existing(row)) for row in candidates], points)
               for name, candidates in parent["panels"].items() if name != "development"}
    results.update({name: tally([(row, shared.existing(row)) for row in rows if row["condition"] == name], points)
                    for name in sine.TREATMENTS})
    return results


def infer(folder):
    import numpy as np

    from app.detectors.audio.calibration import logit
    from app.detectors.audio.content import window_in_speech
    from app.detectors.audio.upgraded import SpeechActivity
    from scripts.audio_strong_traffic_view import network_free
    from scripts.prepare_audio_specialists_v15 import ASSETS
    from scripts.run_audio_specialists_v15 import load_voice

    frozen = source.read(folder / "freeze.json")
    previous = source.read(PARENT)
    source.check(frozen["runner_sha256"] == source.path_sha(Path(__file__))
                 and frozen["shared_reader_sha256"] == source.path_sha(Path(shared.__file__))
                 and previous["freeze"]["asset_hashes"] == ASSETS, "Scoring implementation or model changed")
    network_free()
    weights = source.ROOT / "models/weights/audio-upgrade"
    model = load_voice({"weights": weights, "assets": ASSETS})
    activity = SpeechActivity(weights / "silero_vad.jit")
    records = []
    for native, pcm in sine.native_rows(folder, frozen):
        row = {"group": native["id"], "kind": ("background_only" if native["role"] == "background_only"
                                               else "cut_paste" if native["truth"] else "untouched"),
               **{key: native[key] for key in ("side", "condition", "truth")},
               "pcm_sha256": source.digest(pcm.tobytes())}
        values = pcm.astype(np.float32) / 32768
        spans = activity.spans(values)
        row["existing_scores"] = [logit(model.predict([values[a:b]])[0]) if window_in_speech(spans, a / 16000, b / 16000) else None
                                  for a, b in stock.WINDOWS]
        views = dict.fromkeys(VIEWS[:2], [None] * len(shared.STARTS))
        if shared.existing(row):
            readouts, _ = shared.windows(model, values, spans)
            views = {arm: readouts[arm] for arm in VIEWS[:2]}
        records.append(row | {"views": views})
        if len(records) % 12 == 0:
            print(f"Scored {len(records)} native deliveries", flush=True)
    receipt = {"freeze": frozen, "records": records, "points": previous["points"]}
    receipt["summary"] = summary(receipt)
    source.write(folder / "receipt.json", receipt)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "score", "review"))
    parser.add_argument("path", type=Path)
    parser.add_argument("--cache", type=Path)
    args = parser.parse_args()
    if args.phase == "freeze":
        if args.cache is None:
            parser.error("--cache is required")
        freeze(args.path, args.cache)
    elif args.phase == "score":
        infer(args.path)
    else:
        receipt = source.read(args.path)
        reproduced = summary(receipt)
        source.check(reproduced == receipt["summary"], "Saved comparison does not replay")
        print(json.dumps({k: {a: item["counts"] for a, item in v.items()} for k, v in reproduced.items()}, sort_keys=True))


if __name__ == "__main__":
    main()
