# SPDX-License-Identifier: MIT
"""Render and replay a two-voice factorial localization benchmark."""

from __future__ import annotations

import argparse
import gzip
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from scripts import audio_short_edit_detector_training as phoneme
from scripts import audio_stream_attenuation as prior

SOURCE = phoneme.SOURCE
SCORING = prior.SCORING
RATE = phoneme.RATE
SAMPLES = phoneme.SAMPLES
PHONEME = prior.PARENT.with_name("audio-short-edit-detector-training-2026-11-14.json")
SCHEMA = "chipmunk.audio-two-voice-components/1"
METHOD = {
    "panels": {"development": "surrey", "phoneme_transfer": "bbc", "stock_transfer": "bbc"},
    "render": "Replay source commitments, assign speaker pairs by hash within each role, and position two exact "
    "edits at staggered (1.45 and 3.65 s) or overlapping (2.25 and 2.40 s) clocks. Alternate room directions. "
    "The second voice is 5 dB louder on the untouched mono render. Gains stay fixed across all four combinations "
    "and the untouched control. Ground truth allows the measured direct-path arrival, not the full room tail.",
    "measure": "Use the same pinned one-second classifier and speech gate on the mono mean and each anonymous "
    "stereo directional estimate. Retain the closest known source only for scoring attribution. Measure two "
    "rules: strongest estimate and a same-clock lead of at least log(2) over every other eligible estimate. "
    "For each rule and for mono, use the largest development score on either genuine condition; keep held cutoffs fixed.",
    "control": "Neither means two matched ordinary edits, in addition to a separately measured pair with neither "
    "edit applied. If a round-trip resampling is byte-identical, use the frozen same-reader edit instead. "
    "Measure file and window errors separately. A useful location covers half the known edit "
    "and reaches 0.30 temporal overlap. If both voices changed, score single-window recovery as any or all "
    "known intervals. Score attribution only for estimated streams and require a useful interval on the favored source.",
}
LOCATIONS = {"staggered": (23200, 58400), "overlapping": (36000, 38400)}
CONDITIONS = {"untouched": ("untouched", "untouched"), "neither": ("genuine", "genuine"),
              "foreground": ("synthetic", "genuine"), "background": ("genuine", "synthetic"),
              "both": ("synthetic", "synthetic")}
SCORES = ("mono", "directional", "directional_lead")
MIN_LEAD = math.log(2)


def rank(value):
    return SOURCE.digest(f"{SCHEMA}/{value}".encode())


def contract():
    return {"schema": SCHEMA, "method": METHOD, "phoneme_sha256": SOURCE.path_sha(PHONEME),
            "stock_sha256": SOURCE.path_sha(prior.STOCK), "classifier": SCORING._lock("ami")}


def phoneme_clips(root):
    plan, _ = phoneme.load_plan(root)
    published = SOURCE.read(PHONEME)["sources"]
    SOURCE.check(all(plan[key] == value for key, value in published.items()), "PhonemeDF source plan differs")
    rows = {(row["group"], row["kind"]): row for row in plan["rows"] if row["condition"] == "dry"}
    members = defaultdict(list)
    for source in plan["sources"]:
        members[source["reader_sha256"]].append(source)
    fitting = sorted((speaker for speaker, clips in members.items() if clips[0]["role"] == "fit"), key=rank)
    held = sorted(set(members) - set(fitting), key=rank)[:4]

    def clip(source, family):
        import numpy as np

        group = source["group"]
        variants = {"untouched": "untouched", "genuine": f"resampled/{family}", "synthetic": f"edited/{family}"}
        if rows[group, variants["genuine"]]["pcm_sha256"] == rows[group, "untouched"]["pcm_sha256"]:
            variants["genuine"] = "genuine_splice"
        hashes = {role: rows[group, kind]["pcm_sha256"] for role, kind in variants.items()}
        samples = {}
        for role, digest in hashes.items():
            raw = (root / "pcm" / f"{digest}.pcm").read_bytes()
            SOURCE.check(len(raw) == SAMPLES * 2 and SOURCE.digest(raw) == digest, "PhonemeDF PCM differs")
            samples[role] = np.frombuffer(raw, "<i2")
        return {"speaker": source["reader_sha256"], "group": group, "family": family, "original_span": source["truth"],
                "control": variants["genuine"], "variants": hashes, "samples": samples}

    for cohort, speakers in (("development", fitting), ("phoneme_transfer", held)):
        for pair_index, pair in enumerate(zip(speakers[::2], speakers[1::2], strict=True)):
            choices = []
            for speaker in pair:
                ordered = sorted(members[speaker], key=lambda item: rank(item["group"]))
                choices.append([item for longer in (False, True) for item in
                                [r for r in ordered if (r["truth"][1] - r["truth"][0] >= 9600) == longer][:2]])
            for index, sources in enumerate(zip(*choices, strict=True)):
                family_order = phoneme.sources.FAMILIES[::1 if (index // 2 + pair_index) % 2 == 0 else -1]
                yield cohort, pair_index, index, [clip(source, family) for source, family in
                                                   zip(sources, family_order, strict=True)]


def stock_clips(root):
    import numpy as np

    public = SOURCE.read(prior.STOCK)
    prior.stock.verify(public)
    path = root / "placements-freeze.json"
    SOURCE.check(SOURCE.path_sha(path) == public["native_freeze_sha256"], "Stock source plan differs")
    plan = SOURCE.read(path)
    with np.load(root / "placements-native.npz", allow_pickle=False) as data:
        dry = data["dry"]
    SOURCE.check(dry.dtype == np.dtype("<i2") and dry.shape == (len(public["rows"]), SAMPLES), "Stock clock differs")
    items = {}
    for samples, row in zip(dry, public["rows"], strict=True):
        if row["snr"] == -5:
            digest = row["dry_pcm_sha256"]
            SOURCE.check(SOURCE.digest(samples.tobytes()) == digest, "Stock source audio differs")
            items[row["group"], row["kind"]] = row, samples
    speakers = {ix: phoneme.sources.order(str(item["reader"])) for ix, item in enumerate(plan["source_commitments"])}
    SOURCE.check(len(speakers) == len(set(speakers.values())) == 8, "Stock speakers are not distinct")

    def clip(group, family):
        arms = {"untouched": "untouched", "genuine": "same_reader_splice", "synthetic": family}
        entries = {variant: items[group, arm] for variant, arm in arms.items()}
        truth = entries["synthetic"][0]["edit_clock"]
        SOURCE.check(entries["genuine"][0]["ordinary_splice_clock"] == truth, "Genuine stock edit clock differs")
        return {"speaker": speakers[group], "group": group, "family": family, "original_span": truth,
                "control": arms["genuine"],
                "variants": {variant: entry[0]["dry_pcm_sha256"] for variant, entry in entries.items()},
                "samples": {variant: entry[1] for variant, entry in entries.items()}}

    ordered = sorted(speakers, key=rank)
    for pair_index, pair in enumerate(zip(ordered[::2], ordered[1::2], strict=True)):
        for index in range(2):
            families = ("piper", "kitten") if (index + pair_index) % 2 == 0 else ("kitten", "piper")
            yield "stock_transfer", pair_index, index, [clip(group, family) for group, family in
                                                        zip(pair, families, strict=True)]


def position(samples, shift):
    import numpy as np

    placed = np.zeros(SAMPLES, dtype=np.float64)
    first = max(0, -shift)
    last = min(SAMPLES, SAMPLES - shift)
    placed[first + shift:last + shift] = samples[first:last]
    fade = np.sin(np.linspace(0, np.pi / 2, 320)) ** 2
    placed[first + shift:first + shift + len(fade)] *= fade
    placed[last + shift - len(fade):last + shift] *= fade[::-1]
    return placed


def render(clips, schedule, impulses, directions):
    import numpy as np
    from scipy.signal import fftconvolve

    tracks, descriptors = [], []
    for clip, start, direction in zip(clips, LOCATIONS[schedule], directions, strict=True):
        a, b = clip["original_span"]
        shift = start - a
        ir = impulses[direction]
        arrival = int(np.argmax(np.max(abs(ir), axis=1)))
        stop = start + b - a
        SOURCE.check(start >= max(0, shift) + 320 and stop + 320 <= min(SAMPLES, SAMPLES + shift),
                     "A source placement faded or truncated a staged edit")
        start_rendered, stop_rendered = (start + arrival, stop + arrival)
        SOURCE.check(0 < start_rendered < stop_rendered <= SAMPLES, "Room arrival falls outside the clip")
        variants = {}
        for variant, pcm in clip["samples"].items():
            placed = position(pcm, shift)
            variants[variant] = fftconvolve(placed[:, None], ir, axes=0)[:SAMPLES]
            if variant != "untouched":
                dry = clip["samples"]["untouched"]
                SOURCE.check(np.array_equal(pcm[:a], dry[:a]) and np.array_equal(pcm[b:], dry[b:])
                             and np.any(pcm[a:b] != dry[a:b]), "Native samples do not match the declared edit")
        tracks.append(variants)
        descriptors.append({key: val for key, val in clip.items() if key != "samples"} |
                           {"shift": shift, "direction": direction, "direct_arrival": arrival,
                            "span": [start_rendered, stop_rendered]})
    foreground, background = tracks
    relative = np.linalg.norm(foreground["untouched"].mean(axis=1)) / np.linalg.norm(background["untouched"].mean(axis=1))
    gain = relative * 10 ** .25
    maximum = max(np.max(abs(wave)) for wave in foreground.values()) + gain * max(
        np.max(abs(wave)) for wave in background.values())
    gains = [25000 / maximum, 25000 * gain / maximum]
    quantized = [{kind: np.rint(record * amp).astype("<i2") for kind, record in lane.items()}
                 for lane, amp in zip(tracks, gains, strict=True)]
    return descriptors, quantized


def freeze(args):
    import numpy as np

    from scripts.audio_measured_spatial_transfer import load_rigs

    rigs, assets = load_rigs(args.rooms)
    args.output.mkdir(mode=0o700)
    pcm_root = args.output / "pcm"
    pcm_root.mkdir(mode=0o700)
    scenes = []

    def save(values):
        data = values.tobytes()
        digest = SOURCE.digest(data)
        file = pcm_root / f"{digest}.pcm"
        if not file.exists():
            file.write_bytes(data)
        return digest

    for cohort, pair, index, clips in (*phoneme_clips(args.phoneme), *stock_clips(args.stock)):
        schedule = tuple(LOCATIONS)[index % 2]
        directions = ("left", "right") if (pair + index) % 2 == 0 else ("right", "left")
        metadata, sources = render(clips, schedule, rigs[METHOD["panels"][cohort]], directions)
        mixes = {}
        for name, (kind_a, kind_b) in CONDITIONS.items():
            first, second = sources[0][kind_a], sources[1][kind_b]
            mixed = first.astype(np.int32) + second.astype(np.int32)
            SOURCE.check(np.max(abs(mixed)) < 32768, "The new mixture would clip")
            mixes[name] = {"pcm_sha256": save(mixed.astype("<i2")), "stems": [save(first), save(second)]}
        scenes.append({"cohort": cohort, "pair": pair, "schedule": schedule, "voices": metadata, "mixes": mixes})
    plan = {"contract": contract(), "rooms": assets, "scenes": scenes}
    validate(plan)
    SOURCE.write(args.output / "freeze.json", plan)
    print(f"Froze {len(scenes)} scenes and {len(scenes) * len(CONDITIONS)} source-committed deliveries", flush=True)


def validate(plan):
    SOURCE.check(plan["contract"] == contract(), "Method or source commitments differ")
    counts = Counter()
    speakers = defaultdict(set)
    mixtures = set()
    for scene in plan["scenes"]:
        cohort, pair, schedule = (scene[key] for key in ("cohort", "pair", "schedule"))
        expected = LOCATIONS[schedule]
        voices = scene["voices"]
        keys = tuple(voice["speaker"] for voice in voices)
        SOURCE.check(len(voices) == len(set(keys)) == 2 and len({voice["family"] for voice in voices}) == 2,
                     "A scene reuses its reader or generator")
        speakers[cohort, pair].add(keys)
        counts[cohort, schedule] += 1
        for voice, start in zip(voices, expected, strict=True):
            a, b = voice["original_span"]
            actual = start + voice["direct_arrival"]
            SOURCE.check(0 < a < b <= SAMPLES and voice["shift"] == start - a and
                         voice["span"] == [actual, actual + b - a], "Exact source and delivery clocks differ")
        SOURCE.check(set(scene["mixes"]) == set(CONDITIONS), "A factorial arm or ordinary control is absent")
        for kind, key in CONDITIONS.items():
            item = scene["mixes"][kind]
            p = item["pcm_sha256"]
            SOURCE.check(p not in mixtures and all(prior.stock.prior.sha256(d) for d in [p, *item["stems"]]),
                         "A mixture repeated or lost its hash")
            mixtures.add(p)
            for side, role in enumerate(key):
                same = {scene["mixes"][arm]["stems"][side] for arm, states in CONDITIONS.items() if states[side] == role}
                SOURCE.check(len(same) == 1, "The same voice state used different samples between controls")
    SOURCE.check(counts == {(cohort, schedule): count for cohort, count in
                             (("development", 6), ("phoneme_transfer", 4), ("stock_transfer", 4))
                             for schedule in LOCATIONS}, "Unexpected scene counts")
    pairs = [next(iter(items)) for items in speakers.values()]
    SOURCE.check(all(len(items) == 1 for items in speakers.values()) and len({p for pair in pairs for p in pair}) == 2 * len(pairs),
                 "Speaker reuse crosses a pair or source role")


def cases(plan, phase):
    for index, scene in enumerate(plan["scenes"]):
        if (scene["cohort"] == "development") == (phase == "development"):
            for condition in CONDITIONS:
                yield index, scene, condition, scene["mixes"][condition]


def load_stereo(root, digest):
    import numpy as np

    raw = (root / "pcm" / f"{digest}.pcm").read_bytes()
    SOURCE.check(len(raw) == 4 * SAMPLES and SOURCE.digest(raw) == digest, "Rendered stereo bytes changed")
    return np.frombuffer(raw, "<i2").reshape(SAMPLES, 2)


def strongest(row, arm):
    if arm == "mono":
        result = prior.parent.ranked([row["mono"]])
        return None if result is None else (*result, None)
    picks = []
    for window_index, start in enumerate(prior.parent.GRID):
        values = [(stream["scores"][window_index], stream_index)
                  for stream_index, stream in enumerate(row["streams"])
                  if stream["scores"][window_index] is not None]
        ordered = sorted(values, key=lambda item: (-item[0], item[1]))
        if not ordered or (arm == "directional_lead" and (len(ordered) < 2 or ordered[0][0] - ordered[1][0] < MIN_LEAD)):
            continue
        value, stream = ordered[0]
        picks.append((value, -start, stream))
    return max(picks, default=None)


def bind(plan, record, phase):
    SOURCE.check(record["contract"] == contract() and record["phase"] == phase, "Recording phase differs")
    for (index, scene, condition, original), row in zip(cases(plan, phase), record["rows"], strict=True):
        SOURCE.check((index, condition, original["pcm_sha256"]) == (row["scene"], row["condition"], row["pcm_sha256"])
                     and prior.valid_grid(row["mono"]), "Scored waveform or clock differs")
        SOURCE.check(len(row["streams"]) in (0, 2, 3) and all(prior.stock.prior.sha256(s["pcm_sha256"]) and
                     prior.valid_grid(s["scores"]) and len(s["quality"]) == len(prior.parent.GRID) and
                     all(len(q) == 2 and all(x is None or math.isfinite(x) for x in q) for q in s["quality"])
                     for s in row["streams"]), "A directional stream witness differs")
        yield scene, condition, row


def cutoffs(plan, development):
    maximum = defaultdict(dict)
    for scene, condition, row in bind(plan, development, "development"):
        if condition not in ("untouched", "neither"):
            continue
        for arm in SCORES:
            reading = strongest(row, arm)
            if reading:
                namespace = maximum[arm]
                pair = scene["pair"]
                namespace[pair] = max(namespace.get(pair, -math.inf), reading[0])
    points = {}
    for arm in SCORES:
        ranked = sorted(maximum[arm].values(), reverse=True)
        # No point exists without enough independent genuine controls.
        points[arm] = {"strict": ranked[0] if ranked else None, "one_pair": ranked[1] if len(ranked) >= 2 else None}
    return points


def score(args):
    import numpy as np

    from scripts.audio_source_isolation_diagnostic import acoustic_quality
    from scripts.audio_spatial_mask_voice_transfer import separate

    root = args.inputs
    native_plan = root / "freeze.json"
    plan = SOURCE.read(native_plan)
    validate(plan)
    points = cutoffs(plan, SOURCE.read(args.development)) if args.development is not None else None
    SOURCE.check((args.phase == "held") == (points is not None), "Held inference requires the sealed development record")
    args.out.mkdir(mode=0o700)
    record = {"contract": contract(), "phase": args.phase, "plan_sha256": SOURCE.path_sha(native_plan),
              "points": points, "rows": []}
    SOURCE.write(args.out / "freeze.json", {key: val for key, val in record.items() if key != "rows"})
    classifier, extract, activity = SCORING.load_classifier(args.weights, args.speech)

    def classify(samples):
        wave = samples.astype(np.float32) / 32768
        return SCORING.window_log_odds(classifier, extract, wave, activity.spans(wave), RATE)

    for number, (index, _, condition, row) in enumerate(cases(plan, args.phase), 1):
        digest = row["pcm_sha256"]
        stereo = load_stereo(root, digest)
        left, right = (load_stereo(root, stem) for stem in row["stems"])
        SOURCE.check(np.array_equal(stereo, left.astype(np.int32) + right.astype(np.int32)), "Mixture is not the exact stem sum")
        mono = np.rint(stereo.mean(axis=1)).astype("<i2")
        # Measure the detector before comparing any estimate to a known source.
        baselines = classify(mono)
        streams, _ = separate(stereo.astype(np.float64))
        streams = [s for s in streams if s is not None]
        measurements = [classify(s) for s in streams] if len(streams) >= 2 else []
        estimates = []
        blend = [stem.mean(axis=1) for stem in (left, right)]
        for stream, values in zip(streams if measurements else [], measurements, strict=True):
            quality = [[acoustic_quality(stem[start:start + RATE], stream[start:start + RATE]) for stem in blend]
                       for start in prior.parent.GRID]
            estimates.append({"pcm_sha256": SOURCE.digest(stream.tobytes()), "scores": values, "quality": quality})
        record["rows"].append({"scene": index, "condition": condition, "pcm_sha256": digest,
                               "mono": baselines, "streams": estimates})
        if number % 10 == 0:
            print(f"{args.phase}: scored {number} deliveries", flush=True)
    SOURCE.write(args.out / "scores.json", record)


def current(args):
    import numpy as np
    import torch

    from app.detectors.audio import short_voice_review
    from app.detectors.audio.upgraded import SpeechActivity
    from scripts.audio_strong_traffic_view import network_free
    from scripts.prepare_audio_specialists_v15 import ASSETS
    from scripts.run_audio_specialists_v15 import load_voice

    root = args.inputs
    native_plan = root / "freeze.json"
    plan = SOURCE.read(native_plan)
    validate(plan)
    network_free()
    model = load_voice({"weights": args.weights, "assets": ASSETS})
    torch.set_num_threads(2)
    activity = SpeechActivity(args.weights / "silero_vad.jit")

    class Deadline:
        def check(self, _stage):
            pass  # Repeatable local comparison of the exact app scan.

    readings = []
    for i, (index, _, condition, item) in enumerate(cases(plan, "held"), 1):
        digest = item["pcm_sha256"]
        stereo = load_stereo(root, digest)
        signal = np.rint(stereo.mean(axis=1)).astype(np.float32) / 32768
        result = short_voice_review.scan(signal, activity.spans(signal), model, Deadline(), codec="pcm_s16le")
        reading = result["selected"]
        window = [reading["start_sample"], reading["end_sample"]] if reading else None
        readings.append({"scene": index, "condition": condition, "pcm_sha256": digest, "window": window,
                         "speech_counts": result["speech_counts"], "responses": result["responses"]})
        if i % 10 == 0:
            print(f"Current app scan: measured {i} deliveries", flush=True)
    SOURCE.write(args.out, {"plan_sha256": SOURCE.path_sha(native_plan), "assets": ASSETS, "rows": readings})


def outcome(scene, condition, window, guess):
    if window is None:
        return "missed" if "synthetic" in CONDITIONS[condition] else "clear"
    truth = [voice["span"] if arm == "synthetic" else None for voice, arm in
             zip(scene["voices"], CONDITIONS[condition], strict=True)]
    if not any(truth):
        return "ordinary_flag"
    matches = [index for index, span in enumerate(truth) if span and SCORING._useful(window, span)]
    if not matches:
        return "wrong_time"
    if guess is not None and guess not in matches:
        return "wrong_voice"
    if guess is None:
        return "timed_only"
    return "correct_voice_and_time"


def attributed(window, chosen_stream):
    sample = window[0]
    readings = chosen_stream["quality"][prior.parent.GRID.index(sample)]
    if any(value is None for value in readings) or abs(readings[0] - readings[1]) < 1:
        return None
    return int(readings[1] > readings[0])


def app_reading(row):
    windows = prior.stock.WINDOWS
    counts = row["speech_counts"]
    SOURCE.check(len(counts) == len(windows) and all(type(x) is int and 0 <= x <= 2 * RATE for x in counts),
                 "Current speech coverage differs")
    expected = [pair for pair, count in zip(windows, counts, strict=True) if count >= RATE]
    observed = [(entry["start_sample"], entry["end_sample"], entry["logit"]) for entry in row["responses"]]
    SOURCE.check(expected == [entry[:2] for entry in observed] and all(math.isfinite(entry[2]) for entry in observed),
                 "Current voice responses differ from its admitted windows")
    strongest_window = max(observed, key=lambda entry: (entry[2], -entry[0]), default=None)
    window = list(strongest_window[:2]) if strongest_window and strongest_window[2] > prior.stock.POINT else None
    SOURCE.check(row["window"] == window, "Current scan point or source range differs")
    return (window, None, bool(observed))


def summarize(readings, plan):
    counts = defaultdict(Counter)
    ordinary_pairs = defaultdict(set)
    for (index, arm, condition), (window, guess, assessed) in readings.items():
        scene = plan["scenes"][index]
        decision = outcome(scene, condition, window, guess)
        matches = [v["span"] for v, state in zip(scene["voices"], CONDITIONS[condition], strict=True)
                   if state == "synthetic" and SCORING._useful(window, v["span"])]
        controls = [readings[index, arm, control][0] for control in ("neither", "untouched")]
        most_specific = [span for span in matches if not any(SCORING._useful(reference, span) for reference in controls)]
        for scope in (scene["cohort"], f"{scene['cohort']}/{scene['schedule']}"):
            key = scope, arm, condition
            result = counts[key]
            result["assessed"] += assessed
            result[decision] += 1
            if decision == "ordinary_flag":
                ordinary_pairs[key].add(scene["pair"])
            if matches:
                result["useful_times"] += 1
                result["useful_without_matched_control"] += bool(most_specific)
                if condition == "both":
                    result["both_times"] += len(matches) == 2
    return {"/".join(key): dict(value, ordinary_pairs_flagged=len(ordinary_pairs[key]))
            for key, value in sorted(counts.items())}


def report(payload):
    plan, development, held = (payload[key] for key in ("plan", "development", "held"))
    validate(plan)
    plan_hash = SOURCE.digest((json.dumps(plan, indent=2, sort_keys=True, allow_nan=False) + "\n").encode())
    SOURCE.check(development["plan_sha256"] == held["plan_sha256"] == plan_hash, "Scoring uses another plan")
    points = cutoffs(plan, development)
    SOURCE.check(development["points"] is None and held["points"] == points, "Held cutoffs changed")
    readings = {}
    current_counts = payload["current"]
    SOURCE.check(current_counts["plan_sha256"] == plan_hash, "Current settings use another plan")
    for (_, condition, row), current_row in zip(bind(plan, held, "held"), current_counts["rows"], strict=True):
        SOURCE.check(all(current_row[k] == row[k] for k in ("scene", "condition", "pcm_sha256")), "Current source differs")
        index = row["scene"]
        readings[index, "current", condition] = app_reading(current_row)
        for arm in SCORES:
            strongest_view = strongest(row, arm)
            for point, cutoff in points[arm].items():
                assessed = strongest_view is not None and cutoff is not None
                hit, guess = None, None
                if assessed and strongest_view[0] > cutoff:
                    hit = [-strongest_view[1], RATE - strongest_view[1]]
                    if strongest_view[2] is not None:
                        guess = attributed(hit, row["streams"][strongest_view[2]])
                readings[index, f"{arm}/{point}", condition] = (hit, guess, assessed)
    return summarize(readings, plan)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    modes = cli.add_subparsers(dest="command", required=True)
    intake = modes.add_parser("freeze")
    for name in ("output", "phoneme", "stock", "rooms"):
        intake.add_argument(name, type=Path)
    run = modes.add_parser("score")
    run.add_argument("phase", choices=("development", "held"))
    for name in ("inputs", "out", "weights", "speech"):
        run.add_argument(name, type=Path)
    run.add_argument("--development", type=Path)
    app = modes.add_parser("current")
    for name in ("inputs", "out", "weights"):
        app.add_argument(name, type=Path)
    combine = modes.add_parser("combine")
    for name in ("plan", "development", "held", "current", "out"):
        combine.add_argument(name, type=Path)
    modes.add_parser("review").add_argument("receipt", type=Path)
    args = cli.parse_args()
    if args.command == "freeze":
        freeze(args)
    elif args.command == "score":
        score(args)
    elif args.command == "current":
        current(args)
    elif args.command == "combine":
        payload = {name: SOURCE.read(getattr(args, name)) for name in ("plan", "development", "held", "current")}
        payload["results"] = report(payload)
        SOURCE.write(args.out, payload)
    else:
        open_receipt = gzip.open if args.receipt.suffix == ".gz" else Path.open
        with open_receipt(args.receipt, "rt", encoding="utf-8") as stream:
            payload = json.load(stream)
        result = report(payload)
        SOURCE.check(payload["results"] == result, "Result differs from numeric replay")
        print(f"Replayed {len(payload['plan']['scenes'])} matched scenes")
        for key, stats in result.items():
            print(key, stats)


if __name__ == "__main__":
    main()
