# SPDX-License-Identifier: MIT
"""Compare voice localization on mixed, exact, and estimated speech."""

from __future__ import annotations

import argparse
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

from scripts import audio_second_classifier_overlap as scoring
from scripts import audio_source_isolation_diagnostic as isolation

SOURCE = scoring.SOURCE
PARENT = scoring.parent.PANELS / "audio-second-classifier-overlap-2026-11-14.json"
SAMPLES = scoring.parent.SAMPLES
RATE = scoring.RATE
GRID = scoring.GRIDS[RATE]
SEPARATOR = {
    "repo": "mhavo/mossformer2-ss-16k-onnx",
    "revision": "3be7f95237b5121caf41dec4c62b0af341a640af",
    "sha256": "54f956bd96736b6f8713d6d31599c31f12cfe591875768088cbe893193f7f1b6",
}
VIEWS = ("mix", "exact_target", "blind", "mixture_phase")
SCHEMA = "chipmunk.audio-mixture-phase-isolation/1"
METHOD = {
    "inputs": "Use the fixed minus-five-dB AMI and ICSI panels, including isolated backgrounds. Recover the "
              "exact target for diagnostic use only by integer subtraction of the paired isolated background.",
    "blind": "Separate two four-second segments at 0 and 2 s. Match output ordering and polarity on their "
             "2 s shared interval, decline if either absolute centered correlation is below 0.25, and "
             "linearly crossfade. Keep estimates with RMS at least 8 percent of the submission.",
    "phase": "Convert the two estimated waveforms to complementary power-ratio masks in the 512-sample "
             "short-time transform with a 128-sample hop. Apply each mask to the submitted complex "
             "spectrum. Use the same stream ordering and activity rules as the direct estimates.",
    "scoring": "Use the existing Apache-2.0 one-second detector with its same 0.5 s grid and speech "
               "requirement independently on each estimated stream. Rank the largest response across "
               "streams and windows, with earliest ties. Reuse the parent mixed-window measurements.",
    "cutoffs": "For each view use the AMI ordinary source-group maximum ranked second highest, bounded "
               "below by its highest measured isolated-background score. Use a strict cutoff and seal "
               "the values before ICSI scoring. An exact silent target is unassessed.",
    "decision": "On ICSI require at least two more useful locations on separate readers without a paired "
                "genuine location collision, at most one additional ordinary group and zero background flags "
                "to motivate new-source validation. Source labels are external diagnostic information.",
}


def lock(panel):
    _, parent_sha = scoring._parent(panel)
    return {"schema": SCHEMA, "method": METHOD, "panel": panel, "roster_sha256": parent_sha,
            "classifier_receipt_sha256": SOURCE.path_sha(PARENT), "separator": SEPARATOR}


def previous_rows(panel):
    receipt, _ = scoring._parent(panel)
    previous = SOURCE.read(PARENT)["panels"][panel]["measurements"]
    for clip, scores in zip(receipt["freeze"]["plan"]["clips"], previous, strict=True):
        SOURCE.check(clip["pcm_sha256"] == scores["pcm_sha256"], "Prior score uses another source")
        if clip["condition_db"] == -5:
            yield clip, scores["log_odds"][0]


def select_streams(mixture, inference):
    import numpy as np

    left, right = (inference(mixture[start:start + 4 * RATE][None, :]) for start in (0, 2 * RATE))
    width = 2 * RATE
    correlations = [[isolation.centered_correlation(a[-width:], b[:width]) for b in right] for a in left]
    order = max(((0, 1), (1, 0)), key=lambda pair: sum(abs(correlations[i][j]) for i, j in enumerate(pair)))
    agreements = [correlations[i][j] for i, j in enumerate(order)]
    if min(map(abs, agreements)) < .25:
        return [], agreements
    fade = np.linspace(0, 1, width, dtype=np.float32)
    streams = []
    for i, j in enumerate(order):
        a, b = left[i], right[j] * (1 if agreements[i] >= 0 else -1)
        streams.append(np.concatenate((a[:width], (1 - fade) * a[width:] + fade * b[:width], b[width:])))
    return streams, agreements


def mixture_phase(mixture, streams):
    import numpy as np
    from scipy.signal import istft, stft

    params = {"fs": RATE, "nperseg": 512, "noverlap": 384}
    spectrum = stft(mixture, **params)[2]
    powers = np.stack([abs(stft(signal, **params)[2]) ** 2 for signal in streams])
    total = powers.sum(axis=0)
    masks = np.divide(powers, total, out=np.full_like(powers, .5), where=total > 1e-12)
    estimates = [istft(spectrum * mask, **params)[1] for mask in masks]
    SOURCE.check(all(view.shape == (SAMPLES,) and np.isfinite(view).all() for view in estimates)
                 and np.max(np.abs(estimates[0] + estimates[1] - mixture)) < 1e-5,
                 "Mixture-phase reconstruction lost source samples")
    return estimates


def prepare(args):
    import numpy as np
    import onnxruntime as ort

    SOURCE.check(SOURCE.path_sha(args.separator) == SEPARATOR["sha256"], "Separator digest differs")
    settings = ort.SessionOptions()
    settings.intra_op_num_threads = 2
    settings.inter_op_num_threads = 1
    session = ort.InferenceSession(str(args.separator), sess_options=settings, providers=["CPUExecutionProvider"])

    def infer(signal):
        output, = session.run(["sources"], {"mix": signal})
        SOURCE.check(output.shape == (2, 4 * RATE) and output.dtype == np.float32 and np.isfinite(output).all(),
                     "Separator must return two aligned finite 4 s estimates")
        return output

    args.output.mkdir(mode=0o700)
    samples_dir = args.output / "signals"
    samples_dir.mkdir(mode=0o700)
    roster = list(previous_rows(args.panel))
    clips = {}
    for clip, _ in roster:
        key = clip["pcm_sha256"]
        raw = (args.pcm / f"{key}.pcm").read_bytes()
        SOURCE.check(len(raw) == 2 * SAMPLES and SOURCE.digest(raw) == key, "Unexpected mixture PCM")
        clips[clip["group"], clip["kind"]] = np.frombuffer(raw, "<i2").astype(np.int32)
    records = []
    for clip, _ in roster:
        group = clip["group"]
        submitted = clips[group, clip["kind"]]
        background = clips[group, "background"]
        native = submitted - background
        original = clips[group, "untouched"] - background
        SOURCE.check(np.max(abs(native)) <= 32767, "Exact foreground samples overflowed")
        if clip["truth"]:
            start, stop = clip["truth"]
            SOURCE.check(np.array_equal(native[:start], original[:start]) and
                         np.array_equal(native[stop:], original[stop:]), "Foreground edit escaped its source range")
        wet, dry = (value.astype(np.float32) / 32768. for value in (submitted, native))
        blind, alignments = select_streams(wet, infer)
        phased = mixture_phase(wet, blind) if blind else []
        bound = .08 * np.linalg.norm(wet)
        record = {"pcm_sha256": clip["pcm_sha256"], "stream_alignment": [round(x, 6) for x in alignments],
                  "views": {}, "qualities": {}}
        for name, views in (("mix", [wet]), ("exact_target", [dry]), ("blind", blind), ("mixture_phase", phased)):
            available = [view for view in views if np.any(view) and (name in ("mix", "exact_target") or
                         np.linalg.norm(view) >= bound)]
            digests = []
            for view in available:
                data = view.astype("<f4", copy=False).tobytes()
                digest = SOURCE.digest(data)
                path = samples_dir / f"{digest}.f32"
                if not path.exists():
                    path.write_bytes(data)
                digests.append(digest)
            record["views"][name] = digests
            quality = {}
            if np.any(dry):
                ranges = {"whole": (0, SAMPLES)}
                if clip["truth"]:
                    ranges["edit"] = clip["truth"]
                for area, (a, b) in ranges.items():
                    values = [isolation.acoustic_quality(dry[a:b], view[a:b]) for view in available]
                    quality[area] = [round(value, 6) if value is not None else None for value in values]
            record["qualities"][name] = quality
        records.append(record)
        if len(records) % 8 == 0:
            print(f"{args.panel}: separated {len(records)}/{len(roster)}", flush=True)
    SOURCE.write(args.output / "signals.json", {"freeze": lock(args.panel), "runtime": ort.__version__, "rows": records})


def ranked(scores):
    return max(((value, -start) for stream in scores for value, start in zip(stream, GRID, strict=True)
                if value is not None), default=None)


def chosen(response, point):
    if response is not None and response[0] > point:
        return [-response[1], RATE - response[1]]
    return None


def iter_results(record, panel):
    SOURCE.check(record["freeze"] == lock(panel), "Run commitment differs")
    for (clip, mix), row in zip(previous_rows(panel), record["rows"], strict=True):
        SOURCE.check(clip["pcm_sha256"] == row["pcm_sha256"] and row["scores"]["mix"] == [mix], "Mixed scores differ")
        SOURCE.check(set(row["scores"]) == set(row["views"]) == set(row["qualities"]) == set(VIEWS),
                     "View inventory differs")
        background = clip["kind"] == "background"
        expected_areas = set() if background else {"whole", "edit"} if clip["truth"] else {"whole"}
        SOURCE.check(len(row["views"]["mix"]) == 1 and len(row["views"]["exact_target"]) == int(not background)
                     and len(row["stream_alignment"]) == 2
                     and all(-1 <= value <= 1 for value in row["stream_alignment"]), "Unexpected native view coverage")
        for view in VIEWS:
            streams = len(row["views"][view])
            SOURCE.check(len(row["scores"][view]) == streams <= 2 and all(
                len(stream) == len(GRID) and all(value is None or math.isfinite(value) for value in stream)
                for stream in row["scores"][view]), "Invalid window scores")
            quality = row["qualities"][view]
            SOURCE.check(set(quality) == expected_areas and all(
                len(area) == streams and all(value is None or math.isfinite(value) for value in area)
                for area in quality.values()), "Invalid acoustic measurement")
        yield clip, row


def calibrate(development):
    groups = defaultdict(dict)
    isolated = defaultdict(list)
    for clip, row in iter_results(development, "ami"):
        if clip["truth"] is not None:
            continue
        for view, scores in row["scores"].items():
            response = ranked(scores)
            if response is None:
                continue
            if clip["kind"] == "background":
                isolated[view].append(response[0])
            else:
                group = clip["group"]
                groups[view][group] = max(groups[view].get(group, -math.inf), response[0])
    return {view: max(sorted(groups[view].values())[-2], max(isolated[view], default=-math.inf)) for view in VIEWS}


def measure(args):
    import numpy as np

    points = calibrate(SOURCE.read(args.development)) if args.development else None
    input_record = SOURCE.read(args.views / "signals.json")
    SOURCE.check(input_record["freeze"] == lock(args.panel), "Prepared view method differs")
    args.output.mkdir(mode=0o700)
    SOURCE.write(args.output / "freeze.json", {"source": input_record["freeze"], "points": points})
    model, feature, speech = scoring.load_classifier(args.weights, args.speech)
    cache = {}
    for index, ((clip, mixed_scores), row) in enumerate(zip(previous_rows(args.panel), input_record["rows"], strict=True), 1):
        SOURCE.check(clip["pcm_sha256"] == row["pcm_sha256"], "Wrong source view")
        result = {"mix": [mixed_scores]}
        for name in VIEWS[1:]:
            result[name] = []
            for digest in row["views"][name]:
                if digest not in cache:
                    raw = (args.views / "signals" / f"{digest}.f32").read_bytes()
                    SOURCE.check(SOURCE.digest(raw) == digest and len(raw) == 4 * SAMPLES, "Float samples differ")
                    signal = np.frombuffer(raw, "<f4")
                    spans = speech.spans(signal)
                    cache[digest] = scoring.window_log_odds(model, feature, signal, spans, RATE)
                result[name].append(cache[digest])
        row["scores"] = result
        if index % 8 == 0:
            print(f"{args.panel}: scored {index}/{len(input_record['rows'])}", flush=True)
    input_record["points"] = points
    SOURCE.write(args.output / "scored.json", input_record)


def summarize(record, panel, points):
    rows = list(iter_results(record, panel))
    counts = defaultdict(Counter)
    recovered, genuine, acoustic = defaultdict(set), {}, defaultdict(list)
    ordinary = defaultdict(set)
    for clip, row in rows:
        group, kind = clip["group"], clip["kind"]
        for view in VIEWS:
            response = ranked(row["scores"][view])
            location = chosen(response, points[view])
            counts[view]["assessed"] += response is not None
            if clip["truth"] is not None:
                counts[view]["edited"] += 1
                counts[view]["edited_flags"] += location is not None
                base = row["qualities"]["mix"]["edit"][0]
                quality = row["qualities"][view]["edit"]
                valid_quality = [value for value in quality if value is not None]
                if valid_quality and base is not None:
                    acoustic[view].append(max(valid_quality) - base)
                if view in ("blind", "mixture_phase") and len(valid_quality) == 2 and valid_quality[0] != valid_quality[1]:
                    strengths = [ranked([stream]) for stream in row["scores"][view]]
                    if response is not None and strengths[0] != strengths[1]:
                        winner = strengths.index(response)
                        closest = valid_quality.index(max(valid_quality))
                        counts[view]["ranked_edit_stream_comparisons"] += 1
                        counts[view]["strongest_response_from_closest_stream"] += winner == closest
                        counts[view]["flagged_edit_stream_comparisons"] += location is not None
                        counts[view]["flag_from_closest_stream"] += location is not None and winner == closest
                if scoring._useful(location, clip["truth"]):
                    counts[view]["useful"] += 1
                    recovered[view].add((group, kind, tuple(location)))
            elif kind == "background":
                counts[view]["background"] += 1
                counts[view]["background_flags"] += location is not None
                genuine[view, group, kind] = location
            else:
                counts[view]["ordinary"] += 1
                counts[view]["ordinary_flags"] += location is not None
                if location is not None:
                    ordinary[view].add(group)
                genuine[view, group, kind] = location
    for view, counter in counts.items():
        counter["ordinary_groups_flagged"] = len(ordinary[view])
        counter["useful_without_matched_collision"] = sum(all(
            list(location) != genuine.get((view, group, comparison))
            for comparison in ("untouched", kind.replace("edited/", "ordinary/"), "background"))
            for group, kind, location in recovered[view])
    return {view: {**counts[view], "edit_si_sdr_count": len(acoustic[view]),
                   "edit_si_sdr_gain_median_db": round(median(acoustic[view]), 4) if acoustic[view] else None}
            for view in VIEWS}


def review(payload):
    panels = payload["panels"]
    points = calibrate(panels["ami"])
    SOURCE.check(panels["ami"]["points"] is None and panels["icsi"]["points"] == points, "Held cutoffs differ")
    return {panel: summarize(panels[panel], panel, points) for panel in ("ami", "icsi")}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    commands = cli.add_subparsers(dest="phase", required=True)
    for phase, fields in (("prepare", ("output", "pcm", "separator")),
                          ("measure", ("output", "views", "weights", "speech"))):
        cmd = commands.add_parser(phase)
        cmd.add_argument("panel", choices=("ami", "icsi"))
        for field in fields:
            cmd.add_argument(field, type=Path)
        if phase == "measure":
            cmd.add_argument("--development", type=Path)
    combine = commands.add_parser("combine")
    for field in ("ami", "icsi", "output"):
        combine.add_argument(field, type=Path)
    commands.add_parser("review").add_argument("receipt", type=Path)
    args = cli.parse_args()
    if args.phase == "prepare":
        prepare(args)
    elif args.phase == "measure":
        if (args.panel == "icsi") != (args.development is not None):
            cli.error("ICSI scoring requires the sealed AMI results")
        measure(args)
    elif args.phase == "combine":
        record = {"panels": {panel: SOURCE.read(getattr(args, panel)) for panel in ("ami", "icsi")}}
        record["results"] = review(record)
        SOURCE.write(args.output, record)
    else:
        record = SOURCE.read(args.receipt)
        results = review(record)
        SOURCE.check(results == record["results"], "Stored results do not reproduce")
        print(results)


if __name__ == "__main__":
    main()
