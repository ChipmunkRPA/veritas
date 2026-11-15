# SPDX-License-Identifier: MIT
"""Test ordered lip-release alignment against local and clip-wide soundtrack shifts."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

from scripts import video_bilabial_mouth_transfer as parent

PANEL = Path(__file__).resolve().parents[2] / "docs/evaluation-panels"
SOURCE = PANEL / "video-bilabial-mouth-transfer-2026-11-14.json"
ACOUSTIC = (PANEL / "video-neural-phone-clock-2026-11-14.json", PANEL / "video-neural-phone-context-2026-11-14.json")
SCHEMA = "chipmunk.video-lip-event-alignment/1"
RATE = parent.RATE
SKIP = 1120
CHANGE_COST = 1440
CLOCK_LIMIT = 4160
RESIDUAL_LIMIT = 960
MARGIN = 160
SPAN_PAD = 1280
MIN_FLANK_GAP = 1600
METHOD = {
    "source": "Use the committed B/P detections, decoded inner-lip release observations and native clock from the same "
              "96 Lombard Grid views. Keep the prior development/held speakers and acoustic bias intact.",
    "matching": "Enumerate ordered one-to-one associations. Compare a constant audio-minus-picture offset against a "
                "single interval of changed offset, with distinct ordered support on both sides. Drop unmatched events "
                "with a fixed 70 ms-equivalent penalty each. Penalize an added timing change by 90 ms-equivalent.",
    "extent": "The core range spans the local audio and picture event times with 80 ms outer padding. The supported "
              "range also extends toward the midpoint behind and ahead, by at most 180 ms from each outer event. "
              "Both ranges use shared matches and points. The wider span is a review range between timing anchors.",
    "ambiguity": "Require a plausible constant baseline and a local excess of 110–340 ms. Keep the winning sign only "
                 "if competing near-cost explanations agree in sign and sufficiently overlap the same event range.",
    "points": "Freeze the largest and second-largest ordinary-or-constant-shift score per calibration speaker before "
              "assessing held examples. Use the prior 0.40 temporal IoU and 80 ms signed-error rule. These are research points.",
}


def associations(audio, picture):
    for count in range(2, min(len(audio), len(picture)) + 1):
        for left in combinations(audio, count):
            for right in combinations(picture, count):
                yield list(zip(left, right, strict=True))


def fit(matches, bias, *, limit=CLOCK_LIMIT):
    offsets = [a[1] - v["sample"] for a, v in matches]
    center = round(statistics.median(offsets))
    if abs(center - bias) > limit or max(abs(d - center) for d in offsets) > RESIDUAL_LIMIT:
        return None
    return center, sum(abs(d - center) for d in offsets)


def outer_times(pairs):
    bounds = [value for a, v in pairs for value in (a[1], v["sample"])]
    return min(bounds), max(bounds)


def candidate(matches, begin, end, bias, count_cost, samples):
    local = matches[begin:end]
    before, after = matches[:begin], matches[end:]
    base = fit(before + after, bias)
    if base is None:
        return None
    start, stop = outer_times(local)
    previous = outer_times(before)[1]
    following = outer_times(after)[0]
    if min(start - previous, following - stop) < MIN_FLANK_GAP:
        return None
    local_fit = fit(local, base[0], limit=parent.MAXIMUM_OFFSET)
    if local_fit is None:
        return None
    offset = local_fit[0] - base[0]
    if not parent.MINIMUM_OFFSET <= abs(offset) <= parent.MAXIMUM_OFFSET:
        return None
    return {
        "cost": count_cost + base[1] + local_fit[1] + CHANGE_COST,
        "offset_samples": offset, "baseline_samples": base[0],
        "matches": [{"phone": a[0], "audio_sample": a[1], "picture_sample": v["sample"], "frames": v["frames"],
                     "scope": "local" if begin <= index < end else "before" if index < begin else "after"}
                    for index, (a, v) in enumerate(matches)],
        "core": [max(0, start-SPAN_PAD), min(samples, stop+SPAN_PAD)],
        "supported": [max(0, start-2880, min(start-SPAN_PAD, (previous+start)//2)),
                      min(samples, stop+2880, max(stop+SPAN_PAD, (following+stop)//2))],
    }


def same_explanation(first, second):
    if first["offset_samples"] * second["offset_samples"] < 0:
        return False
    a, b = first["core"]
    c, d = second["core"]
    return 2 * (min(b, d) - max(a, c)) >= max(b, d) - min(a, c)


def measure(row, role, bias):
    audio = sorted(row["cases"][role]["neural"], key=lambda event: event[1])
    picture = row["releases"]
    # Repeated-timestamp events or unexpectedly dense detections leave the assignment unresolved.
    if (not 3 <= min(len(audio), len(picture)) <= max(len(audio), len(picture)) <= 8
            or len({event[1] for event in audio}) != len(audio)):
        return {"status": "insufficient_events"}
    null = (len(audio) + len(picture)) * SKIP
    for a in audio:
        for v in picture:
            if abs(a[1] - v["sample"] - bias) <= CLOCK_LIMIT:
                null = min(null, (len(audio) + len(picture) - 2) * SKIP)
    choices = []
    for matches in associations(audio, picture):
        penalty = (len(audio) + len(picture) - 2*len(matches)) * SKIP
        constant = fit(matches, bias)
        if constant is not None:
            null = min(null, penalty + constant[1])
        for begin in range(1, len(matches)-1):
            for end in range(begin+1, len(matches)):
                option = candidate(matches, begin, end, bias, penalty, row["samples"])
                if option is not None:
                    choices.append(option)
    if not choices:
        return {"status": "no_supported_change", "constant_cost": null}
    choices.sort(key=lambda item: (item["cost"], item["core"], abs(item["offset_samples"])))
    winner = choices[0]
    if any(other["cost"] <= winner["cost"] + MARGIN and not same_explanation(winner, other) for other in choices[1:]):
        return {"status": "ambiguous_change", "constant_cost": null}
    return {"status": "measured", "constant_cost": null, "improvement": null-winner["cost"], **winner}


def provenance(document):
    parent.review(document, ACOUSTIC)
    return {"schema": SCHEMA, "method": METHOD, "parent_sha256": parent.phone.sha(SOURCE),
            "audio_bias_samples": document["result"]["bias"]["neural"]["value"]}


def controls(rows):
    values = defaultdict(int)
    for row in rows:
        if row["cohort"] == 2 and row["split"] == "development" and row["role"] not in parent.EDITS:
            values[row["speaker"]] = max(values[row["speaker"]], row.get("improvement", 0))
    scores = sorted(values.values(), reverse=True)
    parent.phone.require(len(scores) == 4, "The four frozen calibration speakers are required")
    return {"tight": scores[0], "wider": scores[1]}


def summarize(rows, points):
    metrics = defaultdict(Counter)
    for row in rows:
        if row["split"] != "held":
            continue
        role = row["role"]
        signed = parent.EDITS.get(role)
        group = "local" if signed else "constant" if role.startswith("global_") else "ordinary"
        for name, point in points.items():
            for scope in ("core", "supported"):
                stats = metrics[str(row["cohort"]), name, scope, group]
                alert = row.get("improvement", 0) > point
                useful = bool(alert and signed and parent.useful(row[scope], row["offset_samples"], row["truth"], signed))
                stats.update(files=1, measured=row["status"] == "measured", assessed=row["status"] != "insufficient_events",
                             flags=alert, useful=useful, additional_useful=useful and not row["prior_useful"],
                             misplaced=bool(signed and alert and not useful))
    return {"/".join(key): dict(stats) for key, stats in sorted(metrics.items())}


def rows_for(document, split, bias):
    previous = document["result"]["points"]["neural_flanked"]["coverage"]
    for row in document["rows"]:
        if (row["cohort"] != 2 and split == "development") or row["split"] != split:
            continue
        for role, case in row["cases"].items():
            _, old = parent.proposals(row, role, "neural_flanked", bias)
            signed = parent.EDITS.get(role)
            truth = row["edit"] if signed else None
            prior_useful = bool(signed and old and old["strength"] > previous
                                and parent.useful(old["span_samples"], old["offset_samples"], truth, signed))
            yield {"cohort": row["cohort"], "split": split, "speaker": row["speaker"], "condition": row["condition"],
                   "source": row["source"], "role": role, "truth": truth, "pcm_sha256": case["pcm_sha256"],
                   "prior_useful": prior_useful, **measure(row, role, bias)}


def replay(document):
    freeze = provenance(document)
    bias = freeze["audio_bias_samples"]
    calibration = list(rows_for(document, "development", bias))
    points = controls(calibration)
    held = list(rows_for(document, "held", bias))
    return {"freeze": freeze, "calibration": calibration, "points": points,
            "held": held, "outcomes": summarize(held, points)}


def run(destination):
    document = json.loads(SOURCE.read_bytes())
    freeze = provenance(document)
    bias = freeze["audio_bias_samples"]
    calibration = list(rows_for(document, "development", bias))
    points = controls(calibration)
    destination.mkdir(mode=0o700)
    parent.phone.write(destination / "points.json", {"freeze": freeze, "calibration": calibration, "points": points})
    held = list(rows_for(document, "held", bias))
    result = {"freeze": freeze, "calibration": calibration, "points": points, "held": held,
              "outcomes": summarize(held, points)}
    parent.phone.write(destination / "receipt.json", result)


def compression_sources(document):
    groups = defaultdict(list)
    for row in document["rows"]:
        if row["split"] == "held":
            groups[row["cohort"], row["speaker"]].append(row)
    return [min(rows, key=lambda row: parent.phone.digest(str(row["source"]).encode())) for rows in groups.values()]


def aac_clocked(wave, path):
    import numpy as np

    ffmpeg = ["ffmpeg", "-nostdin", "-v", "error"]
    subprocess.run([*ffmpeg, "-y", "-f", "s16le", "-ar", str(RATE), "-ac", "1", "-i", "pipe:0",
                    "-threads", "1", "-c:a", "aac", "-b:a", "48k", str(path)],
                   input=wave.tobytes(), check=True, timeout=15)
    raw = subprocess.check_output([*ffmpeg, "-i", str(path), "-ac", "1", "-ar", str(RATE), "-f", "s16le", "pipe:1"], timeout=15)
    decoded = np.frombuffer(raw, dtype="<i2")[:len(wave)].copy()
    if len(decoded) != len(wave):
        raise ValueError("AAC decoding did not recover the same sample extent")
    expected = wave.astype(np.float64)
    received = decoded.astype(np.float64)
    scored = [(float(np.sum(expected[64:-64] * received[64+lag:len(wave)-64+lag])), lag) for lag in range(-16, 17)]
    if abs(max(scored)[1]) > 1:
        raise ValueError("AAC delivery changed the measured sample origin")
    return decoded


def replay_compressed(document, measurements, points):
    bias = provenance(document)["audio_bias_samples"]
    sources = compression_sources(document)
    candidates = []
    for row, measured in zip(sources, measurements, strict=True):
        parent.phone.require(measured["source"] == row["source"] and list(measured["cases"]) == sorted(parent.ROLES),
                             "Compressed source selection differs")
        for role in parent.ROLES:
            case = measured["cases"][role]
            parent.phone.require(case["earlier_sha256"] == row["cases"][role]["pcm_sha256"]
                                 and parent.SHA256.fullmatch(case["decoded_sha256"])
                                 and all(p in ("b", "p") and 0 <= t < row["samples"] and .35 <= value <= 1
                                         for p, t, value in case["neural"]), "Compressed acoustic clock differs")
            prepared = row | {"cases": {role: {"neural": case["neural"]}}}
            signed = parent.EDITS.get(role)
            truth = row["edit"] if signed else None
            _, prior = parent.proposals(prepared, role, "neural_flanked", bias)
            point = document["result"]["points"]["neural_flanked"]["coverage"]
            earlier_useful = bool(signed and prior and prior["strength"] > point
                                  and parent.useful(prior["span_samples"], prior["offset_samples"], truth, signed))
            candidates.append({"source": row["source"], "cohort": row["cohort"], "split": "held", "role": role,
                               "truth": truth, "prior_useful": earlier_useful,
                               "prior_flagged": bool(prior and prior["strength"] > point), **measure(prepared, role, bias)})
    original = set(row["source"] for row in sources)
    without_codec = [row for row in rows_for(document, "held", bias) if row["source"] in original]
    return {"aac": summarize(candidates, points), "same_subset_before_encoding": summarize(without_codec, points),
            "observations": candidates}


def compression(args):
    from scripts.audio_strong_traffic_view import network_free

    document = json.loads(SOURCE.read_bytes())
    locked = json.loads(args.receipt.read_bytes())
    parent.phone.require(locked == replay(document), "Timing points differ from the frozen comparison")
    points = locked["points"]
    plan = compression_sources(document)
    archives, _ = parent.phone.source_archives(args.sources)
    network_free()
    infer, device = parent.phone.neural_decoder(args.model)
    measured = []
    with ZipFile(archives["audio"]) as audio, TemporaryDirectory(prefix="lip-aac-") as folder:
        for row in plan:
            member = row["members"]["audio"]
            raw = audio.read(member["name"])
            parent.phone.require(parent.phone.digest(raw) == member["sha256"], "Publisher soundtrack differs")
            cases = {}
            for role, wave in parent.treatments(parent.phone.pcm(raw), row["edit"]).items():
                before = parent.phone.digest(wave.tobytes())
                parent.phone.require(before == row["cases"][role]["pcm_sha256"], "Staged soundtrack differs")
                decoded = aac_clocked(wave, Path(folder) / "delivery.m4a")
                cases[role] = {"earlier_sha256": before, "decoded_sha256": parent.phone.digest(decoded.tobytes()),
                               "neural": infer(decoded)}
            measured.append({"source": row["source"], "cases": dict(sorted(cases.items()))})
            print(f"AAC inference {len(measured)}/{len(plan)} source groups", flush=True)
    result = replay_compressed(document, measured, points)
    parent.phone.write(args.path, {"schema": SCHEMA, "parent_sha256": parent.phone.sha(SOURCE), "points": points,
                                   "encoding": "aac-lc-m4a-48k-16k-mono", "device": device, "measurements": measured, "result": result})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("run", "review", "compression", "review-compression"))
    parser.add_argument("path", type=Path)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--sources", type=Path)
    parser.add_argument("--model", type=Path)
    args = parser.parse_args()
    if args.phase == "run":
        run(args.path)
    elif args.phase == "compression":
        if not all((args.receipt, args.sources, args.model)):
            parser.error("Compression requires frozen results, publisher archives and the pinned model")
        compression(args)
    elif args.phase == "review-compression":
        result = json.loads(args.path.read_bytes())
        document = json.loads(SOURCE.read_bytes())
        if (result["schema"] != SCHEMA or result["parent_sha256"] != parent.phone.sha(SOURCE)
                or result["encoding"] != "aac-lc-m4a-48k-16k-mono" or result["points"] != replay(document)["points"]
                or result["result"] != replay_compressed(document, result["measurements"], result["points"])):
            raise ValueError("Soundtrack compression or its measured timing outcomes differ")
        print("Reproduced the result from the recorded compressed soundtrack detections.")
    else:
        result = json.loads(args.path.read_bytes())
        if result != replay(json.loads(SOURCE.read_bytes())):
            raise ValueError("Ordered event links, calibration or timing outcomes differ")
        print("Reproduced the ordered audiovisual links, timing readouts and held decisions.")
