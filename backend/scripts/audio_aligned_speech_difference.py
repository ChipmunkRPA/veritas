"""Measure earlier-copy speech alignment on two released neural-edit panels."""

from __future__ import annotations

import argparse
import json
import math
import re
import wave
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory

from app.evidence.audio_aligned_speech import FLANK, HOP, RATE, SIZES, processing_residual, resampled
from app.evidence.audio_aligned_speech import measure as compare
from scripts import audio_source_isolation_diagnostic as common
from scripts import replay_audio_bandwidth_public as ami
from scripts import review_audio_sine_preview as sine

SCHEMA = "chipmunk.audio-aligned-speech-difference/1"
BANKS = {
    "sine": ("audio-sine-preview-2026-11-13", "1376aeda58d56d22be13ee77cdebdd5c40f28e7883b7ada30ed5f0d351725b40"),
    "partialedit": ("audio-partialedit-dense-transfer-2026-11-12",
                    "749c83cce977529a6181f6fc159e5761f9d38afce8b89ade84512e8ebfe59fac"),
    "ami": ("audio-bandwidth-return-confirmation-2026-11-13",
            "875d6308b572eb9efb52f175b21e055babbf2589e1df192e81e54d477f68843d"),
}
CONTROLS = ("unchanged", "gain", "equalized", "resampled", "wrong_excerpt")
AMI_SELECTION = ("ES2007c", "IS1004c")
AMI_ASSESSMENT = ("ES2012b", "IS1001b")
SHA256 = re.compile(r"[0-9a-f]{64}")
METHOD = {
    "features": "Hann 40 ms spectra every 20 ms; 28 triangular Mel bands from 200 to 6800 Hz. "
                "Keep cepstral coefficients 1 to 12 and remove the per-file median. Add symmetric "
                "four-hop differences. Relative activity rejects very quiet frames.",
    "alignment": "Constrained dynamic time warping with an 800 ms band around the duration-adjusted "
                 "diagonal and a 0.08 penalty for each nondiagonal step. Average the path cost "
                 "on the submitted clock. Neither source transcript nor truth enters matching.",
    "witness": "Require a path median cost at most 0.7 and two matching flanks, each 400 ms, "
               "with at least 30% of frames above the relative activity floor in both copies. "
               "Search 120 ms to 2.4 s spans every 20 ms. The score is interior mean cost minus "
               "the larger flank mean. Rank one positive span, preferring the earlier one on ties.",
    "point": "Freeze the largest eligible ordinary AMI control score before evaluating either "
             "edit bank. Report every row including unassessed, misplaced and correctly located. "
             "A useful range covers at least half the labeled edit and reaches 0.30 temporal IoU.",
    "processing_guard": "A separate follow-up arm after the first results. Explain the whole "
                        "pair with one gain or finite impulse response and at most 2% uniform "
                        "resampling. Fit on alternating 40 ms blocks and require a 95th-percentile "
                        "relative RMS residual at most 0.10 on each held fold; preserve the "
                        "original alignment point and report the unguarded arm alongside it. "
                        "Infer the simplest ratio with denominator at most 256 that produces "
                        "the submitted sample count under upward rounding. Test the follow-up "
                        "on two AMI meetings outside point selection, using a different rate.",
    "current": "Measure the current earlier-copy app route with optional timing and speech enabled. "
               "Check every localized or exact difference range on the submitted clock, including gain-adjusted ranges.",
    "scope": "This is a reference-based difference comparison with no new model or AI attribution. "
             "The SINE and PartialEdit sources were used previously with different checks. "
             "SINE uses one reader and PartialEdit uses three. Two AMI meetings supply selection "
             "controls and two separate meetings supply assessment controls.",
}


def bank(name):
    stem, pin = BANKS[name]
    path = common.PANELS / f"{stem}.json"
    common.check(common.path_sha(path) == pin, f"Frozen {name} inputs differ")
    return common.read(path)


def matched_controls(original, wrong, *, rate_ratio=(101, 100)):
    import numpy as np
    from scipy.signal import butter, sosfilt

    return {
        "unchanged": original,
        "gain": np.rint(original * 0.63).astype("<i2"),
        "equalized": np.clip(np.rint(sosfilt(butter(2, 3300, fs=RATE, output="sos"), original)), -32768, 32767).astype("<i2"),
        "resampled": resampled(original, *rate_ratio),
        "wrong_excerpt": wrong,
    }


def wav(path, pcm):
    with wave.open(str(path), "wb") as stream:
        stream.setparams((1, 2, RATE, 0, "NONE", "not compressed"))
        stream.writeframes(pcm.tobytes())


def current(root, first, second):
    from app.evidence.audio_reference import compare as current_compare

    earlier, submitted = root / "earlier.wav", root / "submitted.wav"
    wav(earlier, first)
    wav(submitted, second)
    result, _ = current_compare(earlier, submitted, include_stereo=False, include_local_timing=True,
                                include_clean_speech=True)
    candidates = []
    for source in ("audio", "level_adjusted"):
        for span in result.get(source, {}).get("ranges", []):
            submitted_clock = span.get("file_ranges", {}).get("submitted", span)
            candidates.append([submitted_clock["start_sample"], submitted_clock["end_sample"]])
    speech = result["clean_speech"].get("range")
    timing = result["local_timing"].get("range")
    if speech:
        candidates.append(speech["samples"])
    if timing:
        candidates.append(timing["submitted_samples"])
    return {"status": result["status"], "spans": [list(span) for span in sorted(set(map(tuple, candidates)))]}


def partial_cases(folder, receipt, *, scenes=("clean", "construction")):
    import numpy as np

    roster = receipt["roster"]
    originals = {r["utterance"]: r for r in roster if (r["family"], r["scene"]) == ("genuine", "clean")}

    def load(row):
        raw = (folder / "pcm" / f"{row['id']:03}.pcm").read_bytes()
        common.check(len(raw) == 2 * row["samples"] and common.digest(raw) == row["pcm_sha256"],
                     "Prepared PartialEdit PCM does not match its published source commitment")
        return np.frombuffer(raw, dtype="<i2")

    for row in roster:
        if row["scene"] in scenes:
            first = load(originals[row["utterance"]])
            yield str(row["id"]), row["utterance"], f"{row['scene']}/{row['family']}", row["truth"], first, load(row)


def sine_cases(folder):
    import numpy as np

    intake = bank("sine")
    common.check(sine.audit(folder) == intake, "Publisher SINE intake differs")
    versions = {}
    for identifier, group in intake["cases"].items():
        versions[identifier] = {}
        for role, row in group.items():
            with wave.open(str(folder / row["filename"]), "rb") as stream:
                pcm = np.frombuffer(stream.readframes(stream.getnframes()), dtype="<i2")
            versions[identifier][role] = resampled(pcm, 2, 3)
    names = list(versions)
    for index, (identifier, roles) in enumerate(versions.items()):
        earlier = roles["real"]
        for role, signal in roles.items():
            if role != "real":
                a, b, _ = intake["cases"][identifier][role]["publisher_intervals"][1]
                yield f"{identifier}/{role}", identifier, role, [[a * 2 // 3, b * 2 // 3]], earlier, signal
        wrong = versions[names[(index + 1) % len(names)]]["real"]
        for role, signal in matched_controls(earlier, wrong).items():
            yield f"{identifier}/{role}", identifier, role, [], earlier, signal


def ordinary_cases(cache, *, assessment=False):
    import numpy as np

    source_rows = list(ami.inputs(cache, bank("ami")["freeze"]))
    meetings = AMI_ASSESSMENT if assessment else AMI_SELECTION
    rate_ratio = (75, 76) if assessment else (101, 100)
    for source, raw, neighbor in source_rows:
        if source["session"] not in meetings:
            continue
        first = np.frombuffer(raw, dtype="<i2")
        wrong = np.frombuffer(neighbor, dtype="<i2")
        for role, signal in matched_controls(first, wrong, rate_ratio=rate_ratio).items():
            if assessment:
                role = f"ami/{role}"
            yield f"{source['id']}/{role}", source["id"], role, [], first, signal


def score_rows(cases, *, baseline=False):
    with TemporaryDirectory(prefix="speech-alignment-") as tmp:
        for identifier, group, role, truth, first, second in cases:
            measurement = compare(first, second)
            measurement["processing_residual"] = processing_residual(first, second) if measurement["selected"] else None
            yield {"id": identifier, "group": group, "role": role, "truth": truth, "samples": [len(first), len(second)],
                   "input_sha256": [common.digest(x.tobytes()) for x in (first, second)],
                   "measurement": measurement,
                   "current": current(Path(tmp), first, second) if baseline else None}


def point(rows):
    return max(x["measurement"]["selected"]["score"] for x in rows if x["measurement"]["selected"] is not None)


def summarize(rows, threshold):
    totals = {}
    for row in rows:
        cell = totals.setdefault(row["role"], Counter())
        selected = row["measurement"]["selected"]
        residual = row["measurement"]["processing_residual"]
        flag = selected is not None and selected["score"] > threshold
        guarded = flag and (residual is None or residual > 0.10)
        useful = selected is not None and any(common.useful(selected["span"], truth) for truth in row["truth"])
        incumbent = row["current"] and row["current"]["spans"]
        cell.update({"planned": 1, "alignment_assessed": selected is not None, "alignment_flags": flag,
                     "alignment_useful": flag and useful, "alignment_useful_before_point": useful,
                     "guarded_flags": guarded, "guarded_useful": guarded and useful,
                     "current_has_difference": bool(incumbent),
                     "current_useful": bool(incumbent and any(common.useful(span, truth)
                                                            for span in incumbent for truth in row["truth"]))})
    return totals


def expected_roster():
    exact = {}
    for identifier, entries in bank("sine")["cases"].items():
        for role in ("infill", "cut_paste", *CONTROLS):
            truth = [[a*2//3, b*2//3] for a, b, state in entries[role]["publisher_intervals"] if state == "F"] if role in entries else []
            exact[f"{identifier}/{role}"] = (identifier, role, truth)
    for row in bank("partialedit")["roster"]:
        if row["scene"] in {"clean", "construction"}:
            exact[str(row["id"])] = (row["utterance"], f"{row['scene']}/{row['family']}", row["truth"])
    exact.update(ordinary_roster(AMI_ASSESSMENT, "ami/"))
    return exact


def ordinary_roster(meetings, prefix=""):
    return {f"{group}/{prefix}{role}": (group, prefix + role, [])
            for group in (f"{meeting}-{i}" for meeting in meetings for i in range(8))
            for role in CONTROLS}


def review(receipt):
    inputs = {key: values[1] for key, values in BANKS.items()}
    common.check(receipt["schema"] == SCHEMA and receipt["method"] == METHOD and receipt["inputs"] == inputs,
                 "Published method or inputs differ")
    expected = expected_roster()
    development = ordinary_roster(AMI_SELECTION)
    partial = bank("partialedit")["roster"]
    partial_by_id = {str(row["id"]): row for row in partial}
    partial_references = {row["utterance"]: row for row in partial
                          if (row["family"], row["scene"]) == ("genuine", "clean")}
    ami_references = {row["id"]: row for row in bank("ami")["freeze"]["sources"]}
    for split, roster in (("development", development), ("assessment", expected)):
        rows = receipt[split]
        common.check(len(rows) == len(roster), "Published row count differs")
        for row in rows:
            common.check(roster.pop(row["id"]) == (row["group"], row["role"], row["truth"]),
                         "Published case or annotation differs")
            measurement = row["measurement"]
            a, b = row["samples"]
            common.check(len(row["input_sha256"]) == 2 and all(SHA256.fullmatch(pin) for pin in row["input_sha256"])
                         and all(0 <= lo < hi <= b for lo, hi in row["truth"]), "Invalid inputs or truth range")
            if row["id"] in partial_by_id:
                submitted = partial_by_id[row["id"]]
                reference = partial_references[submitted["utterance"]]
                common.check(row["input_sha256"] == [reference["pcm_sha256"], submitted["pcm_sha256"]]
                             and row["samples"] == [reference["samples"], submitted["samples"]],
                             "Published PartialEdit clock or audio commitment differs")
            elif row["group"] in ami_references:
                reference = ami_references[row["group"]]
                common.check(a == ami.SAMPLES and row["input_sha256"][0] == reference["pcm_sha256"],
                             "Published AMI clock or audio commitment differs")
            if split == "assessment":
                spans = row["current"]["spans"]
                common.check(spans == sorted(spans) and len(set(map(tuple, spans))) == len(spans)
                             and all(0 <= lo < hi <= b for lo, hi in spans), "Invalid current output intervals")
            residual = measurement["processing_residual"]
            common.check(min(a, b) >= 29 * RATE // 5 and math.isfinite(measurement["median_path_cost"]),
                         "Invalid source clock or median cost")
            common.check(residual is None or (math.isfinite(residual) and residual >= 0
                         and measurement["selected"] is not None and abs(a-b)*50 <= a),
                         "Invalid processing explanation")
            if (selected := measurement["selected"]) is not None:
                start, stop = selected["span"]
                old_start, old_stop = selected["earlier_span"]
                left, middle, right = selected["means"]
                common.check(start % HOP == stop % HOP == 0 and start >= FLANK * HOP
                             and stop + FLANK * HOP <= b and (stop-start)//HOP in SIZES
                             and 0 <= old_start <= old_stop <= a and abs(middle-max(left, right)-selected["score"]) <= 0.000002
                             and 0 <= left <= 0.7 and 0 <= right <= 0.7 and measurement["median_path_cost"] <= 0.7,
                             "Published selected window or score differs")
    threshold = point(receipt["development"])
    summary = summarize(receipt["assessment"], threshold)
    common.check(threshold == receipt["point"] and receipt["summary"] == summary, "Published counts or point differ")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("score", "review"))
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--sine", type=Path)
    parser.add_argument("--partialedit", type=Path)
    parser.add_argument("--ami", type=Path)
    args = parser.parse_args()
    if args.phase == "score":
        if not all((args.sine, args.partialedit, args.ami)):
            parser.error("scoring requires all three verified source caches")
        if args.receipt.exists():
            parser.error("choose a new receipt path")
        development = list(score_rows(ordinary_cases(args.ami)))
        threshold = point(development)
        lock = args.receipt.with_suffix(".point.json")
        common.write(lock, {"development_sha256": common.digest(json.dumps(development, sort_keys=True).encode()),
                            "point": threshold, "method": METHOD})
        print(f"Fixed the point at {threshold:.6f} from {len(development)} AMI controls", flush=True)
        rows = list(score_rows(sine_cases(args.sine), baseline=True))
        rows.extend(score_rows(partial_cases(args.partialedit, bank("partialedit")), baseline=True))
        rows.extend(score_rows(ordinary_cases(args.ami, assessment=True), baseline=True))
        receipt = {"schema": SCHEMA, "method": METHOD, "inputs": {key: values[1] for key, values in BANKS.items()},
                   "point": threshold, "development": development, "assessment": rows, "summary": summarize(rows, threshold)}
        review(receipt)
        common.write(args.receipt, receipt)
    else:
        receipt = common.read(args.receipt)
    print(json.dumps(review(receipt), sort_keys=True))


if __name__ == "__main__":
    main()
