"""Measure earlier-copy speech alignment on two released neural-edit panels."""

from __future__ import annotations

import argparse
import json
import math
import re
import wave
from collections import Counter
from fractions import Fraction
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts import audio_source_isolation_diagnostic as common
from scripts import replay_audio_bandwidth_public as ami
from scripts import review_audio_sine_preview as sine

SCHEMA = "chipmunk.audio-aligned-speech-difference/1"
RATE = 16_000
HOP = 320
WINDOW = 640
FLANK = 20
SIZES = (6, 10, 16, 24, 36, 54, 81, 120)
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


def resampled(pcm, up, down):
    import numpy as np
    from scipy.signal import resample_poly

    return np.clip(np.rint(resample_poly(pcm.astype(np.float64), up, down)), -32768, 32767).astype("<i2")


def spectral_features(pcm):
    import numpy as np
    from scipy.fft import dct

    frame = np.lib.stride_tricks.sliding_window_view(np.pad(pcm.astype(np.float64), WINDOW // 2), WINDOW)[::HOP]
    frame = frame[:len(pcm) // HOP]
    level = np.sqrt(np.mean(frame**2, axis=1))
    active = level >= max(10, np.percentile(level, 85) * 0.12)
    power = abs(np.fft.rfft(frame * np.hanning(WINDOW), n=1024))**2
    hz = np.fft.rfftfreq(1024, 1 / RATE)
    mel = 2595 * np.log10(1 + hz / 700)
    corners = np.linspace(2595 * np.log10(1 + 200 / 700), 2595 * np.log10(1 + 6800 / 700), 30)
    low, center, high = corners[:-2, None], corners[1:-1, None], corners[2:, None]
    filters = np.maximum(0, np.minimum((mel - low) / (center - low), (high - mel) / (high - center)))
    bands = power @ (filters / filters.sum(axis=1, keepdims=True)).T
    floor = max(1., float(np.percentile(bands, 90))) * 1e-5
    features = dct(np.log(np.maximum(bands, floor)), norm="ortho", axis=1)[:, 1:13]
    features -= np.median(features[active], axis=0) if active.any() else np.median(features, axis=0)
    padded = np.pad(features, ((2, 2), (0, 0)), mode="edge")
    delta = (padded[4:] - padded[:-4]) * 0.5
    return np.concatenate((features, delta), axis=1), active


def path_indices(cost):
    """Evaluate the three DTW predecessors on one antidiagonal at a time."""
    import numpy as np

    m, n = cost.shape
    total = np.full((m + 1, n + 1), np.inf)
    previous = np.zeros((m + 1, n + 1), dtype=np.uint8)
    total[0, 0] = 0
    for diagonal in range(2, m + n + 1):
        i = np.arange(max(1, diagonal - n), min(m, diagonal - 1) + 1)
        j = diagonal - i
        within_band = abs(j - 1 - (i - 1) * (n - 1) / (m - 1)) <= RATE * 0.8 / HOP
        i, j = i[within_band], j[within_band]
        values = np.stack((total[i-1, j-1], total[i-1, j] + 0.08, total[i, j-1] + 0.08))
        previous[i, j] = np.argmin(values, axis=0)
        total[i, j] = np.min(values, axis=0) + cost[i-1, j-1]
    i, j = m, n
    result = []
    while i and j:
        result.append((i-1, j-1))
        step = previous[i, j]
        i -= step != 2
        j -= step != 1
    common.check(i == j == 0 and math.isfinite(total[m, n]), "Could not reconstruct a complete alignment")
    return np.array(result[::-1]).T


def compare(first, second):
    import numpy as np
    from scipy.spatial.distance import cdist

    a, active_a = spectral_features(first)
    b, active_b = spectral_features(second)
    cost = np.sqrt(cdist(a, b, metric="sqeuclidean") / a.shape[1])
    earlier, submitted = path_indices(cost)
    n = len(b)
    counts = np.bincount(submitted, minlength=n)
    matches = np.bincount(submitted, weights=cost[earlier, submitted], minlength=n) / counts
    positions = np.bincount(submitted, weights=earlier, minlength=n) / counts
    activity = np.bincount(submitted, weights=active_a[earlier] & active_b[submitted], minlength=n) / counts
    path_median = round(float(np.median(matches)), 6)
    best = None
    if path_median <= 0.7:
        cumulative = np.r_[0., np.cumsum(matches)]
        spoken = np.r_[0., np.cumsum(activity)]
        for width in SIZES:
            start = np.arange(FLANK, n - FLANK - width + 1)
            end = start + width
            middle = (cumulative[end] - cumulative[start]) / width
            left = (cumulative[start] - cumulative[start-FLANK]) / FLANK
            right = (cumulative[end+FLANK] - cumulative[end]) / FLANK
            supported = ((left <= 0.7) & (right <= 0.7)
                         & (spoken[start] - spoken[start-FLANK] >= 0.3 * FLANK)
                         & (spoken[end+FLANK] - spoken[end] >= 0.3 * FLANK))
            scores = np.where(supported, middle - np.maximum(left, right), -np.inf)
            if not np.any(supported):
                continue
            k = int(np.argmax(scores))
            candidate = {"score": round(float(scores[k]), 6), "span": [int(start[k]) * HOP, int(end[k]) * HOP],
                         "earlier_span": [round(positions[start[k]]) * HOP, (round(positions[end[k]-1]) + 1) * HOP],
                         "means": [round(float(value), 6) for value in (left[k], middle[k], right[k])]}
            if best is None or (candidate["score"], -candidate["span"][0]) > (best["score"], -best["span"][0]):
                best = candidate
    return {"median_path_cost": path_median, "selected": best}


def processing_residual(first, second):
    import numpy as np

    if abs(len(second) - len(first)) * 50 > len(first):
        return None
    n = len(first)
    m = len(second)
    center = Fraction(2*m - 1, 2*n)
    ratio = None
    for denominator in range(1, 257):
        nearest = center.limit_denominator(denominator)
        if m - 1 < nearest * n <= m:
            ratio = nearest
            break
    if ratio is None:
        return None
    reference = resampled(first, ratio.numerator, ratio.denominator) if ratio != 1 else first
    common.check(len(reference) == m, "Resampled processing explanation does not match the submitted clock")
    n = m
    half = 24
    features = np.lib.stride_tricks.sliding_window_view(reference[:n].astype(np.float64), 2 * half + 1)
    expected = second[half:n-half].astype(np.float64)
    blocks = len(expected) // WINDOW
    design = features[:blocks * WINDOW].reshape(blocks, WINDOW, -1)
    target = expected[:blocks * WINDOW].reshape(blocks, WINDOW)
    energy = np.sqrt(np.mean(target**2, axis=1))
    active = energy >= max(32, np.percentile(energy, 85) * 0.12)
    fold = np.arange(blocks) % 2
    values = []
    for training in (0, 1):
        fit = (fold == training) & active
        test = (fold != training) & active
        if min(fit.sum(), test.sum()) < 12:
            return None
        weights, *_ = np.linalg.lstsq(design[fit, ::8].reshape(-1, design.shape[-1]), target[fit, ::8].reshape(-1),
                                      rcond=1e-6)
        error = np.sqrt(np.mean((design[test] @ weights - target[test])**2, axis=1))
        values.append(float(np.percentile(error / energy[test], 95)))
    return round(max(values), 6)


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
