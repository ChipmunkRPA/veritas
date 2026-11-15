# SPDX-License-Identifier: MIT
"""Compare missing lip closure with independent speech and picture timing anchors."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

from scripts import video_mouth_closure_transfer as parent

phone = parent.phone
RATE = phone.RATE
SCHEMA = "chipmunk.video-mouth-clock-context/1"
NAMES = (*parent.M_NAMES, "b", "bʲ", "bʰ", "bː", "p", "pʲ", "pʰ", "pː")
SHIFTS = {-2560: "earlier_160", -1280: "earlier_80", 1280: "later_80", 2560: "later_160"}
METHODS = ("nominal", "corrected", "anchored_only")
OFFSET_GRID = range(-3840, 3841, 320)
TOLERANCE = 960
CLEARANCE = 6720
EDITED = {parent.POSITIVE, "dub_earlier_80", "dub_later_160"}
ROLES = (*parent.ROLES, *SHIFTS.values(), "dub_earlier_80", "dub_later_160")
DESIGN = {
    "sources": "Keep the previous six development and ten held speaker groups. Reuse their decoded native aperture "
               "and source clock. Rerun acoustic inference with M, B and P and four new clean soundtrack offsets, "
               "plus two local dubs with an additional global shift. Speakers and native pictures have prior exposure.",
    "anchor": "Use B or P with emission at least .55 and corresponding low-pass proposal within 40 ms. Compare "
              "them with independently detected visible lip releases. Retain the separate points for M midpoint "
              "and B/P release from the preceding studies.",
    "clock": "Exclude B/P anchors within 420 ms of an M proposal. Search audio-minus-picture shifts from -240 to +240 ms. "
             "At each shift match to a unique nearest visible release within 60 ms. Require two different releases "
             "at least 160 ms apart. Use the median residual and decline clusters whose members differ by more than 60 ms. "
             "Among maximal support clusters choose lowest sum of residual errors. Decline another maximal-support "
             "cluster 80 ms away with residual error at most 25 ms larger. Use zero when context is missing in corrected arm.",
    "test": "Score the minimum observed lip aperture in the same 100/160 ms windows and retain the strongest M candidate. "
            "Use the previous fixed points and separately show correction with mandatory anchor support. Display picture and "
            "sound ranges on their own clocks. Require .40 temporal IoU and the same M time tolerance as the parent.",
}


def observed_phones(raw, filtered):
    return [event for event in raw if event[2] >= parent.ANCHOR_POINT
            and any(event[0] == other[0] and abs(event[1] - other[1]) <= 640 for other in filtered)]


def pair_anchors(events, releases, hypothesis, bias):
    matches = []
    for name, sample, _ in events:
        predicted = sample - bias - hypothesis
        choices = sorted((abs(predicted - row["sample"]), row["sample"], row["frames"]) for row in releases)
        if not choices or choices[0][0] > TOLERANCE or (len(choices) > 1 and choices[0][0] == choices[1][0]):
            continue
        error, picture, frames = choices[0]
        matches.append((error, name, sample, picture, frames))
    picked, used = [], set()
    for _, name, sample, picture, frames in sorted(matches):
        if picture not in used:
            picked.append({"phone": name, "audio": sample, "picture": picture, "frames": frames})
            used.add(picture)
    return sorted(picked, key=lambda item: item["picture"])


def estimate_lag(events, releases, query, bias):
    distant = [e for e in events if e[0] in ("b", "p") and abs(e[1] - query) >= CLEARANCE]
    models = {}
    for hypothesis in OFFSET_GRID:
        pairs = pair_anchors(distant, releases, hypothesis, bias)
        if len(pairs) < 2 or pairs[-1]["picture"] - pairs[0]["picture"] < 2560:
            continue
        offsets = [pair["audio"] - pair["picture"] - bias for pair in pairs]
        if max(offsets) - min(offsets) > 960:
            continue
        lag = round(statistics.median(offsets))
        if not OFFSET_GRID.start <= lag < OFFSET_GRID.stop:
            continue
        error = sum(abs(value - lag) for value in offsets)
        key = tuple((pair["audio"], pair["picture"]) for pair in pairs)
        models[key] = (-len(pairs), error, abs(lag), lag, pairs)
    if not models:
        return None
    ranked = sorted(models.values(), key=lambda item: item[:4])
    best = ranked[0]
    if any(other[0] == best[0] and other[1] <= best[1] + 400 and abs(other[3] - best[3]) >= 1280 for other in ranked[1:]):
        return None
    return {"lag": best[3], "pairs": best[4]}


def read_picture(frames, center, half):
    if center - half < frames[0]["sample"] or center + half > frames[-1]["sample"]:
        return None
    kept = [row for row in frames if abs(row["sample"] - center) <= half]
    if (len(kept) < 4 or any(row["gap"] is None for row in kept)
            or any(not 0 < right["sample"] - left["sample"] <= 800 for left, right in zip(kept, kept[1:]))):
        return None
    return {"frames": [row["index"] for row in kept], "minimum_aperture": min(row["gap"] for row in kept),
            "picture_range": [center - half, center + half]}


def readings(case, pictures, closure_bias, release_bias):
    observed = observed_phones(case["acoustic"], case["lowpass"])
    releases = parent.releases.visual_releases(pictures)
    result = {}
    for method in METHODS:
        for name, half in parent.HALF_WINDOWS.items():
            choices = []
            for phoneme, audio_sample, strength in observed:
                if phoneme != "m":
                    continue
                timing = estimate_lag(observed, releases, audio_sample, release_bias) if method != "nominal" else None
                if method == "anchored_only" and timing is None:
                    continue
                lag = timing["lag"] if timing else 0
                mapped = audio_sample - closure_bias
                picture = read_picture(pictures, mapped - lag, half)
                if picture:
                    choices.append({**picture, "audio_event": audio_sample, "posterior": strength, "timing": timing,
                                    "expected_closure": mapped, "range": [mapped - half, mapped + half]})
            result[f"{method}/{name}"] = max(choices, key=lambda item: (item["minimum_aperture"], -item["audio_event"]), default=None)
    return result


def treatments(sounds, plan):
    import numpy as np

    def shift(samples, amount):
        return (np.pad(samples[:-amount], (amount, 0)) if amount > 0
                else np.pad(samples[-amount:], (0, -amount)))

    base = parent.deliveries(sounds, plan)
    return {**base, **{role: shift(sounds["m"], amount) for amount, role in SHIFTS.items()},
            "dub_earlier_80": shift(base[parent.POSITIVE], -1280), "dub_later_160": shift(base[parent.POSITIVE], 2560)}


def side(role):
    return "m" if role in (*parent.M_ROLES, *SHIFTS.values()) else "vowel"


def shift_amount(role):
    amounts = {value: key for key, value in SHIFTS.items()}
    return (amounts | {"dub_earlier_80": -1280, "dub_later_160": 2560, "m_global_early": -1920,
                       "m_global_late": 1920}).get(role, 0)


def truth(row, role):
    center = row["sources"]["vowel"]["center"] + shift_amount(role)
    donor = row["sources"]["m_alternate"]
    return [center - parent.SPLICE_HALF, center + parent.SPLICE_HALF], [v + center - donor["center"] for v in donor["aligned"]]


def outcome(closure, release, rows, selected_split):
    fit = closure["fitted"]["fit"]
    release_bias = release["result"]["bias"]["neural"]["value"]
    records = closure["fitted"]["development"]["rows"] if selected_split == "development" else closure["held"]["rows"]
    fit_rows = [row for row in closure["freeze"]["rows"] if row["split"] == selected_split]
    counts, found = defaultdict(Counter), []
    for row, pictures, record in zip(fit_rows, records, rows, strict=True):
        phone.require(record["speaker"] == row["speaker"] and record["condition"] == row["condition"]
                      and set(record["cases"]) == set(ROLES), "Case identity differs from the frozen source")
        for role in ROLES:
            case = record["cases"][role]
            kind = ("edit" if role == parent.POSITIVE else "shifted_edit" if role in EDITED else
                    "constant_shift" if shift_amount(role) else "ordinary")
            native = row["sources"][side(role)]["samples"]
            original = pictures["cases"].get(role)
            if original:
                phone.require(case["pcm_sha256"] == original["pcm_sha256"], "Earlier delivery PCM differs")
            phone.require(parent.releases.SHA256.fullmatch(case["pcm_sha256"])
                          and all(p in ("m", "b", "p") and 0 <= t < native and .35 <= s <= 1
                                  for key in ("acoustic", "lowpass") for p, t, s in case[key]), "Acoustic event is outside the source")
            view = pictures["picture"][side(role)]["frames"]
            decisions = readings(case, view, fit["clock"]["offset"], release_bias)
            for key, decision in decisions.items():
                method, window = key.split("/")
                measured = decision is not None
                context = measured and decision["timing"] is not None
                for point in ("tight", "broad"):
                    flagged = measured and decision["minimum_aperture"] > fit["points"][window][point]
                    hit = role in EDITED and flagged and parent.useful(decision, *truth(row, role))
                    correct_clock = bool(context and abs(decision["timing"]["lag"] - shift_amount(role)) <= 960)
                    counts[key, point, kind].update(files=1, measured=int(measured), supported_clock=int(context),
                                                   accurate_clock=int(correct_clock), flagged=int(flagged), useful=int(hit))
                    if flagged:
                        found.append({"case": [row["speaker"], row["condition"], role], "method": method, "window": window,
                                      "point": point, "correct_range": hit, "reading": decision})
    return {"counts": {"/".join(keys): dict(value) for keys, value in sorted(counts.items())}, "findings": found}


def measure(args, closure):
    import sys

    media, metadata = phone.source_archives(args.sources)
    phone.require(metadata == closure["freeze"]["archives"], "Publisher archives differ")
    sys.path.insert(0, str(args.runtime))
    infer, device = phone.neural_decoder(args.model, names=NAMES)
    observations = []
    with ZipFile(media["audio"]) as archive:
        for row in (row for row in closure["freeze"]["rows"] if row["split"] == args.phase):
            plan = row["sources"]
            sounds = {key: phone.pcm(archive.read(item["files"]["audio"])) for key, item in plan.items()}
            phone.require(all(phone.digest(pcm.tobytes()) == plan[key]["audio_pcm_sha256"] for key, pcm in sounds.items()),
                          "Publisher sound differs")
            cases = {}
            for role, pcm in treatments(sounds, plan).items():
                acoustic = infer(pcm)
                filtered = infer(parent.filtered(pcm)) if acoustic else []
                cases[role] = {"acoustic": acoustic, "lowpass": filtered, "pcm_sha256": phone.digest(pcm.tobytes())}
            observations.append({"speaker": row["speaker"], "condition": row["condition"], "cases": cases})
            print(args.phase, len(observations), flush=True)
    return {"device": device, "rows": observations}


def replay_native(args, closure, receipt):
    from app.detectors.video.decode import decode_frames, probe_frames

    media, metadata = phone.source_archives(args.sources)
    phone.require(metadata == closure["freeze"]["archives"], "Publisher archives differ")
    selected_split = receipt["split"]
    selected = defaultdict(set)
    for finding in receipt["outcome"]["findings"]:
        speaker, condition, role = finding["case"]
        reading = finding["reading"]
        witness = selected[speaker, condition, side(role)]
        witness.update(reading["frames"])
        if reading["timing"]:
            for pair in reading["timing"]["pairs"]:
                witness.update(pair["frames"])
    planned = [row for row in closure["freeze"]["rows"] if row["split"] == selected_split]
    picture_data = closure["fitted"]["development"]["rows"] if selected_split == "development" else closure["held"]["rows"]
    summary = Counter()
    with (ZipFile(media["front"]) as video, ZipFile(media["audio"]) as audio,
          TemporaryDirectory(prefix="chipmunk-av-clock-replay-") as tmp):
        movie = Path(tmp) / "sample.mov"
        for source, pictures, measured in zip(planned, picture_data, receipt["measurement"]["rows"], strict=True):
            plan = source["sources"]
            sound = {key: phone.pcm(audio.read(item["files"]["audio"])) for key, item in plan.items()}
            phone.require(all(phone.digest(pcm.tobytes()) == plan[key]["audio_pcm_sha256"] for key, pcm in sound.items()),
                          "Original audio differs")
            views = treatments(sound, plan)
            for role, pcm in views.items():
                phone.require(phone.digest(pcm.tobytes()) == measured["cases"][role]["pcm_sha256"], "Submitted PCM differs")
                if shift := shift_amount(role):
                    parent_role = parent.POSITIVE if role in EDITED else "m_original"
                    base = views[parent_role]
                    zero, shifted, original = ((pcm[:shift], pcm[shift:], base[:-shift]) if shift > 0
                                               else (pcm[shift:], pcm[:shift], base[-shift:]))
                    phone.require(not zero.any() and (shifted == original).all(), "The claimed global shift differs")
                    summary["constant_clock_shifts"] += 1
                summary["delivered_waveforms"] += 1
            for key in ("m", "vowel"):
                wanted = selected[source["speaker"], source["condition"], key]
                if not wanted:
                    continue
                data = video.read(plan[key]["files"]["front"])
                phone.require(phone.digest(data) == plan[key]["front_sha256"], "Native picture source differs")
                movie.write_bytes(data)
                table = probe_frames(movie, cap=160, timeout_s=18, max_pixels=1920 * 1080)
                expected = pictures["picture"][key]
                phone.require(table.time_base == expected["time_base"] and not table.interpolated and not table.truncated
                              and table.pts == tuple(row["pts"] for row in expected["frames"]), "Native picture clock differs")
                verified = set()
                for frame in decode_frames(movie, table, sorted(wanted), timeout_s=25):
                    anchor = expected["frames"][frame.index]
                    x0, y0, x1, y1 = anchor["mouth_box"]
                    phone.require(frame.pts_verified and frame.pts == anchor["pts"]
                                  and phone.digest(frame.rgb.tobytes()) == anchor["rgb_sha256"]
                                  and phone.digest(frame.rgb[y0:y1, x0:x1].tobytes()) == anchor["mouth_crop_sha256"],
                                  "A displayed input does not match its committed pixels")
                    verified.add(frame.index)
                phone.require(verified == wanted, "Some supporting pictures could not be verified")
                summary["supported_source_pictures"] += len(verified)
    phone.write(args.output, {"receipt_sha256": phone.sha(args.path), **summary})


def main():
    root = Path(__file__).resolve().parents[2] / "docs/evaluation-panels"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("development", "held", "review", "replay"))
    parser.add_argument("path", type=Path)
    parser.add_argument("--closure", type=Path, default=root / "video-mouth-closure-transfer-2026-11-14.json")
    parser.add_argument("--releases", type=Path, default=root / "video-bilabial-mouth-transfer-2026-11-14.json")
    parser.add_argument("--sources", type=Path)
    parser.add_argument("--runtime", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    closure, release = (json.loads(path.read_bytes()) for path in (args.closure, args.releases))
    frozen = {"schema": SCHEMA, "design": DESIGN, "parents": [phone.sha(args.closure), phone.sha(args.releases)]}
    if args.phase in ("review", "replay"):
        receipt = json.loads(args.path.read_bytes())
        phone.require(receipt["freeze"] == frozen, "Frozen method or parent receipt differs")
        result = outcome(closure, release, receipt["measurement"]["rows"], receipt["split"])
        phone.require(result == receipt["outcome"], "Recorded clock results do not replay")
        if args.phase == "replay":
            if args.sources is None or args.output is None:
                parser.error("Native replay requires --sources and --output")
            replay_native(args, closure, receipt)
        print(json.dumps(result["counts"], sort_keys=True, indent=2))
        return
    if args.sources is None or args.runtime is None or args.model is None:
        parser.error("Measurement requires --sources, --runtime and --model")
    measured = measure(args, closure)
    receipt = {"freeze": frozen, "split": args.phase, "measurement": measured,
               "outcome": outcome(closure, release, measured["rows"], args.phase)}
    phone.write(args.path, receipt)


if __name__ == "__main__":
    main()
