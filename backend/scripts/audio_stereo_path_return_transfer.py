"""Test whether a short stereo recording-path change can be localized in noise."""

from __future__ import annotations

import argparse
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import fft, signal

from scripts import audio_odss_matched_room as acoustic
from scripts import audio_wpe_fresh_room_transfer as source

SCHEMA = "chipmunk.audio-stereo-path-return/1"
RATE = acoustic.RATE
SAMPLES = acoustic.SAMPLES
HOP = 320
FRAME = 1280
CELL = 1600
DURATIONS = (7, 9, 11)
LEVELS = (None, 5, 0, -5)
KINDS = ("untouched", "room_only", "ordinary_same_path", "ordinary_path_change",
         "synthetic_same_path", "synthetic_path_change", "gradual_move")
ARMS = ("level_only", "joint")
METHOD = {
    "measurement": "On 80 ms Hann windows every 20 ms, band-limit both channels to 300–2400 Hz. In each frame measure the strongest positive normalized cross-correlation at lags −16 through +16 samples, and the signed left/right RMS level ratio. Aggregate median lag, level and correlation over each 100 ms cell, using only energetic positively coherent frames.",
    "return": "Scan 0.7, 0.9 and 1.1 second candidate intervals on the 100 ms grid. Require matching five-cell flanks on both sides, with a one-cell guard. Compare the inside against both flanks; subtract any inside fluctuation beyond the flanks and half the flank distance. Compare level alone and equal-unit delay, level and correlation distance. Fit both points to the strict maximum of eligible unedited, room-only and same-path ordinary development views, including all noise levels.",
    "sources": "Use previously licensed and analyzed speech from separate reader groups. Author new matched path interventions using pinned Surrey and BBC measured stereo responses. Match ordinary and published synthetic donor durations and local gain, apply the same 20 ms fades, and reuse each background segment, level, and peak gain across versions. Include unchanged, masker-only, ordinary same-path and ordinary gradual-move controls. Synthetic and ordinary wrong-path inserts are both localization positives. Same-path synthetic inserts are separately reported as coverage probes, with no origin inference from this cue.",
    "decision": "Keep the strongest range in each view. Evaluate 0.30 interval IoU and 50% edit coverage and count additional range flags separately on each control. The range describes a temporary change in the recorded stereo field.",
}


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def feature_frames(stereo, *, alternate=False):
    values = stereo.astype(np.float64)
    windows = np.lib.stride_tricks.sliding_window_view(values, FRAME, axis=0)[::HOP]
    windows = windows * signal.windows.hann(FRAME, sym=False)
    spectrum = fft.rfft(windows, axis=-1) if not alternate else np.fft.rfft(windows, axis=-1)
    freq = np.fft.rfftfreq(FRAME, 1 / RATE)
    spectrum[:, :, (freq < 300) | (freq > 2400)] = 0
    inverse = np.fft.irfft if alternate else fft.irfft
    filtered = inverse(spectrum, n=FRAME, axis=-1)
    energy = np.sqrt(np.mean(filtered ** 2, axis=-1))
    product = spectrum[:, 0, :] * spectrum[:, 1, :].conj()
    cross = inverse(product, n=FRAME, axis=-1)
    lags = np.arange(-16, 17)
    candidates = cross[:, lags] / np.maximum(np.prod(energy, axis=1)[:, None] * FRAME, 1e-8)
    chosen = np.argmax(candidates, axis=1)
    corr = candidates[np.arange(len(chosen)), chosen]
    levels = 20 * np.log10(np.maximum(energy[:, 0], 1e-8) / np.maximum(energy[:, 1], 1e-8))
    strong = np.max(energy, axis=1)
    eligible = (np.min(energy, axis=1) >= max(12, float(np.quantile(strong, .90)) * .12)) & (corr >= .50)
    centers = np.arange(len(windows)) * HOP + FRAME // 2
    features = np.column_stack((lags[chosen], levels, corr))
    result = np.full((SAMPLES // CELL, 3), np.nan)
    for cell in range(len(result)):
        subset = features[eligible & (centers >= cell * CELL) & (centers < (cell + 1) * CELL)]
        if len(subset) >= 2:
            result[cell] = np.median(subset, axis=0)
    return result


def distances(data, center, arm):
    difference = data - center
    if arm == "level_only":
        return np.abs(difference[..., 1]) / 3
    return np.linalg.norm(difference / np.array((4., 3., .15)), axis=-1)


def summary(cells, arm):
    valid = cells[np.isfinite(cells).all(axis=1)]
    if len(valid) < max(3, (len(cells) + 1) // 2):
        return None
    center = np.median(valid, axis=0)
    return center, float(np.median(distances(valid, center, arm)))


def best(cells, arm):
    found = []
    for width in DURATIONS:
        for start in range(6, len(cells) - width - 5):
            stop = start + width
            regions = (summary(cells[start-6:start-1], arm), summary(cells[start:stop], arm),
                       summary(cells[stop+1:stop+6], arm))
            if any(region is None for region in regions):
                continue
            (before, bl), (inside, il), (after, al) = regions
            flank = float(distances(before, after, arm))
            if flank > 1.5 + max(bl, al):
                continue
            change = min(float(distances(inside, before, arm)), float(distances(inside, after, arm)))
            score = max(0., change - .5 * flank - max(0., il - max(bl, al)))
            found.append((round(score, 7), -start, -width))
    if not found:
        return None
    score, start, width = max(found)
    return {"range_samples": [-start * CELL, (-start - width) * CELL], "score": score}


def measure(stereo):
    nominal = feature_frames(stereo)
    repeat = feature_frames(stereo, alternate=True)
    require(np.allclose(nominal, repeat, atol=1e-9, rtol=0, equal_nan=True), "Independent FFT did not reproduce features")
    first, second = ({arm: best(cells, arm) for arm in ARMS} for cells in (nominal, repeat))
    require(first == second, "Independent FFT changed the proposed range")
    return first


def equalize(values, reference):
    return values * acoustic.rms(reference.mean(axis=1)) / max(acoustic.rms(values.mean(axis=1)), 1e-8)


def edit(host, donor, start, width):
    end = start + width
    source_pcm = donor.astype(np.float64)
    dest = host[start:end].astype(np.float64)
    gain = math.sqrt((1 + np.mean(dest ** 2)) / (1 + np.mean(source_pcm ** 2)))
    require(.2 < gain < 5, "Source donor is not sufficiently audible to construct a fair comparison")
    content = np.zeros(SAMPLES, dtype=np.float64)
    alpha = np.zeros(SAMPLES, dtype=np.float64)
    alpha[start:end] = np.r_[np.linspace(0, 1, 320), np.ones(width-640), np.linspace(1, 0, 320)]
    content[start:end] = source_pcm * gain * alpha[start:end]
    return content, alpha


def available_assets(args):
    groups, excluded = source.source_groups(args)
    result = []
    for group, role, reader, pin, host, donors in groups:
        if not donors and len(host) >= SAMPLES:
            result.append((group, role, reader, pin, host, (), ()))
        if donors:
            result.append((group, role, reader, pin, host, donors, source.ANCHORS))
    require(sum(bool(row[5]) for row in result) == 2 and len(result) >= 5, "Earlier qualified donor/reader schedule differs")
    return result, excluded


def renders(host, real, fake, start, width, position, rig):
    convolve = acoustic.convolve
    original = convolve(host, rig["front"])
    result = {"untouched": original}
    different = equalize(convolve(host, rig[position]), original)
    # A smooth move persists across both sides of the interval.
    panning = np.clip((np.arange(SAMPLES) - start + RATE) / (width + 2 * RATE), 0, 1)[:, None]
    result["gradual_move"] = (1 - panning) * original + panning * different
    for name, donor in (("ordinary", real), ("synthetic", fake)):
        if donor is None:
            continue
        insert, alpha = edit(host, donor, start, width)
        retained = convolve(host * (1 - alpha), rig["front"])
        normal = convolve(insert, rig["front"])
        shifted = equalize(convolve(insert, rig[position]), normal)
        result[f"{name}_same_path"] = retained + normal
        result[f"{name}_path_change"] = retained + shifted
    return result


def score_input(item, kind, span, role, identity, measured):
    return {"case": identity, "group": item, "kind": kind, "role": role, "truth": span,
            "measured_snr_db": measured}


def prepare(args):
    groups, excluded = available_assets(args)
    impulse, impulse_pins = acoustic.room_impulses(args.surrey, args.bbc)
    background_specs = source.ami_inputs(args)
    background_used = defaultdict(set)
    prior = acoustic.read(acoustic.ROOT / "docs/evaluation-panels/audio-wpe-fresh-room-transfer-2026-11-13.json")
    before = args.prior_run / "roster.json"
    require(acoustic.file_digest(before) == prior["roster_sha256"], "Earlier room-input roster differs")
    for row in acoustic.read(before):
        background_used[row["room"]].add(row["background"]["start"] // 160)
    rows, waveforms, private = [], [], []
    for group, role, reader, pin, samples, donors, anchors in groups:
        host = samples[:SAMPLES]
        layout, background_room = ("development", "meeting") if role == "development" else ("assessment", "lecture")
        planned = list(zip(donors, anchors, strict=True)) if donors else [(None, source.ANCHORS[0])]
        for index, (donor, (start, stop)) in enumerate(planned):
            name = f"{group}-{index}"
            width = stop - start
            ordinary = (samples[SAMPLES:SAMPLES+width] if len(samples) >= SAMPLES+width else samples[-width:])
            fake = None if donor is None else donor[1]
            where = "left" if index == 0 else "right"
            versions = renders(host, ordinary, fake, start, width, where, impulse[layout])
            masker, desc = source.choose_background(background_specs[background_room], background_room, name,
                                                     background_used[background_room], paired=bool(donors))
            mask = equalize(acoustic.convolve(masker, impulse[layout]["right"]), versions["untouched"])
            peak = max(float(np.max(abs(version) + abs(mask) * 10 ** .25)) for version in versions.values())
            peak_gain = 24000 / peak
            references = {kind: np.rint(values * peak_gain).astype(np.int32) for kind, values in versions.items()}
            inputs = {}
            for level in LEVELS:
                noise = None if level is None else np.rint(mask * peak_gain * 10 ** (-level / 20)).astype(np.int32)
                measured = None if level is None else float(20 * np.log10(acoustic.rms(references["untouched"].mean(axis=1))
                                                                            / acoustic.rms(noise.mean(axis=1))))
                require(measured is None or abs(measured - level) < .02, "Frozen foreground level differs")
                deliveries = references if noise is None else {**{key: value + noise for key, value in references.items()},
                                                               "room_only": noise}
                for kind, waveform in deliveries.items():
                    require(np.max(abs(waveform)) <= 24001, "Unexpected peak")
                    if role == "development" and kind not in ("untouched", "room_only", "ordinary_same_path", "ordinary_path_change", "gradual_move"):
                        continue
                    case = f"{name}:{level}:{kind}"
                    measured_mix = None if kind == "room_only" or measured is None else round(measured, 6)
                    truth = [start, stop] if "path_change" in kind or "same_path" in kind else None
                    record = score_input(name, kind, truth, role, case, measured_mix)
                    delivered = waveform.astype("<i2")
                    inputs[case] = acoustic.digest(delivered.tobytes())
                    rows.append({**record, "level_db": level, "input_sha256": inputs[case],
                                 "donor_family": donor[-1] if donor is not None and kind.startswith("synthetic") else None})
                    waveforms.append(delivered)
            private.append({"group": name, "reader": reader, "host": pin, "donor": None if donor is None else donor[0],
                            "background": desc, "inputs": inputs})
    require(len({row["input_sha256"] for row in rows}) == len(rows), "Frozen deliveries alias")
    lock = {"schema": SCHEMA, "method": METHOD, "clock_hz": RATE, "waveform_samples": SAMPLES, "levels_db": LEVELS,
            "room_pins": impulse_pins, "source_ancestry": source.PREFIX_SHA, "ami_pins": source.AMI_PINS,
            "ancestral_exclusion_count": excluded, "code_sha256": acoustic.file_digest(Path(__file__)),
            "source_code_sha256": acoustic.file_digest(Path(source.__file__)),
            "private_source_roster_sha256": acoustic.digest(acoustic.json_bytes(private)),
            "delivery_roster_sha256": acoustic.digest(acoustic.json_bytes(rows))}
    return lock, rows, waveforms


def results(rows, waveforms, role):
    return [{**row, "findings": measure(stereo)} for row, stereo in zip(rows, waveforms, strict=True) if row["role"] == role]


def fit(development):
    controls = [row for row in development if row["kind"] in ("untouched", "room_only", "ordinary_same_path", "gradual_move")]
    return {arm: max(row["findings"][arm]["score"] for row in controls if row["findings"][arm] is not None) for arm in ARMS}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "fit", "score", "replay"))
    for arg in ("run", "prefix", "publisher", "directional", "aachen", "stress-run", "prior-run", "ami", "surrey", "bbc"):
        parser.add_argument(f"--{arg}", type=Path, required=True)
    args = parser.parse_args()
    lock, rows, waves = prepare(args)
    frozen = args.run / "freeze.json"
    roster = args.run / "delivery-roster.json"
    if args.phase == "freeze":
        args.run.mkdir(mode=0o700)
        acoustic.write(frozen, lock)
        acoustic.write(roster, rows)
        print(f"Froze {len(rows)} stereo deliveries")
        return
    require(frozen.read_bytes() == acoustic.json_bytes(lock) and roster.read_bytes() == acoustic.json_bytes(rows),
            "Frozen code or waveform commitments differ")
    dev = args.run / "development.json"
    if args.phase == "fit":
        scored = results(rows, waves, "development")
        acoustic.write(dev, {"rows": scored, "points": fit(scored)})
        print(f"Fitted on {len(scored)} development deliveries")
        return
    values = acoustic.read(dev)
    require(values["points"] == fit(values["rows"]), "Development point differs")
    scored = results(rows, waves, "assessment")
    path = args.run / "public.json"
    result = {"freeze": lock, "development": values, "assessment": scored}
    if args.phase == "replay":
        require(path.read_bytes() == acoustic.json_bytes(result)
                and results(rows, waves, "development") == values["rows"], "Score replay differs")
        print(f"Replayed {len(rows)} waveform commitments and strongest ranges")
        return
    acoustic.write(path, result)
    print(f"Scored {len(scored)} held deliveries")


if __name__ == "__main__":
    main()
