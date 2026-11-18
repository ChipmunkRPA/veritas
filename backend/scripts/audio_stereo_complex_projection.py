# SPDX-License-Identifier: Apache-2.0
"""Recount a frequency-dependent foreground cancellation trial on stereo Opus."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory

SCHEMA = "chipmunk.audio-stereo-complex-projection/1"
SCAN = 512
HOP = 128
SUPPORT = (160, 3000)
MAX_DEVIATION = .25
SCENES = {
    "development": {
        "DLIVING": ("ES2004a", (82, 117, 161, 247), (181, 213, 275, 296)),
        "OOFFICE": ("TS3004a", (76, 97, 165, 217), (183, 211, 263, 295)),
    },
    "transfer": {
        "DKITCHEN": ("ES2010a", (82, 109, 171), (603, 627, 636)),
        "NFIELD": ("IS1005a", (83, 110, 172), (676, 715, 752)),
        "STRAFFIC": ("ES2011a", (84, 111, 175), (662, 717, 750)),
        "TMETRO": ("IS1004a", (85, 113, 174), (653, 720, 748)),
    },
}
DELAY = (8, -11, 27, -21)
ROLES = {"recorded", "ordinary_panned", "moving_voice", "different_background",
         "repeat_10db", "repeat_18db", "repeated_foreground", "same_channel_twice"}


def digest_bytes(data):
    return hashlib.sha256(data).hexdigest()


def shared_component_frames(spectra, bins):
    import numpy as np
    from scipy.ndimage import gaussian_filter1d

    # Use active foreground frames for the fit. The same constant filter then
    # applies at every position, leaving candidate timing on the original clock.
    energy = np.sum(np.abs(spectra[:, bins, :]) ** 2, axis=(0, 1))
    top = energy >= np.quantile(energy, .65)
    if top.sum() < 16:
        return None
    left, right = spectra[:, :, top]
    cross = np.mean(right * left.conjugate(), axis=1)
    left_power = np.mean(np.abs(left) ** 2, axis=1)
    right_power = np.mean(np.abs(right) ** 2, axis=1)
    def smooth(values):
        return gaussian_filter1d(values, .85, mode="nearest")
    cross = smooth(cross.real) + 1j * smooth(cross.imag)
    left_power, right_power = smooth(left_power), smooth(right_power)
    denominator = left_power * right_power
    known = (np.abs(cross) ** 2 >= .94 * denominator) & (left_power > np.max(left_power) / 3000)
    return cross, left_power, known & bins


def cancelled(stereo, fit):
    import numpy as np
    from scipy.signal import istft, stft

    from app.evidence import audio_native_alignment
    from app.evidence.audio_channel_alignment_binding import GAIN_DENOMINATOR, overlap

    first, end = overlap(len(stereo), fit.right_offset_samples)
    start_16 = (first + 2) // 3
    filtered = audio_native_alignment.narrowband(stereo)
    offset = fit.right_offset_samples
    left = filtered[start_16 * 3:end:3, 0]
    right = filtered[start_16 * 3 + offset:end + offset:3, 1]
    hz, _, spectra = stft(np.column_stack((left, right)).T, fs=16_000, nperseg=SCAN,
                          noverlap=SCAN - HOP, boundary="zeros", padded=True, axis=-1)
    bands = (hz >= SUPPORT[0]) & (hz <= SUPPORT[1])
    response = shared_component_frames(spectra, bands)
    if response is None:
        return None, None
    cross, energy, known = response
    gain = fit.right_gain_numerator / GAIN_DENOMINATOR
    residual = np.zeros_like(cross)
    residual[known] = cross[known] / (energy[known] * gain) - 1
    residual *= np.minimum(1, MAX_DEVIATION / np.maximum(np.abs(residual), MAX_DEVIATION))
    fitted = gain * (1 + residual)
    residual_spec = (fitted[:, None] * spectra[0] - spectra[1]) / (1 + np.abs(fitted[:, None]))
    _, restored = istft(residual_spec, fs=16_000, nperseg=SCAN, noverlap=SCAN - HOP, boundary=True)
    values = np.zeros((len(stereo) + 2) // 3, dtype=float)
    values[start_16:start_16 + len(left)] = restored[:len(left)]
    return values, {"supported_bins": int(known.sum()), "mean_deviation": float(np.mean(np.abs(residual[known]))) if known.any() else 0}


def search(stereo, fit):
    from app.evidence import audio_aac_alignment
    from app.evidence.audio_channel_alignment_binding import overlap

    if fit is None:
        return "insufficient_alignment", (), None
    values, stats = cancelled(stereo, fit)
    if values is None:
        return "insufficient_shared_energy", (), None
    reason, spans = audio_aac_alignment._locate(values, *overlap(len(stereo), fit.right_offset_samples))
    return reason, spans, stats


def useful(pairs, truth):
    if truth is None:
        return False
    def iou(first, second):
        lo, hi = max(first[0], second[0]), min(first[1], second[1])
        return max(0, hi - lo) / (max(first[1], second[1]) - min(first[0], second[0]))
    return any(min(iou(span, label) for span, label in zip(pair, truth, strict=True)) >= .5 for pair in pairs)


def summarize(rows):
    groups = {}
    for row in rows:
        roles = groups.setdefault(row["role"], Counter())
        roles["n"] += 1
        prior, extra = row["baseline"]["spans"], row["frequency"]["spans"]
        combined = prior or extra
        up, after = useful(prior, row["truth"]), useful(combined, row["truth"])
        roles.update(baseline=up, combined=after, new=after and not up, raw_candidate=useful(extra, row["truth"]),
                     ordinary_prior=row["truth"] is None and bool(prior),
                     ordinary_added=row["truth"] is None and not prior and bool(extra),
                     misplaced_added=row["truth"] is not None and not prior and bool(extra) and not after)
    return groups


def inputs(split, native_roots, aac_roots, voices):
    import numpy as np
    from scipy.signal import resample_poly

    from scripts import audio_stereo_local_lag as local
    from scripts import audio_stereo_opus_repetition as opus
    from scripts.audio_native_alignment_panel import deliveries, digest, read, segment

    samples = opus.SAMPLES
    alias, expected = local.ROSTERS["calibration" if split == "development" else "held"]
    roster = local.PANELS / alias
    if digest(roster) != expected:
        raise ValueError("Publisher source commitment changed")
    ledger = {row["scene"]: row["members"] for row in read(roster)}
    root = native_roots if split == "development" else aac_roots
    for scene, (meeting, starts, speech) in SCENES[split].items():
        room_paths = [root / scene / Path(record["member"]).name for record in ledger[scene]]
        if [digest(path) for path in room_paths] != [record["sha256"] for record in ledger[scene]]:
            raise ValueError("Publisher scene audio commitment changed")
        voice_root = aac_roots if split == "development" else voices
        voice_path = voice_root / f"{meeting}.Array1-05.wav"
        voice_sha = digest(voice_path)
        for i, (start, voice_start) in enumerate(zip(starts, speech, strict=True)):
            room = np.column_stack([segment(path, start, samples, 48000) for path in room_paths])
            unrelated = np.column_stack([segment(path, start + 7, 30720, 48000) for path in room_paths])
            source = segment(voice_path, voice_start, samples // 3, 16000)
            voice = resample_poly(source.astype(float), 3, 1)
            variants = list(deliveries(room, unrelated, voice, i, offset=DELAY[i]))
            first, second = variants[-1][2]
            repeated = voice.copy()
            repeated[slice(*second)] = voice[slice(*first)]
            repeated_mix = next(pcm for label, pcm, _ in deliveries(room, unrelated, repeated, i, offset=DELAY[i]) if label == "ordinary_panned")
            variants.append(("repeated_foreground", repeated_mix, None))
            variants.append(("same_channel_twice", np.repeat(room[:, :1], 2, axis=1), None))
            for label, pcm, truth in variants:
                source = {"scene":scene, "meeting":meeting, "source_pcm_hashes":[member["sha256"] for member in ledger[scene]],
                          "voice_file_sha256": voice_sha, "start_seconds":start, "voice_start_seconds":voice_start,
                          "role":label, "truth":[list(pair) for pair in truth] if truth else None, "parent_sha256":digest_bytes(pcm.tobytes())}
                yield source, pcm


def read(path):
    return json.loads(gzip.decompress(path.read_bytes()))


def serialized(document):
    return (json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def save(path, document):
    with path.open("xb") as file:
        file.write(gzip.compress(serialized(document), mtime=0))


def freeze(native, aac, voices, output):
    save(output, {"schema": SCHEMA, "cases": {
        split: [meta for meta, _ in inputs(split, native, aac, voices)] for split in SCENES},
        "implementation_sha256": digest_bytes(Path(__file__).read_bytes())})


def measure(native, aac, voices, plan, output):
    import numpy as np

    from app.detectors.audio.ogg_opus_reference import native_clock
    from app.evidence import audio_aac_alignment
    from scripts import audio_stereo_opus_repetition as opus
    from scripts.audio_native_alignment_panel import save_wav

    lock = read(plan)
    if lock["schema"] != SCHEMA or lock["implementation_sha256"] != digest_bytes(Path(__file__).read_bytes()):
        raise ValueError("The evaluation runner differs from the frozen input plan")
    readings = {}
    with TemporaryDirectory(prefix="chipmunk-complex-stereo.") as scratch:
        wav, ogg = (Path(scratch) / name for name in ("case.wav", "case.ogg"))
        for split in SCENES:
            rows = []
            for (meta, pcm), frozen in zip(inputs(split, native, aac, voices), lock["cases"][split], strict=True):
                if meta != frozen:
                    raise ValueError("Source metadata or samples differ from the frozen case")
                save_wav(wav, pcm)
                opus.encode(wav, ogg)
                clock = native_clock(ogg, channels=2, packet_bytes=opus.PACKET_BYTES, presented=opus.SAMPLES)
                if clock != opus.CLOCK:
                    raise ValueError("The stereo Opus presentation clock differs from the comparison profile")
                samples = opus.native_decoded(ogg)
                stereo = np.frombuffer(samples, dtype="<i2").reshape(-1, 2)
                fitted, reason, baseline = audio_aac_alignment.search(stereo)
                decision, spans, details = search(stereo, fitted)
                rows.append({"opus_sha256": digest_bytes(ogg.read_bytes()), "decoded_sha256": digest_bytes(samples),
                             "clock": clock, "baseline": {"reason": reason, "spans": baseline},
                             "frequency": {"reason": decision, "spans": spans, "details": details}})
                if len(rows) % 8 == 0:
                    print(split, len(rows), flush=True)
            readings[split] = {"rows": rows, "summary": summarize(
                [row | meta for row, meta in zip(rows, lock["cases"][split], strict=True)])}
    save(output, {"schema": SCHEMA, "plan": lock, "frozen_sha256": digest_bytes(serialized(lock)), "readings": readings})


def review(path):
    from scripts.audio_stereo_opus_repetition import CLOCK

    record = read(path)
    plan = record["plan"]
    if (record["schema"] != SCHEMA or plan["schema"] != SCHEMA or
            plan["implementation_sha256"] != digest_bytes(Path(__file__).read_bytes()) or
            record["frozen_sha256"] != digest_bytes(serialized(plan))):
        raise ValueError("The receipt does not bind this evaluation runner")
    expected_counts = {"development": 64, "transfer": 96}
    summaries = {}
    all_inputs = set()
    for split, count in expected_counts.items():
        meta, readout = plan["cases"][split], record["readings"][split]
        rows = readout["rows"]
        if len(rows) != count or len(meta) != count:
            raise ValueError("A split is missing planned cases")
        expected_cases = {(name, start, foreground, role)
                          for name, (_, starts, speech) in SCENES[split].items()
                          for start, foreground in zip(starts, speech, strict=True) for role in ROLES}
        if {(case["scene"], case["start_seconds"], case["voice_start_seconds"], case["role"])
                for case in meta} != expected_cases:
            raise ValueError("The receipt differs from the fixed comparison roster")
        for case, row in zip(meta, rows, strict=True):
            speaker, starts, speech = SCENES[split][case["scene"]]
            identity = row["opus_sha256"]
            if (case["meeting"] != speaker or (case["start_seconds"], case["voice_start_seconds"])
                    not in zip(starts, speech, strict=True) or row["clock"] != CLOCK or identity in all_inputs):
                raise ValueError("The receipt has a changed source assignment, clock, or repeated encoded file")
            all_inputs.add(identity)
        summaries[split] = summarize([row | case for row, case in zip(rows, meta, strict=True)])
        if summaries[split] != readout["summary"]:
            raise ValueError("The saved localization counts do not replay")
    print(json.dumps(summaries, indent=2, sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name, labels in (("freeze", ("native", "aac", "voices", "output")),
                         ("measure", ("native", "aac", "voices", "plan", "output")),
                         ("review", ("path",))):
        cli = commands.add_parser(name)
        for label in labels:
            cli.add_argument(label, type=Path)
    args = vars(parser.parse_args())
    {"freeze": freeze, "measure": measure, "review": review}[args.pop("command")](**args)


if __name__ == "__main__":
    main()
