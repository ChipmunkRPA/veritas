# SPDX-License-Identifier: Apache-2.0
"""Test time-local cross-band lag nominations for stereo repeats under speech."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts import audio_stereo_opus_repetition as parent
from scripts.audio_native_alignment_panel import deliveries, digest, read, save, save_wav, segment

PANELS = Path(__file__).resolve().parents[2] / "docs" / "evaluation-panels"
PLAN = PANELS / "audio-stereo-local-lag-plan-2026-11-18.json"
PLAN_SHA = "6429060e69e250cff4069620ac212d6e5c11bd82e108efbb2a0e41ba95fdeaa5"
SCHEMA = "chipmunk.stereo-local-lag/1"
ROSTERS = {
    "calibration": ("audio-native-alignment-sources-2026-11-17.json", "c7f227b6da9484c30eaad03df3d70b747cba17c83dcb78741f79f98c84e20690"),
    "held": ("audio-aac-repetition-sources-2026-11-17.json", "0b53be6061abd49c9505f7a3ab714e4c5fb3073c49af62342ffdcb72d28245f5"),
}
SCENES = {"calibration": ("DLIVING", "OOFFICE"), "held": ("DKITCHEN", "NFIELD", "STRAFFIC", "TMETRO")}
STARTS = {"calibration": (11, 54, 141, 195), "held": (39, 69, 126, 158, 233, 288)}
VOICE_STARTS = {"calibration": (101, 134, 186, 251), "held": (62, 101, 139, 173, 214, 251)}
VOICES = {
    "calibration": {"ES2004a": "c6f0eafaedd3632da4cb3f14788ee83a8efae2b456f70eefa8e98b2b6992102a",
                    "TS3004a": "3f12bd8637e86b937e7962d2df246c6ffe49d996a8a19412737ce02a8d7ce8b0"},
    "held": {"ES2011b": "5cffddcbd80dbe2265862cf20aec2d38ec85548502ed3c725600d56b115c0c0f",
             "IS1006b": "0d2c35a24d4989384a62c9b270027362e016e64f18811cc4b178e7bbd7717588"},
}
SHIFTS = (8, -11, 27, -21, 19, -32)
POSITIVE = parent.POSITIVES
ROLES = ("recorded", "ordinary_panned", "different_background", "moving_voice", *POSITIVE,
         "repeated_foreground", "same_channel_twice")
WIDTH = 960
STEP = 640
MIN_DISTANCE = 1200
LOCAL_POINT = .62


def local_lags(bands):
    """Nominate lags from short time positions jointly supported by two bands."""
    import numpy as np
    from scipy.fft import irfft, next_fast_len, rfft

    values = np.array([band[::2] for band in bands])
    length = values.shape[1]
    valid = length - WIDTH + 1
    fft_size = next_fast_len(length + WIDTH - 1, real=True)
    spectrum = rfft(values, n=fft_size, axis=1)
    totals = np.column_stack((np.zeros(len(values)), np.cumsum(values, axis=1)))
    sums = totals[:, WIDTH:] - totals[:, :-WIDTH]
    powers = np.column_stack((np.zeros(len(values)), np.cumsum(values**2, axis=1)))
    energy = np.maximum(powers[:, WIDTH:] - powers[:, :-WIDTH] - sums**2 / WIDTH, 0)
    nominations = {}
    for start in range(0, valid - MIN_DISTANCE, STEP):
        right = start + MIN_DISTANCE
        template = values[:, start:start + WIDTH]
        centered = template - template.mean(axis=1, keepdims=True)
        products = irfft(spectrum * rfft(centered, n=fft_size, axis=1).conjugate(), n=fft_size, axis=1)[:, right:valid]
        window_energy = energy[:, right:valid]
        anchor_energy = energy[:, start:start + 1]
        active = np.minimum(anchor_energy, window_energy) >= WIDTH * 4
        scores = np.divide(products, np.sqrt(anchor_energy * window_energy), out=np.zeros_like(products), where=active)
        combined = np.partition(scores, -2, axis=0)[-2].copy()
        for _ in range(2):
            peak = int(np.argmax(combined))
            if combined[peak] < LOCAL_POINT:
                break
            lag = peak + MIN_DISTANCE
            votes, strongest = nominations.get(lag, (0, 0))
            nominations[lag] = votes + 1, max(float(combined[peak]), strongest)
            combined[max(0, peak - 127):peak + 128] = -1
    top = sorted(nominations, key=lambda lag: (-nominations[lag][0], -nominations[lag][1], lag))[:12]
    return sorted({2 * lag + side for lag in top for side in range(-2, 3) if 2 * lag + side >= 2400})


def case_inputs(split, roots):
    import numpy as np
    from scipy.signal import resample_poly

    backgrounds, voices = roots
    name, sha = ROSTERS[split]
    roster = PANELS / name
    if digest(roster) != sha:
        raise ValueError("A frozen publisher source inventory changed")
    files = {row["scene"]: row for row in read(roster)}
    for meeting, expected in VOICES[split].items():
        if digest(voices / f"{meeting}.Array1-05.wav") != expected:
            raise ValueError("The whole AMI recording differs from the earlier verified source copy")
    for position, scene in enumerate(SCENES[split]):
        room = [backgrounds / scene / Path(record["member"]).name for record in files[scene]["members"]]
        if [digest(path) for path in room] != [record["sha256"] for record in files[scene]["members"]]:
            raise ValueError("The two DEMAND channels differ from the exact source commitment")
        meeting = tuple(VOICES[split])[position % 2]
        for index, start in enumerate(STARTS[split]):
            recordings = np.column_stack([segment(path, start, parent.SAMPLES, 48000) for path in room])
            replacement = np.column_stack([segment(path, start + 7, 30720, 48000) for path in room])
            speech_start = VOICE_STARTS[split][index] + 6 * (position // 2)
            source = segment(voices / f"{meeting}.Array1-05.wav", speech_start, parent.SAMPLES // 3, 16000)
            voice = resample_poly(source.astype(float), 3, 1)
            groups = list(deliveries(recordings, replacement, voice, index, offset=SHIFTS[index]))
            first, second = groups[-1][2]
            repeated = voice.copy()
            repeated[slice(*second)] = voice[slice(*first)]
            panned_repeat = next(pcm for label, pcm, _ in deliveries(recordings, replacement, repeated, index, offset=SHIFTS[index])
                                if label == "ordinary_panned")
            all_roles = (*groups, ("repeated_foreground", panned_repeat, None),
                         ("same_channel_twice", np.repeat(recordings[:, :1], 2, axis=1), None))
            for role, pcm, truth in all_roles:
                yield {"scene": scene, "start_seconds": start, "meeting": meeting, "speech_start_seconds": speech_start,
                       "offset_samples": SHIFTS[index], "role": role, "truth": truth}, pcm


def reading(samples, *, nominate=False):
    from unittest.mock import patch

    from app.evidence import audio_aac_alignment as repeat

    alignment, reason, spans = repeat.search(samples)
    baseline = {"status": reason, "spans": spans,
                "alignment": alignment.right_offset_samples if alignment else None}
    if not nominate or reason is not None or spans:
        return baseline, None
    new_lags = []

    def remember(bands):
        new_lags.extend(local_lags(bands))
        return new_lags

    with patch.object(repeat, "_nominate_lags", remember):
        _, status, revised = repeat.search(samples)
    supplementary = {"status": status, "spans": revised, "lags_tested": len(new_lags)}
    return baseline, supplementary


def measure(split, roots, scratch):
    import numpy as np

    from app.detectors.audio.decode import decode_audio, decode_native_stereo
    from app.detectors.audio.ogg_opus_reference import native_clock
    from app.evidence import audio_aac_alignment as repeat

    if digest(Path(repeat.__file__)) != parent.DETECTOR_SHA:
        raise ValueError("The interval verification must remain frozen")
    rows = []
    wav, opus, aac = (scratch / name for name in ("parent.wav", "encoded.opus", "encoded.m4a"))
    for meta, samples in case_inputs(split, roots):
        save_wav(wav, samples)
        parent.encode(wav, opus)
        clock = native_clock(opus, packet_bytes=parent.PACKET_BYTES, channels=2, presented=parent.SAMPLES)
        if clock != parent.CLOCK:
            raise ValueError("Opus packet clock disagrees with the fixed stereo profile")
        raw = parent.native_decoded(opus)
        baseline, local = reading(np.frombuffer(raw, dtype="<i2").reshape(-1, 2), nominate=True)
        row = meta | {"parent_sha256": digest(wav), "opus_sha256": digest(opus), "clock": clock,
                      "opus_pcm_sha256": hashlib.sha256(raw).hexdigest(), "original": baseline, "local": local}
        if split == "held":
            command = ["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(wav), "-threads", "1", "-c:a", "aac",
                       "-b:a", "160k", "-map_metadata", "-1", "-fflags", "+bitexact", str(aac)]
            subprocess.run(command, check=True, capture_output=True, timeout=15)
            sha = digest(aac)
            decoded = decode_audio(aac, source_sha256=sha)
            raw = decode_native_stereo(aac, decoded.mapping, aac=True)
            if raw is None or len(raw) != parent.SAMPLES * 4 or decoded.mapping["source_start_pts"] != 0:
                raise ValueError("The matched AAC clock differs from the exact stereo source")
            paired, _ = reading(np.frombuffer(raw, dtype="<i2").reshape(-1, 2))
            row.update(aac=paired, aac_sha256=sha, aac_pcm_sha256=hashlib.sha256(raw).hexdigest())
        rows.append(row)
        if len(rows) % 24 == 0:
            print(f"Measured {len(rows)} {split} stereo cases", flush=True)
    return rows


def summary(rows, split):
    totals = {role: Counter() for role in ROLES}
    expected = {(scene, second, role) for scene in SCENES[split] for second in STARTS[split] for role in ROLES}
    source_gains, scenes, levels = set(), set(), set()
    for row in rows:
        role = row["role"]
        expected.remove((row["scene"], row["start_seconds"], role))
        if row["clock"] != parent.CLOCK or bool(row["truth"]) != (role in POSITIVE):
            raise ValueError("Source timing or labels changed")
        old, new = row["original"], row["local"]
        if (new is not None) != (old["status"] is None and not old["spans"]):
            raise ValueError("Local search must run only after an assessed result without a pair")
        paired = row.get("aac")
        for result in (old, new, paired):
            if not result:
                continue
            if result["spans"] and result["status"] is not None:
                raise ValueError("An unavailable check claims a result")
            for (a, b), (c, d) in result["spans"]:
                if not 0 <= a < b <= c < d <= parent.SAMPLES:
                    raise ValueError("A result does not lie on the original source clock")
        selected = new if new and new["spans"] else old
        baseline, combined = parent.useful(row, old), parent.useful(row, selected)
        aac = bool(paired and parent.useful(row, paired))
        totals[role].update(total=1, opus_assessed=old["status"] is None, baseline=baseline, combined=combined,
            gained=combined and not baseline, new_ordinary=role not in POSITIVE and bool(selected["spans"] and not old["spans"]),
            new_misplaced=role in POSITIVE and bool(selected["spans"] and not old["spans"]) and not combined,
            aac=aac, aac_preserved=aac and combined, prior_aac_preserved=aac and baseline)
        if combined:
            scenes.add(row["scene"])
            levels.add(role)
        if combined and not baseline:
            source_gains.add((row["scene"], row["start_seconds"]))
    if expected:
        raise ValueError("The receipt is missing a planned role or source excerpt")
    total = sum(totals.values(), Counter())
    improved = total["gained"] >= 3 and len(source_gains) >= 2 and total["new_ordinary"] <= 2 and total["new_misplaced"] <= 2
    qualified = (improved and total["combined"] >= 16 and len(scenes) >= 2 and len(levels) == 2 and total["aac"] > 0
                 and 100 * total["aac_preserved"] >= 85 * total["aac"])
    return {"per_role": totals, "total": total, "experimental_gain": improved, "qualified": qualified,
            "source_groups_gained": len(source_gains), "useful_rooms": len(scenes), "useful_levels": sorted(levels)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="mode", required=True)
    for mode in ("calibration", "held", "review"):
        command = commands.add_parser(mode)
        command.add_argument("receipt", type=Path)
        if mode != "review":
            command.add_argument("rooms", type=Path)
            command.add_argument("voices", type=Path)
        if mode != "calibration":
            command.add_argument("--calibration", type=Path)
    args = parser.parse_args()
    if digest(PLAN) != PLAN_SHA:
        raise ValueError("The frozen plan changed")
    expected = {"schema": SCHEMA, "plan_sha256": PLAN_SHA, "code_sha256": digest(Path(__file__))}
    calibration_hash = None
    if args.mode != "calibration" and args.calibration:
        control = read(args.calibration)
        if (any(control[key] != value for key, value in expected.items()) or control["split"] != "calibration"
                or summary(control["rows"], "calibration") != control["summary"]
                or control["summary"]["total"].get("new_ordinary", 0)):
            raise ValueError("The frozen development guard did not pass")
        calibration_hash = digest(args.calibration)
    if args.mode == "review":
        item = read(args.receipt)
        result = summary(item["rows"], item["split"])
        if (any(item[key] != value for key, value in expected.items()) or item["summary"] != result
                or item["calibration_sha256"] != calibration_hash):
            raise ValueError("The portable receipt differs from the locked comparison")
    else:
        if args.receipt.exists() or (args.mode == "held" and not calibration_hash):
            parser.error("Use a new receipt path and supply the calibration for held scoring")
        with TemporaryDirectory(prefix="chipmunk-local-lag-repetition-") as scratch:
            rows = measure(args.mode, (args.rooms, args.voices), Path(scratch))
        result = summary(rows, args.mode)
        save(args.receipt, expected | {"split": args.mode, "calibration_sha256": calibration_hash, "rows": rows, "summary": result})
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
