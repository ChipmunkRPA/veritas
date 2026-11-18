# SPDX-License-Identifier: Apache-2.0
"""Compare sparse time-local lag proposals with a matched stereo AAC baseline."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import subprocess
import wave
from collections import Counter, defaultdict
from itertools import chain
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

from scripts import audio_stereo_local_lag as prior
from scripts import audio_stereo_opus_repetition as opus
from scripts.audio_native_alignment_panel import deliveries, digest, read, save, save_wav, segment

PLAN = prior.PANELS / "audio-stereo-sparse-lag-plan-2026-11-18.json"
PLAN_SHA256 = "002c53fe7f25c7c3e255afa7145b9b7411c946259f37daef3a45a26b31cc7a8f"
ARCHIVE_SHA1 = "0f8088da2f45bd699bb6e39870dc8c6b624ddb96"
VOICES = {
    "ES2010a": "99ca68da1f4af6d6316ada55978f43c50ddb0569df6fe973559bee074b75f9d7",
    "IS1005a": "2cc5d5715fc75d00f5f70d5693e004485e0693f64c27851877dd58d9b5d46444",
    "ES2011a": "409bd94990397de8706f8f209d01935b532c57323010f178928f2c262b1be455",
    "IS1004a": "3f6d6e61e896c91c5f1ad90726ccc1f5283fa764156394f699750c3a18fc914a",
}
POSITIVE = opus.POSITIVES
NATIVE_ROLES = (*prior.ROLES, "motor_sweep")
TRANSIT_ROLES = ("transit_recorded", "transit_panned")
STRESS = ("motor_sweep", *TRANSIT_ROLES)
TRANSIT_SOURCES = tuple(f"{kind}{index:02d}" for kind in ("bus", "tube", "tubestation") for index in range(1, 11))
ARMS = ("aac", "baseline_opus", "sparse_opus")
MAX_FINE_LAGS = 12


def sparse_lags(bands):
    import numpy as np
    from scipy.fft import irfft, next_fast_len, rfft

    values = np.array([band[::2] for band in bands])
    full = np.asarray(bands)
    width, step, distance = prior.WIDTH, prior.STEP, prior.MIN_DISTANCE
    valid = values.shape[1] - width + 1
    fft_size = next_fast_len(values.shape[1] + width - 1, real=True)
    spectrum = rfft(values, n=fft_size, axis=1)
    totals = np.column_stack((np.zeros(len(values)), np.cumsum(values, axis=1)))
    sums = totals[:, width:] - totals[:, :-width]
    powers = np.column_stack((np.zeros(len(values)), np.cumsum(values**2, axis=1)))
    energy = np.maximum(powers[:, width:] - powers[:, :-width] - sums**2 / width, 0)
    proposals = defaultdict(list)
    for start in range(0, valid - distance, step):
        right = start + distance
        template = values[:, start:start + width]
        centered = template - template.mean(axis=1, keepdims=True)
        products = irfft(spectrum * rfft(centered, n=fft_size, axis=1).conjugate(),
                         n=fft_size, axis=1)[:, right:valid]
        window_energy = energy[:, right:valid]
        anchor_energy = energy[:, start:start + 1]
        active = np.minimum(anchor_energy, window_energy) >= width * 4
        scores = np.divide(products, np.sqrt(anchor_energy * window_energy),
                           out=np.zeros_like(products), where=active)
        combined = np.partition(scores, -2, axis=0)[-2].copy()
        for _ in range(2):
            peak = int(np.argmax(combined))
            if combined[peak] < prior.LOCAL_POINT:
                break
            proposals[peak + distance].append((start, float(combined[peak])))
            combined[max(0, peak - 127):peak + 128] = -1

    def ranked_lag(lag):
        return -len(proposals[lag]), -max(score for _, score in proposals[lag]), lag

    def agrees(first, lag):
        second = first + lag
        if second + 2 * width > full.shape[1]:
            return False
        left, right = (full[:, a:a + 2 * width] for a in (first, second))
        left = left - left.mean(axis=1, keepdims=True)
        right = right - right.mean(axis=1, keepdims=True)
        left_energy, right_energy = (np.einsum("ij,ij->i", side, side) for side in (left, right))
        strong = np.minimum(left_energy, right_energy) >= 2 * width * 4
        scores = np.einsum("ij,ij->i", left, right)
        return np.count_nonzero(strong & (scores >= prior.LOCAL_POINT * np.sqrt(left_energy * right_energy))) >= 2

    fine = {
        lag
        for coarse in sorted(proposals, key=ranked_lag)[:12]
        for lag in range(2 * coarse - 2, 2 * coarse + 3)
        if lag >= 2 * distance and any(agrees(2 * start, lag) for start, _ in proposals[coarse])
    }
    return sorted(fine)


def native_cases(plan, root):
    import numpy as np
    from scipy.signal import resample_poly

    from scripts.audio_channel_difference_panel import rms

    rosters = {}
    for name, split in (("native", "calibration"), ("aac", "held")):
        filename, sha = prior.ROSTERS[split]
        path = prior.PANELS / filename
        if digest(path) != sha:
            raise ValueError("The earlier publisher channel roster changed")
        rosters[name] = {row["scene"]: row["members"] for row in read(path)}
    for scene, room in plan["rooms"].items():
        roster, meeting = room["roster"], room["meeting"]
        records = rosters[roster][scene]
        paths = [root / roster / row["member"] for row in records]
        voice_file = root / f"{meeting}.Array1-05.wav"
        if [digest(path) for path in paths] != [row["sha256"] for row in records]:
            raise ValueError(f"A recorded channel differs from the publisher checksum: {scene}")
        if digest(voice_file) != VOICES[meeting]:
            raise ValueError(f"The foreground track changed: {meeting}")

        for index, ((start, replacement_start), voice_start) in enumerate(
            zip(room["excerpts"], plan["speech_starts_seconds"], strict=True)
        ):
            background = np.column_stack([segment(path, start, opus.SAMPLES, 48_000) for path in paths])
            replacement = np.column_stack([segment(path, replacement_start, 30720, 48_000) for path in paths])
            voice = resample_poly(segment(voice_file, voice_start, opus.SAMPLES // 3, 16_000).astype(float), 3, 1)
            offset = prior.SHIFTS[index]
            variants = list(deliveries(background, replacement, voice, index, offset=offset))
            first, second = variants[-1][2]
            repeated_voice = voice.copy()
            repeated_voice[slice(*second)] = voice[slice(*first)]
            repeated = next(pcm for role, pcm, _ in deliveries(background, replacement, repeated_voice, index,
                                                               offset=offset) if role == "ordinary_panned")
            time = np.arange(opus.SAMPLES) / 48_000
            engine = np.sin(2 * np.pi * ((205 + index * 6) * time + 1.21 * time**2))
            engine += .7 * np.sin(2 * np.pi * ((574 + index * 8) * time - 2.3 * time**2))
            engine *= 1 + .4 * np.sin(2 * np.pi * (.83 * time + .065 * time**2))
            engine *= .55 * rms(background) / rms(engine)
            motor = np.column_stack((engine, -.62 * np.roll(engine, 61 + index * 4)))
            overlaid = np.rint(background + motor).clip(-32768, 32767).astype("<i2")
            sweep = next(pcm for role, pcm, _ in deliveries(overlaid, replacement, voice, index, offset=offset)
                         if role == "ordinary_panned")
            variants.extend((("repeated_foreground", repeated, None), ("motor_sweep", sweep, None),
                             ("same_channel_twice", np.repeat(background[:, :1], 2, axis=1), None)))
            for role, pcm, truth in variants:
                yield {"group": f"{scene}-{start}", "environment": scene, "role": role,
                       "background_start_seconds": start, "background_sha256": [row["sha256"] for row in records],
                       "replacement_start_seconds": replacement_start, "meeting": meeting,
                       "speech_start_seconds": voice_start, "authored_offset_samples": offset,
                       "truth": truth}, pcm


def transit_cases(root):
    import numpy as np
    from scipy.signal import resample_poly

    archive = root / "scenes_stereo.zip"
    with archive.open("rb") as source:
        if hashlib.file_digest(source, "sha1").hexdigest() != ARCHIVE_SHA1:
            raise ValueError("The DCASE archive differs from its publisher checksum")
    meeting = "ES2011a"
    speech_path = root / f"{meeting}.Array1-05.wav"
    if digest(speech_path) != VOICES[meeting]:
        raise ValueError("The transit overlay speech track changed")
    with ZipFile(archive) as source:
        for index, identifier in enumerate(TRANSIT_SOURCES):
            member = f"scenes_stereo/{identifier}.wav"
            raw = source.read(member)
            source_sha = hashlib.sha256(raw).hexdigest()
            for role, start in zip(TRANSIT_ROLES, (17, 24), strict=True):
                with wave.open(io.BytesIO(raw), "rb") as wav:
                    if wav.getparams()[:4] != (2, 2, 44_100, 1_323_000):
                        raise ValueError("Unexpected DCASE source PCM format")
                    wav.setpos(start * 44_100)
                    block = np.frombuffer(wav.readframes(165_375), dtype="<i2").reshape(-1, 2)
                pcm = np.rint(resample_poly(block.astype(float), 160, 147, axis=0)).clip(-32768, 32767).astype("<i2")
                meta = {"group": identifier, "environment": identifier.rstrip("0123456789"), "role": role,
                        "background_start_seconds": start, "background_sha256": [source_sha], "member": member,
                        "truth": None}
                if role == "transit_panned":
                    speech_start = 770 + index * 6
                    voice = resample_poly(segment(speech_path, speech_start, opus.SAMPLES // 3, 16_000).astype(float),
                                          3, 1)
                    variant = index % len(prior.SHIFTS)
                    offset = prior.SHIFTS[variant]
                    pcm = next(delivery for label, delivery, _ in deliveries(pcm, pcm[:30720], voice, variant,
                                                                              offset=offset)
                               if label == "ordinary_panned")
                    meta.update(meeting=meeting, speech_start_seconds=speech_start, authored_offset_samples=offset)
                yield meta, pcm


def assess(samples, *, fallback=False):
    from unittest.mock import patch

    from app.evidence import audio_aac_alignment as repeat

    fit, reason, spans = repeat.search(samples)
    baseline = {"status": reason, "spans": spans, "alignment": fit.right_offset_samples if fit else None}
    if not fallback or reason is not None or spans:
        return baseline, None
    lag_counts = []

    def candidates(bands):
        lags = sparse_lags(bands)
        lag_counts.append(len(lags))
        return lags if len(lags) <= MAX_FINE_LAGS else []

    with patch.object(repeat, "_nominate_lags", candidates):
        _, status, revised = repeat.search(samples)
    if len(lag_counts) != 1:
        raise ValueError("The candidate did not verify the same alignment and frequency support")
    return baseline, {"status": status, "spans": revised, "surviving_lags": lag_counts[0],
                      "tested_lags": lag_counts[0] if lag_counts[0] <= MAX_FINE_LAGS else 0}


def measure(plan, root, scratch):
    import numpy as np

    from app.detectors.audio.decode import decode_audio, decode_native_stereo
    from app.detectors.audio.ogg_opus_reference import native_clock
    from app.evidence import audio_aac_alignment as repeat

    if digest(Path(repeat.__file__)) != opus.DETECTOR_SHA:
        raise ValueError("The unchanged baseline interval matcher differs from the frozen digest")
    ffmpeg = subprocess.check_output(["ffmpeg", "-version"], text=True).splitlines()[0].split()[:3]
    if ffmpeg != ["ffmpeg", "version", "7.0.2"]:
        raise ValueError(f"Unexpected source codec runtime: {ffmpeg}")
    rows = []
    wav, ogg, aac = (scratch / name for name in ("parent.wav", "encoded.opus", "encoded.m4a"))
    for meta, samples in chain(native_cases(plan, root), transit_cases(root)):
        if samples.shape != (opus.SAMPLES, 2):
            raise ValueError("The WAV parent needs exactly 180000 stereo samples")
        save_wav(wav, samples)
        opus.encode(wav, ogg)
        clock = native_clock(ogg, packet_bytes=opus.PACKET_BYTES, channels=2, presented=opus.SAMPLES)
        if clock != opus.CLOCK:
            raise ValueError("Opus packet clock disagrees with the frozen stereo profile")
        ogg_pcm = opus.native_decoded(ogg)
        baseline, sparse = assess(np.frombuffer(ogg_pcm, dtype="<i2").reshape(-1, 2), fallback=True)
        command = ["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(wav), "-threads", "1", "-c:a", "aac",
                   "-b:a", "160k", "-map_metadata", "-1", "-fflags", "+bitexact", str(aac)]
        subprocess.run(command, check=True, capture_output=True, timeout=15)
        aac_sha = digest(aac)
        decoded = decode_audio(aac, source_sha256=aac_sha)
        aac_pcm = decode_native_stereo(aac, decoded.mapping, aac=True)
        if aac_pcm is None or len(aac_pcm) != opus.SAMPLES * 4 or decoded.mapping["source_start_pts"] != 0:
            raise ValueError("The matched AAC clock differs from the exact stereo source")
        paired, _ = assess(np.frombuffer(aac_pcm, dtype="<i2").reshape(-1, 2))
        rows.append(meta | {"parent_sha256": digest(wav), "opus_sha256": digest(ogg), "opus_clock": clock,
                            "opus_pcm_sha256": hashlib.sha256(ogg_pcm).hexdigest(), "aac_sha256": aac_sha,
                            "aac_pcm_sha256": hashlib.sha256(aac_pcm).hexdigest(), "aac_start_pts": 0,
                            "aac_presented_samples": len(aac_pcm) // 4, "aac": paired, "baseline_opus": baseline,
                            "sparse_fallback": sparse})
        if len(rows) % 24 == 0:
            print(f"Measured {len(rows)} paired stereo cases", flush=True)
    return rows


def recount(plan, rows):
    expected = {
        (f"{scene}-{start}", role)
        for scene, room in plan["rooms"].items()
        for start, _ in room["excerpts"]
        for role in NATIVE_ROLES
    } | {(source, role) for source in TRANSIT_SOURCES for role in TRANSIT_ROLES}
    totals = defaultdict(Counter)
    gained, positive_environments, positive_levels = set(), set(), set()
    changes = Counter()
    for row in rows:
        role, group = row["role"], row["group"]
        expected.remove((group, role))
        if (bool(row["truth"]) != (role in POSITIVE) or row["opus_clock"] != opus.CLOCK
                or row["aac_start_pts"] != 0 or row["aac_presented_samples"] != opus.SAMPLES):
            raise ValueError("The native sample clock or labels changed")
        aac, baseline, fallback = row["aac"], row["baseline_opus"], row["sparse_fallback"]
        if (fallback is not None) != (baseline["status"] is None and not baseline["spans"]):
            raise ValueError("The sparse fallback replaced an existing result")
        if fallback:
            surviving, tested = fallback["surviving_lags"], fallback["tested_lags"]
            if not 0 <= surviving <= 60 or tested != (surviving if surviving <= MAX_FINE_LAGS else 0):
                raise ValueError("The predictor exceeded the sparse fine-lag budget")
            if not tested and fallback["spans"]:
                raise ValueError("An abstained fallback claimed a pair")
            changes.update(fallbacks=1, dense_abstentions=surviving > MAX_FINE_LAGS, empty=surviving == 0)
        selected = fallback if fallback and fallback["spans"] else baseline
        for arm, result in zip(ARMS, (aac, baseline, selected), strict=True):
            for (a, b), (c, d) in result["spans"]:
                if not 0 <= a < b <= c < d <= opus.SAMPLES or result["status"] is not None:
                    raise ValueError("A result is outside its assessed source sample clock")
            correct = opus.useful(row, result)
            totals[f"{arm}:{role}"].update(files=1, assessed=result["status"] is None, correct=correct,
                                            ordinary_flag=not row["truth"] and bool(result["spans"]),
                                            misplaced=bool(row["truth"] and result["spans"] and not correct))
        aac_ok, baseline_ok, selected_ok = (opus.useful(row, arm) for arm in (aac, baseline, selected))
        if baseline_ok and not selected_ok:
            raise ValueError("The sparse fallback lost a useful baseline match")
        if selected_ok:
            positive_environments.add(row["environment"])
            positive_levels.add(role)
        if selected_ok and not baseline_ok:
            gained.add(group)
        new = bool(selected["spans"] and not baseline["spans"])
        changes.update(gained=selected_ok and not baseline_ok, aac_correct=aac_ok, baseline_correct=baseline_ok,
                       combined_correct=selected_ok, aac_retained=aac_ok and selected_ok,
                       new_wrong=bool(new and row["truth"] and not selected_ok),
                       new_ordinary=bool(new and not row["truth"]), new_stress=bool(new and role in STRESS))
    if expected:
        raise ValueError("The comparison is missing expected case roles")
    gates = {"gained_three": changes["gained"] >= 3, "gained_two_groups": len(gained) >= 2,
             "ordinary_budget": changes["new_ordinary"] <= 2, "misplaced_budget": changes["new_wrong"] <= 2,
             "zero_new_stress": changes["new_stress"] == 0,
             "useful_sixteen": changes["combined_correct"] >= 16,
             "useful_both_levels": len(positive_levels) == 2, "useful_two_environments": len(positive_environments) >= 2,
             "aac_retention": bool(changes["aac_correct"]) and 100 * changes["aac_retained"] >= 85 * changes["aac_correct"]}
    return {"by_role": totals, "changes": changes, "gates": gates, "qualifies": all(gates.values()),
            "groups_gained": sorted(gained), "useful_environments": sorted(positive_environments)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("measure", "review"):
        action = commands.add_parser(command)
        action.add_argument("receipt", type=Path)
        if command == "measure":
            action.add_argument("sources", type=Path)
    args = parser.parse_args()
    if digest(PLAN) != PLAN_SHA256:
        raise ValueError("The prospective plan changed")
    plan = read(PLAN)
    expected = {"schema": "chipmunk.stereo-sparse-lag/1",
                "plan_sha256": PLAN_SHA256, "runner_sha256": digest(Path(__file__)),
                "baseline_sha256": opus.DETECTOR_SHA, "dcase_sha1": ARCHIVE_SHA1, "voice_sha256": VOICES}
    if args.command == "measure":
        if args.receipt.exists():
            raise ValueError("Use a new receipt filename")
        with TemporaryDirectory(prefix="chipmunk-sparse-lag-", dir=args.receipt.parent) as scratch:
            rows = measure(plan, args.sources, Path(scratch))
        result = recount(plan, rows)
        save(args.receipt, expected | {"rows": rows, "summary": result})
    else:
        receipt = read(args.receipt)
        result = recount(plan, receipt["rows"])
        if any(receipt.get(key) != value for key, value in expected.items()) or receipt["summary"] != result:
            raise ValueError("The receipt does not replay against the frozen plan and runner")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
