"""Test a background-only flank projection on a fresh matched voice panel."""

from __future__ import annotations

import argparse
import csv
import io
import re
import wave
from collections import defaultdict
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from scipy.fft import irfft, rfft
from scipy.io import loadmat
from scipy.signal import resample_poly

from scripts import audio_odss_matched_room as room
from scripts import audio_room_stress_transfer as stress
from scripts import audio_spatial_mask_aachen_transfer as aachen

SCHEMA = "chipmunk.audio-flank-background/1"
AIR_SHA = "d2fd52767505c402e8aed9299dcd046493c8254cfab2f769fcf480dc7c616a5f"
ODSS = room.ROOT / "docs/evaluation-panels/audio-odss-matched-room-2026-11-13.json"
PHONEMEDF = room.ROOT / "docs/evaluation-panels/audio-phonemedf-matched-transfer-2026-11-13.json"
ORIGINS = ("natural", "vits")
ROLES = ("development", "assessment")
METHOD = {
    "scope": "The whole foreground utterance is centered within the known 1.5–4.5-second window of a six-second stereo recording. The two outer one-second flanks are expected to carry background only. This geometry is part of the test design. The proposed check is inapplicable when there are no usable background-reference flanks.",
    "projection": "Use the existing square-root Hann transform. Estimate each complex mid/side regression coefficient from frames wholly contained within 0–1 and 5–6 seconds. Use the same diagonal floor, cap, sub-150 Hz exclusion, subtraction amount and output-availability checks as the existing whole-mixture frontal view. Apply frozen coefficients to all frames and reconstruct at the original sample clock.",
    "point": "Keep the existing mono/WPE/frontal operating points from the published ODSS development role. Fit the flank projection point as the maximum admitted flank score across genuine mixed speech and isolated competing-speech controls from that same development role. Leave all mono flags in place. Add a flag only when the previous selective cleanup rule would add one and the flank score exceeds its independently fitted point.",
    "fresh": "Compare the frozen routes on two matched VCTK prompts per reader, with four readers excluded from the existing ODSS, PhonemeDF, serving, matched-babble, and PACC source IDs. Render the complete native genuine and VITS utterance, matched at publisher ID and transcript, in the Aachen meeting room. Alternate lateral paths and reuse each exact competing-speech segment and gain across origins. Take the earliest four nonoverlapping eligible six-second clips per AMI session, with at least 3.9 seconds of annotated speech, 350 ms of simultaneous speech, and 300 ms of annotated speech in each background-reference flank. Assess 0 and -5 dB, room-only and masker-only controls.",
}


def prior():
    receipt = room.read(ODSS)
    points = room.fit(receipt["development"])
    room.require(points == receipt["operating_points"]["points"], "Prior development operating points differ")
    return receipt, points


def flank_view(stereo, fft=rfft, inverse=irfft):
    # Transform centers are indexed in original samples, including boundary padding.
    waveform = stereo.astype(np.float64)
    spectrum, window = room.wpe.transform(waveform, fft)
    centers = np.arange(spectrum.shape[1]) * room.wpe.HOP
    half = room.wpe.FRAME // 2
    flank = ((centers >= half) & (centers <= room.RATE - half)
             | (centers >= 5 * room.RATE + half) & (centers <= room.SAMPLES - half))
    mid = spectrum.mean(axis=2)
    side = (spectrum[:, :, 0] - spectrum[:, :, 1]) * 0.5
    power = np.mean(abs(side[:, flank]) ** 2, axis=1)
    floor = max(float(np.median(power)) * 0.1, 1e-12)
    gain = np.mean(mid[:, flank] * side[:, flank].conj(), axis=1) / (power + floor)
    gain *= np.minimum(1, 2 / np.maximum(abs(gain), 1e-12))
    gain[np.fft.rfftfreq(room.wpe.FRAME, 1 / room.RATE) < 150] = 0
    candidate = np.rint(room.wpe.restore(mid - .7 * gain[:, None] * side, window, inverse))
    room.require(np.isfinite(candidate).all(), "Nonfinite flank residual")
    if np.max(abs(candidate)) >= 32768 or np.mean(candidate ** 2) < .02 * np.mean(waveform.mean(axis=1) ** 2):
        return None
    return candidate.astype("<i2")


def local_view(stereo, target):
    actual = flank_view(stereo)
    alternate = flank_view(stereo, np.fft.rfft, np.fft.irfft)
    room.require((actual is None) == (alternate is None), "FFT replay changed flank availability")
    delta = 0 if actual is None else int(np.max(abs(actual.astype(np.int32) - alternate.astype(np.int32))))
    room.require(delta <= 1, "FFT replay of flank projection differs by more than one PCM unit")
    if actual is None or target is None:
        return actual, delta, None
    a, b = room.WINDOW
    mono = np.rint(stereo.astype(np.float64).mean(axis=1)).astype("<i2")
    return actual, delta, room.wpe.signal_gain(actual[a:b], mono[a:b], target[a:b].mean(axis=1))


def rank(value):
    return room.digest(f"{SCHEMA}:{value}".encode())


def exclusions():
    directory = room.ROOT / "backend/data/independent-evaluation"
    anchors = (directory / "audio-v9/source-plan.json", directory / "audio-matched-babble-v181/private-plan.json")
    others = sorted(path for path in directory.glob("audio-pacc*/**/*.json") if path.stat().st_size < 4_000_000)
    by_file = {str(path.relative_to(directory)): room.file_digest(path) for path in (*anchors, *others)}
    speakers = {hit.casefold() for path in (*anchors, *others)
                for hit in re.findall(r"(?<![a-z0-9])p[23]\d\d\b", path.read_text(), re.I)}
    return speakers, by_file


def source_plan(catalog_cache):
    entries = room.catalog(catalog_cache, False)
    transcripts = room.member(catalog_cache, entries[room.TRANSCRIPTS], False)
    room.require(room.digest(transcripts) == room.TRANSCRIPTS_SHA, "Published prompt metadata differs")
    texts = defaultdict(set)
    for item in csv.DictReader(io.StringIO(transcripts.decode("utf-8-sig")), delimiter="|"):
        texts[item["id"]].add(" ".join(re.findall(r"\w+", item["text"].casefold())))
    excluded, by_file = exclusions()
    grouped = defaultdict(list)
    for path, natural in entries.items():
        match = re.fullmatch(r"odss/natural/vctk/([^/]+)/([^/]+)\.wav", path)
        if match is None:
            continue
        speaker, key = match.groups()
        vits = entries.get(path.replace("/natural/", "/vits/", 1))
        if speaker not in excluded and len(texts[key]) == 1 and vits is not None and all(
                53100 <= entry.file_size <= 95800 for entry in (natural, vits)):
            text, = texts[key]
            if len(text.split()) >= 4:
                grouped[speaker].append((natural, vits, text))
    selected, used = [], set()
    for speaker in sorted(grouped, key=rank):
        available = []
        for natural, vits, text in sorted(grouped[speaker], key=lambda row: rank(row[0].filename)):
            if text in used:
                continue
            available.append((natural, vits, text))
            used.add(text)
            if len(available) == 2:
                break
        if len(available) == 2:
            selected.append((speaker, available))
        if len(selected) == 4:
            break
    room.require(len(selected) == 4, "Complete distinct reader schedule required")
    return selected, by_file


def source_inputs(args, development):
    selected, prior_rosters = source_plan(args.odss_cache)
    excluded = {row["speaker"] for row in development}
    old_texts = {row["prompt_sha256"] for row in development}
    old_pcm = {source["pcm_sha256"] for row in development for source in row["members"].values()}
    roster, dry = [], {}
    for index, (speaker, pairs) in enumerate(selected):
        room.require(speaker not in excluded, "Prior ODSS reader reused")
        group = f"vctk-reader-{index:02d}"
        for prompt, (natural, vits, text) in enumerate(pairs):
            case = f"{group}-prompt-{prompt}"
            text_pin = room.digest(text.encode())
            room.require(text_pin not in old_texts, "Prior or repeated prompt reused")
            old_texts.add(text_pin)
            bindings = {}
            for origin, entry in zip(ORIGINS, (natural, vits), strict=True):
                body = room.member(args.vctk_cache, entry, False)
                pcm = room.native_wav(body)
                pcm_pin = room.digest(pcm)
                room.require(pcm_pin not in old_pcm, "Prior or repeated PCM reused")
                old_pcm.add(pcm_pin)
                dry[case, origin] = np.frombuffer(pcm, dtype="<i2")
                bindings[origin] = {"publisher_member": entry.filename, "pcm_sha256": pcm_pin, "wav_sha256": room.digest(body)}
            roster.append({"case": case, "group": group, "role": "assessment", "speaker": speaker,
                           "prompt_sha256": text_pin, "sources": bindings})
    return roster, dry, prior_rosters


def aachen_room(path):
    room.require(room.file_digest(path) == AIR_SHA, "Previously pinned AIR archive differs")
    responses, identities = {}, {}
    with ZipFile(path) as archive:
        notice = archive.read("AIR_1_4/license.txt")
        room.require(b"MIT License" in notice, "AIR license differs")
        for role, position in (("front", 4), ("left", 1), ("right", 2)):
            channels, members = [], []
            for channel in (1, 0):
                name = f"AIR_1_4/air_binaural_meeting_{channel}_1_{position}.mat"
                raw = archive.read(name)
                data = loadmat(io.BytesIO(raw), simplify_cells=True)
                impulse, meta = data["h_air"], data["air_info"]
                room.require(meta["room"] == "meeting" and meta["head"] == 1 and meta["channel"] == channel
                             and meta["fs"] == 48000 and impulse.ndim == 1 and np.isfinite(impulse).all(),
                             "Unexpected AIR response metadata")
                channels.append(impulse)
                members.append({"name": name, "sha256": room.digest(raw)})
            impulse = np.ascontiguousarray(resample_poly(np.column_stack(channels), 1, 3, axis=0), dtype="<f8")
            peaks = np.argmax(abs(impulse), axis=0)
            lag = int(peaks[0] - peaks[1])
            room.require(abs(lag) <= 2 if role == "front" else abs(lag) >= 6, "Unexpected early-peak geometry")
            responses[role] = impulse
            identities[role] = {"members": members, "peaks_at_16k": peaks.tolist(), "distance_cm": meta["distance"],
                                "impulse_sha256": room.digest(impulse.tobytes())}
    return responses, identities, room.digest(notice)


def build_deliveries(item, dry, masker, responses, convolve, origins, levels, room_only):
    case = item["case"]
    position = "left" if case.endswith("-0") else "right"
    targets = {}
    for origin in origins:
        signal, _ = room.padded(dry[case, origin])
        rendered = convolve(signal, responses["front"])
        targets[origin] = rendered / room.rms(rendered.mean(axis=1))
    noise = convolve(masker, responses[position])
    noise /= room.rms(noise.mean(axis=1))
    peak = max(np.max(abs(target) + abs(noise) * 10 ** (-min(levels) / 20)) for target in targets.values())
    scale = 24000 / peak
    backgrounds = {level: np.rint(noise * scale * 10 ** (-level / 20)).astype(np.int32) for level in levels}
    for origin, values in targets.items():
        target = np.rint(values * scale).astype(np.int32)
        if room_only:
            yield origin, None, target.astype("<i2"), target, None, position
        for level, bg in backgrounds.items():
            mixed = target + bg
            measured = float(20 * np.log10(room.rms(target.mean(axis=1)) / room.rms(bg.mean(axis=1))))
            room.require(np.max(abs(mixed)) <= 24001 and abs(measured - level) <= .02, "Target level or peak differs")
            yield origin, level, mixed.astype("<i2"), target, round(measured, 6), position
    for level, bg in backgrounds.items():
        yield "background", level, bg.astype("<i2"), None, None, position


def fresh_maskers(spec, used, meeting):
    path, pin, counts, prior = spec
    selected = []
    with wave.open(str(path), "rb") as source:
        room.require(source.getparams()[:3] == (1, 2, room.RATE), "Unexpected native AMI clock")
        stops = min(len(counts), source.getnframes() // 160)
        for tick in range(500, stops - 600 + 1, 25):
            start = tick * 160
            if any(abs(start - old) < room.SAMPLES for old in used) or any(
                    start < (second + 3) * room.RATE and second * room.RATE < start + room.SAMPLES for second in prior):
                continue
            active = counts[tick:tick + 600]
            overlap = int(np.count_nonzero(active >= 2))
            flanks = [int(np.count_nonzero(active[:100])), int(np.count_nonzero(active[-100:]))]
            if np.count_nonzero(active) < 390 or overlap < 35 or min(flanks) < 30:
                continue
            source.setpos(start)
            pcm = source.readframes(room.SAMPLES)
            room.require(len(pcm) == 2 * room.SAMPLES, "Truncated AMI excerpt")
            selected.append((np.frombuffer(pcm, dtype="<i2"), {"meeting": meeting, "start": start,
                             "capture_sha256": pin, "pcm_sha256": room.digest(pcm), "overlap_centiseconds": overlap,
                             "speech_flanks_centiseconds": flanks}))
            used.add(start)
            if len(selected) == 4:
                break
    room.require(len(selected) == 4, "Incomplete annotation-supported masker schedule")
    return iter(selected)


def prepare(args):
    old, points = prior()
    all_odss, dry = room.plan(args.odss_cache, False)
    old_positions, old_background = room.backgrounds(args.ami, all_odss)
    prior_inputs = {row["id"]: row for row in old["development"]["rows"]}
    old_rooms, _ = room.room_impulses(args.surrey, args.bbc)
    new, fresh_dry, exclusion_rosters = source_inputs(args, all_odss)
    fresh_rooms, room_pins, notice = aachen_room(args.air)
    mask_specs = aachen.ami_inputs(args.fresh_ami, stress.annotation_path())
    earlier = room.read(args.prior_aachen / "roster.json")
    meetings = ("ES2005a", "ES2008a")
    used = {meeting: {row["ami"]["start"] for row in earlier if row["ami"]["meeting"] == meeting} for meeting in meetings}
    prior_stress = room.read(args.prior_stress / "roster.json")
    evidence = ODSS.parent
    for root, filename in ((args.prior_aachen, "audio-spatial-mask-aachen-transfer-2026-11-13.json"),
                           (args.prior_stress, "audio-room-stress-transfer-2026-11-13.json")):
        room.require(room.file_digest(root / "roster.json") == room.read(evidence / filename)["private_roster_sha256"],
                     "Earlier room roster differs from its published commitment")
    room.require(sum(map(len, used.values())) == 25 and len(prior_stress) == 167, "Prior acoustic test roles differ")
    maskers = {meeting: fresh_maskers(mask_specs[meeting], used[meeting], meeting) for meeting in meetings}
    fresh_backgrounds = {}
    rows, views = [], {}
    a, b = room.WINDOW
    for item in all_odss:
        if item["role"] != "development":
            continue
        case = item["case"]
        for origin, level, stereo, target, measured, _, pos in room.room_views(item, dry, old_background[case], old_rooms):
            if level is None or origin not in ("natural", "background"):
                continue
            key = f"{case}:{origin}:{level}"
            pin = room.digest(stereo.tobytes())
            room.require(pin == prior_inputs[key]["input_sha256"], "Retained ODSS development input differs")
            flank, delta, gain = local_view(stereo, target)
            views[key] = {"flank": flank}
            rows.append({"id": key, "case": case, "group": item["group"], "role": "development", "origin": origin,
                         "level_db": level, "position": pos, "snr_db": measured, "input_sha256": pin,
                         "window_hashes": {"flank": None if flank is None else room.digest(flank[a:b].tobytes())},
                         "fft_max_lsb": {"flank": delta}, "gain_db": {"flank": gain}})
    for item in new:
        case = item["case"]
        meeting = meetings[int(item["group"].rsplit("-", 1)[1]) % len(meetings)]
        masker, descriptor = next(maskers[meeting])
        fresh_backgrounds[case] = descriptor
        for origin, level, stereo, target, measured, pos in build_deliveries(
                item, fresh_dry, masker, fresh_rooms, stress.convolve, ORIGINS, room.LEVELS, True):
            key = f"{case}:{origin}:{'room' if level is None else level}"
            outputs, delta, gains = room.wave_views(stereo, target, level)
            if level is not None:
                flank, replay, gain = local_view(stereo, target)
                outputs["flank"] = flank
                delta["flank"] = replay
                if gains is not None:
                    gains["flank"] = gain
            views[key] = outputs
            rows.append({"id": key, "case": case, "group": item["group"], "role": "assessment", "origin": origin,
                         "level_db": level, "position": pos, "snr_db": measured, "input_sha256": room.digest(stereo.tobytes()),
                         "window_hashes": {arm: None if values is None else room.digest(values[a:b].tobytes())
                                           for arm, values in outputs.items()}, "fft_max_lsb": delta, "gain_db": gains})
    room.require(len(rows) == 96 and len({row["input_sha256"] for row in rows}) == len(rows), "Frozen delivery coverage differs")
    lock = {"schema": SCHEMA, "method": METHOD, "primary_points": points,
            "odss_receipt_sha256": room.file_digest(ODSS), "phonemedf_receipt_sha256": room.file_digest(PHONEMEDF),
            "code_sha256": room.file_digest(Path(__file__)), "room_code_sha256": room.file_digest(Path(room.__file__)),
            "air_archive_sha256": AIR_SHA, "air_license_sha256": notice, "room_responses": room_pins,
            "odss_catalog_sha256": room.odss.DIRECTORY_SHA256, "odss_transcripts_sha256": room.TRANSCRIPTS_SHA,
            "speech_assets": room.models.ASSETS, "window_samples": list(room.WINDOW), "levels_db": list(room.LEVELS),
            "source_roster_sha256": room.digest(room.json_bytes(new)), "delivery_roster_sha256": room.digest(room.json_bytes(rows)),
            "excluded_source_rosters": exclusion_rosters, "development_background_roster_sha256": room.digest(room.json_bytes(old_positions)),
            "assessment_background_roster_sha256": room.digest(room.json_bytes(fresh_backgrounds)),
            "assessment_background_recording_sha256": {meeting: mask_specs[meeting][1] for meeting in meetings}}
    return lock, rows, views


def fit(rows):
    values = [row["responses"]["flank"] for row in rows if row["responses"]["flank"] is not None]
    room.require(len(rows) == 32 and all(row["origin"] in ("natural", "background") for row in rows)
                 and {row["group"] for row in rows if row["origin"] == "natural" and row["responses"]["flank"] is not None}
                 == {row["group"] for row in rows} and values, "Incomplete development controls")
    return max(values)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "fit", "score", "replay"))
    for arg in ("odss-cache", "vctk-cache", "surrey", "bbc", "ami", "fresh-ami", "air", "prior-aachen", "prior-stress", "run"):
        parser.add_argument(f"--{arg}", type=Path, required=True)
    parser.add_argument("--weights", type=Path, default=room.ROOT / "models/weights/audio-upgrade")
    args = parser.parse_args()
    lock, rows, views = prepare(args)
    frozen = args.run / "freeze.json"
    if args.phase == "freeze":
        args.run.mkdir(mode=0o700)
        room.write(frozen, lock)
        room.write(args.run / "delivery-roster.json", rows)
        print(f"Froze {len(rows)} matched development and assessment deliveries")
        return
    room.require(frozen.read_bytes() == room.json_bytes(lock) and
                 (args.run / "delivery-roster.json").read_bytes() == room.json_bytes(rows), "Frozen source or code differs")
    if args.phase == "replay":
        print(f"Replayed {len(rows)} frozen deliveries and original view hashes")
        return
    dev_path, point_path = (args.run / filename for filename in ("development.json", "flank-point.json"))
    if args.phase == "fit":
        room.require(not dev_path.exists() and not point_path.exists(), "Development results already exist")
        dev = room.score(rows, views, "development", args.weights)
        room.write(dev_path, dev)
        room.write(point_path, {"freeze_sha256": room.file_digest(frozen), "development_sha256": room.file_digest(dev_path),
                               "flank_point": fit(dev["rows"])})
        return
    point, dev = room.read(point_path), room.read(dev_path)
    room.require(point == {"freeze_sha256": room.file_digest(frozen), "development_sha256": room.file_digest(dev_path),
                           "flank_point": fit(dev["rows"])}, "Development-only flank point differs")
    out = args.run / "public.json"
    room.require(not out.exists(), "Fresh assessment already scored")
    assessment = room.score(rows, views, "assessment", args.weights)
    room.write(out, {"freeze": lock, "development": dev, "operating_point": point, "assessment": assessment})


if __name__ == "__main__":
    main()
