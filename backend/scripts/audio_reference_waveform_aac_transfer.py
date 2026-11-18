# SPDX-License-Identifier: Apache-2.0
"""Replay the frozen earlier-copy waveform evaluation across WAV, FLAC and AAC."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from functools import partial
from itertools import product
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

from app.evidence import audio_reference_waveform as waveform
from scripts import audio_aligned_speech_transfer as prior
from scripts import audio_reference_waveform_transfer as probe
from scripts.video_shared_sound_panel import ARCHIVE_SHA1
from scripts.video_stereo_covariance_panel import read, save

FREEZE_SHA256 = "965f3478771b0dae6a7f50b49341873157693b31fcf7b4801f0638ffc5d26a6e"
FREEZE = prior.io.PUBLIC / "audio-reference-waveform-aac-freeze-2026-11-18.json"
ORIGINAL = prior.io.PUBLIC / "audio-reference-waveform-2026-11-14.json"
SAMPLES = 96000
OFFSETS = (2, 8, 14, 20)
SCENES = ("bus", "office")
RATES = (32, 64, 96, 128)
ROLES = ("cut_paste", "unchanged", "resyn")
SCHEMA = "chipmunk.experimental-waveform-codec-transfer/1"
NATIVE_CLOCK = {"sample_rate": 48000, "channels": 1, "time_base": "1/48000",
                "presentation_end_exclusive_pts": 288000}
UNASSESSED = {"equal_duration_required", "insufficient_matching_waveform"}


def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def background(archive):
    import numpy as np
    from scipy.signal import resample_poly

    from scripts.audio_stereo_ambience_panel import _archive_pcm

    with archive.open("rb") as handle:
        if hashlib.file_digest(handle, "sha1").hexdigest() != ARCHIVE_SHA1:
            raise ValueError("DCASE archive differs from its pinned publisher copy")
    recordings, tracks = [], {}
    with ZipFile(archive) as bundle:
        for scene in SCENES:
            member = f"scenes_stereo/{scene}06.wav"
            source = bundle.read(member)
            mono = _archive_pcm(source)[:, 0]
            clips = [resample_poly(mono[start * 44100:(start + 6) * 44100].astype(float), 160, 441)
                     for start in OFFSETS]
            tracks[scene] = [np.clip(np.rint(clip), -32768, 32767).astype("<i2") for clip in clips]
            if any(len(samples) != SAMPLES for samples in tracks[scene]):
                raise ValueError("Prepared room excerpts differ from their frozen extent")
            recordings.append({"member": member, "sha256": hashlib.sha256(source).hexdigest(),
                               "prepared_pcm_sha256": [hashlib.sha256(track.tobytes()).hexdigest() for track in tracks[scene]]})
    return recordings, tracks


def flac(source, output):
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-protocol_whitelist", "file", "-i",
                    str(source), "-map_metadata", "-1", "-c:a", "flac", "-sample_fmt", "s16", str(output)],
                   check=True, capture_output=True, timeout=20)


def aac48(pcm, output, rate):
    subprocess.run(
        ["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-f", "s16le", "-ar", "16000", "-ac", "1",
         "-i", "pipe:0", "-ar", "48000", "-map_metadata", "-1", "-threads", "1", "-c:a", "aac",
         "-profile:a", "aac_low", "-b:a", f"{rate}k", "-fflags", "+bitexact", "-flags:a", "+bitexact", str(output)],
        input=pcm, check=True, capture_output=True, timeout=25,
    )


def bitrate(path):
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-protocol_whitelist", "file", "-select_streams", "a",
         "-show_entries", "stream=bit_rate", "-of", "json", str(path)],
        check=True, capture_output=True, timeout=20,
    )
    stream, = json.loads(result.stdout)["streams"]
    return int(stream["bit_rate"])


def track(path):
    from app.evidence import audio_reference as reference

    return reference._read_track(path, reference._identity(path, path.name), max_seconds=reference.MAX_SECONDS,
                                 origin=Fraction(0), video=False)


def pcm(track_result):
    import numpy as np

    return np.rint(track_result[1] * 32768).astype("<i2")


def useful(measurement, truth):
    span = probe.validate(measurement)
    return bool(span and any(prior.original.common.useful(span, target) for target in truth))


def evaluate_group(cases, *, scratch, rate):
    import numpy as np

    from app.evidence import audio_reference as reference

    with TemporaryDirectory(dir=scratch, prefix="files-") as folder:
        folder = Path(folder)
        native_earlier, flac_earlier = (folder / f"earlier.{suffix}" for suffix in ("wav", "flac"))
        first = cases[0][-2]
        prior.original.wav(native_earlier, first)
        flac(native_earlier, flac_earlier)
        earlier = track(native_earlier), track(flac_earlier)
        e_hashes = sha(native_earlier), sha(flac_earlier)
        if not all(item[4] and np.array_equal(pcm(item[0]), first) for item in earlier):
            raise ValueError("The earlier WAV and FLAC need native sample parity")
        rows = []
        for panel, item, scene, role, truth, earlier_input, submitted in cases:
            if len(earlier_input) != SAMPLES or len(submitted) != SAMPLES or not np.array_equal(first, earlier_input):
                raise ValueError("The same six-second earlier source must be used throughout the group")
            wav, lossless, aac = (folder / f"submitted.{suffix}" for suffix in ("wav", "flac", "m4a"))
            prior.original.wav(wav, submitted)
            flac(wav, lossless)
            aac48(submitted.tobytes(), aac, rate)
            decoded = track(wav), track(lossless), track(aac)
            for source, latter in zip(earlier, decoded[:2], strict=True):
                native, reason, _ = reference._short_reference_tracks((source, latter))
                if (reason is not None or any(track[2]["samples"] != SAMPLES or track[2]["source_pts"] != 0 for track in native)
                        or not np.array_equal(pcm(native[0]), first) or not np.array_equal(pcm(native[1]), submitted)):
                    raise ValueError("The WAV and FLAC delivery must preserve all samples and the same zero-origin clock")
            verified = reference._aac_presented_tracks((flac_earlier, aac), (earlier[1], decoded[2]), high_rate=True)
            if verified is None:
                raise ValueError(f"The exact AAC presentation clock could not be verified for {panel}/{item['id']}/{scene}/{role}")
            presented, proof = verified
            if (proof["native_clock"] != NATIVE_CLOCK or proof["presentation_end_exclusive_pts"] != SAMPLES
                    or any(len(track[1]) != SAMPLES for track in presented)):
                raise ValueError("AAC decode must preserve the verified six-second sample extent")
            base = probe.measurement(first, submitted)
            encoded = probe.measurement(*(pcm(value) for value in presented))
            rows.append({
                "panel": panel, "id": item["id"], "reader": item["reader"], "role": role,
                "scene": scene, "truth": truth, "wav": base, "aac": encoded, "clock": proof,
                "aac_stream_bit_rate": bitrate(aac),
                "file_sha256": {"earlier_wav": e_hashes[0], "earlier_flac": e_hashes[1], "later_wav": sha(wav),
                                "later_flac": sha(lossless), "later_aac": sha(aac)},
            })
    return rows


def summarize(rows):
    grouped = defaultdict(Counter)
    new_ordinary = set()
    for row in rows:
        scene, role = row["scene"], row["role"]
        base, coded = row["wav"], row["aac"]
        left = useful(base, row["truth"])
        right = useful(coded, row["truth"])
        base_flag, coded_flag = (bool(it["selected"]) for it in (base, coded))
        cell = grouped[scene, role]
        cell.update(total=1, wav_useful=left, aac_useful=right, aac_gained=right and not left,
                    aac_lost=left and not right, wav_flags=base_flag, aac_flags=coded_flag,
                    aac_novel_flags=coded_flag and not base_flag, wav_unassessed=base["status"] in UNASSESSED,
                    aac_unassessed=coded["status"] in UNASSESSED)
        if role == "unchanged" and coded_flag and not base_flag:
            new_ordinary.add(row["reader"])
    edits = [count for (_, role), count in grouped.items() if role == "cut_paste"]
    ordinary = [count for (_, role), count in grouped.items() if role == "unchanged"]
    resynthesized = [count for (_, role), count in grouped.items() if role == "resyn"]
    if [sum(cell["total"] for cell in cells) for cells in (edits, ordinary, resynthesized)] != [81, 117, 36]:
        raise ValueError("Results must cover all annotated, ordinary and resynthesized comparisons")
    edit_base = sum(cell["wav_useful"] for cell in edits)
    remaining = edit_base - sum(cell["aac_lost"] for cell in edits)
    ordinary_new = sum(cell["aac_novel_flags"] for cell in ordinary)
    behavior = {
        "wav_useful_noisy_edits": edit_base, "aac_useful_noisy_edits": sum(cell["aac_useful"] for cell in edits),
        "aac_preserved_wav_edits": remaining, "aac_new_ordinary_flags": ordinary_new,
        "newly_flagged_ordinary_readers": len(new_ordinary),
        "bounded_sample_and_clock_for_all_formats": True,
        "approval_criteria_met": (edit_base >= 30 and 100 * remaining >= 85 * edit_base
                                  and all(cell["wav_useful"] - cell["aac_useful"] <= 3 for cell in edits)
                                  and ordinary_new <= 2 and len(new_ordinary) <= 1
                                  and sum(cell["aac_flags"] - cell["wav_flags"] for cell in ordinary) <= 2),
    }
    return {f"{scene}/{role}": dict(counts) for (scene, role), counts in sorted(grouped.items())}, behavior


def spec():
    recorded = read(ORIGINAL)
    if sha(FREEZE) != FREEZE_SHA256 or recorded["implementation_sha256"] != sha(Path(waveform.__file__)):
        raise ValueError("The format plan or frozen earlier-copy waveform method differs")
    return {
        "schema": SCHEMA, "freeze_sha256": FREEZE_SHA256, "original_receipt_sha256": sha(ORIGINAL),
        "source_receipt_sha256": {name: sha(path) for name, path in zip(prior.PANELS, prior.sources(), strict=True)},
        "archive_sha1": ARCHIVE_SHA1,
    }


def review(receipt):
    if any(receipt.get(key) != value for key, value in spec().items()) or receipt["nominal_kbps"] not in RATES:
        raise ValueError("The receipt differs from the pinned comparison plan")
    rows = receipt["rows"]
    roster, rates = defaultdict(set), []
    for row in rows:
        expected = (row["role"], row["scene"])
        choices = roster[row["panel"], row["id"]]
        if expected in choices or row["panel"] not in prior.PANELS or (
            bool(row["truth"]) != (row["role"] == "cut_paste")
            or any(type(start) is not int or type(end) is not int or not 0 <= start < end <= SAMPLES
                   for start, end in row["truth"])
            or row["clock"]["native_clock"] != NATIVE_CLOCK or row["clock"]["presentation_start_pts"] != 0
            or row["clock"]["presentation_end_exclusive_pts"] != SAMPLES or row["clock"]["side"] != "submitted"
            or row["clock"]["time_base"] != "1/16000" or row["clock"]["sample_rate"] != 16000
            or row["clock"]["decoded_samples"] != SAMPLES + row["clock"]["excluded_trailing_samples"]
            or not 0 <= row["clock"]["excluded_trailing_samples"] <= 342
            or any(reading["samples"] != [SAMPLES, SAMPLES] for reading in (row["wav"], row["aac"]))
            or row["wav"]["sha256"][0] != row["aac"]["sha256"][0]
            or type(row["aac_stream_bit_rate"]) is not int or row["aac_stream_bit_rate"] <= 0
        ):
            raise ValueError("A held case differs from its exact sample clock or labeled scope")
        choices.add(expected)
        rates.append(row["aac_stream_bit_rate"])
    scenes = (*SCENES, "competing_speech")
    for choices in roster.values():
        base = set(product(("unchanged",), scenes))
        if choices not in [base | set(product((edit,), scenes)) for edit in ("cut_paste", "resyn")]:
            raise ValueError("Each source needs the same three conditions and matched ordinary controls")
    if len(roster) != 39:
        raise ValueError("The reserved plan must cover 39 source clips")
    counts, behavior = summarize(rows)
    if (receipt["summary"] != counts or receipt["behavior"] != behavior
            or receipt["stream_bit_rate"] != {"min": min(rates), "max": max(rates)}):
        raise ValueError("Reported measurements differ from the retained case results")
    return counts, behavior


def run(scratch, archive, full, short, rate):
    pinned = spec()
    sounds, noises = background(archive)
    groups = defaultdict(list)
    for panel, source, folder in zip(prior.PANELS, prior.sources(), (full, short), strict=True):
        case_inputs = prior.cases(prior.io.read(source), folder, noises)
        for item, scene, role, truth, first, second in case_inputs:
            if scene != "clean":
                groups[panel, item["id"]].append((panel, item, scene, role, truth, first, second))
    if len(groups) != 39 or sum(len(group) for group in groups.values()) != 234:
        raise ValueError("The earlier SINE source pair selection differs")
    print(f"Prepared {len(groups)} source clips and verified the pinned backgrounds", flush=True)
    rows = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        evaluate = partial(evaluate_group, scratch=scratch, rate=rate)
        for finished, group_rows in enumerate(pool.map(evaluate, groups.values()), 1):
            rows.extend(group_rows)
            if finished % 5 == 0 or finished == len(groups):
                print(f"Verified {finished}/{len(groups)} source clips across all three formats", flush=True)
    counts, behavior = summarize(rows)
    rates = [row["aac_stream_bit_rate"] for row in rows]
    return pinned | {
        "nominal_kbps": rate, "backgrounds": sounds, "rows": rows, "summary": counts, "behavior": behavior,
        "stream_bit_rate": {"min": min(rates), "max": max(rates)},
        "ffmpeg": subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True, check=True).stdout.splitlines()[0],
    }


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    commands = cli.add_subparsers(dest="mode", required=True)
    reviewer = commands.add_parser("review")
    reviewer.add_argument("receipt", type=Path)
    replay = commands.add_parser("run")
    for field in ("receipt", "archive", "full", "short"):
        replay.add_argument(field, type=Path)
    replay.add_argument("--kbps", type=int, choices=RATES, default=64)
    args = cli.parse_args()
    if args.mode == "review":
        receipt = read(args.receipt)
    else:
        if args.receipt.exists():
            cli.error("Replay needs a new output file.")
        with TemporaryDirectory(prefix="chipmunk-waveform-codec-transfer-") as location:
            receipt = run(Path(location), args.archive, args.full, args.short, args.kbps)
    summary, behavior = review(receipt)
    if args.mode == "run":
        save(args.receipt, receipt)
    print(json.dumps({"summary": summary, "behavior": behavior, "stream_bit_rate": receipt["stream_bit_rate"]}, sort_keys=True))
