# SPDX-License-Identifier: Apache-2.0
"""Qualify original speech word intervals and native picture clocks for the GRID reserve."""

from __future__ import annotations

import argparse
import hashlib
import io
import math
import wave
import zlib
from collections import defaultdict
from fractions import Fraction
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

import numpy as np
from scipy.signal import correlate, resample_poly

from app.detectors.video.decode import decode_frames, probe_frames
from scripts import acquire_grid_transfer_v130 as acquisition
from scripts import grid_phone_alignment_v161 as words
from scripts import grid_transfer_plan_v143 as planning
from scripts import review_video_lipsync_local_delay_transfer as audio
from scripts.video_common_sources_v89 import read, require, sha, write

PUBLISHER_RATE, PCM_RATE, PICTURE_RATE = 25_000, 16_000, 25
WAV_PADDING = 250
SPEC = {
    "schema": "chipmunk.grid-reserve-word-clock/1",
    "source": "All sixteen reserve originals in eight previously frozen native input groups.",
    "directory_match": "Unique complete equality of each video archive's 1,000 utterance stems with one publisher word directory.",
    "wave_match": "CRC-verified endpointed publisher WAV. Match the complete resampled waveform and each of its six "
                  "published word segments to the actual decoded video soundtrack. Retain the measured offsets and independent per-word residuals.",
    "full_search_samples": 960, "word_search_samples": 320,
    "minimum_wave_correlation": .90, "minimum_word_correlation": .80, "maximum_word_drift_samples": 32,
    "clock": "Use half-open words. Map a publisher sample index at 25 kHz to source time with the measured "
             "common waveform offset. Retain first included canonical sample and source picture at or after each boundary.",
    "phone_truth": "The released alignment contains words. Acoustic phone and release labels require separate evidence.",
}


def publisher_record():
    path = acquisition.METADATA / "grid-record.json"
    require(sha(path) == acquisition.RECORD_SHA, "Exact official GRID metadata required")
    body = read(path)
    require(body["metadata"]["license"]["id"] == "cc-by-4.0", "Official GRID licence differs")
    return {record["key"]: record for record in body["files"]}


def plan():
    folder = planning.TRIAL / "bind-00"
    result, profile = read(folder / "output/result.json"), read(folder / "profile.json")
    artifact = folder / "output/private-plan.json"
    require(result["status"] == "complete" and result["freeze_sha256"] == sha(planning.PUBLIC)
            and result["private_plan_sha256"] == sha(artifact) and profile["child_exit_code"] == 0
            and profile["guard"] is None and profile["sample_errors"] == 0
            and profile["worker_receipt_sha256"] == sha(folder / "output/result.json"),
            "Require the completed archived plan and its exact artifact")
    sources = read(artifact)["sources"]
    reserve = [item for item in sources if item["role"] == "reserve"]
    require(len(reserve) == 16 and len({item["component_index"] for item in reserve}) == 8
            and not {item["component_index"] for item in reserve} & {item["component_index"] for item in sources if item["role"] == "pilot"},
            "Require all eight untouched reserve groups")
    return {item["ordinal"]: item for item in reserve}, result["private_plan_sha256"]


def inventory(source, word_groups):
    slot = source["source_slot"]
    bundle = planning.ARCHIVE / "grid-acquisition-continuation-v135" / f"source-{slot:02}" / "archive.zip"
    publication = source["input"]["planned"]["source"]
    label = Path(publication["publisher_name"]).stem
    require(bundle.stat().st_size == publication["bytes"], "Retained publisher video archive size differs")
    with ZipFile(bundle) as videos:
        members = {}
        for info in videos.infolist():
            path = PurePosixPath(info.filename)
            if len(path.parts) != 2 or path.parent.name != label or path.suffix != ".mpg" or path.name.startswith("._"):
                continue
            require(path.stem not in members, "Video archive utterance is ambiguous")
            members[path.stem] = info
    require(len(members) == 1000, "Require the complete publisher video inventory")
    matches = [group for group, entries in word_groups.items() if set(entries) == set(members)]
    require(len(matches) == 1 and all(not set(members) & set(entries) for group, entries in word_groups.items() if group != matches[0]),
            "Publisher word directory is incomplete or ambiguous")
    return label, matches[0], members


def wave_pcm(path, rate):
    with wave.open(str(path) if isinstance(path, Path) else path, "rb") as stream:
        require((stream.getnchannels(), stream.getframerate(), stream.getsampwidth(), stream.getcomptype()) == (1, rate, 2, "NONE"),
                "Expected publisher or canonical mono PCM")
        count = stream.getnframes()
        pcm = np.frombuffer(stream.readframes(count + 1), dtype="<i2")
    require(len(pcm) == count, "Truncated waveform")
    return pcm


def best_match(source, sample, near, tolerance):
    lo = max(0, near - tolerance)
    hi = min(len(source) - len(sample), near + tolerance)
    require(len(sample) and lo <= hi, "No possible waveform placement")
    target = source[lo:hi+len(sample)].astype(np.float64)
    signal = sample.astype(np.float64) - float(np.mean(sample))
    n = len(signal)
    sums = np.r_[0, np.cumsum(target)]
    squares = np.r_[0, np.cumsum(target * target)]
    norm = np.sqrt(np.maximum(0., squares[n:] - squares[:-n] - (sums[n:] - sums[:-n])**2/n))
    normalizer = np.linalg.norm(signal) * norm
    correlation = correlate(target, signal, mode="valid", method="fft") / np.maximum(normalizer, 1e-12)
    best = int(correlation.argmax())
    return lo + best, float(np.clip(correlation[best], -1, 1))


def exact_native(row):
    path = Path(row["source_file"])
    digest = row["source_sha256"]
    require(path.resolve().is_relative_to(planning.ARCHIVE) and sha(path) == digest, "Retained original source differs")
    table = probe_frames(path, timeout_s=15)
    frozen = row["native_video_table"]
    require(table.to_json() == frozen and not (table.rotation or table.interpolated or table.truncated)
            and [value * table.tb for value in table.pts] == [Fraction(i, PICTURE_RATE) for i in range(75)],
            "Original picture clock differs")
    pixels = []
    for frame in decode_frames(path, table, None, timeout_s=25):
        proof = row["pictures"][frame.index]
        require(frame.pts_verified and frame.pts == proof["pts"] and frame.index == proof["frame_index"]
                and audio.sha(frame.rgb.tobytes()) == proof["rgb_sha256"], "Original native picture differs")
        pixels.append(frame.rgb)
    require(len(pixels) == 75, "All 75 source pictures required")
    pcm, origin, dropped = audio.source_pcm(path, digest)
    wav = Path(row["wav_file"])
    require(sha(wav) == row["wav_file_sha256"] and np.array_equal(pcm, wave_pcm(wav, PCM_RATE))
            and not dropped and origin == Fraction(row["audio_origin_seconds"]) and len(pcm) == row["pcm_mapping"]["samples"],
            "Re-decoded original PCM differs")
    return table, pixels, pcm, origin


def project(alignment, official_pcm, native, origin, table):
    begin, end = alignment[0][1], alignment[-1][2]
    lead = begin-WAV_PADDING
    require(len(official_pcm) == end-begin+2*WAV_PADDING+1, "Publisher WAV endpoint extent differs")
    signal = resample_poly(official_pcm.astype(np.float64), 16, 25)
    expected = round(Fraction(lead * PCM_RATE, PUBLISHER_RATE))
    best, coefficient = best_match(native, signal, expected, SPEC["full_search_samples"])
    require(coefficient >= SPEC["minimum_wave_correlation"], "Official waveform identity is unconfirmed")
    shift = Fraction(best, PCM_RATE) - Fraction(lead, PUBLISHER_RATE)
    source_times = [pts * table.tb for pts in table.pts]
    measurements = []
    for name, a, b in alignment:
        x, y = (round(Fraction((index-lead) * PCM_RATE, PUBLISHER_RATE)) for index in (a,b))
        observed, quality = best_match(native, signal[x:y], best+x, SPEC["word_search_samples"])
        residual = observed - (best+x)
        require(abs(residual) <= SPEC["maximum_word_drift_samples"] and quality >= SPEC["minimum_word_correlation"],
                "Official word timing does not follow the common waveform alignment")
        times = [origin + Fraction(i, PUBLISHER_RATE) + shift for i in (a,b)]
        samples = [math.ceil((value-origin)*PCM_RATE) for value in times]
        pictures = [sum(pts < value for pts in source_times) for value in times]
        require(0 <= samples[0] < samples[1] <= len(native) and 0 <= pictures[0] < pictures[1] < len(source_times),
                "Projected word lies outside the exact source clocks")
        measurements.append({"word": name, "publisher_samples": [a,b], "source_seconds": list(map(str,times)),
                             "source_pcm_samples": samples, "source_native_picture_indices": pictures,
                             "word_residual_samples": residual, "wave_correlation": quality,
                             "source_word_pcm_sha256": audio.sha(native[samples[0]:samples[1]].tobytes())})
    return {"whole_wave_correlation": coefficient, "publisher_to_source_shift_seconds": str(shift), "words": measurements}


def qualify(wavs, output):
    sources, plan_sha = plan()
    downloads = read(wavs / "receipt.json")
    asset = downloads["publisher_asset"]
    published = publisher_record()
    require(asset == published["audio_25k.zip"]
            and {row["ordinal"] for row in downloads["wavs"]} == set(sources),
            "Exact publisher WAV range receipt required")
    downloaded = {row["ordinal"]: row for row in downloads["wavs"]}
    with words.ALIGNMENTS.open("rb") as stream:
        archive_md5 = hashlib.file_digest(stream, lambda: hashlib.md5(usedforsecurity=False)).hexdigest()
    alignment_archive = published["alignments.zip"]
    require(alignment_archive["checksum"] == f"md5:{archive_md5}" and alignment_archive["size"] == words.ALIGNMENTS.stat().st_size,
            "Exact official GRID word-alignment archive required")
    records, groups = [], {}
    with ZipFile(words.ALIGNMENTS) as archive:
        word_groups = defaultdict(dict)
        for entry in archive.infolist():
            path = PurePosixPath(entry.filename)
            if len(path.parts) == 3 and path.parts[0] == "alignments" and path.suffix == ".align" and not path.name.startswith("._"):
                require(path.stem not in word_groups[path.parent.name], "Ambiguous publisher alignment")
                word_groups[path.parent.name][path.stem] = entry
        for ordinal, source in sources.items():
            row, slot = source["input"], source["source_slot"]
            if slot not in groups:
                groups[slot] = inventory(source, word_groups)
            label, matched, members = groups[slot]
            frozen_member = row["planned"]["member"]
            stem = Path(frozen_member["name"]).stem
            candidate = members[stem]
            require(candidate.filename == frozen_member["name"] and frozen_member["crc32"] == candidate.CRC,
                    "Selected original member differs from archive inventory")
            info = word_groups[matched][stem]
            alignment_bytes = archive.read(info)
            word_spans, total = words.publisher_words(alignment_bytes)
            require(total == 74500, "Publisher full word timeline differs")
            official = downloaded[ordinal]
            path = wavs / f"{ordinal:02}.wav"
            payload = path.read_bytes()
            require(len(payload) == official["bytes"] and sha(path) == official["sha256"]
                    and zlib.crc32(payload) == official["crc32"]
                    and PurePosixPath(official["member"]).stem == stem,
                    "Official WAV bytes differ from their publisher archive member")
            table, _, pcm, origin = exact_native(row)
            timing = project(word_spans, wave_pcm(io.BytesIO(payload), PUBLISHER_RATE), pcm, origin, table)
            records.append({"ordinal": ordinal, "group": slot, "literal_label": label == matched, "source_sha256": row["source_sha256"],
                            "alignment_sha256": audio.sha(alignment_bytes), "official_wave_sha256": official["sha256"], **timing})
            print("qualified", ordinal, flush=True)
    output.mkdir(mode=0o700, parents=True)
    write(output / "private-word-clocks.json", {"plan_sha256": plan_sha, "protocol": SPEC, "records": records})
    timings = [word for record in records for word in record["words"]]
    report = {"status": "qualified", "reserve_sources": len(records), "source_groups": len(groups),
              "literal_word_references": sum(row["literal_label"] for row in records),
              "full_inventory_remapped_references": sum(not row["literal_label"] for row in records),
              "minimum_complete_wave_correlation": min(row["whole_wave_correlation"] for row in records),
              "maximum_complete_wave_correlation": max(row["whole_wave_correlation"] for row in records),
              "minimum_individual_word_correlation": min(row["wave_correlation"] for row in timings),
              "maximum_local_alignment_residual_samples": max(abs(row["word_residual_samples"]) for row in timings),
              "distinct_measured_shifts_seconds": sorted({record["publisher_to_source_shift_seconds"] for record in records}, key=Fraction),
              "publisher_alignments_sha256": sha(words.ALIGNMENTS),
              "publisher_wave_ranges_sha256": sha(wavs / "receipt.json"),
              "public_transfer_bytes": sum(item["bytes"] for item in downloads["http_ranges"]),
              "native_pictures": len(records)*75, "canonical_pcm_samples": len(records)*47648,
              "mapped_word_intervals": len(timings), "independently_verified_phoneme_intervals": 0,
              "private_word_clocks_sha256": sha(output / "private-word-clocks.json")}
    write(output / "qualification.json", report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wavs", type=Path, help="Private CRC-verified publisher WAV selection")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    qualify(args.wavs, args.output)


if __name__ == "__main__":
    main()
