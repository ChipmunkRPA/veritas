"""Replay a frozen spatial audio comparison on bounded PhonemeDF release prefixes."""

from __future__ import annotations

import argparse
import gzip
import io
import re
import tarfile
import wave
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from scripts import audio_odss_matched_room as room

SCHEMA = "chipmunk.audio-phonemedf-matched-transfer/1"
SOURCE = "https://zenodo.org/records/19522979"
SOURCE_RECORD = 19522979
ARCHIVES = {
    "natural": ("real", 9420947621, "71371f453e3d56199f047b0ee16edac1"),
    "melotts": ("MeloTTS", 7703256782, "e12abaafe46a405e37ed9b24d1f9d76e"),
    "chatterbox": ("ChatterboxTTS", 7848724770, "ce8e6a0c2a1456ca201347032c906059"),
}
CHUNK = 8 * 1024**2
PREFIX = 96 * 1024**2
PROMPTS = 3
ODSS = room.ROOT / "docs/evaluation-panels/audio-odss-matched-room-2026-11-13.json"
ODSS_SHA = "b664e009c7c62dc890a4817358288d143874257e1d66c0fd250edafea9e41f4a"
METHOD = {
    "source": "Use complete archived WAV members and exact publisher utterance identifiers with unique normalized transcripts. Select three prompts per source reader by salted SHA-256 from utterances of 6 to 14 seconds in each version with a maximum duration ratio of 1.25. Use each origin's first six seconds, then score the fixed middle three seconds. The full prompt matches by construction. This cropped middle interval is not phoneme-aligned across origins.",
    "masking": "Reuse the pinned BBC room and ES2012b AMI annotated speech recording from ODSS. Reserve the eight ODSS assessment excerpts before selecting nonoverlapping new excerpts. Use the same new masker, location and common gain for all three origins per case. Alternate left and right. Measure room mean level at 0 and -5 dB. Use the identical background-only waveform and repeat room-only native scores.",
    "policy": "Freeze each operating point from ODSS development before new model inference. The selective route preserves mono flags and adds a flag only if native mono meets the ODSS development one-genuine-reader point and both cleaned views exceed their respective zero-genuine-reader points. The ODSS assessment remains retrospective. Report each generator and genuine/background counts and original utterance-reader groups independently.",
}


class ArchivePrefix(io.RawIOBase):
    def __init__(self, paths):
        self.paths = iter(paths)
        self.stream = None

    def readable(self):
        return True

    def readinto(self, target):
        while True:
            if self.stream is None:
                path = next(self.paths, None)
                if path is None:
                    return 0
                self.stream = path.open("rb")
            count = self.stream.readinto(target)
            if count:
                return count
            self.stream.close()
            self.stream = None

    def close(self):
        if self.stream is not None:
            self.stream.close()
        super().close()


def archive_members(paths):
    with io.BufferedReader(ArchivePrefix(paths)) as stream, gzip.GzipFile(fileobj=stream) as decoded:
        with tarfile.open(fileobj=decoded, mode="r|") as archive:
            try:
                for item in archive:
                    if item.isfile():
                        room.require(0 < item.size <= 2 * 1024**2, "Unexpected publisher member size")
                        body = archive.extractfile(item).read()
                        room.require(len(body) == item.size, "Unexpected incomplete member")
                        yield item.name, body
            except (EOFError, tarfile.ReadError) as failure:
                # The fixed compressed prefix ends within the next publisher member.
                room.require("end" in str(failure).casefold(), "Unexpected archive structure")


def prefix_paths(cache, origin):
    label, size, _ = ARCHIVES[origin]
    paths = []
    for offset in range(0, PREFIX, CHUNK):
        parent = cache / label
        path = parent / f"{offset}.range"
        header = (parent / f"{offset}.headers").read_text().splitlines()
        expected = f"content-range: bytes {offset}-{offset + CHUNK - 1}/{size}"
        room.require(expected in [line.lower() for line in header] and path.stat().st_size == CHUNK,
                     "Publisher range does not match its bounded request")
        paths.append(path)
    return paths


def pcm_bytes(body):
    with wave.open(io.BytesIO(body), "rb") as stream:
        count = stream.getnframes()
        room.require((stream.getnchannels(), stream.getsampwidth(), stream.getframerate(), stream.getcomptype()) ==
                     (1, 2, room.RATE, "NONE"), "Expected native mono 16-bit 16 kHz PCM")
        pcm = stream.readframes(count)
        room.require(len(pcm) == count * 2, "Publisher PCM length disagrees")
    return pcm


def rank(value):
    return room.digest(f"{SCHEMA}:{value}".encode())


def load_inputs(cache, record):
    archive = room.read(record)
    files = {row["key"]: (row["size"], row["checksum"]) for row in archive["files"]}
    room.require(archive["id"] == SOURCE_RECORD and archive["metadata"]["license"]["id"] == "cc-by-4.0",
                 "Unexpected publisher release or terms")
    room.require(all(files[f"PhonemeDF_{label}.tar.gz"] == (length, f"md5:{md5}")
                     for label, length, md5 in ARCHIVES.values()), "Publisher archive commitments disagree")
    by_origin, chunk_pins, transcripts = {}, {}, {}
    for origin, (flavor, _, _) in ARCHIVES.items():
        paths = prefix_paths(cache, origin)
        chunk_pins[origin] = [room.file_digest(path) for path in paths]
        members = {}
        for path, body in archive_members(paths):
            if origin == "natural" and re.fullmatch(r"real/transcripts/\d+-\d+-\d+\.txt", path):
                key = Path(path).stem
                text = " ".join(re.findall(r"\w+", body.decode("utf-8").casefold()))
                room.require(key not in transcripts and bool(text), "Unexpected duplicate or empty transcript")
                transcripts[key] = text
                continue
            folder = "real/wav" if origin == "natural" else flavor
            match = re.fullmatch(rf"{folder}/(\d+-\d+-\d+)\.wav", path)
            if match:
                key, = match.groups()
                room.require(key not in members, "Duplicate origin key in publisher prefix")
                pcm = pcm_bytes(body)
                samples = len(pcm) // 2
                members[key] = (samples, room.digest(body), room.digest(pcm), pcm[:2 * room.SAMPLES])
        by_origin[origin] = members
    matched = set.intersection(*(set(value) for value in by_origin.values()))
    grouped, common = defaultdict(list), Counter()
    for key in matched:
        speaker, _, _ = key.split("-")
        sizes = [members[key][0] for members in by_origin.values()]
        common[speaker] += 1
        if min(sizes) >= room.SAMPLES and max(sizes) <= 14 * room.RATE and max(sizes) * 4 <= min(sizes) * 5:
            room.require(key in transcripts, "Matched prompt transcript missing")
            grouped[speaker].append(key)
    room.require(len(transcripts) == 28539 and len(common) == 4 and min(map(len, grouped.values())) >= PROMPTS,
                 f"Bounded prefix did not provide the complete planned source roster: {common}")
    roster, dry, used_text, used_pcm = [], {}, set(), set()
    for index, speaker in enumerate(sorted(common, key=rank)):
        group = f"source-reader-{index:02d}"
        selected = 0
        for key in sorted(grouped[speaker], key=rank):
            text = transcripts[key]
            if text in used_text:
                continue
            case = f"{group}-prompt-{selected}"
            bindings = {}
            for origin, members in by_origin.items():
                length, wav_hash, pcm_hash, cropped = members[key]
                window_hash = room.digest(cropped)
                room.require(window_hash not in used_pcm, "Aliased input waveform")
                used_pcm.add(window_hash)
                dry[case, origin] = np.frombuffer(cropped, dtype="<i2")
                bindings[origin] = {"samples": length, "wav_sha256": wav_hash,
                                    "full_pcm_sha256": pcm_hash, "first_six_pcm_sha256": window_hash}
            used_text.add(text)
            roster.append({"case": case, "group": group, "role": "assessment", "source_id": key,
                           "prompt_sha256": room.digest(text.encode()), "sources": bindings})
            selected += 1
            if selected == PROMPTS:
                break
        room.require(selected == PROMPTS, "Required source group is incomplete")
    return roster, dry, {
        "release_record_sha256": room.file_digest(record), "archive_count": {key: len(value) for key, value in by_origin.items()},
        "common_counts": sorted(common.values()), "eligible_counts": sorted(map(len, grouped.values())), "chunks_sha256": chunk_pins,
    }


def deliveries(item, dry, masker, room_inputs):
    case = item["case"]
    position = "left" if int(case.rsplit("-", 1)[1]) % 2 == 0 else "right"
    targets = {}
    for origin in ARCHIVES:
        signal = room.convolve(dry[case, origin], room_inputs["front"])
        targets[origin] = signal / room.rms(signal.mean(axis=1))
    noise = room.convolve(masker, room_inputs[position])
    noise /= room.rms(noise.mean(axis=1))
    maximum = max(float(np.max(abs(target) + abs(noise) * 10 ** (-min(room.LEVELS) / 20))) for target in targets.values())
    scale = 24000 / maximum
    residuals = {level: np.rint(noise * scale * 10 ** (-level / 20)).astype(np.int32) for level in room.LEVELS}
    for origin, target in targets.items():
        signal = np.rint(target * scale).astype(np.int32)
        yield origin, None, signal.astype("<i2"), signal, None, position
        for level, noise_pcm in residuals.items():
            measured = 20 * np.log10(room.rms(signal.mean(axis=1)) / room.rms(noise_pcm.mean(axis=1)))
            room.require(abs(measured - level) <= .02 and np.max(abs(signal + noise_pcm)) <= 24001, "Unexpected level or peak")
            yield origin, level, (signal + noise_pcm).astype("<i2"), signal, round(float(measured), 6), position
    for level, noise_pcm in residuals.items():
        yield "background", level, noise_pcm.astype("<i2"), None, None, position


def prior():
    room.require(room.file_digest(ODSS) == ODSS_SHA, "Published prior panel commitment differs")
    data = room.read(ODSS)
    points = room.fit(data["development"])
    room.require(points == data["operating_points"]["points"], "Prior development fit differs")
    return points


def prepare(args):
    points = prior()
    sources, dry, intake = load_inputs(args.cache, args.record)
    rooms, assets = room.room_impulses(args.surrey, args.bbc)
    old = [{"role": "assessment", "case": f"reader-{index:02d}-prompt-{prompt}"}
           for index in range(4, 8) for prompt in range(2)]
    reserve = [{"role": "development", "case": "unused-development"}] + old + sources
    positions, background = room.backgrounds(args.ami, reserve)
    reserved = [positions[item["case"]]["start_sample"] for item in old]
    received, rows = {}, []
    for source in sources:
        case = source["case"]
        start = positions[case]["start_sample"]
        room.require(all(abs(start - previous) >= room.SAMPLES for previous in reserved),
                     "New background excerpt overlaps an ODSS assessment excerpt")
        for origin, level, stereo, target, measured, position in deliveries(source, dry, background[case], rooms["assessment"]):
            key = f"{case}:{origin}:{'room' if level is None else level}"
            views, delta, gains = room.wave_views(stereo, target, level)
            a, b = room.WINDOW
            bindings = {name: None if pcm is None else room.digest(pcm[a:b].tobytes()) for name, pcm in views.items()}
            received[key] = views
            rows.append({"id": key, "case": case, "group": source["group"], "role": "assessment", "origin": origin,
                         "level_db": level, "measured_snr_db": measured, "noise_position": position, "window_hashes": bindings,
                         "input_sha256": room.digest(stereo.tobytes()), "acoustic_gain_db": gains, "fft_max_lsb": delta})
    room.require(len({row["input_sha256"] for row in rows}) == len(rows), "Unexpected rendered alias")
    lock = {"schema": SCHEMA, "method": METHOD, "publisher": SOURCE, "archive_identities": ARCHIVES, "intake": intake,
            "speech_assets": room.models.ASSETS, "room_assets": assets["assessment"], "background_source": room.AMI["assessment"],
            "window_samples": list(room.WINDOW), "sample_rate": room.RATE, "levels_db": list(room.LEVELS),
            "odss_receipt_sha256": ODSS_SHA, "points": points, "code_sha256": room.file_digest(Path(__file__)),
            "room_code_sha256": room.file_digest(Path(room.__file__)), "wpe_code_sha256": room.file_digest(Path(room.wpe.__file__)),
            "source_roster_sha256": room.digest(room.json_bytes(sources)), "background_roster_sha256": room.digest(room.json_bytes(positions)),
            "delivery_roster_sha256": room.digest(room.json_bytes(rows))}
    return lock, rows, received


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "score", "replay"))
    for arg in ("cache", "record", "run", "surrey", "bbc", "ami"):
        parser.add_argument(f"--{arg}", type=Path, required=True)
    parser.add_argument("--weights", type=Path, default=room.ROOT / "models/weights/audio-upgrade")
    args = parser.parse_args()
    lock, rows, received = prepare(args)
    freeze = args.run / "freeze.json"
    if args.phase == "freeze":
        args.run.mkdir(mode=0o700)
        room.write(freeze, lock)
        room.write(args.run / "delivery-roster.json", rows)
        print(f"Froze {len(rows)} deliveries from four previously untested source readers before inference")
        return
    room.require(freeze.read_bytes() == room.json_bytes(lock) and
                 (args.run / "delivery-roster.json").read_bytes() == room.json_bytes(rows),
                 "Fresh roster or operating points changed after freeze")
    if args.phase == "replay":
        print(f"Verified {len(rows)} frozen deliveries and original view hashes")
        return
    path = args.run / "public.json"
    room.require(not path.exists(), "Fresh assessment has already been scored")
    result = room.score(rows, received, "assessment", args.weights)
    room.write(path, {"freeze": lock, "assessment": result})


if __name__ == "__main__":
    main()
