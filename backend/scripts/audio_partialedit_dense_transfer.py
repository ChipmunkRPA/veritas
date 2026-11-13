# SPDX-License-Identifier: Apache-2.0
"""Compare speech-window schedules on paired PartialEdit and VCTK recordings."""

from __future__ import annotations

import argparse
import csv
import io
import re
import struct
import subprocess
import tarfile
import zipfile
import zlib
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
from pathlib import Path

from scripts import audio_vocal_event_mixtures as common
from scripts import review_audio_reduced_background_localization as temporal
from scripts.audio_reduced_background_localization import SplitArchive, assets_for_scores

RATE, WINDOW = temporal.RATE, temporal.WINDOW
SCHEMA = "chipmunk.audio-partialedit-dense-transfer/1"
CALIBRATION = temporal.METHOD["primary_calibration_sha256"]
PARTIAL_EDIT = "https://zenodo.org/records/18829689"
VCTK = "https://datashare.ed.ac.uk/server/api/core/bitstreams/535f4286-e54c-4038-838c-a02285e32cb2/content"
VCTK_ETAG = '"8a6ba2946b36fcbef0212cad601f4bfa"'
VCTK_DIRECTORY_START, VCTK_SIZE = 11_734_391_133, 11_747_302_977
FAMILIES = ("E1", "E2")
SCENES = ("clean", "construction", "leicester")
ARMS = ("current", "dense", "combined")
MEMBER = re.compile(r"(E[12])/(p\d{3})/(p\d{3}_\d{3})_edited_partial_16k\.wav")
INPUT_PINS = {
    "labels.csv": "adeecb0a7dd39a982223c07970e02e41ca5395ea3d484fcc6ce03174ebdf6cbc",
    "speakers.zip": "8be8f5b907163bcd886c4e487f173347a7972080d4e77bfed14ec66298edf086",
    "vctk-directory": "ff07294e20e89207c2620a5fb0ce8a404478268b80788ce29c6c9837ce2fafac",
    "E1.prefix": "39e84aa4fddb39585368f787788a8ea5f6ce8719b97213fdd24bc6df0c9a8466",
    "E1.more": "8177c5089fc771d5731616d9e6e5b35ad4d1ea94a5dba000441ea136da552889",
    "E2.prefix": "6ecfb5790a6fc3b326a957fed4820e63726b5b0aea26458085ff0f9cd03fde47",
    "E2.more": "8a5c365419981391420ab32558ca7d09ec27ac0d90f7e6389dce5ce868191465",
}
BACKGROUNDS = {
    "construction": ("7b69cda9671d635bd20ed17f1f7db728fbca3ba6a67cc7ce2512777a8e02534a", (5, 30, 55, 80), -5),
    "leicester": ("42d2f7a9989748fe3f190b39665e7519bc205b336b3237c0788e2f36c6bab707", (4, 21, 38, 55), 5),
}
METHOD = {
    "source": "Every common eligible utterance from the two frozen publisher prefixes and official eval speaker list.",
    "eligibility": "Mono 16 kHz WAV at 6–14 s, at least one edit 0.6–2.4 s, label duration tolerance 20 ms. Canonical PCM16.",
    "control": "The same utterance's official VCTK microphone 1 recording, decoded to 16 kHz at its own full length.",
    "comparison": "Preserve incumbent locations. Add only the shared 1.5 s of two adjacent AI-leaning 3 s responses.",
    "useful": "Cover half a known edit with at least 0.30 temporal intersection over union.",
    "min_speakers_for_confirmation": 8,
    "deployment": "Transfer pilot only. No automatic serving change or fit to this panel.",
}
require, sha, digest, save = common.require, common.sha, common.digest, common.save


class OffsetDirectory(io.BytesIO):
    """Expose just the retained tail at its real offset in the large ZIP."""

    def tell(self):
        return super().tell() + VCTK_DIRECTORY_START

    def seek(self, offset, whence=0):
        if whence == 0:
            offset -= VCTK_DIRECTORY_START
        return super().seek(offset, whence) + VCTK_DIRECTORY_START


def native_wav(body):
    import numpy as np

    from scripts.audio_gtcrn_traffic import pcm

    require(body[:4] == b"RIFF" and body[8:12] == b"WAVE", "Publisher member is not a RIFF WAV")
    chunks, pos = {}, 12
    while pos + 8 <= len(body):
        name, length = struct.unpack_from("<4sI", body, pos)
        start = pos + 8
        end = start + length
        require(end <= len(body) and name not in chunks, "Malformed or duplicate WAV chunk")
        chunks[name] = body[start:end]
        pos = end + length % 2
    require(pos == len(body) and len(chunks.get(b"fmt ", b"")) >= 16 and b"data" in chunks, "Incomplete WAV layout")
    kind, channels, rate, byte_rate, block_align, bits = struct.unpack_from("<HHIIHH", chunks[b"fmt "])
    require(channels == 1 and rate == RATE and (kind, bits) in {(1, 16), (3, 32)}
            and block_align == bits // 8 and byte_rate == RATE * block_align
            and len(chunks[b"data"]) % block_align == 0, "Unexpected edited source geometry")
    raw = chunks[b"data"]
    if kind == 1:
        return raw
    values = np.frombuffer(raw, dtype="<f4")
    require(np.isfinite(values).all(), "Nonfinite publisher audio")
    return pcm(values)


def edited_sources(sources):
    with zipfile.ZipFile(sources / "speakers.zip") as archive:
        speakers = set(archive.read("eval/eval.spk").decode().split())
    require(len(speakers) == 70, "Publisher evaluation roster differs")
    with (sources / "labels.csv").open(newline="") as stream:
        labels = {name: list(map(Decimal, values)) for name, *values in csv.reader(stream)}
    require(len(labels) == 85_007, "Publisher label count differs")
    edits = {}
    complete = Counter()
    for family in FAMILIES:
        with (sources / f"{family}.prefix").open("rb") as head, (sources / f"{family}.more").open("rb") as tail:
            try:
                with tarfile.open(fileobj=SplitArchive(head, tail), mode="r|gz") as archive:
                    for member in archive:
                        if not member.isfile():
                            continue
                        match = MEMBER.fullmatch(member.name)
                        require(match and 44 < member.size < 2**20 and member.name in labels, "Unexpected publisher member")
                        _, speaker, utterance = match.groups()
                        require(utterance.startswith(f"{speaker}_") and match[1] == family, "Wrong publisher source")
                        with archive.extractfile(member) as stream:
                            body = stream.read()
                        require(len(body) == member.size, "Incomplete publisher member")
                        complete[family] += 1
                        bounds = labels[member.name]
                        times = list(zip(bounds[:-1:2], bounds[1:-1:2], strict=True))
                        if (speaker not in speakers or not 6 <= bounds[-1] <= 14
                                or not any(Decimal(".6") <= b - a <= Decimal("2.4") for a, b in times)):
                            continue
                        raw = native_wav(body)
                        count = len(raw) // 2
                        spans = [[round(a * RATE), round(b * RATE)] for a, b in times]
                        require(len(bounds) in (3, 5) and abs(count - round(bounds[-1] * RATE)) <= 320
                                and all(0 <= a < b <= count for a, b in spans)
                                and spans == temporal.intervals_union(spans), "Edited truth clock differs")
                        if 6 * RATE <= count <= 14 * RATE:
                            require((utterance, family) not in edits, "Repeated edited source")
                            edits[utterance, family] = (raw, spans, digest(body))
            except (tarfile.ReadError, EOFError) as error:
                require(isinstance(error, EOFError) or "end of data" in str(error), "Unexpected bounded archive error")
    return edits, dict(complete)


def original_spans(sources, utterances):
    with zipfile.ZipFile(OffsetDirectory((sources / "vctk-directory").read_bytes())) as archive:
        for utterance in utterances:
            speaker = utterance.split("_")[0]
            member = archive.getinfo(f"wav48_silence_trimmed/{speaker}/{utterance}_mic1.flac")
            require(member.compress_type == zipfile.ZIP_DEFLATED and member.flag_bits == 0
                    and 0 < member.file_size < 2**20 and 0 < member.compress_size < 2**20, "Unexpected genuine ZIP member")
            yield utterance, member


def original(sources, utterance, member, download):
    segment = sources / f"{utterance}.zipseg"
    headers = segment.with_suffix(".headers")
    first = member.header_offset
    last = first + member.compress_size + 4095
    if download and (not segment.is_file() or not headers.is_file()):
        subprocess.run(["curl", "-fSs", "--retry", "2", "--max-time", "40", "--range", f"{first}-{last}",
                        "--max-filesize", str(last - first + 1), "--dump-header", str(headers),
                        f"{VCTK}?veritas-range={first}", "-o", str(segment)], check=True, timeout=130)
    response = headers.read_text().lower().splitlines()
    require(any(line.startswith(("http/1.1 206 ", "http/2 206")) for line in response)
            and f"content-range: bytes {first}-{last}/{VCTK_SIZE}" in response
            and f"etag: {VCTK_ETAG}" in response, "Official ZIP range response differs")
    raw = segment.read_bytes()
    require(len(raw) == last - first + 1, "Incomplete official ZIP range")
    fields = struct.unpack("<4s5H3I2H", raw[:30])
    signature, _, flags, method, _, _, crc, compressed, size, name_len, extra_len = fields
    begin = 30 + name_len + extra_len
    require(signature == b"PK\x03\x04" and flags == member.flag_bits and method == member.compress_type
            and (crc, compressed, size) == (member.CRC, member.compress_size, member.file_size)
            and raw[30:30 + name_len].decode("ascii") == member.filename, "ZIP local header does not bind to official member")
    body = zlib.decompress(raw[begin:begin + compressed], -15)
    require(len(body) == size and zlib.crc32(body) == crc and body.startswith(b"fLaC"), "Genuine member failed ZIP validation")
    return body, [first, last]


def eligible_sources(sources):
    require({name: sha(sources / name) for name in INPUT_PINS} == INPUT_PINS, "Frozen publisher inputs differ")
    edits, complete = edited_sources(sources)
    utterances = sorted(set.intersection(*(set(key[0] for key in edits if key[1] == family) for family in FAMILIES)))
    require(bool(utterances), "No matched edited utterances")
    return edits, list(original_spans(sources, utterances)), complete


def acquire(sources):
    _, members, _ = eligible_sources(sources)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda pair: original(sources, *pair, download=True), members))
    print(f"Verified {len(members)} exact matched original ZIP members", flush=True)


def prepare(run, sources, backgrounds):
    from scripts.audio_gtcrn_traffic import float_pcm, pcm
    from scripts.audio_noise_views_transfer import match_views

    edits, members, complete = eligible_sources(sources)
    run.mkdir(mode=0o700)
    (run / "pcm").mkdir(mode=0o700)
    (run / "controls").mkdir(mode=0o700)
    noise = {}
    for scene, (pin, _, _) in BACKGROUNDS.items():
        path = backgrounds / f"{scene}.ogg"
        require(sha(path) == pin, "Background source differs")
        noise[scene] = float_pcm(common.decode(path))
    rows, excluded = [], []
    for index, (utterance, member) in enumerate(members):
        body, segment = original(sources, utterance, member, download=False)
        file = run / "controls" / f"{utterance}.flac"
        file.write_bytes(body)
        control = common.decode(file)
        if not 6 * RATE <= len(control) // 2 <= 14 * RATE:
            excluded.append({"utterance": utterance, "samples": len(control) // 2})
            continue
        counterparts = {"genuine": (control, [], digest(body)), **{family: edits[utterance, family] for family in FAMILIES}}
        for family, (source, spans, source_sha256) in counterparts.items():
            signal = float_pcm(source)
            for scene in SCENES:
                offset = None
                raw = source
                if scene != "clean":
                    _, offsets, snr = BACKGROUNDS[scene]
                    offset = offsets[index % len(offsets)] * RATE
                    track = noise[scene][offset:offset + len(signal)]
                    raw = pcm(match_views(signal, {scene: track}, snr)[scene])
                row = {"id": len(rows), "utterance": utterance, "family": family, "scene": scene, "offset": offset,
                       "truth": spans, "samples": len(raw) // 2, "pcm_sha256": digest(raw),
                       "source_sha256": source_sha256, "vctk_range": segment if family == "genuine" else None}
                (run / "pcm" / f"{row['id']:03}.pcm").write_bytes(raw)
                rows.append(row)
    require(bool(rows) and len({row["pcm_sha256"] for row in rows}) == len(rows), "Missing or repeated delivery")
    save(run / "roster.json", rows)
    lock = {"schema": SCHEMA, "method": METHOD, "sources": INPUT_PINS, "backgrounds": BACKGROUNDS,
            "runner_sha256": sha(Path(__file__)), "roster_sha256": sha(run / "roster.json"),
            "voice_assets": common.assets(), "calibration_sha256": CALIBRATION, "excluded_original_clocks": excluded,
            "complete_archive_members": complete, "known_model_responses_at_freeze": 0,
            "ffmpeg": subprocess.check_output(["ffmpeg", "-version"], text=True, timeout=5).splitlines()[0]}
    save(run / "freeze.json", lock)
    print(f"Froze {len(rows)} exact deliveries: {sha(run / 'freeze.json')}", flush=True)


def score(run):
    from app.detectors.audio import calibration
    from app.detectors.audio.short_voice_review import scan as short_scan
    from app.engine.boundary import Deadline
    from scripts.audio_gtcrn_traffic import float_pcm
    from scripts.audio_radar_transfer import adapter_result

    lock = common.read(run / "freeze.json")
    require(lock["runner_sha256"] == sha(Path(__file__)) and lock["roster_sha256"] == sha(run / "roster.json")
            and lock["calibration_sha256"] == calibration.SHA256, "Frozen scoring inputs differ")
    voice, vad = assets_for_scores(lock)
    rows, checks = common.read(run / "roster.json"), []
    for row in rows:
        raw = (run / "pcm" / f"{row['id']:03}.pcm").read_bytes()
        require(len(raw) == 2 * row["samples"] and digest(raw) == row["pcm_sha256"], "Frozen PCM differs")
        signal = float_pcm(raw)
        sparse, dense = temporal.schedules(len(signal))
        measured = []
        for a, b in sorted(set(map(tuple, sparse + dense))):
            result = adapter_result(signal[a:b], voice, vad)
            require((result["scope"] and result["windows"] == [[0, WINDOW]]) or result["decision"] == "unassessed",
                    "Unexpected original speech response")
            measured.append({"span": [a, b], "direction": result["decision"], "pcm_sha256": digest(raw[2*a:2*b])})
        short, short_primary = None, None
        if len(signal) == 6 * RATE:
            short_primary = adapter_result(signal, voice, vad, include_status=True)["status"]
            if short_primary == "ok":
                short = short_scan(signal, vad.spans(signal), voice, Deadline(), codec="pcm_s16le")["selected"]
        checks.append({"id": row["id"], "windows": measured, "short": short, "short_primary": short_primary})
        if len(checks) % 12 == 0 or len(checks) == len(rows):
            print(f"Scored {len(checks)}/{len(rows)}", flush=True)
    save(run / "scores.json", {"freeze_sha256": sha(run / "freeze.json"), "checks": checks})


def recount(rows, checks):
    groups = defaultdict(Counter)
    gains, alarms = defaultdict(set), defaultdict(set)
    for row, record in zip(rows, checks, strict=True):
        sparse, dense = temporal.schedules(row["samples"])
        windows = record["windows"]
        require(record["id"] == row["id"] and [tuple(item["span"]) for item in windows] == sorted(set(map(tuple, sparse + dense)))
                and all(item["direction"] in temporal.RESCUE | {"ai", "original"} for item in windows), "Saved schedule differs")
        directions = {tuple(item["span"]): item["direction"] for item in windows}
        positive = {span for span, direction in directions.items() if direction == "ai"}
        current = [span for span in sparse if tuple(span) in positive]
        if len(current) < 2 or any(directions[tuple(span)] == "original" for span in sparse):
            current = []
        if short := record["short"]:
            a, b = short["start_sample"], short["end_sample"]
            require(row["samples"] == 6 * RATE and a in range(0, 5 * RATE, RATE) and b == a + 2 * RATE,
                    "Short review span differs")
            current.append([a, b])
        proposed = temporal.intersections(dense, positive)
        scene, family, truth = row["scene"], row["family"], row["truth"]
        cell = groups[f"{scene}/{family}"]
        cell["files"] += 1
        cell["assessed"] += any(value != "unassessed" for value in directions.values())
        cell["genuine_samples"] += row["samples"] - sum(b - a for a, b in truth)
        located = {}
        for arm, ranges in zip(ARMS, (current, proposed, current + proposed), strict=True):
            union = temporal.intervals_union(ranges)
            cell[f"{arm}_flagged"] += bool(union)
            located[arm] = temporal.useful(ranges, truth)
            cell[f"{arm}_useful"] += located[arm]
            cell[f"{arm}_genuine_samples_flagged"] += sum(b - a for a, b in union) - sum(
                temporal.overlap(span, edit) for span in union for edit in truth)
        if located["combined"] and not located["current"]:
            gains[scene].add(row["utterance"])
        if family == "genuine" and proposed and not current:
            alarms[scene].add(row["utterance"])
    return {"cells": groups, "added_useful_utterances": {scene: len(gains[scene]) for scene in SCENES},
            "added_genuine_utterances": {scene: len(alarms[scene]) for scene in SCENES},
            "publisher_speakers": len({row["utterance"].split("_")[0] for row in rows}), "admitted_to_serving": False}


def report(run):
    lock, rows, measured = (common.read(run / name) for name in ("freeze.json", "roster.json", "scores.json"))
    require(measured["freeze_sha256"] == sha(run / "freeze.json") and lock["roster_sha256"] == sha(run / "roster.json"),
            "Saved score inputs differ")
    result = recount(rows, measured["checks"])
    save(run / "receipt.json", {"lock": lock, "roster": rows, "checks": measured["checks"],
                                "checks_sha256": digest(common.encoded(measured["checks"])), "result": result})
    print(common.encoded(result).decode())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("acquire", "prepare", "score", "report"))
    parser.add_argument("run", type=Path)
    parser.add_argument("--sources", type=Path)
    parser.add_argument("--backgrounds", type=Path)
    args = parser.parse_args()
    if args.phase == "acquire":
        acquire(args.sources)
    elif args.phase == "prepare":
        prepare(args.run, args.sources, args.backgrounds)
    else:
        {"score": score, "report": report}[args.phase](args.run)


if __name__ == "__main__":
    main()
