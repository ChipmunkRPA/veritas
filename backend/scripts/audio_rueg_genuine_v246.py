"""Bind 24 original Turkish oral recordings through a pinned RUEG ZIP directory."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import struct
import time
import wave
import zlib
from collections import Counter
from dataclasses import asdict, dataclass
from http.client import HTTPException
from pathlib import Path, PurePosixPath
from urllib.error import HTTPError, URLError
from urllib.request import Request, build_opener

import numpy as np

from app.detectors.audio.decode import decode_audio, read_wav, source_ranges, wav_bytes
from scripts import audio_voxforge_tr_genuine_v199 as crop
from scripts.audio_commons_background_v234 import AGENT, NoRedirect, digest, read, require, save
from scripts.video_appearance_native_v108 import ARCHIVE, BASE, ROOT

NAME = "audio-rueg-genuine-v246"
TRIAL = BASE / NAME
CACHE = ARCHIVE / NAME
HISTORY = BASE / "audio-history-v58/catalogue.json"
URL = "https://zenodo.org/records/11234583/files/rueg-tr-1.0-wav.zip"
LICENSE_RECORD = "https://api.datacite.org/dois/10.5281/zenodo.11234583"
ARCHIVE_BYTES = 3_038_375_115
PUBLISHER_ARCHIVE_MD5 = "77dc415a6971de0a4ba7651a78f4fde0"
DIRECTORY_START, DIRECTORY_SIZE = 3_038_331_448, 43_645
DIRECTORY_SHA256 = "26c784410cf41501d67308166052cbb0829b614cc810bc166246e7b7ef7d3b60"
ENTRIES, FORMAL_GROUPS, SLOTS = 392, 67, 24
EXPECTED_COMPRESSED, EXPECTED_EXPANDED = 157_768_001, 200_205_096
FORMAL = re.compile(r"(?P<group>TUmo[0-9]{2,4}[MF]T)_fsT\.wav")
RANK_PREFIX = "chipmunk.audio-rueg-genuine/246:"
END = struct.Struct("<4s4H2LH")
CENTRAL = struct.Struct("<4s6H3L5H2L")
LOCAL = struct.Struct("<4s5H3L2H")
MAX_MEMBER_COMPRESSED, MAX_MEMBER_EXPANDED = 18_000_000, 24_000_000
MAX_NAME_BYTES, MAX_EXTRA_BYTES = 512, 512
MAX_MEDIA_REQUESTS, MAX_MEDIA_BYTES = 2 * SLOTS, 160_000_000
MIN_REQUEST_INTERVAL, NETWORK_TIMEOUT, NETWORK_DEADLINE = 3, 20, 900
MAX_NATIVE_SECONDS = 300
DISK_RESERVE = 3 * 1024**3


@dataclass(frozen=True)
class Member:
    participant: str
    name: str
    compressed: int
    expanded: int
    crc32: int
    local_offset: int


def parse_directory(body, entries, directory_start):
    members, offset, count = [], 0, 0
    while offset < len(body):
        require(offset + CENTRAL.size <= len(body), "Truncated ZIP central entry")
        (
            sig,
            _,
            _,
            flags,
            method,
            _,
            _,
            crc,
            compressed,
            expanded,
            names,
            extra,
            comment,
            disk,
            _,
            _,
            start,
        ) = CENTRAL.unpack_from(body, offset)
        position = offset + CENTRAL.size
        offset = position + names + extra + comment
        require(sig == b"PK\x01\x02" and offset <= len(body), "ZIP directory entry differs")
        count += 1
        name = body[position : position + names].decode("utf-8")
        path = PurePosixPath(name)
        match = FORMAL.fullmatch(path.name)
        if match is None:
            continue
        require(
            not path.is_absolute()
            and all(part not in {".", ".."} for part in name.split("/"))
            and "\\" not in name
            and names <= MAX_NAME_BYTES
            and flags == disk == 0
            and method == 8
            and 0 < compressed <= MAX_MEMBER_COMPRESSED
            and 0 < expanded <= MAX_MEMBER_EXPANDED
            and 0 <= start < start + LOCAL.size + names + compressed <= directory_start,
            "Formal spoken member is outside the frozen ZIP bounds",
        )
        members.append(Member(match["group"], name, compressed, expanded, crc, start))
    require(count == entries and offset == len(body), "ZIP directory entry count differs")
    require(
        len({m.participant for m in members}) == len(members), "Expected one formal member per participant"
    )
    require(len({m.local_offset for m in members}) == len(members), "Duplicate member offset")
    return members


def select(footer, directory):
    expected = (b"PK\x05\x06", 0, 0, ENTRIES, ENTRIES, DIRECTORY_SIZE, DIRECTORY_START, 0)
    require(
        len(footer) == END.size and END.unpack(footer) == expected, "Pinned single-volume ZIP footer differs"
    )
    require(
        len(directory) == DIRECTORY_SIZE and hashlib.sha256(directory).hexdigest() == DIRECTORY_SHA256,
        "Pinned publisher directory differs",
    )
    members = parse_directory(directory, ENTRIES, DIRECTORY_START)
    require(len(members) == FORMAL_GROUPS, "Published monolingual formal groups differ")
    ordered = sorted(members, key=lambda m: hashlib.sha256(f"{RANK_PREFIX}{m.participant}".encode()).digest())
    selected = ordered[:SLOTS]
    require(
        sum(m.compressed for m in selected) == EXPECTED_COMPRESSED
        and sum(m.expanded for m in selected) == EXPECTED_EXPANDED,
        "Frozen group selection byte totals differ",
    )
    return selected


class Rejected(ValueError):
    pass


class RangeReader:
    def __init__(self, max_requests, max_bytes, opener=None):
        self.max_requests, self.max_bytes = max_requests, max_bytes
        self.opener = opener or build_opener(NoRedirect)
        self.requests = self.requested_bytes = 0
        self.last_start = float("-inf")
        self.deadline = time.monotonic() + NETWORK_DEADLINE

    def get(self, start, size):
        require(
            0 <= start < start + size <= ARCHIVE_BYTES
            and self.requests < self.max_requests
            and self.requested_bytes + size <= self.max_bytes,
            "Frozen request or archive bound exceeded",
        )
        time.sleep(max(0, self.last_start + MIN_REQUEST_INTERVAL - time.monotonic()))
        self.last_start = time.monotonic()
        if self.last_start >= self.deadline:
            raise Rejected("transfer_deadline")
        self.requests += 1
        self.requested_bytes += size
        end = start + size - 1
        request = Request(
            URL, headers={"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity", "User-Agent": AGENT}
        )
        with self.opener.open(request, timeout=NETWORK_TIMEOUT) as response:
            if (
                response.status != 206
                or response.geturl() != URL
                or response.headers.get("Content-Range") != f"bytes {start}-{end}/{ARCHIVE_BYTES}"
                or response.headers.get("Content-Length") != str(size)
                or response.headers.get("Content-Encoding", "identity") != "identity"
            ):
                raise Rejected("unexpected_partial_response")
            raw = bytearray()
            while len(raw) <= size:
                if time.monotonic() >= self.deadline:
                    raise Rejected("transfer_deadline")
                chunk = response.read1(min(256 * 1024, size + 1 - len(raw)))
                if not chunk:
                    break
                raw.extend(chunk)
        if len(raw) != size or time.monotonic() >= self.deadline:
            raise Rejected("partial_body_length_or_deadline")
        return bytes(raw)


def extract_member(member, transport):
    header = transport.get(member.local_offset, LOCAL.size)
    sig, _, flags, method, _, _, crc, compressed, expanded, names, extra = LOCAL.unpack(header)
    name = member.name.encode("utf-8")
    start = member.local_offset + LOCAL.size
    if (
        sig != b"PK\x03\x04"
        or flags != 0
        or method != 8
        or (crc, compressed, expanded) != (member.crc32, member.compressed, member.expanded)
        or names != len(name)
        or not 0 <= extra <= MAX_EXTRA_BYTES
        or start + names + extra + compressed > DIRECTORY_START
    ):
        raise Rejected("local_header_mismatch")
    body = transport.get(start, names + extra + compressed)
    if body[:names] != name:
        raise Rejected("local_member_name_mismatch")
    try:
        decoder = zlib.decompressobj(-zlib.MAX_WBITS)
        original = decoder.decompress(body[names + extra :], expanded + 1)
    except zlib.error as exc:
        raise Rejected("bounded_deflate_failed") from exc
    if (
        len(original) != expanded
        or not decoder.eof
        or decoder.unused_data
        or decoder.unconsumed_tail
        or zlib.crc32(original) != crc
    ):
        raise Rejected("member_length_or_crc_mismatch")
    return original


def lock_context():
    code = (
        Path(__file__),
        ROOT / "backend/tests/test_audio_rueg_genuine_v246.py",
        ROOT / "docs/audio-rueg-genuine-v246-method.md",
        Path(crop.__file__),
        ROOT / "backend/scripts/audio_commons_background_v234.py",
        ROOT / "backend/app/detectors/audio/decode.py",
        ROOT / "backend/app/detectors/video/decode.py",
    )
    decoders = {name: shutil.which(name) for name in ("ffmpeg", "ffprobe")}
    require(all(decoders.values()), "Production decoder binaries required")
    return {
        "schema": "chipmunk.audio-rueg-genuine-freeze/246",
        "publisher_archive": URL,
        "publisher_rights_record": LICENSE_RECORD,
        "license": "CC0-1.0",
        "archive_bytes": ARCHIVE_BYTES,
        "publisher_full_archive_md5_unverified": PUBLISHER_ARCHIVE_MD5,
        "full_archive_verified": False,
        "directory_sha256": DIRECTORY_SHA256,
        "directory_start": DIRECTORY_START,
        "directory_bytes": DIRECTORY_SIZE,
        "directory_entries": ENTRIES,
        "available_formal_participants": FORMAL_GROUPS,
        "planned_participants": SLOTS,
        "selected_register": "formal spoken Turkish",
        "selected_speaker_scope": "monolingual participants in Turkey",
        "age_metadata": "not available in verified audio directory",
        "recording_role": "elicited oral speech",
        "planned_member_compressed_bytes": EXPECTED_COMPRESSED,
        "planned_member_expanded_bytes": EXPECTED_EXPANDED,
        "metadata_request_cap": 2,
        "media_request_cap": MAX_MEDIA_REQUESTS,
        "media_request_bytes_cap": MAX_MEDIA_BYTES,
        "minimum_request_interval_s": MIN_REQUEST_INTERVAL,
        "request_timeout_s": NETWORK_TIMEOUT,
        "transfer_deadline_s": NETWORK_DEADLINE,
        "native_duration_cap_s": MAX_NATIVE_SECONDS,
        "known_exposure_catalogue_sha256": digest(HISTORY),
        "code_sha256": {str(path.relative_to(ROOT)): digest(path) for path in code},
        "decoder_sha256": {key: digest(Path(path).resolve()) for key, path in decoders.items()},
        "numpy_version": np.__version__,
        "zlib_version": zlib.ZLIB_RUNTIME_VERSION,
        "model_calls_permitted": False,
    }


def headroom():
    require(
        ARCHIVE.is_dir()
        and ARCHIVE.resolve() == ARCHIVE
        and ARCHIVE.parent.parent.is_mount()
        and shutil.disk_usage(ARCHIVE).free >= DISK_RESERVE
        and shutil.disk_usage(BASE).free >= DISK_RESERVE,
        "Mounted archive and workspace disk reserves required",
    )


def preflight():
    headroom()
    require(not TRIAL.exists() and not CACHE.exists(), "Preserve every previous trial path")
    return {"status": "ready_for_metadata_only", **lock_context()}


def freeze():
    preflight()
    TRIAL.mkdir(mode=0o700)
    CACHE.mkdir(mode=0o700)
    transport = RangeReader(2, END.size + DIRECTORY_SIZE)
    footer = transport.get(ARCHIVE_BYTES - END.size, END.size)
    directory = transport.get(DIRECTORY_START, DIRECTORY_SIZE)
    selected = select(footer, directory)
    (TRIAL / "footer.bin").write_bytes(footer)
    (TRIAL / "directory.bin").write_bytes(directory)
    plan = TRIAL / "private-plan.json"
    save(plan, [asdict(member) for member in selected])
    locked = {
        **lock_context(),
        "private_plan_sha256": digest(plan),
        "metadata_requests": transport.requests,
        "metadata_bytes": transport.requested_bytes,
    }
    save(TRIAL / "freeze.json", locked)
    return {"status": "frozen_metadata_only", "freeze_sha256": digest(TRIAL / "freeze.json"), **locked}


def verify():
    require(
        TRIAL.resolve() == TRIAL and CACHE.resolve() == CACHE and CACHE.is_dir(), "Owned stage paths required"
    )
    selected = select((TRIAL / "footer.bin").read_bytes(), (TRIAL / "directory.bin").read_bytes())
    plan = TRIAL / "private-plan.json"
    require(read(plan) == [asdict(member) for member in selected], "Exact selected groups required")
    frozen = TRIAL / "freeze.json"
    require(
        read(frozen)
        == {
            **lock_context(),
            "private_plan_sha256": digest(plan),
            "metadata_requests": 2,
            "metadata_bytes": END.size + DIRECTORY_SIZE,
        },
        "Frozen metadata or runtime differs",
    )
    return digest(frozen), selected


def stage_receipt(stage, slot):
    return TRIAL / f"{stage}-{slot:02d}.json"


def acquire():
    headroom()
    lock, members = verify()
    save(TRIAL / "acquire-start.json", {"freeze_sha256": lock})
    transport = RangeReader(MAX_MEDIA_REQUESTS, MAX_MEDIA_BYTES)
    for slot, member in enumerate(members, 1):
        row = {"slot": slot, "freeze_sha256": lock}
        try:
            original = extract_member(member, transport)
            with (CACHE / f"slot-{slot:02d}.wav").open("xb") as stream:
                stream.write(original)
            row.update(status="verified_member", source_sha256=hashlib.sha256(original).hexdigest())
        except Rejected as exc:
            row.update(status="unavailable", reason=str(exc))
        except HTTPError as exc:
            reason = "publisher_access_or_rate_limit" if exc.code in {403, 429} else "publisher_http_error"
            row.update(status="unavailable", reason=reason, http_status=exc.code)
        except (URLError, HTTPException, OSError):
            row.update(status="unavailable", reason="transport_error")
        save(stage_receipt("acquire", slot), row)
        if row.get("reason") in {
            "publisher_access_or_rate_limit",
            "transfer_deadline",
            "unexpected_partial_response",
        }:
            break
    save(
        TRIAL / "acquire-end.json",
        {"freeze_sha256": lock, "requests": transport.requests, "requested_bytes": transport.requested_bytes},
    )
    return readout()


def verified_original(slot, member, row):
    source = CACHE / f"slot-{slot:02d}.wav"
    require(
        source.is_file() and not source.is_symlink() and source.stat().st_size == member.expanded,
        "Exact original member file required",
    )
    raw = source.read_bytes()
    require(
        zlib.crc32(raw) == member.crc32 and hashlib.sha256(raw).hexdigest() == row["source_sha256"],
        "Saved member CRC or SHA-256 differs",
    )
    return source


def prepare_clip(source, source_sha, excluded):
    with wave.open(str(source), "rb") as wav:
        frames, rate, channels, width = (
            wav.getnframes(),
            wav.getframerate(),
            wav.getnchannels(),
            wav.getsampwidth(),
        )
        require(
            1 <= channels <= 2
            and 1 <= width <= 4
            and crop.MIN_NATIVE_RATE <= rate <= crop.MAX_NATIVE_RATE
            and 3 * rate <= frames <= MAX_NATIVE_SECONDS * rate
            and wav.getcomptype() == "NONE"
            and len(wav.readframes(frames)) == frames * channels * width,
            "Complete bounded native PCM WAV required",
        )
    decoded = decode_audio(source, source_sha256=source_sha, max_seconds=MAX_NATIVE_SECONDS + 1)
    pcm, mapping = decoded.pcm, decoded.mapping
    require(
        mapping["source_sample_rate"] == rate
        and mapping["source_channels"] == channels
        and mapping["source_codec"].startswith("pcm_")
        and abs(len(pcm) - frames * crop.RATE / rate) <= 1
        and mapping["pcm_sha256"] == crop.pcm_sha(pcm)
        and np.array_equal(read_wav(decoded.wav), pcm),
        "Complete production PCM differs from native WAV",
    )
    full = source_ranges(mapping, 0, len(pcm))
    require(len(full) == 1, "Continuous source audio required")
    (start, end), activity = crop.select_crop(pcm)
    output = pcm[start:end]
    windows = source_ranges(mapping, start, end)
    require(
        windows == [crop.sample_range(start, end, full[0]["source_start_pts"])],
        "Exact crop source clock required",
    )
    fingerprints = {f"bytes:{source_sha}", f"pcm:{mapping['pcm_sha256']}", f"pcm:{crop.pcm_sha(output)}"}
    require(not fingerprints & excluded, "Known byte or PCM overlap")
    wav = wav_bytes(output)
    require(np.array_equal(read_wav(wav), output), "Exact crop WAV round trip required")
    return (
        wav,
        {
            "full_pcm_sha256": mapping["pcm_sha256"],
            "native_frames": frames,
            "native_rate": rate,
            "native_channels": channels,
            "native_width_bytes": width,
            "full_source_range": full[0],
            "source_range": windows[0],
            "samples": len(output),
            "pcm_sha256": crop.pcm_sha(output),
            "wav_sha256": hashlib.sha256(wav).hexdigest(),
            **activity,
        },
        fingerprints,
    )


def qualify():
    headroom()
    lock, members = verify()
    require(
        read(TRIAL / "acquire-end.json")["freeze_sha256"] == lock, "Terminal acquisition receipt required"
    )
    save(TRIAL / "qualify-start.json", {"freeze_sha256": lock})
    excluded = set(read(HISTORY)["keys"])
    for slot, member in enumerate(members, 1):
        path = stage_receipt("acquire", slot)
        if not path.is_file() or (acquisition := read(path))["status"] != "verified_member":
            continue
        require(
            acquisition["freeze_sha256"] == lock and acquisition["slot"] == slot,
            "Exact acquisition receipt required",
        )
        source = verified_original(slot, member, acquisition)
        row = {"slot": slot, "freeze_sha256": lock, "acquisition_sha256": digest(path)}
        try:
            wav, facts, identities = prepare_clip(source, acquisition["source_sha256"], excluded)
            with (CACHE / f"slot-{slot:02d}.crop.wav").open("xb") as stream:
                stream.write(wav)
            row.update(facts, status="qualified_pcm")
            excluded.update(identities)
        except (ValueError, OSError, EOFError, wave.Error):
            row.update(status="unavailable", reason="native_pcm_or_activity_unqualified")
        save(stage_receipt("qualify", slot), row)
    save(TRIAL / "qualify-end.json", {"freeze_sha256": lock})
    return readout()


def readout():
    lock, members = verify()
    counts, commitments = Counter(), []
    for slot, member in enumerate(members, 1):
        acquired, qualified = (stage_receipt(stage, slot) for stage in ("acquire", "qualify"))
        if not acquired.is_file():
            counts["acquisition_unattempted"] += 1
            continue
        original = read(acquired)
        require(
            original["freeze_sha256"] == lock and original["slot"] == slot, "Acquisition identity differs"
        )
        commitments.append(digest(acquired))
        if original["status"] == "unavailable":
            require(not qualified.exists(), "Unavailable source cannot be prepared")
            counts["acquisition_unavailable"] += 1
            continue
        require(original["status"] == "verified_member", "Unexpected acquisition status")
        verified_original(slot, member, original)
        counts["verified_member"] += 1
        if not qualified.is_file():
            counts["qualification_unattempted"] += 1
            continue
        row = read(qualified)
        require(
            row["freeze_sha256"] == lock
            and row["slot"] == slot
            and row["acquisition_sha256"] == digest(acquired),
            "Qualification identity differs",
        )
        commitments.append(digest(qualified))
        if row["status"] == "qualified_pcm":
            path = CACHE / f"slot-{slot:02d}.crop.wav"
            require(
                path.is_file()
                and not path.is_symlink()
                and path.stat().st_size <= 2 * crop.MAX_SAMPLES + 1024,
                "Bounded exact crop artifact required",
            )
            raw = path.read_bytes()
            pcm = read_wav(raw)
            require(
                crop.MIN_SAMPLES <= len(pcm) == row["samples"] <= crop.MAX_SAMPLES
                and crop.pcm_sha(pcm) == row["pcm_sha256"]
                and hashlib.sha256(raw).hexdigest() == row["wav_sha256"]
                and row["source_range"]
                == crop.sample_range(
                    row["source_range"]["start_sample"],
                    row["source_range"]["end_sample"],
                    row["full_source_range"]["source_start_pts"],
                )
                and row["source_range"]["end_sample"] - row["source_range"]["start_sample"] == len(pcm),
                "Stored crop extent or PCM differs",
            )
            counts["qualified_pcm"] += 1
        else:
            require(row["status"] == "unavailable", "Unexpected qualification status")
            counts["qualification_unavailable"] += 1
    terminal = {}
    for stage in ("acquire", "qualify"):
        path = TRIAL / f"{stage}-end.json"
        terminal[stage] = read(path) if path.is_file() else None
        require(
            terminal[stage] is None or terminal[stage]["freeze_sha256"] == lock, "Terminal identity differs"
        )
    transfer = terminal["acquire"] or {"requests": 0, "requested_bytes": 0}
    require(
        0 <= transfer["requests"] <= MAX_MEDIA_REQUESTS
        and 0 <= transfer["requested_bytes"] <= MAX_MEDIA_BYTES,
        "Archive request accounting exceeds frozen bounds",
    )
    return {
        "schema": "chipmunk.audio-rueg-genuine-result/246",
        "freeze_sha256": lock,
        "planned": len(members),
        "counts": dict(counts),
        "age_group_counts": {"unverified": len(members)},
        "media_requests": transfer["requests"],
        "media_requested_bytes": transfer["requested_bytes"],
        "terminal_stages": {stage: value is not None for stage, value in terminal.items()},
        "receipt_set_sha256": hashlib.sha256(
            json.dumps(commitments, separators=(",", ":")).encode()
        ).hexdigest(),
        "full_archive_verified": False,
        "model_calls": 0,
    }


def main():
    commands = {
        "preflight": preflight,
        "freeze": freeze,
        "acquire": acquire,
        "qualify": qualify,
        "readout": readout,
    }
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=tuple(commands))
    args = parser.parse_args()
    print(json.dumps(commands[args.command](), sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
