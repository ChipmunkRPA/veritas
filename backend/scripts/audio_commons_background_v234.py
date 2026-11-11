"""Qualify fixed Wikimedia Commons background recordings without detector calls."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import re
import shutil
import stat
import subprocess
import time
import wave
from collections import Counter
from dataclasses import dataclass, fields
from http.client import HTTPException
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import HTTPRedirectHandler, Request, build_opener

from app.detectors.audio.decode import SAMPLE_RATE, decode_audio, source_ranges

ROOT = Path(__file__).resolve().parents[2]
NAME = "audio-commons-background-v234"
TRIAL = ROOT / "backend/data/independent-evaluation" / NAME
PRIVATE_ROSTER = TRIAL.with_name(f"{NAME}-private-roster.json")
PRIVATE_ROSTER_SHA256 = "8b8dc7da941618e5ff7ce6ac47625cc8635684bc1f9f8aabe961e57366d6fbcf"
PRIVATE_ROSTER_SCHEMA = "chipmunk.audio-commons-background-private-roster/234"
MAX_ROSTER_BYTES = 10_000
PLANNED_ENVIRONMENTS = {"kitchen": 4, "traffic": 3}
PLANNED_LICENSES = {"CC0-1.0": 2, "CC-BY-SA-4.0": 3, "Public domain": 2}
VOLUME = Path("/Volumes/SEAGATE")
CACHE = VOLUME / "Codex-Development-Archive/2026-11-08-Chipmunk-Verify" / NAME
HOST = "https://upload.wikimedia.org/wikipedia/commons/"
AGENT = "ChipmunkVerifyResearch/1.0 (https://github.com/ChipmunkRPA/veritas)"
FREEZE_SCHEMA = "chipmunk.audio-commons-background-freeze/234"
DECODE_SECONDS = 120
CHUNK_BYTES = 256 * 1024
MAX_FILE_BYTES = 125_000_000
MAX_RUN_BYTES = 225_000_000
MAX_RUN_SECONDS = 600
MIN_REQUEST_INTERVAL = 3
SOCKET_TIMEOUT = 20
DISK_RESERVE = 3 * 1024**3
CONTENT_TYPES = {
    ".oga": "application/ogg",
    ".ogg": "application/ogg",
    ".wav": "audio/wav",
    ".webm": "video/webm",
    ".flac": "audio/x-flac",
}
REJECT_REASONS = frozenset(
    {
        "publisher_access_or_rate_limit",
        "publisher_http_error",
        "unexpected_publisher_response",
        "unexpected_body_length",
        "publisher_checksum_mismatch",
        "transport_error",
        "transport_deadline",
    }
)


@dataclass(frozen=True)
class Source:
    title: str
    page_id: int
    revision: int
    uploaded_utc: str
    author: str
    environment: str
    license: str
    provenance: str
    size: int
    publisher_sha1: str
    content_type: str

    @property
    def revision_url(self):
        return f"https://commons.wikimedia.org/w/index.php?oldid={self.revision}"

    @property
    def url(self):
        name = self.title.replace(" ", "_")
        address = hashlib.md5(name.encode(), usedforsecurity=False).hexdigest()
        return f"{HOST}{address[0]}/{address[:2]}/{quote(name)}"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path, algorithm="sha256"):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, algorithm).hexdigest()


def serialized(value):
    return json.dumps(value, indent=2, sort_keys=True) + "\n"


def save(path, value):
    with path.open("x", encoding="utf-8") as stream:
        stream.write(serialized(value))


def read(path):
    require(path.is_file() and not path.is_symlink(), "An owned regular receipt is required")
    return json.loads(path.read_text(encoding="utf-8"))


def load_sources():
    require(PRIVATE_ROSTER.resolve() == PRIVATE_ROSTER, "Owned private source roster path required")
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
    with os.fdopen(os.open(PRIVATE_ROSTER, flags), "rb") as stream:
        info = os.fstat(stream.fileno())
        require(
            stat.S_ISREG(info.st_mode)
            and stat.S_IMODE(info.st_mode) == 0o600
            and info.st_uid == os.geteuid()
            and 0 < info.st_size <= MAX_ROSTER_BYTES,
            "Source roster must be an owner-only regular file",
        )
        raw = stream.read(MAX_ROSTER_BYTES + 1)
    require(hashlib.sha256(raw).hexdigest() == PRIVATE_ROSTER_SHA256, "Private source roster bytes differ")
    value = json.loads(raw)
    require(
        isinstance(value, dict)
        and set(value) == {"schema", "sources"}
        and value["schema"] == PRIVATE_ROSTER_SCHEMA
        and isinstance(value["sources"], list)
        and len(value["sources"]) == sum(PLANNED_ENVIRONMENTS.values()),
        "Exact private source roster schema required",
    )
    required_fields = {field.name for field in fields(Source)}
    require(
        all(isinstance(row, dict) and set(row) == required_fields for row in value["sources"]),
        "Private source record fields differ",
    )
    sources = tuple(Source(**row) for row in value["sources"])
    for source in sources:
        require(
            all(type(v) is int and v > 0 for v in (source.size, source.page_id, source.revision))
            and source.size <= MAX_FILE_BYTES
            and all(
                isinstance(v, str) and bool(v)
                for v in (
                    source.title,
                    source.author,
                    source.provenance,
                    source.uploaded_utc,
                    source.publisher_sha1,
                    source.content_type,
                    source.license,
                    source.environment,
                )
            )
            and "/" not in source.title
            and "\\" not in source.title
            and re.fullmatch(r"[0-9a-f]{40}", source.publisher_sha1)
            and re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", source.uploaded_utc)
            and CONTENT_TYPES.get(Path(source.title).suffix) == source.content_type,
            "One exact commercially permitted Commons asset is required per slot",
        )
    require(
        sum(source.size for source in sources) <= MAX_RUN_BYTES
        and len({source.author for source in sources}) == len(sources)
        and len({source.page_id for source in sources}) == len(sources)
        and len({source.publisher_sha1 for source in sources}) == len(sources)
        and Counter(source.environment for source in sources) == PLANNED_ENVIRONMENTS
        and Counter(source.license for source in sources) == PLANNED_LICENSES,
        "Source size, environment, license or identity bounds differ",
    )
    return sources


def identity(sources):
    files = (
        Path(__file__),
        ROOT / "backend/tests/test_audio_commons_background_v234.py",
        ROOT / "backend/app/detectors/audio/decode.py",
        ROOT / "backend/app/detectors/video/decode.py",
        ROOT / "docs/audio-commons-background-v234-method.md",
    )
    decoders = {name: shutil.which(name) for name in ("ffmpeg", "ffprobe")}
    require(all(decoders.values()), "Production FFmpeg and FFprobe are required")
    return {
        "schema": FREEZE_SCHEMA,
        "private_roster_sha256": PRIVATE_ROSTER_SHA256,
        "planned_by_environment": PLANNED_ENVIRONMENTS,
        "planned_by_license": PLANNED_LICENSES,
        "expected_publisher_bytes": sum(source.size for source in sources),
        "code_sha256": {str(path.relative_to(ROOT)): digest(path) for path in files},
        "decoder_sha256": {name: digest(Path(path).resolve()) for name, path in decoders.items()},
        "numpy_version": importlib.metadata.version("numpy"),
        "maximum_requests": len(sources),
        "maximum_file_bytes": MAX_FILE_BYTES,
        "maximum_run_bytes": MAX_RUN_BYTES,
        "maximum_run_seconds": MAX_RUN_SECONDS,
        "minimum_request_interval_seconds": MIN_REQUEST_INTERVAL,
        "socket_timeout_seconds": SOCKET_TIMEOUT,
        "production_decode_maximum_seconds": DECODE_SECONDS,
        "expected_speech_or_music_review": "pending",
        "model_calls_permitted": False,
    }


def headroom():
    require(
        VOLUME.is_mount() and CACHE.parent.is_dir() and CACHE.parent.resolve() == CACHE.parent,
        "The mounted Seagate project archive is required",
    )
    require(
        shutil.disk_usage(CACHE.parent).free >= DISK_RESERVE
        and shutil.disk_usage(TRIAL.parent).free >= DISK_RESERVE,
        "The existing archive and workspace disk reserves are required",
    )


def verify():
    locked = TRIAL / "freeze.json"
    require(
        TRIAL.resolve() == TRIAL and CACHE.resolve() == CACHE and CACHE.is_dir(), "Owned study paths required"
    )
    sources = load_sources()
    require(read(locked) == identity(sources), "Exact frozen Commons study required")
    return digest(locked), sources


def prospective():
    headroom()
    require(not TRIAL.exists() and not CACHE.exists(), "Preserve all previous study paths")
    return identity(load_sources())


def preflight():
    locked = prospective()
    return {
        "status": "ready",
        "prospective_freeze_sha256": hashlib.sha256(serialized(locked).encode()).hexdigest(),
        **locked,
    }


def freeze():
    locked = prospective()
    TRIAL.mkdir(mode=0o700)
    CACHE.mkdir(mode=0o700)
    save(TRIAL / "freeze.json", locked)
    return {
        "status": "frozen",
        "planned": locked["maximum_requests"],
        "freeze_sha256": digest(TRIAL / "freeze.json"),
    }


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class Rejected(ValueError):
    pass


def asset_paths(slot, source):
    stem = CACHE / f"slot-{slot:02d}"
    return stem.with_suffix(Path(source.title).suffix), stem.with_suffix(".pcm.wav")


def download(source, destination, opener, deadline):
    require(
        not destination.exists() and destination.resolve() == destination, "Owned fresh asset path required"
    )
    partial = destination.with_suffix(".part")
    request = Request(source.url, headers={"User-Agent": AGENT, "Accept-Encoding": "identity"})
    sha1, sha256 = hashlib.sha1(), hashlib.sha256()
    size = 0
    if time.monotonic() >= deadline:
        raise Rejected("transport_deadline")
    with opener.open(request, timeout=SOCKET_TIMEOUT) as response:
        allowed_type = {source.content_type}
        if source.content_type == "audio/wav":
            allowed_type.add("audio/x-wav")
        if source.content_type == "audio/x-flac":
            allowed_type.add("audio/flac")
        if (
            response.geturl() != source.url
            or response.status != 200
            or response.headers.get_content_type() not in allowed_type
            or response.headers.get("Content-Length") != str(source.size)
            or response.headers.get("Content-Encoding", "identity") != "identity"
        ):
            raise Rejected("unexpected_publisher_response")
        with partial.open("xb") as stream:
            while True:
                if time.monotonic() >= deadline:
                    raise Rejected("transport_deadline")
                chunk = response.read1(min(CHUNK_BYTES, source.size + 1 - size))
                if time.monotonic() >= deadline:
                    raise Rejected("transport_deadline")
                if not chunk:
                    break
                size += len(chunk)
                if size > source.size:
                    raise Rejected("unexpected_body_length")
                sha1.update(chunk)
                sha256.update(chunk)
                stream.write(chunk)
    if size != source.size:
        raise Rejected("unexpected_body_length")
    if sha1.hexdigest() != source.publisher_sha1:
        raise Rejected("publisher_checksum_mismatch")
    partial.rename(destination)
    return {"bytes": size, "publisher_sha1": sha1.hexdigest(), "sha256": sha256.hexdigest()}


def acquire():
    headroom()
    locked, sources = verify()
    marker = TRIAL / "acquisition-start.json"
    save(marker, {"freeze_sha256": locked})
    opener = build_opener(NoRedirect)
    deadline = time.monotonic() + MAX_RUN_SECONDS
    for slot, source in enumerate(sources, 1):
        if time.monotonic() >= deadline:
            break
        target, _ = asset_paths(slot, source)
        path = TRIAL / f"acquisition-{slot:02d}.json"
        started = time.monotonic()
        row = {"ordinal": slot, "freeze_sha256": locked}
        try:
            row.update(download(source, target, opener, deadline), status="verified_original")
        except Rejected as exc:
            row.update(status="rejected", reason=str(exc))
        except HTTPError as exc:
            reason = "publisher_access_or_rate_limit" if exc.code in {403, 429} else "publisher_http_error"
            row.update(status="rejected", reason=reason, http_status=exc.code)
        except (URLError, HTTPException, OSError):
            row.update(status="rejected", reason="transport_error")
        save(path, row)
        if row.get("reason") in {"publisher_access_or_rate_limit", "transport_deadline"}:
            break
        if slot < len(sources):
            time.sleep(max(0, MIN_REQUEST_INTERVAL - (time.monotonic() - started)))
    save(TRIAL / "acquisition-end.json", {"freeze_sha256": locked})
    return readout()


def verify_original(source, target, row):
    require(
        target.is_file()
        and not target.is_symlink()
        and target.stat().st_size == row["bytes"] == source.size
        and digest(target, "sha1") == row["publisher_sha1"] == source.publisher_sha1
        and digest(target) == row["sha256"],
        "Saved original bytes differ from the publisher or receipt",
    )


def inspect_pcm(path):
    require(path.is_file() and not path.is_symlink(), "Owned regular PCM output is required")
    with wave.open(str(path), "rb") as stream:
        require(
            (stream.getnchannels(), stream.getsampwidth(), stream.getframerate(), stream.getcomptype())
            == (1, 2, SAMPLE_RATE, "NONE"),
            "Expected exact signed-16-bit mono 16 kHz PCM",
        )
        samples = stream.getnframes()
        require(
            3 * SAMPLE_RATE <= samples <= DECODE_SECONDS * SAMPLE_RATE,
            "PCM extent differs from frozen bounds",
        )
        frames = stream.readframes(samples)
        require(len(frames) == samples * 2, "Truncated PCM")
    require(frames != frames[:2] * samples, "Constant PCM provides no usable recording")
    return {"samples": samples, "wav_sha256": digest(path), "pcm_sha256": hashlib.sha256(frames).hexdigest()}


def qualify():
    headroom()
    locked, sources = verify()
    require(
        read(TRIAL / "acquisition-end.json") == {"freeze_sha256": locked}, "Terminated acquisition required"
    )
    save(TRIAL / "qualification-start.json", {"freeze_sha256": locked})
    for slot, source in enumerate(sources, 1):
        receipt = TRIAL / f"acquisition-{slot:02d}.json"
        if not receipt.is_file() or (original := read(receipt))["status"] != "verified_original":
            continue
        target, pcm = asset_paths(slot, source)
        verify_original(source, target, original)
        row = {
            "ordinal": slot,
            "freeze_sha256": locked,
            "acquisition_receipt_sha256": digest(receipt),
            "auditory_review": "pending",
            "admitted_for_model_evaluation": False,
        }
        try:
            decoded = decode_audio(target, source_sha256=original["sha256"], max_seconds=DECODE_SECONDS)
            timeline = source_ranges(decoded.mapping, 0, len(decoded.pcm))
            with pcm.open("xb") as stream:
                stream.write(decoded.wav)
            row.update(
                inspect_pcm(pcm),
                status="verified_pcm",
                source_ranges=timeline,
                source_codec=decoded.mapping["source_codec"],
                source_channels=decoded.mapping["source_channels"],
                source_sample_rate=decoded.mapping["source_sample_rate"],
                source_duration_s=decoded.mapping["source_duration_s"],
            )
        except (ValueError, EOFError, OSError, wave.Error, subprocess.SubprocessError):
            row.update(status="rejected_pcm", reason="bounded_decode_or_pcm_verification_failed")
        save(TRIAL / f"qualification-{slot:02d}.json", row)
    save(TRIAL / "qualification-end.json", {"freeze_sha256": locked})
    return readout()


def readout():
    locked, sources = verify()
    counts = {name: Counter() for name in ("kitchen", "traffic")}
    commitments = {name: [] for name in ("acquisition", "qualification")}
    rejected = Counter()
    for slot, source in enumerate(sources, 1):
        counts[source.environment]["planned"] += 1
        target, pcm = asset_paths(slot, source)
        parent = None
        for stage in commitments:
            path = TRIAL / f"{stage}-{slot:02d}.json"
            if not path.is_file():
                counts[source.environment][f"{stage}_unattempted"] += 1
                continue
            row = read(path)
            require(row["ordinal"] == slot and row["freeze_sha256"] == locked, "Receipt identity differs")
            commitments[stage].append(digest(path))
            counts[source.environment][row["status"]] += 1
            if stage == "acquisition":
                if row["status"] == "verified_original":
                    verify_original(source, target, row)
                    parent = digest(path)
                else:
                    require(
                        row["status"] == "rejected"
                        and row["reason"] in REJECT_REASONS
                        and not target.exists(),
                        "Unexpected acquisition status or artifact",
                    )
                    rejected[row["reason"]] += 1
                continue
            require(
                parent
                and row["acquisition_receipt_sha256"] == parent
                and row["auditory_review"] == "pending"
                and row["admitted_for_model_evaluation"] is False,
                "Mechanical qualification has no auditory clearance",
            )
            if row["status"] == "verified_pcm":
                require(
                    all(row[k] == value for k, value in inspect_pcm(pcm).items()),
                    "PCM qualification receipt differs",
                )
                cursor = 0
                for bounds in row["source_ranges"]:
                    end = bounds["end_sample"]
                    require(
                        bounds["start_sample"] == cursor < end <= row["samples"]
                        and bounds["time_base"] == f"1/{SAMPLE_RATE}"
                        and bounds["source_end_pts"] - bounds["source_start_pts"] == end - cursor,
                        "Saved source range differs from PCM extent",
                    )
                    cursor = end
                require(cursor == row["samples"], "Saved source ranges must partition the complete PCM")
            else:
                require(
                    row["status"] == "rejected_pcm"
                    and row["reason"] == "bounded_decode_or_pcm_verification_failed",
                    "Unexpected PCM qualification status",
                )
    terminal = {key: (TRIAL / f"{key}-end.json").is_file() for key in commitments}
    require(
        all(
            read(TRIAL / f"{key}-end.json") == {"freeze_sha256": locked} for key in terminal if terminal[key]
        ),
        "Terminal stage markers must bind the same study",
    )
    return {
        "schema": "chipmunk.audio-commons-background-result/234",
        "freeze_sha256": locked,
        "counts_by_environment": {key: dict(value) for key, value in counts.items()},
        "acquisition_rejection_counts": dict(rejected),
        "terminal_stages": terminal,
        "receipt_set_sha256": {
            key: hashlib.sha256(json.dumps(value, separators=(",", ":")).encode()).hexdigest()
            for key, value in commitments.items()
        },
        "auditory_reviewed": 0,
        "admitted_for_model_evaluation": 0,
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
    print(json.dumps(commands[args.command](), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
