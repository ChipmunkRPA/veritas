"""Freeze and prepare twelve source-separated genuine Turkish VoxForge controls."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import http.client
import io
import json
import math
import os
import re
import shutil
import subprocess
import tarfile
import types
import unicodedata
import urllib.request
import wave
from collections import Counter, defaultdict
from html.parser import HTMLParser
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory

from app.detectors.audio.decode import SAMPLE_RATE, decode_audio, read_wav, source_ranges
from scripts import video_appearance_runtime_v108 as runtime
from scripts.video_appearance_corrected_v112 import scoped
from scripts.video_appearance_native_v108 import BASE, ROOT, read, require, sha, write

MODULE = "scripts.audio_voxforge_tr_genuine_v188"
NAME = "audio-voxforge-tr-genuine-v188"
TRIAL = BASE / NAME
STAGE = TRIAL / "bind-00"
OUTPUT = STAGE / "output"
PUBLIC = ROOT / "docs/evaluation-panels/audio-voxforge-tr-genuine-freeze-v188.json"
METHOD = ROOT / "docs/audio-voxforge-tr-genuine-v188-method.md"
LISTING = Path("/private/tmp/chipmunk-voxforge-tr.n0lv8VgL/index.html")
LISTING_SHA = "b652b9add0ec4115d31a68610784ca4b1b3ac8dd1e9dda53f6a0ef74d7ddf854"
HISTORY = BASE / "audio-history-v58/catalogue.json"
ORIGIN = "https://repository.voxforge1.org/downloads/tr/Trunk/Audio/Original/48kHz_16bit/"
SESSION = re.compile(r"([A-Za-z0-9_.-]+?)-(20[0-9]{6})-([A-Za-z0-9]+)\.tgz")
MIN_SAMPLES, MAX_SAMPLES = SAMPLE_RATE * 5 // 2, SAMPLE_RATE * 4
NATIVE_RATE = 48000
GROUPS, SESSIONS_PER_GROUP = 12, 3
MAX_ARCHIVE, MAX_UNPACKED, MAX_MEMBERS, MAX_TEXT, MAX_WAVE = 8_000_000, 24_000_000, 96, 128_000, 2_000_000


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def rank(value):
    return digest(f"chipmunk.audio-voxforge-tr/188:{LISTING_SHA}:{value}".encode())


class IndexLinks(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attributes):
        if tag == "a":
            link = dict(attributes).get("href")
            if link and SESSION.fullmatch(link):
                self.links.append(link)


def source_plan(raw, known_voices):
    links = IndexLinks()
    links.feed(raw.decode("utf-8"))
    require(len(links.links) == len(set(links.links)), "Publisher index includes repeated session URLs")
    groups = defaultdict(list)
    for name in links.links:
        alias = SESSION.fullmatch(name)[1].casefold()
        if not re.search(r"anon|unknown|guest", alias) and f"voxforge:{alias}" not in known_voices:
            groups[alias].append(name)
    aliases = sorted(groups, key=lambda alias: rank(f"speaker:{alias}"))
    require(len(aliases) >= GROUPS, "Not enough named contributor groups in the frozen publisher listing")
    return [
        {
            "slot": f"genuine-{i:02d}",
            "contributor": alias,
            "sessions": sorted(groups[alias], key=lambda name: rank(f"session:{name}"))[:SESSIONS_PER_GROUP],
        }
        for i, alias in enumerate(aliases[:GROUPS])
    ], len(aliases)


def context(raw):
    require(digest(raw) == LISTING_SHA, "Exact captured publisher listing required")
    history = read(HISTORY)
    plan, count = source_plan(raw, {value.casefold() for value in history["voice_keys"]})
    ffmpeg = {name: shutil.which(name) for name in ("ffmpeg", "ffprobe")}
    require(all(ffmpeg.values()), "Production audio decoder tools required")
    files = (
        Path(__file__),
        METHOD,
        ROOT / "backend/tests/test_audio_voxforge_tr_genuine_v188.py",
        ROOT / "backend/app/detectors/audio/decode.py",
        ROOT / "backend/app/detectors/video/decode.py",
        Path(runtime.__file__),
    )
    return {
        "schema": "chipmunk.audio-voxforge-tr-genuine-freeze/188",
        "publisher_index_url": ORIGIN,
        "publisher_index_sha256": LISTING_SHA,
        "historical_identity_catalogue_sha256": sha(HISTORY),
        "code_sha256": {str(path.relative_to(ROOT)): sha(path) for path in files},
        "decoder_tool_sha256": {name: sha(Path(path).resolve()) for name, path in ffmpeg.items()},
        "language": "tr",
        "speech_origin": "genuine",
        "named_contributor_groups_in_plan_pool": count,
        "planned_contributor_slots": GROUPS,
        "planned_session_archives": sum(len(row["sessions"]) for row in plan),
        "maximum_sessions_per_group": SESSIONS_PER_GROUP,
        "maximum_archive_bytes": MAX_ARCHIVE,
        "maximum_unpacked_tar_bytes": MAX_UNPACKED,
        "native_rate": NATIVE_RATE,
        "production_rate": SAMPLE_RATE,
        "production_sample_count_bounds": [MIN_SAMPLES, MAX_SAMPLES],
        "holdout_media_access_permitted": False,
        "model_calls_permitted": False,
    }, plan


def freeze():
    require(not TRIAL.exists() and not PUBLIC.exists(), "Preserve every previous attempt")
    raw = LISTING.read_bytes()
    lock, plan = context(raw)
    TRIAL.mkdir(mode=0o700, exist_ok=False)
    with (TRIAL / "publisher-index.html").open("xb") as stream:
        stream.write(raw)
    plan_path = TRIAL / "private-plan.json"
    write(plan_path, plan)
    lock["private_plan_sha256"] = sha(plan_path)
    write(TRIAL / "freeze.json", lock)
    write(PUBLIC, lock)
    print(
        json.dumps(
            {
                "status": "frozen",
                "planned_groups": GROUPS,
                "planned_archives": lock["planned_session_archives"],
            }
        )
    )


def verify():
    lock, plan = context((TRIAL / "publisher-index.html").read_bytes())
    path = TRIAL / "private-plan.json"
    lock["private_plan_sha256"] = sha(path)
    require(
        read(PUBLIC) == read(TRIAL / "freeze.json") == lock and read(path) == plan,
        "Exact prospective contributor and session plan required",
    )
    return plan


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, msg, headers, new_url):
        return None


def session_list(plan):
    return [(row["slot"], name) for row in plan for name in row["sessions"]]


def download_archive(opener, name):
    require(SESSION.fullmatch(name), "Only a frozen original Turkish session name is permitted")
    url = ORIGIN + name
    request = urllib.request.Request(url, headers={"User-Agent": "Chipmunk Verify source evaluation"})
    with opener.open(request, timeout=20) as response:
        announced = response.headers.get("Content-Length")
        require(
            response.status == 200
            and response.geturl() == url
            and (announced is None or announced.isdecimal() and 0 < int(announced) <= MAX_ARCHIVE),
            "Exact bounded publisher response required",
        )
        payload = response.read(MAX_ARCHIVE + 1)
        require(
            0 < len(payload) <= MAX_ARCHIVE
            and (announced is None or len(payload) == int(announced))
            and payload.startswith(b"\x1f\x8b"),
            "Expected complete bounded publisher TGZ bytes",
        )
        return payload, {
            "etag": response.headers.get("ETag"),
            "last_modified": response.headers.get("Last-Modified"),
        }


def acquire():
    plan = verify()
    folder = TRIAL / "acquired"
    folder.mkdir(mode=0o700, exist_ok=False)
    archives = folder / "archives"
    archives.mkdir(mode=0o700)
    opener = urllib.request.build_opener(NoRedirect())
    rows = []
    journal = folder / "private-acquisition.jsonl"
    with journal.open("x") as stream:
        for slot, name in session_list(plan):
            row = {"slot": slot, "archive": name, "url": ORIGIN + name}
            try:
                payload, headers = download_archive(opener, name)
                with (archives / name).open("xb") as handle:
                    handle.write(payload)
            except (OSError, ValueError, http.client.HTTPException) as error:
                row.update(status="unavailable", reason=type(error).__name__)
            else:
                row.update(status="downloaded", bytes=len(payload), sha256=digest(payload), **headers)
            rows.append(row)
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            stream.flush()
    receipt = {
        "schema": "chipmunk.audio-voxforge-tr-acquisition/188",
        "status": "complete",
        "freeze_sha256": sha(PUBLIC),
        "private_acquisition_sha256": sha(journal),
        "archive_outcomes": dict(Counter(row["status"] for row in rows)),
    }
    write(folder / "result.json", receipt)
    print(json.dumps(receipt, indent=2))


def acquired(plan):
    folder = TRIAL / "acquired"
    journal = folder / "private-acquisition.jsonl"
    with journal.open() as stream:
        rows = [json.loads(line) for line in stream]
    result = read(folder / "result.json")
    require(
        [(row["slot"], row["archive"]) for row in rows] == session_list(plan)
        and all(
            row["url"] == ORIGIN + row["archive"] and row["status"] in {"downloaded", "unavailable"}
            for row in rows
        )
        and result
        == {
            "schema": "chipmunk.audio-voxforge-tr-acquisition/188",
            "status": "complete",
            "freeze_sha256": sha(PUBLIC),
            "private_acquisition_sha256": sha(journal),
            "archive_outcomes": dict(Counter(row["status"] for row in rows)),
        },
        "Exact acquisition receipt for every frozen session required",
    )
    return rows, result


def verify_archive(row):
    folder = TRIAL / "acquired/archives"
    path = folder / row["archive"]
    require(
        SESSION.fullmatch(row["archive"])
        and folder.resolve(strict=True) == folder
        and path.is_file()
        and not path.is_symlink()
        and path.stat().st_size == row["bytes"] <= MAX_ARCHIVE
        and sha(path) == row["sha256"],
        "Exact acquired session archive required",
    )
    with gzip.open(path, "rb") as stream:
        unpacked = stream.read(MAX_UNPACKED + 1)
    require(len(unpacked) <= MAX_UNPACKED, "Session expands past the frozen tar byte bound")
    return tarfile.open(fileobj=io.BytesIO(unpacked), mode="r:")


def regular_members(archive):
    members = archive.getmembers()
    named = [(PurePosixPath(member.name), member) for member in members]
    require(
        0 < len(members) <= MAX_MEMBERS
        and len({name for name, _ in named}) == len(members)
        and all(
            not name.is_absolute() and ".." not in name.parts and (member.isdir() or member.isfile())
            for name, member in named
        ),
        "Only bounded, unique safe regular session members are allowed",
    )
    return {name: member for name, member in named if member.isfile()}


def member_bytes(archive, member, maximum):
    require(0 < member.size <= maximum, "Session member exceeds its byte bound")
    with archive.extractfile(member) as stream:
        data = stream.read(maximum + 1)
    require(len(data) == member.size, "Incomplete session member")
    return data


def norm(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def license_name(text):
    notice = norm(text)
    if "gnu general public license" not in notice or re.search(
        r"non.?commercial|research (use )?only", notice
    ):
        return None
    # A generic GPL copy or its appendix template is not a per-session grant.
    notice = notice.split("how to apply these terms", 1)[0]
    grant = re.search(
        r"(?:under (?:the terms of )?(?:the )?|licen[cs]ed (?:under )?)(?:gnu general public licen[cs]e)"
        r"[^.]{0,130}?\bversion ([23])\b([^.;]{0,130})",
        notice,
    )
    if grant is None:
        return None
    later = "any later version" in grant[2] or "or later" in grant[2]
    return f"GPL-{grant[1]}.0-{'or-later' if later else 'only'}"


def metadata(archive, files, contributor):
    def named(stems):
        return [path for path in files if path.name.casefold() in stems]

    readme_names = named({"readme", "readme.txt", "readme.md"})
    require(len(readme_names) == 1, "Exact per-session README required")
    readme_path = readme_names[0]
    readme = member_bytes(archive, files[readme_path], MAX_TEXT)
    info = readme.decode("utf-8-sig", errors="replace")
    languages = re.findall(r"^\s*(?:language|lang)\s*:\s*([^\r\n]+)", info, re.IGNORECASE | re.MULTILINE)
    usernames = re.findall(
        r"^\s*(?:user\s*name|username|voxforge\s+user\s*name)\s*:\s*([^\r\n]+)",
        info,
        re.IGNORECASE | re.MULTILINE,
    )
    require(
        len(languages) == len(usernames) == 1
        and norm(languages[0]) in {"tr", "tr-tr", "tr_tr", "turkish", "türkçe"}
        and norm(usernames[0]) == contributor,
        "Exact declared Turkish language and matching named contributor required",
    )
    permissions = named(
        {"license", "license.txt", "gpl_license", "gpl_license.txt", "copying", "copying.txt"}
    )
    licenses = {}
    for path in [*permissions, readme_path]:
        data = readme if path == readme_path else member_bytes(archive, files[path], MAX_TEXT)
        notice = data.decode("utf-8-sig", errors="replace")
        require(
            not re.search(r"non.?commercial|research (use )?only", norm(notice)),
            "Conflicting per-session noncommercial or research-only terms",
        )
        if path == readme_path and not re.search(r"redistribut|license:", info, re.IGNORECASE):
            continue
        if (license := license_name(notice)) is not None:
            licenses[license] = {"member": path.as_posix(), "sha256": digest(data)}
    require(len(licenses) == 1, "One explicit, unambiguous per-session GPL permission required")
    license, permission = next(iter(licenses.items()))
    prompt_paths = named({"prompts-original", "prompts-original.txt"}) or named({"prompts", "prompts.txt"})
    require(len(prompt_paths) == 1, "Exact per-session prompt manifest required")
    prompt_path = prompt_paths[0]
    prompt_bytes = member_bytes(archive, files[prompt_path], MAX_TEXT)
    prompts = {}
    for line in prompt_bytes.decode("utf-8-sig", errors="replace").splitlines():
        fields = line.strip().split(None, 1)
        if len(fields) == 2 and "�" not in fields[1]:
            key = PurePosixPath(fields[0]).stem.casefold()
            prompt = norm(fields[1])
            require(key not in prompts and bool(prompt), "Prompt identifiers differ")
            prompts[key] = digest(prompt.encode("utf-8"))
    require(prompts, "No attributable prompt rows")
    return prompts, {
        "license": license,
        "license_receipt": permission,
        "readme_sha256": digest(readme),
        "prompt_manifest_sha256": digest(prompt_bytes),
    }


def native_samples(data, extension):
    if extension == ".wav":
        with wave.open(io.BytesIO(data), "rb") as wav:
            rate, channels, bits, frames = (
                wav.getframerate(),
                wav.getnchannels(),
                wav.getsampwidth() * 8,
                wav.getnframes(),
            )
    else:
        require(
            len(data) >= 42
            and data.startswith(b"fLaC")
            and (data[4] & 0x7F) == 0
            and int.from_bytes(data[5:8], "big") == 34,
            "Native FLAC STREAMINFO unavailable",
        )
        streaminfo = int.from_bytes(data[18:26], "big")
        rate = streaminfo >> 44
        channels = ((streaminfo >> 41) & 7) + 1
        bits = ((streaminfo >> 36) & 31) + 1
        frames = streaminfo & ((1 << 36) - 1)
    require(
        (rate, channels, bits) == (NATIVE_RATE, 1, 16) and 5 * NATIVE_RATE // 2 <= frames <= 4 * NATIVE_RATE,
        "Expected native 2.5–4 s, 48 kHz, mono, signed16 speech",
    )
    return frames


def source_range(samples):
    return {
        "start_sample": 0,
        "end_sample": samples,
        "source_start_pts": 0,
        "source_end_pts": samples,
        "time_base": f"1/{SAMPLE_RATE}",
    }


def production_pcm(payload, extension, frames, folder):
    path = folder / f"original{extension}"
    path.write_bytes(payload)
    decoded = decode_audio(path, source_sha256=digest(payload), max_seconds=5.0)
    mapping, samples = decoded.mapping, len(decoded.pcm)
    require(
        MIN_SAMPLES <= samples <= MAX_SAMPLES
        and mapping["source_sample_rate"] == NATIVE_RATE
        and mapping["source_channels"] == 1
        and abs(samples - frames * SAMPLE_RATE / NATIVE_RATE) <= 1
        and mapping["source_duration_s"] is not None
        and math.isclose(
            mapping["source_duration_s"], frames / NATIVE_RATE, rel_tol=0, abs_tol=1 / SAMPLE_RATE
        )
        and source_ranges(mapping, 0, samples) == [source_range(samples)]
        and digest(read_wav(decoded.wav).astype("<f4").tobytes()) == mapping["pcm_sha256"],
        "Expected full zero-origin native extent and exact production PCM",
    )
    return decoded.wav, {
        "native_samples": frames,
        "samples": samples,
        "sample_rate": SAMPLE_RATE,
        "pcm_sha256": mapping["pcm_sha256"],
        "source_range": source_range(samples),
    }


def prepare_slot(group, records, prompts_used, fingerprints, scratch, prepared, attempts):
    slot = group["slot"]
    for row in records:
        candidate = {"slot": slot, "archive": row["archive"]}
        if row["status"] != "downloaded":
            attempts.append({**candidate, "status": "archive_unavailable", "reason": row["reason"]})
            continue
        try:
            with verify_archive(row) as archive:
                files = regular_members(archive)
                prompts, provenance = metadata(archive, files, group["contributor"])
                clips = sorted(
                    (path for path in files if path.suffix.casefold() in {".wav", ".flac"}),
                    key=lambda path: rank(f"clip:{row['archive']}:{path.as_posix()}"),
                )
                require(clips, "No native WAV or FLAC members")
                attempts.append(
                    {
                        **candidate,
                        "status": "session_qualified",
                        **provenance,
                        "available_members": len(clips),
                    }
                )
                for path in clips:
                    attempt = {**candidate, "member": path.as_posix()}
                    prompt = prompts.get(path.stem.casefold())
                    if not prompt or prompt in prompts_used:
                        attempts.append({**attempt, "status": "prompt_unavailable_or_duplicate"})
                        continue
                    try:
                        extension = path.suffix.casefold()
                        payload = member_bytes(archive, files[path], MAX_WAVE)
                        frames = native_samples(payload, extension)
                        original = digest(payload)
                        require(f"bytes:{original}" not in fingerprints, "Previously known exact audio bytes")
                        wav, receipt = production_pcm(payload, extension, frames, scratch)
                        require(
                            f"pcm:{receipt['pcm_sha256']}" not in fingerprints, "Previously known exact PCM"
                        )
                    except (OSError, ValueError, EOFError, wave.Error) as error:
                        attempts.append(
                            {**attempt, "status": "clip_unavailable", "private_reason": str(error)}
                        )
                        continue
                    target = prepared / f"{slot}.wav"
                    with target.open("xb") as stream:
                        stream.write(wav)
                    prompts_used.add(prompt)
                    fingerprints.update({f"bytes:{original}", f"pcm:{receipt['pcm_sha256']}"})
                    attempts.append({**attempt, "status": "selected"})
                    return {
                        **attempt,
                        **provenance,
                        **receipt,
                        "status": "qualified",
                        "language": "tr",
                        "label": 0,
                        "archive_sha256": row["sha256"],
                        "source_sha256": original,
                        "prompt_sha256": prompt,
                        "wav_sha256": sha(target),
                    }
        except (OSError, ValueError, EOFError, tarfile.TarError) as error:
            attempts.append({**candidate, "status": "session_unavailable", "private_reason": str(error)})
    return {"slot": slot, "status": "unavailable", "language": "tr", "label": 0}


def prepare_worker():
    require(
        (STAGE / "resources.jsonl").is_file()
        and (STAGE / "worker.log").is_file()
        and os.environ.get("HF_HUB_OFFLINE") == os.environ.get("TRANSFORMERS_OFFLINE") == "1",
        "Exact offline guarded preparation required",
    )
    plan = verify()
    acquisition, acquired_result = acquired(plan)
    OUTPUT.mkdir(mode=0o700, exist_ok=False)
    prepared = OUTPUT / "prepared"
    prepared.mkdir(mode=0o700)
    attempts, rows, used_prompts = [], [], set()
    known = set(read(HISTORY)["keys"])
    with TemporaryDirectory(prefix="voxforge-genuine-") as tmp:
        for group in plan:
            candidates = [row for row in acquisition if row["slot"] == group["slot"]]
            rows.append(prepare_slot(group, candidates, used_prompts, known, Path(tmp), prepared, attempts))
    observations, journal = OUTPUT / "private-cases.json", OUTPUT / "private-attempts.json"
    write(observations, rows)
    write(journal, attempts)
    write(
        OUTPUT / "result.json",
        {
            "schema": "chipmunk.audio-voxforge-tr-genuine-result/188",
            "status": "complete",
            "freeze_sha256": sha(PUBLIC),
            "private_acquisition_sha256": acquired_result["private_acquisition_sha256"],
            "private_cases_sha256": sha(observations),
            "private_attempts_sha256": sha(journal),
            "planned_cases": GROUPS,
            "case_outcomes": dict(Counter(row["status"] for row in rows)),
            "attempt_outcomes": dict(Counter(row["status"] for row in attempts)),
            "qualified_licenses": dict(
                Counter(row["license"] for row in rows if row["status"] == "qualified")
            ),
            "language": "tr",
            "origin": "genuine",
            "new_neural_calls": 0,
            "holdout_media_opened": False,
            "holdout_scored": False,
        },
    )


def prepare():
    def launch(argv, **kwargs):
        require(
            argv[1:] == ["-B", "-m", "scripts.video_appearance_native_v108", "bind", str(OUTPUT), "0"],
            "Exact v108 data-only invocation required",
        )
        return subprocess.Popen([*argv[:3], MODULE, "prepare-worker"], **kwargs)

    proxy = types.SimpleNamespace(
        Popen=launch, STDOUT=subprocess.STDOUT, TimeoutExpired=subprocess.TimeoutExpired
    )

    def ready(*_):
        if not (TRIAL / "acquired/result.json").is_file():
            return False
        acquired(verify())
        return True

    scoped(
        runtime.run,
        {"TRIAL": TRIAL, "verify": verify, "subprocess": proxy, "prerequisites": ready, "__file__": __file__},
    )("bind", 0)


def read_prepared_pcm(row):
    samples = row["samples"]
    require(
        row["status"] == "qualified"
        and re.fullmatch(r"genuine-[0-9]{2}", row["slot"])
        and MIN_SAMPLES <= samples <= MAX_SAMPLES
        and row["sample_rate"] == SAMPLE_RATE
        and row["source_range"] == source_range(samples),
        "Exact qualified production WAV receipt required",
    )
    folder = OUTPUT / "prepared"
    path = folder / f"{row['slot']}.wav"
    require(
        folder.resolve(strict=True) == folder and path.is_file() and not path.is_symlink(),
        "Direct prepared WAV required",
    )
    data = path.read_bytes()
    require(digest(data) == row["wav_sha256"], "Prepared WAV bytes differ")
    pcm = read_wav(data)
    require(
        len(pcm) == samples and digest(pcm.astype("<f4").tobytes()) == row["pcm_sha256"],
        "Exact production PCM differs",
    )
    return pcm


def readout():
    plan = verify()
    archive_rows, acquisition = acquired(plan)
    result_path, profile_path = OUTPUT / "result.json", STAGE / "profile.json"
    result, profile = read(result_path), read(profile_path)
    journal, case_path = OUTPUT / "private-attempts.json", OUTPUT / "private-cases.json"
    rows, attempts = read(case_path), read(journal)
    require(
        profile["schema"] == "chipmunk.video-appearance-resource-profile/108"
        and profile["command"] == "bind"
        and profile["batch"] == 0
        and profile["child_started"] is True
        and profile["child_exit_code"] == 0
        and profile["guard"] is None
        and profile["sample_errors"] == 0
        and profile["runtime_sha256"] == sha(Path(__file__))
        and profile["worker_receipt_sha256"] == sha(result_path)
        and profile["resource_journal_sha256"] == sha(STAGE / "resources.jsonl")
        and profile["sampled_tree_rss_lower_bound_bytes"] <= 1536 * 1024**2
        and result["schema"] == "chipmunk.audio-voxforge-tr-genuine-result/188"
        and result["status"] == "complete"
        and result["freeze_sha256"] == sha(PUBLIC)
        and result["private_acquisition_sha256"] == acquisition["private_acquisition_sha256"]
        and result["private_cases_sha256"] == sha(case_path)
        and result["private_attempts_sha256"] == sha(journal)
        and result["planned_cases"] == len(rows) == GROUPS
        and result["new_neural_calls"] == profile["new_neural_calls"] == 0
        and result["holdout_media_opened"] is result["holdout_scored"] is False
        and [row["slot"] for row in rows] == [group["slot"] for group in plan]
        and all(
            row["status"] in {"qualified", "unavailable"} and row["language"] == "tr" and row["label"] == 0
            for row in rows
        )
        and result["case_outcomes"] == Counter(row["status"] for row in rows)
        and result["attempt_outcomes"] == Counter(row["status"] for row in attempts),
        "Exact completed guarded data-only preparation required",
    )
    qualified = [row for row in rows if row["status"] == "qualified"]
    archives = {
        (row["slot"], row["archive"]): row["sha256"] for row in archive_rows if row["status"] == "downloaded"
    }
    require(
        all(archives.get((row["slot"], row["archive"])) == row["archive_sha256"] for row in qualified)
        and all(
            len({row[key] for row in qualified}) == len(qualified)
            for key in ("prompt_sha256", "source_sha256", "pcm_sha256")
        )
        and result["qualified_licenses"] == Counter(row["license"] for row in qualified),
        "Selected clips must match acquired sessions and unique prompt and PCM commitments",
    )
    for row in qualified:
        read_prepared_pcm(row)
    return {
        **result,
        "resource_profile_sha256": sha(profile_path),
        "maximum_sampled_descendant_rss_bytes": profile["sampled_tree_rss_lower_bound_bytes"],
        "worker_duration_seconds": profile["duration_s"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "acquire", "prepare", "prepare-worker", "readout"))
    command = parser.parse_args().command
    if command == "readout":
        print(json.dumps(readout(), indent=2, allow_nan=False))
    else:
        {"freeze": freeze, "acquire": acquire, "prepare": prepare, "prepare-worker": prepare_worker}[
            command
        ]()
