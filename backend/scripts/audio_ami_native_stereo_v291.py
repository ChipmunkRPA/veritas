"""Qualify two fixed AMI microphone pairs without opening a model."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import wave
from collections import Counter
from contextlib import ExitStack
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

import numpy as np

from scripts import audio_native_stereo_preflight_v274 as stereo

ROOT = stereo.ROOT
TRIAL = ROOT / "backend/data/independent-evaluation/audio-ami-native-stereo-v291"
MEDIA = stereo.holdout.ARCHIVE / "audio-ami-native-stereo-v291"
FREEZE, ACQUISITION = TRIAL / "freeze.json", TRIAL / "acquired.json"
PROOF, RESULT = TRIAL / "provenance", TRIAL / "qualified.json"
HOST = "groups.inf.ed.ac.uk"
HOME_URL = f"https://{HOST}/ami"
CORPUS = f"{HOME_URL}/corpus"
MIRROR = f"{HOME_URL}/AMICorpusMirror/amicorpus"
SESSIONS = ("ES2004a", "IS1009a")
MICROPHONES = ("Array1-01", "Array1-05")
RATE, SAMPLES = stereo.RATE, stereo.SAMPLES
WINDOWS = tuple((60 + 30 * index) * RATE for index in range(12))
MEDIA_LIMIT = 40 * 1024**2
PROVENANCE_LIMIT = 512 * 1024
PROVENANCE = {
    "license": (f"{CORPUS}/license.shtml", b"creativecommons.org/licenses/by/4.0/legalcode"),
    "ethics": (f"{CORPUS}/ethicsandconsent.shtml", b"consent form"),
    "consent_v1_1": (f"{CORPUS}/ami_consent_form_081004.pdf", b"%pdf-"),
    "common_timeline": (f"{CORPUS}/overview.shtml", b"synchronized to a common timeline"),
    "signals": (f"{CORPUS}/signals.shtml", b"48khz to 16khz"),
    "issues": (f"{CORPUS}/dataproblems.shtml", b"data problems"),
    "partitions": (f"{CORPUS}/datasets.shtml", b"unseen data for evaluation"),
    **{f"index_{session}": (f"{MIRROR}/{session}/audio/", f"{session}.Array1-05.wav".lower().encode())
       for session in SESSIONS},
}
CODE = (*stereo.CODE, "backend/scripts/audio_ami_native_stereo_v291.py",
        "backend/tests/test_audio_ami_native_stereo_v291.py", "docs/audio-ami-native-stereo-v291-method.md")
require, sha, read, write = stereo.require, stereo.sha, stereo.holdout.read, stereo.holdout.write


class FirstPartyOnly(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def media_sources():
    return {f"{session}.{mic}.wav": f"{MIRROR}/{session}/audio/{session}.{mic}.wav"
            for session in SESSIONS for mic in MICROPHONES}


def receive(url, destination, minimum, maximum):
    parsed = urlsplit(url)
    require(parsed.scheme == "https" and parsed.netloc == HOST and parsed.path.startswith("/ami/")
            and not parsed.query and not parsed.fragment, "Only first-party AMI resources are allowed")
    request = Request(url, headers={"Accept-Encoding": "identity", "User-Agent": "Chipmunk-Verify-AMI-evaluation/1"})
    deadline = time.monotonic() + 180
    count, digest = 0, hashlib.sha256()
    with build_opener(FirstPartyOnly()).open(request, timeout=20) as response:
        size = response.headers.get("Content-Length")
        require(response.status == 200 and response.geturl() == url and response.headers.get("Content-Encoding", "identity") == "identity"
                and (size is None or (size.isdecimal() and minimum <= int(size) <= maximum)), "Unexpected AMI HTTP response")
        with destination.open("xb") as stream:
            while chunk := response.read1(256 * 1024):
                count += len(chunk)
                require(time.monotonic() <= deadline and count <= maximum, "AMI response exceeded time or size bound")
                digest.update(chunk)
                stream.write(chunk)
        require(time.monotonic() <= deadline and minimum <= count <= maximum and (size is None or count == int(size)),
                "Incomplete AMI response")
        return {"url": url, "bytes": count, "sha256": digest.hexdigest(),
                "last_modified": response.headers.get("Last-Modified"), "etag": response.headers.get("ETag")}


def plan():
    return {"schema": "chipmunk.audio-ami-native-stereo-plan/291", "sessions": list(SESSIONS), "microphones": list(MICROPHONES),
            "sources": media_sources(), "license": "CC-BY-4.0", "origin": "genuine recorded human meeting speech",
            "consent_source": "AMI public consent form version 1.1, 8 October 2004, and project ethics documentation",
            "provenance_urls": {name: url for name, (url, _) in PROVENANCE.items()},
            "sample_rate": RATE, "sample_width_bytes": 2, "window_samples": SAMPLES, "window_starts": list(WINDOWS),
            "planned_cases": len(SESSIONS) * len(WINDOWS), "max_media_bytes_per_file": MEDIA_LIMIT, "model_calls": 0}


def identity():
    stereo.holdout.model_free()
    return {**plan(), "code_sha256": {name: sha(ROOT / name) for name in CODE},
            "codec_binaries_sha256": {name: stereo.tool_hash(name) for name in ("ffmpeg", "ffprobe")}}


def freeze():
    require(not TRIAL.exists() and not MEDIA.exists(), "Preserve every previous source attempt")
    bound = identity()
    TRIAL.mkdir(mode=0o700)
    PROOF.mkdir(mode=0o700)
    documents = {}
    for name, (url, marker) in PROVENANCE.items():
        path = PROOF / name
        documents[name] = receive(url, path, 40, PROVENANCE_LIMIT)
        require(marker in b" ".join(path.read_bytes().lower().split()), f"AMI provenance marker differs for {name}")
    write(FREEZE, {**bound, "schema": "chipmunk.audio-ami-native-stereo-freeze/291", "documents": documents})
    return {"status": "frozen", "freeze_sha256": sha(FREEZE), "provenance_documents": len(documents), "media_files_opened": 0}


def verify():
    saved = read(FREEZE)
    expected = {**identity(), "schema": "chipmunk.audio-ami-native-stereo-freeze/291"}
    require(set(saved) == set(expected) | {"documents"} and all(saved[name] == value for name, value in expected.items())
            and set(saved["documents"]) == set(PROVENANCE), "Exact prospective AMI capture freeze required")
    for name, (url, _) in PROVENANCE.items():
        path, document = PROOF / name, saved["documents"][name]
        require(document["url"] == url and path.stat().st_size == document["bytes"] and sha(path) == document["sha256"],
                "Frozen first-party provenance snapshot differs")


def acquire():
    verify()
    require(not ACQUISITION.exists() and not MEDIA.exists(), "Preserve every previous AMI acquisition attempt")
    MEDIA.mkdir(mode=0o700)
    folder = MEDIA / "raw"
    folder.mkdir(mode=0o700)
    files = {name: receive(url, folder / name, 2 * (WINDOWS[-1] + SAMPLES), MEDIA_LIMIT)
             for name, url in media_sources().items()}
    write(ACQUISITION, {"schema": "chipmunk.audio-ami-native-stereo-acquisition/291", "freeze_sha256": sha(FREEZE), "files": files})
    return {"status": "acquired", "files": len(files), "bytes": sum(item["bytes"] for item in files.values()),
            "acquisition_sha256": sha(ACQUISITION), "model_calls": 0}


def exact_files():
    acquired = read(ACQUISITION)
    files = acquired["files"]
    require(acquired["schema"] == "chipmunk.audio-ami-native-stereo-acquisition/291"
            and acquired["freeze_sha256"] == sha(FREEZE) and set(files) == set(media_sources()), "Exact AMI acquisition receipt required")
    for name, url in media_sources().items():
        path = MEDIA / "raw" / name
        row = files[name]
        require(path.is_file() and path.resolve(strict=True) == path and row["url"] == url
                and path.stat().st_size == row["bytes"] <= MEDIA_LIMIT and sha(path) == row["sha256"],
                "First-party capture bytes differ from acquisition")
    return files


def pair_geometry(readers):
    require(len(readers) == 2, "The fixed two-microphone pair is required")
    for wav in readers:
        require((wav.getnchannels(), wav.getsampwidth(), wav.getframerate(), wav.getcomptype()) == (1, 2, RATE, "NONE"),
                "Expected original AMI mono signed-16-bit 16 kHz microphone PCM")
    samples = readers[0].getnframes()
    require(samples == readers[1].getnframes() and samples >= WINDOWS[-1] + SAMPLES,
            "Paired channels must share one complete AMI sample extent")
    return samples


def channel_digest(wav, count):
    digest = hashlib.sha256()
    wav.rewind()
    for start in range(0, count, RATE):
        length = min(count - start, RATE)
        body = wav.readframes(length)
        require(len(body) == 2 * length, "Original AMI PCM is truncated")
        digest.update(body)
    require(not wav.readframes(1), "Original AMI PCM exceeds the declared clock")
    return digest.hexdigest()


def paired_interval(readers, start):
    channels = []
    for wav in readers:
        wav.setpos(start)
        body = wav.readframes(SAMPLES)
        require(len(body) == 2 * SAMPLES, "Paired AMI window is truncated")
        channels.append(np.frombuffer(body, dtype="<i2"))
    return tuple(channels)


def cases():
    for session in SESSIONS:
        with ExitStack() as stack:
            readers = tuple(stack.enter_context(wave.open(str(MEDIA / "raw" / f"{session}.{mic}.wav"), "rb")) for mic in MICROPHONES)
            samples = pair_geometry(readers)
            hashes = [channel_digest(reader, samples) for reader in readers]
            require(len(set(hashes)) == 2, "Separate physical microphones require distinct source PCM")
            for index, start in enumerate(WINDOWS):
                yield {
                    "case": f"{session}-{index:02d}", "session": session, "source_samples": samples,
                    "source_channel_pcm_sha256": hashes, "source_range": {"start_sample": start, "end_sample": start + SAMPLES,
                                                                           "time_base": f"1/{RATE}"},
                }, paired_interval(readers, start)


def raw_replay(path, channels):
    require(path.is_file() and path.resolve(strict=True) == path, "Direct saved stereo WAV required")
    interleaved = np.column_stack(channels).astype("<i2").tobytes()
    with wave.open(str(path), "rb") as wav:
        require((wav.getnchannels(), wav.getsampwidth(), wav.getframerate(), wav.getnframes(), wav.getcomptype()) == (2, 2, RATE, SAMPLES, "NONE")
                and wav.readframes(SAMPLES + 1) == interleaved, "Saved AMI stereo WAV differs from the exact paired source interval")
    return {"stereo_pcm_sha256": stereo.digest(interleaved), "stereo_wav_sha256": sha(path)}


def qualify():
    verify()
    exact_files()
    destination = MEDIA / "qualified"
    require(not RESULT.exists() and not destination.exists(), "Preserve every prior AMI qualification attempt")
    destination.mkdir(mode=0o700)
    deadline = time.monotonic() + 300
    rows = []
    for row, channels in cases():
        require(time.monotonic() < deadline, "AMI qualification exceeded its wall-time budget")
        _, measured = stereo.fixture_readout(destination, row["case"], channels)
        path = destination / f"{row['case']}.wav"
        rows.append({**row, **measured, **raw_replay(path, channels)})
    require(time.monotonic() < deadline and len(rows) == len(SESSIONS) * len(WINDOWS), "Incomplete AMI qualification")
    stereo.holdout.model_free()
    write(RESULT, {"schema": "chipmunk.audio-ami-native-stereo-qualified/291", "freeze_sha256": sha(FREEZE),
                   "acquisition_sha256": sha(ACQUISITION), "cases": rows, "model_calls": 0})
    return {"status": "qualified", "cases": len(rows), "qualification_sha256": sha(RESULT), "model_calls": 0}


def readout():
    verify()
    exact_files()
    saved = read(RESULT)
    require(saved["schema"] == "chipmunk.audio-ami-native-stereo-qualified/291" and saved["freeze_sha256"] == sha(FREEZE)
            and saved["acquisition_sha256"] == sha(ACQUISITION) and saved["model_calls"] == 0, "Exact completed AMI receipt required")
    counts = Counter()
    for (case, channels), row in zip(cases(), saved["cases"], strict=True):
        path = MEDIA / "qualified" / f"{case['case']}.wav"
        require(all(row[key] == value for key, value in {**case, **raw_replay(path, channels)}.items())
                and row["channel_pcm_sha256"] == [stereo.digest(channel.tobytes()) for channel in channels]
                and row["source_timeline"] == [{"start_sample": 0, "end_sample": SAMPLES, "source_pts": 0}],
                "AMI source, channel, or case timeline differs")
        counts[case["session"]] += 1
    stereo.holdout.model_free()
    return {"schema": "chipmunk.audio-ami-native-stereo-readout/291", "status": "complete", "sessions": len(counts),
            "qualified_by_session": dict(counts), "planned_cases": len(SESSIONS) * len(WINDOWS), "cases_replayed": sum(counts.values()),
            "freeze_sha256": sha(FREEZE), "acquisition_sha256": sha(ACQUISITION), "qualification_sha256": sha(RESULT),
            "model_calls": 0}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = {"plan": plan, "freeze": freeze, "acquire": acquire, "qualify": qualify, "readout": readout}
    parser.add_argument("action", choices=commands)
    print(json.dumps(commands[parser.parse_args().action](), indent=2, sort_keys=True, allow_nan=False))
