"""Replay the frozen raw MOSS and independent CC0 vocal-event source roster."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import tarfile
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener

SCHEMA = "chipmunk.audio-vocal-event-sources/1"
RECEIPT = (Path(__file__).resolve().parents[2] / "docs" / "evaluation-panels"
           / "audio-vocal-event-source-roster-2026-11-12.json")
DATASET = "https://huggingface.co/datasets/laion/more-synthetic-vocalbursts-raw"
REVISION = "e2c0c2a7b322d8b89cd01c9982062f87eaa7d742"
CC0 = "https://creativecommons.org/publicdomain/zero/1.0/"
ADULTS = {f"{age}_{gender}" for age in ("young", "middle_aged", "mature") for gender in ("woman", "man")}
ITEM_URL = re.compile(r"https://freesound\.org/people/([a-zA-Z0-9_-]+)/sounds/([1-9][0-9]*)/")
ITEM_LICENSE = re.compile(rb'title="Go to the full license text"\s+href="https?://creativecommons\.org/publicdomain/zero/1\.0/"')
MAX_MOSS_BYTES = 5_000_000
MAX_FREESOUND_BYTES = 2_000_000
RETAINED_EVENTS = {
    "laughter": {"Chuckle", "Giggle", "Chortle", "Belly Laugh", "Snort-Laugh", "Titter",
                 "Polite Fake Laugh", "Snicker / Snigger"},
    "breath": {"Contented Sigh", "Relaxing Exhale", "Genuine Yawn", "Polite Covered Yawn", "Investigative Sniff"},
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


class ShardRedirect(HTTPRedirectHandler):
    def __init__(self, lfs_sha256: str) -> None:
        self.lfs_sha256 = lfs_sha256
        self.commit_confirmed = False

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        origin = urlparse(req.full_url)
        target = urlparse(newurl)
        require(not self.commit_confirmed and origin.hostname == "huggingface.co" and target.scheme == "https"
                and target.hostname is not None and target.hostname.endswith((".hf.co", ".huggingface.co")),
                "Unexpected archive redirect")
        require(headers.get("x-repo-commit") == REVISION
                and headers.get("x-linked-etag") == f'"{self.lfs_sha256}"', "Publisher shard commitment differs")
        self.commit_confirmed = True
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def archive_range(shard: dict, start: int, length: int) -> bytes:
    name = shard["path"]
    require(name in {"moss/shard-0000.tar", "moss/shard-0001.tar"}, "Archive path is outside raw MOSS")
    require(start >= 0 and 0 < length <= MAX_MOSS_BYTES and start + length <= shard["bytes"],
            "Archive range exceeds bounds")
    end = start + length - 1
    redirect = ShardRedirect(shard["lfs_sha256"])
    request = Request(f"{DATASET}/resolve/{REVISION}/{name}?download=true", headers={"Range": f"bytes={start}-{end}"})
    with build_opener(redirect).open(request, timeout=25) as response:
        require(response.status == 206 and redirect.commit_confirmed
                and response.headers.get("Content-Range") == f"bytes {start}-{end}/{shard['bytes']}",
                "Publisher did not return the committed byte range")
        raw = response.read(length + 1)
    require(len(raw) == length, "Publisher range size differs")
    return raw


def moss_members(source: dict, shard: dict) -> dict[str, bytes]:
    members = source["members"]
    start = members["json"]["offset"] - 512
    end = members["wav"]["offset"] + members["wav"]["bytes"]
    envelope = archive_range(shard, start, end - start)
    output = {}
    for ext, spec in members.items():
        relative = spec["offset"] - start
        require(relative >= 512 and relative % 512 == 0, "Unaligned tar member")
        info = tarfile.TarInfo.frombuf(envelope[relative - 512:relative], "utf-8", "strict")
        require(info.isfile() and info.name == f"{source['key']}.{ext}" and info.size == spec["bytes"],
                "Publisher tar member header differs")
        output[ext] = envelope[relative:relative + spec["bytes"]]
    return output


def bounded_download(url: str) -> bytes:
    with build_opener().open(url, timeout=20) as response:
        require(response.status == 200 and response.url == url, "Unexpected Freesound response")
        raw = response.read(MAX_FREESOUND_BYTES + 1)
    require(0 < len(raw) <= MAX_FREESOUND_BYTES, "Freesound response exceeded bounds")
    return raw


def ordinary_audio(source: dict) -> bytes:
    item = ITEM_URL.fullmatch(source["item_url"])
    require(item is not None and item.groups() == (source["contributor"], source["item_id"])
            and source["license_url"] == CC0, "Unrecognized control item rights or identity")
    preview = source["preview_url"]
    require(re.fullmatch(r"https://cdn\.freesound\.org/previews/[0-9]+/"
                         + source["item_id"] + r"_[0-9]+-hq\.mp3", preview) is not None, "Unrecognized control preview")
    page = bounded_download(source["item_url"])
    require(ITEM_LICENSE.search(page) is not None and preview.encode() in page,
            "Freesound page no longer declares the committed item license and preview")
    return bounded_download(preview)


def committed(path: Path, spec: dict, cap: int, fetched: bytes | None = None) -> bytes:
    require(0 < spec["bytes"] <= cap, f"Invalid size commitment for {path.name}")
    if path.exists():
        require(path.stat().st_size == spec["bytes"], f"Size differs for {path.name}")
        raw = path.read_bytes()
    elif fetched is not None:
        raw = fetched
    else:
        raise FileNotFoundError(path)
    require(len(raw) == spec["bytes"] and sha256(raw) == spec["sha256"], f"Bytes differ for {path.name}")
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(raw)
    return raw


def verify_audio(path: Path, expected: dict) -> None:
    probe = subprocess.run(["ffprobe", "-v", "error", "-protocol_whitelist", "file", "-select_streams", "a", "-show_entries",
                            "stream=codec_name,sample_rate,channels,duration", "-of", "json", str(path)],
                           capture_output=True, check=True, timeout=15)
    streams = json.loads(probe.stdout)["streams"]
    fields = ("codec_name", "sample_rate", "channels", "duration")
    require(len(streams) == 1 and all(streams[0].get(field) == expected[field] for field in fields),
            f"Decoded audio geometry differs for {path.name}")
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-xerror", "-protocol_whitelist", "file", "-i", str(path),
                    "-map", "0:a:0", "-f", "null", "-"], capture_output=True, check=True, timeout=15)


def replay(run: Path, receipt: dict, fetch: bool) -> None:
    require(receipt["schema"] == SCHEMA and receipt["dataset"]["revision"] == REVISION, "Unrecognized source receipt")
    for source in receipt["synthetic"]:
        key = source["key"]
        require(re.fullmatch(r"[0-9]{4}", key) is not None and set(source["members"]) == {"json", "wav"},
                "Unrecognized raw MOSS members")
        paths = {ext: run / "moss" / f"{key}.{ext}" for ext in source["members"]}
        fetched = moss_members(source, receipt["dataset"]["shards"][source["shard"]]) if (
            fetch and not all(path.exists() for path in paths.values())) else {}
        sidecar = json.loads(committed(paths["json"], source["members"]["json"], 2048, fetched.get("json")))
        require(sidecar["model"] == "MOSS-SoundEffect-v2.0" and sidecar["id"] == source["publisher_id"]
                and sidecar["age_group"] == source["prompt_age_group"] and sidecar["age_group"] in ADULTS
                and sidecar["vocal_burst_key"] == source["event"]
                and source["event"] in RETAINED_EVENTS[source["family"]], "Raw sidecar is outside roster selection")
        committed(paths["wav"], source["members"]["wav"], MAX_MOSS_BYTES, fetched.get("wav"))
        verify_audio(paths["wav"], source["audio"])
    for source in receipt["ordinary"]:
        item = source["item_id"]
        require(re.fullmatch(r"[1-9][0-9]*", item) is not None, "Invalid Freesound item number")
        path = run / "ordinary" / f"{item}.mp3"
        fetched = ordinary_audio(source) if fetch and not path.exists() else None
        committed(path, source["preview"], MAX_FREESOUND_BYTES, fetched)
        verify_audio(path, source["audio"])
    print(f"Verified {len(receipt['synthetic'])} raw MOSS sources and {len(receipt['ordinary'])} ordinary control previews")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("fetch", "verify"))
    parser.add_argument("run", type=Path, help="Directory for ignored audio source media")
    parser.add_argument("--receipt", type=Path, default=RECEIPT)
    args = parser.parse_args()
    try:
        replay(args.run, json.loads(args.receipt.read_bytes()), args.command == "fetch")
    except HTTPError as error:
        parser.exit(1, f"Publisher HTTP request failed ({error.code})\n")
    except URLError:
        parser.exit(1, "Publisher network request failed\n")


if __name__ == "__main__":
    main()
