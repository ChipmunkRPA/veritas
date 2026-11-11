"""Qualify the frozen VCD2 recordings with Azure's two ETag formats."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import re
import time
from pathlib import Path

from scripts import video_vcd2_genuine_v195 as previous

TRIAL = previous.ARCHIVE / "video-vcd2-genuine-v197"
PRIOR_PRIVATE_SHA = "525522cd539cd7fad5468177762405bea834673339bcba6b6d53d4be5ce798c9"
PRIOR_PUBLIC_SHA = "00df5acecc6268936b2549aa5e55ddd8d2a4a5eb076e884591ca9ca02fbda7e3"
PRIOR_PUBLIC = previous.ROOT / "docs/evaluation-panels/video-vcd2-genuine-roster-freeze-v195.json"
CODE = (
    "backend/scripts/video_vcd2_genuine_v197.py",
    "backend/tests/test_video_vcd2_genuine_v197.py",
    "docs/video-vcd2-genuine-v197-method.md",
)
ETAG = re.compile(r'(?:0x[0-9a-fA-F]+|"0x[0-9a-fA-F]+")')
BYTE_LENGTH = re.compile(r"[1-9][0-9]{0,8}")
require, sha, write = previous.require, previous.sha, previous.write


def azure_etag(value: str | None) -> str:
    require(value is not None and ETAG.fullmatch(value), "Invalid Azure object ETag")
    return value[1:-1] if value.startswith('"') else value


def context(metadata_dir: Path) -> tuple[dict, list[dict]]:
    prior = previous.TRIAL / "freeze.json"
    require(
        prior.resolve() == prior and prior.is_file() and sha(prior) == PRIOR_PRIVATE_SHA,
        "Exact private v195 freeze required",
    )
    require(
        PRIOR_PUBLIC.is_file() and sha(PRIOR_PUBLIC) == PRIOR_PUBLIC_SHA, "Exact public v195 freeze required"
    )
    inherited = json.loads(prior.read_text(encoding="utf-8"))
    public = json.loads(PRIOR_PUBLIC.read_text(encoding="utf-8"))
    require(
        inherited == previous.context(metadata_dir) and public["private_roster_sha256"] == PRIOR_PRIVATE_SHA,
        "v195 roster or source commitments changed",
    )
    failed = previous.TRIAL / "slot-01-attempt-01"
    require(
        failed.resolve() == failed and failed.is_dir() and not any(failed.iterdir()),
        "Original pre-transfer attempt must remain present and empty",
    )
    cases = inherited["cases"]
    return {
        "schema": "chipmunk.video-vcd2-genuine/197",
        "prior_private_freeze_sha256": PRIOR_PRIVATE_SHA,
        "prior_public_freeze_sha256": PRIOR_PUBLIC_SHA,
        "preserved_pre_transfer_attempt": "slot-01-attempt-01",
        "relative_ordinary_interval_s": inherited["relative_ordinary_interval_s"],
        "recording_components": len(cases),
        "code_sha256": {path: sha(previous.ROOT / path) for path in CODE},
    }, cases


def download(url: str, output: Path) -> dict:
    import httpx

    deadline = time.monotonic() + 180
    with httpx.Client(follow_redirects=False, timeout=25, headers={"Accept-Encoding": "identity"}) as client:
        head = client.head(url)
        head.raise_for_status()
        head_etag = head.headers.get("ETag")
        etag = azure_etag(head_etag)
        length_header = head.headers.get("Content-Length", "")
        require(
            head.status_code == 200
            and BYTE_LENGTH.fullmatch(length_header)
            and int(length_header) <= previous.MAX_BYTES,
            "Missing or out-of-bounds official object length",
        )
        length = int(length_header)
        require(
            head.headers.get("Content-Encoding", "identity") == "identity",
            "Expected exact uncompressed HTTP body",
        )
        head_md5 = head.headers.get("Content-MD5")
        sha256, md5 = hashlib.sha256(), hashlib.md5(usedforsecurity=False)
        size = 0
        condition = f'"{etag}"'
        with client.stream("GET", url, headers={"If-Match": condition}) as response:
            response.raise_for_status()
            get_etag = response.headers.get("ETag")
            response_md5 = response.headers.get("Content-MD5")
            require(
                response.status_code == 200
                and "Content-Range" not in response.headers
                and azure_etag(get_etag) == etag
                and response.headers.get("Content-Length") == length_header
                and response.headers.get("Content-Encoding", "identity") == "identity"
                and response.headers.get("Content-Type") == head.headers.get("Content-Type")
                and (not head_md5 or not response_md5 or head_md5 == response_md5),
                "Official full GET did not match the HEAD object",
            )
            with output.open("xb") as stream:
                for chunk in response.iter_raw(1024 * 1024):
                    size += len(chunk)
                    require(
                        time.monotonic() < deadline and size <= length, "Source transfer exceeded its limits"
                    )
                    sha256.update(chunk)
                    md5.update(chunk)
                    stream.write(chunk)
        require(size == length, "Official source length changed during transfer")
        publisher_md5 = head_md5 or response_md5
        require(
            not publisher_md5 or publisher_md5 == base64.b64encode(md5.digest()).decode("ascii"),
            "Official Content-MD5 disagrees with received bytes",
        )
        return {
            "source_url": url,
            "bytes": size,
            "sha256": sha256.hexdigest(),
            "etag": etag,
            "head_etag": head_etag,
            "get_etag": get_etag,
            "if_match": condition,
            "content_type": head.headers.get("Content-Type"),
            "azure_api_version": head.headers.get("x-ms-version"),
            "publisher_content_md5": publisher_md5,
            "last_modified": head.headers.get("Last-Modified"),
        }


def qualify(record: dict, cases: list[dict], slot: int, attempt: int):
    previous.archive_ready()
    freeze = TRIAL / "freeze.json"
    require(
        freeze.resolve() == freeze
        and freeze.is_file()
        and json.loads(freeze.read_text(encoding="utf-8")) == record,
        "Exact private v197 freeze required",
    )
    require(
        1 <= slot <= len(cases) and 1 <= attempt <= 9,
        "Select one roster slot and an attempt number from 1 to 9",
    )
    source = cases[slot - 1]
    folder = TRIAL / f"slot-{slot:02d}-attempt-{attempt:02d}"
    previous.private_directory(folder)
    partial, file = folder / "source.mkv.partial", folder / "source.mkv"
    acquisition = download(previous.AZURE + source["benchmark_path"], partial)
    require(sha(partial) == acquisition["sha256"], "Stored bytes differ from the received official object")
    partial.rename(file)
    write(folder / "acquisition.json", acquisition)
    verified = previous.media_record(file, source["resolution"])
    require(sha(file) == acquisition["sha256"], "Source changed while the picture clock was verified")
    write(
        folder / "qualification.json",
        {"slot": slot, "attempt": attempt, "sha256": acquisition["sha256"], **verified},
    )
    print(json.dumps({"status": "qualified", "slot": slot, "frames": verified["clock"]["frame_count"]}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "freeze", "qualify"))
    parser.add_argument("--metadata-dir", type=Path, required=True)
    parser.add_argument("--slot", type=int)
    parser.add_argument("--attempt", type=int)
    args = parser.parse_args()
    if (args.command == "qualify") != (args.slot is not None):
        parser.error("--slot is required only for qualify")
    if args.command != "qualify" and args.attempt is not None:
        parser.error("--attempt applies only to qualify")
    record, cases = context(args.metadata_dir)
    if args.command == "preflight":
        print(json.dumps({"status": "prior_freeze_verified", "recording_components": len(cases)}))
    elif args.command == "freeze":
        previous.archive_ready()
        previous.private_directory(TRIAL)
        write(TRIAL / "freeze.json", record)
        print(json.dumps({"status": "frozen", "recording_components": len(cases)}))
    else:
        qualify(record, cases, args.slot, 1 if args.attempt is None else args.attempt)


if __name__ == "__main__":
    main()
