"""Rebuild the published bandwidth confirmation from pinned AMI PCM ranges."""

from __future__ import annotations

import argparse
import json
import re
import struct
import sys
from pathlib import Path
from urllib.request import HTTPRedirectHandler, Request, build_opener

if __package__:
    from . import audio_bandwidth_confirmation as trial
else:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import audio_bandwidth_confirmation as trial

ROOT = Path(__file__).resolve().parents[2]
RECEIPT = ROOT / "docs/evaluation-panels/audio-bandwidth-return-confirmation-2026-11-13.json"
MIRROR = "https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus"
# Exact bytes and first 44-byte RIFF header digests of the four parent WAVs.
HEADERS = {
    "ES2007c": (76_077_100, "03b945ca5f13a0f62fefc49edccc7b9ed3b2b2ab6381979ad8716d6fea54585e"),
    "IS1004c": (72_448_785, "ba342b9358bb8aa08c889b51fbd629db3629531b27a00dea6e4f7fef1487dd26"),
    "ES2012b": (71_669_378, "e20e4b90c1dce18363f04f7e7675cf96c395d03f4949f2d83aab8e4b7dd422d0"),
    "IS1001b": (67_741_119, "e4701d6488737da0ccebb68e8a3df70ce9dfce7cda0a9bd527b518559a1f0052"),
}
METHOD_FILES = {
    "runner_sha256": "backend/scripts/audio_bandwidth_confirmation.py",
    "method_sha256": "backend/app/evidence/audio_bandwidth_confirmation.py",
    "bands_sha256": "backend/app/evidence/audio_bandwidth_return.py",
    "baseline_sha256": "backend/app/evidence/audio_continuity.py",
}
PINNED_BASELINE = "77e7ab1cc2cef05bee168584e24824543cf18cb183cfb4ef4664334385de22fd"
PCM_START = 44
RATE = 16_000
SAMPLES = 96_000
PCM_BYTES = SAMPLES * 2
SHA256 = re.compile(r"[0-9a-f]{64}")


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def publisher_range(capture: str, start: int, size: int) -> bytes:
    total, _ = HEADERS[capture]
    stop = start + size - 1
    if not 0 <= start <= stop < total:
        raise ValueError("Requested AMI byte range exceeds its source extent")
    url = f"{MIRROR}/{capture}/audio/{capture}.Array1-01.wav?chipmunk-byte-offset={start}"
    request = Request(url, headers={
        "Range": f"bytes={start}-{stop}",
        "Accept-Encoding": "identity",
        "User-Agent": "Chipmunk-Verify-bandwidth-replay/1",
    })
    with build_opener(NoRedirect()).open(request, timeout=25) as response:
        if (response.status != 206 or response.headers.get("Content-Range") != f"bytes {start}-{stop}/{total}"
                or response.headers.get("Content-Encoding", "identity") != "identity"
                or response.headers.get("Content-Length") not in (None, str(size))):
            raise ValueError(f"AMI did not return the requested byte range for {capture}")
        data = response.read(size + 1)
    if len(data) != size:
        raise ValueError(f"Incomplete AMI payload for {capture}")
    return data


def cached_range(cache, capture, name, start, size, pins, *, acquire):
    path = cache / name
    needed = size // len(pins)

    def verify(data):
        if len(data) != size or any(trial.sha(data[i*needed:(i+1)*needed]) != pin for i, pin in enumerate(pins)):
            raise ValueError(f"Published AMI bytes changed: {name}")

    if path.is_file():
        if path.stat().st_size != size:
            raise ValueError(f"Incorrect cached AMI payload length: {name}")
        data = path.read_bytes()
        verify(data)
        return data
    if not acquire:
        raise ValueError(f"AMI cache file missing: {name}")
    data = publisher_range(capture, start, size)
    verify(data)
    with path.open("xb") as stream:
        stream.write(data)
    return data


def verified_header(cache, capture, *, acquire):
    total, pin = HEADERS[capture]
    header = cached_range(cache, capture, f"{capture}.wavhead", 0, PCM_START, (pin,), acquire=acquire)
    *fields, payload = struct.unpack("<4sI4s4sIHHIIHH4sI", header)
    if (tuple(fields) != (b"RIFF", total-8, b"WAVE", b"fmt ", 16, 1, 1, RATE, RATE*2, 2, 16, b"data")
            or payload % 2 or not 0 < payload <= total - PCM_START):
        raise ValueError(f"Unsupported native AMI WAV layout: {capture}")
    return payload // 2


def start_sample(capture, ordinal, length):
    """Reconstruct the frozen stratum order from the original RIFF extent."""
    begin = max(60 * RATE, ordinal * length // 8)
    end = min(length - 60 * RATE, (ordinal + 1) * length // 8)
    step = 2 * SAMPLES
    starts = range(((begin + step - 1) // step) * step, end - step + 1, step)
    return min(starts, key=lambda tick: trial.sha(f"chipmunk-bandwidth-20261113:{capture}:{tick}".encode()))


def inputs(cache, freeze, *, acquire=False):
    remaining = {row["id"]: row for row in freeze["sources"]}
    if len(remaining) != len(freeze["sources"]) or len(remaining) != 32:
        raise ValueError("Expected 32 unique frozen AMI excerpts")
    for capture in trial.CAPTURES:
        length = verified_header(cache, capture, acquire=acquire)
        for ordinal in range(8):
            name = f"{capture}-{ordinal}"
            source = remaining.pop(name)
            start = start_sample(capture, ordinal, length)
            if source["session"] != capture or source["start_sample"] != start:
                raise ValueError(f"Published source clock differs: {name}")
            pins = (source["pcm_sha256"], source["neighbor_sha256"])
            if not all(SHA256.fullmatch(pin) for pin in pins):
                raise ValueError(f"Published source digest is invalid: {name}")
            data = cached_range(cache, capture, f"{name}.pcm", PCM_START + 2*start,
                                2*PCM_BYTES, pins, acquire=acquire)
            yield source, data[:PCM_BYTES], data[PCM_BYTES:]


def replay(cache: Path, receipt: dict, receipt_sha256: str, *, acquire: bool, measure: bool):
    trial.check(receipt)
    freeze = receipt["freeze"]
    for key, relative_path in METHOD_FILES.items():
        expected = PINNED_BASELINE if key == "baseline_sha256" else freeze[key]
        if trial.file_sha(ROOT / relative_path) != expected:
            raise ValueError(f"Frozen implementation differs: {relative_path}")
    if acquire:
        cache.mkdir(mode=0o700, exist_ok=True)
    roster = list(inputs(cache, freeze, acquire=acquire))
    summary = None
    if measure:
        rows = trial.measure(roster)
        if len(rows) != len(receipt["rows"]):
            raise ValueError("Native replay delivery count differs")
        for actual, published in zip(rows, receipt["rows"], strict=True):
            if actual != published:
                raise ValueError(f"Native replay differs for {actual['id']} {actual['variant']}")
        summary = trial.check({"freeze": freeze, "rows": rows, "summary": trial.report(rows)})
    return {"schema": "chipmunk.audio-bandwidth-public-replay/1", "receipt_sha256": receipt_sha256,
            "source_headers_verified": len(trial.CAPTURES), "pcm_ranges_verified": 2 * len(roster),
            "source_pcm_bytes": len(roster) * 2 * PCM_BYTES, "whole_wav_hashes_checked": False,
            "delivery_rows_reproduced": len(receipt["rows"]) if measure else 0, "summary": summary}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("acquire", "verify-sources", "replay"))
    parser.add_argument("cache", type=Path)
    args = parser.parse_args()
    try:
        data = RECEIPT.read_bytes()
        result = replay(args.cache, json.loads(data), trial.sha(data),
                        acquire=args.phase == "acquire", measure=args.phase == "replay")
    except (KeyError, OSError, TypeError, ValueError) as error:
        parser.exit(1, f"Bandwidth replay failed: {error}\n")
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
