"""Rebuild the published matched audio waveforms from bounded public publisher downloads."""

from __future__ import annotations

import argparse
import struct
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import Request, urlopen
from xml.etree import ElementTree
from zipfile import ZipFile

import numpy as np

from scripts import audio_odss_matched_room as panel

RECEIPT = panel.ROOT / "docs/evaluation-panels/audio-odss-matched-room-2026-11-13.json"
AMI_MIRROR = "https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus"
ANNOTATION = "https://groups.inf.ed.ac.uk/ami/AMICorpusAnnotations/ami_public_manual_1.6.2.zip"
ANNOTATION_BYTES = 22_887_865
ODSS = "https://zenodo.org/records/8370669/files/odss.zip"
SOURCE = {
    "IS1001b": (67_741_119, "e4701d6488737da0ccebb68e8a3df70ce9dfce7cda0a9bd527b518559a1f0052"),
    "ES2012b": (71_669_378, "e20e4b90c1dce18363f04f7e7675cf96c395d03f4949f2d83aab8e4b7dd422d0"),
}
ALLOWED_HOSTS = {"zenodo.org", "groups.inf.ed.ac.uk", "raw.githubusercontent.com"}


def response_bytes(url, size, *, start=None, total=None):
    headers = {"Accept-Encoding": "identity", "User-Agent": "Chipmunk-Verify-matched-speech-replay/1"}
    expected = None
    if start is not None:
        stop = start + size - 1
        panel.require(total is not None and 0 <= start <= stop < total, "Invalid publisher byte range")
        headers["Range"] = f"bytes={start}-{stop}"
        expected = f"bytes {start}-{stop}/{total}"
    request = Request(url, headers=headers)
    with urlopen(request, timeout=25) as response:
        location = urlsplit(response.geturl())
        panel.require(location.scheme == "https" and location.hostname in ALLOWED_HOSTS
                      and response.status == (200 if expected is None else 206)
                      and response.headers.get("Content-Encoding", "identity") == "identity"
                      and (expected is None or response.headers.get("Content-Range") == expected),
                      "Publisher did not return the exact requested payload")
        reported = response.headers.get("Content-Length")
        panel.require(reported is None or reported == str(size), "Publisher response length disagrees")
        data = response.read(size + 1)
    panel.require(len(data) == size, "Publisher payload is incomplete")
    return data


def committed(path, data, sha=None):
    if sha is not None:
        panel.require(panel.digest(data) == sha, f"Publisher bytes changed: {path.name}")
    with path.open("xb") as stream:
        stream.write(data)


def odss_range(start, size):
    return response_bytes(f"{ODSS}?download=1&chipmunk-offset={start}&chunk={size}",
                          size, start=start, total=panel.odss.ARCHIVE_BYTES)


def odss_sources(cache):
    # Only the published central directory, notice, transcript metadata and roster members are read.
    path = cache / "directory.bin"
    if not path.exists():
        first = 2 * 1024**2
        payload = odss_range(panel.odss.DIRECTORY_OFFSET, first)
        payload += odss_range(panel.odss.DIRECTORY_OFFSET + first, panel.odss.DIRECTORY_BYTES - first)
        committed(path, payload, panel.odss.DIRECTORY_SHA256)
    previous = panel.ranged
    try:
        panel.ranged = odss_range
        return panel.plan(cache, True)
    finally:
        panel.ranged = previous


def wav_header(blob, name):
    expected_length, pin = SOURCE[name]
    panel.require(len(blob) == 44 and panel.digest(blob) == pin, "AMI header commitment changed")
    riff, reported, wave, fmt, fmt_size, codec, channels, rate, rate_bytes, block, bits, data, payload = (
        struct.unpack("<4sI4s4sIHHIIHH4sI", blob))
    panel.require((riff, reported, wave, fmt, fmt_size, codec, channels, rate, rate_bytes, block, bits, data) ==
                  (b"RIFF", expected_length - 8, b"WAVE", b"fmt ", 16, 1, 1, 16000, 32000, 2, 16, b"data")
                  and payload % block == 0 and 44 + payload <= expected_length, "Unexpected AMI native PCM layout")
    return payload


def manifest(annotations, roster):
    panel.require(panel.file_digest(annotations) == panel.ami.ANNOTATION_SHA256, "AMI manual annotation archive differs")
    rows = {}
    with ZipFile(annotations) as archive:
        document = ElementTree.fromstring(archive.read("corpusResources/meetings.xml"))
        people = {node.get("observation"): {person.get("global_name") for person in node if person.get("global_name")}
                  for node in document.iter() if node.get("observation")}
        excluded = set().union(*(people[meeting] for meeting in panel.PREVIOUS_AMI))
        for role, (meeting, _) in panel.AMI.items():
            panel.require(len(people[meeting]) == 4 and not people[meeting] & excluded, "AMI participant roles overlap")
            excluded.update(people[meeting])
            tracks = [panel.ami.track_words(archive.read(f"words/{meeting}.{channel}.words.xml")) for channel in "ABCD"]
            length = max(end for track in tracks for _, end in track)
            voices = np.zeros(length, dtype=np.int32)
            for track in tracks:
                active = np.zeros(length, dtype=bool)
                for begin, end in track:
                    active[begin:end] = True
                voices += active
            opportunities = [(tick, int(np.count_nonzero(voices[tick + 150:tick + 450] >= 2)))
                             for tick in range(500, length - 600, 25) if np.count_nonzero(voices[tick:tick + 600]) >= 480]
            accepted = []
            for row in roster:
                if row["role"] != role:
                    continue
                case = row["case"]
                tick, overlap = min((item for item in opportunities if item[1] >= 100
                                     and all(abs(item[0] - other) >= 600 for other in accepted)),
                                    key=lambda item: panel.rank(f"{case}:{item[0]}"))
                accepted.append(tick)
                rows[case] = {"meeting": meeting, "start_sample": tick * 160, "overlap_centiseconds": overlap}
    panel.require(len(rows) == 16, "AMI schedule differs")
    return rows


def backgrounds(ami_dir, rows, *, download):
    samples, private = {}, {}
    bounds = {}
    for name, (size, _) in SOURCE.items():
        header = ami_dir / f"{name}.wavhead"
        url = f"{AMI_MIRROR}/{name}/audio/{name}.Array1-01.wav"
        if download and not header.exists():
            committed(header, response_bytes(f"{url}?chipmunk-byte-offset=0", 44, start=0, total=size))
        bounds[name] = wav_header(header.read_bytes(), name)
    for case, row in rows.items():
        name, start = row["meeting"], row["start_sample"]
        offset, size = 44 + 2 * start, 2 * panel.SAMPLES
        path = ami_dir / f"{case}.pcm"
        panel.require(offset + size <= 44 + bounds[name], "AMI excerpt is outside PCM data")
        if download and not path.exists():
            url = f"{AMI_MIRROR}/{name}/audio/{name}.Array1-01.wav?chipmunk-byte-offset={offset}"
            committed(path, response_bytes(url, size, start=offset, total=SOURCE[name][0]))
        body = path.read_bytes()
        panel.require(len(body) == size, "AMI excerpt length differs")
        samples[case] = np.frombuffer(body, dtype="<i2")
        private[case] = {**row, "pcm_sha256": panel.digest(body)}
    return private, samples


def download_assets(cache):
    odss, ami, surrey, bbc = (cache / name for name in ("odss", "ami", "surrey", "bbc"))
    for path in (odss, ami, surrey, bbc):
        path.mkdir(mode=0o700, exist_ok=True)
    roster, _ = odss_sources(odss)
    annotation = ami / "ami_public_manual_1.6.2.zip"
    if not annotation.exists():
        committed(annotation, response_bytes(ANNOTATION, ANNOTATION_BYTES), panel.ami.ANNOTATION_SHA256)
    backgrounds(ami, manifest(annotation, roster), download=True)
    for directory, assets in ((surrey, (panel.surrey.SURREY_ASSET,)), (bbc, panel.bbc.ASSETS.values())):
        for asset in assets:
            path = directory / asset.filename
            if not path.exists():
                committed(path, response_bytes(asset.url, asset.bytes))
            panel.surrey.pinned_bytes(directory, asset)


def replay(cache, expected):
    lock = expected["freeze"]
    roster, dry = panel.plan(cache / "odss", False)
    panel.require(panel.digest(panel.json_bytes(roster)) == lock["private_source_roster_sha256"], "Speech source roster differs")
    annotation = cache / "ami/ami_public_manual_1.6.2.zip"
    schedules = manifest(annotation, roster)
    masker_roster, masker = backgrounds(cache / "ami", schedules, download=False)
    panel.require(panel.digest(panel.json_bytes(masker_roster)) == lock["private_background_roster_sha256"], "AMI excerpts differ")
    rooms, assets = panel.room_impulses(cache / "surrey", cache / "bbc")
    panel.require(assets == lock["room_assets"], "Measured room bytes differ")
    originals = {row["id"]: {key: value for key, value in row.items() if key != "responses"}
                 for role in ("development", "assessment") for row in expected[role]["rows"]}
    compared, noisy, channels = 0, 0, 0
    a, b = panel.WINDOW
    for entry in roster:
        case = entry["case"]
        for origin, level, stereo, target, measured, extent, position in panel.room_views(entry, dry, masker[case], rooms):
            key = f"{case}:{origin}:{'room' if level is None else level}"
            views, delta, gains = panel.wave_views(stereo, target, level)
            bindings = {name: None if pcm is None else panel.digest(pcm[a:b].tobytes()) for name, pcm in views.items()}
            row = {"id": key, "case": case, "group": entry["group"], "role": entry["role"], "origin": origin,
                   "level_db": level, "measured_snr_db": None if measured is None else round(measured, 6),
                   "input_sha256": panel.digest(stereo.tobytes()), "window_hashes": bindings, "acoustic_gain_db": gains,
                   "fft_max_lsb": delta, "noise_position": position, "utterance_extent": None if extent is None else list(extent)}
            panel.require(row == originals.pop(key, None), f"Published input or view differs: {key}")
            compared += 1
            noisy += level is not None
            channels += len(views)
    panel.require(not originals and compared == 176, "Full delivery comparison is incomplete")
    entries = [path for path in cache.glob("*/*") if path.is_file()]
    return {"schema": "chipmunk.audio-odss-public-waveform-replay/1", "source": "bounded publisher downloads",
            "original_receipt_sha256": panel.file_digest(RECEIPT), "stereo_deliveries": compared,
            "noisy_deliveries": noisy, "comparison_view_slots": channels, "odss_wav_members": len(dry),
            "ami_pcm_excerpts": len(masker), "cached_bytes": sum(path.stat().st_size for path in entries),
            "files": len(entries), "network_required_for_replay": False, "voice_classifier_forwards": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("acquire", "replay"))
    parser.add_argument("cache", type=Path)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    if args.phase == "acquire":
        args.cache.mkdir(mode=0o700, exist_ok=True)
        download_assets(args.cache)
    result = replay(args.cache, panel.read(RECEIPT))
    if args.receipt:
        panel.write(args.receipt, result)
    print(panel.json_bytes(result).decode(), end="")


if __name__ == "__main__":
    main()
