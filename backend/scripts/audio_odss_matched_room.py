"""Compare speech views using exact text-matched ODSS utterances in two rooms."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import re
import time
import wave
import zipfile
import zlib
from collections import Counter, defaultdict
from pathlib import Path
from urllib.request import Request, urlopen
from xml.etree import ElementTree

import numpy as np
from scipy.signal import fftconvolve

from scripts import audio_bbc_room_assets_v303 as bbc
from scripts import audio_odss_inventory_v297 as odss
from scripts import audio_room_assets_v299 as surrey
from scripts import audio_wpe_room_transfer as wpe
from scripts import prepare_audio_specialists_v15 as models
from scripts import prepare_natural_overlap_transfer as ami

SCHEMA = "chipmunk.audio-odss-matched-room/1"
ROOT = Path(__file__).resolve().parents[2]
RATE, SAMPLES = 16000, 96000
WINDOW = (24000, 72000)
LEVELS = (0, -5)
PROMPTS = 2
MAX_MEMBER = 800_000
TRANSCRIPTS = "odss/transcripts.psv"
TRANSCRIPTS_SHA = "3edf8a4978111f696bfa896f420e5d526049e6cdc21693b30ae6108c16eb32f7"
AMI = {"development": ("IS1001b", "164dda162e5e8e851cb56b629116e916835d571b6b4cdf3d164337297b0a853d"),
       "assessment": ("ES2012b", "dd6f2caa7dc2ca8357964efa4d252a025151e11de4ce496b3b12d89bfd6464ba")}
PREVIOUS_AMI = ("ES2004a", "IS1009a", "ES2010d", "IS1007a", "ES2005a", "ES2008a", "IS1004c", "ES2007c")
METHOD = {
    "sources": "Freeze two exact ODSS triplets per publisher speaker. The complete native words fit the same three-second interval; no splices, crops, speed changes or newly generated speech. Match both generators and human by ZIP member suffix and pinned publisher transcript. Deterministically split four speakers into development and four into assessment. Exclude any shared normalized prompt across the whole panel.",
    "rendering": "Development uses Surrey and IS1001b, assessment uses BBC and ES2012b. Front foreground, alternating left/right competing source. The complete speech is centered in a fixed three-second interval of a six-second stereo mixture. The same annotated competing speech, absolute gain and native rendering apply to all three origins of each prompt. Normalize the target room mean energy of each origin to a common level. Render room-only, 0 and -5 dB mixture, and matched background-only controls. Scale all views together before signed integer addition.",
    "scores": "Use the existing voice model and speech admission on the fixed three-second range for native mono, blind WPE, and a frontal mid-side canceller. Fit one threshold per arm using admitted noisy human responses only. Each primary point is the maximum over four development speakers, and the sensitivity point allows one development speaker. Compare all noisy held true and false results by origin and speaker, plus matched background-only, availability and acoustic SNR. Room-only mono is diagnostic.",
    "frontal": "The mid-side canceller assumes a centered desired source. The existing fixed-clock WPE transform supplies the STFT. Regress complex mid on side within each frequency bin over the full six-second upload, using 10% of median side spectral energy as a floor and capping complex gain magnitude at two. Keep frequencies below 150 Hz unchanged. Subtract 0.7 of the fitted component before overlap-add. Reject clipped views and outputs under 2% of original mono energy.",
}


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def file_digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def rank(value):
    return digest(f"{SCHEMA}:{value}".encode())


def json_bytes(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def write(path, value):
    with path.open("xb") as output:
        output.write(json_bytes(value))


def read(path):
    return json.loads(path.read_bytes())


def ranged(start, size):
    require(0 < size <= 2 * 1024**2 and start >= 0 and start + size <= odss.ARCHIVE_BYTES, "Bounded publisher range required")
    end = start + size - 1
    request = Request(odss.URL, headers={"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity",
                                        "User-Agent": "Chipmunk-Verify-transfer-evaluation"})
    with urlopen(request, timeout=25) as response:
        require(response.status == 206 and response.headers.get("Content-Range") ==
                f"bytes {start}-{end}/{odss.ARCHIVE_BYTES}" and response.headers.get("Content-Encoding", "identity") == "identity",
                "Publisher response must provide the exact unencoded bounded range")
        body = response.read(size + 1)
    require(len(body) == size, "Truncated bounded publisher range")
    return body


def catalog(cache, download):
    path = cache / "directory.bin"
    if download and not path.exists():
        size = 2 * 1024**2
        body = ranged(odss.DIRECTORY_OFFSET, size) + ranged(odss.DIRECTORY_OFFSET + size, odss.DIRECTORY_BYTES - size)
        require(digest(body) == odss.DIRECTORY_SHA256, "Pinned ODSS ZIP directory differs")
        path.write_bytes(body)
    body = path.read_bytes()
    require(len(body) == odss.DIRECTORY_BYTES and digest(body) == odss.DIRECTORY_SHA256, "Pinned ODSS ZIP directory differs")
    footer = odss.FOOTER.pack(b"PK\x05\x06", 0, 0, odss.ENTRIES, odss.ENTRIES, odss.DIRECTORY_BYTES, 0, 0)
    with zipfile.ZipFile(io.BytesIO(body + footer)) as archive:
        entries = {entry.filename: entry for entry in archive.infolist()}
    require(len(entries) == odss.ENTRIES, "ODSS catalog has duplicate members")
    return entries


def decode_member(entry, encoded):
    signature, _, flags, method, *_, name_size, extra_size = odss.HEADER.unpack_from(encoded)
    name = entry.filename.encode("ascii")
    start = odss.HEADER.size + name_size + extra_size
    require(signature == b"PK\x03\x04" and flags == entry.flag_bits and not flags & 1 and method == entry.compress_type
            and method in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED) and name_size == len(name) and extra_size <= 1024
            and encoded[odss.HEADER.size:odss.HEADER.size + name_size] == name and start + entry.compress_size <= len(encoded),
            "Local ODSS member header and catalog disagree")
    stored = encoded[start:start + entry.compress_size]
    body = stored
    if method == zipfile.ZIP_DEFLATED:
        decoder = zlib.decompressobj(-15)
        body = decoder.decompress(stored, entry.file_size + 1)
        require(decoder.eof and not decoder.unused_data, "Complete single member stream required")
    require(len(body) == entry.file_size and zlib.crc32(body) == entry.CRC, "ODSS member CRC/length differs")
    return body


def member(cache, entry, download):
    require(0 < entry.file_size <= MAX_MEMBER and 0 < entry.compress_size <= MAX_MEMBER, "ODSS member exceeds bound")
    path = cache / f"{entry.header_offset}.range"
    if download and not path.exists():
        size = odss.HEADER.size + len(entry.filename) + 1024 + entry.compress_size
        require(entry.header_offset + size <= odss.DIRECTORY_OFFSET + 1024, "Local header range overruns catalog")
        encoded = ranged(entry.header_offset, size)
        body = decode_member(entry, encoded)
        path.write_bytes(encoded)
        return body
    return decode_member(entry, path.read_bytes())


def native_wav(body):
    with wave.open(io.BytesIO(body), "rb") as stream:
        count = stream.getnframes()
        require((stream.getnchannels(), stream.getsampwidth(), stream.getframerate(), stream.getcomptype()) == (1, 2, RATE, "NONE")
                and int(1.65 * RATE) <= count <= WINDOW[1] - WINDOW[0], "Expected full 1.65 to 3 second mono ODSS utterance")
        pcm = stream.readframes(count)
        require(len(pcm) == 2 * count, "ODSS native sample length differs")
    return pcm


def plan(cache, download):
    entries = catalog(cache, download)
    trans = member(cache, entries[TRANSCRIPTS], download)
    require(digest(trans) == TRANSCRIPTS_SHA, "Pinned ODSS transcripts differ")
    texts = defaultdict(set)
    for row in csv.DictReader(io.StringIO(trans.decode("utf-8-sig")), delimiter="|"):
        texts[row["id"]].add(row["text"])
    notices = {entry: digest(member(cache, entries[entry], download)) for entry in odss.METADATA}
    require(notices == odss.METADATA, "Publisher license or speaker manifest differs")
    grouped = defaultdict(list)
    for name, entry in entries.items():
        match = re.fullmatch(r"odss/natural/hifi-tts/(\d+)/([^/]+)\.wav", name)
        if not match:
            continue
        speaker, prompt = match.groups()
        peers = [entries.get(name.replace("/natural/", f"/{origin}/", 1)) for origin in odss.ORIGINS]
        # WAV header allowance keeps every selected member away from duration edges.
        if len(texts[prompt]) == 1 and all(peer is not None and 53100 <= peer.file_size <= 95800 for peer in peers):
            text, = texts[prompt]
            normalized = " ".join(re.findall(r"\w+", text.casefold()))
            if len(normalized.split()) >= 4:
                grouped[speaker].append((prompt, normalized, peers))
    require(len(grouped) == 8 and min(map(len, grouped.values())) >= PROMPTS, "Too few exact eligible speaker groups")
    roster, source_pcm, used_texts, used_content = [], {}, set(), set()
    speakers = sorted(grouped, key=rank)
    for index, speaker in enumerate(speakers):
        role = "development" if index < len(speakers) // 2 else "assessment"
        group = f"reader-{index:02d}"
        picked = 0
        for prompt, text, peers in sorted(grouped[speaker], key=lambda row: rank(f"{speaker}:{row[0]}")):
            if text in used_texts:
                continue
            item = f"{group}-prompt-{picked}"
            members = {}
            for origin, entry in zip(odss.ORIGINS, peers, strict=True):
                body = member(cache, entry, download)
                pcm = native_wav(body)
                pin = digest(pcm)
                require(pin not in used_content, "ODSS native PCM aliases a source")
                used_content.add(pin)
                source_pcm[item, origin] = np.frombuffer(pcm, dtype="<i2")
                members[origin] = {"name": entry.filename, "wav_sha256": digest(body), "pcm_sha256": pin, "samples": len(pcm) // 2}
            roster.append({"case": item, "group": group, "speaker": speaker, "role": role, "prompt": prompt,
                           "prompt_sha256": digest(text.encode()), "members": members})
            used_texts.add(text)
            picked += 1
            if picked == PROMPTS:
                break
        require(picked == PROMPTS, "Complete speaker prompt schedule required")
    return roster, source_pcm


def room_impulses(surrey_dir, bbc_dir):
    data, identity = surrey.pinned_bytes(surrey_dir, surrey.SURREY_ASSET)
    from_surrey = surrey.surrey_impulses(data)
    inputs = {"development": {label: from_surrey[member] for label, member in
                              (("front", "FC"), ("left", "SiL"), ("right", "SiR"))}, "assessment": {}}
    assets = {"development": [identity], "assessment": []}
    for position, asset in bbc.ASSETS.items():
        data, identity = surrey.pinned_bytes(bbc_dir, asset)
        impulse, _ = bbc.parse_impulse(data, position)
        inputs["assessment"][position] = impulse
        assets["assessment"].append(identity)
    return inputs, assets


def backgrounds(ami_dir, roster):
    annotation = ami_dir / "ami_public_manual_1.6.2.zip"
    require(file_digest(annotation) == ami.ANNOTATION_SHA256, "Pinned AMI word annotations differ")
    positions, signals = {}, {}
    with zipfile.ZipFile(annotation) as archive:
        meeting_data = ElementTree.fromstring(archive.read("corpusResources/meetings.xml"))
        people = {node.get("observation"): {child.get("global_name") for child in node if child.get("global_name")}
                  for node in meeting_data.iter() if node.get("observation")}
        excluded = set().union(*(people[key] for key in PREVIOUS_AMI))
        for role, (meeting, pin) in AMI.items():
            require(len(people[meeting]) == 4 and not people[meeting] & excluded, "AMI participant roles overlap")
            excluded.update(people[meeting])
            path = ami_dir / f"{meeting}.Array1-01.wav"
            require(file_digest(path) == pin, "Pinned AMI recording differs")
            tracks = [ami.track_words(archive.read(f"words/{meeting}.{channel}.words.xml")) for channel in "ABCD"]
            occupancy = np.zeros((len(tracks), max(b for track in tracks for _, b in track)), dtype=bool)
            for row, track in zip(occupancy, tracks, strict=True):
                for start, end in track:
                    row[start:end] = True
            speakers = occupancy.sum(axis=0)
            possible = []
            for start in range(500, len(speakers) - 600, 25):
                cover = int(np.count_nonzero(speakers[start + 150:start + 450] >= 2))
                if cover >= 100 and np.count_nonzero(speakers[start:start + 600]) >= 480:
                    possible.append((start, cover))
            selected = []
            with wave.open(str(path), "rb") as stream:
                require((stream.getnchannels(), stream.getsampwidth(), stream.getframerate(), stream.getcomptype()) ==
                        (1, 2, RATE, "NONE"), "Expected native mono AMI recording")
                for row in (row for row in roster if row["role"] == role):
                    case = row["case"]
                    start, cover = min((item for item in possible if all(abs(item[0] - x) >= 600 for x in selected)),
                                       key=lambda item: rank(f"{case}:{item[0]}"))
                    selected.append(start)
                    start *= 160
                    stream.setpos(start)
                    pcm = stream.readframes(SAMPLES)
                    require(len(pcm) == 2 * SAMPLES, "AMI excerpt is truncated")
                    positions[case] = {"meeting": meeting, "start_sample": start, "overlap_centiseconds": cover,
                                       "pcm_sha256": digest(pcm)}
                    signals[case] = np.frombuffer(pcm, dtype="<i2")
    return positions, signals


def convolve(signal, impulse):
    require(impulse.delay == (0, 0) and impulse.rate == 48000, "Expected publisher embedded-delay 48 kHz impulse")
    impulse.identity()
    expanded = surrey.room.resample(signal.astype(np.float64), RATE, impulse.rate)
    rendered = fftconvolve(expanded[:, None], impulse.coefficients, axes=0)[:len(expanded)]
    result = surrey.room.resample(rendered, impulse.rate, RATE)
    require(result.shape == (SAMPLES, 2) and np.isfinite(result).all(), "Unexpected rendered clock or samples")
    return result


def padded(pcm):
    start = WINDOW[0] + (WINDOW[1] - WINDOW[0] - len(pcm)) // 2
    end = start + len(pcm)
    return np.pad(pcm, (start, SAMPLES - end)), (start, end)


def rms(signal):
    return float(np.sqrt(np.mean(signal ** 2)))


def room_views(row, dry, masker, rooms):
    case, role = row["case"], row["role"]
    position = "left" if case.endswith("-0") else "right"
    spectra, extents = {}, {}
    for origin in odss.ORIGINS:
        padded_pcm, extents[origin] = padded(dry[case, origin])
        signal = convolve(padded_pcm, rooms[role]["front"])
        spectra[origin] = signal / rms(signal.mean(axis=1))
    noise = convolve(masker, rooms[role][position])
    noise /= rms(noise.mean(axis=1))
    maximum = max(float(np.max(abs(target) + abs(noise) * 10 ** (-min(LEVELS) / 20))) for target in spectra.values())
    scale = 24000 / maximum
    residuals = {level: np.rint(noise * scale * 10 ** (-level / 20)).astype(np.int32) for level in LEVELS}
    for origin, signal in spectra.items():
        target = np.rint(signal * scale).astype(np.int32)
        yield origin, None, target.astype("<i2"), target, None, extents[origin], position
        for level, noise_pcm in residuals.items():
            measured = 20 * np.log10(rms(target.mean(axis=1)) / rms(noise_pcm.mean(axis=1)))
            require(abs(measured - level) <= .02 and np.max(abs(target + noise_pcm)) <= 24001, "Rendered level or peak differs")
            yield origin, level, (target + noise_pcm).astype("<i2"), target, float(measured), extents[origin], position
    for level, noise_pcm in residuals.items():
        yield "background", level, noise_pcm.astype("<i2"), None, None, None, position


def frontal(stereo, fft=np.fft.rfft, inverse=np.fft.irfft):
    waveform = stereo.astype(np.float64)
    spectrum, window = wpe.transform(waveform, fft)
    mid = spectrum.mean(axis=2)
    side = (spectrum[:, :, 0] - spectrum[:, :, 1]) * 0.5
    power = np.mean(abs(side) ** 2, axis=1)
    floor = max(float(np.median(power)) * 0.1, 1e-12)
    gain = np.mean(mid * side.conj(), axis=1) / (power + floor)
    gain *= np.minimum(1, 2 / np.maximum(abs(gain), 1e-12))
    gain[np.fft.rfftfreq(wpe.FRAME, 1 / RATE) < 150] = 0
    candidate = np.rint(wpe.restore(mid - 0.7 * gain[:, None] * side, window, inverse))
    require(np.isfinite(candidate).all(), "Nonfinite frontal residual")
    if np.max(abs(candidate)) >= 32768 or np.mean(candidate ** 2) < .02 * np.mean(waveform.mean(axis=1) ** 2):
        return None
    return candidate.astype("<i2")


def wave_views(stereo, target, level):
    from scipy.fft import irfft, rfft

    mono = np.rint(stereo.astype(np.float64).mean(axis=1)).astype("<i2")
    result = {"mono": mono}
    if level is None:
        return result, {}, None
    result.update(wpe=wpe.wpe(stereo), frontal=frontal(stereo))
    alternates = {"wpe": wpe.wpe(stereo, fft=np.fft.rfft, inverse=np.fft.irfft), "frontal": frontal(stereo, rfft, irfft)}
    limits = {}
    for name, alternate in alternates.items():
        actual = result[name]
        require((actual is None) == (alternate is None), "FFT replay changed view availability")
        limits[name] = 0 if actual is None else int(np.max(abs(actual.astype(np.int32) - alternate.astype(np.int32))))
        require(limits[name] <= 1, "FFT replay differs by more than one PCM unit")
    improvement = None
    if target is not None:
        a, b = WINDOW
        reference = target[a:b].mean(axis=1)
        improvement = {name: wpe.signal_gain(values[a:b], mono[a:b], reference) if values is not None else None
                       for name, values in result.items() if name != "mono"}
    return result, limits, improvement


def prepare(args):
    roster, dry = plan(args.cache, False)
    rooms, assets = room_impulses(args.surrey, args.bbc)
    background, masker = backgrounds(args.ami, roster)
    received = {}
    private_roster = []
    for row in roster:
        case = row["case"]
        for origin, level, stereo, target, measured, extent, position in room_views(row, dry, masker[case], rooms):
            key = f"{case}:{origin}:{'room' if level is None else level}"
            views, delta, gains = wave_views(stereo, target, level)
            bind = {name: digest(pcm[WINDOW[0]:WINDOW[1]].tobytes()) if pcm is not None else None for name, pcm in views.items()}
            received[key] = views
            private_roster.append({"id": key, "case": case, "group": row["group"], "role": row["role"], "origin": origin,
                                   "level_db": level, "measured_snr_db": None if measured is None else round(measured, 6),
                                   "input_sha256": digest(stereo.tobytes()), "window_hashes": bind, "acoustic_gain_db": gains,
                                   "fft_max_lsb": delta, "noise_position": position, "utterance_extent": extent})
    lock = {"schema": SCHEMA, "method": METHOD, "windows": list(WINDOW), "clock_hz": RATE, "stereo_samples": SAMPLES,
            "levels_db": LEVELS, "code_sha256": file_digest(Path(__file__)), "wpe_sha256": file_digest(Path(wpe.__file__)),
            "odss_catalog_sha256": odss.DIRECTORY_SHA256, "odss_transcripts_sha256": TRANSCRIPTS_SHA, "odss_notice_sha256": odss.METADATA,
            "speech_assets": models.ASSETS, "room_assets": assets, "native_background": AMI, "background_annotations": ami.ANNOTATION_SHA256,
            "private_source_roster_sha256": digest(json_bytes(roster)), "private_background_roster_sha256": digest(json_bytes(background)),
            "delivery_roster_sha256": digest(json_bytes(private_roster))}
    return lock, private_roster, received


def score(rows, received, role, weights):
    from app.detectors.audio.calibration import logit
    from app.detectors.audio.content import window_in_speech
    from app.detectors.audio.upgraded import SpeechActivity
    from scripts.audio_strong_traffic_view import network_free
    from scripts.run_audio_specialists_v15 import load_voice

    network_free()
    voice = load_voice({"weights": str(weights), "assets": models.ASSETS})
    gate = SpeechActivity(weights / "silero_vad.jit")
    counts = Counter()
    result = []
    started = time.monotonic()
    a, b = WINDOW
    for row in rows:
        if row["role"] != role:
            continue
        responses = {}
        for arm, pcm in received[row["id"]].items():
            response = None
            if pcm is not None:
                waveform = pcm.astype(np.float32) / 32768
                if window_in_speech(gate.spans(waveform), a / RATE, b / RATE):
                    probability, = voice.predict([waveform[a:b]])
                    require(math.isfinite(probability) and 0 <= probability <= 1, "Invalid voice response")
                    response = float(logit(probability))
                    counts[arm] += 1
            responses[arm] = response
        result.append({**row, "responses": responses})
        if len(result) % 12 == 0:
            print(role, len(result), "deliveries", sum(counts.values()), "forwards", round(time.monotonic() - started, 2), flush=True)
    print(role, len(result), "deliveries", sum(counts.values()), "forwards", round(time.monotonic() - started, 2), flush=True)
    return {"rows": result, "forwards": dict(counts)}


def fit(development):
    values = defaultdict(lambda: defaultdict(list))
    groups = {row["group"] for row in development["rows"]}
    require(len(groups) == 4 and all(row["role"] == "development" for row in development["rows"]), "Development role differs")
    for row in development["rows"]:
        if row["origin"] == "natural" and row["level_db"] is not None:
            for arm, value in row["responses"].items():
                if value is not None:
                    values[arm][row["group"]].append(value)
    points = {}
    for arm in ("mono", "wpe", "frontal"):
        require(set(values[arm]) == groups, f"Incomplete admitted development readers in {arm}")
        highest = sorted((max(answers) for answers in values[arm].values()), reverse=True)
        points[arm] = {"zero": highest[0], "one": highest[1]}
    return points


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("download", "freeze", "fit", "score", "replay"))
    parser.add_argument("cache", type=Path)
    parser.add_argument("--run", type=Path)
    for flag in ("surrey", "bbc", "ami"):
        parser.add_argument(f"--{flag}", type=Path)
    parser.add_argument("--weights", type=Path, default=ROOT / "models/weights/audio-upgrade")
    args = parser.parse_args()
    if args.phase == "download":
        args.cache.mkdir(mode=0o700, exist_ok=True)
        roster, pcm = plan(args.cache, True)
        print(f"Verified {len(roster)} exact transcript triplets and {len(pcm)} WAVs")
        return
    require(all((args.run, args.surrey, args.bbc, args.ami)), "Every offline stage needs the run, room and AMI paths")
    lock, roster, received = prepare(args)
    freeze = args.run / "freeze.json"
    if args.phase == "freeze":
        args.run.mkdir(mode=0o700)
        write(freeze, lock)
        write(args.run / "delivery-roster.json", roster)
        print(f"Froze {len(roster)} clock-bound deliveries before model inference")
        return
    require(freeze.read_bytes() == json_bytes(lock) and (args.run / "delivery-roster.json").read_bytes() == json_bytes(roster),
            "Frozen study assets, waveform rows or code differ")
    if args.phase == "replay":
        print(f"Replayed {len(roster)} delivery and view commitments")
        return
    dev = args.run / "development.json"
    points_path = args.run / "points.json"
    if args.phase == "fit":
        require(not dev.exists() and not points_path.exists(), "Development results already exist")
        development = score(roster, received, "development", args.weights)
        points = fit(development)
        write(dev, development)
        write(points_path, {"freeze_sha256": file_digest(freeze), "development_sha256": file_digest(dev), "points": points})
        print(points)
        return
    points = read(points_path)
    require(points == {"freeze_sha256": file_digest(freeze), "development_sha256": file_digest(dev), "points": fit(read(dev))},
            "Operating points must be frozen before assessment")
    public = args.run / "public.json"
    require(not public.exists(), "Assessment results already exist")
    held = score(roster, received, "assessment", args.weights)
    write(public, {"freeze": lock, "operating_points": points, "development": read(dev), "assessment": held})


if __name__ == "__main__":
    main()
