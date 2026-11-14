# SPDX-License-Identifier: Apache-2.0
"""Compare the existing background review after 48 kHz lossless mono delivery."""

from __future__ import annotations

import argparse
import binascii
import hashlib
import json
import math
import struct
import subprocess
import wave
import zlib
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory

SCHEMA = "chipmunk.audio-background-lossless-rate-transfer/1"
ROOT = Path(__file__).resolve().parents[2]
PUBLIC = ROOT / "docs/evaluation-panels"
PARENTS = {
    "frozen": ("audio-nemotron-noise-transfer-2026-11-12.json", "004eb428ec6bfbe86ab50e1173ddc4db857bafb419d832038a55bf8583eddadb", 74),
    "confirmation": ("audio-nemotron-noise-confirmation-2026-11-12.json", "0615aa36e53fc092b6458161e6d27de65e92ba9f20d45ac334402c0db276977b", 92),
}
POINT = 7.422796281843017
RENDER = "Three seconds of 48 kHz mono PCM16, rendered with un-dithered FFmpeg swresample from the pinned 16 kHz held mixture. Encode the same PCM losslessly as WAV and FLAC."
POLICY = {"minimum_retained_noisy_synthetic_flags_fraction": 0.80, "maximum_added_genuine_flags_per_panel": 2,
          "maximum_added_gain_or_background_flags": 0}
CODECS = {"wav": "pcm_s16le", "flac": "flac"}
ELIGIBLE_DIRECTIONS = {"inconclusive", "unassessed"}
VCTK_FREEZE = "d3a2412b4572976314566ab62ff4c307442c39f4c9597df783b138746acd8831"
VCTK_MAX_NOISY_GENUINE = 1


def require(valid, reason):
    if not valid:
        raise ValueError(reason)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def serialized(value):
    return json.dumps(value, sort_keys=True, indent=2, allow_nan=False).encode() + b"\n"


def load(path):
    return json.loads(path.read_bytes())


def save(path, doc):
    with path.open("xb") as out:
        out.write(serialized(doc))


def parent_rows():
    for panel, (name, digest, count) in PARENTS.items():
        contents = (PUBLIC / name).read_bytes()
        require(sha(contents) == digest, "Published parent receipt differs")
        parent = json.loads(contents)
        rows = parent["held"]
        require(len(rows) == count, "Held set differs")
        for row in rows:
            yield panel, row, parent["freeze"]["voice_assets"]


def clock(mapping, codec):
    require(mapping.get("source_codec") == codec and mapping.get("source_channels") == 1
            and mapping.get("source_sample_rate") == 48_000 and mapping.get("source_bit_depth") == 16
            and mapping.get("source_stream_index") == 0 and mapping.get("timeline_time_base") == "1/16000"
            and mapping.get("samples") == 48_000
            and mapping.get("timeline") == [{"start_sample": 0, "end_sample": 48_000, "source_pts": 0}],
            "Lossless delivery changed source extent")


def ffmpeg(*arguments, pcm=None):
    return subprocess.run(["ffmpeg", "-nostdin", "-v", "error", *arguments], input=pcm, check=True,
                          timeout=10, capture_output=True).stdout


def make_files(raw, base):
    rendered = ffmpeg("-f", "s16le", "-ar", "16000", "-ac", "1", "-i", "pipe:0", "-af",
                      "aresample=48000:dither_method=none", "-c:a", "pcm_s16le", "-f", "s16le", "pipe:1", pcm=raw)
    require(len(rendered) == 48_000 * 3 * 2, "Unexpected 48 kHz render extent")
    with wave.open(str(base.with_suffix(".wav")), "wb") as file:
        file.setparams((1, 2, 48_000, 0, "NONE", "not compressed"))
        file.writeframes(rendered)
    base.with_suffix(".flac").write_bytes(ffmpeg("-i", str(base.with_suffix(".wav")), "-c:a", "flac", "-f", "flac", "pipe:1"))
    return sha(rendered)


def expected_legacy(row):
    enhanced = row["candidates"]["new_enhancer"]
    return {"direction": row["incumbent"], "admitted": enhanced["admitted"],
            "logit": enhanced["logit"] if enhanced["admitted"] else None}


def flagged(response):
    return (response["direction"] in ELIGIBLE_DIRECTIONS and response["admitted"] and response["logit"] > POINT)


def summarize(rows):
    result = {}
    for panel in PARENTS:
        result[panel] = {}
        for carrier in CODECS:
            counts = Counter()
            for row in rows:
                if row["panel"] != panel:
                    continue
                prior, new = flagged(row["prior"]), flagged(row["rate_48k"][carrier])
                category = ("clean" if row["scene"] == "gain_control" else "background" if row["label"] == "isolated"
                            else "noisy_ai" if row["label"] == "spoof" else "noisy_genuine")
                counts[f"{category}_views"] += 1
                counts[f"{category}_prior_flags"] += int(prior)
                counts[f"{category}_new_flags"] += int(new)
                counts[f"{category}_lost_flags"] += int(prior and not new)
                counts[f"{category}_added_flags"] += int(new and not prior)
                counts["primary_direction_changes"] += int(row["prior"]["direction"] != row["rate_48k"][carrier]["direction"])
            retained = counts["noisy_ai_prior_flags"] - counts["noisy_ai_lost_flags"]
            result[panel][carrier] = {"counts": dict(counts), "passes": {
                "retains_noisy_flags": retained >= counts["noisy_ai_prior_flags"] * POLICY["minimum_retained_noisy_synthetic_flags_fraction"],
                "ordinary_flags": counts["noisy_genuine_added_flags"] <= POLICY["maximum_added_genuine_flags_per_panel"],
                "clean_or_background_flags": (counts["clean_added_flags"] + counts["background_added_flags"]
                                              <= POLICY["maximum_added_gain_or_background_flags"]),
            }}
    return result


def freeze(inputs, run):
    from app.detectors.audio.decode import decode_audio

    run.mkdir(mode=0o700)
    (run / "media").mkdir()
    deliveries = []
    for panel, row, assets in parent_rows():
        order = row["id"]
        source = (inputs / panel / "pcm" / f"{order:03}.pcm").read_bytes()
        require(len(source) == 96_000 and sha(source) == row["pcm_sha256"], "Held PCM differs from prior receipt")
        base = run / "media" / f"{panel}-{order:03}"
        native_pcm = make_files(source, base)
        pins, views = {}, {}
        for extension, codec in CODECS.items():
            path = base.with_suffix(f".{extension}")
            pins[extension] = sha(path.read_bytes())
            require(sha(ffmpeg("-i", str(path), "-c:a", "pcm_s16le", "-f", "s16le", "pipe:1")) == native_pcm,
                    "Lossless carrier changed native 48 kHz PCM")
            decoded = decode_audio(path, source_sha256=pins[extension])
            clock(decoded.mapping, codec)
            views[extension] = decoded.mapping["pcm_sha256"]
        deliveries.append({"panel": panel, "id": order, "scene": row["scene"], "label": row["label"],
                           "prior_pcm_sha256": row["pcm_sha256"], "carrier_sha256": pins, "native_48k_pcm_sha256": native_pcm,
                           "decoded_16k_sha256": views, "prior": expected_legacy(row)})
        if len(deliveries) % 24 == 0:
            print(f"{len(deliveries)} lossless pairs bound", flush=True)
    save(run / "freeze.json", {"schema": SCHEMA, "runner_sha256": sha(Path(__file__).read_bytes()), "point": POINT,
         "render": RENDER, "policy": POLICY, "voice_assets": assets, "ffmpeg": ffmpeg("-version").decode().splitlines()[0],
         "panels": {key: spec[1] for key, spec in PARENTS.items()}, "rows": deliveries})


def score(run, runtime, weights):
    import numpy as np

    from app.core.config import settings
    from app.detectors.audio import calibration
    from app.detectors.audio.background_voice_review import OPERATING_POINT, _enhance
    from app.detectors.audio.content import window_in_speech
    from app.detectors.audio.decode import decode_audio
    from app.detectors.audio.upgraded import SpeechActivity
    from app.engine.boundary import Deadline
    from scripts.audio_radar_transfer import adapter_result
    from scripts.audio_strong_traffic_view import network_free
    from scripts.run_audio_specialists_v15 import load_voice

    frozen = load(run / "freeze.json")
    require(frozen["runner_sha256"] == sha(Path(__file__).read_bytes()) and POINT == OPERATING_POINT, "Frozen policy differs")
    settings.environment = "dev"
    settings.audio_detector_mode = "candidate"
    settings.audio_background_review_model = runtime / "model"
    settings.audio_background_review_packages = runtime / "python312"
    network_free()
    model = load_voice({"weights": str(weights), "assets": {key: value for key, value in frozen["voice_assets"].items()
                                                           if key in {"pellav2_detector.pt", "w2v-bert-config.json", "silero_vad.jit"}}})
    vad = SpeechActivity(weights / "silero_vad.jit")
    rows, calls = [], Counter()
    for row in frozen["rows"]:
        readings, measured = {}, {}
        for carrier, codec in CODECS.items():
            path = run / "media" / f"{row['panel']}-{row['id']:03}.{carrier}"
            decoded = decode_audio(path, source_sha256=row["carrier_sha256"][carrier])
            clock(decoded.mapping, codec)
            pin = decoded.mapping["pcm_sha256"]
            require(pin == row["decoded_16k_sha256"][carrier], "Analyzed PCM differs")
            if pin not in measured:
                direction = adapter_result(decoded.pcm, model, vad)["decision"]
                calls["primary"] += 1
                response = {"direction": direction, "admitted": None, "logit": None}
                if direction in ELIGIBLE_DIRECTIONS:
                    enhanced = _enhance(decoded.pcm, Deadline(20))
                    calls["enhancer"] += 1
                    pcm = np.frombuffer(enhanced, dtype="<i2").astype(np.float32) / 32768
                    admitted = window_in_speech(vad.spans(pcm), 0., 3.)
                    response.update(admitted=admitted, processed_16k_sha256=sha(enhanced))
                    if admitted:
                        prediction, = model.predict([pcm])
                        response["logit"] = float(calibration.logit(prediction))
                measured[pin] = response
            readings[carrier] = measured[pin]
        rows.append({**row, "rate_48k": readings})
        if len(rows) % 12 == 0 or len(rows) == len(frozen["rows"]):
            print(f"{len(rows)} paired dispositions assessed", flush=True)
    summary = summarize(rows)
    save(run / "receipt.json", {"freeze": {key: value for key, value in frozen.items() if key != "rows"},
                                "freeze_sha256": sha((run / "freeze.json").read_bytes()), "rows": rows, "calls": dict(calls),
                                "result": summary})
    print(serialized(summary).decode(), end="")


def native_input(root, row, noise):
    import numpy as np

    cached = root / f"{row['offset']}.range"
    raw = cached.read_bytes()
    start, stop = row["offset"], row["offset"] + row["size"] + 255
    header = (root / f"{start}.h").read_text().lower()
    require(sha(raw) == row["range_sha256"] and len(raw) == stop - start + 1
            and f"content-range: bytes {start}-{stop}/11747302977" in header, "VCTK publisher range differs")
    fixed = struct.unpack_from("<4s5H3I2H", raw)
    name_length, extra_length = fixed[-2:]
    require(fixed[0] == b"PK\x03\x04" and fixed[3] == row["method"] in {0, 8}
            and raw[30:30+name_length].decode() == row["name"], "VCTK native member differs")
    a = 30 + name_length + extra_length
    compressed = raw[a:a+row["size"]]
    original = zlib.decompress(compressed, -15) if row["method"] == 8 else compressed
    require(sha(original) == row["flac_sha256"] and len(original) == row["expected_flac_bytes"]
            and binascii.crc32(original) == row["crc"], "VCTK member provenance differs")
    decoded = ffmpeg("-f", "flac", "-i", "pipe:0", "-map", "0:a:0", "-ac", "1", "-c:a", "pcm_s16le", "-f", "s16le",
                     "pipe:1", pcm=original)
    require(sha(decoded) == row["native_pcm_sha256"] and len(decoded) == 2 * row["native_samples"], "Native PCM differed")
    foreground = np.frombuffer(decoded, "<i2")[row["start"]:row["start"]+144000].astype(np.float64) / 32768
    masker = noise[row["background_start"]:row["background_start"]+144000]
    mixed = foreground + masker * (np.linalg.norm(foreground) / np.linalg.norm(masker) * 10**0.25)
    gain = min(1., .98 / max(np.max(abs(mixed)), np.max(abs(foreground))))
    return {key: np.rint(view * gain * 32768).astype("<i2").tobytes() for key, view in {"clean": foreground, "road": mixed}.items()}


def native_replay(root, background, receipt):
    import numpy as np

    from app.detectors.audio.decode import decode_audio

    frozen = load(receipt)["native_freeze"]
    require(sha(serialized(frozen)) == VCTK_FREEZE and sha(background.read_bytes()) == frozen["background_sha256"],
            "Native VCTK inputs or background changed")
    raw = ffmpeg("-i", str(background), "-map", "0:a:0", "-ac", "1", "-c:a", "pcm_s16le", "-f", "s16le", "pipe:1")
    masker = np.frombuffer(raw, "<i2").astype(np.float64) / 32768
    with TemporaryDirectory() as directory:
        temp = Path(directory) / "test.wav"
        for row in frozen["rows"]:
            for name, pcm in native_input(root, row, masker).items():
                with wave.open(str(temp), "wb") as file:
                    file.setparams((1, 2, 48_000, 0, "NONE", "not compressed"))
                    file.writeframes(pcm)
                pin = sha(temp.read_bytes())
                require(pin == row["deliveries"][name], "Native test delivery differs")
                decoded = decode_audio(temp, source_sha256=pin)
                clock(decoded.mapping, "pcm_s16le")
                require(decoded.mapping["pcm_sha256"] == row["decoded_delivery"][name], "Native test input differs")
    print(f"Verified {2*len(frozen['rows'])} native 48 kHz control inputs and clocks")


def native_results(document):
    frozen, rows = document["native_freeze"], document["native_controls"]
    require(sha(serialized(frozen)) == VCTK_FREEZE and len(frozen["rows"]) == 12
            and len(rows) == 24, "Native genuine roster differs")
    counts = {condition: Counter() for condition in ("clean", "road")}
    for observed, (row, condition) in zip(rows, ((row, k) for row in frozen["rows"] for k in counts), strict=True):
        direction, score = observed["direction"], observed["logit"]
        require(observed["id"] == row["id"] and observed["condition"] == condition
                and direction in {*ELIGIBLE_DIRECTIONS, "original", "ai"}
                and (observed["processed_pcm_sha256"] is not None) == (direction in ELIGIBLE_DIRECTIONS)
                and (score is None or math.isfinite(score) and direction in ELIGIBLE_DIRECTIONS)
                and observed["flagged"] == (score is not None and score > POINT), "Native genuine outcome differs")
        counts[condition].update({"files": 1, "primary_ai": int(direction == "ai"), "added_flags": int(observed["flagged"]),
                                  "assessed_after_reduction": int(score is not None)})
    return {"counts": {name: dict(values) for name, values in counts.items()},
            "passes": counts["clean"]["added_flags"] == 0 and counts["road"]["added_flags"] <= VCTK_MAX_NOISY_GENUINE}


def review(path):
    document = load(path)
    freeze, rows = document["freeze"], document["rows"]
    require(freeze["schema"] == SCHEMA and freeze["render"] == RENDER and freeze["policy"] == POLICY
            and freeze["point"] == POINT and freeze["panels"] == {key: spec[1] for key, spec in PARENTS.items()}, "Policy or panels differ")
    for observed, (panel, historical, _) in zip(rows, parent_rows(), strict=True):
        require((observed["panel"], observed["id"], observed["prior_pcm_sha256"]) == (panel, historical["id"], historical["pcm_sha256"])
                and all(observed[field] == historical[field] for field in ("scene", "label"))
                and observed["prior"] == expected_legacy(historical), "Historical comparison data differs")
        require(observed["rate_48k"].keys() == CODECS.keys(), "Missing carrier responses")
        for new in observed["rate_48k"].values():
            direction, admitted, response = (new[key] for key in ("direction", "admitted", "logit"))
            require(direction in {*ELIGIBLE_DIRECTIONS, "original", "ai"}
                    and ((type(admitted) is bool and "processed_16k_sha256" in new) if direction in ELIGIBLE_DIRECTIONS
                         else admitted is None and "processed_16k_sha256" not in new)
                    and (type(response) is float and math.isfinite(response) if admitted is True else response is None),
                    "Invalid 48 kHz decision")
    file_rows = [{key: value for key, value in row.items() if key != "rate_48k"} for row in rows]
    require(sha(serialized({**freeze, "rows": file_rows})) == document["freeze_sha256"], "Frozen carrier commitment differs")
    require(summarize(rows) == document["result"], "Summary differs from bound decisions")
    native = native_results(document)
    require(native == document["native_result"] and document["serving_admission"] is False, "Native results or admission differ")
    print(serialized({"carrier_transfer": document["result"], "native_48k": native, "serving_admission": False}).decode(), end="")


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("phase", choices=("freeze", "score", "review", "native-replay"))
    cli.add_argument("path", type=Path)
    cli.add_argument("--inputs", type=Path)
    cli.add_argument("--background", type=Path)
    cli.add_argument("--runtime", type=Path, default=ROOT / "backend/data/background-speech-runtime")
    cli.add_argument("--weights", type=Path, default=ROOT / "models/weights/audio-upgrade")
    args = cli.parse_args()
    if args.phase == "freeze":
        freeze(args.inputs, args.path)
    elif args.phase == "score":
        score(args.path, args.runtime, args.weights)
    elif args.phase == "native-replay":
        native_replay(args.inputs, args.background, args.path)
    else:
        review(args.path)


if __name__ == "__main__":
    main()
