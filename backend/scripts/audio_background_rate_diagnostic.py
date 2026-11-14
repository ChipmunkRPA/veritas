# SPDX-License-Identifier: Apache-2.0
"""Separate traffic and lossless carrier effects in the background speech review."""

from __future__ import annotations

import argparse
import math
import sys
import wave
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts import audio_background_lossless_rate_transfer as previous

ROOT = previous.ROOT
PARENT = ROOT / "docs/evaluation-panels/audio-background-lossless-rate-transfer-2026-11-13.json"
SCHEMA = "chipmunk.audio-background-rate-diagnostic/1"
CONDITIONS = ("clean48", "road48", "road16_repacked", "road16_direct", "road0", "road5", "road_only")
METHOD = {
    "source": "Same fixed roster, source offsets, three-second range, and Vienna traffic used in the 12-speaker native control test.",
    "carrier": "Compare the retained native road WAV against the byte-identical analysed PCM rewrapped in 16 kHz PCM16 WAV, and an independent direct 16 kHz mixture. The direct mixture resamples clean speech and the background separately at their existing gain and then sums and quantizes.",
    "dose": "Mix the same 48 kHz speech and traffic excerpt at the prior -5 dB, 0 dB and +5 dB. Keep clean foreground gain fixed. Background alone uses the exact -5 dB gain.",
    "rule": "Rerun the existing primary decision, speech gate, and background-reduced response at the fixed point. Compute enhancement only for primary-inconclusive or primary-unassessed inputs.",
    "question": "A reproduced flag on a 16 kHz WAV carrying exactly the same analysed PCM establishes that accepting only native 16 kHz uploads cannot prevent that case. An independently mixed 16 kHz WAV checks that outcome under quantization differences. No point fitting or changes to serving.",
    "point": previous.POINT,
}
ELIGIBLE = previous.ELIGIBLE_DIRECTIONS
require, sha, load, save = previous.require, previous.sha, previous.load, previous.save


def original_plan():
    reading = load(PARENT)
    require(sha(previous.serialized(reading["native_freeze"])) == previous.VCTK_FREEZE, "Parent speaker freeze changed")
    return reading


def decode(path, rate, digest):
    from app.detectors.audio.decode import decode_audio

    value = decode_audio(path, source_sha256=digest)
    mapping = value.mapping
    require(mapping["source_sample_rate"] == rate and mapping["source_codec"] == "pcm_s16le"
            and mapping["source_channels"] == mapping["channels"] == 1 and mapping["sample_rate"] == 16000
            and mapping["samples"] == 48000 and mapping["timeline_time_base"] == "1/16000"
            and mapping["timeline"] == [{"start_sample": 0, "end_sample": 48000, "source_pts": 0}],
            "WAV input does not preserve the complete planned analysis clock")
    return value.pcm, mapping["pcm_sha256"]


def wav(path, raw, rate):
    require(len(raw) == rate * 3 * 2, "Input length differs")
    with wave.open(str(path), "wb") as stream:
        stream.setparams((1, 2, rate, 0, "NONE", "not compressed"))
        stream.writeframes(raw)


def inputs(parent, cache, background, staging):
    import numpy as np

    def native(path):
        return previous.ffmpeg("-i", str(path), "-map", "0:a:0", "-ac", "1", "-c:a", "pcm_s16le", "-f", "s16le", "pipe:1")

    def samples(raw):
        return np.frombuffer(raw, "<i2").astype(np.float64) / 32768

    def quantize(signal):
        codes = np.rint(signal * 32768)
        require(np.isfinite(codes).all() and np.min(codes) >= -32768 and np.max(codes) <= 32767, "A mixture clipped")
        return codes.astype("<i2").tobytes()

    def resample(raw):
        return samples(previous.ffmpeg("-f", "s16le", "-ar", "48000", "-ac", "1", "-i", "pipe:0", "-af",
                                       "aresample=16000:dither_method=none", "-c:a", "pcm_s16le", "-f", "s16le", "pipe:1",
                                       pcm=raw))

    frozen = parent["native_freeze"]
    require(sha(background.read_bytes()) == frozen["background_sha256"], "Background recording changed")
    masker = samples(native(background))
    for row in frozen["rows"]:
        index = row["id"]
        source = cache / f"{index:02}-source.flac"
        source_bytes = native(source)
        require(sha(source.read_bytes()) == row["flac_sha256"] and sha(source_bytes) == row["native_pcm_sha256"],
                "Speaker file did not reproduce")
        source_start, masker_start = (row[key] for key in ("start", "background_start"))
        voice = samples(source_bytes)[source_start:source_start + 144000]
        noise = masker[masker_start:masker_start + 144000]
        require(len(voice) == len(noise) == 144000 and np.linalg.norm(noise) > 0, "Planned source extent differs")
        power_ratio = np.linalg.norm(voice) / np.linalg.norm(noise)
        loud = noise * (power_ratio * 10**.25)
        gain = min(1., .98 / max(np.max(abs(voice + loud)), np.max(abs(voice))))
        clean_native, road_native = quantize(voice * gain), quantize((voice + loud) * gain)
        independent = resample(clean_native) + resample(quantize(loud * gain))
        cases = {
            "clean48": (48000, clean_native),
            "road48": (48000, road_native),
            "road16_repacked": (16000, None),
            "road16_direct": (16000, quantize(independent)),
            "road0": (48000, quantize((voice + noise * power_ratio) * gain)),
            "road5": (48000, quantize((voice + noise * (power_ratio * 10**-.25)) * gain)),
            "road_only": (48000, quantize(loud * gain)),
        }
        prior_road = None
        for condition, (rate, raw) in cases.items():
            path = staging / f"{index:02}-{condition}.wav"
            wav(path, quantize(prior_road) if raw is None else raw, rate)
            digest = sha(path.read_bytes())
            pcm, commitment = decode(path, rate, digest)
            if condition in {"clean48", "road48"}:
                prior = "road" if condition == "road48" else "clean"
                require(digest == row["deliveries"][prior] and commitment == row["decoded_delivery"][prior],
                        "Previously scored native WAV or PCM differs")
            if condition == "road48":
                prior_road = pcm.copy()
            drift = None
            if condition.startswith("road16"):
                require(prior_road is not None, "Expected native mixture first")
                difference = np.rint((pcm - prior_road) * 32768).astype(np.int32)
                drift = {"different_samples": int(np.count_nonzero(difference)), "maximum_pcm_step": int(abs(difference).max())}
                if condition == "road16_repacked":
                    require(not drift["different_samples"], "Exact PCM repack changed samples")
            yield {"id": index, "condition": condition, "input_sha256": digest, "analysis_pcm_sha256": commitment,
                   "source_rate": rate, "samples": len(pcm), "versus_road48": drift}, pcm


def pin():
    return {"schema": SCHEMA, "method": METHOD, "parent_sha256": sha(PARENT.read_bytes()),
            "implementation_sha256": sha(Path(__file__).read_bytes())}


def restored_weights():
    from importlib.metadata import distributions

    from app.core.config import settings
    from app.detectors.audio.background_review_worker import ASSETS, PACKAGES
    from app.detectors.audio.upgraded import SpeechActivity
    from scripts.audio_strong_traffic_view import network_free
    from scripts.prepare_audio_specialists_v15 import ASSETS as VOICE_ASSETS
    from scripts.run_audio_specialists_v15 import load_voice

    runtime = ROOT / "backend/data/background-speech-runtime"
    weights = ROOT / "models/weights/audio-upgrade"
    model, packages = runtime / "model", runtime / "python312"
    require(all(sha((model / name).read_bytes()) == digest for name, digest in ASSETS.items()), "Enhancer files differ")
    versions = {item.metadata["Name"].lower(): item.version for item in distributions(path=[str(packages)])}
    require(all(versions.get(name) == version for name, version in PACKAGES.items()), "Pinned enhancement packages differ")
    network_free()
    settings.environment = "dev"
    settings.audio_detector_mode = "candidate"
    voice = load_voice({"weights": str(weights), "assets": VOICE_ASSETS})
    activity = SpeechActivity(weights / "silero_vad.jit")
    sys.path.insert(0, str(packages))
    import mlx.core as mx
    from mlx_audio.sts.utils import load_model

    mx.set_cache_limit(256 * 1024**2)
    enhancer = load_model(str(model))
    return voice, activity, enhancer


def read_views(source, planned):
    import numpy as np

    from app.detectors.audio import calibration
    from app.detectors.audio.background_voice_review import OPERATING_POINT
    from app.detectors.audio.content import window_in_speech
    from scripts.audio_radar_transfer import adapter_result

    require(METHOD["point"] == OPERATING_POINT, "Serving point differs from frozen study")
    voice, activity, enhancer = restored_weights()
    cache = {}
    for (row, pcm), key in zip(source, planned, strict=True):
        require(row == key, "Input differs from sealed plan")
        commitment = row["analysis_pcm_sha256"]
        if commitment not in cache:
            primary = adapter_result(pcm, voice, activity)
            measured = {"primary": primary["decision"], "primary_raw": primary["raw"], "processed_pcm_sha256": None,
                        "logit": None, "admitted": False, "flagged": False}
            if measured["primary"] in ELIGIBLE:
                output = np.asarray(enhancer.generate(pcm, sample_rate=16000, chunk_seconds=None).audio, dtype=np.float32)
                require(output.shape == pcm.shape and np.isfinite(output).all(), "Enhanced samples or clock differ")
                raw = np.clip(np.rint(output * 32768), -32768, 32767).astype("<i2").tobytes()
                enhanced = np.frombuffer(raw, "<i2").astype(np.float32) / 32768
                measured["processed_pcm_sha256"] = sha(raw)
                measured["admitted"] = window_in_speech(activity.spans(enhanced), 0., 3.)
                if measured["admitted"]:
                    probability, = voice.predict([enhanced])
                    measured["logit"] = float(calibration.logit(probability))
                    measured["flagged"] = measured["logit"] > METHOD["point"]
            cache[commitment] = measured
        yield row | cache[commitment]
        if row["condition"] == CONDITIONS[-1]:
            print(f"Measured speaker {row['id'] + 1}/12", flush=True)


def audit(receipt, parent):
    parent_control = {(row["id"], row["condition"]): row for row in parent["native_controls"]}
    freeze = receipt["freeze"]
    require(freeze["schema"] == SCHEMA and freeze["method"] == METHOD and freeze["parent_sha256"] == sha(PARENT.read_bytes())
            and len(freeze["cases"]) == len(receipt["rows"]) == 12 * len(CONDITIONS), "Unexpected source plan")
    groups = {}
    for row, key, (speaker, condition) in zip(receipt["rows"], freeze["cases"],
          ((speaker, condition) for speaker in parent["native_freeze"]["rows"] for condition in CONDITIONS), strict=True):
        require(all(row[field] == value for field, value in key.items()) and row["id"] == speaker["id"]
                and row["condition"] == condition and row["samples"] == 48000
                and row["source_rate"] == (16000 if condition.startswith("road16") else 48000), "Case binding differs")
        for field in ("input_sha256", "analysis_pcm_sha256"):
            require(len(row[field]) == 64 and all(c in "0123456789abcdef" for c in row[field]), "Case commitment invalid")
        groups.setdefault(row["id"], {})[condition] = row
        response = row["logit"]
        direction = row["primary"]
        require(direction in {"original", "ai", *ELIGIBLE}
                and (row["primary_raw"] is None if direction == "unassessed" else 0 <= row["primary_raw"] <= 1.)
                and type(row["flagged"]) is type(row["admitted"]) is bool
                and row["admitted"] == (response is not None)
                and (row["processed_pcm_sha256"] is not None) == (direction in ELIGIBLE)
                and row["flagged"] == (response is not None and math.isfinite(response) and response > METHOD["point"]),
                "Decision or response differs from the fixed rule")
        if condition in {"clean48", "road48"}:
            prior = parent_control[speaker["id"], "clean" if condition == "clean48" else "road"]
            require(row["input_sha256"] == speaker["deliveries"][prior["condition"]]
                    and row["analysis_pcm_sha256"] == speaker["decoded_delivery"][prior["condition"]], "Old native input differs")
    counts = {condition: Counter() for condition in CONDITIONS}
    paired_flips = Counter()
    maximum_step, max_prior_delta = 0, 0.
    prior_flag_changes = 0
    fields = ("primary", "primary_raw", "processed_pcm_sha256", "logit", "admitted", "flagged", "analysis_pcm_sha256")
    for index, group in groups.items():
        old = group["road48"]
        repacked = group["road16_repacked"]
        require(all(old[key] == repacked[key] for key in fields)
                and repacked["versus_road48"] == {"different_samples": 0, "maximum_pcm_step": 0},
                "Identical analysed PCM must retain the same decisions")
        direct = group["road16_direct"]
        drift = direct["versus_road48"]
        require(0 <= drift["different_samples"] <= 48000 and 0 <= drift["maximum_pcm_step"] <= 65535, "Invalid PCM drift")
        maximum_step = max(maximum_step, drift["maximum_pcm_step"])
        for condition, row in group.items():
            counts[condition].update(files=1, primary_ai=int(row["primary"] == "ai"), assessed=int(row["admitted"]), flags=int(row["flagged"]))
            if condition in {"clean48", "road48"}:
                prior = parent_control[index, "clean" if condition == "clean48" else "road"]
                prior_flag_changes += row["flagged"] != prior["flagged"]
                if row["logit"] is not None and prior["logit"] is not None:
                    max_prior_delta = max(max_prior_delta, abs(row["logit"] - prior["logit"]))
            if condition in {"road16_direct", "road0", "road5", "road_only"}:
                paired_flips[condition + "_added"] += int(row["flagged"] and not old["flagged"])
                paired_flips[condition + "_removed"] += int(old["flagged"] and not row["flagged"])
    return {"counts": counts, "versus_road48": paired_flips, "maximum_direct_pcm_step": maximum_step,
            "prior_native_flag_changes": prior_flag_changes, "prior_native_max_logit_difference": max_prior_delta}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "score", "review"))
    parser.add_argument("path", type=Path)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--background", type=Path)
    args = parser.parse_args()
    parent = original_plan()
    if args.phase == "review":
        recorded = load(args.path)
        measured = audit(recorded, parent)
        require(recorded["result"] == measured, "Published results do not recount")
        print(previous.serialized(measured).decode(), end="")
        return
    require(args.cache and args.background, "Supply checked VCTK cache and the original traffic recording")
    with TemporaryDirectory() as temp:
        data = inputs(parent, args.cache, args.background, Path(temp))
        if args.phase == "freeze":
            args.path.mkdir(mode=0o700)
            save(args.path / "freeze.json", pin() | {"cases": [row for row, _ in data]})
            return
        planned = load(args.path / "freeze.json")
        require(all(planned[key] == value for key, value in pin().items()), "Runner or frozen procedure differs")
        result = {"freeze": planned, "rows": list(read_views(data, planned["cases"]))}
        result["result"] = audit(result, parent)
        save(args.path / "receipt.json", result)
        print(previous.serialized(result["result"]).decode(), end="")


if __name__ == "__main__":
    main()
