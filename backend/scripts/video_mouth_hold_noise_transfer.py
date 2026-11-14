# SPDX-License-Identifier: Apache-2.0
"""Compare robust periodicity views on rights-reviewed speech and matched backgrounds."""
from __future__ import annotations

import argparse
import json
from hashlib import sha256
from pathlib import Path

from scripts.review_video_mouth_hold_noise_transfer import SCENES, SCHEMA, VIEWS, summarize

RATE = 16000
WINDOWS = ((12480, 18880), (30080, 36480), (47680, 54080), (65280, 71680))
PARENT = "4a6dcb2ebbd6c3451228f7439e1c2f9fe3234f9392c02b8892c0ad6893a2f1c8"
BANDS = ((85, 400), (400, 900), (900, 1800), (1800, 3600))
DESIGN = {
    "scope": "First source pair per SINE reader, original and alternate. Four fixed 400 ms windows per delivery.",
    "mix": "Original and alternate each use their own nominal -5 dB foreground-to-background RMS. "
           "Per source side, one common scale covers every delivery, including matched background-only controls.",
    "baseline": "Raw 40 ms autocorrelation, .78 peak and sustained in two-thirds of windows.",
    "band_balance": "Autocorrelation after RMS balancing four fixed bands with an amplitude floor. "
                    "The mean equally weights bands rather than allowing the loudest band to dominate.",
    "flank_whitening": "Subtract a local power estimate from 40 ms window spectra using 200–400 ms of "
                       "source context on each side. Preserve timing and search peaks using normalized autocorrelation.",
    "review": "Per arm, maximize valid recovery of clean-positive windows on development only, "
              "with no more than one added distinct background window and one added mixture without clean support. "
              "Prefer the larger cutoff on ties. Compare candidate unions with the current raw decision.",
}
THRESHOLDS = (.35, .45, .55, .65, .75)


def sha(data: bytes) -> str:
    return sha256(data).hexdigest()


def _source_plan(cache: Path, sounds: Path) -> dict:
    from scripts.audio_background_confirmation import BACKGROUNDS

    raw = (cache / "freeze.json").read_bytes()
    if sha(raw) != PARENT:
        raise ValueError("SINE source plan differs from the archived comparison")
    parent = json.loads(raw)
    choices = {}
    for pair in parent["pairs"]:
        choices.setdefault(pair["reader"], pair)
    ranked = sorted(choices.values(), key=lambda row: sha(f"video-mouth-noise/1/{row['reader']}".encode()))
    rows = []
    for index, pair in enumerate(ranked):
        rows.append({"group": index, "source_id": pair["id"], "split": "development" if index < 8 else "held",
                     "side_hashes": {key: entry["wav_sha256"] for key, entry in pair["files"].items()}})
    for scene, facts in BACKGROUNDS.items():
        if sha((sounds / f"{scene}.ogg").read_bytes()) != facts["sha256"]:
            raise ValueError(f"{scene} original differs from its publisher commitment")
    return {"schema": SCHEMA, "method": DESIGN, "source_parent_sha256": PARENT,
            "runtime_sha256": sha(Path(__file__).read_bytes()), "windows": [list(row) for row in WINDOWS],
            "review_sha256": sha(Path(__file__).with_name("review_video_mouth_hold_noise_transfer.py").read_bytes()),
            "threshold_grid": list(THRESHOLDS), "groups": rows, "environmental_sources": BACKGROUNDS,
            "overlap_source_sha256": [row["pcm_sha256"] for row in parent["backgrounds"]]}


def _peaks(window, lags):
    import numpy as np

    centered = window - window.mean()
    values = np.correlate(centered, centered, "full")[len(centered) - 1:][lags]
    squared = np.r_[0., np.cumsum(centered ** 2)]
    norm = np.sqrt(squared[len(centered) - lags] * np.maximum(0, squared[-1] - squared[lags]))
    coeff = values / (norm + 1e-12)
    usable = (coeff[1:-1] > coeff[:-2]) & (coeff[1:-1] >= coeff[2:])
    if not usable.any():
        return 0.
    return max(0., float(np.max(coeff[1:-1][usable])))


def _block_peak(data, sample, view, background_power=None):
    import numpy as np

    from app.evidence.mouth_hold_sound import HOP

    block = data[sample:sample + 2 * HOP]
    if np.sqrt(np.mean((block - block.mean()) ** 2)) < .007:
        return 0.
    lags = np.arange(35, 221)
    if view == "raw":
        return _peaks(block, lags)

    tapered = (block - block.mean()) * np.hanning(len(block))
    spectrum = np.fft.rfft(tapered)
    hz = np.fft.rfftfreq(len(block), 1 / RATE)
    if view == "flank_whitening":
        power = np.abs(spectrum) ** 2
        gain = np.sqrt(np.maximum(.04, 1 - 1.5 * background_power / (power + 1e-12)))
        return _peaks(np.fft.irfft(spectrum * gain), lags)
    band_signals = []
    for lower, upper in BANDS:
        filtered = np.fft.irfft(spectrum * ((hz >= lower) & (hz < upper)))
        band_signals.append(filtered / max(.003, float(np.sqrt(np.mean(filtered ** 2)))))
    return _peaks(np.sum(band_signals, axis=0), lags)


def _watch(waveform):
    import numpy as np

    from app.detectors.video.mouth_hold_sound import measure
    from app.evidence.mouth_hold_sound import HOP, block_starts

    data = waveform.astype(np.float64) / 32768
    mapping = {"samples": len(data), "timeline_time_base": "1/16000",
               "timeline": [{"start_sample": 0, "end_sample": len(data), "source_pts": 0}]}
    output = []
    for left, right in WINDOWS:
        around = (*range(left - 6400, left - 3200, HOP), *range(right + 3200, right + 6400, HOP))
        frames = np.stack([data[i:i + 2 * HOP] for i in around])
        power = np.abs(np.fft.rfft((frames - frames.mean(axis=1, keepdims=True)) * np.hanning(2 * HOP), axis=1)) ** 2
        background = np.median(power, axis=0)
        stats = {}
        for view in VIEWS:
            peaks = [_block_peak(data, i, view, background) for i in block_starts(left, right)]
            # Use a stable scalar for the weakest of the required two-thirds.
            # 19 blocks require the top 13 to support an interval.
            order = len(peaks) - ((2 * len(peaks) + 2) // 3)
            stats[view] = round(sorted(peaks)[order], 10)
        review = {"schema": "chipmunk.video-mouth-local-hold/2", "status": "matched",
                  "pts": [left, right], "match": {"frames": [0, 1]}}
        raw = measure(data, mapping, review, "1/16000")
        if (stats["raw"] >= .78) != raw["sustained_periodicity"]:
            raise ValueError("The raw response disagrees with the current implementation")
        output.append(stats)
    return output


def _environment(sounds, manifest):
    import numpy as np

    from scripts.audio_vocal_event_mixtures import decode

    result = {}
    for scene, facts in manifest.items():
        pcm = np.frombuffer(decode(sounds / f"{scene}.ogg"), dtype="<i2")
        excerpts = [pcm[(offset + 2) * RATE:(offset + 8) * RATE] for offset in facts["offsets"]]
        if any(len(row) != 6 * RATE for row in excerpts):
            raise ValueError(f"The locked offsets cannot provide six seconds of {scene}")
        result[scene] = excerpts
    return result


def _inputs(cache, plan, sounds):
    from scripts.audio_sine_resynthesis_transfer import native_rows

    parent = json.loads((cache / "freeze.json").read_bytes())
    selected = {source["source_id"] for source in plan["groups"]}
    sources = {int(row["id"]): row for row in parent["pairs"]}
    parent["pairs"] = [sources[index] for index in sorted(selected)]
    original, backgrounds = {}, []
    for row, pcm in native_rows(cache, parent):
        if row["role"] == "background_only":
            backgrounds.append(pcm)
        elif row["condition"] == "clean":
            original[row["id"], row["side"]] = pcm
    environment = _environment(sounds, plan["environmental_sources"])
    yield from _mix(original, backgrounds, environment, plan)


def _mix(original, backgrounds, environment, plan):
    import numpy as np

    for group in plan["groups"]:
        index = group["group"]
        is_held = group["split"] == "held"
        offset = (2 if is_held else 0) + index % 2
        room = (4 if is_held else 0) + index % 4
        for side in ("original", "variant"):
            foreground = original[group["source_id"], side].astype(np.float64)
            disturbances = []
            for scene in SCENES:
                noise = backgrounds[room] if scene == "competing" else environment[scene][offset]
                noise_id = room if scene == "competing" else offset
                values = noise.astype(np.float64)
                if not np.linalg.norm(values):
                    raise ValueError("A background excerpt contains no measurable sound")
                added = values * np.linalg.norm(foreground) / np.linalg.norm(values) * 10 ** .25
                disturbances.append((scene, noise_id, added))
            peak = max(float(np.max(np.abs(foreground + values))) for _, _, values in disturbances)
            peak = max(peak, float(np.max(np.abs(foreground))),
                       *(float(np.max(np.abs(values))) for _, _, values in disturbances))
            scale = min(1., 32000 / peak)
            speech = np.rint(foreground * scale).astype("<i2")
            yield index, side, "clean", None, speech
            for scene, noise_id, values in disturbances:
                background = np.rint(values * scale).astype("<i2")
                yield index, side, scene, noise_id, speech + background
                yield index, side, "background_" + scene, noise_id, background


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("phase", choices=("freeze", "measure"))
    cli.add_argument("source", type=Path)
    cli.add_argument("sounds", type=Path)
    cli.add_argument("folder", type=Path)
    args = cli.parse_args()
    plan = _source_plan(args.source, args.sounds)
    freeze = args.folder / "freeze.json"
    if args.phase == "freeze":
        args.folder.mkdir(mode=0o700, exist_ok=False)
        freeze.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
        return
    if plan != json.loads(freeze.read_bytes()):
        raise ValueError("The route or inputs changed after the freeze")
    rows = [{"group": group, "side": side, "scene": scene, "background_id": background,
             "pcm_sha256": sha(samples.tobytes()), "windows": _watch(samples)}
            for group, side, scene, background, samples in _inputs(args.source, plan, args.sounds)]
    receipt = {"plan": plan, "rows": rows}
    receipt["summary"] = summarize(receipt)
    with (args.folder / "result.json").open("x") as output:
        output.write(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
