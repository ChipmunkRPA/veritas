# SPDX-License-Identifier: Apache-2.0
"""Fit separate enhanced-audio points and check transfer to two held recordings."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from scripts import audio_gtcrn_traffic as enhancement
from scripts import audio_radar_transfer as radar

common, require = radar.common, radar.require
RATE, COUNT = 16_000, 48_000
FROZEN = common.PUBLIC / "audio-radar-transfer-2026-11-12.json"
DEV = common.PUBLIC / "audio-radar-long-confirmation-2026-11-12.json"
COMMONS = common.BASE / "audio-commons-slice-screen-v247/authored/output/private-admitted-slices.json"
ROLE_TAPES = {"fit": (1, 2), "calibration": (6, 4)}
HEADS = ("enhanced", "joint")
RULE = {"min_added_synthetic_groups": 5, "min_scenes_with_additions": 2, "max_added_genuine_groups": 3,
        "max_added_gain_control": 0, "max_isolated_flags": 0}


def save(path, value):
    with path.open("xb") as stream:
        stream.write(common.encoded(value))


def canonical(pcm):
    return enhancement.float_pcm(enhancement.pcm(pcm))


def rms(pcm):
    return float(np.sqrt(np.mean(np.asarray(pcm, dtype=np.float64) ** 2)))


def match_views(voice, noises, ratio_db):
    source = voice.astype(np.float64)
    foreground = rms(source)
    require(foreground > 0, "Silent foreground")
    views = {"gain_control": source}
    for key, noise in noises.items():
        level = rms(noise)
        require(level > 0 and len(noise) == len(voice), "Invalid matched recording excerpt")
        views[key] = source + noise * (foreground / level) * 10 ** (-ratio_db / 20)
    gain = min(1., .98 / max(float(np.abs(pcm).max()) for pcm in views.values()))
    return {key: canonical(pcm * gain) for key, pcm in views.items()}


def source_clip(path, row):
    from app.detectors.audio.decode import decode_audio

    require(common.sha(path) == row["flac_sha256"], "Publisher source hash differs")
    clip = decode_audio(path, source_sha256=row["flac_sha256"], max_seconds=15)
    n = len(clip.pcm)
    require(clip.mapping["timeline"] == [{"start_sample": 0, "end_sample": n, "source_pts": 0}]
            and clip.mapping["pcm_sha256"] == row["pcm_sha256"], "Exact native speech differs from the published ledger")
    return clip.pcm


def development_tapes(source):
    clips, pins = {}, {}
    for slot in set().union(*ROLE_TAPES.values()):
        qualification = common.BASE / f"audio-commons-background-v234/qualification-{slot:02}.json"
        path = source / f"slot-{slot:02}.pcm.wav"
        require(common.sha(path) == common.read(qualification)["wav_sha256"], "Development recording hash differs")
        values = enhancement.float_pcm(common.decode(path))
        selected = sorted((row for row in common.read(COMMONS)["excerpts"] if row["slot"] == slot),
                          key=lambda row: row["start_sample"])[:4]
        require(len(selected) == 4, "Incomplete development scene")
        clips[slot] = []
        for row in selected:
            start = row["start_sample"]
            pcm = values[start:start + COUNT]
            require(common.digest(enhancement.pcm(pcm)) == row["pcm_sha256"], "Frozen background samples differ")
            clips[slot].append(pcm)
        pins[str(slot)] = {"wav_sha256": common.sha(path), "excerpts": selected}
    return clips, pins


def prepare(run, radar_source, commons_source, assets):
    from app.detectors.audio.decode import decode_audio

    require(common.sha(assets / enhancement.MODEL) == enhancement.MODEL_SHA
            and common.sha(assets / "LICENSE") == enhancement.LICENSE_SHA, "Enhancement assets differ")
    tapes, pins = development_tapes(commons_source)
    scenes = {}
    for key, starts in radar.SCENES.items():
        path = radar_source / f"{key}.ogg"
        require(common.sha(path) == radar.SOURCE_FILES[path.name], "Held noise recording differs")
        scenes[key] = decode_audio(path, source_sha256=radar.SOURCE_FILES[path.name], max_seconds=max(starts) + 5).pcm
    run.mkdir(mode=0o700)
    (run / "pcm").mkdir()
    rows = []

    def keep(pcm, *, role, group, label, scene, baseline=None):
        raw = enhancement.pcm(pcm)
        number = len(rows)
        (run / "pcm" / f"{number:03}.pcm").write_bytes(raw)
        rows.append({"id": number, "role": role, "group": group, "label": label, "scene": scene,
                     "baseline": baseline, "samples": len(pcm), "pcm_sha256": common.digest(raw)})

    dev = common.read(DEV)["rows"]
    for rank, row in enumerate(dev):
        rank %= 20
        role = "fit" if rank < 12 else "calibration"
        pcm = source_clip(radar_source / "confirmation" / f"{row['id']}.flac", row)
        start = (len(pcm) - COUNT) // 2
        voices = match_views(pcm[start:start + COUNT], {f"slot-{slot}": tapes[slot][rank % 4] for slot in ROLE_TAPES[role]},
                             -5 if rank % 2 == 0 else -10)
        for name, values in voices.items():
            keep(values, role=role, group=row["id"], label=row["label"], scene=name)
    for role, slots in ROLE_TAPES.items():
        for slot in slots:
            for rank, pcm in enumerate(tapes[slot]):
                keep(pcm, role=role, group=f"slot-{slot}", label="isolated", scene=f"isolated-{rank}")
    hold = [row for row in common.read(FROZEN)["rows"] if row["stratum"] == "short"]
    for rank, row in enumerate(hold):
        pcm = source_clip(radar_source / "panel" / "clips" / f"{row['id']}.flac", row)
        _, views = radar.acoustic_views(pcm, scenes, rank % radar.SIZE)
        for scene, values in views.items():
            receipt = row["mixtures"][scene]
            require(radar.pcm_hash(values) == receipt["pcm_sha256"], "Held audio did not reproduce published mixture")
            keep(values, role="held", group=row["id"], label=row["label"], scene=scene, baseline=receipt["response"]["decision"])
    for scene, starts in radar.SCENES.items():
        for rank, start in enumerate(starts):
            keep(scenes[scene][start * RATE:start * RATE + COUNT], role="held", group=scene, label="isolated",
                 scene=f"isolated-{rank}")
    save(run / "roster.json", rows)
    save(run / "freeze.json", {"schema": "chipmunk.audio-noise-views-freeze/1", "runner_sha256": common.sha(Path(__file__)),
                               "roster_sha256": common.sha(run / "roster.json"), "dev_sha256": common.sha(DEV),
                               "held_sha256": common.sha(FROZEN), "tapes": pins, "enhancer_sha256": enhancement.MODEL_SHA,
                               "heads": list(HEADS), "rule": RULE, "voice_assets": common.assets()})


def enhanced_view(wave, session):
    spectra, window = enhancement.signal_spectra(wave)
    states = {name: np.zeros(shape, dtype=np.float32) for name, shape in enhancement.INPUTS.items() if name != "mix"}
    outputs = []
    for frame in spectra:
        mix = np.stack((frame.real, frame.imag), axis=-1).astype(np.float32)[None, :, None, :]
        out, *cache = session.run(None, {"mix": mix, **states})
        states = dict(zip(states, cache, strict=True))
        outputs.append(out[0, :, 0, 0] + 1j * out[0, :, 0, 1])
    frames = np.fft.irfft(outputs, n=enhancement.FFT, axis=1) * window
    out = np.zeros((len(frames) - 1) * enhancement.HOP + enhancement.FFT)
    weight = np.zeros_like(out)
    for i, frame in enumerate(frames):
        start = i * enhancement.HOP
        out[start:start + len(window)] += frame
        weight[start:start + len(window)] += window**2
    span = slice(enhancement.FFT // 2, enhancement.FFT // 2 + len(wave))
    result = out[span] / weight[span]
    require(result.shape == wave.shape and np.isfinite(result).all(), "Enhancement changed the source sample clock")
    return canonical(result)


def load_frozen(run):
    frozen, rows = (common.read(run / name) for name in ("freeze.json", "roster.json"))
    require(frozen["runner_sha256"] == common.sha(Path(__file__)) and frozen["roster_sha256"] == common.sha(run / "roster.json")
            and frozen["rule"] == RULE and frozen["dev_sha256"] == common.sha(DEV) and frozen["held_sha256"] == common.sha(FROZEN),
            "Frozen study differs")
    return frozen, rows


def score(run, assets, role):
    import onnxruntime as ort

    from app.detectors.audio import calibration
    from app.detectors.audio.content import window_in_speech
    from app.detectors.audio.upgraded import SpeechActivity
    from scripts.run_audio_specialists_v15 import load_voice

    frozen, all_rows = load_frozen(run)
    heads_hash = None
    if role == "held":
        heads_hash = common.sha(run / "heads.json")
        require(common.read(run / "heads.json")["development_sha256"] == common.sha(run / "development.json"),
                "Freeze operating points before held inference")
    checkpoint = assets / enhancement.MODEL
    require(common.sha(checkpoint) == enhancement.MODEL_SHA and common.sha(assets / "LICENSE") == enhancement.LICENSE_SHA,
            "Enhancement checkpoint differs")
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(checkpoint), sess_options=options, providers=["CPUExecutionProvider"])
    require({item.name: item.shape for item in session.get_inputs()} == enhancement.INPUTS, "Unexpected enhancement graph")
    weights = {key: value for key, value in frozen["voice_assets"].items() if key not in common.SOUND_ASSETS}
    require(all(common.sha(common.WEIGHTS / key) == pin for key, pin in weights.items()), "Speech weights differ")
    voice = load_voice({"weights": str(common.WEIGHTS), "assets": weights})
    activity = SpeechActivity(common.WEIGHTS / "silero_vad.jit")
    selected = [row for row in all_rows if (row["role"] == "held") == (role == "held")]
    output = []
    for i, row in enumerate(selected, 1):
        raw = (run / "pcm" / f"{row['id']:03}.pcm").read_bytes()
        require(common.digest(raw) == row["pcm_sha256"] and len(raw) == 2 * row["samples"], "Frozen mixture changed")
        wave = enhancement.float_pcm(raw)
        enhanced = enhanced_view(wave, session)
        original_response, enhanced_response = map(calibration.logit, voice.predict([wave, enhanced]))
        usable = window_in_speech(activity.spans(enhanced), 0., len(enhanced) / RATE)
        output.append({**row, "enhanced_admitted": usable, "raw_logit": original_response, "enhanced_logit": enhanced_response,
                       "suppression_db": enhancement.db_ratio(enhancement.energy(wave), enhancement.energy(enhanced)),
                       "enhanced_sha256": common.digest(enhancement.pcm(enhanced))})
        if i % 24 == 0 or i == len(selected):
            print(f"{role}: {i}/{len(selected)} matched views scored", flush=True)
    save(run / f"{role}.json", {"freeze_sha256": common.sha(run / "freeze.json"), "onnxruntime": ort.__version__,
                               "heads_sha256": heads_hash, "rows": output})


def inputs(row):
    return [row["raw_logit"], row["enhanced_logit"], row["suppression_db"]]


def response(row, head, parameters):
    if head == "enhanced":
        return row["enhanced_logit"]
    values = (np.asarray(inputs(row)) - parameters["center"]) / parameters["scale"]
    return float(values @ parameters["coef"][:-1] + parameters["coef"][-1])


def fit(run):
    from scipy.optimize import minimize

    frozen, _ = load_frozen(run)
    source = common.read(run / "development.json")
    require(source["freeze_sha256"] == common.sha(run / "freeze.json"), "Development scores differ")
    rows = [row for row in source["rows"] if row["role"] == "fit" and row["label"] != "isolated"]
    x = np.asarray([inputs(row) for row in rows])
    y = np.asarray([row["label"] == "spoof" for row in rows], dtype=float)
    center, scale = x.mean(axis=0), np.maximum(x.std(axis=0), 1e-6)
    x = np.column_stack(((x - center) / scale, np.ones(len(x))))

    def loss(coef):
        score = x @ coef
        return float(np.mean(np.logaddexp(0, score) - y * score) + .10 * np.sum(coef[:-1]**2))

    solution = minimize(loss, np.zeros(x.shape[1]), method="BFGS")
    require(solution.success and np.isfinite(solution.x).all(), "Joint point fit did not converge")
    parameters = {"center": center.tolist(), "scale": scale.tolist(), "coef": solution.x.tolist()}
    genuine = [row for row in source["rows"] if row["role"] == "calibration" and row["label"] != "spoof"
               and row["enhanced_admitted"]]
    require(len({row["group"] for row in genuine if row["label"] == "bonafide"}) >= 6,
            "Too few separate genuine groups to select the new points")
    points = {head: max(response(row, head, parameters) for row in genuine) for head in HEADS}
    save(run / "heads.json", {"schema": frozen["schema"], "development_sha256": common.sha(run / "development.json"),
                              "calibration_admitted": len(genuine), "joint_parameters": parameters, "points": points})


def report(run):
    _, roster = load_frozen(run)
    heads, held = (common.read(run / name) for name in ("heads.json", "held.json"))
    require(heads["development_sha256"] == common.sha(run / "development.json")
            and held["freeze_sha256"] == common.sha(run / "freeze.json")
            and held["heads_sha256"] == common.sha(run / "heads.json"), "Holdout result differs")
    require(all(all(row[key] == value for key, value in source.items()) for row, source in zip(
                held["rows"], (row for row in roster if row["role"] == "held"), strict=True)),
            "Held scores do not cover the exact frozen roster")
    results = {}
    for head in HEADS:
        cells, gains, alarms, gain_scenes = defaultdict(Counter), set(), set(), set()
        for row in held["rows"]:
            baseline = row["baseline"]
            isolated = row["label"] == "isolated"
            admitted = row["enhanced_admitted"]
            candidate = admitted and response(row, head, heads["joint_parameters"]) > heads["points"][head]
            additional = candidate and baseline in {None, "unassessed", "inconclusive"}
            kind = "isolated" if isolated else f"{row['scene']}/{row['label']}"
            cells[kind].update({"files": 1, "incumbent_ai": baseline == "ai", "enhanced_admitted": admitted,
                                "additional_ai": additional, "combined_ai": baseline == "ai" or additional})
            if additional and row["scene"] in radar.SCENES:
                if row["label"] == "spoof":
                    gains.add(row["group"])
                    gain_scenes.add(row["scene"])
                else:
                    alarms.add(row["group"])
        advances = (len(gains) >= RULE["min_added_synthetic_groups"] and len(gain_scenes) >= RULE["min_scenes_with_additions"]
                    and len(alarms) <= RULE["max_added_genuine_groups"]
                    and cells["gain_control/bonafide"]["additional_ai"] <= RULE["max_added_gain_control"]
                    and cells["isolated"]["additional_ai"] <= RULE["max_isolated_flags"])
        results[head] = {"cells": cells, "added_synthetic_groups": len(gains), "scenes_with_additions": sorted(gain_scenes),
                         "added_genuine_groups": len(alarms), "advances": advances}
    output = {"schema": "chipmunk.audio-noise-views-transfer/1", "freeze": common.read(run / "freeze.json"),
              "heads": heads, "held_sha256": common.sha(run / "held.json"), "results": results}
    save(run / "result.json", output)
    print(json.dumps(results, indent=2, sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "development", "fit", "held", "report"))
    parser.add_argument("run", type=Path)
    parser.add_argument("--assets", type=Path)
    parser.add_argument("--radar-source", type=Path)
    parser.add_argument("--commons-source", type=Path)
    args = parser.parse_args()
    if args.phase == "prepare":
        if not (args.assets and args.radar_source and args.commons_source):
            parser.error("prepare requires all three source directories")
        prepare(args.run, args.radar_source, args.commons_source, args.assets)
    elif args.phase in {"development", "held"}:
        if not args.assets:
            parser.error("score requires --assets")
        score(args.run, args.assets, args.phase)
    elif args.phase == "fit":
        fit(args.run)
    else:
        report(args.run)


if __name__ == "__main__":
    main()
