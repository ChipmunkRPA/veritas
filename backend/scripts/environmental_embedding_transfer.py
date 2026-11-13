# SPDX-License-Identifier: Apache-2.0
"""Compare learned and handcrafted evidence for environmental audio origin."""

from __future__ import annotations

import argparse
import math
from collections import Counter, defaultdict
from pathlib import Path

from scripts import audio_vocal_event_mixtures as io

BASELINE = io.PUBLIC / "environmental-sound-transfer-2026-11-11.json"
SCHEMA = "chipmunk.environmental-embedding-transfer/1"
PINS = {
    "environment_freeze": "6933be668bdd576ee024de08713d4819b8253126a0ee744b567dece19c0af7a8",
    "environment_acquisition": "a5c5009a9079bd47d59a9f98ba3ad40679eb57bbbfe10337f8fb6c6a1d6cfd4c",
    "speech_roster": "2d252123922194a073dc9ed16a03ae16dfd883b72acc04296d9171dce963ff86",
    "feature_code": "909f010e1b23e1ea9cd606279263e33e211b136ffb900ed6aac551aa6e5da4ac",
    "feature_weights": io.SOUND_ASSETS["panns-Cnn10.pth"],
}
ARMS = ("spectral", "learned", "fused", "learned_native_fit")
ROLES = ("fit",) * 12 + ("selection",) * 10 + ("held",) * 18
SCENES = {"fit": ("native", "speech_mix"), "selection": ("native", "speech_mix"),
          "held": ("native", "speech_mix", "balanced_mix", "opus")}
POINTS = {"native": "native", "speech_mix": "speech_mix", "balanced_mix": "speech_mix", "opus": "native"}
METHOD = {
    "samples": 48000, "sample_rate": 16000, "environment_to_speech_db": {"speech_mix": -6, "balanced_mix": 0},
    "fit": "Balanced source, scene and origin weights. Ridge penalty 0.50, with equally scaled feature blocks.",
    "arms": ["Previous spectral magnitude vector", "Frozen sound embedding",
             "Equal-energy combination of these feature blocks", "Same embedding fit on native only"],
    "selection": "The second-highest genuine response in each delivery condition, exceeded strictly.",
    "speech": "Use distinct retained clean genuine voices by role, same foreground across all origin lanes within a source.",
    "advance": {"min_each_generator_at_loud_speech": 11, "max_ordinary_at_loud_speech": 2,
                "min_each_generator_in_other_views": 9, "max_ordinary_in_other_views": 3},
}


def peak_mix(environment, speech, db):
    import numpy as np

    if db == -6:
        from scripts.environmental_sound_transfer import mix

        return mix(environment, speech)
    a, b = (np.frombuffer(raw, "<i2").astype(float) for raw in (environment, speech))
    blended = a * np.linalg.norm(b) / np.linalg.norm(a) + b
    blended *= min(1.0, 30_000 / np.max(np.abs(blended)))
    return np.rint(blended).astype("<i2").tobytes()


def feature_inputs(environment, speech, run, output):
    from scripts.audio_phase_corroboration import sample_bytes
    from scripts.environmental_sound_transfer import clip, convert, intake

    frozen, acquisition = intake(environment)
    io.require(io.sha(environment / "freeze.json") == PINS["environment_freeze"]
               and io.sha(environment / "acquisition.json") == PINS["environment_acquisition"]
               and io.sha(speech / "roster.json") == PINS["speech_roster"], "Frozen sources differ")
    voice_roster = [row for row in io.read(speech / "roster.json")
                    if row["placement"] == "clean" and row["family"] == "genuine"]
    voices = {"fit": [row for row in voice_roster if row["role"] == "archived"][:4],
              "selection": [row for row in voice_roster if row["role"] == "archived"][4:],
              "held": [row for row in voice_roster if row["role"] == "transfer"]}
    groups = [row["group"] for rows in voices.values() for row in rows]
    io.require(list(map(len, voices.values())) == [4, 4, 7] and len(set(groups)) == 15, "Speech roles overlap")
    output.mkdir(mode=0o700)
    rows = []
    offsets = Counter()
    for source in frozen["groups"]:
        role, group = source["role"], source["group"]
        voice = voices[role][offsets[role] % len(voices[role])]
        offsets[role] += 1
        speech_bytes = sample_bytes(speech, voice)
        for lane in sorted(source["parts"]):
            native = clip(environment, acquisition, f"{group:02}-{lane}")
            for scene in SCENES[role]:
                if scene == "native":
                    raw = native
                elif scene == "opus":
                    raw = convert(run, native)
                else:
                    raw = peak_mix(native, speech_bytes, METHOD["environment_to_speech_db"][scene])
                index = len(rows)
                (output / f"{index:03}.pcm").write_bytes(raw)
                rows.append({"id": index, "role": role, "group": group, "lane": lane, "scene": scene,
                             "pcm_sha256": io.digest(raw), "native_pcm_sha256": io.digest(native),
                             "voice_pcm_sha256": io.digest(speech_bytes) if scene.endswith("mix") else None})
    for name in ("delivery.wav", "delivery.opus"):
        (run / name).unlink()
    return rows


def prepare(environment, speech, run):
    import subprocess

    run.mkdir(mode=0o700)
    rows = feature_inputs(environment, speech, run, run / "pcm")
    io.save(run / "roster.json", rows)
    freeze = {"schema": SCHEMA, "method": METHOD, "pins": PINS, "runner": io.sha(Path(__file__)),
              "roster_sha256": io.sha(run / "roster.json"), "prior_public_sha256": io.sha(BASELINE),
              "ffmpeg": subprocess.check_output(["ffmpeg", "-version"], text=True, timeout=5).splitlines()[0],
              "responses_at_freeze": 0}
    io.save(run / "freeze.json", freeze)
    print("Frozen", len(rows), "source-bound deliveries.", flush=True)


def read_run(run):
    freeze, rows = [io.read(run / name) for name in ("freeze.json", "roster.json")]
    io.require(freeze["schema"] == SCHEMA and freeze["method"] == METHOD and freeze["pins"] == PINS
               and freeze["runner"] == io.sha(Path(__file__)) and freeze["roster_sha256"] == io.sha(run / "roster.json")
               and freeze["prior_public_sha256"] == io.sha(BASELINE), "Procedure or frozen roster changed")
    return freeze, rows


def window(run, row):
    data = (run / "pcm" / f"{row['id']:03}.pcm").read_bytes()
    io.require(len(data) == 2 * METHOD["samples"] and io.digest(data) == row["pcm_sha256"], "PCM binding changed")
    return data


def extract(run, held=False):
    import numpy as np
    import torch
    from scipy.signal import resample_poly

    from app.third_party.panns.cnn10 import Cnn10
    from scripts.environmental_sound_transfer import features

    _, rows = read_run(run)
    weights = io.WEIGHTS / "panns-Cnn10.pth"
    source = io.ROOT / "backend/app/third_party/panns/cnn10.py"
    io.require(io.sha(weights) == PINS["feature_weights"] and io.sha(source) == PINS["feature_code"],
               "Frozen representation differs")
    if held:
        head = io.read(run / "heads.json")
        io.require(head["freeze_sha256"] == io.sha(run / "freeze.json")
                   and head["development_sha256"] == io.sha(run / "development.json"), "Held points were not fixed")
    torch.set_num_threads(1)
    model = Cnn10(32000, 1024, 320, 64, 50, 14000, 527)
    model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True)["model"], strict=True)
    model.eval()
    measured = []
    for row in rows:
        if (row["role"] == "held") != held:
            continue
        raw = window(run, row)
        pcm = np.frombuffer(raw, "<i2").astype(np.float32) / 32768
        wave = resample_poly(pcm, 2, 1).astype(np.float32)
        with torch.inference_mode():
            vector = model(torch.from_numpy(wave)[None])["embedding"][0].numpy()
        io.require(vector.shape == (512,) and np.isfinite(vector).all(), "Sound embedding is incomplete")
        measured.append({"id": row["id"], "spectral": features(raw)["magnitude"], "learned": vector.tolist()})
        if len(measured) % 36 == 0:
            print("Held" if held else "Development", len(measured), flush=True)
    io.save(run / ("held.json" if held else "development.json"), measured)
    print("Held" if held else "Development", len(measured), "complete", flush=True)


def design(measured, arm):
    import numpy as np

    parts = ("spectral", "learned") if arm == "fused" else (("spectral",) if arm == "spectral" else ("learned",))
    widths = [len(measured[0][part]) for part in parts]
    matrix = np.asarray([[value for part in parts for value in row[part]] for row in measured])
    scales = np.repeat([math.sqrt(width) for width in widths], widths)
    return matrix, scales


def scores(head, measured, arm):
    matrix, block_scale = design(measured, arm)
    return (((matrix - head["mean"]) / head["scale"] / block_scale) @ head["coef"]).tolist()


def fit(run):
    import numpy as np

    _, rows = read_run(run)
    dev = io.read(run / "development.json")
    observations = {row["id"]: row for row in dev}
    heads, selected = {}, []
    scope = [row for row in rows if row["role"] == "selection"]
    for arm in ARMS:
        train = [row for row in rows if row["role"] == "fit"
                 and (arm != "learned_native_fit" or row["scene"] == "native")]
        matrix, block_scale = design([observations[row["id"]] for row in train], arm)
        truth = np.asarray([row["lane"] != "real" for row in train])
        weight = np.where(truth, .5 / truth.sum(), .5 / (~truth).sum())
        mean = weight @ matrix
        scale = np.maximum(np.sqrt(weight @ (matrix-mean)**2), 1e-5)
        z = (matrix-mean)/scale/block_scale
        # Solve in sample space to keep the fit small for 512 features.
        weighted = z * np.sqrt(weight[:, None])
        coef = weighted.T @ np.linalg.solve(weighted @ weighted.T + .50*np.eye(len(z)),
                                             np.sqrt(weight)*(truth-.5))
        head = {"mean": mean.tolist(), "scale": scale.tolist(), "coef": coef.tolist()}
        values = scores(head, [observations[row["id"]] for row in scope], arm)
        selected.append(values)
        heads[arm] = head
    decisions = [{"id": row["id"], "responses": dict(zip(ARMS, values, strict=True))}
                 for row, values in zip(scope, zip(*selected, strict=True), strict=True)]
    points = fit_points(scope, decisions)
    io.save(run / "heads.json", {"freeze_sha256": io.sha(run / "freeze.json"),
                                  "development_sha256": io.sha(run / "development.json"),
                                  "heads": heads, "selection": decisions, "points": points})
    print("Native and competing-speech points fixed.", flush=True)


def fit_points(rows, responses):
    points = {}
    for arm in ARMS:
        points[arm] = {}
        for scene in SCENES["selection"]:
            genuine = [record["responses"][arm] for row, record in zip(rows, responses, strict=True)
                       if row["scene"] == scene and row["lane"] == "real"]
            io.require(len(genuine) == 10, "The ordinary selection panel is incomplete")
            points[arm][scene] = sorted(genuine)[-2]
    return points


def summarize(rows, responses, points):
    cells = {arm: defaultdict(Counter) for arm in ARMS}
    for row, item in zip(rows, responses, strict=True):
        scene, lane = row["scene"], row["lane"]
        for arm in ARMS:
            score = item["responses"][arm]
            flag = score > points[arm][POINTS[scene]]
            cells[arm][scene][lane] += flag
    outcomes = {}
    for arm, scenes in cells.items():
        gates = {scene: counts["real"] <= (2 if scene == "speech_mix" else 3)
                 and all(counts[lane] >= (11 if scene == "speech_mix" else 9) for lane in ("audiogen", "audiolcm"))
                 for scene, counts in scenes.items()}
        outcomes[arm] = {"flags": dict(scenes), "passes_view": gates, "passes": all(gates.values())}
    return outcomes


def report(run):
    freeze, rows = read_run(run)
    trained, held = [io.read(run / name) for name in ("heads.json", "held.json")]
    selection = [row for row in rows if row["role"] == "selection"]
    tests = [row for row in rows if row["role"] == "held"]
    io.require([row["id"] for row in tests] == [row["id"] for row in held], "Held order changed")
    all_scores = [scores(trained["heads"][arm], held, arm) for arm in ARMS]
    outcomes = [{"id": row["id"], "responses": dict(zip(ARMS, values, strict=True))}
                for row, values in zip(tests, zip(*all_scores, strict=True), strict=True)]
    result = summarize(tests, outcomes, trained["points"])
    receipt = {"freeze": freeze, "rows": rows, "points": trained["points"], "selection": trained["selection"],
               "held": outcomes, "result": result, "heads_sha256": io.sha(run / "heads.json"),
               "held_features_sha256": io.sha(run / "held.json")}
    io.require(fit_points(selection, receipt["selection"]) == receipt["points"], "Selection point changed")
    io.save(run / "receipt.json", receipt)
    print(io.encoded(result).decode(), end="")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "development", "fit", "held", "report", "run"))
    parser.add_argument("run", type=Path)
    parser.add_argument("--environment", type=Path, default=io.BASE / "environmental-origin-2026-11-11")
    parser.add_argument("--speech", type=Path, default=io.BASE / "audio-spatial-interference-2026-11-11/run")
    args = parser.parse_args()
    if args.phase in ("prepare", "run"):
        prepare(args.environment, args.speech, args.run)
    if args.phase in ("development", "run"):
        extract(args.run)
    if args.phase in ("fit", "run"):
        fit(args.run)
    if args.phase in ("held", "run"):
        extract(args.run, held=True)
    if args.phase in ("report", "run"):
        report(args.run)


if __name__ == "__main__":
    main()
