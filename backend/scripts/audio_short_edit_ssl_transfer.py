# SPDX-License-Identifier: MIT
"""Compare temporal speech representations on the same frozen short-edit task."""

import argparse
import json
from pathlib import Path

from scripts import audio_short_edit_detector_training as parent

SOURCE = parent.SOURCE
SCHEMA = "chipmunk.audio-short-edit-ssl-transfer/1"
EXPERIMENT = {
    "parent_method": parent.METHOD,
    "change": "Replace log-Mel inputs with frozen WavLM hidden states from layers 4, 8 and 12, without changing "
              "source roles, the temporal model, objective, speech gate, proposal grid, or operating-point rules.",
    "encoding": "One original six-second native mono PCM per call. Match frame centers at 200 + 320i samples. "
                "Reduce each 768-wide layer by a fixed signed projection, then interpolate to the common 10 ms "
                "head grid with edge hold. This does not add time resolution to the 20 ms input representations.",
    "projection": "NumPy PCG64 seed 611407. Signed projections with orthogonal columns from reduced QR, "
                  "taking 22, 21 and 21 columns from the three layers. No source-dependent fit.",
    "readout": "Report matched-control ordering, localization, ordinary and isolated-background alerts, and "
               "unassessed files for each held generator. Compare to the unchanged log-Mel and app checks.",
}
LAYERS = (4, 8, 12)
WIDTHS = (22, 21, 21)
FRAMES = 299
SPEECH = "e1122837f4154c511485fe0b9c64455f7b929c96fbb8d79fbdb336383ebd3720"
TRANSFER_ROSTER = "72cddb5d4dd8621d8d179d474b79a3f48262a887100dcc6a87658dfee05d605f"
FLANK_METHOD = EXPERIMENT | {"selection": "Require 300 ms flanks on both sides, separated by 100 ms guards. Rank "
              "each speech-supported proposal by its mean content score minus the larger flank mean, with the same "
              "learned endpoint term. At least half of each flank must carry detected speech. Refit genuine-only "
              "cutoffs on the original calibration group before running the holdout; keep both heads frozen."}


def frozen_encoder(weights, *, variable_context=False):
    import numpy as np
    import torch

    from scripts import audio_wavlm_features_v13 as encoder

    model = encoder.load(weights)
    rng = np.random.default_rng(611407)
    projections = [torch.from_numpy(np.linalg.qr(rng.choice([-1., 1.], size=(encoder.WIDTH, width)))[0].astype(np.float32))
                   for width in WIDTHS]
    source_clock = 200 + 320 * np.arange(FRAMES)
    head_clock = np.arange(parent.FRAMES) * parent.HOP

    @torch.inference_mode()
    def encode(audio):
        waveform = torch.from_numpy(audio)[None]
        response = model(waveform, attention_mask=torch.ones_like(waveform, dtype=torch.long),
                         output_hidden_states=True, return_dict=True)
        frames = (len(audio) - 400) // 320 + 1 if variable_context else FRAMES
        SOURCE.check(response.last_hidden_state.shape == (1, frames, encoder.WIDTH), "Encoder clock changed")
        encoded = torch.cat([response.hidden_states[layer][0] @ projection
                             for layer, projection in zip(LAYERS, projections, strict=True)], dim=1).numpy()
        source = 200 + 320 * np.arange(frames) if variable_context else source_clock
        target = np.arange(len(audio) // parent.HOP) * parent.HOP if variable_context else head_clock
        values = np.asarray([np.interp(target, source, column) for column in encoded.T], dtype="<f4")
        SOURCE.check(np.isfinite(values).all(), "Non-finite speech representation")
        return values

    return encode


def cache_features(args):
    import numpy as np
    import torch

    from scripts import audio_wavlm_features_v13 as encoder
    from scripts.audio_strong_traffic_view import network_free

    plan, digest = parent.load_plan(args.data)
    args.cache.mkdir(mode=0o700, exist_ok=True)
    asset_keys = ("repository", "revision", "assets_sha256", "parameters", "weight_terms", "runtime")
    lock = {"schema": SCHEMA, "method": EXPERIMENT, "source_sha256": digest,
            "encoder": {key: encoder.SPEC[key] for key in asset_keys}, "epochs": parent.EPOCHS,
            "seed": parent.SEED, "speech_sha256": SPEECH}
    freeze = args.cache / "freeze.json"
    if freeze.exists():
        SOURCE.check(SOURCE.read(freeze) == lock, "Feature cache belongs to a different run")
    else:
        SOURCE.write(freeze, lock)
    network_free()
    torch.set_num_threads(3)
    torch.use_deterministic_algorithms(True)
    encode = frozen_encoder(args.encoder)
    rows = {row["pcm_sha256"]: row for row in plan["rows"]}
    feature_hashes = {}
    for i, (key, row) in enumerate(sorted(rows.items()), 1):
        audio = parent.pcm(args.data, row)
        path = args.cache / (key + ".npy")
        if not path.exists():
            values = encode(audio)
            temp = path.with_suffix(".tmp.npy")
            np.save(temp, values, allow_pickle=False)
            temp.replace(path)
        feature_hashes[key] = SOURCE.path_sha(path)
        if i % 80 == 0 or i == len(rows):
            print(f"Encoded {i}/{len(rows)} distinct source waveforms.", flush=True)
    SOURCE.write(args.cache / "features.json", feature_hashes)


def feature_reader(root, plan, digest):
    import numpy as np

    lock = SOURCE.read(root / "freeze.json")
    hashes = SOURCE.read(root / "features.json")
    SOURCE.check(lock["source_sha256"] == digest and lock["method"] == EXPERIMENT and lock["schema"] == SCHEMA
                 and set(hashes) == {row["pcm_sha256"] for row in plan["rows"]}, "Feature input bank differs")

    def feature_of(_data, row):
        key = row["pcm_sha256"]
        path = root / (key + ".npy")
        SOURCE.check(SOURCE.path_sha(path) == hashes[key], "Frozen frame features differ")
        features = np.load(path, allow_pickle=False)
        SOURCE.check(features.shape == (parent.FREQUENCIES, parent.FRAMES) and features.dtype == np.float32
                     and np.isfinite(features).all(), "Invalid frame features")
        return features

    return lock, feature_of


def train(args, *, model_factory=parent.model, schema=SCHEMA, method=EXPERIMENT):
    import torch
    from safetensors.torch import save_file

    from app.detectors.audio.upgraded import SpeechActivity
    from scripts.audio_strong_traffic_view import network_free

    plan, digest = parent.load_plan(args.data)
    origin, feature_of = feature_reader(args.cache, plan, digest)
    lock = origin | {"schema": schema, "method": method}
    SOURCE.check(SOURCE.path_sha(args.speech) == SPEECH, "Speech activity asset differs")
    args.output.mkdir(mode=0o700)
    network_free()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    activity = SpeechActivity(args.speech)
    fitted = []
    for known, unknown in (parent.sources.FAMILIES, parent.sources.FAMILIES[::-1]):
        model, mean, scale, losses = parent.train(args.data, plan, known, None, feature_of=feature_of,
                                                 model_factory=model_factory)
        weights = args.output / (known + ".safetensors")
        save_file({**model.state_dict(), "feature_mean": torch.from_numpy(mean),
                   "feature_scale": torch.from_numpy(scale)}, str(weights))
        calibration = parent.relevant(plan, "calibration", known, calibration=True)
        scored = parent.score_rows(model, (mean, scale), args.data, calibration, None, activity, feature_of=feature_of)
        seal = {"trained_family": known, "held_family": unknown, "calibration_scores": scored, "losses": losses,
                "points": parent.scores_at_points(calibration, scored), "weights_sha256": SOURCE.path_sha(weights)}
        SOURCE.write(args.output / (known + "-points.json"), seal)
        fitted.append((model, (mean, scale), seal))
        print(f"Sealed {known} fit and calibration before scoring {unknown}.", flush=True)
    cases = []
    for model, norm, seal in fitted:
        held = parent.relevant(plan, "held", seal["held_family"])
        cases.append(seal | {"held_scores": parent.score_rows(model, norm, args.data, held, None, activity, feature_of=feature_of)})
        print(f"Scored {len(held)} held deliveries for {seal['held_family']}.", flush=True)
    SOURCE.write(args.output / "candidate.json", {"freeze": lock, "cases": cases,
                                                  "feature_ledger_sha256": SOURCE.path_sha(args.cache / "features.json")})


def flank_interval(local, edge, spans):
    from app.detectors.audio.content import window_in_speech

    def supported(a, b):
        return window_in_speech(spans, a / 100, b / 100, fraction=.5)

    peak = None
    for duration in parent.LENGTHS:
        for start in range(40, parent.FRAMES - duration - 39, parent.STRIDE):
            end = start + duration
            if not all(supported(a, b) for a, b in ((start-40, start-10), (start, end), (end+10, end+40))):
                continue
            here = float(local[start:end].mean())
            context = max(float(local[start-40:start-10].mean()), float(local[end+10:end+40].mean()))
            boundaries = float(max(edge[start-4:start+5]) + max(edge[end-4:end+5]))
            proposal = here - context + .075 * boundaries, -start, -end
            if peak is None or proposal > peak:
                peak = proposal
    return None if peak is None else {"score": round(peak[0], 8), "range": [-parent.HOP * peak[1], -parent.HOP * peak[2]]}


def load_head(root, case, *, model_factory=parent.model):
    from safetensors.torch import load_file

    weights = root / (case["trained_family"] + ".safetensors")
    SOURCE.check(SOURCE.path_sha(weights) == case["weights_sha256"], "Fitted head differs")
    tensors = load_file(str(weights))
    mean, scale = tensors.pop("feature_mean").numpy(), tensors.pop("feature_scale").numpy()
    model = model_factory().eval()
    model.load_state_dict(tensors, strict=True)
    return model, (mean, scale)


def flank_readout(args):
    import torch

    from app.detectors.audio.upgraded import SpeechActivity
    from scripts.audio_strong_traffic_view import network_free

    plan, digest = parent.load_plan(args.data)
    lock, feature_of = feature_reader(args.cache, plan, digest)
    fitted = SOURCE.read(args.fit / "candidate.json")
    SOURCE.check(fitted["freeze"] == lock and SOURCE.path_sha(args.speech) == SPEECH, "Flank inputs differ")
    args.output.mkdir(mode=0o700)
    freeze = lock | {"method": FLANK_METHOD, "head_receipt_sha256": SOURCE.path_sha(args.fit / "candidate.json")}
    SOURCE.write(args.output / "freeze.json", freeze)
    network_free()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    activity = SpeechActivity(args.speech)
    sealed = []
    for case in fitted["cases"]:
        known, unknown = case["trained_family"], case["held_family"]
        model, norm = load_head(args.fit, case)
        rows = parent.relevant(plan, "calibration", known, calibration=True)
        scored = parent.score_rows(model, norm, args.data, rows, None, activity, feature_of=feature_of, select=flank_interval)
        seal = {"trained_family": known, "held_family": unknown, "weights_sha256": case["weights_sha256"],
                "points": parent.scores_at_points(rows, scored), "calibration_scores": scored}
        SOURCE.write(args.output / (known + "-points.json"), seal)
        sealed.append((model, norm, seal))
    cases = []
    for model, norm, seal in sealed:
        rows = parent.relevant(plan, "held", seal["held_family"])
        scored = parent.score_rows(model, norm, args.data, rows, None, activity, feature_of=feature_of, select=flank_interval)
        cases.append(seal | {"held_scores": scored})
    SOURCE.write(args.output / "candidate.json", {"freeze": freeze, "cases": cases})


def transfer_inputs(data, transfer):
    plan, digest = parent.load_plan(data)
    roster = transfer / "roster.json"
    SOURCE.check(SOURCE.path_sha(roster) == TRANSFER_ROSTER, "Transfer roster differs")
    training_rooms = {row["background"]["room"] for row in plan["sources"]}
    training_readers = {row["reader_sha256"] for row in plan["sources"]}
    used_pcm = {row["pcm_sha256"] for row in plan["rows"]}
    admitted, excluded = [], set()
    for row in SOURCE.read(roster):
        SOURCE.check(row["pcm_sha256"] not in used_pcm and row["room"] not in training_rooms,
                     "Transfer reuses model development PCM or a meeting")
        readers = [row.get("host"), row.get("donor")]
        for source in readers:
            voice = (source or {}).get("canonical_speaker", "")
            if voice.startswith("librispeech:") and parent.sources.order(voice.removeprefix("librispeech:")) in training_readers:
                excluded.add(row["group"])
        admitted.append(row)
    return [row for row in admitted if row["group"] not in excluded], excluded, digest


def transfer(args):
    import gc

    import numpy as np
    import torch
    from app.detectors.audio import short_voice_review
    from app.detectors.audio.upgraded import SpeechActivity
    from scripts.audio_strong_traffic_view import network_free
    from scripts.prepare_audio_specialists_v15 import ASSETS
    from scripts.run_audio_specialists_v15 import load_voice

    rows, excluded, digest = transfer_inputs(args.data, args.roster)
    args.output.mkdir(mode=0o700)
    speech_path = args.weights / "silero_vad.jit"
    SOURCE.check(SOURCE.path_sha(speech_path) == SPEECH, "Speech activity asset differs")
    network_free()
    torch.set_num_threads(3)
    torch.use_deterministic_algorithms(True)
    activity = SpeechActivity(speech_path)
    head_specs = {}
    heads = {}
    for name, root in (("log_mel", args.log_mel), ("frozen_speech_layers", args.speech_layers)):
        result = SOURCE.read(root / "candidate.json")
        SOURCE.check(result["freeze"]["source_sha256"] == digest, "Head source panel differs")
        for case in result["cases"]:
            family = case["trained_family"]
            key = f"{name}/{family}"
            head_specs[key] = {"weights_sha256": case["weights_sha256"], "points": case["points"]}
            heads[key] = load_head(root, case)
    if args.flank:
        result = SOURCE.read(args.flank / "candidate.json")
        SOURCE.check(result["freeze"]["head_receipt_sha256"] == SOURCE.path_sha(args.speech_layers / "candidate.json")
                     and result["freeze"]["method"] == FLANK_METHOD, "Frozen flank method differs")
        for case in result["cases"]:
            family = case["trained_family"]
            key = f"frozen_speech_layers/{family}"
            SOURCE.check(case["weights_sha256"] == head_specs[key]["weights_sha256"], "Flank head differs")
            head_specs[f"flank/{family}"] = {"weights_sha256": case["weights_sha256"], "points": case["points"]}
    lock = {"schema": SCHEMA, "transfer_roster_sha256": TRANSFER_ROSTER, "fit_roster_sha256": digest,
            "excluded_groups": len(excluded), "head_specs": head_specs, "planned_pcm_sha256": [r["pcm_sha256"] for r in rows]}
    SOURCE.write(args.output / "freeze.json", lock)
    encode, bank = frozen_encoder(args.encoder), parent.mel_filterbank()
    portable, cached = [], []
    with torch.inference_mode():
        for row in rows:
            raw = (args.roster / "pcm" / (row["id"] + ".pcm")).read_bytes()
            SOURCE.check(len(raw) == 2 * parent.SAMPLES and SOURCE.digest(raw) == row["pcm_sha256"],
                         "Transfer PCM differs")
            audio = np.frombuffer(raw, "<i2").astype(np.float32) / 32768.
            source_features = {"log_mel": parent.features(audio, bank), "frozen_speech_layers": encode(audio)}
            spans = activity.spans(audio)
            output = {key: row[key] for key in ("group", "arm", "family", "truth", "pcm_sha256")}
            output["heads"] = {}
            for key, (model, (mean, scale)) in heads.items():
                values = (source_features[key.split("/", 1)[0]] - mean[:, None]) / scale[:, None]
                local, edge = model(torch.from_numpy(values[None]))[0].numpy()
                output["heads"][key] = parent.best_interval(local, edge, spans)
                flank = key.replace("frozen_speech_layers/", "flank/")
                if args.flank and flank != key:
                    output["heads"][flank] = flank_interval(local, edge, spans)
            cached.append((audio, spans))
            portable.append(output)
    del encode
    gc.collect()
    torch.set_num_threads(2)
    stock = load_voice({"weights": args.weights, "assets": ASSETS})

    class Deadline:
        def check(self, _stage):
            pass  # Offline timing is not part of this comparison.

    for output, (audio, spans) in zip(portable, cached, strict=True):
        result = short_voice_review.scan(audio, spans, stock, Deadline(), codec="pcm_s16le")
        window = result["selected"]
        output["incumbent"] = {"assessed": bool(result["responses"]),
                               "range": [window["start_sample"], window["end_sample"]] if window is not None else None}
    record = {"freeze": lock, "rows": portable}
    record["counts"] = transfer_counts(record)
    SOURCE.write(args.output / "receipt.json", record)
    print(f"Scored {len(rows)} separate transfer waveforms across {len(head_specs)} scoring arms.", flush=True)


def transfer_counts(record):
    from collections import Counter

    from scripts.review_audio_stock_voice_transfer import useful

    freeze, rows = record["freeze"], record["rows"]
    SOURCE.check(freeze["schema"] == SCHEMA and freeze["transfer_roster_sha256"] == TRANSFER_ROSTER
                 and [row["pcm_sha256"] for row in rows] == freeze["planned_pcm_sha256"], "Transfer commitments differ")
    SOURCE.check(len(rows) == len({row["pcm_sha256"] for row in rows}), "Transfer PCM repeated")
    matched = Counter((row["group"], row["arm"]) for row in rows)
    families = Counter(row["family"] for row in rows if row["truth"] is not None)
    SOURCE.check(freeze["excluded_groups"] == 1 and families == {"clone": 3, "expressive": 4, "stock": 4}
                 and len(rows) == 48 and len(matched) == len(rows)
                 and sum(row["arm"] == "background_only" for row in rows) == 4
                 and all((row["truth"] is not None) == (row["arm"] == "synthetic") for row in rows), "Transfer cohort differs")
    for row in rows:
        if row["arm"] == "synthetic":
            SOURCE.check(all(matched[row["group"], arm] == 1 for arm in ("untouched", "ordinary_same_host", "ordinary_other_voice")),
                         "Matched ordinary transfer control is missing")
    points = {f"{key}/{name}": (key, point) for key, spec in freeze["head_specs"].items()
              for name, point in spec["points"].items()}
    decisions = {"incumbent": [(row["incumbent"]["assessed"], row["incumbent"]["range"]) for row in rows]}
    for policy, (head, point) in points.items():
        scores = [{"pcm_sha256": row["pcm_sha256"], "best": row["heads"][head]} for row in rows]
        decisions[policy] = list(zip((item["best"] is not None for item in scores), parent.verdicts(rows, scores, point), strict=True))
    outputs = {}
    for policy, calls in decisions.items():
        counts = Counter()
        groups = set()
        for row, (assessed, span) in zip(rows, calls, strict=True):
            key = row["family"] if row["truth"] is not None else "background" if row["arm"] == "background_only" else "genuine"
            counts[key + "/files"] += 1
            counts[key + "/unassessed"] += not assessed
            counts[key + "/flags"] += span is not None
            if row["truth"] is not None:
                counts[key + "/useful"] += span is not None and useful(span, row["truth"])
            elif span is not None and key == "genuine":
                groups.add(row["group"])
        outputs[policy] = dict(counts) | {"genuine/flagged_groups": len(groups)}
    if "counts" in record:
        SOURCE.check(record["counts"] == outputs, "Transfer results failed scalar replay")
    return outputs


def review(baseline, candidate, flank=None):
    original = parent.result(baseline)
    improved = parent.result(baseline | {"candidate": candidate}, schema=SCHEMA, method=EXPERIMENT)
    outputs = {"log_mel": original, "frozen_speech_layers": improved}
    if flank:
        serialized = json.dumps(candidate, indent=2, sort_keys=True, allow_nan=False) + "\n"
        SOURCE.check(flank["freeze"]["head_receipt_sha256"] == SOURCE.digest(serialized.encode("utf-8"))
                     and {case["trained_family"]: case["weights_sha256"] for case in flank["cases"]}
                     == {case["trained_family"]: case["weights_sha256"] for case in candidate["cases"]},
                     "The flank readout must use the exact same fitted heads")
        outputs["flank"] = parent.result(baseline | {"candidate": flank}, schema=SCHEMA, method=FLANK_METHOD)
    return outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="phase", required=True)
    for phase, fields in (("cache", ("data", "cache", "encoder")), ("train", ("data", "cache", "speech", "output")),
                          ("transfer", ("data", "roster", "encoder", "weights", "log_mel", "speech_layers", "output")),
                          ("flank", ("data", "cache", "speech", "fit", "output")),
                          ("review", ("baseline", "candidate")), ("review-transfer", ("receipt",))):
        command = commands.add_parser(phase)
        for field in fields:
            command.add_argument(field, type=Path)
        if phase in {"review", "transfer"}:
            command.add_argument("--flank", type=Path)
    args = parser.parse_args()
    if args.phase == "cache":
        cache_features(args)
    elif args.phase == "train":
        train(args)
    elif args.phase == "transfer":
        transfer(args)
    elif args.phase == "flank":
        flank_readout(args)
    elif args.phase == "review":
        flank = SOURCE.read(args.flank) if args.flank else None
        print(json.dumps(review(SOURCE.read(args.baseline), SOURCE.read(args.candidate), flank), indent=2, sort_keys=True))
    else:
        print(json.dumps(transfer_counts(SOURCE.read(args.receipt)), indent=2, sort_keys=True))
