"""Compare fitted speech contexts over the same six-second source interval."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from scripts import audio_clean_teacher_transfer as teacher
from scripts import audio_focused_context_transfer as event
from scripts import audio_long_context_fit as fit
from scripts import audio_long_context_holdout as holdout
from scripts import audio_long_context_inputs as data
from scripts import audio_short_edit_ssl_transfer as model
from scripts import audio_two_second_transfer as serving
from scripts.audio_long_context_review import EXPERIMENT, SCHEMA, summarize


def official_input(folder, row):
    return fit.pcm(folder, {"sha256": row["pcm_sha256"], "samples": row["samples"]})


def measurements(args):
    fit.start_cpu()
    frozen = args.media / "freeze.json"
    plan = event.read(frozen)
    lock = {"schema": SCHEMA, "source_sha256": fit.file_sha(frozen), "scope": EXPERIMENT["scope"],
            "encoder_sha256": fit.file_sha(Path(model.__file__)), "assets_sha256": fit.encoder.SPEC["assets_sha256"]}
    args.output.mkdir(mode=0o700, exist_ok=True)
    commitment = args.output / "freeze.json"
    if commitment.exists() and event.read(commitment) != lock:
        raise ValueError("Existing held features differ from their source plan")
    if not commitment.exists():
        event.save(commitment, lock)
    encode = model.frozen_encoder(args.encoder, variable_context=True)
    activity = teacher.runtime(args.speech)
    observations = {}
    for number, row in enumerate(plan["cases"], 1):
        source_digest = row["pcm_sha256"]
        if source_digest not in observations:
            scope = holdout.centered_views(row["samples"])
            start, length = scope["short"]
            directory = args.output / source_digest
            directory.mkdir(mode=0o700, exist_ok=True)
            spans = directory / "speech.json"
            filenames = {context: directory / f"{context}.npy" for context in data.CONTEXTS}
            if not spans.exists() or not all(path.exists() for path in filenames.values()):
                wave = official_input(args.media, row)
                if not spans.exists():
                    event.save(spans, activity.spans(wave[start:start + length]))
                for context, path in filenames.items():
                    if path.exists():
                        continue
                    offset, count = scope[context]
                    temporary = path.with_suffix(".tmp.npy")
                    np.save(temporary, encode(wave[offset:offset + count]), allow_pickle=False)
                    temporary.replace(path)
            observations[source_digest] = {"views": scope, "speech_sha256": fit.file_sha(spans),
                                           "features": {context: fit.file_sha(path) for context, path in filenames.items()}}
        if number % 20 == 0 or number == len(plan["cases"]):
            print(f"Prepared feature and speech readouts for {number}/{len(plan['cases'])} held deliveries", flush=True)
    event.save(args.output / "files.json", observations)


def deployed(args):
    from scripts.audio_joint_generator_transfer import panels, prior
    from scripts.run_audio_specialists_v15 import load_voice

    fit.start_cpu()
    plan = event.read(args.media / "freeze.json")
    activity = teacher.runtime(args.weights / "silero_vad.jit")
    assets = prior.read(panels.PRIMARY)["incumbent"]["assets"]
    voice = load_voice({"weights": args.weights, "assets": assets})
    rows = []
    for i, row in enumerate(plan["cases"], 1):
        rows.append(serving.score_case(row, official_input(args.media, row), voice, activity))
        if i % 20 == 0 or i == len(plan["cases"]):
            print(f"Measured the current detector on {i}/{len(plan['cases'])} held deliveries", flush=True)
    event.save(args.output, {"source_sha256": fit.file_sha(args.media / "freeze.json"), "assets": assets, "rows": rows})


def fitted_heads(directory):
    from safetensors.torch import load_file

    document = event.read(directory / "freeze.json")
    if document["schema"] != fit.SCHEMA or document["method"] != fit.METHOD:
        raise ValueError("Unexpected fitted speech heads")
    result = {}
    for head in document["heads"]:
        path = directory / head["weights"]
        if fit.file_sha(path) != head["sha256"]:
            raise ValueError("Fitted weights changed")
        state = load_file(str(path), device="cpu")
        mean, scale = (state.pop(key).numpy() for key in ("feature_mean", "feature_scale"))
        predictor = fit.parent.model().eval().requires_grad_(False)
        predictor.load_state_dict(state, strict=True)
        result[head["context"]] = predictor, mean, scale
    return document, result


def common_reading(head, features, shift, spans):
    import torch

    predictor, mean, scale = head
    if features.shape[0] != 64 or features.dtype != np.dtype("<f4"):
        raise ValueError("Unexpected encoder output shape")
    with torch.inference_mode():
        normalized = torch.from_numpy(((features - mean[:, None]) / scale[:, None])[None])
        local, edge = predictor(normalized)[0].numpy()
    source = np.arange(len(local)) * fit.HOP
    common = shift + np.arange(data.SAMPLES["short"] // fit.HOP) * fit.HOP
    if common[0] < source[0] or common[-1] > source[-1] + fit.HOP - 1:
        raise ValueError("The shared interval falls outside the encoded samples")
    return fit.pool(np.interp(common, source, local), np.interp(common, source, edge), spans)


def common_calibration(args):
    fit.start_cpu()
    frozen, heads = fitted_heads(args.fit)
    plan, manifest = fit.feature_manifest(args.data)
    if any(frozen[key] != value for key, value in manifest.items()):
        raise ValueError("The calibration data differ from the fit")
    load = fit.feature_reader(args.cache, manifest)
    sources = {item["source"]: item for item in plan["sources"]}
    responses = {context: [] for context in data.CONTEXTS}
    for row in plan["rows"]:
        if row["role"] != "calibration" or row["truth"]:
            continue
        item = next(item for item in row["inputs"] if item["context"] == "short")
        spans = event.read(args.cache / f"{item['sha256']}.json")
        offsets = sources[row["source"]]["offsets"]
        for context, head in heads.items():
            readings = common_reading(head, load(row, context), offsets["short"] - offsets[context], spans)
            responses[context].append({key: row[key] for key in ("source", "reader", "condition", "kind", "room")}
                                      | {"readings": readings})
    points = {context: fit.calibrate(rows) for context, rows in responses.items()}
    commitments = {"schema": SCHEMA, "method": EXPERIMENT, "fit_sha256": fit.file_sha(args.fit / "freeze.json"),
                   "points": points, "calibration": responses}
    event.save(args.output, commitments)
    print("Froze both pairs of operating points on the common sample scope", flush=True)


def compare(args):
    fit.start_cpu()
    frozen, heads = fitted_heads(args.fit)
    cal = event.read(args.calibration)
    if cal["fit_sha256"] != fit.file_sha(args.fit / "freeze.json") or cal["method"] != EXPERIMENT:
        raise ValueError("The shared points do not belong to this fit")
    plan = event.read(args.media / "freeze.json")
    reference = event.read(args.baseline)
    measured = event.read(args.features / "files.json")
    lock = event.read(args.features / "freeze.json")
    if (fit.file_sha(args.media / "freeze.json") != lock["source_sha256"] or
            {str(row["group"]) for row in plan["origins"]} & set(frozen["development_readers"])):
        raise ValueError("The held input plan does not match or contains development readers")
    rows = []
    for case, baseline in zip(plan["cases"], reference["rows"], strict=True):
        if not all(baseline[key] == field for key, field in case.items()):
            raise ValueError("The serving panel differs from the held input")
        record = measured[case["pcm_sha256"]]
        directory = args.features / case["pcm_sha256"]
        speech = directory / "speech.json"
        views = holdout.centered_views(case["samples"])
        if record["views"] != {key: list(value) for key, value in views.items()} or fit.file_sha(speech) != record["speech_sha256"]:
            raise ValueError("Held admission differs")
        spans = event.read(speech)
        a, length = views["short"]
        readings = {}
        for context, head in heads.items():
            filename = directory / f"{context}.npy"
            if fit.file_sha(filename) != record["features"][context]:
                raise ValueError("Held feature array differs")
            readings[context] = common_reading(head, np.load(filename, allow_pickle=False), a - views[context][0], spans)
        current = serving.locations(baseline)["adjacent_support"]
        current_assessed = (any(item["direction"] != "unassessed" for item in baseline["current_windows"])
                            or any(item["responses"] for item in baseline["excerpts"]))
        rows.append(case | {"scope": [a, a + length], "readings": readings, "current": current,
                            "current_assessed": current_assessed})
    summary = summarize(rows, cal["points"])
    event.save(args.output, {"schema": SCHEMA, "experiment": EXPERIMENT, "source_plan": plan,
                            "fit": frozen, "calibration": cal, "rows": rows, "summary": summary,
                            "baseline_assets": reference.get("model_sha256", reference.get("assets"))})
    print(f"Compared {len(rows)} external inputs on the shared source clock", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    cli = {
        "measure": (measurements, ("media", "output", "encoder", "speech")),
        "deployed": (deployed, ("media", "weights", "output")),
        "calibrate": (common_calibration, ("fit", "data", "cache", "output")),
        "compare": (compare, ("fit", "calibration", "media", "features", "baseline", "output")),
    }
    for label, (runner, paths) in cli.items():
        command = commands.add_parser(label)
        command.set_defaults(runner=runner)
        for name in paths:
            command.add_argument(name, type=Path)
    args = parser.parse_args()
    args.runner(args)
