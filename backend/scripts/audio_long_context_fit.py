"""Fit the same local speech detector on small edits in two matched input durations."""
from __future__ import annotations

import argparse
import hashlib
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from scripts import audio_clean_teacher_transfer as teacher
from scripts import audio_long_context_inputs as data
from scripts import audio_short_edit_detector_training as parent
from scripts import audio_short_edit_ssl_transfer as projected
from scripts import audio_wavlm_features_v13 as encoder
from scripts.audio_focused_context_transfer import read, save

SCHEMA = "chipmunk.audio-native-context-fit/1"
HOP = parent.HOP
WIDTHS = (4000, 5600, 8000, 9600, 12000, 16000)
STRIDE = 800
EPOCHS = 24
METHOD = {
    "fit": "Use identical six- and ten-second content and boundary heads with the same starting seed. Match source, "
           "duration of the edit, donor, reprocessing control and background samples. Normalize positive and negative "
           "frame losses separately. Compare each synthetic interval with its matched ordinary reprocessing interval.",
    "windows": "Scan six fixed durations from 250 ms to 1 s at 50 ms steps. Require detected speech over at least half "
               "the window. Use mean content and the larger responses near its endpoints.",
    "points": "Use independent fitting and calibration readers and meeting groups. Normalize the maximum responses for "
              "each duration by the median and 90th percentile of the ordinary prompt maxima. Select 90th-percentile "
              "and maximum whole-input points over ordinary prompts with isolated background responses as a floor.",
    "comparison": "Keep the existing check as an independent comparator. Use no held cases to fit either model or any point.",
}


def file_sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def pcm(root, item):
    path = root / "pcm" / f"{item['sha256']}.pcm"
    raw = path.read_bytes()
    if len(raw) != 2 * item["samples"] or hashlib.sha256(raw).hexdigest() != item["sha256"]:
        raise ValueError("A prepared waveform does not match its input commitment")
    return np.frombuffer(raw, "<i2").astype(np.float32) / 32768


def start_cpu():
    import torch

    from scripts.audio_strong_traffic_view import network_free

    network_free()
    torch.set_num_threads(3)
    torch.use_deterministic_algorithms(True)


def feature_manifest(root):
    source = root / "freeze.json"
    plan = read(source)
    if plan["schema"] != data.SCHEMA:
        raise ValueError("Unexpected paired speech preparation")
    if {row["reader"] for row in plan["rows"] if row["role"] == "fit"} & {
        row["reader"] for row in plan["rows"] if row["role"] == "calibration"
    }:
        raise ValueError("A fitting reader also appears in calibration")
    return plan, {
        "schema": SCHEMA, "source_sha256": file_sha(source), "method": METHOD, "epochs": EPOCHS, "seed": parent.SEED,
        "encoder": {key: encoder.SPEC[key] for key in ("repository", "revision", "assets_sha256", "weight_terms")},
        "speech_sha256": projected.SPEECH,
        "code_sha256": {Path(module.__file__).name: file_sha(Path(module.__file__)) for module in (parent, projected, data)}
        | {Path(__file__).name: file_sha(Path(__file__))},
    }


def features(args):
    start_cpu()
    plan, commitment = feature_manifest(args.data)
    folder = args.cache
    folder.mkdir(mode=0o700, exist_ok=True)
    lock = folder / "freeze.json"
    if lock.exists() and read(lock) != commitment:
        raise ValueError("The cached features belong to different input waveforms")
    if not lock.exists():
        save(lock, commitment)
    encode = projected.frozen_encoder(args.encoder, variable_context=True)
    activity = teacher.runtime(args.speech)
    wanted = {}
    for row in plan["rows"]:
        for item in row["inputs"]:
            if row["role"] == "calibration" and not row["truth"]:
                wanted[item["sha256"]] = (item, True)
            elif row["role"] == "fit":
                wanted.setdefault(item["sha256"], (item, False))
    for number, (digest, (item, speaking)) in enumerate(wanted.items(), 1):
        base = folder / digest
        feature, spans = base.with_suffix(".npy"), base.with_suffix(".json")
        if not feature.exists() or (speaking and not spans.exists()):
            values = pcm(args.data, item)
            if not feature.exists():
                temporary = base.with_suffix(".tmp.npy")
                np.save(temporary, encode(values), allow_pickle=False)
                temporary.replace(feature)
            if speaking and not spans.exists():
                save(spans, activity.spans(values))
        if number % 40 == 0 or number == len(wanted):
            print(f"Measured {number} of {len(wanted)} distinct paired-context model inputs", flush=True)
    save(folder / "files.json", {digest: file_sha(folder / f"{digest}.npy") for digest in wanted})


def feature_reader(root, manifest):
    if read(root / "freeze.json") != manifest:
        raise ValueError("The paired feature cache differs from the input plan")
    hashes = read(root / "files.json")

    def load(row, context):
        item = next(item for item in row["inputs"] if item["context"] == context)
        filename = root / f"{item['sha256']}.npy"
        if file_sha(filename) != hashes[item["sha256"]]:
            raise ValueError("Frozen encoder output differs from its commitment")
        array = np.load(filename, allow_pickle=False)
        if array.shape != (64, item["samples"] // HOP) or array.dtype != np.dtype("<f4") or not np.isfinite(array).all():
            raise ValueError("Invalid encoder output")
        return array

    return load


def training_array(rows, load, context):
    by_prompt = defaultdict(lambda: defaultdict(dict))
    for row in rows:
        if row["role"] == "fit":
            by_prompt[row["source"]][row["condition"]][row["kind"]] = row
    waveforms, masks, edges = [], [], []
    clock = np.arange(data.SAMPLES[context] // HOP) * HOP
    for conditions in by_prompt.values():
        cases = [conditions[None], *(value for key, value in conditions.items() if key is not None)]
        if len(cases) != 2:
            raise ValueError("Every fitting prompt needs dry and background mixtures")
        a, b = cases[0][f"edited/{data.GENERATORS[0]}"]["truth"][context]
        waveforms.append([[load(items[kind], context) for kind in (*data.KINDS, "background" if "background" in items else "ordinary")]
                          for items in cases])
        masks.append(np.clip(np.minimum(clock + HOP, b) - np.maximum(clock, a), 0, HOP) / HOP)
        edges.append((np.abs(clock - a) <= 480) | (np.abs(clock - b) <= 480))
    return np.asarray(waveforms), np.asarray(masks, dtype=np.float32), np.asarray(edges, dtype=np.float32)


def fit_head(rows, load, context):
    import torch
    from torch.nn import functional as F

    values, mask, edges = training_array(rows, load, context)
    axes = (0, 1, 2, 4)
    mean = values.mean(axis=axes, dtype=np.float64).astype(np.float32)
    scale = np.maximum(.001, values.std(axis=axes, dtype=np.float64)).astype(np.float32)
    inputs = torch.from_numpy((values - mean[:, None]) / scale[:, None])
    mask, edges = torch.from_numpy(mask), torch.from_numpy(edges)
    torch.manual_seed(parent.SEED)
    model = parent.model().train()
    optim = torch.optim.AdamW(model.parameters(), lr=.002, weight_decay=.001)
    losses = []
    for _ in range(EPOCHS):
        epoch = []
        for i in torch.randperm(len(inputs)).split(6):
            batch = inputs[i]
            logits = model(batch.flatten(0, 2)).reshape(*batch.shape[:3], 2, batch.shape[-1])
            local, edge = logits.unbind(dim=3)
            edited = local[:, :, 3:5]
            genuine = local[:, :, 1:3]
            interval = mask[i, None, None]
            bounds = edges[i, None, None]
            core = (F.softplus(-edited) * interval).sum(dim=-1) / interval.sum(dim=-1)
            other = (F.softplus(edited) * (1 - interval)).sum(dim=-1) / (1 - interval).sum(dim=-1)
            normal = F.softplus(local[:, :, [0, 1, 2, 5]]).mean()
            boundary = (F.softplus(-edge[:, :, 3:5]) * bounds).sum(dim=-1) / bounds.sum(dim=-1)
            boundary_normal = (F.softplus(edge[:, :, 3:5]) * (1 - bounds)).mean() + F.softplus(edge[:, :, [0, 1, 2, 5]]).mean()
            margins = ((edited - genuine) * interval).sum(dim=-1) / interval.sum(dim=-1)
            loss = (core.mean() + .5 * (other.mean() + normal) + .25 * (boundary.mean() + .5 * boundary_normal)
                    + .35 * F.softplus(-margins).mean() + .12 * F.smooth_l1_loss(margins[:, 0], margins[:, 1]))
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            epoch.append(float(loss.detach()))
        losses.append(round(sum(epoch) / len(epoch), 8))
    return model.eval(), mean, scale, losses


def pool(local, edge, spans):
    clock = np.arange(len(local)) * HOP
    speech = np.zeros(len(local), dtype=np.float64)
    for start, stop in spans:
        speech += np.clip(np.minimum(clock + HOP, stop * data.RATE) - np.maximum(clock, start * data.RATE), 0, HOP)
    mass = np.r_[0., np.cumsum(np.minimum(speech, HOP))]
    content = np.r_[0., np.cumsum(local, dtype=np.float64)]
    edges = np.lib.stride_tricks.sliding_window_view(np.pad(edge, (4, 5), constant_values=-np.inf), 9).max(axis=1)
    readings = []
    for width in WIDTHS:
        count = width // HOP
        left = np.arange(0, len(local) - count + 1, STRIDE // HOP)
        right = left + count
        valid = 2 * (mass[right] - mass[left]) >= width - .00001
        if not np.any(valid):
            readings.append(None)
            continue
        scores = (content[right] - content[left]) / count + .075 * (edges[left] + edges[right])
        i = int(np.argmax(np.where(valid, scores, -np.inf)))
        readings.append({"score": round(float(scores[i]), 8), "range": [int(left[i]) * HOP, int(right[i]) * HOP]})
    return readings


def infer(model, mean, scale, features, spans):
    import torch

    with torch.inference_mode():
        input_tensor = torch.from_numpy(((features - mean[:, None]) / scale[:, None])[None])
        local, edge = model(input_tensor)[0].numpy()
    if not np.isfinite(local).all() or not np.isfinite(edge).all():
        raise ValueError("Nonfinite speech-localization readout")
    return pool(local, edge, spans)


def percentile(values, percent):
    ordered = sorted(values)
    return ordered[math.ceil(percent * len(ordered)) - 1]


def normalized(item, calibration):
    if item is None or calibration is None:
        return None
    center, scale = calibration
    return ((item["score"] - center) / scale, item["range"])


def calibrate(calibration):
    width_scales = []
    for i in range(len(WIDTHS)):
        per_prompt = defaultdict(list)
        for row in calibration:
            item = row["readings"][i]
            if row["kind"] != "background" and item is not None:
                per_prompt[row["source"]].append(item["score"])
        values = [max(history) for history in per_prompt.values()]
        if not values:
            width_scales.append(None)
            continue
        middle = percentile(values, .50)
        width_scales.append([middle, round(max(.15, percentile(values, .90) - middle), 8)])
    foreground, background = defaultdict(list), []
    for row in calibration:
        scored = [observed[0] for item, scale in zip(row["readings"], width_scales, strict=True)
                  if (observed := normalized(item, scale)) is not None]
        if scored:
            if row["kind"] == "background":
                background.append(max(scored))
            else:
                foreground[row["source"]].append(max(scored))
    maxima = [max(readings) for readings in foreground.values()]
    if not maxima or not background:
        raise ValueError("No complete ordinary and isolated background calibration")
    return {"width_scales": width_scales, "wider": max(percentile(maxima, .90), max(background)),
            "strict": max(max(maxima), max(background)), "foreground_prompts": len(maxima), "isolated_deliveries": len(background)}


def fit(args):
    import torch
    from safetensors.torch import save_file

    start_cpu()
    plan, manifest = feature_manifest(args.data)
    load = feature_reader(args.cache, manifest)
    args.output.mkdir(mode=0o700)
    reports = []
    for context in data.CONTEXTS:
        model, mean, scale, losses = fit_head(plan["rows"], load, context)
        file = args.output / f"{context}.safetensors"
        save_file(model.state_dict() | {"feature_mean": torch.from_numpy(mean),
                                        "feature_scale": torch.from_numpy(scale)}, str(file))
        calibration = []
        for row in plan["rows"]:
            if row["role"] != "calibration" or row["truth"]:
                continue
            item = next(item for item in row["inputs"] if item["context"] == context)
            spans = read(args.cache / f"{item['sha256']}.json")
            readings = infer(model, mean, scale, load(row, context), spans)
            calibration.append({key: row[key] for key in ("source", "reader", "condition", "kind", "room")}
                               | {"sha256": item["sha256"], "readings": readings})
        points = calibrate(calibration)
        reports.append({"context": context, "weights": file.name, "sha256": file_sha(file), "losses": losses,
                        "points": points, "calibration": calibration})
        print(f"Fit {context} with separate ordinary and isolated-background calibration", flush=True)
    save(args.output / "freeze.json", manifest | {"heads": reports, "held_manifest_sha256": plan["held_manifest_sha256"],
                                                 "development_readers": sorted({row["reader"] for row in plan["rows"]})})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    paths = {"features": ("data", "cache", "encoder", "speech"), "fit": ("data", "cache", "output")}
    for command, names in paths.items():
        runner = commands.add_parser(command)
        for name in names:
            runner.add_argument(name, type=Path)
    args = parser.parse_args()
    (features if args.command == "features" else fit)(args)
