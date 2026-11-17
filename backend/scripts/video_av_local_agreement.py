# SPDX-License-Identifier: Apache-2.0
"""Fit a local speech and mouth comparison on licensed genuine recordings."""

from __future__ import annotations

import argparse
import gzip
import json
from collections import defaultdict
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import cv2
import numpy as np
import torch
from safetensors.torch import load_file, save_file
from torch.nn import functional as F

from app.bench import video_av_local_agreement as bench
from app.detectors.video import av_local_agreement as detector
from scripts.export_video_lip_correspondence import make_soundtracks, waveform
from scripts.video_lip_delivery_transfer import qualified_48k
from scripts.video_word_content_model import pin

SEED = 417031
EPOCHS = 18


def save(path, receipt):
    with path.open("xb") as stream:
        stream.write(gzip.compress(json.dumps(receipt, sort_keys=True, allow_nan=False).encode(), mtime=0))


def load_crops(capture, row):
    body = (capture / row["file"]).read_bytes()
    if pin(body) != row["sha256"]:
        raise ValueError("Mouth pixels differ from their bound source frames")
    with np.load(BytesIO(body), allow_pickle=False) as packet:
        return tuple(packet[key] for key in ("patches", "valid", "samples", "targets"))


def fit_examples(sources, plan_path, capture, eyes):
    plan = bench.read(plan_path)
    images = bench.read(capture / "capture.json.gz")
    if images["plan"] != bench.digest(plan_path):
        raise ValueError("The mouth inputs belong to a different source plan")
    rosters = {row["target"]["front_sha256"]: row for row in plan["rows"] if row["split"] == "fit"}
    rows = []
    with ZipFile(sources / "lombardgrid_audio.zip") as audio:
        for index, row in enumerate(images["rows"], 1):
            origin = rosters.pop(row["picture_sha256"])
            if origin["speaker"] != row["speaker"] or row["split"] != "fit":
                raise ValueError("A genuine training clip has the wrong speaker or partition")
            pcm = waveform(audio, origin["target"])
            patches, valid, samples, targets = load_crops(capture, row)
            positions, visual, visual_ok = detector.picture_windows(eyes, patches, valid)
            prepared = detector.sound(pcm)
            acoustic, audible = detector.acoustic_windows(prepared, samples[positions])
            keep = visual_ok & audible
            if keep.sum() >= 20:
                rows.append(
                    (
                        row["speaker"],
                        samples[positions][keep],
                        visual[keep],
                        acoustic[keep],
                        targets[positions][keep],
                    )
                )
            if index % 48 == 0:
                print(f"Read {index}/{len(images['rows'])} checked training clips", flush=True)
    if rosters:
        raise ValueError("The fitting input is incomplete")
    return rows


def paired_examples(rows):
    random = np.random.default_rng(SEED)
    grouped = defaultdict(list)
    for clip, (speaker, *_) in enumerate(rows):
        grouped[speaker].append(clip)
    packed = []
    for clip, (speaker, times, picture, sound, classes) in enumerate(rows):
        other_clips = [i for i in grouped[speaker] if i != clip]
        if not other_clips:
            continue
        separation = np.abs(times[:, None] - times)
        local = (separation >= 3200) & (separation <= 12200)
        for j, sample in enumerate(times):
            temporal = np.flatnonzero(local[j])
            if len(temporal) == 0:
                continue
            peer_times, _, peer_sound, _ = rows[random.choice(other_clips)][1:]
            nearby = np.flatnonzero(np.abs(peer_times - sample) <= 4200)
            if len(nearby) == 0:
                nearby = np.arange(len(peer_times))
            packed.append(
                (
                    picture[j],
                    sound[j],
                    sound[random.choice(temporal)],
                    peer_sound[random.choice(nearby)],
                    int(classes[j]),
                )
            )
    columns = zip(*packed, strict=True)
    return tuple(torch.from_numpy(np.stack(column)) for column in columns)


def train(inputs, path, *, phonetic):
    torch.manual_seed(SEED)
    eyes, ears, shifted, peer, labels = inputs
    labels = labels.to(torch.int64)
    model = detector.Agreement()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0028, weight_decay=0.009)
    response = []
    for epoch in range(EPOCHS):
        losses = []
        model.train()
        for group in torch.randperm(len(eyes)).split(192):
            x = eyes[group] + 0.025 * torch.randn_like(eyes[group])
            visual = model.visual(x)
            actual, displaced, peer_voice = model.acoustic(
                torch.cat((ears[group], shifted[group], peer[group]))
            ).chunk(3)
            paired = torch.sum(visual * actual, dim=-1)
            negative = torch.stack(
                (torch.sum(visual * displaced, dim=-1), torch.sum(visual * peer_voice, dim=-1))
            )
            loss = F.softplus(9 * (0.14 + negative - paired)).mean() + 0.12 * (1 - paired).mean()
            if phonetic:
                target = labels[group]
                # The fit knows broad visible speech groups. Inference needs neither words nor phonetic labels.
                loss = loss + 0.31 * (
                    F.cross_entropy(model.phonetic(8 * visual), target, reduction="sum")
                    + F.cross_entropy(model.phonetic(8 * actual), target, reduction="sum")
                ) / (2 * (target != -100).sum().clamp(1))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        response.append(round(float(np.mean(losses)), 7))
        print(f"{path.stem} epoch {epoch + 1}/{EPOCHS}, loss {response[-1]}", flush=True)
    save_file(model.state_dict(), str(path))
    return response


def implementation():
    backend = Path(__file__).resolve().parents[1]
    paths = (Path(value).resolve() for value in (__file__, detector.__file__, bench.__file__))
    return {path.relative_to(backend).as_posix(): bench.digest(path) for path in paths}


def fitting(args):
    out = args.output
    out.mkdir(mode=0o700)
    image = detector.load_picture(load_file(str(args.vision)))
    rows = fit_examples(args.sources, args.plan, args.crops, image)
    if len({row[0] for row in rows}) < 10:
        raise ValueError("The fit lacks enough licensed speakers")
    data = paired_examples(rows)
    print(f"Fit {len(data[0])} frame pairs from {len(rows)} original clips", flush=True)
    losses = {arm: train(data, out / f"{arm}.safetensors", phonetic=arm == "phonetic") for arm in bench.ARMS}
    save(
        out / "fit.json.gz",
        {
            "schema": bench.SCHEMA,
            "implementation": implementation(),
            "vision": bench.digest(args.vision),
            "plan": bench.digest(args.plan),
            "pictures": bench.digest(args.crops / "capture.json.gz"),
            "clips": len(rows),
            "pairs": len(data[0]),
            "speakers": sorted({row[0] for row in rows}),
            "weights": {arm: bench.digest(out / f"{arm}.safetensors") for arm in bench.ARMS},
            "losses": losses,
        },
    )


def read_models(fit, vision):
    stored = bench.read(fit / "fit.json.gz")
    if stored["implementation"] != implementation() or stored["vision"] != bench.digest(vision):
        raise ValueError("The comparison code or visual features differ from the fit")
    models = {}
    for arm, expected in stored["weights"].items():
        path = fit / f"{arm}.safetensors"
        if bench.digest(path) != expected:
            raise ValueError("The audio and mouth projection weights differ from the fit")
        model = detector.Agreement()
        model.load_state_dict(load_file(str(path)), strict=True)
        models[arm] = model.eval()
    return detector.load_picture(load_file(str(vision))), models, stored


def deliver(tracks, baseline, carrier, role, scratch):
    pcm = tracks[role]
    old = baseline["cases"][carrier][role]
    if carrier == "aac48":
        pcm, _, metadata = qualified_48k(pcm, scratch)
        encoded = metadata["file_sha256"]
        if encoded != old["carrier_sha256"]:
            raise ValueError("The audio encoder differs from the app comparator")
    else:
        encoded = old["carrier_sha256"]
    signed = pin(pcm.tobytes())
    if signed != old["source_pcm_sha256"]:
        raise ValueError("Candidate and incumbent must measure identical presented audio")
    return pcm, {
        "audio_sha256": signed,
        "carrier_sha256": encoded,
        "baseline": old["current_span"],
        "baseline_status": old["current_status"],
    }


def evaluate(sources, plan, crop_dir, incumbent, phase, eyes, models):
    planned = bench.read(plan)
    image_pin = bench.read(crop_dir / "capture.json.gz")
    if image_pin["plan"] != bench.digest(plan):
        raise ValueError("The mouth crop inventory differs from the source plan")
    measured = {row["picture_sha256"]: row for row in bench.read(incumbent)["rows"]}
    captures = {row["picture_sha256"]: row for row in image_pin["rows"]}
    rows = [row for row in planned["rows"] if row["split"] == phase]
    result = []
    with (
        ZipFile(sources / "lombardgrid_audio.zip") as audio,
        TemporaryDirectory(prefix="av-lip-sound-") as root,
    ):
        scratch = Path(root) / "track.m4a"
        for i, record in enumerate(rows, 1):
            sha = record["target"]["front_sha256"]
            old, cap = measured[sha], captures[sha]
            truth = record["target"]["span"]
            if (old["pts"], old["time_base"], old["truth"]) != (cap["pts"], cap["time_base"], truth):
                raise ValueError("Mouth samples and baseline results are on different native clocks")
            patches, valid, times, _ = load_crops(crop_dir, cap)
            visual = detector.picture_windows(eyes, patches, valid)
            tracks = make_soundtracks(audio, record)
            results = {}
            for carrier in bench.CARRIERS:
                cases = {}
                for role in bench.ROLES:
                    pcm, delivery = deliver(tracks, old, carrier, role, scratch)
                    sound = detector.sound(pcm)
                    proposals = {}
                    for name, model in models.items():
                        status, candidate = detector.localize(
                            times,
                            detector.similarities(model, visual, sound, times),
                            len(pcm),
                        )
                        proposals[name] = {"status": status, "candidate": candidate}
                    cases[role] = delivery | {"proposals": proposals}
                results[carrier] = cases
            result.append(
                {
                    "source_sha256": sha,
                    "speaker": record["speaker"],
                    "condition": record["condition"],
                    "native_times": times.tolist(),
                    "frame_pts": cap["pts"],
                    "time_base": cap["time_base"],
                    "truth": truth,
                    "samples": len(tracks["original"]),
                    "cases": results,
                }
            )
            if i % 6 == 0 or i == len(rows):
                print(f"Measured {phase} recording {i}/{len(rows)}", flush=True)
    return result


def comparison(args):
    eyes, projections, fit = read_models(args.fit, args.vision)
    phase = "fresh" if args.freeze else "calibration"
    rows = [
        row
        for plan, crops, prior in args.input
        for row in evaluate(args.sources, Path(plan), Path(crops), Path(prior), phase, eyes, projections)
    ]
    speakers = {row["speaker"] for row in rows}
    if speakers & set(fit["speakers"]):
        raise ValueError("A fitting speaker entered the comparison")
    models = fit["weights"] | {"vision": fit["vision"]}
    passed = {
        "schema": bench.SCHEMA,
        "rows": rows,
        "implementation": implementation(),
        "models": models,
        "inputs": [
            [bench.digest(Path(p)) for p in (plan, Path(crops) / "capture.json.gz", prior)]
            for plan, crops, prior in args.input
        ],
    }
    if args.freeze is None:
        options, choices = bench.freeze(rows)
        passed.update(development_speakers=sorted(speakers | set(fit["speakers"])), freeze=None)
    else:
        locked = bench.read(args.freeze)
        if (
            speakers & set(locked["development_speakers"])
            or locked["models"] != models
            or locked["implementation"] != implementation()
        ):
            raise ValueError("The held inputs cross the fixed comparison")
        options, choices = locked["options"], locked["choices"]
        passed["freeze"] = bench.digest(args.freeze)
    outcomes = bench.results(rows, options, choices)
    save(args.output, passed | {"options": options, "choices": choices, "outcomes": outcomes})
    bench.review(args.output, args.freeze)
    print(json.dumps({key: value["selected"] for key, value in outcomes.items()}, sort_keys=True), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_subparsers(dest="stage", required=True)
    fit, measure = (mode.add_parser(name) for name in ("fit", "measure"))
    for step in (fit, measure):
        for name in ("sources", "vision", "output"):
            step.add_argument(name, type=Path)
    for name in ("plan", "crops"):
        fit.add_argument(name, type=Path)
    measure.add_argument("fit", type=Path)
    measure.add_argument("--freeze", type=Path)
    measure.add_argument(
        "--input", nargs=3, action="append", required=True, metavar=("PLAN", "MOUTH_CROPS", "BASELINE")
    )
    args = parser.parse_args()
    torch.set_num_threads(3)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    cv2.setNumThreads(1)
    (fitting if args.stage == "fit" else comparison)(args)
