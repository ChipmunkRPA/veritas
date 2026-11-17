"""Fit a compact visual speech model on publisher-aligned genuine talkers."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from contextlib import closing
from fractions import Fraction
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

from scripts import video_lip_articulation_transfer as prior
from scripts import video_lip_correspondence as receipt

SCHEMA = "chipmunk.video-visual-speech/1"
FIT_PER_CONDITION = 12
RATE = 16_000
LEAD = 320
SEED = 261117
EPOCHS = 18
CLIP_LIMIT = 120


def rank(uid):
    return prior.phone.digest(f"{SCHEMA}/{uid}".encode())


def prepare(source, parent, fresh, output):
    from scripts.lombard_mouth_source_intake import index_zip

    original, transfer = receipt.read(parent), receipt.read(fresh)
    actors = {row["target"]["id"].split("_", 1)[0]: row["speaker"] for row in original["rows"] if row["split"] == "fit"}
    calibration = [row for row in original["rows"] if row["split"] == "calibration"]
    second = [row | {"split": "second"} for row in transfer["rows"]]
    reserved = {row["speaker"] for row in second}
    first = [row | {"split": "first"} for row in original["rows"] if row["split"] == "held" and row["speaker"] not in reserved]
    groups = [set(actors.values()), *(set(row["speaker"] for row in rows) for rows in (calibration, first, second))]
    assert all(not (a & b) for i, a in enumerate(groups) for b in groups[i + 1:])
    qualified = defaultdict(list)
    paths, publisher = prior.phone.source_archives(source)
    with (ZipFile(source / "publisher-json.zip") as metadata, ZipFile(source / "alignment.zip") as aligned,
          ZipFile(paths["front"]) as front, ZipFile(paths["audio"]) as audio):
        labels, faces, voices = (index_zip(archive, extension) for archive, extension in
                                 ((aligned, ".json"), (front, ".mov"), (audio, ".wav")))
        for name in index_zip(metadata, ".json").values():
            for row in json.loads(metadata.read(name)):
                actor, condition = row["SPKR"], row["COND"]
                uid = f"{actor}_{condition}_{row['UTTERANCE']}"
                if (actor in actors and row["STATUS"] == "CORRECT" and condition in {"p", "l"}
                        and all(uid in archive for archive in (labels, faces, voices))):
                    qualified[actor, condition].append(uid)
        fitting = []
        for (actor, condition), values in sorted(qualified.items()):
            done = 0
            for uid in sorted(set(values), key=rank):
                label = aligned.read(labels[uid])
                timing = json.loads(label)[uid]
                if not 12 <= len(timing) <= 35:
                    continue
                waveform = audio.read(voices[uid])
                pcm = prior.phone.pcm(waveform)
                if not 30000 <= len(pcm) <= 56000:
                    continue
                target = {"id": uid, "front": faces[uid].filename, "samples": len(pcm),
                          "front_sha256": prior.phone.digest(front.read(faces[uid])), "audio": voices[uid].filename,
                          "audio_sha256": prior.phone.digest(waveform), "alignment_sha256": prior.phone.digest(label)}
                fitting.append({"speaker": actors[actor], "condition": condition, "split": "fit", "target": target})
                done += 1
                if done == FIT_PER_CONDITION:
                    break
            assert done == FIT_PER_CONDITION, (actor, condition, done)
    assert len(fitting) == 2 * FIT_PER_CONDITION * len(actors) == 336
    rows = fitting + calibration + first + second
    assert len({row["target"]["front_sha256"] for row in rows}) == len(rows)
    receipt.save(output, {"schema": SCHEMA, "publisher": publisher, "parent": receipt.sha(parent), "transfer": receipt.sha(fresh),
                          "split_counts": dict(Counter(row["split"] for row in rows)), "rows": rows})
    print(dict(Counter(row["split"] for row in rows)), flush=True)


def picture_patches(path, mp, task):
    import numpy as np

    from app.detectors.video import viseme_patch
    from app.detectors.video.decode import decode_stream, probe_frames

    table = probe_frames(path, cap=CLIP_LIMIT, timeout_s=18, max_pixels=1280 * 720)
    if (table.truncated or table.interpolated or table.rotation or not 35 <= len(table) <= CLIP_LIMIT
            or any(not Fraction(1, 28) <= (b - a) * table.tb <= Fraction(1, 22) for a, b in zip(table.pts, table.pts[1:]))):
        raise ValueError("Recording does not have the qualified source picture clock")
    images = np.zeros((len(table), viseme_patch.HEIGHT, viseme_patch.WIDTH), dtype=np.uint8)
    valid = np.zeros(len(table), dtype=bool)
    count = 0
    with closing(decode_stream(path, table, max_side=None, timeout_s=20)) as frames:
        for picture in frames:
            if not picture.pts_verified or (picture.index, picture.pts) != (count, table.pts[count]):
                raise ValueError("A decoded picture lacks its declared native timestamp")
            mesh = task.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(picture.rgb)))
            if (patch := viseme_patch.crop(picture.rgb, mesh.face_landmarks)) is not None:
                images[count] = patch
                valid[count] = True
            count += 1
    if count != len(table):
        raise ValueError("Missing source pictures")
    samples = np.asarray([round((pts - table.pts[0]) * table.tb * RATE) for pts in table.pts])
    return images, valid, samples, table


def labels(frames, raw, *, samples):
    import numpy as np

    from app.detectors.video.viseme_patch import MOUTH_GROUPS, PHONE_GROUPS

    definitions = {phone: MOUTH_GROUPS.index(name) for name, phones in PHONE_GROUPS.items() for phone in phones}
    time = frames + LEAD
    out = np.full(len(time), -100, dtype=np.int8)
    others = MOUTH_GROUPS.index("other")
    for phone in raw:
        a = round(Fraction(str(phone["offset"])) * RATE)
        b = round((Fraction(str(phone["offset"])) + Fraction(str(phone["duration"]))) * RATE)
        name = phone["phone"].split("_", 1)[0]
        out[(a <= time) & (time < min(b, samples))] = definitions.get(name, others)
    return out


def capture(plan, source, model_assets, phase, output):
    import cv2
    import numpy as np

    from app.evidence.video_reference_timing import _landmarker
    from scripts.lombard_mouth_source_intake import index_zip

    cv2.setNumThreads(1)
    sys.path.insert(0, str((model_assets / "runtime").resolve()))
    model = model_assets / "face_landmarker.task"
    assert receipt.sha(model) == prior.study.parent.FACE_SHA256
    frozen = receipt.read(plan)
    assert frozen["schema"] == SCHEMA
    output.mkdir(mode=0o700)
    seen = []
    rows = [row for row in frozen["rows"] if row["split"] == phase] if phase == "fit" else [
        row for row in frozen["rows"] if row["split"] != "fit"]
    with (ZipFile(source / "lombardgrid_front.zip") as front, ZipFile(source / "alignment.zip") as labeled,
          _landmarker(model) as (mp, task), TemporaryDirectory() as work):
        archive = index_zip(labeled, ".json")
        picture = Path(work) / "input.mov"
        for position, row in enumerate(rows, 1):
            uid, pin = (row["target"][key] for key in ("id", "front_sha256"))
            data = front.read(row["target"]["front"])
            raw = labeled.read(archive[uid])
            assert prior.phone.digest(data) == pin and prior.phone.digest(raw) == row["target"]["alignment_sha256"]
            picture.write_bytes(data)
            images, valid, samples, table = picture_patches(picture, mp, task)
            target = labels(samples, json.loads(raw)[uid], samples=row["target"]["samples"])
            target[~valid] = -100
            name = f"{pin}.npz"
            path = output / name
            np.savez_compressed(path, patches=images, valid=valid, samples=samples, targets=target)
            seen.append({"file": name, "sha256": receipt.sha(path), "picture_sha256": pin, "speaker": row["speaker"],
                         "split": row["split"], "pts": list(table.pts), "time_base": str(table.tb), "valid": int(valid.sum())})
            if position % 12 == 0:
                print(f"{phase}: {position}/{len(rows)} clips", flush=True)
    receipt.save(output / "capture.json.gz", {"schema": SCHEMA, "plan": receipt.sha(plan), "landmark_asset": receipt.sha(model),
                 "phase": phase, "rows": seen})


def train(captured, output):
    import numpy as np
    import torch
    from safetensors.torch import save_file
    from torch.nn import functional as F

    from app.detectors.video.viseme_patch import HEIGHT, MOUTH_GROUPS, WIDTH, VisualSpeech, normalize

    torch.manual_seed(SEED)
    torch.set_num_threads(3)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    packet = receipt.read(captured / "capture.json.gz")
    assert packet["phase"] == "fit"
    crops, ground = [], []
    for row in packet["rows"]:
        path = captured / row["file"]
        assert receipt.sha(path) == row["sha256"]
        with np.load(path, allow_pickle=False) as data:
            good = data["valid"]
            if good.sum() >= .80 * len(good):
                crops.append(normalize(data["patches"], good))
                ground.append(data["targets"])
    counts = np.bincount(np.concatenate([y[y >= 0] for y in ground]), minlength=len(MOUTH_GROUPS))
    assert len(crops) >= 300 and np.all(counts >= 200), counts
    weights = torch.tensor(np.sqrt(counts.mean() / counts).clip(.5, 3.), dtype=torch.float32)
    model = VisualSpeech()
    optim = torch.optim.AdamW(model.parameters(), lr=.002, weight_decay=.003)
    losses = []
    for epoch in range(EPOCHS):
        for indices in torch.randperm(len(crops)).split(12):
            size = max(len(crops[i]) for i in indices)
            x = torch.zeros((len(indices), size, 2, HEIGHT, WIDTH))
            y = torch.full((len(indices), size), -100, dtype=torch.long)
            for slot, index in enumerate(indices.tolist()):
                array = torch.from_numpy(crops[index])
                x[slot, :len(array)] = array.flip(-1) if torch.rand(()).item() < .5 else array
                y[slot, :len(array)] = torch.from_numpy(ground[index]).long()
            score = model(x)
            loss = F.cross_entropy(score.flatten(0, 1), y.flatten(), weight=weights)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
        losses.append(round(loss.item(), 6))
        print(f"Visual epoch {epoch + 1}/{EPOCHS}: loss {losses[-1]}", flush=True)
    output.mkdir(mode=0o700)
    path = output / "visual-speech.safetensors"
    save_file(model.state_dict(), str(path))
    receipt.save(output / "fit.json.gz", {"schema": SCHEMA, "capture": receipt.sha(captured / "capture.json.gz"),
        "plan": packet["plan"], "weight": receipt.sha(path), "spatial_model": receipt.sha(Path(sys.modules[VisualSpeech.__module__].__file__)),
        "fit_clips": len(crops), "label_frames": dict(zip(MOUTH_GROUPS, map(int, counts), strict=True)), "losses": losses,
        "lead_samples": LEAD, "seed": SEED, "epochs": EPOCHS})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    for name, fields in (("prepare", ("source", "parent", "fresh", "output")),
                         ("capture", ("plan", "source", "model_assets", "output")), ("train", ("captured", "output"))):
        command = sub.add_parser(name)
        for name in fields:
            command.add_argument(name, type=Path)
        if "model_assets" in fields:
            command.add_argument("--phase", required=True, choices=("fit", "evaluation"))
    args = vars(parser.parse_args())
    {"prepare": prepare, "capture": capture, "train": train}[args.pop("mode")](**args)
