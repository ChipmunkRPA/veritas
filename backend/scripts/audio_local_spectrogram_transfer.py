"""Fit two small local spectrogram checks on source-matched partial-speech edits."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from scripts import audio_partial_residual_inputs as inputs
from scripts import audio_partial_residual_transfer as old
from scripts import audio_periodic_excitation_transfer as replay
from scripts.audio_short_edit_controls import UnsupportedControl

SCHEMA = "chipmunk.audio-local-spectrogram-transfer/1"
CELL, RATE = 3200, 16000
FFT, FRAME, HOP, BANDS = 512, 400, 160, 40
SEED, EPOCHS, BATCH = 20261112, 12, 3
ARMS = ("spectrogram_local", "spectrogram_context", "context_fusion")
BASELINE, ROOM_BASELINE = replay.BASELINE, replay.ROOM_BASELINE
ROLES = ("confidence_cal", "selection", "confirmation")
METHOD = {
    "source": "New 4 s splice episodes only from the qualified v81/v82 fusion_fit role. Three episodes per each of 7 clone and 4 stock donor groups, with genuine same-host and other-voice controls and matched LibriSpeech backgrounds. Select hosts by fixed order. Reject unusable signal extents or component clipping before scoring. Preserve source-role separation. Expressive appears only in held comparison.",
    "fit_conditions": "Dry and -5 dB. Never train on diagnostic or confirmation tracks, AMI audio, or any expressive generator.",
    "features": "25 ms Hann, 10 ms hop, 512-point STFT, 40 triangular mel bands from 80 to 7600 Hz. Log powers relative to full-file geometric mean, clamped [-8,8], plus 20 ms temporal differences. Fixed frame centers [0,10,20,...] ms. Boundary padding is zero.",
    "model": "80 to 16 pointwise convolution, depthwise temporal width 5, 16 to 24 pointwise convolution, depthwise temporal width 5, then mean pool each 200 ms. Local head uses that cell; contextual head also uses mean of cells at offsets -3,-2,+2,+3 and target-minus-context. Edge offsets clip to the nearest cell. Fit both independently.",
    "objective": "Overlap-weighted, class-balanced logistic cell loss. Add .15 softplus ranking loss on synthetic vs each matching genuine splice within known edit, and .04 squared deviation outside the edit and two neighboring cells. Fixed 12 epochs and AdamW, learning rate .003, weight decay .01, 3 host episodes per batch, seeded fixed-order random permutations.",
    "fusion": "Mean of incumbent and contextual cell z-scores. Centers and scales are median and robust 1.4826 times MAD across all exact genuine calibration cells, with scale floor .10. No fusion on the six-second panel because its original incumbent cell arrays were not retained.",
    "points": "Every candidate strictly exceeds its maximum exact genuine calibration track response. Incumbent points unchanged. Compare held ranking at exact genuine source-group false counts 0,1,2. Select strongest 0.6-second mean over 3 adjacent cells. Useful means >=.30 IoU and >=.50 edit coverage.",
    "advance": "Need two net useful strong-noise gains across two families, at most one extra noisy false group, preserve number of dry useful locations, and flag no isolated background. Need >=1 useful noisy AMI location and no extra genuine false AMI flag in the contextual arm. Successful pilot would require fresh acquisition.",
}
read, write, sha, require = replay.read, replay.write, replay.sha, replay.require


def digest(data):
    return hashlib.sha256(data).hexdigest()


def order(value):
    return digest(f"{SCHEMA}:{value}".encode())


def select_rows(sources, kind):
    groups = defaultdict(list)
    for row in sources.values():
        if row["role"] == "fusion_fit" and row["kind"] == kind:
            groups[row["study_group"]].append(row)
    return [sorted(groups[key], key=lambda row: order(row["raw_sha256"])) for key in sorted(groups, key=order)]


def training_sources():
    catalog, clean, root, _ = inputs.catalogues()
    positives = [*select_rows(catalog, "clone"), *select_rows(catalog, "stock")]
    hosts, rooms = (select_rows(catalog, kind) for kind in ("genuine", "speech"))
    require((len(positives), len(hosts), len(rooms)) == (11, 14, 4), "Expected qualified source groups for expanded fit")
    tasks, rows, waveforms = [], [], []
    for repeat in range(3):
        for index, eligible in enumerate(positives):
            ordinal = repeat*len(positives)+index
            donor = eligible[repeat % len(eligible)]
            rendered = None
            for offset in range(len(hosts)):
                source_group = hosts[(ordinal+offset) % len(hosts)]
                host = source_group[repeat % len(source_group)]
                if host["canonical_speaker"] == donor["canonical_speaker"] or host["study_group"] == donor["study_group"]:
                    continue
                other_groups = [g for g in hosts if g[0]["study_group"] not in {host["study_group"], donor["study_group"]}
                                and g[0]["canonical_speaker"] not in {host["canonical_speaker"], donor["canonical_speaker"]}]
                ordinary = other_groups[(ordinal+offset*3+5) % len(other_groups)]
                task = {"id": f"spectrogram-fit-{ordinal:02}", "role": "fusion_fit", "duration": inputs.DURATIONS[(ordinal+repeat) % 3],
                        "host": host["raw_sha256"], "ordinary": ordinary[(repeat+offset) % len(ordinary)]["raw_sha256"],
                        "synthetic": donor["raw_sha256"], "background": rooms[ordinal % len(rooms)][0]["raw_sha256"]}
                try:
                    rendered = list(inputs.construct({"pairs": [task], "background_controls": []}, catalog, clean, root))
                except (ValueError, UnsupportedControl) as error:
                    if str(error) not in {"Fixed comparison contains no usable source extent", "pcm16_clipping"}:
                        raise
                    continue
                break
            require(rendered is not None, "No usable score-blind training episode for donor")
            tasks.append(task)
            for row, values in rendered:
                raw = values.astype("<i2", copy=False).tobytes()
                rows.append(row | {"pcm_sha256": digest(raw)})
                waveforms.append(raw)
    require(len(tasks) == 33 and len(rows) == 330, "Expanded training construction is incomplete")
    require(inputs.geometry_review(rows, [np.frombuffer(x, "<i2") for x in waveforms]) == 33, "Matched waveforms failed replay")
    return tasks, rows, waveforms


def mel_bank():
    hz = np.fft.rfftfreq(FFT, 1/RATE)
    def mel(frequency):
        return 2595*np.log10(1+frequency/700)
    points = 700*(10**(np.linspace(mel(80), mel(7600), BANDS+2)/2595)-1)
    lower, upper = (hz[None]-points[:-2, None])/(points[1:-1, None]-points[:-2, None]), (points[2:, None]-hz[None])/(points[2:, None]-points[1:-1, None])
    triangles = np.maximum(0, np.minimum(lower, upper))
    return (triangles/triangles.sum(axis=1, keepdims=True)).T


BANK = mel_bank()
WINDOW = np.hanning(FRAME)


def features(raw):
    pcm = np.frombuffer(raw, "<i2").astype(np.float64)/32768
    require(len(pcm) in {64_000, 96_000}, "Unexpected source waveform extent")
    frames = np.lib.stride_tricks.sliding_window_view(np.pad(pcm, FRAME//2), FRAME)[::HOP][:-1]
    power = abs(np.fft.rfft(frames*WINDOW, FFT))**2
    logarithmic = np.log(np.maximum(power @ BANK, 1e-8))
    logarithmic = np.clip(logarithmic-logarithmic.mean(), -8., 8.)
    delta = logarithmic - logarithmic[np.maximum(0, np.arange(len(logarithmic))-2)]
    return np.concatenate((logarithmic, delta), axis=1).T.astype(np.float32)


class LocalNet(nn.Module):
    def __init__(self, contextual):
        super().__init__()
        self.contextual = contextual
        self.encoder = nn.Sequential(nn.Conv1d(80, 16, 1), nn.GELU(), nn.Conv1d(16, 16, 5, padding=2, groups=16),
                                     nn.Conv1d(16, 24, 1), nn.GELU(), nn.Conv1d(24, 24, 5, padding=2, groups=24), nn.GELU())
        self.output = nn.Sequential(nn.Linear(72 if contextual else 24, 16), nn.GELU(), nn.Linear(16, 1))

    def forward(self, value):
        encoded = self.encoder(value)
        cells = F.avg_pool1d(encoded, 20).transpose(1, 2)
        if self.contextual:
            indices = torch.arange(cells.shape[1])
            flanks = torch.stack([cells[:, (indices+offset).clamp(0, cells.shape[1]-1)] for offset in (-3, -2, 2, 3)]).mean(0)
            cells = torch.cat((cells, flanks, cells-flanks), dim=-1)
        return self.output(cells).squeeze(-1)


def objective(logits, labels):
    positive = labels/labels.sum().clamp(min=1.)
    negative = (1.-labels)/(1.-labels).sum().clamp(min=1.)
    balanced = .5*((positive*F.softplus(-logits)).sum() + (negative*F.softplus(logits)).sum())
    inside = labels[..., 2:3, :]
    delta = logits[..., 2:3, :] - logits[..., :2, :]
    measured = (delta*inside).sum(-1)/inside.sum(-1).clamp(min=1.)
    protected = F.max_pool1d((inside > 0).to(logits.dtype).flatten(0, 1), 5, padding=2, stride=1).reshape_as(inside)
    return balanced + .15*F.softplus(-measured).mean() + .04*((1-protected)*delta.square()).mean()


def train_net(data, labels, contextual):
    torch.manual_seed(SEED)
    model = LocalNet(contextual)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.003, weight_decay=.01)
    random = np.random.Generator(np.random.PCG64(SEED))
    for _ in range(EPOCHS):
        for episodes in np.array_split(random.permutation(len(data)), len(data)//BATCH):
            batch = data[episodes]
            target = labels[episodes]
            logits = model(batch.flatten(0, 2)).reshape(target.shape)
            loss = objective(logits, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    require(bool(torch.isfinite(loss)), "Fitting produced a non-finite loss")
    return model.eval(), float(loss.detach())


def fit_arrays(rows, raw):
    chosen = [(r, x) for r, x in zip(rows, raw, strict=True) if r["condition"] in {"dry", "speech_minus5db"} and r["arm"] in inputs.ARMS]
    require(len(chosen) == 198, "Missing fitted variants")
    data = torch.from_numpy(np.stack([features(x) for _, x in chosen])).reshape(33, 2, 3, 80, 400)
    labels = torch.from_numpy(np.stack([old.overlap_weights(r["truth"]) for r, _ in chosen]).astype(np.float32)).reshape(33, 2, 3, 20)
    return data, labels


def bindings(args):
    parent, rows, rooms = replay.bind(args.parent, args.meeting)
    source_files = {f"{role}-features.{suffix}": sha(args.parent / f"{role}-features.{suffix}") for role in ROLES for suffix in ("npz", "json")}
    freeze = {"schema": SCHEMA, "method": METHOD, "code_sha256": sha(Path(__file__)), "constructor_sha256": sha(Path(inputs.__file__)),
              "parent": parent["parent"] | source_files, "meeting": parent["meeting"],
              "runtime": {key: importlib.metadata.version(key) for key in ("numpy", "torch")}, "seed": SEED,
              "training_views": 198, "training_episodes": 33, "training_generator_groups": 11,
              "external_weights": False, "new_incumbent_model_forwards": 0}
    return freeze, rows, rooms


def current(parent, role, rows):
    paths = [parent / f"{role}-features.{suffix}" for suffix in ("npz", "json")]
    raw = read(paths[1])
    require(raw["arrays_sha256"] == sha(paths[0]), "Exact incumbent features changed")
    entries = [r for r in rows if r["role"] == role]
    require([(r["id"], r["pcm_sha256"]) for r in entries] == [(r["id"], r["pcm_sha256"]) for r in raw["rows"]], "Incumbent cell source changed")
    with np.load(paths[0], allow_pickle=False) as packed:
        local = packed["local"].astype(np.float64)
    require(local.shape == (len(entries), 20) and np.isfinite(local).all(), "Exact incumbent cells differ")
    return entries, local


def predict(model, raw):
    matrix = torch.from_numpy(np.stack([features(x) for x in raw]))
    with torch.inference_mode():
        values = np.concatenate([model(chunk).numpy().astype(np.float64) for chunk in matrix.split(24)])
    require(np.isfinite(values).all(), "Non-finite local response")
    return values


def robust(values):
    median = float(np.median(values))
    scale = max(.10, float(1.4826*np.median(abs(values-median))))
    return {"center": median, "scale": scale}


def fuse(incumbent, context, factors):
    first, second = (factors[name] for name in (BASELINE, ARMS[1]))
    return ((incumbent-first["center"])/first["scale"]+(context-second["center"])/second["scale"])*.5


def measure(values, truth, point):
    windows = np.lib.stride_tricks.sliding_window_view(values, 3).mean(axis=1)
    index = int(windows.argmax())
    start, end = index*CELL, (index+3)*CELL
    score = float(windows[index])
    return {"score": score, "range": [start, end], "flag": score > point, "located": old.useful_range(start, end, truth)}


def fit(args, rows):
    tasks, train_rows, raw = training_sources()
    bind = read(args.source / "training.json")
    require(bind == {"tasks": tasks, "rows": train_rows}, "Regenerated training waveforms differ from frozen episode commitments")
    data, targets = fit_arrays(train_rows, raw)
    models, losses = zip(*(train_net(data, targets, contextual) for contextual in (False, True)), strict=True)
    entries, incumbent = current(args.parent, "confidence_cal", rows)
    selection = np.array([not row["label"] for row in entries])
    raws = [old.pcm_bytes(args.parent, row) for row, keep in zip(entries, selection, strict=True) if keep]
    absolute, context = (predict(model, raws) for model in models)
    factors = {BASELINE: robust(incumbent[selection]), ARMS[1]: robust(context)}
    fused = fuse(incumbent[selection], context, factors)
    points = {arm: max(measure(values, None, 0.)["score"] for values in matrix)
              for arm, matrix in zip(ARMS, (absolute, context, fused), strict=True)}
    archive = args.source / "heads.pt"
    with archive.open("xb") as stream:
        torch.save(dict(zip(ARMS[:2], (model.state_dict() for model in models), strict=True)), stream)
    write(args.source / "points.json", {"freeze_sha256": sha(args.source / "freeze.json"), "heads_sha256": sha(archive),
          "points": points, "factors": factors, "last_batch_training_losses": losses, "calibration_genuine_views": len(raws)})
    print("Two model heads, late fusion and genuine operating points frozen", flush=True)


def inspect(args, rows, rooms):
    seal = read(args.source / "points.json")
    archive = args.source / "heads.pt"
    require(seal["freeze_sha256"] == sha(args.source / "freeze.json") and seal["heads_sha256"] == sha(archive), "Fitted heads changed")
    saved = torch.load(archive, weights_only=True, map_location="cpu")
    models = (LocalNet(False).eval(), LocalNet(True).eval())
    for model, name in zip(models, ARMS[:2], strict=True):
        model.load_state_dict(saved[name], strict=True)
    comparator = read(args.parent / "fit.json")["points"][BASELINE]["point"]
    previous_matched = {row["id"]: row["checks"][BASELINE] for row in read(args.parent / "decisions.json")}
    results = []
    for role in ("selection", "confirmation"):
        entries, baseline = current(args.parent, role, rows)
        raw = [old.pcm_bytes(args.parent, row) for row in entries]
        local, contextual = (predict(model, raw) for model in models)
        fused = fuse(baseline, contextual, seal["factors"])
        for index, row in enumerate(entries):
            values = {BASELINE: baseline[index], **dict(zip(ARMS, (local[index], contextual[index], fused[index]), strict=True))}
            checks = {name: measure(cell, row["truth"], comparator if name == BASELINE else seal["points"][name])
                      for name, cell in values.items()}
            require(all(checks[BASELINE][k] == previous_matched[row["id"]][k] for k in ("score", "range", "flag")),
                    "Current score and location differ from retained comparator")
            results.append({"panel": "matched", "role": role, "condition": row["condition"], "family": row["family"],
                            "id": row["id"], "arm": row["arm"], "truth": row["truth"], "pcm_sha256": row["pcm_sha256"],
                            "group": row["pair_id"] if row["pair_id"] is not None else row["background_group"],
                            "checks": checks})
    baseline = {row["id"]: row for row in read(args.parent / "meeting-replay.json")["decisions"]}
    short = {row["id"]: row for role in ("selection", "confirmation") for row in read(args.meeting / f"{role}.json")["rows"]}
    point = read(args.meeting / "points.json")["points"]["short_odd"]["point"]
    local, contextual = (predict(model, [raw for _, raw in rooms]) for model in models)
    for index, (row, raw) in enumerate(rooms):
        previous = baseline[row["id"]]["checks"][BASELINE]
        value, bounds = short[row["id"]]["features"]["short_odd"]
        results.append({"panel": "meeting", "role": row["panel"], "condition": row["condition"], "family": row["family"],
                        "id": row["id"], "arm": row["kind"], "truth": row["truth"], "pcm_sha256": digest(raw), "group": row["host_group"],
                        "checks": {BASELINE: {**previous, "located": previous["location"]},
                                   ROOM_BASELINE: {"score": value, "range": bounds, "flag": value > point,
                                                   "located": old.useful_range(*bounds, row["truth"])},
                                   **{name: measure(matrix[index], row["truth"], seal["points"][name])
                                      for name, matrix in zip(ARMS[:2], (local, contextual), strict=True)}}})
    return results


def summarize(results):
    groups = defaultdict(list)
    for row in results:
        groups[row["panel"], row["role"], row["condition"]].append(row)
    tables = {}
    for (panel, role, condition), rows in groups.items():
        arms = list(rows[0]["checks"])
        cell = tables[f"{panel}/{role}/{condition}"] = {name: replay.counts(rows, name) for name in arms}
        if role == "confirmation":
            scored = [name for name in arms if name != BASELINE or panel != "meeting"]
            controls = old.HUMAN if panel == "matched" else ("ordinary",)
            cell["matched_ranking"] = {name: replay.pair_ranking(rows, name, controls) for name in scored}
            cell["equal_false_groups"] = {str(n): {name: replay.counts(rows, name, n) for name in scored} for n in (0, 1, 2)}
    dry = tables["matched/confirmation/dry"]
    noisy = tables["matched/confirmation/speech_minus5db"]
    room = tables["meeting/confirmation/room"]
    context = ARMS[1]
    details = {}
    for arm in ARMS:
        def location(row, name):
            item = row["checks"][name]
            return item["flag"] and item["located"]

        new = [r for r in groups["matched", "confirmation", "speech_minus5db"] if location(r, arm) and not location(r, BASELINE)]
        detail = {"net_noisy_useful": noisy[arm]["useful"]-noisy[BASELINE]["useful"],
                  "new_noisy_families": sorted({r["family"] for r in new}),
                  "additional_false_groups": noisy[arm]["genuine_false_groups"]-noisy[BASELINE]["genuine_false_groups"],
                  "dry_useful_change": dry[arm]["useful"]-dry[BASELINE]["useful"], "background_false": noisy[arm]["background_only_flags"],
                  "room_useful": room[context]["useful"], "room_false_change": room[context]["genuine_false_flags"]-room[BASELINE]["genuine_false_flags"]}
        detail["qualifies"] = (detail["net_noisy_useful"] >= 2 and len(detail["new_noisy_families"]) >= 2
                               and detail["additional_false_groups"] <= 1 and detail["dry_useful_change"] >= 0 and detail["background_false"] == 0
                               and detail["room_useful"] >= 1 and detail["room_false_change"] <= 0)
        details[arm] = detail
    return {"cells": tables, "gate": details}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "fit", "report", "audit"))
    parser.add_argument("source", type=Path)
    parser.add_argument("--parent", type=Path, default=replay.PARENT)
    parser.add_argument("--meeting", type=Path, default=replay.MEETING)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    started = time.monotonic()
    lock, rows, rooms = bindings(args)
    frozen, training = (args.source / name for name in ("freeze.json", "training.json"))
    if args.phase == "freeze":
        tasks, train_rows, _ = training_sources()
        require(not {row["pcm_sha256"] for row in train_rows} & {row["pcm_sha256"] for row in rows if row["role"] != "fusion_fit"},
                "Training audio matches a different source role")
        args.source.mkdir(mode=0o700)
        write(training, {"tasks": tasks, "rows": train_rows})
        write(frozen, lock | {"training_sha256": sha(training)})
        print(f"Frozen {len(tasks)} source-matched fit episodes and disjoint comparison inputs", flush=True)
        return
    require(read(frozen) == lock | {"training_sha256": sha(training)}, "Frozen method or source qualification changed")
    if args.phase == "fit":
        fit(args, rows)
        return
    observations = inspect(args, rows, rooms)
    outcome = summarize(observations)
    if args.phase == "report":
        write(args.source / "observations.json", observations)
        write(args.source / "result.json", lock | outcome | {"commitments": {name: sha(args.source / f"{name}.json") for name in
              ("freeze", "training", "points", "observations")}, "observation_views": len(observations), "seconds": round(time.monotonic()-started, 3)})
        print(outcome["gate"], flush=True)
    else:
        recorded = read(args.source / "result.json")
        require(all(sha(args.source / f"{name}.json") == checksum for name, checksum in recorded["commitments"].items())
                and read(args.source / "observations.json") == observations and all(recorded[k] == v for k, v in outcome.items()),
                "Exact source features, predictions, or outcome cells failed replay")
        print(f"Reproduced {len(observations)} source-bound comparisons and all location cells", flush=True)


if __name__ == "__main__":
    main()
