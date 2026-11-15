# SPDX-License-Identifier: MIT
"""Compare local acoustic input contributions with dense voice responses."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
from collections import Counter
from pathlib import Path

from scripts import audio_shared_mel_context as parent

source = parent.source
PANELS = ("development", "random_placement", "families", "sine")
PUBLISHED_PARENT = parent.PANELS / "audio-shared-mel-context-2026-11-14.json"
PUBLISHED_PARENT_SHA256 = "e4de086f269d7682bfd48a5b5bed9f92f666d3e577e367a079474b8e2f419bce"
SCHEMA = "chipmunk.audio-input-contribution-transfer/1"
ARMS = ("dense", "contribution", "agreement")
FIRST_UPPER_BAND = 40
METHOD = {
    "input": "Use the unchanged primary mixture and speech gate. Use original waveform conditioning, normalized Mel "
             "features and the frozen voice model over two four-second contexts at zero and two seconds.",
    "contribution": "Take the derivative of the full context's pre-sigmoid mean frame response with respect to its "
                    "normalized Mel input. For each time position, multiply the gradient by the input and sum the signed "
                    "terms from the upper 40 of 80 bands in both Mel frames. Multiply by the number of feature positions "
                    "to cancel mean pooling's length factor. The lower bands stay fixed. This estimates local sensitivity "
                    "to drawing the upper bands toward the context-average Mel input.",
    "geometry": "Use the established one-second windows at half-second steps. Use the native 280+320k sample centers, "
                "at least 40 measured frames, the median signed term in each window, and the lower result when both contexts qualify.",
    "arms": "Compare conventional dense frame responses and signed input contributions separately. The agreement arm "
            "keeps a dense window only if its contribution is at least the median among measured windows in that same recording.",
    "point": "Use the larger of the second-highest previously unflagged ordinary development group maximum and the "
             "highest previously unflagged isolated-background maximum for each arm. Apply strictly. Preserve current flags "
             "in supplement comparisons. A useful position covers at least half the labeled edit and reaches 0.30 time IoU.",
    "advancement": "A supplement needs at least two added useful locations across at least two generator families, "
                   "at most one additional ordinary source group per held panel and no additional isolated-background flags.",
}


def _normalized_input(pcm):
    import torch

    from app.detectors.audio.upgraded import prepare_voice_window
    from app.third_party.kaldi.fbank import fbank

    signal = torch.from_numpy(prepare_voice_window(pcm))
    present = torch.nonzero(signal).flatten()
    extent = int(present[-1]) + 1 if len(present) else len(signal)
    mel = fbank(signal[None, :max(extent, 400)] * 32768., num_mel_bins=80,
               sample_frequency=parent.stock.RATE, dither=0., window_type="povey")
    standardized = (mel - mel.mean(0)) / torch.sqrt(mel.var(0) + 1e-7)
    count = len(standardized) // 2 * 2
    return signal, standardized[:count].reshape(1, -1, 160)


def _frames(model, features):
    import torch

    mask = torch.ones(1, features.shape[1], dtype=torch.long)
    states = model.backbone(input_features=features, attention_mask=mask, output_hidden_states=True).hidden_states
    weights = model.layer_weights.softmax(0)[:, None, None, None]
    hidden = (torch.stack(states) * weights).sum(0)[0]
    return model.head(hidden).squeeze(-1)


def _check_gradient(model, signal, features, frame_scores, contributions):
    import torch

    with torch.no_grad():
        pooled = model(signal[None]).item()
        source.check(abs(frame_scores.mean().item() - pooled) <= .00002, "Dense responses do not reconstruct the context score")
        a, b = len(frame_scores) // 3, len(frame_scores) * 2 // 3
        expected = contributions[a:b].sum().item() / len(frame_scores)
        epsilon = .01
        positive, negative = features.clone(), features.clone()
        positive.reshape(1, -1, 2, 80)[:, a:b, :, FIRST_UPPER_BAND:] *= 1 + epsilon
        negative.reshape(1, -1, 2, 80)[:, a:b, :, FIRST_UPPER_BAND:] *= 1 - epsilon
        observed = (_frames(model, positive).mean().item() - _frames(model, negative).mean().item()) / (2 * epsilon)
        source.check(abs(expected - observed) <= max(.002, .08 * abs(expected)),
                     f"Input derivative {expected:.8f} differs from finite perturbation {observed:.8f}")
    return {"pooling_absolute_error": abs(frame_scores.mean().item() - pooled),
            "signed_derivative": expected, "finite_perturbation": observed, "epsilon": epsilon}


def _readouts(model, pcm, spans, audit):
    import numpy as np
    import torch

    from app.detectors.audio.content import window_in_speech

    admitted = [window_in_speech(spans, start / parent.stock.RATE, (start + parent.WIDTH) / parent.stock.RATE)
                for start in parent.STARTS]
    values = {arm: [[] for _ in parent.STARTS] for arm in ARMS[:2]}
    for offset in source.CUTS:
        local = [(i, start) for i, (start, allowed) in enumerate(zip(parent.STARTS, admitted, strict=True))
                 if allowed and offset <= start and start + parent.WIDTH <= offset + 64000]
        if not local:
            continue
        signal, features = _normalized_input(pcm[offset:offset + 64000])
        features.requires_grad_(True)
        dense = _frames(model, features)
        gradient, = torch.autograd.grad(dense.mean(), features)
        band_terms = (gradient * features.detach()).reshape(-1, 2, 80)
        contribution = band_terms[:, :, FIRST_UPPER_BAND:].sum((1, 2)) * len(dense)
        source.check(torch.isfinite(dense).all().item() and torch.isfinite(contribution).all().item(),
                     "A native response is nonfinite")
        if not audit:
            audit.update(_check_gradient(model, signal, features.detach(), dense.detach(), contribution))
        centers = offset + 280 + np.arange(len(dense)) * 320
        for name, vector in zip(ARMS[:2], (dense.detach().numpy(), contribution.numpy()), strict=True):
            for index, start in local:
                inside = vector[(centers >= start) & (centers < start + parent.WIDTH)]
                if len(inside) >= 40:
                    values[name][index].append(float(np.median(inside)))
    views = {name: [round(min(window), 8) if window else None for window in windows] for name, windows in values.items()}
    return views | {"agreement": _agreement(views)}


def _agreement(views):
    from statistics import median

    dense, contributions = views["dense"], views["contribution"]
    available = [value for value in contributions if value is not None]
    if not available:
        return [None] * len(dense)
    within_recording = median(available)
    return [response if measure is not None and measure >= within_recording else None
            for response, measure in zip(dense, contributions, strict=True)]


def _points(rows):
    points = {}
    for arm in ARMS:
        ordinary, backgrounds = {}, []
        for row in rows:
            top = parent.candidate(row, arm)
            if row["truth"] or top is None or parent.existing(row):
                continue
            if row["kind"] in parent.interval.BACKGROUNDS:
                backgrounds.append(top[0])
            else:
                group = row["group"]
                ordinary[group] = max(ordinary.get(group, -math.inf), top[0])
        source.check(len(ordinary) >= 2, "Too few admitted ordinary development source groups")
        second = sorted(ordinary.values(), reverse=True)[1]
        points[arm] = max([second, *backgrounds])
    return points


def _summary(rows, arm, point, *, supplement=False):
    stats = parent.tally(rows, arm, point, supplement=supplement)
    source_maxima = Counter()
    for row in rows:
        if top := parent.candidate(row, arm):
            source_maxima["measured"] += 1
            source_maxima["useful_top_before_point"] += bool(row["truth"] and parent.stock.useful(top[1], row["truth"]))
    return stats | dict(source_maxima)


def review(receipt):
    lock, panels = receipt["freeze"], receipt["panels"]
    source.check(source.path_sha(PUBLISHED_PARENT) == PUBLISHED_PARENT_SHA256, "Published comparison receipt differs")
    previous = source.read(PUBLISHED_PARENT)
    source.check(lock["schema"] == SCHEMA and lock["method"] == METHOD and lock["bindings"] == previous["freeze"]["bindings"]
                 and set(panels) == set(PANELS), "Method or input commitments differ")
    for panel in PANELS:
        for row, expected in zip(panels[panel], previous["panels"][panel], strict=True):
            source.check(all(row[key] == value for key, value in expected.items() if key not in {"views", "pooled_frames"}),
                         "A source differs from its parent receipt")
            views = row["views"]
            source.check(set(views) == set(ARMS) and all(len(values) == len(parent.STARTS) for values in views.values())
                         and all(value is None or math.isfinite(value) for values in views.values() for value in values)
                         and views["dense"] == expected["views"]["local"]
                         and [value is None for value in views["dense"]] ==
                         [value is None for value in views["contribution"]] and views["agreement"] == _agreement(views),
                         "Source windows or agreement differ")
    points = _points(panels["development"])
    source.check(points == receipt["points"], "Frozen points differ")
    outcomes = {panel: {"existing": parent.tally(rows, "existing"), **{
        arm: {"standalone": _summary(rows, arm, points[arm]),
              "supplement": _summary(rows, arm, points[arm], supplement=True)} for arm in ARMS}}
                for panel, rows in panels.items()}
    advanced = {}
    for arm in ARMS:
        supplements = [outcomes[panel][arm]["supplement"] for panel in PANELS[1:]]
        gains = [gain for panel in supplements for gain in panel["added_useful"]]
        advanced[arm] = (len(gains) >= 2 and len({gain[2] for gain in gains}) >= 2 and
                         all(len(result["new_ordinary_groups"]) <= 1 and result["counts"]["background_flags"] == 0
                             for result in supplements))
    return {"panels": outcomes, "advance_to_fresh_validation": advanced}


def run(args):
    import numpy as np
    import torch

    from app.detectors.audio.upgraded import SpeechActivity
    from scripts.audio_strong_traffic_view import network_free
    from scripts.prepare_audio_specialists_v15 import ASSETS
    from scripts.run_audio_specialists_v15 import load_voice

    lock = {"schema": SCHEMA, "method": METHOD, "bindings": parent.ledger_roles(), "asset_hashes": ASSETS,
            "runner_sha256": source.path_sha(Path(__file__)),
            "runtime": {name: importlib.metadata.version(name) for name in parent.sine.RUNTIME}}
    args.path.mkdir(mode=0o700)
    source.write(args.path / "freeze.json", lock)
    _, bank, _ = parent.interval.native_rows(args.archive)
    sine_receipt = source.read(parent.SINE_PARENT)
    parent.sine.audit_native(args.sine, sine_receipt)
    for _, pcm in parent.sine.native_rows(args.sine, sine_receipt["freeze"]):
        raw = pcm.tobytes()
        bank[source.digest(raw)] = raw
    network_free()
    model = load_voice({"weights": args.weights, "assets": ASSETS})
    torch.set_num_threads(2)
    activity = SpeechActivity(args.weights / "silero_vad.jit")
    panels, audit = {}, {}
    for panel in PANELS:
        inputs = parent.input_rows(panel)
        rows = []
        with (args.path / f"{panel}.jsonl").open("x") as journal:
            for row in inputs:
                raw = bank[row["pcm_sha256"]]
                source.check(source.digest(raw) == row["pcm_sha256"], "Native mixture bytes differ")
                pcm = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768
                views = _readouts(model, pcm, activity.spans(pcm), audit)
                record = row | {"views": views, "existing_summary": panel in {"families", "sine"}}
                rows.append(record)
                journal.write(json.dumps(record, sort_keys=True) + "\n")
                journal.flush()
                if len(rows) % 8 == 0 or len(rows) == len(inputs):
                    print(f"{panel}: {len(rows)}/{len(inputs)}", flush=True)
        panels[panel] = rows
        if panel == "development":
            points = _points(rows)
            source.write(args.path / "points.json", points)
            print(f"Frozen operating points: {points}", flush=True)
    receipt = {"freeze": lock, "panels": panels, "points": points, "numerical_check": audit}
    receipt["outcome"] = review(receipt)
    source.write(args.path / "receipt.json", receipt)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("score", "review"))
    parser.add_argument("path", type=Path)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--sine", type=Path)
    parser.add_argument("--weights", type=Path, default=source.ROOT / "models/weights/audio-upgrade")
    args = parser.parse_args()
    if args.phase == "score" and (args.archive is None or args.sine is None):
        parser.error("Scoring requires --archive and --sine")
    receipt = run(args) if args.phase == "score" else source.read(args.path)
    checked = review(receipt)
    source.check(checked == receipt["outcome"], "Saved outcome differs from the replay")
    print(json.dumps(checked, sort_keys=True))
