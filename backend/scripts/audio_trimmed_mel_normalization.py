"""Compare robust Mel normalization on native mixtures before voice localization."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
from pathlib import Path

from scripts import audio_source_isolation_diagnostic as prior

ROOT = prior.ROOT
PARENT = prior.PANELS / "audio-source-isolation-diagnostic-2026-11-13.json"
PARENT_SHA = "b431596eb48f1c7339427873e45a4d4f583dc455ff50be3624873d2053f0daf9"
SCHEMA = "chipmunk.audio-trimmed-mel-normalization/1"
LOW, HIGH = .1, .9
BLEND = .5
METHOD = {
    "sources": "Use the same source-separated development and held native mixtures as the isolation diagnostic. This is a new method comparison on previously scored source audio.",
    "inputs": "Use the exact original six-second mixtures only. Apply the existing speech gate and four-second windows at 0 and 2 seconds. Keep the original waveform preprocessing, filterbank, backbone and affine frame head.",
    "normalization": "For each Mel band, clamp a copy of its raw log-energy values at that context's 10th and 90th percentiles. Average the original and clamped means and the original and clamped sample variances equally. Standardize the original log-energy values with these blended moments and the existing variance floor. Keep all original feature values and temporal order before the affine normalization. Neither score reads a clean stem or an isolated background.",
    "location": "Use the same one-second intervals every half second. Report the median frame logit within each interval and the lower median where two native contexts fully cover that interval. An interval must pass the original half-window speech gate. Preserve the source clock.",
    "fixed_points": "Set the candidate point at the maximum assessed genuine development response. Reuse the locked native dense development point and the prior stock two-second check. Count an edit as useful at 0.30 IoU and 50 percent edit coverage. Count alarms by ordinary group and show unassessed cases.",
    "comparison": "The robust view must retain the existing useful location, add at least two useful edits across two generator families, flag at most one additional ordinary group and never flag an isolated room. Also report a descriptive false-group sweep for the dense original and robust routes. No held responses select a point. Passing would justify fresh tests, never serving on this reused bank alone.",
}


def read_parent():
    prior.check(prior.path_sha(PARENT) == PARENT_SHA, "Published parent results differ")
    return prior.read(PARENT)


def parent_rows(parent, phase):
    return {r["id"]: r for r in parent[phase]}


def bindings(weights):
    from scripts.prepare_audio_specialists_v15 import ASSETS

    roster, parents = prior.sources()
    prior.check({name: prior.path_sha(weights / name) for name in ASSETS} == ASSETS, "Pinned voice assets differ")
    lock = {
        "schema": SCHEMA,
        "method": METHOD,
        "parent_sha256": PARENT_SHA,
        "source_inputs": parents,
        "voice_assets": ASSETS,
        "runner_sha256": prior.path_sha(Path(__file__)),
        "parent_runner_sha256": prior.path_sha(Path(prior.__file__)),
        "voice_code_sha256": prior.path_sha(ROOT / "backend/app/detectors/audio/upgraded.py"),
        "filterbank_sha256": prior.path_sha(ROOT / "backend/app/third_party/kaldi/fbank.py"),
        "runtime": {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "torch", "transformers", "pyloudnorm")},
    }
    return lock, roster


def frame_scores(model, pcm, *, robust):
    import torch

    from app.detectors.audio.upgraded import prepare_voice_window
    from app.third_party.kaldi.fbank import fbank

    signal = torch.from_numpy(prepare_voice_window(pcm))
    nonzero = torch.nonzero(signal).flatten()
    extent = int(nonzero[-1]) + 1 if len(nonzero) else len(signal)
    mel = fbank(signal[None, :max(extent, 400)] * 32768., num_mel_bins=80,
               sample_frequency=prior.RATE, dither=0., window_type="povey")
    mean, variance = mel.mean(0, keepdim=True), mel.var(0, keepdim=True)
    if robust:
        lower = torch.quantile(mel, LOW, dim=0)
        upper = torch.quantile(mel, HIGH, dim=0)
        bounded = mel.clamp(lower, upper)
        mean = torch.lerp(mean, bounded.mean(0, keepdim=True), BLEND)
        variance = torch.lerp(variance, bounded.var(0, keepdim=True), BLEND)
    normalized = (mel - mean) / torch.sqrt(variance + 1e-7)
    count = len(normalized) // 2 * 2
    frames = normalized[:count].reshape(1, count // 2, 160)
    mask = torch.ones(1, frames.shape[1], dtype=torch.long)
    states = model.backbone(input_features=frames, attention_mask=mask, output_hidden_states=True).hidden_states
    weighted = torch.stack(states).mul(model.layer_weights.softmax(dim=0)[:, None, None, None]).sum(0)[0]
    response = model.head(weighted).squeeze(-1)
    prior.check(0 < len(response) <= 199 and torch.isfinite(response).all().item(), "Unexpected source response")
    return response.cpu().numpy()


def capture_windows(pcm, spans, predictor):
    import numpy as np

    from app.detectors.audio.content import window_in_speech

    width, starts = prior.GRIDS["dense"]
    admitted = [window_in_speech(spans, start / prior.RATE, (start + width) / prior.RATE) for start in starts]
    contexts = []
    for offset in prior.CUTS:
        end = offset + 64000
        if not any(allowed and offset <= start <= end - width for start, allowed in zip(starts, admitted, strict=True)):
            continue
        scores = predictor(offset, pcm[offset:end])
        contexts.append((offset, offset + 280 + np.arange(len(scores)) * 320, scores))
    readouts = []
    for start, allowed in zip(starts, admitted, strict=True):
        end = start + width
        values = []
        if allowed:
            for offset, centers, scores in contexts:
                if offset <= start and end <= offset + 64000:
                    chosen = scores[(start <= centers) & (centers < end)]
                    if len(chosen) >= 40:
                        values.append(float(np.median(chosen)))
        readouts.append(min(values) if values else None)
    return readouts


def score_bank(roster, expected, phase, weights):
    import numpy as np
    import torch

    from app.detectors.audio.upgraded import SpeechActivity
    from scripts.audio_strong_traffic_view import network_free
    from scripts.prepare_audio_specialists_v15 import ASSETS
    from scripts.run_audio_specialists_v15 import load_voice

    torch.set_num_threads(1)
    network_free()
    model = load_voice({"weights": str(weights), "assets": ASSETS})
    speech = SpeechActivity(weights / "silero_vad.jit")
    scores, cache = {}, {}
    forwards = 0
    for row in roster:
        if (row["role"] == "assessment") != (phase == "assessment"):
            continue
        ident = prior.digest(row["pcm_sha256"].encode())
        original = expected[ident]
        prior.check(all(original[k] == row[k] for k in ("role", "truth", "kind", "family", "pcm_sha256")), "Parent case differs")
        raw = next(prior.source_data(prior.DATA / row["bank"], row))
        pcm = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.
        spans = speech.spans(pcm)
        expected_windows = [items[0]["score"] for items in original["measurements"]["mixture"]["dense"]]

        def predict(offset, values, raw=raw):
            nonlocal forwards
            pin = prior.digest(raw[2 * offset:2 * (offset + len(values))])
            if pin not in cache:
                cache[pin] = frame_scores(model, values, robust=True)
                forwards += 1
            return cache[pin]

        with torch.inference_mode():
            if not scores:
                vanilla = capture_windows(pcm, spans, lambda _, samples: frame_scores(model, samples, robust=False))
                prior.check(any(x is not None for x in vanilla) and all(
                    left is None and right is None or left is not None and right is not None and abs(left - right) <= .0001
                    for left, right in zip(vanilla, expected_windows, strict=True)), "Ordinary normalization failed native parity")
            values = capture_windows(pcm, spans, predict)
        prior.check([v is not None for v in values] == [v is not None for v in expected_windows], "Source interval coverage differs")
        scores[ident] = values
        if len(scores) % 8 == 0:
            print(phase, len(scores), "/", len(expected), "forwards", forwards, flush=True)
    return {"forwards": forwards, "scores": scores}


def calibrated(rows, scores):
    ordinary = [value for row in rows if row["truth"] is None for value in scores[row["id"]] if value is not None]
    prior.check(len(ordinary) >= 8, "Not enough admitted ordinary development scores")
    return max(ordinary)


def with_measurements(rows, scores):
    original = []
    for row in rows:
        dense = row["measurements"]["mixture"]["dense"]
        observed = [[{**items[0], "score": score}] for items, score in zip(dense, scores[row["id"]], strict=True)]
        original.append({**row, "measurements": {"mixture": {"dense": observed}}})
    return original


def summarize(parent, development, assessment, point):
    rows = parent["assessment"]
    robust_rows = with_measurements(rows, assessment)
    stock_point = parent["freeze"]["prior_point"]
    original_point = parent["points"]["mixture/dense"]["point"]
    routes = {"stock": (rows, "stock", stock_point), "dense_original": (rows, "dense", original_point),
              "dense_robust": (robust_rows, "dense", point)}
    outcomes = {name: prior.result(bank, "mixture", grid, fixed) for name, (bank, grid, fixed) in routes.items()}
    _, found, alarms = outcomes["dense_robust"]
    _, old_found, old_alarms = outcomes["stock"]
    gained = found - old_found
    families = {row["family"] for row in rows if row["id"] in gained}
    comparison = {"gained": len(gained), "lost": len(old_found - found), "new_false_groups": len(alarms - old_alarms),
                  "gain_families": len(families), "ordinary_groups": len({row["group"] for row in rows if row["truth"] is None})}
    comparison["qualifies_for_fresh_test"] = (comparison["lost"] == 0 and comparison["gained"] >= 2
        and comparison["gain_families"] >= 2 and comparison["new_false_groups"] <= 1
        and outcomes["dense_robust"][0].get("isolated_room_flags", 0) == 0)
    equal = {}
    for name, (bank, grid, _) in routes.items():
        counts = {}
        for row in bank:
            if row["truth"] is not None:
                continue
            response = prior.best(row, "mixture", grid)
            if response is not None:
                group = row["group"]
                counts[group] = max(response[0], counts.get(group, -math.inf))
        equal[name] = [prior.result(bank, "mixture", grid, cut)[0] for cut in sorted(counts.values(), reverse=True)[:4]]
    matched = {}
    for name, (bank, grid, _) in routes.items():
        successes = 0
        width, starts = prior.GRIDS[grid]
        for row in bank:
            if row["truth"] is None:
                continue
            peers = [r for r in bank if r["group"] == row["group"]
                     and r["kind"] in {"ordinary_same_host", "ordinary_other_voice"}]
            if len(peers) != 2:
                continue
            values = [r["measurements"]["mixture"][grid] for r in (row, *peers)]
            qualified = [(r[0]["score"], a[0]["score"], b[0]["score"])
                         for start, r, a, b in zip(starts, *values, strict=True)
                         if prior.useful([start, start + width], row["truth"])
                         and all(x[0]["score"] is not None for x in (r, a, b))]
            successes += bool(qualified) and max(x[0] for x in qualified) > max(max(x[1:]) for x in qualified)
        matched[name] = successes
    return {"fixed": {name: value[0] for name, value in outcomes.items()}, "comparison": comparison,
            "equal_false_groups": equal, "higher_than_both_splices_at_known_interval": matched,
            "ordinary_development_files": sum(row["truth"] is None for row in parent["development"]),
            "point": calibrated(parent["development"], development)}


def review(ledger, parent):
    lock = ledger["freeze"]
    prior.check(lock["schema"] == SCHEMA and lock["method"] == METHOD and lock["parent_sha256"] == PARENT_SHA,
                "Frozen method or source commitment differs")
    for phase in ("development", "assessment"):
        expected = parent_rows(parent, phase)
        scores = ledger[phase]["scores"]
        prior.check(scores.keys() == expected.keys(), "Missing source delivery")
        for ident, values in scores.items():
            old = expected[ident]["measurements"]["mixture"]["dense"]
            prior.check(all(v is None if p[0]["score"] is None else type(v) in (float, int) and math.isfinite(v)
                            for v, p in zip(values, old, strict=True)), "Invalid score or source coverage")
    point = calibrated(parent["development"], ledger["development"]["scores"])
    outcome = summarize(parent, ledger["development"]["scores"], ledger["assessment"]["scores"], point)
    prior.check(ledger["summary"] == outcome, "Published outcomes differ from recorded source ranges")
    return outcome


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "development", "assessment", "review"))
    parser.add_argument("run", type=Path)
    parser.add_argument("--voice", type=Path, default=ROOT / "models/weights/audio-upgrade")
    args = parser.parse_args()
    parent = read_parent()
    if args.phase == "review":
        print(json.dumps(review(prior.read(args.run), parent), sort_keys=True))
        return
    lock, roster = bindings(args.voice)
    freeze = args.run / "freeze.json"
    if args.phase == "freeze":
        args.run.mkdir(mode=0o700, exist_ok=True)
        prior.write(freeze, lock)
        return
    prior.check(prior.read(freeze) == lock, "Frozen runner or inputs differ")
    if args.phase == "development":
        bank = score_bank(roster, parent_rows(parent, args.phase), args.phase, args.voice)
        prior.write(args.run / "development.json", bank)
        prior.write(args.run / "point.json", {"value": calibrated(parent["development"], bank["scores"]),
                                              "sha256": prior.path_sha(args.run / "development.json")})
        return
    development = prior.read(args.run / "development.json")
    point = prior.read(args.run / "point.json")
    prior.check(point == {"value": calibrated(parent["development"], development["scores"]),
                         "sha256": prior.path_sha(args.run / "development.json")}, "Seal the development point before held inference")
    assessment = score_bank(roster, parent_rows(parent, args.phase), args.phase, args.voice)
    summary = summarize(parent, development["scores"], assessment["scores"], point["value"])
    ledger = {"freeze": lock, "development": development, "assessment": assessment, "summary": summary}
    review(ledger, parent)
    prior.write(args.run / "public.json", ledger)
    print(json.dumps({"fixed": summary["fixed"], "comparison": summary["comparison"]}, sort_keys=True))


if __name__ == "__main__":
    main()
