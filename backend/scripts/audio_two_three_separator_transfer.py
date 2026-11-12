"""Test additional voice evidence by comparing two- and three-stream separators."""
from __future__ import annotations

import argparse
import itertools
import time
import wave
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from scripts import audio_independent_overlap_count as voice
from scripts import audio_measured_speaker_routing as receipt
from scripts import audio_separation_third_voice_pilot as sep
from scripts import audio_stream_presence_transfer as prior

REVISION = "eb43c5bfbb2aa654630adbf849373bcec0a20ed4"
ASSETS = {"hyperparams.yaml": "19b0c807be63a5767509c30adb5c0176688ff2d077fd9bb95d0cf188cbd4d4e5",
          "encoder.ckpt": "740d082e75b92370bc12f176d5b5554f6383cf8bc607da8ac8a550255987ed4a",
          "decoder.ckpt": "05340634088efb00ec448f548af107d5d932057ea214dd64497f34106247e184",
          "masknet.ckpt": "f54762647082a8fe75a1cacc6c3d463b967223fc77c836cc4fb8ca2a272d9a75"}
ARMS = ("novel_energy", "qualified_novel_energy")
METHOD = {"matching": "At native alignment, match two outputs to the best distinct pair of three outputs by total absolute correlation. Score the unmatched third output by residual RMS after least-squares projection onto the two-output subspace, divided by mixture RMS.",
          "qualification": "Require both matched correlations >=0.60 and at least 0.48 s independently detected speech in the unmatched output. The reference labels never enter this calculation.",
          "points": "Second-highest group maximum over the 16 constructed development two-voice controls. Replay all constructed and natural confirmation cases.",
          "advance": "At least half the three-voice cases and at most one in six two-voice controls must receive compound support in both panels."}


def extra_measurement(mixture, two, three):
    def unit(rows):
        rows = rows - rows.mean(axis=1, keepdims=True)
        return rows / np.maximum(np.linalg.norm(rows, axis=1, keepdims=True), 1e-12)

    correlations = np.clip(abs(unit(two) @ unit(three).T), 0, 1)
    match = max(itertools.permutations(range(3), 2), key=lambda pair: correlations[0, pair[0]] + correlations[1, pair[1]])
    remaining = next(index for index in range(3) if index not in match)
    extra = three[remaining]
    relative = np.linalg.norm(extra) / max(np.linalg.norm(mixture), 1e-12)
    residual = (1 - prior.explained_energy(extra, two)) ** .5 * relative
    return extra, float(residual), min(float(correlations[index, value]) for index, value in enumerate(match))


def load_two(assets):
    import torch
    from speechbrain.lobes.models.dual_path import Decoder, Dual_Path_Model, Encoder, SBTransformerBlock

    def transformer():
        return SBTransformerBlock(num_layers=8, d_model=256, nhead=8, d_ffn=1024, dropout=0,
                                  use_positional_encoding=True, norm_before=True)

    model = {"encoder": Encoder(kernel_size=16, out_channels=256),
             "masknet": Dual_Path_Model(num_spks=2, in_channels=256, out_channels=256, num_layers=2, K=250,
                                        intra_model=transformer(), inter_model=transformer(), norm="ln",
                                        linear_layer_after_inter_intra=False, skip_around_intra=True),
             "decoder": Decoder(in_channels=256, out_channels=1, kernel_size=16, stride=8, bias=False)}
    for name, module in model.items():
        module.load_state_dict(torch.load(assets / f"{name}.ckpt", map_location="cpu", weights_only=True), strict=True)
        module.eval()
    return model


def run_model(model, mixture, channels):
    import torch

    with torch.inference_mode():
        encoded = model["encoder"](torch.from_numpy(mixture.astype("float32"))[None, ...])
        decoded = torch.stack([model["decoder"](mask * encoded)[0] for mask in model["masknet"](encoded)]).numpy()
    result = np.zeros((channels, len(mixture)))
    width = min(len(mixture), decoded.shape[1])
    result[:, :width] = decoded[:, :width]
    if not np.isfinite(result).all():
        raise ValueError("Separator outputs must be finite")
    return result


def measure(root, parent, three_assets, two_assets, vad, rows, natural):
    import torch
    from scipy.signal import resample_poly

    from app.detectors.audio.upgraded import SpeechActivity

    torch.set_num_threads(2)
    two_model, three_model = load_two(two_assets), sep.load_separator(three_assets)
    activity = SpeechActivity(vad)
    observed = []
    for index, row in enumerate(rows, 1):
        if row["panel"] == "natural_confirmation":
            room = natural["rooms"][row["group"]]["meeting"]
            with wave.open(str(root / "natural-raw" / f"{room}.wav"), "rb") as source:
                source.setpos(row["start_sample"])
                pcm = source.readframes(48000)
        else:
            pcm = receipt.pcm(parent, row)
        if len(pcm) != 96000 or sep.overlap.digest(pcm) != row["pcm_sha256"]:
            raise ValueError("Audio does not match the frozen input")
        mixture = resample_poly(voice.decode(pcm), 1, 2)
        two, three = run_model(two_model, mixture, 2), run_model(three_model, mixture, 3)
        extra, novel, match = extra_measurement(mixture, two, three)
        waveform = resample_poly(extra, 2, 1)
        peak = float(np.max(abs(waveform)))
        active = sum(round(end * 16000) - round(begin * 16000) for begin, end in
                     activity.spans((waveform * (.7 / peak)).astype("float32"))) if peak > 1e-6 else 0
        values = {"novel_energy": novel, "qualified_novel_energy": novel if match >= .6 and active >= 7680 else None}
        observed.append({**row, "scores": values, "minimum_match": match, "unmatched_active_samples": active})
        if index % 24 == 0:
            print(f"Compared {index}/{len(rows)} exact inputs", flush=True)
    return observed


def point_fit(rows):
    maxima = {arm: defaultdict(float) for arm in ARMS}
    for row in rows:
        for arm, score in row["scores"].items():
            maxima[arm][row["group"]] = max(maxima[arm][row["group"]], score or 0.)
    if any(len(groups) != 8 for groups in maxima.values()):
        raise ValueError("Expected all eight development control groups")
    return {arm: sorted(groups.values())[-2] for arm, groups in maxima.items()}


def summarize(rows, points):
    totals = defaultdict(Counter)
    groups = defaultdict(set)
    for row in rows:
        kind = "two" if row["expected_voices"] == 2 else "three"
        for condition in (row["panel"], f"{row['panel']}/{row['delivery']}"):
            totals[condition][f"planned_{kind}"] += 1
            for arm, score in row["scores"].items():
                hit = score is not None and score > points[arm]
                totals[condition][f"{arm}_{kind}_flag"] += int(hit)
                totals[condition][f"{arm}_{kind}_unassessed"] += int(score is None)
                if hit:
                    groups[condition, arm, kind].add(row["group"])
    def passes(panel):
        counts = totals[panel]
        return (2 * counts["qualified_novel_energy_three_flag"] >= counts["planned_three"] > 0
                and 6 * counts["qualified_novel_energy_two_flag"] <= counts["planned_two"] and counts["planned_two"] > 0)

    return {"totals": totals, "affected_groups": {"/".join(key): len(value) for key, value in groups.items()},
            "advance": passes("constructed_confirmation") and passes("natural_confirmation")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "fit", "confirm"))
    for name in ("output", "stream_run", "parent", "three_assets", "two_assets", "vad"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    rows, parent_lock, natural = prior.input_plan(args.stream_run, args.parent, args.three_assets, args.vad)
    lock_path = args.stream_run / "stream-presence-freeze.json"
    if (receipt.read(lock_path) != parent_lock
            or {key: receipt.sha(args.two_assets / key) for key in ASSETS} != ASSETS):
        raise ValueError("Exact earlier source roster and Apache-licensed assets required")
    plan = {"schema": "chipmunk.two-three-separator-transfer/1", "method": METHOD,
            "source_roster_freeze_sha256": receipt.sha(lock_path), "implementation_sha256": receipt.sha(Path(__file__)),
            "two_voice_revision": REVISION, "two_voice_assets": ASSETS, "three_voice_revision": sep.REVISION,
            "cases": dict(Counter(r["panel"] for r in rows if r["panel"] != "development" or r["expected_voices"] == 2))}
    path = args.output / "paired-separator-freeze.json"
    if args.action == "freeze":
        return receipt.write(path, plan)
    if receipt.read(path) != plan:
        raise ValueError("Use the exact frozen comparison")
    development = args.action == "fit"
    points_path = args.output / "points.json"
    if not development:
        points = receipt.read(points_path)
        if points["freeze_sha256"] != receipt.sha(path) or points["development_sha256"] != receipt.sha(args.output / "fit-private.json"):
            raise ValueError("Development point commitment differs")
    if development:
        selected = [r for r in rows if r["panel"] == "development" and r["expected_voices"] == 2]
    else:
        selected = [r for r in rows if r["panel"] != "development"]
    started = time.monotonic()
    observed = measure(args.stream_run, args.parent, args.three_assets, args.two_assets, args.vad, selected, natural)
    response = args.output / f"{args.action}-private.json"
    receipt.write(response, observed)
    if development:
        points = {"values": point_fit(observed), "freeze_sha256": receipt.sha(path), "development_sha256": receipt.sha(response)}
        receipt.write(points_path, points)
    result = summarize(observed, points["values"])
    receipt.write(args.output / f"{args.action}-aggregate.json", {**result, "freeze_sha256": receipt.sha(path),
        "points_sha256": receipt.sha(points_path), "response_sha256": receipt.sha(response),
        "separation_forwards": 2 * len(observed), "seconds": time.monotonic() - started})


if __name__ == "__main__":
    main()
