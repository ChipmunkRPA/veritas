"""Qualify a pinned separator on exact retained two- and three-voice mixtures."""
from __future__ import annotations

import argparse
import itertools
import math
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from scripts import audio_hard_overlap_transfer as overlap
from scripts import audio_measured_speaker_routing as receipt

REVISION = "1f2d824e2333aec9e60d0030fb144885c1afc426"
ASSETS = {"hyperparams.yaml": "4f0ae05600c66d7f8cf1269f54fd32dad576d4d473ad7169ef6ab26190658233",
          "encoder.ckpt": "e949e80c1a96965b723460209bb155eac1fdb2f6d2cf4e7b3e1d3152716f38d0",
          "decoder.ckpt": "9afdda4e4ecb9eefb5d49329193c7a9215ae7443aab206f2fe4eb0f50988ca53",
          "masknet.ckpt": "416af7b09d912c486261dd824788dd92d3b21373c8ec4d029209792522f4dbcb"}
PARENT_MANIFEST = "207f2084f1dd036f197c404c5b727f7740266b6eaa2037cb6488da33c954399a"
QUOTAS = {"et": 1, "fi": 2, "lt": 1, "nl": 3, "sl": 1}
TEST_CASES = {"two", "three_quiet", "three_faint"}


def signal_quality(estimate, reference):
    """Scale-invariant source-to-distortion ratio with fixed sample alignment."""
    estimate = estimate - estimate.mean()
    reference = reference - reference.mean()
    if np.dot(reference, reference) <= 1e-12:
        raise ValueError("The known source must contain a measurable waveform")
    projection = reference * (np.dot(estimate, reference) / np.dot(reference, reference))
    return 10 * math.log10((np.dot(projection, projection) + 1e-12) /
                           (np.dot(estimate - projection, estimate - projection) + 1e-12))


def matches(mixture, references, estimates):
    matrix = [[signal_quality(output, target) for output in estimates] for target in references]
    assignment = max(itertools.permutations(range(len(estimates)), len(references)),
                     key=lambda permutation: sum(matrix[i][j] for i, j in enumerate(permutation)))
    measurements = [{"si_sdr_db": matrix[i][j], "gain_db": matrix[i][j] - signal_quality(mixture, references[i])}
                    for i, j in enumerate(assignment)]
    unmatched = set(range(len(estimates))).difference(assignment)
    ratio = max((np.linalg.norm(estimates[j]) / np.linalg.norm(mixture) for j in unmatched), default=0.)
    return measurements, ratio


def sources(pcm: Path, row: dict):
    """Recover the known reference stems using the already frozen mixing recipe."""
    def decode(blob):
        return np.frombuffer(blob, dtype="<i2").astype("float64") / 32768

    loaded = [decode((pcm / "stems" / f"{row['group']:03}-{i}.pcm").read_bytes()) for i in range(3)]
    if overlap.digest(overlap.mix(loaded)[row["role_rotation"], row["delivery"], row["condition"]]) != row["pcm_sha256"]:
        raise ValueError("Frozen mixture cannot be reconstructed from the reference stems")
    if row["delivery"] == "reflective":
        loaded = [overlap.reflected(voice) for voice in loaded]
    return loaded[:row["expected_voices"]]  # Pilot only evaluates rotation 0. SI-SDR is invariant to each source gain.


def load_separator(assets: Path):
    import torch
    from speechbrain.lobes.models.dual_path import Decoder, Dual_Path_Model, Encoder, SBTransformerBlock

    def transformer():
        return SBTransformerBlock(num_layers=8, d_model=256, nhead=8, d_ffn=1024, dropout=0,
                                  use_positional_encoding=True, norm_before=True)

    model = {"encoder": Encoder(kernel_size=16, out_channels=256),
             "masknet": Dual_Path_Model(num_spks=3, in_channels=256, out_channels=256, num_layers=2, K=250,
                                        intra_model=transformer(), inter_model=transformer(), norm="ln",
                                        linear_layer_after_inter_intra=False, skip_around_intra=True),
             "decoder": Decoder(in_channels=256, out_channels=1, kernel_size=16, stride=8, bias=False)}
    for name, layer in model.items():
        layer.load_state_dict(torch.load(assets / f"{name}.ckpt", map_location="cpu", weights_only=True), strict=True)
        layer.eval()
    return model


def execute(folder: Path, assets: Path, parent: Path, action: str):
    if receipt.sha(parent / "manifest.json") != PARENT_MANIFEST or {n: receipt.sha(assets / n) for n in ASSETS} != ASSETS:
        raise ValueError("Exact original licensed weights and parent panel are required")
    manifest = receipt.read(parent / "manifest.json")
    groups = []
    for lang, limit in QUOTAS.items():
        candidates = [r["group"] for r in manifest["groups"] if r["language"] == lang]
        groups.extend(sorted(candidates, key=lambda value: overlap.digest(f"separation/1:{value}".encode()))[:limit])
    rows = [r for r in manifest["rows"] if r["group"] in groups and r["role_rotation"] == 0 and r["condition"] in TEST_CASES]
    if len(groups) != 8 or len(rows) != 48:
        raise ValueError("The eight disjoint groups and their fixed 48 selected views are required")
    fixed = {"schema": "chipmunk.third-voice-separation-pilot/1", "revision": REVISION, "assets": ASSETS,
             "parent_manifest_sha256": PARENT_MANIFEST, "implementation_sha256": receipt.sha(Path(__file__)),
             "inputs_sha256": overlap.digest(str([(r["id"], r["pcm_sha256"]) for r in rows]).encode()),
             "groups": len(groups), "views": len(rows),
             "advance": "Recover the third reference at >=0 dB SI-SDR and >=3 dB above the mixture in at least 12 of 16 views at each level. At most 2 of 16 two-voice controls may have an unmatched channel at or above 10% of mixture RMS."}
    frozen = folder / "separator-freeze.json"
    if action == "freeze":
        return receipt.write(frozen, fixed)
    if receipt.read(frozen) != fixed:
        raise ValueError("Use the exact frozen comparison")
    import torch
    from scipy.signal import resample_poly

    torch.set_num_threads(2)
    model = load_separator(assets)
    started = time.monotonic()
    local = []
    totals = defaultdict(Counter)
    for row in rows:
        mixture = resample_poly(np.frombuffer(receipt.pcm(parent, row), "<i2").astype("float64") / 32768, 1, 2)
        stems = [resample_poly(item, 1, 2) for item in sources(parent, row)]
        with torch.inference_mode():
            encoded = model["encoder"](torch.from_numpy(mixture.astype("float32"))[None, ...])
            masks = model["masknet"](encoded)
            decoded = torch.stack([model["decoder"](mask * encoded)[0] for mask in masks])
        estimate = np.zeros((3, len(mixture)), dtype="float64")
        width = min(estimate.shape[1], decoded.shape[1])
        estimate[:, :width] = decoded[:, :width].numpy()
        if not np.isfinite(estimate).all():
            raise ValueError("Invalid three-channel separator output")
        qualities, extra = matches(mixture, stems, estimate)
        tally = totals[row["condition"]]
        tally["measured_views"] += 1
        if row["expected_voices"] == 3:
            third = qualities[2]
            tally["third_recovered"] += int(third["si_sdr_db"] >= 0 and third["gain_db"] >= 3)
        else:
            tally["extra_output_over_10_pct_mixture_rms"] += int(extra >= .10)
        local.append({"id": row["id"], "measurements": qualities, "unmatched_rms_ratio": extra})
    receipt.write(folder / "separator-private.json", local)
    receipt.write(folder / "separator-aggregate.json", {"protocol": fixed, "freeze_sha256": receipt.sha(frozen),
        "private_result_sha256": receipt.sha(folder / "separator-private.json"), "summary": totals,
        "forwards": len(local), "seconds": time.monotonic() - started})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "run"))
    parser.add_argument("folder", type=Path)
    parser.add_argument("assets", type=Path)
    parser.add_argument("parent", type=Path)
    args = parser.parse_args()
    execute(args.folder, args.assets, args.parent, args.action)
