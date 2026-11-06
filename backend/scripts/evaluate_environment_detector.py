"""Research-only specialized environmental detector on unseen source datasets.

The architecture is the reviewed official MIT implementation. Weights are loaded
with weights_only=True and hashes are pinned. This script never changes service
models, calibration, reports or historical analyses.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import io
import json
import math
import urllib.parse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import soundfile as sf
from evaluate_background_continuity import REVISION, ROOT, fetch, order, save, wilson
from scipy.signal import resample_poly

from app.core.canonical import canonical_hash

CODE_SHA = "45e5fb35d59e3945b021430280697fe4fb33730fe59af92180563d4c9c1dadde"
WEIGHTS_SHA = "3b40c5ee8f51bc5f6173350f1b011ec37071f0cd5b5acbc220db24bb52a0a321"
CONFIG = {
    "first_conv": 128,
    "filts": [70, [1, 32], [32, 32], [32, 64], [64, 64]],
    "gat_dims": [64, 32],
    "pool_ratios": [0.5, 0.7, 0.5, 0.5],
    "temperatures": [2.0, 2.0, 100.0, 100.0],
}


def prepare():
    folder = ROOT / "environment-detector"
    folder.mkdir(exist_ok=True)
    (folder / "metadata").mkdir(exist_ok=True)
    (folder / "source").mkdir(exist_ok=True)
    metadata = list(csv.DictReader((ROOT / "upstream/test_metadata.csv").open()))
    selected = []
    for dataset in ("DCASE2023Task7", "Clotho"):
        real = [
            (i, row)
            for i, row in enumerate(metadata)
            if row["source dataset"] == dataset and row["faketype"] == "real"
        ]
        chosen = sorted(real, key=lambda pair: order("unseen:" + dataset + ":" + pair[1]["source wavename"]))[
            :32
        ]
        for index, row in chosen:
            selected.append(
                {
                    "row_offset": index,
                    "source_group": dataset + ":" + row["source wavename"],
                    "dataset": dataset,
                    "source_filename": row["source wavename"],
                    "expected_filename": row["wavename"],
                }
            )
    protocol = {
        "schema": "chipmunk.environment-detector-evaluation/1",
        "dataset_revision": REVISION,
        "upstream_split": "test",
        "selection": selected,
        "selected_offsets": [0, 2, 7],
        "offset_meaning": [
            "recorded original",
            "audio-to-audio regeneration",
            "unseen text-to-audio generator",
        ],
        "selection_seed": "chipmunk-background-continuity-20261104-v1",
        "source_metadata_sha256": hashlib.sha256(
            (ROOT / "upstream/test_metadata.csv").read_bytes()
        ).hexdigest(),
        "checkpoint_sha256": WEIGHTS_SHA,
        "architecture_sha256": CODE_SHA,
        "configuration": CONFIG,
        "input_policy": "16kHz mono via scipy polyphase resampling; first64600samples or cyclic repeat ifshort; raw amplitude; float32 exact scored input saved.",
        "calibration_policy": "Frozen34 recorded calibration groups from background protocol; threshold is95th percentile of fake-minus-real logit margin, higherquantile. No test tuning.",
        "test_policy": "64 recorded source files from two datasets absent from EnvSDD development; paired ATA and unseen-generator TTA variants keptwithsource. Originalfilename/PCM duplicates audited.",
        "production_admission": {
            "maximum_false_positive_rate": 0.05,
            "maximum_wilson_upper95": 0.10,
            "minimum_fake_recall": 0.8,
        },
        "limitations": [
            "Research candidate only. Checkpoint redistribution/commercial license is not established by its download page.",
            "This small test is not representative of allroomtone/music or codecs.",
            "Source-file groups cannot establish original recording-session independence.",
            "Whole-window logits cannot locate an edited word or background boundary.",
            "Uncalibrated logits and softmax values are not authenticity probabilities.",
            "Smallinputresampling difference from upstream librosa pipeline is documented; no authorbenchmark reproduction is claimed.",
        ],
    }
    save(folder / "protocol.json", protocol)

    def download(item):
        path = folder / "metadata" / f"{item['row_offset']}.json"
        if path.exists():
            data = json.loads(path.read_bytes())
        else:
            query = urllib.parse.urlencode(
                {
                    "dataset": "EnvSDD/EnvSDD",
                    "config": "default",
                    "split": "test",
                    "offset": item["row_offset"],
                    "length": 8,
                }
            )
            data = json.loads(fetch("https://datasets-server.huggingface.co/rows?" + query))
            save(path, data)
        rows = []
        for offset in (0, 2, 7):
            entry = data["rows"][offset]
            row = entry["row"]
            expected = metadata[item["row_offset"] + offset]
            url = row["audio"][0]["src"]
            if (
                row["filename"] != expected["wavename"]
                or expected["source wavename"] != item["source_filename"]
                or REVISION not in url
            ):
                raise RuntimeError("Frozen officialtest mapping changed")
            destination = folder / "source" / f"{entry['row_idx']}.wav"
            if not destination.exists():
                destination.write_bytes(fetch(url))
            raw = destination.read_bytes()
            rows.append(
                {
                    **item,
                    "row_idx": entry["row_idx"],
                    "condition": "real" if offset == 0 else "ata" if offset == 2 else "tta_unseen",
                    "label": row["label"],
                    "generator": row["generative_model"],
                    "attack_type": row["attack_type"],
                    "source_path": str(destination.relative_to(ROOT)),
                    "source_sha256": hashlib.sha256(raw).hexdigest(),
                    "source_url": url.split("?")[0],
                }
            )
        print("official test source", item["dataset"], item["source_filename"], flush=True)
        return rows

    rows = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        for batch in pool.map(download, selected):
            rows.extend(batch)
    old = json.loads((ROOT / "manifest.json").read_bytes())
    calibration = [
        {**r, "condition": "calibration_real"}
        for r in old["records"]
        if r["split"] == "calibration" and r["label"] == "real" and not r.get("excluded_reason")
    ]
    save(
        folder / "manifest.json",
        {"protocol_sha256": canonical_hash(protocol), "calibration": calibration, "test": rows},
    )
    print({"calibration": len(calibration), "test": len(rows)}, flush=True)


def score():
    import torch

    folder = ROOT / "environment-detector"
    manifest = json.loads((folder / "manifest.json").read_bytes())
    upstream = ROOT / "upstream"
    if (
        hashlib.sha256((upstream / "aasist.py").read_bytes()).hexdigest() != CODE_SHA
        or hashlib.sha256((upstream / "tta-aasist.pth").read_bytes()).hexdigest() != WEIGHTS_SHA
    ):
        raise RuntimeError("Unreviewed architecture or changed weights")
    spec = importlib.util.spec_from_file_location("reviewed_environment_architecture", upstream / "aasist.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    torch.set_num_threads(2)
    torch.manual_seed(0)
    model = module.Model(CONFIG)
    model.load_state_dict(
        torch.load(upstream / "tta-aasist.pth", map_location="cpu", weights_only=True), strict=True
    )
    model.eval()
    inputs = folder / "inputs"
    inputs.mkdir(exist_ok=True)
    rows = []
    for number, row in enumerate([*manifest["calibration"], *manifest["test"]]):
        raw = (ROOT / row["source_path"]).read_bytes()
        if hashlib.sha256(raw).hexdigest() != row["source_sha256"]:
            raise RuntimeError("Changed source")
        pcm, rate = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
        pcm = pcm.mean(axis=1)
        if rate != 16000:
            gcd = math.gcd(rate, 16000)
            pcm = resample_poly(pcm, 16000 // gcd, rate // gcd)
        if not len(pcm):
            raise RuntimeError("Empty environmental input")
        pcm = np.tile(pcm, int(np.ceil(64600 / len(pcm))))[:64600].astype("<f4")
        sha = hashlib.sha256(pcm.tobytes()).hexdigest()
        np.save(inputs / f"{sha}.npy", pcm)
        with torch.inference_mode():
            logits = model(torch.from_numpy(pcm.copy()).unsqueeze(0))[0].numpy()
        rows.append(
            {
                "source_group": row["source_group"],
                "condition": row["condition"],
                "source_sha256": row["source_sha256"],
                "dataset": row.get("dataset"),
                "generator": row.get("generator", row.get("generative_model")),
                "input_sha256": sha,
                "fake_logit": float(logits[0]),
                "real_logit": float(logits[1]),
                "fake_margin": float(logits[0] - logits[1]),
            }
        )
        if number % 20 == 0:
            print("environment inference", number, flush=True)
    save(
        folder / "scores.json",
        {
            "manifest_sha256": canonical_hash(manifest),
            "architecture_sha256": CODE_SHA,
            "checkpoint_sha256": WEIGHTS_SHA,
            "rows": rows,
        },
    )


def evaluate():
    folder = ROOT / "environment-detector"
    scores = json.loads((folder / "scores.json").read_bytes())
    protocol = json.loads((folder / "protocol.json").read_bytes())
    calibration = [r for r in scores["rows"] if r["condition"] == "calibration_real"]
    test = [r for r in scores["rows"] if r["condition"] != "calibration_real"]
    threshold = float(np.quantile([r["fake_margin"] for r in calibration], 0.95, method="higher"))
    frozen = {
        "scores_calibration_sha256": canonical_hash(calibration),
        "threshold": threshold,
        "positive": "fake_margin > threshold",
    }
    save(folder / "threshold.json", frozen)
    counts = {}
    for condition in ("real", "ata", "tta_unseen"):
        rows = [r for r in test if r["condition"] == condition]
        n = len(rows)
        flagged = sum(r["fake_margin"] > threshold for r in rows)
        counts[condition] = {
            "n": n,
            "flagged": flagged,
            "rate": flagged / n,
            "wilson_ci95": wilson(flagged, n),
        }
    dataset_counts = {}
    for dataset in sorted({r["dataset"] for r in test}):
        dataset_counts[dataset] = {}
        for condition in counts:
            rows = [r for r in test if r["dataset"] == dataset and r["condition"] == condition]
            flagged = sum(r["fake_margin"] > threshold for r in rows)
            dataset_counts[dataset][condition] = {
                "n": len(rows),
                "flagged": flagged,
                "rate": flagged / len(rows),
            }
    real = [r["fake_margin"] for r in test if r["condition"] == "real"]
    fake = [r["fake_margin"] for r in test if r["condition"] != "real"]
    auc = float(np.mean([float(f > r) + 0.5 * float(f == r) for f in fake for r in real]))
    a = protocol["production_admission"]
    admitted = (
        counts["real"]["rate"] <= a["maximum_false_positive_rate"]
        and counts["real"]["wilson_ci95"]["high"] <= a["maximum_wilson_upper95"]
        and min(counts["ata"]["rate"], counts["tta_unseen"]["rate"]) >= a["minimum_fake_recall"]
    )
    record = {
        "protocol_sha256": canonical_hash(protocol),
        "scores_sha256": canonical_hash(scores),
        "threshold": frozen,
        "counts": counts,
        "datasets": dataset_counts,
        "auc_real_vs_all_generated": auc,
        "statistical_admission_passed": admitted,
        "deployment_eligible": False,
        "commercial_weights_license_established": False,
        "duplicate_test_input_hashes": {
            k: v for k, v in Counter(r["input_sha256"] for r in test).items() if v > 1
        },
        "calibration_test_pcm_overlap": sorted(
            {r["input_sha256"] for r in calibration} & {r["input_sha256"] for r in test}
        ),
        "scope": "Frozen pretrained environmental detector; publisher-held-out datasets and unseen generator; a small research evaluation, not a localised manipulation proof.",
    }
    save(folder / "evaluation.json", record)
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["prepare", "score", "evaluate"])
    args = parser.parse_args()
    {"prepare": prepare, "score": score, "evaluate": evaluate}[args.stage]()
