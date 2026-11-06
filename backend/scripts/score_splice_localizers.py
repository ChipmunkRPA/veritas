"""Run two published splice localizers with pinned, offline CPU inference."""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as transform

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.detectors.image import jpeg_bitstream  # noqa: E402
from app.detectors.image import localiser as loc  # noqa: E402
from scripts.audit_fused_production_grid import component_masks, to_source  # noqa: E402
from scripts.localization_benchmark import mask_metrics, read_manifest, sha, source  # noqa: E402

ROOT = Path(__file__).resolve().parents[1] / "data/localization-eval/alternative-checkpoints"
PUBLISHER_FILES = {
    "tidal-v30.1.pt": "ca3aa144e6ca08f2bb2eae93ac45d76c27a2922770e028bf90ef76ce974ee061",
    "tidal-model_architecture.py": "06ba9c339ab926bcd1e87aaeb6d5ece52101eb7ffed1c91eb937f73857a48ba5",
    "tidal-preprocessing.py": "881d8850f191cea76050016c620c63f656f5abe95723acbec75ed3684937b57e",
    "madl-dualstream.pt": "65a96f53964a8370af0984ef915a6aac513742762b5f805e406bb6261838f10d",
    "madl-dualstream-model.py": "626b86238d56357c364943389d5eb2a7b44f8dd2ea060ea746dae0958f6eb1ce",
    "madl-dataset.py": "e4f29c2012b1de2008cad1362bd3d67d41f86ebeca9a11d335656e69e84e2113",
}


def pinned(name: str) -> Path:
    path = ROOT / name
    if sha(path.read_bytes()) != PUBLISHER_FILES[name]:
        raise ValueError(f"Publisher asset digest mismatch for {name}")
    return path


def reviewed_source(name: str):
    path = pinned(name)
    identifier = "chipmunk_eval_" + path.stem.replace("-", "_")
    spec = importlib.util.spec_from_file_location(identifier, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[identifier] = module
    spec.loader.exec_module(module)
    return module


def state(name: str) -> dict:
    path = pinned(name)
    if torch.serialization.get_unsafe_globals_in_checkpoint(path):
        raise ValueError(f"Unsafe pickle globals in {name}")
    return torch.load(path, map_location="cpu", weights_only=True, mmap=True)


def tidal():
    sys.path.insert(0, str(ROOT / "deps"))
    architecture = reviewed_source("tidal-model_architecture.py")
    preprocess = reviewed_source("tidal-preprocessing.py")
    model = architecture.build_vrp301_model()
    model.load_state_dict(state("tidal-v30.1.pt"), strict=True)
    model.eval()

    def predict(image):
        values = model(preprocess.preprocess_image(image)).sigmoid()[0, 0].numpy()
        above = int((values > 0.7).sum())
        return values, above / values.size, above >= 400

    return predict, 0.7


def madl():
    architecture = reviewed_source("madl-dualstream-model.py")
    preprocess = reviewed_source("madl-dataset.py")
    saved = state("madl-dualstream.pt")
    expected = {"backbone_name": "convnext_tiny", "image_size": 512, "fusion_dim": 256, "num_classes": 3,
                "proposal_threshold": 0.45, "proposal_top_k": 5}
    if (saved["format_version"] != "madl.checkpoint.v1" or saved["architecture"] != expected
            or saved["class_names"] != ["real", "synthetic", "tampered"]):
        raise ValueError("Unexpected dual-stream architecture or class order")
    model = architecture.DualStreamPixelAgent(**expected, use_pretrained=False)
    model.load_state_dict(saved["state_dict"], strict=True)
    model.eval()

    def predict(image):
        resized = transform.resize(image, [512, 512], antialias=True)
        rgb = transform.to_tensor(resized).unsqueeze(0)
        noise = torch.from_numpy(preprocess.compute_noise_map(np.asarray(resized, dtype=np.uint8))).unsqueeze(0)
        result = model(rgb, noise)
        probabilities = result["logits"].softmax(dim=-1)[0]
        return result["heatmap_logits"].sigmoid()[0, 0].numpy(), float(probabilities[2]), int(probabilities.argmax()) == 2

    return predict, 0.45


def score_map(raw_map, image_gate, threshold, image_bytes, pixels, truth):
    height, width = truth.shape
    grid = loc.grid_shape(width, height, loc.GRID_MAX_SIDE["screening"])
    floating = cv2.resize(raw_map, grid, interpolation=cv2.INTER_LINEAR)
    values = np.rint(np.clip(floating.astype(np.float64), 0, 1) * 255) / 255.0
    structure = jpeg_bitstream.read(image_bytes)
    excluded, _ = loc.conditions_mask(pixels, structure.quality_luma if structure else None)
    reliable = loc.reliable_grid(excluded, grid)
    masks, _ = component_masks(values, reliable, width, height, threshold=threshold)
    retained = to_source(masks["per_map_display_cap"], width, height)
    return {"ungated": mask_metrics(retained, truth), "metrics": mask_metrics(retained & image_gate, truth)}


def evaluate(check: str, root: Path, output: Path) -> None:
    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    manifest, digest = read_manifest(root)
    predict, threshold = {"tidal": tidal, "madl": madl}[check]()
    output.mkdir(parents=True, exist_ok=False)
    policy = {"model": check, "device": "cpu", "manifest_sha256": digest,
              "asset_sha256": {name: value for name, value in PUBLISHER_FILES.items() if name.startswith(check)},
              "map_threshold": threshold,
              "image_gate": "native pixels above 0.70 >= 400" if check == "tidal" else "tampered class wins",
              "script_sha256": sha(Path(__file__).read_bytes())}
    (output / "policy.json").write_text(json.dumps(policy, indent=2, allow_nan=False) + "\n")
    with (output / "predictions.jsonl").open("x") as log, torch.inference_mode():
        for index, item in enumerate(manifest["items"], 1):
            image_bytes, pixels, truth = source(root, item)
            raw, score, eligible = predict(Image.fromarray(pixels))
            if not np.isfinite(raw).all() or not 0 <= raw.min() <= raw.max() <= 1 or not 0 <= score <= 1:
                raise ValueError("Invalid model response")
            path = output / f"map-{index:04d}.npy"
            np.save(path, raw, allow_pickle=False)
            result = {"id": item["id"], "label": item["label"], "family": item["family"],
                      "image_sha256": item["image_sha256"], "mask_sha256": item["mask_sha256"],
                      "score": score, "gate": eligible, "raw_map": path.name, "raw_map_sha256": sha(path.read_bytes()),
                      **score_map(raw, eligible, threshold, image_bytes, pixels, truth)}
            log.write(json.dumps(result, allow_nan=False) + "\n")
            log.flush()
            if index == 1 or index % 20 == 0 or index == len(manifest["items"]):
                print(f"{check}: {index}/{len(manifest['items'])}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=("tidal", "madl"))
    parser.add_argument("manifest_root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    evaluate(args.model, args.manifest_root, args.output)
