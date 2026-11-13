# SPDX-License-Identifier: Apache-2.0
"""Evaluate exact released FLAME regional weights on two frozen JPEG source banks."""

from __future__ import annotations

import argparse
import functools
import importlib
import sys
import tarfile
import tempfile
from importlib import metadata
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import torch
from PIL import Image
from torch.nn.functional import interpolate

from app.detectors.image import localiser as loc
from scripts import image_regional_tile_rescue_transfer as panel
from scripts import resume_image_regional_tile_rescue as prior
from scripts.localization_benchmark import mask_metrics

SCHEMA = "chipmunk.image-flame-jpeg-transfer/1"
FLAME_REV = "5d3d42f32934d64a898fd4fd05a815a9a3e32ebf"
SAM_REV = "2b90b9f5ceec907a1c18123530e92e794ad901a4"
FLAME_FILES = {
    "flame-source.tar.gz": "9a6a2fc81bab875132ff53de5e3bb7197b6b00c576a22397a6cb6f43d23fe538",
    "sam2-source.tar.gz": "1f2fbfad3ffa38110368abac76c6ef9df9c282a66d5c2807bc94abf4d2fb30f8",
    "flame_g2_ladmulti_sam2.pth": "9b6c67e3c35f647a3c9207275b1f1406c36780a2a3e0ee6b0f696d82161991bb",
    "model_params.json": "e46391983ca732313d1dc306e14ef3799c67a5bdf899a54a5fac8fef7e87ed42",
    "sam2.1_hiera_base_plus.pt": "a2345aede8715ab1d5d31b4a509fb160c5a4af1970f199d9054ccfb746c004c5",
}
DEPS = {"hydra-core": "1.3.2", "omegaconf": "2.3.0", "antlr4-python3-runtime": "4.9.3",
        "iopath": "0.1.10", "portalocker": "3.2.0"}
ARMS = ("current_luma", "native", "publisher_fixed", "ordinary_point", "native_plus_flame")
DESIGN = {
    "source_roles": {"retained_b": "development, point selected from ordinary controls only",
                     "external": "source-disjoint transfer, frozen points"},
    "input": "Exact source JPEG upright RGB. Pillow bilinear full-image 512 square resize; float32 unit RGB.",
    "model": "Released FLAME v0.1.0 trainable delta and official SAM2.1 Hiera Base Plus, CPU float32.",
    "selection": "Bilinear sigmoid projection to source coordinates; > 0.5, same JPEG reliability mask and minimum component area, top two by mean and size.",
    "ordinary_point": "Strictly above the second-largest development ordinary source maximum of top component means. Keep both qualifying components.",
    "additive": "Use candidate regions only on deliveries where the incumbent native JPEG check has no retained region.",
    "useful": {"iou": .30, "recall": .50},
    "advancement": {"additional_feathered_lama_groups": 2, "added_ordinary_groups": 2, "added_misplaced_edit_deliveries": 3},
    "confidence_weight": 0,
}
read, write, require, sha, file_sha = panel.read, panel.write, panel.require, panel.sha, panel.file_sha


def extract(archive, destination, prefixes):
    with tarfile.open(archive, "r:gz") as compressed:
        selected = [m for m in compressed.getmembers()
                    if m.isfile() and any(m.name == key or m.name.startswith(key + "/") for key in prefixes)]
        require(0 < len(selected) < 500 and sum(m.size for m in selected) < 8_000_000, "Unexpected source package extent")
        compressed.extractall(destination, members=selected, filter="data")


def asset_binding(folder):
    require(all(file_sha(folder / name) == digest for name, digest in FLAME_FILES.items()),
            "FLAME or SAM2.1 release file digest differs")
    sys.path.insert(0, str(folder / "deps"))
    require(all(metadata.version(name) == version for name, version in DEPS.items()),
            "Isolated SAM2.1 dependencies differ")
    return FLAME_FILES


def load_model(folder, source):
    flame = source / f"FLAME-{FLAME_REV}" / "FLAME"
    sam = source / f"sam2-{SAM_REV}"
    extract(folder / "flame-source.tar.gz", source, [str((flame / name).relative_to(source))
            for name in ("model", "sam2configs/sam2.1_hiera_b+.yaml")])
    extract(folder / "sam2-source.tar.gz", source, [str((sam / "sam2").relative_to(source))])
    sys.path[:0] = [str(flame), str(sam)]
    from hydra import initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    module = importlib.import_module("model.forgerylocalizer")
    config = read(folder / "model_params.json")
    model_cfg = config["model_config"]
    require(model_cfg["sam_backend"] == "sam2" and model_cfg["adapter_scales"] is None
            and config["training_config"]["img_size"] == 512 and config["training_config"]["val_force_resize"],
            "Unexpected released FLAME graph or geometry")
    kwargs = {key: model_cfg[key] for key in (
        "prompt_dim", "downscale", "train_sam_iou", "dropout_rate", "lad_tau", "lad_multi_taus",
        "forensic_operator", "sam_backend", "adapter_residual_scale", "adapter_type", "adapter_forensic_source",
        "sam3_prompt_mode", "coarse_prompt_head", "mask_compressor_kernel_size", "mask_compressor_output", "legacy_logit_head",
    )}
    GlobalHydra.instance().clear()
    base_builder = functools.partial(module.build_sam2, device="cpu")
    with initialize_config_dir(config_dir=str(flame / "sam2configs"), version_base=None):
        with patch.object(module, "build_sam2", base_builder):
            model = module.ForgeryLocalizer(config["sam_config"]["sam_config_file"],
                                            str(folder / "sam2.1_hiera_base_plus.pt"),
                                            output_resolution=(512, 512),
                                            use_detection_probe=config["data_config"]["authentic_ratio"] > 0,
                                            **kwargs)
    # The pinned Meta builder requires a strict weights-only base-state load.
    released = torch.load(folder / "flame_g2_ladmulti_sam2.pth", map_location="cpu", weights_only=True)
    trainable = {name for name, param in model.named_parameters() if param.requires_grad}
    state = released["model"]
    require(released["checkpoint_save_mode"] == "trainable_only" and set(state) == trainable and len(state) == 100,
            "Released delta does not cover the complete intended trainable tensor set")
    expected = model.state_dict()
    require(all(isinstance(v, torch.Tensor) and v.shape == expected[k].shape and v.dtype == expected[k].dtype
                and torch.isfinite(v).all().item() for k, v in state.items()), "Unexpected FLAME state tensor")
    missing, unexpected = model.load_state_dict(state, strict=False)
    require(not unexpected and set(missing) == set(expected) - trainable
            and all(torch.equal(v, model.state_dict()[k]) for k, v in state.items()), "FLAME assignment differs")
    released_buffers = set(dict(model.named_buffers())) & set(missing)
    return model.eval().requires_grad_(False), {"loaded_delta_tensors": len(state),
             "inherited_or_default_buffers": len(released_buffers), "checkpoint_epoch": released["epoch"]}


@torch.inference_mode()
def predict(model, rgb):
    image = Image.fromarray(rgb).resize((512, 512), Image.Resampling.BILINEAR)
    tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).unsqueeze(0).contiguous().float().div_(255)
    logits = model(tensor)
    require(logits.shape == (1, 1, 512, 512) and logits.dtype == torch.float32 and torch.isfinite(logits).all().item(),
            "Complete finite released FLAME response required")
    field = interpolate(logits.sigmoid(), size=rgb.shape[:2], mode="bilinear", align_corners=False)[0, 0].numpy()
    return field, sha(tensor.numpy().tobytes())


def components(field, rgb):
    excluded, _ = loc.conditions_mask(rgb, 75)
    selected = ((field > .5) & ~excluded).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(selected, connectivity=8)
    minimum = np.ceil(loc.MIN_REGION_FRACTION * field.size)
    regions = []
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area >= minimum:
            mask = labels == label
            regions.append((float(field[mask].mean()), area, mask))
    regions.sort(key=lambda hit: (hit[0], hit[1]), reverse=True)
    return regions[:2]


def union(regions, shape, point):
    out = np.zeros(shape, bool)
    for score, _, mask in regions:
        if score > point:
            out |= mask
    return out


def baseline(args, index, recorded, shape):
    payload = (args.tile_run / "masks" / f"{index:03}.bits").read_bytes()
    size = int(np.prod(shape))
    stride = (size + 7) // 8
    require(len(payload) == 3 * stride and sha(payload) == recorded["mask_sha256"], "Recorded comparator masks differ")
    return tuple(np.unpackbits(np.frombuffer(payload[j*stride:(j+1)*stride], np.uint8), count=size)
                 .reshape(shape).astype(bool) for j in range(2))


def freeze(args):
    previous = args.tile_run / "freeze.json"
    require(file_sha(previous) == prior.FROZEN_SHA256, "Original matched two-bank freeze required")
    frozen = read(previous)
    require(panel.profile(args) == frozen, "Frozen previous JPEG source identities differ")
    rows = prior.evidence_rows(args.tile_run, frozen)
    require(len(rows) == 200, "Original full comparator masks required")
    return {
        "schema": SCHEMA, "design": DESIGN, "assets": asset_binding(args.assets), "runner_sha256": file_sha(Path(__file__)),
        "jpeg_freeze_sha256": file_sha(previous), "incumbent_journal_sha256": file_sha(args.tile_run / "rows.jsonl"),
        "versions": {"torch": str(torch.__version__), "opencv": cv2.__version__, "numpy": np.__version__, **DEPS},
        "sources": frozen["source_counts"],
    }, frozen, rows


def run(args, frozen, incumbents, model):
    rows, maxima, point = [], {}, None
    model_forwards = 0
    for index, ((key, body, truth, _), expected, old) in enumerate(zip(
            panel.material(args), frozen["rows"], incumbents, strict=True)):
        require(panel.witness(key, body, truth) == expected, "Frozen JPEG or source truth changed")
        if key["panel"] == "external" and point is None:
            require(len(maxima) == 15 and index == prior.MAIN_ROWS, "Ordinary point requires all 15 development groups")
            point = sorted(maxima.values())[-2]
            write(args.output / "point.json", {"point": point, "ordinary_development_groups": len(maxima)})
            print("point", point, flush=True)
        rgb = panel.source_panel.common.decode(body)
        field, input_hash = predict(model, rgb)
        model_forwards += 1
        if index == 0:
            repeated, repeat_hash = predict(model, rgb)
            model_forwards += 1
            require(np.array_equal(field, repeated) and input_hash == repeat_hash, "First model forward is not repeatable")
        regions = components(field, rgb)
        fixed = union(regions, truth.shape, .5)
        luma, native = baseline(args, index, old, truth.shape)
        for arm, mask in zip(ARMS[:2], (luma, native), strict=True):
            require(mask_metrics(mask, truth) == old["measures"][arm], "Incumbent source pixels or overlap differs")
        score = max((value for value, _, _ in regions), default=.5)
        if key["panel"] == "retained_b":
            if key["role"] not in panel.previous.POSITIVE:
                maxima[key["group"]] = max(maxima.get(key["group"], .5), score)
            masks = (luma, native, fixed)
        else:
            selected = union(regions, truth.shape, point)
            masks = (luma, native, fixed, selected, native if native.any() else selected)
        packed = b"".join(np.packbits(mask, axis=None).tobytes() for mask in masks)
        (args.output / "masks" / f"{index:03}.bits").write_bytes(packed)
        rows.append(expected | {"input_tensor_sha256": input_hash, "model_response_sha256": sha(field.tobytes()),
                    "mask_sha256": sha(packed), "component_scores": [v for v, _, _ in regions],
                    "measures": {arm: mask_metrics(mask, truth) for arm, mask in zip(ARMS[:len(masks)], masks, strict=True)}})
        if (index + 1) % 8 == 0:
            print("scored", index + 1, flush=True)
    write(args.output / "rows.json", rows)
    return rows, point, model_forwards


def summary(rows, point):
    result = {}
    for phase in ("retained_b", "external"):
        selected = [r for r in rows if r["panel"] == phase]
        bank, groups = {}, {}
        for arm in selected[0]["measures"]:
            totals = {}
            groups[arm] = set()
            for row in selected:
                metric = row["measures"][arm]
                edited = row["role"] in panel.previous.POSITIVE
                flag = bool(metric["predicted_positive_pixels"])
                useful = edited and panel.previous.is_useful(metric)
                for family in (row["family"], "all"):
                    name = f"{family}/{row['role'] if edited else 'ordinary'}"
                    cell = totals.setdefault(name, {"files": 0, "flags": 0, "useful": 0, "misplaced": 0})
                    cell["files"] += 1
                    cell["flags"] += flag
                    cell["useful"] += useful
                    cell["misplaced"] += edited and flag and not useful
                if flag and not edited:
                    groups[arm].add(row["group"])
            bank[arm] = {"strata": totals, "ordinary_groups": len(groups[arm])}
        if phase == "external":
            gains = {r["group"] for r in selected if r["family"] == "lama" and r["role"] == "feathered"
                     and panel.previous.is_useful(r["measures"]["native_plus_flame"])
                     and not panel.previous.is_useful(r["measures"]["native"])}
            added_wrong = (sum(c["misplaced"] for k, c in bank["native_plus_flame"]["strata"].items() if k.startswith("all/"))
                           - sum(c["misplaced"] for k, c in bank["native"]["strata"].items() if k.startswith("all/")))
            values = {"additional_feathered_lama_groups": len(gains),
                      "added_ordinary_groups": len(groups["native_plus_flame"] - groups["native"]),
                      "added_misplaced_edit_deliveries": added_wrong}
            required = DESIGN["advancement"]
            bank["paired"] = values | {"passes": len(gains) >= required["additional_feathered_lama_groups"]
                                      and values["added_ordinary_groups"] <= required["added_ordinary_groups"]
                                      and added_wrong <= required["added_misplaced_edit_deliveries"]}
        result[phase] = bank
    return {"points": {"pixel": .5, "ordinary_component_mean": point}, "banks": result}


def audit(args, frozen, incumbents, rows):
    maxima = {}
    for index, ((key, body, truth, _), expected, old, row) in enumerate(zip(
            panel.material(args), frozen["rows"], incumbents, rows, strict=True)):
        require(panel.witness(key, body, truth) == expected and all(row[k] == v for k, v in expected.items()),
                "Native JPEG or truth replay differs")
        payload = (args.output / "masks" / f"{index:03}.bits").read_bytes()
        stride = (truth.size + 7) // 8
        require(len(payload) == stride * len(row["measures"]) and sha(payload) == row["mask_sha256"],
                "Retained source mask differs")
        masks = [np.unpackbits(np.frombuffer(payload[j*stride:(j+1)*stride], np.uint8), count=truth.size)
                 .reshape(truth.shape).astype(bool) for j in range(len(row["measures"]))]
        for arm, mask in zip(ARMS, masks):
            require(mask_metrics(mask, truth) == row["measures"][arm], "Source pixel metrics differ")
        require(all(np.array_equal(a, b) for a, b in zip(masks[:2], baseline(args, index, old, truth.shape), strict=True)),
                "Incumbent source evidence differs")
        if key["panel"] == "retained_b" and key["role"] not in panel.previous.POSITIVE:
            score = max(row["component_scores"], default=.5)
            maxima[key["group"]] = max(maxima.get(key["group"], .5), score)
        if key["panel"] == "external":
            require(not (masks[3] & ~masks[2]).any() and np.array_equal(masks[4], masks[1] if masks[1].any() else masks[3]),
                    "Frozen additive selection differs")
    value = read(args.output / "point.json")
    require(len(maxima) == value["ordinary_development_groups"] == 15 and sorted(maxima.values())[-2] == value["point"],
            "Ordinary group point does not replay")
    return value["point"]


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("stage", choices=("smoke", "freeze", "run", "review"))
    cli.add_argument("--assets", type=Path, required=True)
    for name in ("tile-run", "archive", "prior-probe", "residual", "early-f5", "later-f5", "checkpoint",
                 "output-fresh", "external", "output", "publication"):
        cli.add_argument(f"--{name}", type=Path)
    args = cli.parse_args()
    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    if args.stage == "smoke":
        asset_binding(args.assets)
        with tempfile.TemporaryDirectory(dir=args.assets) as folder:
            model, info = load_model(args.assets, Path(folder))
            rgb = np.random.default_rng(612).integers(16, 240, (512, 768, 3), dtype=np.uint8)
            field, digest = predict(model, rgb)
            print(info, field.shape, float(field.min()), float(field.max()), digest, flush=True)
        return
    profile, frozen, incumbents = freeze(args)
    freeze_path = args.output / "freeze.json"
    if args.stage == "freeze":
        args.output.mkdir(mode=0o700)
        (args.output / "masks").mkdir()
        write(freeze_path, profile)
        return
    require(profile == read(freeze_path), "FLAME study freeze differs")
    if args.stage == "run":
        with tempfile.TemporaryDirectory(dir=args.assets) as folder:
            model, info = load_model(args.assets, Path(folder))
            rows, point, calls = run(args, frozen, incumbents, model)
        write(args.output / "result.json", summary(rows, point) | {"model_load": info, "model_forwards": calls})
    else:
        rows = read(args.output / "rows.json")
        result = read(args.output / "result.json")
        require(len(rows) == 200 and args.publication is not None, "Both complete banks required")
        point = audit(args, frozen, incumbents, rows)
        require(all(result[k] == v for k, v in summary(rows, point).items()), "Summary differs on replay")
        write(args.publication, profile | result | {"native_source_replays": len(rows),
              "freeze_sha256": file_sha(freeze_path), "rows_sha256": file_sha(args.output / "rows.json")})
        print(result["banks"]["external"]["paired"], flush=True)


if __name__ == "__main__":
    main()
