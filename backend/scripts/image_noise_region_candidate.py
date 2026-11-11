"""Evaluate a pinned noise-guided checkpoint on sharp and blended local edits."""

from __future__ import annotations

import argparse
import json
import resource
import socket
import sys
import tarfile
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance

from app.detectors.image import localiser, regional_support
from scripts import image_background_patch_score as panel
from scripts import image_bg_real_insertion_score as scoring
from scripts import image_boundary_blend_transfer as blend
from scripts import image_relative_patch_transfer as shared
from scripts.image_native_comparison_readout_v45 import source_mask

SCHEMA = "chipmunk.image-noise-region-candidate/1"
CODE_REVISION = "9cf1535e050e97490e11634ae2faa6ae55c8611d"
CODE_SHA256 = "5cb7a6a91257635e5bdcdb452d72f091b1de2e39971f86f1d62c2854df97b27f"
MODEL_REVISION = "027d6f86b7761eafd0e237cc52a8e1070f05339e"
WEIGHTS_SHA256 = "1906e0d51a0a284008b847a4a2a3835f4263616ce6e7423915b97f46395e7112"
POINTS = (.5, .55, .6, .65, .7, .75, .8, .85, .9, .95, .98, 1.000001)
sha, write = shared.file_digest, shared.write


def offline(*args, **kwargs):
    raise RuntimeError("Network unavailable during the image study")


def checked_code(archive: Path, source: Path) -> None:
    if sha(archive) != CODE_SHA256 or source.name != f"ngiml-infer-{CODE_REVISION}":
        raise ValueError("Pinned upstream source archive or extracted source name differs")
    with tarfile.open(archive) as tar:
        frozen = {}
        for member in tar.getmembers():
            _, _, path = member.name.partition("/")
            if member.isfile() and (path.startswith("src/") and path.endswith(".py") or path == "LICENSE"):
                stream = tar.extractfile(member)
                frozen[path] = stream.read()
    actual = {str(path.relative_to(source)) for path in (source / "src").rglob("*.py")}
    if actual | {"LICENSE"} != set(frozen) or any((source / rel).read_bytes() != data for rel, data in frozen.items()):
        raise ValueError("The Python or license bytes differ from the pinned archive")


def predictor(args):
    import torch

    sys.path.insert(0, str(args.source))
    socket.socket.connect = socket.socket.connect_ex = socket.getaddrinfo = offline
    from src import runtime

    assert Path(runtime.__file__).resolve().is_relative_to(args.source.resolve())
    torch.set_num_threads(2)
    assert torch.serialization.get_unsafe_globals_in_checkpoint(args.weights) == []
    checkpoint = torch.load(args.weights, map_location="cpu", weights_only=True)
    config, origin = runtime.build_model_config_from_checkpoint(checkpoint)
    train = checkpoint["train_config"]
    assert origin == "train_config.model_config" and config.use_low_level and config.use_residual and not config.use_context
    assert train["resize_max_side"] == 448 and checkpoint["epoch"] == 14
    model = runtime.HybridNGIML(runtime.disable_pretrained_backbones(config))
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    normalization = runtime.resolve_normalization_mode_for_inference(checkpoint_train_config=train)
    device = torch.device("cpu")

    def predict(image):
        source = torch.from_numpy(np.asarray(image, dtype=np.float32) / 255).permute(2, 0, 1).contiguous()
        resized, _ = runtime.resize_image_for_inference(source, resize_max_side=448)
        response = runtime.predict_probability_map(model, resized, device, normalization_mode=normalization)
        assert response.shape == resized.shape[-2:]
        return response.numpy().astype(np.float32, copy=False)

    return predict, {"epoch": checkpoint["epoch"], "normalization": normalization, "torch": torch.__version__}


def points_from_controls(folder, predict, journal):
    _, plan, attempts = scoring.checked_inputs(folder)
    controls = {fmt: [] for fmt in panel.FORMATS}
    for slot, _, views, _ in scoring.panels(folder, plan, attempts):
        roles = ("original", "processed", "blur", "contrast")
        variants = blend.control_views(views["original"], views["processed"])
        for fmt in panel.FORMATS:
            group_flags = np.zeros(len(POINTS), dtype=bool)
            for role, rgb in zip(roles, variants, strict=True):
                data, image = panel.encoded(rgb, fmt)
                output = predict(image)
                read = regional_support.readout(output, True, POINTS[0], data, np.asarray(image))
                flags = [bool(localiser.components(read.recorded, read.reliable, image.width, image.height,
                                                   threshold=point)[0]) for point in POINTS]
                group_flags |= flags
                journal.write(json.dumps({"slot": slot, "role": role, "format": fmt, "input_sha256": shared.digest(data),
                                          "map_sha256": shared.digest(output.tobytes()), "flags": flags}) + "\n")
            controls[fmt].append(group_flags)
        journal.flush()
    result = {}
    for fmt, outcomes in controls.items():
        assert len(outcomes) == 8
        flags = np.count_nonzero(outcomes, axis=0).tolist()
        result[fmt] = {"point": next(point for point, count in zip(POINTS, flags, strict=True) if count <= 1),
                       "point_grid": list(POINTS), "flagged_source_groups": flags, "source_groups": len(outcomes)}
    return result


def confirmation(intake, receipt, predict, points, journal):
    locked = {(row["slot"], row["role"], row["format"]): row for row in receipt}
    assert len(locked) == 192
    outcomes = []
    for row, original, hard_truth in panel.inputs(intake):
        source, box = original["original"], row["box"]
        contrast = np.asarray(ImageEnhance.Contrast(Image.fromarray(source)).enhance(1.2))
        views = original | {"feathered_edit": blend.local(source, original["edited"], box, feather=True),
                            "feathered_processing": blend.local(source, original["local_processing"], box, feather=True),
                            "local_contrast": blend.local(source, contrast, box)}
        for role, rgb in views.items():
            expected = np.any(rgb != source, axis=2) if role in ("edited", "feathered_edit") else np.zeros_like(hard_truth)
            for fmt in panel.FORMATS:
                data, image = panel.encoded(rgb, fmt)
                binding = locked.pop((row["slot"], role, fmt))
                assert shared.digest(data) == binding["input_sha256"] and row["generator"] == binding["generator"]
                assert int(expected.sum()) == binding["measurements"]["boundary_prominence"]["target"]
                output = predict(image)
                pixels = np.asarray(image)
                metrics = {}
                for name, point in (("half_point", POINTS[0]), ("format_control_point", points[fmt]["point"])):
                    selection = regional_support.readout(output, True, point, data, pixels).source_mask(image.width, image.height)
                    assert np.array_equal(selection, source_mask(output, True, point, data, pixels, expected))
                    metrics[name] = scoring.measure(selection, expected)
                entry = {**{key: binding[key] for key in ("slot", "role", "generator", "format", "input_sha256")},
                         "map_sha256": shared.digest(output.tobytes()), "map_extent": list(output.shape),
                         "response_max": float(output.max()), "measurements": metrics}
                journal.write(json.dumps(entry, allow_nan=False) + "\n")
                outcomes.append(entry)
        journal.flush()
        print(f"Group {row['slot']} scored", flush=True)
    assert not locked and len(outcomes) == len(receipt)
    return outcomes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "run"))
    for name in ("archive", "source", "weights", "development", "intake", "blend_receipt", "destination"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    checked_code(args.archive, args.source)
    assert sha(args.weights) == WEIGHTS_SHA256
    candidate = json.loads((args.blend_receipt / "result.json").read_bytes())
    rows_file = args.blend_receipt / "rows.json"
    assert sha(rows_file) == candidate["journal_sha256"]
    identity = {"schema": SCHEMA, "implementation_sha256": sha(Path(__file__)), "upstream_code_sha256": CODE_SHA256,
                "code_revision": CODE_REVISION, "model_revision": MODEL_REVISION, "weights_sha256": WEIGHTS_SHA256,
                "development_sha256": sha(args.development / "result.json"), "intake_sha256": sha(args.intake / "result.json"),
                "paired_journal_sha256": sha(rows_file), "point_grid": list(POINTS),
                "rule": "smallest point flagging at most one of eight separate control groups per format",
                "preparation": "full aspect-preserved view, cap short side at 448, no central crop, FP32 CPU"}
    if args.stage == "freeze":
        args.destination.mkdir(mode=0o700)
        write(args.destination / "freeze.json", identity)
        return
    assert identity == json.loads((args.destination / "freeze.json").read_bytes())
    started = time.monotonic()
    predict, runtime = predictor(args)
    calibration = args.destination / "control-rows.jsonl"
    with calibration.open("x") as journal:
        points = points_from_controls(args.development, predict, journal)
    write(args.destination / "points.json", points)
    print("Fixed points sealed before confirmation", flush=True)
    outcomes = args.destination / "rows.jsonl"
    with outcomes.open("x") as journal:
        rows = confirmation(args.intake, json.loads(rows_file.read_bytes()), predict, points, journal)
    feathered = [{**row, "delivery": row["format"], "role": "edited"} for row in rows if row["role"] == "feathered_edit"]
    write(args.destination / "result.json", {"schema": SCHEMA, "freeze_sha256": sha(args.destination / "freeze.json"),
          "points_sha256": sha(args.destination / "points.json"), "control_journal_sha256": sha(calibration),
          "results_sha256": sha(outcomes), "development_forwards": 64, "confirmation_forwards": len(rows),
          "runtime": runtime, "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
          "elapsed_s": round(time.monotonic() - started, 2), "feathered": scoring.summarize(feathered),
          "sharp_and_controls": panel.summarise([row for row in rows if row["role"] != "feathered_edit"])})


if __name__ == "__main__":
    main()
