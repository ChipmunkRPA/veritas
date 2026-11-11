"""Qualify exact FLUX.2 Klein reference editing on authored pixels, offline."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
import sysconfig
import time
import types
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "backend/data/independent-evaluation"
V40 = DATA / "generator-components-v40"
TRIAL = DATA / "image-flux-reference-edit-v271"
STAGE = TRIAL / "authored"
OUTPUT = STAGE / "output"
PUBLIC = ROOT / "docs/evaluation-panels/image-flux-reference-edit-smoke-freeze-v271.json"
ARCHIVE = Path("/Volumes/SEAGATE/Codex-Development-Archive/2026-11-08-Chipmunk-Verify")
BUNDLE = ARCHIVE / "image-flux-klein-4b-v40"
ACQUISITION_SHA = "f27d41114e0853b407a794f683a9151566f2a415502cb91beea50c86c6b374b3"
ENV = Path("/private/tmp/chipmunk-generator-runtime-v40")
MODEL = "mlx-community/flux2-klein-4b-4bit"
REVISION = "860e87183ceb29e39627c0612ebd66d8ea66e68c"
METADATA = V40 / "model-0-response.json"
METADATA_SHA = "e7c45af4cba93e8f6c27fdbd2194de7ce126b753bd9593388b7e3e41086bc086"
LOCK = ROOT / "docs/evaluation-panels/generator-runtime-v40-requirements.txt"
LOCK_SHA = "7963b0a3baff27df29efd171be663ff98f002234d61201b6e0492987399e1ce3"
WHEEL = V40 / "mflux-0.24.0-py3-none-any.whl"
WHEEL_SHA = "682368f04a1b2b0871144a8d57b7f7866e84dd0fd2bc3a765d1d57918e21237e"
NOTICE_SHA = "4a232d09478519580a5c00d6eb9aa402ba4f672ac15fb80edd459ba863769064"
MODULE = "scripts.image_flux_reference_edit_v271"
SIZE, SEED, STEPS = 256, 271_004, 4
REPEAT_MAX_SECONDS = 600
REPEAT_MAX_MLX_PEAK = 5 * 1024**3
SCHEDULER = "flow_match_euler_discrete"
PROMPT = "Remove the bright orange rectangle and fill the gap with the surrounding blue patterned background. Keep the framing and other objects unchanged."
CASES = {"interior": (82, 96, 146, 160), "edge": (0, 104, 64, 168)}
ASSETS = {
    "text_encoder/0.safetensors",
    "text_encoder/1.safetensors",
    "text_encoder/model.safetensors.index.json",
    "tokenizer/chat_template.jinja",
    "tokenizer/tokenizer.json",
    "tokenizer/tokenizer_config.json",
    "transformer/0.safetensors",
    "transformer/1.safetensors",
    "transformer/model.safetensors.index.json",
    "vae/0.safetensors",
    "vae/model.safetensors.index.json",
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def digest(array):
    return hashlib.sha256(array.tobytes()).hexdigest()


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def inventory():
    require(sha(METADATA) == METADATA_SHA, "Pinned v40 publisher response required")
    metadata = json.loads(METADATA.read_bytes())
    require(metadata["id"] == MODEL and metadata["sha"] == REVISION, "Publisher model revision differs")
    rows = {}
    for item in metadata["siblings"]:
        name = item["rfilename"]
        if name in ASSETS:
            rows[name] = {
                "bytes": item["size"],
                "lfs_sha256": item.get("lfs", {}).get("sha256"),
                "git_blob": item["blobId"],
            }
    require(
        set(rows) == ASSETS and sum(row["bytes"] for row in rows.values()) == 4_619_700_117,
        "Complete pinned bundle required",
    )
    return rows


def check_asset(root, name, row):
    path = root / name
    require(path.is_file() and path.resolve(strict=True) == path, f"Direct pinned asset required: {name}")
    require(path.stat().st_size == row["bytes"], f"Asset size differs: {name}")
    if row["lfs_sha256"] is not None:
        require(sha(path) == row["lfs_sha256"], f"Asset LFS digest differs: {name}")
    else:
        data = path.read_bytes()
        blob = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
        require(blob == row["git_blob"], f"Asset Git blob differs: {name}")


def verify_bundle(root, rows):
    receipt = root / "verified-assets.json"
    require(
        receipt.is_file() and receipt.resolve(strict=True) == receipt and sha(receipt) == ACQUISITION_SHA,
        "Exact independently verified v271 acquisition receipt required",
    )
    checked = json.loads(receipt.read_bytes())
    require(
        checked["revision"] == REVISION and checked["verified_bytes"] == 4_619_700_724,
        "Acquisition revision or byte count differs",
    )
    for folder in ("text_encoder", "tokenizer", "transformer", "vae"):
        found = {
            f"{folder}/{item.name}" for item in (root / folder).iterdir() if not item.name.startswith("._")
        }
        require(
            found == {name for name in rows if name.startswith(folder + "/")},
            f"Unexpected or absent files in {folder}",
        )
    for name, row in rows.items():
        check_asset(root, name, row)
    for folder in ("text_encoder", "transformer", "vae"):
        index = json.loads((root / folder / "model.safetensors.index.json").read_bytes())
        expected = {
            Path(name).name
            for name in rows
            if name.startswith(folder + "/") and name.endswith(".safetensors")
        }
        require(set(index["weight_map"].values()) == expected, f"Index shard map differs: {folder}")
        require(
            index["metadata"]["quantization_level"] == "4"
            and index["metadata"]["quantization_group_size"] == "64",
            f"Saved quantization differs: {folder}",
        )


def runtime_identity():
    require(
        Path(sys.prefix) == ENV and platform.python_version() == "3.12.6" and platform.machine() == "arm64",
        "Exact v40 isolated Mac runtime required",
    )
    require(sha(LOCK) == LOCK_SHA and sha(WHEEL) == WHEEL_SHA, "v40 runtime lock or wheel changed")
    expected = dict(re.findall(r"^([a-z0-9_.-]+)==([^\s]+)", LOCK.read_text(), re.M))
    site = Path(sysconfig.get_path("purelib"))
    require(
        site.resolve(strict=True) == site and site.is_relative_to(ENV),
        "Exact isolated site directory required",
    )
    installed = {
        row.metadata["Name"].lower().replace("_", "-"): row.version
        for row in importlib.metadata.distributions(path=[str(site)])
    }
    require(len(expected) == 56 and installed == expected, "Runtime packages differ from 56 v40 pins")
    with zipfile.ZipFile(WHEEL) as archive:
        names = {name for name in archive.namelist() if name.startswith("mflux/") and not name.endswith("/")}
        installed_names = {
            str(path.relative_to(site))
            for path in (site / "mflux").rglob("*")
            if path.is_file() and "__pycache__" not in path.parts
        }
        require(len(names) == 1059 and names == installed_names, "MFlux files differ from the reviewed wheel")
        for name in names:
            path = site / name
            require(
                path.resolve(strict=True) == path and path.read_bytes() == archive.read(name),
                f"MFlux file differs: {name}",
            )
    require(sha(ENV / "THIRD_PARTY_NOTICES/MFLUX-MIT.txt") == NOTICE_SHA, "Exact MIT notice required")
    return {"python": "3.12.6", "architecture": "arm64", "packages": 56, "mflux_wheel_files_exact": 1059}


def fixture(case):
    import numpy as np

    y, x = np.indices((SIZE, SIZE))
    rgb = np.stack((30 + (3 * x + y) % 59, 72 + (x + 3 * y) % 71, 145 + (2 * x + y) % 76), axis=2).astype(
        np.uint8
    )
    rgb[25:43, 182:215] = [220, 241, 94]
    rgb[204:226, 55:81] = [211, 71, 176]
    left, top, right, bottom = CASES[case]
    mask = np.zeros((SIZE, SIZE), np.uint8)
    mask[top:bottom, left:right] = 255
    rgb[mask > 0] = [255, 122, 21]
    return rgb, mask


def compose(source, raw, mask):
    import numpy as np

    from scripts.image_pair_truth_v21 import qualify

    require(
        source.shape == raw.shape == (SIZE, SIZE, 3) and source.dtype == raw.dtype == np.uint8,
        "Exact 256x256 native RGB geometry required",
    )
    require(
        mask.shape == (SIZE, SIZE)
        and mask.dtype == np.uint8
        and np.isin(mask, (0, 255)).all()
        and mask.min() == 0
        and mask.max() == 255,
        "Nondegenerate native binary mask required",
    )
    support = mask == 255
    raw_changed = np.any(source != raw, axis=2)
    composite = source.copy()
    composite[support] = raw[support]
    changed, proof = qualify(source, composite, mask)
    proof.update(
        {
            "scope": "Observed RGB changes after an explicit source-coordinate composition of a raw reference-edit output.",
            "raw_changes_inside": int((raw_changed & support).sum()),
            "raw_changes_outside": int((raw_changed & ~support).sum()),
            "source_rgb_sha256": digest(source),
            "raw_rgb_sha256": digest(raw),
            "declared_mask_sha256": digest(mask),
            "composite_rgb_sha256": digest(composite),
            "outside_source_rgb_sha256": digest(source[~support]),
            "outside_composite_rgb_sha256": digest(composite[~support]),
        }
    )
    return composite, changed.astype(np.uint8) * 255, proof


def plan():
    assets = inventory()
    code = (
        Path(__file__),
        ROOT / "backend/scripts/image_pair_truth_v21.py",
        ROOT / "backend/scripts/local_probe_limits_v24.py",
        ROOT / "backend/scripts/audio_tree_sampler_v64.py",
        ROOT / "backend/scripts/video_appearance_runtime_v108.py",
        ROOT / "backend/scripts/video_appearance_native_v108.py",
        ROOT / "backend/scripts/video_appearance_corrected_v112.py",
        ROOT / "backend/tests/test_image_flux_reference_edit_v271.py",
    )
    fixtures = {}
    for case, bounds in CASES.items():
        rgb, mask = fixture(case)
        fixtures[case] = {
            "bounds_ltrb_exclusive": list(bounds),
            "source_rgb_sha256": digest(rgb),
            "mask_sha256": digest(mask),
        }
    return {
        "schema": "chipmunk.image-flux-reference-edit-freeze/271",
        "code_sha256": {str(path.relative_to(ROOT)): sha(path) for path in code},
        "repo": MODEL,
        "revision": REVISION,
        "publisher_response_sha256": METADATA_SHA,
        "assets": assets,
        "acquisition_receipt_sha256": ACQUISITION_SHA,
        "mflux_version": "0.24.0",
        "mflux_wheel_sha256": WHEEL_SHA,
        "mflux_notice_sha256": NOTICE_SHA,
        "runtime_lock_sha256": LOCK_SHA,
        "size": [SIZE, SIZE],
        "fixtures": fixtures,
        "seed": SEED,
        "prompt": PROMPT,
        "steps": STEPS,
        "guidance": 1.0,
        "scheduler": SCHEDULER,
        "use_kv_cache": False,
        "device": "mlx.gpu",
        "max_generator_calls": 3,
        "conditional_repeat_max_elapsed_seconds": REPEAT_MAX_SECONDS,
        "conditional_repeat_max_mlx_peak_bytes": REPEAT_MAX_MLX_PEAK,
        "initial_ram_bytes": 6 * 1024**3,
        "max_sampled_tree_rss_bytes": 6 * 1024**3,
        "seconds_limit": 900,
        "scope": "Authored RGB geometry and runtime only. The generator takes a reference image. The mask is applied in a separate source-coordinate composition. No project photos, detector calls, or performance claim.",
    }


def verify():
    frozen = plan()
    require(
        PUBLIC.is_file() and json.loads(PUBLIC.read_bytes()) == frozen,
        "Exact prospective v271 freeze required",
    )
    return frozen


def save_png(path, pixels):
    import numpy as np
    from PIL import Image

    with path.open("xb") as stream:
        Image.fromarray(pixels).save(stream, format="PNG")
    with Image.open(path) as image:
        require(np.array_equal(np.asarray(image), pixels), "Lossless PNG round-trip failed")
    return sha(path)


def worker():
    require(
        (STAGE / "resources.jsonl").is_file()
        and (STAGE / "worker.log").is_file()
        and os.environ.get("HF_HUB_OFFLINE") == os.environ.get("TRANSFORMERS_OFFLINE") == "1",
        "Guarded offline worker required",
    )
    frozen = verify()
    started = time.monotonic()
    network = []

    def offline(event, _args):
        if event in {
            "socket.connect",
            "socket.sendto",
            "socket.getaddrinfo",
            "socket.gethostbyname",
            "socket.gethostbyaddr",
        }:
            network.append(event)
            raise RuntimeError("Offline worker refused network activity")

    sys.addaudithook(offline)
    identity = runtime_identity()
    verify_bundle(BUNDLE, frozen["assets"])
    import mlx.core as mx
    import numpy as np
    from mflux.models.flux2.variants.edit.flux2_klein_edit import Flux2KleinEdit

    for name in ("mflux", "numpy", "PIL", "mlx.core"):
        require(
            Path(sys.modules[name].__file__).resolve().is_relative_to(ENV),
            f"Unexpected dependency origin: {name}",
        )
    mx.set_default_device(mx.gpu)
    mx.set_cache_limit(128 * 1024**2)
    model = Flux2KleinEdit(model_path=str(BUNDLE), small_text_encoder=False, small_decoder=False)
    require(model.bits == 4, "Loaded model did not report 4-bit weights")
    OUTPUT.mkdir(mode=0o700)
    rows = []
    journal = OUTPUT / "attempts.jsonl"

    def generate(slot, case, stream):
        source, mask = fixture(case)
        folder = OUTPUT / slot
        folder.mkdir(mode=0o700)
        encoded = {
            "source": save_png(folder / "source.png", source),
            "mask": save_png(folder / "mask.png", mask),
        }
        stream.write(json.dumps({"slot": slot, "case": case, "status": "generator_started"}) + "\n")
        stream.flush()
        output = model.generate_image(
            seed=SEED,
            prompt=PROMPT,
            num_inference_steps=STEPS,
            width=SIZE,
            height=SIZE,
            guidance=1.0,
            scheduler=SCHEDULER,
            use_kv_cache=False,
            image_paths=[folder / "source.png"],
        )
        require(
            output.image.mode == "RGB" and output.image.size == (SIZE, SIZE),
            "Generator did not return exact RGB geometry",
        )
        raw = np.asarray(output.image)
        encoded["raw"] = save_png(folder / "raw.png", raw)
        raw_changes = np.any(source != raw, axis=2)
        stream.write(
            json.dumps(
                {
                    "slot": slot,
                    "case": case,
                    "status": "raw_observed",
                    "raw_png_sha256": encoded["raw"],
                    "inside_changes": int((raw_changes & (mask > 0)).sum()),
                    "outside_changes": int((raw_changes & (mask == 0)).sum()),
                }
            )
            + "\n"
        )
        stream.flush()
        composite, changed, proof = compose(source, raw, mask)
        encoded["composite"] = save_png(folder / "composite.png", composite)
        encoded["changed"] = save_png(folder / "changed.png", changed)
        row = {"slot": slot, "case": case, "status": "qualified_geometry", "png_sha256": encoded, **proof}
        rows.append(row)
        stream.write(json.dumps(row) + "\n")
        stream.flush()
        mx.clear_cache()

    with journal.open("x") as stream:
        generate("interior", "interior", stream)
        generate("edge", "edge", stream)
        if time.monotonic() - started <= REPEAT_MAX_SECONDS and mx.get_peak_memory() <= REPEAT_MAX_MLX_PEAK:
            generate("interior-repeat", "interior", stream)
    require(not network, "Unexpected network attempt")
    repeated = rows[0]["raw_rgb_sha256"] == rows[2]["raw_rgb_sha256"] if len(rows) == 3 else None
    write(
        OUTPUT / "result.json",
        {
            "schema": "chipmunk.image-flux-reference-edit-result/271",
            "status": "complete",
            "freeze_sha256": sha(PUBLIC),
            "attempt_journal_sha256": sha(journal),
            "runtime": identity,
            "cases": rows,
            "new_neural_calls": len(rows),
            "detector_calls": 0,
            "repeat_exact_raw_rgb": repeated,
            "mlx_peak_active_memory_bytes": int(mx.get_peak_memory()),
            "network_attempts": 0,
            "project_photos_used": 0,
        },
    )


def run():
    from scripts import video_appearance_runtime_v108 as guard
    from scripts.video_appearance_corrected_v112 import scoped

    require(
        (ENV / "bin/python").is_file() and BUNDLE.is_dir(),
        "Expected local runtime and acquired archive bundle",
    )
    TRIAL.mkdir(exist_ok=True)

    def launch(argv, **kwargs):
        require(
            argv[1:] == ["-B", "-m", "scripts.video_appearance_native_v108", "authored", str(OUTPUT)],
            "Exact v271 guarded worker required",
        )
        kwargs["env"].update(
            PYTHONNOUSERSITE="1",
            HF_HUB_DISABLE_TELEMETRY="1",
            HF_HUB_DISABLE_IMPLICIT_TOKEN="1",
            HF_HOME=str(STAGE / "hub-offline"),
        )
        return subprocess.Popen([str(ENV / "bin/python"), "-B", "-m", MODULE, "worker"], **kwargs)

    proxy = types.SimpleNamespace(
        Popen=launch, STDOUT=subprocess.STDOUT, TimeoutExpired=subprocess.TimeoutExpired
    )
    scoped(
        guard.run,
        {
            "TRIAL": TRIAL,
            "verify": verify,
            "prerequisites": lambda *_: True,
            "subprocess": proxy,
            "__file__": __file__,
        },
    )("authored")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "run", "worker"))
    args = parser.parse_args()
    if args.command == "freeze":
        write(PUBLIC, plan())
        print(json.dumps({"freeze_sha256": sha(PUBLIC), "new_neural_calls": 0}), flush=True)
    elif args.command == "run":
        run()
    else:
        worker()
