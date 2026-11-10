"""Pinned isolated geometry wheels and import probe; no model execution."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import subprocess
import sys
import types
import zipfile
from pathlib import Path, PurePosixPath

from scripts import acquire_grid_transfer_v130 as acquisition
from scripts.video_appearance_corrected_v112 import scoped

ROOT, BASE, PANEL = acquisition.ROOT, acquisition.BASE, acquisition.PANEL
TRIAL = BASE / "portrait-geometry-runtime-v132"
SITE = TRIAL / "site"
PUBLIC = PANEL / "portrait-geometry-runtime-freeze-v132.json"
CODE = (
    "backend/scripts/portrait_geometry_runtime_v132.py",
    "backend/tests/test_portrait_geometry_runtime_v132.py",
    "docs/portrait-geometry-runtime-v132-method.md",
)
PINS = {
    "mediapipe": (
        "0.10.32",
        "mediapipe-0.10.32-py3-none-macosx_11_0_arm64.whl",
        19386286,
        "b62178b7585e0bb8789075c43bbb3e352fbc4a8f765797fded509f86a098b29b",
    ),
    "absl-py": (
        "2.3.1",
        "absl_py-2.3.1-py3-none-any.whl",
        135811,
        "eeecf07f0c2a93ace0772c92e596ace6d3d3996c042b2128459aaae2a76de11d",
    ),
    "flatbuffers": (
        "25.9.23",
        "flatbuffers-25.9.23-py2.py3-none-any.whl",
        30869,
        "255538574d6cb6d0a79a17ec8bc0d30985913b87513a01cce8bcdb6b4c44d0e2",
    ),
    "sounddevice": (
        "0.5.3",
        "sounddevice-0.5.3-py3-none-macosx_10_6_x86_64.macosx_10_6_universal2.whl",
        108306,
        "278dc4451fff70934a176df048b77d80d7ce1623a6ec9db8b34b806f3112f9c2",
    ),
}
SERVING_PACKAGES = ("opencv-python-headless", "numpy", "cffi", "protobuf", "matplotlib")
read, write, sha, require = acquisition.read, acquisition.write, acquisition.sha, acquisition.require


def serving_identity():
    return {
        name: {
            "version": (dist := importlib.metadata.distribution(name)).version,
            "metadata_sha256": hashlib.sha256(dist.read_text("METADATA").encode()).hexdigest(),
        }
        for name in SERVING_PACKAGES
    }


def plan():
    rows = []
    for name, (version, filename, size, digest) in PINS.items():
        source = acquisition.METADATA / f"{name}-pypi.json"
        metadata = read(source)
        require(metadata["info"]["version"] == version, "Exact publisher package version required")
        candidates = [r for r in metadata["urls"] if r["filename"] == filename]
        require(len(candidates) == 1, "One exact platform wheel required")
        item = candidates[0]
        require(
            item["size"] == size
            and item["digests"]["sha256"] == digest
            and item["url"].startswith("https://files.pythonhosted.org/packages/")
            and item["url"].endswith("/" + filename),
            "Pinned publisher wheel identity required",
        )
        rows.append(
            {
                "package": name,
                "version": version,
                "filename": filename,
                "url": item["url"],
                "bytes": size,
                "sha256": digest,
                "md5": item["digests"]["md5"],
                "publisher_metadata_sha256": sha(source),
            }
        )
    return rows


def identity():
    acquisition.monitor_verified()
    return {
        "schema": "chipmunk.portrait-geometry-runtime-freeze/132",
        "code_sha256": {p: sha(ROOT / p) for p in CODE},
        "wheels": plan(),
        "existing_application_packages": serving_identity(),
        "install": "pip --no-deps --no-index --no-compile --target exclusive-research-site",
        "opencv_policy": "Reuse pinned existing headless OpenCV; do not install declared contrib dependency or change the application environment. Actual used API imports are qualified separately; this is not a full dependency-solver/pip-check claim.",
        "licenses": {
            "mediapipe": "Apache-2.0",
            "absl-py": "Apache-2.0",
            "flatbuffers": "Apache-2.0",
            "sounddevice": "MIT",
        },
        "new_neural_calls": 0,
    }


def freeze():
    body = identity()
    TRIAL.mkdir(exist_ok=False)
    write(TRIAL / "freeze.json", body)
    write(PUBLIC, body)
    print('{"status":"frozen","wheels":4,"new_neural_calls":0}', flush=True)


def verify():
    require(read(TRIAL / "freeze.json") == read(PUBLIC) == identity(), "Unchanged isolated runtime required")


def installed_manifest(wheels, site):
    observed = []
    for wheel in wheels:
        with zipfile.ZipFile(wheel) as archive:
            entries = archive.infolist()
            require(len({r.filename for r in entries}) == len(entries), "Unique wheel paths required")
            for member in entries:
                if member.is_dir():
                    continue
                parts = PurePosixPath(member.filename).parts
                require(
                    parts
                    and not PurePosixPath(member.filename).is_absolute()
                    and ".." not in parts
                    and "\\" not in member.filename
                    and not any(p.endswith(".data") for p in parts),
                    "Only bounded direct library wheel paths are admitted",
                )
                if parts[-1] == "RECORD" and parts[-2].endswith(".dist-info"):
                    continue  # pip regenerates RECORD, not the original source/native/library bytes.
                target = site / member.filename
                require(
                    target.resolve().is_relative_to(site.resolve()), "Installed path escaped isolated site"
                )
                expected = hashlib.sha256(archive.read(member)).hexdigest()
                require(
                    target.is_file() and sha(target) == expected,
                    "Installed wheel member differs from publisher bytes",
                )
                observed.append({"file": member.filename, "sha256": expected})
    require(observed, "Complete installed wheel source/native files required")
    return sorted(observed, key=lambda r: r["file"])


def install(output):
    before = serving_identity()
    wheels = TRIAL / "wheels"
    wheels.mkdir(exist_ok=False)
    downloaded = []
    for row in plan():
        path = wheels / row["filename"]
        evidence = acquisition.download(row, path)
        require(evidence["sha256"] == row["sha256"], "Publisher SHA-256 mismatch")
        downloaded.append(path)
    require(not SITE.exists(), "Never overwrite an installed research runtime")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--no-index",
            "--no-compile",
            "--disable-pip-version-check",
            "--target",
            str(SITE),
            *map(str, downloaded),
        ],
        check=True,
    )
    manifest = installed_manifest(downloaded, SITE)
    write(output / "installed-files.json", manifest)
    sys.path.insert(0, str(SITE))
    importlib.invalidate_caches()
    modules = {
        name: importlib.import_module(name) for name in ("mediapipe", "absl", "flatbuffers", "sounddevice")
    }
    for name, module in modules.items():
        require(
            Path(module.__file__).resolve().is_relative_to(SITE.resolve()), "Isolated import path required"
        )
    for name, row in PINS.items():
        require(importlib.metadata.version(name) == row[0], "Exact isolated imported distribution required")
    import cv2
    from mediapipe.tasks.python import vision
    from mediapipe.tasks.python.core.base_options import BaseOptions

    require(
        hasattr(vision, "FaceLandmarker") and hasattr(BaseOptions.Delegate, "CPU"),
        "Expected synchronous geometry API required",
    )
    require(
        not Path(cv2.__file__).resolve().is_relative_to(SITE.resolve()),
        "Reuse existing application OpenCV only",
    )
    require(
        serving_identity() == before == read(PUBLIC)["existing_application_packages"],
        "Application packages changed",
    )
    require(
        "insightface" not in sys.modules and "torch" not in sys.modules,
        "No restricted model or neural renderer import",
    )
    write(
        output / "result.json",
        {
            "schema": "chipmunk.portrait-geometry-runtime-result/132",
            "status": "complete",
            "wheels_verified": len(downloaded),
            "installed_files_verified": len(manifest),
            "private_installed_manifest_sha256": sha(output / "installed-files.json"),
            "isolated_versions": {name: row[0] for name, row in PINS.items()},
            "existing_opencv_version": cv2.__version__,
            "application_packages_unchanged": True,
            "freeze_sha256": sha(PUBLIC),
            "new_neural_calls": 0,
            "defaults_changed": False,
            "scope": "Publisher-verified isolated wheel bytes and import/API probe. No face task instance, model inference, microphone/device recording, detector response or customer-runtime change.",
        },
    )


def routed_popen(argv, **kwargs):
    require(
        argv[1:5] == ["-B", "-m", "scripts.video_appearance_native_v108", "bind"],
        "Data-only runtime installation required",
    )
    routed = list(argv)
    routed[3] = "scripts.portrait_geometry_runtime_v132"
    return subprocess.Popen(routed, **kwargs)


def run():
    proxy = types.SimpleNamespace(
        Popen=routed_popen, STDOUT=subprocess.STDOUT, TimeoutExpired=subprocess.TimeoutExpired
    )
    scoped(
        acquisition.runtime.run,
        {
            "TRIAL": TRIAL,
            "verify": verify,
            "prerequisites": lambda c, b: True,
            "subprocess": proxy,
            "__file__": str(Path(__file__)),
        },
    )("bind", 0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "run", "bind"))
    parser.add_argument("arguments", nargs="*")
    args = parser.parse_args()
    if args.command in {"freeze", "run"}:
        require(not args.arguments, "No extra arguments required")
        (freeze if args.command == "freeze" else run)()
    else:
        verify()
        require(len(args.arguments) == 2 and args.arguments[1] == "0", "One fixed guarded stage required")
        output = Path(args.arguments[0])
        require(
            output.resolve() == (TRIAL / "bind-00/output").resolve(),
            "Exact exclusive guarded output required",
        )
        output.mkdir(exist_ok=False)
        install(output)
