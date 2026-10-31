"""Check the reviewed Python dependency and torch wheel identity; no models or downloads.

An image (`--backend cpu` or `cu126`) also names the commit it was built from: `VERITAS_BUILD_COMMIT` is a full commit, or the
check fails at the stage `identity` and with it the image build (N-15; D-58). A checkout (`--backend local`) reads its own HEAD.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.metadata as metadata
import io
import json
import os
import platform
import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import unquote, urlsplit

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name, parse_wheel_filename
from packaging.version import Version

BACKEND = Path(__file__).resolve().parents[1]
BACKENDS = ("local", "cpu", "cu126")
TORCH_PACKAGES = frozenset({"torch", "torchvision"})
WHEEL_HOSTS = frozenset({"download.pytorch.org", "download-r2.pytorch.org"})
_HASH = re.compile(r"(?:^|\s)--hash=sha256:[a-fA-F0-9]{64}(?=\s|$)")
_COMMIT = re.compile(r"[0-9a-f]{40}")
NO_COMMIT = ("VERITAS_BUILD_COMMIT is not a full commit (40 lowercase hexadecimal characters), so no result of this image could "
             "name its build: build with `make up`, or pass --build-arg VERITAS_BUILD_COMMIT=$(git rev-parse HEAD)")


class ContractError(ValueError):
    """The runtime does not match the reviewed dependency contract."""


def parse_lock(content: str, backend: str) -> dict[str, str]:
    """Read exact pins / official torch wheel URLs, ignoring pip hash tokens and comments.

    Installation verifies wheel hashes. This check records the lock's SHA and verifies the installed versions, not wheel bytes.
    """
    if backend not in BACKENDS:
        raise ContractError("unknown runtime backend")
    pins = {}
    for number, original in enumerate(re.sub(r"\\\r?\n", " ", content).splitlines(), 1):
        line = original.strip()
        if not line or line.startswith("#"):
            continue
        line = re.sub(r"\s+#.*$", "", line)
        line = _HASH.sub("", line).strip()
        if not line:
            continue
        try:
            requirement = Requirement(line)
        except ValueError:
            raise ContractError(f"lock line {number}: malformed or unpinned requirement") from None
        name = canonicalize_name(requirement.name)
        if name in pins or requirement.marker is not None or requirement.extras:
            raise ContractError(f"lock line {number}: duplicate, conditional or extra requirement")
        if requirement.url:
            url = urlsplit(requirement.url)
            if (backend == "local" or name not in TORCH_PACKAGES or url.scheme != "https"
                    or url.netloc not in WHEEL_HOSTS or url.query
                    or (url.fragment and not re.fullmatch(r"sha256=[a-fA-F0-9]{64}", url.fragment))
                    or not url.path.startswith(f"/whl/{backend}/")):
                raise ContractError(f"{name}: require an official {backend} torch wheel URL")
            try:
                wheel_name, version, _, tags = parse_wheel_filename(unquote(url.path.rsplit("/", 1)[-1]))
            except ValueError:
                raise ContractError(f"{name}: invalid wheel filename") from None
            if (canonicalize_name(wheel_name) != name or version.local != backend
                    or not all(tag.interpreter == "cp312" and tag.abi == "cp312"
                               and (tag.platform == "linux_x86_64"
                                    or tag.platform.startswith("manylinux") and tag.platform.endswith("_x86_64"))
                               for tag in tags)):
                raise ContractError(f"{name}: wheel must match {backend}, CPython 3.12 and Linux x86_64")
        else:
            specifiers = list(requirement.specifier)
            if len(specifiers) != 1 or specifiers[0].operator != "==" or "*" in specifiers[0].version:
                raise ContractError(f"{name}: an exact version is required")
            try:
                version = Version(specifiers[0].version)
            except ValueError:
                raise ContractError(f"{name}: invalid pinned version") from None
            if backend != "local" and name in TORCH_PACKAGES:
                raise ContractError(f"{name}: Linux contracts require an official backend-specific wheel URL")
        pins[name] = str(version)
    if not pins.keys() >= TORCH_PACKAGES:
        raise ContractError("lock must include torch and torchvision")
    return dict(sorted(pins.items()))


def _pip_check() -> None:
    # pip check is local metadata inspection. Do not inherit index credentials or user pip configuration.
    env = {name: os.environ[name] for name in ("PATH", "LD_LIBRARY_PATH", "SYSTEMROOT", "LANG", "LC_ALL") if name in os.environ}
    env.update({"PIP_CONFIG_FILE": os.devnull, "PIP_NO_INDEX": "1", "PIP_DISABLE_PIP_VERSION_CHECK": "1",
                "PIP_NO_CACHE_DIR": "1", "PYTHONDONTWRITEBYTECODE": "1"})
    result = subprocess.run([sys.executable, "-B", "-m", "pip", "--disable-pip-version-check", "--no-cache-dir", "check"],
                            env=env, capture_output=True, text=True, timeout=60, check=False)
    if result.returncode:
        raise ContractError("pip check failed: installed dependency metadata is inconsistent")


def _native_check(backend: str, versions: dict[str, str]) -> dict:
    import torch
    import torchvision
    from torchvision.ops import nms

    for name, actual in (("torch", torch.__version__), ("torchvision", torchvision.__version__)):
        if Version(actual) != Version(versions[name]):
            raise ContractError(f"{name}: imported version differs from installed package metadata")
    cuda = torch.version.cuda
    expected_cuda = "12.6" if backend == "cu126" else None
    if cuda != expected_cuda:
        raise ContractError(f"torch CUDA backend mismatch: expected {expected_cuda!r}, found {cuda!r}")
    # CPU-only fixture exercises torchvision's compiled operator without a model or a GPU/driver requirement.
    with torch.inference_mode():
        boxes = torch.tensor([[0., 0., 10., 10.], [1., 1., 9., 9.], [20., 20., 30., 30.]], dtype=torch.float32, device="cpu")
        scores = torch.tensor([0.9, 0.8, 0.7], dtype=torch.float32, device="cpu")
        indices = nms(boxes, scores, 0.5).tolist()
    if indices != [0, 2]:
        raise ContractError("torchvision CPU NMS returned unexpected indices")
    return {"torch": str(torch.__version__), "torchvision": str(torchvision.__version__), "cuda": cuda,
            "native_operation": "torchvision.ops.nms on CPU", "indices": indices}


def check_runtime(backend: str) -> dict:
    """Return JSON-safe success/failure evidence; perform no application/model imports or downloads."""
    result = {"schema_version": 1, "ok": False, "backend": backend,
              "python": platform.python_version(), "platform": {"system": platform.system(), "machine": platform.machine()},
              "lock_sha256": None, "versions": {}, "pip_check": "not_run", "native": None, "stage": "runtime",
              "build_commit": None}
    try:
        if backend not in BACKENDS:
            raise ContractError("unknown runtime backend")
        if sys.version_info[:2] != (3, 12):
            raise ContractError("runtime contract requires Python 3.12")
        if backend != "local" and (platform.system() != "Linux" or platform.machine() not in ("x86_64", "AMD64")):
            raise ContractError("image runtime contract requires Linux x86_64")
        if backend != "local":
            result["stage"] = "identity"
            commit = os.environ.get("VERITAS_BUILD_COMMIT", "")
            if not _COMMIT.fullmatch(commit):
                raise ContractError(NO_COMMIT)
            result["build_commit"] = commit
        result["stage"] = "lock"
        name = "tested-versions.txt" if backend == "local" else f"runtime-{backend}.lock"
        path = BACKEND / "requirements" / name
        content = path.read_bytes()
        result["lock"] = name
        result["lock_sha256"] = hashlib.sha256(content).hexdigest()
        pins = parse_lock(content.decode("utf-8"), backend)
        result["stage"] = "versions"
        for name, expected in pins.items():
            try:
                actual = metadata.version(name)
            except metadata.PackageNotFoundError:
                raise ContractError(f"{name}: required package is missing") from None
            result["versions"][name] = actual
            if Version(actual) != Version(expected):
                raise ContractError(f"{name}: expected {expected}, installed {actual}")
        result["stage"] = "pip_check"
        _pip_check()
        result["pip_check"] = "passed"
        result["stage"] = "native"
        # Incidental import output must not corrupt the command's single JSON stdout record.
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            result["native"] = _native_check(backend, result["versions"])
        result.update(ok=True, stage="complete")
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"[:600]
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=BACKENDS, required=True)
    args = parser.parse_args(argv)
    result = check_runtime(args.backend)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
