"""Read runtime secrets without placing values in Terraform, VM metadata, or Docker configuration."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def validate_secret(secret: dict) -> None:
    if set(secret) != {"api_key", "signing_key"}:
        raise ValueError("runtime secret must contain exactly api_key and signing_key")
    for key, value in secret.items():
        if not isinstance(value, str) or len(value) < 32 or "\x00" in value or "\n" in value:
            raise ValueError(f"{key} must be a single-line secret of at least 32 characters")
    if secret["api_key"] == secret["signing_key"]:
        raise ValueError("bootstrap and signing keys must be independent")


def main() -> None:
    secret = json.loads(Path("/run/secrets/runtime.json").read_text())
    validate_secret(secret)
    os.environ["VERITAS_API_KEY"] = secret["api_key"]
    os.environ["VERITAS_SIGNING_KEY"] = secret["signing_key"]
    config = json.loads(Path("/opt/veritas/config.json").read_text())
    if os.environ.get("VERITAS_BUILD_COMMIT") != config["build_commit"]:
        raise SystemExit("container build commit differs from the reviewed deployment")
    if os.environ.get("VERITAS_DEVICE") == "cuda":
        import torch
        if not torch.cuda.is_available():
            raise SystemExit("CUDA worker has no usable NVIDIA device")
    if len(sys.argv) < 2:
        raise SystemExit("entrypoint needs a command")
    os.execvp(sys.argv[1], sys.argv[1:])


if __name__ == "__main__":
    main()
