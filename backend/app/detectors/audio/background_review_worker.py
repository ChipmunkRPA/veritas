"""Isolated, pinned Apple Silicon speech enhancement for one native PCM interval."""

import hashlib
import importlib.metadata
import sys
from pathlib import Path

PACKAGES = {"mlx": "0.32.4", "mlx-metal": "0.32.4", "mlx-audio": "0.6.5"}
ASSETS = {
    "model.safetensors": "06badaf4029b297f97f7fe48d62990b48d5482608210df577354e0d4e0ed70e1",
    "config.json": "e69986483774f01893d2727503addc49ef201d19e6e1a266b8d4e971e28c478d",
    "LICENSE": "a3d882953f026d3e2fdd57842d9c522c8320d4c0388af2278bcf4d438d255a2c",
}
SAMPLES = 48_000


def main():
    model_path, packages = map(Path, sys.argv[1:])
    installed = {dist.metadata["Name"].lower(): dist.version
                 for dist in importlib.metadata.distributions(path=[str(packages)])}
    if not all(installed.get(package) == version for package, version in PACKAGES.items()):
        raise ValueError("Unexpected local enhancement runtime")
    for name, digest in ASSETS.items():
        with (model_path / name).open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != digest:
                raise ValueError("Unexpected local enhancement asset")
    raw = sys.stdin.buffer.read(SAMPLES * 2 + 1)
    if len(raw) != SAMPLES * 2:
        raise ValueError("Unexpected source sample count")
    sys.path.insert(0, str(packages))
    # Keep any library progress messages off the binary pipe.
    output = sys.stdout.buffer
    sys.stdout = sys.stderr
    import mlx.core as mx
    import numpy as np
    from mlx_audio.sts.utils import load_model

    mx.set_cache_limit(256 * 1024**2)
    model = load_model(str(model_path))
    wave = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    values = np.asarray(model.generate(wave, sample_rate=16_000, chunk_seconds=None).audio, dtype=np.float32)
    if values.shape != wave.shape or not np.isfinite(values).all():
        raise ValueError("Enhancement changed the native sample clock")
    output.write(np.clip(np.rint(values * 32768.0), -32768, 32767).astype("<i2").tobytes())
    output.flush()


if __name__ == "__main__":
    main()
