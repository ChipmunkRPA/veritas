# SPDX-License-Identifier: Apache-2.0
"""Restore exact public BG-REAL photo and mask bytes from a completed JPEG intake."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts import image_jpeg_band_sources


def restore(receipt, output):
    import requests

    from scripts.image_bg_real_acquisition_v221 import fingerprint, png_pixels
    from scripts.image_bg_real_transport_v225 import download

    encoded = receipt.read_bytes()
    body = gzip.decompress(encoded) if receipt.suffix == ".gz" else encoded
    intake = json.loads(body)
    rows = intake["cases"]
    if (intake["schema"] != image_jpeg_band_sources.SCHEMA or not 1 <= len(rows) <= 48
            or [row["tag"] for row in rows] != [f"native-{number:02d}" for number in range(len(rows))]):
        raise ValueError("Expected a complete, numbered BG-REAL JPEG intake")
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    deadline = time.monotonic() + 240

    def fetch(row):
        source = row["source"]
        original = source["original"]
        with requests.Session() as session:
            session.trust_env = False
            image, _ = download(session, original["path"], original["sha256"], deadline)
            mask, _ = download(session, source["subject"], row["subject_sha256"], deadline)
        observed = fingerprint(image, png_pixels(image, source["size"]))
        if observed != row["fingerprints"]["original"] or hashlib.sha256(mask).hexdigest() != row["subject_sha256"]:
            raise ValueError(f"The published {row['tag']} asset differs from the completed intake")
        return row["tag"], image, mask

    with TemporaryDirectory(dir=output.parent, prefix=".restore-jpeg-intake-") as temporary:
        staged = Path(temporary) / "intake"
        staged.mkdir(mode=0o700)
        with ThreadPoolExecutor(max_workers=4) as pool:
            for tag, image, mask in pool.map(fetch, rows):
                (staged / f"{tag}.png").write_bytes(image)
                (staged / f"{tag}-subject.png").write_bytes(mask)
        (staged / "intake.json").write_bytes(body)
        if output.exists() or output.is_symlink():
            raise FileExistsError(output)
        staged.rename(output)
    print(f"Verified {len(rows)} original photos and their subject masks.")


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("receipt", type=Path)
    cli.add_argument("output", type=Path)
    restore(**vars(cli.parse_args()))
