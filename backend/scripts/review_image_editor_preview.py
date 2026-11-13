# SPDX-License-Identifier: Apache-2.0
"""Read back saved embedded-thumbnail evidence against the submitted source."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.evidence.image_editor_preview_source import MAX_BYTES, require, verify

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("export", type=Path)
    parser.add_argument("assets", type=Path, help="Directory containing each crop as <sha256>.png")
    args = parser.parse_args()
    with args.source.open("rb") as stream:
        body = stream.read(MAX_BYTES + 1)
    require(len(body) <= MAX_BYTES, "Source is larger than the verification limit")
    value = json.loads(args.export.read_bytes())
    print(json.dumps(verify(body, value.get("thumbnail_evidence", value),
                            lambda pin: (args.assets / f"{pin}.png").read_bytes()), sort_keys=True))
