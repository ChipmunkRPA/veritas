"""Bound authored public-watermark native tests; no serving changes."""

import argparse
from pathlib import Path

from scripts import specialist_runtime_v4 as guard

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    guard.MODULES = {"scripts.run_public_audio_watermark_v1"}
    guard.run("scripts.run_public_audio_watermark_v1", args.arguments, args.folder, 900)
